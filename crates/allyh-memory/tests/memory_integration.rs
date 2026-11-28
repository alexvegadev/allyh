use std::sync::Arc;
use std::time::Duration;

use allyh_memory::{
    CandidateDecision, CodeGraphSnapshot, ConsolidationAction, DEFAULT_SCOPE_ID,
    DeterministicHashEmbedder, HeuristicConflictJudge, MemoryConsolidator, MemoryStore,
    SessionRecord, SleepJob, SleepTimeWorkerHandle, StorageBackendKind, StorageConfig, TaskRecord,
    TaskStatus, VerificationProof, compute_repo_id,
};
use chrono::Utc;
use tempfile::TempDir;

#[tokio::test]
async fn test_lancedb_bitemporal_isolation_and_task_gate() {
    let dir = TempDir::new().expect("tempdir");
    let cfg = StorageConfig::from_local_path(dir.path().join("db"));
    assert_eq!(cfg.backend_kind(), StorageBackendKind::Local);

    let s3_cfg = StorageConfig::from_uri("s3://allyh-memory-bucket/shared-db");
    assert_eq!(s3_cfg.backend_kind(), StorageBackendKind::S3Shared);

    let embedder = Arc::new(DeterministicHashEmbedder);
    let store = MemoryStore::connect(cfg, embedder).await.expect("connect");

    let repo_a = compute_repo_id("/repos/alpha");
    let repo_b = compute_repo_id("/repos/beta");

    // 1. Session persistence (including prompt-cache & cost telemetry)
    let session = SessionRecord {
        session_id: "sess-1".into(),
        repo_id: repo_a.clone(),
        scope_id: DEFAULT_SCOPE_ID.into(),
        started_at: Utc::now(),
        ended_at: None,
        rolling_summary: "Initialized workspace".into(),
        token_budget_used: 1200,
        input_tokens: 18_500,
        output_tokens: 1_250,
        cached_input_tokens: 12_000,
        cache_creation_tokens: 2_500,
        estimated_cost_usd: 0.0425,
    };
    store.upsert_session(&session).await.expect("upsert session");
    let latest = store
        .latest_session(&repo_a, DEFAULT_SCOPE_ID)
        .await
        .expect("latest")
        .expect("some session");
    assert_eq!(latest.session_id, "sess-1");
    assert_eq!(latest.input_tokens, 18_500);
    assert_eq!(latest.output_tokens, 1_250);
    assert_eq!(latest.cached_input_tokens, 12_000);
    assert_eq!(latest.cache_creation_tokens, 2_500);
    assert!((latest.estimated_cost_usd - 0.0425).abs() < 1e-6);

    // 2. Write-gated bi-temporal consolidation
    let consolidator = MemoryConsolidator::new(store.clone(), Arc::new(HeuristicConflictJudge));
    let added = consolidator
        .consolidate_decision(
            &repo_a,
            DEFAULT_SCOPE_ID,
            CandidateDecision {
                topic: "error-handling".into(),
                rule_statement: "Use thiserror in allyh-core and anyhow only in allyh-tui".into(),
                rationale: "Structured library errors with ergonomic binary top-level errors"
                    .into(),
                affected_paths: vec!["crates/allyh-core".into(), "crates/allyh-tui".into()],
            },
        )
        .await
        .expect("add decision");
    assert_eq!(added.action, ConsolidationAction::Add);
    let old_id = added.committed_id.clone().expect("old_id");

    // Duplicate is NOOP
    let noop = consolidator
        .consolidate_decision(
            &repo_a,
            DEFAULT_SCOPE_ID,
            CandidateDecision {
                topic: "error-handling".into(),
                rule_statement: "Use thiserror in allyh-core and anyhow only in allyh-tui".into(),
                rationale: "Repeated rule".into(),
                affected_paths: vec!["crates/allyh-core".into()],
            },
        )
        .await
        .expect("noop");
    assert!(matches!(noop.action, ConsolidationAction::Noop { .. }));

    // Hybrid search finds it in repo_a, NOT in repo_b
    let hits_a = store
        .hybrid_search_decisions(&repo_a, DEFAULT_SCOPE_ID, "thiserror allyh-core", 5)
        .await
        .expect("search a");
    assert_eq!(hits_a.len(), 1);
    assert_eq!(hits_a[0].decision.id, old_id);

    let hits_b = store
        .hybrid_search_decisions(&repo_b, DEFAULT_SCOPE_ID, "thiserror allyh-core", 5)
        .await
        .expect("search b");
    assert!(hits_b.is_empty());

    // Override supersedes old rule via invalid_at + superseded_by
    let updated = consolidator
        .consolidate_decision(
            &repo_a,
            DEFAULT_SCOPE_ID,
            CandidateDecision {
                topic: "error-handling".into(),
                rule_statement: "Use miette across all crates including allyh-core".into(),
                rationale: "Richer diagnostic spans".into(),
                affected_paths: vec!["crates/allyh-core".into()],
            },
        )
        .await
        .expect("update");
    assert!(matches!(
        updated.action,
        ConsolidationAction::Update { ref target_id } if target_id == &old_id
    ));
    let new_id = updated.committed_id.clone().expect("new_id");

    let active = store
        .list_active_decisions(&repo_a, DEFAULT_SCOPE_ID)
        .await
        .expect("active");
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].id, new_id);

    let all = store
        .list_all_decisions(&repo_a, DEFAULT_SCOPE_ID)
        .await
        .expect("all");
    assert_eq!(all.len(), 2);
    let old_rec = all.iter().find(|d| d.id == old_id).expect("old record");
    assert!(old_rec.invalid_at.is_some());
    assert_eq!(old_rec.superseded_by.as_deref(), Some(new_id.as_str()));

    // 3. Task verification proof gate
    let task = TaskRecord {
        id: "task-1".into(),
        repo_id: repo_a.clone(),
        scope_id: DEFAULT_SCOPE_ID.into(),
        parent_id: None,
        title: "Implement RepoGraph".into(),
        acceptance_criteria: vec!["cargo test passes".into()],
        status: TaskStatus::InProgress,
        last_error_trace: None,
        updated_at: Utc::now(),
    };
    store.upsert_task(&task).await.expect("upsert task");

    let rejected = store
        .transition_task_status(
            &repo_a,
            DEFAULT_SCOPE_ID,
            "task-1",
            TaskStatus::Verified,
            None,
            None,
        )
        .await;
    assert!(rejected.is_err());

    let proof = VerificationProof {
        command: "cargo test".into(),
        passed_at: Utc::now(),
        warnings: 0,
        summary: "all passed".into(),
    };
    let verified = store
        .transition_task_status(
            &repo_a,
            DEFAULT_SCOPE_ID,
            "task-1",
            TaskStatus::Verified,
            Some(&proof),
            None,
        )
        .await
        .expect("verified with proof");
    assert_eq!(verified.status, TaskStatus::Verified);

    // 4. Code snapshot upsert/get/delete
    let snap = CodeGraphSnapshot {
        repo_id: repo_a.clone(),
        scope_id: DEFAULT_SCOPE_ID.into(),
        file_path: "src/lib.rs".into(),
        language: "rust".into(),
        sha256_hash: "deadbeef".into(),
        last_read_turn: 1,
        symbols_json: "[]".into(),
        dependencies: vec!["std::sync::Arc".into()],
        vector: DeterministicHashEmbedder::embed_sync("src/lib.rs"),
    };
    store
        .upsert_code_snapshot(&snap)
        .await
        .expect("upsert snapshot");
    let fetched = store
        .get_code_snapshot(&repo_a, DEFAULT_SCOPE_ID, "src/lib.rs")
        .await
        .expect("get snapshot")
        .expect("snapshot exists");
    assert_eq!(fetched.sha256_hash, "deadbeef");

    // 5. Sleep-time worker processes async candidate + optimize
    let worker = SleepTimeWorkerHandle::spawn(consolidator, Duration::from_secs(60));
    worker.enqueue(SleepJob::ConsolidateCandidates {
        repo_id: repo_a.clone(),
        scope_id: DEFAULT_SCOPE_ID.into(),
        candidates: vec![CandidateDecision {
            topic: "async-runtime".into(),
            rule_statement: "Use Tokio multi-threaded runtime everywhere".into(),
            rationale: "Single executor".into(),
            affected_paths: vec!["crates/".into()],
        }],
    });
    worker.enqueue(SleepJob::OptimizeTables);
    tokio::time::sleep(Duration::from_millis(350)).await;
    worker.enqueue(SleepJob::Shutdown);

    let report = store
        .status_report(&repo_a, DEFAULT_SCOPE_ID)
        .await
        .expect("status report");
    assert_eq!(report.sessions_count, 1);
    assert_eq!(report.active_decisions_count, 2);
    assert_eq!(report.invalidated_decisions_count, 1);
    assert_eq!(report.tasks_count, 1);
    assert_eq!(report.code_snapshots_count, 1);
}

#[test]
fn test_cross_platform_uri_expansion_and_repo_id_normalization() {
    use allyh_memory::expand_local_uri;

    // Remote URIs stay untouched
    assert_eq!(
        expand_local_uri("s3://allyh-bucket/shared-db"),
        "s3://allyh-bucket/shared-db"
    );

    // Tilde expansion resolves against home directory with both `/` and `\`
    if let Some(home) = dirs::home_dir() {
        let expected = home.join("allyh").join("db").to_string_lossy().into_owned();
        assert_eq!(expand_local_uri("~/allyh/db"), expected);
        assert_eq!(expand_local_uri(r"~\allyh\db"), expected);
        assert_eq!(
            StorageConfig::from_uri("~/allyh/db").uri,
            expected
        );
    }

    // Windows verbatim prefixes, backslashes, and trailing slashes produce identical repo_id
    let standard = compute_repo_id("C:/Users/dev/project");
    let backslash = compute_repo_id(r"C:\Users\dev\project");
    let verbatim = compute_repo_id(r"\\?\C:\Users\dev\project");
    let trailing = compute_repo_id("C:/Users/dev/project/");
    assert_eq!(standard, backslash);
    assert_eq!(standard, verbatim);
    assert_eq!(standard, trailing);
}

#[tokio::test]
async fn test_skills_crud_matching_and_partition_isolation() {
    use allyh_memory::{MemoryError, normalize_skill_slug};

    let dir = TempDir::new().expect("tempdir");
    let store = MemoryStore::connect(
        StorageConfig::from_local_path(dir.path().join("db")),
        Arc::new(DeterministicHashEmbedder),
    )
    .await
    .expect("connect");

    let repo_a = compute_repo_id("/repos/skills-alpha");
    let repo_b = compute_repo_id("/repos/skills-beta");

    assert_eq!(normalize_skill_slug("Add  LanceDB!! Table"), "add-lancedb-table");
    assert_eq!(normalize_skill_slug("---"), "");

    // Punctuation-only names are rejected cleanly
    let bad = store
        .save_skill(&repo_a, DEFAULT_SCOPE_ID, "---", "desc", "steps", vec![])
        .await;
    assert!(matches!(bad, Err(MemoryError::InvalidSkill(_))));

    // Create skill
    let (created_rec, created) = store
        .save_skill(
            &repo_a,
            DEFAULT_SCOPE_ID,
            "add-lancedb-table",
            "Add a new Apache Arrow table to LanceDB",
            "1. Define Arrow schema.\n2. Register table in store.rs.",
            vec!["new table".into(), "lancedb schema".into(), "new table".into()],
        )
        .await
        .expect("create skill");
    assert!(created);
    assert_eq!(created_rec.triggers, vec!["new table", "lancedb schema"]);

    // Update preserves created_at
    let (updated_rec, created_again) = store
        .save_skill(
            &repo_a,
            DEFAULT_SCOPE_ID,
            "add-lancedb-table",
            "Add a new Arrow table to LanceDB",
            "1. Define Arrow schema.\n2. Register table in store.rs.\n3. Add tests.",
            vec!["new table".into(), "lancedb schema".into()],
        )
        .await
        .expect("update skill");
    assert!(!created_again);
    assert_eq!(updated_rec.created_at, created_rec.created_at);
    assert!(updated_rec.instructions.contains("Add tests."));

    // Lookup by exact name, case-insensitive name, slug, and id
    assert!(
        store
            .get_skill(&repo_a, DEFAULT_SCOPE_ID, "ADD-LANCEDB-TABLE")
            .await
            .expect("get ci")
            .is_some()
    );
    assert!(
        store
            .get_skill(&repo_a, DEFAULT_SCOPE_ID, &updated_rec.id)
            .await
            .expect("get id")
            .is_some()
    );

    // Partition isolation across repos and scopes
    assert!(store.list_skills(&repo_b, DEFAULT_SCOPE_ID).await.expect("repo b").is_empty());
    assert!(store.list_skills(&repo_a, "other-scope").await.expect("other scope").is_empty());

    // Trigger matching
    let matched = store
        .match_skills(&repo_a, DEFAULT_SCOPE_ID, "How do I add a new table here?", 3)
        .await
        .expect("match");
    assert_eq!(matched.len(), 1);
    assert_eq!(matched[0].name, "add-lancedb-table");

    let unmatched = store
        .match_skills(&repo_a, DEFAULT_SCOPE_ID, "Fix CSS button alignment", 3)
        .await
        .expect("unmatched");
    assert!(unmatched.is_empty());

    // DbStatusReport includes skills_count
    let report = store
        .status_report(&repo_a, DEFAULT_SCOPE_ID)
        .await
        .expect("status");
    assert_eq!(report.skills_count, 1);

    // Delete skill
    let removed = store
        .delete_skill(&repo_a, DEFAULT_SCOPE_ID, "add-lancedb-table")
        .await
        .expect("delete");
    assert!(removed.is_some());
    assert!(store.list_skills(&repo_a, DEFAULT_SCOPE_ID).await.expect("empty").is_empty());
    assert_eq!(
        store
            .status_report(&repo_a, DEFAULT_SCOPE_ID)
            .await
            .expect("status after delete")
            .skills_count,
        0
    );
}

#[tokio::test]
async fn test_sim_ledger_storage_lineage_aliases_and_json_roundtrip() {
    use allyh_memory::{
        HarnessVersionRecord, MemoryError, SimAttemptInput, SimExportDocument, SimLedger,
        SimMemoryMode, SimProblemEntry, SimProblemRef,
    };

    let dir = TempDir::new().expect("tempdir");
    let store = MemoryStore::connect(
        StorageConfig::from_local_path(dir.path().join("db")),
        Arc::new(DeterministicHashEmbedder),
    )
    .await
    .expect("connect");

    let repo_a = compute_repo_id("/repos/sim-alpha");
    let repo_b = compute_repo_id("/repos/sim-beta");

    // 1. Harness version persistence & single-active switching
    let h0 = HarnessVersionRecord {
        id: "hv-seed-0".into(),
        repo_id: repo_a.clone(),
        scope_id: DEFAULT_SCOPE_ID.into(),
        parent_version_id: None,
        run_id: None,
        round: 0,
        label: "seed".into(),
        spec_json: r#"{"version":1}"#.into(),
        sha256_hash: "hash0".into(),
        policy_model: "anthropic:claude-sonnet-4-6".into(),
        domain: "coding".into(),
        is_active: true,
        created_at: Utc::now(),
    };
    let h1 = HarnessVersionRecord {
        id: "hv-evolved-1".into(),
        parent_version_id: Some("hv-seed-0".into()),
        round: 1,
        label: "round-1".into(),
        spec_json: r#"{"version":2}"#.into(),
        sha256_hash: "hash1".into(),
        is_active: false,
        ..h0.clone()
    };
    store.upsert_harness_version(&h0).await.expect("upsert h0");
    store.upsert_harness_version(&h1).await.expect("upsert h1");
    let activated = store
        .set_active_harness_version(&repo_a, DEFAULT_SCOPE_ID, "hv-evolved")
        .await
        .expect("activate h1");
    assert_eq!(activated.id, "hv-evolved-1");
    let active_now = store
        .get_active_harness_version(&repo_a, DEFAULT_SCOPE_ID)
        .await
        .expect("get active")
        .expect("active exists");
    assert_eq!(active_now.id, "hv-evolved-1");

    // 2. Source SIM ledger recording & rejected-attempt guard
    let mut source = SimLedger::new_source(
        &repo_a,
        DEFAULT_SCOPE_ID,
        "source-tb21",
        "coding_enriched",
        "tb2.1",
        "anthropic:claude-sonnet-4-6",
        "anthropic:claude-sonnet-4-6",
        "hv-seed-0",
        SimMemoryMode::FullSim,
    );

    let rejected = source.record_accepted_attempt(SimAttemptInput {
        problem_ref: SimProblemRef::New("premature completion".into()),
        round: 1,
        sample: "s1".into(),
        accepted: false,
        backfilled: false,
        what: "rejected experiment".into(),
        files: vec!["interlock.rs".into()],
        predicted: vec!["task-a".into()],
        fixed: vec![],
        missed: vec!["task-a".into()],
        passed: 1,
        total: 5,
        score: 20.0,
        domain: "coding_enriched".into(),
        benchmark: "tb2.1".into(),
        policy_model: "anthropic:claude-sonnet-4-6".into(),
        harness_version_id: "hv-seed-0".into(),
    });
    assert!(matches!(rejected, Err(MemoryError::InvalidSim(_))));
    assert!(source.problems.is_empty());

    let u1 = source
        .record_accepted_attempt(SimAttemptInput {
            problem_ref: SimProblemRef::New("premature completion without verification".into()),
            round: 1,
            sample: "s2".into(),
            accepted: true,
            backfilled: false,
            what: "Add execution gating interlock".into(),
            files: vec!["crates/allyh-core/src/interlock.rs".into()],
            predicted: vec!["task-a".into(), "task-b".into()],
            fixed: vec!["task-a".into()],
            missed: vec!["task-b".into()],
            passed: 4,
            total: 5,
            score: 80.0,
            domain: "coding_enriched".into(),
            benchmark: "tb2.1".into(),
            policy_model: "anthropic:claude-sonnet-4-6".into(),
            harness_version_id: "hv-evolved-1".into(),
        })
        .expect("record P1");
    assert_eq!(u1.canonical_problem_id, "P1");
    assert!(u1.opened_new_problem);

    // Deterministic exact-text matching merges duplicate New() into P1 automatically
    let u2 = source
        .record_accepted_attempt(SimAttemptInput {
            problem_ref: SimProblemRef::New("  Premature   Completion Without Verification ".into()),
            round: 2,
            sample: "s1".into(),
            accepted: true,
            backfilled: false,
            what: "Tighten interlock check after failed tool calls".into(),
            files: vec!["crates/allyh-core/src/interlock.rs".into()],
            predicted: vec!["task-b".into()],
            fixed: vec!["task-b".into()],
            missed: vec![],
            passed: 5,
            total: 5,
            score: 100.0,
            domain: "coding_enriched".into(),
            benchmark: "tb2.1".into(),
            policy_model: "anthropic:claude-sonnet-4-6".into(),
            harness_version_id: "hv-evolved-1".into(),
        })
        .expect("merge into P1");
    assert_eq!(u2.canonical_problem_id, "P1");
    assert!(!u2.opened_new_problem);
    assert_eq!(source.problems.len(), 1);
    assert_eq!(source.problems[0].attempts.len(), 2);

    // Explicit alias merge when duplicate entries are loaded/imported
    source.problems.push(SimProblemEntry {
        id: "P6".into(),
        name: "premature completion without verification".into(),
        opened_round: 3,
        origin_domain: "coding_enriched".into(),
        origin_run_id: source.run.run_id.clone(),
        attempts: vec![source.problems[0].attempts[1].clone()],
    });
    let merged = source.merge_identical_problems();
    assert_eq!(merged, vec![("P6".to_string(), "P1".to_string())]);
    assert_eq!(source.resolve_alias("P6"), "P1");
    assert_eq!(source.problems[0].attempts.len(), 3);

    // Persist to LanceDB & reload
    store.save_sim_ledger(&source).await.expect("save source");
    let loaded = store
        .load_sim_ledger(&repo_a, DEFAULT_SCOPE_ID, &source.run.run_id)
        .await
        .expect("load source")
        .expect("source exists");
    assert_eq!(loaded.problems.len(), 1);
    assert_eq!(loaded.problems[0].attempts.len(), 3);
    assert_eq!(loaded.resolve_alias("P6"), "P1");

    // Partition isolation
    assert!(
        store
            .list_sim_runs(&repo_b, DEFAULT_SCOPE_ID)
            .await
            .expect("repo b runs")
            .is_empty()
    );

    // 3. Carryover into FullSim vs ProblemOnly vs HarnessTransfer
    let target_full = loaded.carry_into(
        &repo_a,
        DEFAULT_SCOPE_ID,
        "target-tb4-full",
        "tb4",
        "tb4",
        "openai:gpt-4o",
        "openai:gpt-4o",
        "hv-evolved-1",
        SimMemoryMode::FullSim,
    );
    assert_eq!(
        target_full.run.carried_from_run_id.as_deref(),
        Some(loaded.run.run_id.as_str())
    );
    assert_eq!(target_full.problems.len(), 1);
    assert_eq!(target_full.problems[0].attempts.len(), 3);
    assert!(
        target_full
            .render_for_prompt(SimMemoryMode::FullSim)
            .contains("Add execution gating interlock")
    );

    let target_prob_only = loaded.carry_into(
        &repo_a,
        DEFAULT_SCOPE_ID,
        "target-tb4-prob-only",
        "tb4",
        "tb4",
        "openai:gpt-4o",
        "openai:gpt-4o",
        "hv-evolved-1",
        SimMemoryMode::ProblemOnly,
    );
    assert_eq!(target_prob_only.problems.len(), 1);
    assert!(target_prob_only.problems[0].attempts.is_empty());
    let prob_prompt = target_prob_only.render_for_prompt(SimMemoryMode::ProblemOnly);
    assert!(prob_prompt.contains("premature completion without verification"));
    assert!(!prob_prompt.contains("Add execution gating interlock"));

    // 4. Appendix D.3 JSON Export / Import round-trip
    let doc = target_full.to_export_document();
    let json = serde_json::to_string_pretty(&doc).expect("serialize doc");
    let parsed: SimExportDocument = serde_json::from_str(&json).expect("deserialize doc");
    assert_eq!(parsed.aliases.get("P6").map(String::as_str), Some("P1"));
    assert_eq!(
        parsed.carried_from.as_ref().map(|c| c.into.as_str()),
        Some("target-tb4-full")
    );

    let imported = store
        .import_sim_document(
            &repo_b,
            DEFAULT_SCOPE_ID,
            parsed,
            "imported-arm",
            "tb4",
            "tb4",
            "local:qwen-35b",
            "local:qwen-35b",
            "hv-evolved-1",
        )
        .await
        .expect("import into repo_b");
    assert_eq!(imported.problems.len(), 1);
    assert_eq!(imported.problems[0].attempts.len(), 3);

    // 5. DbStatusReport includes SIM counts
    let status = store
        .status_report(&repo_a, DEFAULT_SCOPE_ID)
        .await
        .expect("status");
    assert_eq!(status.harness_versions_count, 2);
    assert_eq!(status.sim_runs_count, 1);
    assert_eq!(status.sim_problems_count, 1);
    assert_eq!(status.sim_attempts_count, 3);
}

#[tokio::test]
async fn test_repo_profile_crud_and_partition_isolation() {
    use allyh_memory::RepoProfile;

    let dir = TempDir::new().expect("tempdir");
    let store = MemoryStore::connect(
        StorageConfig::from_local_path(dir.path().join("db")),
        Arc::new(DeterministicHashEmbedder),
    )
    .await
    .expect("connect");

    let repo_a = compute_repo_id("/repos/poly-alpha");
    let repo_b = compute_repo_id("/repos/poly-beta");

    assert!(
        store
            .get_repo_profile(&repo_a, DEFAULT_SCOPE_ID)
            .await
            .expect("empty get")
            .is_none()
    );

    let mut profile = RepoProfile::empty(&repo_a, DEFAULT_SCOPE_ID);
    profile.languages_override = vec!["TypeScript".into(), "Python".into()];
    profile.check_command = Some("npx tsc --noEmit".into());
    profile.lint_command = Some("ruff check .".into());
    profile.test_command = Some("pytest -q".into());
    profile.extra_ignore_globs = vec!["generated/**".into()];
    profile.auto_verify_override = Some(true);

    store
        .upsert_repo_profile(&profile)
        .await
        .expect("upsert profile");

    let loaded = store
        .get_repo_profile(&repo_a, DEFAULT_SCOPE_ID)
        .await
        .expect("get profile")
        .expect("profile exists");
    assert_eq!(loaded.check_command.as_deref(), Some("npx tsc --noEmit"));
    assert_eq!(loaded.lint_command.as_deref(), Some("ruff check ."));
    assert_eq!(loaded.test_command.as_deref(), Some("pytest -q"));
    assert_eq!(loaded.extra_ignore_globs, vec!["generated/**"]);
    assert_eq!(loaded.auto_verify_override, Some(true));

    // Partition isolation across repos and scopes
    assert!(
        store
            .get_repo_profile(&repo_b, DEFAULT_SCOPE_ID)
            .await
            .expect("repo b")
            .is_none()
    );
    assert!(
        store
            .get_repo_profile(&repo_a, "other-scope")
            .await
            .expect("other scope")
            .is_none()
    );

    store
        .delete_repo_profile(&repo_a, DEFAULT_SCOPE_ID)
        .await
        .expect("delete profile");
    assert!(
        store
            .get_repo_profile(&repo_a, DEFAULT_SCOPE_ID)
            .await
            .expect("after delete")
            .is_none()
    );
}
