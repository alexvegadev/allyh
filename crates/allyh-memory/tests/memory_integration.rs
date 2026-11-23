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

    // 1. Session persistence
    let session = SessionRecord {
        session_id: "sess-1".into(),
        repo_id: repo_a.clone(),
        scope_id: DEFAULT_SCOPE_ID.into(),
        started_at: Utc::now(),
        ended_at: None,
        rolling_summary: "Initialized workspace".into(),
        token_budget_used: 1200,
    };
    store.upsert_session(&session).await.expect("upsert session");
    let latest = store
        .latest_session(&repo_a, DEFAULT_SCOPE_ID)
        .await
        .expect("latest")
        .expect("some session");
    assert_eq!(latest.session_id, "sess-1");

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
