use std::sync::Arc;
use std::time::Duration;

use allyh_core::{
    AllyHOrchestrator, HybridRouter, ModelProfile, OrchestratorConfig, RigCompletionBackend,
};
use allyh_memory::StorageConfig;
use tempfile::TempDir;

fn offline_router() -> HybridRouter {
    HybridRouter::new(
        ModelProfile::offline_mock("Test-Primary"),
        ModelProfile::offline_mock("Test-Local-9B"),
        Arc::new(RigCompletionBackend),
    )
}

fn make_config(repo_dir: &std::path::Path, db_dir: &std::path::Path, scope_id: &str) -> OrchestratorConfig {
    OrchestratorConfig {
        repo_root: repo_dir.to_path_buf(),
        storage: StorageConfig::from_local_path(db_dir),
        scope_id: scope_id.to_string(),
        auto_verify: false,
        idle_optimize_interval: Duration::from_secs(60),
        enable_fs_watcher: false,
    }
}

/// Phase 4 — 5-Step Isolation & Coherence Acceptance Suite from PLAN.md:
/// 1. Establish an architectural constraint in Session A without polluting the repo with `.md` files.
/// 2. Close AllyH, reopen in Session B, request a new module — verify automatic adherence.
/// 3. Manually edit a file outside AllyH — verify SHA-256 mismatch detection + diff summary injection.
/// 4. Override the architectural rule in Session C — verify `invalid_at` + `superseded_by` without conflict.
/// 5. Switch to a separate repository (and second scope) on the same LanceDB URI — verify strict isolation.
#[tokio::test]
async fn five_step_isolation_and_coherence_acceptance_suite() {
    let tmp = TempDir::new().expect("tempdir");
    let db_dir = tmp.path().join("shared_lancedb");
    let repo_a_dir = tmp.path().join("repo_a");
    let repo_b_dir = tmp.path().join("repo_b");

    std::fs::create_dir_all(repo_a_dir.join("src")).expect("mkdir repo_a");
    std::fs::create_dir_all(repo_b_dir.join("src")).expect("mkdir repo_b");

    let repo_a_lib = repo_a_dir.join("src/lib.rs");
    std::fs::write(&repo_a_lib, "pub fn init() -> bool { true }\n").expect("write repo_a lib");
    std::fs::write(repo_b_dir.join("src/lib.rs"), "pub fn other() -> i32 { 7 }\n")
        .expect("write repo_b lib");

    // -----------------------------------------------------------------------
    // Step 1: Session A establishes an architectural constraint
    // -----------------------------------------------------------------------
    {
        let mut session_a = AllyHOrchestrator::initialize(
            make_config(&repo_a_dir, &db_dir, "default"),
            offline_router(),
            None,
        )
        .await
        .expect("init session A");

        let turn_a = session_a
            .handle_prompt(
                "Rule: Use thiserror in allyh-core and anyhow only in allyh-tui",
                None,
            )
            .await
            .expect("turn A");

        assert_eq!(
            turn_a.consolidated_decisions.len(),
            1,
            "constraint should be persisted through bi-temporal consolidator"
        );
        // Verify zero Markdown plan files polluted the target repository.
        assert!(!repo_a_dir.join("PLAN.md").exists());
        assert!(!repo_a_dir.join("TODO.md").exists());
    } // Session A closed/dropped

    // -----------------------------------------------------------------------
    // Step 2: Reopen in Session B — automatic adherence without in-repo `.md`
    // -----------------------------------------------------------------------
    let mut session_b = AllyHOrchestrator::initialize(
        make_config(&repo_a_dir, &db_dir, "default"),
        offline_router(),
        None,
    )
    .await
    .expect("init session B");

    let turn_b = session_b
        .handle_prompt("Add a new error module in allyh-core", None)
        .await
        .expect("turn B");

    assert!(
        turn_b
            .preamble
            .text
            .contains("Use thiserror in allyh-core and anyhow only in allyh-tui"),
        "Session B preamble must automatically inject persisted invariant from LanceDB"
    );
    assert!(
        turn_b.assistant_text.contains("thiserror"),
        "Session B response should reflect active invariant"
    );

    // -----------------------------------------------------------------------
    // Step 3: Manual out-of-band file edit — SHA-256 mismatch + diff summary
    // -----------------------------------------------------------------------
    std::fs::write(
        &repo_a_lib,
        "pub fn init() -> bool { true }\npub fn manual_hook() -> usize { 42 }\n",
    )
    .expect("external manual edit");

    let turn_b2 = session_b
        .handle_prompt("Check manual_hook in src/lib.rs", None)
        .await
        .expect("turn B2");

    assert_eq!(
        turn_b2.env_delta.out_of_band_edits.len(),
        1,
        "out-of-band edit must be detected via SHA-256 comparison"
    );
    assert_eq!(turn_b2.env_delta.out_of_band_edits[0].file_path, "src/lib.rs");
    assert!(
        turn_b2.preamble.text.contains("Out-of-band manual edits detected"),
        "diff summary must be injected into the next turn's preamble"
    );
    assert!(
        turn_b2.preamble.text.contains("manual_hook"),
        "new symbol from incremental tree-sitter re-indexing should be visible in preamble"
    );
    drop(session_b);

    // -----------------------------------------------------------------------
    // Step 4: Session C overrides the architectural rule -> bi-temporal invalidation
    // -----------------------------------------------------------------------
    {
        let mut session_c = AllyHOrchestrator::initialize(
            make_config(&repo_a_dir, &db_dir, "default"),
            offline_router(),
            None,
        )
        .await
        .expect("init session C");

        let turn_c1 = session_c
            .handle_prompt(
                "Override: Use miette for error handling across allyh-core instead of thiserror",
                None,
            )
            .await
            .expect("turn C1");
        assert_eq!(turn_c1.consolidated_decisions.len(), 1);
        assert!(
            turn_c1.consolidated_decisions[0].invalidated_id.is_some(),
            "old rule must be logically invalidated via invalid_at"
        );

        let all = session_c
            .store()
            .list_all_decisions(session_c.repo_id(), session_c.scope_id())
            .await
            .expect("list all");
        assert_eq!(all.len(), 2, "history preserved bi-temporally without hard delete");
        let old = all
            .iter()
            .find(|d| d.rule_statement.contains("anyhow only in allyh-tui"))
            .expect("old rule exists in history");
        assert!(old.invalid_at.is_some());
        assert!(old.superseded_by.is_some());

        // Subsequent query in Session C sees ONLY the new rule, no retrieval conflict.
        let turn_c2 = session_c
            .handle_prompt("How should we implement error types in allyh-core?", None)
            .await
            .expect("turn C2");
        assert!(turn_c2.preamble.text.contains("miette"));
        assert!(!turn_c2.preamble.text.contains("anyhow only in allyh-tui"));
    }

    // -----------------------------------------------------------------------
    // Step 5: Separate repository (and separate scope) on same LanceDB URI
    // -----------------------------------------------------------------------
    let mut session_repo_b = AllyHOrchestrator::initialize(
        make_config(&repo_b_dir, &db_dir, "default"),
        offline_router(),
        None,
    )
    .await
    .expect("init repo B");

    let turn_repo_b = session_repo_b
        .handle_prompt("Create an error module", None)
        .await
        .expect("turn repo B");

    assert!(
        !turn_repo_b.preamble.text.contains("miette")
            && !turn_repo_b.preamble.text.contains("thiserror"),
        "strict repo_id partition isolation must prevent cross-repository memory leakage"
    );

    // Also verify scope_id isolation for a second subagent on repo_a with a different scope.
    let mut session_other_scope = AllyHOrchestrator::initialize(
        make_config(&repo_a_dir, &db_dir, "agent-isolated"),
        offline_router(),
        None,
    )
    .await
    .expect("init isolated scope");

    let turn_scope = session_other_scope
        .handle_prompt("Check error handling rules", None)
        .await
        .expect("turn isolated scope");
    assert!(
        !turn_scope.preamble.text.contains("miette"),
        "scope_id partition isolation must isolate distinct agent scopes"
    );
}

#[tokio::test]
async fn test_control_gate_and_non_destructive_compaction() {
    use allyh_core::{CompactingHistory, ControlGateState};
    use rig_core::message::{
        AssistantContent, AssistantMessage, CallId, Message, ToolName, ToolResultContent,
        UserContent,
    };

    let tmp = TempDir::new().expect("tempdir");
    let db_dir = tmp.path().join("db");
    let repo_dir = tmp.path().join("repo");
    std::fs::create_dir_all(repo_dir.join("src")).expect("mkdir");
    std::fs::write(repo_dir.join("src/lib.rs"), "pub fn v() -> u8 { 1 }\n").expect("write");

    let mut orch = AllyHOrchestrator::initialize(
        make_config(&repo_dir, &db_dir, "default"),
        offline_router(),
        None,
    )
    .await
    .expect("init");

    // Initially Discovery (writes locked)
    assert!(!orch.toolset().writes_enabled());

    // Propose plan -> still locked
    let state = orch.propose_plan(AllyHOrchestrator::build_quick_plan("Add feature X"));
    assert!(matches!(state, ControlGateState::PlanApproval { .. }));
    assert!(!orch.toolset().writes_enabled());

    // Approve plan -> tasks persisted in LanceDB & writes unlocked
    let tasks = orch.approve_plan().await.expect("approve");
    assert_eq!(tasks.len(), 1);
    assert!(orch.toolset().writes_enabled());

    // Non-destructive history compaction preserves intact tool call / tool result pairs
    let mut history = CompactingHistory::new(String::new());
    for i in 0..8 {
        history.push_message(Message::user(format!("Prompt {i}")));
        let tool_call_msg = Message::Assistant(AssistantMessage::new(vec![
            AssistantContent::tool_call(
                format!("call-{i}"),
                ToolName::new("inspect_symbol").unwrap(),
                serde_json::json!({"query": "v"}),
            ),
        ]));
        let tool_res_msg = Message::User {
            content: vec![UserContent::tool_result(
                CallId::from_wire(format!("call-{i}")),
                ToolName::new("inspect_symbol").unwrap(),
                vec![ToolResultContent::text("ok")],
            )],
        };
        history.push_atomic_group(vec![tool_call_msg, tool_res_msg]);
    }

    let router = offline_router();
    let compacted = history.compact_if_needed(&router, 0, 1000, true).await;
    assert!(compacted);
    assert!(!history.rolling_summary().is_empty());

    // Verify every retained ToolCall has its matching ToolResult immediately after it.
    let msgs = history.messages();
    for (idx, msg) in msgs.iter().enumerate() {
        if let Message::Assistant(ast) = msg
            && ast.tool_calls().next().is_some()
        {
            let next = msgs.get(idx + 1).expect("matching tool result must remain intact");
            assert!(matches!(next, Message::User { .. }));
        }
    }
}

#[tokio::test]
async fn test_skills_progressive_disclosure_and_actor_commands() {
    use allyh_core::{UiEvent, UserCommand};

    let tmp = TempDir::new().expect("tempdir");
    let db_dir = tmp.path().join("db");
    let repo_a_dir = tmp.path().join("repo_a");
    let repo_b_dir = tmp.path().join("repo_b");
    std::fs::create_dir_all(repo_a_dir.join("src")).expect("mkdir a");
    std::fs::create_dir_all(repo_b_dir.join("src")).expect("mkdir b");
    std::fs::write(repo_a_dir.join("src/lib.rs"), "pub fn a() {}\n").expect("write a");
    std::fs::write(repo_b_dir.join("src/lib.rs"), "pub fn b() {}\n").expect("write b");

    // Save a skill in Session A (Discovery mode)
    {
        let session_a = AllyHOrchestrator::initialize(
            make_config(&repo_a_dir, &db_dir, "default"),
            offline_router(),
            None,
        )
        .await
        .expect("init A");

        let (saved, created) = session_a
            .save_skill(
                "add-lancedb-table",
                "Steps to register a new LanceDB table",
                "SECRET_STEP_42: Update schema.rs and store.rs together.",
                vec!["new lancedb table".into()],
            )
            .await
            .expect("save skill");
        assert!(created);
        assert_eq!(saved.name, "add-lancedb-table");
    }

    // Reopen Session B -> progressive disclosure
    let mut session_b = AllyHOrchestrator::initialize(
        make_config(&repo_a_dir, &db_dir, "default"),
        offline_router(),
        None,
    )
    .await
    .expect("init B");

    let snap = session_b.inspector_snapshot().await.expect("snap");
    assert_eq!(snap.skills_count, 1);

    // 1. Unrelated prompt: catalog is present, full instructions are NOT expanded
    let turn_unrelated = session_b
        .handle_prompt("Explain how error formatting works", None)
        .await
        .expect("turn unrelated");
    assert!(turn_unrelated.preamble.text.contains("## Available Skills (LanceDB)"));
    assert!(turn_unrelated.preamble.text.contains("`add-lancedb-table`"));
    assert!(!turn_unrelated.preamble.text.contains("SECRET_STEP_42"));

    // 2. Trigger-matching prompt: full instructions ARE expanded automatically
    let turn_matched = session_b
        .handle_prompt("We need a new lancedb table for telemetry", None)
        .await
        .expect("turn matched");
    assert!(turn_matched.preamble.text.contains("## Active Skill Instructions"));
    assert!(turn_matched.preamble.text.contains("SECRET_STEP_42"));

    // 3. Pinning expands instructions even for unrelated prompts
    let (_, pinned) = session_b
        .toggle_pin_skill("add-lancedb-table")
        .await
        .expect("pin")
        .expect("skill exists");
    assert!(pinned);
    let turn_pinned = session_b
        .handle_prompt("Say hello", None)
        .await
        .expect("turn pinned");
    assert!(turn_pinned.preamble.text.contains("SECRET_STEP_42"));

    // 4. Separate repo stays isolated
    let mut session_repo_b = AllyHOrchestrator::initialize(
        make_config(&repo_b_dir, &db_dir, "default"),
        offline_router(),
        None,
    )
    .await
    .expect("init repo B");
    assert_eq!(session_repo_b.inspector_snapshot().await.expect("snap b").skills_count, 0);
    let turn_b = session_repo_b
        .handle_prompt("We need a new lancedb table", None)
        .await
        .expect("turn b");
    assert!(!turn_b.preamble.text.contains("add-lancedb-table"));

    // 5. Actor commands round-trip
    let (cmd_tx, mut ui_rx) = session_b.spawn_actor();
    // Consume initial snapshot
    let _ = ui_rx.recv().await;

    cmd_tx.send(UserCommand::RequestSkills).unwrap();
    let ev = ui_rx.recv().await.expect("skills list event");
    assert!(matches!(
        ev,
        UiEvent::SkillsList { ref skills, ref pinned_ids }
            if skills.len() == 1 && pinned_ids.contains(&skills[0].id)
    ));

    cmd_tx.send(UserCommand::ShowSkill("add-lancedb-table".into())).unwrap();
    let ev = ui_rx.recv().await.expect("skill detail event");
    assert!(matches!(
        ev,
        UiEvent::SkillDetail { ref skill, pinned: true } if skill.name == "add-lancedb-table"
    ));

    cmd_tx.send(UserCommand::RemoveSkill("add-lancedb-table".into())).unwrap();
    let notice = ui_rx.recv().await.expect("remove notice");
    assert!(matches!(notice, UiEvent::SystemNotice(msg) if msg.contains("Removed skill")));
    let updated_snap = ui_rx.recv().await.expect("updated snap");
    assert!(matches!(updated_snap, UiEvent::InspectorUpdated(s) if s.skills_count == 0));

    cmd_tx.send(UserCommand::Shutdown).unwrap();
}

/// Milestone 1 Polyglot Acceptance Suite:
/// Verifies end-to-end indexing, RepoGraph ego-graphs, monorepo toolchain detection,
/// out-of-tree `RepoProfile` custom verification gating `TaskStatus::Verified`,
/// and backward-compatible `HarnessSpec` / `CompletionInterlock` deserialization.
#[tokio::test]
async fn polyglot_monorepo_acceptance_suite() {
    use allyh_core::{CompletionInterlock, CompletionInterlockPolicy, HarnessSpec, InterlockDecision};
    use allyh_memory::{RepoProfile, TaskStatus};
    use allyh_tools::{EditByHashArgs, InspectSymbolArgs, RunVerificationArgs};

    let tmp = TempDir::new().expect("tempdir");
    let db_dir = tmp.path().join("db");
    let repo_dir = tmp.path().join("poly_monorepo");

    std::fs::create_dir_all(repo_dir.join("backend/src")).unwrap();
    std::fs::create_dir_all(repo_dir.join("web/src")).unwrap();
    std::fs::create_dir_all(repo_dir.join("ml")).unwrap();
    std::fs::create_dir_all(repo_dir.join("svc")).unwrap();

    std::fs::write(
        repo_dir.join("backend/Cargo.toml"),
        "[package]\nname = \"backend\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    std::fs::write(
        repo_dir.join("backend/src/lib.rs"),
        "pub fn health_check() -> bool { true }\n",
    )
    .unwrap();
    std::fs::write(repo_dir.join("web/tsconfig.json"), "{}").unwrap();
    std::fs::write(
        repo_dir.join("web/src/client.ts"),
        "export function fetchStatus(): string { return 'ok'; }\n",
    )
    .unwrap();
    std::fs::write(
        repo_dir.join("web/src/App.tsx"),
        "import { fetchStatus } from './client';\nexport function StatusBanner() { return <span>{fetchStatus()}</span>; }\n",
    )
    .unwrap();
    std::fs::write(
        repo_dir.join("ml/pyproject.toml"),
        "[project]\nname = \"ml\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    std::fs::write(
        repo_dir.join("ml/score.py"),
        "def compute_score(x: int) -> int:\n    return normalize(x)\n\ndef normalize(v: int) -> int:\n    return v * 2\n",
    )
    .unwrap();
    std::fs::write(repo_dir.join("svc/go.mod"), "module example.com/svc\n").unwrap();
    std::fs::write(
        repo_dir.join("svc/main.go"),
        "package main\n\ntype Engine struct{}\nfunc (e *Engine) Start() int { return boot() }\nfunc boot() int { return 1 }\n",
    )
    .unwrap();

    let mut cfg = make_config(&repo_dir, &db_dir, "default");
    cfg.auto_verify = true;
    let mut orch = AllyHOrchestrator::initialize(cfg, offline_router(), None)
        .await
        .expect("init polyglot orchestrator");

    // 1. Inspector detects all four languages & toolchains automatically
    let snap = orch.inspector_snapshot().await.expect("snap");
    for lang in ["Rust", "TypeScript", "Python", "Go"] {
        assert!(
            snap.detected_languages.iter().any(|l| l == lang),
            "expected language {lang} in {:?}",
            snap.detected_languages
        );
    }
    assert_eq!(snap.detected_toolchains.len(), 4);

    // 2. Inspect polyglot symbols with accurate 1-hop ego-graphs and language tags
    let tsx_inspect = orch
        .toolset()
        .inspect_tool()
        .execute(InspectSymbolArgs {
            query: "StatusBanner".into(),
            hops: Some(1),
            start_line: None,
            end_line: None,
        })
        .await
        .expect("inspect tsx");
    assert_eq!(tsx_inspect.matched_files[0].language, "tsx");
    assert!(
        tsx_inspect
            .ego_graph
            .hop1_neighbors
            .iter()
            .any(|e| e.to_symbol == "fetchStatus")
    );

    let py_inspect = orch
        .toolset()
        .inspect_tool()
        .execute(InspectSymbolArgs {
            query: "compute_score".into(),
            hops: Some(1),
            start_line: None,
            end_line: None,
        })
        .await
        .expect("inspect py");
    assert_eq!(py_inspect.matched_files[0].language, "python");
    assert!(
        py_inspect
            .ego_graph
            .hop1_neighbors
            .iter()
            .any(|e| e.to_symbol == "normalize")
    );

    let go_inspect = orch
        .toolset()
        .inspect_tool()
        .execute(InspectSymbolArgs {
            query: "Engine.Start".into(),
            hops: Some(1),
            start_line: None,
            end_line: None,
        })
        .await
        .expect("inspect go");
    assert_eq!(go_inspect.matched_files[0].language, "go");
    assert!(
        go_inspect
            .ego_graph
            .hop1_neighbors
            .iter()
            .any(|e| e.to_symbol == "boot")
    );

    // 3. Configure out-of-tree RepoProfile custom verifier (no dotfiles written into repo!)
    let pass_cmd = "echo verified-ok";
    let mut profile = RepoProfile::empty(orch.repo_id(), orch.scope_id());
    profile.check_command = Some(pass_cmd.into());
    profile.test_command = Some(pass_cmd.into());
    orch.save_repo_profile(profile).await.expect("save profile");
    assert!(!repo_dir.join(".allyh.toml").exists());

    // 4. Approve a plan, edit a TypeScript file with auto-verify, and verify the LanceDB task
    orch.propose_plan(AllyHOrchestrator::build_quick_plan("Update web client"));
    let tasks = orch.approve_plan().await.expect("approve plan");
    let task_id = tasks[0].id.clone();

    let client_inspect = orch
        .toolset()
        .inspect_tool()
        .execute(InspectSymbolArgs {
            query: "web/src/client.ts".into(),
            hops: Some(1),
            start_line: None,
            end_line: None,
        })
        .await
        .expect("inspect client.ts");
    let hash = client_inspect.matched_files[0].sha256_hash.clone();

    let edit_out = orch
        .toolset()
        .edit_tool()
        .execute(EditByHashArgs {
            file_path: "web/src/client.ts".into(),
            expected_sha256: Some(hash),
            target_snippet: Some("'ok'".into()),
            replacement_snippet: Some("'healthy'".into()),
            hunks: None,
            full_content: None,
            create_if_missing: None,
            target_scope: None,
            toolchain: None,
        })
        .await
        .expect("edit web/src/client.ts");
    assert!(
        edit_out
            .verification
            .as_ref()
            .is_some_and(|v| v.success && v.toolchain == "custom")
    );

    let verify_out = orch
        .toolset()
        .verify_tool()
        .execute(RunVerificationArgs {
            kind: Some("test".into()),
            target: None,
            test_filter: None,
            toolchain: None,
            affected_file: Some("web/src/client.ts".into()),
            task_id: Some(task_id.clone()),
        })
        .await
        .expect("run_verification");
    assert_eq!(verify_out.verified_task_id.as_deref(), Some(task_id.as_str()));

    let updated_tasks = orch
        .store()
        .list_tasks(orch.repo_id(), orch.scope_id())
        .await
        .expect("list tasks");
    assert_eq!(updated_tasks[0].status, TaskStatus::Verified);

    // 5. Legacy HarnessSpec JSON (`auto_cargo_check`) + CompletionInterlock (`verification` payload)
    let mut legacy_json = serde_json::to_value(HarnessSpec::default_seed()).unwrap();
    legacy_json["tools"].as_object_mut().unwrap().remove("auto_verify");
    legacy_json["tools"]["auto_cargo_check"] = serde_json::json!(false);
    let parsed_legacy: HarnessSpec =
        serde_json::from_value(legacy_json).expect("legacy auto_cargo_check alias must deserialize");
    assert!(!parsed_legacy.tools.auto_verify);

    let mut interlock = CompletionInterlock::new(CompletionInterlockPolicy {
        enabled: true,
        max_consecutive_rejections: 2,
        require_verification_after_edits: true,
        reject_on_recent_tool_error: true,
        quality_audit_notice: None,
        budget_reminder_ratio: None,
    });
    interlock.observe_tool_round(&[allyh_tools::ToolExecutionRecord {
        tool_name: "edit_by_hash".into(),
        compact_header: "edit_by_hash(web/src/client.ts) -> custom check: PASS".into(),
        succeeded: true,
        output_text: String::new(),
        output_json: serde_json::json!({ "verification": { "success": true } }),
    }]);
    assert_eq!(interlock.check_completion("Done"), InterlockDecision::Allow);
}

#[tokio::test]
async fn test_atomic_plan_checkpoint_diff_and_rollback() {
    use allyh_tools::{EditByHashArgs, InspectSymbolArgs};

    let tmp = TempDir::new().expect("tempdir");
    let db_dir = tmp.path().join("db");
    let repo_dir = tmp.path().join("repo");
    std::fs::create_dir_all(repo_dir.join("src")).unwrap();
    let lib_path = repo_dir.join("src/lib.rs");
    let orig_code = "pub fn answer() -> i32 { 42 }\n";
    std::fs::write(&lib_path, orig_code).unwrap();

    let mut orch = AllyHOrchestrator::initialize(
        make_config(&repo_dir, &db_dir, "default"),
        offline_router(),
        None,
    )
    .await
    .expect("init");

    // Approving a plan automatically captures an out-of-tree checkpoint.
    orch.propose_plan(AllyHOrchestrator::build_quick_plan("Refactor answer"));
    orch.approve_plan().await.expect("approve");
    let snap = orch.inspector_snapshot().await.expect("snap");
    assert_eq!(snap.active_checkpoint_label.as_deref(), Some("Refactor answer"));

    // Modify existing file AND create a new file.
    let inspected = orch
        .toolset()
        .inspect_tool()
        .execute(InspectSymbolArgs {
            query: "src/lib.rs".into(),
            hops: Some(1),
            start_line: None,
            end_line: None,
        })
        .await
        .expect("inspect");
    orch.toolset()
        .edit_tool()
        .execute(EditByHashArgs {
            file_path: "src/lib.rs".into(),
            expected_sha256: Some(inspected.matched_files[0].sha256_hash.clone()),
            target_snippet: Some("42".into()),
            replacement_snippet: Some("99".into()),
            hunks: None,
            full_content: None,
            create_if_missing: None,
            target_scope: None,
            toolchain: None,
        })
        .await
        .expect("edit existing");
    orch.toolset()
        .edit_tool()
        .execute(EditByHashArgs {
            file_path: "src/extra.rs".into(),
            expected_sha256: None,
            target_snippet: None,
            replacement_snippet: None,
            hunks: None,
            full_content: Some("pub fn extra() {}\n".into()),
            create_if_missing: Some(true),
            target_scope: None,
            toolchain: None,
        })
        .await
        .expect("create extra");

    let diff = orch.checkpoint_diff().expect("checkpoint diff");
    assert!(diff.has_changes());
    assert_eq!(diff.modified_files, vec!["src/lib.rs"]);
    assert_eq!(diff.created_files, vec!["src/extra.rs"]);
    assert!(diff.unified_diff.contains("+pub fn answer() -> i32 { 99 }"));

    // Rollback restores modified file byte-for-byte AND removes newly created file.
    let rollback = orch
        .rollback_checkpoint()
        .await
        .expect("rollback")
        .expect("rollback report");
    assert_eq!(rollback.restored_files, vec!["src/lib.rs"]);
    assert_eq!(rollback.removed_files, vec!["src/extra.rs"]);
    assert_eq!(std::fs::read_to_string(&lib_path).unwrap(), orig_code);
    assert!(!repo_dir.join("src/extra.rs").exists());
    assert!(!orch.checkpoint_diff().unwrap().has_changes());
}

