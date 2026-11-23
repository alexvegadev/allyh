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
        auto_cargo_check: false,
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

