use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use allyh_env::EnvironmentMonitor;
use allyh_memory::{
    DEFAULT_SCOPE_ID, DeterministicHashEmbedder, MemoryStore, StorageConfig, compute_repo_id,
};
use allyh_tools::{AciToolset, EditByHashArgs, InspectSymbolArgs, ToolError};
use tempfile::TempDir;

#[tokio::test]
async fn test_aci_write_gate_and_stale_hash_rejection() {
    let tmp = TempDir::new().expect("tmp");
    let repo_dir = tmp.path().join("repo");
    let db_dir = tmp.path().join("db");
    std::fs::create_dir_all(repo_dir.join("src")).expect("mkdir");
    std::fs::write(
        repo_dir.join("src/lib.rs"),
        "pub fn greet() -> &'static str { \"hello\" }\n",
    )
    .expect("write");

    let store = MemoryStore::connect(
        StorageConfig::from_local_path(&db_dir),
        Arc::new(DeterministicHashEmbedder),
    )
    .await
    .expect("connect");

    let repo_id = compute_repo_id(&repo_dir.to_string_lossy());
    let env = Arc::new(EnvironmentMonitor::new(
        &repo_dir,
        repo_id,
        DEFAULT_SCOPE_ID,
        store.clone(),
    ));
    env.sync_and_collect_delta(1, false).await.expect("sync");

    let writes_enabled = Arc::new(AtomicBool::new(false));
    let toolset = AciToolset::new(env, store, writes_enabled, false);

    // 1. Inspect symbol works in read-only Discovery mode
    let inspected = toolset
        .inspect_tool()
        .execute(InspectSymbolArgs {
            query: "greet".into(),
            hops: Some(1),
            start_line: None,
            end_line: None,
        })
        .await
        .expect("inspect");
    assert_eq!(inspected.matched_files.len(), 1);
    let initial_hash = inspected.matched_files[0].sha256_hash.clone();

    // 2. Edit is refused while Control Gate is locked
    let blocked = toolset
        .edit_tool()
        .execute(EditByHashArgs {
            file_path: "src/lib.rs".into(),
            expected_sha256: Some(initial_hash.clone()),
            target_snippet: Some("\"hello\"".into()),
            replacement_snippet: Some("\"world\"".into()),
            hunks: None,
            full_content: None,
            create_if_missing: None,
            target_scope: None,
            toolchain: None,
        })
        .await;
    assert!(matches!(blocked, Err(ToolError::WriteGateLocked(_))));

    // 3. Unlock gate, verify stale hash is rejected
    toolset.set_writes_enabled(true);
    let stale = toolset
        .edit_tool()
        .execute(EditByHashArgs {
            file_path: "src/lib.rs".into(),
            expected_sha256: Some("0000000000000000".into()),
            target_snippet: Some("\"hello\"".into()),
            replacement_snippet: Some("\"world\"".into()),
            hunks: None,
            full_content: None,
            create_if_missing: None,
            target_scope: None,
            toolchain: None,
        })
        .await;
    assert!(matches!(stale, Err(ToolError::StaleHash { .. })));

    // 4. Matching hash succeeds and updates file + hash
    let ok = toolset
        .edit_tool()
        .execute(EditByHashArgs {
            file_path: "src/lib.rs".into(),
            expected_sha256: Some(initial_hash.clone()),
            target_snippet: Some("\"hello\"".into()),
            replacement_snippet: Some("\"world\"".into()),
            hunks: None,
            full_content: None,
            create_if_missing: None,
            target_scope: None,
            toolchain: None,
        })
        .await
        .expect("edit succeeded");
    assert_ne!(ok.new_sha256, initial_hash);
    assert!(ok.diff_summary.contains("world"));
}

#[tokio::test]
async fn test_edit_by_hash_preserves_windows_crlf_with_lf_snippets() {
    let tmp = TempDir::new().expect("tmp");
    let repo_dir = tmp.path().join("repo");
    let db_dir = tmp.path().join("db");
    std::fs::create_dir_all(repo_dir.join("src")).expect("mkdir");
    let lib_path = repo_dir.join("src").join("lib.rs");
    std::fs::write(&lib_path, "pub fn a() -> i32 {\r\n    1\r\n}\r\n").expect("write crlf");

    let store = MemoryStore::connect(
        StorageConfig::from_local_path(&db_dir),
        Arc::new(DeterministicHashEmbedder),
    )
    .await
    .expect("connect");

    let env = Arc::new(EnvironmentMonitor::new(
        &repo_dir,
        compute_repo_id(&repo_dir.to_string_lossy()),
        DEFAULT_SCOPE_ID,
        store.clone(),
    ));
    env.sync_and_collect_delta(1, false).await.expect("sync");

    let writes_enabled = Arc::new(AtomicBool::new(true));
    let toolset = AciToolset::new(env, store, writes_enabled, false);

    let inspected = toolset
        .inspect_tool()
        .execute(InspectSymbolArgs {
            query: "a".into(),
            hops: Some(1),
            start_line: None,
            end_line: None,
        })
        .await
        .expect("inspect");
    let hash = inspected.matched_files[0].sha256_hash.clone();

    // Model sends LF-only target/replacement snippets for a CRLF file on disk
    toolset
        .edit_tool()
        .execute(EditByHashArgs {
            file_path: "src/lib.rs".into(),
            expected_sha256: Some(hash),
            target_snippet: Some("pub fn a() -> i32 {\n    1\n}".into()),
            replacement_snippet: Some("pub fn a() -> i32 {\n    2\n}".into()),
            hunks: None,
            full_content: None,
            create_if_missing: None,
            target_scope: None,
            toolchain: None,
        })
        .await
        .expect("crlf edit should succeed");

    let updated = std::fs::read_to_string(&lib_path).expect("read updated");
    assert_eq!(updated, "pub fn a() -> i32 {\r\n    2\r\n}\r\n");
}

#[tokio::test]
async fn test_skill_tools_work_in_discovery_mode_and_dispatch() {
    let tmp = TempDir::new().expect("tmp");
    let repo_dir = tmp.path().join("repo");
    let db_dir = tmp.path().join("db");
    std::fs::create_dir_all(repo_dir.join("src")).expect("mkdir");
    std::fs::write(repo_dir.join("src/lib.rs"), "pub fn ok() -> bool { true }\n").expect("write");

    let store = MemoryStore::connect(
        StorageConfig::from_local_path(&db_dir),
        Arc::new(DeterministicHashEmbedder),
    )
    .await
    .expect("connect");

    let env = Arc::new(EnvironmentMonitor::new(
        &repo_dir,
        compute_repo_id(&repo_dir.to_string_lossy()),
        DEFAULT_SCOPE_ID,
        store.clone(),
    ));

    // Discovery mode: writes_enabled = false
    let writes_enabled = Arc::new(AtomicBool::new(false));
    let toolset = AciToolset::new(env, store, writes_enabled, false);

    let defs = toolset.definitions();
    assert!(defs.iter().any(|d| d.name.as_str() == "add_skill"));
    assert!(defs.iter().any(|d| d.name.as_str() == "get_skill"));

    // add_skill is allowed in Discovery mode (out-of-tree LanceDB write)
    let (add_rec, _) = toolset
        .dispatch(
            "add_skill",
            serde_json::json!({
                "name": "release-checklist",
                "description": "Verify version and changelog before tagging",
                "instructions": "1. Bump Cargo.toml.\n2. Run cargo test --workspace.",
                "triggers": ["release", "tag version"]
            }),
        )
        .await;
    assert!(add_rec.succeeded, "add_skill should succeed in Discovery mode: {}", add_rec.output_text);
    assert!(add_rec.compact_header.contains("created"));

    // get_skill loads full instructions on demand in Discovery mode
    let (get_rec, _) = toolset
        .dispatch(
            "get_skill",
            serde_json::json!({ "name": "release-checklist" }),
        )
        .await;
    assert!(get_rec.succeeded);
    assert!(get_rec.output_text.contains("Bump Cargo.toml."));
    assert!(get_rec.output_text.contains("tag version"));

    // Unknown skill gives a helpful error listing available skills
    let (missing_rec, _) = toolset
        .dispatch("get_skill", serde_json::json!({ "name": "nope" }))
        .await;
    assert!(!missing_rec.succeeded);
    assert!(missing_rec.output_text.contains("release-checklist"));
}

#[tokio::test]
async fn test_inspect_sim_tool_reads_ledger_and_resolves_aliases() {
    use allyh_memory::{SimAttemptInput, SimLedger, SimMemoryMode, SimProblemEntry, SimProblemRef};

    let tmp = TempDir::new().expect("tmp");
    let repo_dir = tmp.path().join("repo");
    let db_dir = tmp.path().join("db");
    std::fs::create_dir_all(repo_dir.join("src")).expect("mkdir");
    std::fs::write(repo_dir.join("src/lib.rs"), "pub fn sim() {}\n").expect("write");

    let store = MemoryStore::connect(
        StorageConfig::from_local_path(&db_dir),
        Arc::new(DeterministicHashEmbedder),
    )
    .await
    .expect("connect");

    let repo_id = compute_repo_id(&repo_dir.to_string_lossy());
    let env = Arc::new(EnvironmentMonitor::new(
        &repo_dir,
        repo_id.clone(),
        DEFAULT_SCOPE_ID,
        store.clone(),
    ));
    let toolset = AciToolset::new(env, store.clone(), Arc::new(AtomicBool::new(false)), false);

    assert!(
        toolset
            .definitions()
            .iter()
            .any(|d| d.name.as_str() == "inspect_sim")
    );

    let mut ledger = SimLedger::new_source(
        &repo_id,
        DEFAULT_SCOPE_ID,
        "source-arm",
        "coding",
        "bench-1",
        "offline",
        "offline",
        "hv-0",
        SimMemoryMode::FullSim,
    );
    ledger
        .record_accepted_attempt(SimAttemptInput {
            problem_ref: SimProblemRef::New("premature stop".into()),
            round: 1,
            sample: "s1".into(),
            accepted: true,
            backfilled: false,
            what: "Enable completion interlock".into(),
            files: vec!["interlock.rs".into()],
            predicted: vec!["t1".into()],
            fixed: vec!["t1".into()],
            missed: vec![],
            passed: 1,
            total: 1,
            score: 100.0,
            domain: "coding".into(),
            benchmark: "bench-1".into(),
            policy_model: "offline".into(),
            harness_version_id: "hv-1".into(),
        })
        .expect("record");
    ledger.problems.push(SimProblemEntry {
        id: "P6".into(),
        name: "premature stop".into(),
        opened_round: 2,
        origin_domain: "coding".into(),
        origin_run_id: ledger.run.run_id.clone(),
        attempts: vec![],
    });
    ledger.merge_identical_problems();
    store.save_sim_ledger(&ledger).await.expect("save ledger");

    // Full ledger inspection in Discovery mode
    let (full_rec, _) = toolset.dispatch("inspect_sim", serde_json::json!({})).await;
    assert!(full_rec.succeeded);
    assert!(full_rec.output_text.contains("Enable completion interlock"));

    // Alias lookup (`P6` -> `P1`)
    let (alias_rec, _) = toolset
        .dispatch("inspect_sim", serde_json::json!({ "problem_id": "P6" }))
        .await;
    assert!(alias_rec.succeeded);
    assert!(alias_rec.output_text.contains("Resolved alias `P6` -> `P1`"));
}

#[tokio::test]
async fn test_search_code_hybrid_regex_and_semantic_ranking() {
    use allyh_tools::SearchCodeArgs;

    let tmp = TempDir::new().expect("tmp");
    let repo_dir = tmp.path().join("repo");
    let db_dir = tmp.path().join("db");
    std::fs::create_dir_all(repo_dir.join("src")).unwrap();
    std::fs::create_dir_all(repo_dir.join("ignored_cache")).unwrap();
    std::fs::write(repo_dir.join(".gitignore"), "ignored_cache/\n").unwrap();
    std::fs::write(
        repo_dir.join("src/auth.rs"),
        "pub fn verify_jwt_token(token: &str) -> bool { !token.is_empty() }\n",
    )
    .unwrap();
    std::fs::write(
        repo_dir.join("src/math.ts"),
        "export function addNumbers(a: number, b: number): number { return a + b; }\n",
    )
    .unwrap();
    std::fs::write(
        repo_dir.join("ignored_cache/leak.rs"),
        "pub fn verify_jwt_token() {}\n",
    )
    .unwrap();

    let store = MemoryStore::connect(
        StorageConfig::from_local_path(&db_dir),
        Arc::new(DeterministicHashEmbedder),
    )
    .await
    .expect("connect");
    let env = Arc::new(EnvironmentMonitor::new(
        &repo_dir,
        compute_repo_id(&repo_dir.to_string_lossy()),
        DEFAULT_SCOPE_ID,
        store.clone(),
    ));
    env.sync_and_collect_delta(1, false).await.expect("sync");

    let toolset = AciToolset::new(env, store, Arc::new(AtomicBool::new(false)), false);

    // Hybrid search finds exact match + LanceDB semantic hit while respecting .gitignore
    let out = toolset
        .search_tool()
        .execute(SearchCodeArgs {
            query: "verify_jwt_token".into(),
            mode: Some("hybrid".into()),
            path_prefix: None,
            file_glob: None,
            case_sensitive: Some(false),
            max_results: Some(10),
        })
        .await
        .expect("hybrid search");
    assert_eq!(out.text_matches.len(), 1);
    assert_eq!(out.text_matches[0].file_path, "src/auth.rs");
    assert!(!out.semantic_hits.is_empty());
    assert_eq!(out.semantic_hits[0].file_path, "src/auth.rs");

    // Regex search with file_glob filter
    let rx = toolset
        .search_tool()
        .execute(SearchCodeArgs {
            query: r"function\s+add\w+".into(),
            mode: Some("regex".into()),
            path_prefix: None,
            file_glob: Some("*.ts".into()),
            case_sensitive: Some(true),
            max_results: Some(10),
        })
        .await
        .expect("regex search");
    assert_eq!(rx.text_matches.len(), 1);
    assert_eq!(rx.text_matches[0].file_path, "src/math.ts");
}

#[tokio::test]
async fn test_edit_by_hash_atomic_multi_hunk_and_rollback_on_mismatch() {
    use allyh_tools::EditHunk;

    let tmp = TempDir::new().expect("tmp");
    let repo_dir = tmp.path().join("repo");
    let db_dir = tmp.path().join("db");
    std::fs::create_dir_all(repo_dir.join("src")).unwrap();
    let file = repo_dir.join("src/lib.rs");
    let v1 = "pub const A: i32 = 1;\npub const B: i32 = 2;\npub const C: i32 = 3;\n";
    std::fs::write(&file, v1).unwrap();

    let store = MemoryStore::connect(
        StorageConfig::from_local_path(&db_dir),
        Arc::new(DeterministicHashEmbedder),
    )
    .await
    .expect("connect");
    let env = Arc::new(EnvironmentMonitor::new(
        &repo_dir,
        compute_repo_id(&repo_dir.to_string_lossy()),
        DEFAULT_SCOPE_ID,
        store.clone(),
    ));
    env.sync_and_collect_delta(1, false).await.expect("sync");
    let toolset = AciToolset::new(env, store, Arc::new(AtomicBool::new(true)), false);
    let hash1 = EnvironmentMonitor::sha256_str(v1);

    // 1. Failing hunk #2 must abort atomically — hunk #1 MUST NOT be written to disk!
    let err = toolset
        .edit_tool()
        .execute(EditByHashArgs {
            file_path: "src/lib.rs".into(),
            expected_sha256: Some(hash1.clone()),
            target_snippet: None,
            replacement_snippet: None,
            hunks: Some(vec![
                EditHunk {
                    target_snippet: "A: i32 = 1".into(),
                    replacement_snippet: "A: i32 = 10".into(),
                },
                EditHunk {
                    target_snippet: "NON_EXISTENT_SNIPPET".into(),
                    replacement_snippet: "BOOM".into(),
                },
            ]),
            full_content: None,
            create_if_missing: None,
            target_scope: None,
            toolchain: None,
        })
        .await;
    assert!(matches!(err, Err(ToolError::EditTargetNotFound { .. })));
    assert_eq!(std::fs::read_to_string(&file).unwrap(), v1);

    // 2. All-matching multi-hunk batch applies atomically in one call
    let ok = toolset
        .edit_tool()
        .execute(EditByHashArgs {
            file_path: "src/lib.rs".into(),
            expected_sha256: Some(hash1),
            target_snippet: None,
            replacement_snippet: None,
            hunks: Some(vec![
                EditHunk {
                    target_snippet: "A: i32 = 1".into(),
                    replacement_snippet: "A: i32 = 10".into(),
                },
                EditHunk {
                    target_snippet: "C: i32 = 3".into(),
                    replacement_snippet: "C: i32 = 30".into(),
                },
            ]),
            full_content: None,
            create_if_missing: None,
            target_scope: None,
            toolchain: None,
        })
        .await
        .expect("atomic multi-hunk edit");
    assert!(ok.diff_summary.contains("30"));
    let updated = std::fs::read_to_string(&file).unwrap();
    assert!(updated.contains("A: i32 = 10") && updated.contains("C: i32 = 30"));
}

#[tokio::test]
async fn test_run_command_policy_gate_and_git_inspect() {
    use allyh_tools::{GitInspectArgs, RunCommandArgs};

    let tmp = TempDir::new().expect("tmp");
    let repo_dir = tmp.path().join("repo");
    let db_dir = tmp.path().join("db");
    std::fs::create_dir_all(repo_dir.join("src")).unwrap();
    std::fs::write(repo_dir.join("src/main.rs"), "fn main() { println!(\"v1\"); }\n").unwrap();

    // Initialize a real git repo for git_inspect
    let init_status = std::process::Command::new("git")
        .args(["init"])
        .current_dir(&repo_dir)
        .status()
        .expect("git init");
    assert!(init_status.success());
    let _ = std::process::Command::new("git")
        .args(["config", "user.email", "test@allyh.dev"])
        .current_dir(&repo_dir)
        .status();
    let _ = std::process::Command::new("git")
        .args(["config", "user.name", "AllyH Tester"])
        .current_dir(&repo_dir)
        .status();
    let _ = std::process::Command::new("git")
        .args(["add", "."])
        .current_dir(&repo_dir)
        .status();
    let _ = std::process::Command::new("git")
        .args(["commit", "-m", "initial commit"])
        .current_dir(&repo_dir)
        .status();

    // Make a working-tree edit so diff has content
    std::fs::write(repo_dir.join("src/main.rs"), "fn main() { println!(\"v2\"); }\n").unwrap();

    let store = MemoryStore::connect(
        StorageConfig::from_local_path(&db_dir),
        Arc::new(DeterministicHashEmbedder),
    )
    .await
    .expect("connect");
    let env = Arc::new(EnvironmentMonitor::new(
        &repo_dir,
        compute_repo_id(&repo_dir.to_string_lossy()),
        DEFAULT_SCOPE_ID,
        store.clone(),
    ));
    let toolset = AciToolset::new(env, store, Arc::new(AtomicBool::new(false)), false);

    // 1. git_inspect log, diff, blame
    let log_out = toolset
        .git_tool()
        .execute(GitInspectArgs {
            action: "log".into(),
            path: None,
            limit: Some(5),
            staged: None,
            start_line: None,
            end_line: None,
        })
        .await
        .expect("git log");
    assert_eq!(log_out.log.len(), 1);
    assert_eq!(log_out.log[0].summary, "initial commit");

    let diff_out = toolset
        .git_tool()
        .execute(GitInspectArgs {
            action: "diff".into(),
            path: Some("src/main.rs".into()),
            limit: None,
            staged: Some(false),
            start_line: None,
            end_line: None,
        })
        .await
        .expect("git diff");
    assert_eq!(diff_out.diff.as_ref().unwrap().files_changed, 1);
    assert!(diff_out.rendered.contains("v2"));

    let blame_out = toolset
        .git_tool()
        .execute(GitInspectArgs {
            action: "blame".into(),
            path: Some("src/main.rs".into()),
            limit: None,
            staged: None,
            start_line: Some(1),
            end_line: Some(1),
        })
        .await
        .expect("git blame");
    assert_eq!(blame_out.blame.len(), 1);

    // 2. run_command Policy Gate: read-only command succeeds immediately
    let safe = toolset
        .command_tool()
        .execute(RunCommandArgs {
            command: "echo safe_readonly_ok".into(),
            working_dir: None,
            timeout_secs: Some(10),
            max_output_chars: None,
        })
        .await
        .expect("readonly command allowed");
    assert!(safe.success);
    assert!(safe.stdout.contains("safe_readonly_ok"));

    // 3. Chained / mutating command is blocked until approved
    let chained = "echo one && echo two";
    let (blocked_rec, _) = toolset
        .dispatch("run_command", serde_json::json!({ "command": chained }))
        .await;
    assert!(!blocked_rec.succeeded);
    assert_eq!(blocked_rec.output_json["approval_required"], true);

    // 4. Approve once -> succeeds once, then re-locks
    toolset.command_policy().approve_once(chained);
    let (once_rec, _) = toolset
        .dispatch("run_command", serde_json::json!({ "command": chained }))
        .await;
    assert!(once_rec.succeeded);
    assert!(once_rec.output_text.contains("two"));

    let (relocked_rec, _) = toolset
        .dispatch("run_command", serde_json::json!({ "command": chained }))
        .await;
    assert!(!relocked_rec.succeeded);
    assert_eq!(relocked_rec.output_json["approval_required"], true);
}
