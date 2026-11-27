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
            full_content: None,
            create_if_missing: None,
            cargo_package: None,
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
            full_content: None,
            create_if_missing: None,
            cargo_package: None,
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
            full_content: None,
            create_if_missing: None,
            cargo_package: None,
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
            full_content: None,
            create_if_missing: None,
            cargo_package: None,
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
