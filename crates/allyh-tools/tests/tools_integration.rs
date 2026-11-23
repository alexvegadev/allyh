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
