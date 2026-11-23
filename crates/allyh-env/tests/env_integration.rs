use std::sync::Arc;

use allyh_env::{
    EnvironmentMonitor, RustIndexer, SymbolKind, parse_cargo_json_diagnostics,
};
use allyh_memory::{
    DEFAULT_SCOPE_ID, DeterministicHashEmbedder, MemoryStore, StorageConfig, compute_repo_id,
};
use tempfile::TempDir;

#[test]
fn test_rust_indexer_and_ego_graph() {
    let src_a = r#"
use std::sync::Arc;

pub struct Store {
    pub count: usize,
}

impl Store {
    pub fn consolidate(&self) -> usize {
        helper_rank(self.count)
    }
}
"#;
    let src_b = r#"
pub fn helper_rank(n: usize) -> usize {
    n + 1
}

pub fn run_pipeline(store: &Store) -> usize {
    store.consolidate()
}
"#;

    let graph_a = RustIndexer::index_source(src_a).expect("parse a");
    assert!(graph_a.symbols.iter().any(|s| s.name == "Store" && s.kind == SymbolKind::Struct));
    assert!(
        graph_a
            .symbols
            .iter()
            .any(|s| s.name == "Store::consolidate" && s.kind == SymbolKind::Function)
    );
    assert!(graph_a.dependencies.iter().any(|d| d.contains("std::sync::Arc")));

    let graph_b = RustIndexer::index_source(src_b).expect("parse b");
    assert!(graph_b.symbols.iter().any(|s| s.name == "helper_rank"));
    assert!(graph_b.symbols.iter().any(|s| s.name == "run_pipeline"));
}

#[test]
fn test_parse_cargo_json_diagnostics() {
    let sample = r#"{"reason":"compiler-message","message":{"level":"error","code":{"code":"E0308"},"message":"mismatched types","spans":[{"file_name":"src/lib.rs","line_start":12,"column_start":5,"is_primary":true,"suggested_replacement":"42_u64"}],"children":[]}}"#;
    let diags = parse_cargo_json_diagnostics(sample);
    assert_eq!(diags.len(), 1);
    assert_eq!(diags[0].code.as_deref(), Some("E0308"));
    assert_eq!(diags[0].file.as_deref(), Some("src/lib.rs"));
    assert_eq!(diags[0].line, Some(12));
    assert!(diags[0].suggestion.as_deref().unwrap().contains("42_u64"));
}

#[tokio::test]
async fn test_out_of_band_edit_detection_and_reindexing() {
    let tmp = TempDir::new().expect("tmp");
    let repo_dir = tmp.path().join("repo");
    let db_dir = tmp.path().join("db");
    std::fs::create_dir_all(repo_dir.join("src")).expect("mkdir");

    let file_path = repo_dir.join("src/lib.rs");
    std::fs::write(&file_path, "pub fn alpha() -> i32 { 1 }\n").expect("write v1");

    let store = MemoryStore::connect(
        StorageConfig::from_local_path(&db_dir),
        Arc::new(DeterministicHashEmbedder),
    )
    .await
    .expect("connect");

    let repo_id = compute_repo_id(&repo_dir.to_string_lossy());
    let monitor = EnvironmentMonitor::new(&repo_dir, repo_id, DEFAULT_SCOPE_ID, store);

    // Initial scan indexes src/lib.rs with no out-of-band edit flagged
    let delta1 = monitor.sync_and_collect_delta(1, false).await.expect("delta1");
    assert!(delta1.out_of_band_edits.is_empty());
    assert!(delta1.reindexed_files.contains(&"src/lib.rs".to_string()));

    // Simulate manual external user edit
    std::fs::write(
        &file_path,
        "pub fn alpha() -> i32 { 2 }\npub fn beta() -> i32 { alpha() }\n",
    )
    .expect("write v2");

    let delta2 = monitor.sync_and_collect_delta(2, false).await.expect("delta2");
    assert_eq!(delta2.out_of_band_edits.len(), 1);
    assert_eq!(delta2.out_of_band_edits[0].file_path, "src/lib.rs");
    assert!(delta2.out_of_band_edits[0].diff_summary.contains("beta"));

    // RepoGraph ego-graph now knows about `beta` -> `alpha`
    let ego = monitor.ego_graph_for("beta", 2).await.expect("ego");
    assert!(!ego.center_definitions.is_empty());
    assert!(ego.hop1_neighbors.iter().any(|e| e.to_symbol == "alpha"));
}
