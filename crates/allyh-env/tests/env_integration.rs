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

#[tokio::test]
async fn test_cross_platform_safe_path_and_verbatim_normalization() {
    use std::path::Path;
    use allyh_env::{EnvError, strip_verbatim_prefix};

    assert_eq!(
        strip_verbatim_prefix(Path::new(r"\\?\C:\Users\demo\repo")),
        Path::new(r"C:\Users\demo\repo")
    );
    assert_eq!(
        strip_verbatim_prefix(Path::new(r"\\?\UNC\server\share\repo")),
        Path::new(r"\\server\share\repo")
    );

    let tmp = TempDir::new().expect("tmp");
    let repo_dir = tmp.path().join("repo");
    let db_dir = tmp.path().join("db");
    std::fs::create_dir_all(repo_dir.join("src")).expect("mkdir");

    let store = MemoryStore::connect(
        StorageConfig::from_local_path(&db_dir),
        Arc::new(DeterministicHashEmbedder),
    )
    .await
    .expect("connect");

    let monitor = EnvironmentMonitor::new(
        &repo_dir,
        compute_repo_id(&repo_dir.to_string_lossy()),
        DEFAULT_SCOPE_ID,
        store,
    );

    // Valid relative paths with forward slashes, backslashes, `./`, or `..` in filename
    let (rel1, _) = monitor.resolve_safe_path(r".\src\foo..bar.rs").expect("valid path");
    assert_eq!(rel1, "src/foo..bar.rs");

    let (rel2, _) = monitor.resolve_safe_path("src/./lib.rs").expect("valid curdir");
    assert_eq!(rel2, "src/lib.rs");

    // Cross-platform path escapes must all be rejected
    for bad in [
        "../outside.rs",
        "src/../../outside.rs",
        "/etc/passwd",
        r"\Windows\System32",
        r"C:\Windows\System32",
        "C:relative_escape.rs",
        "C:/escape.rs",
        "",
        ".",
    ] {
        assert!(
            matches!(monitor.resolve_safe_path(bad), Err(EnvError::PathEscape(_))),
            "expected PathEscape for `{bad}`"
        );
    }
}

#[test]
fn test_polyglot_tree_sitter_indexers_and_cross_file_ego_graphs() {
    use allyh_env::{LanguageRegistry, RepoGraph, SourceLanguage};
    use allyh_memory::CodeGraphSnapshot;

    let reg = LanguageRegistry::default_all();

    // 1. TypeScript / TSX
    let ts_api = r#"
import { UserRecord } from "./types";

export interface AuthService {
    verify(token: string): Promise<UserRecord>;
}

export class TokenAuthenticator implements AuthService {
    verify(token: string): Promise<UserRecord> {
        return fetchUser(token);
    }
}

export function fetchUser(token: string): Promise<UserRecord> {
    return Promise.resolve({ id: token });
}
"#;
    let tsx_view = r#"
import { TokenAuthenticator } from "./auth";

export function ProfileCard() {
    const auth = new TokenAuthenticator();
    auth.verify("abc");
    return <div>Profile</div>;
}
"#;
    let (lang_ts, g_ts) = reg.index_path("web/src/auth.ts", ts_api).expect("index ts");
    assert_eq!(lang_ts, SourceLanguage::TypeScript);
    assert!(g_ts.symbols.iter().any(|s| s.name == "AuthService" && s.kind == SymbolKind::Interface));
    assert!(g_ts.symbols.iter().any(|s| s.name == "TokenAuthenticator" && s.kind == SymbolKind::Class));
    assert!(g_ts.symbols.iter().any(|s| s.name == "TokenAuthenticator.verify" && s.kind == SymbolKind::Method));
    assert!(g_ts.dependencies.iter().any(|d| d == "./types"));

    let (lang_tsx, g_tsx) = reg.index_path("web/src/ProfileCard.tsx", tsx_view).expect("index tsx");
    assert_eq!(lang_tsx, SourceLanguage::Tsx);
    assert!(g_tsx.symbols.iter().any(|s| s.name == "ProfileCard" && s.kind == SymbolKind::Component));

    // 2. Python
    let py_src = r#"
from typing import Optional
import json

MAX_RETRIES = 3

class PipelineRunner:
    def execute(self, payload: str) -> Optional[str]:
        return transform_payload(payload)

def transform_payload(raw: str) -> str:
    return raw.strip()
"#;
    let (lang_py, g_py) = reg.index_path("ml/pipeline.py", py_src).expect("index py");
    assert_eq!(lang_py, SourceLanguage::Python);
    assert!(g_py.symbols.iter().any(|s| s.name == "PipelineRunner" && s.kind == SymbolKind::Class));
    assert!(g_py.symbols.iter().any(|s| s.name == "PipelineRunner.execute" && s.kind == SymbolKind::Method));
    assert!(g_py.symbols.iter().any(|s| s.name == "transform_payload" && s.kind == SymbolKind::Function));
    assert!(g_py.symbols.iter().any(|s| s.name == "MAX_RETRIES" && s.kind == SymbolKind::Const));
    assert!(g_py.dependencies.iter().any(|d| d == "typing"));
    assert!(g_py.dependencies.iter().any(|d| d == "json"));

    // 3. Go
    let go_src = r#"
package worker

import (
    "context"
    "fmt"
)

type JobStore interface {
    Load(ctx context.Context, id string) error
}

type Worker struct {
    Store JobStore
}

func (w *Worker) RunJob(ctx context.Context, id string) error {
    return dispatchTask(w.Store, ctx, id)
}

func dispatchTask(store JobStore, ctx context.Context, id string) error {
    return store.Load(ctx, id)
}
"#;
    let (lang_go, g_go) = reg.index_path("svc/worker.go", go_src).expect("index go");
    assert_eq!(lang_go, SourceLanguage::Go);
    assert!(g_go.symbols.iter().any(|s| s.name == "JobStore" && s.kind == SymbolKind::Interface));
    assert!(g_go.symbols.iter().any(|s| s.name == "Worker" && s.kind == SymbolKind::Struct));
    assert!(g_go.symbols.iter().any(|s| s.name == "Worker.RunJob" && s.kind == SymbolKind::Method));
    assert!(g_go.symbols.iter().any(|s| s.name == "dispatchTask" && s.kind == SymbolKind::Function));
    assert!(g_go.dependencies.iter().any(|d| d == "context"));

    // Cross-file EgoGraph resolution across TS/TSX
    let snapshots = vec![
        CodeGraphSnapshot {
            repo_id: "r".into(),
            scope_id: "s".into(),
            file_path: "web/src/auth.ts".into(),
            language: "typescript".into(),
            sha256_hash: "h1".into(),
            last_read_turn: 1,
            symbols_json: serde_json::to_string(&g_ts.symbols).unwrap(),
            dependencies: g_ts.dependencies,
            vector: vec![0.0; 384],
        },
        CodeGraphSnapshot {
            repo_id: "r".into(),
            scope_id: "s".into(),
            file_path: "web/src/ProfileCard.tsx".into(),
            language: "tsx".into(),
            sha256_hash: "h2".into(),
            last_read_turn: 1,
            symbols_json: serde_json::to_string(&g_tsx.symbols).unwrap(),
            dependencies: g_tsx.dependencies,
            vector: vec![0.0; 384],
        },
    ];
    let ego = RepoGraph::build_ego_graph(&snapshots, "ProfileCard", 2);
    assert!(!ego.center_definitions.is_empty());
    assert!(
        ego.hop1_neighbors
            .iter()
            .any(|e| e.to_symbol == "TokenAuthenticator" || e.to_symbol == "TokenAuthenticator.verify"),
        "expected ProfileCard -> TokenAuthenticator edge, got: {:?}",
        ego.hop1_neighbors
    );
}

#[test]
fn test_polyglot_diagnostic_parsers_and_monorepo_routing() {
    use allyh_env::{
        ToolchainVerifier, VerificationKind, VerificationTarget, parse_biome_json_diagnostics,
        parse_eslint_json_diagnostics, parse_generic_file_line_diagnostics,
        parse_go_test_json_diagnostics, parse_go_vet_diagnostics, parse_golangci_json_diagnostics,
        parse_mypy_json_diagnostics, parse_pyright_json_diagnostics, parse_pytest_output,
        parse_ruff_json_diagnostics, parse_sarif_diagnostics, parse_tsc_diagnostics,
    };

    // tsc
    let tsc = parse_tsc_diagnostics(
        "src/app.ts(12,5): error TS2322: Type 'number' is not assignable to type 'string'.\n\
         src/util.ts:4:2 - error TS2304: Cannot find name 'foo'.",
    );
    assert_eq!(tsc.len(), 2);
    assert_eq!(tsc[0].code.as_deref(), Some("TS2322"));
    assert_eq!(tsc[0].line, Some(12));
    assert_eq!(tsc[1].file.as_deref(), Some("src/util.ts"));

    // eslint
    let eslint = parse_eslint_json_diagnostics(
        r#"[{"filePath":"src/index.ts","messages":[{"ruleId":"no-unused-vars","severity":2,"message":"'x' is unused","line":3,"column":7}]}]"#,
    );
    assert_eq!(eslint.len(), 1);
    assert_eq!(eslint[0].level, "error");
    assert_eq!(eslint[0].code.as_deref(), Some("no-unused-vars"));

    // biome
    let biome = parse_biome_json_diagnostics(
        r#"{"diagnostics":[{"category":"lint/style/useConst","severity":"warning","description":"Use const","location":{"path":{"file":"src/a.ts"}}}]}"#,
    );
    assert_eq!(biome.len(), 1);
    assert_eq!(biome[0].level, "warning");

    // pyright
    let pyright = parse_pyright_json_diagnostics(
        r#"{"generalDiagnostics":[{"file":"app.py","severity":"error","message":"Argument of type int","rule":"reportArgumentType","range":{"start":{"line":9,"character":4}}}]}"#,
    );
    assert_eq!(pyright.len(), 1);
    assert_eq!(pyright[0].line, Some(10));
    assert_eq!(pyright[0].column, Some(5));

    // mypy
    let mypy = parse_mypy_json_diagnostics(
        r#"{"file":"ml/train.py","line":21,"column":3,"message":"Incompatible return value","code":"return-value","severity":"error"}"#,
    );
    assert_eq!(mypy.len(), 1);
    assert_eq!(mypy[0].code.as_deref(), Some("return-value"));

    // ruff
    let ruff = parse_ruff_json_diagnostics(
        r#"[{"code":"F401","message":"`os` imported but unused","filename":"ml/train.py","location":{"row":1,"column":8},"fix":{"message":"Remove unused import"}}]"#,
    );
    assert_eq!(ruff.len(), 1);
    assert_eq!(ruff[0].suggestion.as_deref(), Some("Remove unused import"));

    // pytest
    let pytest = parse_pytest_output("FAILED tests/test_math.py::test_add - AssertionError: 2 != 3\n");
    assert_eq!(pytest.len(), 1);
    assert_eq!(pytest[0].file.as_deref(), Some("tests/test_math.py"));

    // go vet / golangci / go test -json
    let govet = parse_go_vet_diagnostics("./pkg/server.go:42:9: printf format %d has arg of wrong type");
    assert_eq!(govet.len(), 1);
    assert_eq!(govet[0].file.as_deref(), Some("pkg/server.go"));
    assert_eq!(govet[0].line, Some(42));

    let golangci = parse_golangci_json_diagnostics(
        r#"{"Issues":[{"FromLinter":"errcheck","Text":"Error return value is not checked","Severity":"warning","Pos":{"Filename":"pkg/server.go","Line":15,"Column":2}}]}"#,
    );
    assert_eq!(golangci.len(), 1);
    assert_eq!(golangci[0].code.as_deref(), Some("errcheck"));

    let gotest = parse_go_test_json_diagnostics(
        "{\"Action\":\"output\",\"Package\":\"example.com/svc\",\"Test\":\"TestAdd\",\"Output\":\"expected 4 got 5\\n\"}\n\
         {\"Action\":\"fail\",\"Package\":\"example.com/svc\",\"Test\":\"TestAdd\"}\n",
    );
    assert_eq!(gotest.len(), 1);
    assert!(gotest[0].message.contains("expected 4 got 5"));

    // SARIF 2.1.0 + generic fallback
    let sarif = parse_sarif_diagnostics(
        r#"{"version":"2.1.0","runs":[{"results":[{"ruleId":"C401","level":"error","message":{"text":"Null deref"},"locations":[{"physicalLocation":{"artifactLocation":{"uri":"src/main.c"},"region":{"startLine":7,"startColumn":3}}}]}]}]}"#,
    );
    assert_eq!(sarif.len(), 1);
    assert_eq!(sarif[0].file.as_deref(), Some("src/main.c"));
    assert_eq!(sarif[0].line, Some(7));

    let generic = parse_generic_file_line_diagnostics("src/Main.java:18:4: error: ';' expected");
    assert_eq!(generic.len(), 1);
    assert_eq!(generic[0].file.as_deref(), Some("src/Main.java"));

    // Polyglot monorepo routing
    let tmp = TempDir::new().expect("tmp");
    let root = tmp.path();
    std::fs::create_dir_all(root.join("backend/src")).unwrap();
    std::fs::create_dir_all(root.join("web/src")).unwrap();
    std::fs::create_dir_all(root.join("ml")).unwrap();
    std::fs::create_dir_all(root.join("svc")).unwrap();
    std::fs::write(root.join("backend/Cargo.toml"), "[package]\nname=\"api\"\n").unwrap();
    std::fs::write(root.join("web/tsconfig.json"), "{}").unwrap();
    std::fs::write(root.join("ml/pyproject.toml"), "[project]\nname=\"ml\"\n").unwrap();
    std::fs::write(root.join("svc/go.mod"), "module example.com/svc\n").unwrap();

    let verifier = ToolchainVerifier::new(root);
    let detected = verifier.detect_toolchains();
    assert_eq!(detected.len(), 4);

    let ts_tc = verifier
        .resolve_toolchain(&VerificationTarget {
            kind: VerificationKind::Check,
            affected_file: Some("web/src/App.tsx".into()),
            ..Default::default()
        })
        .expect("ts toolchain");
    assert_eq!(ts_tc.id, "typescript");
    assert_eq!(ts_tc.root_rel, "web");

    let py_tc = verifier
        .resolve_toolchain(&VerificationTarget {
            kind: VerificationKind::Test,
            affected_file: Some("ml/train.py".into()),
            ..Default::default()
        })
        .expect("py toolchain");
    assert_eq!(py_tc.id, "python");
    assert_eq!(py_tc.root_rel, "ml");

    let go_tc = verifier
        .resolve_toolchain(&VerificationTarget {
            kind: VerificationKind::Check,
            affected_file: Some("svc/worker.go".into()),
            ..Default::default()
        })
        .expect("go toolchain");
    assert_eq!(go_tc.id, "go");
    assert_eq!(go_tc.root_rel, "svc");
}

#[tokio::test]
async fn test_gitignore_and_repo_profile_extra_ignores() {
    use allyh_memory::RepoProfile;

    let tmp = TempDir::new().expect("tmp");
    let repo_dir = tmp.path().join("repo");
    let db_dir = tmp.path().join("db");
    std::fs::create_dir_all(repo_dir.join("src")).unwrap();
    std::fs::create_dir_all(repo_dir.join("ignored_dir")).unwrap();
    std::fs::create_dir_all(repo_dir.join("generated")).unwrap();

    std::fs::write(repo_dir.join(".gitignore"), "ignored_dir/\n").unwrap();
    std::fs::write(repo_dir.join("src/app.ts"), "export function live() { return 1; }\n").unwrap();
    std::fs::write(repo_dir.join("ignored_dir/skip.ts"), "export function dead() {}\n").unwrap();
    std::fs::write(repo_dir.join("generated/gen.py"), "def auto_gen(): pass\n").unwrap();

    let store = MemoryStore::connect(
        StorageConfig::from_local_path(&db_dir),
        Arc::new(DeterministicHashEmbedder),
    )
    .await
    .expect("connect");

    let repo_id = compute_repo_id(&repo_dir.to_string_lossy());
    let mut profile = RepoProfile::empty(&repo_id, DEFAULT_SCOPE_ID);
    profile.extra_ignore_globs = vec!["generated/**".into()];
    store.upsert_repo_profile(&profile).await.expect("save profile");

    let monitor = EnvironmentMonitor::new(&repo_dir, repo_id.clone(), DEFAULT_SCOPE_ID, store.clone());
    let delta = monitor.sync_and_collect_delta(1, false).await.expect("sync");
    assert!(delta.reindexed_files.contains(&"src/app.ts".to_string()));
    assert!(!delta.reindexed_files.iter().any(|f| f.contains("ignored_dir")));
    assert!(!delta.reindexed_files.iter().any(|f| f.contains("generated")));
    assert!(delta.detected_languages.contains(&"TypeScript".to_string()));

    let snap = store
        .get_code_snapshot(&repo_id, DEFAULT_SCOPE_ID, "src/app.ts")
        .await
        .expect("get")
        .expect("exists");
    assert_eq!(snap.language, "typescript");
}
