//! AllyH environment watcher, git tracker, polyglot tree-sitter RepoGraph,
//! and pluggable multi-language toolchain verifier.

pub mod cargo;
pub mod checkpoint;
pub mod error;
pub mod git;
pub mod graph;
pub mod lang;
pub mod lsp;
pub mod path;
pub mod search;
pub mod verify;
pub mod watcher;
pub mod worktree;

pub use cargo::{
    CargoRunner, CargoVerificationReport, CompilerDiagnostic, VerificationKind,
    parse_cargo_json_diagnostics,
};
pub use checkpoint::{CheckpointDiffReport, PlanCheckpoint, RollbackReport};
pub use error::{EnvError, Result};
pub use git::{GitBlameLine, GitDiffReport, GitLogEntry, GitState, GitTracker};
pub use graph::{
    EgoEdge, EgoGraph, EgoNode, FileGraph, RepoGraph, RustIndexer, SymbolInfo, SymbolKind,
};
pub use lang::{LanguageIndexer, LanguageRegistry, SourceLanguage};
pub use lsp::{
    DEFAULT_LSP_DIAGNOSTIC_WAIT, LspCallHierarchyItem, LspClient, LspDiagnostic, LspLocation,
    LspManager, LspPosition, LspRange, LspServerSpec, LspServerState, LspServerStatus,
    collect_lsp_diagnostics_report, encode_lsp_frame, enrich_ego_graph_with_lsp, file_uri_to_path,
    file_uri_to_rel_path, lsp_language_id_for_path, path_to_file_uri,
};
pub use path::{canonicalize_normalized, path_to_slash_string, strip_verbatim_prefix};
pub use search::{CodeSearchMatch, CodeSearchQuery, TextSearchResult, search_workspace_text};
pub use verify::custom::{parse_generic_file_line_diagnostics, parse_sarif_diagnostics};
pub use verify::go::{
    GoVerificationAdapter, parse_go_test_json_diagnostics, parse_go_vet_diagnostics,
    parse_golangci_json_diagnostics,
};
pub use verify::python::{
    PythonVerificationAdapter, parse_mypy_json_diagnostics, parse_pyright_json_diagnostics,
    parse_pytest_output, parse_ruff_json_diagnostics,
};
pub use verify::rust::RustVerificationAdapter;
pub use verify::typescript::{
    TypeScriptVerificationAdapter, parse_biome_json_diagnostics, parse_eslint_json_diagnostics,
    parse_tsc_diagnostics,
};
pub use verify::{
    DetectedToolchain, DiagnosticItem, ToolchainVerifier, VerificationAdapter, VerificationReport,
    VerificationTarget,
};
pub use watcher::{
    CrossScopeClaim, EnvDelta, EnvironmentMonitor, FsChangeEvent, LiveFsWatcher, OutOfBandEdit,
    normalize_rel_path, summarize_text_diff,
};
pub use worktree::{WorktreeInfo, WorktreeManager};
