//! AllyH environment watcher, git tracker, polyglot tree-sitter RepoGraph,
//! and pluggable multi-language toolchain verifier.

pub mod cargo;
pub mod error;
pub mod git;
pub mod graph;
pub mod lang;
pub mod path;
pub mod verify;
pub mod watcher;

pub use cargo::{
    CargoRunner, CargoVerificationReport, CompilerDiagnostic, VerificationKind,
    parse_cargo_json_diagnostics,
};
pub use error::{EnvError, Result};
pub use git::{GitState, GitTracker};
pub use graph::{
    EgoEdge, EgoGraph, EgoNode, FileGraph, RepoGraph, RustIndexer, SymbolInfo, SymbolKind,
};
pub use lang::{LanguageIndexer, LanguageRegistry, SourceLanguage};
pub use path::{canonicalize_normalized, path_to_slash_string, strip_verbatim_prefix};
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
    EnvDelta, EnvironmentMonitor, FsChangeEvent, LiveFsWatcher, OutOfBandEdit, normalize_rel_path,
    summarize_text_diff,
};
