//! AllyH environment watcher, git tracker, tree-sitter RepoGraph, and cargo diagnostics runner.

pub mod cargo;
pub mod error;
pub mod git;
pub mod graph;
pub mod watcher;

pub use cargo::{
    CargoRunner, CargoVerificationReport, CompilerDiagnostic, VerificationKind,
    parse_cargo_json_diagnostics,
};
pub use error::{EnvError, Result};
pub use git::{GitState, GitTracker};
pub use graph::{EgoEdge, EgoGraph, EgoNode, FileGraph, RustIndexer, SymbolInfo, SymbolKind};
pub use watcher::{
    EnvDelta, EnvironmentMonitor, FsChangeEvent, LiveFsWatcher, OutOfBandEdit, normalize_rel_path,
    summarize_text_diff,
};
