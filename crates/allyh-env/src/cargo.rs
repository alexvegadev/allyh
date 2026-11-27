use std::path::{Path, PathBuf};

use crate::error::Result;
use crate::path::strip_verbatim_prefix;
use crate::verify::rust::RustVerificationAdapter;
use crate::verify::{VerificationAdapter, VerificationTarget};

pub use crate::verify::rust::parse_cargo_json_diagnostics;
pub use crate::verify::{
    CargoVerificationReport, CompilerDiagnostic, VerificationKind, VerificationReport,
};

/// Backward-compatible Rust-specific runner delegating to [`RustVerificationAdapter`].
#[derive(Debug, Clone)]
pub struct CargoRunner {
    repo_root: PathBuf,
}

impl CargoRunner {
    pub fn new(repo_root: impl Into<PathBuf>) -> Self {
        Self {
            repo_root: strip_verbatim_prefix(&repo_root.into()),
        }
    }

    pub fn has_cargo_manifest(&self) -> bool {
        self.repo_root.join("Cargo.toml").exists()
    }

    pub async fn run(
        &self,
        kind: VerificationKind,
        package: Option<&str>,
    ) -> Result<CargoVerificationReport> {
        let target = VerificationTarget {
            kind,
            target_scope: package.map(str::to_string),
            affected_file: None,
            toolchain_override: Some("rust".to_string()),
        };
        RustVerificationAdapter
            .run(&self.repo_root, ".", &target)
            .await
    }

    pub fn repo_root(&self) -> &Path {
        &self.repo_root
    }
}
