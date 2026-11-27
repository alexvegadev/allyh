use std::sync::Arc;

use allyh_env::{
    EnvironmentMonitor, VerificationKind, VerificationReport, VerificationTarget,
};
use allyh_memory::{MemoryStore, TaskStatus};
use rig_core::completion::ToolAnnotations;
use rig_core::tool::{Tool, ToolContext, ToolExecutionError, args_schema};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::ToolError;

/// Arguments for [`RunVerificationTool`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct RunVerificationArgs {
    /// Verification kind: `"check"`, `"lint"`, `"clippy"`, or `"test"` (defaults to `"check"`).
    pub kind: Option<String>,
    /// Optional workspace package, subdirectory, or test scope (alias: `package`).
    #[serde(default, alias = "package")]
    pub target: Option<String>,
    /// Optional targeted test name, file, or module filter (for `cargo test`, `pytest`, `vitest`, `go test`).
    #[serde(default)]
    pub test_filter: Option<String>,
    /// Optional explicit toolchain override (`rust`, `typescript`, `python`, `go`, `custom`).
    #[serde(default)]
    pub toolchain: Option<String>,
    /// Optional repository-relative file path to route verification to its enclosing sub-project.
    #[serde(default)]
    pub affected_file: Option<String>,
    /// Optional LanceDB task ID to transition to `verified` upon passing verification.
    pub task_id: Option<String>,
}

/// Structured result of running automated workspace verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunVerificationOutput {
    pub report: VerificationReport,
    pub prescriptive_diagnostics: String,
    pub verified_task_id: Option<String>,
}

/// Runs language-aware automated verification (`check`, `lint`/`clippy`, or `test`) across
/// Rust, TypeScript/JS, Python, Go, or custom `RepoProfile` commands, and optionally advances
/// a LanceDB task to `verified` with a timestamped [`allyh_memory::VerificationProof`].
#[derive(Clone)]
pub struct RunVerificationTool {
    env: Arc<EnvironmentMonitor>,
    store: MemoryStore,
}

impl RunVerificationTool {
    pub fn new(env: Arc<EnvironmentMonitor>, store: MemoryStore) -> Self {
        Self { env, store }
    }

    pub async fn execute(
        &self,
        args: RunVerificationArgs,
    ) -> Result<RunVerificationOutput, ToolError> {
        let raw_kind = args.kind.as_deref().unwrap_or("check");
        let kind = VerificationKind::parse(raw_kind).ok_or_else(|| {
            ToolError::InvalidArgs(format!(
                "unknown verification kind '{raw_kind}' (expected check, lint, clippy, or test)"
            ))
        })?;
        let target = VerificationTarget {
            kind,
            target_scope: args.target.clone(),
            test_filter: args.test_filter.clone(),
            affected_file: args.affected_file.clone(),
            toolchain_override: args.toolchain.clone(),
        };
        let report = self.env.run_verification(&target).await?;
        let prescriptive_diagnostics = report.render_prescriptive();

        let mut verified_task_id = None;
        if let Some(task_id) = args.task_id.as_deref() {
            if let Some(proof) = report.to_verification_proof() {
                self.store
                    .transition_task_status(
                        self.env.repo_id(),
                        self.env.scope_id(),
                        task_id,
                        TaskStatus::Verified,
                        Some(&proof),
                        None,
                    )
                    .await?;
                verified_task_id = Some(task_id.to_string());
            } else {
                let _ = self
                    .store
                    .transition_task_status(
                        self.env.repo_id(),
                        self.env.scope_id(),
                        task_id,
                        TaskStatus::InProgress,
                        None,
                        Some(prescriptive_diagnostics.clone()),
                    )
                    .await;
            }
        }

        Ok(RunVerificationOutput {
            report,
            prescriptive_diagnostics,
            verified_task_id,
        })
    }
}

impl Tool for RunVerificationTool {
    const NAME: &'static str = "run_verification";
    type Args = RunVerificationArgs;
    type Output = RunVerificationOutput;
    type Error = ToolError;

    fn description(&self) -> String {
        "Run automated verification (`check`, `lint`/`clippy`, or `test`) using the detected project toolchain (Rust, TypeScript/JS, Python, Go, or custom RepoProfile command) with structured diagnostics, and optionally mark a LanceDB task `verified` when passing.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<RunVerificationArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_title("Run Automated Verification")
            .with_read_only(true)
            .with_open_world(false)
    }

    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        ToolExecutionError::other(error.to_string())
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.execute(args).await
    }
}
