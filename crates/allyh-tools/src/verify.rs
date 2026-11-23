use std::sync::Arc;

use allyh_env::{CargoVerificationReport, EnvironmentMonitor, VerificationKind};
use allyh_memory::{MemoryStore, TaskStatus};
use rig_core::completion::ToolAnnotations;
use rig_core::tool::{Tool, ToolContext, ToolExecutionError, args_schema};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::ToolError;

/// Arguments for [`RunVerificationTool`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct RunVerificationArgs {
    /// Verification command kind: `"check"`, `"clippy"`, or `"test"` (defaults to `"check"`).
    pub kind: Option<String>,
    /// Optional workspace package (`-p <package>`).
    pub package: Option<String>,
    /// Optional LanceDB task ID to transition to `verified` upon passing verification.
    pub task_id: Option<String>,
}

/// Structured result of running automated workspace verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunVerificationOutput {
    pub report: CargoVerificationReport,
    pub prescriptive_diagnostics: String,
    pub verified_task_id: Option<String>,
}

/// Runs `cargo check` / `clippy` / `test` and optionally advances a LanceDB task to `verified`
/// with cryptographic/timestamped [`allyh_memory::VerificationProof`].
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
        let kind = match args
            .kind
            .as_deref()
            .unwrap_or("check")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "check" | "" => VerificationKind::Check,
            "clippy" => VerificationKind::Clippy,
            "test" => VerificationKind::Test,
            other => {
                return Err(ToolError::InvalidArgs(format!(
                    "unknown verification kind '{other}' (expected check, clippy, or test)"
                )));
            }
        };
        let report = self.env.run_cargo(kind, args.package.as_deref()).await?;
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
        "Run automated verification (`cargo check`, `cargo clippy`, or `cargo test`) with structured JSON diagnostics, and optionally mark a LanceDB task `verified` when passing.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<RunVerificationArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_title("Run Cargo Verification")
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
