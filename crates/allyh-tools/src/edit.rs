use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use allyh_env::{
    EnvironmentMonitor, VerificationKind, VerificationReport, VerificationTarget,
    summarize_text_diff,
};
use rig_core::completion::ToolAnnotations;
use rig_core::tool::{Tool, ToolContext, ToolExecutionError, args_schema};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::ToolError;

/// Arguments for [`EditByHashTool`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct EditByHashArgs {
    /// Repository-relative path to modify or create.
    pub file_path: String,
    /// Required SHA-256 hex digest of the file last inspected (leave `None` only when `create_if_missing` is `true` for a new file).
    pub expected_sha256: Option<String>,
    /// Exact substring to replace in the existing file.
    pub target_snippet: Option<String>,
    /// Replacement text for `target_snippet`.
    pub replacement_snippet: Option<String>,
    /// Full replacement file contents (used when creating a file or replacing the entire file).
    pub full_content: Option<String>,
    /// Allow creating a new file when it does not exist yet.
    pub create_if_missing: Option<bool>,
    /// Optional package, subdirectory, or test scope for faster post-edit verification (alias: `cargo_package`).
    #[serde(default, alias = "cargo_package")]
    pub target_scope: Option<String>,
    /// Optional explicit toolchain override (`rust`, `typescript`, `python`, `go`, `custom`).
    #[serde(default)]
    pub toolchain: Option<String>,
}

/// Prescriptive output returned by [`EditByHashTool`], including post-edit diagnostics from
/// the affected sub-project's toolchain verifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditByHashOutput {
    pub file_path: String,
    pub previous_sha256: Option<String>,
    pub new_sha256: String,
    pub diff_summary: String,
    #[serde(alias = "cargo_check")]
    pub verification: Option<VerificationReport>,
    pub prescriptive_diagnostics: String,
}

/// Hash-verified file editor with automatic post-edit verification hook across supported toolchains.
#[derive(Clone)]
pub struct EditByHashTool {
    env: Arc<EnvironmentMonitor>,
    /// Enforces the Control Gate: edits are rejected while AllyH is in Discovery / PlanApproval mode.
    writes_enabled: Arc<AtomicBool>,
    /// Automatically run post-edit check verification on the affected sub-project.
    auto_verify: bool,
}

impl EditByHashTool {
    pub fn new(
        env: Arc<EnvironmentMonitor>,
        writes_enabled: Arc<AtomicBool>,
        auto_verify: bool,
    ) -> Self {
        Self {
            env,
            writes_enabled,
            auto_verify,
        }
    }

    pub async fn execute(&self, args: EditByHashArgs) -> Result<EditByHashOutput, ToolError> {
        if !self.writes_enabled.load(Ordering::SeqCst) {
            return Err(ToolError::WriteGateLocked(
                "Plan approval gate is active; approve the plan before modifying files."
                    .to_string(),
            ));
        }

        let (rel_path, abs_path) = self.env.resolve_safe_path(&args.file_path)?;
        let exists = abs_path.exists();

        let (previous_content, previous_sha256) = if exists {
            let content = std::fs::read_to_string(&abs_path)?;
            let actual_hash = EnvironmentMonitor::sha256_str(&content);
            let expected = args.expected_sha256.as_deref().ok_or_else(|| {
                ToolError::InvalidArgs(format!(
                    "expected_sha256 is required when editing existing file '{rel_path}' (current hash: {actual_hash})"
                ))
            })?;
            if expected.trim() != actual_hash {
                return Err(ToolError::StaleHash {
                    file_path: rel_path,
                    expected: expected.to_string(),
                    actual: actual_hash,
                });
            }
            (content, Some(actual_hash))
        } else {
            if !args.create_if_missing.unwrap_or(false) {
                return Err(ToolError::EditTargetNotFound {
                    file_path: rel_path,
                    reason: "file does not exist and create_if_missing is false".to_string(),
                });
            }
            (String::new(), None)
        };

        let updated_content = if let Some(full) = args.full_content {
            full
        } else if let (Some(target), Some(replacement)) = (
            args.target_snippet.as_deref(),
            args.replacement_snippet.as_deref(),
        ) {
            if previous_content.contains(target) {
                previous_content.replacen(target, replacement, 1)
            } else if previous_content.contains("\r\n") {
                // Preserve Windows CRLF files when models emit LF-only snippets from `inspect_symbol`.
                let crlf_target = to_crlf(target);
                if !previous_content.contains(&crlf_target) {
                    return Err(ToolError::EditTargetNotFound {
                        file_path: rel_path,
                        reason: "target_snippet was not found verbatim in file".to_string(),
                    });
                }
                let crlf_replacement = to_crlf(replacement);
                previous_content.replacen(&crlf_target, &crlf_replacement, 1)
            } else {
                return Err(ToolError::EditTargetNotFound {
                    file_path: rel_path,
                    reason: "target_snippet was not found verbatim in file".to_string(),
                });
            }
        } else {
            return Err(ToolError::InvalidArgs(
                "provide either full_content or both target_snippet and replacement_snippet"
                    .to_string(),
            ));
        };

        if let Some(parent) = abs_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&abs_path, &updated_content)?;

        let new_sha256 = EnvironmentMonitor::sha256_str(&updated_content);
        let diff_summary = summarize_text_diff(&previous_content, &updated_content);

        // Immediately refresh snapshot & cache so this authorized edit is NOT mistaken for out-of-band drift.
        self.env.cache_file_content(&rel_path, &updated_content);
        let _ = self.env.index_file(&rel_path, 0).await;

        let profile = self.env.load_repo_profile().await;
        let should_auto_verify = profile
            .as_ref()
            .and_then(|p| p.auto_verify_override)
            .unwrap_or(self.auto_verify);
        let has_custom_check = profile
            .as_ref()
            .and_then(|p| p.check_command.as_deref())
            .is_some_and(|c| !c.trim().is_empty());

        let verification =
            if should_auto_verify && (has_custom_check || self.env.has_any_toolchain()) {
                let target = VerificationTarget {
                    kind: VerificationKind::Check,
                    target_scope: args.target_scope.clone(),
                    affected_file: Some(rel_path.clone()),
                    toolchain_override: args.toolchain.clone(),
                };
                Some(self.env.run_verification(&target).await?)
            } else {
                None
            };

        let prescriptive_diagnostics = verification
            .as_ref()
            .map(VerificationReport::render_prescriptive)
            .unwrap_or_else(|| {
                "verification skipped (no detected toolchain or auto-check disabled)".to_string()
            });

        Ok(EditByHashOutput {
            file_path: rel_path,
            previous_sha256,
            new_sha256,
            diff_summary,
            verification,
            prescriptive_diagnostics,
        })
    }
}

impl Tool for EditByHashTool {
    const NAME: &'static str = "edit_by_hash";
    type Args = EditByHashArgs;
    type Output = EditByHashOutput;
    type Error = ToolError;

    fn description(&self) -> String {
        "Edit or create a workspace file only if its SHA-256 hash matches expected_sha256, then automatically run the affected sub-project's type-check/compiler verification and return prescriptive diagnostics.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<EditByHashArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_title("Hash-Verified File Edit + Auto Check")
            .with_read_only(false)
            .with_destructive(true)
            .with_open_world(false)
    }

    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        match error {
            ToolError::WriteGateLocked(msg) => ToolExecutionError::refused(msg),
            ToolError::StaleHash { .. } | ToolError::InvalidArgs(_) => {
                ToolExecutionError::invalid_args(error.to_string())
            }
            other => ToolExecutionError::other(other.to_string()),
        }
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.execute(args).await
    }
}

fn to_crlf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\n', "\r\n")
}
