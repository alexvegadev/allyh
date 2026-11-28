use std::sync::Arc;

use allyh_env::{EnvironmentMonitor, GitBlameLine, GitDiffReport, GitLogEntry, GitState};
use rig_core::completion::ToolAnnotations;
use rig_core::tool::{Tool, ToolContext, ToolExecutionError, args_schema};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::ToolError;

/// Arguments for [`GitInspectTool`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct GitInspectArgs {
    /// Read-only git inspection action: `"status"`, `"log"`, `"diff"`, or `"blame"`.
    pub action: String,
    /// Optional repository-relative file or directory path filter (required for `"blame"`).
    #[serde(default)]
    pub path: Option<String>,
    /// Maximum commit count for `"log"` (defaults to 10, max 50).
    #[serde(default)]
    pub limit: Option<usize>,
    /// For `"diff"`: inspect staged index changes instead of working-tree changes.
    #[serde(default)]
    pub staged: Option<bool>,
    /// For `"blame"`: 1-based start line (defaults to 1).
    #[serde(default)]
    pub start_line: Option<usize>,
    /// For `"blame"`: 1-based end line (defaults to `start_line + 40`).
    #[serde(default)]
    pub end_line: Option<usize>,
}

/// Structured output returned by [`GitInspectTool`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitInspectOutput {
    pub action: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<GitState>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub log: Vec<GitLogEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diff: Option<GitDiffReport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blame: Vec<GitBlameLine>,
    pub rendered: String,
}

/// Read-only structured Git inspection ACI tool (`status`, `log`, `diff`, `blame`).
#[derive(Clone)]
pub struct GitInspectTool {
    env: Arc<EnvironmentMonitor>,
}

impl GitInspectTool {
    pub fn new(env: Arc<EnvironmentMonitor>) -> Self {
        Self { env }
    }

    pub async fn execute(&self, args: GitInspectArgs) -> Result<GitInspectOutput, ToolError> {
        let action = args.action.trim().to_ascii_lowercase();
        let safe_path = match args.path.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            Some(p) => Some(self.env.resolve_safe_path(p)?.0),
            None => None,
        };

        match action.as_str() {
            "status" => {
                let st = self.env.git().inspect()?;
                let rendered = format!(
                    "Branch: {} (HEAD: {})\nDirty: {}\nUntracked: {}",
                    st.branch,
                    st.head_commit.as_deref().unwrap_or("none"),
                    if st.dirty_files.is_empty() {
                        "none".to_string()
                    } else {
                        st.dirty_files.join(", ")
                    },
                    if st.untracked_files.is_empty() {
                        "none".to_string()
                    } else {
                        st.untracked_files.join(", ")
                    },
                );
                Ok(GitInspectOutput {
                    action,
                    status: Some(st),
                    log: Vec::new(),
                    diff: None,
                    blame: Vec::new(),
                    rendered,
                })
            }
            "log" => {
                let limit = args.limit.unwrap_or(10).clamp(1, 50);
                let entries = self.env.git().log(limit, safe_path.as_deref())?;
                let rendered = if entries.is_empty() {
                    "No commits found.".to_string()
                } else {
                    entries
                        .iter()
                        .map(|e| format!("{} | {} | {} | {}", e.short_commit, e.timestamp, e.author, e.summary))
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                Ok(GitInspectOutput {
                    action,
                    status: None,
                    log: entries,
                    diff: None,
                    blame: Vec::new(),
                    rendered,
                })
            }
            "diff" => {
                let staged = args.staged.unwrap_or(false);
                let diff = self.env.git().diff(staged, safe_path.as_deref(), 8000)?;
                let rendered = if diff.files_changed == 0 {
                    if staged {
                        "No staged changes.".to_string()
                    } else {
                        "No working-tree diff.".to_string()
                    }
                } else {
                    format!(
                        "{} file(s) changed, +{} -{}\n{}",
                        diff.files_changed, diff.insertions, diff.deletions, diff.patch
                    )
                };
                Ok(GitInspectOutput {
                    action,
                    status: None,
                    log: Vec::new(),
                    diff: Some(diff),
                    blame: Vec::new(),
                    rendered,
                })
            }
            "blame" => {
                let Some(rel_path) = safe_path else {
                    return Err(ToolError::InvalidArgs(
                        "git_inspect action 'blame' requires a relative 'path'".to_string(),
                    ));
                };
                let start = args.start_line.unwrap_or(1).max(1);
                let end = args.end_line.unwrap_or(start + 40);
                let lines = self.env.git().blame(&rel_path, start, end)?;
                let rendered = lines
                    .iter()
                    .map(|l| format!("{:>4} | {} ({}) | {}", l.line, l.commit, l.author, l.content))
                    .collect::<Vec<_>>()
                    .join("\n");
                Ok(GitInspectOutput {
                    action,
                    status: None,
                    log: Vec::new(),
                    diff: None,
                    blame: lines,
                    rendered,
                })
            }
            other => Err(ToolError::InvalidArgs(format!(
                "unknown git_inspect action '{other}' (expected status, log, diff, or blame)"
            ))),
        }
    }
}

impl Tool for GitInspectTool {
    const NAME: &'static str = "git_inspect";
    type Args = GitInspectArgs;
    type Output = GitInspectOutput;
    type Error = ToolError;

    fn description(&self) -> String {
        "Read-only Git inspection tool supporting `status`, `log`, `diff` (working tree or staged), and `blame` without invoking a shell.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<GitInspectArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_title("Inspect Git History, Diff & Blame")
            .with_read_only(true)
            .with_open_world(false)
    }

    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        match error {
            ToolError::InvalidArgs(msg) => ToolExecutionError::invalid_args(msg),
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
