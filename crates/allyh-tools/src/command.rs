use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use allyh_env::EnvironmentMonitor;
use rig_core::completion::ToolAnnotations;
use rig_core::tool::{Tool, ToolContext, ToolExecutionError, args_schema};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::process::Command;

use crate::error::ToolError;

/// Default shell execution timeout (seconds).
const DEFAULT_COMMAND_TIMEOUT_SECS: u64 = 30;
/// Hard ceiling for command execution timeout (seconds).
const MAX_COMMAND_TIMEOUT_SECS: u64 = 120;
/// Default combined output character cap before head+tail truncation.
const DEFAULT_MAX_OUTPUT_CHARS: usize = 8_000;

/// Pending shell command awaiting explicit user approval in the TUI Command Approval Gate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingCommandApproval {
    pub command: String,
    pub working_dir: String,
    pub reason: String,
    pub suggested_pattern: String,
}

/// Classification produced by [`CommandPolicyGate::evaluate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandPolicyDecision {
    AllowReadOnly,
    AllowPattern(String),
    AllowOnce,
    RequireApproval {
        reason: String,
        suggested_pattern: String,
    },
}

/// Thread-safe command policy gate controlling which shell commands [`RunCommandTool`] may execute.
///
/// Read-only inspection commands (without shell chaining/redirection metacharacters) are allowed
/// automatically. Mutating, chained, or network commands require explicit approval via
/// `approve_once` (`[Allow Once]`) or `allow_pattern` (`[Allow Pattern]`).
#[derive(Debug, Default)]
pub struct CommandPolicyGate {
    allowed_patterns: Mutex<Vec<String>>,
    one_shot_approvals: Mutex<HashSet<String>>,
}

impl CommandPolicyGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Approve one exact command string for a single execution.
    pub fn approve_once(&self, command: &str) {
        if let Ok(mut guard) = self.one_shot_approvals.lock() {
            guard.insert(command.trim().to_string());
        }
    }

    /// Allow all future commands starting with `pattern` (prefix match, e.g. `"cargo test"` or `"pytest"`).
    pub fn allow_pattern(&self, pattern: &str) {
        let norm = pattern.trim().to_string();
        if norm.is_empty() {
            return;
        }
        if let Ok(mut guard) = self.allowed_patterns.lock()
            && !guard.iter().any(|p| p.eq_ignore_ascii_case(&norm))
        {
            guard.push(norm);
        }
    }

    pub fn list_allowed_patterns(&self) -> Vec<String> {
        self.allowed_patterns
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    /// Evaluate policy and consume any matching one-shot approval.
    pub fn evaluate_and_consume(&self, command: &str) -> CommandPolicyDecision {
        let trimmed = command.trim();
        if let Ok(mut once) = self.one_shot_approvals.lock()
            && once.remove(trimmed)
        {
            return CommandPolicyDecision::AllowOnce;
        }

        if let Ok(patterns) = self.allowed_patterns.lock() {
            for pat in patterns.iter() {
                if command_matches_pattern(trimmed, pat) {
                    return CommandPolicyDecision::AllowPattern(pat.clone());
                }
            }
        }

        if is_safe_readonly_command(trimmed) {
            return CommandPolicyDecision::AllowReadOnly;
        }

        let suggested_pattern = suggest_pattern(trimmed);
        let reason = if contains_shell_metachars(trimmed) {
            "Command uses shell chaining, piping, or redirection and requires explicit approval."
                .to_string()
        } else {
            "Command may mutate files, run arbitrary code, or access the network and requires explicit approval."
                .to_string()
        };
        CommandPolicyDecision::RequireApproval {
            reason,
            suggested_pattern,
        }
    }
}

/// Arguments for [`RunCommandTool`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct RunCommandArgs {
    /// Command line to execute inside the workspace.
    pub command: String,
    /// Optional repository-relative working directory (defaults to repository root).
    #[serde(default)]
    pub working_dir: Option<String>,
    /// Timeout in seconds (defaults to 30, max 120).
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    /// Maximum output characters retained (head + tail truncation; defaults to 8000).
    #[serde(default)]
    pub max_output_chars: Option<usize>,
}

/// Structured output returned by [`RunCommandTool`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunCommandOutput {
    pub command: String,
    pub working_dir: String,
    pub exit_code: i32,
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub truncated: bool,
}

/// Policy-gated shell execution ACI tool with strict timeout and bounded output truncation.
#[derive(Clone)]
pub struct RunCommandTool {
    env: Arc<EnvironmentMonitor>,
    policy: Arc<CommandPolicyGate>,
}

impl RunCommandTool {
    pub fn new(env: Arc<EnvironmentMonitor>, policy: Arc<CommandPolicyGate>) -> Self {
        Self { env, policy }
    }

    pub fn policy(&self) -> &Arc<CommandPolicyGate> {
        &self.policy
    }

    pub async fn execute(&self, args: RunCommandArgs) -> Result<RunCommandOutput, ToolError> {
        let cmd_str = args.command.trim();
        if cmd_str.is_empty() {
            return Err(ToolError::InvalidArgs(
                "run_command 'command' must not be empty".to_string(),
            ));
        }

        let (rel_dir, abs_dir) = match args
            .working_dir
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty() && *s != ".")
        {
            Some(rel) => self.env.resolve_safe_path(rel)?,
            None => (".".to_string(), self.env.repo_root().to_path_buf()),
        };

        if let CommandPolicyDecision::RequireApproval { reason, .. } =
            self.policy.evaluate_and_consume(cmd_str)
        {
            return Err(ToolError::CommandApprovalRequired {
                command: cmd_str.to_string(),
                reason,
            });
        }

        let timeout_secs = args
            .timeout_secs
            .unwrap_or(DEFAULT_COMMAND_TIMEOUT_SECS)
            .clamp(1, MAX_COMMAND_TIMEOUT_SECS);
        let max_chars = args
            .max_output_chars
            .unwrap_or(DEFAULT_MAX_OUTPUT_CHARS)
            .clamp(256, 64_000);

        let mut cmd = if cfg!(windows) {
            let mut c = Command::new("cmd");
            c.args(["/C", cmd_str]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args(["-c", cmd_str]);
            c
        };
        cmd.current_dir(&abs_dir).kill_on_drop(true);

        let output = tokio::time::timeout(Duration::from_secs(timeout_secs), cmd.output())
            .await
            .map_err(|_| ToolError::CommandTimedOut {
                command: cmd_str.to_string(),
                timeout_secs,
            })??;

        let raw_stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let raw_stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let (stdout, out_trunc) = truncate_head_tail(&raw_stdout, max_chars);
        let (stderr, err_trunc) = truncate_head_tail(&raw_stderr, max_chars / 2);

        Ok(RunCommandOutput {
            command: cmd_str.to_string(),
            working_dir: rel_dir,
            exit_code: output.status.code().unwrap_or(-1),
            success: output.status.success(),
            stdout,
            stderr,
            truncated: out_trunc || err_trunc,
        })
    }
}

impl Tool for RunCommandTool {
    const NAME: &'static str = "run_command";
    type Args = RunCommandArgs;
    type Output = RunCommandOutput;
    type Error = ToolError;

    fn description(&self) -> String {
        "Execute a workspace command subject to AllyH's Command Approval Gate. Safe read-only commands run immediately; mutating, chained, or network commands require explicit user approval (`[Allow Once]` / `[Allow Pattern]`).".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<RunCommandArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_title("Policy-Gated Shell Command")
            .with_read_only(false)
            .with_destructive(true)
            .with_open_world(true)
    }

    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        match error {
            ToolError::CommandApprovalRequired { .. } => {
                ToolExecutionError::refused(error.to_string())
            }
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

/// Derive a clean prefix pattern suggestion for `[Allow Pattern]` (first 1-2 tokens).
pub fn suggest_pattern(command: &str) -> String {
    let tokens: Vec<&str> = command.split_whitespace().collect();
    match tokens.as_slice() {
        [] => String::new(),
        [only] => (*only).to_string(),
        [first, second, ..] => {
            if second.starts_with('-') {
                (*first).to_string()
            } else {
                format!("{first} {second}")
            }
        }
    }
}

fn command_matches_pattern(command: &str, pattern: &str) -> bool {
    if contains_shell_metachars(command) && !contains_shell_metachars(pattern) {
        return false;
    }
    let cmd_lower = command.to_ascii_lowercase();
    let pat_lower = pattern.trim().to_ascii_lowercase();
    cmd_lower == pat_lower || cmd_lower.starts_with(&format!("{pat_lower} "))
}

fn contains_shell_metachars(cmd: &str) -> bool {
    cmd.contains("&&")
        || cmd.contains("||")
        || cmd.contains(';')
        || cmd.contains('|')
        || cmd.contains('>')
        || cmd.contains('<')
        || cmd.contains('`')
        || cmd.contains("$(")
        || cmd.contains('\n')
}

fn is_safe_readonly_command(cmd: &str) -> bool {
    if cmd.is_empty() || contains_shell_metachars(cmd) {
        return false;
    }
    let tokens: Vec<&str> = cmd.split_whitespace().collect();
    let Some(&head) = tokens.first() else {
        return false;
    };
    let head_lower = head.to_ascii_lowercase();
    let second = tokens.get(1).map(|s| s.to_ascii_lowercase());

    match head_lower.as_str() {
        "pwd" | "ls" | "dir" | "which" | "where" | "whoami" | "uname" | "echo" | "rg" => true,
        "git" => matches!(
            second.as_deref(),
            Some("status" | "log" | "diff" | "show" | "branch" | "rev-parse" | "describe" | "tag")
        ),
        "cargo" => matches!(second.as_deref(), Some("--version" | "-v" | "metadata" | "tree")),
        "rustc" | "node" | "npm" | "python" | "python3" | "pytest" | "go" => matches!(
            second.as_deref(),
            Some("--version" | "-v" | "version" | "env")
        ),
        _ => false,
    }
}

fn truncate_head_tail(text: &str, max_chars: usize) -> (String, bool) {
    if text.len() <= max_chars || max_chars < 64 {
        return (text.to_string(), false);
    }
    let head_budget = max_chars * 2 / 3;
    let tail_budget = max_chars - head_budget;
    let head_cut = text
        .char_indices()
        .take_while(|(i, _)| *i < head_budget)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0);
    let tail_target = text.len().saturating_sub(tail_budget);
    let tail_start = text
        .char_indices()
        .find(|(i, _)| *i >= tail_target)
        .map(|(i, _)| i)
        .unwrap_or(text.len());
    if head_cut >= tail_start {
        return (text.to_string(), false);
    }
    let omitted = tail_start - head_cut;
    (
        format!(
            "{}\n... [truncated {omitted} chars] ...\n{}",
            &text[..head_cut],
            &text[tail_start..]
        ),
        true,
    )
}
