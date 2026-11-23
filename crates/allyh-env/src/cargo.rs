use std::path::{Path, PathBuf};

use allyh_memory::VerificationProof;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::process::Command;

use crate::error::{EnvError, Result};
use crate::path::strip_verbatim_prefix;

/// Verification mode supported by [`CargoRunner`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationKind {
    Check,
    Clippy,
    Test,
}

impl VerificationKind {
    pub fn command_label(&self) -> &'static str {
        match self {
            Self::Check => "cargo check --message-format=json",
            Self::Clippy => "cargo clippy --message-format=json",
            Self::Test => "cargo test --message-format=json",
        }
    }
}

/// Structured compiler diagnostic extracted from `cargo --message-format=json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerDiagnostic {
    pub level: String,
    pub code: Option<String>,
    pub message: String,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub column: Option<u32>,
    pub suggestion: Option<String>,
}

/// Prescriptive compiler / test verification report attached to ACI tool outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CargoVerificationReport {
    pub kind: VerificationKind,
    pub command: String,
    pub success: bool,
    pub error_count: usize,
    pub warning_count: usize,
    pub diagnostics: Vec<CompilerDiagnostic>,
    pub raw_summary: String,
}

impl CargoVerificationReport {
    /// Convert a passing verification report into a [`VerificationProof`] required to mark
    /// a LanceDB task `verified`.
    pub fn to_verification_proof(&self) -> Option<VerificationProof> {
        if !self.success {
            return None;
        }
        Some(VerificationProof {
            command: self.command.clone(),
            passed_at: Utc::now(),
            warnings: self.warning_count,
            summary: self.compact_status_line(),
        })
    }

    /// Compact one-line summary for TUI Tool Cards (e.g., `cargo check: PASS (0 warnings)`).
    pub fn compact_status_line(&self) -> String {
        let tool_name = match self.kind {
            VerificationKind::Check => "cargo check",
            VerificationKind::Clippy => "cargo clippy",
            VerificationKind::Test => "cargo test",
        };
        if self.success {
            format!("{tool_name}: PASS ({} warnings)", self.warning_count)
        } else {
            format!(
                "{tool_name}: FAIL ({} errors, {} warnings)",
                self.error_count, self.warning_count
            )
        }
    }

    /// Prescriptive diagnostics formatted for model repair loops (*CoDAIM 2026*).
    pub fn render_prescriptive(&self) -> String {
        let mut lines = vec![self.compact_status_line()];
        for diag in self.diagnostics.iter().take(12) {
            let loc = match (&diag.file, diag.line, diag.column) {
                (Some(f), Some(l), Some(c)) => format!("{f}:{l}:{c}"),
                (Some(f), Some(l), None) => format!("{f}:{l}"),
                (Some(f), None, None) => f.clone(),
                _ => "<unknown>".to_string(),
            };
            let code = diag
                .code
                .as_deref()
                .map(|c| format!("[{c}] "))
                .unwrap_or_default();
            lines.push(format!("- {} {code}{} ({loc})", diag.level.to_uppercase(), diag.message));
            if let Some(hint) = &diag.suggestion {
                lines.push(format!("  -> Fix hint: {hint}"));
            }
        }
        if self.diagnostics.is_empty() && !self.success && !self.raw_summary.is_empty() {
            lines.push(self.raw_summary.clone());
        }
        lines.join("\n")
    }
}

/// Runs `cargo check`, `cargo clippy`, and `cargo test` in a target repository.
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
        if !self.has_cargo_manifest() {
            return Ok(CargoVerificationReport {
                kind,
                command: kind.command_label().to_string(),
                success: true,
                error_count: 0,
                warning_count: 0,
                diagnostics: Vec::new(),
                raw_summary: "No Cargo.toml found in workspace root; skipped cargo check."
                    .to_string(),
            });
        }

        let mut args: Vec<String> = match kind {
            VerificationKind::Check => vec!["check".into(), "--message-format=json".into()],
            VerificationKind::Clippy => vec!["clippy".into(), "--message-format=json".into()],
            VerificationKind::Test => vec!["test".into(), "--message-format=json".into()],
        };
        if let Some(pkg) = package
            && !pkg.trim().is_empty()
        {
            args.push("-p".into());
            args.push(pkg.trim().to_string());
        }

        let cargo_bin = std::env::var_os("CARGO")
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| std::ffi::OsString::from("cargo"));
        let output = Command::new(cargo_bin)
            .args(&args)
            .current_dir(&self.repo_root)
            .output()
            .await
            .map_err(|e| EnvError::Command(format!("failed to spawn cargo: {e}")))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let diagnostics = parse_cargo_json_diagnostics(&stdout);
        let error_count = diagnostics.iter().filter(|d| d.level == "error").count();
        let warning_count = diagnostics.iter().filter(|d| d.level == "warning").count();

        let success = output.status.success();
        let raw_summary = if !success && diagnostics.is_empty() {
            truncate_tail(&format!("{stdout}\n{stderr}"), 1200)
        } else {
            truncate_tail(&stderr, 600)
        };

        Ok(CargoVerificationReport {
            kind,
            command: format!("cargo {}", args.join(" ")),
            success,
            error_count,
            warning_count,
            diagnostics,
            raw_summary,
        })
    }

    pub fn repo_root(&self) -> &Path {
        &self.repo_root
    }
}

/// Parse newline-delimited JSON output from `cargo --message-format=json`.
pub fn parse_cargo_json_diagnostics(stdout: &str) -> Vec<CompilerDiagnostic> {
    let mut out = Vec::new();
    for line in stdout.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with('{') {
            continue;
        }
        let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) else {
            continue;
        };
        if val.get("reason").and_then(|v| v.as_str()) != Some("compiler-message") {
            continue;
        }
        let Some(msg) = val.get("message") else {
            continue;
        };
        let level = msg
            .get("level")
            .and_then(|v| v.as_str())
            .unwrap_or("info")
            .to_string();
        if level != "error" && level != "warning" {
            continue;
        }
        let message = msg
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let code = msg
            .get("code")
            .and_then(|c| c.get("code"))
            .and_then(|v| v.as_str())
            .map(str::to_string);

        let primary_span = msg
            .get("spans")
            .and_then(|s| s.as_array())
            .and_then(|arr| {
                arr.iter()
                    .find(|sp| sp.get("is_primary").and_then(|b| b.as_bool()) == Some(true))
                    .or_else(|| arr.first())
            });

        let file = primary_span
            .and_then(|sp| sp.get("file_name"))
            .and_then(|v| v.as_str())
            .map(|s| s.replace('\\', "/"));
        let line_no = primary_span
            .and_then(|sp| sp.get("line_start"))
            .and_then(|v| v.as_u64())
            .map(|n| n as u32);
        let col_no = primary_span
            .and_then(|sp| sp.get("column_start"))
            .and_then(|v| v.as_u64())
            .map(|n| n as u32);

        let mut suggestion = primary_span
            .and_then(|sp| sp.get("suggested_replacement"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| format!("replace with `{s}`"));

        if suggestion.is_none()
            && let Some(children) = msg.get("children").and_then(|c| c.as_array())
        {
            for child in children {
                let child_level = child.get("level").and_then(|v| v.as_str()).unwrap_or("");
                if matches!(child_level, "help" | "note")
                    && let Some(cmsg) = child.get("message").and_then(|v| v.as_str())
                {
                    suggestion = Some(cmsg.to_string());
                    break;
                }
            }
        }

        out.push(CompilerDiagnostic {
            level,
            code,
            message,
            file,
            line: line_no,
            column: col_no,
            suggestion,
        });
    }
    out
}

fn truncate_tail(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.len() <= max_chars {
        return trimmed.to_string();
    }
    let start = trimmed.len().saturating_sub(max_chars);
    let safe_start = trimmed
        .char_indices()
        .find(|(idx, _)| *idx >= start)
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    format!("...{}", &trimmed[safe_start..])
}
