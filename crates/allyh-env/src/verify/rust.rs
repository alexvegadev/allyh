use std::path::Path;

use async_trait::async_trait;
use tokio::process::Command;

use crate::error::{EnvError, Result};
use crate::verify::{
    CompilerDiagnostic, VerificationAdapter, VerificationKind, VerificationReport,
    VerificationTarget, subproject_dir, truncate_tail,
};

/// Rust verification adapter (`cargo check` / `cargo clippy` / `cargo test --message-format=json`).
pub struct RustVerificationAdapter;

#[async_trait]
impl VerificationAdapter for RustVerificationAdapter {
    fn id(&self) -> &'static str {
        "rust"
    }

    fn display_name(&self) -> &'static str {
        "Rust (cargo)"
    }

    fn manifest_names(&self) -> &'static [&'static str] {
        &["Cargo.toml"]
    }

    async fn run(
        &self,
        repo_root: &Path,
        subproject_rel: &str,
        target: &VerificationTarget,
    ) -> Result<VerificationReport> {
        let workdir = subproject_dir(repo_root, subproject_rel);
        if !workdir.join("Cargo.toml").exists() {
            return Ok(VerificationReport {
                toolchain: "cargo".to_string(),
                working_dir: subproject_rel.to_string(),
                kind: target.kind,
                command: target.kind.command_label().to_string(),
                success: true,
                error_count: 0,
                warning_count: 0,
                diagnostics: Vec::new(),
                raw_summary: "No Cargo.toml found in workspace root; skipped cargo check."
                    .to_string(),
            });
        }

        let mut args: Vec<String> = match target.kind {
            VerificationKind::Check => vec!["check".into(), "--message-format=json".into()],
            VerificationKind::Clippy | VerificationKind::Lint => {
                vec!["clippy".into(), "--message-format=json".into()]
            }
            VerificationKind::Test => vec!["test".into(), "--message-format=json".into()],
        };
        if let Some(pkg) = target.target_scope.as_deref()
            && !pkg.trim().is_empty()
            && pkg.trim() != subproject_rel
        {
            args.push("-p".into());
            args.push(pkg.trim().to_string());
        }

        let cargo_bin = std::env::var_os("CARGO")
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| std::ffi::OsString::from("cargo"));
        let output = Command::new(cargo_bin)
            .args(&args)
            .current_dir(&workdir)
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

        Ok(VerificationReport {
            toolchain: "cargo".to_string(),
            working_dir: subproject_rel.to_string(),
            kind: target.kind,
            command: format!("cargo {}", args.join(" ")),
            success,
            error_count,
            warning_count,
            diagnostics,
            raw_summary,
        })
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
