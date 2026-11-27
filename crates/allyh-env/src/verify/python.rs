use std::path::Path;

use async_trait::async_trait;
use tokio::process::Command;

use crate::error::Result;
use crate::verify::{
    CompilerDiagnostic, VerificationAdapter, VerificationKind, VerificationReport,
    VerificationTarget, subproject_dir, truncate_tail,
};

/// Python verification adapter (`pyright`/`mypy`, `ruff`, `pytest`).
pub struct PythonVerificationAdapter;

#[async_trait]
impl VerificationAdapter for PythonVerificationAdapter {
    fn id(&self) -> &'static str {
        "python"
    }

    fn display_name(&self) -> &'static str {
        "Python"
    }

    fn manifest_names(&self) -> &'static [&'static str] {
        &["pyproject.toml", "setup.py", "requirements.txt"]
    }

    async fn run(
        &self,
        repo_root: &Path,
        subproject_rel: &str,
        target: &VerificationTarget,
    ) -> Result<VerificationReport> {
        let workdir = subproject_dir(repo_root, subproject_rel);

        match target.kind {
            VerificationKind::Check => {
                // Prefer pyright --outputjson, fallback to mypy --output=json.
                if let Ok(out) = Command::new("pyright")
                    .arg("--outputjson")
                    .current_dir(&workdir)
                    .output()
                    .await
                {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    let diagnostics = parse_pyright_json_diagnostics(&stdout);
                    return Ok(build_report(
                        subproject_rel,
                        target.kind,
                        "pyright --outputjson".to_string(),
                        out.status.success(),
                        diagnostics,
                        &format!("{stdout}\n{stderr}"),
                    ));
                }
                if let Ok(out) = Command::new("mypy")
                    .args(["--output=json", "."])
                    .current_dir(&workdir)
                    .output()
                    .await
                {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    let diagnostics = parse_mypy_json_diagnostics(&stdout);
                    return Ok(build_report(
                        subproject_rel,
                        target.kind,
                        "mypy --output=json .".to_string(),
                        out.status.success(),
                        diagnostics,
                        &format!("{stdout}\n{stderr}"),
                    ));
                }
                Ok(skipped_missing_tool(
                    subproject_rel,
                    target.kind,
                    "pyright --outputjson",
                    "pyright/mypy",
                ))
            }
            VerificationKind::Lint | VerificationKind::Clippy => {
                let Ok(out) = Command::new("ruff")
                    .args(["check", "--output-format", "json", "."])
                    .current_dir(&workdir)
                    .output()
                    .await
                else {
                    return Ok(skipped_missing_tool(
                        subproject_rel,
                        target.kind,
                        "ruff check --output-format json .",
                        "ruff",
                    ));
                };
                let stdout = String::from_utf8_lossy(&out.stdout);
                let stderr = String::from_utf8_lossy(&out.stderr);
                let diagnostics = parse_ruff_json_diagnostics(&stdout);
                Ok(build_report(
                    subproject_rel,
                    target.kind,
                    "ruff check --output-format json .".to_string(),
                    out.status.success(),
                    diagnostics,
                    &format!("{stdout}\n{stderr}"),
                ))
            }
            VerificationKind::Test => {
                let mut args = vec!["-q".to_string(), "--tb=short".to_string()];
                if let Some(scope) = target.target_scope.as_deref()
                    && !scope.trim().is_empty()
                    && scope.trim() != subproject_rel
                {
                    args.push(scope.trim().to_string());
                }
                if let Some(filter) = target
                    .test_filter
                    .as_deref()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                {
                    if filter.contains('/') || filter.contains(".py") || filter.contains("::") {
                        args.push(filter.to_string());
                    } else {
                        args.push("-k".to_string());
                        args.push(filter.to_string());
                    }
                }
                let cmd_label = format!("pytest {}", args.join(" "));
                let Ok(out) = Command::new("pytest")
                    .args(&args)
                    .current_dir(&workdir)
                    .output()
                    .await
                else {
                    return Ok(skipped_missing_tool(
                        subproject_rel,
                        target.kind,
                        &cmd_label,
                        "pytest",
                    ));
                };
                let combined = format!(
                    "{}\n{}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                );
                let diagnostics = parse_pytest_output(&combined);
                Ok(build_report(
                    subproject_rel,
                    target.kind,
                    cmd_label,
                    out.status.success(),
                    diagnostics,
                    &combined,
                ))
            }
        }
    }
}

/// Parse `pyright --outputjson` diagnostic payload.
pub fn parse_pyright_json_diagnostics(stdout: &str) -> Vec<CompilerDiagnostic> {
    let Ok(root) = serde_json::from_str::<serde_json::Value>(stdout.trim()) else {
        return Vec::new();
    };
    let Some(items) = root.get("generalDiagnostics").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for item in items {
        let sev = item
            .get("severity")
            .and_then(|v| v.as_str())
            .unwrap_or("error")
            .to_ascii_lowercase();
        let level = if sev == "warning" {
            "warning"
        } else if sev == "error" {
            "error"
        } else {
            continue;
        }
        .to_string();
        let file = item
            .get("file")
            .and_then(|v| v.as_str())
            .map(|s| s.replace('\\', "/"));
        let message = item
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let code = item
            .get("rule")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let start = item.get("range").and_then(|r| r.get("start"));
        // Pyright range lines/characters are 0-based.
        let line = start
            .and_then(|s| s.get("line"))
            .and_then(|v| v.as_u64())
            .map(|n| n as u32 + 1);
        let column = start
            .and_then(|s| s.get("character"))
            .and_then(|v| v.as_u64())
            .map(|n| n as u32 + 1);

        out.push(CompilerDiagnostic {
            level,
            code,
            message,
            file,
            line,
            column,
            suggestion: None,
        });
    }
    out
}

/// Parse `mypy --output=json` newline-delimited JSON diagnostics.
pub fn parse_mypy_json_diagnostics(stdout: &str) -> Vec<CompilerDiagnostic> {
    let mut out = Vec::new();
    for line in stdout.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with('{') {
            continue;
        }
        let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) else {
            continue;
        };
        let sev = val
            .get("severity")
            .and_then(|v| v.as_str())
            .unwrap_or("error")
            .to_ascii_lowercase();
        if sev != "error" && sev != "warning" {
            continue;
        }
        out.push(CompilerDiagnostic {
            level: sev,
            code: val.get("code").and_then(|v| v.as_str()).map(str::to_string),
            message: val
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            file: val
                .get("file")
                .and_then(|v| v.as_str())
                .map(|s| s.replace('\\', "/")),
            line: val.get("line").and_then(|v| v.as_u64()).map(|n| n as u32),
            column: val.get("column").and_then(|v| v.as_u64()).map(|n| n as u32),
            suggestion: val.get("hint").and_then(|v| v.as_str()).map(str::to_string),
        });
    }
    out
}

/// Parse `ruff check --output-format json` array output.
pub fn parse_ruff_json_diagnostics(stdout: &str) -> Vec<CompilerDiagnostic> {
    let Ok(items) = serde_json::from_str::<Vec<serde_json::Value>>(stdout.trim()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for item in items {
        let code = item
            .get("code")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let message = item
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let file = item
            .get("filename")
            .and_then(|v| v.as_str())
            .map(|s| s.replace('\\', "/"));
        let loc = item.get("location");
        let line = loc
            .and_then(|l| l.get("row"))
            .and_then(|v| v.as_u64())
            .map(|n| n as u32);
        let column = loc
            .and_then(|l| l.get("column"))
            .and_then(|v| v.as_u64())
            .map(|n| n as u32);
        let suggestion = item
            .get("fix")
            .and_then(|f| f.get("message"))
            .and_then(|v| v.as_str())
            .map(str::to_string);

        out.push(CompilerDiagnostic {
            level: "warning".to_string(),
            code,
            message,
            file,
            line,
            column,
            suggestion,
        });
    }
    out
}

/// Parse `pytest --tb=short` failure lines (`FAILED path/test_mod.py::test_fn - AssertionError: ...`).
pub fn parse_pytest_output(output: &str) -> Vec<CompilerDiagnostic> {
    let mut out = Vec::new();
    for line in output.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("FAILED ") {
            let (node_id, msg) = rest
                .split_once(" - ")
                .map(|(a, b)| (a.trim(), b.trim()))
                .unwrap_or((rest.trim(), "pytest test failed"));
            let file = node_id
                .split("::")
                .next()
                .map(|s| s.replace('\\', "/"));
            out.push(CompilerDiagnostic {
                level: "error".to_string(),
                code: Some(node_id.to_string()),
                message: msg.to_string(),
                file,
                line: None,
                column: None,
                suggestion: None,
            });
        }
    }
    out
}

fn build_report(
    subproject_rel: &str,
    kind: VerificationKind,
    command: String,
    success: bool,
    diagnostics: Vec<CompilerDiagnostic>,
    combined_raw: &str,
) -> VerificationReport {
    let error_count = diagnostics.iter().filter(|d| d.level == "error").count();
    let warning_count = diagnostics.iter().filter(|d| d.level == "warning").count();
    let raw_summary = if !success && diagnostics.is_empty() {
        truncate_tail(combined_raw, 1200)
    } else {
        truncate_tail(combined_raw, 400)
    };
    VerificationReport {
        toolchain: "python".to_string(),
        working_dir: subproject_rel.to_string(),
        kind,
        command,
        success,
        error_count,
        warning_count,
        diagnostics,
        raw_summary,
    }
}

fn skipped_missing_tool(
    subproject_rel: &str,
    kind: VerificationKind,
    command: &str,
    tool_hint: &str,
) -> VerificationReport {
    VerificationReport {
        toolchain: "python".to_string(),
        working_dir: subproject_rel.to_string(),
        kind,
        command: command.to_string(),
        success: true,
        error_count: 0,
        warning_count: 0,
        diagnostics: Vec::new(),
        raw_summary: format!("Tool `{tool_hint}` not found on PATH; skipped Python verification."),
    }
}
