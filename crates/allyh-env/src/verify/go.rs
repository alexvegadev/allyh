use std::path::Path;

use async_trait::async_trait;
use tokio::process::Command;

use crate::error::Result;
use crate::verify::{
    CompilerDiagnostic, VerificationAdapter, VerificationKind, VerificationReport,
    VerificationTarget, subproject_dir, truncate_tail,
};

/// Go verification adapter (`go vet`, `golangci-lint`, `go test -json`).
pub struct GoVerificationAdapter;

#[async_trait]
impl VerificationAdapter for GoVerificationAdapter {
    fn id(&self) -> &'static str {
        "go"
    }

    fn display_name(&self) -> &'static str {
        "Go"
    }

    fn manifest_names(&self) -> &'static [&'static str] {
        &["go.mod"]
    }

    async fn run(
        &self,
        repo_root: &Path,
        subproject_rel: &str,
        target: &VerificationTarget,
    ) -> Result<VerificationReport> {
        let workdir = subproject_dir(repo_root, subproject_rel);
        let pkg_scope = target
            .target_scope
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty() && *s != subproject_rel)
            .unwrap_or("./...");

        match target.kind {
            VerificationKind::Check => {
                let cmd_label = format!("go vet {pkg_scope}");
                let Ok(out) = Command::new("go")
                    .args(["vet", pkg_scope])
                    .current_dir(&workdir)
                    .output()
                    .await
                else {
                    return Ok(skipped_missing_tool(subproject_rel, target.kind, &cmd_label, "go"));
                };
                let combined = format!(
                    "{}\n{}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                );
                let diagnostics = parse_go_vet_diagnostics(&combined);
                Ok(build_report(
                    subproject_rel,
                    target.kind,
                    cmd_label,
                    out.status.success(),
                    diagnostics,
                    &combined,
                ))
            }
            VerificationKind::Lint | VerificationKind::Clippy => {
                if let Ok(out) = Command::new("golangci-lint")
                    .args(["run", "--out-format", "json", pkg_scope])
                    .current_dir(&workdir)
                    .output()
                    .await
                {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    let diagnostics = parse_golangci_json_diagnostics(&stdout);
                    return Ok(build_report(
                        subproject_rel,
                        target.kind,
                        format!("golangci-lint run --out-format json {pkg_scope}"),
                        out.status.success(),
                        diagnostics,
                        &format!("{stdout}\n{stderr}"),
                    ));
                }
                // Fallback to `go vet` if `golangci-lint` is not installed.
                let cmd_label = format!("go vet {pkg_scope}");
                let Ok(out) = Command::new("go")
                    .args(["vet", pkg_scope])
                    .current_dir(&workdir)
                    .output()
                    .await
                else {
                    return Ok(skipped_missing_tool(
                        subproject_rel,
                        target.kind,
                        &cmd_label,
                        "golangci-lint/go",
                    ));
                };
                let combined = format!(
                    "{}\n{}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                );
                let diagnostics = parse_go_vet_diagnostics(&combined);
                Ok(build_report(
                    subproject_rel,
                    target.kind,
                    cmd_label,
                    out.status.success(),
                    diagnostics,
                    &combined,
                ))
            }
            VerificationKind::Test => {
                let cmd_label = format!("go test -json {pkg_scope}");
                let Ok(out) = Command::new("go")
                    .args(["test", "-json", pkg_scope])
                    .current_dir(&workdir)
                    .output()
                    .await
                else {
                    return Ok(skipped_missing_tool(subproject_rel, target.kind, &cmd_label, "go"));
                };
                let stdout = String::from_utf8_lossy(&out.stdout);
                let stderr = String::from_utf8_lossy(&out.stderr);
                let mut diagnostics = parse_go_test_json_diagnostics(&stdout);
                if diagnostics.is_empty() && !out.status.success() {
                    diagnostics = parse_go_vet_diagnostics(&stderr);
                }
                Ok(build_report(
                    subproject_rel,
                    target.kind,
                    cmd_label,
                    out.status.success(),
                    diagnostics,
                    &format!("{stdout}\n{stderr}"),
                ))
            }
        }
    }
}

/// Parse `go vet` / `go build` diagnostics of the form `./pkg/foo.go:14:9: message`.
pub fn parse_go_vet_diagnostics(output: &str) -> Vec<CompilerDiagnostic> {
    let mut out = Vec::new();
    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // Try `file.go:line:col: msg`
        let parts: Vec<&str> = trimmed.splitn(4, ':').collect();
        if parts.len() == 4
            && parts[0].ends_with(".go")
            && let (Ok(line_no), Ok(col_no)) = (parts[1].trim().parse::<u32>(), parts[2].trim().parse::<u32>())
        {
            out.push(CompilerDiagnostic {
                level: "error".to_string(),
                code: None,
                message: parts[3].trim().to_string(),
                file: Some(parts[0].trim_start_matches("./").replace('\\', "/")),
                line: Some(line_no),
                column: Some(col_no),
                suggestion: None,
            });
            continue;
        }
        // Try `file.go:line: msg`
        let parts3: Vec<&str> = trimmed.splitn(3, ':').collect();
        if parts3.len() == 3
            && parts3[0].ends_with(".go")
            && let Ok(line_no) = parts3[1].trim().parse::<u32>()
        {
            out.push(CompilerDiagnostic {
                level: "error".to_string(),
                code: None,
                message: parts3[2].trim().to_string(),
                file: Some(parts3[0].trim_start_matches("./").replace('\\', "/")),
                line: Some(line_no),
                column: None,
                suggestion: None,
            });
        }
    }
    out
}

/// Parse `golangci-lint run --out-format json` payload.
pub fn parse_golangci_json_diagnostics(stdout: &str) -> Vec<CompilerDiagnostic> {
    let Ok(root) = serde_json::from_str::<serde_json::Value>(stdout.trim()) else {
        return Vec::new();
    };
    let Some(issues) = root.get("Issues").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for issue in issues {
        let sev = issue
            .get("Severity")
            .and_then(|v| v.as_str())
            .unwrap_or("warning")
            .to_ascii_lowercase();
        let level = if sev == "error" { "error" } else { "warning" }.to_string();
        let code = issue
            .get("FromLinter")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let message = issue
            .get("Text")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let pos = issue.get("Pos");
        let file = pos
            .and_then(|p| p.get("Filename"))
            .and_then(|v| v.as_str())
            .map(|s| s.replace('\\', "/"));
        let line = pos
            .and_then(|p| p.get("Line"))
            .and_then(|v| v.as_u64())
            .map(|n| n as u32);
        let column = pos
            .and_then(|p| p.get("Column"))
            .and_then(|v| v.as_u64())
            .map(|n| n as u32);

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

/// Parse `go test -json` newline-delimited test events into failure diagnostics.
pub fn parse_go_test_json_diagnostics(stdout: &str) -> Vec<CompilerDiagnostic> {
    use std::collections::HashMap;

    let mut outputs: HashMap<(String, String), Vec<String>> = HashMap::new();
    let mut out = Vec::new();

    for line in stdout.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with('{') {
            continue;
        }
        let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) else {
            continue;
        };
        let action = val.get("Action").and_then(|v| v.as_str()).unwrap_or("");
        let pkg = val
            .get("Package")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let test = val
            .get("Test")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if action == "output"
            && let Some(text) = val.get("Output").and_then(|v| v.as_str())
        {
            outputs
                .entry((pkg, test))
                .or_default()
                .push(text.trim_end().to_string());
        } else if action == "fail" && !test.is_empty() {
            let captured = outputs
                .remove(&(pkg.clone(), test.clone()))
                .unwrap_or_default();
            let detail = captured
                .into_iter()
                .filter(|l| !l.starts_with("=== RUN") && !l.starts_with("--- FAIL"))
                .collect::<Vec<_>>()
                .join(" | ");
            let message = if detail.is_empty() {
                format!("Go test `{test}` failed in `{pkg}`")
            } else {
                detail
            };
            out.push(CompilerDiagnostic {
                level: "error".to_string(),
                code: Some(test),
                message,
                file: Some(pkg),
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
        toolchain: "go".to_string(),
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
        toolchain: "go".to_string(),
        working_dir: subproject_rel.to_string(),
        kind,
        command: command.to_string(),
        success: true,
        error_count: 0,
        warning_count: 0,
        diagnostics: Vec::new(),
        raw_summary: format!("Tool `{tool_hint}` not found on PATH; skipped Go verification."),
    }
}
