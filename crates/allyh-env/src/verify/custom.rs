use std::path::Path;

use tokio::process::Command;

use crate::error::{EnvError, Result};
use crate::verify::{
    CompilerDiagnostic, VerificationKind, VerificationReport, subproject_dir, truncate_tail,
};

/// Execute a custom verification command from a [`allyh_memory::RepoProfile`],
/// automatically parsing SARIF 2.1.0 JSON or generic `file:line:col: severity: message` lines.
pub async fn run_custom_command(
    repo_root: &Path,
    subproject_rel: &str,
    kind: VerificationKind,
    command_line: &str,
) -> Result<VerificationReport> {
    let workdir = subproject_dir(repo_root, subproject_rel);
    let output = if cfg!(windows) {
        Command::new("cmd")
            .args(["/C", command_line])
            .current_dir(&workdir)
            .output()
            .await
    } else {
        Command::new("sh")
            .args(["-c", command_line])
            .current_dir(&workdir)
            .output()
            .await
    }
    .map_err(|e| EnvError::Command(format!("failed to run custom verifier `{command_line}`: {e}")))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}\n{stderr}");

    let mut diagnostics = parse_sarif_diagnostics(&stdout);
    if diagnostics.is_empty() {
        diagnostics = parse_generic_file_line_diagnostics(&combined);
    }

    let error_count = diagnostics.iter().filter(|d| d.level == "error").count();
    let warning_count = diagnostics.iter().filter(|d| d.level == "warning").count();
    let success = output.status.success();
    let raw_summary = if !success && diagnostics.is_empty() {
        truncate_tail(&combined, 1200)
    } else {
        truncate_tail(&combined, 400)
    };

    Ok(VerificationReport {
        toolchain: "custom".to_string(),
        working_dir: subproject_rel.to_string(),
        kind,
        command: command_line.to_string(),
        success,
        error_count,
        warning_count,
        diagnostics,
        raw_summary,
    })
}

/// Parse SARIF 2.1.0 JSON output (supported by clang, Roslyn/C#, Java SpotBugs, CodeQL, ruff, etc.).
pub fn parse_sarif_diagnostics(stdout: &str) -> Vec<CompilerDiagnostic> {
    let trimmed = stdout.trim();
    if !trimmed.starts_with('{') {
        return Vec::new();
    }
    let Ok(root) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        return Vec::new();
    };
    let Some(runs) = root.get("runs").and_then(|v| v.as_array()) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for run in runs {
        let Some(results) = run.get("results").and_then(|v| v.as_array()) else {
            continue;
        };
        for res in results {
            let raw_level = res
                .get("level")
                .and_then(|v| v.as_str())
                .unwrap_or("error")
                .to_ascii_lowercase();
            let level = match raw_level.as_str() {
                "warning" => "warning",
                "note" | "none" => continue,
                _ => "error",
            }
            .to_string();
            let code = res
                .get("ruleId")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let message = res
                .get("message")
                .and_then(|m| m.get("text"))
                .and_then(|v| v.as_str())
                .unwrap_or("SARIF diagnostic")
                .to_string();
            let phys = res
                .get("locations")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|l| l.get("physicalLocation"));
            let file = phys
                .and_then(|p| p.get("artifactLocation"))
                .and_then(|a| a.get("uri"))
                .and_then(|v| v.as_str())
                .map(|s| s.trim_start_matches("file:///").replace('\\', "/"));
            let region = phys.and_then(|p| p.get("region"));
            let line = region
                .and_then(|r| r.get("startLine"))
                .and_then(|v| v.as_u64())
                .map(|n| n as u32);
            let column = region
                .and_then(|r| r.get("startColumn"))
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
    }
    out
}

/// Generic fallback parser for compiler lines like:
/// `path/file.c:42:10: error: expected ';'` or `src/Main.java:18: warning: unchecked cast`.
pub fn parse_generic_file_line_diagnostics(output: &str) -> Vec<CompilerDiagnostic> {
    let mut out = Vec::new();
    for line in output.lines() {
        let trimmed = line.trim();
        let parts: Vec<&str> = trimmed.splitn(5, ':').collect();
        if parts.len() == 5
            && let (Ok(line_no), Ok(col_no)) =
                (parts[1].trim().parse::<u32>(), parts[2].trim().parse::<u32>())
        {
            let sev = parts[3].trim().to_ascii_lowercase();
            if sev == "error" || sev == "warning" {
                out.push(CompilerDiagnostic {
                    level: sev,
                    code: None,
                    message: parts[4].trim().to_string(),
                    file: Some(parts[0].trim().replace('\\', "/")),
                    line: Some(line_no),
                    column: Some(col_no),
                    suggestion: None,
                });
                continue;
            }
        }
        let parts4: Vec<&str> = trimmed.splitn(4, ':').collect();
        if parts4.len() == 4
            && let Ok(line_no) = parts4[1].trim().parse::<u32>()
        {
            let sev = parts4[2].trim().to_ascii_lowercase();
            if sev == "error" || sev == "warning" {
                out.push(CompilerDiagnostic {
                    level: sev,
                    code: None,
                    message: parts4[3].trim().to_string(),
                    file: Some(parts4[0].trim().replace('\\', "/")),
                    line: Some(line_no),
                    column: None,
                    suggestion: None,
                });
            }
        }
    }
    out
}
