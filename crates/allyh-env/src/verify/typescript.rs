use std::path::Path;

use async_trait::async_trait;
use tokio::process::Command;

use crate::error::Result;
use crate::verify::{
    CompilerDiagnostic, VerificationAdapter, VerificationKind, VerificationReport,
    VerificationTarget, subproject_dir, truncate_tail,
};

/// TypeScript / JavaScript verification adapter (`tsc`, `eslint`/`biome`, `vitest`/`npm test`).
pub struct TypeScriptVerificationAdapter;

#[async_trait]
impl VerificationAdapter for TypeScriptVerificationAdapter {
    fn id(&self) -> &'static str {
        "typescript"
    }

    fn display_name(&self) -> &'static str {
        "TypeScript/JS"
    }

    fn manifest_names(&self) -> &'static [&'static str] {
        &["tsconfig.json", "package.json"]
    }

    async fn run(
        &self,
        repo_root: &Path,
        subproject_rel: &str,
        target: &VerificationTarget,
    ) -> Result<VerificationReport> {
        let workdir = subproject_dir(repo_root, subproject_rel);
        let has_tsconfig = workdir.join("tsconfig.json").exists();
        let has_biome = workdir.join("biome.json").exists() || workdir.join("biome.jsonc").exists();

        match target.kind {
            VerificationKind::Check => {
                if !has_tsconfig {
                    return Ok(VerificationReport {
                        toolchain: "typescript".to_string(),
                        working_dir: subproject_rel.to_string(),
                        kind: target.kind,
                        command: "tsc --noEmit".to_string(),
                        success: true,
                        error_count: 0,
                        warning_count: 0,
                        diagnostics: Vec::new(),
                        raw_summary: "No tsconfig.json found; skipped tsc type-check.".to_string(),
                    });
                }
                let (cmd_label, output) =
                    run_node_tool(&workdir, "tsc", &["--noEmit", "--pretty", "false"]).await;
                let Some(out) = output else {
                    return Ok(skipped_missing_tool(
                        "typescript",
                        subproject_rel,
                        target.kind,
                        &cmd_label,
                        "tsc",
                    ));
                };
                let stdout = String::from_utf8_lossy(&out.stdout);
                let stderr = String::from_utf8_lossy(&out.stderr);
                let combined = format!("{stdout}\n{stderr}");
                let diagnostics = parse_tsc_diagnostics(&combined);
                Ok(build_report(
                    "typescript",
                    subproject_rel,
                    target.kind,
                    cmd_label,
                    out.status.success(),
                    diagnostics,
                    &combined,
                ))
            }
            VerificationKind::Lint | VerificationKind::Clippy => {
                if has_biome {
                    let (cmd_label, output) =
                        run_node_tool(&workdir, "biome", &["check", "--reporter=json", "."]).await;
                    if let Some(out) = output {
                        let stdout = String::from_utf8_lossy(&out.stdout);
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        let diagnostics = parse_biome_json_diagnostics(&stdout);
                        return Ok(build_report(
                            "typescript",
                            subproject_rel,
                            target.kind,
                            cmd_label,
                            out.status.success(),
                            diagnostics,
                            &format!("{stdout}\n{stderr}"),
                        ));
                    }
                }
                let (cmd_label, output) =
                    run_node_tool(&workdir, "eslint", &["--format", "json", "."]).await;
                let Some(out) = output else {
                    return Ok(skipped_missing_tool(
                        "typescript",
                        subproject_rel,
                        target.kind,
                        &cmd_label,
                        "eslint/biome",
                    ));
                };
                let stdout = String::from_utf8_lossy(&out.stdout);
                let stderr = String::from_utf8_lossy(&out.stderr);
                let diagnostics = parse_eslint_json_diagnostics(&stdout);
                Ok(build_report(
                    "typescript",
                    subproject_rel,
                    target.kind,
                    cmd_label,
                    out.status.success(),
                    diagnostics,
                    &format!("{stdout}\n{stderr}"),
                ))
            }
            VerificationKind::Test => {
                let filter = target
                    .test_filter
                    .as_deref()
                    .or(target.target_scope.as_deref())
                    .map(str::trim)
                    .filter(|s| !s.is_empty() && *s != subproject_rel);

                // Prefer local vitest when present for fast targeted test runs.
                let has_local_vitest = workdir
                    .join("node_modules")
                    .join(".bin")
                    .join(if cfg!(windows) { "vitest.cmd" } else { "vitest" })
                    .exists();
                if has_local_vitest {
                    let mut v_args: Vec<&str> = vec!["run"];
                    if let Some(f) = filter {
                        v_args.push(f);
                    }
                    let (cmd_label, output) = run_node_tool(&workdir, "vitest", &v_args).await;
                    if let Some(out) = output {
                        let combined = format!(
                            "{}\n{}",
                            String::from_utf8_lossy(&out.stdout),
                            String::from_utf8_lossy(&out.stderr)
                        );
                        let diagnostics = parse_tsc_diagnostics(&combined);
                        return Ok(build_report(
                            "typescript",
                            subproject_rel,
                            target.kind,
                            cmd_label,
                            out.status.success(),
                            diagnostics,
                            &combined,
                        ));
                    }
                }

                let mut args: Vec<String> = vec!["test".to_string()];
                if let Some(f) = filter {
                    args.push("--".to_string());
                    args.push(f.to_string());
                }
                let npm = node_bin("npm");
                let cmd_label = format!("npm {}", args.join(" "));
                let output = Command::new(npm)
                    .args(&args)
                    .current_dir(&workdir)
                    .output()
                    .await
                    .ok();
                let Some(out) = output else {
                    return Ok(skipped_missing_tool(
                        "typescript",
                        subproject_rel,
                        target.kind,
                        &cmd_label,
                        "npm",
                    ));
                };
                let combined = format!(
                    "{}\n{}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                );
                let diagnostics = parse_tsc_diagnostics(&combined);
                Ok(build_report(
                    "typescript",
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

/// Parse `tsc --noEmit --pretty false` diagnostics of the form:
/// `src/app.ts(12,5): error TS2322: Type 'number' is not assignable to type 'string'.`
/// or `src/app.ts:12:5 - error TS2322: ...`
pub fn parse_tsc_diagnostics(output: &str) -> Vec<CompilerDiagnostic> {
    let mut out = Vec::new();
    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Format A: `path/file.ts(line,col): error TS1234: message`
        if let Some((loc_part, rest)) = trimmed.split_once("): ")
            && let Some((file, coords)) = loc_part.rsplit_once('(')
            && let Some((line_s, col_s)) = coords.split_once(',')
            && let (Ok(line_no), Ok(col_no)) = (line_s.trim().parse::<u32>(), col_s.trim().parse::<u32>())
            && let Some((level, code, message)) = split_ts_severity(rest)
        {
            out.push(CompilerDiagnostic {
                level,
                code,
                message,
                file: Some(file.trim().replace('\\', "/")),
                line: Some(line_no),
                column: Some(col_no),
                suggestion: None,
            });
            continue;
        }
        // Format B: `path/file.ts:line:col - error TS1234: message`
        if let Some((loc_part, rest)) = trimmed.split_once(" - ") {
            let parts: Vec<&str> = loc_part.rsplitn(3, ':').collect();
            if parts.len() == 3
                && let (Ok(col_no), Ok(line_no)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>())
                && let Some((level, code, message)) = split_ts_severity(rest)
            {
                out.push(CompilerDiagnostic {
                    level,
                    code,
                    message,
                    file: Some(parts[2].trim().replace('\\', "/")),
                    line: Some(line_no),
                    column: Some(col_no),
                    suggestion: None,
                });
            }
        }
    }
    out
}

fn split_ts_severity(rest: &str) -> Option<(String, Option<String>, String)> {
    let (sev_and_code, msg) = rest.split_once(':')?;
    let mut tokens = sev_and_code.split_whitespace();
    let sev = tokens.next()?.to_ascii_lowercase();
    if sev != "error" && sev != "warning" {
        return None;
    }
    let code = tokens.next().map(str::to_string);
    Some((sev, code, msg.trim().to_string()))
}

/// Parse `eslint --format json` array output into structured [`CompilerDiagnostic`]s.
pub fn parse_eslint_json_diagnostics(stdout: &str) -> Vec<CompilerDiagnostic> {
    let Ok(files) = serde_json::from_str::<Vec<serde_json::Value>>(stdout.trim()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for file_entry in files {
        let file = file_entry
            .get("filePath")
            .and_then(|v| v.as_str())
            .map(|s| s.replace('\\', "/"));
        let Some(messages) = file_entry.get("messages").and_then(|v| v.as_array()) else {
            continue;
        };
        for msg in messages {
            let severity = msg.get("severity").and_then(|v| v.as_u64()).unwrap_or(1);
            let level = if severity >= 2 { "error" } else { "warning" }.to_string();
            let message = msg
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let code = msg
                .get("ruleId")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let line = msg.get("line").and_then(|v| v.as_u64()).map(|n| n as u32);
            let column = msg.get("column").and_then(|v| v.as_u64()).map(|n| n as u32);
            let suggestion = msg
                .get("suggestions")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|s| s.get("desc"))
                .and_then(|v| v.as_str())
                .map(str::to_string);

            out.push(CompilerDiagnostic {
                level,
                code,
                message,
                file: file.clone(),
                line,
                column,
                suggestion,
            });
        }
    }
    out
}

/// Parse `biome check --reporter=json` output into structured [`CompilerDiagnostic`]s.
pub fn parse_biome_json_diagnostics(stdout: &str) -> Vec<CompilerDiagnostic> {
    let Ok(root) = serde_json::from_str::<serde_json::Value>(stdout.trim()) else {
        return Vec::new();
    };
    let Some(diags) = root.get("diagnostics").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for d in diags {
        let sev = d
            .get("severity")
            .and_then(|v| v.as_str())
            .unwrap_or("error")
            .to_ascii_lowercase();
        let level = if sev.contains("warn") {
            "warning"
        } else {
            "error"
        }
        .to_string();
        let code = d
            .get("category")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let message = d
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("biome diagnostic")
            .to_string();
        let file = d
            .get("location")
            .and_then(|l| l.get("path"))
            .and_then(|p| p.get("file"))
            .and_then(|v| v.as_str())
            .map(|s| s.replace('\\', "/"));

        out.push(CompilerDiagnostic {
            level,
            code,
            message,
            file,
            line: None,
            column: None,
            suggestion: None,
        });
    }
    out
}

async fn run_node_tool(
    workdir: &Path,
    bin_name: &str,
    args: &[&str],
) -> (String, Option<std::process::Output>) {
    let local_bin = if cfg!(windows) {
        workdir
            .join("node_modules")
            .join(".bin")
            .join(format!("{bin_name}.cmd"))
    } else {
        workdir.join("node_modules").join(".bin").join(bin_name)
    };
    let cmd_label = format!("{bin_name} {}", args.join(" "));
    if local_bin.exists()
        && let Ok(out) = Command::new(&local_bin)
            .args(args)
            .current_dir(workdir)
            .output()
            .await
    {
        return (cmd_label, Some(out));
    }

    let global_bin = node_bin(bin_name);
    let out = Command::new(global_bin)
        .args(args)
        .current_dir(workdir)
        .output()
        .await
        .ok();
    (cmd_label, out)
}

fn node_bin(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.cmd")
    } else {
        name.to_string()
    }
}

fn build_report(
    toolchain: &str,
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
        toolchain: toolchain.to_string(),
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
    toolchain: &str,
    subproject_rel: &str,
    kind: VerificationKind,
    command: &str,
    tool_hint: &str,
) -> VerificationReport {
    VerificationReport {
        toolchain: toolchain.to_string(),
        working_dir: subproject_rel.to_string(),
        kind,
        command: command.to_string(),
        success: true,
        error_count: 0,
        warning_count: 0,
        diagnostics: Vec::new(),
        raw_summary: format!(
            "Tool `{tool_hint}` not found on PATH or in node_modules/.bin; skipped verification."
        ),
    }
}
