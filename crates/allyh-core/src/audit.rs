use std::path::Path;

use allyh_env::{GitDiffReport, GitTracker, ToolchainVerifier, VerificationTarget};
use allyh_memory::{ArchitecturalDecision, MemoryStore, VerificationProof};
use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::orchestrator::AllyHOrchestrator;

/// Single architectural invariant violation detected in a Git diff during `allyh audit`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditViolation {
    pub decision_id: String,
    pub topic: String,
    pub rule_statement: String,
    pub file_path: String,
    pub line_preview: String,
    pub reason: String,
}

/// Structured PR/branch architectural drift report produced by `allyh audit --base <ref>`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitecturalAuditReport {
    pub repo_id: String,
    pub scope_id: String,
    pub base_ref: String,
    pub files_changed: usize,
    pub changed_files: Vec<String>,
    pub invariants_checked: usize,
    pub violations: Vec<AuditViolation>,
    pub passed: bool,
}

/// Execution mode for headless CI runs (`allyh run`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HeadlessMode {
    /// Strict read-only mode (Control Gate permanently locked). Ideal for PR review & auditing.
    DiscoveryOnly,
    /// Execution mode with automatic plan approval and mandatory verification check.
    Execute,
}

impl HeadlessMode {
    pub fn parse_spec(raw: &str) -> std::result::Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "discovery" | "discovery-only" | "read-only" | "readonly" => Ok(Self::DiscoveryOnly),
            "execute" | "exec" | "autofix" => Ok(Self::Execute),
            other => Err(format!(
                "invalid headless mode '{other}' (expected 'discovery-only' or 'execute')"
            )),
        }
    }
}

/// Structured JSONL event emitted by `allyh run --output jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HeadlessStreamEvent {
    Start {
        repo_id: String,
        scope_id: String,
        mode: HeadlessMode,
        model: String,
    },
    ToolCard {
        tool_name: String,
        succeeded: bool,
        header: String,
        output: String,
    },
    AssistantReply {
        text: String,
        preamble_tokens: u64,
        stable_prefix_tokens: u64,
    },
    Verification {
        command: String,
        passed: bool,
        warnings: usize,
        summary: String,
    },
    Summary {
        passed: bool,
        tool_calls: usize,
        decisions_consolidated: usize,
        input_tokens: u64,
        output_tokens: u64,
        cached_input_tokens: u64,
        estimated_cost_usd: f64,
        verification_proof: Option<VerificationProof>,
    },
}

/// Final outcome returned by [`run_headless_turn`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadlessRunReport {
    pub passed: bool,
    pub assistant_text: String,
    pub tool_calls: usize,
    pub verification_proof: Option<VerificationProof>,
    pub events: Vec<HeadlessStreamEvent>,
}

/// Execute a non-interactive headless CI turn on an initialized [`AllyHOrchestrator`].
pub async fn run_headless_turn(
    orch: &mut AllyHOrchestrator,
    repo_root: &Path,
    prompt: &str,
    mode: HeadlessMode,
    auto_approve: bool,
    require_verification: bool,
) -> Result<HeadlessRunReport> {
    let mut events = vec![HeadlessStreamEvent::Start {
        repo_id: orch.repo_id().to_string(),
        scope_id: orch.scope_id().to_string(),
        mode,
        model: orch.router().primary().label.clone(),
    }];

    if mode == HeadlessMode::Execute && auto_approve {
        let plan = AllyHOrchestrator::build_quick_plan(prompt);
        let _ = orch.propose_plan(plan);
        let _ = orch.approve_plan().await?;
    }

    let outcome = orch.handle_prompt(prompt, None).await?;
    for rec in &outcome.tool_records {
        events.push(HeadlessStreamEvent::ToolCard {
            tool_name: rec.tool_name.clone(),
            succeeded: rec.succeeded,
            header: rec.compact_header.clone(),
            output: rec.output_text.clone(),
        });
    }
    events.push(HeadlessStreamEvent::AssistantReply {
        text: outcome.assistant_text.clone(),
        preamble_tokens: outcome.preamble.estimated_tokens,
        stable_prefix_tokens: outcome.preamble.stable_prefix_tokens,
    });

    let mut verification_proof = None;
    let mut passed = true;

    if mode == HeadlessMode::Execute && require_verification {
        let profile = orch.get_repo_profile().await.ok().flatten();
        let verifier = ToolchainVerifier::new(repo_root);
        let has_tc = !verifier.detect_toolchains().is_empty()
            || profile
                .as_ref()
                .is_some_and(|p| p.check_command.is_some() || p.test_command.is_some());
        if has_tc {
            match verifier
                .run(&VerificationTarget::default(), profile.as_ref())
                .await
            {
                Ok(rep) => {
                    let summary = rep.compact_status_line();
                    events.push(HeadlessStreamEvent::Verification {
                        command: rep.command.clone(),
                        passed: rep.success,
                        warnings: rep.warning_count,
                        summary: summary.clone(),
                    });
                    if rep.success {
                        verification_proof = Some(VerificationProof {
                            command: rep.command,
                            passed_at: Utc::now(),
                            warnings: rep.warning_count,
                            summary,
                        });
                    } else {
                        passed = false;
                    }
                }
                Err(err) => {
                    events.push(HeadlessStreamEvent::Verification {
                        command: "auto".to_string(),
                        passed: false,
                        warnings: 0,
                        summary: format!("verification error: {err}"),
                    });
                    passed = false;
                }
            }
        }
    }

    events.push(HeadlessStreamEvent::Summary {
        passed,
        tool_calls: outcome.tool_records.len(),
        decisions_consolidated: outcome.consolidated_decisions.len(),
        input_tokens: outcome.turn_telemetry.input_tokens,
        output_tokens: outcome.turn_telemetry.output_tokens,
        cached_input_tokens: outcome.turn_telemetry.cached_input_tokens,
        estimated_cost_usd: outcome.turn_telemetry.estimated_cost_usd,
        verification_proof: verification_proof.clone(),
    });

    Ok(HeadlessRunReport {
        passed,
        assistant_text: outcome.assistant_text,
        tool_calls: outcome.tool_records.len(),
        verification_proof,
        events,
    })
}

/// Compare a Git diff (`base_ref` -> working tree) against all active bi-temporal
/// `architectural_decisions` stored in LanceDB for `(repo_id, scope_id)`.
pub async fn audit_repo_against_base(
    store: &MemoryStore,
    repo_root: &Path,
    repo_id: &str,
    scope_id: &str,
    base_ref: &str,
) -> Result<ArchitecturalAuditReport> {
    let git = GitTracker::new(repo_root);
    let diff = git.diff_against_base(base_ref, 64_000)?;
    let invariants = store.list_active_decisions(repo_id, scope_id).await?;
    Ok(audit_diff_against_invariants(
        repo_id,
        scope_id,
        base_ref,
        &diff,
        &invariants,
    ))
}

/// Pure deterministic diff-vs-invariant drift analyzer (also directly unit-testable).
pub fn audit_diff_against_invariants(
    repo_id: &str,
    scope_id: &str,
    base_ref: &str,
    diff: &GitDiffReport,
    invariants: &[ArchitecturalDecision],
) -> ArchitecturalAuditReport {
    let added_lines = parse_added_lines_by_file(&diff.patch);
    let mut violations = Vec::new();

    for decision in invariants {
        let checks = extract_prohibited_checks(decision);
        if checks.is_empty() {
            continue;
        }
        for (file_path, lines) in &added_lines {
            for check in &checks {
                if !check.applies_to_file(file_path) {
                    continue;
                }
                for line in lines {
                    if line_contains_token_ci(line, &check.forbidden_token) {
                        violations.push(AuditViolation {
                            decision_id: decision.id.clone(),
                            topic: decision.topic.clone(),
                            rule_statement: decision.rule_statement.clone(),
                            file_path: file_path.clone(),
                            line_preview: line.trim().chars().take(140).collect(),
                            reason: check.reason.clone(),
                        });
                        break;
                    }
                }
            }
        }
    }

    let passed = violations.is_empty();
    ArchitecturalAuditReport {
        repo_id: repo_id.to_string(),
        scope_id: scope_id.to_string(),
        base_ref: base_ref.to_string(),
        files_changed: diff.files_changed,
        changed_files: diff.changed_files.clone(),
        invariants_checked: invariants.len(),
        violations,
        passed,
    }
}

#[derive(Debug, Clone)]
struct DriftCheck {
    forbidden_token: String,
    /// When `Some`, the forbidden token is allowed ONLY inside paths matching one of these prefixes.
    allowed_only_in: Vec<String>,
    /// When non-empty, restrict check to files matching one of these path prefixes.
    scoped_paths: Vec<String>,
    reason: String,
}

impl DriftCheck {
    fn applies_to_file(&self, file_path: &str) -> bool {
        let norm = file_path.replace('\\', "/");
        if !self.allowed_only_in.is_empty() {
            let in_allowed = self
                .allowed_only_in
                .iter()
                .any(|allowed| norm.contains(allowed.as_str()));
            return !in_allowed;
        }
        if !self.scoped_paths.is_empty() {
            return self
                .scoped_paths
                .iter()
                .any(|p| norm.contains(p.as_str()));
        }
        true
    }
}

/// Extract structured prohibition rules from natural-language architectural invariants:
/// - `"Never use X ..."` / `"Do not use X ..."` / `"Avoid X ..."` / `"Ban X ..."` / `"Forbid X ..."`
/// - `"Use X instead of Y"`
/// - `"<token> only in <path>"` (e.g., `"Use thiserror in allyh-core and anyhow only in allyh-tui"`)
fn extract_prohibited_checks(decision: &ArchitecturalDecision) -> Vec<DriftCheck> {
    let mut checks = Vec::new();
    let rule = decision.rule_statement.as_str();
    let lower = rule.to_ascii_lowercase();

    // Pattern 1: "<token> only in <allowed_path>" (e.g., "anyhow only in allyh-tui")
    if let Some(pos) = lower.find(" only in ") {
        let before = rule[..pos].trim();
        let after = rule[pos + " only in ".len()..].trim();
        let forbidden = before
            .split_whitespace()
            .last()
            .map(clean_ident_token)
            .unwrap_or_default();
        let allowed_path = after
            .split(|c: char| c.is_whitespace() || c == ',' || c == ';' || c == ')')
            .next()
            .map(clean_path_token)
            .unwrap_or_default();
        if forbidden.len() >= 3 && !is_stopword(&forbidden) && !allowed_path.is_empty() {
            checks.push(DriftCheck {
                forbidden_token: forbidden.clone(),
                allowed_only_in: vec![allowed_path.clone()],
                scoped_paths: Vec::new(),
                reason: format!(
                    "`{forbidden}` is restricted to `{allowed_path}` by invariant '{}'",
                    decision.rule_statement
                ),
            });
        }
    }

    // Pattern 2: "never use <X>", "do not use <X>", "don't use <X>", "avoid <X>", "forbid <X>", "ban <X>", "no <X>"
    for marker in [
        "never use ",
        "do not use ",
        "don't use ",
        "must not use ",
        "avoid using ",
        "avoid ",
        "forbid ",
        "ban ",
        "instead of ",
    ] {
        let mut search_from = 0usize;
        while let Some(rel) = lower[search_from..].find(marker) {
            let start = search_from + rel + marker.len();
            let tail = rule[start..].trim();
            if let Some(raw_tok) = tail.split_whitespace().next() {
                let tok = clean_ident_token(raw_tok);
                if tok.len() >= 3 && !is_stopword(&tok) {
                    // Check if rule scopes the prohibition with "in <path>"
                    let mut scoped = decision.affected_paths.clone();
                    if scoped.is_empty()
                        && let Some(in_pos) = lower[start..].find(" in ")
                    {
                        let p_tail = rule[start + in_pos + 4..].trim();
                        if let Some(p_raw) = p_tail.split_whitespace().next() {
                            let p = clean_path_token(p_raw);
                            if p.contains('/') || p.contains('-') || p.contains('_') {
                                scoped.push(p);
                            }
                        }
                    }
                    checks.push(DriftCheck {
                        forbidden_token: tok.clone(),
                        allowed_only_in: Vec::new(),
                        scoped_paths: scoped,
                        reason: format!(
                            "Prohibited token `{tok}` added in violation of invariant '{}'",
                            decision.rule_statement
                        ),
                    });
                }
            }
            search_from = start;
        }
    }

    checks
}

fn parse_added_lines_by_file(patch: &str) -> Vec<(String, Vec<String>)> {
    let mut result: Vec<(String, Vec<String>)> = Vec::new();
    let mut current_file: Option<String> = None;

    for line in patch.lines() {
        if let Some(rest) = line.strip_prefix("+++ b/") {
            let path = rest.trim().to_string();
            current_file = Some(path.clone());
            if !result.iter().any(|(p, _)| p == &path) {
                result.push((path, Vec::new()));
            }
        } else if let Some(rest) = line.strip_prefix("diff --git ") {
            if let Some(b_part) = rest.split(" b/").nth(1) {
                let path = b_part.trim().to_string();
                current_file = Some(path.clone());
                if !result.iter().any(|(p, _)| p == &path) {
                    result.push((path, Vec::new()));
                }
            }
        } else if line.starts_with('+') && !line.starts_with("+++") {
            if let Some(ref cur) = current_file
                && let Some((_, lines)) = result.iter_mut().find(|(p, _)| p == cur)
            {
                lines.push(line[1..].to_string());
            }
        }
    }
    result
}

fn line_contains_token_ci(line: &str, token: &str) -> bool {
    let lower_line = line.to_ascii_lowercase();
    let lower_tok = token.to_ascii_lowercase();
    if lower_tok.is_empty() {
        return false;
    }
    let bytes = lower_line.as_bytes();
    let mut idx = 0usize;
    while let Some(found) = lower_line[idx..].find(&lower_tok) {
        let start = idx + found;
        let end = start + lower_tok.len();
        let left_ok = start == 0 || !is_ident_byte(bytes[start - 1]);
        let right_ok = end == bytes.len() || !is_ident_byte(bytes[end]);
        if left_ok && right_ok {
            return true;
        }
        idx = start + 1;
    }
    false
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn clean_ident_token(raw: &str) -> String {
    raw.trim_matches(|c: char| {
        matches!(
            c,
            '`' | '"' | '\'' | ',' | '.' | ';' | ':' | '(' | ')' | '[' | ']' | '{' | '}'
        )
    })
    .to_string()
}

fn clean_path_token(raw: &str) -> String {
    clean_ident_token(raw).replace('\\', "/")
}

fn is_stopword(tok: &str) -> bool {
    matches!(
        tok.to_ascii_lowercase().as_str(),
        "the" | "and" | "for" | "with" | "from" | "that" | "this" | "into" | "only" | "any" | "all"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_decision(rule: &str, paths: Vec<&str>) -> ArchitecturalDecision {
        ArchitecturalDecision {
            id: "dec-1".into(),
            repo_id: "repo".into(),
            scope_id: "default".into(),
            topic: "errors".into(),
            rule_statement: rule.into(),
            rationale: "Strict library error discipline".into(),
            affected_paths: paths.into_iter().map(str::to_string).collect(),
            vector: Vec::new(),
            valid_at: Utc::now(),
            invalid_at: None,
            superseded_by: None,
        }
    }

    #[test]
    fn detects_only_in_and_never_use_invariant_violations_on_diff() {
        let inv1 = sample_decision(
            "Use thiserror in allyh-core and anyhow only in allyh-tui",
            vec![],
        );
        let inv2 = sample_decision("Never use unwrap in crates/allyh-core", vec!["crates/allyh-core"]);

        let patch = "\
diff --git a/crates/allyh-core/src/foo.rs b/crates/allyh-core/src/foo.rs
+++ b/crates/allyh-core/src/foo.rs
+use anyhow::Result;
+pub fn boom() { Some(1).unwrap(); }
diff --git a/crates/allyh-tui/src/bar.rs b/crates/allyh-tui/src/bar.rs
+++ b/crates/allyh-tui/src/bar.rs
+use anyhow::Result;
";
        let diff = GitDiffReport {
            files_changed: 2,
            insertions: 3,
            deletions: 0,
            changed_files: vec![
                "crates/allyh-core/src/foo.rs".into(),
                "crates/allyh-tui/src/bar.rs".into(),
            ],
            patch: patch.into(),
            truncated: false,
        };

        let report = audit_diff_against_invariants("repo", "default", "main", &diff, &[inv1, inv2]);
        assert!(!report.passed);
        assert_eq!(report.violations.len(), 2);
        assert!(report
            .violations
            .iter()
            .all(|v| v.file_path == "crates/allyh-core/src/foo.rs"));
    }
}
