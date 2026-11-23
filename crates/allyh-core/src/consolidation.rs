use allyh_memory::{
    ArchitecturalDecision, CandidateDecision, ConflictJudge, ConsolidationAction,
    HeuristicConflictJudge, Result as MemoryResult,
};
use async_trait::async_trait;

use crate::router::HybridRouter;

/// Local 9B–12B LLM-backed conflict judge for AllyH's pre-write bi-temporal gate.
///
/// Falls back automatically to [`HeuristicConflictJudge`] when the local model server
/// is unavailable or returns unparseable JSON.
#[derive(Clone)]
pub struct LocalLlmConflictJudge {
    router: HybridRouter,
    fallback: HeuristicConflictJudge,
}

impl LocalLlmConflictJudge {
    pub fn new(router: HybridRouter) -> Self {
        Self {
            router,
            fallback: HeuristicConflictJudge,
        }
    }
}

#[async_trait]
impl ConflictJudge for LocalLlmConflictJudge {
    async fn judge(
        &self,
        candidate: &CandidateDecision,
        existing_active: &[ArchitecturalDecision],
    ) -> MemoryResult<ConsolidationAction> {
        if existing_active.is_empty() {
            return Ok(ConsolidationAction::Add);
        }

        let existing_summary = existing_active
            .iter()
            .map(|d| {
                serde_json::json!({
                    "id": d.id,
                    "topic": d.topic,
                    "rule_statement": d.rule_statement,
                    "affected_paths": d.affected_paths,
                })
            })
            .collect::<Vec<_>>();

        let system = "You are AllyH's bi-temporal memory conflict gate. \
            Compare the candidate architectural decision against existing active decisions and output ONLY valid JSON matching one of:\n\
            {\"action\":\"ADD\"}\n\
            {\"action\":\"UPDATE\",\"target_id\":\"<id>\"}\n\
            {\"action\":\"INVALIDATE\",\"target_id\":\"<id>\",\"reason\":\"<why>\"}\n\
            {\"action\":\"NOOP\",\"existing_id\":\"<id>\"}";

        let user = format!(
            "Candidate:\n{}\n\nExisting Active Decisions:\n{}",
            serde_json::to_string_pretty(candidate).unwrap_or_default(),
            serde_json::to_string_pretty(&existing_summary).unwrap_or_default(),
        );

        if let Ok(raw) = self.router.run_local_worker(system, &user).await
            && let Some(json_slice) = extract_json_object(&raw)
            && let Ok(action) = serde_json::from_str::<ConsolidationAction>(json_slice)
        {
            return Ok(action);
        }

        self.fallback.judge(candidate, existing_active).await
    }
}

/// Extract candidate architectural invariants from a completed turn (using the local worker model
/// when available, plus deterministic rule detection for explicit user constraints and overrides).
pub async fn extract_candidate_decisions(
    router: &HybridRouter,
    user_prompt: &str,
    assistant_reply: &str,
) -> Vec<CandidateDecision> {
    // 1. Try local 9B-12B extractor first when configured.
    let system = "Extract durable architectural constraints or rule overrides stated by the user. \
        Return ONLY a JSON array of objects with fields: topic, rule_statement, rationale, affected_paths (string array). \
        Return [] if the turn contains no durable architectural rules.";
    let prompt = format!("User:\n{user_prompt}\n\nAssistant:\n{assistant_reply}");

    if let Ok(raw) = router.run_local_worker(system, &prompt).await
        && let Some(arr_slice) = extract_json_array(&raw)
        && let Ok(parsed) = serde_json::from_str::<Vec<CandidateDecision>>(arr_slice)
        && !parsed.is_empty()
    {
        return parsed;
    }

    // 2. Deterministic extractor for explicit architectural rules / overrides.
    extract_heuristic_decisions(user_prompt)
}

/// Deterministic extractor recognizing explicit user rules, constraints, conventions, and overrides.
pub fn extract_heuristic_decisions(user_prompt: &str) -> Vec<CandidateDecision> {
    let trimmed = user_prompt.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    let lower = trimmed.to_ascii_lowercase();

    let is_explicit_rule = lower.starts_with("rule:")
        || lower.starts_with("constraint:")
        || lower.starts_with("decision:")
        || lower.starts_with("override:")
        || lower.starts_with("invariant:")
        || lower.contains("architectural constraint")
        || lower.contains("architectural rule")
        || lower.contains("override the ")
        || lower.contains("instead of ")
        || lower.contains("always use ")
        || lower.contains("never use ")
        || (lower.contains("use ") && lower.contains(" only in "));

    if !is_explicit_rule {
        return Vec::new();
    }

    let cleaned = trimmed
        .trim_start_matches("Rule:")
        .trim_start_matches("rule:")
        .trim_start_matches("Constraint:")
        .trim_start_matches("constraint:")
        .trim_start_matches("Decision:")
        .trim_start_matches("decision:")
        .trim_start_matches("Override:")
        .trim_start_matches("override:")
        .trim_start_matches("Invariant:")
        .trim_start_matches("invariant:")
        .trim()
        .to_string();

    let topic = infer_topic(&cleaned);
    let affected_paths = infer_affected_paths(&cleaned);

    vec![CandidateDecision {
        topic,
        rule_statement: cleaned,
        rationale: "Explicitly established by user turn".to_string(),
        affected_paths,
    }]
}

fn infer_topic(statement: &str) -> String {
    let lower = statement.to_ascii_lowercase();
    if lower.contains("thiserror")
        || lower.contains("anyhow")
        || lower.contains("miette")
        || lower.contains("error")
    {
        "error-handling".to_string()
    } else if lower.contains("lancedb") || lower.contains("s3://") || lower.contains("storage") {
        "storage".to_string()
    } else if lower.contains("tokio") || lower.contains("async") {
        "concurrency".to_string()
    } else if lower.contains("ratatui") || lower.contains("tui") {
        "tui".to_string()
    } else if lower.contains("test") || lower.contains("cargo") {
        "verification".to_string()
    } else {
        "architecture".to_string()
    }
}

fn infer_affected_paths(statement: &str) -> Vec<String> {
    let mut paths = Vec::new();
    for token in statement.split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_' && c != '/' && c != '.') {
        let t = token.trim_matches('.');
        if t.starts_with("allyh-") || t.contains('/') || t.ends_with(".rs") {
            let normalized = if t.starts_with("allyh-") {
                format!("crates/{t}")
            } else {
                t.to_string()
            };
            if !paths.contains(&normalized) {
                paths.push(normalized);
            }
        }
    }
    paths
}

fn extract_json_object(raw: &str) -> Option<&str> {
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    (end >= start).then(|| &raw[start..=end])
}

fn extract_json_array(raw: &str) -> Option<&str> {
    let start = raw.find('[')?;
    let end = raw.rfind(']')?;
    (end >= start).then(|| &raw[start..=end])
}
