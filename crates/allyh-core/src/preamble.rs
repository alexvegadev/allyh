use allyh_env::{EgoGraph, EnvDelta};
use allyh_memory::{ScoredDecision, SkillRecord, TaskRecord, TaskStatus};
use serde::{Deserialize, Serialize};

use crate::gate::ControlGateState;
use crate::harness::HarnessSpec;

/// Assembled system preamble + token estimate for the current turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssembledPreamble {
    pub text: String,
    pub estimated_tokens: u64,
    pub activated_decision_ids: Vec<String>,
}

/// Inputs required to construct AllyH's dynamic per-turn system preamble.
pub struct PreambleInputs<'a> {
    pub repo_id: &'a str,
    pub scope_id: &'a str,
    pub gate_state: &'a ControlGateState,
    pub harness: &'a HarnessSpec,
    pub env_delta: &'a EnvDelta,
    pub tasks: &'a [TaskRecord],
    pub memories: &'a [ScoredDecision],
    pub available_skills: &'a [SkillRecord],
    pub active_skills: &'a [SkillRecord],
    pub ego_graphs: &'a [EgoGraph],
    pub rolling_summary: Option<&'a str>,
}

/// Builds the dynamic per-turn preamble combining `HarnessSpec` policies, `EnvDelta`, active `tasks`,
/// top-k active `architectural_decisions` (`invalid_at IS NULL`), and 1-hop `RepoGraph` neighborhoods.
pub struct DynamicPreambleAssembler;

impl DynamicPreambleAssembler {
    pub fn assemble(inputs: &PreambleInputs<'_>) -> AssembledPreamble {
        let mut sections = Vec::new();
        let harness = inputs.harness;

        let mut header_lines = vec![
            "# AllyH Coherence-First Agent Harness".to_string(),
            format!(
                "Repository Partition: `repo_id={}` | `scope_id={}`",
                &inputs.repo_id[..inputs.repo_id.len().min(12)],
                inputs.scope_id
            ),
            format!("Control Gate Mode: `{}`", inputs.gate_state.label()),
            "Operating Rules:".to_string(),
        ];
        for rule in &harness.preamble.base_operating_rules {
            header_lines.push(format!("- {rule}"));
        }
        if !harness.tools.stale_hash_hint.trim().is_empty() {
            header_lines.push(format!("- {}", harness.tools.stale_hash_hint.trim()));
        }
        sections.push(header_lines.join("\n"));

        if !harness.preamble.verification_guidelines.is_empty() {
            let mut lines = vec!["## Verification Guidelines".to_string()];
            for g in &harness.preamble.verification_guidelines {
                lines.push(format!("- {g}"));
            }
            sections.push(lines.join("\n"));
        }

        if !harness.preamble.domain_guidance.is_empty() {
            let mut lines = vec!["## Domain Guidance".to_string()];
            for g in &harness.preamble.domain_guidance {
                lines.push(format!("- {g}"));
            }
            sections.push(lines.join("\n"));
        }

        if let Some(fmt_hint) = harness
            .model_adaptation
            .output_format_guidance
            .as_deref()
            .filter(|s| !s.trim().is_empty())
        {
            sections.push(format!("## Model Output Guidelines\n{}", fmt_hint.trim()));
        }

        if harness.preamble.include_rolling_summary
            && let Some(summary) = inputs.rolling_summary
            && !summary.trim().is_empty()
        {
            sections.push(format!("## Rolling Episodic Summary\n{}", summary.trim()));
        }

        // 1. Live Environment Delta
        sections.push(inputs.env_delta.render_for_preamble());

        // 2. Active Architectural Invariants (invalid_at IS NULL)
        let mut activated_ids = Vec::new();
        if !inputs.memories.is_empty() {
            let mut lines = vec!["## Active Architectural Invariants (LanceDB)".to_string()];
            for scored in inputs.memories {
                activated_ids.push(scored.decision.id.clone());
                let pin_tag = if scored.pinned { " [PINNED]" } else { "" };
                let paths = if scored.decision.affected_paths.is_empty() {
                    String::new()
                } else {
                    format!(" (paths: {})", scored.decision.affected_paths.join(", "))
                };
                lines.push(format!(
                    "- [{}]{pin_tag} {} — Rationale: {}{paths}",
                    scored.decision.topic, scored.decision.rule_statement, scored.decision.rationale
                ));
            }
            sections.push(lines.join("\n"));
        }

        // 3. Structured Execution Plan (`tasks`)
        let active_tasks: Vec<&TaskRecord> = inputs
            .tasks
            .iter()
            .filter(|t| t.status != TaskStatus::Verified)
            .collect();
        if !active_tasks.is_empty() {
            let mut lines = vec!["## Active Task Queue (LanceDB)".to_string()];
            for task in active_tasks.iter().take(6) {
                lines.push(format!(
                    "- [{}] {} (id: `{}`)",
                    task.status, task.title, task.id
                ));
                for crit in &task.acceptance_criteria {
                    lines.push(format!("  * Acceptance: {crit}"));
                }
                if let Some(err) = &task.last_error_trace {
                    lines.push(format!("  * Last Error: {}", err.lines().next().unwrap_or("")));
                }
            }
            sections.push(lines.join("\n"));
        }

        // 4. Reusable Skills (Progressive Disclosure: compact catalog + expanded active skills)
        if !inputs.available_skills.is_empty() {
            let mut lines = vec!["## Available Skills (LanceDB)".to_string()];
            for skill in inputs.available_skills {
                let triggers = if skill.triggers.is_empty() {
                    String::new()
                } else {
                    format!(" [triggers: {}]", skill.triggers.join(", "))
                };
                let desc = if skill.description.is_empty() {
                    "Reusable procedural skill"
                } else {
                    skill.description.as_str()
                };
                lines.push(format!("- `{}`: {desc}{triggers}", skill.name));
            }
            sections.push(lines.join("\n"));
        }

        if !inputs.active_skills.is_empty() {
            let mut blocks = vec!["## Active Skill Instructions".to_string()];
            for skill in inputs.active_skills {
                let mut block = format!("### {}", skill.name);
                if !skill.description.is_empty() {
                    block.push_str(&format!("\n_{}_", skill.description));
                }
                block.push_str(&format!("\n{}", skill.instructions.trim()));
                blocks.push(block);
            }
            sections.push(blocks.join("\n\n"));
        }

        // 5. Compact 1-hop RepoGraph Neighborhoods
        let mut graph_blocks = Vec::new();
        for ego in inputs.ego_graphs {
            let rendered = ego.render_compact();
            if !rendered.is_empty() {
                graph_blocks.push(rendered);
            }
        }
        if !graph_blocks.is_empty() {
            sections.push(format!(
                "## Code Dependency Context (RepoGraph)\n{}",
                graph_blocks.join("\n\n")
            ));
        }

        let text = sections.join("\n\n");
        let estimated_tokens = estimate_tokens(&text);
        AssembledPreamble {
            text,
            estimated_tokens,
            activated_decision_ids: activated_ids,
        }
    }

    /// Extract candidate symbol or path queries from a prompt and recent environment delta.
    pub fn extract_symbol_queries(prompt: &str, env_delta: &EnvDelta) -> Vec<String> {
        let mut queries = Vec::new();
        for token in prompt.split(|c: char| !c.is_alphanumeric() && c != '_' && c != ':' && c != '.' && c != '/') {
            let t = token.trim_matches(|c: char| c == ':' || c == '.');
            if t.contains("::") || t.ends_with(".rs") || (t.len() >= 5 && t.chars().any(|c| c.is_uppercase() || c == '_')) {
                if !queries.iter().any(|q| q == t) {
                    queries.push(t.to_string());
                }
            }
            if queries.len() >= 3 {
                break;
            }
        }
        for edit in &env_delta.out_of_band_edits {
            if queries.len() >= 4 {
                break;
            }
            if !queries.iter().any(|q| q == &edit.file_path) {
                queries.push(edit.file_path.clone());
            }
        }
        queries
    }
}

/// Fast approximate token estimator (~4 chars per token).
pub fn estimate_tokens(text: &str) -> u64 {
    ((text.len() as u64) + 3) / 4
}
