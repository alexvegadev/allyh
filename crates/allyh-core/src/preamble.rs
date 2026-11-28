use allyh_env::{EgoGraph, EnvDelta};
use allyh_memory::{ScoredDecision, SkillRecord, TaskRecord, TaskStatus};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::gate::ControlGateState;
use crate::harness::HarnessSpec;

/// Assembled cache-aware system preamble split into a byte-stable prefix and a per-turn volatile suffix.
///
/// Keeping operating rules, harness guidelines, pinned architectural invariants, and the sorted skills
/// catalog inside [`stable_prefix`](Self::stable_prefix) maximizes provider prompt-cache hit rates
/// (Anthropic `cache_control`, OpenAI `prompt_cache_key`, Gemini prefix/context caching) across turns
/// while live [`EnvDelta`], active tasks, and per-turn RepoGraph neighborhoods stay in
/// [`volatile_suffix`](Self::volatile_suffix).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssembledPreamble {
    /// Byte-invariant prefix across turns when only `EnvDelta`, tasks, or turn-local graphs change.
    pub stable_prefix: String,
    /// Per-turn dynamic context (`ControlGate`, `EnvDelta`, tasks, non-pinned memories, ego-graphs).
    pub volatile_suffix: String,
    /// Full concatenated preamble (`stable_prefix` + `"\n\n"` + `volatile_suffix`).
    pub text: String,
    pub stable_prefix_tokens: u64,
    pub volatile_suffix_tokens: u64,
    pub estimated_tokens: u64,
    /// Deterministic SHA-256 hex digest of `stable_prefix` used for prompt-cache routing keys.
    pub stable_prefix_hash: String,
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

/// Builds the cache-aware per-turn preamble combining `HarnessSpec` policies, `EnvDelta`, active `tasks`,
/// top-k active `architectural_decisions` (`invalid_at IS NULL`), and 1-hop `RepoGraph` neighborhoods.
pub struct DynamicPreambleAssembler;

impl DynamicPreambleAssembler {
    pub fn assemble(inputs: &PreambleInputs<'_>) -> AssembledPreamble {
        let mut stable_sections = Vec::new();
        let mut volatile_sections = Vec::new();
        let harness = inputs.harness;

        // =====================================================================
        // STABLE PREFIX (byte-invariant across turns unless rules/pins/catalog change)
        // =====================================================================
        let mut header_lines = vec![
            "# AllyH Coherence-First Agent Harness".to_string(),
            format!(
                "Repository Partition: `repo_id={}` | `scope_id={}`",
                &inputs.repo_id[..inputs.repo_id.len().min(12)],
                inputs.scope_id
            ),
            "Operating Rules:".to_string(),
        ];
        for rule in &harness.preamble.base_operating_rules {
            header_lines.push(format!("- {rule}"));
        }
        if !harness.tools.stale_hash_hint.trim().is_empty() {
            header_lines.push(format!("- {}", harness.tools.stale_hash_hint.trim()));
        }
        stable_sections.push(header_lines.join("\n"));

        if !harness.preamble.verification_guidelines.is_empty() {
            let mut lines = vec!["## Verification Guidelines".to_string()];
            for g in &harness.preamble.verification_guidelines {
                lines.push(format!("- {g}"));
            }
            stable_sections.push(lines.join("\n"));
        }

        if !harness.preamble.domain_guidance.is_empty() {
            let mut lines = vec!["## Domain Guidance".to_string()];
            for g in &harness.preamble.domain_guidance {
                lines.push(format!("- {g}"));
            }
            stable_sections.push(lines.join("\n"));
        }

        if let Some(fmt_hint) = harness
            .model_adaptation
            .output_format_guidance
            .as_deref()
            .filter(|s| !s.trim().is_empty())
        {
            stable_sections.push(format!("## Model Output Guidelines\n{}", fmt_hint.trim()));
        }

        // Pinned architectural invariants stay in the stable prefix (sorted by id for byte invariance).
        let mut activated_ids = Vec::new();
        let mut pinned_memories: Vec<&ScoredDecision> =
            inputs.memories.iter().filter(|m| m.pinned).collect();
        pinned_memories.sort_by(|a, b| a.decision.id.cmp(&b.decision.id));
        if !pinned_memories.is_empty() {
            let mut lines = vec!["## Pinned Architectural Invariants (LanceDB)".to_string()];
            for scored in &pinned_memories {
                activated_ids.push(scored.decision.id.clone());
                let paths = if scored.decision.affected_paths.is_empty() {
                    String::new()
                } else {
                    format!(" (paths: {})", scored.decision.affected_paths.join(", "))
                };
                lines.push(format!(
                    "- [{}] [PINNED] {} — Rationale: {}{paths}",
                    scored.decision.topic, scored.decision.rule_statement, scored.decision.rationale
                ));
            }
            stable_sections.push(lines.join("\n"));
        }

        // Compact skills catalog is stable across turns (sorted by skill name for byte invariance).
        if !inputs.available_skills.is_empty() {
            let mut sorted_skills: Vec<&SkillRecord> = inputs.available_skills.iter().collect();
            sorted_skills.sort_by(|a, b| a.name.cmp(&b.name));
            let mut lines = vec!["## Available Skills (LanceDB)".to_string()];
            for skill in sorted_skills {
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
            stable_sections.push(lines.join("\n"));
        }

        // =====================================================================
        // VOLATILE SUFFIX (changes per turn: gate state, summary, EnvDelta, tasks, graphs)
        // =====================================================================
        volatile_sections.push(format!("Control Gate Mode: `{}`", inputs.gate_state.label()));

        if harness.preamble.include_rolling_summary
            && let Some(summary) = inputs.rolling_summary
            && !summary.trim().is_empty()
        {
            volatile_sections.push(format!("## Rolling Episodic Summary\n{}", summary.trim()));
        }

        // 1. Live Environment Delta
        volatile_sections.push(inputs.env_delta.render_for_preamble());

        // 2. Per-turn Retrieved Architectural Invariants (non-pinned, invalid_at IS NULL)
        let dynamic_memories: Vec<&ScoredDecision> =
            inputs.memories.iter().filter(|m| !m.pinned).collect();
        if !dynamic_memories.is_empty() {
            let mut lines = vec!["## Active Architectural Invariants (LanceDB)".to_string()];
            for scored in dynamic_memories {
                activated_ids.push(scored.decision.id.clone());
                let paths = if scored.decision.affected_paths.is_empty() {
                    String::new()
                } else {
                    format!(" (paths: {})", scored.decision.affected_paths.join(", "))
                };
                lines.push(format!(
                    "- [{}] {} — Rationale: {}{paths}",
                    scored.decision.topic, scored.decision.rule_statement, scored.decision.rationale
                ));
            }
            volatile_sections.push(lines.join("\n"));
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
            volatile_sections.push(lines.join("\n"));
        }

        // 4. Expanded Active Skill Instructions (progressive disclosure per turn)
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
            volatile_sections.push(blocks.join("\n\n"));
        }

        // 5. Compact 1-hop / 2-hop RepoGraph Neighborhoods
        let mut graph_blocks = Vec::new();
        for ego in inputs.ego_graphs {
            let rendered = ego.render_compact();
            if !rendered.is_empty() {
                graph_blocks.push(rendered);
            }
        }
        if !graph_blocks.is_empty() {
            volatile_sections.push(format!(
                "## Code Dependency Context (RepoGraph)\n{}",
                graph_blocks.join("\n\n")
            ));
        }

        let stable_prefix = stable_sections.join("\n\n");
        let volatile_suffix = volatile_sections.join("\n\n");
        let text = if volatile_suffix.is_empty() {
            stable_prefix.clone()
        } else {
            format!("{stable_prefix}\n\n{volatile_suffix}")
        };

        let stable_prefix_tokens = estimate_tokens(&stable_prefix);
        let volatile_suffix_tokens = estimate_tokens(&volatile_suffix);
        let estimated_tokens = estimate_tokens(&text);

        let mut hasher = Sha256::new();
        hasher.update(stable_prefix.as_bytes());
        let stable_prefix_hash = hex::encode(hasher.finalize());

        AssembledPreamble {
            stable_prefix,
            volatile_suffix,
            text,
            stable_prefix_tokens,
            volatile_suffix_tokens,
            estimated_tokens,
            stable_prefix_hash,
            activated_decision_ids: activated_ids,
        }
    }

    /// Extract candidate symbol or path queries from a prompt and recent environment delta.
    pub fn extract_symbol_queries(prompt: &str, env_delta: &EnvDelta) -> Vec<String> {
        let mut queries = Vec::new();
        for token in prompt.split(|c: char| {
            !c.is_alphanumeric() && c != '_' && c != ':' && c != '.' && c != '/' && c != '-'
        }) {
            let t = token.trim_matches(|c: char| c == ':' || c == '.' || c == '-');
            if t.is_empty() {
                continue;
            }
            let has_code_ext = [
                ".rs", ".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs", ".py", ".pyi",
                ".go", ".toml", ".json", ".mod",
            ]
            .iter()
            .any(|ext| t.ends_with(ext));
            let is_member_expr = (t.contains("::") || (t.contains('.') && !t.starts_with("http")))
                && t.len() >= 4;
            let is_ident = t.len() >= 5 && t.chars().any(|c| c.is_uppercase() || c == '_');

            if has_code_ext || is_member_expr || is_ident {
                if !queries.iter().any(|q| q == t) {
                    queries.push(t.to_string());
                }
            }
            if queries.len() >= 4 {
                break;
            }
        }
        for edit in &env_delta.out_of_band_edits {
            if queries.len() >= 5 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use allyh_env::{GitState, OutOfBandEdit};
    use allyh_memory::ArchitecturalDecision;
    use chrono::Utc;

    fn sample_pinned_memory() -> ScoredDecision {
        ScoredDecision {
            decision: ArchitecturalDecision {
                id: "dec-pinned-1".into(),
                repo_id: "repo-1".into(),
                scope_id: "default".into(),
                topic: "errors".into(),
                rule_statement: "Use thiserror in core crates".into(),
                rationale: "Structured error matching".into(),
                affected_paths: vec!["crates/allyh-core".into()],
                vector: Vec::new(),
                valid_at: Utc::now(),
                invalid_at: None,
                superseded_by: None,
            },
            score: 2.0,
            pinned: true,
        }
    }

    fn sample_skill() -> SkillRecord {
        SkillRecord {
            id: "repo-1:default:cargo-verify".into(),
            repo_id: "repo-1".into(),
            scope_id: "default".into(),
            name: "cargo-verify".into(),
            description: "Run cargo test after edits".into(),
            instructions: "Always run cargo test --workspace".into(),
            triggers: vec!["test".into()],
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    #[test]
    fn stable_prefix_is_byte_invariant_across_env_delta_and_task_changes() {
        let harness = HarnessSpec::default_seed();
        let pinned = vec![sample_pinned_memory()];
        let skills = vec![sample_skill()];

        let delta_turn_1 = EnvDelta {
            git: GitState {
                branch: "main".into(),
                dirty_files: vec!["src/a.rs".into()],
                ..GitState::default()
            },
            build_status: Some("cargo check: PASS".into()),
            detected_languages: vec!["Rust".into()],
            detected_toolchains: vec!["Rust (cargo)".into()],
            ..EnvDelta::default()
        };

        let delta_turn_2 = EnvDelta {
            git: GitState {
                branch: "main".into(),
                dirty_files: vec!["src/a.rs".into(), "src/b.rs".into()],
                ..GitState::default()
            },
            out_of_band_edits: vec![OutOfBandEdit {
                file_path: "src/b.rs".into(),
                previous_hash: Some("111".into()),
                current_hash: Some("222".into()),
                diff_summary: "+4/-1 lines".into(),
            }],
            build_status: Some("cargo check: FAIL (1 error)".into()),
            detected_languages: vec!["Rust".into(), "TypeScript".into()],
            detected_toolchains: vec!["Rust (cargo)".into()],
            ..EnvDelta::default()
        };

        let task_turn_2 = vec![TaskRecord {
            id: "task-1".into(),
            repo_id: "repo-1".into(),
            scope_id: "default".into(),
            parent_id: None,
            title: "Fix compile error in src/b.rs".into(),
            acceptance_criteria: vec!["cargo check passes".into()],
            status: TaskStatus::InProgress,
            last_error_trace: Some("E0308 mismatched types".into()),
            updated_at: Utc::now(),
        }];

        let p1 = DynamicPreambleAssembler::assemble(&PreambleInputs {
            repo_id: "repo-1234567890abcdef",
            scope_id: "default",
            gate_state: &ControlGateState::Discovery,
            harness: &harness,
            env_delta: &delta_turn_1,
            tasks: &[],
            memories: &pinned,
            available_skills: &skills,
            active_skills: &[],
            ego_graphs: &[],
            rolling_summary: Some("Turn 1 summary"),
        });

        let p2 = DynamicPreambleAssembler::assemble(&PreambleInputs {
            repo_id: "repo-1234567890abcdef",
            scope_id: "default",
            gate_state: &ControlGateState::Execution {
                active_task_id: Some("task-1".into()),
            },
            harness: &harness,
            env_delta: &delta_turn_2,
            tasks: &task_turn_2,
            memories: &pinned,
            available_skills: &skills,
            active_skills: &skills,
            ego_graphs: &[],
            rolling_summary: Some("Turn 2 updated summary with new error"),
        });

        // Core Milestone 5 guarantee: byte-for-byte stable prefix invariance!
        assert_eq!(p1.stable_prefix.as_bytes(), p2.stable_prefix.as_bytes());
        assert_eq!(p1.stable_prefix_hash, p2.stable_prefix_hash);
        assert!(p1.text.starts_with(&p1.stable_prefix));
        assert!(p2.text.starts_with(&p2.stable_prefix));

        // Meanwhile volatile suffix reflects the live workspace & task state delta.
        assert_ne!(p1.volatile_suffix, p2.volatile_suffix);
        assert!(p2.volatile_suffix.contains("E0308 mismatched types"));
        assert!(p2.volatile_suffix.contains("src/b.rs"));
        assert!(p1.stable_prefix.contains("Use thiserror in core crates"));
        assert!(p1.stable_prefix.contains("cargo-verify"));
    }
}
