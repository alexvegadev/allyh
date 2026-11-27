use allyh_memory::HarnessVersionRecord;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::error::{CoreError, Result};

/// Prompt and preamble assembly configuration managed by the harness.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreamblePolicy {
    pub base_operating_rules: Vec<String>,
    #[serde(default)]
    pub verification_guidelines: Vec<String>,
    #[serde(default)]
    pub domain_guidance: Vec<String>,
    #[serde(default = "default_true")]
    pub include_rolling_summary: bool,
}

/// Memory, skill, and RepoGraph retrieval budgets managed by the harness.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetrievalPolicy {
    pub decision_top_k: usize,
    pub fallback_decision_count: usize,
    pub skill_match_top_k: usize,
    pub ego_graph_hops: usize,
    pub max_symbol_queries: usize,
}

/// Non-destructive conversation compaction thresholds.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompactionPolicy {
    pub keep_recent_units: usize,
    pub compaction_ratio: f64,
    pub max_rollup_chars: usize,
}

/// ACI tool execution and output-budget controls.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolPolicy {
    pub max_tool_rounds: usize,
    /// Maximum characters retained from the head of oversized tool outputs (0 disables truncation).
    pub output_head_chars: usize,
    /// Maximum characters retained from the tail of oversized tool outputs (0 disables truncation).
    pub output_tail_chars: usize,
    #[serde(alias = "auto_cargo_check")]
    pub auto_verify: bool,
    #[serde(default)]
    pub stale_hash_hint: String,
}

/// Execution-verified completion interlock (inspired by P5 & P8 in arXiv:2611.18856 Appendix F.2).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompletionInterlockPolicy {
    pub enabled: bool,
    /// Hard upper bound on consecutive rejections per turn to guarantee no completion deadlocks.
    pub max_consecutive_rejections: usize,
    /// Reject premature completion if files were edited with no subsequent passing verification.
    pub require_verification_after_edits: bool,
    /// Reject premature completion immediately after a failed tool call without recovery.
    pub reject_on_recent_tool_error: bool,
    /// Optional one-shot soft audit notice injected before final completion (e.g., optimality/quality check).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality_audit_notice: Option<String>,
    /// Optional step-budget ratio (0.0..=1.0) at which to inject a single remaining-budget reminder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_reminder_ratio: Option<f32>,
}

/// Setting-dependent model formatting/repair hints reassessed during cross-model CRSI (Table 8).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelAdaptationPolicy {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format_guidance: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parser_repair_notes: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_policy_hint: Option<String>,
}

/// Versioned, serializable executable harness specification (`H_t`) for AllyH.
///
/// Captures the prompts, context/retrieval budgets, tool policies, completion interlocks,
/// and model-adaptation rules that CRSI + SIM can inspect, mutate, evaluate, and carry
/// across datasets and policy models without recompiling Rust.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HarnessSpec {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    pub preamble: PreamblePolicy,
    pub retrieval: RetrievalPolicy,
    pub compaction: CompactionPolicy,
    pub tools: ToolPolicy,
    pub completion_interlock: CompletionInterlockPolicy,
    #[serde(default)]
    pub model_adaptation: ModelAdaptationPolicy,
}

fn default_true() -> bool {
    true
}

fn default_schema_version() -> u32 {
    1
}

impl Default for HarnessSpec {
    fn default() -> Self {
        Self::default_seed()
    }
}

impl HarnessSpec {
    /// Seed harness `H_0` preserving AllyH's exact baseline behavior.
    pub fn default_seed() -> Self {
        Self {
            schema_version: 1,
            preamble: PreamblePolicy {
                base_operating_rules: vec![
                    "Enforce all active Architectural Invariants below without requiring in-repo Markdown plans.".to_string(),
                    "Always inspect symbols (`inspect_symbol`) to obtain fresh SHA-256 hashes before calling `edit_by_hash`.".to_string(),
                    "When a skill in the Available Skills catalog applies, call `get_skill` to load its full procedure; use `add_skill` to persist new reusable workflows.".to_string(),
                    "Work incrementally on the single active `in_progress` task and verify with `run_verification`.".to_string(),
                ],
                verification_guidelines: Vec::new(),
                domain_guidance: Vec::new(),
                include_rolling_summary: true,
            },
            retrieval: RetrievalPolicy {
                decision_top_k: 8,
                fallback_decision_count: 5,
                skill_match_top_k: 3,
                ego_graph_hops: 1,
                max_symbol_queries: 3,
            },
            compaction: CompactionPolicy {
                keep_recent_units: 6,
                compaction_ratio: 0.70,
                max_rollup_chars: 2400,
            },
            tools: ToolPolicy {
                max_tool_rounds: 6,
                output_head_chars: 0,
                output_tail_chars: 0,
                auto_verify: true,
                stale_hash_hint: String::new(),
            },
            completion_interlock: CompletionInterlockPolicy {
                enabled: false,
                max_consecutive_rejections: 3,
                require_verification_after_edits: true,
                reject_on_recent_tool_error: true,
                quality_audit_notice: None,
                budget_reminder_ratio: None,
            },
            model_adaptation: ModelAdaptationPolicy::default(),
        }
    }

    /// Validate bounds so evolved candidate patches cannot create degenerate or deadlocking harnesses.
    pub fn validate(&self) -> Result<()> {
        if self.retrieval.decision_top_k == 0 || self.retrieval.decision_top_k > 64 {
            return Err(CoreError::HarnessSpec(
                "retrieval.decision_top_k must be in 1..=64".to_string(),
            ));
        }
        if self.retrieval.ego_graph_hops > 3 {
            return Err(CoreError::HarnessSpec(
                "retrieval.ego_graph_hops must be <= 3".to_string(),
            ));
        }
        if self.compaction.keep_recent_units == 0 || self.compaction.keep_recent_units > 64 {
            return Err(CoreError::HarnessSpec(
                "compaction.keep_recent_units must be in 1..=64".to_string(),
            ));
        }
        if !(0.20..=0.95).contains(&self.compaction.compaction_ratio) {
            return Err(CoreError::HarnessSpec(
                "compaction.compaction_ratio must be in 0.20..=0.95".to_string(),
            ));
        }
        if self.tools.max_tool_rounds == 0 || self.tools.max_tool_rounds > 24 {
            return Err(CoreError::HarnessSpec(
                "tools.max_tool_rounds must be in 1..=24".to_string(),
            ));
        }
        if self.completion_interlock.max_consecutive_rejections > 5 {
            return Err(CoreError::HarnessSpec(
                "completion_interlock.max_consecutive_rejections must be <= 5 to prevent deadlocks"
                    .to_string(),
            ));
        }
        if let Some(ratio) = self.completion_interlock.budget_reminder_ratio
            && !(0.10..=0.95).contains(&ratio)
        {
            return Err(CoreError::HarnessSpec(
                "completion_interlock.budget_reminder_ratio must be in 0.10..=0.95".to_string(),
            ));
        }
        Ok(())
    }

    /// Deterministic JSON serialization + SHA-256 content digest.
    pub fn canonical_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn sha256_hash(&self) -> String {
        let json = serde_json::to_string(self).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(json.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Apply an RFC 7396 JSON Merge Patch onto a clone of this spec and validate the result.
    pub fn apply_merge_patch(&self, patch: &serde_json::Value) -> Result<Self> {
        let mut base = serde_json::to_value(self)?;
        json_merge_patch(&mut base, patch);
        let updated: Self = serde_json::from_value(base)
            .map_err(|e| CoreError::HarnessSpec(format!("invalid harness patch: {e}")))?;
        updated.validate()?;
        Ok(updated)
    }

    /// Wrap this spec into a persistable [`HarnessVersionRecord`].
    #[allow(clippy::too_many_arguments)]
    pub fn to_version_record(
        &self,
        repo_id: impl Into<String>,
        scope_id: impl Into<String>,
        parent_version_id: Option<String>,
        run_id: Option<String>,
        round: u32,
        label: impl Into<String>,
        policy_model: impl Into<String>,
        domain: impl Into<String>,
        is_active: bool,
    ) -> Result<HarnessVersionRecord> {
        self.validate()?;
        let spec_json = self.canonical_json()?;
        let sha256_hash = self.sha256_hash();
        Ok(HarnessVersionRecord {
            id: Uuid::new_v4().to_string(),
            repo_id: repo_id.into(),
            scope_id: scope_id.into(),
            parent_version_id,
            run_id,
            round,
            label: label.into(),
            spec_json,
            sha256_hash,
            policy_model: policy_model.into(),
            domain: domain.into(),
            is_active,
            created_at: Utc::now(),
        })
    }

    /// Parse a persisted [`HarnessVersionRecord`] back into a validated [`HarnessSpec`].
    pub fn from_version_record(record: &HarnessVersionRecord) -> Result<Self> {
        let spec: Self = serde_json::from_str(&record.spec_json)
            .map_err(|e| CoreError::HarnessSpec(format!("corrupt harness spec JSON: {e}")))?;
        spec.validate()?;
        Ok(spec)
    }

    /// Truncate oversized tool output using configured head + tail quotas (when non-zero).
    pub fn format_tool_output(&self, text: &str) -> String {
        let head = self.tools.output_head_chars;
        let tail = self.tools.output_tail_chars;
        if head == 0 && tail == 0 {
            return text.to_string();
        }
        let max_total = head.saturating_add(tail);
        if text.len() <= max_total || max_total == 0 {
            return text.to_string();
        }
        let head_cut = text
            .char_indices()
            .take_while(|(i, _)| *i < head)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        let tail_start_target = text.len().saturating_sub(tail);
        let tail_start = text
            .char_indices()
            .find(|(i, _)| *i >= tail_start_target)
            .map(|(i, _)| i)
            .unwrap_or(text.len());
        if head_cut >= tail_start {
            return text.to_string();
        }
        let omitted = tail_start - head_cut;
        format!(
            "{}\n... [truncated {omitted} chars by HarnessSpec head/tail budget] ...\n{}",
            &text[..head_cut],
            &text[tail_start..]
        )
    }
}

fn json_merge_patch(target: &mut serde_json::Value, patch: &serde_json::Value) {
    match (target, patch) {
        (serde_json::Value::Object(t_map), serde_json::Value::Object(p_map)) => {
            for (k, v) in p_map {
                if v.is_null() {
                    t_map.remove(k);
                } else if let Some(existing) = t_map.get_mut(k) {
                    json_merge_patch(existing, v);
                } else {
                    t_map.insert(k.clone(), v.clone());
                }
            }
        }
        (target_slot, patch_val) => {
            *target_slot = patch_val.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harness_spec_patch_validation_and_truncation() {
        let seed = HarnessSpec::default_seed();
        assert!(seed.validate().is_ok());

        let patch = serde_json::json!({
            "completion_interlock": {
                "enabled": true,
                "max_consecutive_rejections": 2,
                "quality_audit_notice": "Check optimality before stopping."
            },
            "tools": {
                "output_head_chars": 20,
                "output_tail_chars": 15
            }
        });
        let evolved = seed.apply_merge_patch(&patch).expect("valid merge patch");
        assert!(evolved.completion_interlock.enabled);
        assert_eq!(evolved.completion_interlock.max_consecutive_rejections, 2);
        assert_ne!(seed.sha256_hash(), evolved.sha256_hash());

        let long_out = "HEAD_SECTION_1234567890_MIDDLE_STUFF_HERE_TAIL_SECTION_987654321";
        let formatted = evolved.format_tool_output(long_out);
        assert!(formatted.contains("HEAD_SECTION"));
        assert!(formatted.contains("truncated"));
        assert!(formatted.contains("987654321"));

        // Invalid deadlock-prone rejection count is rejected by validate()
        let bad_patch = serde_json::json!({
            "completion_interlock": {
                "max_consecutive_rejections": 99
            }
        });
        assert!(seed.apply_merge_patch(&bad_patch).is_err());
    }
}
