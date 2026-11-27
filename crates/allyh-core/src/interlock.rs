use allyh_tools::ToolExecutionRecord;

use crate::harness::CompletionInterlockPolicy;

/// Decision emitted by [`CompletionInterlock`] when the model tries to finish a turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterlockDecision {
    Allow,
    Reject { reason: String, reminder: String },
}

/// Execution-verified completion interlock state machine for AllyH's tool loop.
///
/// Directly inspired by the P5/P8 `ExecutionGatingInterlock` case study in arXiv:2611.18856
/// Appendix F.2: intercepts premature completion after failed tool executions or unverified
/// file edits, while bounding consecutive rejections (`max_consecutive_rejections`) so a done
/// task is never locked out.
#[derive(Debug, Clone)]
pub struct CompletionInterlock {
    policy: CompletionInterlockPolicy,
    consecutive_rejections: usize,
    total_tool_calls: usize,
    edited_since_verification: bool,
    last_round_had_error: bool,
    last_error_header: Option<String>,
    quality_notice_emitted: bool,
    budget_notice_emitted: bool,
}

impl CompletionInterlock {
    pub fn new(policy: CompletionInterlockPolicy) -> Self {
        Self {
            policy,
            consecutive_rejections: 0,
            total_tool_calls: 0,
            edited_since_verification: false,
            last_round_had_error: false,
            last_error_header: None,
            quality_notice_emitted: false,
            budget_notice_emitted: false,
        }
    }

    pub fn consecutive_rejections(&self) -> usize {
        self.consecutive_rejections
    }

    /// Observe all tool executions performed in a single round of the agent loop.
    pub fn observe_tool_round(&mut self, records: &[ToolExecutionRecord]) {
        if records.is_empty() {
            return;
        }
        self.total_tool_calls += records.len();
        self.last_round_had_error = false;
        self.last_error_header = None;

        for rec in records {
            if !rec.succeeded {
                self.last_round_had_error = true;
                self.last_error_header = Some(rec.compact_header.clone());
            }
            match rec.tool_name.as_str() {
                "edit_by_hash" if rec.succeeded => {
                    // Check if automatic post-edit verification already passed cleanly
                    // (supports both `verification` and legacy `cargo_check` payload keys).
                    let auto_passed = rec
                        .output_json
                        .get("verification")
                        .or_else(|| rec.output_json.get("cargo_check"))
                        .and_then(|c| c.get("success"))
                        .and_then(|b| b.as_bool())
                        .unwrap_or(false);
                    if !auto_passed {
                        self.edited_since_verification = true;
                    }
                }
                "run_verification" if rec.succeeded => {
                    self.edited_since_verification = false;
                }
                _ => {}
            }
        }
    }

    /// Emit an optional one-shot remaining-budget reminder once the round threshold is reached.
    pub fn budget_reminder_if_due(
        &mut self,
        current_round_zero_based: usize,
        max_rounds: usize,
    ) -> Option<String> {
        if !self.policy.enabled || self.budget_notice_emitted || max_rounds == 0 {
            return None;
        }
        let ratio = self.policy.budget_reminder_ratio?;
        let progress = (current_round_zero_based + 1) as f32 / max_rounds as f32;
        if progress >= ratio {
            self.budget_notice_emitted = true;
            let remaining = max_rounds.saturating_sub(current_round_zero_based + 1);
            return Some(format!(
                "[CompletionInterlock BUDGET NOTICE]: You have {remaining} tool round(s) remaining out of {max_rounds}. Prioritize verification and final deliverable checks."
            ));
        }
        None
    }

    /// Evaluate whether the model may terminate the turn now, or must continue with a prescriptive notice.
    pub fn check_completion(&mut self, _proposed_final_text: &str) -> InterlockDecision {
        if !self.policy.enabled {
            return InterlockDecision::Allow;
        }
        if self.consecutive_rejections >= self.policy.max_consecutive_rejections {
            return InterlockDecision::Allow;
        }

        // Case A: last tool round failed with an unrecovered error.
        if self.policy.reject_on_recent_tool_error && self.last_round_had_error {
            self.consecutive_rejections += 1;
            let err_hint = self
                .last_error_header
                .as_deref()
                .unwrap_or("recent tool invocation failed");
            return InterlockDecision::Reject {
                reason: "unrecovered_tool_error".to_string(),
                reminder: format!(
                    "[CompletionInterlock REJECT ({}/{})]: The previous tool round failed ({err_hint}). Inspect or fix the error and re-verify before completing.",
                    self.consecutive_rejections, self.policy.max_consecutive_rejections
                ),
            };
        }

        // Case B: edits were applied without a passing verification check.
        if self.policy.require_verification_after_edits && self.edited_since_verification {
            self.consecutive_rejections += 1;
            return InterlockDecision::Reject {
                reason: "unverified_edits".to_string(),
                reminder: format!(
                    "[CompletionInterlock REJECT ({}/{})]: Files were modified via `edit_by_hash` without a passing verification check. Run `run_verification` before declaring completion.",
                    self.consecutive_rejections, self.policy.max_consecutive_rejections
                ),
            };
        }

        // Case C: optional one-shot soft quality/optimality audit gate (only after active tool work).
        if self.total_tool_calls > 0
            && !self.quality_notice_emitted
            && let Some(notice) = self
                .policy
                .quality_audit_notice
                .as_deref()
                .filter(|s| !s.trim().is_empty())
        {
            self.quality_notice_emitted = true;
            self.consecutive_rejections += 1;
            return InterlockDecision::Reject {
                reason: "quality_audit_gate".to_string(),
                reminder: format!(
                    "[CompletionInterlock QUALITY AUDIT ({}/{})]: {}",
                    self.consecutive_rejections,
                    self.policy.max_consecutive_rejections,
                    notice.trim()
                ),
            };
        }

        InterlockDecision::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_tool(name: &str, succeeded: bool, auto_pass: Option<bool>) -> ToolExecutionRecord {
        let output_json = match auto_pass {
            Some(ok) => serde_json::json!({ "cargo_check": { "success": ok } }),
            None => serde_json::json!({}),
        };
        ToolExecutionRecord {
            tool_name: name.to_string(),
            compact_header: format!("{name} -> {}", if succeeded { "OK" } else { "ERR" }),
            succeeded,
            output_text: String::new(),
            output_json,
        }
    }

    #[test]
    fn test_interlock_rejects_unverified_edit_and_respects_max_cap() {
        let mut interlock = CompletionInterlock::new(CompletionInterlockPolicy {
            enabled: true,
            max_consecutive_rejections: 2,
            require_verification_after_edits: true,
            reject_on_recent_tool_error: true,
            quality_audit_notice: Some("Verify edge cases and optimality.".into()),
            budget_reminder_ratio: Some(0.75),
        });

        // Edit without auto cargo check -> completion rejected
        interlock.observe_tool_round(&[mock_tool("edit_by_hash", true, None)]);
        assert!(matches!(
            interlock.check_completion("Done!"),
            InterlockDecision::Reject { ref reason, .. } if reason == "unverified_edits"
        ));

        // Second rejection hits cap (2/2), third attempt is ALWAYS allowed (no deadlock)
        assert!(matches!(
            interlock.check_completion("Still done"),
            InterlockDecision::Reject { .. }
        ));
        assert_eq!(interlock.check_completion("Final"), InterlockDecision::Allow);
    }

    #[test]
    fn test_interlock_allows_after_verification_and_emits_single_quality_audit() {
        let mut interlock = CompletionInterlock::new(CompletionInterlockPolicy {
            enabled: true,
            max_consecutive_rejections: 3,
            require_verification_after_edits: true,
            reject_on_recent_tool_error: true,
            quality_audit_notice: Some("Confirm solution quality.".into()),
            budget_reminder_ratio: Some(0.5),
        });

        interlock.observe_tool_round(&[mock_tool("edit_by_hash", true, None)]);
        interlock.observe_tool_round(&[mock_tool("run_verification", true, None)]);

        // First stop triggers one-shot quality audit
        assert!(matches!(
            interlock.check_completion("All verified."),
            InterlockDecision::Reject { ref reason, .. } if reason == "quality_audit_gate"
        ));
        // Second stop is allowed
        assert_eq!(
            interlock.check_completion("Confirmed optimal."),
            InterlockDecision::Allow
        );

        // Budget reminder fires once at >= 50%
        assert!(interlock.budget_reminder_if_due(2, 6).is_some());
        assert!(interlock.budget_reminder_if_due(3, 6).is_none());
    }
}
