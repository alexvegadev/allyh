use std::collections::HashSet;

use allyh_core::{HarnessSpec, HybridRouter, ModelProfile, ModelProviderKind};
use allyh_memory::{SIM_PROMPT_RULES, SimLedger, SimMemoryMode};
use async_trait::async_trait;
use rig_core::message::Message;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::eval::EvalFeedback;
use crate::manifest::CandidateProposal;

/// Stage-1 pre-execution screening decision for a candidate proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreeningDecision {
    pub sample_id: String,
    pub approved: bool,
    pub rationale: String,
}

/// Stage-2 post-execution selection verdict (`H_{t+1}` or retain `H_t`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionVerdict {
    pub accepted_sample_id: Option<String>,
    pub rationale: String,
}

/// Pair of candidate proposal + its measured evolve-set feedback `F_t^C`.
#[derive(Debug, Clone)]
pub struct EvaluatedCandidate {
    pub proposal: CandidateProposal,
    pub harness: HarnessSpec,
    pub harness_version_id: String,
    pub feedback: EvalFeedback,
}

/// Two-stage Selector `S(H_t, F_t, C_t, F_t^C; M_t)` (arXiv:2611.18856 §3.1–3.2).
#[async_trait]
pub trait SimSelector: Send + Sync {
    /// Stage 1: Screen candidates before paying evaluation rollout cost.
    async fn screen_candidates(
        &self,
        current_harness: &HarnessSpec,
        baseline_feedback: &EvalFeedback,
        candidates: &[CandidateProposal],
        ledger: &SimLedger,
        memory_mode: SimMemoryMode,
    ) -> Result<Vec<ScreeningDecision>>;

    /// Stage 2: Choose among evaluated candidates (or reject all -> `None`).
    async fn select_winner(
        &self,
        current_harness: &HarnessSpec,
        baseline_feedback: &EvalFeedback,
        evaluated: &[EvaluatedCandidate],
        ledger: &SimLedger,
        memory_mode: SimMemoryMode,
    ) -> Result<SelectionVerdict>;
}

/// LLM-backed two-stage SIM-conditioned selector with deterministic offline fallback.
#[derive(Clone)]
pub struct LlmSimSelector {
    router: HybridRouter,
    optimizer_profile: ModelProfile,
}

impl LlmSimSelector {
    pub fn new(router: HybridRouter, optimizer_profile: ModelProfile) -> Self {
        Self {
            router,
            optimizer_profile,
        }
    }

    pub fn build_system_prompt(ledger: &SimLedger, memory_mode: SimMemoryMode) -> String {
        let rules = SIM_PROMPT_RULES.join("\n");
        let sim_block = ledger.render_for_prompt(memory_mode);
        format!(
            "You are AllyH's Self-Improvement Harness Selector.\n\
             Weigh performance gains, complexity, domain generality, and regression risk against the complete SIM record.\n\n\
             ## SIM Usage Rules (arXiv:2611.18856 Appendix D.2)\n{rules}\n\n\
             {sim_block}"
        )
    }
}

#[async_trait]
impl SimSelector for LlmSimSelector {
    async fn screen_candidates(
        &self,
        current_harness: &HarnessSpec,
        baseline_feedback: &EvalFeedback,
        candidates: &[CandidateProposal],
        ledger: &SimLedger,
        memory_mode: SimMemoryMode,
    ) -> Result<Vec<ScreeningDecision>> {
        // First apply deterministic structural validity & deduplication gate.
        let mut valid_candidates = Vec::new();
        let mut decisions = Vec::new();
        let mut seen_hashes = HashSet::new();
        seen_hashes.insert(current_harness.sha256_hash());

        for cand in candidates {
            match current_harness.apply_merge_patch(&cand.patch) {
                Ok(patched) => {
                    let hash = patched.sha256_hash();
                    if seen_hashes.insert(hash) {
                        valid_candidates.push(cand.clone());
                    } else {
                        decisions.push(ScreeningDecision {
                            sample_id: cand.sample_id.clone(),
                            approved: false,
                            rationale: "no-op or duplicate of another candidate harness".to_string(),
                        });
                    }
                }
                Err(err) => {
                    decisions.push(ScreeningDecision {
                        sample_id: cand.sample_id.clone(),
                        approved: false,
                        rationale: format!("invalid HarnessSpec patch: {err}"),
                    });
                }
            }
        }

        if valid_candidates.is_empty() {
            return Ok(decisions);
        }

        if self.optimizer_profile.provider == ModelProviderKind::OfflineMock {
            for cand in valid_candidates {
                decisions.push(ScreeningDecision {
                    sample_id: cand.sample_id,
                    approved: true,
                    rationale: "validated structurally; approved for evolve evaluation".to_string(),
                });
            }
            return Ok(decisions);
        }

        let system = format!(
            "{}\n\nReturn ONLY a JSON array of `{{\"sample_id\":\"s1\",\"approved\":true|false,\"rationale\":\"...\"}}`.",
            Self::build_system_prompt(ledger, memory_mode)
        );
        let user = format!(
            "Baseline Evolve Feedback:\n{}\n\nProposed Candidates:\n{}",
            baseline_feedback.render_summary(),
            serde_json::to_string_pretty(&valid_candidates).unwrap_or_default()
        );

        let mut opt_router = self.router.clone();
        opt_router.set_primary(self.optimizer_profile.clone());
        if let Ok(resp) = opt_router
            .run_primary(&system, &[Message::user(user)], &[], None)
            .await
            && let Some(arr) = extract_json_array(&resp.text())
            && let Ok(parsed) = serde_json::from_str::<Vec<ScreeningDecision>>(arr)
            && !parsed.is_empty()
        {
            decisions.extend(parsed);
            return Ok(decisions);
        }

        for cand in valid_candidates {
            decisions.push(ScreeningDecision {
                sample_id: cand.sample_id,
                approved: true,
                rationale: "fallback screen approval".to_string(),
            });
        }
        Ok(decisions)
    }

    async fn select_winner(
        &self,
        current_harness: &HarnessSpec,
        baseline_feedback: &EvalFeedback,
        evaluated: &[EvaluatedCandidate],
        ledger: &SimLedger,
        memory_mode: SimMemoryMode,
    ) -> Result<SelectionVerdict> {
        if evaluated.is_empty() {
            return Ok(SelectionVerdict {
                accepted_sample_id: None,
                rationale: "no candidates survived screening".to_string(),
            });
        }

        if self.optimizer_profile.provider == ModelProviderKind::OfflineMock {
            return HeuristicSimSelector
                .select_winner(current_harness, baseline_feedback, evaluated, ledger, memory_mode)
                .await;
        }

        let summaries: Vec<serde_json::Value> = evaluated
            .iter()
            .map(|c| {
                let (fixed, missed) = EvalFeedback::evaluate_predictions(
                    baseline_feedback,
                    &c.feedback,
                    &c.proposal.predicted,
                );
                serde_json::json!({
                    "sample_id": c.proposal.sample_id,
                    "problem_ref": c.proposal.problem_ref,
                    "what": c.proposal.what,
                    "score": c.feedback.score,
                    "delta": c.feedback.score - baseline_feedback.score,
                    "passed": c.feedback.passed_rollouts,
                    "total": c.feedback.total_rollouts,
                    "predicted": c.proposal.predicted,
                    "fixed": fixed,
                    "missed": missed,
                })
            })
            .collect();

        let system = format!(
            "{}\n\nSelect at most ONE winning candidate or `null` if none improves on baseline safely.\n\
             Return ONLY valid JSON: `{{\"accepted_sample_id\":\"s1\"|null,\"rationale\":\"...\"}}`.",
            Self::build_system_prompt(ledger, memory_mode)
        );
        let user = format!(
            "Baseline Evolve Score: {:.2}\n{}\n\nEvaluated Candidates:\n{}",
            baseline_feedback.score,
            baseline_feedback.render_summary(),
            serde_json::to_string_pretty(&summaries).unwrap_or_default()
        );

        let mut opt_router = self.router.clone();
        opt_router.set_primary(self.optimizer_profile.clone());
        if let Ok(resp) = opt_router
            .run_primary(&system, &[Message::user(user)], &[], None)
            .await
            && let Some(obj) = extract_json_object(&resp.text())
            && let Ok(verdict) = serde_json::from_str::<SelectionVerdict>(obj)
        {
            if let Some(ref sid) = verdict.accepted_sample_id {
                if evaluated.iter().any(|c| c.proposal.sample_id == *sid) {
                    return Ok(verdict);
                }
            } else {
                return Ok(verdict);
            }
        }

        HeuristicSimSelector
            .select_winner(current_harness, baseline_feedback, evaluated, ledger, memory_mode)
            .await
    }
}

/// Deterministic score-maximizing selector with strict improvement threshold (offline fallback & tests).
#[derive(Debug, Default, Clone)]
pub struct HeuristicSimSelector;

#[async_trait]
impl SimSelector for HeuristicSimSelector {
    async fn screen_candidates(
        &self,
        current_harness: &HarnessSpec,
        _baseline_feedback: &EvalFeedback,
        candidates: &[CandidateProposal],
        _ledger: &SimLedger,
        _memory_mode: SimMemoryMode,
    ) -> Result<Vec<ScreeningDecision>> {
        let mut seen = HashSet::new();
        seen.insert(current_harness.sha256_hash());
        let mut out = Vec::new();
        for cand in candidates {
            match current_harness.apply_merge_patch(&cand.patch) {
                Ok(patched) if seen.insert(patched.sha256_hash()) => {
                    out.push(ScreeningDecision {
                        sample_id: cand.sample_id.clone(),
                        approved: true,
                        rationale: "valid unique candidate".to_string(),
                    });
                }
                Ok(_) => out.push(ScreeningDecision {
                    sample_id: cand.sample_id.clone(),
                    approved: false,
                    rationale: "duplicate or no-op patch".to_string(),
                }),
                Err(err) => out.push(ScreeningDecision {
                    sample_id: cand.sample_id.clone(),
                    approved: false,
                    rationale: format!("invalid patch: {err}"),
                }),
            }
        }
        Ok(out)
    }

    async fn select_winner(
        &self,
        _current_harness: &HarnessSpec,
        baseline_feedback: &EvalFeedback,
        evaluated: &[EvaluatedCandidate],
        _ledger: &SimLedger,
        _memory_mode: SimMemoryMode,
    ) -> Result<SelectionVerdict> {
        let best = evaluated
            .iter()
            .filter(|c| c.feedback.score > baseline_feedback.score + 1e-4)
            .max_by(|a, b| {
                a.feedback
                    .score
                    .partial_cmp(&b.feedback.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        match best {
            Some(winner) => Ok(SelectionVerdict {
                accepted_sample_id: Some(winner.proposal.sample_id.clone()),
                rationale: format!(
                    "evolve score improved {:.2} -> {:.2} (+{:.2})",
                    baseline_feedback.score,
                    winner.feedback.score,
                    winner.feedback.score - baseline_feedback.score
                ),
            }),
            None => Ok(SelectionVerdict {
                accepted_sample_id: None,
                rationale: format!(
                    "no candidate exceeded baseline evolve score {:.2}; keeping H_t unchanged",
                    baseline_feedback.score
                ),
            }),
        }
    }
}

fn extract_json_array(raw: &str) -> Option<&str> {
    let start = raw.find('[')?;
    let end = raw.rfind(']')?;
    (end >= start).then(|| &raw[start..=end])
}

fn extract_json_object(raw: &str) -> Option<&str> {
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    (end >= start).then(|| &raw[start..=end])
}
