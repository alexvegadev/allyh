use std::collections::HashSet;

use allyh_core::{HarnessSpec, HybridRouter, ModelProfile, ModelProviderKind};
use allyh_memory::{SIM_PROMPT_RULES, SimLedger, SimMemoryMode};
use async_trait::async_trait;
use rig_core::message::Message;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::eval::EvalFeedback;
use crate::manifest::CandidateProposal;

/// Weights controlling multi-objective ranking across pass rate, token budget, tool-round
/// efficiency, and wall-clock latency when selecting candidate [`HarnessSpec`] patches.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MultiObjectiveWeights {
    /// Weight on pass-rate delta (`score_cand - score_base`, on 0..=100 scale).
    pub pass_rate_weight: f32,
    /// Bonus/penalty points for relative token-budget reduction vs. baseline (clamped to `[-1, 1]`).
    pub token_efficiency_weight: f32,
    /// Bonus/penalty points for relative tool-call count reduction vs. baseline (clamped to `[-1, 1]`).
    pub tool_round_weight: f32,
    /// Bonus/penalty points for relative latency reduction vs. baseline (clamped to `[-1, 1]`).
    pub latency_weight: f32,
    /// Minimum composite objective gain required over baseline (`0.0`) to accept a candidate.
    pub min_composite_gain: f32,
}

impl Default for MultiObjectiveWeights {
    fn default() -> Self {
        Self {
            pass_rate_weight: 1.0,
            token_efficiency_weight: 4.0,
            tool_round_weight: 2.5,
            latency_weight: 1.5,
            min_composite_gain: 0.25,
        }
    }
}

/// Detailed multi-objective scoring breakdown for one evaluated candidate vs. baseline.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CandidateObjectiveBreakdown {
    pub pass_rate_delta: f32,
    pub token_saving_ratio: f32,
    pub tool_saving_ratio: f32,
    pub latency_saving_ratio: f32,
    pub composite_delta: f32,
}

/// Compute multi-objective gain relative to baseline (`0.0` = identical to baseline).
///
/// Positive `saving_ratio` means the candidate used *fewer* tokens / tool calls / milliseconds
/// than the baseline (clamped to `[-1.0, 1.0]` so extreme outliers never overwhelm pass rate).
pub fn evaluate_multi_objective(
    baseline: &EvalFeedback,
    candidate: &EvalFeedback,
    weights: &MultiObjectiveWeights,
) -> CandidateObjectiveBreakdown {
    let pass_rate_delta = candidate.score - baseline.score;
    let token_saving_ratio = relative_saving(baseline.mean_tokens(), candidate.mean_tokens());
    let tool_saving_ratio =
        relative_saving(baseline.mean_tool_calls(), candidate.mean_tool_calls());
    let latency_saving_ratio =
        relative_saving(baseline.mean_duration_ms(), candidate.mean_duration_ms());

    let composite_delta = pass_rate_delta * weights.pass_rate_weight
        + token_saving_ratio * weights.token_efficiency_weight
        + tool_saving_ratio * weights.tool_round_weight
        + latency_saving_ratio * weights.latency_weight;

    CandidateObjectiveBreakdown {
        pass_rate_delta,
        token_saving_ratio,
        tool_saving_ratio,
        latency_saving_ratio,
        composite_delta,
    }
}

fn relative_saving(baseline_val: f64, candidate_val: f64) -> f32 {
    if baseline_val <= 1e-6 {
        return 0.0;
    }
    ((baseline_val - candidate_val) / baseline_val).clamp(-1.0, 1.0) as f32
}

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

/// LLM-backed two-stage SIM-conditioned selector with multi-objective telemetry and deterministic fallback.
#[derive(Clone)]
pub struct LlmSimSelector {
    router: HybridRouter,
    optimizer_profile: ModelProfile,
    weights: MultiObjectiveWeights,
}

impl LlmSimSelector {
    pub fn new(router: HybridRouter, optimizer_profile: ModelProfile) -> Self {
        Self {
            router,
            optimizer_profile,
            weights: MultiObjectiveWeights::default(),
        }
    }

    pub fn with_weights(mut self, weights: MultiObjectiveWeights) -> Self {
        self.weights = weights;
        self
    }

    pub fn build_system_prompt(ledger: &SimLedger, memory_mode: SimMemoryMode) -> String {
        let rules = SIM_PROMPT_RULES.join("\n");
        let sim_block = ledger.render_for_prompt(memory_mode);
        format!(
            "You are AllyH's Multi-Objective Self-Improvement Harness Selector.\n\
             Weigh pass-rate gains, token budget efficiency, tool-round efficiency, wall-clock latency, \
             complexity, domain generality, and regression risk against the complete SIM record.\n\n\
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
            return HeuristicSimSelector::with_weights(self.weights)
                .select_winner(
                    current_harness,
                    baseline_feedback,
                    evaluated,
                    ledger,
                    memory_mode,
                )
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
                let obj = evaluate_multi_objective(baseline_feedback, &c.feedback, &self.weights);
                serde_json::json!({
                    "sample_id": c.proposal.sample_id,
                    "problem_ref": c.proposal.problem_ref,
                    "what": c.proposal.what,
                    "score": c.feedback.score,
                    "delta": obj.pass_rate_delta,
                    "composite_delta": obj.composite_delta,
                    "mean_tokens": c.feedback.mean_tokens(),
                    "mean_tool_calls": c.feedback.mean_tool_calls(),
                    "mean_duration_ms": c.feedback.mean_duration_ms(),
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
             Never accept a candidate that regresses pass rate.\n\
             Return ONLY valid JSON: `{{\"accepted_sample_id\":\"s1\"|null,\"rationale\":\"...\"}}`.",
            Self::build_system_prompt(ledger, memory_mode)
        );
        let user = format!(
            "Baseline Evolve Score: {:.2} (mean_tokens={:.0}, mean_tools={:.2}, mean_ms={:.0})\n{}\n\nEvaluated Candidates:\n{}",
            baseline_feedback.score,
            baseline_feedback.mean_tokens(),
            baseline_feedback.mean_tool_calls(),
            baseline_feedback.mean_duration_ms(),
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

        HeuristicSimSelector::with_weights(self.weights)
            .select_winner(
                current_harness,
                baseline_feedback,
                evaluated,
                ledger,
                memory_mode,
            )
            .await
    }
}

/// Deterministic multi-objective selector maximizing pass rate + token/tool/latency efficiency
/// while strictly rejecting any pass-rate regression.
#[derive(Debug, Default, Clone)]
pub struct HeuristicSimSelector {
    pub weights: MultiObjectiveWeights,
}

impl HeuristicSimSelector {
    pub fn with_weights(weights: MultiObjectiveWeights) -> Self {
        Self { weights }
    }
}

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
        let mut scored: Vec<(&EvaluatedCandidate, CandidateObjectiveBreakdown)> = evaluated
            .iter()
            .map(|c| {
                let breakdown =
                    evaluate_multi_objective(baseline_feedback, &c.feedback, &self.weights);
                (c, breakdown)
            })
            // Guardrail: never accept a candidate that regresses pass rate below baseline.
            .filter(|(_, b)| {
                b.pass_rate_delta >= -1e-4
                    && (b.pass_rate_delta > 1e-4 || b.composite_delta >= self.weights.min_composite_gain)
            })
            .collect();

        scored.sort_by(|(ca, ba), (cb, bb)| {
            bb.composite_delta
                .partial_cmp(&ba.composite_delta)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    bb.pass_rate_delta
                        .partial_cmp(&ba.pass_rate_delta)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| ca.proposal.sample_id.cmp(&cb.proposal.sample_id))
        });

        match scored.first() {
            Some((winner, b)) => Ok(SelectionVerdict {
                accepted_sample_id: Some(winner.proposal.sample_id.clone()),
                rationale: format!(
                    "multi-objective gain +{:.2} (pass {:.2}->{:.2} [{:+.2}], tokens {:+.0}%, tools {:+.0}%, latency {:+.0}%)",
                    b.composite_delta,
                    baseline_feedback.score,
                    winner.feedback.score,
                    b.pass_rate_delta,
                    b.token_saving_ratio * 100.0,
                    b.tool_saving_ratio * 100.0,
                    b.latency_saving_ratio * 100.0,
                ),
            }),
            None => Ok(SelectionVerdict {
                accepted_sample_id: None,
                rationale: format!(
                    "no candidate improved multi-objective score without pass-rate regression (baseline {:.2}); keeping H_t unchanged",
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{EvalTask, TaskRolloutResult};
    use allyh_memory::{SimLedger, SimMemoryMode, SimProblemRef};
    use std::collections::BTreeMap;

    fn dummy_tasks() -> Vec<EvalTask> {
        vec![
            EvalTask {
                id: "t1".into(),
                title: "Task 1".into(),
                prompt: "p1".into(),
                setup_files: BTreeMap::new(),
                verifier_command: None,
                expected_substrings: Vec::new(),
                domain_tags: Vec::new(),
            },
            EvalTask {
                id: "t2".into(),
                title: "Task 2".into(),
                prompt: "p2".into(),
                setup_files: BTreeMap::new(),
                verifier_command: None,
                expected_substrings: Vec::new(),
                domain_tags: Vec::new(),
            },
        ]
    }

    fn make_feedback(
        version: &str,
        passes: [bool; 2],
        tokens: u64,
        tools: usize,
        ms: u64,
    ) -> EvalFeedback {
        let tasks = dummy_tasks();
        let rollouts = vec![
            TaskRolloutResult {
                task_id: "t1".into(),
                rollout_idx: 0,
                passed: passes[0],
                score: if passes[0] { 1.0 } else { 0.0 },
                trace_summary: String::new(),
                error_trace: None,
                tool_calls: tools,
                duration_ms: ms,
                estimated_tokens: tokens,
            },
            TaskRolloutResult {
                task_id: "t2".into(),
                rollout_idx: 0,
                passed: passes[1],
                score: if passes[1] { 1.0 } else { 0.0 },
                trace_summary: String::new(),
                error_trace: None,
                tool_calls: tools,
                duration_ms: ms,
                estimated_tokens: tokens,
            },
        ];
        EvalFeedback::from_rollouts(version, "test-model", &tasks, 1, rollouts)
    }

    fn make_candidate(sample_id: &str, fb: EvalFeedback) -> EvaluatedCandidate {
        EvaluatedCandidate {
            proposal: CandidateProposal {
                sample_id: sample_id.into(),
                problem_ref: SimProblemRef::New("efficiency".into()),
                what: format!("candidate {sample_id}"),
                files: Vec::new(),
                predicted: Vec::new(),
                patch: serde_json::json!({}),
            },
            harness: HarnessSpec::default_seed(),
            harness_version_id: sample_id.into(),
            feedback: fb,
        }
    }

    #[tokio::test]
    async fn multi_objective_selector_prefers_efficient_candidate_and_blocks_pass_regressions() {
        let selector = HeuristicSimSelector::default();
        let harness = HarnessSpec::default_seed();
        let ledger = SimLedger::new_source(
            "repo",
            "default",
            "test",
            "coding",
            "bench",
            "m",
            "m",
            "seed",
            SimMemoryMode::FullSim,
        );
        let baseline = make_feedback("base", [true, false], 20_000, 6, 4_000);

        // 1. Two candidates achieve identical 100% pass rate: leaner token/tool/latency candidate wins.
        let bloated = make_candidate("s_bloated", make_feedback("s1", [true, true], 40_000, 10, 8_000));
        let lean = make_candidate("s_lean", make_feedback("s2", [true, true], 10_000, 3, 2_000));
        let verdict = selector
            .select_winner(&harness, &baseline, &[bloated, lean], &ledger, SimMemoryMode::FullSim)
            .await
            .unwrap();
        assert_eq!(verdict.accepted_sample_id.as_deref(), Some("s_lean"));

        // 2. Equal pass rate (50%) with major token/tool/latency savings still wins cleanly!
        let faster_same_pass =
            make_candidate("s_fast", make_feedback("s3", [true, false], 8_000, 2, 1_500));
        let verdict_eff = selector
            .select_winner(
                &harness,
                &baseline,
                &[faster_same_pass],
                &ledger,
                SimMemoryMode::FullSim,
            )
            .await
            .unwrap();
        assert_eq!(verdict_eff.accepted_sample_id.as_deref(), Some("s_fast"));

        // 3. Pass-rate regression (50% -> 0%) is STRICTLY rejected even if 95% cheaper/faster!
        let regressed =
            make_candidate("s_regressed", make_feedback("s4", [false, false], 500, 1, 100));
        let verdict_rej = selector
            .select_winner(&harness, &baseline, &[regressed], &ledger, SimMemoryMode::FullSim)
            .await
            .unwrap();
        assert!(verdict_rej.accepted_sample_id.is_none());
    }
}
