use allyh_core::{HarnessSpec, HybridRouter, ModelProfile, ModelProviderKind};
use allyh_memory::{SIM_PROMPT_RULES, SimLedger, SimMemoryMode, SimProblemRef};
use async_trait::async_trait;
use rig_core::message::Message;
use serde::Deserialize;

use crate::error::Result;
use crate::eval::EvalFeedback;
use crate::manifest::{CandidateProposal, RunHistory};

/// Proposer `P(H_t, F_t, D_t; M_t)` generating `k` candidate harness modifications per round.
#[async_trait]
pub trait SimProposer: Send + Sync {
    #[allow(clippy::too_many_arguments)]
    async fn propose(
        &self,
        round: u32,
        current_harness: &HarnessSpec,
        feedback: &EvalFeedback,
        run_history: &RunHistory,
        ledger: &SimLedger,
        memory_mode: SimMemoryMode,
        num_candidates: usize,
    ) -> Result<Vec<CandidateProposal>>;
}

/// LLM-backed SIM-conditioned proposer with deterministic offline fallback.
#[derive(Clone)]
pub struct LlmSimProposer {
    router: HybridRouter,
    optimizer_profile: ModelProfile,
}

impl LlmSimProposer {
    pub fn new(router: HybridRouter, optimizer_profile: ModelProfile) -> Self {
        Self {
            router,
            optimizer_profile,
        }
    }

    /// Build the exact SIM-conditioned system prompt embedding all 8 Appendix D.2 rules
    /// and the **complete** SIM ledger (zero truncation).
    pub fn build_system_prompt(ledger: &SimLedger, memory_mode: SimMemoryMode) -> String {
        let rules = SIM_PROMPT_RULES.join("\n");
        let sim_block = ledger.render_for_prompt(memory_mode);
        format!(
            "You are AllyH's Self-Improvement Harness Proposer.\n\
             Your task is to diagnose recurring harness failures from evolve-set feedback and propose \
             focused, high-leverage RFC 7396 JSON Merge Patches to `HarnessSpec`.\n\n\
             ## SIM Usage Rules (arXiv:2611.18856 Appendix D.2)\n{rules}\n\n\
             {sim_block}\n\n\
             ## Output Contract\n\
             Return ONLY a valid JSON array of candidate proposals. Each object must have:\n\
             - `problem_ref`: `{{\"kind\":\"existing\",\"value\":\"P<n>\"}}` OR `{{\"kind\":\"new\",\"value\":\"<root-cause hypothesis>\"}}`\n\
             - `what`: concise edit description and rationale\n\
             - `files`: array of harness sections/files affected (e.g. `[\"completion_interlock\", \"preamble\"]`)\n\
             - `predicted`: array of evolve-set task IDs expected to be fixed BEFORE evaluation\n\
             - `patch`: JSON Merge Patch object targeting `HarnessSpec` fields"
        )
    }

    pub fn build_user_prompt(
        round: u32,
        current_harness: &HarnessSpec,
        feedback: &EvalFeedback,
        run_history: &RunHistory,
        num_candidates: usize,
    ) -> String {
        let harness_json = current_harness.canonical_json().unwrap_or_default();
        format!(
            "Round: {round}\n\
             Requested candidates (k): {num_candidates}\n\n\
             ## Current HarnessSpec (H_t)\n```json\n{harness_json}\n```\n\n\
             ## Evolve-Set Feedback (F_t)\n{}\n\n\
             {}",
            feedback.render_summary(),
            run_history.render_for_prompt()
        )
    }
}

#[derive(Debug, Deserialize)]
struct RawProposalJson {
    problem_ref: SimProblemRef,
    what: String,
    #[serde(default)]
    files: Vec<String>,
    #[serde(default)]
    predicted: Vec<String>,
    patch: serde_json::Value,
}

#[async_trait]
impl SimProposer for LlmSimProposer {
    async fn propose(
        &self,
        round: u32,
        current_harness: &HarnessSpec,
        feedback: &EvalFeedback,
        run_history: &RunHistory,
        ledger: &SimLedger,
        memory_mode: SimMemoryMode,
        num_candidates: usize,
    ) -> Result<Vec<CandidateProposal>> {
        let k = num_candidates.max(1);
        if self.optimizer_profile.provider == ModelProviderKind::OfflineMock {
            return HeuristicSimProposer
                .propose(round, current_harness, feedback, run_history, ledger, memory_mode, k)
                .await;
        }

        let system = Self::build_system_prompt(ledger, memory_mode);
        let user = Self::build_user_prompt(round, current_harness, feedback, run_history, k);

        let mut opt_router = self.router.clone();
        opt_router.set_primary(self.optimizer_profile.clone());
        let messages = vec![Message::user(user)];

        if let Ok(resp) = opt_router.run_primary(&system, &messages, &[], None).await {
            let raw_text = resp.text();
            if let Some(arr_slice) = extract_json_array(&raw_text)
                && let Ok(parsed) = serde_json::from_str::<Vec<RawProposalJson>>(arr_slice)
                && !parsed.is_empty()
            {
                let mut proposals = Vec::new();
                for (idx, item) in parsed.into_iter().take(k).enumerate() {
                    if current_harness.apply_merge_patch(&item.patch).is_ok() {
                        proposals.push(CandidateProposal {
                            sample_id: format!("s{}", idx + 1),
                            problem_ref: item.problem_ref,
                            what: item.what,
                            files: if item.files.is_empty() {
                                vec!["harness_spec.json".to_string()]
                            } else {
                                item.files
                            },
                            predicted: item.predicted,
                            patch: item.patch,
                        });
                    }
                }
                if !proposals.is_empty() {
                    return Ok(proposals);
                }
            }
        }

        // Graceful fallback to deterministic heuristic proposer when LLM returns invalid JSON.
        HeuristicSimProposer
            .propose(round, current_harness, feedback, run_history, ledger, memory_mode, k)
            .await
    }
}

/// Deterministic offline proposer exercising SIM conditioning, existing-problem reuse,
/// and target-native problem creation without requiring network calls.
#[derive(Debug, Default, Clone)]
pub struct HeuristicSimProposer;

#[async_trait]
impl SimProposer for HeuristicSimProposer {
    async fn propose(
        &self,
        _round: u32,
        current_harness: &HarnessSpec,
        feedback: &EvalFeedback,
        _run_history: &RunHistory,
        ledger: &SimLedger,
        memory_mode: SimMemoryMode,
        num_candidates: usize,
    ) -> Result<Vec<CandidateProposal>> {
        let failing = feedback.failing_task_ids();
        let mut candidates = Vec::new();
        let k = num_candidates.max(1);

        // Candidate 1: Enable execution-verified completion interlock (P5 pattern).
        if !current_harness.completion_interlock.enabled {
            let problem_ref = match_existing_or_new(
                ledger,
                memory_mode,
                "premature completion without execution verification",
            );
            candidates.push(CandidateProposal {
                sample_id: format!("s{}", candidates.len() + 1),
                problem_ref,
                what: "Enable CompletionInterlock to intercept premature completion after failed tools or unverified edits (bounded at 3 rejections).".to_string(),
                files: vec!["crates/allyh-core/src/interlock.rs".to_string(), "harness.completion_interlock".to_string()],
                predicted: failing.iter().take(2).cloned().collect(),
                patch: serde_json::json!({
                    "completion_interlock": {
                        "enabled": true,
                        "max_consecutive_rejections": 3,
                        "require_verification_after_edits": true,
                        "reject_on_recent_tool_error": true
                    }
                }),
            });
        }

        // Candidate 2: Add verification guidelines & contract audit (P3 pattern).
        if current_harness.preamble.verification_guidelines.is_empty() {
            let problem_ref = match_existing_or_new(
                ledger,
                memory_mode,
                "deliverable violates external verifier contract or schema",
            );
            candidates.push(CandidateProposal {
                sample_id: format!("s{}", candidates.len() + 1),
                problem_ref,
                what: "Add explicit external-contract and schema verification guidelines to the dynamic preamble.".to_string(),
                files: vec!["crates/allyh-core/src/preamble.rs".to_string(), "harness.preamble".to_string()],
                predicted: failing.iter().take(2).cloned().collect(),
                patch: serde_json::json!({
                    "preamble": {
                        "verification_guidelines": [
                            "Verify exact external API signatures, schemas, and file deliverables against task requirements.",
                            "Re-run automated verification after every code modification before finalizing."
                        ]
                    }
                }),
            });
        }

        // Candidate 3: Extend completion interlock with optimality / quality audit gate (P8 pattern).
        if current_harness.completion_interlock.enabled
            && current_harness.completion_interlock.quality_audit_notice.is_none()
        {
            let problem_ref = match_existing_or_new(
                ledger,
                memory_mode,
                "premature termination on feasible but suboptimal solutions",
            );
            candidates.push(CandidateProposal {
                sample_id: format!("s{}", candidates.len() + 1),
                problem_ref,
                what: "Add one-shot quality and optimality audit notice to CompletionInterlock before final completion.".to_string(),
                files: vec!["crates/allyh-core/src/interlock.rs".to_string(), "harness.completion_interlock".to_string()],
                predicted: failing.iter().take(2).cloned().collect(),
                patch: serde_json::json!({
                    "completion_interlock": {
                        "quality_audit_notice": "Passing basic checks establishes feasibility, not completeness or optimality. Audit edge cases and contract requirements before stopping.",
                        "budget_reminder_ratio": 0.75
                    }
                }),
            });
        }

        // Candidate 4: Add head/tail tool output truncation + higher retrieval depth.
        if current_harness.tools.output_head_chars == 0 {
            let problem_ref = match_existing_or_new(
                ledger,
                memory_mode,
                "oversized tool outputs saturate context window",
            );
            candidates.push(CandidateProposal {
                sample_id: format!("s{}", candidates.len() + 1),
                problem_ref,
                what: "Configure head/tail tool output truncation quotas (4000/2000 chars) and widen retrieval context.".to_string(),
                files: vec!["harness.tools".to_string(), "harness.retrieval".to_string()],
                predicted: failing.iter().take(1).cloned().collect(),
                patch: serde_json::json!({
                    "tools": {
                        "output_head_chars": 4000,
                        "output_tail_chars": 2000,
                        "max_tool_rounds": 8
                    },
                    "retrieval": {
                        "decision_top_k": 10,
                        "skill_match_top_k": 4
                    }
                }),
            });
        }

        // Candidate 5: Model-specific output format guidance for cross-model adaptation.
        if current_harness.model_adaptation.output_format_guidance.is_none() {
            let problem_ref = match_existing_or_new(
                ledger,
                memory_mode,
                "model-specific output formatting and tool-argument drift",
            );
            candidates.push(CandidateProposal {
                sample_id: format!("s{}", candidates.len() + 1),
                problem_ref,
                what: "Add concise model output contract guidance to reduce malformed tool calls and incomplete final answers.".to_string(),
                files: vec!["harness.model_adaptation".to_string()],
                predicted: failing.iter().take(2).cloned().collect(),
                patch: serde_json::json!({
                    "model_adaptation": {
                        "output_format_guidance": "Always emit valid JSON tool arguments and verify every required output artifact before finishing."
                    }
                }),
            });
        }

        candidates.truncate(k);
        Ok(candidates)
    }
}

fn match_existing_or_new(
    ledger: &SimLedger,
    mode: SimMemoryMode,
    hypothesis: &str,
) -> SimProblemRef {
    if matches!(mode, SimMemoryMode::FullSim | SimMemoryMode::ProblemOnly) {
        let norm = allyh_memory::normalize_root_cause(hypothesis);
        if let Some(existing) = ledger
            .problems
            .iter()
            .find(|p| allyh_memory::normalize_root_cause(&p.name) == norm)
        {
            return SimProblemRef::Existing(existing.id.clone());
        }
    }
    SimProblemRef::New(hypothesis.to_string())
}

fn extract_json_array(raw: &str) -> Option<&str> {
    let start = raw.find('[')?;
    let end = raw.rfind(']')?;
    (end >= start).then(|| &raw[start..=end])
}
