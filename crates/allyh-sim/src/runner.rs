use std::sync::Arc;

use allyh_core::{HarnessSpec, ModelProfile};
use allyh_memory::{
    HarnessVersionRecord, MemoryStore, SimAttemptInput, SimLedger, SimMemoryMode,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::error::{Result, SimError};
use crate::eval::{BenchmarkEvaluator, EvalFeedback, EvolveSplit, HeldOutSplit};
use crate::manifest::{ChangeManifest, RoundHistoryEntry, RunHistory};
use crate::proposer::SimProposer;
use crate::selector::{EvaluatedCandidate, SimSelector};

/// Configuration for a source or continuation CRSI optimization run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrsiConfig {
    pub repo_id: String,
    pub scope_id: String,
    pub arm_name: String,
    pub domain: String,
    pub benchmark: String,
    pub rounds: u32,
    pub candidates_per_round: usize,
    pub rollouts_per_task: usize,
    pub memory_mode: SimMemoryMode,
    pub policy_model: ModelProfile,
    pub optimizer_model: ModelProfile,
    /// When true, activates the final retained harness in LanceDB for interactive AllyH sessions.
    pub activate_winner: bool,
}

/// Summary of a single optimization round `t`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundSummary {
    pub round: u32,
    pub baseline_score: f32,
    pub proposed_count: usize,
    pub screened_in_count: usize,
    pub evaluated_count: usize,
    pub accepted_sample: Option<String>,
    pub accepted_problem_id: Option<String>,
    pub opened_new_problem: bool,
    pub resulting_harness_id: String,
    pub resulting_score: f32,
    pub selection_rationale: String,
}

/// Aggregate report returned after completing a source or continuation CRSI run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrsiRunReport {
    pub run_id: String,
    pub arm_name: String,
    pub memory_mode: SimMemoryMode,
    pub carried_from_run_id: Option<String>,
    pub starting_harness_id: String,
    pub final_harness_id: String,
    pub starting_evolve_score: f32,
    pub final_evolve_score: f32,
    pub rounds: Vec<RoundSummary>,
}

/// Continual Recursive Self-Improvement (CRSI) runner orchestrating Equations (1)–(3)
/// of arXiv:2611.18856 with strict Evolve/Held-Out split isolation.
pub struct CrsiRunner {
    store: MemoryStore,
    evaluator: Arc<dyn BenchmarkEvaluator>,
    proposer: Arc<dyn SimProposer>,
    selector: Arc<dyn SimSelector>,
}

impl CrsiRunner {
    pub fn new(
        store: MemoryStore,
        evaluator: Arc<dyn BenchmarkEvaluator>,
        proposer: Arc<dyn SimProposer>,
        selector: Arc<dyn SimSelector>,
    ) -> Self {
        Self {
            store,
            evaluator,
            proposer,
            selector,
        }
    }

    /// Run source RSI from `(H_0, ∅)` on `evolve_split` and return `(H_S, M_S, report)`.
    pub async fn run_source(
        &self,
        config: CrsiConfig,
        seed_harness: HarnessSpec,
        evolve_split: &EvolveSplit,
    ) -> Result<(HarnessVersionRecord, SimLedger, CrsiRunReport)> {
        seed_harness.validate()?;
        let seed_record = seed_harness.to_version_record(
            &config.repo_id,
            &config.scope_id,
            None,
            None,
            0,
            format!("{}-seed", config.arm_name),
            &config.policy_model.model_id,
            &config.domain,
            false,
        )?;
        self.store.upsert_harness_version(&seed_record).await?;

        let ledger = SimLedger::new_source(
            &config.repo_id,
            &config.scope_id,
            &config.arm_name,
            &config.domain,
            &config.benchmark,
            &config.policy_model.model_id,
            &config.optimizer_model.model_id,
            &seed_record.id,
            config.memory_mode,
        );

        self.execute_rounds(config, seed_harness, seed_record, ledger, evolve_split)
            .await
    }

    /// Run cross-dataset or cross-model CRSI continuation from a prior source run `(H_S, M_S)`.
    ///
    /// Supports all four arXiv:2611.18856 transfer conditions:
    /// - `ColdStart`: ignores source harness & SIM; starts from `HarnessSpec::default_seed()` + empty SIM.
    /// - `HarnessTransfer`: inherits `H_S` with empty SIM `(H_S, ∅)`.
    /// - `ProblemOnly`: inherits `H_S` with root causes only `(H_S, M_S^c)`.
    /// - `FullSim`: inherits `H_S` with full SIM record `(H_S, M_S)` and extends it with target attempts.
    pub async fn run_continuation(
        &self,
        source_repo_id: &str,
        source_scope_id: &str,
        source_run_id_or_arm: &str,
        config: CrsiConfig,
        evolve_split: &EvolveSplit,
    ) -> Result<(HarnessVersionRecord, SimLedger, CrsiRunReport)> {
        let Some(source_ledger) = self
            .store
            .load_sim_ledger(source_repo_id, source_scope_id, source_run_id_or_arm)
            .await?
        else {
            return Err(SimError::InvalidSuite(format!(
                "source SIM run '{source_run_id_or_arm}' not found"
            )));
        };

        let (starting_spec, starting_record) = if config.memory_mode == SimMemoryMode::ColdStart {
            let seed = HarnessSpec::default_seed();
            let rec = seed.to_version_record(
                &config.repo_id,
                &config.scope_id,
                None,
                None,
                0,
                format!("{}-cold-seed", config.arm_name),
                &config.policy_model.model_id,
                &config.domain,
                false,
            )?;
            self.store.upsert_harness_version(&rec).await?;
            (seed, rec)
        } else {
            let Some(src_hv) = self
                .store
                .get_harness_version(
                    source_repo_id,
                    source_scope_id,
                    &source_ledger.run.current_harness_id,
                )
                .await?
            else {
                return Err(SimError::InvalidSuite(format!(
                    "source harness version '{}' not found",
                    source_ledger.run.current_harness_id
                )));
            };
            let spec = HarnessSpec::from_version_record(&src_hv)?;
            // Clone starting harness into target partition if crossing repo/scope boundaries.
            let target_start = spec.to_version_record(
                &config.repo_id,
                &config.scope_id,
                Some(src_hv.id.clone()),
                None,
                0,
                format!("{}-start", config.arm_name),
                &config.policy_model.model_id,
                &config.domain,
                false,
            )?;
            self.store.upsert_harness_version(&target_start).await?;
            (spec, target_start)
        };

        let target_ledger = source_ledger.carry_into(
            &config.repo_id,
            &config.scope_id,
            &config.arm_name,
            &config.domain,
            &config.benchmark,
            &config.policy_model.model_id,
            &config.optimizer_model.model_id,
            &starting_record.id,
            config.memory_mode,
        );

        self.execute_rounds(
            config,
            starting_spec,
            starting_record,
            target_ledger,
            evolve_split,
        )
        .await
    }

    /// Evaluate a harness version strictly on the disjoint [`HeldOutSplit`] outside the loop.
    pub async fn evaluate_held_out(
        &self,
        repo_id: &str,
        scope_id: &str,
        harness_version_id: &str,
        policy_model: &ModelProfile,
        held_out: &HeldOutSplit,
        rollouts_per_task: usize,
    ) -> Result<EvalFeedback> {
        let Some(rec) = self
            .store
            .get_harness_version(repo_id, scope_id, harness_version_id)
            .await?
        else {
            return Err(SimError::InvalidSuite(format!(
                "harness version '{harness_version_id}' not found"
            )));
        };
        let spec = HarnessSpec::from_version_record(&rec)?;
        self.evaluator
            .evaluate(
                &spec,
                &rec.id,
                policy_model,
                held_out.tasks(),
                rollouts_per_task,
            )
            .await
    }

    async fn execute_rounds(
        &self,
        config: CrsiConfig,
        mut current_harness: HarnessSpec,
        mut current_record: HarnessVersionRecord,
        mut ledger: SimLedger,
        evolve_split: &EvolveSplit,
    ) -> Result<(HarnessVersionRecord, SimLedger, CrsiRunReport)> {
        if evolve_split.is_empty() {
            return Err(SimError::InvalidSuite(
                "evolve split must contain at least one task".to_string(),
            ));
        }

        let starting_harness_id = current_record.id.clone();
        self.store.save_sim_ledger(&ledger).await?;

        // Initial baseline evaluation F_0 on evolve set.
        let mut current_feedback = self
            .evaluator
            .evaluate(
                &current_harness,
                &current_record.id,
                &config.policy_model,
                evolve_split.tasks(),
                config.rollouts_per_task,
            )
            .await?;
        let starting_evolve_score = current_feedback.score;

        let mut run_history = RunHistory::default();
        let mut round_summaries = Vec::with_capacity(config.rounds as usize);

        for round in 1..=config.rounds {
            let baseline_score = current_feedback.score;

            // 1. Proposer P(H_t, F_t, D_t; M_t) generates k candidates and locks ChangeManifest pre-eval.
            let candidates = self
                .proposer
                .propose(
                    round,
                    &current_harness,
                    &current_feedback,
                    &run_history,
                    &ledger,
                    config.memory_mode,
                    config.candidates_per_round,
                )
                .await?;
            let manifest = ChangeManifest {
                round,
                base_harness_id: current_record.id.clone(),
                candidates,
                created_at: Utc::now(),
            };
            let proposed_count = manifest.candidates.len();

            // 2. Selector Stage 1: pre-execution screening using full SIM M_t.
            let screening = self
                .selector
                .screen_candidates(
                    &current_harness,
                    &current_feedback,
                    &manifest.candidates,
                    &ledger,
                    config.memory_mode,
                )
                .await?;
            let surviving: Vec<_> = manifest
                .candidates
                .iter()
                .filter(|c| {
                    screening
                        .iter()
                        .find(|s| s.sample_id == c.sample_id)
                        .map(|s| s.approved)
                        .unwrap_or(false)
                })
                .cloned()
                .collect();
            let screened_in_count = surviving.len();

            // 3. Evaluate surviving candidates on evolve set ONLY -> F_t^C.
            let mut evaluated = Vec::with_capacity(surviving.len());
            for cand in surviving {
                let Ok(patched_spec) = current_harness.apply_merge_patch(&cand.patch) else {
                    continue;
                };
                let cand_record = patched_spec.to_version_record(
                    &config.repo_id,
                    &config.scope_id,
                    Some(current_record.id.clone()),
                    Some(ledger.run.run_id.clone()),
                    round,
                    format!("{}-r{}-{}", config.arm_name, round, cand.sample_id),
                    &config.policy_model.model_id,
                    &config.domain,
                    false,
                )?;
                let fb = self
                    .evaluator
                    .evaluate(
                        &patched_spec,
                        &cand_record.id,
                        &config.policy_model,
                        evolve_split.tasks(),
                        config.rollouts_per_task,
                    )
                    .await?;
                evaluated.push(EvaluatedCandidate {
                    proposal: cand,
                    harness: patched_spec,
                    harness_version_id: cand_record.id,
                    feedback: fb,
                });
            }
            let evaluated_count = evaluated.len();

            // 4. Selector Stage 2: choose winning candidate or retain H_t (e_t = ∅).
            let verdict = self
                .selector
                .select_winner(
                    &current_harness,
                    &current_feedback,
                    &evaluated,
                    &ledger,
                    config.memory_mode,
                )
                .await?;

            let mut accepted_sample = None;
            let mut accepted_problem_id = None;
            let mut opened_new_problem = false;
            let mut accepted_what = None;

            if let Some(winner_id) = verdict.accepted_sample_id
                && let Some(winner) = evaluated
                    .into_iter()
                    .find(|c| c.proposal.sample_id == winner_id)
            {
                let (fixed, missed) = EvalFeedback::evaluate_predictions(
                    &current_feedback,
                    &winner.feedback,
                    &winner.proposal.predicted,
                );

                // Persist winning HarnessVersionRecord.
                let mut saved_rec = winner.harness.to_version_record(
                    &config.repo_id,
                    &config.scope_id,
                    Some(current_record.id.clone()),
                    Some(ledger.run.run_id.clone()),
                    round,
                    format!("{}-r{}-{}", config.arm_name, round, winner.proposal.sample_id),
                    &config.policy_model.model_id,
                    &config.domain,
                    false,
                )?;
                saved_rec.id = winner.harness_version_id.clone();
                self.store.upsert_harness_version(&saved_rec).await?;

                // 5. Update SIM U(M_t, e_t) for memory modes that maintain SIM.
                if matches!(
                    config.memory_mode,
                    SimMemoryMode::FullSim | SimMemoryMode::ProblemOnly
                ) {
                    let update = ledger.record_accepted_attempt(SimAttemptInput {
                        problem_ref: winner.proposal.problem_ref.clone(),
                        round,
                        sample: winner.proposal.sample_id.clone(),
                        accepted: true,
                        backfilled: false,
                        what: winner.proposal.what.clone(),
                        files: winner.proposal.files.clone(),
                        predicted: winner.proposal.predicted.clone(),
                        fixed,
                        missed,
                        passed: winner.feedback.passed_rollouts,
                        total: winner.feedback.total_rollouts,
                        score: winner.feedback.score,
                        domain: config.domain.clone(),
                        benchmark: config.benchmark.clone(),
                        policy_model: config.policy_model.model_id.clone(),
                        harness_version_id: saved_rec.id.clone(),
                    })?;
                    accepted_problem_id = Some(update.canonical_problem_id);
                    opened_new_problem = update.opened_new_problem;
                } else {
                    ledger.run.current_harness_id = saved_rec.id.clone();
                    ledger.run.updated_at = Utc::now();
                }

                accepted_sample = Some(winner.proposal.sample_id);
                accepted_what = Some(winner.proposal.what);
                current_harness = winner.harness;
                current_record = saved_rec;
                current_feedback = winner.feedback;
            }

            ledger.run.rounds_completed = round;
            ledger.run.updated_at = Utc::now();
            self.store.save_sim_ledger(&ledger).await?;

            run_history.push(RoundHistoryEntry {
                round,
                baseline_score,
                accepted_sample: accepted_sample.clone(),
                accepted_what,
                resulting_score: current_feedback.score,
                notes: verdict.rationale.clone(),
            });

            round_summaries.push(RoundSummary {
                round,
                baseline_score,
                proposed_count,
                screened_in_count,
                evaluated_count,
                accepted_sample,
                accepted_problem_id,
                opened_new_problem,
                resulting_harness_id: current_record.id.clone(),
                resulting_score: current_feedback.score,
                selection_rationale: verdict.rationale,
            });
        }

        if config.activate_winner {
            current_record = self
                .store
                .set_active_harness_version(&config.repo_id, &config.scope_id, &current_record.id)
                .await?;
        }

        let report = CrsiRunReport {
            run_id: ledger.run.run_id.clone(),
            arm_name: ledger.run.arm_name.clone(),
            memory_mode: ledger.run.memory_mode,
            carried_from_run_id: ledger.run.carried_from_run_id.clone(),
            starting_harness_id,
            final_harness_id: current_record.id.clone(),
            starting_evolve_score,
            final_evolve_score: current_feedback.score,
            rounds: round_summaries,
        };

        Ok((current_record, ledger, report))
    }
}
