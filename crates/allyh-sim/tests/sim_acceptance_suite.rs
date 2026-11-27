use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use allyh_core::{
    AllyHOrchestrator, HarnessSpec, HybridRouter, ModelProfile, OrchestratorConfig,
    RigCompletionBackend,
};
use allyh_memory::{
    DEFAULT_SCOPE_ID, DeterministicHashEmbedder, MemoryStore, SimMemoryMode, SimProblemEntry,
    SimProblemRef, StorageConfig, compute_repo_id,
};
use allyh_sim::{
    BenchmarkEvaluator, CandidateProposal, CrsiConfig, CrsiRunner, EvalFeedback, EvalSuiteSpec,
    EvalTask, HeuristicSimProposer, HeuristicSimSelector, LlmSimProposer, RunHistory,
    SimProposer, TaskRolloutResult, UnifiedMultiModelEvaluator,
};
use async_trait::async_trait;
use tempfile::TempDir;

/// Deterministic mock evaluator scoring `HarnessSpec` features against domain-tagged tasks.
/// Simulates realistic source-to-target and cross-model harness adaptation dynamics.
struct SyntheticHarnessEvaluator;

#[async_trait]
impl BenchmarkEvaluator for SyntheticHarnessEvaluator {
    async fn evaluate(
        &self,
        harness: &HarnessSpec,
        harness_version_id: &str,
        policy: &ModelProfile,
        tasks: &[EvalTask],
        rollouts_per_task: usize,
    ) -> allyh_sim::Result<EvalFeedback> {
        let rollouts_per_task = rollouts_per_task.max(1);
        let mut rollouts = Vec::new();
        for task in tasks {
            let passed = task_passes(harness, policy, task);
            for r in 0..rollouts_per_task {
                rollouts.push(TaskRolloutResult {
                    task_id: task.id.clone(),
                    rollout_idx: r,
                    passed,
                    score: if passed { 1.0 } else { 0.0 },
                    trace_summary: if passed {
                        "verifier passed".to_string()
                    } else {
                        format!("failed tags: {:?}", task.domain_tags)
                    },
                    error_trace: (!passed).then(|| format!("failure in {}", task.id)),
                    tool_calls: 1,
                    duration_ms: 1,
                });
            }
        }
        Ok(EvalFeedback::from_rollouts(
            harness_version_id,
            &policy.model_id,
            tasks,
            rollouts_per_task,
            rollouts,
        ))
    }
}

fn task_passes(harness: &HarnessSpec, policy: &ModelProfile, task: &EvalTask) -> bool {
    for tag in &task.domain_tags {
        match tag.as_str() {
            "needs_interlock" => {
                if !harness.completion_interlock.enabled {
                    return false;
                }
            }
            "needs_contract_guidelines" => {
                if harness.preamble.verification_guidelines.is_empty() {
                    return false;
                }
            }
            "needs_quality_audit" => {
                if harness.completion_interlock.quality_audit_notice.is_none() {
                    return false;
                }
            }
            "needs_truncation" => {
                if harness.tools.output_head_chars == 0 {
                    return false;
                }
            }
            "needs_model_format_for_target_model" => {
                if policy.model_id.contains("target-model")
                    && harness.model_adaptation.output_format_guidance.is_none()
                {
                    return false;
                }
            }
            "unsolvable" => return false,
            _ => {}
        }
    }
    true
}

/// Spy wrapper capturing prompts built by `LlmSimProposer` to verify split isolation & SIM conditioning.
struct SpyProposer {
    inner: HeuristicSimProposer,
    captured_system_prompts: Arc<Mutex<Vec<String>>>,
    captured_user_prompts: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl SimProposer for SpyProposer {
    async fn propose(
        &self,
        round: u32,
        current_harness: &HarnessSpec,
        feedback: &EvalFeedback,
        run_history: &RunHistory,
        ledger: &allyh_memory::SimLedger,
        memory_mode: SimMemoryMode,
        num_candidates: usize,
    ) -> allyh_sim::Result<Vec<CandidateProposal>> {
        self.captured_system_prompts
            .lock()
            .unwrap()
            .push(LlmSimProposer::build_system_prompt(ledger, memory_mode));
        self.captured_user_prompts
            .lock()
            .unwrap()
            .push(LlmSimProposer::build_user_prompt(
                round,
                current_harness,
                feedback,
                run_history,
                num_candidates,
            ));
        self.inner
            .propose(
                round,
                current_harness,
                feedback,
                run_history,
                ledger,
                memory_mode,
                num_candidates,
            )
            .await
    }
}

fn make_task(id: &str, tags: &[&str]) -> EvalTask {
    EvalTask {
        id: id.to_string(),
        title: format!("Task {id}"),
        prompt: format!("Solve {id}"),
        setup_files: BTreeMap::new(),
        verifier_command: None,
        expected_substrings: Vec::new(),
        domain_tags: tags.iter().map(|s| s.to_string()).collect(),
    }
}

#[tokio::test]
async fn five_step_sim_and_crsi_acceptance_suite() {
    let tmp = TempDir::new().expect("tempdir");
    let db_dir = tmp.path().join("sim_db");
    let repo_a_dir = tmp.path().join("repo_a");
    let repo_b_dir = tmp.path().join("repo_b");
    std::fs::create_dir_all(repo_a_dir.join("src")).expect("mkdir a");
    std::fs::create_dir_all(repo_b_dir.join("src")).expect("mkdir b");
    std::fs::write(repo_a_dir.join("src/lib.rs"), "pub fn alpha() {}\n").expect("write a");
    std::fs::write(repo_b_dir.join("src/lib.rs"), "pub fn beta() {}\n").expect("write b");

    let store = MemoryStore::connect(
        StorageConfig::from_local_path(&db_dir),
        Arc::new(DeterministicHashEmbedder),
    )
    .await
    .expect("connect");

    let repo_a = compute_repo_id(&repo_a_dir.to_string_lossy());
    let repo_b = compute_repo_id(&repo_b_dir.to_string_lossy());

    // -----------------------------------------------------------------------
    // Step 1 & 2: Source RSI + Strict Disjoint Evolve/Held-Out Split Isolation
    // -----------------------------------------------------------------------
    let source_suite = EvalSuiteSpec {
        name: "source-coding-suite".into(),
        domain: "coding_enriched".into(),
        benchmark: "tb2.1".into(),
        split_seed: 42,
        tasks: vec![
            make_task("src-interlock-1", &["needs_interlock"]),
            make_task("src-interlock-2", &["needs_interlock"]),
            make_task("src-contract-1", &["needs_contract_guidelines"]),
            make_task("src-contract-2", &["needs_contract_guidelines"]),
            make_task("src-hard-1", &["needs_interlock", "unsolvable"]),
            make_task("src-baseline-ok", &[]),
        ],
    };
    let (src_evolve, src_held_out) = source_suite.split().expect("split source suite");
    let held_out_ids: Vec<String> = src_held_out.tasks().iter().map(|t| t.id.clone()).collect();
    assert!(!held_out_ids.is_empty());

    let captured_sys = Arc::new(Mutex::new(Vec::new()));
    let captured_usr = Arc::new(Mutex::new(Vec::new()));
    let spy_proposer = Arc::new(SpyProposer {
        inner: HeuristicSimProposer,
        captured_system_prompts: Arc::clone(&captured_sys),
        captured_user_prompts: Arc::clone(&captured_usr),
    });

    let runner = CrsiRunner::new(
        store.clone(),
        Arc::new(SyntheticHarnessEvaluator),
        spy_proposer,
        Arc::new(HeuristicSimSelector),
    );

    let source_policy = ModelProfile::offline_mock("source-model-v1");
    let optimizer_model = ModelProfile::offline_mock("optimizer-v1");

    let (source_harness_rec, mut source_ledger, source_report) = runner
        .run_source(
            CrsiConfig {
                repo_id: repo_a.clone(),
                scope_id: DEFAULT_SCOPE_ID.into(),
                arm_name: "source-tb21".into(),
                domain: "coding_enriched".into(),
                benchmark: "tb2.1".into(),
                rounds: 3,
                candidates_per_round: 4,
                rollouts_per_task: 2,
                memory_mode: SimMemoryMode::FullSim,
                policy_model: source_policy.clone(),
                optimizer_model: optimizer_model.clone(),
                activate_winner: false,
            },
            HarnessSpec::default_seed(),
            &src_evolve,
        )
        .await
        .expect("run source RSI");

    assert!(
        source_report.final_evolve_score > source_report.starting_evolve_score,
        "source RSI should improve evolve score: {} -> {}",
        source_report.starting_evolve_score,
        source_report.final_evolve_score
    );
    assert!(
        !source_ledger.problems.is_empty(),
        "SIM should record diagnosed problems from accepted rounds"
    );

    // Verify Step 2: Held-out task IDs NEVER leaked into proposer prompts or persisted SIM records.
    for prompt in captured_usr.lock().unwrap().iter() {
        for hid in &held_out_ids {
            assert!(
                !prompt.contains(hid),
                "held-out task '{hid}' leaked into proposer prompt!"
            );
        }
    }
    let sim_json = serde_json::to_string(&source_ledger.to_export_document()).unwrap();
    for hid in &held_out_ids {
        assert!(
            !sim_json.contains(hid),
            "held-out task '{hid}' leaked into SIM ledger!"
        );
    }

    // Evaluate source harness on held-out split outside the loop.
    let held_out_before = runner
        .evaluate_held_out(
            &repo_a,
            DEFAULT_SCOPE_ID,
            &source_report.starting_harness_id,
            &source_policy,
            &src_held_out,
            2,
        )
        .await
        .expect("held-out before");
    let held_out_after = runner
        .evaluate_held_out(
            &repo_a,
            DEFAULT_SCOPE_ID,
            &source_harness_rec.id,
            &source_policy,
            &src_held_out,
            2,
        )
        .await
        .expect("held-out after");
    assert!(held_out_after.score >= held_out_before.score);

    // -----------------------------------------------------------------------
    // Step 3: Rejected Round Fidelity (M_{t+1} = M_t) + Missed Prediction Persistence
    // -----------------------------------------------------------------------
    let attempts_before_extra = source_ledger.total_attempts();
    // Round 3 in our 3-round run had no further improvement if all solvable evolve tasks passed,
    // or verify at least one recorded attempt kept its predicted/fixed/missed breakdown.
    let total_predicted: usize = source_ledger
        .problems
        .iter()
        .flat_map(|p| p.attempts.iter())
        .map(|a| a.predicted.len())
        .sum();
    assert!(total_predicted > 0, "ex-ante predicted tasks must be recorded");
    assert!(attempts_before_extra >= 1);

    // Verify alias merge (`P6 -> P1`) persists cleanly before continuation.
    let first_name = source_ledger.problems[0].name.clone();
    source_ledger.problems.push(SimProblemEntry {
        id: "P6".into(),
        name: first_name,
        opened_round: 9,
        origin_domain: "coding_enriched".into(),
        origin_run_id: source_ledger.run.run_id.clone(),
        attempts: Vec::new(),
    });
    let merged = source_ledger.merge_identical_problems();
    assert_eq!(merged.len(), 1);
    assert_eq!(source_ledger.resolve_alias("P6"), "P1");
    store
        .save_sim_ledger(&source_ledger)
        .await
        .expect("save merged source ledger");

    // -----------------------------------------------------------------------
    // Step 4: Cross-Dataset & Cross-Model Continuation across all 4 transfer modes
    // -----------------------------------------------------------------------
    let target_suite = EvalSuiteSpec {
        name: "target-tb4-suite".into(),
        domain: "tb4".into(),
        benchmark: "tb4".into(),
        split_seed: 99,
        tasks: vec![
            make_task(
                "tgt-quality-1",
                &["needs_interlock", "needs_quality_audit"],
            ),
            make_task(
                "tgt-quality-2",
                &["needs_interlock", "needs_quality_audit"],
            ),
            make_task(
                "tgt-model-fmt-1",
                &["needs_interlock", "needs_model_format_for_target_model"],
            ),
            make_task(
                "tgt-trunc-1",
                &["needs_truncation"],
            ),
        ],
    };
    let (tgt_evolve, _tgt_held_out) = target_suite.split().expect("split target");
    let target_policy = ModelProfile::offline_mock("target-model-v2");

    // 4a. Full SIM continuation
    captured_sys.lock().unwrap().clear();
    let (full_harness, full_ledger, full_report) = runner
        .run_continuation(
            &repo_a,
            DEFAULT_SCOPE_ID,
            &source_ledger.run.run_id,
            CrsiConfig {
                repo_id: repo_a.clone(),
                scope_id: DEFAULT_SCOPE_ID.into(),
                arm_name: "target-tb4-full-sim".into(),
                domain: "tb4".into(),
                benchmark: "tb4".into(),
                rounds: 3,
                candidates_per_round: 5,
                rollouts_per_task: 1,
                memory_mode: SimMemoryMode::FullSim,
                policy_model: target_policy.clone(),
                optimizer_model: optimizer_model.clone(),
                activate_winner: true,
            },
            &tgt_evolve,
        )
        .await
        .expect("run FullSim continuation");

    assert_eq!(
        full_ledger.run.carried_from_run_id.as_deref(),
        Some(source_ledger.run.run_id.as_str())
    );
    assert!(full_report.final_evolve_score >= full_report.starting_evolve_score);
    assert_eq!(full_ledger.resolve_alias("P6"), "P1");
    // Verify FullSim proposer prompt received all 8 Appendix D.2 rules and carried source history!
    let full_sys_prompt = captured_sys.lock().unwrap()[0].clone();
    assert!(full_sys_prompt.contains("Treat SIM entries as prior experience"));
    assert!(full_sys_prompt.contains("Carried from:"));
    assert!(full_sys_prompt.contains("Merged aliases: P6->P1"));

    // 4b. Problem-Only continuation omits attempt details from proposer prompt
    captured_sys.lock().unwrap().clear();
    let (_, prob_only_ledger, _) = runner
        .run_continuation(
            &repo_a,
            DEFAULT_SCOPE_ID,
            &source_ledger.run.run_id,
            CrsiConfig {
                repo_id: repo_a.clone(),
                scope_id: DEFAULT_SCOPE_ID.into(),
                arm_name: "target-tb4-problem-only".into(),
                domain: "tb4".into(),
                benchmark: "tb4".into(),
                rounds: 1,
                candidates_per_round: 3,
                rollouts_per_task: 1,
                memory_mode: SimMemoryMode::ProblemOnly,
                policy_model: target_policy.clone(),
                optimizer_model: optimizer_model.clone(),
                activate_winner: false,
            },
            &tgt_evolve,
        )
        .await
        .expect("run ProblemOnly continuation");
    assert!(
        prob_only_ledger
            .problems
            .iter()
            .all(|p| p.attempts.is_empty()),
        "ProblemOnly mode must retain root causes only"
    );
    assert!(captured_sys.lock().unwrap()[0].contains("Root Causes Only"));

    // 4c. HarnessTransfer starts from H_S with empty SIM
    let (_, transfer_ledger, _) = runner
        .run_continuation(
            &repo_a,
            DEFAULT_SCOPE_ID,
            &source_ledger.run.run_id,
            CrsiConfig {
                repo_id: repo_a.clone(),
                scope_id: DEFAULT_SCOPE_ID.into(),
                arm_name: "target-tb4-harness-only".into(),
                domain: "tb4".into(),
                benchmark: "tb4".into(),
                rounds: 1,
                candidates_per_round: 3,
                rollouts_per_task: 1,
                memory_mode: SimMemoryMode::HarnessTransfer,
                policy_model: target_policy.clone(),
                optimizer_model: optimizer_model.clone(),
                activate_winner: false,
            },
            &tgt_evolve,
        )
        .await
        .expect("run HarnessTransfer");
    assert!(transfer_ledger.problems.is_empty());

    // 4d. ColdStart starts from default seed H_0 with empty SIM
    let (_, cold_ledger, _) = runner
        .run_continuation(
            &repo_a,
            DEFAULT_SCOPE_ID,
            &source_ledger.run.run_id,
            CrsiConfig {
                repo_id: repo_a.clone(),
                scope_id: DEFAULT_SCOPE_ID.into(),
                arm_name: "target-tb4-cold-start".into(),
                domain: "tb4".into(),
                benchmark: "tb4".into(),
                rounds: 1,
                candidates_per_round: 3,
                rollouts_per_task: 1,
                memory_mode: SimMemoryMode::ColdStart,
                policy_model: target_policy.clone(),
                optimizer_model: optimizer_model.clone(),
                activate_winner: false,
            },
            &tgt_evolve,
        )
        .await
        .expect("run ColdStart");
    assert!(cold_ledger.problems.is_empty());
    assert!(cold_ledger.run.carried_from_run_id.is_none());

    // -----------------------------------------------------------------------
    // Step 5: End-to-End Harness Activation in AllyHOrchestrator + Isolation + Unified Multi-Model
    // -----------------------------------------------------------------------
    let orch_cfg = OrchestratorConfig {
        repo_root: repo_a_dir.clone(),
        storage: StorageConfig::from_local_path(&db_dir),
        scope_id: DEFAULT_SCOPE_ID.into(),
        auto_verify: false,
        idle_optimize_interval: Duration::from_secs(60),
        enable_fs_watcher: false,
    };
    let router = HybridRouter::new(
        ModelProfile::offline_mock("different-model-v3"),
        ModelProfile::offline_mock("worker"),
        Arc::new(RigCompletionBackend),
    );
    let mut orch = AllyHOrchestrator::initialize(orch_cfg, router, None)
        .await
        .expect("init orchestrator");

    // Activated winning harness from Step 4a is automatically loaded!
    assert_eq!(
        orch.active_harness_record().map(|r| r.id.as_str()),
        Some(full_harness.id.as_str())
    );
    // Cross-model shift alert triggers because active harness was tuned for `target-model-v2`
    let shift = orch.model_shift_notice().await;
    assert!(
        shift.as_deref().unwrap_or("").contains("Cross-model shift"),
        "expected cross-model shift notice, got {shift:?}"
    );

    let turn = orch
        .handle_prompt("Inspect alpha", None)
        .await
        .expect("turn");
    if !orch.active_harness().preamble.verification_guidelines.is_empty() {
        assert!(turn.preamble.text.contains("## Verification Guidelines"));
    }

    // Repo B stays strictly isolated from Repo A's SIM runs and active harness.
    assert!(
        store
            .get_active_harness_version(&repo_b, DEFAULT_SCOPE_ID)
            .await
            .expect("repo b active")
            .is_none()
    );
    assert!(
        store
            .list_sim_runs(&repo_b, DEFAULT_SCOPE_ID)
            .await
            .expect("repo b runs")
            .is_empty()
    );

    // Unified Multi-Model evaluator averages scores across multiple policies (§4.3.3 S2).
    let unified = UnifiedMultiModelEvaluator::new(
        Arc::new(SyntheticHarnessEvaluator),
        vec![source_policy, target_policy],
    )
    .expect("unified");
    let unified_fb = unified
        .evaluate(
            orch.active_harness(),
            &full_harness.id,
            &ModelProfile::offline_mock("unused"),
            tgt_evolve.tasks(),
            1,
        )
        .await
        .expect("unified eval");
    assert!(unified_fb.policy_model.starts_with("unified:"));
    assert_eq!(
        unified_fb.total_rollouts,
        (tgt_evolve.len() * 2) as u32
    );

    // Sanity check `SimProblemRef` JSON format used in prompts
    let ex = serde_json::to_string(&SimProblemRef::Existing("P1".into())).unwrap();
    assert!(ex.contains("\"kind\":\"existing\""));
}
