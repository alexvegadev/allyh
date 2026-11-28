use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use allyh_core::{
    AllyHOrchestrator, CompletionBackend, HarnessSpec, HybridRouter, ModelProfile,
    OrchestratorConfig, RigCompletionBackend,
};
use allyh_memory::StorageConfig;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tempfile::TempDir;
use tokio::process::Command;

use crate::error::{Result, SimError};

/// Single benchmark task inside an [`EvalSuiteSpec`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvalTask {
    pub id: String,
    pub title: String,
    pub prompt: String,
    /// Optional workspace files seeded into an isolated temp repo before running the task.
    #[serde(default)]
    pub setup_files: BTreeMap<String, String>,
    /// Optional shell verification command executed in the isolated temp repo (exit 0 = pass).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verifier_command: Option<String>,
    /// Optional expected substrings checked against assistant reply + preamble/workspace output.
    #[serde(default)]
    pub expected_substrings: Vec<String>,
    /// Optional failure/domain tags used for diagnostics and offline simulation.
    #[serde(default)]
    pub domain_tags: Vec<String>,
}

/// Portable benchmark suite definition loaded from JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSuiteSpec {
    pub name: String,
    pub domain: String,
    pub benchmark: String,
    #[serde(default = "default_split_seed")]
    pub split_seed: u64,
    pub tasks: Vec<EvalTask>,
}

fn default_split_seed() -> u64 {
    20_261_113
}

impl EvalSuiteSpec {
    pub fn load_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let spec: Self = serde_json::from_str(&raw)?;
        if spec.tasks.is_empty() {
            return Err(SimError::InvalidSuite(
                "evaluation suite must contain at least one task".to_string(),
            ));
        }
        Ok(spec)
    }

    /// Deterministically partition tasks uniformly at random (without stratification) into
    /// disjoint [`EvolveSplit`] and [`HeldOutSplit`] halves (arXiv:2611.18856 Appendix B.2:
    /// when total count is odd, evolve half takes the even count).
    pub fn split(&self) -> Result<(EvolveSplit, HeldOutSplit)> {
        if self.tasks.len() < 2 {
            return Err(SimError::InvalidSuite(
                "evaluation suite needs at least 2 tasks to form disjoint evolve and held-out splits".to_string(),
            ));
        }
        let mut shuffled = self.tasks.clone();
        deterministic_shuffle(&mut shuffled, self.split_seed);

        let total = shuffled.len();
        let evolve_count = if total.is_multiple_of(2) {
            total / 2
        } else {
            let half = total / 2;
            if half.is_multiple_of(2) { half } else { half + 1 }
        };
        let held_out = shuffled.split_off(evolve_count);
        Ok((EvolveSplit(shuffled), HeldOutSplit(held_out)))
    }
}

/// Disjoint evolve split (`X^evo`): ONLY this split may be passed to proposer, selector, and SIM.
#[derive(Debug, Clone)]
pub struct EvolveSplit(Vec<EvalTask>);

impl EvolveSplit {
    pub fn from_tasks(tasks: Vec<EvalTask>) -> Self {
        Self(tasks)
    }

    pub fn tasks(&self) -> &[EvalTask] {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Disjoint held-out split (`X^eval`): strictly outside the optimization loop.
#[derive(Debug, Clone)]
pub struct HeldOutSplit(Vec<EvalTask>);

impl HeldOutSplit {
    pub fn from_tasks(tasks: Vec<EvalTask>) -> Self {
        Self(tasks)
    }

    pub fn tasks(&self) -> &[EvalTask] {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Outcome of executing one rollout of a single [`EvalTask`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRolloutResult {
    pub task_id: String,
    pub rollout_idx: usize,
    pub passed: bool,
    /// Per-rollout score on `[0.0, 1.0]`.
    pub score: f32,
    pub trace_summary: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_trace: Option<String>,
    pub tool_calls: usize,
    pub duration_ms: u64,
}

/// Aggregated evaluation feedback `F_t` returned by executing a harness on a task split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalFeedback {
    pub harness_version_id: String,
    pub policy_model: String,
    pub rollouts_per_task: usize,
    pub total_tasks: usize,
    pub passed_rollouts: u32,
    pub total_rollouts: u32,
    /// Benchmark score on 0..=100 scale (Eq. 4 pass@1 when rollouts are binary).
    pub score: f32,
    /// Mean pass rate per task ID in `[0.0, 1.0]`.
    pub task_pass_rates: BTreeMap<String, f32>,
    pub rollouts: Vec<TaskRolloutResult>,
}

impl EvalFeedback {
    /// Aggregate rollout outcomes using arXiv:2611.18856 Eq. 4 (`pass@1` on 0..=100 scale),
    /// keeping every failed/infrastructure rollout in the denominator with score 0.
    pub fn from_rollouts(
        harness_version_id: impl Into<String>,
        policy_model: impl Into<String>,
        tasks: &[EvalTask],
        rollouts_per_task: usize,
        rollouts: Vec<TaskRolloutResult>,
    ) -> Self {
        let rollouts_per_task = rollouts_per_task.max(1);
        let total_tasks = tasks.len();
        let total_rollouts = (total_tasks * rollouts_per_task) as u32;
        let mut passed_rollouts = 0u32;
        let mut task_sums: BTreeMap<String, f32> = tasks
            .iter()
            .map(|t| (t.id.clone(), 0.0f32))
            .collect();

        for r in &rollouts {
            if r.passed {
                passed_rollouts += 1;
            }
            if let Some(entry) = task_sums.get_mut(&r.task_id) {
                *entry += r.score.clamp(0.0, 1.0);
            }
        }

        let mut task_pass_rates = BTreeMap::new();
        let mut mean_sum = 0.0f32;
        for (task_id, sum) in task_sums {
            let rate = sum / rollouts_per_task as f32;
            mean_sum += rate;
            task_pass_rates.insert(task_id, rate);
        }

        let score = if total_tasks == 0 {
            0.0
        } else {
            (100.0 / total_tasks as f32) * mean_sum
        };

        Self {
            harness_version_id: harness_version_id.into(),
            policy_model: policy_model.into(),
            rollouts_per_task,
            total_tasks,
            passed_rollouts,
            total_rollouts,
            score,
            task_pass_rates,
            rollouts,
        }
    }

    /// Return task IDs whose pass rate is below 1.0 (failing or flaky tasks).
    pub fn failing_task_ids(&self) -> Vec<String> {
        self.task_pass_rates
            .iter()
            .filter_map(|(id, &rate)| (rate < 1.0).then_some(id.clone()))
            .collect()
    }

    /// Check ex-ante `predicted` task fixes against ex-post measured evolve outcomes,
    /// returning `(fixed, missed)` (arXiv:2611.18856 §3.2 & Appendix D.3).
    pub fn evaluate_predictions(
        baseline: &EvalFeedback,
        candidate: &EvalFeedback,
        predicted: &[String],
    ) -> (Vec<String>, Vec<String>) {
        let mut fixed = Vec::new();
        let mut missed = Vec::new();
        for task_id in predicted {
            let before = baseline.task_pass_rates.get(task_id).copied().unwrap_or(0.0);
            let after = candidate
                .task_pass_rates
                .get(task_id)
                .copied()
                .unwrap_or(0.0);
            if after > before + 1e-4 {
                fixed.push(task_id.clone());
            } else {
                missed.push(task_id.clone());
            }
        }
        (fixed, missed)
    }

    /// Compact summary rendered for Proposer/Selector prompts.
    pub fn render_summary(&self) -> String {
        let mut lines = vec![format!(
            "Harness `{}` | Policy `{}` | Score: {:.2} ({}/{} rollouts passed)",
            &self.harness_version_id[..self.harness_version_id.len().min(12)],
            self.policy_model,
            self.score,
            self.passed_rollouts,
            self.total_rollouts
        )];
        for (task_id, rate) in &self.task_pass_rates {
            let status = if *rate >= 1.0 {
                "PASS"
            } else if *rate > 0.0 {
                "PARTIAL"
            } else {
                "FAIL"
            };
            lines.push(format!("  - {task_id}: {status} ({:.0}%)", rate * 100.0));
            if *rate < 1.0
                && let Some(err) = self
                    .rollouts
                    .iter()
                    .find(|r| r.task_id == *task_id && !r.passed)
                    .and_then(|r| r.error_trace.as_deref())
            {
                lines.push(format!("    trace: {}", truncate_str(err, 220)));
            }
        }
        lines.join("\n")
    }
}

/// Pluggable evaluator executing a candidate [`HarnessSpec`] on a slice of [`EvalTask`]s.
#[async_trait]
pub trait BenchmarkEvaluator: Send + Sync {
    async fn evaluate(
        &self,
        harness: &HarnessSpec,
        harness_version_id: &str,
        policy: &ModelProfile,
        tasks: &[EvalTask],
        rollouts_per_task: usize,
    ) -> Result<EvalFeedback>;
}

/// Executes AllyH's own runtime [`AllyHOrchestrator`] with the candidate [`HarnessSpec`]
/// inside isolated temporary workspaces for each task rollout.
pub struct AllyHRuntimeEvaluator {
    backend: Arc<dyn CompletionBackend>,
    worker_profile: ModelProfile,
}

impl AllyHRuntimeEvaluator {
    pub fn new(backend: Arc<dyn CompletionBackend>, worker_profile: ModelProfile) -> Self {
        Self {
            backend,
            worker_profile,
        }
    }

    pub fn live_default() -> Self {
        Self {
            backend: Arc::new(RigCompletionBackend),
            worker_profile: ModelProfile::offline_mock("Local-Eval-Worker"),
        }
    }
}

#[async_trait]
impl BenchmarkEvaluator for AllyHRuntimeEvaluator {
    async fn evaluate(
        &self,
        harness: &HarnessSpec,
        harness_version_id: &str,
        policy: &ModelProfile,
        tasks: &[EvalTask],
        rollouts_per_task: usize,
    ) -> Result<EvalFeedback> {
        let rollouts_per_task = rollouts_per_task.max(1);
        let mut results = Vec::with_capacity(tasks.len() * rollouts_per_task);

        for task in tasks {
            for rollout_idx in 0..rollouts_per_task {
                let start = Instant::now();
                let outcome = self
                    .run_single_rollout(harness, policy, task, rollout_idx)
                    .await;
                let duration_ms = start.elapsed().as_millis() as u64;
                match outcome {
                    Ok(mut res) => {
                        res.duration_ms = duration_ms;
                        results.push(res);
                    }
                    Err(err) => {
                        // Infrastructure failures stay in denominator with score 0 (Appendix B.3).
                        results.push(TaskRolloutResult {
                            task_id: task.id.clone(),
                            rollout_idx,
                            passed: false,
                            score: 0.0,
                            trace_summary: "infrastructure error".to_string(),
                            error_trace: Some(err.to_string()),
                            tool_calls: 0,
                            duration_ms,
                        });
                    }
                }
            }
        }

        Ok(EvalFeedback::from_rollouts(
            harness_version_id,
            &policy.model_id,
            tasks,
            rollouts_per_task,
            results,
        ))
    }
}

impl AllyHRuntimeEvaluator {
    async fn run_single_rollout(
        &self,
        harness: &HarnessSpec,
        policy: &ModelProfile,
        task: &EvalTask,
        rollout_idx: usize,
    ) -> Result<TaskRolloutResult> {
        let tmp = TempDir::new()?;
        let repo_dir = tmp.path().join("repo");
        let db_dir = tmp.path().join("db");
        std::fs::create_dir_all(&repo_dir)?;

        for (rel_path, content) in &task.setup_files {
            let full = repo_dir.join(rel_path);
            if let Some(parent) = full.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&full, content)?;
        }

        let cfg = OrchestratorConfig {
            repo_root: repo_dir.clone(),
            storage: StorageConfig::from_local_path(&db_dir),
            scope_id: format!("sim-eval-{rollout_idx}"),
            auto_verify: harness.tools.auto_verify,
            idle_optimize_interval: Duration::from_secs(3600),
            enable_fs_watcher: false,
            extra_mcp_servers: Vec::new(),
        };
        let router = HybridRouter::new(
            policy.clone(),
            self.worker_profile.clone(),
            Arc::clone(&self.backend),
        );
        let mut orch = AllyHOrchestrator::initialize(cfg, router, None).await?;
        orch.set_active_harness_spec(harness.clone(), None)?;
        // Unlock writes inside isolated eval sandbox so tasks can exercise edit_by_hash.
        orch.toolset().set_writes_enabled(true);

        let turn = orch.handle_prompt(&task.prompt, None).await?;

        let mut passed = true;
        let mut failure_reasons = Vec::new();

        // Check expected substrings against assistant text + assembled preamble context.
        let combined = format!("{}\n{}", turn.preamble.text, turn.assistant_text);
        for needle in &task.expected_substrings {
            if !combined.contains(needle) {
                passed = false;
                failure_reasons.push(format!("missing expected substring '{needle}'"));
            }
        }

        // Run optional frozen shell verifier inside the isolated temp repo.
        if let Some(cmd_str) = task
            .verifier_command
            .as_deref()
            .filter(|s| !s.trim().is_empty())
        {
            let output = run_shell_command(&repo_dir, cmd_str).await?;
            if !output.status.success() {
                passed = false;
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                failure_reasons.push(format!(
                    "verifier `{cmd_str}` failed: {}",
                    truncate_str(&format!("{stdout}\n{stderr}"), 300)
                ));
            }
        }

        Ok(TaskRolloutResult {
            task_id: task.id.clone(),
            rollout_idx,
            passed,
            score: if passed { 1.0 } else { 0.0 },
            trace_summary: truncate_str(&turn.assistant_text, 240),
            error_trace: if failure_reasons.is_empty() {
                None
            } else {
                Some(failure_reasons.join("; "))
            },
            tool_calls: turn.tool_records.len(),
            duration_ms: 0,
        })
    }
}

/// Pluggable external benchmark evaluator for code-level or container-backed harnesses
/// (e.g. Terminal-Bench / SWE / Workspace / Engineering runners).
///
/// Invokes `program [args...] --harness-file <json> --tasks-file <json> --model <spec> --rollouts <N>`
/// and expects a JSON-serialized [`EvalFeedback`] on stdout.
pub struct CommandHarnessEvaluator {
    pub program: String,
    pub base_args: Vec<String>,
    pub working_dir: Option<PathBuf>,
}

#[async_trait]
impl BenchmarkEvaluator for CommandHarnessEvaluator {
    async fn evaluate(
        &self,
        harness: &HarnessSpec,
        harness_version_id: &str,
        policy: &ModelProfile,
        tasks: &[EvalTask],
        rollouts_per_task: usize,
    ) -> Result<EvalFeedback> {
        let tmp = TempDir::new()?;
        let harness_path = tmp.path().join("harness_spec.json");
        let tasks_path = tmp.path().join("tasks.json");
        std::fs::write(&harness_path, harness.canonical_json()?)?;
        std::fs::write(&tasks_path, serde_json::to_string_pretty(tasks)?)?;

        let mut cmd = Command::new(&self.program);
        cmd.args(&self.base_args)
            .arg("--harness-file")
            .arg(&harness_path)
            .arg("--tasks-file")
            .arg(&tasks_path)
            .arg("--model")
            .arg(&policy.model_id)
            .arg("--rollouts")
            .arg(rollouts_per_task.max(1).to_string());
        if let Some(dir) = &self.working_dir {
            cmd.current_dir(dir);
        }

        let output = cmd.output().await?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(SimError::Evaluation(format!(
                "external evaluator `{}` failed: {}",
                self.program,
                truncate_str(&stderr, 400)
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut feedback: EvalFeedback = serde_json::from_str(&stdout).map_err(|e| {
            SimError::Evaluation(format!(
                "failed to parse EvalFeedback JSON from `{}`: {e}",
                self.program
            ))
        })?;
        feedback.harness_version_id = harness_version_id.to_string();
        Ok(feedback)
    }
}

/// Multi-model unified synthesis evaluator (arXiv:2611.18856 §4.3.3 S2 / Appendix C.6):
/// pools rollout feedback from multiple policy models per round and averages scores equally.
pub struct UnifiedMultiModelEvaluator {
    inner: Arc<dyn BenchmarkEvaluator>,
    policies: Vec<ModelProfile>,
}

impl UnifiedMultiModelEvaluator {
    pub fn new(inner: Arc<dyn BenchmarkEvaluator>, policies: Vec<ModelProfile>) -> Result<Self> {
        if policies.is_empty() {
            return Err(SimError::InvalidSuite(
                "UnifiedMultiModelEvaluator requires at least one policy model".to_string(),
            ));
        }
        Ok(Self { inner, policies })
    }
}

#[async_trait]
impl BenchmarkEvaluator for UnifiedMultiModelEvaluator {
    async fn evaluate(
        &self,
        harness: &HarnessSpec,
        harness_version_id: &str,
        _policy: &ModelProfile,
        tasks: &[EvalTask],
        rollouts_per_task: usize,
    ) -> Result<EvalFeedback> {
        let mut all_rollouts = Vec::new();
        let mut total_score_sum = 0.0f32;
        let mut passed_total = 0u32;
        let mut total_rollouts = 0u32;
        let mut task_rate_sums: BTreeMap<String, f32> = tasks
            .iter()
            .map(|t| (t.id.clone(), 0.0f32))
            .collect();

        for policy in &self.policies {
            let fb = self
                .inner
                .evaluate(harness, harness_version_id, policy, tasks, rollouts_per_task)
                .await?;
            total_score_sum += fb.score;
            passed_total += fb.passed_rollouts;
            total_rollouts += fb.total_rollouts;
            for (tid, rate) in fb.task_pass_rates {
                *task_rate_sums.entry(tid).or_insert(0.0) += rate;
            }
            for mut r in fb.rollouts {
                r.trace_summary = format!("[{}] {}", policy.model_id, r.trace_summary);
                all_rollouts.push(r);
            }
        }

        let n_models = self.policies.len() as f32;
        let task_pass_rates = task_rate_sums
            .into_iter()
            .map(|(k, v)| (k, v / n_models))
            .collect();
        let policy_label = self
            .policies
            .iter()
            .map(|p| p.model_id.as_str())
            .collect::<Vec<_>>()
            .join("+");

        Ok(EvalFeedback {
            harness_version_id: harness_version_id.to_string(),
            policy_model: format!("unified:{policy_label}"),
            rollouts_per_task,
            total_tasks: tasks.len(),
            passed_rollouts: passed_total,
            total_rollouts,
            score: total_score_sum / n_models,
            task_pass_rates,
            rollouts: all_rollouts,
        })
    }
}

async fn run_shell_command(cwd: &Path, cmd_str: &str) -> Result<std::process::Output> {
    #[cfg(target_os = "windows")]
    let output = Command::new("cmd")
        .args(["/C", cmd_str])
        .current_dir(cwd)
        .output()
        .await?;
    #[cfg(not(target_os = "windows"))]
    let output = Command::new("sh")
        .args(["-c", cmd_str])
        .current_dir(cwd)
        .output()
        .await?;
    Ok(output)
}

fn deterministic_shuffle<T>(slice: &mut [T], mut state: u64) {
    for i in (1..slice.len()).rev() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let j = (z as usize) % (i + 1);
        slice.swap(i, j);
    }
}

fn truncate_str(text: &str, max_chars: usize) -> String {
    let single = text.replace('\n', " ");
    if single.len() <= max_chars {
        return single;
    }
    let cut = single
        .char_indices()
        .take_while(|(idx, _)| *idx < max_chars)
        .last()
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    format!("{}...", &single[..cut])
}
