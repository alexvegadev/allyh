use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use allyh_env::{EnvDelta, EnvironmentMonitor, LiveFsWatcher};
use allyh_memory::{
    CandidateDecision, ConsolidationOutcome, DbStatusReport, DeterministicHashEmbedder,
    EmbeddingProvider, HarnessVersionRecord, MemoryConsolidator, MemoryStore, ScoredDecision,
    SessionRecord, SimExportDocument, SimLedger, SimProblemEntry, SimRunRecord, SkillRecord,
    SleepJob, SleepTimeWorkerHandle, StorageConfig, TaskRecord, WorkerStatus, compute_repo_id,
};
use allyh_tools::{AciToolset, ToolExecutionRecord};
use chrono::Utc;
use rig_core::message::{Message, ToolResultContent, UserContent};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::compaction::CompactingHistory;
use crate::consolidation::{LocalLlmConflictJudge, extract_candidate_decisions};
use crate::error::Result;
use crate::gate::{ControlGate, ControlGateState, ProposedPlan, ProposedTaskSpec};
use crate::harness::HarnessSpec;
use crate::interlock::{CompletionInterlock, InterlockDecision};
use crate::preamble::{AssembledPreamble, DynamicPreambleAssembler, PreambleInputs};
use crate::router::{HybridRouter, ModelProfile};

/// Live snapshot pushed to the TUI Coherence Inspector sidebar & status bar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectorSnapshot {
    pub repo_id: String,
    pub scope_id: String,
    pub storage_uri: String,
    pub storage_backend_label: String,
    pub git_branch: String,
    pub dirty_files: Vec<String>,
    pub out_of_band_edits: Vec<String>,
    pub build_status: String,
    pub gate_state: ControlGateState,
    pub active_tasks: Vec<TaskRecord>,
    pub activated_memories: Vec<ScoredDecision>,
    pub skills_count: usize,
    pub active_harness_label: String,
    pub sim_problems_count: usize,
    pub sim_run_id: Option<String>,
    pub active_model_label: String,
    pub worker_model_label: String,
    pub worker_status: WorkerStatus,
    pub tokens_used: u64,
    pub token_budget: u64,
}

/// Commands dispatched from the TUI (or programmatic client) to the core orchestrator actor.
#[derive(Debug, Clone)]
pub enum UserCommand {
    SendPrompt(String),
    ProposePlan(ProposedPlan),
    ApprovePlan,
    RefinePlan(String),
    SwitchModel(String),
    PinMemory(String),
    InvalidateMemory {
        decision_id: String,
        reason: Option<String>,
    },
    RecordDecision(CandidateDecision),
    RequestMemories,
    RequestSkills,
    ShowSkill(String),
    PinSkill(String),
    RemoveSkill(String),
    UpsertSkill {
        name: String,
        description: String,
        instructions: String,
        triggers: Vec<String>,
    },
    RequestSimSummary,
    RequestSimProblems,
    ShowSimProblem(String),
    ExportSim(PathBuf),
    ShowActiveHarness,
    UseHarness(String),
    RequestDbStatus,
    RefreshEnv { run_build_check: bool },
    CompactNow,
    Shutdown,
}

/// Events streamed from the core orchestrator back to the 60 FPS Ratatui loop.
#[derive(Debug, Clone)]
pub enum UiEvent {
    TokenDelta(String),
    AssistantMessageFinished {
        text: String,
        preamble_tokens: u64,
    },
    ToolCard(ToolExecutionRecord),
    GateChanged(ControlGateState),
    InspectorUpdated(InspectorSnapshot),
    WorkerStatusChanged(WorkerStatus),
    MemoriesList(Vec<ScoredDecision>),
    SkillsList {
        skills: Vec<SkillRecord>,
        pinned_ids: HashSet<String>,
    },
    SkillDetail {
        skill: SkillRecord,
        pinned: bool,
    },
    SimSummary {
        ledger: Option<SimLedger>,
        active_harness_label: String,
        active_harness_hash: String,
    },
    SimProblems(Option<SimLedger>),
    SimProblemDetail {
        problem: SimProblemEntry,
        resolved_from: Option<String>,
    },
    ActiveHarnessDetail {
        label: String,
        version_id: Option<String>,
        spec: HarnessSpec,
    },
    DbStatus(DbStatusReport),
    SystemNotice(String),
    Error(String),
}

/// Outcome returned by [`AllyHOrchestrator::handle_prompt`].
#[derive(Debug, Clone)]
pub struct TurnOutcome {
    pub assistant_text: String,
    pub preamble: AssembledPreamble,
    pub env_delta: EnvDelta,
    pub tool_records: Vec<ToolExecutionRecord>,
    pub consolidated_decisions: Vec<ConsolidationOutcome>,
}

/// Configuration for initializing an [`AllyHOrchestrator`] session.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    pub repo_root: PathBuf,
    pub storage: StorageConfig,
    pub scope_id: String,
    pub auto_cargo_check: bool,
    pub idle_optimize_interval: Duration,
    pub enable_fs_watcher: bool,
}

impl OrchestratorConfig {
    pub fn new(repo_root: impl Into<PathBuf>) -> Self {
        Self {
            repo_root: repo_root.into(),
            storage: StorageConfig::default_local(),
            scope_id: allyh_memory::DEFAULT_SCOPE_ID.to_string(),
            auto_cargo_check: true,
            idle_optimize_interval: Duration::from_secs(45),
            enable_fs_watcher: true,
        }
    }
}

/// Core AllyH orchestrator wiring LanceDB memory, environment monitoring, compact ACI tools,
/// Control Gate, hybrid model routing, history compaction, and sleep-time consolidation.
pub struct AllyHOrchestrator {
    session_id: String,
    repo_id: String,
    scope_id: String,
    turn_counter: u64,
    store: MemoryStore,
    consolidator: MemoryConsolidator,
    sleep_worker: SleepTimeWorkerHandle,
    env: Arc<EnvironmentMonitor>,
    toolset: AciToolset,
    gate: ControlGate,
    router: HybridRouter,
    history: CompactingHistory,
    active_harness: HarnessSpec,
    active_harness_record: Option<HarnessVersionRecord>,
    pinned_decision_ids: HashSet<String>,
    pinned_skill_ids: HashSet<String>,
    last_env_delta: EnvDelta,
    last_activated_memories: Vec<ScoredDecision>,
    last_preamble_tokens: u64,
    _fs_watcher: Option<LiveFsWatcher>,
}

impl AllyHOrchestrator {
    /// Initialize a new AllyH session connected to out-of-tree LanceDB storage.
    /// Runs mandatory startup environment checks & restores the latest rolling session summary.
    pub async fn initialize(
        config: OrchestratorConfig,
        router: HybridRouter,
        embedder: Option<Arc<dyn EmbeddingProvider>>,
    ) -> Result<Self> {
        let embedder = embedder.unwrap_or_else(|| Arc::new(DeterministicHashEmbedder));
        let store = MemoryStore::connect(config.storage, embedder).await?;

        let git_tracker = allyh_env::GitTracker::new(&config.repo_root);
        let canonical_id = git_tracker.canonical_repo_identifier();
        let repo_id = compute_repo_id(&canonical_id);
        let scope_id = config.scope_id;

        let env = Arc::new(EnvironmentMonitor::new(
            &config.repo_root,
            repo_id.clone(),
            scope_id.clone(),
            store.clone(),
        ));

        let fs_watcher = if config.enable_fs_watcher {
            env.start_watching().ok().map(|(w, _rx)| w)
        } else {
            None
        };

        let writes_enabled = Arc::new(AtomicBool::new(false));
        let gate = ControlGate::new(Arc::clone(&writes_enabled));
        let toolset = AciToolset::new(
            Arc::clone(&env),
            store.clone(),
            writes_enabled,
            config.auto_cargo_check,
        );

        let judge = Arc::new(LocalLlmConflictJudge::new(router.clone()));
        let consolidator = MemoryConsolidator::new(store.clone(), judge);
        let sleep_worker =
            SleepTimeWorkerHandle::spawn(consolidator.clone(), config.idle_optimize_interval);

        // Restore previous session's rolling summary for cross-session continuity.
        let prior_summary = store
            .latest_session(&repo_id, &scope_id)
            .await?
            .map(|s| s.rolling_summary)
            .unwrap_or_default();

        // Mandatory startup environment check & initial RepoGraph indexing.
        let initial_delta = env.sync_and_collect_delta(0, false).await?;

        let session_id = Uuid::new_v4().to_string();
        let initial_session = SessionRecord {
            session_id: session_id.clone(),
            repo_id: repo_id.clone(),
            scope_id: scope_id.clone(),
            started_at: Utc::now(),
            ended_at: None,
            rolling_summary: prior_summary.clone(),
            token_budget_used: 0,
        };
        store.upsert_session(&initial_session).await?;

        let initial_active = store
            .list_active_decisions(&repo_id, &scope_id)
            .await?
            .into_iter()
            .take(8)
            .map(|decision| ScoredDecision {
                decision,
                score: 1.0,
                pinned: false,
            })
            .collect();

        let active_harness_record = store
            .get_active_harness_version(&repo_id, &scope_id)
            .await?;
        let active_harness = match &active_harness_record {
            Some(rec) => HarnessSpec::from_version_record(rec).unwrap_or_else(|err| {
                tracing::warn!(error = %err, version_id = %rec.id, "falling back to default seed harness");
                HarnessSpec::default_seed()
            }),
            None => HarnessSpec::default_seed(),
        };

        let mut history = CompactingHistory::new(prior_summary);
        history.apply_policy(
            active_harness.compaction.keep_recent_units,
            active_harness.compaction.compaction_ratio,
            active_harness.compaction.max_rollup_chars,
        );

        Ok(Self {
            session_id,
            repo_id,
            scope_id,
            turn_counter: 0,
            store,
            consolidator,
            sleep_worker,
            env,
            toolset,
            gate,
            router,
            history,
            active_harness,
            active_harness_record,
            pinned_decision_ids: HashSet::new(),
            pinned_skill_ids: HashSet::new(),
            last_env_delta: initial_delta,
            last_activated_memories: initial_active,
            last_preamble_tokens: 0,
            _fs_watcher: fs_watcher,
        })
    }

    pub fn repo_id(&self) -> &str {
        &self.repo_id
    }

    pub fn scope_id(&self) -> &str {
        &self.scope_id
    }

    pub fn store(&self) -> &MemoryStore {
        &self.store
    }

    pub fn env(&self) -> &Arc<EnvironmentMonitor> {
        &self.env
    }

    pub fn toolset(&self) -> &AciToolset {
        &self.toolset
    }

    pub fn gate(&self) -> &ControlGate {
        &self.gate
    }

    pub fn router(&self) -> &HybridRouter {
        &self.router
    }

    pub fn active_harness(&self) -> &HarnessSpec {
        &self.active_harness
    }

    pub fn active_harness_record(&self) -> Option<&HarnessVersionRecord> {
        self.active_harness_record.as_ref()
    }

    pub fn active_harness_label(&self) -> String {
        match &self.active_harness_record {
            Some(rec) => format!("{} (r{})", rec.label, rec.round),
            None => "seed (default)".to_string(),
        }
    }

    /// Apply an in-memory [`HarnessSpec`] immediately (used by CRSI evaluators and tests).
    pub fn set_active_harness_spec(
        &mut self,
        spec: HarnessSpec,
        record: Option<HarnessVersionRecord>,
    ) -> Result<()> {
        spec.validate()?;
        self.history.apply_policy(
            spec.compaction.keep_recent_units,
            spec.compaction.compaction_ratio,
            spec.compaction.max_rollup_chars,
        );
        self.active_harness = spec;
        self.active_harness_record = record;
        Ok(())
    }

    /// Activate a persisted [`HarnessVersionRecord`] from LanceDB by ID, prefix, or label.
    pub async fn activate_harness_version(
        &mut self,
        version_id_or_prefix: &str,
    ) -> Result<HarnessVersionRecord> {
        let rec = self
            .store
            .set_active_harness_version(&self.repo_id, &self.scope_id, version_id_or_prefix)
            .await?;
        let spec = HarnessSpec::from_version_record(&rec)?;
        self.set_active_harness_spec(spec, Some(rec.clone()))?;
        Ok(rec)
    }

    /// Detect whether the active reasoning model differs from the policy model recorded on
    /// the active evolved harness or latest SIM run (cross-model CRSI shift signal).
    pub async fn model_shift_notice(&self) -> Option<String> {
        let current_model = &self.router.primary().model_id;
        let source_policy = self
            .active_harness_record
            .as_ref()
            .map(|r| r.policy_model.as_str())
            .filter(|p| !p.trim().is_empty());

        let latest_run = self
            .store
            .latest_sim_run(&self.repo_id, &self.scope_id)
            .await
            .ok()
            .flatten();

        let prev_policy = source_policy.or_else(|| {
            latest_run
                .as_ref()
                .map(|r| r.policy_model.as_str())
                .filter(|p| !p.trim().is_empty())
        })?;

        if !prev_policy.eq_ignore_ascii_case(current_model)
            && !prev_policy.ends_with(current_model)
        {
            let run_hint = latest_run
                .as_ref()
                .map(|r| format!(" (`allyh sim continue --from-run {}`)", &r.run_id[..r.run_id.len().min(8)]))
                .unwrap_or_default();
            return Some(format!(
                "Cross-model shift detected: active harness/SIM was tuned for `{prev_policy}`, now running `{current_model}`. Inspect with `/sim` or continue CRSI{run_hint}."
            ));
        }
        None
    }

    pub async fn load_sim_ledger(&self, run_id_or_arm: &str) -> Result<Option<SimLedger>> {
        Ok(self
            .store
            .load_sim_ledger(&self.repo_id, &self.scope_id, run_id_or_arm)
            .await?)
    }

    pub async fn list_sim_runs(&self) -> Result<Vec<SimRunRecord>> {
        Ok(self
            .store
            .list_sim_runs(&self.repo_id, &self.scope_id)
            .await?)
    }

    pub async fn export_sim_document(
        &self,
        run_id_or_arm: Option<&str>,
    ) -> Result<Option<SimExportDocument>> {
        Ok(self
            .store
            .export_sim_document(&self.repo_id, &self.scope_id, run_id_or_arm)
            .await?)
    }

    pub fn switch_model(&mut self, spec: &str) -> Result<ModelProfile> {
        let profile = ModelProfile::parse_spec(spec)?;
        self.router.set_primary(profile.clone());
        Ok(profile)
    }

    pub fn propose_plan(&mut self, plan: ProposedPlan) -> ControlGateState {
        self.gate.propose_plan(plan).clone()
    }

    pub async fn approve_plan(&mut self) -> Result<Vec<TaskRecord>> {
        self.gate
            .approve_plan(&self.store, &self.repo_id, &self.scope_id)
            .await
    }

    pub fn refine_plan(&mut self) -> ControlGateState {
        self.gate.refine_plan().clone()
    }

    /// Pin or unpin an architectural decision so it is always injected into the dynamic preamble.
    pub fn toggle_pin_memory(&mut self, decision_id: &str) -> bool {
        if self.pinned_decision_ids.contains(decision_id) {
            self.pinned_decision_ids.remove(decision_id);
            false
        } else {
            self.pinned_decision_ids.insert(decision_id.to_string());
            true
        }
    }

    /// Logically invalidate a stale architectural decision (`invalid_at = NOW()`).
    pub async fn invalidate_memory(&mut self, decision_id: &str) -> Result<u64> {
        self.pinned_decision_ids.remove(decision_id);
        let updated = self
            .store
            .invalidate_decision(&self.repo_id, &self.scope_id, decision_id, None, Utc::now())
            .await?;
        self.refresh_activated_memories("").await?;
        Ok(updated)
    }

    /// Explicitly commit a candidate architectural decision through the write-gated bi-temporal consolidator.
    pub async fn commit_decision(
        &mut self,
        candidate: CandidateDecision,
    ) -> Result<ConsolidationOutcome> {
        let outcome = self
            .consolidator
            .consolidate_decision(&self.repo_id, &self.scope_id, candidate)
            .await?;
        self.refresh_activated_memories("").await?;
        Ok(outcome)
    }

    /// Query live LanceDB metrics for `/db-status`.
    pub async fn db_status(&self) -> Result<DbStatusReport> {
        Ok(self.store.status_report(&self.repo_id, &self.scope_id).await?)
    }

    /// List all skills registered for this `(repo_id, scope_id)` partition.
    pub async fn list_skills(&self) -> Result<Vec<SkillRecord>> {
        Ok(self.store.list_skills(&self.repo_id, &self.scope_id).await?)
    }

    /// Look up a single skill by name, slug, or id.
    pub async fn get_skill(&self, name_or_id: &str) -> Result<Option<SkillRecord>> {
        Ok(self
            .store
            .get_skill(&self.repo_id, &self.scope_id, name_or_id)
            .await?)
    }

    /// Create or update a skill in LanceDB (user-initiated; works in Discovery mode).
    pub async fn save_skill(
        &self,
        name: &str,
        description: &str,
        instructions: &str,
        triggers: Vec<String>,
    ) -> Result<(SkillRecord, bool)> {
        Ok(self
            .store
            .save_skill(
                &self.repo_id,
                &self.scope_id,
                name,
                description,
                instructions,
                triggers,
            )
            .await?)
    }

    /// Remove a skill by name, slug, or id from LanceDB.
    pub async fn remove_skill(&mut self, name_or_id: &str) -> Result<Option<SkillRecord>> {
        let removed = self
            .store
            .delete_skill(&self.repo_id, &self.scope_id, name_or_id)
            .await?;
        if let Some(ref skill) = removed {
            self.pinned_skill_ids.remove(&skill.id);
        }
        Ok(removed)
    }

    /// Pin or unpin a skill so its full instructions are always injected into the preamble.
    pub async fn toggle_pin_skill(&mut self, name_or_id: &str) -> Result<Option<(SkillRecord, bool)>> {
        let Some(skill) = self.get_skill(name_or_id).await? else {
            return Ok(None);
        };
        let pinned = if self.pinned_skill_ids.contains(&skill.id) {
            self.pinned_skill_ids.remove(&skill.id);
            false
        } else {
            self.pinned_skill_ids.insert(skill.id.clone());
            true
        };
        Ok(Some((skill, pinned)))
    }

    pub fn pinned_skill_ids(&self) -> &HashSet<String> {
        &self.pinned_skill_ids
    }

    /// Refresh environment delta (`git2`, SHA-256 hash comparison, incremental `tree-sitter` indexing).
    pub async fn refresh_env(&mut self, run_build_check: bool) -> Result<EnvDelta> {
        let delta = self
            .env
            .sync_and_collect_delta(self.turn_counter, run_build_check)
            .await?;
        self.last_env_delta = delta.clone();
        Ok(delta)
    }

    /// Build the live [`InspectorSnapshot`] displayed in the TUI sidebar & status bar.
    pub async fn inspector_snapshot(&self) -> Result<InspectorSnapshot> {
        let tasks = self.store.list_tasks(&self.repo_id, &self.scope_id).await?;
        let skills_count = self
            .store
            .list_skills(&self.repo_id, &self.scope_id)
            .await?
            .len();
        let latest_sim = self
            .store
            .load_sim_ledger(&self.repo_id, &self.scope_id, "")
            .await?;
        let sim_problems_count = latest_sim.as_ref().map(|l| l.problems.len()).unwrap_or(0);
        let sim_run_id = latest_sim.map(|l| l.run.run_id);
        let mut memories = self.last_activated_memories.clone();
        for m in &mut memories {
            m.pinned = self.pinned_decision_ids.contains(&m.decision.id);
        }

        let oob = self
            .last_env_delta
            .out_of_band_edits
            .iter()
            .map(|e| format!("{}: {}", e.file_path, e.diff_summary))
            .collect();

        Ok(InspectorSnapshot {
            repo_id: self.repo_id.clone(),
            scope_id: self.scope_id.clone(),
            storage_uri: self.store.config().uri.clone(),
            storage_backend_label: self
                .store
                .config()
                .backend_kind()
                .display_label()
                .to_string(),
            git_branch: self.last_env_delta.git.branch.clone(),
            dirty_files: self.last_env_delta.git.dirty_files.clone(),
            out_of_band_edits: oob,
            build_status: self
                .last_env_delta
                .build_status
                .clone()
                .unwrap_or_else(|| "Idle".to_string()),
            gate_state: self.gate.state().clone(),
            active_tasks: tasks,
            activated_memories: memories,
            skills_count,
            active_harness_label: self.active_harness_label(),
            sim_problems_count,
            sim_run_id,
            active_model_label: self.router.primary().label.clone(),
            worker_model_label: self.router.local_worker().label.clone(),
            worker_status: self.sleep_worker.status(),
            tokens_used: self.last_preamble_tokens + self.history.estimated_tokens(),
            token_budget: self.router.primary().context_window_tokens,
        })
    }

    /// Execute one complete interactive agent turn:
    /// 1. Collect live `EnvDelta` (detecting out-of-band SHA-256 manual edits).
    /// 2. Retrieve top-k active `architectural_decisions` (`invalid_at IS NULL`) + pinned rules.
    /// 3. Extract 1-hop `RepoGraph` neighborhoods and assemble the dynamic preamble.
    /// 4. Compact history non-destructively if near context budget.
    /// 5. Run the `rig-core` completion + ACI tool-call loop (bounded to 6 tool rounds).
    /// 6. Extract & consolidate architectural decisions through the bi-temporal write gate
    ///    and enqueue sleep-time optimization.
    pub async fn handle_prompt(
        &mut self,
        prompt: &str,
        ui_tx: Option<&mpsc::UnboundedSender<UiEvent>>,
    ) -> Result<TurnOutcome> {
        self.turn_counter += 1;

        // 1. Live environment check & out-of-band edit detection.
        let env_delta = self
            .env
            .sync_and_collect_delta(self.turn_counter, false)
            .await?;
        self.last_env_delta = env_delta.clone();

        // 2. Hybrid search active architectural decisions + merge pinned rules.
        let memories = self.refresh_activated_memories(prompt).await?;

        // 3. Gather active tasks, reusable skills (catalog + pinned/matched), and 1-hop ego-graphs.
        let tasks = self.store.list_tasks(&self.repo_id, &self.scope_id).await?;
        let available_skills = self.store.list_skills(&self.repo_id, &self.scope_id).await?;
        let mut active_skills: Vec<SkillRecord> = available_skills
            .iter()
            .filter(|s| self.pinned_skill_ids.contains(&s.id))
            .cloned()
            .collect();
        let matched_skills = self
            .store
            .match_skills(
                &self.repo_id,
                &self.scope_id,
                prompt,
                self.active_harness.retrieval.skill_match_top_k,
            )
            .await?;
        for skill in matched_skills {
            if !active_skills.iter().any(|s| s.id == skill.id) {
                active_skills.push(skill);
            }
        }

        let mut sym_queries = DynamicPreambleAssembler::extract_symbol_queries(prompt, &env_delta);
        sym_queries.truncate(self.active_harness.retrieval.max_symbol_queries.max(1));
        let mut ego_graphs = Vec::new();
        for q in &sym_queries {
            if let Ok(ego) = self
                .env
                .ego_graph_for(q, self.active_harness.retrieval.ego_graph_hops)
                .await
                && (!ego.center_definitions.is_empty() || !ego.hop1_neighbors.is_empty())
            {
                ego_graphs.push(ego);
            }
        }

        let preamble = DynamicPreambleAssembler::assemble(&PreambleInputs {
            repo_id: &self.repo_id,
            scope_id: &self.scope_id,
            gate_state: self.gate.state(),
            harness: &self.active_harness,
            env_delta: &env_delta,
            tasks: &tasks,
            memories: &memories,
            available_skills: &available_skills,
            active_skills: &active_skills,
            ego_graphs: &ego_graphs,
            rolling_summary: Some(self.history.rolling_summary()),
        });
        self.last_preamble_tokens = preamble.estimated_tokens;

        // 4. Non-destructive history compaction if approaching token budget.
        self.history
            .compact_if_needed(
                &self.router,
                preamble.estimated_tokens,
                self.router.primary().context_window_tokens,
                false,
            )
            .await;

        self.history.push_message(Message::user(prompt));

        // 5. Run completion + compact ACI tool-use loop with CompletionInterlock.
        let tool_defs = self.toolset.definitions();
        let mut tool_records = Vec::new();
        let mut final_text = String::new();
        let max_rounds = self.active_harness.tools.max_tool_rounds.max(1);
        let mut interlock = CompletionInterlock::new(self.active_harness.completion_interlock.clone());

        for round in 0..max_rounds {
            let (tok_tx, mut tok_rx) = mpsc::unbounded_channel::<String>();
            let ui_clone = ui_tx.cloned();
            let forward_handle = tokio::spawn(async move {
                while let Some(chunk) = tok_rx.recv().await {
                    if let Some(tx) = &ui_clone {
                        let _ = tx.send(UiEvent::TokenDelta(chunk));
                    }
                }
            });

            let messages = self.history.messages();
            let response = self
                .router
                .run_primary(&preamble.text, &messages, &tool_defs, Some(tok_tx))
                .await?;
            let _ = forward_handle.await;

            let calls: Vec<_> = response.tool_calls().cloned().collect();
            let text = response.text();
            if !text.is_empty() {
                final_text = text;
            }

            if calls.is_empty() {
                if let Some(msg) = response.message() {
                    self.history.push_message(msg);
                }
                if round + 1 < max_rounds
                    && let InterlockDecision::Reject { reminder, .. } =
                        interlock.check_completion(&final_text)
                {
                    if let Some(tx) = ui_tx {
                        let _ = tx.send(UiEvent::SystemNotice(reminder.clone()));
                    }
                    self.history.push_message(Message::user(reminder));
                    continue;
                }
                break;
            }

            // Preserve intact assistant tool-calls + user tool-results as ONE atomic history unit!
            let mut atomic_pair = Vec::new();
            if let Some(ast_msg) = response.message() {
                atomic_pair.push(ast_msg);
            }
            let mut result_contents = Vec::new();
            let mut round_records = Vec::new();
            for call in calls {
                let (record, output) = self
                    .toolset
                    .dispatch(
                        call.function.name.as_str(),
                        serde_json::Value::Object(call.function.arguments.clone()),
                    )
                    .await;
                if let Some(tx) = ui_tx {
                    let _ = tx.send(UiEvent::ToolCard(record.clone()));
                }
                round_records.push(record.clone());
                tool_records.push(record);
                let formatted = self.active_harness.format_tool_output(&output.render());
                result_contents.push(UserContent::tool_result(
                    call.id,
                    call.function.name,
                    vec![ToolResultContent::text(formatted)],
                ));
            }
            interlock.observe_tool_round(&round_records);
            if !result_contents.is_empty() {
                atomic_pair.push(Message::User {
                    content: result_contents,
                });
            }
            self.history.push_atomic_group(atomic_pair);

            if round + 1 < max_rounds
                && let Some(budget_notice) = interlock.budget_reminder_if_due(round, max_rounds)
            {
                self.history.push_message(Message::user(budget_notice));
            }
        }

        // 6. Post-turn bi-temporal decision consolidation & sleep-time optimization.
        let candidates = extract_candidate_decisions(&self.router, prompt, &final_text).await;
        let mut consolidated_decisions = Vec::new();
        for cand in candidates {
            if let Ok(outcome) = self
                .consolidator
                .consolidate_decision(&self.repo_id, &self.scope_id, cand)
                .await
            {
                consolidated_decisions.push(outcome);
            }
        }
        if !consolidated_decisions.is_empty() {
            let _ = self.refresh_activated_memories(prompt).await;
            self.sleep_worker.enqueue(SleepJob::OptimizeTables);
        }

        // Persist updated session record out-of-tree.
        let session_record = SessionRecord {
            session_id: self.session_id.clone(),
            repo_id: self.repo_id.clone(),
            scope_id: self.scope_id.clone(),
            started_at: Utc::now(),
            ended_at: None,
            rolling_summary: self.history.rolling_summary().to_string(),
            token_budget_used: self.last_preamble_tokens + self.history.estimated_tokens(),
        };
        let _ = self.store.upsert_session(&session_record).await;

        if let Some(tx) = ui_tx {
            let _ = tx.send(UiEvent::AssistantMessageFinished {
                text: final_text.clone(),
                preamble_tokens: preamble.estimated_tokens,
            });
            if let Ok(snap) = self.inspector_snapshot().await {
                let _ = tx.send(UiEvent::InspectorUpdated(snap));
            }
        }

        Ok(TurnOutcome {
            assistant_text: final_text,
            preamble,
            env_delta,
            tool_records,
            consolidated_decisions,
        })
    }

    async fn refresh_activated_memories(&mut self, query: &str) -> Result<Vec<ScoredDecision>> {
        let top_k = self.active_harness.retrieval.decision_top_k.max(1);
        let fallback_k = self.active_harness.retrieval.fallback_decision_count;
        let mut results = if query.trim().is_empty() {
            self.store
                .list_active_decisions(&self.repo_id, &self.scope_id)
                .await?
                .into_iter()
                .take(top_k)
                .map(|decision| ScoredDecision {
                    decision,
                    score: 1.0,
                    pinned: false,
                })
                .collect()
        } else {
            self.store
                .hybrid_search_decisions(&self.repo_id, &self.scope_id, query, top_k)
                .await?
        };

        // Always include all active pinned decisions even if not top-k for the current query.
        if !self.pinned_decision_ids.is_empty() {
            let active = self
                .store
                .list_active_decisions(&self.repo_id, &self.scope_id)
                .await?;
            for dec in active {
                if self.pinned_decision_ids.contains(&dec.id)
                    && !results.iter().any(|r| r.decision.id == dec.id)
                {
                    results.insert(
                        0,
                        ScoredDecision {
                            decision: dec,
                            score: 2.0,
                            pinned: true,
                        },
                    );
                }
            }
        }
        // If query returned nothing (e.g. brand-new wording), still surface up to fallback_k active invariants.
        if results.is_empty() && fallback_k > 0 {
            results = self
                .store
                .list_active_decisions(&self.repo_id, &self.scope_id)
                .await?
                .into_iter()
                .take(fallback_k)
                .map(|decision| ScoredDecision {
                    decision,
                    score: 0.5,
                    pinned: false,
                })
                .collect();
        }

        for r in &mut results {
            r.pinned = self.pinned_decision_ids.contains(&r.decision.id);
        }
        self.last_activated_memories = results.clone();
        Ok(results)
    }

    /// Spawn the orchestrator as a background Tokio actor driven by `UserCommand`s
    /// and emitting `UiEvent`s for the non-blocking Ratatui loop.
    pub fn spawn_actor(
        mut self,
    ) -> (
        mpsc::UnboundedSender<UserCommand>,
        mpsc::UnboundedReceiver<UiEvent>,
    ) {
        let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<UserCommand>();
        let (ui_tx, ui_rx) = mpsc::unbounded_channel::<UiEvent>();

        let mut worker_status_rx = self.sleep_worker.subscribe_status();
        let ui_tx_status = ui_tx.clone();
        tokio::spawn(async move {
            while worker_status_rx.changed().await.is_ok() {
                let status = worker_status_rx.borrow().clone();
                let _ = ui_tx_status.send(UiEvent::WorkerStatusChanged(status));
            }
        });

        tokio::spawn(async move {
            if let Ok(snap) = self.inspector_snapshot().await {
                let _ = ui_tx.send(UiEvent::InspectorUpdated(snap));
            }

            while let Some(cmd) = cmd_rx.recv().await {
                match cmd {
                    UserCommand::SendPrompt(prompt) => {
                        if let Err(err) = self.handle_prompt(&prompt, Some(&ui_tx)).await {
                            let _ = ui_tx.send(UiEvent::Error(err.to_string()));
                        }
                    }
                    UserCommand::ProposePlan(plan) => {
                        let state = self.propose_plan(plan);
                        let _ = ui_tx.send(UiEvent::GateChanged(state));
                        if let Ok(snap) = self.inspector_snapshot().await {
                            let _ = ui_tx.send(UiEvent::InspectorUpdated(snap));
                        }
                    }
                    UserCommand::ApprovePlan => match self.approve_plan().await {
                        Ok(tasks) => {
                            let _ = ui_tx.send(UiEvent::GateChanged(self.gate.state().clone()));
                            let _ = ui_tx.send(UiEvent::SystemNotice(format!(
                                "Plan approved: {} task(s) persisted in LanceDB; write tools unlocked.",
                                tasks.len()
                            )));
                            if let Ok(snap) = self.inspector_snapshot().await {
                                let _ = ui_tx.send(UiEvent::InspectorUpdated(snap));
                            }
                        }
                        Err(err) => {
                            let _ = ui_tx.send(UiEvent::Error(err.to_string()));
                        }
                    },
                    UserCommand::RefinePlan(feedback) => {
                        let state = self.refine_plan();
                        let _ = ui_tx.send(UiEvent::GateChanged(state));
                        let _ = ui_tx.send(UiEvent::SystemNotice(format!(
                            "Plan returned to Discovery (read-only). Refinement note: {feedback}"
                        )));
                        if let Ok(snap) = self.inspector_snapshot().await {
                            let _ = ui_tx.send(UiEvent::InspectorUpdated(snap));
                        }
                    }
                    UserCommand::SwitchModel(spec) => match self.switch_model(&spec) {
                        Ok(profile) => {
                            let _ = ui_tx.send(UiEvent::SystemNotice(format!(
                                "Switched active reasoning model to {}",
                                profile.label
                            )));
                            if let Some(shift_msg) = self.model_shift_notice().await {
                                let _ = ui_tx.send(UiEvent::SystemNotice(shift_msg));
                            }
                            if let Ok(snap) = self.inspector_snapshot().await {
                                let _ = ui_tx.send(UiEvent::InspectorUpdated(snap));
                            }
                        }
                        Err(err) => {
                            let _ = ui_tx.send(UiEvent::Error(err.to_string()));
                        }
                    },
                    UserCommand::PinMemory(id) => {
                        let pinned = self.toggle_pin_memory(&id);
                        let short_id: String = id.chars().take(8).collect();
                        let _ = ui_tx.send(UiEvent::SystemNotice(format!(
                            "Memory `{short_id}` {}",
                            if pinned { "pinned" } else { "unpinned" }
                        )));
                        if let Ok(snap) = self.inspector_snapshot().await {
                            let _ = ui_tx.send(UiEvent::InspectorUpdated(snap));
                        }
                    }
                    UserCommand::InvalidateMemory {
                        decision_id,
                        reason,
                    } => match self.invalidate_memory(&decision_id).await {
                        Ok(n) => {
                            let short_id: String = decision_id.chars().take(8).collect();
                            let suffix = reason
                                .map(|r| format!(" ({r})"))
                                .unwrap_or_default();
                            let _ = ui_tx.send(UiEvent::SystemNotice(format!(
                                "Logically invalidated {n} memory record(s) for `{short_id}`{suffix}"
                            )));
                            if let Ok(snap) = self.inspector_snapshot().await {
                                let _ = ui_tx.send(UiEvent::InspectorUpdated(snap));
                            }
                        }
                        Err(err) => {
                            let _ = ui_tx.send(UiEvent::Error(err.to_string()));
                        }
                    },
                    UserCommand::RecordDecision(candidate) => {
                        match self.commit_decision(candidate).await {
                            Ok(outcome) => {
                                let _ = ui_tx.send(UiEvent::SystemNotice(format!(
                                    "Consolidated architectural invariant: {:?}",
                                    outcome.action
                                )));
                                if let Ok(snap) = self.inspector_snapshot().await {
                                    let _ = ui_tx.send(UiEvent::InspectorUpdated(snap));
                                }
                            }
                            Err(err) => {
                                let _ = ui_tx.send(UiEvent::Error(err.to_string()));
                            }
                        }
                    }
                    UserCommand::RequestMemories => match self.refresh_activated_memories("").await {
                        Ok(list) => {
                            let _ = ui_tx.send(UiEvent::MemoriesList(list));
                        }
                        Err(err) => {
                            let _ = ui_tx.send(UiEvent::Error(err.to_string()));
                        }
                    },
                    UserCommand::RequestSkills => match self.list_skills().await {
                        Ok(skills) => {
                            let _ = ui_tx.send(UiEvent::SkillsList {
                                skills,
                                pinned_ids: self.pinned_skill_ids.clone(),
                            });
                        }
                        Err(err) => {
                            let _ = ui_tx.send(UiEvent::Error(err.to_string()));
                        }
                    },
                    UserCommand::ShowSkill(name_or_id) => match self.get_skill(&name_or_id).await {
                        Ok(Some(skill)) => {
                            let pinned = self.pinned_skill_ids.contains(&skill.id);
                            let _ = ui_tx.send(UiEvent::SkillDetail { skill, pinned });
                        }
                        Ok(None) => {
                            let _ = ui_tx.send(UiEvent::Error(format!(
                                "Skill '{name_or_id}' not found. Run /skills to see registered skills."
                            )));
                        }
                        Err(err) => {
                            let _ = ui_tx.send(UiEvent::Error(err.to_string()));
                        }
                    },
                    UserCommand::PinSkill(name_or_id) => {
                        match self.toggle_pin_skill(&name_or_id).await {
                            Ok(Some((skill, pinned))) => {
                                let _ = ui_tx.send(UiEvent::SystemNotice(format!(
                                    "Skill `{}` {}",
                                    skill.name,
                                    if pinned {
                                        "pinned (always expanded in preamble)"
                                    } else {
                                        "unpinned"
                                    }
                                )));
                                if let Ok(snap) = self.inspector_snapshot().await {
                                    let _ = ui_tx.send(UiEvent::InspectorUpdated(snap));
                                }
                            }
                            Ok(None) => {
                                let _ = ui_tx.send(UiEvent::Error(format!(
                                    "Skill '{name_or_id}' not found. Run /skills to see registered skills."
                                )));
                            }
                            Err(err) => {
                                let _ = ui_tx.send(UiEvent::Error(err.to_string()));
                            }
                        }
                    }
                    UserCommand::RemoveSkill(name_or_id) => {
                        match self.remove_skill(&name_or_id).await {
                            Ok(Some(skill)) => {
                                let _ = ui_tx.send(UiEvent::SystemNotice(format!(
                                    "Removed skill `{}`.",
                                    skill.name
                                )));
                                if let Ok(snap) = self.inspector_snapshot().await {
                                    let _ = ui_tx.send(UiEvent::InspectorUpdated(snap));
                                }
                            }
                            Ok(None) => {
                                let _ = ui_tx.send(UiEvent::Error(format!(
                                    "Skill '{name_or_id}' not found. Run /skills to see registered skills."
                                )));
                            }
                            Err(err) => {
                                let _ = ui_tx.send(UiEvent::Error(err.to_string()));
                            }
                        }
                    }
                    UserCommand::UpsertSkill {
                        name,
                        description,
                        instructions,
                        triggers,
                    } => {
                        match self
                            .save_skill(&name, &description, &instructions, triggers)
                            .await
                        {
                            Ok((skill, created)) => {
                                let verb = if created { "Saved" } else { "Updated" };
                                let _ = ui_tx.send(UiEvent::SystemNotice(format!(
                                    "{verb} skill `{}` ({} trigger(s)).",
                                    skill.name,
                                    skill.triggers.len()
                                )));
                                if let Ok(snap) = self.inspector_snapshot().await {
                                    let _ = ui_tx.send(UiEvent::InspectorUpdated(snap));
                                }
                            }
                            Err(err) => {
                                let _ = ui_tx.send(UiEvent::Error(err.to_string()));
                            }
                        }
                    }
                    UserCommand::RequestSimSummary => match self.load_sim_ledger("").await {
                        Ok(ledger) => {
                            let _ = ui_tx.send(UiEvent::SimSummary {
                                ledger,
                                active_harness_label: self.active_harness_label(),
                                active_harness_hash: self.active_harness.sha256_hash(),
                            });
                        }
                        Err(err) => {
                            let _ = ui_tx.send(UiEvent::Error(err.to_string()));
                        }
                    },
                    UserCommand::RequestSimProblems => match self.load_sim_ledger("").await {
                        Ok(ledger) => {
                            let _ = ui_tx.send(UiEvent::SimProblems(ledger));
                        }
                        Err(err) => {
                            let _ = ui_tx.send(UiEvent::Error(err.to_string()));
                        }
                    },
                    UserCommand::ShowSimProblem(raw_id) => match self.load_sim_ledger("").await {
                        Ok(Some(ledger)) => {
                            let canonical = ledger.resolve_alias(&raw_id);
                            let resolved_from = (!canonical.eq_ignore_ascii_case(raw_id.trim()))
                                .then(|| raw_id.trim().to_string());
                            if let Some(problem) = ledger.get_problem(&canonical).cloned() {
                                let _ = ui_tx.send(UiEvent::SimProblemDetail {
                                    problem,
                                    resolved_from,
                                });
                            } else {
                                let _ = ui_tx.send(UiEvent::Error(format!(
                                    "SIM problem '{raw_id}' not found. Run `/sim list` to see recorded problems."
                                )));
                            }
                        }
                        Ok(None) => {
                            let _ = ui_tx.send(UiEvent::SystemNotice(
                                "No Self-Improvement Memory (SIM) runs recorded for this partition yet.".to_string(),
                            ));
                        }
                        Err(err) => {
                            let _ = ui_tx.send(UiEvent::Error(err.to_string()));
                        }
                    },
                    UserCommand::ExportSim(path) => match self.export_sim_document(None).await {
                        Ok(Some(doc)) => match serde_json::to_string_pretty(&doc) {
                            Ok(json) => {
                                if let Some(parent) = path.parent()
                                    && !parent.as_os_str().is_empty()
                                {
                                    let _ = std::fs::create_dir_all(parent);
                                }
                                match std::fs::write(&path, json) {
                                    Ok(()) => {
                                        let _ = ui_tx.send(UiEvent::SystemNotice(format!(
                                            "Exported SIM ledger ({} problem(s)) to `{}`.",
                                            doc.problems.len(),
                                            path.display()
                                        )));
                                    }
                                    Err(err) => {
                                        let _ = ui_tx.send(UiEvent::Error(format!(
                                            "Failed to write SIM export '{}': {err}",
                                            path.display()
                                        )));
                                    }
                                }
                            }
                            Err(err) => {
                                let _ = ui_tx.send(UiEvent::Error(err.to_string()));
                            }
                        },
                        Ok(None) => {
                            let _ = ui_tx.send(UiEvent::SystemNotice(
                                "No SIM run available to export.".to_string(),
                            ));
                        }
                        Err(err) => {
                            let _ = ui_tx.send(UiEvent::Error(err.to_string()));
                        }
                    },
                    UserCommand::ShowActiveHarness => {
                        let _ = ui_tx.send(UiEvent::ActiveHarnessDetail {
                            label: self.active_harness_label(),
                            version_id: self
                                .active_harness_record
                                .as_ref()
                                .map(|r| r.id.clone()),
                            spec: self.active_harness.clone(),
                        });
                    }
                    UserCommand::UseHarness(id_or_prefix) => {
                        match self.activate_harness_version(&id_or_prefix).await {
                            Ok(rec) => {
                                let _ = ui_tx.send(UiEvent::SystemNotice(format!(
                                    "Activated harness `{}` (id: `{}`, round {}).",
                                    rec.label,
                                    &rec.id[..rec.id.len().min(8)],
                                    rec.round
                                )));
                                if let Ok(snap) = self.inspector_snapshot().await {
                                    let _ = ui_tx.send(UiEvent::InspectorUpdated(snap));
                                }
                            }
                            Err(err) => {
                                let _ = ui_tx.send(UiEvent::Error(err.to_string()));
                            }
                        }
                    }
                    UserCommand::RequestDbStatus => match self.db_status().await {
                        Ok(report) => {
                            let _ = ui_tx.send(UiEvent::DbStatus(report));
                        }
                        Err(err) => {
                            let _ = ui_tx.send(UiEvent::Error(err.to_string()));
                        }
                    },
                    UserCommand::RefreshEnv { run_build_check } => {
                        match self.refresh_env(run_build_check).await {
                            Ok(_) => {
                                if let Ok(snap) = self.inspector_snapshot().await {
                                    let _ = ui_tx.send(UiEvent::InspectorUpdated(snap));
                                }
                            }
                            Err(err) => {
                                let _ = ui_tx.send(UiEvent::Error(err.to_string()));
                            }
                        }
                    }
                    UserCommand::CompactNow => {
                        let compacted = self
                            .history
                            .compact_if_needed(
                                &self.router,
                                self.last_preamble_tokens,
                                self.router.primary().context_window_tokens,
                                true,
                            )
                            .await;
                        let _ = ui_tx.send(UiEvent::SystemNotice(if compacted {
                            "Compacted older conversation turns into rolling summary.".to_string()
                        } else {
                            "History already compact (nothing to evict).".to_string()
                        }));
                        if let Ok(snap) = self.inspector_snapshot().await {
                            let _ = ui_tx.send(UiEvent::InspectorUpdated(snap));
                        }
                    }
                    UserCommand::Shutdown => {
                        self.sleep_worker.enqueue(SleepJob::Shutdown);
                        break;
                    }
                }
            }
        });

        (cmd_tx, ui_rx)
    }

    /// Build a quick default plan proposal from a user description (used by `/plan <title>`).
    pub fn build_quick_plan(description: &str) -> ProposedPlan {
        let title = if description.trim().is_empty() {
            "Incremental Feature Implementation".to_string()
        } else {
            description.trim().to_string()
        };
        ProposedPlan {
            title: title.clone(),
            summary: format!("Single-feature gated execution plan for: {title}"),
            affected_files: Vec::new(),
            tasks: vec![ProposedTaskSpec {
                title,
                acceptance_criteria: vec![
                    "Changes respect all active LanceDB architectural invariants".to_string(),
                    "cargo check and cargo test pass with 0 errors".to_string(),
                ],
            }],
        }
    }
}
