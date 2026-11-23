use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use allyh_memory::{MemoryStore, TaskRecord, TaskStatus};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{CoreError, Result};

/// Proposed single-feature task specification awaiting user approval at the Control Gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedTaskSpec {
    pub title: String,
    pub acceptance_criteria: Vec<String>,
}

/// Interactive execution plan presented to the user before granting autonomous write permissions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedPlan {
    pub title: String,
    pub summary: String,
    pub affected_files: Vec<String>,
    pub tasks: Vec<ProposedTaskSpec>,
}

/// Explicit Control Gate state machine (*SWE-Interact* mitigation).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum ControlGateState {
    /// Read-only exploration & requirement discovery (`writes_enabled = false`).
    Discovery,
    /// Plan proposed and awaiting explicit `[Approve Plan]` or `[Refine]` (`writes_enabled = false`).
    PlanApproval { plan: ProposedPlan },
    /// User approved the plan; autonomous write tools are unlocked for the active task (`writes_enabled = true`).
    Execution { active_task_id: Option<String> },
}

impl ControlGateState {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Discovery => "Discovery (Read-Only)",
            Self::PlanApproval { .. } => "Plan Approval Gate (Locked)",
            Self::Execution { .. } => "Execution (Approved)",
        }
    }

    pub fn allows_writes(&self) -> bool {
        matches!(self, Self::Execution { .. })
    }
}

/// Manages transitions between `Discovery -> PlanApproval -> Execution` while synchronizing
/// the shared atomic write-permission lock read by `allyh-tools::EditByHashTool`.
#[derive(Clone)]
pub struct ControlGate {
    state: ControlGateState,
    writes_enabled: Arc<AtomicBool>,
}

impl ControlGate {
    pub fn new(writes_enabled: Arc<AtomicBool>) -> Self {
        writes_enabled.store(false, Ordering::SeqCst);
        Self {
            state: ControlGateState::Discovery,
            writes_enabled,
        }
    }

    pub fn state(&self) -> &ControlGateState {
        &self.state
    }

    pub fn writes_enabled_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.writes_enabled)
    }

    /// Move to `PlanApproval` while keeping write tools strictly locked.
    pub fn propose_plan(&mut self, plan: ProposedPlan) -> &ControlGateState {
        self.writes_enabled.store(false, Ordering::SeqCst);
        self.state = ControlGateState::PlanApproval { plan };
        &self.state
    }

    /// Approve the pending plan: persists structured `TaskRecord`s in LanceDB (marking the first
    /// `in_progress` and remaining `approved`) and unlocks autonomous write tools.
    pub async fn approve_plan(
        &mut self,
        store: &MemoryStore,
        repo_id: &str,
        scope_id: &str,
    ) -> Result<Vec<TaskRecord>> {
        let plan = match &self.state {
            ControlGateState::PlanApproval { plan } => plan.clone(),
            _ => {
                return Err(CoreError::ControlGate(
                    "no pending plan awaiting approval".to_string(),
                ));
            }
        };

        let now = Utc::now();
        let mut created = Vec::new();
        for (idx, spec) in plan.tasks.iter().enumerate() {
            let task = TaskRecord {
                id: Uuid::new_v4().to_string(),
                repo_id: repo_id.to_string(),
                scope_id: scope_id.to_string(),
                parent_id: None,
                title: spec.title.clone(),
                acceptance_criteria: spec.acceptance_criteria.clone(),
                status: if idx == 0 {
                    TaskStatus::InProgress
                } else {
                    TaskStatus::Approved
                },
                last_error_trace: None,
                updated_at: now,
            };
            store.upsert_task(&task).await?;
            created.push(task);
        }

        let active_task_id = created.first().map(|t| t.id.clone());
        self.state = ControlGateState::Execution { active_task_id };
        self.writes_enabled.store(true, Ordering::SeqCst);
        Ok(created)
    }

    /// Reject/refine the pending plan, returning to read-only `Discovery` mode.
    pub fn refine_plan(&mut self) -> &ControlGateState {
        self.writes_enabled.store(false, Ordering::SeqCst);
        self.state = ControlGateState::Discovery;
        &self.state
    }

    /// Lock writes back down to `Discovery` after completing an incremental feature turn.
    pub fn lock_to_discovery(&mut self) {
        self.writes_enabled.store(false, Ordering::SeqCst);
        self.state = ControlGateState::Discovery;
    }
}
