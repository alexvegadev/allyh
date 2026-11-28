use std::sync::Arc;

use allyh_env::EnvironmentMonitor;
use allyh_memory::{MemoryStore, SimMemoryMode};
use rig_core::completion::ToolAnnotations;
use rig_core::tool::{Tool, ToolContext, ToolExecutionError, args_schema};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::ToolError;

/// Arguments for [`InspectSimTool`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct InspectSimArgs {
    /// Optional SIM problem ID or merged alias (e.g. `P3` or `P6`). Omit to inspect the full ledger summary.
    pub problem_id: Option<String>,
    /// Optional SIM run ID prefix or arm name. Omit to inspect the latest SIM run for this partition.
    pub run_id: Option<String>,
}

/// Structured read-only output returned by [`InspectSimTool`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectSimOutput {
    pub run_id: Option<String>,
    pub arm_name: Option<String>,
    pub domain: Option<String>,
    pub benchmark: Option<String>,
    pub policy_model: Option<String>,
    pub memory_mode: Option<String>,
    pub carried_from: Option<String>,
    pub problem_count: usize,
    pub attempt_count: usize,
    pub rendered: String,
}

/// Read-only ACI tool that inspects Self-Improvement Memory (SIM) problem entries,
/// edit rationales, predicted/fixed/missed outcomes, and carryover provenance.
#[derive(Clone)]
pub struct InspectSimTool {
    env: Arc<EnvironmentMonitor>,
    store: MemoryStore,
}

impl InspectSimTool {
    pub fn new(env: Arc<EnvironmentMonitor>, store: MemoryStore) -> Self {
        Self { env, store }
    }

    pub async fn execute(&self, args: InspectSimArgs) -> Result<InspectSimOutput, ToolError> {
        let repo_id = self.env.repo_id();
        let scope_id = self.env.scope_id();
        let run_needle = args.run_id.as_deref().unwrap_or("");

        let Some(ledger) = self
            .store
            .load_sim_ledger(repo_id, scope_id, run_needle)
            .await?
        else {
            return Ok(InspectSimOutput {
                run_id: None,
                arm_name: None,
                domain: None,
                benchmark: None,
                policy_model: None,
                memory_mode: None,
                carried_from: None,
                problem_count: 0,
                attempt_count: 0,
                rendered: "No Self-Improvement Memory (SIM) runs recorded for this repository partition yet.".to_string(),
            });
        };

        let carried_from = match (
            &ledger.run.carried_from_run_id,
            &ledger.run.carried_from_domain,
        ) {
            (Some(arm), Some(dom)) => Some(format!("{arm} ({dom})")),
            (Some(arm), None) => Some(arm.clone()),
            _ => None,
        };

        let rendered = if let Some(raw_pid) = args.problem_id.as_deref().filter(|s| !s.trim().is_empty())
        {
            let resolved = ledger.resolve_alias(raw_pid);
            let Some(problem) = ledger.get_problem(&resolved) else {
                let ids: Vec<&str> = ledger.problems.iter().map(|p| p.id.as_str()).collect();
                let hint = if ids.is_empty() {
                    "No problems recorded in this SIM run.".to_string()
                } else {
                    format!("Available problem IDs: {}", ids.join(", "))
                };
                return Err(ToolError::InvalidArgs(format!(
                    "SIM problem '{raw_pid}' not found. {hint}"
                )));
            };
            let mut lines = vec![format!(
                "SIM Problem {} (opened round {}, origin: `{}`): {}",
                problem.id, problem.opened_round, problem.origin_domain, problem.name
            )];
            if !resolved.eq_ignore_ascii_case(raw_pid.trim()) {
                lines.push(format!("(Resolved alias `{}` -> `{}`)", raw_pid.trim(), problem.id));
            }
            for attempt in &problem.attempts {
                lines.push(format!(
                    "- Round {} [{}] (domain=`{}`, benchmark=`{}`, policy=`{}`, passed={}/{}, score={:.2})",
                    attempt.round,
                    attempt.sample,
                    attempt.domain,
                    attempt.benchmark,
                    attempt.policy_model,
                    attempt.passed,
                    attempt.total,
                    attempt.score
                ));
                lines.push(format!("  What: {}", attempt.what));
                if !attempt.files.is_empty() {
                    lines.push(format!("  Files: {}", attempt.files.join(", ")));
                }
                lines.push(format!(
                    "  Predicted: [{}] | Fixed: [{}] | Missed: [{}]",
                    attempt.predicted.join(", "),
                    attempt.fixed.join(", "),
                    attempt.missed.join(", ")
                ));
            }
            lines.join("\n")
        } else {
            ledger.render_for_prompt(SimMemoryMode::FullSim)
        };

        Ok(InspectSimOutput {
            run_id: Some(ledger.run.run_id.clone()),
            arm_name: Some(ledger.run.arm_name.clone()),
            domain: Some(ledger.run.domain.clone()),
            benchmark: Some(ledger.run.benchmark.clone()),
            policy_model: Some(ledger.run.policy_model.clone()),
            memory_mode: Some(ledger.run.memory_mode.to_string()),
            carried_from,
            problem_count: ledger.problems.len(),
            attempt_count: ledger.total_attempts(),
            rendered,
        })
    }
}

impl Tool for InspectSimTool {
    const NAME: &'static str = "inspect_sim";
    type Args = InspectSimArgs;
    type Output = InspectSimOutput;
    type Error = ToolError;

    fn description(&self) -> String {
        "Inspect the Self-Improvement Memory (SIM) ledger for this repository partition (read-only): recurring harness problem diagnoses, past modification rationales, predicted/fixed/missed outcomes, and cross-model/cross-dataset provenance.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<InspectSimArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_title("Inspect Self-Improvement Memory")
            .with_read_only(true)
            .with_destructive(false)
            .with_open_world(false)
    }

    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        match error {
            ToolError::InvalidArgs(msg) => ToolExecutionError::invalid_args(msg),
            other => ToolExecutionError::other(other.to_string()),
        }
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.execute(args).await
    }
}
