use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use allyh_env::EnvironmentMonitor;
use allyh_memory::{MemoryStore, SkillRecord};
use chrono::Utc;
use rig_core::completion::ToolAnnotations;
use rig_core::tool::{Tool, ToolContext, ToolExecutionError, args_schema};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::ToolError;

/// Arguments for [`AddSkillTool`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct AddSkillArgs {
    /// Short, unique, human-readable skill name (e.g. `add-lancedb-table`).
    pub name: String,
    /// One-line summary of what the skill accomplishes.
    pub description: String,
    /// Full reusable procedure / instructions the agent should follow when the skill applies.
    pub instructions: String,
    /// Optional trigger hints (keywords or phrases) that indicate when to recall this skill.
    pub triggers: Option<Vec<String>>,
}

/// Compact confirmation returned by [`AddSkillTool`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddSkillOutput {
    pub id: String,
    pub name: String,
    pub repo_id: String,
    pub scope_id: String,
    pub triggers: Vec<String>,
    pub created: bool,
    pub updated_at: String,
}

/// ACI tool that registers a reusable procedural skill into the LanceDB `skills` table.
#[derive(Clone)]
pub struct AddSkillTool {
    env: Arc<EnvironmentMonitor>,
    store: MemoryStore,
    /// Enforces the Control Gate: skills cannot be persisted while AllyH is in Discovery / PlanApproval mode.
    writes_enabled: Arc<AtomicBool>,
}

impl AddSkillTool {
    pub fn new(
        env: Arc<EnvironmentMonitor>,
        store: MemoryStore,
        writes_enabled: Arc<AtomicBool>,
    ) -> Self {
        Self {
            env,
            store,
            writes_enabled,
        }
    }

    /// Derive a stable, collision-resistant skill id from repo/scope/name.
    fn skill_id(repo_id: &str, scope_id: &str, name: &str) -> String {
        let slug: String = name
            .trim()
            .to_ascii_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '-' })
            .collect();
        let slug = slug.trim_matches('-').to_string();
        format!("{repo_id}:{scope_id}:{slug}")
    }

    pub async fn execute(&self, args: AddSkillArgs) -> Result<AddSkillOutput, ToolError> {
        if !self.writes_enabled.load(Ordering::SeqCst) {
            return Err(ToolError::WriteGateLocked(
                "Plan approval gate is active; approve the plan before registering skills."
                    .to_string(),
            ));
        }

        let name = args.name.trim();
        if name.is_empty() {
            return Err(ToolError::InvalidArgs(
                "skill 'name' must not be empty".to_string(),
            ));
        }
        let instructions = args.instructions.trim();
        if instructions.is_empty() {
            return Err(ToolError::InvalidArgs(
                "skill 'instructions' must not be empty".to_string(),
            ));
        }

        let repo_id = self.env.repo_id().to_string();
        let scope_id = self.env.scope_id().to_string();
        let id = Self::skill_id(&repo_id, &scope_id, name);

        let triggers: Vec<String> = args
            .triggers
            .unwrap_or_default()
            .into_iter()
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect();

        // Preserve original creation timestamp when updating an existing skill.
        let existing = self.store.list_skills(&repo_id, &scope_id).await?;
        let created_at = existing
            .iter()
            .find(|s| s.id == id)
            .map(|s| s.created_at)
            .unwrap_or_else(Utc::now);
        let created = !existing.iter().any(|s| s.id == id);

        let now = Utc::now();
        let record = SkillRecord {
            id: id.clone(),
            repo_id: repo_id.clone(),
            scope_id: scope_id.clone(),
            name: name.to_string(),
            description: args.description.trim().to_string(),
            instructions: instructions.to_string(),
            triggers: triggers.clone(),
            created_at,
            updated_at: now,
        };
        self.store.upsert_skill(&record).await?;

        Ok(AddSkillOutput {
            id,
            name: record.name,
            repo_id,
            scope_id,
            triggers,
            created,
            updated_at: now.to_rfc3339(),
        })
    }
}

impl Tool for AddSkillTool {
    const NAME: &'static str = "add_skill";
    type Args = AddSkillArgs;
    type Output = AddSkillOutput;
    type Error = ToolError;

    fn description(&self) -> String {
        "Register a reusable procedural skill (name, description, instructions, optional triggers) into persistent LanceDB memory so it can be recalled across turns and sessions.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<AddSkillArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_title("Add Reusable Skill")
            .with_read_only(false)
            .with_destructive(false)
            .with_open_world(false)
    }

    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        ToolExecutionError::other(error.to_string())
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.execute(args).await
    }
}
