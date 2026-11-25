use std::sync::Arc;

use allyh_env::EnvironmentMonitor;
use allyh_memory::{MemoryError, MemoryStore};
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

/// ACI tool that registers a reusable procedural skill into the out-of-tree LanceDB `skills` table.
///
/// Allowed in Discovery mode as well as Execution mode because skills live in out-of-tree
/// LanceDB memory and never modify files in the target repository.
#[derive(Clone)]
pub struct AddSkillTool {
    env: Arc<EnvironmentMonitor>,
    store: MemoryStore,
}

impl AddSkillTool {
    pub fn new(env: Arc<EnvironmentMonitor>, store: MemoryStore) -> Self {
        Self { env, store }
    }

    pub async fn execute(&self, args: AddSkillArgs) -> Result<AddSkillOutput, ToolError> {
        let repo_id = self.env.repo_id().to_string();
        let scope_id = self.env.scope_id().to_string();

        let (record, created) = self
            .store
            .save_skill(
                &repo_id,
                &scope_id,
                &args.name,
                &args.description,
                &args.instructions,
                args.triggers.unwrap_or_default(),
            )
            .await
            .map_err(|err| match err {
                MemoryError::InvalidSkill(msg) => ToolError::InvalidArgs(msg),
                other => ToolError::Memory(other),
            })?;

        Ok(AddSkillOutput {
            id: record.id,
            name: record.name,
            repo_id,
            scope_id,
            triggers: record.triggers,
            created,
            updated_at: record.updated_at.to_rfc3339(),
        })
    }
}

impl Tool for AddSkillTool {
    const NAME: &'static str = "add_skill";
    type Args = AddSkillArgs;
    type Output = AddSkillOutput;
    type Error = ToolError;

    fn description(&self) -> String {
        "Register or update a reusable procedural skill (name, description, instructions, optional triggers) in persistent LanceDB memory so it can be recalled across turns and sessions. Available in Discovery and Execution modes.".to_string()
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

/// Arguments for [`GetSkillTool`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct GetSkillArgs {
    /// Skill name, normalized slug, or full skill id to load from LanceDB.
    pub name: String,
}

/// Full procedural skill payload returned by [`GetSkillTool`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetSkillOutput {
    pub id: String,
    pub name: String,
    pub description: String,
    pub instructions: String,
    pub triggers: Vec<String>,
    pub updated_at: String,
}

/// Read-only ACI tool that loads the full instructions of a registered skill on demand.
#[derive(Clone)]
pub struct GetSkillTool {
    env: Arc<EnvironmentMonitor>,
    store: MemoryStore,
}

impl GetSkillTool {
    pub fn new(env: Arc<EnvironmentMonitor>, store: MemoryStore) -> Self {
        Self { env, store }
    }

    pub async fn execute(&self, args: GetSkillArgs) -> Result<GetSkillOutput, ToolError> {
        let name = args.name.trim();
        if name.is_empty() {
            return Err(ToolError::InvalidArgs(
                "skill 'name' must not be empty".to_string(),
            ));
        }

        let repo_id = self.env.repo_id();
        let scope_id = self.env.scope_id();
        let Some(skill) = self.store.get_skill(repo_id, scope_id, name).await? else {
            let available: Vec<String> = self
                .store
                .list_skills(repo_id, scope_id)
                .await?
                .into_iter()
                .map(|s| s.name)
                .collect();
            let hint = if available.is_empty() {
                "No skills are registered for this repository partition.".to_string()
            } else {
                format!("Available skills: {}", available.join(", "))
            };
            return Err(ToolError::InvalidArgs(format!(
                "skill '{name}' not found. {hint}"
            )));
        };

        Ok(GetSkillOutput {
            id: skill.id,
            name: skill.name,
            description: skill.description,
            instructions: skill.instructions,
            triggers: skill.triggers,
            updated_at: skill.updated_at.to_rfc3339(),
        })
    }
}

impl Tool for GetSkillTool {
    const NAME: &'static str = "get_skill";
    type Args = GetSkillArgs;
    type Output = GetSkillOutput;
    type Error = ToolError;

    fn description(&self) -> String {
        "Load the full reusable instructions and triggers for a named skill from persistent LanceDB memory (read-only). Use when a skill in the Available Skills catalog applies to the current task.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<GetSkillArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_title("Load Reusable Skill")
            .with_read_only(true)
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
