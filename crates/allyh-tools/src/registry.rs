use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use allyh_env::EnvironmentMonitor;
use allyh_memory::MemoryStore;
use rig_core::completion::ToolDefinition;
use rig_core::message::ToolName;
use rig_core::tool::{IntoToolOutput, Tool, ToolContext, ToolOutput};
use serde::{Deserialize, Serialize};

use crate::edit::{EditByHashArgs, EditByHashTool};
use crate::error::ToolError;
use crate::inspect::{InspectSymbolArgs, InspectSymbolTool};
use crate::skill::{AddSkillArgs, AddSkillTool};
use crate::verify::{RunVerificationArgs, RunVerificationTool};

/// Compact execution card emitted after every ACI tool invocation for the TUI and audit log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionRecord {
    pub tool_name: String,
    pub compact_header: String,
    pub succeeded: bool,
    pub output_text: String,
    pub output_json: serde_json::Value,
}

/// Bundled compact ACI toolset (`inspect_symbol`, `edit_by_hash`, `run_verification`)
/// sharing the same repository monitor, LanceDB store, and write-approval gate.
#[derive(Clone)]
pub struct AciToolset {
    inspect: InspectSymbolTool,
    edit: EditByHashTool,
    verify: RunVerificationTool,
    skill: AddSkillTool,
    writes_enabled: Arc<AtomicBool>,
}

impl AciToolset {
    pub fn new(
        env: Arc<EnvironmentMonitor>,
        store: MemoryStore,
        writes_enabled: Arc<AtomicBool>,
        auto_cargo_check: bool,
    ) -> Self {
        Self {
            inspect: InspectSymbolTool::new(Arc::clone(&env), store.clone()),
            edit: EditByHashTool::new(Arc::clone(&env), Arc::clone(&writes_enabled), auto_cargo_check),
            verify: RunVerificationTool::new(Arc::clone(&env), store.clone()),
            skill: AddSkillTool::new(env, store, Arc::clone(&writes_enabled)),
            writes_enabled,
        }
    }

    pub fn set_writes_enabled(&self, enabled: bool) {
        self.writes_enabled.store(enabled, Ordering::SeqCst);
    }

    pub fn writes_enabled(&self) -> bool {
        self.writes_enabled.load(Ordering::SeqCst)
    }

    pub fn inspect_tool(&self) -> &InspectSymbolTool {
        &self.inspect
    }

    pub fn edit_tool(&self) -> &EditByHashTool {
        &self.edit
    }

    pub fn verify_tool(&self) -> &RunVerificationTool {
        &self.verify
    }

    pub fn skill_tool(&self) -> &AddSkillTool {
        &self.skill
    }

    /// Return `rig-core` [`ToolDefinition`]s exposed to the active completion model.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        let mut defs = vec![
            ToolDefinition::new(
                ToolName::new(InspectSymbolTool::NAME).expect("valid tool name"),
                self.inspect.description(),
                self.inspect.parameters(),
            )
            .with_annotations(self.inspect.annotations()),
            ToolDefinition::new(
                ToolName::new(RunVerificationTool::NAME).expect("valid tool name"),
                self.verify.description(),
                self.verify.parameters(),
            )
            .with_annotations(self.verify.annotations()),
        ];

        // Expose `edit_by_hash` definition always so the model knows it exists, while the
        // tool itself strictly enforces the write-approval gate at execution time.
        defs.push(
            ToolDefinition::new(
                ToolName::new(EditByHashTool::NAME).expect("valid tool name"),
                self.edit.description(),
                self.edit.parameters(),
            )
            .with_annotations(self.edit.annotations()),
        );

        // Expose `add_skill` definition always; the tool enforces the write-approval gate
        // at execution time before persisting a reusable procedural skill.
        defs.push(
            ToolDefinition::new(
                ToolName::new(AddSkillTool::NAME).expect("valid tool name"),
                self.skill.description(),
                self.skill.parameters(),
            )
            .with_annotations(self.skill.annotations()),
        );
        defs
    }

    /// Dispatch a model-requested tool call by name with JSON arguments, returning a
    /// compact [`ToolExecutionRecord`] plus canonical [`ToolOutput`].
    pub async fn dispatch(
        &self,
        tool_name: &str,
        args: serde_json::Value,
    ) -> (ToolExecutionRecord, ToolOutput) {
        let mut ctx = ToolContext::default();
        match tool_name {
            InspectSymbolTool::NAME => match serde_json::from_value::<InspectSymbolArgs>(args) {
                Ok(parsed) => {
                    let q = parsed.query.clone();
                    match self.inspect.call(&mut ctx, parsed).await {
                        Ok(out) => {
                            let header = format!(
                                "inspect_symbol({q}) -> {} snippet(s), {} 1-hop edge(s)",
                                out.matched_files.len(),
                                out.ego_graph.hop1_neighbors.len()
                            );
                            let json_val = serde_json::to_value(&out).unwrap_or_default();
                            let rendered = format!(
                                "{header}\n{}\n{}",
                                out.ego_graph_summary,
                                out.matched_files
                                    .iter()
                                    .map(|f| format!(
                                        "--- {} (sha256: {}, lines {}-{}) ---\n{}",
                                        f.file_path, f.sha256_hash, f.start_line, f.end_line, f.snippet
                                    ))
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            );
                            let tool_output = out
                                .into_tool_output()
                                .unwrap_or_else(|_| ToolOutput::text(rendered.clone()));
                            (
                                ToolExecutionRecord {
                                    tool_name: tool_name.to_string(),
                                    compact_header: header,
                                    succeeded: true,
                                    output_text: rendered,
                                    output_json: json_val,
                                },
                                tool_output,
                            )
                        }
                        Err(err) => failed_record(tool_name, &format!("inspect_symbol({q})"), err),
                    }
                }
                Err(err) => failed_record(
                    tool_name,
                    "inspect_symbol(invalid args)",
                    ToolError::InvalidArgs(err.to_string()),
                ),
            },
            EditByHashTool::NAME => match serde_json::from_value::<EditByHashArgs>(args) {
                Ok(parsed) => {
                    let path = parsed.file_path.clone();
                    match self.edit.call(&mut ctx, parsed).await {
                        Ok(out) => {
                            let check_label = out
                                .cargo_check
                                .as_ref()
                                .map(|c| c.compact_status_line())
                                .unwrap_or_else(|| "check skipped".to_string());
                            let header = format!("edit_by_hash({path}) -> {check_label}");
                            let json_val = serde_json::to_value(&out).unwrap_or_default();
                            let rendered = format!(
                                "{header}\nDiff: {}\nNew SHA-256: {}\n{}",
                                out.diff_summary, out.new_sha256, out.prescriptive_diagnostics
                            );
                            let tool_output = out
                                .into_tool_output()
                                .unwrap_or_else(|_| ToolOutput::text(rendered.clone()));
                            (
                                ToolExecutionRecord {
                                    tool_name: tool_name.to_string(),
                                    compact_header: header,
                                    succeeded: true,
                                    output_text: rendered,
                                    output_json: json_val,
                                },
                                tool_output,
                            )
                        }
                        Err(err) => failed_record(tool_name, &format!("edit_by_hash({path})"), err),
                    }
                }
                Err(err) => failed_record(
                    tool_name,
                    "edit_by_hash(invalid args)",
                    ToolError::InvalidArgs(err.to_string()),
                ),
            },
            RunVerificationTool::NAME => match serde_json::from_value::<RunVerificationArgs>(args) {
                Ok(parsed) => match self.verify.call(&mut ctx, parsed).await {
                    Ok(out) => {
                        let header = format!(
                            "run_verification -> {}",
                            out.report.compact_status_line()
                        );
                        let succeeded = out.report.success;
                        let json_val = serde_json::to_value(&out).unwrap_or_default();
                        let rendered = out.prescriptive_diagnostics.clone();
                        let tool_output = out
                            .into_tool_output()
                            .unwrap_or_else(|_| ToolOutput::text(rendered.clone()));
                        (
                            ToolExecutionRecord {
                                tool_name: tool_name.to_string(),
                                compact_header: header,
                                succeeded,
                                output_text: rendered,
                                output_json: json_val,
                            },
                            tool_output,
                        )
                    }
                    Err(err) => failed_record(tool_name, "run_verification", err),
                },
                Err(err) => failed_record(
                    tool_name,
                    "run_verification(invalid args)",
                    ToolError::InvalidArgs(err.to_string()),
                ),
            },
            AddSkillTool::NAME => match serde_json::from_value::<AddSkillArgs>(args) {
                Ok(parsed) => match self.skill.execute(parsed).await {
                    Ok(out) => {
                        let verb = if out.created { "created" } else { "updated" };
                        let header = format!(
                            "add_skill({}) -> {verb}, {} trigger(s)",
                            out.name,
                            out.triggers.len()
                        );
                        let json_val = serde_json::to_value(&out).unwrap_or_default();
                        let rendered = format!(
                            "{header}\nid: {}\nrepo: {} / scope: {}\nupdated_at: {}",
                            out.id, out.repo_id, out.scope_id, out.updated_at
                        );
                        (
                            ToolExecutionRecord {
                                tool_name: tool_name.to_string(),
                                compact_header: header,
                                succeeded: true,
                                output_text: rendered.clone(),
                                output_json: json_val,
                            },
                            ToolOutput::text(rendered),
                        )
                    }
                    Err(err) => failed_record(tool_name, "add_skill", err),
                },
                Err(err) => failed_record(
                    tool_name,
                    "add_skill(invalid args)",
                    ToolError::InvalidArgs(err.to_string()),
                ),
            },
            other => failed_record(other, other, ToolError::UnknownTool(other.to_string())),
        }
    }
}

fn failed_record(
    tool_name: &str,
    prefix: &str,
    err: ToolError,
) -> (ToolExecutionRecord, ToolOutput) {
    let msg = err.to_string();
    let header = format!("{prefix} -> ERROR");
    let output = ToolOutput::text(msg.clone());
    let rec = ToolExecutionRecord {
        tool_name: tool_name.to_string(),
        compact_header: header,
        succeeded: false,
        output_text: msg.clone(),
        output_json: serde_json::json!({ "error": msg }),
    };
    (rec, output)
}
