use std::sync::Arc;

use allyh_env::{EgoGraph, EnvironmentMonitor, SourceLanguage};
use allyh_memory::MemoryStore;
use rig_core::completion::ToolAnnotations;
use rig_core::tool::{Tool, ToolContext, ToolExecutionError, args_schema};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::ToolError;

/// Arguments for [`InspectSymbolTool`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct InspectSymbolArgs {
    /// Symbol name (e.g. `MemoryStore::consolidate`) or relative file path (e.g. `src/lib.rs`).
    pub query: String,
    /// RepoGraph traversal depth (1 or 2; defaults to 1).
    pub hops: Option<usize>,
    /// Optional line start/end window when reading a specific file directly.
    pub start_line: Option<usize>,
    pub end_line: Option<usize>,
}

/// Compact structured response returned by [`InspectSymbolTool`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectSymbolOutput {
    pub query: String,
    pub matched_files: Vec<InspectedFileSnippet>,
    pub ego_graph_summary: String,
    pub ego_graph: EgoGraph,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectedFileSnippet {
    pub file_path: String,
    #[serde(default)]
    pub language: String,
    pub sha256_hash: String,
    pub start_line: usize,
    pub end_line: usize,
    pub total_lines: usize,
    pub snippet: String,
}

/// Compact ACI tool that inspects a source symbol or file slice across supported languages
/// together with its SHA-256 hash and 1-hop / 2-hop RepoGraph neighborhood.
#[derive(Clone)]
pub struct InspectSymbolTool {
    env: Arc<EnvironmentMonitor>,
    store: MemoryStore,
}

impl InspectSymbolTool {
    pub fn new(env: Arc<EnvironmentMonitor>, store: MemoryStore) -> Self {
        Self { env, store }
    }

    pub async fn execute(&self, args: InspectSymbolArgs) -> Result<InspectSymbolOutput, ToolError> {
        let hops = args.hops.unwrap_or(1).clamp(1, 2);
        let mut ego = self.env.ego_graph_for(&args.query, hops).await?;

        // If the query looks like a relative file path that wasn't indexed yet, index it on demand.
        if ego.center_definitions.is_empty()
            && let Ok((rel, abs)) = self.env.resolve_safe_path(&args.query)
            && abs.is_file()
        {
            let _ = self.env.index_file(&rel, 0).await;
            ego = self.env.ego_graph_for(&rel, hops).await?;
        }

        let mut matched_files = Vec::new();
        let mut seen_ranges = std::collections::HashSet::new();

        // Direct file read if query resolves to an existing file.
        if let Ok((rel, abs)) = self.env.resolve_safe_path(&args.query)
            && abs.is_file()
        {
            let content = std::fs::read_to_string(&abs)?;
            let hash = EnvironmentMonitor::sha256_str(&content);
            self.env.cache_file_content(&rel, &content);
            let lines: Vec<&str> = content.lines().collect();
            let total_lines = lines.len();
            let start = args.start_line.unwrap_or(1).max(1);
            let end = args
                .end_line
                .unwrap_or((start + 120).min(total_lines.max(1)))
                .min(total_lines.max(1));
            let snippet = slice_lines(&lines, start, end);
            seen_ranges.insert((rel.clone(), start, end));
            matched_files.push(InspectedFileSnippet {
                language: SourceLanguage::from_path(&rel).as_str().to_string(),
                file_path: rel,
                sha256_hash: hash,
                start_line: start,
                end_line: end,
                total_lines,
                snippet,
            });
        }

        // Symbol-scoped slices from RepoGraph definitions (avoids dumping whole files).
        for def in ego.center_definitions.iter().take(4) {
            let Ok((rel, abs)) = self.env.resolve_safe_path(&def.file_path) else {
                continue;
            };
            if !abs.is_file() {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(&abs) else {
                continue;
            };
            let hash = EnvironmentMonitor::sha256_str(&content);
            self.env.cache_file_content(&rel, &content);

            let lines: Vec<&str> = content.lines().collect();
            let total_lines = lines.len();
            let start = def.symbol.start_line.saturating_sub(2).max(1);
            let end = (def.symbol.end_line + 2).min(total_lines.max(1)).min(start + 120);
            if !seen_ranges.insert((rel.clone(), start, end)) {
                continue;
            }
            matched_files.push(InspectedFileSnippet {
                language: SourceLanguage::from_path(&rel).as_str().to_string(),
                file_path: rel,
                sha256_hash: hash,
                start_line: start,
                end_line: end,
                total_lines,
                snippet: slice_lines(&lines, start, end),
            });
        }

        // Fallback: if no file or symbol matched, list indexed files matching substring.
        if matched_files.is_empty() {
            let snapshots = self
                .store
                .list_code_snapshots(self.env.repo_id(), self.env.scope_id())
                .await?;
            let q = args.query.to_ascii_lowercase();
            for snap in snapshots.into_iter().filter(|s| {
                s.file_path.to_ascii_lowercase().contains(&q)
                    || s.symbols_json.to_ascii_lowercase().contains(&q)
            }).take(3) {
                if let Ok((rel, abs)) = self.env.resolve_safe_path(&snap.file_path)
                    && let Ok(content) = std::fs::read_to_string(&abs)
                {
                    let hash = EnvironmentMonitor::sha256_str(&content);
                    self.env.cache_file_content(&rel, &content);
                    let lines: Vec<&str> = content.lines().collect();
                    let total_lines = lines.len();
                    let end = total_lines.min(80).max(1);
                    matched_files.push(InspectedFileSnippet {
                        language: SourceLanguage::from_path(&rel).as_str().to_string(),
                        file_path: rel,
                        sha256_hash: hash,
                        start_line: 1,
                        end_line: end,
                        total_lines,
                        snippet: slice_lines(&lines, 1, end),
                    });
                }
            }
        }

        let ego_graph_summary = ego.render_compact();
        Ok(InspectSymbolOutput {
            query: args.query,
            matched_files,
            ego_graph_summary,
            ego_graph: ego,
        })
    }
}

impl Tool for InspectSymbolTool {
    const NAME: &'static str = "inspect_symbol";
    type Args = InspectSymbolArgs;
    type Output = InspectSymbolOutput;
    type Error = ToolError;

    fn description(&self) -> String {
        "Inspect a source symbol or relative file slice across supported languages, returning its SHA-256 hash (required for edit_by_hash) and compact 1-hop/2-hop RepoGraph neighborhood.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<InspectSymbolArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_title("Inspect Symbol & RepoGraph")
            .with_read_only(true)
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

fn slice_lines(lines: &[&str], start_line: usize, end_line: usize) -> String {
    if lines.is_empty() {
        return String::new();
    }
    let s = start_line.saturating_sub(1).min(lines.len().saturating_sub(1));
    let e = end_line.min(lines.len());
    lines[s..e]
        .iter()
        .enumerate()
        .map(|(idx, line)| format!("{:>4} | {line}", s + idx + 1))
        .collect::<Vec<_>>()
        .join("\n")
}
