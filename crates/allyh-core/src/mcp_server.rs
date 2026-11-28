use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use allyh_env::{EnvironmentMonitor, GitTracker};
use allyh_memory::{
    CandidateDecision, DeterministicHashEmbedder, EmbeddingProvider, HeuristicConflictJudge,
    MemoryConsolidator, MemoryStore, StorageConfig, compute_repo_id,
};
use allyh_tools::{
    AddSkillArgs, AddSkillTool, GetSkillArgs, GetSkillTool, InspectSimArgs, InspectSimTool,
    InspectSymbolArgs, InspectSymbolTool, MCP_PROTOCOL_VERSION,
};
use chrono::Utc;
use serde_json::{Value, json};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;

use crate::error::Result;

/// Configuration for launching [`AllyHMcpServer`].
#[derive(Debug, Clone)]
pub struct McpServerLaunchConfig {
    pub repo_root: PathBuf,
    pub storage: StorageConfig,
    pub scope_id: String,
}

/// Exposes AllyH's out-of-tree bi-temporal architectural memory, polyglot RepoGraph,
/// procedural skills, structured task queue, and SIM ledger over the Model Context Protocol (MCP)
/// JSON-RPC 2.0 stdio transport (`allyh mcp serve`).
pub struct AllyHMcpServer {
    repo_id: String,
    scope_id: String,
    store: MemoryStore,
    consolidator: MemoryConsolidator,
    env: Arc<EnvironmentMonitor>,
    inspect_symbol: InspectSymbolTool,
    get_skill: GetSkillTool,
    add_skill: AddSkillTool,
    inspect_sim: InspectSimTool,
    pinned_decisions: Mutex<HashSet<String>>,
}

impl AllyHMcpServer {
    /// Initialize a headless MCP server partition against LanceDB and the target repository.
    pub async fn initialize(
        config: McpServerLaunchConfig,
        embedder: Option<Arc<dyn EmbeddingProvider>>,
    ) -> Result<Self> {
        let embedder = embedder.unwrap_or_else(|| Arc::new(DeterministicHashEmbedder));
        let store = MemoryStore::connect(config.storage, embedder).await?;
        let git = GitTracker::new(&config.repo_root);
        let repo_id = compute_repo_id(&git.canonical_repo_identifier());
        let scope_id = config.scope_id;

        let env = Arc::new(EnvironmentMonitor::new(
            &config.repo_root,
            repo_id.clone(),
            scope_id.clone(),
            store.clone(),
        ));
        // Perform initial polyglot RepoGraph scan so symbol & ego-graph queries are immediately warm.
        let _ = env.sync_and_collect_delta(0, false).await;

        let consolidator = MemoryConsolidator::new(store.clone(), Arc::new(HeuristicConflictJudge));
        let inspect_symbol = InspectSymbolTool::new(Arc::clone(&env), store.clone());
        let get_skill = GetSkillTool::new(Arc::clone(&env), store.clone());
        let add_skill = AddSkillTool::new(Arc::clone(&env), store.clone());
        let inspect_sim = InspectSimTool::new(Arc::clone(&env), store.clone());

        Ok(Self {
            repo_id,
            scope_id,
            store,
            consolidator,
            env,
            inspect_symbol,
            get_skill,
            add_skill,
            inspect_sim,
            pinned_decisions: Mutex::new(HashSet::new()),
        })
    }

    pub fn repo_id(&self) -> &str {
        &self.repo_id
    }

    pub fn scope_id(&self) -> &str {
        &self.scope_id
    }

    /// Run the newline-delimited JSON-RPC 2.0 MCP server loop on `stdin`/`stdout`.
    pub async fn serve_stdio(&self) -> Result<()> {
        let stdin = BufReader::new(tokio::io::stdin());
        let stdout = tokio::io::stdout();
        self.serve(stdin, stdout).await
    }

    /// Serve MCP JSON-RPC 2.0 over any async reader/writer pair until EOF.
    pub async fn serve<R, W>(&self, mut reader: R, mut writer: W) -> Result<()>
    where
        R: AsyncBufRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let mut line = String::new();
        loop {
            line.clear();
            let n = reader.read_line(&mut line).await?;
            if n == 0 {
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let req: Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(err) => {
                    let err_resp = json!({
                        "jsonrpc": "2.0",
                        "id": Value::Null,
                        "error": { "code": -32700, "message": format!("Parse error: {err}") }
                    });
                    write_json_line(&mut writer, &err_resp).await?;
                    continue;
                }
            };

            if let Some(resp) = self.handle_request(&req).await {
                write_json_line(&mut writer, &resp).await?;
            }
        }
        self.env.shutdown_lsp().await;
        Ok(())
    }

    /// Process a single JSON-RPC 2.0 MCP message, returning `None` for notifications.
    pub async fn handle_request(&self, req: &Value) -> Option<Value> {
        let id = req.get("id").cloned();
        let method = req.get("method").and_then(|v| v.as_str()).unwrap_or("");
        let params = req.get("params").cloned().unwrap_or_else(|| json!({}));

        // JSON-RPC notifications (no `id`) must never receive a response.
        let id = id?;

        let response = match method {
            "initialize" => json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": { "listChanged": false }
                    },
                    "serverInfo": {
                        "name": "allyh",
                        "version": env!("CARGO_PKG_VERSION"),
                    }
                }
            }),
            "ping" => json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {}
            }),
            "tools/list" => json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "tools": self.tool_descriptors()
                }
            }),
            "tools/call" => {
                let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let args = params
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                let call_result = self.dispatch_tool(tool_name, args).await;
                json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": call_result,
                })
            }
            other => json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {
                    "code": -32601,
                    "message": format!("Method not found: '{other}'")
                }
            }),
        };

        Some(response)
    }

    /// Canonical MCP tool descriptors exposed by `allyh mcp serve`.
    pub fn tool_descriptors(&self) -> Vec<Value> {
        vec![
            mcp_tool_desc(
                "query_invariants",
                "Hybrid vector + BM25 search over active bi-temporal architectural invariants in LanceDB (invalid_at IS NULL).",
                json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Search query or topic (empty lists recent active invariants)" },
                        "top_k": { "type": "integer", "description": "Maximum invariants to return (default 8)" }
                    }
                }),
                true,
                false,
            ),
            mcp_tool_desc(
                "record_decision",
                "Commit a candidate architectural decision through AllyH's bi-temporal conflict gate (ADD/UPDATE/INVALIDATE/NOOP).",
                json!({
                    "type": "object",
                    "required": ["topic", "rule_statement"],
                    "properties": {
                        "topic": { "type": "string" },
                        "rule_statement": { "type": "string" },
                        "rationale": { "type": "string" },
                        "affected_paths": { "type": "array", "items": { "type": "string" } }
                    }
                }),
                false,
                false,
            ),
            mcp_tool_desc(
                "invalidate_decision",
                "Logically invalidate a stale architectural decision (sets invalid_at = NOW() without hard-deleting history).",
                json!({
                    "type": "object",
                    "required": ["decision_id"],
                    "properties": {
                        "decision_id": { "type": "string", "description": "Decision ID or unique prefix" }
                    }
                }),
                false,
                true,
            ),
            mcp_tool_desc(
                "pin_decision",
                "Pin or unpin an active architectural decision so it is prioritized in queries.",
                json!({
                    "type": "object",
                    "required": ["decision_id"],
                    "properties": {
                        "decision_id": { "type": "string" }
                    }
                }),
                false,
                false,
            ),
            mcp_tool_desc(
                "inspect_symbol",
                "Inspect a source symbol or file slice across supported languages, returning its SHA-256 hash and 1-hop/2-hop RepoGraph neighborhood.",
                json!({
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": { "type": "string" },
                        "hops": { "type": "integer" },
                        "start_line": { "type": "integer" },
                        "end_line": { "type": "integer" }
                    }
                }),
                true,
                false,
            ),
            mcp_tool_desc(
                "get_ego_graph",
                "Return the compact 1-hop/2-hop polyglot RepoGraph ego-graph (with LSP semantic enrichment when available) for a symbol or file.",
                json!({
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": { "type": "string" },
                        "hops": { "type": "integer" }
                    }
                }),
                true,
                false,
            ),
            mcp_tool_desc(
                "get_env_delta",
                "Scan workspace SHA-256 hashes, git status, out-of-band edits, detected languages/toolchains, and active LSP servers.",
                json!({
                    "type": "object",
                    "properties": {
                        "run_build_check": { "type": "boolean" }
                    }
                }),
                true,
                false,
            ),
            mcp_tool_desc(
                "list_skills",
                "List all reusable procedural skills stored in LanceDB for this repository partition.",
                json!({ "type": "object", "properties": {} }),
                true,
                false,
            ),
            mcp_tool_desc(
                "get_skill",
                "Load the full instructions and triggers of a registered skill by name or slug.",
                json!({
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": { "type": "string" }
                    }
                }),
                true,
                false,
            ),
            mcp_tool_desc(
                "add_skill",
                "Register or update a reusable procedural skill in out-of-tree LanceDB storage.",
                json!({
                    "type": "object",
                    "required": ["name", "instructions"],
                    "properties": {
                        "name": { "type": "string" },
                        "description": { "type": "string" },
                        "instructions": { "type": "string" },
                        "triggers": { "type": "array", "items": { "type": "string" } }
                    }
                }),
                false,
                false,
            ),
            mcp_tool_desc(
                "list_tasks",
                "List structured execution tasks and verification statuses stored in LanceDB.",
                json!({ "type": "object", "properties": {} }),
                true,
                false,
            ),
            mcp_tool_desc(
                "inspect_sim",
                "Inspect Self-Improvement Memory (SIM) runs, problem entries (P1..Pn), and alias resolutions.",
                json!({
                    "type": "object",
                    "properties": {
                        "problem_id": { "type": "string" },
                        "run_id": { "type": "string" }
                    }
                }),
                true,
                false,
            ),
        ]
    }

    async fn dispatch_tool(&self, name: &str, args: Value) -> Value {
        match self.dispatch_tool_inner(name, args).await {
            Ok(payload) => {
                let text = serde_json::to_string_pretty(&payload).unwrap_or_default();
                json!({
                    "content": [{ "type": "text", "text": text }],
                    "structuredContent": payload,
                    "isError": false
                })
            }
            Err(err) => json!({
                "content": [{ "type": "text", "text": err }],
                "isError": true
            }),
        }
    }

    async fn dispatch_tool_inner(&self, name: &str, args: Value) -> std::result::Result<Value, String> {
        match name {
            "query_invariants" => {
                let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let top_k = args
                    .get("top_k")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(8)
                    .clamp(1, 50) as usize;
                let pinned = self.pinned_decisions.lock().await.clone();
                let results = if query.trim().is_empty() {
                    self.store
                        .list_active_decisions(&self.repo_id, &self.scope_id)
                        .await
                        .map_err(|e| e.to_string())?
                        .into_iter()
                        .take(top_k)
                        .map(|d| {
                            let is_pinned = pinned.contains(&d.id);
                            json!({
                                "id": d.id,
                                "topic": d.topic,
                                "rule_statement": d.rule_statement,
                                "rationale": d.rationale,
                                "affected_paths": d.affected_paths,
                                "valid_at": d.valid_at,
                                "pinned": is_pinned,
                            })
                        })
                        .collect::<Vec<_>>()
                } else {
                    self.store
                        .hybrid_search_decisions(&self.repo_id, &self.scope_id, query, top_k)
                        .await
                        .map_err(|e| e.to_string())?
                        .into_iter()
                        .map(|s| {
                            let is_pinned = pinned.contains(&s.decision.id);
                            json!({
                                "id": s.decision.id,
                                "topic": s.decision.topic,
                                "rule_statement": s.decision.rule_statement,
                                "rationale": s.decision.rationale,
                                "affected_paths": s.decision.affected_paths,
                                "valid_at": s.decision.valid_at,
                                "score": s.score,
                                "pinned": is_pinned,
                            })
                        })
                        .collect::<Vec<_>>()
                };
                Ok(json!({
                    "repo_id": self.repo_id,
                    "scope_id": self.scope_id,
                    "count": results.len(),
                    "invariants": results,
                }))
            }
            "record_decision" => {
                let topic = args
                    .get("topic")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                let rule_statement = args
                    .get("rule_statement")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if topic.is_empty() || rule_statement.is_empty() {
                    return Err("record_decision requires non-empty 'topic' and 'rule_statement'".to_string());
                }
                let rationale = args
                    .get("rationale")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Recorded via AllyH MCP server")
                    .to_string();
                let affected_paths: Vec<String> = args
                    .get("affected_paths")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|x| x.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();

                let outcome = self
                    .consolidator
                    .consolidate_decision(
                        &self.repo_id,
                        &self.scope_id,
                        CandidateDecision {
                            topic,
                            rule_statement,
                            rationale,
                            affected_paths,
                        },
                    )
                    .await
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(outcome).map_err(|e| e.to_string())
            }
            "invalidate_decision" => {
                let prefix = args
                    .get("decision_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim();
                if prefix.is_empty() {
                    return Err("invalidate_decision requires 'decision_id'".to_string());
                }
                let active = self
                    .store
                    .list_active_decisions(&self.repo_id, &self.scope_id)
                    .await
                    .map_err(|e| e.to_string())?;
                let resolved_id = active
                    .iter()
                    .find(|d| d.id == prefix || d.id.starts_with(prefix))
                    .map(|d| d.id.clone())
                    .unwrap_or_else(|| prefix.to_string());
                let updated = self
                    .store
                    .invalidate_decision(&self.repo_id, &self.scope_id, &resolved_id, None, Utc::now())
                    .await
                    .map_err(|e| e.to_string())?;
                self.pinned_decisions.lock().await.remove(&resolved_id);
                Ok(json!({
                    "decision_id": resolved_id,
                    "invalidated_count": updated,
                }))
            }
            "pin_decision" => {
                let prefix = args
                    .get("decision_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim();
                if prefix.is_empty() {
                    return Err("pin_decision requires 'decision_id'".to_string());
                }
                let active = self
                    .store
                    .list_active_decisions(&self.repo_id, &self.scope_id)
                    .await
                    .map_err(|e| e.to_string())?;
                let resolved_id = active
                    .iter()
                    .find(|d| d.id == prefix || d.id.starts_with(prefix))
                    .map(|d| d.id.clone())
                    .unwrap_or_else(|| prefix.to_string());
                let mut guard = self.pinned_decisions.lock().await;
                let pinned = if guard.contains(&resolved_id) {
                    guard.remove(&resolved_id);
                    false
                } else {
                    guard.insert(resolved_id.clone());
                    true
                };
                Ok(json!({ "decision_id": resolved_id, "pinned": pinned }))
            }
            "inspect_symbol" => {
                let parsed: InspectSymbolArgs =
                    serde_json::from_value(args).map_err(|e| e.to_string())?;
                let out = self
                    .inspect_symbol
                    .execute(parsed)
                    .await
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(out).map_err(|e| e.to_string())
            }
            "get_ego_graph" => {
                let query = args
                    .get("query")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim();
                if query.is_empty() {
                    return Err("get_ego_graph requires 'query'".to_string());
                }
                let hops = args.get("hops").and_then(|v| v.as_u64()).unwrap_or(1).clamp(1, 2) as usize;
                let ego = self
                    .env
                    .ego_graph_for(query, hops)
                    .await
                    .map_err(|e| e.to_string())?;
                Ok(json!({
                    "summary": ego.render_compact(),
                    "ego_graph": ego,
                }))
            }
            "get_env_delta" => {
                let run_build = args
                    .get("run_build_check")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let delta = self
                    .env
                    .sync_and_collect_delta(0, run_build)
                    .await
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(delta).map_err(|e| e.to_string())
            }
            "list_skills" => {
                let skills = self
                    .store
                    .list_skills(&self.repo_id, &self.scope_id)
                    .await
                    .map_err(|e| e.to_string())?;
                Ok(json!({ "count": skills.len(), "skills": skills }))
            }
            "get_skill" => {
                let parsed: GetSkillArgs =
                    serde_json::from_value(args).map_err(|e| e.to_string())?;
                let out = self
                    .get_skill
                    .execute(parsed)
                    .await
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(out).map_err(|e| e.to_string())
            }
            "add_skill" => {
                let parsed: AddSkillArgs =
                    serde_json::from_value(args).map_err(|e| e.to_string())?;
                let out = self
                    .add_skill
                    .execute(parsed)
                    .await
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(out).map_err(|e| e.to_string())
            }
            "list_tasks" => {
                let tasks = self
                    .store
                    .list_tasks(&self.repo_id, &self.scope_id)
                    .await
                    .map_err(|e| e.to_string())?;
                Ok(json!({ "count": tasks.len(), "tasks": tasks }))
            }
            "inspect_sim" => {
                let parsed: InspectSimArgs =
                    serde_json::from_value(args).map_err(|e| e.to_string())?;
                let out = self
                    .inspect_sim
                    .execute(parsed)
                    .await
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(out).map_err(|e| e.to_string())
            }
            other => Err(format!("unknown AllyH MCP tool '{other}'")),
        }
    }
}

fn mcp_tool_desc(
    name: &str,
    description: &str,
    input_schema: Value,
    read_only: bool,
    destructive: bool,
) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": input_schema,
        "annotations": {
            "readOnlyHint": read_only,
            "destructiveHint": destructive,
            "openWorldHint": false
        }
    })
}

async fn write_json_line<W: AsyncWrite + Unpin>(writer: &mut W, payload: &Value) -> Result<()> {
    let mut line = serde_json::to_string(payload)?;
    line.push('\n');
    writer.write_all(line.as_bytes()).await?;
    writer.flush().await?;
    Ok(())
}
