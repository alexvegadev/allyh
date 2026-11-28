use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock as StdRwLock};
use std::time::Duration;

use allyh_memory::{McpServerConfig, McpServerTransport};
use rig_core::completion::{ToolAnnotations, ToolDefinition};
use rig_core::message::ToolName;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{Mutex, oneshot};

use crate::error::{Result, ToolError};

/// MCP JSON-RPC protocol revision negotiated by AllyH client & server.
pub const MCP_PROTOCOL_VERSION: &str = "2025-03-26";
const DEFAULT_MCP_TIMEOUT: Duration = Duration::from_secs(15);

/// Tool discovered from an external MCP server and mounted into AllyH's [`crate::AciToolset`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRemoteTool {
    pub server_name: String,
    pub remote_name: String,
    pub qualified_name: String,
    pub description: String,
    pub input_schema: Value,
    /// Conservative read-only flag: true ONLY when server explicitly sets `readOnlyHint: true`
    /// and does NOT set `destructiveHint: true`. All other external tools are write-gated.
    pub read_only: bool,
    pub destructive: bool,
    pub open_world: bool,
}

impl McpRemoteTool {
    pub fn to_rig_definition(&self) -> Option<ToolDefinition> {
        let name = ToolName::new(&self.qualified_name).ok()?;
        let desc = format!("[MCP:{}] {}", self.server_name, self.description);
        let annotations = ToolAnnotations::default()
            .with_title(format!("MCP {}: {}", self.server_name, self.remote_name))
            .with_read_only(self.read_only)
            .with_destructive(self.destructive)
            .with_open_world(self.open_world);
        Some(ToolDefinition::new(name, desc, self.input_schema.clone()).with_annotations(annotations))
    }
}

/// Live connection status snapshot for an external MCP server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerStatus {
    pub name: String,
    pub transport: String,
    pub target: String,
    pub connected: bool,
    pub tool_count: usize,
    pub tools: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}

/// Active transport backing a connected MCP server.
enum McpTransportHandle {
    Stdio(Arc<StdioMcpTransport>),
    Http(Arc<HttpMcpTransport>),
}

impl McpTransportHandle {
    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        match self {
            Self::Stdio(t) => t.request(method, params).await,
            Self::Http(t) => t.request(method, params).await,
        }
    }

    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        match self {
            Self::Stdio(t) => t.notify(method, params).await,
            Self::Http(t) => t.notify(method, params).await,
        }
    }

    fn is_alive(&self) -> bool {
        match self {
            Self::Stdio(t) => t.is_alive(),
            Self::Http(_) => true,
        }
    }

    async fn shutdown(&self) {
        if let Self::Stdio(t) = self {
            t.shutdown().await;
        }
    }
}

struct StdioMcpTransport {
    server_name: String,
    stdin: Arc<Mutex<ChildStdin>>,
    child: Arc<Mutex<Child>>,
    next_id: AtomicU64,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value>>>>>,
    alive: Arc<AtomicBool>,
}

impl StdioMcpTransport {
    async fn spawn(
        server_name: &str,
        command: &str,
        args: &[String],
        env: &[(String, String)],
    ) -> Result<Self> {
        let mut cmd = Command::new(command);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        for (k, v) in env {
            cmd.env(k, v);
        }

        let mut child = cmd.spawn().map_err(|e| {
            ToolError::InvalidArgs(format!(
                "failed to spawn MCP stdio server `{server_name}` (`{command}`): {e}"
            ))
        })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| ToolError::InvalidArgs(format!("MCP `{server_name}` missing stdin")))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| ToolError::InvalidArgs(format!("MCP `{server_name}` missing stdout")))?;

        let stdin = Arc::new(Mutex::new(stdin));
        let child = Arc::new(Mutex::new(child));
        let pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value>>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let alive = Arc::new(AtomicBool::new(true));

        let pending_reader = Arc::clone(&pending);
        let alive_reader = Arc::clone(&alive);
        let name_log = server_name.to_string();

        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        let trimmed = line.trim();
                        if trimmed.is_empty() || !trimmed.starts_with('{') {
                            continue;
                        }
                        let Ok(msg) = serde_json::from_str::<Value>(trimmed) else {
                            continue;
                        };
                        if let Some(id_val) = msg.get("id").and_then(|v| v.as_u64()) {
                            let tx = pending_reader.lock().await.remove(&id_val);
                            if let Some(tx) = tx {
                                if let Some(err) = msg.get("error") {
                                    let err_msg = err
                                        .get("message")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("MCP JSON-RPC error");
                                    let _ = tx.send(Err(ToolError::InvalidArgs(format!(
                                        "MCP `{name_log}` error: {err_msg}"
                                    ))));
                                } else {
                                    let _ = tx.send(Ok(msg
                                        .get("result")
                                        .cloned()
                                        .unwrap_or(Value::Null)));
                                }
                            }
                        }
                    }
                }
            }
            alive_reader.store(false, Ordering::SeqCst);
            let mut map = pending_reader.lock().await;
            for (_, tx) in map.drain() {
                let _ = tx.send(Err(ToolError::InvalidArgs(format!(
                    "MCP stdio server `{name_log}` closed connection"
                ))));
            }
        });

        Ok(Self {
            server_name: server_name.to_string(),
            stdin,
            child,
            next_id: AtomicU64::new(1),
            pending,
            alive,
        })
    }

    fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        if !self.is_alive() {
            return Err(ToolError::InvalidArgs(format!(
                "MCP server `{}` is disconnected",
                self.server_name
            )));
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);

        let payload = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        let mut line = serde_json::to_string(&payload)?;
        line.push('\n');

        {
            let mut guard = self.stdin.lock().await;
            if let Err(err) = guard.write_all(line.as_bytes()).await {
                self.pending.lock().await.remove(&id);
                return Err(ToolError::Io(err));
            }
            let _ = guard.flush().await;
        }

        match tokio::time::timeout(DEFAULT_MCP_TIMEOUT, rx).await {
            Ok(Ok(res)) => res,
            Ok(Err(_)) => Err(ToolError::InvalidArgs(format!(
                "MCP server `{}` dropped response for `{method}`",
                self.server_name
            ))),
            Err(_) => {
                self.pending.lock().await.remove(&id);
                Err(ToolError::InvalidArgs(format!(
                    "MCP server `{}` timed out on `{method}`",
                    self.server_name
                )))
            }
        }
    }

    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        if !self.is_alive() {
            return Ok(());
        }
        let payload = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        let mut line = serde_json::to_string(&payload)?;
        line.push('\n');
        let mut guard = self.stdin.lock().await;
        guard.write_all(line.as_bytes()).await?;
        guard.flush().await?;
        Ok(())
    }

    async fn shutdown(&self) {
        self.alive.store(false, Ordering::SeqCst);
        let mut child = self.child.lock().await;
        let _ = child.start_kill();
        let _ = tokio::time::timeout(Duration::from_millis(300), child.wait()).await;
    }
}

/// Streamable HTTP / SSE MCP transport (supports JSON and `text/event-stream` responses).
struct HttpMcpTransport {
    server_name: String,
    url: String,
    headers: Vec<(String, String)>,
    client: reqwest::Client,
    next_id: AtomicU64,
    session_id: Mutex<Option<String>>,
}

impl HttpMcpTransport {
    fn new(server_name: &str, url: String, headers: Vec<(String, String)>) -> Self {
        Self {
            server_name: server_name.to_string(),
            url,
            headers,
            client: reqwest::Client::builder()
                .timeout(DEFAULT_MCP_TIMEOUT)
                .build()
                .unwrap_or_default(),
            next_id: AtomicU64::new(1),
            session_id: Mutex::new(None),
        }
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let payload = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let mut req = self
            .client
            .post(&self.url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream");
        if let Some(sid) = self.session_id.lock().await.as_deref() {
            req = req.header("Mcp-Session-Id", sid);
        }
        for (k, v) in &self.headers {
            req = req.header(k.as_str(), v.as_str());
        }

        let resp = req.json(&payload).send().await.map_err(|e| {
            ToolError::InvalidArgs(format!(
                "MCP HTTP request to `{}` (`{}`) failed: {e}",
                self.server_name, self.url
            ))
        })?;

        if let Some(sid) = resp
            .headers()
            .get("mcp-session-id")
            .and_then(|v| v.to_str().ok())
        {
            *self.session_id.lock().await = Some(sid.to_string());
        }

        let status = resp.status();
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();
        let body = resp.text().await.map_err(|e| {
            ToolError::InvalidArgs(format!(
                "MCP HTTP body read failed for `{}`: {e}",
                self.server_name
            ))
        })?;

        if !status.is_success() {
            return Err(ToolError::InvalidArgs(format!(
                "MCP HTTP server `{}` returned {status}: {}",
                self.server_name,
                body.chars().take(240).collect::<String>()
            )));
        }

        let rpc_msg = if content_type.contains("text/event-stream") {
            parse_sse_json_rpc(&body, Some(id)).ok_or_else(|| {
                ToolError::InvalidArgs(format!(
                    "MCP SSE stream from `{}` contained no JSON-RPC response for id {id}",
                    self.server_name
                ))
            })?
        } else {
            serde_json::from_str::<Value>(&body)?
        };

        if let Some(err) = rpc_msg.get("error") {
            let err_msg = err
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("MCP HTTP JSON-RPC error");
            return Err(ToolError::InvalidArgs(format!(
                "MCP `{}` error: {err_msg}",
                self.server_name
            )));
        }
        Ok(rpc_msg.get("result").cloned().unwrap_or(Value::Null))
    }

    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let payload = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        let mut req = self
            .client
            .post(&self.url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream");
        if let Some(sid) = self.session_id.lock().await.as_deref() {
            req = req.header("Mcp-Session-Id", sid);
        }
        for (k, v) in &self.headers {
            req = req.header(k.as_str(), v.as_str());
        }
        let _ = req.json(&payload).send().await;
        Ok(())
    }
}

struct ConnectedMcpServer {
    config: McpServerConfig,
    transport: Option<McpTransportHandle>,
    tools: Vec<McpRemoteTool>,
    last_error: Option<String>,
}

/// Manages external MCP server connections (`stdio` & Streamable HTTP/SSE) and enforces
/// AllyH's Control Gate (`writes_enabled`) on every remote tool invocation.
#[derive(Clone, Default)]
pub struct McpClientManager {
    servers: Arc<Mutex<HashMap<String, ConnectedMcpServer>>>,
    /// Synchronously readable tool snapshot for `AciToolset::definitions()`.
    tool_catalog: Arc<StdRwLock<Vec<McpRemoteTool>>>,
}

impl McpClientManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Connect to (or reconnect) an external MCP server, run `initialize` + `tools/list`,
    /// and mount its tools into the live catalog.
    pub async fn connect_server(&self, config: McpServerConfig) -> Result<McpServerStatus> {
        let name = sanitize_ident(&config.name);
        if name.is_empty() {
            return Err(ToolError::InvalidArgs(
                "MCP server name must contain at least one ASCII alphanumeric character".to_string(),
            ));
        }

        let transport_res: Result<McpTransportHandle> = match &config.transport {
            McpServerTransport::Stdio { command, args, env } => {
                StdioMcpTransport::spawn(&name, command, args, env)
                    .await
                    .map(|t| McpTransportHandle::Stdio(Arc::new(t)))
            }
            McpServerTransport::Http { url, headers } => Ok(McpTransportHandle::Http(Arc::new(
                HttpMcpTransport::new(&name, url.clone(), headers.clone()),
            ))),
        };

        let transport = match transport_res {
            Ok(t) => t,
            Err(err) => {
                let err_str = err.to_string();
                let mut guard = self.servers.lock().await;
                guard.insert(
                    name.clone(),
                    ConnectedMcpServer {
                        config: config.clone(),
                        transport: None,
                        tools: Vec::new(),
                        last_error: Some(err_str.clone()),
                    },
                );
                drop(guard);
                self.refresh_catalog().await;
                return Err(ToolError::InvalidArgs(err_str));
            }
        };

        // MCP handshake: initialize -> notifications/initialized -> tools/list
        let init_params = json!({
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "allyh",
                "version": env!("CARGO_PKG_VERSION"),
            }
        });
        if let Err(err) = transport.request("initialize", init_params).await {
            transport.shutdown().await;
            let err_str = err.to_string();
            let mut guard = self.servers.lock().await;
            guard.insert(
                name.clone(),
                ConnectedMcpServer {
                    config: config.clone(),
                    transport: None,
                    tools: Vec::new(),
                    last_error: Some(err_str.clone()),
                },
            );
            drop(guard);
            self.refresh_catalog().await;
            return Err(ToolError::InvalidArgs(err_str));
        }
        let _ = transport.notify("notifications/initialized", json!({})).await;

        let tools_val = transport.request("tools/list", json!({})).await?;
        let remote_tools = parse_remote_tools(&name, &tools_val);

        let old = {
            let mut guard = self.servers.lock().await;
            guard.insert(
                name.clone(),
                ConnectedMcpServer {
                    config,
                    transport: Some(transport),
                    tools: remote_tools,
                    last_error: None,
                },
            )
        };
        if let Some(prev) = old
            && let Some(t) = prev.transport
        {
            t.shutdown().await;
        }
        self.refresh_catalog().await;

        let statuses = self.status().await;
        statuses
            .into_iter()
            .find(|s| s.name == name)
            .ok_or_else(|| ToolError::UnknownTool(format!("mcp server '{name}'")))
    }

    /// Best-effort connect all enabled MCP server configs (recording errors without aborting startup).
    pub async fn sync_configs(&self, configs: &[McpServerConfig]) {
        for cfg in configs {
            if cfg.enabled {
                let _ = self.connect_server(cfg.clone()).await;
            }
        }
    }

    /// Disconnect and unmount an external MCP server by name.
    pub async fn disconnect_server(&self, name: &str) -> bool {
        let key = sanitize_ident(name);
        let removed = self.servers.lock().await.remove(&key);
        let existed = removed.is_some();
        if let Some(entry) = removed
            && let Some(t) = entry.transport
        {
            t.shutdown().await;
        }
        self.refresh_catalog().await;
        existed
    }

    /// Synchronously return all mounted external MCP tools.
    pub fn list_tools(&self) -> Vec<McpRemoteTool> {
        self.tool_catalog
            .read()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    /// Look up a single mounted MCP tool by its qualified name (`mcp__<server>__<tool>`).
    pub fn find_tool(&self, qualified_name: &str) -> Option<McpRemoteTool> {
        self.tool_catalog
            .read()
            .ok()?
            .iter()
            .find(|t| t.qualified_name == qualified_name)
            .cloned()
    }

    /// Return status snapshots for all registered MCP servers.
    pub async fn status(&self) -> Vec<McpServerStatus> {
        let guard = self.servers.lock().await;
        let mut out: Vec<McpServerStatus> = guard
            .iter()
            .map(|(name, entry)| {
                let (transport_label, target) = match &entry.config.transport {
                    McpServerTransport::Stdio { command, args, .. } => {
                        let cmd_str = if args.is_empty() {
                            command.clone()
                        } else {
                            format!("{command} {}", args.join(" "))
                        };
                        ("stdio".to_string(), cmd_str)
                    }
                    McpServerTransport::Http { url, .. } => ("http".to_string(), url.clone()),
                };
                let connected = entry
                    .transport
                    .as_ref()
                    .is_some_and(McpTransportHandle::is_alive);
                McpServerStatus {
                    name: name.clone(),
                    transport: transport_label,
                    target,
                    connected,
                    tool_count: entry.tools.len(),
                    tools: entry.tools.iter().map(|t| t.qualified_name.clone()).collect(),
                    last_error: entry.last_error.clone(),
                }
            })
            .collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    /// Invoke a remote MCP tool while enforcing AllyH's Control Gate (`writes_enabled`).
    ///
    /// Returns `(tool_meta, raw_result_json, rendered_text, succeeded)`.
    pub async fn call_tool(
        &self,
        qualified_name: &str,
        arguments: Value,
        writes_enabled: bool,
    ) -> Result<(McpRemoteTool, Value, String, bool)> {
        let Some(tool) = self.find_tool(qualified_name) else {
            return Err(ToolError::UnknownTool(qualified_name.to_string()));
        };

        // Enforce AllyH Control Gate on every non-read-only external MCP tool!
        if !tool.read_only && !writes_enabled {
            return Err(ToolError::WriteGateLocked(format!(
                "MCP tool `{}` (server `{}`) is not marked read-only; approve a plan before executing mutating external tools.",
                tool.remote_name, tool.server_name
            )));
        }

        let transport_req = {
            let guard = self.servers.lock().await;
            let entry = guard.get(&tool.server_name).ok_or_else(|| {
                ToolError::UnknownTool(format!("MCP server `{}` not connected", tool.server_name))
            })?;
            match &entry.transport {
                Some(McpTransportHandle::Stdio(t)) => McpTransportHandle::Stdio(Arc::clone(t)),
                Some(McpTransportHandle::Http(t)) => McpTransportHandle::Http(Arc::clone(t)),
                None => {
                    return Err(ToolError::InvalidArgs(format!(
                        "MCP server `{}` is not connected",
                        tool.server_name
                    )));
                }
            }
        };

        let result = transport_req
            .request(
                "tools/call",
                json!({
                    "name": tool.remote_name,
                    "arguments": arguments,
                }),
            )
            .await?;

        let is_error = result
            .get("isError")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let rendered = render_mcp_call_result(&result);
        Ok((tool, result, rendered, !is_error))
    }

    /// Cleanly shut down all connected stdio MCP child processes.
    pub async fn shutdown_all(&self) {
        let entries: Vec<ConnectedMcpServer> = {
            let mut guard = self.servers.lock().await;
            guard.drain().map(|(_, v)| v).collect()
        };
        for entry in entries {
            if let Some(t) = entry.transport {
                t.shutdown().await;
            }
        }
        self.refresh_catalog().await;
    }

    async fn refresh_catalog(&self) {
        let mut all = Vec::new();
        {
            let guard = self.servers.lock().await;
            for entry in guard.values() {
                if entry
                    .transport
                    .as_ref()
                    .is_some_and(McpTransportHandle::is_alive)
                {
                    all.extend(entry.tools.clone());
                }
            }
        }
        all.sort_by(|a, b| a.qualified_name.cmp(&b.qualified_name));
        if let Ok(mut cat) = self.tool_catalog.write() {
            *cat = all;
        }
    }
}

/// Parse an MCP `tools/list` result into [`McpRemoteTool`] entries.
fn parse_remote_tools(server_name: &str, result: &Value) -> Vec<McpRemoteTool> {
    let Some(arr) = result.get("tools").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for item in arr {
        let Some(remote_name) = item.get("name").and_then(|v| v.as_str()) else {
            continue;
        };
        let description = item
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("External MCP tool")
            .to_string();
        let input_schema = item
            .get("inputSchema")
            .cloned()
            .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));

        let ann = item.get("annotations");
        let explicit_read_only = ann
            .and_then(|a| a.get("readOnlyHint"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let explicit_destructive = ann
            .and_then(|a| a.get("destructiveHint"))
            .and_then(|v| v.as_bool());
        let open_world = ann
            .and_then(|a| a.get("openWorldHint"))
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let read_only = explicit_read_only && explicit_destructive != Some(true);
        let destructive = explicit_destructive.unwrap_or(!read_only);
        let qualified_name = format!(
            "mcp__{}__{}",
            sanitize_ident(server_name),
            sanitize_ident(remote_name)
        );

        out.push(McpRemoteTool {
            server_name: server_name.to_string(),
            remote_name: remote_name.to_string(),
            qualified_name,
            description,
            input_schema,
            read_only,
            destructive,
            open_world,
        });
    }
    out
}

/// Extract text blocks from an MCP `tools/call` response payload.
fn render_mcp_call_result(result: &Value) -> String {
    if let Some(arr) = result.get("content").and_then(|v| v.as_array()) {
        let mut parts = Vec::new();
        for block in arr {
            if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                parts.push(text.to_string());
            } else if let Ok(s) = serde_json::to_string(block) {
                parts.push(s);
            }
        }
        if !parts.is_empty() {
            return parts.join("\n");
        }
    }
    serde_json::to_string_pretty(result).unwrap_or_default()
}

/// Parse an SSE (`text/event-stream`) response body to extract the matching JSON-RPC payload.
pub fn parse_sse_json_rpc(body: &str, expected_id: Option<u64>) -> Option<Value> {
    let mut last_valid = None;
    for line in body.lines() {
        let trimmed = line.trim();
        let Some(data) = trimmed.strip_prefix("data:") else {
            continue;
        };
        let payload = data.trim();
        if payload.is_empty() || payload == "[DONE]" {
            continue;
        }
        if let Ok(val) = serde_json::from_str::<Value>(payload) {
            if let Some(want) = expected_id
                && val.get("id").and_then(|v| v.as_u64()) == Some(want)
            {
                return Some(val);
            }
            last_valid = Some(val);
        }
    }
    last_valid
}

/// Parse a CLI `--mcp-server` spec of the form `name=http://...` or `name=cmd arg1 arg2`.
pub fn parse_mcp_server_spec(spec: &str) -> Result<McpServerConfig> {
    let Some((raw_name, raw_target)) = spec.split_once('=') else {
        return Err(ToolError::InvalidArgs(format!(
            "invalid MCP server spec '{spec}' (expected `name=http://...` or `name=command arg1 ...`)"
        )));
    };
    let name = sanitize_ident(raw_name);
    let target = raw_target.trim();
    if name.is_empty() || target.is_empty() {
        return Err(ToolError::InvalidArgs(format!(
            "MCP server spec '{spec}' must have non-empty name and target"
        )));
    }
    let transport = if target.starts_with("http://") || target.starts_with("https://") {
        McpServerTransport::Http {
            url: target.to_string(),
            headers: Vec::new(),
        }
    } else {
        let mut parts = target.split_whitespace();
        let command = parts.next().unwrap_or("").to_string();
        let args = parts.map(str::to_string).collect();
        McpServerTransport::Stdio {
            command,
            args,
            env: Vec::new(),
        }
    };
    Ok(McpServerConfig {
        name,
        transport,
        enabled: true,
    })
}

fn sanitize_ident(raw: &str) -> String {
    let mut out = String::new();
    for ch in raw.trim().chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    out.trim_matches('_').to_string()
}
