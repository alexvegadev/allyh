use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{Mutex, Notify, RwLock, oneshot};

use crate::error::{EnvError, Result};
use crate::lsp::protocol::{
    LspCallHierarchyItem, LspDiagnostic, LspLocation, LspPosition, encode_lsp_frame,
    parse_locations_value, path_to_file_uri,
};

/// Default per-request timeout for LSP calls.
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(6);

/// Configuration describing how to spawn an external stdio LSP server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LspServerSpec {
    /// Server family key (`rust`, `typescript`, `python`, `go`, `cpp`).
    pub family: String,
    /// Human-readable server name (e.g., `rust-analyzer`, `vtsls`, `pyright`, `gopls`).
    pub name: String,
    /// Executable path or command name.
    pub command: String,
    /// CLI arguments (e.g. `["--stdio"]`).
    pub args: Vec<String>,
    /// Optional extra environment variables passed to the spawned process.
    pub env: Vec<(String, String)>,
    /// When true, only lazily spawn if the workspace actually contains a matching project manifest
    /// (prevents e.g. spawning `rust-analyzer` on bare temp directories without `Cargo.toml`).
    pub require_manifest: bool,
}

/// Cached published diagnostics for a single document URI.
#[derive(Debug, Clone)]
struct PublishedDiagnosticsEntry {
    generation: u64,
    diagnostics: Vec<LspDiagnostic>,
}

/// Tracked open-document state in an active LSP session.
#[derive(Debug, Clone)]
struct OpenDocument {
    version: i32,
    content_hash: String,
}

/// Live stdio JSON-RPC 2.0 Language Server Protocol client.
pub struct LspClient {
    spec: LspServerSpec,
    workspace_root: PathBuf,
    stdin: Arc<Mutex<ChildStdin>>,
    child: Arc<Mutex<Child>>,
    next_id: AtomicU64,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value>>>>>,
    open_docs: Arc<Mutex<HashMap<String, OpenDocument>>>,
    diagnostics: Arc<RwLock<HashMap<String, PublishedDiagnosticsEntry>>>,
    diag_gen: Arc<AtomicU64>,
    diag_notify: Arc<Notify>,
    alive: Arc<AtomicBool>,
    last_activity: Arc<Mutex<Instant>>,
    supports_pull_diagnostics: AtomicBool,
}

impl LspClient {
    /// Spawn a stdio LSP server and complete the `initialize` -> `initialized` handshake.
    pub async fn start(
        spec: LspServerSpec,
        workspace_root: impl Into<PathBuf>,
        init_timeout: Duration,
    ) -> Result<Self> {
        let workspace_root = workspace_root.into();
        let mut cmd = Command::new(&spec.command);
        cmd.args(&spec.args)
            .current_dir(&workspace_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        for (k, v) in &spec.env {
            cmd.env(k, v);
        }

        let mut child = cmd.spawn().map_err(|e| {
            EnvError::Command(format!(
                "failed to spawn LSP server `{}` (`{}`): {e}",
                spec.name, spec.command
            ))
        })?;

        let stdin = child.stdin.take().ok_or_else(|| {
            EnvError::Command(format!("LSP server `{}` has no stdin pipe", spec.name))
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            EnvError::Command(format!("LSP server `{}` has no stdout pipe", spec.name))
        })?;

        let stdin = Arc::new(Mutex::new(stdin));
        let child = Arc::new(Mutex::new(child));
        let pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value>>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let diagnostics: Arc<RwLock<HashMap<String, PublishedDiagnosticsEntry>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let diag_gen = Arc::new(AtomicU64::new(0));
        let diag_notify = Arc::new(Notify::new());
        let alive = Arc::new(AtomicBool::new(true));

        // Background stdout reader task decoding `Content-Length` framed JSON-RPC messages.
        let stdin_reader = Arc::clone(&stdin);
        let pending_reader = Arc::clone(&pending);
        let diagnostics_reader = Arc::clone(&diagnostics);
        let diag_gen_reader = Arc::clone(&diag_gen);
        let diag_notify_reader = Arc::clone(&diag_notify);
        let alive_reader = Arc::clone(&alive);
        let server_name_log = spec.name.clone();

        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout);
            while let Some(msg) = read_lsp_message(&mut reader).await {
                if let Some(method) = msg.get("method").and_then(|v| v.as_str()) {
                    if let Some(req_id) = msg.get("id").cloned() {
                        // Server-to-client request: reply cleanly so servers never stall.
                        let reply = build_server_request_reply(method, &req_id);
                        let frame = encode_lsp_frame(&reply);
                        let mut guard = stdin_reader.lock().await;
                        let _ = guard.write_all(&frame).await;
                        let _ = guard.flush().await;
                    } else if method == "textDocument/publishDiagnostics"
                        && let Some(params) = msg.get("params")
                        && let Some(uri) = params.get("uri").and_then(|v| v.as_str())
                    {
                        let parsed: Vec<LspDiagnostic> = params
                            .get("diagnostics")
                            .cloned()
                            .and_then(|v| serde_json::from_value(v).ok())
                            .unwrap_or_default();
                        let generation = diag_gen_reader.fetch_add(1, Ordering::SeqCst) + 1;
                        let mut map = diagnostics_reader.write().await;
                        map.insert(
                            uri.to_string(),
                            PublishedDiagnosticsEntry {
                                generation,
                                diagnostics: parsed,
                            },
                        );
                        drop(map);
                        diag_notify_reader.notify_waiters();
                    }
                } else if let Some(id_val) = msg.get("id").and_then(|v| v.as_u64()) {
                    let tx = {
                        let mut map = pending_reader.lock().await;
                        map.remove(&id_val)
                    };
                    if let Some(tx) = tx {
                        if let Some(err) = msg.get("error") {
                            let err_msg = err
                                .get("message")
                                .and_then(|v| v.as_str())
                                .unwrap_or("LSP error");
                            let _ = tx.send(Err(EnvError::Command(format!(
                                "LSP `{server_name_log}` error: {err_msg}"
                            ))));
                        } else {
                            let result = msg.get("result").cloned().unwrap_or(Value::Null);
                            let _ = tx.send(Ok(result));
                        }
                    }
                }
            }

            // Server closed stdout -> mark dead and fail all inflight requests immediately.
            alive_reader.store(false, Ordering::SeqCst);
            let mut map = pending_reader.lock().await;
            for (_, tx) in map.drain() {
                let _ = tx.send(Err(EnvError::Command(format!(
                    "LSP server `{server_name_log}` exited unexpectedly"
                ))));
            }
            diag_notify_reader.notify_waiters();
        });

        let client = Self {
            spec,
            workspace_root,
            stdin,
            child,
            next_id: AtomicU64::new(1),
            pending,
            open_docs: Arc::new(Mutex::new(HashMap::new())),
            diagnostics,
            diag_gen,
            diag_notify,
            alive,
            last_activity: Arc::new(Mutex::new(Instant::now())),
            supports_pull_diagnostics: AtomicBool::new(false),
        };

        client.initialize(init_timeout).await?;
        Ok(client)
    }

    pub fn spec(&self) -> &LspServerSpec {
        &self.spec
    }

    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    pub async fn last_activity(&self) -> Instant {
        *self.last_activity.lock().await
    }

    pub async fn open_document_count(&self) -> usize {
        self.open_docs.lock().await.len()
    }

    pub async fn cached_diagnostic_files_count(&self) -> usize {
        self.diagnostics.read().await.len()
    }

    async fn touch(&self) {
        *self.last_activity.lock().await = Instant::now();
    }

    async fn initialize(&self, timeout: Duration) -> Result<()> {
        let root_uri = path_to_file_uri(&self.workspace_root);
        let root_name = self
            .workspace_root
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("workspace");

        let params = json!({
            "processId": std::process::id(),
            "clientInfo": {
                "name": "allyh",
                "version": env!("CARGO_PKG_VERSION"),
            },
            "rootUri": root_uri,
            "workspaceFolders": [{
                "uri": root_uri,
                "name": root_name,
            }],
            "capabilities": {
                "textDocument": {
                    "synchronization": {
                        "didSave": true,
                        "dynamicRegistration": false,
                    },
                    "references": { "dynamicRegistration": false },
                    "definition": { "dynamicRegistration": false, "linkSupport": true },
                    "implementation": { "dynamicRegistration": false, "linkSupport": true },
                    "callHierarchy": { "dynamicRegistration": false },
                    "publishDiagnostics": {
                        "relatedInformation": false,
                    },
                    "diagnostic": {
                        "dynamicRegistration": false,
                    },
                },
                "workspace": {
                    "workspaceFolders": true,
                },
            },
        });

        let init_result = self.request("initialize", params, timeout).await?;
        let has_diag_provider = init_result
            .get("capabilities")
            .and_then(|c| c.get("diagnosticProvider"))
            .is_some_and(|v| !v.is_null() && *v != Value::Bool(false));
        self.supports_pull_diagnostics
            .store(has_diag_provider, Ordering::SeqCst);

        self.notify("initialized", json!({})).await?;
        Ok(())
    }

    /// Send a JSON-RPC request and await its correlated response within `timeout`.
    pub async fn request(&self, method: &str, params: Value, timeout: Duration) -> Result<Value> {
        if !self.is_alive() {
            return Err(EnvError::Command(format!(
                "LSP server `{}` is not running",
                self.spec.name
            )));
        }
        self.touch().await;
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        {
            let mut map = self.pending.lock().await;
            map.insert(id, tx);
        }

        let payload = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        if let Err(err) = self.write_frame(&payload).await {
            self.pending.lock().await.remove(&id);
            return Err(err);
        }

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(res)) => res,
            Ok(Err(_)) => Err(EnvError::Command(format!(
                "LSP server `{}` dropped response channel for `{method}`",
                self.spec.name
            ))),
            Err(_) => {
                self.pending.lock().await.remove(&id);
                Err(EnvError::Command(format!(
                    "LSP `{}` timed out on `{method}` after {}ms",
                    self.spec.name,
                    timeout.as_millis()
                )))
            }
        }
    }

    /// Send a JSON-RPC notification (no response expected).
    pub async fn notify(&self, method: &str, params: Value) -> Result<()> {
        if !self.is_alive() {
            return Err(EnvError::Command(format!(
                "LSP server `{}` is not running",
                self.spec.name
            )));
        }
        self.touch().await;
        let payload = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        self.write_frame(&payload).await
    }

    async fn write_frame(&self, payload: &Value) -> Result<()> {
        let frame = encode_lsp_frame(payload);
        let mut guard = self.stdin.lock().await;
        guard
            .write_all(&frame)
            .await
            .map_err(|e| EnvError::Command(format!("LSP stdin write error: {e}")))?;
        guard
            .flush()
            .await
            .map_err(|e| EnvError::Command(format!("LSP stdin flush error: {e}")))?;
        Ok(())
    }

    /// Synchronize a file with the LSP server via `didOpen`, `didChange`, and optional `didSave`.
    /// Returns the document's `file://` URI and the diagnostics generation counter prior to syncing.
    pub async fn sync_document(
        &self,
        abs_path: &Path,
        language_id: &str,
        content: &str,
        content_hash: &str,
        emit_save: bool,
    ) -> Result<(String, u64)> {
        let uri = path_to_file_uri(abs_path);
        let pre_gen = self.diag_gen.load(Ordering::SeqCst);

        let mut docs = self.open_docs.lock().await;
        match docs.get_mut(&uri) {
            None => {
                docs.insert(
                    uri.clone(),
                    OpenDocument {
                        version: 1,
                        content_hash: content_hash.to_string(),
                    },
                );
                drop(docs);
                self.notify(
                    "textDocument/didOpen",
                    json!({
                        "textDocument": {
                            "uri": uri,
                            "languageId": language_id,
                            "version": 1,
                            "text": content,
                        }
                    }),
                )
                .await?;
            }
            Some(entry) if entry.content_hash != content_hash => {
                entry.version += 1;
                entry.content_hash = content_hash.to_string();
                let ver = entry.version;
                drop(docs);
                self.notify(
                    "textDocument/didChange",
                    json!({
                        "textDocument": {
                            "uri": uri,
                            "version": ver,
                        },
                        "contentChanges": [{ "text": content }],
                    }),
                )
                .await?;
            }
            Some(_) => {
                drop(docs);
            }
        }

        if emit_save {
            self.notify(
                "textDocument/didSave",
                json!({
                    "textDocument": { "uri": uri },
                    "text": content,
                }),
            )
            .await?;
        }

        Ok((uri, pre_gen))
    }

    /// Close a previously opened document.
    pub async fn close_document(&self, abs_path: &Path) -> Result<()> {
        let uri = path_to_file_uri(abs_path);
        let removed = self.open_docs.lock().await.remove(&uri).is_some();
        if removed && self.is_alive() {
            let _ = self
                .notify(
                    "textDocument/didClose",
                    json!({ "textDocument": { "uri": uri } }),
                )
                .await;
        }
        Ok(())
    }

    /// Fetch diagnostics for a synced document URI using LSP 3.17 pull diagnostics when advertised,
    /// otherwise awaiting a fresh `textDocument/publishDiagnostics` push notification within `wait_budget`.
    pub async fn fetch_diagnostics(
        &self,
        uri: &str,
        since_generation: u64,
        wait_budget: Duration,
    ) -> Result<Option<Vec<LspDiagnostic>>> {
        // 1. Prefer LSP 3.17 pull diagnostics (`textDocument/diagnostic`) if the server supports it.
        if self.supports_pull_diagnostics.load(Ordering::SeqCst)
            && let Ok(val) = self
                .request(
                    "textDocument/diagnostic",
                    json!({ "textDocument": { "uri": uri } }),
                    wait_budget.max(Duration::from_millis(500)),
                )
                .await
            && let Some(items) = val.get("items").cloned()
            && let Ok(parsed) = serde_json::from_value::<Vec<LspDiagnostic>>(items)
        {
            return Ok(Some(parsed));
        }

        // 2. Otherwise await a fresh `publishDiagnostics` notification for this URI.
        let deadline = Instant::now() + wait_budget;
        loop {
            {
                let map = self.diagnostics.read().await;
                if let Some(entry) = map.get(uri)
                    && entry.generation > since_generation
                {
                    return Ok(Some(entry.diagnostics.clone()));
                }
            }
            if !self.is_alive() {
                return Ok(None);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                // Return cached diagnostics if any exist, otherwise None (caller falls back to CLI).
                let map = self.diagnostics.read().await;
                return Ok(map.get(uri).map(|e| e.diagnostics.clone()));
            }
            let _ = tokio::time::timeout(remaining, self.diag_notify.notified()).await;
        }
    }

    /// Query `textDocument/references` (excluding declaration).
    pub async fn references(
        &self,
        uri: &str,
        position: LspPosition,
    ) -> Result<Vec<LspLocation>> {
        let val = self
            .request(
                "textDocument/references",
                json!({
                    "textDocument": { "uri": uri },
                    "position": position,
                    "context": { "includeDeclaration": false },
                }),
                DEFAULT_REQUEST_TIMEOUT,
            )
            .await?;
        Ok(parse_locations_value(&val))
    }

    /// Query `textDocument/definition`.
    pub async fn definition(
        &self,
        uri: &str,
        position: LspPosition,
    ) -> Result<Vec<LspLocation>> {
        let val = self
            .request(
                "textDocument/definition",
                json!({
                    "textDocument": { "uri": uri },
                    "position": position,
                }),
                DEFAULT_REQUEST_TIMEOUT,
            )
            .await?;
        Ok(parse_locations_value(&val))
    }

    /// Query `textDocument/implementation`.
    pub async fn implementation(
        &self,
        uri: &str,
        position: LspPosition,
    ) -> Result<Vec<LspLocation>> {
        let val = self
            .request(
                "textDocument/implementation",
                json!({
                    "textDocument": { "uri": uri },
                    "position": position,
                }),
                DEFAULT_REQUEST_TIMEOUT,
            )
            .await?;
        Ok(parse_locations_value(&val))
    }

    /// Query `textDocument/prepareCallHierarchy` -> `incomingCalls` & `outgoingCalls`.
    pub async fn call_hierarchy(
        &self,
        uri: &str,
        position: LspPosition,
    ) -> Result<(Vec<LspCallHierarchyItem>, Vec<LspCallHierarchyItem>)> {
        let prepared_val = self
            .request(
                "textDocument/prepareCallHierarchy",
                json!({
                    "textDocument": { "uri": uri },
                    "position": position,
                }),
                DEFAULT_REQUEST_TIMEOUT,
            )
            .await?;
        let items: Vec<LspCallHierarchyItem> =
            serde_json::from_value(prepared_val).unwrap_or_default();
        let Some(first) = items.into_iter().next() else {
            return Ok((Vec::new(), Vec::new()));
        };

        let item_val = serde_json::to_value(&first).unwrap_or(Value::Null);
        let mut incoming = Vec::new();
        if let Ok(in_val) = self
            .request(
                "callHierarchy/incomingCalls",
                json!({ "item": item_val }),
                DEFAULT_REQUEST_TIMEOUT,
            )
            .await
            && let Some(arr) = in_val.as_array()
        {
            for entry in arr {
                if let Some(from) = entry.get("from").cloned()
                    && let Ok(parsed) = serde_json::from_value::<LspCallHierarchyItem>(from)
                {
                    incoming.push(parsed);
                }
            }
        }

        let mut outgoing = Vec::new();
        if let Ok(out_val) = self
            .request(
                "callHierarchy/outgoingCalls",
                json!({ "item": item_val }),
                DEFAULT_REQUEST_TIMEOUT,
            )
            .await
            && let Some(arr) = out_val.as_array()
        {
            for entry in arr {
                if let Some(to) = entry.get("to").cloned()
                    && let Ok(parsed) = serde_json::from_value::<LspCallHierarchyItem>(to)
                {
                    outgoing.push(parsed);
                }
            }
        }

        Ok((incoming, outgoing))
    }

    /// Perform a clean LSP `shutdown` -> `exit` sequence.
    pub async fn shutdown(&self) {
        if self.is_alive() {
            let _ = self
                .request("shutdown", Value::Null, Duration::from_secs(2))
                .await;
            let _ = self.notify("exit", Value::Null).await;
            self.alive.store(false, Ordering::SeqCst);
        }
        let mut child = self.child.lock().await;
        let _ = tokio::time::timeout(Duration::from_millis(500), child.wait()).await;
        let _ = child.start_kill();
    }
}

fn build_server_request_reply(method: &str, id: &Value) -> Value {
    let result = match method {
        "workspace/configuration" => json!([{}]),
        "window/workDoneProgress/create"
        | "client/registerCapability"
        | "client/unregisterCapability" => Value::Null,
        "workspace/applyEdit" => json!({ "applied": false }),
        _ => Value::Null,
    };
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    })
}

/// Read a single `Content-Length` framed JSON-RPC message from an async buffered reader.
async fn read_lsp_message<R: AsyncBufReadExt + Unpin>(reader: &mut R) -> Option<Value> {
    let mut content_length: Option<usize> = None;
    loop {
        let mut line = String::new();
        let bytes_read = reader.read_line(&mut line).await.ok()?;
        if bytes_read == 0 {
            return None;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if content_length.is_some() {
                break;
            }
            continue;
        }
        if let Some((name, val)) = trimmed.split_once(':')
            && name.trim().eq_ignore_ascii_case("content-length")
            && let Ok(len) = val.trim().parse::<usize>()
        {
            content_length = Some(len);
        }
    }

    let len = content_length?;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).await.ok()?;
    serde_json::from_slice(&buf).ok()
}
