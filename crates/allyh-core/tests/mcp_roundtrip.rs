use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use allyh_core::{
    AllyHMcpServer, AllyHOrchestrator, HybridRouter, McpServerLaunchConfig, ModelProfile,
    OrchestratorConfig, RigCompletionBackend,
};
use allyh_memory::{McpServerConfig, McpServerTransport, RepoProfile, StorageConfig};
use allyh_tools::{McpClientManager, ToolError};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const SERVER_ENV: &str = "ALLYH_MCP_SERVE_TEST";
const REPO_ENV: &str = "ALLYH_MCP_TEST_REPO";
const DB_ENV: &str = "ALLYH_MCP_TEST_DB";
const SCOPE_ENV: &str = "ALLYH_MCP_TEST_SCOPE";

#[tokio::main]
async fn main() {
    if std::env::var_os(SERVER_ENV).is_some() {
        run_mcp_server_child().await;
        return;
    }

    test_stdio_mcp_client_server_roundtrip().await;
    test_streamable_http_and_sse_transport().await;
    println!("mcp_roundtrip: 2 passed, 0 failed");
}

async fn run_mcp_server_child() {
    let repo_root = PathBuf::from(std::env::var(REPO_ENV).expect("ALLYH_MCP_TEST_REPO"));
    let db_dir = PathBuf::from(std::env::var(DB_ENV).expect("ALLYH_MCP_TEST_DB"));
    let scope_id = std::env::var(SCOPE_ENV).unwrap_or_else(|_| "default".to_string());

    let server = AllyHMcpServer::initialize(
        McpServerLaunchConfig {
            repo_root,
            storage: StorageConfig::from_local_path(&db_dir),
            scope_id,
        },
        None,
    )
    .await
    .expect("initialize AllyHMcpServer");

    server
        .serve_stdio()
        .await
        .expect("serve_stdio should exit cleanly on EOF");
}

/// End-to-end test: real `AllyHMcpServer::serve_stdio()` child process connected to
/// `AllyHOrchestrator` / `AciToolset` via `McpClientManager` over stdio JSON-RPC 2.0.
async fn test_stdio_mcp_client_server_roundtrip() {
    let tmp = TempDir::new().expect("tempdir");
    let server_repo = tmp.path().join("server_repo");
    let server_db = tmp.path().join("server_db");
    let client_repo = tmp.path().join("client_repo");
    let client_db = tmp.path().join("client_db");

    std::fs::create_dir_all(server_repo.join("src")).expect("mkdir server_repo");
    std::fs::create_dir_all(client_repo.join("src")).expect("mkdir client_repo");

    // Seed a TypeScript file in the MCP server's repository so polyglot RepoGraph queries work.
    std::fs::write(
        server_repo.join("src/pricing.ts"),
        "export function computeTotal(price: number, tax: number): number {\n  return price + tax;\n}\n\nexport function checkout(price: number): number {\n  return computeTotal(price, price * 0.1);\n}\n",
    )
    .expect("write pricing.ts");
    std::fs::write(
        client_repo.join("src/lib.rs"),
        "pub fn client_entry() -> bool { true }\n",
    )
    .expect("write client lib.rs");

    let exe = std::env::current_exe().expect("current_exe");
    let mcp_cfg = McpServerConfig {
        name: "allyh".to_string(),
        transport: McpServerTransport::Stdio {
            command: exe.to_string_lossy().into_owned(),
            args: Vec::new(),
            env: vec![
                (SERVER_ENV.to_string(), "1".to_string()),
                (
                    REPO_ENV.to_string(),
                    server_repo.to_string_lossy().into_owned(),
                ),
                (DB_ENV.to_string(), server_db.to_string_lossy().into_owned()),
                (SCOPE_ENV.to_string(), "mcp-e2e".to_string()),
            ],
        },
        enabled: true,
    };

    let router = HybridRouter::new(
        ModelProfile::offline_mock("Primary"),
        ModelProfile::offline_mock("Worker"),
        Arc::new(RigCompletionBackend),
    );
    let orch = AllyHOrchestrator::initialize(
        OrchestratorConfig {
            repo_root: client_repo.clone(),
            storage: StorageConfig::from_local_path(&client_db),
            scope_id: "client-scope".to_string(),
            auto_verify: false,
            idle_optimize_interval: Duration::from_secs(60),
            enable_fs_watcher: false,
            extra_mcp_servers: vec![mcp_cfg.clone()],
        },
        router,
        None,
    )
    .await
    .expect("initialize client orchestrator with MCP server");

    // 1. Verify all 12 AllyH MCP tools are mounted in AciToolset::definitions()
    let defs = orch.toolset().definitions();
    let mounted_mcp: Vec<&str> = defs
        .iter()
        .map(|d| d.name.as_str())
        .filter(|n| n.starts_with("mcp__allyh__"))
        .collect();
    assert_eq!(
        mounted_mcp.len(),
        12,
        "expected 12 mounted AllyH MCP tools, got {mounted_mcp:?}"
    );
    assert!(mounted_mcp.contains(&"mcp__allyh__query_invariants"));
    assert!(mounted_mcp.contains(&"mcp__allyh__record_decision"));
    assert!(mounted_mcp.contains(&"mcp__allyh__get_ego_graph"));
    assert!(mounted_mcp.contains(&"mcp__allyh__add_skill"));

    let snap = orch.inspector_snapshot().await.expect("inspector snapshot");
    assert_eq!(snap.mcp_servers_count, 1);

    // 2. Control Gate Enforcement: Discovery mode (writes_enabled = false)
    assert!(!orch.toolset().writes_enabled());

    // Mutating MCP tool MUST be rejected while Control Gate is locked (both at McpClientManager
    // level and as a failed ToolExecutionRecord card through AciToolset::dispatch).
    let direct_refused = orch
        .toolset()
        .mcp_clients()
        .call_tool(
            "mcp__allyh__record_decision",
            json!({
                "topic": "error-handling",
                "rule_statement": "Use thiserror in library crates",
                "rationale": "Structured errors"
            }),
            false,
        )
        .await;
    assert!(
        matches!(direct_refused, Err(ToolError::WriteGateLocked(_))),
        "expected WriteGateLocked for mutating MCP tool in Discovery mode, got {direct_refused:?}"
    );
    let (refused_rec, _) = orch
        .toolset()
        .dispatch(
            "mcp__allyh__record_decision",
            json!({
                "topic": "error-handling",
                "rule_statement": "Use thiserror in library crates",
                "rationale": "Structured errors"
            }),
        )
        .await;
    assert!(
        !refused_rec.succeeded && refused_rec.output_text.contains("read-only"),
        "expected Control Gate refusal tool card, got: {refused_rec:?}"
    );

    // Read-only MCP tool (`get_ego_graph`) MUST succeed in Discovery mode on polyglot TS repo!
    let (ego_rec, _) = orch
        .toolset()
        .dispatch(
            "mcp__allyh__get_ego_graph",
            json!({ "query": "computeTotal" }),
        )
        .await;
    assert!(ego_rec.succeeded, "ego graph failed: {}", ego_rec.output_text);
    assert!(
        ego_rec.output_text.contains("computeTotal") && ego_rec.output_text.contains("pricing.ts"),
        "expected TypeScript symbol in ego graph output, got: {}",
        ego_rec.output_text
    );

    // 3. Unlock Control Gate -> bi-temporal decision consolidation + superseding over MCP
    orch.toolset().set_writes_enabled(true);

    let (rec1, _) = orch
        .toolset()
        .dispatch(
            "mcp__allyh__record_decision",
            json!({
                "topic": "http-client",
                "rule_statement": "Use hyper directly for HTTP calls",
                "rationale": "Initial rule"
            }),
        )
        .await;
    assert!(rec1.succeeded, "record_decision #1 failed: {}", rec1.output_text);

    // Supersede on the same topic ("http-client") — conflict gate should mark old rule invalid_at
    let (rec2, _) = orch
        .toolset()
        .dispatch(
            "mcp__allyh__record_decision",
            json!({
                "topic": "http-client",
                "rule_statement": "Use reqwest with rustls-tls for all HTTP calls",
                "rationale": "Unified TLS and simpler client ergonomics"
            }),
        )
        .await;
    assert!(rec2.succeeded, "record_decision #2 failed: {}", rec2.output_text);

    let (queried, _) = orch
        .toolset()
        .dispatch(
            "mcp__allyh__query_invariants",
            json!({ "query": "HTTP client library", "top_k": 5 }),
        )
        .await;
    assert!(queried.succeeded);
    assert!(
        queried.output_text.contains("reqwest"),
        "expected active reqwest rule, got: {}",
        queried.output_text
    );
    assert!(
        !queried.output_text.contains("Use hyper directly"),
        "superseded hyper rule must not appear as active: {}",
        queried.output_text
    );

    // 4. Skills & Tasks round-trip over MCP
    let (added_skill, _) = orch
        .toolset()
        .dispatch(
            "mcp__allyh__add_skill",
            json!({
                "name": "ts-pricing-check",
                "description": "Validate pricing calculations",
                "instructions": "Run unit tests on src/pricing.ts before merging.",
                "triggers": ["pricing", "checkout"]
            }),
        )
        .await;
    assert!(added_skill.succeeded, "add_skill failed: {}", added_skill.output_text);

    let (listed_skills, _) = orch
        .toolset()
        .dispatch("mcp__allyh__list_skills", json!({}))
        .await;
    assert!(listed_skills.succeeded);
    assert!(listed_skills.output_text.contains("ts-pricing-check"));

    let (fetched_skill, _) = orch
        .toolset()
        .dispatch(
            "mcp__allyh__get_skill",
            json!({ "name": "ts-pricing-check" }),
        )
        .await;
    assert!(fetched_skill.succeeded);
    assert!(fetched_skill.output_text.contains("src/pricing.ts"));

    let (tasks, _) = orch
        .toolset()
        .dispatch("mcp__allyh__list_tasks", json!({}))
        .await;
    assert!(tasks.succeeded);

    // 5. Verify out-of-tree RepoProfile MCP server config persistence in LanceDB
    let mut profile = RepoProfile::empty(orch.repo_id(), orch.scope_id());
    profile.mcp_servers.push(mcp_cfg.clone());
    orch.store()
        .upsert_repo_profile(&profile)
        .await
        .expect("upsert repo profile with mcp_servers");
    let reloaded = orch
        .store()
        .get_repo_profile(orch.repo_id(), orch.scope_id())
        .await
        .expect("reload repo profile")
        .expect("profile should exist");
    assert_eq!(reloaded.mcp_servers.len(), 1);
    assert_eq!(reloaded.mcp_servers[0].name, "allyh");

    // Cleanly shut down spawned stdio child process
    orch.toolset().mcp_clients().shutdown_all().await;
}

/// End-to-end test for Streamable HTTP + `text/event-stream` SSE transport + Control Gate.
async fn test_streamable_http_and_sse_transport() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind local mock MCP HTTP server");
    let addr = listener.local_addr().expect("local_addr");

    let server_task = tokio::spawn(async move {
        // Serve up to 8 HTTP requests cleanly.
        for _ in 0..8 {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            let mut buf = vec![0u8; 16_384];
            let Ok(n) = socket.read(&mut buf).await else {
                continue;
            };
            if n == 0 {
                continue;
            }
            let raw = String::from_utf8_lossy(&buf[..n]);
            let body = raw.split("\r\n\r\n").nth(1).unwrap_or("").trim();
            let req: Value = match serde_json::from_str(body) {
                Ok(v) => v,
                Err(_) => {
                    let resp = "HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                    let _ = socket.write_all(resp.as_bytes()).await;
                    continue;
                }
            };
            let method = req.get("method").and_then(|v| v.as_str()).unwrap_or("");
            let id = req.get("id").cloned();

            // Notification (no id) -> 202 Accepted
            if id.is_none() {
                let resp = "HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                let _ = socket.write_all(resp.as_bytes()).await;
                continue;
            }
            let id_val = id.unwrap();

            match method {
                "initialize" => {
                    let payload = json!({
                        "jsonrpc": "2.0",
                        "id": id_val,
                        "result": {
                            "protocolVersion": "2025-03-26",
                            "capabilities": { "tools": {} },
                            "serverInfo": { "name": "mock-http-mcp", "version": "1.0.0" }
                        }
                    })
                    .to_string();
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nMcp-Session-Id: sess-42\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        payload.len(),
                        payload
                    );
                    let _ = socket.write_all(resp.as_bytes()).await;
                }
                "tools/list" => {
                    let payload = json!({
                        "jsonrpc": "2.0",
                        "id": id_val,
                        "result": {
                            "tools": [
                                {
                                    "name": "search_docs",
                                    "description": "Search remote documentation",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": { "q": { "type": "string" } }
                                    },
                                    "annotations": { "readOnlyHint": true, "destructiveHint": false }
                                },
                                {
                                    "name": "delete_index",
                                    "description": "Mutating/unannotated remote tool",
                                    "inputSchema": { "type": "object", "properties": {} }
                                }
                            ]
                        }
                    })
                    .to_string();
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        payload.len(),
                        payload
                    );
                    let _ = socket.write_all(resp.as_bytes()).await;
                }
                "tools/call" => {
                    // Return an SSE (`text/event-stream`) stream to verify SSE response parsing!
                    let tool_name = req["params"]["name"].as_str().unwrap_or("");
                    let q = req["params"]["arguments"]["q"].as_str().unwrap_or("");
                    let rpc = json!({
                        "jsonrpc": "2.0",
                        "id": id_val,
                        "result": {
                            "content": [{ "type": "text", "text": format!("SSE ok from {tool_name}: {q}") }],
                            "isError": false
                        }
                    })
                    .to_string();
                    let sse = format!("event: message\ndata: {rpc}\n\n");
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        sse.len(),
                        sse
                    );
                    let _ = socket.write_all(resp.as_bytes()).await;
                }
                _ => {
                    let resp = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                    let _ = socket.write_all(resp.as_bytes()).await;
                }
            }
        }
    });

    let mgr = McpClientManager::new();
    let cfg = McpServerConfig {
        name: "remote_docs".to_string(),
        transport: McpServerTransport::Http {
            url: format!("http://{addr}/mcp"),
            headers: vec![("X-Test-Client".to_string(), "allyh".to_string())],
        },
        enabled: true,
    };

    let status = mgr
        .connect_server(cfg)
        .await
        .expect("HTTP MCP server should connect");
    assert!(
        status.connected,
        "HTTP MCP server should connect, error: {:?}",
        status.last_error
    );
    assert_eq!(status.tool_count, 2);

    // Read-only HTTP/SSE tool works with writes_enabled = false
    let (_, _, text, ok) = mgr
        .call_tool("mcp__remote_docs__search_docs", json!({ "q": "lancedb" }), false)
        .await
        .expect("SSE tools/call should succeed");
    assert!(ok);
    assert_eq!(text, "SSE ok from search_docs: lancedb");

    // Unannotated tool (`delete_index`) is conservatively write-gated!
    let blocked = mgr
        .call_tool("mcp__remote_docs__delete_index", json!({}), false)
        .await;
    assert!(
        matches!(blocked, Err(ToolError::WriteGateLocked(_))),
        "unannotated HTTP MCP tool must be blocked when writes_enabled=false, got {blocked:?}"
    );

    // Unlocking writes allows `delete_index`
    let (_, _, text2, ok2) = mgr
        .call_tool("mcp__remote_docs__delete_index", json!({}), true)
        .await
        .expect("delete_index should succeed when writes_enabled=true");
    assert!(ok2);
    assert!(text2.contains("delete_index"));

    mgr.shutdown_all().await;
    server_task.abort();
}
