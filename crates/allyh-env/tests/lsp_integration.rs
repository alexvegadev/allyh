use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Write};
use std::sync::Arc;
use std::time::Duration;

use allyh_env::{
    EnvironmentMonitor, LspManager, LspServerSpec, LspServerState, encode_lsp_frame,
    file_uri_to_path, file_uri_to_rel_path, path_to_file_uri,
};
use allyh_memory::{
    DEFAULT_SCOPE_ID, DeterministicHashEmbedder, MemoryStore, StorageConfig, compute_repo_id,
};
use serde_json::{Value, json};
use tempfile::TempDir;

const MOCK_ENV_KEY: &str = "ALLYH_MOCK_LSP_MODE";

fn main() {
    if std::env::var(MOCK_ENV_KEY).as_deref() == Ok("1") {
        run_mock_lsp_server();
        return;
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");

    rt.block_on(async {
        test_uri_roundtrip();
        test_semantic_repograph_disambiguation_and_diagnostics().await;
        test_lsp_lifecycle_restart_idle_and_graceful_fallback().await;
    });
    println!("lsp_integration: all 3 tests passed");
}

fn test_uri_roundtrip() {
    let tmp = TempDir::new().expect("tmp");
    let nested = tmp.path().join("my folder").join("src").join("alpha.ts");
    let uri = path_to_file_uri(&nested);
    assert!(uri.starts_with("file://"));
    assert!(uri.contains("my%20folder"));
    let back = file_uri_to_path(&uri).expect("decode uri");
    assert!(back.to_string_lossy().contains("my folder"));
    let rel = file_uri_to_rel_path(&uri, tmp.path()).expect("rel path");
    assert_eq!(rel, "my folder/src/alpha.ts");
}

async fn test_semantic_repograph_disambiguation_and_diagnostics() {
    let tmp = TempDir::new().expect("tmp");
    let repo_dir = tmp.path().join("repo");
    let db_dir = tmp.path().join("db");
    std::fs::create_dir_all(repo_dir.join("src")).expect("mkdir");

    // Two modules both define `parse` -> pure syntactic tree-sitter collides by name!
    let alpha_path = repo_dir.join("src/alpha.ts");
    let beta_path = repo_dir.join("src/beta.ts");
    let consumer_alpha_path = repo_dir.join("src/consumer_alpha.ts");
    let consumer_beta_path = repo_dir.join("src/consumer_beta.ts");

    std::fs::write(
        &alpha_path,
        "export function parse(input: string): number {\n    return input.length;\n}\n",
    )
    .unwrap();
    std::fs::write(
        &beta_path,
        "export function parse(raw: number): number {\n    return raw + 1;\n}\n",
    )
    .unwrap();
    std::fs::write(
        &consumer_alpha_path,
        "import { parse } from './alpha';\nexport function runAlpha() {\n    return parse('ok');\n}\n",
    )
    .unwrap();
    std::fs::write(
        &consumer_beta_path,
        "import { parse } from './beta';\nexport function runBeta() {\n    return parse(42);\n}\n",
    )
    .unwrap();

    let store = MemoryStore::connect(
        StorageConfig::from_local_path(&db_dir),
        Arc::new(DeterministicHashEmbedder),
    )
    .await
    .expect("connect");

    let exe = std::env::current_exe().expect("current_exe");
    let lsp = LspManager::empty(&repo_dir);
    lsp.register_server(LspServerSpec {
        family: "typescript".to_string(),
        name: "mock-ts-lsp".to_string(),
        command: exe.to_string_lossy().into_owned(),
        args: Vec::new(),
        env: vec![
            (MOCK_ENV_KEY.to_string(), "1".to_string()),
            (
                "ALLYH_MOCK_REF_URI".to_string(),
                path_to_file_uri(&consumer_alpha_path),
            ),
        ],
        require_manifest: false,
    })
    .await;

    let monitor = EnvironmentMonitor::new(
        &repo_dir,
        compute_repo_id(&repo_dir.to_string_lossy()),
        DEFAULT_SCOPE_ID,
        store,
    )
    .with_lsp_manager(lsp);

    let delta = monitor.sync_and_collect_delta(1, false).await.expect("sync");
    assert!(
        delta
            .active_lsp_servers
            .iter()
            .any(|s| s.contains("mock-ts-lsp")),
        "expected mock-ts-lsp badge in EnvDelta: {:?}",
        delta.active_lsp_servers
    );

    // 1. Semantic RepoGraph disambiguation:
    // Querying `src/alpha.ts` should get LSP-resolved `consumer_alpha.ts` and NOT collide with `consumer_beta.ts`.
    let ego = monitor.ego_graph_for("src/alpha.ts", 1).await.expect("ego");
    assert_eq!(ego.semantic_source.as_deref(), Some("lsp:mock-ts-lsp"));
    assert!(
        ego.hop1_neighbors
            .iter()
            .any(|e| e.from_file == "src/consumer_alpha.ts" && e.from_symbol == "runAlpha" && e.relation.starts_with("lsp_")),
        "expected LSP semantic edge from consumer_alpha.ts, got: {:?}",
        ego.hop1_neighbors
    );
    assert!(
        !ego.hop1_neighbors
            .iter()
            .any(|e| e.from_file == "src/consumer_beta.ts"),
        "LSP disambiguation must eliminate unrelated consumer_beta.ts name collision: {:?}",
        ego.hop1_neighbors
    );

    // 2. Sub-second post-edit LSP diagnostics (error case -> clean fix case)
    let broken = "export function parse(input: string): number {\n    return BROKEN_TYPE;\n}\n";
    std::fs::write(&alpha_path, broken).unwrap();
    let broken_hash = EnvironmentMonitor::sha256_str(broken);
    let report = monitor
        .lsp_diagnostics_for_edit("src/alpha.ts", &alpha_path, broken, &broken_hash)
        .await
        .expect("LSP should return diagnostic report");
    assert_eq!(report.toolchain, "lsp:mock-ts-lsp");
    assert!(!report.success);
    assert_eq!(report.error_count, 1);
    assert_eq!(report.diagnostics[0].code.as_deref(), Some("TS2322"));
    assert_eq!(report.diagnostics[0].line, Some(2));
    assert!(report.diagnostics[0].message.contains("BROKEN_TYPE"));

    let fixed = "export function parse(input: string): number {\n    return input.length;\n}\n";
    std::fs::write(&alpha_path, fixed).unwrap();
    let fixed_hash = EnvironmentMonitor::sha256_str(fixed);
    let clean_report = monitor
        .lsp_diagnostics_for_edit("src/alpha.ts", &alpha_path, fixed, &fixed_hash)
        .await
        .expect("LSP should return clean report");
    assert!(clean_report.success);
    assert_eq!(clean_report.error_count, 0);

    monitor.shutdown_lsp().await;
}

async fn test_lsp_lifecycle_restart_idle_and_graceful_fallback() {
    let tmp = TempDir::new().expect("tmp");
    let repo_dir = tmp.path().join("repo");
    let db_dir = tmp.path().join("db");
    std::fs::create_dir_all(repo_dir.join("src")).unwrap();
    let py_file = repo_dir.join("src/main.py");
    std::fs::write(&py_file, "def compute():\n    return 1\n").unwrap();

    let store = MemoryStore::connect(
        StorageConfig::from_local_path(&db_dir),
        Arc::new(DeterministicHashEmbedder),
    )
    .await
    .expect("connect");

    // 1. Graceful fallback when no LSP is configured
    let empty_lsp = LspManager::empty(&repo_dir);
    let fallback_monitor = EnvironmentMonitor::new(
        &repo_dir,
        compute_repo_id(&repo_dir.to_string_lossy()),
        DEFAULT_SCOPE_ID,
        store.clone(),
    )
    .with_lsp_manager(empty_lsp);
    fallback_monitor
        .sync_and_collect_delta(1, false)
        .await
        .expect("sync");
    let ego = fallback_monitor
        .ego_graph_for("compute", 1)
        .await
        .expect("ego");
    assert!(ego.semantic_source.is_none());
    assert!(
        fallback_monitor
            .lsp_diagnostics_for_edit("src/main.py", &py_file, "def compute(): return 1\n", "abc")
            .await
            .is_none()
    );

    // 2. Restart + idle shutdown lifecycle
    let exe = std::env::current_exe().expect("current_exe");
    let lsp = LspManager::empty(&repo_dir)
        .with_timeouts(Duration::from_secs(5), Duration::from_millis(50));
    lsp.register_server(LspServerSpec {
        family: "python".to_string(),
        name: "mock-py-lsp".to_string(),
        command: exe.to_string_lossy().into_owned(),
        args: Vec::new(),
        env: vec![(MOCK_ENV_KEY.to_string(), "1".to_string())],
        require_manifest: false,
    })
    .await;

    let before = lsp.status().await;
    assert_eq!(before[0].state, LspServerState::Ready);

    let client = lsp.get_or_start("python").await.expect("start mock py lsp");
    assert!(client.is_alive());
    assert_eq!(lsp.status().await[0].state, LspServerState::Running);

    let restarted = lsp.restart(Some("python")).await.expect("restart");
    assert_eq!(restarted[0].state, LspServerState::Running);
    assert!(restarted[0].restart_count >= 1);

    // Wait past 50ms idle timeout -> shutdown_idle reaps the idle server cleanly
    tokio::time::sleep(Duration::from_millis(80)).await;
    let reaped = lsp.shutdown_idle().await;
    assert_eq!(reaped, 1);
    assert_eq!(lsp.status().await[0].state, LspServerState::Ready);
}

// ---------------------------------------------------------------------------
// Minimal Pure-Rust Stdio JSON-RPC 2.0 Mock LSP Server
// ---------------------------------------------------------------------------

fn run_mock_lsp_server() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = stdout.lock();
    let mut docs: HashMap<String, String> = HashMap::new();
    let target_ref_uri = std::env::var("ALLYH_MOCK_REF_URI").unwrap_or_default();

    while let Some(msg) = read_sync_frame(&mut reader) {
        let Some(method) = msg.get("method").and_then(|v| v.as_str()) else {
            continue;
        };
        let id = msg.get("id").cloned();

        match method {
            "initialize" => {
                // Exercise server-to-client request handling (`workspace/configuration`).
                let cfg_req = json!({
                    "jsonrpc": "2.0",
                    "id": 999,
                    "method": "workspace/configuration",
                    "params": { "items": [{ "section": "typescript" }] }
                });
                write_sync_frame(&mut writer, &cfg_req);

                if let Some(id) = id {
                    let resp = json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "capabilities": {
                                "textDocumentSync": 1,
                                "referencesProvider": true,
                                "definitionProvider": true,
                                "implementationProvider": true,
                                "callHierarchyProvider": true
                            }
                        }
                    });
                    write_sync_frame(&mut writer, &resp);
                }
            }
            "textDocument/didOpen" => {
                if let Some(td) = msg.get("params").and_then(|p| p.get("textDocument"))
                    && let (Some(uri), Some(text)) = (
                        td.get("uri").and_then(|v| v.as_str()),
                        td.get("text").and_then(|v| v.as_str()),
                    )
                {
                    docs.insert(uri.to_string(), text.to_string());
                    publish_mock_diagnostics(&mut writer, uri, text);
                }
            }
            "textDocument/didChange" => {
                if let Some(params) = msg.get("params")
                    && let Some(uri) = params
                        .get("textDocument")
                        .and_then(|t| t.get("uri"))
                        .and_then(|v| v.as_str())
                    && let Some(text) = params
                        .get("contentChanges")
                        .and_then(|c| c.as_array())
                        .and_then(|a| a.first())
                        .and_then(|c| c.get("text"))
                        .and_then(|v| v.as_str())
                {
                    docs.insert(uri.to_string(), text.to_string());
                    publish_mock_diagnostics(&mut writer, uri, text);
                }
            }
            "textDocument/didSave" => {
                if let Some(params) = msg.get("params")
                    && let Some(uri) = params
                        .get("textDocument")
                        .and_then(|t| t.get("uri"))
                        .and_then(|v| v.as_str())
                {
                    let text = params
                        .get("text")
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                        .or_else(|| docs.get(uri).cloned())
                        .unwrap_or_default();
                    publish_mock_diagnostics(&mut writer, uri, &text);
                }
            }
            "textDocument/references" => {
                if let Some(id) = id {
                    let result = if !target_ref_uri.is_empty() {
                        json!([{
                            "uri": target_ref_uri,
                            "range": {
                                "start": { "line": 2, "character": 11 },
                                "end": { "line": 2, "character": 16 }
                            }
                        }])
                    } else {
                        json!([])
                    };
                    write_sync_frame(
                        &mut writer,
                        &json!({ "jsonrpc": "2.0", "id": id, "result": result }),
                    );
                }
            }
            "textDocument/implementation" | "textDocument/definition" => {
                if let Some(id) = id {
                    write_sync_frame(
                        &mut writer,
                        &json!({ "jsonrpc": "2.0", "id": id, "result": [] }),
                    );
                }
            }
            "textDocument/prepareCallHierarchy" => {
                if let Some(id) = id {
                    let uri = msg
                        .get("params")
                        .and_then(|p| p.get("textDocument"))
                        .and_then(|t| t.get("uri"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let item = json!([{
                        "name": "parse",
                        "kind": 12,
                        "uri": uri,
                        "range": { "start": { "line": 0, "character": 0 }, "end": { "line": 2, "character": 1 } },
                        "selectionRange": { "start": { "line": 0, "character": 16 }, "end": { "line": 0, "character": 21 } }
                    }]);
                    write_sync_frame(
                        &mut writer,
                        &json!({ "jsonrpc": "2.0", "id": id, "result": item }),
                    );
                }
            }
            "callHierarchy/incomingCalls" => {
                if let Some(id) = id {
                    let result = if !target_ref_uri.is_empty() {
                        json!([{
                            "from": {
                                "name": "runAlpha",
                                "kind": 12,
                                "uri": target_ref_uri,
                                "range": { "start": { "line": 1, "character": 0 }, "end": { "line": 3, "character": 1 } },
                                "selectionRange": { "start": { "line": 1, "character": 16 }, "end": { "line": 1, "character": 24 } }
                            },
                            "fromRanges": []
                        }])
                    } else {
                        json!([])
                    };
                    write_sync_frame(
                        &mut writer,
                        &json!({ "jsonrpc": "2.0", "id": id, "result": result }),
                    );
                }
            }
            "callHierarchy/outgoingCalls" => {
                if let Some(id) = id {
                    write_sync_frame(
                        &mut writer,
                        &json!({ "jsonrpc": "2.0", "id": id, "result": [] }),
                    );
                }
            }
            "shutdown" => {
                if let Some(id) = id {
                    write_sync_frame(
                        &mut writer,
                        &json!({ "jsonrpc": "2.0", "id": id, "result": null }),
                    );
                }
            }
            "exit" => break,
            _ => {
                if let Some(id) = id {
                    write_sync_frame(
                        &mut writer,
                        &json!({ "jsonrpc": "2.0", "id": id, "result": null }),
                    );
                }
            }
        }
    }
}

fn publish_mock_diagnostics<W: Write>(writer: &mut W, uri: &str, text: &str) {
    let mut diags = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        if let Some(col) = line.find("BROKEN_TYPE") {
            diags.push(json!({
                "range": {
                    "start": { "line": idx, "character": col },
                    "end": { "line": idx, "character": col + 11 }
                },
                "severity": 1,
                "code": "TS2322",
                "source": "mock-ts-lsp",
                "message": "Type 'string' is not assignable to type 'number' (BROKEN_TYPE)"
            }));
        }
    }
    let notif = json!({
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": {
            "uri": uri,
            "diagnostics": diags
        }
    });
    write_sync_frame(writer, &notif);
}

fn read_sync_frame<R: BufRead>(reader: &mut R) -> Option<Value> {
    let mut content_length: Option<usize> = None;
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).ok()?;
        if n == 0 {
            return None;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if content_length.is_some() {
                break;
            }
            continue;
        }
        if let Some((k, v)) = trimmed.split_once(':')
            && k.trim().eq_ignore_ascii_case("content-length")
            && let Ok(len) = v.trim().parse::<usize>()
        {
            content_length = Some(len);
        }
    }
    let len = content_length?;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).ok()?;
    serde_json::from_slice(&buf).ok()
}

fn write_sync_frame<W: Write>(writer: &mut W, payload: &Value) {
    let frame = encode_lsp_frame(payload);
    let _ = writer.write_all(&frame);
    let _ = writer.flush();
}
