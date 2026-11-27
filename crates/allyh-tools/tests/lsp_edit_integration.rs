use std::io::{self, BufRead, BufReader, Write};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use allyh_env::{EnvironmentMonitor, LspManager, LspServerSpec, encode_lsp_frame};
use allyh_memory::{
    DEFAULT_SCOPE_ID, DeterministicHashEmbedder, MemoryStore, StorageConfig, compute_repo_id,
};
use allyh_tools::{AciToolset, EditByHashArgs, InspectSymbolArgs};
use serde_json::{Value, json};
use tempfile::TempDir;

const MOCK_ENV_KEY: &str = "ALLYH_TOOLS_MOCK_LSP";

fn main() {
    if std::env::var(MOCK_ENV_KEY).as_deref() == Ok("1") {
        run_pull_diag_mock_server();
        return;
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio rt");
    rt.block_on(test_edit_by_hash_attaches_instant_lsp_diagnostics());
    println!("lsp_edit_integration: passed");
}

async fn test_edit_by_hash_attaches_instant_lsp_diagnostics() {
    let tmp = TempDir::new().expect("tmp");
    let repo_dir = tmp.path().join("repo");
    let db_dir = tmp.path().join("db");
    std::fs::create_dir_all(repo_dir.join("src")).unwrap();
    std::fs::write(
        repo_dir.join("src/app.py"),
        "def greet(name: str) -> str:\n    return name\n",
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
        family: "python".to_string(),
        name: "mock-pyright".to_string(),
        command: exe.to_string_lossy().into_owned(),
        args: Vec::new(),
        env: vec![(MOCK_ENV_KEY.to_string(), "1".to_string())],
        require_manifest: false,
    })
    .await;

    let env = Arc::new(
        EnvironmentMonitor::new(
            &repo_dir,
            compute_repo_id(&repo_dir.to_string_lossy()),
            DEFAULT_SCOPE_ID,
            store.clone(),
        )
        .with_lsp_manager(lsp),
    );
    env.sync_and_collect_delta(1, false).await.expect("sync");

    let toolset = AciToolset::new(Arc::clone(&env), store, Arc::new(AtomicBool::new(true)), true);

    let inspected = toolset
        .inspect_tool()
        .execute(InspectSymbolArgs {
            query: "greet".into(),
            hops: Some(1),
            start_line: None,
            end_line: None,
        })
        .await
        .expect("inspect");
    let hash1 = inspected.matched_files[0].sha256_hash.clone();

    // Introduce an invalid type -> EditByHashTool attaches LSP pull diagnostics in milliseconds.
    let edit_fail = toolset
        .edit_tool()
        .execute(EditByHashArgs {
            file_path: "src/app.py".into(),
            expected_sha256: Some(hash1),
            target_snippet: Some("return name".into()),
            replacement_snippet: Some("return TYPE_ERR".into()),
            full_content: None,
            create_if_missing: None,
            target_scope: None,
            toolchain: None,
        })
        .await
        .expect("edit fail");
    let rep = edit_fail
        .verification
        .as_ref()
        .expect("LSP report must be attached");
    assert_eq!(rep.toolchain, "lsp:mock-pyright");
    assert!(!rep.success);
    assert_eq!(rep.error_count, 1);
    assert!(edit_fail.prescriptive_diagnostics.contains("reportReturnType"));

    // Fix the error -> LSP report flips to PASS.
    let edit_ok = toolset
        .edit_tool()
        .execute(EditByHashArgs {
            file_path: "src/app.py".into(),
            expected_sha256: Some(edit_fail.new_sha256),
            target_snippet: Some("return TYPE_ERR".into()),
            replacement_snippet: Some("return name.upper()".into()),
            full_content: None,
            create_if_missing: None,
            target_scope: None,
            toolchain: None,
        })
        .await
        .expect("edit ok");
    let rep_ok = edit_ok
        .verification
        .as_ref()
        .expect("LSP clean report must be attached");
    assert_eq!(rep_ok.toolchain, "lsp:mock-pyright");
    assert!(rep_ok.success);
    assert_eq!(rep_ok.error_count, 0);

    env.shutdown_lsp().await;
}

/// Minimal mock LSP server advertising LSP 3.17 pull diagnostics (`diagnosticProvider: true`).
fn run_pull_diag_mock_server() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = stdout.lock();
    let mut last_text = String::new();

    while let Some(msg) = read_frame(&mut reader) {
        let Some(method) = msg.get("method").and_then(|v| v.as_str()) else {
            continue;
        };
        let id = msg.get("id").cloned();
        match method {
            "initialize" => {
                if let Some(id) = id {
                    write_frame(
                        &mut writer,
                        &json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "capabilities": {
                                    "textDocumentSync": 1,
                                    "diagnosticProvider": { "interFileDependencies": false, "workspaceDiagnostics": false }
                                }
                            }
                        }),
                    );
                }
            }
            "textDocument/didOpen" => {
                if let Some(text) = msg
                    .get("params")
                    .and_then(|p| p.get("textDocument"))
                    .and_then(|t| t.get("text"))
                    .and_then(|v| v.as_str())
                {
                    last_text = text.to_string();
                }
            }
            "textDocument/didChange" => {
                if let Some(text) = msg
                    .get("params")
                    .and_then(|p| p.get("contentChanges"))
                    .and_then(|c| c.as_array())
                    .and_then(|a| a.first())
                    .and_then(|c| c.get("text"))
                    .and_then(|v| v.as_str())
                {
                    last_text = text.to_string();
                }
            }
            "textDocument/diagnostic" => {
                if let Some(id) = id {
                    let mut items = Vec::new();
                    for (idx, line) in last_text.lines().enumerate() {
                        if let Some(col) = line.find("TYPE_ERR") {
                            items.push(json!({
                                "range": {
                                    "start": { "line": idx, "character": col },
                                    "end": { "line": idx, "character": col + 8 }
                                },
                                "severity": 1,
                                "code": "reportReturnType",
                                "source": "mock-pyright",
                                "message": "Expression of type 'int' cannot be assigned to return type 'str'"
                            }));
                        }
                    }
                    write_frame(
                        &mut writer,
                        &json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": { "kind": "full", "items": items }
                        }),
                    );
                }
            }
            "shutdown" => {
                if let Some(id) = id {
                    write_frame(
                        &mut writer,
                        &json!({ "jsonrpc": "2.0", "id": id, "result": null }),
                    );
                }
            }
            "exit" => break,
            _ => {}
        }
    }
}

fn read_frame<R: BufRead>(reader: &mut R) -> Option<Value> {
    let mut len: Option<usize> = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).ok()? == 0 {
            return None;
        }
        let t = line.trim();
        if t.is_empty() {
            if len.is_some() {
                break;
            }
            continue;
        }
        if let Some((k, v)) = t.split_once(':')
            && k.trim().eq_ignore_ascii_case("content-length")
            && let Ok(n) = v.trim().parse::<usize>()
        {
            len = Some(n);
        }
    }
    let mut buf = vec![0u8; len?];
    reader.read_exact(&mut buf).ok()?;
    serde_json::from_slice(&buf).ok()
}

fn write_frame<W: Write>(writer: &mut W, payload: &Value) {
    let _ = writer.write_all(&encode_lsp_frame(payload));
    let _ = writer.flush();
}
