use std::path::Path;

use allyh_env::LspManager;
use anyhow::Result;

use crate::cli::{LspArgs, LspCommand};

/// Execute `allyh lsp <subcommand>` to inspect auto-discovered LSP servers for a repository.
pub async fn run_cli(args: LspArgs, repo: &Path) -> Result<()> {
    let manager = LspManager::discover_default(repo);

    match args.action {
        LspCommand::Status { json } => {
            let statuses = manager.status().await;
            if json {
                println!("{}", serde_json::to_string_pretty(&statuses)?);
                return Ok(());
            }
            if statuses.is_empty() {
                println!(
                    "No supported LSP binaries detected on PATH (rust-analyzer, vtsls/typescript-language-server, basedpyright/pyright/pylsp, gopls, clangd)."
                );
                println!("AllyH will use tree-sitter RepoGraph + CLI verifiers automatically.");
            } else {
                println!("Discovered LSP Servers ({}):", statuses.len());
                for s in statuses {
                    println!(
                        "  - {:<12} {:<26} [{}] ({})",
                        s.family,
                        s.server_name,
                        s.state.as_str(),
                        s.command
                    );
                }
            }
        }
    }

    Ok(())
}
