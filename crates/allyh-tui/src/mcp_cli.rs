use std::path::Path;
use std::sync::Arc;

use allyh_core::{AllyHMcpServer, McpServerLaunchConfig};
use allyh_env::GitTracker;
use allyh_memory::{
    DeterministicHashEmbedder, EmbeddingProfile, McpServerConfig, McpServerTransport, MemoryStore,
    RepoProfile, StorageConfig, compute_repo_id,
};
use anyhow::{Context, Result, bail};

use crate::cli::{McpArgs, McpCommand};

/// Execute `allyh mcp <serve|status|add|remove>`.
pub async fn run_cli(
    args: McpArgs,
    repo: &Path,
    db_uri: Option<&str>,
    scope: &str,
    embedding_model: Option<&str>,
) -> Result<()> {
    let storage = match db_uri {
        Some(uri) => StorageConfig::from_uri(uri.to_string()),
        None => StorageConfig::default_local(),
    };

    match args.action {
        McpCommand::Serve => {
            let embedding_profile = match embedding_model {
                Some(spec) => {
                    EmbeddingProfile::parse_spec(spec).context("invalid --embedding-model spec")?
                }
                None => EmbeddingProfile::from_env(),
            };
            let embedder = embedding_profile.build_provider();
            let server = AllyHMcpServer::initialize(
                McpServerLaunchConfig {
                    repo_root: repo.to_path_buf(),
                    storage,
                    scope_id: scope.to_string(),
                },
                Some(embedder),
            )
            .await
            .context("failed to initialize AllyH MCP server")?;
            server
                .serve_stdio()
                .await
                .context("AllyH MCP stdio server error")
        }
        McpCommand::Status { json } => {
            let store = MemoryStore::connect(storage, Arc::new(DeterministicHashEmbedder))
                .await
                .context("failed to connect to LanceDB")?;
            let git = GitTracker::new(repo);
            let repo_id = compute_repo_id(&git.canonical_repo_identifier());
            let profile = store.get_repo_profile(&repo_id, scope).await?;
            let servers = profile
                .map(|p| p.mcp_servers)
                .unwrap_or_default();

            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "repo_id": repo_id,
                        "scope_id": scope,
                        "mcp_servers": servers,
                    }))?
                );
                return Ok(());
            }

            println!(
                "Repository Partition: {} | Scope: {scope}",
                &repo_id[..repo_id.len().min(12)]
            );
            if servers.is_empty() {
                println!("Configured MCP Servers: none");
                println!(
                    "Tip: register one with `allyh mcp add <name> --command <cmd> [--arg <arg>...]` or `allyh mcp add <name> --url <http-url>`."
                );
            } else {
                println!("Configured MCP Servers ({}):", servers.len());
                for srv in &servers {
                    let state = if srv.enabled { "enabled" } else { "disabled" };
                    match &srv.transport {
                        McpServerTransport::Stdio { command, args, .. } => {
                            let target = if args.is_empty() {
                                command.clone()
                            } else {
                                format!("{command} {}", args.join(" "))
                            };
                            println!("  - {} [stdio, {state}]: {target}", srv.name);
                        }
                        McpServerTransport::Http { url, .. } => {
                            println!("  - {} [http, {state}]: {url}", srv.name);
                        }
                    }
                }
            }
            Ok(())
        }
        McpCommand::Add {
            name,
            url,
            command,
            args: cmd_args,
            header,
        } => {
            let trimmed_name = name.trim();
            if trimmed_name.is_empty() {
                bail!("MCP server name cannot be empty");
            }
            let transport = match (url, command) {
                (Some(u), None) => {
                    let mut headers = Vec::new();
                    for raw_h in header {
                        let Some((k, v)) = raw_h.split_once(':') else {
                            bail!("invalid --header '{raw_h}' (expected `Key: Value`)");
                        };
                        headers.push((k.trim().to_string(), v.trim().to_string()));
                    }
                    McpServerTransport::Http {
                        url: u.trim().to_string(),
                        headers,
                    }
                }
                (None, Some(cmd)) => McpServerTransport::Stdio {
                    command: cmd.trim().to_string(),
                    args: cmd_args,
                    env: Vec::new(),
                },
                (Some(_), Some(_)) => {
                    bail!("specify either `--url` or `--command`, not both")
                }
                (None, None) => {
                    bail!("specify either `--url <http-url>` or `--command <executable>`")
                }
            };

            let store = MemoryStore::connect(storage, Arc::new(DeterministicHashEmbedder))
                .await
                .context("failed to connect to LanceDB")?;
            let git = GitTracker::new(repo);
            let repo_id = compute_repo_id(&git.canonical_repo_identifier());

            let mut profile = store
                .get_repo_profile(&repo_id, scope)
                .await?
                .unwrap_or_else(|| RepoProfile::empty(&repo_id, scope));
            profile.mcp_servers.retain(|s| s.name != trimmed_name);
            profile.mcp_servers.push(McpServerConfig {
                name: trimmed_name.to_string(),
                transport,
                enabled: true,
            });
            store.upsert_repo_profile(&profile).await?;
            println!(
                "Registered MCP server `{trimmed_name}` in out-of-tree LanceDB RepoProfile (scope: {scope})."
            );
            Ok(())
        }
        McpCommand::Remove { name } => {
            let store = MemoryStore::connect(storage, Arc::new(DeterministicHashEmbedder))
                .await
                .context("failed to connect to LanceDB")?;
            let git = GitTracker::new(repo);
            let repo_id = compute_repo_id(&git.canonical_repo_identifier());

            let Some(mut profile) = store.get_repo_profile(&repo_id, scope).await? else {
                println!("No MCP server named `{name}` found for scope `{scope}`.");
                return Ok(());
            };
            let before = profile.mcp_servers.len();
            profile.mcp_servers.retain(|s| s.name != name);
            if profile.mcp_servers.len() == before {
                println!("No MCP server named `{name}` found for scope `{scope}`.");
            } else {
                store.upsert_repo_profile(&profile).await?;
                println!("Removed MCP server `{name}` from LanceDB RepoProfile (scope: {scope}).");
            }
            Ok(())
        }
    }
}
