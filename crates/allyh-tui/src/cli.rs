use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

use crate::update::{DEFAULT_REPO_NAME, DEFAULT_REPO_OWNER};

/// AllyH: Coherence-First AI Agent Harness powered by rig-core and LanceDB.
#[derive(Debug, Clone, Parser)]
#[command(name = "allyh", version, about)]
pub struct CliArgs {
    /// Optional maintenance subcommand (defaults to launching the TUI).
    #[command(subcommand)]
    pub command: Option<CliCommand>,

    /// Target repository directory (defaults to current working directory).
    #[arg(short, long, default_value = ".")]
    pub repo: PathBuf,

    /// Out-of-tree LanceDB storage URI (local path or `s3://...` / `gs://...` / `az://...`).
    #[arg(long, env = "ALLYH_DB_URI")]
    pub db_uri: Option<String>,

    /// Partition scope identifier for multi-agent isolation or shared team state.
    #[arg(long, env = "ALLYH_SCOPE_ID", default_value = "default")]
    pub scope: String,

    /// Primary reasoning model spec (e.g. `anthropic:claude-sonnet-4-6`, `openai:gpt-4o`, `local:qwen-35b@http://localhost:9931/v1`, `offline`).
    #[arg(long, env = "ALLYH_MODEL")]
    pub model: Option<String>,

    /// Background local worker model spec (e.g. `local:qwen2.5-coder-9b@http://localhost:9931/v1`).
    #[arg(long, env = "ALLYH_WORKER_MODEL")]
    pub worker_model: Option<String>,

    /// Embedding model spec (e.g. `local:Qwen3-Embedding-0.6B`, `ollama:nomic-embed-text`, `voyage:voyage-code-3`, `jina:jina-embeddings-v3`, `gemini:gemini-embedding-001`, `openai:text-embedding-3-small`, `offline`).
    #[arg(long, env = "ALLYH_EMBEDDING_MODEL")]
    pub embedding_model: Option<String>,

    /// Disable automatic post-edit `cargo check --message-format=json`.
    #[arg(long, default_value_t = false)]
    pub no_auto_check: bool,
}

/// Top-level CLI subcommands.
#[derive(Debug, Clone, Subcommand)]
pub enum CliCommand {
    /// Check for and install a newer AllyH release from GitHub.
    Update(UpdateArgs),
}

/// Options for `allyh update`.
#[derive(Debug, Clone, Args)]
pub struct UpdateArgs {
    /// Only check whether a newer release is available without installing it.
    #[arg(long, default_value_t = false)]
    pub check: bool,

    /// Skip interactive confirmation prompts before installing an update.
    #[arg(short = 'y', long, default_value_t = false)]
    pub yes: bool,

    /// Install a specific GitHub release tag (e.g. `v0.2.0`) instead of the latest release.
    #[arg(long, value_name = "TAG")]
    pub tag: Option<String>,

    /// GitHub repository owner to fetch releases from.
    #[arg(long, env = "ALLYH_UPDATE_REPO_OWNER", default_value = DEFAULT_REPO_OWNER)]
    pub repo_owner: String,

    /// GitHub repository name to fetch releases from.
    #[arg(long, env = "ALLYH_UPDATE_REPO_NAME", default_value = DEFAULT_REPO_NAME)]
    pub repo_name: String,

    /// Optional GitHub personal access token (falls back to `GH_TOKEN` / `GITHUB_TOKEN`).
    #[arg(long, hide_env_values = true)]
    pub token: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_default_tui_invocation_without_subcommand() {
        let args = CliArgs::try_parse_from([
            "allyh",
            "--repo",
            "workspace",
            "--embedding-model",
            "local:Qwen3-Embedding-0.6B",
            "--no-auto-check",
        ])
        .expect("parse default CLI args");
        assert!(args.command.is_none());
        assert_eq!(args.repo, PathBuf::from("workspace"));
        assert_eq!(
            args.embedding_model.as_deref(),
            Some("local:Qwen3-Embedding-0.6B")
        );
        assert!(args.no_auto_check);
    }

    #[test]
    fn parses_update_subcommand_flags() {
        let args = CliArgs::try_parse_from([
            "allyh",
            "update",
            "--check",
            "--yes",
            "--tag",
            "v0.2.0",
            "--repo-owner",
            "acme",
            "--repo-name",
            "allyh-custom",
        ])
        .expect("parse update subcommand");

        let Some(CliCommand::Update(update)) = args.command else {
            panic!("expected update subcommand");
        };
        assert!(update.check);
        assert!(update.yes);
        assert_eq!(update.tag.as_deref(), Some("v0.2.0"));
        assert_eq!(update.repo_owner, "acme");
        assert_eq!(update.repo_name, "allyh-custom");
    }
}
