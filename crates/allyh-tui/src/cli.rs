use std::path::PathBuf;

use clap::Parser;

/// AllyH: Coherence-First AI Agent Harness powered by rig-core and LanceDB.
#[derive(Debug, Clone, Parser)]
#[command(name = "allyh", version, about)]
pub struct CliArgs {
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

    /// Disable automatic post-edit `cargo check --message-format=json`.
    #[arg(long, default_value_t = false)]
    pub no_auto_check: bool,
}
