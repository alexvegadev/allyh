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
    #[arg(short, long, global = true, default_value = ".")]
    pub repo: PathBuf,

    /// Out-of-tree LanceDB storage URI (local path or `s3://...` / `gs://...` / `az://...`).
    #[arg(long, global = true, env = "ALLYH_DB_URI")]
    pub db_uri: Option<String>,

    /// Partition scope identifier for multi-agent isolation or shared team state.
    #[arg(long, global = true, env = "ALLYH_SCOPE_ID", default_value = "default")]
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
    /// Manage reusable procedural skills stored in LanceDB for this repository.
    Skills(SkillsArgs),
}

/// Options for `allyh skills`.
#[derive(Debug, Clone, Args)]
pub struct SkillsArgs {
    #[command(subcommand)]
    pub action: SkillsCommand,
}

/// Subcommands under `allyh skills`.
#[derive(Debug, Clone, Subcommand)]
pub enum SkillsCommand {
    /// List all registered skills for the repository partition.
    #[command(visible_alias = "ls")]
    List {
        /// Output machine-readable JSON.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Show the full metadata and instructions of a registered skill.
    Show {
        /// Skill name, normalized slug, or full id.
        name: String,
        /// Output machine-readable JSON.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Register or update a skill directly from the command line.
    Add {
        /// Short, unique skill name (e.g. `add-lancedb-table`).
        name: String,
        /// One-line summary of what the skill accomplishes.
        #[arg(short, long, default_value = "")]
        description: String,
        /// Inline instructions text (mutually exclusive with `--file`).
        #[arg(short, long, conflicts_with = "file")]
        instructions: Option<String>,
        /// Read instructions body from a file (mutually exclusive with `--instructions`).
        #[arg(short, long, conflicts_with = "instructions")]
        file: Option<PathBuf>,
        /// Trigger keywords or phrases (repeatable).
        #[arg(short = 't', long = "trigger")]
        triggers: Vec<String>,
    },
    /// Import a `SKILL.md` file (YAML frontmatter + Markdown body) into LanceDB.
    Import {
        /// Path to the `SKILL.md` file to import.
        path: PathBuf,
        /// Override the skill name parsed from the file.
        #[arg(long)]
        name: Option<String>,
    },
    /// Export a stored skill as a portable `SKILL.md` document.
    Export {
        /// Skill name, normalized slug, or full id.
        name: String,
        /// Destination file path (prints to stdout when omitted).
        #[arg(short, long)]
        out: Option<PathBuf>,
    },
    /// Remove a skill from LanceDB.
    #[command(visible_aliases = ["rm", "delete"])]
    Remove {
        /// Skill name, normalized slug, or full id.
        name: String,
    },
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

    #[test]
    fn parses_skills_subcommands_with_global_flags() {
        let list_args = CliArgs::try_parse_from([
            "allyh",
            "skills",
            "list",
            "--json",
            "--repo",
            "workspace",
            "--scope",
            "team-a",
        ])
        .expect("parse skills list");
        assert_eq!(list_args.repo, PathBuf::from("workspace"));
        assert_eq!(list_args.scope, "team-a");
        assert!(matches!(
            list_args.command,
            Some(CliCommand::Skills(SkillsArgs {
                action: SkillsCommand::List { json: true }
            }))
        ));

        let add_args = CliArgs::try_parse_from([
            "allyh",
            "skills",
            "add",
            "release-flow",
            "--description",
            "Release checklist",
            "--instructions",
            "Run tests then tag.",
            "--trigger",
            "release",
            "--trigger",
            "publish",
        ])
        .expect("parse skills add");
        match add_args.command {
            Some(CliCommand::Skills(SkillsArgs {
                action:
                    SkillsCommand::Add {
                        name,
                        description,
                        instructions,
                        file,
                        triggers,
                    },
            })) => {
                assert_eq!(name, "release-flow");
                assert_eq!(description, "Release checklist");
                assert_eq!(instructions.as_deref(), Some("Run tests then tag."));
                assert!(file.is_none());
                assert_eq!(triggers, vec!["release", "publish"]);
            }
            other => panic!("unexpected parsed command: {other:?}"),
        }

        let import_args = CliArgs::try_parse_from([
            "allyh",
            "skills",
            "import",
            "docs/SKILL.md",
            "--name",
            "custom-name",
        ])
        .expect("parse skills import");
        assert!(matches!(
            import_args.command,
            Some(CliCommand::Skills(SkillsArgs {
                action: SkillsCommand::Import { ref path, ref name }
            })) if path == &PathBuf::from("docs/SKILL.md") && name.as_deref() == Some("custom-name")
        ));

        let remove_args =
            CliArgs::try_parse_from(["allyh", "skills", "rm", "release-flow"]).expect("parse rm");
        assert!(matches!(
            remove_args.command,
            Some(CliCommand::Skills(SkillsArgs {
                action: SkillsCommand::Remove { ref name }
            })) if name == "release-flow"
        ));
    }
}
