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

    /// Disable automatic post-edit project check/type-check verification.
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
    /// Manage Self-Improvement Memory (SIM) and run Continual Recursive Self-Improvement (CRSI).
    Sim(SimArgs),
    /// Manage out-of-tree repository polyglot toolchain and ignore overrides in LanceDB.
    Config(ConfigArgs),
    /// Inspect Language Server Protocol (LSP) semantic bridge availability.
    Lsp(LspArgs),
}

/// Options for `allyh lsp`.
#[derive(Debug, Clone, Args)]
pub struct LspArgs {
    #[command(subcommand)]
    pub action: LspCommand,
}

/// Subcommands under `allyh lsp`.
#[derive(Debug, Clone, Subcommand)]
pub enum LspCommand {
    /// Show auto-discovered LSP servers on PATH for this workspace.
    Status {
        /// Output machine-readable JSON.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
}

/// Options for `allyh config`.
#[derive(Debug, Clone, Args)]
pub struct ConfigArgs {
    #[command(subcommand)]
    pub action: ConfigCommand,
}

/// Subcommands under `allyh config`.
#[derive(Debug, Clone, Subcommand)]
pub enum ConfigCommand {
    /// Show detected toolchains and any stored out-of-tree repository profile overrides.
    Show {
        /// Output machine-readable JSON.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Persist custom check/lint/test commands, language overrides, or extra ignore globs in LanceDB.
    Set {
        /// Custom check/type-check command (pass `auto` to clear).
        #[arg(long)]
        check: Option<String>,
        /// Custom lint command (pass `auto` to clear).
        #[arg(long)]
        lint: Option<String>,
        /// Custom test command (pass `auto` to clear).
        #[arg(long)]
        test: Option<String>,
        /// Override detected languages (repeatable, e.g. `--lang rust --lang typescript`).
        #[arg(long = "lang")]
        languages: Vec<String>,
        /// Extra ignore globs applied during workspace scanning (repeatable).
        #[arg(long = "ignore")]
        ignore_globs: Vec<String>,
        /// Override automatic post-edit verification (`true` or `false`).
        #[arg(long)]
        auto_verify: Option<bool>,
    },
    /// Remove the stored repository profile and restore automatic detection.
    Reset,
}

/// Options for `allyh sim`.
#[derive(Debug, Clone, Args)]
pub struct SimArgs {
    #[command(subcommand)]
    pub action: SimCommand,
}

/// Subcommands under `allyh sim`.
#[derive(Debug, Clone, Subcommand)]
pub enum SimCommand {
    /// Show active harness version, SIM lineage (`carried_from`), and problem/attempt counts.
    Status {
        /// Optional SIM run ID prefix or arm name (defaults to latest run).
        #[arg(long)]
        run: Option<String>,
        /// Output machine-readable JSON.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// List recurring harness problem entries (`P1..Pn`) in a SIM run.
    #[command(visible_alias = "ls")]
    List {
        /// Optional SIM run ID prefix or arm name (defaults to latest run).
        #[arg(long)]
        run: Option<String>,
        /// Output machine-readable JSON.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Show the complete attempt history and outcomes of a SIM problem ID (or merged alias).
    Show {
        /// Problem identifier or alias (e.g. `P3` or `P6`).
        problem_id: String,
        /// Optional SIM run ID prefix or arm name (defaults to latest run).
        #[arg(long)]
        run: Option<String>,
        /// Output machine-readable JSON.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Export a SIM ledger to the canonical arXiv:2611.18856 Appendix D.3 JSON format.
    Export {
        /// Optional SIM run ID prefix or arm name (defaults to latest run).
        #[arg(long)]
        run: Option<String>,
        /// Destination JSON path (prints to stdout when omitted).
        #[arg(short, long)]
        out: Option<PathBuf>,
    },
    /// Import an external Appendix D.3 SIM JSON file into this repository partition.
    Import {
        /// Path to the SIM JSON document.
        path: PathBuf,
        /// Optional arm label for the imported run.
        #[arg(long)]
        arm: Option<String>,
        /// Optional domain label.
        #[arg(long, default_value = "coding")]
        domain: String,
        /// Optional benchmark label.
        #[arg(long, default_value = "imported")]
        benchmark: String,
    },
    /// Run source Recursive Self-Improvement (RSI) on an evaluation suite's disjoint evolve split.
    Evolve {
        /// Path to a JSON evaluation suite (`EvalSuiteSpec`).
        #[arg(long)]
        suite: PathBuf,
        /// Run arm label.
        #[arg(long, default_value = "source-rsi")]
        arm: String,
        /// Number of optimization rounds N (paper default: 10).
        #[arg(long, default_value_t = 3)]
        rounds: u32,
        /// Candidate proposals k per round (paper default: 5).
        #[arg(long, default_value_t = 5)]
        candidates: usize,
        /// Evaluation rollouts per evolve task.
        #[arg(long, default_value_t = 1)]
        rollouts: usize,
        /// SIM memory mode (`full-sim`, `problem-only`, `harness-transfer`, `cold-start`).
        #[arg(long, default_value = "full-sim")]
        mode: String,
        /// Optional dedicated optimizer model spec for proposer & selector (defaults to `--model`).
        #[arg(long)]
        optimizer_model: Option<String>,
        /// Optional external evaluator program (`CommandHarnessEvaluator`).
        #[arg(long)]
        eval_cmd: Option<String>,
        /// Activate the final evolved harness in LanceDB for interactive sessions.
        #[arg(long, default_value_t = true)]
        activate: bool,
        /// Output machine-readable JSON report.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Run cross-dataset or cross-model CRSI continuation carrying `(H_S, M_S)` from a source run.
    Continue {
        /// Source SIM run ID prefix or arm name.
        #[arg(long)]
        from_run: String,
        /// Optional source repository path if carrying across repositories.
        #[arg(long)]
        from_repo: Option<PathBuf>,
        /// Optional source scope ID if carrying across scopes.
        #[arg(long)]
        from_scope: Option<String>,
        /// Path to the target JSON evaluation suite (`EvalSuiteSpec`).
        #[arg(long)]
        suite: PathBuf,
        /// Target arm label.
        #[arg(long, default_value = "target-crsi")]
        arm: String,
        /// Number of target optimization rounds N.
        #[arg(long, default_value_t = 3)]
        rounds: u32,
        /// Candidate proposals k per round.
        #[arg(long, default_value_t = 5)]
        candidates: usize,
        /// Evaluation rollouts per evolve task.
        #[arg(long, default_value_t = 1)]
        rollouts: usize,
        /// Transfer condition (`full-sim`, `problem-only`, `harness-transfer`, `cold-start`).
        #[arg(long, default_value = "full-sim")]
        mode: String,
        /// Optional dedicated optimizer model spec for proposer & selector.
        #[arg(long)]
        optimizer_model: Option<String>,
        /// Optional external evaluator program (`CommandHarnessEvaluator`).
        #[arg(long)]
        eval_cmd: Option<String>,
        /// Activate the final retained target harness in LanceDB.
        #[arg(long, default_value_t = true)]
        activate: bool,
        /// Output machine-readable JSON report.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Evaluate a harness version on the held-out (or evolve) split outside the optimization loop.
    Eval {
        /// Harness version ID prefix, label, or `active`.
        #[arg(long, default_value = "active")]
        harness: String,
        /// Path to the JSON evaluation suite (`EvalSuiteSpec`).
        #[arg(long)]
        suite: PathBuf,
        /// Split to evaluate (`held-out` or `evolve`).
        #[arg(long, default_value = "held-out")]
        split: String,
        /// Evaluation rollouts per task.
        #[arg(long, default_value_t = 1)]
        rollouts: usize,
        /// Optional external evaluator program (`CommandHarnessEvaluator`).
        #[arg(long)]
        eval_cmd: Option<String>,
        /// Output machine-readable JSON feedback.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Activate a stored harness version for interactive TUI sessions.
    Use {
        /// Harness version ID prefix or label.
        version: String,
    },
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

    #[test]
    fn parses_sim_subcommands_with_global_flags() {
        let status_args = CliArgs::try_parse_from([
            "allyh",
            "sim",
            "status",
            "--json",
            "--repo",
            "workspace",
            "--scope",
            "sim-team",
        ])
        .expect("parse sim status");
        assert_eq!(status_args.repo, PathBuf::from("workspace"));
        assert_eq!(status_args.scope, "sim-team");
        assert!(matches!(
            status_args.command,
            Some(CliCommand::Sim(SimArgs {
                action: SimCommand::Status { run: None, json: true }
            }))
        ));

        let evolve_args = CliArgs::try_parse_from([
            "allyh",
            "sim",
            "evolve",
            "--suite",
            "bench.json",
            "--rounds",
            "10",
            "--candidates",
            "5",
            "--mode",
            "full-sim",
        ])
        .expect("parse sim evolve");
        assert!(matches!(
            evolve_args.command,
            Some(CliCommand::Sim(SimArgs {
                action: SimCommand::Evolve {
                    rounds: 10,
                    candidates: 5,
                    ref mode,
                    ..
                }
            })) if mode == "full-sim"
        ));

        let cont_args = CliArgs::try_parse_from([
            "allyh",
            "sim",
            "continue",
            "--from-run",
            "source-tb21",
            "--suite",
            "tb4.json",
            "--mode",
            "problem-only",
        ])
        .expect("parse sim continue");
        assert!(matches!(
            cont_args.command,
            Some(CliCommand::Sim(SimArgs {
                action: SimCommand::Continue {
                    ref from_run,
                    ref mode,
                    ..
                }
            })) if from_run == "source-tb21" && mode == "problem-only"
        ));

        let show_args = CliArgs::try_parse_from(["allyh", "sim", "show", "P5"]).expect("sim show");
        assert!(matches!(
            show_args.command,
            Some(CliCommand::Sim(SimArgs {
                action: SimCommand::Show { ref problem_id, .. }
            })) if problem_id == "P5"
        ));
    }

    #[test]
    fn parses_config_subcommands_with_global_flags() {
        let show = CliArgs::try_parse_from(["allyh", "config", "show", "--json"])
            .expect("parse config show");
        assert!(matches!(
            show.command,
            Some(CliCommand::Config(ConfigArgs {
                action: ConfigCommand::Show { json: true }
            }))
        ));

        let set = CliArgs::try_parse_from([
            "allyh",
            "config",
            "set",
            "--check",
            "npx tsc --noEmit",
            "--test",
            "pytest -q",
            "--lang",
            "typescript",
            "--ignore",
            "generated/**",
            "--auto-verify",
            "false",
        ])
        .expect("parse config set");
        assert!(matches!(
            set.command,
            Some(CliCommand::Config(ConfigArgs {
                action: ConfigCommand::Set {
                    ref check,
                    ref test,
                    ref languages,
                    ref ignore_globs,
                    auto_verify: Some(false),
                    ..
                }
            })) if check.as_deref() == Some("npx tsc --noEmit")
                && test.as_deref() == Some("pytest -q")
                && languages == &vec!["typescript".to_string()]
                && ignore_globs == &vec!["generated/**".to_string()]
        ));

        let reset = CliArgs::try_parse_from(["allyh", "config", "reset"]).expect("parse config reset");
        assert!(matches!(
            reset.command,
            Some(CliCommand::Config(ConfigArgs {
                action: ConfigCommand::Reset
            }))
        ));

        let lsp_status = CliArgs::try_parse_from(["allyh", "lsp", "status", "--json"])
            .expect("parse lsp status");
        assert!(matches!(
            lsp_status.command,
            Some(CliCommand::Lsp(LspArgs {
                action: LspCommand::Status { json: true }
            }))
        ));
    }
}
