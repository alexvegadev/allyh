use std::path::Path;
use std::sync::Arc;

use allyh_core::{
    AllyHOrchestrator, HeadlessMode, HybridRouter, ModelProfile, OrchestratorConfig,
    ReasoningEffort, audit_repo_against_base, run_headless_turn,
};
use allyh_env::{GitTracker, WorktreeManager};
use allyh_memory::{
    DeterministicHashEmbedder, EmbeddingProfile, HeuristicConflictJudge, MemoryConsolidator,
    MemoryStore, StorageConfig, compute_repo_id,
};
use anyhow::{Context, Result, bail};

use crate::cli::{AuditArgs, MemoryArgs, MemoryCommand, RunArgs, WorktreeArgs, WorktreeCommand};

/// Execute `allyh run` non-interactively (CI / headless mode).
#[allow(clippy::too_many_arguments)]
pub async fn run_headless_cli(
    args: RunArgs,
    repo: &Path,
    db_uri: Option<&str>,
    scope: &str,
    model_spec: Option<&str>,
    worker_spec: Option<&str>,
    reasoning_effort: Option<&str>,
    embedding_model: Option<&str>,
) -> Result<()> {
    let prompt = match (args.prompt, args.task_file) {
        (Some(p), None) => p,
        (None, Some(path)) => std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read task file '{}'", path.display()))?,
        (Some(_), Some(_)) => bail!("specify either a prompt or `--task-file`, not both"),
        (None, None) => bail!("provide a prompt argument or `--task-file <path>`"),
    };

    let mode = HeadlessMode::parse_spec(&args.mode).map_err(|e| anyhow::anyhow!(e))?;
    if mode == HeadlessMode::Execute && !args.auto_approve {
        bail!("`allyh run --mode execute` requires explicit `--auto-approve` in headless CI mode");
    }

    let mut cfg = OrchestratorConfig::new(repo);
    cfg.scope_id = scope.to_string();
    cfg.auto_verify = !args.no_verify;
    cfg.enable_fs_watcher = false;
    if let Some(uri) = db_uri {
        cfg.storage = StorageConfig::from_uri(uri.to_string());
    }

    let mut router = HybridRouter::from_env();
    if let Some(spec) = model_spec {
        router.set_primary(ModelProfile::parse_spec(spec).context("invalid --model spec")?);
    }
    if let Some(effort_spec) = reasoning_effort {
        let effort = ReasoningEffort::parse_spec(effort_spec)
            .context("invalid --reasoning-effort value")?;
        router.set_reasoning_effort(Some(effort));
    }
    if let Some(spec) = worker_spec {
        router.set_local_worker(
            ModelProfile::parse_spec(spec).context("invalid --worker-model spec")?,
        );
    }

    let embedding_profile = match embedding_model {
        Some(spec) => {
            EmbeddingProfile::parse_spec(spec).context("invalid --embedding-model spec")?
        }
        None => EmbeddingProfile::from_env(),
    };
    let embedder = embedding_profile.build_provider();

    let mut orch = AllyHOrchestrator::initialize(cfg, router, Some(embedder))
        .await
        .context("failed to initialize headless AllyH orchestrator")?;

    let report = run_headless_turn(
        &mut orch,
        repo,
        &prompt,
        mode,
        args.auto_approve,
        !args.no_verify,
    )
    .await?;

    if args.output.eq_ignore_ascii_case("jsonl") {
        for ev in &report.events {
            println!("{}", serde_json::to_string(ev)?);
        }
    } else {
        println!("{}", report.assistant_text);
        if let Some(proof) = &report.verification_proof {
            println!("\n[Verification Proof] {} -> {}", proof.command, proof.summary);
        }
    }

    if !report.passed {
        bail!("headless run failed verification");
    }
    Ok(())
}

/// Execute `allyh audit --base <ref>` to check a Git diff against active LanceDB invariants.
pub async fn run_audit_cli(
    args: AuditArgs,
    repo: &Path,
    db_uri: Option<&str>,
    scope: &str,
) -> Result<()> {
    let storage = match db_uri {
        Some(uri) => StorageConfig::from_uri(uri.to_string()),
        None => StorageConfig::default_local(),
    };
    let store = MemoryStore::connect(storage, Arc::new(DeterministicHashEmbedder))
        .await
        .context("failed to connect to LanceDB")?;
    let git = GitTracker::new(repo);
    let repo_id = compute_repo_id(&git.canonical_repo_identifier());

    let report = audit_repo_against_base(&store, repo, &repo_id, scope, &args.base).await?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "Architectural Drift Audit (base: `{}`, scope: `{scope}`)",
            report.base_ref
        );
        println!(
            "Files changed: {} | Active invariants checked: {} | Violations: {}",
            report.files_changed,
            report.invariants_checked,
            report.violations.len()
        );
        for v in &report.violations {
            println!(
                "  [VIOLATION] {} [{}]: {}\n    Line: {}\n    Why:  {}",
                v.file_path, v.topic, v.rule_statement, v.line_preview, v.reason
            );
        }
        if report.passed {
            println!("Result: PASS (zero architectural invariant violations)");
        }
    }

    if !report.passed {
        bail!(
            "architectural audit failed with {} invariant violation(s)",
            report.violations.len()
        );
    }
    Ok(())
}

/// Execute `allyh memory <promote|timeline>` against out-of-tree LanceDB.
pub async fn run_memory_cli(
    args: MemoryArgs,
    repo: &Path,
    db_uri: Option<&str>,
    scope: &str,
) -> Result<()> {
    let storage = match db_uri {
        Some(uri) => StorageConfig::from_uri(uri.to_string()),
        None => StorageConfig::default_local(),
    };
    let store = MemoryStore::connect(storage, Arc::new(DeterministicHashEmbedder))
        .await
        .context("failed to connect to LanceDB")?;
    let git = GitTracker::new(repo);
    let repo_id = compute_repo_id(&git.canonical_repo_identifier());

    match args.action {
        MemoryCommand::Promote {
            from,
            to,
            no_skills,
            json,
        } => {
            let consolidator = MemoryConsolidator::new(store, Arc::new(HeuristicConflictJudge));
            let report = consolidator
                .promote_scope(&repo_id, &from, &to, !no_skills)
                .await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "Promoted memory `{}` -> `{}` (repo `{}`):",
                    report.from_scope,
                    report.to_scope,
                    &report.repo_id[..report.repo_id.len().min(12)]
                );
                println!(
                    "  Decisions: +{} added, {} superseded/updated, {} noop | Skills promoted: {}",
                    report.decisions_added,
                    report.decisions_updated,
                    report.decisions_noop,
                    report.skills_promoted
                );
            }
            Ok(())
        }
        MemoryCommand::Timeline { json } => {
            let all = store.list_all_decisions(&repo_id, scope).await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&all)?);
                return Ok(());
            }
            println!(
                "Bi-Temporal Memory Timeline ({} total decision(s), scope: `{scope}`):",
                all.len()
            );
            for dec in &all {
                let status = if dec.invalid_at.is_some() {
                    match &dec.superseded_by {
                        Some(by) => format!("SUPERSEDED by `{}`", &by[..by.len().min(8)]),
                        None => "INVALIDATED".to_string(),
                    }
                } else {
                    "ACTIVE".to_string()
                };
                println!(
                    "  - `{}` [{status}] [{}] {} (valid: {})",
                    &dec.id[..dec.id.len().min(8)],
                    dec.topic,
                    dec.rule_statement,
                    dec.valid_at.format("%Y-%m-%d %H:%M UTC")
                );
            }
            Ok(())
        }
    }
}

/// Execute `allyh worktree <create|list|remove>` for isolated parallel agent scopes.
pub async fn run_worktree_cli(args: WorktreeArgs, repo: &Path) -> Result<()> {
    match args.action {
        WorktreeCommand::Create {
            scope,
            base,
            path,
            json,
        } => {
            let info = WorktreeManager::create_scoped_worktree(
                repo,
                &scope,
                base.as_deref(),
                path.as_deref(),
            )?;
            if json {
                println!("{}", serde_json::to_string_pretty(&info)?);
            } else {
                println!(
                    "Created isolated worktree for scope `{scope}`:\n  Path:   {}\n  Branch: {}",
                    info.path.display(),
                    info.branch
                );
                println!(
                    "  Run:    allyh --repo \"{}\" --scope {scope}",
                    info.path.display()
                );
            }
            Ok(())
        }
        WorktreeCommand::List { json } => {
            let list = WorktreeManager::list_worktrees(repo)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&list)?);
            } else {
                println!("Git Worktrees ({}):", list.len());
                for wt in &list {
                    let badge = if wt.is_main {
                        "main".to_string()
                    } else {
                        wt.scope_id
                            .as_deref()
                            .map(|s| format!("scope={s}"))
                            .unwrap_or_else(|| "linked".to_string())
                    };
                    println!("  - [{}] {} ({})", badge, wt.path.display(), wt.branch);
                }
            }
            Ok(())
        }
        WorktreeCommand::Remove { target, force } => {
            let removed = WorktreeManager::remove_worktree(repo, &target, force)?;
            println!("Removed git worktree '{}'.", removed.display());
            Ok(())
        }
    }
}
