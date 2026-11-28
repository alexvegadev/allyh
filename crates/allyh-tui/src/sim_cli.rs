use std::path::Path;
use std::sync::Arc;

use allyh_core::{HarnessSpec, HybridRouter, ModelProfile};
use allyh_env::GitTracker;
use allyh_memory::{
    DeterministicHashEmbedder, MemoryStore, SimExportDocument, SimMemoryMode, StorageConfig,
    compute_repo_id,
};
use allyh_sim::{
    AllyHRuntimeEvaluator, BenchmarkEvaluator, CommandHarnessEvaluator, CrsiConfig, CrsiRunner,
    EvalSuiteSpec, LlmSimProposer, LlmSimSelector, harvest_sessions_into_sim,
    polyglot_suite_preset,
};
use anyhow::{Context, Result, bail};

use crate::cli::{SimArgs, SimCommand};

#[allow(clippy::too_many_arguments)]
pub async fn run_cli(
    args: SimArgs,
    repo: &Path,
    db_uri: Option<&str>,
    scope_id: &str,
    model_spec: Option<&str>,
    worker_spec: Option<&str>,
) -> Result<()> {
    let storage = match db_uri {
        Some(uri) => StorageConfig::from_uri(uri.to_string()),
        None => StorageConfig::default_local(),
    };
    let store = MemoryStore::connect(storage, Arc::new(DeterministicHashEmbedder))
        .await
        .context("failed to connect to LanceDB memory store")?;

    let canonical_id = GitTracker::new(repo).canonical_repo_identifier();
    let repo_id = compute_repo_id(&canonical_id);

    let mut router = HybridRouter::from_env();
    if let Some(spec) = model_spec {
        router.set_primary(ModelProfile::parse_spec(spec).context("invalid --model spec")?);
    }
    if let Some(spec) = worker_spec {
        router.set_local_worker(
            ModelProfile::parse_spec(spec).context("invalid --worker-model spec")?,
        );
    }

    match args.action {
        SimCommand::Status { run, json } => {
            let active_hv = store.get_active_harness_version(&repo_id, scope_id).await?;
            let ledger = store
                .load_sim_ledger(&repo_id, scope_id, run.as_deref().unwrap_or(""))
                .await?;
            if json {
                let payload = serde_json::json!({
                    "repo_id": repo_id,
                    "scope_id": scope_id,
                    "active_harness": active_hv,
                    "sim_run": ledger.as_ref().map(|l| &l.run),
                    "problems_count": ledger.as_ref().map(|l| l.problems.len()).unwrap_or(0),
                    "attempts_count": ledger.as_ref().map(|l| l.total_attempts()).unwrap_or(0),
                    "aliases": ledger.as_ref().map(|l| &l.aliases),
                });
                println!("{}", serde_json::to_string_pretty(&payload)?);
                return Ok(());
            }

            println!(
                "SIM Status — repo `{}` | scope `{scope_id}`",
                &repo_id[..repo_id.len().min(12)]
            );
            match &active_hv {
                Some(hv) => println!(
                    "Active Harness: {} (id: `{}`, round {}, policy: `{}`)",
                    hv.label,
                    &hv.id[..hv.id.len().min(8)],
                    hv.round,
                    hv.policy_model
                ),
                None => println!("Active Harness: seed (default)"),
            }
            match ledger {
                Some(l) => {
                    println!(
                        "Latest SIM Run: `{}` (arm: `{}`, mode: `{}`, rounds: {})",
                        &l.run.run_id[..l.run.run_id.len().min(8)],
                        l.run.arm_name,
                        l.run.memory_mode,
                        l.run.rounds_completed
                    );
                    println!(
                        "Context:        domain=`{}` | benchmark=`{}` | policy=`{}`",
                        l.run.domain, l.run.benchmark, l.run.policy_model
                    );
                    if let Some(ref from_run) = l.run.carried_from_run_id {
                        println!(
                            "Carried From:   `{}` (domain: `{}`)",
                            from_run,
                            l.run.carried_from_domain.as_deref().unwrap_or("—")
                        );
                    }
                    println!(
                        "Entries:        {} problem(s) | {} attempt(s) | {} alias(es)",
                        l.problems.len(),
                        l.total_attempts(),
                        l.aliases.len()
                    );
                }
                None => println!("Latest SIM Run: none yet (run `allyh sim evolve --suite <path>`)."),
            }
            Ok(())
        }
        SimCommand::List { run, json } => {
            let Some(ledger) = store
                .load_sim_ledger(&repo_id, scope_id, run.as_deref().unwrap_or(""))
                .await?
            else {
                if json {
                    println!("[]");
                } else {
                    println!("No SIM runs recorded for this repository partition.");
                }
                return Ok(());
            };
            if json {
                println!("{}", serde_json::to_string_pretty(&ledger.problems)?);
                return Ok(());
            }
            println!(
                "SIM Problems ({}) — run `{}` ({})\n",
                ledger.problems.len(),
                ledger.run.arm_name,
                ledger.run.memory_mode
            );
            for p in &ledger.problems {
                let pred: usize = p.attempts.iter().map(|a| a.predicted.len()).sum();
                let fixed: usize = p.attempts.iter().map(|a| a.fixed.len()).sum();
                println!(
                    "  {}  (r{}, origin `{}`): {} — {} attempt(s), {fixed}/{pred} predicted fixes realized",
                    p.id,
                    p.opened_round,
                    p.origin_domain,
                    p.name,
                    p.attempts.len()
                );
            }
            Ok(())
        }
        SimCommand::Show {
            problem_id,
            run,
            json,
        } => {
            let Some(ledger) = store
                .load_sim_ledger(&repo_id, scope_id, run.as_deref().unwrap_or(""))
                .await?
            else {
                bail!("no SIM runs recorded for this repository partition");
            };
            let canonical = ledger.resolve_alias(&problem_id);
            let Some(problem) = ledger.get_problem(&canonical) else {
                bail!("SIM problem '{problem_id}' not found");
            };
            if json {
                println!("{}", serde_json::to_string_pretty(problem)?);
                return Ok(());
            }
            println!("Problem:      {} — {}", problem.id, problem.name);
            if !canonical.eq_ignore_ascii_case(problem_id.trim()) {
                println!("Resolved:     {} -> {}", problem_id.trim(), problem.id);
            }
            println!(
                "Opened Round: {} (origin: `{}`)",
                problem.opened_round, problem.origin_domain
            );
            println!("Attempts ({}):", problem.attempts.len());
            for a in &problem.attempts {
                println!(
                    "\n  [Round {} / {}] domain=`{}` benchmark=`{}` policy=`{}` score={:.2} ({}/{})",
                    a.round, a.sample, a.domain, a.benchmark, a.policy_model, a.score, a.passed, a.total
                );
                println!("    What:      {}", a.what);
                if !a.files.is_empty() {
                    println!("    Files:     {}", a.files.join(", "));
                }
                println!("    Predicted: [{}]", a.predicted.join(", "));
                println!("    Fixed:     [{}]", a.fixed.join(", "));
                println!("    Missed:    [{}]", a.missed.join(", "));
            }
            Ok(())
        }
        SimCommand::Export { run, out } => {
            let Some(doc) = store
                .export_sim_document(&repo_id, scope_id, run.as_deref())
                .await?
            else {
                bail!("no SIM run available to export");
            };
            let rendered = serde_json::to_string_pretty(&doc)?;
            if let Some(path) = out {
                if let Some(parent) = path.parent()
                    && !parent.as_os_str().is_empty()
                {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&path, rendered)
                    .with_context(|| format!("failed to write '{}'", path.display()))?;
                println!(
                    "Exported SIM ledger ({} problem(s)) to '{}'.",
                    doc.problems.len(),
                    path.display()
                );
            } else {
                println!("{rendered}");
            }
            Ok(())
        }
        SimCommand::Import {
            path,
            arm,
            domain,
            benchmark,
        } => {
            let raw = std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read '{}'", path.display()))?;
            let doc: SimExportDocument = serde_json::from_str(&raw)
                .with_context(|| format!("invalid SIM JSON in '{}'", path.display()))?;
            let arm_name = arm.unwrap_or_else(|| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("imported-sim")
                    .to_string()
            });
            let active_id = store
                .get_active_harness_version(&repo_id, scope_id)
                .await?
                .map(|h| h.id)
                .unwrap_or_else(|| "seed".to_string());
            let ledger = store
                .import_sim_document(
                    &repo_id,
                    scope_id,
                    doc,
                    &arm_name,
                    &domain,
                    &benchmark,
                    &router.primary().model_id,
                    &router.primary().model_id,
                    &active_id,
                )
                .await?;
            println!(
                "Imported SIM run `{}` (arm: `{}`, {} problem(s), {} attempt(s)).",
                &ledger.run.run_id[..ledger.run.run_id.len().min(8)],
                ledger.run.arm_name,
                ledger.problems.len(),
                ledger.total_attempts()
            );
            Ok(())
        }
        SimCommand::Evolve {
            suite,
            preset,
            arm,
            rounds,
            candidates,
            rollouts,
            mode,
            optimizer_model,
            eval_cmd,
            activate,
            json,
        } => {
            let suite_spec = match (suite, preset) {
                (Some(path), None) => EvalSuiteSpec::load_file(&path)
                    .with_context(|| format!("failed to load suite '{}'", path.display()))?,
                (None, Some(lang)) => polyglot_suite_preset(&lang)?,
                (None, None) => bail!(
                    "specify either `--suite <path.json>` or `--preset <rust|typescript|python|go|polyglot>`"
                ),
                (Some(_), Some(_)) => bail!("cannot specify both `--suite` and `--preset`"),
            };
            let (evolve_split, _held_out) = suite_spec.split()?;
            let memory_mode = SimMemoryMode::parse(&mode)?;
            let optimizer_profile = match optimizer_model.as_deref() {
                Some(spec) => ModelProfile::parse_spec(spec)?,
                None => router.primary().clone(),
            };

            let evaluator = build_evaluator(eval_cmd.as_deref(), repo);
            let runner = CrsiRunner::new(
                store.clone(),
                evaluator,
                Arc::new(LlmSimProposer::new(router.clone(), optimizer_profile.clone())),
                Arc::new(LlmSimSelector::new(router.clone(), optimizer_profile.clone())),
            );
            let cfg = CrsiConfig {
                repo_id,
                scope_id: scope_id.to_string(),
                arm_name: arm,
                domain: suite_spec.domain,
                benchmark: suite_spec.benchmark,
                rounds,
                candidates_per_round: candidates,
                rollouts_per_task: rollouts,
                memory_mode,
                policy_model: router.primary().clone(),
                optimizer_model: optimizer_profile,
                activate_winner: activate,
            };

            let (_hv, ledger, report) = runner
                .run_source(cfg, HarnessSpec::default_seed(), &evolve_split)
                .await?;
            print_crsi_report(&report, &ledger, json)?;
            Ok(())
        }
        SimCommand::Continue {
            from_run,
            from_repo,
            from_scope,
            suite,
            arm,
            rounds,
            candidates,
            rollouts,
            mode,
            optimizer_model,
            eval_cmd,
            activate,
            json,
        } => {
            let suite_spec = EvalSuiteSpec::load_file(&suite)
                .with_context(|| format!("failed to load suite '{}'", suite.display()))?;
            let (evolve_split, _held_out) = suite_spec.split()?;
            let memory_mode = SimMemoryMode::parse(&mode)?;
            let optimizer_profile = match optimizer_model.as_deref() {
                Some(spec) => ModelProfile::parse_spec(spec)?,
                None => router.primary().clone(),
            };

            let source_repo_id = match from_repo.as_deref() {
                Some(path) => compute_repo_id(&GitTracker::new(path).canonical_repo_identifier()),
                None => repo_id.clone(),
            };
            let source_scope_id = from_scope.as_deref().unwrap_or(scope_id);

            let evaluator = build_evaluator(eval_cmd.as_deref(), repo);
            let runner = CrsiRunner::new(
                store.clone(),
                evaluator,
                Arc::new(LlmSimProposer::new(router.clone(), optimizer_profile.clone())),
                Arc::new(LlmSimSelector::new(router.clone(), optimizer_profile.clone())),
            );
            let cfg = CrsiConfig {
                repo_id,
                scope_id: scope_id.to_string(),
                arm_name: arm,
                domain: suite_spec.domain,
                benchmark: suite_spec.benchmark,
                rounds,
                candidates_per_round: candidates,
                rollouts_per_task: rollouts,
                memory_mode,
                policy_model: router.primary().clone(),
                optimizer_model: optimizer_profile,
                activate_winner: activate,
            };

            let (_hv, ledger, report) = runner
                .run_continuation(
                    &source_repo_id,
                    source_scope_id,
                    &from_run,
                    cfg,
                    &evolve_split,
                )
                .await?;
            print_crsi_report(&report, &ledger, json)?;
            Ok(())
        }
        SimCommand::Eval {
            harness,
            suite,
            split,
            rollouts,
            eval_cmd,
            json,
        } => {
            let suite_spec = EvalSuiteSpec::load_file(&suite)
                .with_context(|| format!("failed to load suite '{}'", suite.display()))?;
            let (evolve_split, held_out_split) = suite_spec.split()?;
            let evaluator = build_evaluator(eval_cmd.as_deref(), repo);

            let (spec, hv_id) = if harness.eq_ignore_ascii_case("active") {
                match store.get_active_harness_version(&repo_id, scope_id).await? {
                    Some(rec) => (HarnessSpec::from_version_record(&rec)?, rec.id),
                    None => (HarnessSpec::default_seed(), "seed-default".to_string()),
                }
            } else if harness.eq_ignore_ascii_case("seed") {
                (HarnessSpec::default_seed(), "seed-default".to_string())
            } else {
                let Some(rec) = store
                    .get_harness_version(&repo_id, scope_id, &harness)
                    .await?
                else {
                    bail!("harness version '{harness}' not found");
                };
                (HarnessSpec::from_version_record(&rec)?, rec.id)
            };

            let tasks = match split.trim().to_ascii_lowercase().as_str() {
                "held-out" | "held_out" | "eval" => held_out_split.tasks(),
                "evolve" | "evo" => evolve_split.tasks(),
                other => bail!("unknown split '{other}' (expected 'held-out' or 'evolve')"),
            };

            let fb = evaluator
                .evaluate(&spec, &hv_id, router.primary(), tasks, rollouts)
                .await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&fb)?);
            } else {
                println!("Evaluation ({split} split):\n{}", fb.render_summary());
            }
            Ok(())
        }
        SimCommand::Use { version } => {
            let rec = store
                .set_active_harness_version(&repo_id, scope_id, &version)
                .await?;
            println!(
                "Activated harness `{}` (id: `{}`, round {}, policy: `{}`).",
                rec.label,
                &rec.id[..rec.id.len().min(8)],
                rec.round,
                rec.policy_model
            );
            Ok(())
        }
        SimCommand::Preset { lang, out } => {
            let suite = polyglot_suite_preset(&lang)?;
            let rendered = serde_json::to_string_pretty(&suite)?;
            if let Some(path) = out {
                if let Some(parent) = path.parent()
                    && !parent.as_os_str().is_empty()
                {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&path, rendered)
                    .with_context(|| format!("failed to write preset '{}'", path.display()))?;
                println!(
                    "Wrote `{}` preset ({} tasks) to '{}'.",
                    suite.name,
                    suite.tasks.len(),
                    path.display()
                );
            } else {
                println!("{rendered}");
            }
            Ok(())
        }
        SimCommand::Harvest { run, json } => {
            let report = harvest_sessions_into_sim(
                &store,
                &repo_id,
                scope_id,
                run.as_deref(),
                &router.primary().model_id,
            )
            .await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "SIM Session Harvest — run `{}` ({})",
                    &report.run_id[..report.run_id.len().min(8)],
                    report.arm_name
                );
                println!(
                    "Scanned:   {} session(s) | {} task(s)",
                    report.sessions_scanned, report.tasks_scanned
                );
                println!(
                    "Harvested: {} new problem(s) | {} attempt(s) | IDs: [{}]",
                    report.problems_opened,
                    report.attempts_recorded,
                    report.problem_ids.join(", ")
                );
            }
            Ok(())
        }
    }
}

fn build_evaluator(eval_cmd: Option<&str>, repo: &Path) -> Arc<dyn BenchmarkEvaluator> {
    if let Some(cmd_line) = eval_cmd.filter(|s| !s.trim().is_empty()) {
        let mut parts = cmd_line.split_whitespace();
        if let Some(program) = parts.next() {
            return Arc::new(CommandHarnessEvaluator {
                program: program.to_string(),
                base_args: parts.map(str::to_string).collect(),
                working_dir: Some(repo.to_path_buf()),
            });
        }
    }
    Arc::new(AllyHRuntimeEvaluator::live_default())
}

fn print_crsi_report(
    report: &allyh_sim::CrsiRunReport,
    ledger: &allyh_memory::SimLedger,
    json: bool,
) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(report)?);
        return Ok(());
    }
    println!(
        "CRSI Run `{}` ({}) — mode: `{}`",
        &report.run_id[..report.run_id.len().min(8)],
        report.arm_name,
        report.memory_mode
    );
    if let Some(ref carried) = report.carried_from_run_id {
        println!("Carried from:   `{carried}`");
    }
    println!(
        "Evolve Score:   {:.2} -> {:.2} ({:+.2})",
        report.starting_evolve_score,
        report.final_evolve_score,
        report.final_evolve_score - report.starting_evolve_score
    );
    println!(
        "Final Harness:  `{}`",
        &report.final_harness_id[..report.final_harness_id.len().min(12)]
    );
    println!(
        "SIM Ledger:     {} problem(s) | {} attempt(s)\n",
        ledger.problems.len(),
        ledger.total_attempts()
    );
    for r in &report.rounds {
        let accepted = r
            .accepted_sample
            .as_deref()
            .map(|s| {
                format!(
                    "accepted {s} ({})",
                    r.accepted_problem_id.as_deref().unwrap_or("no-sim")
                )
            })
            .unwrap_or_else(|| "no edit accepted".to_string());
        println!(
            "  Round {}: {:.2} -> {:.2} | {accepted} — {}",
            r.round, r.baseline_score, r.resulting_score, r.selection_rationale
        );
    }
    Ok(())
}
