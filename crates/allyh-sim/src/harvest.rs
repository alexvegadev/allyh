use allyh_memory::{
    MemoryStore, SimAttemptInput, SimLedger, SimMemoryMode, SimProblemRef, TaskStatus,
};
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Summary of harvesting out-of-tree interactive sessions and task failure traces into SIM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionHarvestReport {
    pub run_id: String,
    pub arm_name: String,
    pub sessions_scanned: usize,
    pub tasks_scanned: usize,
    pub problems_opened: usize,
    pub attempts_recorded: usize,
    pub problem_ids: Vec<String>,
}

/// Distill real failed/recovered interactive TUI sessions and task error traces stored out-of-tree
/// in LanceDB into deduplicated SIM problem diagnoses (`P1..Pn`) as backfilled attempts.
pub async fn harvest_sessions_into_sim(
    store: &MemoryStore,
    repo_id: &str,
    scope_id: &str,
    target_run: Option<&str>,
    policy_model: &str,
) -> Result<SessionHarvestReport> {
    let sessions = store.list_sessions(repo_id, scope_id).await?;
    let tasks = store.list_tasks(repo_id, scope_id).await?;

    let active_harness_id = store
        .get_active_harness_version(repo_id, scope_id)
        .await?
        .map(|h| h.id)
        .unwrap_or_else(|| "seed".to_string());

    let mut ledger = if let Some(run_ref) = target_run {
        match store.load_sim_ledger(repo_id, scope_id, run_ref).await? {
            Some(existing) => existing,
            None => SimLedger::new_source(
                repo_id,
                scope_id,
                run_ref,
                "interactive",
                "session-harvest",
                policy_model,
                policy_model,
                &active_harness_id,
                SimMemoryMode::FullSim,
            ),
        }
    } else if let Some(existing) = store
        .load_sim_ledger(repo_id, scope_id, "session-harvest")
        .await?
    {
        existing
    } else {
        SimLedger::new_source(
            repo_id,
            scope_id,
            "session-harvest",
            "interactive",
            "session-harvest",
            policy_model,
            policy_model,
            &active_harness_id,
            SimMemoryMode::FullSim,
        )
    };

    let mut problems_opened = 0usize;
    let mut attempts_recorded = 0usize;
    let mut problem_ids = Vec::new();
    let round = ledger.run.rounds_completed + 1;

    // 1. Harvest structured task error traces (both failed and subsequently recovered/verified tasks).
    for (idx, task) in tasks.iter().enumerate() {
        let Some(err_trace) = task
            .last_error_trace
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        else {
            continue;
        };

        let sample_key = format!("harvest-task-{}", &task.id[..task.id.len().min(8)]);
        if already_harvested(&ledger, &sample_key) {
            continue;
        }

        let category = classify_trace_root_cause(err_trace);
        let recovered = task.status == TaskStatus::Verified;
        let first_line = err_trace.lines().next().unwrap_or(err_trace);
        let outcome = ledger.record_accepted_attempt(SimAttemptInput {
            problem_ref: SimProblemRef::New(category.to_string()),
            round,
            sample: sample_key,
            accepted: false,
            backfilled: true,
            what: format!(
                "Harvested {} task `{}` (#{}) error trace: {}",
                if recovered { "recovered" } else { "failed" },
                task.title,
                idx + 1,
                truncate_chars(first_line, 180)
            ),
            files: Vec::new(),
            predicted: vec![task.id.clone()],
            fixed: if recovered {
                vec![task.id.clone()]
            } else {
                Vec::new()
            },
            missed: if recovered {
                Vec::new()
            } else {
                vec![task.id.clone()]
            },
            passed: u32::from(recovered),
            total: 1,
            score: if recovered { 100.0 } else { 0.0 },
            domain: "interactive".to_string(),
            benchmark: "session-harvest".to_string(),
            policy_model: policy_model.to_string(),
            harness_version_id: active_harness_id.clone(),
        })?;
        if outcome.opened_new_problem {
            problems_opened += 1;
        }
        attempts_recorded += 1;
        if !problem_ids.contains(&outcome.canonical_problem_id) {
            problem_ids.push(outcome.canonical_problem_id);
        }
    }

    // 2. Harvest high-signal session rolling summaries containing failure/recovery indicators
    //    or extreme uncached token burn.
    for session in &sessions {
        let summary = session.rolling_summary.trim();
        let lower = summary.to_ascii_lowercase();
        let has_failure_signal = lower.contains("error")
            || lower.contains("failed")
            || lower.contains("rollback")
            || lower.contains("stale hash")
            || lower.contains("rejected")
            || lower.contains("timeout");
        let low_cache_high_burn = session.input_tokens >= 80_000
            && session.cached_input_tokens * 4 < session.input_tokens;

        if !has_failure_signal && !low_cache_high_burn {
            continue;
        }

        let sample_key = format!("harvest-sess-{}", &session.session_id[..session.session_id.len().min(8)]);
        if already_harvested(&ledger, &sample_key) {
            continue;
        }

        let category = if !has_failure_signal && low_cache_high_burn {
            "Low prompt-cache hit ratio on high-token session"
        } else {
            classify_trace_root_cause(summary)
        };
        let recovered = lower.contains("pass") || lower.contains("fixed") || lower.contains("resolved");

        let outcome = ledger.record_accepted_attempt(SimAttemptInput {
            problem_ref: SimProblemRef::New(category.to_string()),
            round,
            sample: sample_key,
            accepted: false,
            backfilled: true,
            what: format!(
                "Harvested session `{}` (in={}tok, cached={}tok, ${:.4}): {}",
                &session.session_id[..session.session_id.len().min(8)],
                session.input_tokens,
                session.cached_input_tokens,
                session.estimated_cost_usd,
                truncate_chars(summary, 180)
            ),
            files: Vec::new(),
            predicted: vec![session.session_id.clone()],
            fixed: if recovered {
                vec![session.session_id.clone()]
            } else {
                Vec::new()
            },
            missed: if recovered {
                Vec::new()
            } else {
                vec![session.session_id.clone()]
            },
            passed: u32::from(recovered),
            total: 1,
            score: if recovered { 100.0 } else { 0.0 },
            domain: "interactive".to_string(),
            benchmark: "session-harvest".to_string(),
            policy_model: policy_model.to_string(),
            harness_version_id: active_harness_id.clone(),
        })?;
        if outcome.opened_new_problem {
            problems_opened += 1;
        }
        attempts_recorded += 1;
        if !problem_ids.contains(&outcome.canonical_problem_id) {
            problem_ids.push(outcome.canonical_problem_id);
        }
    }

    if attempts_recorded > 0 {
        ledger.run.rounds_completed = round;
        store.save_sim_ledger(&ledger).await?;
    }

    Ok(SessionHarvestReport {
        run_id: ledger.run.run_id,
        arm_name: ledger.run.arm_name,
        sessions_scanned: sessions.len(),
        tasks_scanned: tasks.len(),
        problems_opened,
        attempts_recorded,
        problem_ids,
    })
}

fn already_harvested(ledger: &SimLedger, sample_key: &str) -> bool {
    ledger
        .problems
        .iter()
        .any(|p| p.attempts.iter().any(|a| a.sample == sample_key))
}

fn classify_trace_root_cause(text: &str) -> &'static str {
    let lower = text.to_ascii_lowercase();
    if lower.contains("stale") && lower.contains("hash") {
        "Stale SHA-256 hash mismatch during edit_by_hash"
    } else if lower.contains("write gate") || lower.contains("approval") {
        "Control gate or command approval interruption"
    } else if lower.contains("borrow") || lower.contains("e0") || lower.contains("type") || lower.contains("tsc") {
        "Static type-check or compiler diagnostic failure"
    } else if lower.contains("test") || lower.contains("assert") || lower.contains("panic") {
        "Test assertion or runtime verification failure"
    } else if lower.contains("timeout") || lower.contains("rate limit") {
        "Tool or provider execution timeout"
    } else {
        "Interactive session execution failure or recovery"
    }
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut out: String = text.chars().take(max_chars).collect();
    if text.chars().count() > max_chars {
        out.push('…');
    }
    out
}
