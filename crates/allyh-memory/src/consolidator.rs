use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, watch};
use uuid::Uuid;

use crate::embeddings::cosine_similarity;
use crate::error::Result;
use crate::schema::ArchitecturalDecision;
use crate::store::MemoryStore;

/// Candidate architectural decision extracted from a turn or explicit user command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateDecision {
    pub topic: String,
    pub rule_statement: String,
    pub rationale: String,
    pub affected_paths: Vec<String>,
}

/// Resolution verdict emitted by the pre-write conflict gate before mutating LanceDB.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "UPPERCASE")]
pub enum ConsolidationAction {
    /// Novel invariant with no active conflicts.
    Add,
    /// Supersedes an existing active decision (`invalid_at` + `superseded_by` on target, insert new).
    Update { target_id: String },
    /// Logically invalidates an existing active decision without replacing it.
    Invalidate { target_id: String, reason: String },
    /// Already captured by an active decision; no write required.
    Noop { existing_id: String },
}

/// Result returned after applying the pre-write conflict gate to LanceDB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationOutcome {
    pub action: ConsolidationAction,
    pub committed_id: Option<String>,
    pub invalidated_id: Option<String>,
}

/// Conflict resolver invoked by AllyH's pre-write gate (implemented by local 9B–12B model or heuristic).
#[async_trait]
pub trait ConflictJudge: Send + Sync {
    async fn judge(
        &self,
        candidate: &CandidateDecision,
        existing_active: &[ArchitecturalDecision],
    ) -> Result<ConsolidationAction>;
}

/// Deterministic conflict judge used as a fast offline fallback and in unit/acceptance tests.
#[derive(Debug, Default, Clone)]
pub struct HeuristicConflictJudge;

impl HeuristicConflictJudge {
    fn normalize_tokens(text: &str) -> HashSet<String> {
        text.to_ascii_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
            .filter(|w| w.len() >= 3)
            .map(str::to_string)
            .collect()
    }

    fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }
        let inter = a.intersection(b).count() as f32;
        let union = a.union(b).count() as f32;
        if union == 0.0 { 0.0 } else { inter / union }
    }
}

#[async_trait]
impl ConflictJudge for HeuristicConflictJudge {
    async fn judge(
        &self,
        candidate: &CandidateDecision,
        existing_active: &[ArchitecturalDecision],
    ) -> Result<ConsolidationAction> {
        let cand_topic = candidate.topic.trim().to_ascii_lowercase();
        let cand_rule = candidate.rule_statement.trim().to_ascii_lowercase();
        let cand_tokens = Self::normalize_tokens(&candidate.rule_statement);

        let is_pure_deprecation = (cand_rule.starts_with("remove rule")
            || cand_rule.starts_with("deprecate ")
            || cand_rule.starts_with("invalidate "))
            && !cand_rule.contains("instead");

        for existing in existing_active {
            let ex_topic = existing.topic.trim().to_ascii_lowercase();
            let ex_rule = existing.rule_statement.trim().to_ascii_lowercase();

            if ex_rule == cand_rule {
                return Ok(ConsolidationAction::Noop {
                    existing_id: existing.id.clone(),
                });
            }

            let ex_tokens = Self::normalize_tokens(&existing.rule_statement);
            let overlap = Self::jaccard(&cand_tokens, &ex_tokens);

            if overlap > 0.85 && ex_topic == cand_topic {
                return Ok(ConsolidationAction::Noop {
                    existing_id: existing.id.clone(),
                });
            }

            let same_topic = !cand_topic.is_empty() && cand_topic == ex_topic;
            let overlapping_paths = !candidate.affected_paths.is_empty()
                && candidate
                    .affected_paths
                    .iter()
                    .any(|p| existing.affected_paths.iter().any(|ep| ep == p));

            if same_topic || (overlapping_paths && overlap >= 0.25) || overlap >= 0.45 {
                if is_pure_deprecation {
                    return Ok(ConsolidationAction::Invalidate {
                        target_id: existing.id.clone(),
                        reason: candidate.rule_statement.clone(),
                    });
                }
                return Ok(ConsolidationAction::Update {
                    target_id: existing.id.clone(),
                });
            }
        }

        Ok(ConsolidationAction::Add)
    }
}

/// Write-gated bi-temporal consolidator enforcing pre-write hybrid conflict resolution.
#[derive(Clone)]
pub struct MemoryConsolidator {
    store: MemoryStore,
    judge: Arc<dyn ConflictJudge>,
}

impl MemoryConsolidator {
    pub fn new(store: MemoryStore, judge: Arc<dyn ConflictJudge>) -> Self {
        Self { store, judge }
    }

    pub fn store(&self) -> &MemoryStore {
        &self.store
    }

    /// Execute the pre-write conflict gate and atomically commit `ADD`, `UPDATE`, `INVALIDATE`, or `NOOP`.
    pub async fn consolidate_decision(
        &self,
        repo_id: &str,
        scope_id: &str,
        candidate: CandidateDecision,
    ) -> Result<ConsolidationOutcome> {
        let query_text = format!("{} {}", candidate.topic, candidate.rule_statement);

        // 1. Pre-write hybrid search over active decisions (`invalid_at IS NULL`).
        let mut active_candidates: Vec<ArchitecturalDecision> = self
            .store
            .hybrid_search_decisions(repo_id, scope_id, &query_text, 8)
            .await?
            .into_iter()
            .map(|s| s.decision)
            .collect();

        // Also include exact topic matches even if vector rank was low.
        let all_active = self.store.list_active_decisions(repo_id, scope_id).await?;
        let cand_topic_lower = candidate.topic.trim().to_ascii_lowercase();
        for dec in all_active {
            if dec.topic.trim().to_ascii_lowercase() == cand_topic_lower
                && !active_candidates.iter().any(|existing| existing.id == dec.id)
            {
                active_candidates.push(dec);
            }
        }

        let vector = self
            .store
            .embedder()
            .embed(&format!(
                "{}: {} ({})",
                candidate.topic, candidate.rule_statement, candidate.rationale
            ))
            .await?;

        // Check near-identical embedding + identical rule short-circuit.
        for existing in &active_candidates {
            if existing.rule_statement.trim().eq_ignore_ascii_case(candidate.rule_statement.trim())
                && cosine_similarity(&vector, &existing.vector) > 0.98
            {
                return Ok(ConsolidationOutcome {
                    action: ConsolidationAction::Noop {
                        existing_id: existing.id.clone(),
                    },
                    committed_id: None,
                    invalidated_id: None,
                });
            }
        }

        // 2. Ask the conflict judge (local 9B-12B model or heuristic).
        let action = self.judge.judge(&candidate, &active_candidates).await?;
        let now = Utc::now();

        match &action {
            ConsolidationAction::Noop { .. } => Ok(ConsolidationOutcome {
                action,
                committed_id: None,
                invalidated_id: None,
            }),
            ConsolidationAction::Invalidate { target_id, .. } => {
                self.store
                    .invalidate_decision(repo_id, scope_id, target_id, None, now)
                    .await?;
                Ok(ConsolidationOutcome {
                    invalidated_id: Some(target_id.clone()),
                    committed_id: None,
                    action,
                })
            }
            ConsolidationAction::Update { target_id } => {
                let new_id = Uuid::new_v4().to_string();
                let new_decision = ArchitecturalDecision {
                    id: new_id.clone(),
                    repo_id: repo_id.to_string(),
                    scope_id: scope_id.to_string(),
                    topic: candidate.topic,
                    rule_statement: candidate.rule_statement,
                    rationale: candidate.rationale,
                    affected_paths: candidate.affected_paths,
                    vector,
                    valid_at: now,
                    invalid_at: None,
                    superseded_by: None,
                };
                // First logically invalidate the superseded record, then upsert the replacement.
                self.store
                    .invalidate_decision(repo_id, scope_id, target_id, Some(&new_id), now)
                    .await?;
                self.store.upsert_decision(&new_decision).await?;
                Ok(ConsolidationOutcome {
                    invalidated_id: Some(target_id.clone()),
                    committed_id: Some(new_id),
                    action,
                })
            }
            ConsolidationAction::Add => {
                let new_id = Uuid::new_v4().to_string();
                let new_decision = ArchitecturalDecision {
                    id: new_id.clone(),
                    repo_id: repo_id.to_string(),
                    scope_id: scope_id.to_string(),
                    topic: candidate.topic,
                    rule_statement: candidate.rule_statement,
                    rationale: candidate.rationale,
                    affected_paths: candidate.affected_paths,
                    vector,
                    valid_at: now,
                    invalid_at: None,
                    superseded_by: None,
                };
                self.store.upsert_decision(&new_decision).await?;
                Ok(ConsolidationOutcome {
                    committed_id: Some(new_id),
                    invalidated_id: None,
                    action,
                })
            }
        }
    }
}

/// Live status reported by the asynchronous sleep-time compute worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerStatus {
    Idle,
    Consolidating,
    Optimizing,
}

impl std::fmt::Display for WorkerStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => f.write_str("Idle"),
            Self::Consolidating => f.write_str("Consolidating"),
            Self::Optimizing => f.write_str("Optimizing"),
        }
    }
}

/// Asynchronous jobs queued to the read-only sleep-time compute worker.
#[derive(Debug, Clone)]
pub enum SleepJob {
    ConsolidateCandidates {
        repo_id: String,
        scope_id: String,
        candidates: Vec<CandidateDecision>,
    },
    OptimizeTables,
    Shutdown,
}

/// Asynchronous sleep-time compute worker (UC Berkeley / Letta principle).
///
/// Holds ONLY a [`MemoryConsolidator`] (no mutable filesystem handles), distilling
/// architectural invariants and compacting LanceDB fragments during idle intervals.
pub struct SleepTimeWorkerHandle {
    tx: mpsc::UnboundedSender<SleepJob>,
    status_rx: watch::Receiver<WorkerStatus>,
}

impl SleepTimeWorkerHandle {
    /// Spawn the background sleep-time worker with an idle compaction interval.
    pub fn spawn(consolidator: MemoryConsolidator, idle_optimize_interval: Duration) -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel::<SleepJob>();
        let (status_tx, status_rx) = watch::channel(WorkerStatus::Idle);

        tokio::spawn(async move {
            let mut dirty_writes: usize = 0;
            loop {
                let next_job = tokio::time::timeout(idle_optimize_interval, rx.recv()).await;
                match next_job {
                    Ok(Some(SleepJob::ConsolidateCandidates {
                        repo_id,
                        scope_id,
                        candidates,
                    })) => {
                        let _ = status_tx.send(WorkerStatus::Consolidating);
                        for candidate in candidates {
                            match consolidator
                                .consolidate_decision(&repo_id, &scope_id, candidate)
                                .await
                            {
                                Ok(outcome) => {
                                    if outcome.committed_id.is_some()
                                        || outcome.invalidated_id.is_some()
                                    {
                                        dirty_writes += 1;
                                    }
                                }
                                Err(err) => {
                                    tracing::warn!(error = %err, "sleep-time decision consolidation failed");
                                }
                            }
                        }
                        let _ = status_tx.send(WorkerStatus::Idle);
                    }
                    Ok(Some(SleepJob::OptimizeTables)) => {
                        let _ = status_tx.send(WorkerStatus::Optimizing);
                        if let Err(err) = consolidator.store().optimize_all().await {
                            tracing::warn!(error = %err, "LanceDB optimize_all failed");
                        }
                        dirty_writes = 0;
                        let _ = status_tx.send(WorkerStatus::Idle);
                    }
                    Ok(Some(SleepJob::Shutdown)) | Ok(None) => {
                        break;
                    }
                    Err(_idle_elapsed) => {
                        // User idle timeout elapsed: compact Lance fragments if writes occurred.
                        if dirty_writes > 0 {
                            let _ = status_tx.send(WorkerStatus::Optimizing);
                            if let Err(err) = consolidator.store().optimize_all().await {
                                tracing::warn!(error = %err, "idle LanceDB optimize failed");
                            }
                            dirty_writes = 0;
                            let _ = status_tx.send(WorkerStatus::Idle);
                        }
                    }
                }
            }
        });

        Self { tx, status_rx }
    }

    pub fn enqueue(&self, job: SleepJob) {
        let _ = self.tx.send(job);
    }

    pub fn status(&self) -> WorkerStatus {
        self.status_rx.borrow().clone()
    }

    pub fn subscribe_status(&self) -> watch::Receiver<WorkerStatus> {
        self.status_rx.clone()
    }
}
