use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::RecordBatchIterator;
use chrono::{DateTime, Utc};
use futures::TryStreamExt;
use lancedb::index::Index;
use lancedb::index::scalar::{BTreeIndexBuilder, FtsIndexBuilder, FullTextSearchQuery};
use lancedb::index::vector::IvfPqIndexBuilder;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::OptimizeAction;
use lancedb::{Connection, DistanceType, Table};
use rig_core::DynModel;
use rig_core::operation::Embedding;
use rig_lancedb::{LanceDbVectorIndex, SearchParams};

use crate::config::{StorageConfig, escape_sql_literal};
use crate::embeddings::{EmbeddingProvider, cosine_similarity};
use crate::error::{MemoryError, Result};
use crate::schema::{
    ArchitecturalDecision, CodeGraphSnapshot, DbStatusReport, ScoredDecision, SessionRecord,
    TABLE_ARCHITECTURAL_DECISIONS, TABLE_CODE_GRAPH_SNAPSHOTS, TABLE_SESSIONS, TABLE_TASKS,
    TaskRecord, TaskStatus, VerificationProof, architectural_decisions_schema, batch_to_decisions,
    batch_to_sessions, batch_to_snapshots, batch_to_tasks, code_graph_snapshots_schema,
    decisions_to_batch, sessions_schema, sessions_to_batch, snapshots_to_batch, tasks_schema,
    tasks_to_batch,
};

/// Minimum row count required by LanceDB before training an IVF-PQ vector index.
const MIN_ROWS_FOR_IVF_INDEX: usize = 256;

/// Out-of-tree LanceDB memory store shared across AllyH sessions and subagents.
#[derive(Clone)]
pub struct MemoryStore {
    config: StorageConfig,
    connection: Connection,
    embedder: Arc<dyn EmbeddingProvider>,
}

impl MemoryStore {
    /// Connect to a local disk directory or shared S3/MinIO/GCS LanceDB URI
    /// and ensure all four AllyH tables and indexes exist.
    pub async fn connect(
        config: StorageConfig,
        embedder: Arc<dyn EmbeddingProvider>,
    ) -> Result<Self> {
        if config.is_local() {
            std::fs::create_dir_all(&config.uri)?;
        }

        let mut builder = lancedb::connect(&config.uri);
        if !config.storage_options.is_empty() {
            builder = builder.storage_options(config.storage_options.clone());
        }
        let connection = builder.execute().await?;

        let store = Self {
            config,
            connection,
            embedder,
        };
        store.ensure_tables().await?;
        Ok(store)
    }

    pub fn config(&self) -> &StorageConfig {
        &self.config
    }

    pub fn embedder(&self) -> &Arc<dyn EmbeddingProvider> {
        &self.embedder
    }

    async fn ensure_tables(&self) -> Result<()> {
        let existing = self.connection.table_names().execute().await?;

        if !existing.iter().any(|n| n == TABLE_SESSIONS) {
            self.connection
                .create_empty_table(TABLE_SESSIONS, sessions_schema())
                .execute()
                .await?;
        }
        if !existing.iter().any(|n| n == TABLE_ARCHITECTURAL_DECISIONS) {
            self.connection
                .create_empty_table(
                    TABLE_ARCHITECTURAL_DECISIONS,
                    architectural_decisions_schema(),
                )
                .execute()
                .await?;
        }
        if !existing.iter().any(|n| n == TABLE_TASKS) {
            self.connection
                .create_empty_table(TABLE_TASKS, tasks_schema())
                .execute()
                .await?;
        }
        if !existing.iter().any(|n| n == TABLE_CODE_GRAPH_SNAPSHOTS) {
            self.connection
                .create_empty_table(TABLE_CODE_GRAPH_SNAPSHOTS, code_graph_snapshots_schema())
                .execute()
                .await?;
        }

        // Best-effort scalar / FTS index initialization.
        if let Err(err) = self.ensure_indexes().await {
            tracing::debug!(error = %err, "deferred index creation until tables are populated");
        }
        Ok(())
    }

    async fn open_table(&self, name: &str) -> Result<Table> {
        Ok(self.connection.open_table(name).execute().await?)
    }

    /// Create BTree scalar indexes on `repo_id` and `invalid_at`, FTS index on `rule_statement`,
    /// and IVF-PQ vector index once tables exceed the training threshold.
    pub async fn ensure_indexes(&self) -> Result<()> {
        for table_name in [
            TABLE_SESSIONS,
            TABLE_ARCHITECTURAL_DECISIONS,
            TABLE_TASKS,
            TABLE_CODE_GRAPH_SNAPSHOTS,
        ] {
            let table = self.open_table(table_name).await?;
            let count = table.count_rows(None).await?;
            if count == 0 {
                continue;
            }
            let _ = table
                .create_index(&["repo_id"], Index::BTree(BTreeIndexBuilder::default()))
                .replace(true)
                .execute()
                .await;

            if table_name == TABLE_ARCHITECTURAL_DECISIONS {
                let _ = table
                    .create_index(&["invalid_at"], Index::BTree(BTreeIndexBuilder::default()))
                    .replace(true)
                    .execute()
                    .await;
                let _ = table
                    .create_index(&["rule_statement"], Index::FTS(FtsIndexBuilder::default()))
                    .replace(true)
                    .execute()
                    .await;
                if count >= MIN_ROWS_FOR_IVF_INDEX {
                    let _ = table
                        .create_index(
                            &["vector"],
                            Index::IvfPq(
                                IvfPqIndexBuilder::default().distance_type(DistanceType::Cosine),
                            ),
                        )
                        .replace(true)
                        .execute()
                        .await;
                }
            }
        }
        Ok(())
    }

    /// Expose the `architectural_decisions` table through `rig-lancedb`'s [`LanceDbVectorIndex`].
    pub async fn rig_decisions_vector_index(
        &self,
        embedding_model: impl Into<DynModel<Embedding>>,
    ) -> Result<LanceDbVectorIndex> {
        let table = self.open_table(TABLE_ARCHITECTURAL_DECISIONS).await?;
        let params = SearchParams::default()
            .distance_type(DistanceType::Cosine)
            .column("vector");
        Ok(LanceDbVectorIndex::new(table, embedding_model, "id", params).await?)
    }

    // -----------------------------------------------------------------------
    // Sessions
    // -----------------------------------------------------------------------

    pub async fn upsert_session(&self, record: &SessionRecord) -> Result<()> {
        let table = self.open_table(TABLE_SESSIONS).await?;
        let batch = sessions_to_batch(std::slice::from_ref(record))?;
        let schema = batch.schema();
        let reader = Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));
        let mut merge = table.merge_insert(&["session_id"]);
        merge
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        merge.execute(reader).await?;
        Ok(())
    }

    pub async fn list_sessions(&self, repo_id: &str, scope_id: &str) -> Result<Vec<SessionRecord>> {
        let table = self.open_table(TABLE_SESSIONS).await?;
        let filter = format!(
            "repo_id = '{}' AND scope_id = '{}'",
            escape_sql_literal(repo_id),
            escape_sql_literal(scope_id)
        );
        let batches = table
            .query()
            .only_if(filter)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let mut sessions = Vec::new();
        for batch in &batches {
            sessions.extend(batch_to_sessions(batch)?);
        }
        sessions.sort_by(|a, b| b.started_at.cmp(&a.started_at));
        Ok(sessions)
    }

    pub async fn latest_session(
        &self,
        repo_id: &str,
        scope_id: &str,
    ) -> Result<Option<SessionRecord>> {
        Ok(self.list_sessions(repo_id, scope_id).await?.into_iter().next())
    }

    // -----------------------------------------------------------------------
    // Architectural Decisions (Bi-Temporal + Hybrid Search)
    // -----------------------------------------------------------------------

    pub async fn upsert_decision(&self, decision: &ArchitecturalDecision) -> Result<()> {
        let table = self.open_table(TABLE_ARCHITECTURAL_DECISIONS).await?;
        let batch = decisions_to_batch(std::slice::from_ref(decision))?;
        let schema = batch.schema();
        let reader = Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));
        let mut merge = table.merge_insert(&["id"]);
        merge
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        merge.execute(reader).await?;

        // Keep FTS index fresh for hybrid search.
        let _ = table
            .create_index(&["rule_statement"], Index::FTS(FtsIndexBuilder::default()))
            .replace(true)
            .execute()
            .await;
        Ok(())
    }

    /// Logically invalidate an existing decision by setting `invalid_at` (and optional `superseded_by`).
    /// Never hard-deletes so full bi-temporal history is preserved.
    pub async fn invalidate_decision(
        &self,
        repo_id: &str,
        scope_id: &str,
        decision_id: &str,
        superseded_by: Option<&str>,
        invalidated_at: DateTime<Utc>,
    ) -> Result<u64> {
        let table = self.open_table(TABLE_ARCHITECTURAL_DECISIONS).await?;
        let filter = format!(
            "repo_id = '{}' AND scope_id = '{}' AND id = '{}' AND invalid_at IS NULL",
            escape_sql_literal(repo_id),
            escape_sql_literal(scope_id),
            escape_sql_literal(decision_id)
        );
        let ts_expr = format!("CAST({} AS TIMESTAMP)", invalidated_at.timestamp_micros());
        let mut updater = table.update().only_if(filter).column("invalid_at", ts_expr);
        if let Some(new_id) = superseded_by {
            updater = updater.column(
                "superseded_by",
                format!("'{}'", escape_sql_literal(new_id)),
            );
        }
        let res = updater.execute().await?;
        Ok(res.rows_updated)
    }

    /// Return all active (`invalid_at IS NULL`) decisions for a repository/scope partition.
    pub async fn list_active_decisions(
        &self,
        repo_id: &str,
        scope_id: &str,
    ) -> Result<Vec<ArchitecturalDecision>> {
        let table = self.open_table(TABLE_ARCHITECTURAL_DECISIONS).await?;
        let filter = format!(
            "repo_id = '{}' AND scope_id = '{}' AND invalid_at IS NULL",
            escape_sql_literal(repo_id),
            escape_sql_literal(scope_id)
        );
        let batches = table
            .query()
            .only_if(filter)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let mut decisions = Vec::new();
        for batch in &batches {
            decisions.extend(batch_to_decisions(batch)?);
        }
        decisions.sort_by(|a, b| b.valid_at.cmp(&a.valid_at));
        Ok(decisions)
    }

    /// Return all decisions (both active and superseded/invalidated) for audit and verification.
    pub async fn list_all_decisions(
        &self,
        repo_id: &str,
        scope_id: &str,
    ) -> Result<Vec<ArchitecturalDecision>> {
        let table = self.open_table(TABLE_ARCHITECTURAL_DECISIONS).await?;
        let filter = format!(
            "repo_id = '{}' AND scope_id = '{}'",
            escape_sql_literal(repo_id),
            escape_sql_literal(scope_id)
        );
        let batches = table
            .query()
            .only_if(filter)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let mut decisions = Vec::new();
        for batch in &batches {
            decisions.extend(batch_to_decisions(batch)?);
        }
        decisions.sort_by(|a, b| b.valid_at.cmp(&a.valid_at));
        Ok(decisions)
    }

    /// Hybrid search (`vector` nearest-neighbor + BM25 FTS / lexical fusion via RRF)
    /// strictly filtered by `repo_id = '<id>' AND scope_id = '<scope>' AND invalid_at IS NULL`.
    pub async fn hybrid_search_decisions(
        &self,
        repo_id: &str,
        scope_id: &str,
        query_text: &str,
        top_k: usize,
    ) -> Result<Vec<ScoredDecision>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let table = self.open_table(TABLE_ARCHITECTURAL_DECISIONS).await?;
        let filter = format!(
            "repo_id = '{}' AND scope_id = '{}' AND invalid_at IS NULL",
            escape_sql_literal(repo_id),
            escape_sql_literal(scope_id)
        );

        let query_vec = self.embedder.embed(query_text).await?;
        let candidate_limit = (top_k * 4).max(20);

        // 1. Vector similarity search via LanceDB.
        let mut vec_candidates: Vec<ArchitecturalDecision> = Vec::new();
        let vec_stream = table
            .query()
            .only_if(&filter)
            .nearest_to(query_vec.as_slice())?
            .distance_type(DistanceType::Cosine)
            .limit(candidate_limit)
            .execute()
            .await;

        if let Ok(stream) = vec_stream {
            let batches = stream.try_collect::<Vec<_>>().await?;
            for batch in &batches {
                vec_candidates.extend(batch_to_decisions(batch)?);
            }
        }

        // 2. Full-text search (BM25) via LanceDB FTS index (gracefully degrades if index absent).
        let mut fts_candidates: Vec<ArchitecturalDecision> = Vec::new();
        if !query_text.trim().is_empty() {
            let fts_res = table
                .query()
                .only_if(&filter)
                .full_text_search(FullTextSearchQuery::new(query_text.to_string()))
                .limit(candidate_limit)
                .execute()
                .await;
            if let Ok(stream) = fts_res
                && let Ok(batches) = stream.try_collect::<Vec<_>>().await
            {
                for batch in &batches {
                    if let Ok(rows) = batch_to_decisions(batch) {
                        fts_candidates.extend(rows);
                    }
                }
            }
        }

        // 3. Reciprocal Rank Fusion (RRF) + exact cosine & token overlap boost.
        const RRF_K: f32 = 60.0;
        let mut fused: HashMap<String, (ArchitecturalDecision, f32)> = HashMap::new();

        for (rank, decision) in vec_candidates.into_iter().enumerate() {
            let sim = cosine_similarity(&query_vec, &decision.vector).max(0.0);
            let rrf = 1.0 / (RRF_K + rank as f32 + 1.0);
            let score = rrf + 0.5 * sim;
            fused
                .entry(decision.id.clone())
                .and_modify(|(_, s)| *s += score)
                .or_insert((decision, score));
        }

        for (rank, decision) in fts_candidates.into_iter().enumerate() {
            let rrf = 1.2 / (RRF_K + rank as f32 + 1.0);
            fused
                .entry(decision.id.clone())
                .and_modify(|(_, s)| *s += rrf)
                .or_insert((decision, rrf));
        }

        // Lexical topic/path boost for small repositories where FTS hasn't indexed a token yet.
        let query_terms: Vec<String> = query_text
            .to_ascii_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
            .filter(|t| t.len() >= 2)
            .map(str::to_string)
            .collect();

        let mut scored: Vec<ScoredDecision> = fused
            .into_values()
            .map(|(decision, mut score)| {
                if !query_terms.is_empty() {
                    let haystack = format!(
                        "{} {} {} {}",
                        decision.topic,
                        decision.rule_statement,
                        decision.rationale,
                        decision.affected_paths.join(" ")
                    )
                    .to_ascii_lowercase();
                    let hits = query_terms
                        .iter()
                        .filter(|term| haystack.contains(term.as_str()))
                        .count();
                    score += (hits as f32 / query_terms.len() as f32) * 0.35;
                }
                ScoredDecision {
                    decision,
                    score,
                    pinned: false,
                }
            })
            .collect();

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        Ok(scored)
    }

    // -----------------------------------------------------------------------
    // Tasks (Structured Execution Plan)
    // -----------------------------------------------------------------------

    pub async fn upsert_task(&self, task: &TaskRecord) -> Result<()> {
        let table = self.open_table(TABLE_TASKS).await?;
        let batch = tasks_to_batch(std::slice::from_ref(task))?;
        let schema = batch.schema();
        let reader = Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));
        let mut merge = table.merge_insert(&["id"]);
        merge
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        merge.execute(reader).await?;
        Ok(())
    }

    pub async fn list_tasks(&self, repo_id: &str, scope_id: &str) -> Result<Vec<TaskRecord>> {
        let table = self.open_table(TABLE_TASKS).await?;
        let filter = format!(
            "repo_id = '{}' AND scope_id = '{}'",
            escape_sql_literal(repo_id),
            escape_sql_literal(scope_id)
        );
        let batches = table
            .query()
            .only_if(filter)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let mut tasks = Vec::new();
        for batch in &batches {
            tasks.extend(batch_to_tasks(batch)?);
        }
        tasks.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(tasks)
    }

    /// Enforce strict task state machine rules:
    /// a task CANNOT transition to `TaskStatus::Verified` without a [`VerificationProof`].
    pub async fn transition_task_status(
        &self,
        repo_id: &str,
        scope_id: &str,
        task_id: &str,
        new_status: TaskStatus,
        proof: Option<&VerificationProof>,
        last_error_trace: Option<String>,
    ) -> Result<TaskRecord> {
        if new_status == TaskStatus::Verified && proof.is_none() {
            return Err(MemoryError::InvalidTransition(
                "task cannot transition to 'verified' without passing automated verification (cargo check/test)".to_string(),
            ));
        }

        let tasks = self.list_tasks(repo_id, scope_id).await?;
        let mut task = tasks
            .into_iter()
            .find(|t| t.id == task_id)
            .ok_or_else(|| MemoryError::NotFound(format!("task '{task_id}'")))?;

        task.status = new_status;
        task.last_error_trace = last_error_trace;
        task.updated_at = Utc::now();
        self.upsert_task(&task).await?;
        Ok(task)
    }

    // -----------------------------------------------------------------------
    // Code Graph Snapshots (RepoGraph + File Integrity)
    // -----------------------------------------------------------------------

    pub async fn upsert_code_snapshot(&self, snapshot: &CodeGraphSnapshot) -> Result<()> {
        let table = self.open_table(TABLE_CODE_GRAPH_SNAPSHOTS).await?;
        let batch = snapshots_to_batch(std::slice::from_ref(snapshot))?;
        let schema = batch.schema();
        let reader = Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));
        let mut merge = table.merge_insert(&["repo_id", "scope_id", "file_path"]);
        merge
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        merge.execute(reader).await?;
        Ok(())
    }

    pub async fn list_code_snapshots(
        &self,
        repo_id: &str,
        scope_id: &str,
    ) -> Result<Vec<CodeGraphSnapshot>> {
        let table = self.open_table(TABLE_CODE_GRAPH_SNAPSHOTS).await?;
        let filter = format!(
            "repo_id = '{}' AND scope_id = '{}'",
            escape_sql_literal(repo_id),
            escape_sql_literal(scope_id)
        );
        let batches = table
            .query()
            .only_if(filter)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let mut snapshots = Vec::new();
        for batch in &batches {
            snapshots.extend(batch_to_snapshots(batch)?);
        }
        snapshots.sort_by(|a, b| a.file_path.cmp(&b.file_path));
        Ok(snapshots)
    }

    pub async fn get_code_snapshot(
        &self,
        repo_id: &str,
        scope_id: &str,
        file_path: &str,
    ) -> Result<Option<CodeGraphSnapshot>> {
        let table = self.open_table(TABLE_CODE_GRAPH_SNAPSHOTS).await?;
        let filter = format!(
            "repo_id = '{}' AND scope_id = '{}' AND file_path = '{}'",
            escape_sql_literal(repo_id),
            escape_sql_literal(scope_id),
            escape_sql_literal(file_path)
        );
        let batches = table
            .query()
            .only_if(filter)
            .limit(1)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        for batch in &batches {
            let mut rows = batch_to_snapshots(batch)?;
            if let Some(first) = rows.pop() {
                return Ok(Some(first));
            }
        }
        Ok(None)
    }

    pub async fn delete_code_snapshot(
        &self,
        repo_id: &str,
        scope_id: &str,
        file_path: &str,
    ) -> Result<()> {
        let table = self.open_table(TABLE_CODE_GRAPH_SNAPSHOTS).await?;
        let filter = format!(
            "repo_id = '{}' AND scope_id = '{}' AND file_path = '{}'",
            escape_sql_literal(repo_id),
            escape_sql_literal(scope_id),
            escape_sql_literal(file_path)
        );
        table.delete(&filter).await?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Maintenance & Status
    // -----------------------------------------------------------------------

    /// Compact Lance fragments and optimize indexes across all AllyH tables.
    pub async fn optimize_all(&self) -> Result<()> {
        for table_name in [
            TABLE_SESSIONS,
            TABLE_ARCHITECTURAL_DECISIONS,
            TABLE_TASKS,
            TABLE_CODE_GRAPH_SNAPSHOTS,
        ] {
            let table = self.open_table(table_name).await?;
            if table.count_rows(None).await? > 0 {
                let _ = table.optimize(OptimizeAction::All).await;
            }
        }
        let _ = self.ensure_indexes().await;
        Ok(())
    }

    /// Collect live row counts and backend metadata for `/db-status`.
    pub async fn status_report(&self, repo_id: &str, scope_id: &str) -> Result<DbStatusReport> {
        let sessions = self.list_sessions(repo_id, scope_id).await?;
        let all_decisions = self.list_all_decisions(repo_id, scope_id).await?;
        let active_decisions_count = all_decisions
            .iter()
            .filter(|d| d.invalid_at.is_none())
            .count();
        let invalidated_decisions_count = all_decisions.len() - active_decisions_count;
        let tasks = self.list_tasks(repo_id, scope_id).await?;
        let snapshots = self.list_code_snapshots(repo_id, scope_id).await?;

        Ok(DbStatusReport {
            uri: self.config.uri.clone(),
            backend_label: self.config.backend_kind().display_label().to_string(),
            repo_id: repo_id.to_string(),
            scope_id: scope_id.to_string(),
            sessions_count: sessions.len(),
            active_decisions_count,
            invalidated_decisions_count,
            tasks_count: tasks.len(),
            code_snapshots_count: snapshots.len(),
        })
    }
}
