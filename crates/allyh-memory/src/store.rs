use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{RecordBatchIterator, RecordBatchReader};
use chrono::{DateTime, Utc};
use futures::TryStreamExt;
use lancedb::index::Index;
use lancedb::index::scalar::{BTreeIndexBuilder, FtsIndexBuilder, FullTextSearchQuery};
use lancedb::index::vector::IvfPqIndexBuilder;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::{NewColumnTransform, OptimizeAction};
use lancedb::{Connection, DistanceType, Table};
use rig_core::DynModel;
use rig_core::operation::Embedding;
use rig_lancedb::{LanceDbVectorIndex, SearchParams};

use crate::config::{StorageConfig, escape_sql_literal};
use crate::embeddings::{EmbeddingProvider, cosine_similarity};
use crate::error::{MemoryError, Result};
use crate::schema::{
    ArchitecturalDecision, CodeGraphSnapshot, DbStatusReport, HarnessVersionRecord, RepoProfile,
    ScoredDecision, SessionRecord, SimRunRecord, SkillRecord, TABLE_ARCHITECTURAL_DECISIONS,
    TABLE_CODE_GRAPH_SNAPSHOTS, TABLE_HARNESS_VERSIONS, TABLE_REPO_PROFILES, TABLE_SESSIONS,
    TABLE_SIM_ATTEMPTS, TABLE_SIM_PROBLEMS, TABLE_SIM_RUNS, TABLE_SKILLS, TABLE_TASKS, TaskRecord,
    TaskStatus, VerificationProof, architectural_decisions_schema, batch_to_decisions,
    batch_to_harness_versions, batch_to_repo_profiles, batch_to_sessions, batch_to_sim_attempts,
    batch_to_sim_problems, batch_to_sim_runs, batch_to_skills, batch_to_snapshots, batch_to_tasks,
    code_graph_snapshots_schema, decisions_to_batch, harness_versions_schema,
    harness_versions_to_batch, normalize_skill_slug, repo_profile_id, repo_profiles_schema,
    repo_profiles_to_batch, sessions_schema, sessions_to_batch, sim_attempts_schema,
    sim_attempts_to_batch, sim_problems_schema, sim_problems_to_batch, sim_runs_schema,
    sim_runs_to_batch, skill_id, skills_schema, skills_to_batch, snapshots_to_batch, tasks_schema,
    tasks_to_batch,
};
use crate::sim_ledger::{SimExportDocument, SimLedger};

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
        } else if let Ok(table) = self.open_table(TABLE_SESSIONS).await
            && let Ok(schema) = table.schema().await
        {
            let mut missing_cols = Vec::new();
            for (col, expr) in [
                ("input_tokens", "CAST(0 AS BIGINT UNSIGNED)"),
                ("output_tokens", "CAST(0 AS BIGINT UNSIGNED)"),
                ("cached_input_tokens", "CAST(0 AS BIGINT UNSIGNED)"),
                ("cache_creation_tokens", "CAST(0 AS BIGINT UNSIGNED)"),
                ("estimated_cost_usd", "CAST(0.0 AS DOUBLE)"),
            ] {
                if schema.field_with_name(col).is_err() {
                    missing_cols.push((col.to_string(), expr.to_string()));
                }
            }
            if !missing_cols.is_empty() {
                let _ = table
                    .add_columns(NewColumnTransform::SqlExpressions(missing_cols), None)
                    .await;
            }
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
        } else {
            // Seamlessly migrate pre-v0.3.0 `code_graph_snapshots` tables lacking `language`.
            if let Ok(table) = self.open_table(TABLE_CODE_GRAPH_SNAPSHOTS).await
                && let Ok(schema) = table.schema().await
                && schema.field_with_name("language").is_err()
            {
                let _ = table
                    .add_columns(
                        NewColumnTransform::SqlExpressions(vec![(
                            "language".to_string(),
                            "'unknown'".to_string(),
                        )]),
                        None,
                    )
                    .await;
            }
        }
        if !existing.iter().any(|n| n == TABLE_SKILLS) {
            self.connection
                .create_empty_table(TABLE_SKILLS, skills_schema())
                .execute()
                .await?;
        }
        if !existing.iter().any(|n| n == TABLE_REPO_PROFILES) {
            self.connection
                .create_empty_table(TABLE_REPO_PROFILES, repo_profiles_schema())
                .execute()
                .await?;
        } else if let Ok(table) = self.open_table(TABLE_REPO_PROFILES).await
            && let Ok(schema) = table.schema().await
            && schema.field_with_name("mcp_servers_json").is_err()
        {
            let _ = table
                .add_columns(
                    NewColumnTransform::SqlExpressions(vec![(
                        "mcp_servers_json".to_string(),
                        "'[]'".to_string(),
                    )]),
                    None,
                )
                .await;
        }
        if !existing.iter().any(|n| n == TABLE_HARNESS_VERSIONS) {
            self.connection
                .create_empty_table(TABLE_HARNESS_VERSIONS, harness_versions_schema())
                .execute()
                .await?;
        }
        if !existing.iter().any(|n| n == TABLE_SIM_RUNS) {
            self.connection
                .create_empty_table(TABLE_SIM_RUNS, sim_runs_schema())
                .execute()
                .await?;
        }
        if !existing.iter().any(|n| n == TABLE_SIM_PROBLEMS) {
            self.connection
                .create_empty_table(TABLE_SIM_PROBLEMS, sim_problems_schema())
                .execute()
                .await?;
        }
        if !existing.iter().any(|n| n == TABLE_SIM_ATTEMPTS) {
            self.connection
                .create_empty_table(TABLE_SIM_ATTEMPTS, sim_attempts_schema())
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
            TABLE_SKILLS,
            TABLE_REPO_PROFILES,
            TABLE_HARNESS_VERSIONS,
            TABLE_SIM_RUNS,
            TABLE_SIM_PROBLEMS,
            TABLE_SIM_ATTEMPTS,
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

            if matches!(
                table_name,
                TABLE_SIM_RUNS | TABLE_SIM_PROBLEMS | TABLE_SIM_ATTEMPTS
            ) {
                let _ = table
                    .create_index(&["run_id"], Index::BTree(BTreeIndexBuilder::default()))
                    .replace(true)
                    .execute()
                    .await;
            }

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
                "task cannot transition to 'verified' without passing automated verification (check/lint/test)".to_string(),
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
    // Skills (Reusable Procedural Knowledge)
    // -----------------------------------------------------------------------

    /// Insert or update a reusable skill, keyed by its stable `id`.
    pub async fn upsert_skill(&self, skill: &SkillRecord) -> Result<()> {
        let table = self.open_table(TABLE_SKILLS).await?;
        let batch = skills_to_batch(std::slice::from_ref(skill))?;
        let schema = batch.schema();
        let reader = Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));
        let mut merge = table.merge_insert(&["id"]);
        merge
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        merge.execute(reader).await?;
        Ok(())
    }

    /// Validate, normalize, and upsert a reusable skill, preserving `created_at` on updates.
    /// Returns `(record, created)`.
    pub async fn save_skill(
        &self,
        repo_id: &str,
        scope_id: &str,
        name: &str,
        description: &str,
        instructions: &str,
        triggers: Vec<String>,
    ) -> Result<(SkillRecord, bool)> {
        let name = name.trim();
        if name.is_empty() {
            return Err(MemoryError::InvalidSkill(
                "skill 'name' must not be empty".to_string(),
            ));
        }
        let instructions = instructions.trim();
        if instructions.is_empty() {
            return Err(MemoryError::InvalidSkill(
                "skill 'instructions' must not be empty".to_string(),
            ));
        }

        let id = skill_id(repo_id, scope_id, name)?;
        let mut cleaned_triggers = Vec::new();
        for trigger in triggers {
            let t = trigger.trim();
            if !t.is_empty() && !cleaned_triggers.iter().any(|existing: &String| existing.eq_ignore_ascii_case(t)) {
                cleaned_triggers.push(t.to_string());
            }
        }

        let existing = self.get_skill(repo_id, scope_id, &id).await?;
        let raw_now = Utc::now();
        let now = DateTime::from_timestamp_micros(raw_now.timestamp_micros()).unwrap_or(raw_now);
        let created_at = existing.as_ref().map(|s| s.created_at).unwrap_or(now);
        let created = existing.is_none();

        let record = SkillRecord {
            id,
            repo_id: repo_id.to_string(),
            scope_id: scope_id.to_string(),
            name: name.to_string(),
            description: description.trim().to_string(),
            instructions: instructions.to_string(),
            triggers: cleaned_triggers,
            created_at,
            updated_at: now,
        };
        self.upsert_skill(&record).await?;
        Ok((record, created))
    }

    /// List all skills registered for a repo/scope, newest-updated first.
    pub async fn list_skills(&self, repo_id: &str, scope_id: &str) -> Result<Vec<SkillRecord>> {
        let table = self.open_table(TABLE_SKILLS).await?;
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

        let mut skills = Vec::new();
        for batch in &batches {
            skills.extend(batch_to_skills(batch)?);
        }
        skills.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(skills)
    }

    /// Look up a single skill within `(repo_id, scope_id)` by exact id, derived slug id,
    /// case-insensitive name, or normalized slug.
    pub async fn get_skill(
        &self,
        repo_id: &str,
        scope_id: &str,
        name_or_id: &str,
    ) -> Result<Option<SkillRecord>> {
        let needle = name_or_id.trim();
        if needle.is_empty() {
            return Ok(None);
        }
        let derived_id = skill_id(repo_id, scope_id, needle).ok();
        let needle_slug = normalize_skill_slug(needle);

        let skills = self.list_skills(repo_id, scope_id).await?;
        Ok(skills.into_iter().find(|s| {
            s.id == needle
                || derived_id.as_deref() == Some(s.id.as_str())
                || s.name.eq_ignore_ascii_case(needle)
                || (!needle_slug.is_empty() && normalize_skill_slug(&s.name) == needle_slug)
        }))
    }

    /// Permanently delete a skill within `(repo_id, scope_id)` by name or id.
    /// Returns the removed record if it existed.
    pub async fn delete_skill(
        &self,
        repo_id: &str,
        scope_id: &str,
        name_or_id: &str,
    ) -> Result<Option<SkillRecord>> {
        let Some(skill) = self.get_skill(repo_id, scope_id, name_or_id).await? else {
            return Ok(None);
        };
        let table = self.open_table(TABLE_SKILLS).await?;
        let filter = format!(
            "repo_id = '{}' AND scope_id = '{}' AND id = '{}'",
            escape_sql_literal(repo_id),
            escape_sql_literal(scope_id),
            escape_sql_literal(&skill.id)
        );
        table.delete(&filter).await?;
        Ok(Some(skill))
    }

    /// Deterministically match skills relevant to `query` using trigger phrases, skill names,
    /// and lexical token overlap. Returns up to `top_k` highest-scoring matches.
    pub async fn match_skills(
        &self,
        repo_id: &str,
        scope_id: &str,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<SkillRecord>> {
        if top_k == 0 || query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let skills = self.list_skills(repo_id, scope_id).await?;
        if skills.is_empty() {
            return Ok(Vec::new());
        }

        let query_lower = query.to_ascii_lowercase();
        let query_tokens: Vec<String> = tokenize_skill_text(&query_lower);

        let mut scored: Vec<(SkillRecord, f32)> = skills
            .into_iter()
            .filter_map(|skill| {
                let score = score_skill_match(&skill, &query_lower, &query_tokens);
                (score >= 1.5).then_some((skill, score))
            })
            .collect();

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.0.updated_at.cmp(&a.0.updated_at))
        });
        scored.truncate(top_k);
        Ok(scored.into_iter().map(|(s, _)| s).collect())
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
    // Repository Profiles (Out-of-Tree Polyglot Toolchain & Ignore Overrides)
    // -----------------------------------------------------------------------

    /// Insert or update an out-of-tree [`RepoProfile`] for `(repo_id, scope_id)`.
    pub async fn upsert_repo_profile(&self, profile: &RepoProfile) -> Result<()> {
        let mut normalized = profile.clone();
        normalized.id = repo_profile_id(&normalized.repo_id, &normalized.scope_id);
        normalized.updated_at = Utc::now();

        let table = self.open_table(TABLE_REPO_PROFILES).await?;
        let batch = repo_profiles_to_batch(std::slice::from_ref(&normalized))?;
        let schema = batch.schema();
        let reader = Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));
        let mut merge = table.merge_insert(&["id"]);
        merge
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        merge.execute(reader).await?;
        Ok(())
    }

    /// Retrieve the stored [`RepoProfile`] for `(repo_id, scope_id)`, if configured.
    pub async fn get_repo_profile(
        &self,
        repo_id: &str,
        scope_id: &str,
    ) -> Result<Option<RepoProfile>> {
        let table = self.open_table(TABLE_REPO_PROFILES).await?;
        let filter = format!(
            "repo_id = '{}' AND scope_id = '{}'",
            escape_sql_literal(repo_id),
            escape_sql_literal(scope_id)
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
            let mut rows = batch_to_repo_profiles(batch)?;
            if let Some(first) = rows.pop() {
                return Ok(Some(first));
            }
        }
        Ok(None)
    }

    /// Delete the stored [`RepoProfile`] for `(repo_id, scope_id)`, restoring auto-detection.
    pub async fn delete_repo_profile(&self, repo_id: &str, scope_id: &str) -> Result<()> {
        let table = self.open_table(TABLE_REPO_PROFILES).await?;
        let filter = format!(
            "repo_id = '{}' AND scope_id = '{}'",
            escape_sql_literal(repo_id),
            escape_sql_literal(scope_id)
        );
        table.delete(&filter).await?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Harness Versions & Self-Improvement Memory (SIM)
    // -----------------------------------------------------------------------

    /// Insert or update a versioned executable harness specification.
    pub async fn upsert_harness_version(&self, record: &HarnessVersionRecord) -> Result<()> {
        let table = self.open_table(TABLE_HARNESS_VERSIONS).await?;
        let batch = harness_versions_to_batch(std::slice::from_ref(record))?;
        let schema = batch.schema();
        let reader = Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));
        let mut merge = table.merge_insert(&["id"]);
        merge
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        merge.execute(reader).await?;
        Ok(())
    }

    /// List all harness versions stored for `(repo_id, scope_id)`, newest first.
    pub async fn list_harness_versions(
        &self,
        repo_id: &str,
        scope_id: &str,
    ) -> Result<Vec<HarnessVersionRecord>> {
        let table = self.open_table(TABLE_HARNESS_VERSIONS).await?;
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

        let mut versions = Vec::new();
        for batch in &batches {
            versions.extend(batch_to_harness_versions(batch)?);
        }
        versions.sort_by_key(|a| std::cmp::Reverse(a.created_at));
        Ok(versions)
    }

    /// Look up a single harness version by exact ID or unique prefix within `(repo_id, scope_id)`.
    pub async fn get_harness_version(
        &self,
        repo_id: &str,
        scope_id: &str,
        version_id_or_prefix: &str,
    ) -> Result<Option<HarnessVersionRecord>> {
        let needle = version_id_or_prefix.trim();
        if needle.is_empty() {
            return Ok(None);
        }
        let versions = self.list_harness_versions(repo_id, scope_id).await?;
        Ok(versions
            .into_iter()
            .find(|v| v.id == needle || v.id.starts_with(needle) || v.label.eq_ignore_ascii_case(needle)))
    }

    /// Return the currently active harness version for `(repo_id, scope_id)`, if any.
    pub async fn get_active_harness_version(
        &self,
        repo_id: &str,
        scope_id: &str,
    ) -> Result<Option<HarnessVersionRecord>> {
        let versions = self.list_harness_versions(repo_id, scope_id).await?;
        Ok(versions.into_iter().find(|v| v.is_active))
    }

    /// Mark one harness version active for `(repo_id, scope_id)` and deactivate all others.
    pub async fn set_active_harness_version(
        &self,
        repo_id: &str,
        scope_id: &str,
        version_id_or_prefix: &str,
    ) -> Result<HarnessVersionRecord> {
        let Some(mut target) = self
            .get_harness_version(repo_id, scope_id, version_id_or_prefix)
            .await?
        else {
            return Err(MemoryError::NotFound(format!(
                "harness version '{version_id_or_prefix}'"
            )));
        };

        let table = self.open_table(TABLE_HARNESS_VERSIONS).await?;
        let deactivate_filter = format!(
            "repo_id = '{}' AND scope_id = '{}' AND is_active = true",
            escape_sql_literal(repo_id),
            escape_sql_literal(scope_id)
        );
        let _ = table
            .update()
            .only_if(deactivate_filter)
            .column("is_active", "false")
            .execute()
            .await?;

        target.is_active = true;
        self.upsert_harness_version(&target).await?;
        Ok(target)
    }

    /// Persist a full [`SimLedger`] (run metadata, problem entries, and ordered attempts)
    /// into LanceDB, keeping problem/attempt rows synchronized after alias merges.
    pub async fn save_sim_ledger(&self, ledger: &SimLedger) -> Result<()> {
        let (run, problems, attempts) = ledger.to_records();

        // 1. Upsert run metadata.
        let runs_table = self.open_table(TABLE_SIM_RUNS).await?;
        let run_batch = sim_runs_to_batch(std::slice::from_ref(&run))?;
        let run_schema = run_batch.schema();
        let run_reader = Box::new(RecordBatchIterator::new(vec![Ok(run_batch)], run_schema));
        let mut run_merge = runs_table.merge_insert(&["run_id"]);
        run_merge
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        run_merge.execute(run_reader).await?;

        let run_filter = format!(
            "repo_id = '{}' AND scope_id = '{}' AND run_id = '{}'",
            escape_sql_literal(&run.repo_id),
            escape_sql_literal(&run.scope_id),
            escape_sql_literal(&run.run_id)
        );

        // 2. Replace problem rows for this run_id so merged alias duplicates do not linger.
        let prob_table = self.open_table(TABLE_SIM_PROBLEMS).await?;
        prob_table.delete(&run_filter).await?;
        if !problems.is_empty() {
            let batch = sim_problems_to_batch(&problems)?;
            let schema = batch.schema();
            let reader: Box<dyn RecordBatchReader + Send> =
                Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));
            prob_table.add(reader).execute().await?;
        }

        // 3. Replace attempt rows for this run_id in deterministic sequence order.
        let att_table = self.open_table(TABLE_SIM_ATTEMPTS).await?;
        att_table.delete(&run_filter).await?;
        if !attempts.is_empty() {
            let batch = sim_attempts_to_batch(&attempts)?;
            let schema = batch.schema();
            let reader: Box<dyn RecordBatchReader + Send> =
                Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));
            att_table.add(reader).execute().await?;
        }

        Ok(())
    }

    /// List all SIM runs for `(repo_id, scope_id)`, most recently updated first.
    pub async fn list_sim_runs(&self, repo_id: &str, scope_id: &str) -> Result<Vec<SimRunRecord>> {
        let table = self.open_table(TABLE_SIM_RUNS).await?;
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

        let mut runs = Vec::new();
        for batch in &batches {
            runs.extend(batch_to_sim_runs(batch)?);
        }
        runs.sort_by_key(|a| std::cmp::Reverse(a.updated_at));
        Ok(runs)
    }

    /// Return the most recently updated SIM run for `(repo_id, scope_id)`.
    pub async fn latest_sim_run(
        &self,
        repo_id: &str,
        scope_id: &str,
    ) -> Result<Option<SimRunRecord>> {
        Ok(self.list_sim_runs(repo_id, scope_id).await?.into_iter().next())
    }

    /// Load a complete [`SimLedger`] by exact `run_id`, ID prefix, or `arm_name`.
    pub async fn load_sim_ledger(
        &self,
        repo_id: &str,
        scope_id: &str,
        run_id_or_arm: &str,
    ) -> Result<Option<SimLedger>> {
        let needle = run_id_or_arm.trim();
        let runs = self.list_sim_runs(repo_id, scope_id).await?;
        let run = if needle.is_empty() {
            runs.into_iter().next()
        } else {
            runs.into_iter().find(|r| {
                r.run_id == needle
                    || r.run_id.starts_with(needle)
                    || r.arm_name.eq_ignore_ascii_case(needle)
            })
        };
        let Some(run) = run else {
            return Ok(None);
        };

        let filter = format!(
            "repo_id = '{}' AND scope_id = '{}' AND run_id = '{}'",
            escape_sql_literal(repo_id),
            escape_sql_literal(scope_id),
            escape_sql_literal(&run.run_id)
        );

        let prob_table = self.open_table(TABLE_SIM_PROBLEMS).await?;
        let prob_batches = prob_table
            .query()
            .only_if(&filter)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        let mut problems = Vec::new();
        for batch in &prob_batches {
            problems.extend(batch_to_sim_problems(batch)?);
        }

        let att_table = self.open_table(TABLE_SIM_ATTEMPTS).await?;
        let att_batches = att_table
            .query()
            .only_if(&filter)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        let mut attempts = Vec::new();
        for batch in &att_batches {
            attempts.extend(batch_to_sim_attempts(batch)?);
        }

        Ok(Some(SimLedger::from_records(run, problems, attempts)?))
    }

    /// Export a stored SIM run to its canonical Appendix D.3 [`SimExportDocument`].
    pub async fn export_sim_document(
        &self,
        repo_id: &str,
        scope_id: &str,
        run_id_or_arm: Option<&str>,
    ) -> Result<Option<SimExportDocument>> {
        let ledger = self
            .load_sim_ledger(repo_id, scope_id, run_id_or_arm.unwrap_or(""))
            .await?;
        Ok(ledger.map(|l| l.to_export_document()))
    }

    /// Import an external Appendix D.3 [`SimExportDocument`] into `(repo_id, scope_id)` and persist it.
    #[allow(clippy::too_many_arguments)]
    pub async fn import_sim_document(
        &self,
        repo_id: &str,
        scope_id: &str,
        doc: SimExportDocument,
        arm_name: &str,
        domain: &str,
        benchmark: &str,
        policy_model: &str,
        optimizer_model: &str,
        harness_version_id: &str,
    ) -> Result<SimLedger> {
        let ledger = SimLedger::from_export_document(
            doc,
            repo_id,
            scope_id,
            arm_name,
            domain,
            benchmark,
            policy_model,
            optimizer_model,
            harness_version_id,
        )?;
        self.save_sim_ledger(&ledger).await?;
        Ok(ledger)
    }

    async fn count_sim_problems(&self, repo_id: &str, scope_id: &str) -> Result<usize> {
        let table = self.open_table(TABLE_SIM_PROBLEMS).await?;
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
        let mut rows = 0usize;
        for batch in &batches {
            rows += batch.num_rows();
        }
        Ok(rows)
    }

    async fn count_sim_attempts(&self, repo_id: &str, scope_id: &str) -> Result<usize> {
        let table = self.open_table(TABLE_SIM_ATTEMPTS).await?;
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
        let mut rows = 0usize;
        for batch in &batches {
            rows += batch.num_rows();
        }
        Ok(rows)
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
            TABLE_SKILLS,
            TABLE_REPO_PROFILES,
            TABLE_HARNESS_VERSIONS,
            TABLE_SIM_RUNS,
            TABLE_SIM_PROBLEMS,
            TABLE_SIM_ATTEMPTS,
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
        let skills = self.list_skills(repo_id, scope_id).await?;
        let harness_versions = self.list_harness_versions(repo_id, scope_id).await?;
        let sim_runs = self.list_sim_runs(repo_id, scope_id).await?;
        let sim_problems_count = self.count_sim_problems(repo_id, scope_id).await?;
        let sim_attempts_count = self.count_sim_attempts(repo_id, scope_id).await?;

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
            skills_count: skills.len(),
            harness_versions_count: harness_versions.len(),
            sim_runs_count: sim_runs.len(),
            sim_problems_count,
            sim_attempts_count,
        })
    }
}

fn tokenize_skill_text(text: &str) -> Vec<String> {
    text.to_ascii_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| t.len() >= 3)
        .map(str::to_string)
        .collect()
}

fn score_skill_match(skill: &SkillRecord, query_lower: &str, query_tokens: &[String]) -> f32 {
    let mut score = 0.0f32;
    let name_lower = skill.name.to_ascii_lowercase();
    let slug = normalize_skill_slug(&skill.name);

    if (!name_lower.is_empty() && query_lower.contains(&name_lower))
        || (!slug.is_empty() && query_lower.contains(&slug))
    {
        score += 6.0;
    }

    for trigger in &skill.triggers {
        let t_lower = trigger.trim().to_ascii_lowercase();
        if t_lower.is_empty() {
            continue;
        }
        if query_lower.contains(&t_lower) {
            score += 5.0;
        } else {
            let t_tokens = tokenize_skill_text(&t_lower);
            if !t_tokens.is_empty() {
                let hits = t_tokens
                    .iter()
                    .filter(|tok| query_tokens.iter().any(|q| q == *tok))
                    .count();
                if hits > 0 {
                    score += 2.5 * (hits as f32 / t_tokens.len() as f32);
                }
            }
        }
    }

    if !query_tokens.is_empty() {
        let name_tokens = tokenize_skill_text(&name_lower);
        if !name_tokens.is_empty() {
            let hits = name_tokens
                .iter()
                .filter(|tok| query_tokens.iter().any(|q| q == *tok))
                .count();
            if hits > 0 {
                score += 2.0 * (hits as f32 / name_tokens.len() as f32);
            }
        }

        let desc_tokens = tokenize_skill_text(&skill.description);
        if !desc_tokens.is_empty() {
            let hits = query_tokens
                .iter()
                .filter(|q| desc_tokens.iter().any(|d| d == *q))
                .count();
            if hits >= 2 {
                score += 1.8 * (hits as f32 / query_tokens.len().min(desc_tokens.len()) as f32);
            }
        }
    }

    score
}
