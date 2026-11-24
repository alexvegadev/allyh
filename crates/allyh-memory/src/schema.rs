use std::sync::Arc;

use arrow_array::builder::{FixedSizeListBuilder, Float32Builder, ListBuilder, StringBuilder};
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, ListArray, RecordBatch, StringArray,
    TimestampMicrosecondArray, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{MemoryError, Result};

/// Fixed embedding vector width used across AllyH LanceDB tables.
pub const EMBEDDING_DIM: i32 = 384;

pub const TABLE_SESSIONS: &str = "sessions";
pub const TABLE_ARCHITECTURAL_DECISIONS: &str = "architectural_decisions";
pub const TABLE_TASKS: &str = "tasks";
pub const TABLE_CODE_GRAPH_SNAPSHOTS: &str = "code_graph_snapshots";
pub const TABLE_SKILLS: &str = "skills";

/// Episodic session summary record stored in `sessions`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionRecord {
    pub session_id: String,
    pub repo_id: String,
    pub scope_id: String,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    pub rolling_summary: String,
    pub token_budget_used: u64,
}

/// Bi-temporal architectural invariant stored in `architectural_decisions`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitecturalDecision {
    pub id: String,
    pub repo_id: String,
    pub scope_id: String,
    pub topic: String,
    pub rule_statement: String,
    pub rationale: String,
    pub affected_paths: Vec<String>,
    pub vector: Vec<f32>,
    pub valid_at: DateTime<Utc>,
    pub invalid_at: Option<DateTime<Utc>>,
    pub superseded_by: Option<String>,
}

/// Ranked architectural decision returned by hybrid search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredDecision {
    pub decision: ArchitecturalDecision,
    pub score: f32,
    pub pinned: bool,
}

/// Strict state machine status for structured execution tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    Approved,
    InProgress,
    Verified,
}

impl TaskStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Approved => "approved",
            Self::InProgress => "in_progress",
            Self::Verified => "verified",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "pending" => Ok(Self::Pending),
            "approved" => Ok(Self::Approved),
            "in_progress" => Ok(Self::InProgress),
            "verified" => Ok(Self::Verified),
            other => Err(MemoryError::InvalidTransition(format!(
                "unknown task status '{other}'"
            ))),
        }
    }
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Proof of automated compiler/test verification required to mark a task `verified`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationProof {
    pub command: String,
    pub passed_at: DateTime<Utc>,
    pub warnings: usize,
    pub summary: String,
}

/// Structured execution task stored in `tasks`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskRecord {
    pub id: String,
    pub repo_id: String,
    pub scope_id: String,
    pub parent_id: Option<String>,
    pub title: String,
    pub acceptance_criteria: Vec<String>,
    pub status: TaskStatus,
    pub last_error_trace: Option<String>,
    pub updated_at: DateTime<Utc>,
}

/// Reusable procedural skill stored in `skills`.
///
/// A skill captures a named, reusable procedure (instructions + optional trigger hints)
/// that the agent can register once and recall across turns and sessions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillRecord {
    pub id: String,
    pub repo_id: String,
    pub scope_id: String,
    pub name: String,
    pub description: String,
    pub instructions: String,
    pub triggers: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Normalize a human-readable skill name into a stable lowercase kebab-case slug.
pub fn normalize_skill_slug(name: &str) -> String {
    let mut slug = String::new();
    let mut last_was_dash = false;
    for ch in name.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            last_was_dash = false;
        } else if !slug.is_empty() && !last_was_dash {
            slug.push('-');
            last_was_dash = true;
        }
    }
    while slug.ends_with('-') {
        slug.pop();
    }
    slug
}

/// Derive a stable, collision-resistant skill id from `(repo_id, scope_id, name)`.
pub fn skill_id(repo_id: &str, scope_id: &str, name: &str) -> Result<String> {
    let slug = normalize_skill_slug(name);
    if slug.is_empty() {
        return Err(MemoryError::InvalidSkill(
            "skill 'name' must contain at least one ASCII letter or digit".to_string(),
        ));
    }
    Ok(format!("{repo_id}:{scope_id}:{slug}"))
}

/// Code integrity and RepoGraph snapshot stored in `code_graph_snapshots`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeGraphSnapshot {
    pub repo_id: String,
    pub scope_id: String,
    pub file_path: String,
    pub sha256_hash: String,
    pub last_read_turn: u64,
    pub symbols_json: String,
    pub dependencies: Vec<String>,
    pub vector: Vec<f32>,
}

/// Diagnostics snapshot reported by `/db-status`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbStatusReport {
    pub uri: String,
    pub backend_label: String,
    pub repo_id: String,
    pub scope_id: String,
    pub sessions_count: usize,
    pub active_decisions_count: usize,
    pub invalidated_decisions_count: usize,
    pub tasks_count: usize,
    pub code_snapshots_count: usize,
    pub skills_count: usize,
}

// ---------------------------------------------------------------------------
// Apache Arrow Schemas
// ---------------------------------------------------------------------------

fn timestamp_utc() -> DataType {
    DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
}

fn string_list_type() -> DataType {
    DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)))
}

fn vector_type() -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        EMBEDDING_DIM,
    )
}

pub fn sessions_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("session_id", DataType::Utf8, false),
        Field::new("repo_id", DataType::Utf8, false),
        Field::new("scope_id", DataType::Utf8, false),
        Field::new("started_at", timestamp_utc(), false),
        Field::new("ended_at", timestamp_utc(), true),
        Field::new("rolling_summary", DataType::Utf8, false),
        Field::new("token_budget_used", DataType::UInt64, false),
    ]))
}

pub fn architectural_decisions_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("repo_id", DataType::Utf8, false),
        Field::new("scope_id", DataType::Utf8, false),
        Field::new("topic", DataType::Utf8, false),
        Field::new("rule_statement", DataType::Utf8, false),
        Field::new("rationale", DataType::Utf8, false),
        Field::new("affected_paths", string_list_type(), false),
        Field::new("vector", vector_type(), false),
        Field::new("valid_at", timestamp_utc(), false),
        Field::new("invalid_at", timestamp_utc(), true),
        Field::new("superseded_by", DataType::Utf8, true),
    ]))
}

pub fn tasks_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("repo_id", DataType::Utf8, false),
        Field::new("scope_id", DataType::Utf8, false),
        Field::new("parent_id", DataType::Utf8, true),
        Field::new("title", DataType::Utf8, false),
        Field::new("acceptance_criteria", string_list_type(), false),
        Field::new("status", DataType::Utf8, false),
        Field::new("last_error_trace", DataType::Utf8, true),
        Field::new("updated_at", timestamp_utc(), false),
    ]))
}

pub fn code_graph_snapshots_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("repo_id", DataType::Utf8, false),
        Field::new("scope_id", DataType::Utf8, false),
        Field::new("file_path", DataType::Utf8, false),
        Field::new("sha256_hash", DataType::Utf8, false),
        Field::new("last_read_turn", DataType::UInt64, false),
        Field::new("symbols", DataType::Utf8, false),
        Field::new("dependencies", string_list_type(), false),
        Field::new("vector", vector_type(), false),
    ]))
}

pub fn skills_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("repo_id", DataType::Utf8, false),
        Field::new("scope_id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("description", DataType::Utf8, false),
        Field::new("instructions", DataType::Utf8, false),
        Field::new("triggers", string_list_type(), false),
        Field::new("created_at", timestamp_utc(), false),
        Field::new("updated_at", timestamp_utc(), false),
    ]))
}

// ---------------------------------------------------------------------------
// Arrow RecordBatch Serializers & Deserializers
// ---------------------------------------------------------------------------

fn build_string_list_array(rows: &[Vec<String>]) -> ListArray {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for row in rows {
        for item in row {
            builder.values().append_value(item);
        }
        builder.append(true);
    }
    builder.finish()
}

fn build_vector_array(rows: &[Vec<f32>]) -> Result<FixedSizeListArray> {
    let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), EMBEDDING_DIM);
    for vec in rows {
        if vec.len() != EMBEDDING_DIM as usize {
            return Err(MemoryError::Embedding(format!(
                "expected vector length {EMBEDDING_DIM}, got {}",
                vec.len()
            )));
        }
        for &v in vec {
            builder.values().append_value(v);
        }
        builder.append(true);
    }
    Ok(builder.finish())
}

fn read_string_list(array: &ListArray, row: usize) -> Vec<String> {
    if array.is_null(row) {
        return Vec::new();
    }
    let values = array.value(row);
    let Some(str_arr) = values.as_any().downcast_ref::<StringArray>() else {
        return Vec::new();
    };
    (0..str_arr.len())
        .filter_map(|i| {
            if str_arr.is_null(i) {
                None
            } else {
                Some(str_arr.value(i).to_string())
            }
        })
        .collect()
}

fn read_vector(array: &FixedSizeListArray, row: usize) -> Vec<f32> {
    if array.is_null(row) {
        return vec![0.0; EMBEDDING_DIM as usize];
    }
    let values = array.value(row);
    let Some(f_arr) = values.as_any().downcast_ref::<Float32Array>() else {
        return vec![0.0; EMBEDDING_DIM as usize];
    };
    f_arr.values().to_vec()
}

fn micros_to_utc(micros: i64) -> DateTime<Utc> {
    DateTime::from_timestamp_micros(micros).unwrap_or_else(Utc::now)
}

pub fn sessions_to_batch(records: &[SessionRecord]) -> Result<RecordBatch> {
    let schema = sessions_schema();
    let session_ids = StringArray::from_iter_values(records.iter().map(|r| r.session_id.as_str()));
    let repo_ids = StringArray::from_iter_values(records.iter().map(|r| r.repo_id.as_str()));
    let scope_ids = StringArray::from_iter_values(records.iter().map(|r| r.scope_id.as_str()));
    let started_at = TimestampMicrosecondArray::from_iter_values(
        records.iter().map(|r| r.started_at.timestamp_micros()),
    )
    .with_timezone("UTC");
    let ended_at = TimestampMicrosecondArray::from_iter(
        records
            .iter()
            .map(|r| r.ended_at.map(|dt| dt.timestamp_micros())),
    )
    .with_timezone("UTC");
    let summaries =
        StringArray::from_iter_values(records.iter().map(|r| r.rolling_summary.as_str()));
    let budgets = UInt64Array::from_iter_values(records.iter().map(|r| r.token_budget_used));

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(session_ids),
            Arc::new(repo_ids),
            Arc::new(scope_ids),
            Arc::new(started_at),
            Arc::new(ended_at),
            Arc::new(summaries),
            Arc::new(budgets),
        ],
    )?)
}

pub fn batch_to_sessions(batch: &RecordBatch) -> Result<Vec<SessionRecord>> {
    let session_ids = col_str(batch, "session_id")?;
    let repo_ids = col_str(batch, "repo_id")?;
    let scope_ids = col_str(batch, "scope_id")?;
    let started_at = col_ts(batch, "started_at")?;
    let ended_at = col_ts(batch, "ended_at")?;
    let summaries = col_str(batch, "rolling_summary")?;
    let budgets = col_u64(batch, "token_budget_used")?;

    let mut out = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        out.push(SessionRecord {
            session_id: session_ids.value(i).to_string(),
            repo_id: repo_ids.value(i).to_string(),
            scope_id: scope_ids.value(i).to_string(),
            started_at: micros_to_utc(started_at.value(i)),
            ended_at: if ended_at.is_null(i) {
                None
            } else {
                Some(micros_to_utc(ended_at.value(i)))
            },
            rolling_summary: summaries.value(i).to_string(),
            token_budget_used: budgets.value(i),
        });
    }
    Ok(out)
}

pub fn decisions_to_batch(records: &[ArchitecturalDecision]) -> Result<RecordBatch> {
    let schema = architectural_decisions_schema();
    let ids = StringArray::from_iter_values(records.iter().map(|r| r.id.as_str()));
    let repo_ids = StringArray::from_iter_values(records.iter().map(|r| r.repo_id.as_str()));
    let scope_ids = StringArray::from_iter_values(records.iter().map(|r| r.scope_id.as_str()));
    let topics = StringArray::from_iter_values(records.iter().map(|r| r.topic.as_str()));
    let statements =
        StringArray::from_iter_values(records.iter().map(|r| r.rule_statement.as_str()));
    let rationales = StringArray::from_iter_values(records.iter().map(|r| r.rationale.as_str()));
    let paths_vec: Vec<Vec<String>> = records.iter().map(|r| r.affected_paths.clone()).collect();
    let affected_paths = build_string_list_array(&paths_vec);
    let vectors_vec: Vec<Vec<f32>> = records.iter().map(|r| r.vector.clone()).collect();
    let vectors = build_vector_array(&vectors_vec)?;
    let valid_at = TimestampMicrosecondArray::from_iter_values(
        records.iter().map(|r| r.valid_at.timestamp_micros()),
    )
    .with_timezone("UTC");
    let invalid_at = TimestampMicrosecondArray::from_iter(
        records
            .iter()
            .map(|r| r.invalid_at.map(|dt| dt.timestamp_micros())),
    )
    .with_timezone("UTC");
    let superseded_by =
        StringArray::from_iter(records.iter().map(|r| r.superseded_by.as_deref()));

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(repo_ids),
            Arc::new(scope_ids),
            Arc::new(topics),
            Arc::new(statements),
            Arc::new(rationales),
            Arc::new(affected_paths),
            Arc::new(vectors),
            Arc::new(valid_at),
            Arc::new(invalid_at),
            Arc::new(superseded_by),
        ],
    )?)
}

pub fn batch_to_decisions(batch: &RecordBatch) -> Result<Vec<ArchitecturalDecision>> {
    let ids = col_str(batch, "id")?;
    let repo_ids = col_str(batch, "repo_id")?;
    let scope_ids = col_str(batch, "scope_id")?;
    let topics = col_str(batch, "topic")?;
    let statements = col_str(batch, "rule_statement")?;
    let rationales = col_str(batch, "rationale")?;
    let affected_paths = col_list(batch, "affected_paths")?;
    let vectors = col_fixed_list(batch, "vector")?;
    let valid_at = col_ts(batch, "valid_at")?;
    let invalid_at = col_ts(batch, "invalid_at")?;
    let superseded_by = col_str(batch, "superseded_by")?;

    let mut out = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        out.push(ArchitecturalDecision {
            id: ids.value(i).to_string(),
            repo_id: repo_ids.value(i).to_string(),
            scope_id: scope_ids.value(i).to_string(),
            topic: topics.value(i).to_string(),
            rule_statement: statements.value(i).to_string(),
            rationale: rationales.value(i).to_string(),
            affected_paths: read_string_list(affected_paths, i),
            vector: read_vector(vectors, i),
            valid_at: micros_to_utc(valid_at.value(i)),
            invalid_at: if invalid_at.is_null(i) {
                None
            } else {
                Some(micros_to_utc(invalid_at.value(i)))
            },
            superseded_by: if superseded_by.is_null(i) {
                None
            } else {
                Some(superseded_by.value(i).to_string())
            },
        });
    }
    Ok(out)
}

pub fn tasks_to_batch(records: &[TaskRecord]) -> Result<RecordBatch> {
    let schema = tasks_schema();
    let ids = StringArray::from_iter_values(records.iter().map(|r| r.id.as_str()));
    let repo_ids = StringArray::from_iter_values(records.iter().map(|r| r.repo_id.as_str()));
    let scope_ids = StringArray::from_iter_values(records.iter().map(|r| r.scope_id.as_str()));
    let parent_ids = StringArray::from_iter(records.iter().map(|r| r.parent_id.as_deref()));
    let titles = StringArray::from_iter_values(records.iter().map(|r| r.title.as_str()));
    let criteria_vec: Vec<Vec<String>> = records
        .iter()
        .map(|r| r.acceptance_criteria.clone())
        .collect();
    let criteria = build_string_list_array(&criteria_vec);
    let statuses = StringArray::from_iter_values(records.iter().map(|r| r.status.as_str()));
    let error_traces =
        StringArray::from_iter(records.iter().map(|r| r.last_error_trace.as_deref()));
    let updated_at = TimestampMicrosecondArray::from_iter_values(
        records.iter().map(|r| r.updated_at.timestamp_micros()),
    )
    .with_timezone("UTC");

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(repo_ids),
            Arc::new(scope_ids),
            Arc::new(parent_ids),
            Arc::new(titles),
            Arc::new(criteria),
            Arc::new(statuses),
            Arc::new(error_traces),
            Arc::new(updated_at),
        ],
    )?)
}

pub fn batch_to_tasks(batch: &RecordBatch) -> Result<Vec<TaskRecord>> {
    let ids = col_str(batch, "id")?;
    let repo_ids = col_str(batch, "repo_id")?;
    let scope_ids = col_str(batch, "scope_id")?;
    let parent_ids = col_str(batch, "parent_id")?;
    let titles = col_str(batch, "title")?;
    let criteria = col_list(batch, "acceptance_criteria")?;
    let statuses = col_str(batch, "status")?;
    let error_traces = col_str(batch, "last_error_trace")?;
    let updated_at = col_ts(batch, "updated_at")?;

    let mut out = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        out.push(TaskRecord {
            id: ids.value(i).to_string(),
            repo_id: repo_ids.value(i).to_string(),
            scope_id: scope_ids.value(i).to_string(),
            parent_id: if parent_ids.is_null(i) {
                None
            } else {
                Some(parent_ids.value(i).to_string())
            },
            title: titles.value(i).to_string(),
            acceptance_criteria: read_string_list(criteria, i),
            status: TaskStatus::parse(statuses.value(i))?,
            last_error_trace: if error_traces.is_null(i) {
                None
            } else {
                Some(error_traces.value(i).to_string())
            },
            updated_at: micros_to_utc(updated_at.value(i)),
        });
    }
    Ok(out)
}

pub fn skills_to_batch(records: &[SkillRecord]) -> Result<RecordBatch> {
    let schema = skills_schema();
    let ids = StringArray::from_iter_values(records.iter().map(|r| r.id.as_str()));
    let repo_ids = StringArray::from_iter_values(records.iter().map(|r| r.repo_id.as_str()));
    let scope_ids = StringArray::from_iter_values(records.iter().map(|r| r.scope_id.as_str()));
    let names = StringArray::from_iter_values(records.iter().map(|r| r.name.as_str()));
    let descriptions = StringArray::from_iter_values(records.iter().map(|r| r.description.as_str()));
    let instructions = StringArray::from_iter_values(records.iter().map(|r| r.instructions.as_str()));
    let triggers_vec: Vec<Vec<String>> = records.iter().map(|r| r.triggers.clone()).collect();
    let triggers = build_string_list_array(&triggers_vec);
    let created_at = TimestampMicrosecondArray::from_iter_values(
        records.iter().map(|r| r.created_at.timestamp_micros()),
    )
    .with_timezone("UTC");
    let updated_at = TimestampMicrosecondArray::from_iter_values(
        records.iter().map(|r| r.updated_at.timestamp_micros()),
    )
    .with_timezone("UTC");

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(repo_ids),
            Arc::new(scope_ids),
            Arc::new(names),
            Arc::new(descriptions),
            Arc::new(instructions),
            Arc::new(triggers),
            Arc::new(created_at),
            Arc::new(updated_at),
        ],
    )?)
}

pub fn batch_to_skills(batch: &RecordBatch) -> Result<Vec<SkillRecord>> {
    let ids = col_str(batch, "id")?;
    let repo_ids = col_str(batch, "repo_id")?;
    let scope_ids = col_str(batch, "scope_id")?;
    let names = col_str(batch, "name")?;
    let descriptions = col_str(batch, "description")?;
    let instructions = col_str(batch, "instructions")?;
    let triggers = col_list(batch, "triggers")?;
    let created_at = col_ts(batch, "created_at")?;
    let updated_at = col_ts(batch, "updated_at")?;

    let mut out = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        out.push(SkillRecord {
            id: ids.value(i).to_string(),
            repo_id: repo_ids.value(i).to_string(),
            scope_id: scope_ids.value(i).to_string(),
            name: names.value(i).to_string(),
            description: descriptions.value(i).to_string(),
            instructions: instructions.value(i).to_string(),
            triggers: read_string_list(triggers, i),
            created_at: micros_to_utc(created_at.value(i)),
            updated_at: micros_to_utc(updated_at.value(i)),
        });
    }
    Ok(out)
}

pub fn snapshots_to_batch(records: &[CodeGraphSnapshot]) -> Result<RecordBatch> {
    let schema = code_graph_snapshots_schema();
    let repo_ids = StringArray::from_iter_values(records.iter().map(|r| r.repo_id.as_str()));
    let scope_ids = StringArray::from_iter_values(records.iter().map(|r| r.scope_id.as_str()));
    let file_paths = StringArray::from_iter_values(records.iter().map(|r| r.file_path.as_str()));
    let hashes = StringArray::from_iter_values(records.iter().map(|r| r.sha256_hash.as_str()));
    let turns = UInt64Array::from_iter_values(records.iter().map(|r| r.last_read_turn));
    let symbols = StringArray::from_iter_values(records.iter().map(|r| r.symbols_json.as_str()));
    let deps_vec: Vec<Vec<String>> = records.iter().map(|r| r.dependencies.clone()).collect();
    let dependencies = build_string_list_array(&deps_vec);
    let vecs: Vec<Vec<f32>> = records.iter().map(|r| r.vector.clone()).collect();
    let vectors = build_vector_array(&vecs)?;

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(repo_ids),
            Arc::new(scope_ids),
            Arc::new(file_paths),
            Arc::new(hashes),
            Arc::new(turns),
            Arc::new(symbols),
            Arc::new(dependencies),
            Arc::new(vectors),
        ],
    )?)
}

pub fn batch_to_snapshots(batch: &RecordBatch) -> Result<Vec<CodeGraphSnapshot>> {
    let repo_ids = col_str(batch, "repo_id")?;
    let scope_ids = col_str(batch, "scope_id")?;
    let file_paths = col_str(batch, "file_path")?;
    let hashes = col_str(batch, "sha256_hash")?;
    let turns = col_u64(batch, "last_read_turn")?;
    let symbols = col_str(batch, "symbols")?;
    let dependencies = col_list(batch, "dependencies")?;
    let vectors = col_fixed_list(batch, "vector")?;

    let mut out = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        out.push(CodeGraphSnapshot {
            repo_id: repo_ids.value(i).to_string(),
            scope_id: scope_ids.value(i).to_string(),
            file_path: file_paths.value(i).to_string(),
            sha256_hash: hashes.value(i).to_string(),
            last_read_turn: turns.value(i),
            symbols_json: symbols.value(i).to_string(),
            dependencies: read_string_list(dependencies, i),
            vector: read_vector(vectors, i),
        });
    }
    Ok(out)
}

fn col_str<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| MemoryError::Embedding(format!("missing Utf8 column '{name}'")))
}

fn col_u64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt64Array> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| MemoryError::Embedding(format!("missing UInt64 column '{name}'")))
}

fn col_ts<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a TimestampMicrosecondArray> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<TimestampMicrosecondArray>())
        .ok_or_else(|| MemoryError::Embedding(format!("missing Timestamp column '{name}'")))
}

fn col_list<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ListArray> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<ListArray>())
        .ok_or_else(|| MemoryError::Embedding(format!("missing List column '{name}'")))
}

fn col_fixed_list<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a FixedSizeListArray> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
        .ok_or_else(|| MemoryError::Embedding(format!("missing FixedSizeList column '{name}'")))
}
