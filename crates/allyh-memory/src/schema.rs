use std::sync::Arc;

use arrow_array::builder::{FixedSizeListBuilder, Float32Builder, ListBuilder, StringBuilder};
use arrow_array::{
    Array, BooleanArray, FixedSizeListArray, Float32Array, ListArray, RecordBatch, StringArray,
    TimestampMicrosecondArray, UInt32Array, UInt64Array,
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
pub const TABLE_REPO_PROFILES: &str = "repo_profiles";
pub const TABLE_HARNESS_VERSIONS: &str = "harness_versions";
pub const TABLE_SIM_RUNS: &str = "sim_runs";
pub const TABLE_SIM_PROBLEMS: &str = "sim_problems";
pub const TABLE_SIM_ATTEMPTS: &str = "sim_attempts";

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
    #[serde(default = "default_unknown_language")]
    pub language: String,
    pub sha256_hash: String,
    pub last_read_turn: u64,
    pub symbols_json: String,
    pub dependencies: Vec<String>,
    pub vector: Vec<f32>,
}

fn default_unknown_language() -> String {
    "unknown".to_string()
}

/// Out-of-tree per-repository polyglot configuration stored in `repo_profiles`.
///
/// Allows overriding detected languages, check/lint/test commands, ignore globs, and auto-verify
/// behavior without writing dotfiles into the target repository.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RepoProfile {
    pub id: String,
    pub repo_id: String,
    pub scope_id: String,
    #[serde(default)]
    pub languages_override: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub check_command: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lint_command: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test_command: Option<String>,
    #[serde(default)]
    pub extra_ignore_globs: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_verify_override: Option<bool>,
    pub updated_at: DateTime<Utc>,
}

impl RepoProfile {
    pub fn empty(repo_id: impl Into<String>, scope_id: impl Into<String>) -> Self {
        let repo_id = repo_id.into();
        let scope_id = scope_id.into();
        let id = repo_profile_id(&repo_id, &scope_id);
        Self {
            id,
            repo_id,
            scope_id,
            languages_override: Vec::new(),
            check_command: None,
            lint_command: None,
            test_command: None,
            extra_ignore_globs: Vec::new(),
            auto_verify_override: None,
            updated_at: Utc::now(),
        }
    }

    pub fn is_default_empty(&self) -> bool {
        self.languages_override.is_empty()
            && self.check_command.is_none()
            && self.lint_command.is_none()
            && self.test_command.is_none()
            && self.extra_ignore_globs.is_empty()
            && self.auto_verify_override.is_none()
    }
}

/// Stable partition key for a [`RepoProfile`] row.
pub fn repo_profile_id(repo_id: &str, scope_id: &str) -> String {
    format!("{repo_id}:{scope_id}")
}

/// Best-effort language inference from a relative file path (used for legacy schema migration).
pub fn infer_language_from_path(file_path: &str) -> &'static str {
    let lower = file_path.to_ascii_lowercase();
    if lower.ends_with(".rs") {
        "rust"
    } else if lower.ends_with(".tsx") {
        "tsx"
    } else if lower.ends_with(".ts") || lower.ends_with(".mts") || lower.ends_with(".cts") {
        "typescript"
    } else if lower.ends_with(".jsx") {
        "jsx"
    } else if lower.ends_with(".js") || lower.ends_with(".mjs") || lower.ends_with(".cjs") {
        "javascript"
    } else if lower.ends_with(".py") || lower.ends_with(".pyi") {
        "python"
    } else if lower.ends_with(".go") {
        "go"
    } else if lower.ends_with(".toml") || lower.ends_with(".json") || lower.ends_with(".mod") {
        "config"
    } else {
        "unknown"
    }
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
    pub harness_versions_count: usize,
    pub sim_runs_count: usize,
    pub sim_problems_count: usize,
    pub sim_attempts_count: usize,
}

// ---------------------------------------------------------------------------
// Self-Improvement Memory (SIM) & Harness Versioning Records (arXiv:2611.18856)
// ---------------------------------------------------------------------------

/// Transfer / memory condition governing how SIM conditions target optimization (§3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimMemoryMode {
    /// Start from seed harness H_0 with empty SIM.
    ColdStart,
    /// Start from source-evolved harness H_S with empty SIM.
    HarnessTransfer,
    /// Start from H_S with root-cause hypotheses only (M_S^c), omitting attempt evidence.
    ProblemOnly,
    /// Start from H_S with complete SIM record (M_S) and append target attempts.
    FullSim,
}

impl SimMemoryMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ColdStart => "cold_start",
            Self::HarnessTransfer => "harness_transfer",
            Self::ProblemOnly => "problem_only",
            Self::FullSim => "full_sim",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "cold_start" | "cold" => Ok(Self::ColdStart),
            "harness_transfer" | "harness_only" | "no_sim" => Ok(Self::HarnessTransfer),
            "problem_only" | "root_cause_only" => Ok(Self::ProblemOnly),
            "full_sim" | "full" | "sim" => Ok(Self::FullSim),
            other => Err(MemoryError::InvalidSim(format!(
                "unknown SIM memory mode '{other}' (expected cold_start, harness_transfer, problem_only, or full_sim)"
            ))),
        }
    }
}

impl std::fmt::Display for SimMemoryMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Versioned executable harness specification persisted out-of-tree in `harness_versions`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HarnessVersionRecord {
    pub id: String,
    pub repo_id: String,
    pub scope_id: String,
    pub parent_version_id: Option<String>,
    pub run_id: Option<String>,
    pub round: u32,
    pub label: String,
    pub spec_json: String,
    pub sha256_hash: String,
    pub policy_model: String,
    pub domain: String,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
}

/// Metadata for a source or continuation CRSI optimization run stored in `sim_runs`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SimRunRecord {
    pub run_id: String,
    pub repo_id: String,
    pub scope_id: String,
    pub arm_name: String,
    pub domain: String,
    pub benchmark: String,
    pub policy_model: String,
    pub optimizer_model: String,
    pub memory_mode: SimMemoryMode,
    pub seed_harness_id: String,
    pub current_harness_id: String,
    pub carried_from_run_id: Option<String>,
    pub carried_from_domain: Option<String>,
    pub next_problem_num: u32,
    pub aliases_json: String,
    pub rounds_completed: u32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Recurring harness problem entry m_i = (c_i, ...) stored in `sim_problems`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SimProblemRecord {
    /// Composite primary key `{run_id}:{problem_id}` so each run's ledger snapshot stays intact.
    pub row_id: String,
    pub run_id: String,
    pub repo_id: String,
    pub scope_id: String,
    pub problem_id: String,
    pub name: String,
    pub normalized_name: String,
    pub opened_round: u32,
    pub origin_domain: String,
    pub origin_run_id: String,
    pub created_at: DateTime<Utc>,
}

/// Recorded optimization attempt (a_ij, o_ij, p_ij) stored in `sim_attempts`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SimAttemptRecord {
    pub id: String,
    pub run_id: String,
    pub repo_id: String,
    pub scope_id: String,
    pub problem_id: String,
    pub sequence_idx: u32,
    pub round: u32,
    pub sample: String,
    pub accepted: bool,
    pub backfilled: bool,
    pub what: String,
    pub files: Vec<String>,
    pub predicted: Vec<String>,
    pub fixed: Vec<String>,
    pub missed: Vec<String>,
    pub passed: u32,
    pub total: u32,
    pub score: f32,
    pub domain: String,
    pub benchmark: String,
    pub policy_model: String,
    pub harness_version_id: String,
    pub created_at: DateTime<Utc>,
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
        Field::new("language", DataType::Utf8, false),
        Field::new("sha256_hash", DataType::Utf8, false),
        Field::new("last_read_turn", DataType::UInt64, false),
        Field::new("symbols", DataType::Utf8, false),
        Field::new("dependencies", string_list_type(), false),
        Field::new("vector", vector_type(), false),
    ]))
}

pub fn repo_profiles_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("repo_id", DataType::Utf8, false),
        Field::new("scope_id", DataType::Utf8, false),
        Field::new("languages_override", string_list_type(), false),
        Field::new("check_command", DataType::Utf8, true),
        Field::new("lint_command", DataType::Utf8, true),
        Field::new("test_command", DataType::Utf8, true),
        Field::new("extra_ignore_globs", string_list_type(), false),
        Field::new("auto_verify_override", DataType::Boolean, true),
        Field::new("updated_at", timestamp_utc(), false),
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

pub fn harness_versions_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("repo_id", DataType::Utf8, false),
        Field::new("scope_id", DataType::Utf8, false),
        Field::new("parent_version_id", DataType::Utf8, true),
        Field::new("run_id", DataType::Utf8, true),
        Field::new("round", DataType::UInt32, false),
        Field::new("label", DataType::Utf8, false),
        Field::new("spec_json", DataType::Utf8, false),
        Field::new("sha256_hash", DataType::Utf8, false),
        Field::new("policy_model", DataType::Utf8, false),
        Field::new("domain", DataType::Utf8, false),
        Field::new("is_active", DataType::Boolean, false),
        Field::new("created_at", timestamp_utc(), false),
    ]))
}

pub fn sim_runs_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("run_id", DataType::Utf8, false),
        Field::new("repo_id", DataType::Utf8, false),
        Field::new("scope_id", DataType::Utf8, false),
        Field::new("arm_name", DataType::Utf8, false),
        Field::new("domain", DataType::Utf8, false),
        Field::new("benchmark", DataType::Utf8, false),
        Field::new("policy_model", DataType::Utf8, false),
        Field::new("optimizer_model", DataType::Utf8, false),
        Field::new("memory_mode", DataType::Utf8, false),
        Field::new("seed_harness_id", DataType::Utf8, false),
        Field::new("current_harness_id", DataType::Utf8, false),
        Field::new("carried_from_run_id", DataType::Utf8, true),
        Field::new("carried_from_domain", DataType::Utf8, true),
        Field::new("next_problem_num", DataType::UInt32, false),
        Field::new("aliases_json", DataType::Utf8, false),
        Field::new("rounds_completed", DataType::UInt32, false),
        Field::new("created_at", timestamp_utc(), false),
        Field::new("updated_at", timestamp_utc(), false),
    ]))
}

pub fn sim_problems_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Utf8, false),
        Field::new("run_id", DataType::Utf8, false),
        Field::new("repo_id", DataType::Utf8, false),
        Field::new("scope_id", DataType::Utf8, false),
        Field::new("problem_id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("normalized_name", DataType::Utf8, false),
        Field::new("opened_round", DataType::UInt32, false),
        Field::new("origin_domain", DataType::Utf8, false),
        Field::new("origin_run_id", DataType::Utf8, false),
        Field::new("created_at", timestamp_utc(), false),
    ]))
}

pub fn sim_attempts_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("run_id", DataType::Utf8, false),
        Field::new("repo_id", DataType::Utf8, false),
        Field::new("scope_id", DataType::Utf8, false),
        Field::new("problem_id", DataType::Utf8, false),
        Field::new("sequence_idx", DataType::UInt32, false),
        Field::new("round", DataType::UInt32, false),
        Field::new("sample", DataType::Utf8, false),
        Field::new("accepted", DataType::Boolean, false),
        Field::new("backfilled", DataType::Boolean, false),
        Field::new("what", DataType::Utf8, false),
        Field::new("files", string_list_type(), false),
        Field::new("predicted", string_list_type(), false),
        Field::new("fixed", string_list_type(), false),
        Field::new("missed", string_list_type(), false),
        Field::new("passed", DataType::UInt32, false),
        Field::new("total", DataType::UInt32, false),
        Field::new("score", DataType::Float32, false),
        Field::new("domain", DataType::Utf8, false),
        Field::new("benchmark", DataType::Utf8, false),
        Field::new("policy_model", DataType::Utf8, false),
        Field::new("harness_version_id", DataType::Utf8, false),
        Field::new("created_at", timestamp_utc(), false),
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
    let languages = StringArray::from_iter_values(records.iter().map(|r| {
        if r.language.is_empty() {
            infer_language_from_path(&r.file_path)
        } else {
            r.language.as_str()
        }
    }));
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
            Arc::new(languages),
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
    let languages = batch
        .column_by_name("language")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let hashes = col_str(batch, "sha256_hash")?;
    let turns = col_u64(batch, "last_read_turn")?;
    let symbols = col_str(batch, "symbols")?;
    let dependencies = col_list(batch, "dependencies")?;
    let vectors = col_fixed_list(batch, "vector")?;

    let mut out = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        let file_path = file_paths.value(i).to_string();
        let language = match languages {
            Some(arr) if !arr.is_null(i) && !arr.value(i).is_empty() && arr.value(i) != "unknown" => {
                arr.value(i).to_string()
            }
            _ => infer_language_from_path(&file_path).to_string(),
        };
        out.push(CodeGraphSnapshot {
            repo_id: repo_ids.value(i).to_string(),
            scope_id: scope_ids.value(i).to_string(),
            file_path,
            language,
            sha256_hash: hashes.value(i).to_string(),
            last_read_turn: turns.value(i),
            symbols_json: symbols.value(i).to_string(),
            dependencies: read_string_list(dependencies, i),
            vector: read_vector(vectors, i),
        });
    }
    Ok(out)
}

pub fn repo_profiles_to_batch(records: &[RepoProfile]) -> Result<RecordBatch> {
    let schema = repo_profiles_schema();
    let ids = StringArray::from_iter_values(records.iter().map(|r| r.id.as_str()));
    let repo_ids = StringArray::from_iter_values(records.iter().map(|r| r.repo_id.as_str()));
    let scope_ids = StringArray::from_iter_values(records.iter().map(|r| r.scope_id.as_str()));
    let langs_vec: Vec<Vec<String>> = records
        .iter()
        .map(|r| r.languages_override.clone())
        .collect();
    let languages_override = build_string_list_array(&langs_vec);
    let check_cmds =
        StringArray::from_iter(records.iter().map(|r| r.check_command.as_deref()));
    let lint_cmds = StringArray::from_iter(records.iter().map(|r| r.lint_command.as_deref()));
    let test_cmds = StringArray::from_iter(records.iter().map(|r| r.test_command.as_deref()));
    let ignores_vec: Vec<Vec<String>> = records
        .iter()
        .map(|r| r.extra_ignore_globs.clone())
        .collect();
    let extra_ignore_globs = build_string_list_array(&ignores_vec);
    let auto_verify = BooleanArray::from_iter(records.iter().map(|r| r.auto_verify_override));
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
            Arc::new(languages_override),
            Arc::new(check_cmds),
            Arc::new(lint_cmds),
            Arc::new(test_cmds),
            Arc::new(extra_ignore_globs),
            Arc::new(auto_verify),
            Arc::new(updated_at),
        ],
    )?)
}

pub fn batch_to_repo_profiles(batch: &RecordBatch) -> Result<Vec<RepoProfile>> {
    let ids = col_str(batch, "id")?;
    let repo_ids = col_str(batch, "repo_id")?;
    let scope_ids = col_str(batch, "scope_id")?;
    let languages_override = col_list(batch, "languages_override")?;
    let check_cmds = col_str(batch, "check_command")?;
    let lint_cmds = col_str(batch, "lint_command")?;
    let test_cmds = col_str(batch, "test_command")?;
    let extra_ignore_globs = col_list(batch, "extra_ignore_globs")?;
    let auto_verify = col_bool(batch, "auto_verify_override")?;
    let updated_at = col_ts(batch, "updated_at")?;

    let mut out = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        out.push(RepoProfile {
            id: ids.value(i).to_string(),
            repo_id: repo_ids.value(i).to_string(),
            scope_id: scope_ids.value(i).to_string(),
            languages_override: read_string_list(languages_override, i),
            check_command: (!check_cmds.is_null(i)).then(|| check_cmds.value(i).to_string()),
            lint_command: (!lint_cmds.is_null(i)).then(|| lint_cmds.value(i).to_string()),
            test_command: (!test_cmds.is_null(i)).then(|| test_cmds.value(i).to_string()),
            extra_ignore_globs: read_string_list(extra_ignore_globs, i),
            auto_verify_override: (!auto_verify.is_null(i)).then(|| auto_verify.value(i)),
            updated_at: micros_to_utc(updated_at.value(i)),
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

fn col_u32<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt32Array> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
        .ok_or_else(|| MemoryError::Embedding(format!("missing UInt32 column '{name}'")))
}

fn col_f32<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float32Array> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| MemoryError::Embedding(format!("missing Float32 column '{name}'")))
}

fn col_bool<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a BooleanArray> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<BooleanArray>())
        .ok_or_else(|| MemoryError::Embedding(format!("missing Boolean column '{name}'")))
}

pub fn harness_versions_to_batch(records: &[HarnessVersionRecord]) -> Result<RecordBatch> {
    let schema = harness_versions_schema();
    let ids = StringArray::from_iter_values(records.iter().map(|r| r.id.as_str()));
    let repo_ids = StringArray::from_iter_values(records.iter().map(|r| r.repo_id.as_str()));
    let scope_ids = StringArray::from_iter_values(records.iter().map(|r| r.scope_id.as_str()));
    let parent_ids =
        StringArray::from_iter(records.iter().map(|r| r.parent_version_id.as_deref()));
    let run_ids = StringArray::from_iter(records.iter().map(|r| r.run_id.as_deref()));
    let rounds = UInt32Array::from_iter_values(records.iter().map(|r| r.round));
    let labels = StringArray::from_iter_values(records.iter().map(|r| r.label.as_str()));
    let specs = StringArray::from_iter_values(records.iter().map(|r| r.spec_json.as_str()));
    let hashes = StringArray::from_iter_values(records.iter().map(|r| r.sha256_hash.as_str()));
    let policies = StringArray::from_iter_values(records.iter().map(|r| r.policy_model.as_str()));
    let domains = StringArray::from_iter_values(records.iter().map(|r| r.domain.as_str()));
    let actives = BooleanArray::from_iter(records.iter().map(|r| Some(r.is_active)));
    let created_at = TimestampMicrosecondArray::from_iter_values(
        records.iter().map(|r| r.created_at.timestamp_micros()),
    )
    .with_timezone("UTC");

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(repo_ids),
            Arc::new(scope_ids),
            Arc::new(parent_ids),
            Arc::new(run_ids),
            Arc::new(rounds),
            Arc::new(labels),
            Arc::new(specs),
            Arc::new(hashes),
            Arc::new(policies),
            Arc::new(domains),
            Arc::new(actives),
            Arc::new(created_at),
        ],
    )?)
}

pub fn batch_to_harness_versions(batch: &RecordBatch) -> Result<Vec<HarnessVersionRecord>> {
    let ids = col_str(batch, "id")?;
    let repo_ids = col_str(batch, "repo_id")?;
    let scope_ids = col_str(batch, "scope_id")?;
    let parent_ids = col_str(batch, "parent_version_id")?;
    let run_ids = col_str(batch, "run_id")?;
    let rounds = col_u32(batch, "round")?;
    let labels = col_str(batch, "label")?;
    let specs = col_str(batch, "spec_json")?;
    let hashes = col_str(batch, "sha256_hash")?;
    let policies = col_str(batch, "policy_model")?;
    let domains = col_str(batch, "domain")?;
    let actives = col_bool(batch, "is_active")?;
    let created_at = col_ts(batch, "created_at")?;

    let mut out = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        out.push(HarnessVersionRecord {
            id: ids.value(i).to_string(),
            repo_id: repo_ids.value(i).to_string(),
            scope_id: scope_ids.value(i).to_string(),
            parent_version_id: if parent_ids.is_null(i) {
                None
            } else {
                Some(parent_ids.value(i).to_string())
            },
            run_id: if run_ids.is_null(i) {
                None
            } else {
                Some(run_ids.value(i).to_string())
            },
            round: rounds.value(i),
            label: labels.value(i).to_string(),
            spec_json: specs.value(i).to_string(),
            sha256_hash: hashes.value(i).to_string(),
            policy_model: policies.value(i).to_string(),
            domain: domains.value(i).to_string(),
            is_active: actives.value(i),
            created_at: micros_to_utc(created_at.value(i)),
        });
    }
    Ok(out)
}

pub fn sim_runs_to_batch(records: &[SimRunRecord]) -> Result<RecordBatch> {
    let schema = sim_runs_schema();
    let run_ids = StringArray::from_iter_values(records.iter().map(|r| r.run_id.as_str()));
    let repo_ids = StringArray::from_iter_values(records.iter().map(|r| r.repo_id.as_str()));
    let scope_ids = StringArray::from_iter_values(records.iter().map(|r| r.scope_id.as_str()));
    let arm_names = StringArray::from_iter_values(records.iter().map(|r| r.arm_name.as_str()));
    let domains = StringArray::from_iter_values(records.iter().map(|r| r.domain.as_str()));
    let benchmarks = StringArray::from_iter_values(records.iter().map(|r| r.benchmark.as_str()));
    let policies = StringArray::from_iter_values(records.iter().map(|r| r.policy_model.as_str()));
    let optimizers =
        StringArray::from_iter_values(records.iter().map(|r| r.optimizer_model.as_str()));
    let modes =
        StringArray::from_iter_values(records.iter().map(|r| r.memory_mode.as_str()));
    let seed_ids = StringArray::from_iter_values(records.iter().map(|r| r.seed_harness_id.as_str()));
    let curr_ids =
        StringArray::from_iter_values(records.iter().map(|r| r.current_harness_id.as_str()));
    let carried_runs =
        StringArray::from_iter(records.iter().map(|r| r.carried_from_run_id.as_deref()));
    let carried_domains =
        StringArray::from_iter(records.iter().map(|r| r.carried_from_domain.as_deref()));
    let next_nums = UInt32Array::from_iter_values(records.iter().map(|r| r.next_problem_num));
    let aliases = StringArray::from_iter_values(records.iter().map(|r| r.aliases_json.as_str()));
    let rounds = UInt32Array::from_iter_values(records.iter().map(|r| r.rounds_completed));
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
            Arc::new(run_ids),
            Arc::new(repo_ids),
            Arc::new(scope_ids),
            Arc::new(arm_names),
            Arc::new(domains),
            Arc::new(benchmarks),
            Arc::new(policies),
            Arc::new(optimizers),
            Arc::new(modes),
            Arc::new(seed_ids),
            Arc::new(curr_ids),
            Arc::new(carried_runs),
            Arc::new(carried_domains),
            Arc::new(next_nums),
            Arc::new(aliases),
            Arc::new(rounds),
            Arc::new(created_at),
            Arc::new(updated_at),
        ],
    )?)
}

pub fn batch_to_sim_runs(batch: &RecordBatch) -> Result<Vec<SimRunRecord>> {
    let run_ids = col_str(batch, "run_id")?;
    let repo_ids = col_str(batch, "repo_id")?;
    let scope_ids = col_str(batch, "scope_id")?;
    let arm_names = col_str(batch, "arm_name")?;
    let domains = col_str(batch, "domain")?;
    let benchmarks = col_str(batch, "benchmark")?;
    let policies = col_str(batch, "policy_model")?;
    let optimizers = col_str(batch, "optimizer_model")?;
    let modes = col_str(batch, "memory_mode")?;
    let seed_ids = col_str(batch, "seed_harness_id")?;
    let curr_ids = col_str(batch, "current_harness_id")?;
    let carried_runs = col_str(batch, "carried_from_run_id")?;
    let carried_domains = col_str(batch, "carried_from_domain")?;
    let next_nums = col_u32(batch, "next_problem_num")?;
    let aliases = col_str(batch, "aliases_json")?;
    let rounds = col_u32(batch, "rounds_completed")?;
    let created_at = col_ts(batch, "created_at")?;
    let updated_at = col_ts(batch, "updated_at")?;

    let mut out = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        out.push(SimRunRecord {
            run_id: run_ids.value(i).to_string(),
            repo_id: repo_ids.value(i).to_string(),
            scope_id: scope_ids.value(i).to_string(),
            arm_name: arm_names.value(i).to_string(),
            domain: domains.value(i).to_string(),
            benchmark: benchmarks.value(i).to_string(),
            policy_model: policies.value(i).to_string(),
            optimizer_model: optimizers.value(i).to_string(),
            memory_mode: SimMemoryMode::parse(modes.value(i))?,
            seed_harness_id: seed_ids.value(i).to_string(),
            current_harness_id: curr_ids.value(i).to_string(),
            carried_from_run_id: if carried_runs.is_null(i) {
                None
            } else {
                Some(carried_runs.value(i).to_string())
            },
            carried_from_domain: if carried_domains.is_null(i) {
                None
            } else {
                Some(carried_domains.value(i).to_string())
            },
            next_problem_num: next_nums.value(i),
            aliases_json: aliases.value(i).to_string(),
            rounds_completed: rounds.value(i),
            created_at: micros_to_utc(created_at.value(i)),
            updated_at: micros_to_utc(updated_at.value(i)),
        });
    }
    Ok(out)
}

pub fn sim_problems_to_batch(records: &[SimProblemRecord]) -> Result<RecordBatch> {
    let schema = sim_problems_schema();
    let row_ids = StringArray::from_iter_values(records.iter().map(|r| r.row_id.as_str()));
    let run_ids = StringArray::from_iter_values(records.iter().map(|r| r.run_id.as_str()));
    let repo_ids = StringArray::from_iter_values(records.iter().map(|r| r.repo_id.as_str()));
    let scope_ids = StringArray::from_iter_values(records.iter().map(|r| r.scope_id.as_str()));
    let problem_ids = StringArray::from_iter_values(records.iter().map(|r| r.problem_id.as_str()));
    let names = StringArray::from_iter_values(records.iter().map(|r| r.name.as_str()));
    let norm_names =
        StringArray::from_iter_values(records.iter().map(|r| r.normalized_name.as_str()));
    let opened = UInt32Array::from_iter_values(records.iter().map(|r| r.opened_round));
    let origin_domains =
        StringArray::from_iter_values(records.iter().map(|r| r.origin_domain.as_str()));
    let origin_runs =
        StringArray::from_iter_values(records.iter().map(|r| r.origin_run_id.as_str()));
    let created_at = TimestampMicrosecondArray::from_iter_values(
        records.iter().map(|r| r.created_at.timestamp_micros()),
    )
    .with_timezone("UTC");

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(row_ids),
            Arc::new(run_ids),
            Arc::new(repo_ids),
            Arc::new(scope_ids),
            Arc::new(problem_ids),
            Arc::new(names),
            Arc::new(norm_names),
            Arc::new(opened),
            Arc::new(origin_domains),
            Arc::new(origin_runs),
            Arc::new(created_at),
        ],
    )?)
}

pub fn batch_to_sim_problems(batch: &RecordBatch) -> Result<Vec<SimProblemRecord>> {
    let row_ids = col_str(batch, "row_id")?;
    let run_ids = col_str(batch, "run_id")?;
    let repo_ids = col_str(batch, "repo_id")?;
    let scope_ids = col_str(batch, "scope_id")?;
    let problem_ids = col_str(batch, "problem_id")?;
    let names = col_str(batch, "name")?;
    let norm_names = col_str(batch, "normalized_name")?;
    let opened = col_u32(batch, "opened_round")?;
    let origin_domains = col_str(batch, "origin_domain")?;
    let origin_runs = col_str(batch, "origin_run_id")?;
    let created_at = col_ts(batch, "created_at")?;

    let mut out = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        out.push(SimProblemRecord {
            row_id: row_ids.value(i).to_string(),
            run_id: run_ids.value(i).to_string(),
            repo_id: repo_ids.value(i).to_string(),
            scope_id: scope_ids.value(i).to_string(),
            problem_id: problem_ids.value(i).to_string(),
            name: names.value(i).to_string(),
            normalized_name: norm_names.value(i).to_string(),
            opened_round: opened.value(i),
            origin_domain: origin_domains.value(i).to_string(),
            origin_run_id: origin_runs.value(i).to_string(),
            created_at: micros_to_utc(created_at.value(i)),
        });
    }
    Ok(out)
}

pub fn sim_attempts_to_batch(records: &[SimAttemptRecord]) -> Result<RecordBatch> {
    let schema = sim_attempts_schema();
    let ids = StringArray::from_iter_values(records.iter().map(|r| r.id.as_str()));
    let run_ids = StringArray::from_iter_values(records.iter().map(|r| r.run_id.as_str()));
    let repo_ids = StringArray::from_iter_values(records.iter().map(|r| r.repo_id.as_str()));
    let scope_ids = StringArray::from_iter_values(records.iter().map(|r| r.scope_id.as_str()));
    let problem_ids = StringArray::from_iter_values(records.iter().map(|r| r.problem_id.as_str()));
    let seqs = UInt32Array::from_iter_values(records.iter().map(|r| r.sequence_idx));
    let rounds = UInt32Array::from_iter_values(records.iter().map(|r| r.round));
    let samples = StringArray::from_iter_values(records.iter().map(|r| r.sample.as_str()));
    let accepted = BooleanArray::from_iter(records.iter().map(|r| Some(r.accepted)));
    let backfilled = BooleanArray::from_iter(records.iter().map(|r| Some(r.backfilled)));
    let whats = StringArray::from_iter_values(records.iter().map(|r| r.what.as_str()));
    let files_vec: Vec<Vec<String>> = records.iter().map(|r| r.files.clone()).collect();
    let files = build_string_list_array(&files_vec);
    let pred_vec: Vec<Vec<String>> = records.iter().map(|r| r.predicted.clone()).collect();
    let predicted = build_string_list_array(&pred_vec);
    let fixed_vec: Vec<Vec<String>> = records.iter().map(|r| r.fixed.clone()).collect();
    let fixed = build_string_list_array(&fixed_vec);
    let missed_vec: Vec<Vec<String>> = records.iter().map(|r| r.missed.clone()).collect();
    let missed = build_string_list_array(&missed_vec);
    let passed = UInt32Array::from_iter_values(records.iter().map(|r| r.passed));
    let total = UInt32Array::from_iter_values(records.iter().map(|r| r.total));
    let scores = Float32Array::from_iter_values(records.iter().map(|r| r.score));
    let domains = StringArray::from_iter_values(records.iter().map(|r| r.domain.as_str()));
    let benchmarks = StringArray::from_iter_values(records.iter().map(|r| r.benchmark.as_str()));
    let policies = StringArray::from_iter_values(records.iter().map(|r| r.policy_model.as_str()));
    let harness_ids =
        StringArray::from_iter_values(records.iter().map(|r| r.harness_version_id.as_str()));
    let created_at = TimestampMicrosecondArray::from_iter_values(
        records.iter().map(|r| r.created_at.timestamp_micros()),
    )
    .with_timezone("UTC");

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(run_ids),
            Arc::new(repo_ids),
            Arc::new(scope_ids),
            Arc::new(problem_ids),
            Arc::new(seqs),
            Arc::new(rounds),
            Arc::new(samples),
            Arc::new(accepted),
            Arc::new(backfilled),
            Arc::new(whats),
            Arc::new(files),
            Arc::new(predicted),
            Arc::new(fixed),
            Arc::new(missed),
            Arc::new(passed),
            Arc::new(total),
            Arc::new(scores),
            Arc::new(domains),
            Arc::new(benchmarks),
            Arc::new(policies),
            Arc::new(harness_ids),
            Arc::new(created_at),
        ],
    )?)
}

pub fn batch_to_sim_attempts(batch: &RecordBatch) -> Result<Vec<SimAttemptRecord>> {
    let ids = col_str(batch, "id")?;
    let run_ids = col_str(batch, "run_id")?;
    let repo_ids = col_str(batch, "repo_id")?;
    let scope_ids = col_str(batch, "scope_id")?;
    let problem_ids = col_str(batch, "problem_id")?;
    let seqs = col_u32(batch, "sequence_idx")?;
    let rounds = col_u32(batch, "round")?;
    let samples = col_str(batch, "sample")?;
    let accepted = col_bool(batch, "accepted")?;
    let backfilled = col_bool(batch, "backfilled")?;
    let whats = col_str(batch, "what")?;
    let files = col_list(batch, "files")?;
    let predicted = col_list(batch, "predicted")?;
    let fixed = col_list(batch, "fixed")?;
    let missed = col_list(batch, "missed")?;
    let passed = col_u32(batch, "passed")?;
    let total = col_u32(batch, "total")?;
    let scores = col_f32(batch, "score")?;
    let domains = col_str(batch, "domain")?;
    let benchmarks = col_str(batch, "benchmark")?;
    let policies = col_str(batch, "policy_model")?;
    let harness_ids = col_str(batch, "harness_version_id")?;
    let created_at = col_ts(batch, "created_at")?;

    let mut out = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        out.push(SimAttemptRecord {
            id: ids.value(i).to_string(),
            run_id: run_ids.value(i).to_string(),
            repo_id: repo_ids.value(i).to_string(),
            scope_id: scope_ids.value(i).to_string(),
            problem_id: problem_ids.value(i).to_string(),
            sequence_idx: seqs.value(i),
            round: rounds.value(i),
            sample: samples.value(i).to_string(),
            accepted: accepted.value(i),
            backfilled: backfilled.value(i),
            what: whats.value(i).to_string(),
            files: read_string_list(files, i),
            predicted: read_string_list(predicted, i),
            fixed: read_string_list(fixed, i),
            missed: read_string_list(missed, i),
            passed: passed.value(i),
            total: total.value(i),
            score: scores.value(i),
            domain: domains.value(i).to_string(),
            benchmark: benchmarks.value(i).to_string(),
            policy_model: policies.value(i).to_string(),
            harness_version_id: harness_ids.value(i).to_string(),
            created_at: micros_to_utc(created_at.value(i)),
        });
    }
    Ok(out)
}
