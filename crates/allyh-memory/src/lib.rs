//! AllyH out-of-tree LanceDB memory engine:
//! - Configurable local disk (`~/.local/share/allyh/db`) or shared object store (`s3://...`) URI
//! - Apache Arrow schemas for `sessions`, `architectural_decisions`, `tasks`, and `code_graph_snapshots`
//! - Hybrid vector + BM25 FTS search partitioned by `repo_id` and `scope_id`
//! - Write-gated bi-temporal conflict consolidator (`ADD` / `UPDATE` / `INVALIDATE` / `NOOP`)
//! - Read-only asynchronous sleep-time compute worker

pub mod config;
pub mod consolidator;
pub mod embeddings;
pub mod error;
pub mod schema;
pub mod store;

pub use config::{DEFAULT_SCOPE_ID, StorageBackendKind, StorageConfig, compute_repo_id};
pub use consolidator::{
    CandidateDecision, ConflictJudge, ConsolidationAction, ConsolidationOutcome,
    HeuristicConflictJudge, MemoryConsolidator, SleepJob, SleepTimeWorkerHandle, WorkerStatus,
};
pub use embeddings::{
    DeterministicHashEmbedder, EmbeddingProvider, OpenAiCompatibleEmbedder, cosine_similarity,
    fit_embedding_dim,
};
pub use error::{MemoryError, Result};
pub use schema::{
    ArchitecturalDecision, CodeGraphSnapshot, DbStatusReport, EMBEDDING_DIM, ScoredDecision,
    SessionRecord, TABLE_ARCHITECTURAL_DECISIONS, TABLE_CODE_GRAPH_SNAPSHOTS, TABLE_SESSIONS,
    TABLE_TASKS, TaskRecord, TaskStatus, VerificationProof,
};
pub use store::MemoryStore;
