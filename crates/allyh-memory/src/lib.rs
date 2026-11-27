//! AllyH out-of-tree LanceDB memory engine:
//! - Configurable OS-native local disk directory or shared object store (`s3://...`) URI
//! - Apache Arrow schemas for `sessions`, `architectural_decisions`, `tasks`, and `code_graph_snapshots`
//! - Hybrid vector + BM25 FTS search partitioned by `repo_id` and `scope_id`
//! - Write-gated bi-temporal conflict consolidator (`ADD` / `UPDATE` / `INVALIDATE` / `NOOP`)
//! - Read-only asynchronous sleep-time compute worker

pub mod config;
pub mod consolidator;
pub mod embeddings;
pub mod error;
pub mod schema;
pub mod sim_ledger;
pub mod skill_doc;
pub mod store;

pub use config::{
    DEFAULT_SCOPE_ID, StorageBackendKind, StorageConfig, compute_repo_id, expand_local_uri,
};
pub use consolidator::{
    CandidateDecision, ConflictJudge, ConsolidationAction, ConsolidationOutcome,
    HeuristicConflictJudge, MemoryConsolidator, SleepJob, SleepTimeWorkerHandle, WorkerStatus,
};
pub use embeddings::{
    DEFAULT_LOCAL_EMBEDDING_BASE_URL, DEFAULT_LOCAL_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_EMBEDDING_BASE_URL, DeterministicHashEmbedder, EmbeddingProfile,
    EmbeddingProvider, EmbeddingProviderKind, OpenAiCompatibleEmbedder, cosine_similarity,
    fit_embedding_dim,
};
pub use error::{MemoryError, Result};
pub use schema::{
    ArchitecturalDecision, CodeGraphSnapshot, DbStatusReport, EMBEDDING_DIM, HarnessVersionRecord,
    ScoredDecision, SessionRecord, SimAttemptRecord, SimMemoryMode, SimProblemRecord, SimRunRecord,
    SkillRecord, TABLE_ARCHITECTURAL_DECISIONS, TABLE_CODE_GRAPH_SNAPSHOTS,
    TABLE_HARNESS_VERSIONS, TABLE_SESSIONS, TABLE_SIM_ATTEMPTS, TABLE_SIM_PROBLEMS, TABLE_SIM_RUNS,
    TABLE_SKILLS, TABLE_TASKS, TaskRecord, TaskStatus, VerificationProof, normalize_skill_slug,
    skill_id,
};
pub use sim_ledger::{
    SIM_PROMPT_RULES, SimAttemptEntry, SimAttemptInput, SimCarriedFrom, SimExportDocument,
    SimLedger, SimProblemEntry, SimProblemRef, SimUpdateResult, normalize_root_cause,
};
pub use skill_doc::{
    SkillDocument, load_skill_markdown_file, parse_skill_markdown, render_skill_markdown,
};
pub use store::MemoryStore;
