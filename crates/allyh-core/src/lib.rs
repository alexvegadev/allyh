//! AllyH core orchestrator, control gates, hybrid router, dynamic preamble assembler,
//! non-destructive history compaction, and bi-temporal memory consolidation.

pub mod compaction;
pub mod consolidation;
pub mod error;
pub mod gate;
pub mod harness;
pub mod interlock;
pub mod orchestrator;
pub mod preamble;
pub mod router;

pub use compaction::{CompactingHistory, HistoryUnit};
pub use consolidation::{
    LocalLlmConflictJudge, extract_candidate_decisions, extract_heuristic_decisions,
};
pub use error::{CoreError, Result};
pub use gate::{ControlGate, ControlGateState, ProposedPlan, ProposedTaskSpec};
pub use harness::{
    CompactionPolicy, CompletionInterlockPolicy, HarnessSpec, ModelAdaptationPolicy,
    PreamblePolicy, RetrievalPolicy, ToolPolicy,
};
pub use interlock::{CompletionInterlock, InterlockDecision};
pub use orchestrator::{
    AllyHOrchestrator, InspectorSnapshot, OrchestratorConfig, TurnOutcome, UiEvent, UserCommand,
};
pub use preamble::{AssembledPreamble, DynamicPreambleAssembler, PreambleInputs, estimate_tokens};
pub use router::{
    CompletionBackend, HybridRouter, ModelProfile, ModelProviderKind, RigCompletionBackend,
};
