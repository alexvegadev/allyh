//! Self-Improvement Memory (SIM) and Continual Recursive Self-Improvement (CRSI) engine
//! implementing arXiv:2611.18856 (Google Cloud AI, Google DeepMind, Penn State).
//!
//! Key guarantees:
//! - Strict separation between meta-level SIM (`allyh-sim` / `sim_*` tables) and task-solving memory
//! - Type-enforced disjoint `EvolveSplit` vs. `HeldOutSplit` boundary
//! - Complete-SIM conditioning (zero lossy truncation) for both Proposer and two-stage Selector
//! - Pre-evaluation `ChangeManifest` locking ex-ante `predicted` fixes before ex-post verification
//! - All four CRSI transfer conditions (`ColdStart`, `HarnessTransfer`, `ProblemOnly`, `FullSim`)

pub mod error;
pub mod eval;
pub mod harvest;
pub mod manifest;
pub mod presets;
pub mod proposer;
pub mod runner;
pub mod selector;

pub use error::{Result, SimError};
pub use eval::{
    AllyHRuntimeEvaluator, BenchmarkEvaluator, CommandHarnessEvaluator, EvalFeedback,
    EvalSuiteSpec, EvalTask, EvolveSplit, HeldOutSplit, TaskRolloutResult,
    UnifiedMultiModelEvaluator,
};
pub use harvest::{SessionHarvestReport, harvest_sessions_into_sim};
pub use manifest::{CandidateProposal, ChangeManifest, RoundHistoryEntry, RunHistory};
pub use presets::{SUITE_PRESET_NAMES, list_suite_presets, polyglot_suite_preset};
pub use proposer::{HeuristicSimProposer, LlmSimProposer, SimProposer};
pub use runner::{CrsiConfig, CrsiRunReport, CrsiRunner, RoundSummary};
pub use selector::{
    CandidateObjectiveBreakdown, EvaluatedCandidate, HeuristicSimSelector, LlmSimSelector,
    MultiObjectiveWeights, ScreeningDecision, SelectionVerdict, SimSelector,
    evaluate_multi_objective,
};
