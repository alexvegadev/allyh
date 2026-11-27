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
pub mod manifest;
pub mod proposer;
pub mod runner;
pub mod selector;

pub use error::{Result, SimError};
pub use eval::{
    AllyHRuntimeEvaluator, BenchmarkEvaluator, CommandHarnessEvaluator, EvalFeedback,
    EvalSuiteSpec, EvalTask, EvolveSplit, HeldOutSplit, TaskRolloutResult,
    UnifiedMultiModelEvaluator,
};
pub use manifest::{CandidateProposal, ChangeManifest, RoundHistoryEntry, RunHistory};
pub use proposer::{HeuristicSimProposer, LlmSimProposer, SimProposer};
pub use runner::{CrsiConfig, CrsiRunReport, CrsiRunner, RoundSummary};
pub use selector::{
    EvaluatedCandidate, HeuristicSimSelector, LlmSimSelector, ScreeningDecision, SelectionVerdict,
    SimSelector,
};
