use thiserror::Error;

/// Errors returned by `allyh-sim` CRSI optimization, evaluation, and proposal/selection loops.
#[derive(Debug, Error)]
pub enum SimError {
    #[error("Memory engine error: {0}")]
    Memory(#[from] allyh_memory::MemoryError),

    #[error("Core orchestrator error: {0}")]
    Core(Box<allyh_core::CoreError>),

    #[error("Environment error: {0}")]
    Env(#[from] allyh_env::EnvError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid evaluation suite: {0}")]
    InvalidSuite(String),

    #[error("Evaluation failed: {0}")]
    Evaluation(String),

    #[error("Proposer error: {0}")]
    Proposal(String),

    #[error("Selector error: {0}")]
    Selection(String),
}

pub type Result<T> = std::result::Result<T, SimError>;

impl From<allyh_core::CoreError> for SimError {
    fn from(err: allyh_core::CoreError) -> Self {
        Self::Core(Box::new(err))
    }
}
