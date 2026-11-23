use thiserror::Error;

/// Errors returned by `allyh-core` orchestration, routing, and gate operations.
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("Memory engine error: {0}")]
    Memory(#[from] allyh_memory::MemoryError),

    #[error("Environment monitor error: {0}")]
    Env(#[from] allyh_env::EnvError),

    #[error("ACI tool error: {0}")]
    Tool(#[from] allyh_tools::ToolError),

    #[error("Rig provider error: {0}")]
    Provider(#[from] rig_core::ProviderError),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid model specification '{0}'")]
    InvalidModelSpec(String),

    #[error("Control gate error: {0}")]
    ControlGate(String),

    #[error("Orchestrator channel closed")]
    ChannelClosed,
}

pub type Result<T> = std::result::Result<T, CoreError>;
