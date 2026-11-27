use thiserror::Error;

/// Errors returned by `allyh-tools` ACI tool executions.
#[derive(Debug, Error)]
pub enum ToolError {
    #[error("Environment error: {0}")]
    Env(#[from] allyh_env::EnvError),

    #[error("Memory store error: {0}")]
    Memory(#[from] allyh_memory::MemoryError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Write gate locked: {0}")]
    WriteGateLocked(String),

    #[error("Stale file hash for '{file_path}': expected {expected}, actual {actual}. Re-inspect the file before editing.")]
    StaleHash {
        file_path: String,
        expected: String,
        actual: String,
    },

    #[error("Edit target not found in '{file_path}': {reason}")]
    EditTargetNotFound { file_path: String, reason: String },

    #[error("Command approval required for '{command}': {reason}")]
    CommandApprovalRequired { command: String, reason: String },

    #[error("Command '{command}' timed out after {timeout_secs}s")]
    CommandTimedOut { command: String, timeout_secs: u64 },

    #[error("Invalid tool arguments: {0}")]
    InvalidArgs(String),

    #[error("Unknown tool '{0}'")]
    UnknownTool(String),
}

pub type Result<T> = std::result::Result<T, ToolError>;
