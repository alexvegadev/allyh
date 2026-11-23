use thiserror::Error;

/// Errors returned by `allyh-env` operations.
#[derive(Debug, Error)]
pub enum EnvError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Git error: {0}")]
    Git(#[from] git2::Error),

    #[error("File watcher error: {0}")]
    Notify(#[from] notify::Error),

    #[error("Memory store error: {0}")]
    Memory(#[from] allyh_memory::MemoryError),

    #[error("JSON serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Tree-sitter error: {0}")]
    TreeSitter(String),

    #[error("Invalid path '{0}': must stay inside repository root")]
    PathEscape(String),

    #[error("Command execution failed: {0}")]
    Command(String),
}

pub type Result<T> = std::result::Result<T, EnvError>;
