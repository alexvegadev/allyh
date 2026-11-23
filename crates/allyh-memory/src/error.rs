use thiserror::Error;

/// Errors returned by `allyh-memory` operations.
#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("LanceDB error: {0}")]
    LanceDb(#[from] lancedb::Error),

    #[error("Apache Arrow error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("HTTP embedding request error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Invalid task state transition: {0}")]
    InvalidTransition(String),

    #[error("Record not found: {0}")]
    NotFound(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Invalid storage configuration: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, MemoryError>;
