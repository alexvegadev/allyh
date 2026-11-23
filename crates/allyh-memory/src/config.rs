use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Default multi-agent scope identifier.
pub const DEFAULT_SCOPE_ID: &str = "default";

/// Indicates whether LanceDB is operating on local disk or shared object storage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageBackendKind {
    Local,
    S3Shared,
    GcsShared,
    AzureShared,
    Remote,
}

impl StorageBackendKind {
    /// Human-readable label shown in the TUI Coherence Inspector sidebar.
    pub fn display_label(&self) -> &'static str {
        match self {
            Self::Local => "LanceDB: Local",
            Self::S3Shared => "LanceDB: S3 Shared",
            Self::GcsShared => "LanceDB: GCS Shared",
            Self::AzureShared => "LanceDB: Azure Shared",
            Self::Remote => "LanceDB: Remote",
        }
    }
}

/// Configuration for connecting AllyH's out-of-tree LanceDB storage backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Local filesystem directory or object-store URI (`s3://...`, `gs://...`, `az://...`).
    pub uri: String,
    /// Optional object-store credentials / endpoint options (e.g., MinIO endpoint, AWS keys).
    #[serde(default)]
    pub storage_options: HashMap<String, String>,
}

impl StorageConfig {
    /// Default local out-of-tree path (`~/.local/share/allyh/db`).
    pub fn default_local() -> Self {
        let base = dirs::data_local_dir()
            .or_else(dirs::home_dir)
            .unwrap_or_else(|| PathBuf::from("."));
        let db_path = base.join("allyh").join("db");
        Self {
            uri: db_path.to_string_lossy().into_owned(),
            storage_options: HashMap::new(),
        }
    }

    /// Build a storage config from an explicit URI or local path string.
    pub fn from_uri(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            storage_options: HashMap::new(),
        }
    }

    /// Build a storage config pointing to a local filesystem path.
    pub fn from_local_path(path: impl AsRef<Path>) -> Self {
        Self {
            uri: path.as_ref().to_string_lossy().into_owned(),
            storage_options: HashMap::new(),
        }
    }

    /// Attach object-store options (e.g., `aws_endpoint`, `aws_access_key_id`, `allow_http`).
    pub fn with_storage_option(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.storage_options.insert(key.into(), value.into());
        self
    }

    /// Detect the backend kind from the configured URI.
    pub fn backend_kind(&self) -> StorageBackendKind {
        let lower = self.uri.to_ascii_lowercase();
        if lower.starts_with("s3://") || lower.starts_with("s3a://") {
            StorageBackendKind::S3Shared
        } else if lower.starts_with("gs://") || lower.starts_with("gcs://") {
            StorageBackendKind::GcsShared
        } else if lower.starts_with("az://") || lower.starts_with("abfs://") {
            StorageBackendKind::AzureShared
        } else if lower.starts_with("db://") {
            StorageBackendKind::Remote
        } else {
            StorageBackendKind::Local
        }
    }

    /// Returns `true` if this URI targets a local filesystem directory.
    pub fn is_local(&self) -> bool {
        self.backend_kind() == StorageBackendKind::Local
    }
}

/// Compute a deterministic `repo_id` (SHA-256 hex) from a repository's origin URL
/// or canonical root path.
pub fn compute_repo_id(canonical_identifier: &str) -> String {
    let normalized = canonical_identifier.trim().replace('\\', "/");
    let mut hasher = Sha256::new();
    hasher.update(normalized.as_bytes());
    hex::encode(hasher.finalize())
}

/// Escape single quotes for LanceDB SQL string literals.
pub(crate) fn escape_sql_literal(value: &str) -> String {
    value.replace('\'', "''")
}
