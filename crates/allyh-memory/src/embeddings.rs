use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{MemoryError, Result};
use crate::schema::EMBEDDING_DIM;

/// Default local `llama.cpp` / OpenAI-compatible embedding endpoint.
pub const DEFAULT_LOCAL_EMBEDDING_BASE_URL: &str = "http://localhost:9931/v1";
/// Default local Ollama OpenAI-compatible embedding endpoint.
pub const DEFAULT_OLLAMA_EMBEDDING_BASE_URL: &str = "http://localhost:11434/v1";
/// Default local embedding model when only `ALLYH_EMBEDDING_BASE_URL` is configured.
pub const DEFAULT_LOCAL_EMBEDDING_MODEL: &str = "Qwen3-Embedding-0.6B";
/// Default OpenAI embedding API base URL.
pub const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
/// Default Voyage AI embedding API base URL.
pub const DEFAULT_VOYAGE_BASE_URL: &str = "https://api.voyageai.com/v1";
/// Default Jina AI embedding API base URL.
pub const DEFAULT_JINA_BASE_URL: &str = "https://api.jina.ai/v1";
/// Default Google Gemini OpenAI-compatible API base URL.
pub const DEFAULT_GEMINI_OPENAI_BASE_URL: &str =
    "https://generativelanguage.googleapis.com/v1beta/openai";

/// Provider-agnostic embedding generator for AllyH hybrid vector search.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    fn dimensions(&self) -> usize {
        EMBEDDING_DIM as usize
    }
}

/// Deterministic, zero-dependency feature-hashing embedder (unigrams + bigrams + char 3-grams).
///
/// Used as a reliable offline fallback and for deterministic acceptance tests.
#[derive(Debug, Default, Clone)]
pub struct DeterministicHashEmbedder;

impl DeterministicHashEmbedder {
    pub fn embed_sync(text: &str) -> Vec<f32> {
        let dim = EMBEDDING_DIM as usize;
        let mut vec = vec![0.0_f32; dim];

        let tokens: Vec<String> = text
            .to_ascii_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
            .filter(|t| t.len() >= 2)
            .map(str::to_string)
            .collect();

        if tokens.is_empty() {
            vec[0] = 1.0;
            return vec;
        }

        // Token unigrams (higher weight)
        for token in &tokens {
            accumulate_feature(&mut vec, token, 2.0);
            // Character trigrams for morphological / identifier overlap
            let chars: Vec<char> = token.chars().collect();
            if chars.len() >= 3 {
                for window in chars.windows(3) {
                    let gram: String = window.iter().collect();
                    accumulate_feature(&mut vec, &format!("g3:{gram}"), 0.5);
                }
            }
        }

        // Token bigrams
        for pair in tokens.windows(2) {
            accumulate_feature(&mut vec, &format!("bi:{}_{}", pair[0], pair[1]), 1.25);
        }

        l2_normalize(&mut vec);
        vec
    }
}

fn accumulate_feature(vec: &mut [f32], feature: &str, weight: f32) {
    let mut hasher = Sha256::new();
    hasher.update(feature.as_bytes());
    let digest = hasher.finalize();
    let idx = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]) as usize % vec.len();
    let sign = if digest[4] & 1 == 0 { 1.0 } else { -1.0 };
    vec[idx] += sign * weight;
}

pub(crate) fn l2_normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for v in vec.iter_mut() {
            *v /= norm;
        }
    } else if let Some(first) = vec.first_mut() {
        *first = 1.0;
    }
}

/// Resize any upstream embedding vector to `EMBEDDING_DIM` via Matryoshka (MRL)
/// prefix-truncation (or zero-padding if shorter) followed by L2 normalization.
pub fn fit_embedding_dim(raw: &[f32]) -> Vec<f32> {
    let dim = EMBEDDING_DIM as usize;
    let mut out = vec![0.0_f32; dim];
    if raw.is_empty() {
        out[0] = 1.0;
        return out;
    }
    let copy_len = raw.len().min(dim);
    out[..copy_len].copy_from_slice(&raw[..copy_len]);
    l2_normalize(&mut out);
    out
}

#[async_trait]
impl EmbeddingProvider for DeterministicHashEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(Self::embed_sync(text))
    }
}

/// Embedding provider family supported by AllyH.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingProviderKind {
    LocalOpenAi,
    OpenAi,
    Voyage,
    Jina,
    Gemini,
    OfflineHash,
}

/// Resolved configuration for constructing an [`EmbeddingProvider`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingProfile {
    pub label: String,
    pub provider: EmbeddingProviderKind,
    pub model_id: String,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    /// Optional OpenAI-style `dimensions` hint sent in `/embeddings` requests.
    /// Left `None` for servers/providers where MRL client-side prefix-truncation is safer.
    pub request_dimensions: Option<usize>,
}

impl EmbeddingProfile {
    /// Deterministic offline feature-hashing profile (zero network dependency).
    pub fn offline() -> Self {
        Self {
            label: "Offline: DeterministicHash".to_string(),
            provider: EmbeddingProviderKind::OfflineHash,
            model_id: "deterministic-hash-384".to_string(),
            base_url: None,
            api_key: None,
            request_dimensions: None,
        }
    }

    /// Auto-detect the best configured embedding provider from environment variables,
    /// falling back to [`Self::offline`] when no remote/local embedding provider is configured.
    pub fn from_env() -> Self {
        Self::from_env_with(read_non_empty_env)
    }

    /// Parse an explicit embedding model selector such as:
    /// - `local:Qwen3-Embedding-0.6B` or `local:bge-small-en-v1.5@http://localhost:9931/v1`
    /// - `ollama:nomic-embed-text` or `ollama:qwen3-embedding:0.6b@http://localhost:11434/v1`
    /// - `voyage:voyage-code-3`
    /// - `jina:jina-embeddings-v3`
    /// - `gemini:gemini-embedding-001`
    /// - `openai:text-embedding-3-small`
    /// - `offline` / `hash`
    pub fn parse_spec(spec: &str) -> Result<Self> {
        Self::parse_spec_with_env(spec, read_non_empty_env)
    }

    fn from_env_with<F>(env: F) -> Self
    where
        F: Fn(&str) -> Option<String>,
    {
        if let Some(spec) = env("ALLYH_EMBEDDING_MODEL") {
            match Self::parse_spec_with_env(&spec, &env) {
                Ok(profile) => return profile,
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        spec = %spec,
                        "invalid ALLYH_EMBEDDING_MODEL; falling back to offline hash embedder"
                    );
                    return Self::offline();
                }
            }
        }

        if let Some(base_url) = env("ALLYH_EMBEDDING_BASE_URL") {
            return Self {
                label: format!("Local Embed: {DEFAULT_LOCAL_EMBEDDING_MODEL}"),
                provider: EmbeddingProviderKind::LocalOpenAi,
                model_id: DEFAULT_LOCAL_EMBEDDING_MODEL.to_string(),
                base_url: Some(base_url),
                api_key: env("ALLYH_EMBEDDING_API_KEY").or_else(|| env("LLAMACPP_API_KEY")),
                request_dimensions: None,
            };
        }

        if let Some(api_key) = env("VOYAGE_API_KEY") {
            return Self {
                label: "Voyage Embed: voyage-code-3".to_string(),
                provider: EmbeddingProviderKind::Voyage,
                model_id: "voyage-code-3".to_string(),
                base_url: Some(
                    env("VOYAGE_BASE_URL").unwrap_or_else(|| DEFAULT_VOYAGE_BASE_URL.to_string()),
                ),
                api_key: Some(api_key),
                request_dimensions: None,
            };
        }

        if let Some(api_key) = env("JINA_API_KEY") {
            return Self {
                label: "Jina Embed: jina-embeddings-v3".to_string(),
                provider: EmbeddingProviderKind::Jina,
                model_id: "jina-embeddings-v3".to_string(),
                base_url: Some(
                    env("JINA_BASE_URL").unwrap_or_else(|| DEFAULT_JINA_BASE_URL.to_string()),
                ),
                api_key: Some(api_key),
                request_dimensions: Some(EMBEDDING_DIM as usize),
            };
        }

        if let Some(api_key) = env("GEMINI_API_KEY").or_else(|| env("GOOGLE_API_KEY")) {
            return Self {
                label: "Gemini Embed: gemini-embedding-001".to_string(),
                provider: EmbeddingProviderKind::Gemini,
                model_id: "gemini-embedding-001".to_string(),
                base_url: Some(
                    env("GEMINI_BASE_URL")
                        .unwrap_or_else(|| DEFAULT_GEMINI_OPENAI_BASE_URL.to_string()),
                ),
                api_key: Some(api_key),
                request_dimensions: None,
            };
        }

        if let Some(api_key) = env("OPENAI_API_KEY") {
            return Self {
                label: "OpenAI Embed: text-embedding-3-small".to_string(),
                provider: EmbeddingProviderKind::OpenAi,
                model_id: "text-embedding-3-small".to_string(),
                base_url: Some(
                    env("OPENAI_BASE_URL").unwrap_or_else(|| DEFAULT_OPENAI_BASE_URL.to_string()),
                ),
                api_key: Some(api_key),
                request_dimensions: Some(EMBEDDING_DIM as usize),
            };
        }

        Self::offline()
    }

    fn parse_spec_with_env<F>(spec: &str, env: F) -> Result<Self>
    where
        F: Fn(&str) -> Option<String>,
    {
        let trimmed = spec.trim();
        if trimmed.eq_ignore_ascii_case("offline")
            || trimmed.eq_ignore_ascii_case("hash")
            || trimmed.eq_ignore_ascii_case("deterministic")
        {
            return Ok(Self::offline());
        }

        let Some((scheme, rest)) = trimmed.split_once(':') else {
            return Err(MemoryError::Embedding(format!(
                "invalid embedding model spec '{trimmed}' (expected local:<model>[@url], ollama:<model>[@url], voyage:<model>, jina:<model>, gemini:<model>, openai:<model>, or offline)"
            )));
        };

        let (model_id, explicit_url) = split_model_and_url(rest)?;
        let override_url = explicit_url.or_else(|| env("ALLYH_EMBEDDING_BASE_URL"));
        let override_key = env("ALLYH_EMBEDDING_API_KEY");

        match scheme.to_ascii_lowercase().as_str() {
            "local" | "llamacpp" => {
                let base_url = override_url
                    .or_else(|| env("ALLYH_LOCAL_BASE_URL"))
                    .or_else(|| env("LLAMACPP_API_BASE_URL"))
                    .unwrap_or_else(|| DEFAULT_LOCAL_EMBEDDING_BASE_URL.to_string());
                Ok(Self {
                    label: format!("Local Embed: {model_id}"),
                    provider: EmbeddingProviderKind::LocalOpenAi,
                    model_id,
                    base_url: Some(base_url),
                    api_key: override_key.or_else(|| env("LLAMACPP_API_KEY")),
                    request_dimensions: None,
                })
            }
            "ollama" => {
                let base_url = override_url
                    .or_else(|| env("OLLAMA_HOST").map(|h| normalize_ollama_v1_url(&h)))
                    .unwrap_or_else(|| DEFAULT_OLLAMA_EMBEDDING_BASE_URL.to_string());
                Ok(Self {
                    label: format!("Ollama Embed: {model_id}"),
                    provider: EmbeddingProviderKind::LocalOpenAi,
                    model_id,
                    base_url: Some(base_url),
                    api_key: override_key,
                    request_dimensions: None,
                })
            }
            "openai" => {
                let base_url = override_url
                    .or_else(|| env("OPENAI_BASE_URL"))
                    .unwrap_or_else(|| DEFAULT_OPENAI_BASE_URL.to_string());
                let request_dimensions = model_id
                    .starts_with("text-embedding-3")
                    .then_some(EMBEDDING_DIM as usize);
                Ok(Self {
                    label: format!("OpenAI Embed: {model_id}"),
                    provider: EmbeddingProviderKind::OpenAi,
                    model_id,
                    base_url: Some(base_url),
                    api_key: override_key.or_else(|| env("OPENAI_API_KEY")),
                    request_dimensions,
                })
            }
            "voyage" => {
                let base_url = override_url
                    .or_else(|| env("VOYAGE_BASE_URL"))
                    .unwrap_or_else(|| DEFAULT_VOYAGE_BASE_URL.to_string());
                Ok(Self {
                    label: format!("Voyage Embed: {model_id}"),
                    provider: EmbeddingProviderKind::Voyage,
                    model_id,
                    base_url: Some(base_url),
                    api_key: override_key.or_else(|| env("VOYAGE_API_KEY")),
                    request_dimensions: None,
                })
            }
            "jina" => {
                let base_url = override_url
                    .or_else(|| env("JINA_BASE_URL"))
                    .unwrap_or_else(|| DEFAULT_JINA_BASE_URL.to_string());
                Ok(Self {
                    label: format!("Jina Embed: {model_id}"),
                    provider: EmbeddingProviderKind::Jina,
                    model_id,
                    base_url: Some(base_url),
                    api_key: override_key.or_else(|| env("JINA_API_KEY")),
                    request_dimensions: Some(EMBEDDING_DIM as usize),
                })
            }
            "gemini" | "google" => {
                let base_url = override_url
                    .or_else(|| env("GEMINI_BASE_URL"))
                    .unwrap_or_else(|| DEFAULT_GEMINI_OPENAI_BASE_URL.to_string());
                Ok(Self {
                    label: format!("Gemini Embed: {model_id}"),
                    provider: EmbeddingProviderKind::Gemini,
                    model_id,
                    base_url: Some(base_url),
                    api_key: override_key
                        .or_else(|| env("GEMINI_API_KEY"))
                        .or_else(|| env("GOOGLE_API_KEY")),
                    request_dimensions: None,
                })
            }
            _ => Err(MemoryError::Embedding(format!(
                "unknown embedding provider scheme '{scheme}' in '{trimmed}' (expected local, llamacpp, ollama, voyage, jina, gemini, openai, or offline)"
            ))),
        }
    }

    /// Build a runtime [`EmbeddingProvider`] backed by this profile.
    ///
    /// Remote/local HTTP embedders are configured with automatic fallback to
    /// [`DeterministicHashEmbedder`] so AllyH remains functional offline.
    pub fn build_provider(&self) -> Arc<dyn EmbeddingProvider> {
        if self.provider == EmbeddingProviderKind::OfflineHash {
            return Arc::new(DeterministicHashEmbedder);
        }
        let Some(base_url) = self.base_url.as_deref() else {
            return Arc::new(DeterministicHashEmbedder);
        };
        if matches!(
            self.provider,
            EmbeddingProviderKind::OpenAi
                | EmbeddingProviderKind::Voyage
                | EmbeddingProviderKind::Jina
                | EmbeddingProviderKind::Gemini
        ) && self.api_key.as_deref().is_none_or(str::is_empty)
        {
            tracing::warn!(
                profile = %self.label,
                "cloud embedding provider configured without an API key; requests will fall back to deterministic hash embeddings"
            );
        }
        Arc::new(
            OpenAiCompatibleEmbedder::new(base_url, &self.model_id, self.api_key.clone(), true)
                .with_dimensions(self.request_dimensions),
        )
    }
}

fn read_non_empty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn split_model_and_url(rest: &str) -> Result<(String, Option<String>)> {
    let trimmed = rest.trim();
    if trimmed.is_empty() {
        return Err(MemoryError::Embedding(
            "embedding model id cannot be empty".to_string(),
        ));
    }
    match trimmed.split_once('@') {
        Some((model, url)) => {
            let model = model.trim();
            let url = url.trim();
            if model.is_empty() || url.is_empty() {
                return Err(MemoryError::Embedding(format!(
                    "invalid embedding model/url pair '{trimmed}'"
                )));
            }
            Ok((model.to_string(), Some(url.to_string())))
        }
        None => Ok((trimmed.to_string(), None)),
    }
}

fn normalize_ollama_v1_url(host: &str) -> String {
    let trimmed = host.trim().trim_end_matches('/');
    let with_scheme = if trimmed.contains("://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    };
    if with_scheme.ends_with("/v1") {
        with_scheme
    } else {
        format!("{with_scheme}/v1")
    }
}

/// OpenAI-compatible embedding client (works with local `llama.cpp`, Ollama, Voyage AI, Jina AI,
/// Gemini OpenAI-compat, or OpenAI `/v1/embeddings`).
///
/// Automatically falls back to [`DeterministicHashEmbedder`] when `fallback_on_error` is enabled.
#[derive(Debug, Clone)]
pub struct OpenAiCompatibleEmbedder {
    client: reqwest::Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
    dimensions: Option<usize>,
    fallback_on_error: bool,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedItem>,
}

#[derive(Deserialize)]
struct EmbedItem {
    embedding: Vec<f32>,
}

impl OpenAiCompatibleEmbedder {
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
        fallback_on_error: bool,
    ) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_default();
        Self {
            client,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
            api_key,
            dimensions: None,
            fallback_on_error,
        }
    }

    /// Optionally request a specific Matryoshka output dimensionality from the server.
    pub fn with_dimensions(mut self, dimensions: Option<usize>) -> Self {
        self.dimensions = dimensions;
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn requested_dimensions(&self) -> Option<usize> {
        self.dimensions
    }

    async fn fetch_remote(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/embeddings", self.base_url);
        let mut req = self.client.post(url).json(&EmbedRequest {
            model: &self.model,
            input: text,
            dimensions: self.dimensions,
        });
        if let Some(key) = &self.api_key
            && !key.is_empty()
        {
            req = req.bearer_auth(key);
        }
        let resp = req.send().await?;
        if !resp.status().is_success() {
            return Err(MemoryError::Embedding(format!(
                "embedding endpoint returned HTTP {}",
                resp.status()
            )));
        }
        let body: EmbedResponse = resp.json().await?;
        let first = body
            .data
            .into_iter()
            .next()
            .ok_or_else(|| MemoryError::Embedding("empty embedding response".to_string()))?;
        Ok(fit_embedding_dim(&first.embedding))
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAiCompatibleEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        match self.fetch_remote(text).await {
            Ok(vec) => Ok(vec),
            Err(err) if self.fallback_on_error => {
                tracing::debug!(error = %err, "remote embedder unavailable; falling back to deterministic hash embedder");
                Ok(DeterministicHashEmbedder::embed_sync(text))
            }
            Err(err) => Err(err),
        }
    }
}

/// Cosine similarity in `[-1.0, 1.0]` for L2-normalized vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom > 1e-8 { dot / denom } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;

    #[test]
    fn fit_embedding_dim_uses_mrl_prefix_truncation_and_l2_norm() {
        let dim = EMBEDDING_DIM as usize;
        // 768-dim vector where prefix has [3.0, 4.0, 0, ...] and tail has huge values
        // that would corrupt the vector if modulo-folded.
        let mut wide = vec![0.0_f32; dim * 2];
        wide[0] = 3.0;
        wide[1] = 4.0;
        wide[dim] = 999.0;
        wide[dim + 1] = -999.0;

        let fitted = fit_embedding_dim(&wide);
        assert_eq!(fitted.len(), dim);
        assert!((fitted[0] - 0.6).abs() < 1e-5, "got {}", fitted[0]);
        assert!((fitted[1] - 0.8).abs() < 1e-5, "got {}", fitted[1]);
        assert!(fitted[2..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn fit_embedding_dim_pads_short_and_handles_empty() {
        let dim = EMBEDDING_DIM as usize;
        let short = fit_embedding_dim(&[0.0, 3.0, 4.0]);
        assert_eq!(short.len(), dim);
        assert!((short[1] - 0.6).abs() < 1e-5);
        assert!((short[2] - 0.8).abs() < 1e-5);
        assert!(short[3..].iter().all(|&v| v == 0.0));

        let empty = fit_embedding_dim(&[]);
        assert_eq!(empty.len(), dim);
        assert_eq!(empty[0], 1.0);
    }

    #[test]
    fn parse_spec_supports_local_ollama_and_free_apis() {
        let empty_env = |_: &str| None;

        let local =
            EmbeddingProfile::parse_spec_with_env("local:Qwen3-Embedding-0.6B", empty_env).unwrap();
        assert_eq!(local.provider, EmbeddingProviderKind::LocalOpenAi);
        assert_eq!(local.model_id, "Qwen3-Embedding-0.6B");
        assert_eq!(
            local.base_url.as_deref(),
            Some(DEFAULT_LOCAL_EMBEDDING_BASE_URL)
        );
        assert_eq!(local.request_dimensions, None);

        // Ollama tags frequently contain `:` (e.g. `qwen3-embedding:0.6b`)
        let ollama = EmbeddingProfile::parse_spec_with_env(
            "ollama:qwen3-embedding:0.6b@http://127.0.0.1:11434/v1",
            empty_env,
        )
        .unwrap();
        assert_eq!(ollama.provider, EmbeddingProviderKind::LocalOpenAi);
        assert_eq!(ollama.model_id, "qwen3-embedding:0.6b");
        assert_eq!(ollama.base_url.as_deref(), Some("http://127.0.0.1:11434/v1"));

        let ollama_env = |k: &str| (k == "OLLAMA_HOST").then(|| "127.0.0.1:11435".to_string());
        let ollama_default =
            EmbeddingProfile::parse_spec_with_env("ollama:nomic-embed-text", ollama_env).unwrap();
        assert_eq!(
            ollama_default.base_url.as_deref(),
            Some("http://127.0.0.1:11435/v1")
        );

        let voyage =
            EmbeddingProfile::parse_spec_with_env("voyage:voyage-code-3", empty_env).unwrap();
        assert_eq!(voyage.provider, EmbeddingProviderKind::Voyage);
        assert_eq!(voyage.base_url.as_deref(), Some(DEFAULT_VOYAGE_BASE_URL));
        assert_eq!(voyage.request_dimensions, None);

        let jina =
            EmbeddingProfile::parse_spec_with_env("jina:jina-embeddings-v3", empty_env).unwrap();
        assert_eq!(jina.provider, EmbeddingProviderKind::Jina);
        assert_eq!(jina.base_url.as_deref(), Some(DEFAULT_JINA_BASE_URL));
        assert_eq!(jina.request_dimensions, Some(EMBEDDING_DIM as usize));

        let gemini =
            EmbeddingProfile::parse_spec_with_env("gemini:gemini-embedding-001", empty_env)
                .unwrap();
        assert_eq!(gemini.provider, EmbeddingProviderKind::Gemini);
        assert_eq!(
            gemini.base_url.as_deref(),
            Some(DEFAULT_GEMINI_OPENAI_BASE_URL)
        );
        assert_eq!(gemini.request_dimensions, None);

        let openai_v3 =
            EmbeddingProfile::parse_spec_with_env("openai:text-embedding-3-small", empty_env)
                .unwrap();
        assert_eq!(openai_v3.request_dimensions, Some(EMBEDDING_DIM as usize));

        let openai_ada =
            EmbeddingProfile::parse_spec_with_env("openai:text-embedding-ada-002", empty_env)
                .unwrap();
        assert_eq!(openai_ada.request_dimensions, None);

        let offline = EmbeddingProfile::parse_spec_with_env("offline", empty_env).unwrap();
        assert_eq!(offline.provider, EmbeddingProviderKind::OfflineHash);

        assert!(EmbeddingProfile::parse_spec_with_env("unknown:foo", empty_env).is_err());
        assert!(EmbeddingProfile::parse_spec_with_env("local:", empty_env).is_err());
    }

    #[test]
    fn from_env_auto_detection_priority() {
        let map_env = |pairs: &[(&str, &str)]| {
            let map: HashMap<String, String> = pairs
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect();
            move |k: &str| map.get(k).cloned()
        };

        assert_eq!(
            EmbeddingProfile::from_env_with(map_env(&[])).provider,
            EmbeddingProviderKind::OfflineHash
        );

        let explicit = EmbeddingProfile::from_env_with(map_env(&[
            ("ALLYH_EMBEDDING_MODEL", "local:bge-small-en-v1.5"),
            ("VOYAGE_API_KEY", "vk-test"),
        ]));
        assert_eq!(explicit.provider, EmbeddingProviderKind::LocalOpenAi);
        assert_eq!(explicit.model_id, "bge-small-en-v1.5");

        let voyage = EmbeddingProfile::from_env_with(map_env(&[
            ("VOYAGE_API_KEY", "vk-test"),
            ("OPENAI_API_KEY", "sk-test"),
        ]));
        assert_eq!(voyage.provider, EmbeddingProviderKind::Voyage);
        assert_eq!(voyage.model_id, "voyage-code-3");
        assert_eq!(voyage.api_key.as_deref(), Some("vk-test"));

        let jina = EmbeddingProfile::from_env_with(map_env(&[("JINA_API_KEY", "jina-test")]));
        assert_eq!(jina.provider, EmbeddingProviderKind::Jina);
        assert_eq!(jina.request_dimensions, Some(EMBEDDING_DIM as usize));

        let gemini = EmbeddingProfile::from_env_with(map_env(&[("GEMINI_API_KEY", "gem-test")]));
        assert_eq!(gemini.provider, EmbeddingProviderKind::Gemini);
        assert_eq!(gemini.model_id, "gemini-embedding-001");
    }

    #[tokio::test]
    async fn openai_compatible_embedder_sends_optional_dimensions_and_mrl_truncates() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local_addr");

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut buf = vec![0u8; 8192];
            let n = socket.read(&mut buf).await.expect("read");
            let req_str = String::from_utf8_lossy(&buf[..n]).to_string();

            // Return a 768-dim vector where prefix is [3.0, 4.0, 0, ...] and tail is non-zero.
            let mut vec = vec![0.0_f32; (EMBEDDING_DIM as usize) * 2];
            vec[0] = 3.0;
            vec[1] = 4.0;
            vec[EMBEDDING_DIM as usize] = 500.0;
            let body = serde_json::json!({ "data": [{ "embedding": vec }] }).to_string();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(resp.as_bytes()).await.expect("write");
            req_str
        });

        let embedder = OpenAiCompatibleEmbedder::new(
            format!("http://{addr}/v1"),
            "jina-embeddings-v3",
            Some("secret-token".to_string()),
            false,
        )
        .with_dimensions(Some(EMBEDDING_DIM as usize));

        let out = embedder.embed("architectural invariant").await.expect("embed");
        let captured_req = server.await.expect("server join");

        assert!(captured_req.contains("POST /v1/embeddings HTTP/1.1"));
        assert!(captured_req.contains("authorization: Bearer secret-token"));
        assert!(captured_req.contains("\"dimensions\":384"));
        assert_eq!(out.len(), EMBEDDING_DIM as usize);
        assert!((out[0] - 0.6).abs() < 1e-5);
        assert!((out[1] - 0.8).abs() < 1e-5);
    }
}
