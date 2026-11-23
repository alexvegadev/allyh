use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{MemoryError, Result};
use crate::schema::EMBEDDING_DIM;

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

/// Resize any upstream embedding vector to `EMBEDDING_DIM` via folding/padding + L2 normalization.
pub fn fit_embedding_dim(raw: &[f32]) -> Vec<f32> {
    let dim = EMBEDDING_DIM as usize;
    let mut out = vec![0.0_f32; dim];
    if raw.is_empty() {
        out[0] = 1.0;
        return out;
    }
    for (i, &val) in raw.iter().enumerate() {
        out[i % dim] += val;
    }
    l2_normalize(&mut out);
    out
}

#[async_trait]
impl EmbeddingProvider for DeterministicHashEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(Self::embed_sync(text))
    }
}

/// OpenAI-compatible embedding client (works with local `llama.cpp`, Ollama, or OpenAI `/v1/embeddings`).
///
/// Automatically falls back to [`DeterministicHashEmbedder`] when `fallback_on_error` is enabled.
#[derive(Debug, Clone)]
pub struct OpenAiCompatibleEmbedder {
    client: reqwest::Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
    fallback_on_error: bool,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a str,
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
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
            api_key,
            fallback_on_error,
        }
    }

    async fn fetch_remote(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/embeddings", self.base_url);
        let mut req = self.client.post(url).json(&EmbedRequest {
            model: &self.model,
            input: text,
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
