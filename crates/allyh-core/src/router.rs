use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use rig_core::DynModel;
use rig_core::completion::{
    CacheCost, CacheRates, CacheRetention, CompletionRequest, CompletionResponse, Effort,
    GenerationOptions, OnUnsupported, ToolDefinition, Usage,
};
use rig_core::http_client::{
    BoxedStream, Bytes, Error as RigHttpError, HttpClientExt, LazyBody, MultipartForm, Request,
    Response, Result as RigHttpResult, StreamingResponse,
};
use rig_core::message::{AssistantContent, Message, Origin};
use rig_core::operation::Completion;
use rig_core::providers::anthropic::{self, AnthropicConfig};
use rig_core::providers::gemini::GeminiConfig;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::wire::OPENROUTER;
use rig_core::streaming::{Item, StreamEvent};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::error::{CoreError, Result};

/// Portable reasoning effort level mapped onto OpenAI Responses API, Anthropic thinking,
/// and Gemini reasoning controls via `rig-core` [`GenerationOptions`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    pub fn parse_spec(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "low" => Ok(Self::Low),
            "medium" | "med" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            other => Err(CoreError::InvalidModelSpec(format!(
                "invalid reasoning effort '{other}' (expected low, medium, or high)"
            ))),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    pub fn to_rig_effort(self) -> Effort {
        match self {
            Self::Low => Effort::Low,
            Self::Medium => Effort::Medium,
            Self::High => Effort::High,
        }
    }
}

/// Optional per-request prompt caching and reasoning hints attached to a completion call.
#[derive(Debug, Clone, Default)]
pub struct CompletionCallOptions {
    /// Deterministic cache-routing key (e.g. SHA-256 of the byte-stable preamble prefix).
    pub cache_key: Option<String>,
    /// Enable provider prompt-caching (`cache_control` / prefix caching) when supported.
    pub enable_prompt_cache: bool,
    /// Optional reasoning effort override (falls back to `ModelProfile.reasoning_effort`).
    pub reasoning_effort: Option<ReasoningEffort>,
}

/// Pricing rates in USD per 1M tokens for accurate session cost estimation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModelPricing {
    pub input_per_million: f64,
    pub output_per_million: f64,
    pub cached_read_per_million: f64,
    pub cache_write_per_million: f64,
}

impl ModelPricing {
    pub fn to_cache_rates(self) -> CacheRates {
        CacheRates {
            input: self.input_per_million,
            cached_read: self.cached_read_per_million,
            cache_write: self.cache_write_per_million,
            storage_per_hour: 0.0,
        }
    }

    /// Resolve per-1M-token USD pricing for a model identifier across Anthropic, OpenAI,
    /// Gemini, OpenRouter, enterprise gateways, and local/offline tiers.
    pub fn for_model(provider: &ModelProviderKind, model_id: &str) -> Self {
        if matches!(
            provider,
            ModelProviderKind::LocalOpenAi | ModelProviderKind::OfflineMock
        ) {
            return Self {
                input_per_million: 0.0,
                output_per_million: 0.0,
                cached_read_per_million: 0.0,
                cache_write_per_million: 0.0,
            };
        }

        let lower = model_id.to_ascii_lowercase();
        let name = lower.rsplit('/').next().unwrap_or(&lower);

        if name.contains("claude-opus") {
            Self {
                input_per_million: 15.0,
                output_per_million: 75.0,
                cached_read_per_million: 1.5,
                cache_write_per_million: 18.75,
            }
        } else if name.contains("claude-haiku") {
            Self {
                input_per_million: 0.80,
                output_per_million: 4.0,
                cached_read_per_million: 0.08,
                cache_write_per_million: 1.0,
            }
        } else if name.contains("claude") {
            // Sonnet default
            Self {
                input_per_million: 3.0,
                output_per_million: 15.0,
                cached_read_per_million: 0.30,
                cache_write_per_million: 3.75,
            }
        } else if name.contains("gemini-2.5-flash") || name.contains("gemini-2.0-flash") || name.contains("flash") {
            Self {
                input_per_million: 0.15,
                output_per_million: 0.60,
                cached_read_per_million: 0.0375,
                cache_write_per_million: 0.15,
            }
        } else if name.contains("gemini") {
            // Gemini Pro default
            Self {
                input_per_million: 1.25,
                output_per_million: 10.0,
                cached_read_per_million: 0.3125,
                cache_write_per_million: 1.25,
            }
        } else if name.contains("gpt-5") || name.starts_with("o3") || name.starts_with("o4") {
            Self {
                input_per_million: 2.0,
                output_per_million: 8.0,
                cached_read_per_million: 0.50,
                cache_write_per_million: 2.0,
            }
        } else if name.contains("gpt-4o-mini") || name.contains("mini") {
            Self {
                input_per_million: 0.15,
                output_per_million: 0.60,
                cached_read_per_million: 0.075,
                cache_write_per_million: 0.15,
            }
        } else {
            // Standard frontier default (e.g. gpt-4o / OpenRouter frontier)
            Self {
                input_per_million: 2.50,
                output_per_million: 10.0,
                cached_read_per_million: 1.25,
                cache_write_per_million: 2.50,
            }
        }
    }

    /// Compute total USD cost for one completion [`Usage`] combining `rig-core`'s [`CacheCost`]
    /// input accounting with output-token billing.
    pub fn cost_usd_for_usage(self, usage: &Usage) -> f64 {
        let cache_cost = CacheCost::from_usage(usage);
        let input_usd = cache_cost.usd(&self.to_cache_rates());
        let output_usd = (usage.output_tokens.unwrap_or(0) as f64) * self.output_per_million / 1e6;
        input_usd + output_usd
    }
}

/// Cumulative token, prompt-cache hit, and USD cost telemetry for an AllyH session.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionTelemetry {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_input_tokens: u64,
    pub cache_creation_tokens: u64,
    pub estimated_cost_usd: f64,
}

impl SessionTelemetry {
    /// Record one completion's [`Usage`] against the active [`ModelProfile`].
    pub fn record_usage(&mut self, profile: &ModelProfile, usage: &Usage) {
        let pricing = ModelPricing::for_model(&profile.provider, &profile.model_id);
        self.input_tokens = self
            .input_tokens
            .saturating_add(usage.input_tokens.unwrap_or(0));
        self.output_tokens = self
            .output_tokens
            .saturating_add(usage.output_tokens.unwrap_or(0));
        self.cached_input_tokens = self
            .cached_input_tokens
            .saturating_add(usage.cached_input_tokens.unwrap_or(0));
        self.cache_creation_tokens = self
            .cache_creation_tokens
            .saturating_add(usage.cache_creation_input_tokens.unwrap_or(0));
        self.estimated_cost_usd += pricing.cost_usd_for_usage(usage);
    }

    /// Share of cumulative input tokens served from provider prompt cache in `[0.0, 1.0]`.
    pub fn cache_hit_ratio(&self) -> f64 {
        if self.input_tokens == 0 {
            0.0
        } else {
            (self.cached_input_tokens as f64 / self.input_tokens as f64).clamp(0.0, 1.0)
        }
    }
}

/// Lightweight `rig-core` HTTP transport backed by workspace `reqwest::Client`.
#[derive(Clone, Default)]
pub struct ReqwestRigHttp {
    client: reqwest::Client,
}

impl ReqwestRigHttp {
    fn build_response<B>(
        status_u16: u16,
        headers: &reqwest::header::HeaderMap,
        body: B,
    ) -> RigHttpResult<Response<B>> {
        let mut resp_builder = Response::builder().status(status_u16);
        for (k, v) in headers {
            resp_builder = resp_builder.header(k.as_str(), v.as_bytes());
        }
        Ok(resp_builder.body(body)?)
    }
}

impl HttpClientExt for ReqwestRigHttp {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = RigHttpResult<Response<LazyBody<U>>>> + Send + 'static
    where
        T: Into<Bytes> + Send,
        U: From<Bytes> + Send + 'static,
    {
        let client = self.client.clone();
        let (parts, raw_body) = req.into_parts();
        let body: Bytes = raw_body.into();
        let method_str = parts.method.as_str().to_string();
        let uri_str = parts.uri.to_string();
        let headers_vec: Vec<(String, Vec<u8>)> = parts
            .headers
            .iter()
            .map(|(k, v)| (k.as_str().to_string(), v.as_bytes().to_vec()))
            .collect();

        async move {
            let method = reqwest::Method::from_bytes(method_str.as_bytes())
                .map_err(RigHttpError::instance)?;
            let mut builder = client.request(method, uri_str);
            for (k, v) in headers_vec {
                builder = builder.header(k, v);
            }
            if !body.is_empty() {
                builder = builder.body(body.to_vec());
            }
            let resp = builder.send().await.map_err(RigHttpError::instance)?;
            let status = resp.status().as_u16();
            let headers = resp.headers().clone();
            let lazy: LazyBody<U> = Box::pin(async move {
                let bytes = resp.bytes().await.map_err(RigHttpError::instance)?;
                Ok(U::from(bytes))
            });
            Self::build_response(status, &headers, lazy)
        }
    }

    fn send_multipart<U>(
        &self,
        _req: Request<MultipartForm>,
    ) -> impl Future<Output = RigHttpResult<Response<LazyBody<U>>>> + Send + 'static
    where
        U: From<Bytes> + Send + 'static,
    {
        async {
            Err(RigHttpError::instance(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "multipart upload is not used by AllyH text/tool completion wires",
            )))
        }
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = RigHttpResult<StreamingResponse>> + Send
    where
        T: Into<Bytes> + Send,
    {
        let client = self.client.clone();
        let (parts, raw_body) = req.into_parts();
        let body: Bytes = raw_body.into();
        let method_str = parts.method.as_str().to_string();
        let uri_str = parts.uri.to_string();
        let headers_vec: Vec<(String, Vec<u8>)> = parts
            .headers
            .iter()
            .map(|(k, v)| (k.as_str().to_string(), v.as_bytes().to_vec()))
            .collect();

        async move {
            let method = reqwest::Method::from_bytes(method_str.as_bytes())
                .map_err(RigHttpError::instance)?;
            let mut builder = client.request(method, uri_str);
            for (k, v) in headers_vec {
                builder = builder.header(k, v);
            }
            if !body.is_empty() {
                builder = builder.body(body.to_vec());
            }
            let resp = builder.send().await.map_err(RigHttpError::instance)?;
            let status = resp.status().as_u16();
            let headers = resp.headers().clone();
            let stream: BoxedStream = Box::pin(futures::stream::unfold(resp, |mut r| async move {
                match r.chunk().await {
                    Ok(Some(chunk)) => Some((Ok(chunk), r)),
                    Ok(None) => None,
                    Err(err) => Some((Err(RigHttpError::instance(err)), r)),
                }
            }));
            Self::build_response(status, &headers, stream)
        }
    }
}

/// Provider family supported by AllyH's hybrid router.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelProviderKind {
    Anthropic,
    OpenAi,
    OpenAiResponses,
    Gemini,
    OpenRouter,
    Bedrock,
    VertexAi,
    LocalOpenAi,
    OfflineMock,
}

/// Configuration for either the primary reasoning tier or the fast local worker tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub label: String,
    pub provider: ModelProviderKind,
    pub model_id: String,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub context_window_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
}

impl ModelProfile {
    /// Attach or clear a reasoning effort level (`low`, `medium`, `high`).
    pub fn with_reasoning_effort(mut self, effort: Option<ReasoningEffort>) -> Self {
        self.reasoning_effort = effort;
        self
    }

    /// Default frontier Anthropic Claude profile from environment (`ANTHROPIC_API_KEY`).
    pub fn anthropic_from_env(model_id: impl Into<String>) -> Option<Self> {
        let key = std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())?;
        let model_id = model_id.into();
        let base_url = std::env::var("ANTHROPIC_BASE_URL")
            .ok()
            .filter(|u| !u.is_empty());
        Some(Self {
            label: format!("API: {model_id}"),
            provider: ModelProviderKind::Anthropic,
            model_id,
            base_url,
            api_key: Some(key),
            context_window_tokens: 200_000,
            reasoning_effort: None,
        })
    }

    /// Default frontier OpenAI profile from environment (`OPENAI_API_KEY`).
    pub fn openai_from_env(model_id: impl Into<String>) -> Option<Self> {
        let key = std::env::var("OPENAI_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())?;
        let model_id = model_id.into();
        let base_url = std::env::var("OPENAI_BASE_URL")
            .ok()
            .filter(|u| !u.is_empty());
        Some(Self {
            label: format!("API: {model_id}"),
            provider: ModelProviderKind::OpenAi,
            model_id,
            base_url,
            api_key: Some(key),
            context_window_tokens: 128_000,
            reasoning_effort: None,
        })
    }

    /// Google Gemini profile from environment (`GEMINI_API_KEY`).
    pub fn gemini_from_env(model_id: impl Into<String>) -> Option<Self> {
        let key = std::env::var("GEMINI_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())?;
        let model_id = model_id.into();
        let base_url = std::env::var("GEMINI_BASE_URL")
            .ok()
            .filter(|u| !u.is_empty());
        Some(Self {
            label: format!("Gemini: {model_id}"),
            provider: ModelProviderKind::Gemini,
            model_id,
            base_url,
            api_key: Some(key),
            context_window_tokens: 1_000_000,
            reasoning_effort: None,
        })
    }

    /// OpenRouter multi-provider gateway profile from environment (`OPENROUTER_API_KEY`).
    pub fn openrouter_from_env(model_id: impl Into<String>) -> Option<Self> {
        let key = std::env::var("OPENROUTER_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())?;
        let model_id = model_id.into();
        let base_url = std::env::var("OPENROUTER_BASE_URL")
            .ok()
            .filter(|u| !u.is_empty());
        Some(Self {
            label: format!("OpenRouter: {model_id}"),
            provider: ModelProviderKind::OpenRouter,
            model_id,
            base_url,
            api_key: Some(key),
            context_window_tokens: 200_000,
            reasoning_effort: None,
        })
    }

    /// Local OpenAI-compatible server (`llama.cpp` / Ollama default `http://localhost:9931/v1`).
    pub fn local_openai(
        base_url: impl Into<String>,
        model_id: impl Into<String>,
        context_window_tokens: u64,
    ) -> Self {
        let model_id = model_id.into();
        let base_url = base_url.into();
        Self {
            label: format!("Local: {model_id}"),
            provider: ModelProviderKind::LocalOpenAi,
            model_id,
            base_url: Some(base_url),
            api_key: std::env::var("LLAMACPP_API_KEY")
                .ok()
                .filter(|k| !k.is_empty()),
            context_window_tokens,
            reasoning_effort: None,
        }
    }

    /// Deterministic offline fallback profile (zero network dependency).
    pub fn offline_mock(label: impl Into<String>) -> Self {
        let label = label.into();
        Self {
            label: label.clone(),
            provider: ModelProviderKind::OfflineMock,
            model_id: label,
            base_url: None,
            api_key: None,
            context_window_tokens: 128_000,
            reasoning_effort: None,
        }
    }

    /// Parse a CLI or slash-command model selector such as:
    /// - `anthropic:claude-sonnet-4-6`
    /// - `openai:gpt-4o` / `openai-responses:gpt-5.2` / `responses:o3`
    /// - `gemini:gemini-2.5-pro`
    /// - `openrouter:anthropic/claude-sonnet-4`
    /// - `bedrock:<model-id>` (requires `--features bedrock` / `enterprise-providers`)
    /// - `vertex:<model-id>` (requires `--features vertex` / `enterprise-providers`)
    /// - `local:qwen-35b` or `local:qwen-35b@http://localhost:9931/v1`
    /// - `offline`
    pub fn parse_spec(spec: &str) -> Result<Self> {
        let trimmed = spec.trim();
        if trimmed.eq_ignore_ascii_case("offline") {
            return Ok(Self::offline_mock("Offline: Deterministic"));
        }
        let Some((scheme, rest)) = trimmed.split_once(':') else {
            return Err(CoreError::InvalidModelSpec(format!(
                "{trimmed} (expected anthropic:<model>, openai:<model>, openai-responses:<model>, gemini:<model>, openrouter:<org/model>, local:<model>[@url], or offline)"
            )));
        };
        let rest = rest.trim();
        if rest.is_empty() {
            return Err(CoreError::InvalidModelSpec(format!(
                "{trimmed} (model identifier after ':' cannot be empty)"
            )));
        }

        match scheme.to_ascii_lowercase().as_str() {
            "anthropic" | "claude" => Ok(Self {
                label: format!("API: {rest}"),
                provider: ModelProviderKind::Anthropic,
                model_id: rest.to_string(),
                base_url: std::env::var("ANTHROPIC_BASE_URL")
                    .ok()
                    .filter(|u| !u.is_empty()),
                api_key: std::env::var("ANTHROPIC_API_KEY")
                    .ok()
                    .filter(|k| !k.is_empty()),
                context_window_tokens: 200_000,
                reasoning_effort: None,
            }),
            "openai" => Ok(Self {
                label: format!("API: {rest}"),
                provider: if prefers_openai_responses(rest) {
                    ModelProviderKind::OpenAiResponses
                } else {
                    ModelProviderKind::OpenAi
                },
                model_id: rest.to_string(),
                base_url: std::env::var("OPENAI_BASE_URL")
                    .ok()
                    .filter(|u| !u.is_empty()),
                api_key: std::env::var("OPENAI_API_KEY")
                    .ok()
                    .filter(|k| !k.is_empty()),
                context_window_tokens: 200_000,
                reasoning_effort: None,
            }),
            "openai-responses" | "responses" => Ok(Self {
                label: format!("Responses: {rest}"),
                provider: ModelProviderKind::OpenAiResponses,
                model_id: rest.to_string(),
                base_url: std::env::var("OPENAI_BASE_URL")
                    .ok()
                    .filter(|u| !u.is_empty()),
                api_key: std::env::var("OPENAI_API_KEY")
                    .ok()
                    .filter(|k| !k.is_empty()),
                context_window_tokens: 200_000,
                reasoning_effort: None,
            }),
            "gemini" | "google" => Ok(Self {
                label: format!("Gemini: {rest}"),
                provider: ModelProviderKind::Gemini,
                model_id: rest.to_string(),
                base_url: std::env::var("GEMINI_BASE_URL")
                    .ok()
                    .filter(|u| !u.is_empty()),
                api_key: std::env::var("GEMINI_API_KEY")
                    .ok()
                    .filter(|k| !k.is_empty()),
                context_window_tokens: 1_000_000,
                reasoning_effort: None,
            }),
            "openrouter" => Ok(Self {
                label: format!("OpenRouter: {rest}"),
                provider: ModelProviderKind::OpenRouter,
                model_id: rest.to_string(),
                base_url: std::env::var("OPENROUTER_BASE_URL")
                    .ok()
                    .filter(|u| !u.is_empty()),
                api_key: std::env::var("OPENROUTER_API_KEY")
                    .ok()
                    .filter(|k| !k.is_empty()),
                context_window_tokens: 200_000,
                reasoning_effort: None,
            }),
            "bedrock" => {
                #[cfg(not(feature = "bedrock"))]
                {
                    Err(CoreError::InvalidModelSpec(format!(
                        "{trimmed} (AWS Bedrock requires compiling allyh-core with `--features bedrock` or `--features enterprise-providers`)"
                    )))
                }
                #[cfg(feature = "bedrock")]
                {
                    let region =
                        std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());
                    let base_url = std::env::var("AWS_BEDROCK_BASE_URL")
                        .ok()
                        .filter(|u| !u.is_empty())
                        .or_else(|| {
                            Some(format!("https://bedrock-runtime.{region}.amazonaws.com/openai/v1"))
                        });
                    let api_key = std::env::var("AWS_BEARER_TOKEN_BEDROCK")
                        .or_else(|_| std::env::var("AWS_BEDROCK_API_KEY"))
                        .ok()
                        .filter(|k| !k.is_empty());
                    Ok(Self {
                        label: format!("Bedrock: {rest}"),
                        provider: ModelProviderKind::Bedrock,
                        model_id: rest.to_string(),
                        base_url,
                        api_key,
                        context_window_tokens: 200_000,
                        reasoning_effort: None,
                    })
                }
            }
            "vertex" | "vertexai" => {
                #[cfg(not(feature = "vertex"))]
                {
                    Err(CoreError::InvalidModelSpec(format!(
                        "{trimmed} (Google Vertex AI requires compiling allyh-core with `--features vertex` or `--features enterprise-providers`)"
                    )))
                }
                #[cfg(feature = "vertex")]
                {
                    let location = std::env::var("GOOGLE_CLOUD_LOCATION")
                        .unwrap_or_else(|_| "us-central1".to_string());
                    let base_url = std::env::var("VERTEX_AI_BASE_URL")
                        .ok()
                        .filter(|u| !u.is_empty())
                        .or_else(|| {
                            Some(format!("https://{location}-aiplatform.googleapis.com"))
                        });
                    let api_key = std::env::var("VERTEX_AI_ACCESS_TOKEN")
                        .or_else(|_| std::env::var("GEMINI_API_KEY"))
                        .ok()
                        .filter(|k| !k.is_empty());
                    Ok(Self {
                        label: format!("Vertex: {rest}"),
                        provider: ModelProviderKind::VertexAi,
                        model_id: rest.to_string(),
                        base_url,
                        api_key,
                        context_window_tokens: 1_000_000,
                        reasoning_effort: None,
                    })
                }
            }
            "local" | "llamacpp" | "ollama" => {
                let (model_id, base_url) = match rest.split_once('@') {
                    Some((m, u)) => (m.to_string(), u.to_string()),
                    None => (
                        rest.to_string(),
                        std::env::var("ALLYH_LOCAL_BASE_URL")
                            .or_else(|_| std::env::var("LLAMACPP_API_BASE_URL"))
                            .unwrap_or_else(|_| "http://localhost:9931/v1".to_string()),
                    ),
                };
                Ok(Self::local_openai(base_url, model_id, 64_000))
            }
            _ => Err(CoreError::InvalidModelSpec(trimmed.to_string())),
        }
    }

    /// Build a `rig-core` erased [`DynModel<Completion>`] for live API/local requests.
    pub fn build_rig_model(&self) -> Option<DynModel<Completion>> {
        let http = ReqwestRigHttp::default();
        match self.provider {
            ModelProviderKind::Anthropic => {
                let key = self.api_key.as_deref().filter(|k| !k.is_empty())?;
                let mut cfg = AnthropicConfig::new(key);
                if let Some(url) = &self.base_url {
                    cfg = cfg.with_base_url(url);
                }
                Some(cfg.connect(http).completion(&self.model_id).erase())
            }
            ModelProviderKind::OpenAi => {
                let key = self.api_key.as_deref().filter(|k| !k.is_empty())?;
                let mut cfg = OpenAIConfig::new(key);
                if let Some(url) = &self.base_url {
                    cfg = cfg.with_base_url(url);
                }
                let client = cfg.connect(http);
                if self.reasoning_effort.is_some() {
                    Some(client.responses(&self.model_id).erase())
                } else {
                    Some(client.chat(&self.model_id).erase())
                }
            }
            ModelProviderKind::OpenAiResponses => {
                let key = self.api_key.as_deref().filter(|k| !k.is_empty())?;
                let mut cfg = OpenAIConfig::new(key);
                if let Some(url) = &self.base_url {
                    cfg = cfg.with_base_url(url);
                }
                Some(cfg.connect(http).responses(&self.model_id).erase())
            }
            ModelProviderKind::Gemini => {
                let key = self.api_key.as_deref().filter(|k| !k.is_empty())?;
                let mut cfg = GeminiConfig::new(key);
                if let Some(url) = &self.base_url {
                    cfg = cfg.with_base_url(url);
                }
                Some(cfg.connect(http).completion(&self.model_id).erase())
            }
            ModelProviderKind::OpenRouter => {
                let key = self.api_key.as_deref().filter(|k| !k.is_empty())?;
                let mut cfg = OpenAIConfig::with_key(&OPENROUTER, key);
                if let Some(url) = &self.base_url {
                    cfg = cfg.with_base_url(url);
                }
                Some(cfg.connect(http).chat(&self.model_id).erase())
            }
            ModelProviderKind::Bedrock => {
                #[cfg(feature = "bedrock")]
                {
                    let key = self.api_key.as_deref().filter(|k| !k.is_empty())?;
                    let mut cfg = OpenAIConfig::new(key);
                    if let Some(url) = &self.base_url {
                        cfg = cfg.with_base_url(url);
                    }
                    Some(cfg.connect(http).chat(&self.model_id).erase())
                }
                #[cfg(not(feature = "bedrock"))]
                {
                    None
                }
            }
            ModelProviderKind::VertexAi => {
                #[cfg(feature = "vertex")]
                {
                    let key = self.api_key.as_deref().filter(|k| !k.is_empty())?;
                    let mut cfg = GeminiConfig::new(key);
                    if let Some(url) = &self.base_url {
                        cfg = cfg.with_base_url(url);
                    }
                    Some(cfg.connect(http).completion(&self.model_id).erase())
                }
                #[cfg(not(feature = "vertex"))]
                {
                    None
                }
            }
            ModelProviderKind::LocalOpenAi => {
                let url = self
                    .base_url
                    .as_deref()
                    .unwrap_or("http://localhost:9931/v1");
                let key = self.api_key.as_deref().unwrap_or("local-no-key");
                let cfg = OpenAIConfig::new(key).with_base_url(url);
                Some(cfg.connect(http).chat(&self.model_id).erase())
            }
            ModelProviderKind::OfflineMock => None,
        }
    }
}

fn prefers_openai_responses(model_id: &str) -> bool {
    let lower = model_id.to_ascii_lowercase();
    let name = lower.rsplit('/').next().unwrap_or(&lower);
    name.starts_with("gpt-5")
        || name.starts_with("o1")
        || name.starts_with("o3")
        || name.starts_with("o4")
        || name.contains("codex")
}

/// Abstract completion backend enabling both live `rig-core` providers and deterministic tests.
#[async_trait]
pub trait CompletionBackend: Send + Sync {
    async fn complete(
        &self,
        profile: &ModelProfile,
        preamble: &str,
        history: &[Message],
        tools: &[ToolDefinition],
        token_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> Result<CompletionResponse>;

    /// Completion entry point carrying cache-routing and reasoning-effort options.
    /// Defaults to [`Self::complete`] for custom test backends.
    async fn complete_with_options(
        &self,
        profile: &ModelProfile,
        preamble: &str,
        history: &[Message],
        tools: &[ToolDefinition],
        _options: &CompletionCallOptions,
        token_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> Result<CompletionResponse> {
        self.complete(profile, preamble, history, tools, token_tx)
            .await
    }
}

/// Live `rig-core` completion backend supporting Anthropic, OpenAI (Chat & Responses API),
/// Google Gemini, OpenRouter, optional enterprise providers (Bedrock/Vertex), and local servers,
/// with graceful offline synthesis when no credentials/server are configured.
#[derive(Debug, Default, Clone)]
pub struct RigCompletionBackend;

impl RigCompletionBackend {
    /// Build a `rig-core` [`CompletionRequest`] wired with portable prompt-caching breakpoints
    /// (`CacheRetention::Short`, `cache_key`) and reasoning effort controls (`Effort`).
    pub fn build_request(
        profile: &ModelProfile,
        preamble: &str,
        history: &[Message],
        tools: &[ToolDefinition],
        options: &CompletionCallOptions,
    ) -> Option<CompletionRequest> {
        let last = history.last()?;
        let mut req = CompletionRequest::new(last.clone()).preamble(preamble.to_string());
        if history.len() > 1 {
            req = req.messages(history[..history.len() - 1].iter().cloned());
        }
        if !tools.is_empty() {
            req = req.tools(tools.to_vec());
        }

        let effort = options.reasoning_effort.or(profile.reasoning_effort);
        let mut gen_opts = GenerationOptions::new().on_unsupported(OnUnsupported::Ignore);
        if options.enable_prompt_cache {
            gen_opts = gen_opts.cache(CacheRetention::Short);
            if let Some(key) = &options.cache_key {
                gen_opts = gen_opts.cache_key(key.clone());
            }
        }
        if let Some(e) = effort {
            gen_opts = gen_opts.reasoning(e.to_rig_effort());
        } else {
            // Reasoning models on OpenAI Responses reject explicit temperature; only send
            // low temperature when reasoning is unset and model isn't an o-series/GPT-5 model.
            if !prefers_openai_responses(&profile.model_id) {
                req = req.temperature(0.2);
            }
        }
        if !gen_opts.is_default() {
            req = req.options(gen_opts);
        }
        Some(req)
    }
}

#[async_trait]
impl CompletionBackend for RigCompletionBackend {
    async fn complete(
        &self,
        profile: &ModelProfile,
        preamble: &str,
        history: &[Message],
        tools: &[ToolDefinition],
        token_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> Result<CompletionResponse> {
        self.complete_with_options(
            profile,
            preamble,
            history,
            tools,
            &CompletionCallOptions::default(),
            token_tx,
        )
        .await
    }

    async fn complete_with_options(
        &self,
        profile: &ModelProfile,
        preamble: &str,
        history: &[Message],
        tools: &[ToolDefinition],
        options: &CompletionCallOptions,
        token_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> Result<CompletionResponse> {
        let Some(model) = profile.build_rig_model() else {
            return Ok(synthesize_offline_response(
                profile, preamble, history, options, token_tx,
            ));
        };

        let Some(req) = Self::build_request(profile, preamble, history, tools, options) else {
            return Ok(synthesize_offline_response(
                profile, preamble, history, options, token_tx,
            ));
        };

        if let Some(tx) = token_tx {
            match model.stream(req) {
                Ok(mut stream) => {
                    while let Some(item) = stream.next().await {
                        if let Ok(Item::Event(StreamEvent::Text { text, .. })) = item {
                            let _ = tx.send(text);
                        }
                    }
                    match stream.finish().await {
                        Ok(resp) => Ok(resp),
                        Err(err) => Err(CoreError::Provider(err)),
                    }
                }
                Err(err) => Err(CoreError::Provider(err)),
            }
        } else {
            Ok(model.call(req).await?)
        }
    }
}

fn synthesize_offline_response(
    profile: &ModelProfile,
    preamble: &str,
    history: &[Message],
    options: &CompletionCallOptions,
    token_tx: Option<mpsc::UnboundedSender<String>>,
) -> CompletionResponse {
    let last_user = history
        .iter()
        .rev()
        .find_map(|m| match m {
            Message::User { content } => content.iter().find_map(|c| match c {
                rig_core::message::UserContent::Text(t) => Some(t.text.as_str()),
                _ => None,
            }),
            _ => None,
        })
        .unwrap_or("");

    let active_rules: Vec<&str> = preamble
        .lines()
        .filter(|l| l.trim_start().starts_with("- ["))
        .take(5)
        .collect();
    let oob_lines: Vec<&str> = preamble
        .lines()
        .filter(|l| l.contains("Out-of-band") || l.trim_start().starts_with("  - `"))
        .take(4)
        .collect();

    let mut reply = String::new();
    reply.push_str(&format!(
        "[{}] Grounded response for: {}\n",
        profile.label,
        last_user.trim()
    ));
    if !active_rules.is_empty() {
        reply.push_str("\nEnforcing active LanceDB architectural invariants:\n");
        for rule in active_rules {
            reply.push_str(rule);
            reply.push('\n');
        }
    }
    if !oob_lines.is_empty() {
        reply.push_str("\nAcknowledged live workspace changes:\n");
        for line in oob_lines {
            reply.push_str(line);
            reply.push('\n');
        }
    }
    if reply.ends_with('\n') {
        reply.pop();
    }

    if let Some(tx) = token_tx {
        for chunk in reply.split_inclusive(' ') {
            let _ = tx.send(chunk.to_string());
        }
    }

    let est_in = crate::preamble::estimate_tokens(preamble)
        + crate::preamble::estimate_tokens(last_user);
    let est_out = crate::preamble::estimate_tokens(&reply);
    let mut usage = Usage::new();
    usage.input_tokens = Some(est_in);
    usage.output_tokens = Some(est_out);
    if options.enable_prompt_cache && options.cache_key.is_some() {
        // Approximate half of the preamble prefix being cached on repeated calls offline.
        usage.cached_input_tokens = Some(est_in / 2);
    }

    CompletionResponse::new(
        vec![AssistantContent::text(reply)],
        usage,
        Origin::new("offline", "allyh", profile.model_id.clone()),
        serde_json::json!({ "offline": true }),
    )
}

/// Hierarchical router choosing between the frontier/large primary model and the fast local
/// background worker model (*MemoHarness* / *TO-Router* principle).
#[derive(Clone)]
pub struct HybridRouter {
    primary: ModelProfile,
    local_worker: ModelProfile,
    backend: Arc<dyn CompletionBackend>,
}

impl HybridRouter {
    pub fn new(
        primary: ModelProfile,
        local_worker: ModelProfile,
        backend: Arc<dyn CompletionBackend>,
    ) -> Self {
        Self {
            primary,
            local_worker,
            backend,
        }
    }

    /// Auto-detect best available models from environment variables (`ANTHROPIC_API_KEY`,
    /// `OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `ALLYH_LOCAL_BASE_URL`),
    /// falling back to offline mode when unset.
    pub fn from_env() -> Self {
        let local_url = std::env::var("ALLYH_LOCAL_BASE_URL")
            .or_else(|_| std::env::var("LLAMACPP_API_BASE_URL"))
            .unwrap_or_else(|_| "http://localhost:9931/v1".to_string());
        let local_model =
            std::env::var("ALLYH_LOCAL_MODEL").unwrap_or_else(|_| "qwen2.5-coder-9b".to_string());

        let primary = ModelProfile::anthropic_from_env(anthropic::CLAUDE_SONNET_4_6)
            .or_else(|| ModelProfile::openai_from_env("gpt-4o"))
            .or_else(|| ModelProfile::gemini_from_env("gemini-2.5-pro"))
            .or_else(|| ModelProfile::openrouter_from_env("anthropic/claude-sonnet-4"))
            .unwrap_or_else(|| ModelProfile::offline_mock("Offline: Local/Mock"));

        let local_worker = if std::env::var("ALLYH_LOCAL_BASE_URL").is_ok()
            || std::env::var("LLAMACPP_API_BASE_URL").is_ok()
        {
            ModelProfile::local_openai(local_url, local_model, 32_000)
        } else {
            ModelProfile::offline_mock("Local 9B (Heuristic Fallback)")
        };

        Self::new(primary, local_worker, Arc::new(RigCompletionBackend))
    }

    pub fn primary(&self) -> &ModelProfile {
        &self.primary
    }

    pub fn local_worker(&self) -> &ModelProfile {
        &self.local_worker
    }

    pub fn set_primary(&mut self, profile: ModelProfile) {
        self.primary = profile;
    }

    pub fn set_local_worker(&mut self, profile: ModelProfile) {
        self.local_worker = profile;
    }

    /// Set or clear reasoning effort (`low`, `medium`, `high`) on the primary model profile.
    pub fn set_reasoning_effort(&mut self, effort: Option<ReasoningEffort>) {
        self.primary.reasoning_effort = effort;
    }

    /// Route heavy reasoning / coding turns to the primary frontier or large local model.
    pub async fn run_primary(
        &self,
        preamble: &str,
        history: &[Message],
        tools: &[ToolDefinition],
        token_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> Result<CompletionResponse> {
        self.backend
            .complete(&self.primary, preamble, history, tools, token_tx)
            .await
    }

    /// Route a primary turn with explicit prompt-caching and reasoning-effort options.
    pub async fn run_primary_with_options(
        &self,
        preamble: &str,
        history: &[Message],
        tools: &[ToolDefinition],
        options: &CompletionCallOptions,
        token_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> Result<CompletionResponse> {
        self.backend
            .complete_with_options(&self.primary, preamble, history, tools, options, token_tx)
            .await
    }

    /// Route auxiliary tasks (context compaction, diff summarization, conflict resolution)
    /// to the fast local 9B–12B worker model.
    pub async fn run_local_worker(&self, system_prompt: &str, user_prompt: &str) -> Result<String> {
        if self.local_worker.provider == ModelProviderKind::OfflineMock {
            return Err(CoreError::InvalidModelSpec(
                "local worker running in offline heuristic mode".to_string(),
            ));
        }
        let history = vec![Message::user(user_prompt)];
        let resp = self
            .backend
            .complete(&self.local_worker, system_prompt, &history, &[], None)
            .await?;
        Ok(resp.text())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gemini_openrouter_responses_and_reasoning_effort() {
        let gemini = ModelProfile::parse_spec("gemini:gemini-2.5-pro").expect("parse gemini");
        assert_eq!(gemini.provider, ModelProviderKind::Gemini);
        assert_eq!(gemini.model_id, "gemini-2.5-pro");
        assert_eq!(gemini.context_window_tokens, 1_000_000);

        let openrouter =
            ModelProfile::parse_spec("openrouter:anthropic/claude-sonnet-4").expect("openrouter");
        assert_eq!(openrouter.provider, ModelProviderKind::OpenRouter);
        assert_eq!(openrouter.model_id, "anthropic/claude-sonnet-4");

        let resp_explicit =
            ModelProfile::parse_spec("openai-responses:gpt-4o").expect("openai-responses");
        assert_eq!(resp_explicit.provider, ModelProviderKind::OpenAiResponses);

        let gpt5_auto = ModelProfile::parse_spec("openai:gpt-5.2").expect("openai gpt-5");
        assert_eq!(gpt5_auto.provider, ModelProviderKind::OpenAiResponses);

        assert_eq!(
            ReasoningEffort::parse_spec("high").unwrap(),
            ReasoningEffort::High
        );
        assert_eq!(
            ReasoningEffort::parse_spec("medium").unwrap(),
            ReasoningEffort::Medium
        );
        assert_eq!(
            ReasoningEffort::parse_spec("low").unwrap(),
            ReasoningEffort::Low
        );
        assert!(ReasoningEffort::parse_spec("ultra").is_err());

        // Enterprise providers require opt-in Cargo features when built with defaults.
        #[cfg(not(feature = "bedrock"))]
        assert!(ModelProfile::parse_spec("bedrock:anthropic.claude-3-7-sonnet").is_err());
        #[cfg(not(feature = "vertex"))]
        assert!(ModelProfile::parse_spec("vertex:gemini-2.5-pro").is_err());
    }

    #[test]
    fn builds_cache_and_reasoning_generation_options_on_request() {
        let profile = ModelProfile::parse_spec("anthropic:claude-sonnet-4-6")
            .unwrap()
            .with_reasoning_effort(Some(ReasoningEffort::Medium));
        let opts = CompletionCallOptions {
            cache_key: Some("deadbeef1234".into()),
            enable_prompt_cache: true,
            reasoning_effort: None,
        };
        let req = RigCompletionBackend::build_request(
            &profile,
            "Stable prefix\n\nVolatile suffix",
            &[Message::user("hello")],
            &[],
            &opts,
        )
        .expect("request");
        assert_eq!(req.options.cache, Some(CacheRetention::Short));
        assert_eq!(req.options.cache_key.as_deref(), Some("deadbeef1234"));
        assert_eq!(
            req.options.reasoning,
            Some(rig_core::completion::Reasoning::Effort(Effort::Medium))
        );
        assert_eq!(req.options.on_unsupported, Some(OnUnsupported::Ignore));
    }

    #[test]
    fn computes_accurate_cache_and_usd_cost_telemetry() {
        let sonnet = ModelProfile::parse_spec("anthropic:claude-sonnet-4-6").unwrap();
        let mut telemetry = SessionTelemetry::default();
        let mut usage = Usage::new();
        usage.input_tokens = Some(100_000);
        usage.cached_input_tokens = Some(80_000);
        usage.cache_creation_input_tokens = Some(10_000);
        usage.output_tokens = Some(4_000);

        telemetry.record_usage(&sonnet, &usage);
        // Uncached input: 10,000 * $3/1M = $0.03
        // Cached read:    80,000 * $0.30/1M = $0.024
        // Cache write:    10,000 * $3.75/1M = $0.0375
        // Output:          4,000 * $15/1M = $0.06
        // Total = $0.1515
        assert_eq!(telemetry.input_tokens, 100_000);
        assert_eq!(telemetry.cached_input_tokens, 80_000);
        assert_eq!(telemetry.cache_creation_tokens, 10_000);
        assert_eq!(telemetry.output_tokens, 4_000);
        assert!((telemetry.cache_hit_ratio() - 0.80).abs() < 1e-6);
        assert!((telemetry.estimated_cost_usd - 0.1515).abs() < 1e-6);

        // Local/offline models always incur $0.00 cost.
        let local = ModelProfile::local_openai("http://localhost:9931/v1", "qwen-35b", 64_000);
        let mut local_tel = SessionTelemetry::default();
        local_tel.record_usage(&local, &usage);
        assert_eq!(local_tel.estimated_cost_usd, 0.0);
    }
}
