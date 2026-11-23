use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use rig_core::DynModel;
use rig_core::completion::{CompletionRequest, CompletionResponse, ToolDefinition};
use rig_core::http_client::{
    BoxedStream, Bytes, Error as RigHttpError, HttpClientExt, LazyBody, MultipartForm, Request,
    Response, Result as RigHttpResult, StreamingResponse,
};
use rig_core::message::{AssistantContent, Message, Origin};
use rig_core::operation::Completion;
use rig_core::providers::anthropic::{self, AnthropicConfig};
use rig_core::providers::openai::OpenAIConfig;
use rig_core::streaming::{Item, StreamEvent};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::error::{CoreError, Result};

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
}

impl ModelProfile {
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
        }
    }

    /// Parse a slash-command model selector such as:
    /// - `anthropic:claude-sonnet-4-6`
    /// - `openai:gpt-4o`
    /// - `local:qwen-35b` or `local:qwen-35b@http://localhost:9931/v1`
    /// - `offline`
    pub fn parse_spec(spec: &str) -> Result<Self> {
        let trimmed = spec.trim();
        if trimmed.eq_ignore_ascii_case("offline") {
            return Ok(Self::offline_mock("Offline: Deterministic"));
        }
        let Some((scheme, rest)) = trimmed.split_once(':') else {
            return Err(CoreError::InvalidModelSpec(format!(
                "{trimmed} (expected anthropic:<model>, openai:<model>, local:<model>[@url], or offline)"
            )));
        };
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
            }),
            "openai" => Ok(Self {
                label: format!("API: {rest}"),
                provider: ModelProviderKind::OpenAi,
                model_id: rest.to_string(),
                base_url: std::env::var("OPENAI_BASE_URL")
                    .ok()
                    .filter(|u| !u.is_empty()),
                api_key: std::env::var("OPENAI_API_KEY")
                    .ok()
                    .filter(|k| !k.is_empty()),
                context_window_tokens: 128_000,
            }),
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
                Some(cfg.connect(http).chat(&self.model_id).erase())
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
}

/// Live `rig-core` completion backend supporting Anthropic, OpenAI, and local OpenAI-compatible servers,
/// with graceful offline synthesis when no credentials/server are configured.
#[derive(Debug, Default, Clone)]
pub struct RigCompletionBackend;

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
        let Some(model) = profile.build_rig_model() else {
            return Ok(synthesize_offline_response(
                profile, preamble, history, token_tx,
            ));
        };

        let Some(last) = history.last() else {
            return Ok(synthesize_offline_response(
                profile, preamble, history, token_tx,
            ));
        };

        let mut req = CompletionRequest::new(last.clone()).preamble(preamble.to_string());
        if history.len() > 1 {
            req = req.messages(history[..history.len() - 1].iter().cloned());
        }
        if !tools.is_empty() {
            req = req.tools(tools.to_vec());
        }
        req = req.temperature(0.2);

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

    CompletionResponse::new(
        vec![AssistantContent::text(reply)],
        rig_core::completion::Usage::new(),
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
    /// `OPENAI_API_KEY`, `ALLYH_LOCAL_BASE_URL`), falling back to offline mode when unset.
    pub fn from_env() -> Self {
        let local_url = std::env::var("ALLYH_LOCAL_BASE_URL")
            .or_else(|_| std::env::var("LLAMACPP_API_BASE_URL"))
            .unwrap_or_else(|_| "http://localhost:9931/v1".to_string());
        let local_model =
            std::env::var("ALLYH_LOCAL_MODEL").unwrap_or_else(|_| "qwen2.5-coder-9b".to_string());

        let primary = ModelProfile::anthropic_from_env(anthropic::CLAUDE_SONNET_4_6)
            .or_else(|| ModelProfile::openai_from_env("gpt-4o"))
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
