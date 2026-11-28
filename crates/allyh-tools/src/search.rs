use std::sync::Arc;

use allyh_env::{
    CodeSearchMatch, CodeSearchQuery, EnvironmentMonitor, SymbolInfo, search_workspace_text,
};
use allyh_memory::{MemoryStore, cosine_similarity};
use rig_core::completion::ToolAnnotations;
use rig_core::tool::{Tool, ToolContext, ToolExecutionError, args_schema};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::ToolError;

/// Arguments for [`SearchCodeTool`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct SearchCodeArgs {
    /// Text literal, regular expression, or natural-language concept to search for.
    pub query: String,
    /// Search mode: `"hybrid"` (default: literal/regex + LanceDB semantic ranking), `"regex"`, `"literal"`, or `"semantic"`.
    #[serde(default)]
    pub mode: Option<String>,
    /// Optional repository-relative subdirectory prefix to scope the search (e.g. `crates/allyh-env`).
    #[serde(default)]
    pub path_prefix: Option<String>,
    /// Optional filename glob filter (e.g. `*.rs`, `*.ts`, `*.py`).
    #[serde(default)]
    pub file_glob: Option<String>,
    /// Enable case-sensitive lexical matching (defaults to `false`).
    #[serde(default)]
    pub case_sensitive: Option<bool>,
    /// Maximum lexical matches to return (defaults to 25, capped at 100).
    #[serde(default)]
    pub max_results: Option<usize>,
}

/// Semantic snapshot hit ranked against LanceDB `code_graph_snapshots`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCodeHit {
    pub file_path: String,
    pub language: String,
    pub score: f32,
    pub matched_symbols: Vec<String>,
}

/// Structured output returned by [`SearchCodeTool`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchCodeOutput {
    pub query: String,
    pub mode: String,
    pub text_matches: Vec<CodeSearchMatch>,
    pub semantic_hits: Vec<SemanticCodeHit>,
    pub files_scanned: usize,
    pub truncated: bool,
}

/// Hybrid `.gitignore`-aware regex/literal + LanceDB semantic code search ACI tool.
#[derive(Clone)]
pub struct SearchCodeTool {
    env: Arc<EnvironmentMonitor>,
    store: MemoryStore,
}

impl SearchCodeTool {
    pub fn new(env: Arc<EnvironmentMonitor>, store: MemoryStore) -> Self {
        Self { env, store }
    }

    pub async fn execute(&self, args: SearchCodeArgs) -> Result<SearchCodeOutput, ToolError> {
        let query = args.query.trim();
        if query.is_empty() {
            return Err(ToolError::InvalidArgs(
                "search_code 'query' must not be empty".to_string(),
            ));
        }

        let mode = args
            .mode
            .as_deref()
            .unwrap_or("hybrid")
            .trim()
            .to_ascii_lowercase();
        if !matches!(mode.as_str(), "hybrid" | "regex" | "literal" | "semantic") {
            return Err(ToolError::InvalidArgs(format!(
                "unknown search mode '{mode}' (expected hybrid, regex, literal, or semantic)"
            )));
        }

        let max_results = args.max_results.unwrap_or(25).clamp(1, 100);
        let profile = self.env.load_repo_profile().await;
        let extra_ignores = profile
            .as_ref()
            .map(|p| p.extra_ignore_globs.as_slice())
            .unwrap_or(&[]);

        let mut text_matches = Vec::new();
        let mut files_scanned = 0usize;
        let mut truncated = false;

        if mode != "semantic" {
            let search_res = search_workspace_text(
                self.env.repo_root(),
                &CodeSearchQuery {
                    pattern: query.to_string(),
                    is_regex: mode == "regex",
                    case_sensitive: args.case_sensitive.unwrap_or(false),
                    path_prefix: args.path_prefix.clone(),
                    file_glob: args.file_glob.clone(),
                    max_matches: max_results,
                },
                extra_ignores,
            )?;
            text_matches = search_res.matches;
            files_scanned = search_res.files_scanned;
            truncated = search_res.truncated;
        }

        let mut semantic_hits = Vec::new();
        if mode == "hybrid" || mode == "semantic" {
            semantic_hits = self
                .rank_semantic_snapshots(query, args.path_prefix.as_deref(), max_results.min(10))
                .await?;
        }

        Ok(SearchCodeOutput {
            query: query.to_string(),
            mode,
            text_matches,
            semantic_hits,
            files_scanned,
            truncated,
        })
    }

    async fn rank_semantic_snapshots(
        &self,
        query: &str,
        path_prefix: Option<&str>,
        top_k: usize,
    ) -> Result<Vec<SemanticCodeHit>, ToolError> {
        let snapshots = self
            .store
            .list_code_snapshots(self.env.repo_id(), self.env.scope_id())
            .await?;
        if snapshots.is_empty() {
            return Ok(Vec::new());
        }

        let query_vec = self.store.embedder().embed(query).await?;
        let query_lower = query.to_ascii_lowercase();
        let query_tokens: Vec<String> = query_lower
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|t| t.len() >= 2)
            .map(str::to_string)
            .collect();
        let norm_prefix = path_prefix
            .map(|p| p.trim().replace('\\', "/").trim_matches('/').to_string())
            .filter(|p| !p.is_empty());

        let mut scored: Vec<SemanticCodeHit> = Vec::new();
        for snap in snapshots {
            if let Some(ref pref) = norm_prefix
                && snap.file_path != *pref
                && !snap.file_path.starts_with(&format!("{pref}/"))
            {
                continue;
            }
            let vec_sim = cosine_similarity(&query_vec, &snap.vector).max(0.0);
            let syms: Vec<SymbolInfo> = serde_json::from_str(&snap.symbols_json).unwrap_or_default();

            let mut matched_symbols = Vec::new();
            let mut lexical_boost = 0.0f32;
            let path_lower = snap.file_path.to_ascii_lowercase();
            for tok in &query_tokens {
                if path_lower.contains(tok) {
                    lexical_boost += 0.35;
                }
            }
            for sym in &syms {
                let name_lower = sym.name.to_ascii_lowercase();
                if name_lower.contains(&query_lower)
                    || query_tokens.iter().any(|t| name_lower.contains(t))
                {
                    lexical_boost += 0.5;
                    if matched_symbols.len() < 6 {
                        matched_symbols.push(sym.name.clone());
                    }
                }
            }
            for dep in &snap.dependencies {
                let dep_lower = dep.to_ascii_lowercase();
                if query_tokens.iter().any(|t| dep_lower.contains(t)) {
                    lexical_boost += 0.15;
                }
            }

            let total_score = vec_sim + lexical_boost;
            if total_score > 0.12 {
                scored.push(SemanticCodeHit {
                    file_path: snap.file_path,
                    language: snap.language,
                    score: total_score,
                    matched_symbols,
                });
            }
        }

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.file_path.cmp(&b.file_path))
        });
        scored.truncate(top_k);
        Ok(scored)
    }
}

impl Tool for SearchCodeTool {
    const NAME: &'static str = "search_code";
    type Args = SearchCodeArgs;
    type Output = SearchCodeOutput;
    type Error = ToolError;

    fn description(&self) -> String {
        "Search the repository using .gitignore-aware regex/literal matching fused with LanceDB semantic ranking over indexed code_graph_snapshots.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<SearchCodeArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_title("Hybrid Code & Semantic Search")
            .with_read_only(true)
            .with_open_world(false)
    }

    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        match error {
            ToolError::InvalidArgs(msg) => ToolExecutionError::invalid_args(msg),
            other => ToolExecutionError::other(other.to_string()),
        }
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.execute(args).await
    }
}
