use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use allyh_memory::{RepoProfile, VerificationProof};
use async_trait::async_trait;
use chrono::Utc;
use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::path::strip_verbatim_prefix;

pub mod custom;
pub mod go;
pub mod python;
pub mod rust;
pub mod typescript;

/// Language-agnostic verification mode supported by [`ToolchainVerifier`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationKind {
    Check,
    Lint,
    Clippy,
    Test,
}

impl VerificationKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Check => "check",
            Self::Lint => "lint",
            Self::Clippy => "clippy",
            Self::Test => "test",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "check" | "typecheck" | "type_check" | "" => Some(Self::Check),
            "lint" => Some(Self::Lint),
            "clippy" => Some(Self::Clippy),
            "test" | "tests" => Some(Self::Test),
            _ => None,
        }
    }

    pub fn command_label(&self) -> &'static str {
        match self {
            Self::Check => "cargo check --message-format=json",
            Self::Clippy | Self::Lint => "cargo clippy --message-format=json",
            Self::Test => "cargo test --message-format=json",
        }
    }
}

/// Structured compiler, linter, or test diagnostic parsed from any supported toolchain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompilerDiagnostic {
    pub level: String,
    pub code: Option<String>,
    pub message: String,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub column: Option<u32>,
    pub suggestion: Option<String>,
}

/// Alias for polyglot callers preferring language-agnostic naming.
pub type DiagnosticItem = CompilerDiagnostic;

/// Prescriptive verification report attached to ACI tool outputs across all toolchains.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    #[serde(default = "default_toolchain_label")]
    pub toolchain: String,
    #[serde(default = "default_working_dir")]
    pub working_dir: String,
    pub kind: VerificationKind,
    pub command: String,
    pub success: bool,
    pub error_count: usize,
    pub warning_count: usize,
    pub diagnostics: Vec<CompilerDiagnostic>,
    pub raw_summary: String,
}

fn default_toolchain_label() -> String {
    "cargo".to_string()
}

fn default_working_dir() -> String {
    ".".to_string()
}

/// Backward-compatible alias preserving [`CargoVerificationReport`] across existing callers.
pub type CargoVerificationReport = VerificationReport;

impl VerificationReport {
    /// Convert a passing verification report into a [`VerificationProof`] required to mark
    /// a LanceDB task `verified`.
    pub fn to_verification_proof(&self) -> Option<VerificationProof> {
        if !self.success {
            return None;
        }
        Some(VerificationProof {
            command: self.command.clone(),
            passed_at: Utc::now(),
            warnings: self.warning_count,
            summary: self.compact_status_line(),
        })
    }

    /// Compact one-line summary for TUI Tool Cards (e.g., `cargo check: PASS (0 warnings)`).
    pub fn compact_status_line(&self) -> String {
        let tool_name = match (self.toolchain.as_str(), self.kind) {
            ("cargo" | "rust", VerificationKind::Check) => "cargo check".to_string(),
            ("cargo" | "rust", VerificationKind::Clippy | VerificationKind::Lint) => {
                "cargo clippy".to_string()
            }
            ("cargo" | "rust", VerificationKind::Test) => "cargo test".to_string(),
            (tc, kind) => format!("{tc} {}", kind.as_str()),
        };
        let scope_suffix = if self.working_dir.is_empty() || self.working_dir == "." {
            String::new()
        } else {
            format!(" [{}]", self.working_dir)
        };
        if self.success {
            format!("{tool_name}{scope_suffix}: PASS ({} warnings)", self.warning_count)
        } else {
            format!(
                "{tool_name}{scope_suffix}: FAIL ({} errors, {} warnings)",
                self.error_count, self.warning_count
            )
        }
    }

    /// Prescriptive diagnostics formatted for model repair loops (*CoDAIM 2026*).
    pub fn render_prescriptive(&self) -> String {
        let mut lines = vec![self.compact_status_line()];
        for diag in self.diagnostics.iter().take(12) {
            let loc = match (&diag.file, diag.line, diag.column) {
                (Some(f), Some(l), Some(c)) => format!("{f}:{l}:{c}"),
                (Some(f), Some(l), None) => format!("{f}:{l}"),
                (Some(f), None, None) => f.clone(),
                _ => "<unknown>".to_string(),
            };
            let code = diag
                .code
                .as_deref()
                .map(|c| format!("[{c}] "))
                .unwrap_or_default();
            lines.push(format!(
                "- {} {code}{} ({loc})",
                diag.level.to_uppercase(),
                diag.message
            ));
            if let Some(hint) = &diag.suggestion {
                lines.push(format!("  -> Fix hint: {hint}"));
            }
        }
        if self.diagnostics.is_empty() && !self.success && !self.raw_summary.is_empty() {
            lines.push(self.raw_summary.clone());
        }
        lines.join("\n")
    }
}

/// Detected project toolchain inside a repository (supports polyglot monorepos).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetectedToolchain {
    pub id: String,
    pub label: String,
    pub root_rel: String,
    pub manifest_file: String,
}

impl DetectedToolchain {
    pub fn display_badge(&self) -> String {
        if self.root_rel == "." || self.root_rel.is_empty() {
            self.label.clone()
        } else {
            format!("{} ({})", self.label, self.root_rel)
        }
    }
}

/// Request targeting a specific verification mode, optional scope, test filter, affected file, or toolchain.
#[derive(Debug, Clone, Default)]
pub struct VerificationTarget {
    pub kind: VerificationKind,
    pub target_scope: Option<String>,
    /// Optional targeted test name, file, or module filter (for `cargo test`, `pytest`, `vitest`/`jest`, `go test`).
    pub test_filter: Option<String>,
    pub affected_file: Option<String>,
    pub toolchain_override: Option<String>,
}

impl Default for VerificationKind {
    fn default() -> Self {
        Self::Check
    }
}

/// Pluggable toolchain adapter capable of detecting project roots and running structured checks.
#[async_trait]
pub trait VerificationAdapter: Send + Sync {
    fn id(&self) -> &'static str;
    fn display_name(&self) -> &'static str;
    fn manifest_names(&self) -> &'static [&'static str];
    async fn run(
        &self,
        repo_root: &Path,
        subproject_rel: &str,
        target: &VerificationTarget,
    ) -> Result<VerificationReport>;
}

/// Multi-language toolchain detector & verifier supporting single-language repos and polyglot monorepos.
#[derive(Clone)]
pub struct ToolchainVerifier {
    repo_root: PathBuf,
    adapters: Vec<Arc<dyn VerificationAdapter>>,
}

impl ToolchainVerifier {
    pub fn new(repo_root: impl Into<PathBuf>) -> Self {
        Self {
            repo_root: strip_verbatim_prefix(&repo_root.into()),
            adapters: vec![
                Arc::new(rust::RustVerificationAdapter),
                Arc::new(typescript::TypeScriptVerificationAdapter),
                Arc::new(python::PythonVerificationAdapter),
                Arc::new(go::GoVerificationAdapter),
            ],
        }
    }

    pub fn repo_root(&self) -> &Path {
        &self.repo_root
    }

    /// Detect all supported project toolchains across the workspace (up to depth 4, honoring `.gitignore`).
    pub fn detect_toolchains(&self) -> Vec<DetectedToolchain> {
        let mut found = Vec::new();
        let mut seen: BTreeSet<(String, String)> = BTreeSet::new();

        let walker = WalkBuilder::new(&self.repo_root)
            .max_depth(Some(4))
            .hidden(true)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .ignore(true)
            .filter_entry(|e| {
                if e.depth() == 0 {
                    return true;
                }
                let name = e.file_name().to_str().unwrap_or("");
                !is_ignored_toolchain_dir(name)
            })
            .build();

        for entry in walker.flatten() {
            if !entry.file_type().is_some_and(|ft| ft.is_file()) {
                continue;
            }
            let Some(file_name) = entry.file_name().to_str() else {
                continue;
            };
            let Some(parent) = entry.path().parent() else {
                continue;
            };
            let root_rel = parent
                .strip_prefix(&self.repo_root)
                .ok()
                .map(|p| {
                    let s = p.to_string_lossy().replace('\\', "/");
                    if s.is_empty() { ".".to_string() } else { s }
                })
                .unwrap_or_else(|| ".".to_string());

            for adapter in &self.adapters {
                if adapter
                    .manifest_names()
                    .iter()
                    .any(|m| m.eq_ignore_ascii_case(file_name))
                {
                    // Avoid treating nested Cargo workspace member manifests as independent roots
                    // if a root Cargo.toml already exists at `.`.
                    if adapter.id() == "rust"
                        && root_rel != "."
                        && self.repo_root.join("Cargo.toml").exists()
                    {
                        continue;
                    }
                    let key = (adapter.id().to_string(), root_rel.clone());
                    if seen.insert(key) {
                        found.push(DetectedToolchain {
                            id: adapter.id().to_string(),
                            label: adapter.display_name().to_string(),
                            root_rel: root_rel.clone(),
                            manifest_file: file_name.to_string(),
                        });
                    }
                }
            }
        }

        found.sort_by(|a, b| {
            let a_root = a.root_rel == ".";
            let b_root = b.root_rel == ".";
            b_root
                .cmp(&a_root)
                .then_with(|| a.root_rel.cmp(&b.root_rel))
                .then_with(|| a.id.cmp(&b.id))
        });
        found
    }

    pub fn has_any_toolchain(&self) -> bool {
        !self.detect_toolchains().is_empty()
    }

    /// Resolve the best matching detected toolchain for a [`VerificationTarget`].
    ///
    /// In polyglot monorepos, when `affected_file` is supplied (e.g. `web/src/App.tsx`),
    /// selects the deepest enclosing toolchain root (`web/` -> TypeScript) rather than
    /// running an unrelated sub-project (`backend/` -> Rust).
    pub fn resolve_toolchain(&self, target: &VerificationTarget) -> Option<DetectedToolchain> {
        let detected = self.detect_toolchains();
        if detected.is_empty() {
            return None;
        }

        let requested_id = target.toolchain_override.as_deref().map(normalize_toolchain_id);

        let candidates: Vec<DetectedToolchain> = match requested_id {
            Some(ref id) => detected.into_iter().filter(|d| d.id == *id).collect(),
            None => detected,
        };
        if candidates.is_empty() {
            return None;
        }

        // 1. Match deepest enclosing directory for `affected_file` (polyglot monorepo routing).
        if let Some(ref rel_file) = target.affected_file {
            let norm_file = rel_file.trim().replace('\\', "/").trim_start_matches("./").to_string();
            let file_lang_hint = toolchain_hint_from_file(&norm_file);
            let mut enclosing: Vec<&DetectedToolchain> = candidates
                .iter()
                .filter(|tc| is_ancestor_root(&tc.root_rel, &norm_file))
                .collect();
            enclosing.sort_by(|a, b| {
                let a_lang_match = file_lang_hint == Some(a.id.as_str());
                let b_lang_match = file_lang_hint == Some(b.id.as_str());
                let a_depth = root_depth(&a.root_rel);
                let b_depth = root_depth(&b.root_rel);
                b_depth
                    .cmp(&a_depth)
                    .then_with(|| b_lang_match.cmp(&a_lang_match))
            });
            if let Some(best) = enclosing.first() {
                return Some((*best).clone());
            }
        }

        // 2. Match explicit `target_scope` if it names a detected sub-project directory.
        if let Some(ref scope) = target.target_scope {
            let norm_scope = scope.trim().replace('\\', "/").trim_matches('/').to_string();
            if let Some(exact) = candidates
                .iter()
                .find(|tc| tc.root_rel == norm_scope || tc.root_rel.ends_with(&format!("/{norm_scope}")))
            {
                return Some(exact.clone());
            }
        }

        // 3. Fall back to root-level or first detected toolchain.
        candidates.into_iter().next()
    }

    /// Run automated verification against the resolved toolchain or custom [`RepoProfile`] override.
    pub async fn run(
        &self,
        target: &VerificationTarget,
        profile: Option<&RepoProfile>,
    ) -> Result<VerificationReport> {
        // Custom command override from out-of-tree LanceDB `RepoProfile`.
        if let Some(prof) = profile {
            let custom_cmd = match target.kind {
                VerificationKind::Check => prof.check_command.as_deref(),
                VerificationKind::Lint | VerificationKind::Clippy => prof.lint_command.as_deref(),
                VerificationKind::Test => prof.test_command.as_deref(),
            };
            if let Some(cmd) = custom_cmd.map(str::trim).filter(|s| !s.is_empty()) {
                return custom::run_custom_command(
                    &self.repo_root,
                    ".",
                    target.kind,
                    cmd,
                )
                .await;
            }
        }

        let Some(selected) = self.resolve_toolchain(target) else {
            return Ok(VerificationReport {
                toolchain: "none".to_string(),
                working_dir: ".".to_string(),
                kind: target.kind,
                command: "no-toolchain-detected".to_string(),
                success: true,
                error_count: 0,
                warning_count: 0,
                diagnostics: Vec::new(),
                raw_summary:
                    "No supported project manifest (Cargo.toml, package.json, tsconfig.json, pyproject.toml, go.mod) found; skipped verification."
                        .to_string(),
            });
        };

        let adapter = self
            .adapters
            .iter()
            .find(|a| a.id() == selected.id)
            .expect("detected toolchain always has matching adapter");

        adapter
            .run(&self.repo_root, &selected.root_rel, target)
            .await
    }
}

pub(crate) fn subproject_dir(repo_root: &Path, subproject_rel: &str) -> PathBuf {
    if subproject_rel.is_empty() || subproject_rel == "." {
        repo_root.to_path_buf()
    } else {
        repo_root.join(subproject_rel)
    }
}

pub(crate) fn normalize_toolchain_id(raw: &str) -> String {
    match raw.trim().to_ascii_lowercase().as_str() {
        "rs" | "rust" | "cargo" => "rust".to_string(),
        "ts" | "tsx" | "js" | "jsx" | "typescript" | "javascript" | "node" | "npm" => {
            "typescript".to_string()
        }
        "py" | "python" | "pytest" | "ruff" | "pyright" | "mypy" => "python".to_string(),
        "go" | "golang" => "go".to_string(),
        other => other.to_string(),
    }
}

fn toolchain_hint_from_file(rel_path: &str) -> Option<&'static str> {
    let lower = rel_path.to_ascii_lowercase();
    if lower.ends_with(".rs") || lower.ends_with("cargo.toml") {
        Some("rust")
    } else if lower.ends_with(".ts")
        || lower.ends_with(".tsx")
        || lower.ends_with(".mts")
        || lower.ends_with(".cts")
        || lower.ends_with(".js")
        || lower.ends_with(".jsx")
        || lower.ends_with(".mjs")
        || lower.ends_with(".cjs")
        || lower.ends_with("package.json")
        || lower.ends_with("tsconfig.json")
    {
        Some("typescript")
    } else if lower.ends_with(".py")
        || lower.ends_with(".pyi")
        || lower.ends_with("pyproject.toml")
        || lower.ends_with("requirements.txt")
    {
        Some("python")
    } else if lower.ends_with(".go") || lower.ends_with("go.mod") {
        Some("go")
    } else {
        None
    }
}

fn is_ancestor_root(root_rel: &str, file_rel: &str) -> bool {
    if root_rel == "." || root_rel.is_empty() {
        return true;
    }
    let prefix = format!("{}/", root_rel.trim_end_matches('/'));
    file_rel == root_rel || file_rel.starts_with(&prefix)
}

fn root_depth(root_rel: &str) -> usize {
    if root_rel == "." || root_rel.is_empty() {
        0
    } else {
        root_rel.split('/').filter(|s| !s.is_empty()).count()
    }
}

fn is_ignored_toolchain_dir(name: &str) -> bool {
    matches!(
        name,
        ".git"
            | "target"
            | "node_modules"
            | ".venv"
            | "venv"
            | "__pycache__"
            | ".pytest_cache"
            | ".mypy_cache"
            | ".ruff_cache"
            | "dist"
            | "build"
            | ".next"
            | "vendor"
            | ".idea"
            | ".vscode"
    )
}

pub(crate) fn truncate_tail(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.len() <= max_chars {
        return trimmed.to_string();
    }
    let start = trimmed.len().saturating_sub(max_chars);
    let safe_start = trimmed
        .char_indices()
        .find(|(idx, _)| *idx >= start)
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    format!("...{}", &trimmed[safe_start..])
}
