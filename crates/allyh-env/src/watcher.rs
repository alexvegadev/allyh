use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use allyh_memory::{CodeGraphSnapshot, MemoryStore, RepoProfile};
use chrono::{DateTime, Utc};
use ignore::WalkBuilder;
use ignore::overrides::OverrideBuilder;
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use similar::TextDiff;
use tokio::sync::mpsc;

use crate::cargo::{CargoRunner, CargoVerificationReport, VerificationKind};
use crate::error::{EnvError, Result};
use crate::git::{GitState, GitTracker};
use crate::graph::{EgoGraph, RepoGraph};
use crate::lang::{LanguageRegistry, SourceLanguage};
use crate::lsp::{
    DEFAULT_LSP_DIAGNOSTIC_WAIT, LspManager, LspServerStatus, collect_lsp_diagnostics_report,
    enrich_ego_graph_with_lsp,
};
use crate::path::{
    canonicalize_normalized, has_windows_drive_prefix, path_to_slash_string, strip_root_prefix,
    strip_verbatim_prefix,
};
use crate::verify::{DetectedToolchain, ToolchainVerifier, VerificationReport, VerificationTarget};

/// Filesystem change notification emitted by the live `notify` watcher.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsChangeEvent {
    pub path: String,
    pub kind: String,
    pub observed_at: DateTime<Utc>,
}

/// Out-of-band manual user edit detected via SHA-256 comparison against `code_graph_snapshots`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutOfBandEdit {
    pub file_path: String,
    pub previous_hash: Option<String>,
    pub current_hash: Option<String>,
    pub diff_summary: String,
}

/// Structured environment delta computed before every agent turn.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnvDelta {
    pub generated_at: Option<DateTime<Utc>>,
    pub git: GitState,
    pub out_of_band_edits: Vec<OutOfBandEdit>,
    pub recent_fs_events: Vec<FsChangeEvent>,
    pub reindexed_files: Vec<String>,
    pub build_status: Option<String>,
    #[serde(default)]
    pub detected_languages: Vec<String>,
    #[serde(default)]
    pub detected_toolchains: Vec<String>,
    #[serde(default)]
    pub active_lsp_servers: Vec<String>,
}

impl EnvDelta {
    /// Render a concise environment block for the Dynamic Preamble Assembler.
    pub fn render_for_preamble(&self) -> String {
        let mut lines = Vec::new();
        lines.push("## Live Environment (EnvDelta)".to_string());
        lines.push(format!(
            "- Git branch: `{}` (HEAD: {})",
            self.git.branch,
            self.git.head_commit.as_deref().unwrap_or("none")
        ));
        if !self.detected_languages.is_empty() {
            lines.push(format!(
                "- Languages: {}",
                self.detected_languages.join(", ")
            ));
        }
        if !self.detected_toolchains.is_empty() {
            lines.push(format!(
                "- Toolchains: {}",
                self.detected_toolchains.join(", ")
            ));
        }
        if !self.active_lsp_servers.is_empty() {
            lines.push(format!("- LSP: {}", self.active_lsp_servers.join(", ")));
        }
        if let Some(status) = &self.build_status {
            lines.push(format!("- Last verification: {status}"));
        }
        if !self.git.dirty_files.is_empty() {
            lines.push(format!(
                "- Dirty files: {}",
                self.git
                    .dirty_files
                    .iter()
                    .take(15)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if !self.out_of_band_edits.is_empty() {
            lines.push("- Out-of-band manual edits detected (SHA-256 mismatch):".to_string());
            for edit in &self.out_of_band_edits {
                lines.push(format!("  - `{}`: {}", edit.file_path, edit.diff_summary));
            }
        }
        if !self.git.diff_summary.is_empty() && self.out_of_band_edits.is_empty() {
            lines.push(format!(
                "- Working-tree diff:\n```diff\n{}\n```",
                self.git.diff_summary
            ));
        }
        lines.join("\n")
    }
}

/// Live filesystem watcher backed by `notify::RecommendedWatcher`.
pub struct LiveFsWatcher {
    _watcher: RecommendedWatcher,
}

impl LiveFsWatcher {
    /// Start watching `repo_root` recursively, forwarding non-ignored file events into `tx`.
    pub fn start(
        repo_root: PathBuf,
        tx: mpsc::UnboundedSender<FsChangeEvent>,
    ) -> Result<Self> {
        let raw_root = strip_verbatim_prefix(&repo_root);
        let canonical_root = canonicalize_normalized(&raw_root);
        let watch_root = canonical_root.clone();
        let raw_root_cb = raw_root.clone();
        let canonical_root_cb = canonical_root.clone();

        let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
            let Ok(event) = res else {
                return;
            };
            let kind = match event.kind {
                EventKind::Create(_) => "created",
                EventKind::Modify(_) => "modified",
                EventKind::Remove(_) => "removed",
                _ => return,
            };
            for path in event.paths {
                let rel = strip_root_prefix(&path, &raw_root_cb, &canonical_root_cb)
                    .unwrap_or_else(|| path_to_slash_string(&path));
                if rel.is_empty() || should_ignore_rel_path(&rel) {
                    continue;
                }
                let _ = tx.send(FsChangeEvent {
                    path: rel,
                    kind: kind.to_string(),
                    observed_at: Utc::now(),
                });
            }
        })?;

        watcher.watch(&watch_root, RecursiveMode::Recursive)?;
        Ok(Self { _watcher: watcher })
    }
}

/// Coordinates repository hashing, incremental polyglot `tree-sitter` indexing, `git2` status,
/// `notify` event buffering, and pluggable toolchain verification for a workspace.
#[derive(Clone)]
pub struct EnvironmentMonitor {
    repo_root: PathBuf,
    repo_id: String,
    scope_id: String,
    store: MemoryStore,
    git: GitTracker,
    cargo: CargoRunner,
    registry: LanguageRegistry,
    verifier: ToolchainVerifier,
    lsp: LspManager,
    /// In-memory last-known file contents for concise out-of-band diff generation.
    last_known_contents: Arc<Mutex<HashMap<String, String>>>,
    /// Pending filesystem events received from `LiveFsWatcher`.
    pending_events: Arc<Mutex<Vec<FsChangeEvent>>>,
}

impl EnvironmentMonitor {
    pub fn new(
        repo_root: impl Into<PathBuf>,
        repo_id: impl Into<String>,
        scope_id: impl Into<String>,
        store: MemoryStore,
    ) -> Self {
        let root = strip_verbatim_prefix(&repo_root.into());
        let git = GitTracker::new(&root);
        let cargo = CargoRunner::new(&root);
        let registry = LanguageRegistry::default_all();
        let verifier = ToolchainVerifier::new(&root);
        let lsp = LspManager::discover_default(&root);
        Self {
            repo_root: root,
            repo_id: repo_id.into(),
            scope_id: scope_id.into(),
            store,
            git,
            cargo,
            registry,
            verifier,
            lsp,
            last_known_contents: Arc::new(Mutex::new(HashMap::new())),
            pending_events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Replace the auto-discovered [`LspManager`] (e.g., to disable LSP or inject custom/mock servers).
    pub fn with_lsp_manager(mut self, lsp: LspManager) -> Self {
        self.lsp = lsp;
        self
    }

    pub fn lsp(&self) -> &LspManager {
        &self.lsp
    }

    pub async fn lsp_status(&self) -> Vec<LspServerStatus> {
        self.lsp.status().await
    }

    pub async fn restart_lsp(&self, family: Option<&str>) -> Result<Vec<LspServerStatus>> {
        self.lsp.restart(family).await
    }

    pub async fn shutdown_lsp(&self) {
        self.lsp.shutdown_all().await;
    }

    pub fn repo_root(&self) -> &Path {
        &self.repo_root
    }

    pub fn repo_id(&self) -> &str {
        &self.repo_id
    }

    pub fn scope_id(&self) -> &str {
        &self.scope_id
    }

    pub fn git(&self) -> &GitTracker {
        &self.git
    }

    pub fn cargo(&self) -> &CargoRunner {
        &self.cargo
    }

    pub fn language_registry(&self) -> &LanguageRegistry {
        &self.registry
    }

    pub fn verifier(&self) -> &ToolchainVerifier {
        &self.verifier
    }

    pub fn detect_toolchains(&self) -> Vec<DetectedToolchain> {
        self.verifier.detect_toolchains()
    }

    pub fn has_any_toolchain(&self) -> bool {
        self.verifier.has_any_toolchain()
    }

    /// Load the optional out-of-tree [`RepoProfile`] stored in LanceDB for this partition.
    pub async fn load_repo_profile(&self) -> Option<RepoProfile> {
        self.store
            .get_repo_profile(&self.repo_id, &self.scope_id)
            .await
            .ok()
            .flatten()
    }

    /// Spawn a live `notify` watcher and buffer incoming events into this monitor.
    pub fn start_watching(&self) -> Result<(LiveFsWatcher, mpsc::UnboundedReceiver<FsChangeEvent>)> {
        let (raw_tx, mut raw_rx) = mpsc::unbounded_channel::<FsChangeEvent>();
        let (ui_tx, ui_rx) = mpsc::unbounded_channel::<FsChangeEvent>();
        let watcher = LiveFsWatcher::start(self.repo_root.clone(), raw_tx)?;

        let pending = Arc::clone(&self.pending_events);
        tokio::spawn(async move {
            while let Some(ev) = raw_rx.recv().await {
                if let Ok(mut guard) = pending.lock() {
                    guard.push(ev.clone());
                    if guard.len() > 200 {
                        let drain_count = guard.len() - 200;
                        guard.drain(0..drain_count);
                    }
                }
                let _ = ui_tx.send(ev);
            }
        });

        Ok((watcher, ui_rx))
    }

    /// Record the current content of a file after an authorized AllyH tool read/edit so
    /// subsequent out-of-band edits produce accurate unified diffs.
    pub fn cache_file_content(&self, rel_path: &str, content: &str) {
        if let Ok(mut guard) = self.last_known_contents.lock() {
            guard.insert(normalize_rel_path(rel_path), content.to_string());
        }
    }

    /// Resolve and validate a relative path so tools never escape the repository root
    /// across Windows, macOS, and Linux (blocking absolute paths, drive-relative paths,
    /// parent traversal `..`, and symlinks pointing outside the repository root).
    pub fn resolve_safe_path(&self, rel_path: &str) -> Result<(String, PathBuf)> {
        let raw_trimmed = rel_path.trim();
        if raw_trimmed.is_empty()
            || raw_trimmed.starts_with('/')
            || raw_trimmed.starts_with('\\')
            || has_windows_drive_prefix(raw_trimmed)
        {
            return Err(EnvError::PathEscape(rel_path.to_string()));
        }

        let slash_form = raw_trimmed.replace('\\', "/");
        if Path::new(&slash_form).is_absolute() || has_windows_drive_prefix(&slash_form) {
            return Err(EnvError::PathEscape(rel_path.to_string()));
        }

        let mut segments: Vec<String> = Vec::new();
        let mut abs_path = self.repo_root.clone();
        for comp in Path::new(&slash_form).components() {
            match comp {
                Component::CurDir => continue,
                Component::Normal(seg) => {
                    let s = seg.to_string_lossy();
                    if s.is_empty() || has_windows_drive_prefix(&s) {
                        return Err(EnvError::PathEscape(rel_path.to_string()));
                    }
                    segments.push(s.into_owned());
                    abs_path.push(seg);
                }
                Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                    return Err(EnvError::PathEscape(rel_path.to_string()));
                }
            }
        }

        if segments.is_empty() {
            return Err(EnvError::PathEscape(rel_path.to_string()));
        }

        // Prevent symlink escapes for existing targets or their nearest existing ancestor.
        if let Ok(canon_root) = std::fs::canonicalize(&self.repo_root) {
            let canon_root = strip_verbatim_prefix(&canon_root);
            let mut probe = abs_path.as_path();
            while !probe.exists() {
                match probe.parent() {
                    Some(parent) => probe = parent,
                    None => break,
                }
            }
            if probe.exists()
                && let Ok(canon_probe) = std::fs::canonicalize(probe)
            {
                let canon_probe = strip_verbatim_prefix(&canon_probe);
                if !canon_probe.starts_with(&canon_root) {
                    return Err(EnvError::PathEscape(rel_path.to_string()));
                }
            }
        }

        Ok((segments.join("/"), abs_path))
    }

    /// Compute SHA-256 hex digest of a file's bytes.
    pub fn sha256_str(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Re-index a single source file into `code_graph_snapshots` using the polyglot `LanguageRegistry`.
    pub async fn index_file(&self, rel_path: &str, turn: u64) -> Result<CodeGraphSnapshot> {
        let (normalized, abs_path) = self.resolve_safe_path(rel_path)?;
        let content = std::fs::read_to_string(&abs_path)?;
        let hash = Self::sha256_str(&content);
        self.cache_file_content(&normalized, &content);

        let (lang, file_graph) = self
            .registry
            .index_path(&normalized, &content)
            .unwrap_or_else(|_| (SourceLanguage::from_path(&normalized), Default::default()));

        let symbols_json = serde_json::to_string(&file_graph.symbols)?;
        let embed_text = format!(
            "{} {} {} {}",
            normalized,
            lang.as_str(),
            file_graph
                .symbols
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            file_graph.dependencies.join(" ")
        );
        let vector = self.store.embedder().embed(&embed_text).await?;

        let snapshot = CodeGraphSnapshot {
            repo_id: self.repo_id.clone(),
            scope_id: self.scope_id.clone(),
            file_path: normalized,
            language: lang.as_str().to_string(),
            sha256_hash: hash,
            last_read_turn: turn,
            symbols_json,
            dependencies: file_graph.dependencies,
            vector,
        };
        self.store.upsert_code_snapshot(&snapshot).await?;
        Ok(snapshot)
    }

    /// Scan workspace source files (respecting `.gitignore` and `RepoProfile` ignore overrides),
    /// compare SHA-256 hashes against LanceDB `code_graph_snapshots`, detect out-of-band manual
    /// user edits, trigger incremental polyglot `tree-sitter` re-indexing, and produce an
    /// [`EnvDelta`] ready for the Dynamic Preamble Assembler.
    pub async fn sync_and_collect_delta(
        &self,
        turn: u64,
        run_build_check: bool,
    ) -> Result<EnvDelta> {
        let git = self.git.inspect()?;
        let profile = self.load_repo_profile().await;
        let extra_ignores = profile
            .as_ref()
            .map(|p| p.extra_ignore_globs.as_slice())
            .unwrap_or(&[]);

        let existing_snapshots = self
            .store
            .list_code_snapshots(&self.repo_id, &self.scope_id)
            .await?;
        let existing_map: HashMap<String, CodeGraphSnapshot> = existing_snapshots
            .into_iter()
            .map(|s| (s.file_path.clone(), s))
            .collect();

        let disk_files = self.discover_source_files(extra_ignores);
        let disk_set: HashSet<String> = disk_files.iter().cloned().collect();

        let mut out_of_band_edits = Vec::new();
        let mut reindexed_files = Vec::new();
        let mut detected_langs: BTreeSet<String> = BTreeSet::new();

        for rel_path in &disk_files {
            let lang = SourceLanguage::from_path(rel_path);
            if lang.is_code() {
                detected_langs.insert(lang.display_name().to_string());
            }

            let abs = self.repo_root.join(rel_path);
            let Ok(content) = std::fs::read_to_string(&abs) else {
                continue;
            };
            let current_hash = Self::sha256_str(&content);

            match existing_map.get(rel_path) {
                Some(prev) if prev.sha256_hash == current_hash => {
                    if let Ok(mut guard) = self.last_known_contents.lock() {
                        guard.entry(rel_path.clone()).or_insert(content);
                    }
                }
                Some(prev) => {
                    let prev_content = self
                        .last_known_contents
                        .lock()
                        .ok()
                        .and_then(|g| g.get(rel_path).cloned());
                    let diff_summary = match prev_content {
                        Some(old) => summarize_text_diff(&old, &content),
                        None => format!(
                            "hash changed {}.. -> {}.. ({} lines)",
                            &prev.sha256_hash[..prev.sha256_hash.len().min(8)],
                            &current_hash[..current_hash.len().min(8)],
                            content.lines().count()
                        ),
                    };
                    out_of_band_edits.push(OutOfBandEdit {
                        file_path: rel_path.clone(),
                        previous_hash: Some(prev.sha256_hash.clone()),
                        current_hash: Some(current_hash),
                        diff_summary,
                    });
                    self.index_file(rel_path, turn).await?;
                    reindexed_files.push(rel_path.clone());
                }
                None => {
                    let is_after_initial_scan = !existing_map.is_empty();
                    if is_after_initial_scan {
                        out_of_band_edits.push(OutOfBandEdit {
                            file_path: rel_path.clone(),
                            previous_hash: None,
                            current_hash: Some(current_hash),
                            diff_summary: format!(
                                "new file created ({} lines)",
                                content.lines().count()
                            ),
                        });
                    }
                    self.index_file(rel_path, turn).await?;
                    reindexed_files.push(rel_path.clone());
                }
            }
        }

        // Remove snapshots for files deleted out-of-band.
        for (old_path, prev) in &existing_map {
            if !disk_set.contains(old_path) {
                out_of_band_edits.push(OutOfBandEdit {
                    file_path: old_path.clone(),
                    previous_hash: Some(prev.sha256_hash.clone()),
                    current_hash: None,
                    diff_summary: "file deleted outside AllyH".to_string(),
                });
                self.store
                    .delete_code_snapshot(&self.repo_id, &self.scope_id, old_path)
                    .await?;
            }
        }

        let recent_fs_events = self
            .pending_events
            .lock()
            .map(|mut g| std::mem::take(&mut *g))
            .unwrap_or_default();

        let detected_languages = if let Some(ref prof) = profile
            && !prof.languages_override.is_empty()
        {
            prof.languages_override.clone()
        } else {
            detected_langs.into_iter().collect()
        };

        let mut detected_toolchains: Vec<String> = self
            .verifier
            .detect_toolchains()
            .into_iter()
            .map(|tc| tc.display_badge())
            .collect();
        if let Some(ref prof) = profile
            && (prof.check_command.is_some()
                || prof.lint_command.is_some()
                || prof.test_command.is_some())
        {
            detected_toolchains.insert(0, "Custom profile".to_string());
        }

        let has_verifier = !detected_toolchains.is_empty();
        let build_status = if run_build_check && has_verifier {
            match self
                .verifier
                .run(&VerificationTarget::default(), profile.as_ref())
                .await
            {
                Ok(report) => Some(report.compact_status_line()),
                Err(err) => Some(format!("verification error: {err}")),
            }
        } else {
            None
        };

        let active_lsp_servers = self.lsp.active_badges(&detected_languages).await;

        Ok(EnvDelta {
            generated_at: Some(Utc::now()),
            git,
            out_of_band_edits,
            recent_fs_events,
            reindexed_files,
            build_status,
            detected_languages,
            detected_toolchains,
            active_lsp_servers,
        })
    }

    /// Build a 1-hop or 2-hop RepoGraph ego-graph around a symbol/file from LanceDB snapshots,
    /// automatically enriching edges with type-resolved LSP references/call-hierarchy when available.
    pub async fn ego_graph_for(&self, query_symbol: &str, max_hops: usize) -> Result<EgoGraph> {
        let snapshots = self
            .store
            .list_code_snapshots(&self.repo_id, &self.scope_id)
            .await?;
        let base = RepoGraph::build_ego_graph(&snapshots, query_symbol, max_hops);
        Ok(enrich_ego_graph_with_lsp(base, &snapshots, &self.repo_root, &self.lsp).await)
    }

    /// Attempt sub-second post-edit diagnostics via an active LSP server before falling back to CLI.
    pub async fn lsp_diagnostics_for_edit(
        &self,
        rel_path: &str,
        abs_path: &Path,
        content: &str,
        content_hash: &str,
    ) -> Option<VerificationReport> {
        collect_lsp_diagnostics_report(
            rel_path,
            abs_path,
            content,
            content_hash,
            &self.repo_root,
            &self.lsp,
            DEFAULT_LSP_DIAGNOSTIC_WAIT,
        )
        .await
    }

    /// Run language-agnostic verification using the pluggable [`ToolchainVerifier`] and
    /// any configured out-of-tree [`RepoProfile`] overrides.
    pub async fn run_verification(
        &self,
        target: &VerificationTarget,
    ) -> Result<VerificationReport> {
        let profile = self.load_repo_profile().await;
        self.verifier.run(target, profile.as_ref()).await
    }

    /// Backward-compatible Rust helper running `cargo check` / `clippy` / `test`.
    pub async fn run_cargo(
        &self,
        kind: VerificationKind,
        package: Option<&str>,
    ) -> Result<CargoVerificationReport> {
        self.cargo.run(kind, package).await
    }

    fn discover_source_files(&self, extra_ignore_globs: &[String]) -> Vec<String> {
        let mut builder = WalkBuilder::new(&self.repo_root);
        builder
            .hidden(true)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .ignore(true)
            .require_git(false)
            .filter_entry(|e| {
                e.depth() == 0 || !is_ignored_entry(e.file_name().to_str().unwrap_or(""))
            });

        if !extra_ignore_globs.is_empty() {
            let mut overrides = OverrideBuilder::new(&self.repo_root);
            for glob in extra_ignore_globs {
                let trimmed = glob.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let negated = if trimmed.starts_with('!') {
                    trimmed.to_string()
                } else {
                    format!("!{trimmed}")
                };
                let _ = overrides.add(&negated);
            }
            if let Ok(built) = overrides.build() {
                builder.overrides(built);
            }
        }

        let mut files = Vec::new();
        for entry in builder.build().flatten() {
            if !entry.file_type().is_some_and(|ft| ft.is_file()) {
                continue;
            }
            let path = entry.path();
            if let Ok(rel) = path.strip_prefix(&self.repo_root) {
                let rel_str = rel.to_string_lossy().replace('\\', "/");
                if !should_ignore_rel_path(&rel_str) && self.registry.is_tracked_file(&rel_str) {
                    files.push(rel_str);
                }
            }
        }
        files.sort();
        files
    }
}

/// Produce a compact diff summary between two file versions using `similar`.
pub fn summarize_text_diff(old: &str, new: &str) -> String {
    let diff = TextDiff::from_lines(old, new);
    let mut added = 0usize;
    let mut removed = 0usize;
    let mut preview = Vec::new();

    for change in diff.iter_all_changes() {
        match change.tag() {
            similar::ChangeTag::Insert => {
                added += 1;
                if preview.len() < 8 {
                    preview.push(format!("+{}", change.to_string().trim_end()));
                }
            }
            similar::ChangeTag::Delete => {
                removed += 1;
                if preview.len() < 8 {
                    preview.push(format!("-{}", change.to_string().trim_end()));
                }
            }
            similar::ChangeTag::Equal => {}
        }
    }

    if preview.is_empty() {
        format!("+{added} -{removed} lines")
    } else {
        format!("+{added} -{removed} lines ({})", preview.join(" | "))
    }
}

pub fn normalize_rel_path(path: &str) -> String {
    path.trim()
        .replace('\\', "/")
        .trim_start_matches("./")
        .to_string()
}

fn is_ignored_entry(name: &str) -> bool {
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
            | ".next"
            | "vendor"
            | ".idea"
            | ".vscode"
            | "dist"
            | "build"
    )
}

fn should_ignore_rel_path(rel: &str) -> bool {
    let rel = rel.trim_start_matches("./");
    rel.is_empty()
        || rel == ".git"
        || rel == "target"
        || rel.starts_with(".git/")
        || rel.starts_with("target/")
        || rel.starts_with("node_modules/")
        || rel.starts_with(".venv/")
        || rel.starts_with("venv/")
        || rel.starts_with("vendor/")
        || rel.contains("/target/")
        || rel.contains("/.git/")
        || rel.contains("/node_modules/")
        || rel.contains("/__pycache__/")
        || rel.contains("/.venv/")
        || rel.contains("/vendor/")
        || rel == "Cargo.lock"
        || rel == "package-lock.json"
        || rel == "pnpm-lock.yaml"
        || rel == "yarn.lock"
}
