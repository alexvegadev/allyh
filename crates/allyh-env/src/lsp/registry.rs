use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::error::Result;
use crate::lsp::client::{LspClient, LspServerSpec};
use crate::lsp::protocol::server_family_for_language_id;
use crate::path::strip_verbatim_prefix;

/// Maximum consecutive crash restarts before cooling down a failing server.
const MAX_CONSECUTIVE_CRASHES: u32 = 3;
/// Default cooldown after consecutive server crashes.
const CRASH_COOLDOWN: Duration = Duration::from_secs(30);
/// Default initialization timeout for real language servers.
const DEFAULT_INIT_TIMEOUT: Duration = Duration::from_secs(10);

/// Lifecycle state of a managed LSP server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LspServerState {
    /// Binary found on PATH (or configured via custom spec), awaiting lazy startup.
    Ready,
    /// Live stdio LSP client initialized and responding.
    Running,
    /// Server binary not installed on PATH; transparent `tree-sitter` + CLI fallback active.
    Unavailable,
    /// Server exited unexpectedly and is in restart cooldown.
    Crashed,
}

impl LspServerState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Running => "running",
            Self::Unavailable => "unavailable",
            Self::Crashed => "crashed",
        }
    }
}

/// Public diagnostic status snapshot for an LSP server (surfaced in TUI Inspector & `/lsp`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LspServerStatus {
    pub family: String,
    pub server_name: String,
    pub command: String,
    pub state: LspServerState,
    pub open_documents: usize,
    pub cached_diagnostic_files: usize,
    pub restart_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}

impl LspServerStatus {
    pub fn display_badge(&self) -> String {
        format!("{} ({})", self.server_name, self.state.as_str())
    }
}

struct ManagedServerSlot {
    spec: LspServerSpec,
    client: Option<Arc<LspClient>>,
    consecutive_crashes: u32,
    restart_count: u32,
    last_crash: Option<Instant>,
    last_error: Option<String>,
}

/// Manages per-language stdio LSP servers with lazy startup, crash recovery, and idle shutdown.
#[derive(Clone)]
pub struct LspManager {
    repo_root: PathBuf,
    slots: Arc<Mutex<HashMap<String, ManagedServerSlot>>>,
    init_timeout: Duration,
    idle_timeout: Duration,
}

impl LspManager {
    /// Create a manager pre-populated by auto-discovering installed LSP binaries on `PATH`.
    pub fn discover_default(repo_root: impl Into<PathBuf>) -> Self {
        let root = strip_verbatim_prefix(&repo_root.into());
        let mut slots = HashMap::new();
        for (family, spec) in discover_installed_presets() {
            slots.insert(
                family,
                ManagedServerSlot {
                    spec,
                    client: None,
                    consecutive_crashes: 0,
                    restart_count: 0,
                    last_crash: None,
                    last_error: None,
                },
            );
        }
        Self {
            repo_root: root,
            slots: Arc::new(Mutex::new(slots)),
            init_timeout: DEFAULT_INIT_TIMEOUT,
            idle_timeout: Duration::from_secs(300),
        }
    }

    /// Create an empty manager (useful for tests registering explicit mock LSP specs).
    pub fn empty(repo_root: impl Into<PathBuf>) -> Self {
        Self {
            repo_root: strip_verbatim_prefix(&repo_root.into()),
            slots: Arc::new(Mutex::new(HashMap::new())),
            init_timeout: DEFAULT_INIT_TIMEOUT,
            idle_timeout: Duration::from_secs(300),
        }
    }

    pub fn with_timeouts(mut self, init_timeout: Duration, idle_timeout: Duration) -> Self {
        self.init_timeout = init_timeout;
        self.idle_timeout = idle_timeout;
        self
    }

    /// Register or override a language server spec (shuts down any prior client for that family).
    pub async fn register_server(&self, spec: LspServerSpec) {
        let family = server_family_for_language_id(&spec.family)
            .map(str::to_string)
            .unwrap_or_else(|| spec.family.to_ascii_lowercase());
        let old_client = {
            let mut guard = self.slots.lock().await;
            let prev = guard.remove(&family).and_then(|s| s.client);
            guard.insert(
                family.clone(),
                ManagedServerSlot {
                    spec: LspServerSpec { family, ..spec },
                    client: None,
                    consecutive_crashes: 0,
                    restart_count: 0,
                    last_crash: None,
                    last_error: None,
                },
            );
            prev
        };
        if let Some(old) = old_client {
            old.shutdown().await;
        }
    }

    /// Return whether an LSP server is configured/available for `language_id`.
    pub async fn has_server_for(&self, language_id: &str) -> bool {
        let Some(family) = server_family_for_language_id(language_id) else {
            return false;
        };
        self.slots.lock().await.contains_key(family)
    }

    /// Lazily start (or recover after crash) and return the live [`LspClient`] for `language_id`.
    /// Returns `Ok(None)` cleanly if no server binary is installed or if the server is in cooldown.
    pub async fn get_or_start(&self, language_id: &str) -> Option<Arc<LspClient>> {
        let family = server_family_for_language_id(language_id)?;
        let spec_to_start = {
            let mut guard = self.slots.lock().await;
            let slot = guard.get_mut(family)?;

            if let Some(ref client) = slot.client {
                if client.is_alive() {
                    return Some(Arc::clone(client));
                }
                // Detected server crash -> record and clear stale client.
                slot.client = None;
                slot.consecutive_crashes += 1;
                slot.restart_count += 1;
                slot.last_crash = Some(Instant::now());
                slot.last_error = Some(format!(
                    "LSP `{}` exited unexpectedly (restart #{})",
                    slot.spec.name, slot.restart_count
                ));
            }

            if slot.consecutive_crashes >= MAX_CONSECUTIVE_CRASHES
                && slot
                    .last_crash
                    .is_some_and(|t| t.elapsed() < CRASH_COOLDOWN)
            {
                return None;
            }
            if slot.consecutive_crashes >= MAX_CONSECUTIVE_CRASHES {
                slot.consecutive_crashes = 0;
            }

            if slot.spec.require_manifest
                && !workspace_has_manifest_for_family(&self.repo_root, family)
            {
                return None;
            }

            slot.spec.clone()
        };

        match LspClient::start(spec_to_start.clone(), &self.repo_root, self.init_timeout).await {
            Ok(client) => {
                let arc = Arc::new(client);
                let mut guard = self.slots.lock().await;
                if let Some(slot) = guard.get_mut(family) {
                    slot.client = Some(Arc::clone(&arc));
                    slot.consecutive_crashes = 0;
                    slot.last_error = None;
                }
                Some(arc)
            }
            Err(err) => {
                let mut guard = self.slots.lock().await;
                if let Some(slot) = guard.get_mut(family) {
                    slot.consecutive_crashes += 1;
                    slot.last_crash = Some(Instant::now());
                    slot.last_error = Some(err.to_string());
                }
                tracing::debug!(server = %spec_to_start.name, error = %err, "LSP startup failed; falling back to tree-sitter/CLI");
                None
            }
        }
    }

    /// Return current status of all configured/discovered LSP servers (sorted by family).
    pub async fn status(&self) -> Vec<LspServerStatus> {
        let snapshot: Vec<(LspServerSpec, Option<Arc<LspClient>>, u32, u32, Option<String>)> = {
            let guard = self.slots.lock().await;
            guard
                .values()
                .map(|s| {
                    (
                        s.spec.clone(),
                        s.client.clone(),
                        s.consecutive_crashes,
                        s.restart_count,
                        s.last_error.clone(),
                    )
                })
                .collect()
        };

        let mut out = Vec::with_capacity(snapshot.len());
        for (spec, client_opt, crashes, restarts, last_error) in snapshot {
            let (state, open_documents, cached_diagnostic_files) = match client_opt {
                Some(client) if client.is_alive() => (
                    LspServerState::Running,
                    client.open_document_count().await,
                    client.cached_diagnostic_files_count().await,
                ),
                Some(_) => (LspServerState::Crashed, 0, 0),
                None if crashes > 0 => (LspServerState::Crashed, 0, 0),
                None => (LspServerState::Ready, 0, 0),
            };
            out.push(LspServerStatus {
                family: spec.family,
                server_name: spec.name,
                command: spec.command,
                state,
                open_documents,
                cached_diagnostic_files,
                restart_count: restarts,
                last_error,
            });
        }
        out.sort_by(|a, b| a.family.cmp(&b.family));
        out
    }

    /// Return status badges suitable for the TUI Inspector sidebar (filtering to relevant workspace languages if provided).
    pub async fn active_badges(&self, detected_languages: &[String]) -> Vec<String> {
        let statuses = self.status().await;
        let lang_families: Vec<&'static str> = detected_languages
            .iter()
            .filter_map(|l| server_family_for_language_id(l))
            .collect();

        statuses
            .into_iter()
            .filter(|s| {
                s.state == LspServerState::Running
                    || lang_families.is_empty()
                    || lang_families.contains(&s.family.as_str())
            })
            .map(|s| s.display_badge())
            .collect()
    }

    /// Restart one language server family (or all active servers when `family_filter` is `None`).
    pub async fn restart(&self, family_filter: Option<&str>) -> Result<Vec<LspServerStatus>> {
        let target_family = family_filter.map(|raw| {
            server_family_for_language_id(raw)
                .map(str::to_string)
                .unwrap_or_else(|| raw.trim().to_ascii_lowercase())
        });

        let to_stop: Vec<(String, Option<Arc<LspClient>>)> = {
            let mut guard = self.slots.lock().await;
            let mut list = Vec::new();
            for (fam, slot) in guard.iter_mut() {
                if target_family
                    .as_deref()
                    .is_none_or(|wanted| fam.eq_ignore_ascii_case(wanted))
                {
                    let prev = slot.client.take();
                    slot.consecutive_crashes = 0;
                    slot.last_crash = None;
                    slot.last_error = None;
                    if prev.is_some() {
                        slot.restart_count += 1;
                    }
                    list.push((fam.clone(), prev));
                }
            }
            list
        };

        for (_, client) in &to_stop {
            if let Some(c) = client {
                c.shutdown().await;
            }
        }

        // Eagerly restart explicitly requested server (or previously running servers).
        for (fam, had_client) in to_stop {
            if target_family.is_some() || had_client.is_some() {
                let _ = self.get_or_start(&fam).await;
            }
        }

        Ok(self.status().await)
    }

    /// Shut down any running LSP servers that have been idle longer than `self.idle_timeout`.
    pub async fn shutdown_idle(&self) -> usize {
        let clients: Vec<(String, Arc<LspClient>)> = {
            let guard = self.slots.lock().await;
            guard
                .iter()
                .filter_map(|(fam, slot)| slot.client.as_ref().map(|c| (fam.clone(), Arc::clone(c))))
                .collect()
        };

        let mut stopped = 0usize;
        for (fam, client) in clients {
            let idle_for = client.last_activity().await.elapsed();
            if idle_for >= self.idle_timeout || !client.is_alive() {
                let removed = {
                    let mut guard = self.slots.lock().await;
                    guard.get_mut(&fam).and_then(|s| s.client.take())
                };
                if let Some(c) = removed {
                    c.shutdown().await;
                    stopped += 1;
                }
            }
        }
        stopped
    }

    /// Cleanly shut down all running LSP servers.
    pub async fn shutdown_all(&self) {
        let clients: Vec<Arc<LspClient>> = {
            let mut guard = self.slots.lock().await;
            guard
                .values_mut()
                .filter_map(|slot| slot.client.take())
                .collect()
        };
        for client in clients {
            client.shutdown().await;
        }
    }
}

/// Built-in preset list of well-known LSP servers per language family, in preference order.
fn builtin_presets() -> Vec<(&'static str, Vec<(&'static str, &'static str, &'static [&'static str])>)> {
    vec![
        ("rust", vec![("rust-analyzer", "rust-analyzer", &[])]),
        (
            "typescript",
            vec![
                ("vtsls", "vtsls", &["--stdio"]),
                (
                    "typescript-language-server",
                    "typescript-language-server",
                    &["--stdio"],
                ),
            ],
        ),
        (
            "python",
            vec![
                ("basedpyright", "basedpyright-langserver", &["--stdio"]),
                ("pyright", "pyright-langserver", &["--stdio"]),
                ("pylsp", "pylsp", &[]),
            ],
        ),
        ("go", vec![("gopls", "gopls", &[])]),
        ("cpp", vec![("clangd", "clangd", &[])]),
    ]
}

/// Probe `PATH` for the highest-priority installed LSP server per language family.
fn discover_installed_presets() -> Vec<(String, LspServerSpec)> {
    let mut found = Vec::new();
    for (family, candidates) in builtin_presets() {
        for (name, bin, args) in candidates {
            if let Some(resolved) = find_binary_on_path(bin) {
                found.push((
                    family.to_string(),
                    LspServerSpec {
                        family: family.to_string(),
                        name: name.to_string(),
                        command: resolved.to_string_lossy().into_owned(),
                        args: args.iter().map(|s| (*s).to_string()).collect(),
                        env: Vec::new(),
                        require_manifest: true,
                    },
                ));
                break;
            }
        }
    }
    found
}

/// Cross-platform PATH executable lookup (honoring `PATHEXT` on Windows).
pub fn find_binary_on_path(binary: &str) -> Option<PathBuf> {
    let candidate = Path::new(binary);
    if candidate.is_absolute() || binary.contains('/') || binary.contains('\\') {
        return candidate.is_file().then(|| candidate.to_path_buf());
    }

    let path_var = std::env::var_os("PATH")?;
    let exts: Vec<String> = if cfg!(windows) {
        std::env::var("PATHEXT")
            .unwrap_or_else(|_| ".EXE;.CMD;.BAT;.COM".to_string())
            .split(';')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    } else {
        vec![String::new()]
    };

    for dir in std::env::split_paths(&path_var) {
        let direct = dir.join(binary);
        if direct.is_file() {
            return Some(direct);
        }
        if cfg!(windows) {
            for ext in &exts {
                let with_ext = dir.join(format!("{binary}{ext}"));
                if with_ext.is_file() {
                    return Some(with_ext);
                }
            }
        }
    }
    None
}

fn workspace_has_manifest_for_family(repo_root: &Path, family: &str) -> bool {
    let manifests: &[&str] = match family {
        "rust" => &["Cargo.toml"],
        "typescript" => &["tsconfig.json", "package.json", "jsconfig.json"],
        "python" => &["pyproject.toml", "setup.py", "requirements.txt", "pyrightconfig.json"],
        "go" => &["go.mod"],
        "cpp" => &["compile_commands.json", "CMakeLists.txt"],
        _ => return true,
    };
    if manifests.iter().any(|m| repo_root.join(m).exists()) {
        return true;
    }
    // Check immediate sub-project directories (common polyglot monorepo layout).
    if let Ok(entries) = std::fs::read_dir(repo_root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && manifests.iter().any(|m| path.join(m).exists()) {
                return true;
            }
        }
    }
    false
}
