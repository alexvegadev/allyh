use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::error::{EnvError, Result};
use crate::path::{canonicalize_normalized, path_to_slash_string};

/// Metadata describing an active Git worktree (main checkout or isolated agent worktree).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorktreeInfo {
    pub path: PathBuf,
    pub branch: String,
    pub head_commit: Option<String>,
    pub scope_id: Option<String>,
    pub is_main: bool,
}

/// Manages isolated `git worktree` checkouts per agent scope (`scope_id = task-<id>`)
/// so parallel AllyH agents can edit files independently while sharing one LanceDB URI.
pub struct WorktreeManager;

impl WorktreeManager {
    /// Create a new isolated `git worktree` for `scope_id` rooted at `destination_path`.
    ///
    /// Creates branch `allyh/<sanitized-scope>` from `base_ref` (defaulting to `HEAD`),
    /// leaving the main repository working tree completely clean and untouched.
    pub fn create_scoped_worktree(
        repo_root: &Path,
        scope_id: &str,
        base_ref: Option<&str>,
        destination_path: Option<&Path>,
    ) -> Result<WorktreeInfo> {
        let slug = sanitize_scope_slug(scope_id);
        if slug.is_empty() {
            return Err(EnvError::Command(
                "scope_id must contain at least one ASCII alphanumeric character".to_string(),
            ));
        }

        let dest = match destination_path {
            Some(p) => p.to_path_buf(),
            None => default_worktree_dir(repo_root, &slug),
        };
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let branch = format!("allyh/{slug}");
        let mut cmd = Command::new("git");
        cmd.current_dir(repo_root)
            .arg("worktree")
            .arg("add")
            .arg("-B")
            .arg(&branch)
            .arg(&dest);
        if let Some(base) = base_ref.map(str::trim).filter(|s| !s.is_empty()) {
            cmd.arg(base);
        }

        let output = cmd.output().map_err(|e| {
            EnvError::Command(format!("failed to invoke `git worktree add`: {e}"))
        })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(EnvError::Command(format!(
                "git worktree add failed: {}",
                stderr.trim()
            )));
        }

        let canonical_dest = canonicalize_normalized(&dest);
        Ok(WorktreeInfo {
            path: canonical_dest,
            branch,
            head_commit: None,
            scope_id: Some(scope_id.to_string()),
            is_main: false,
        })
    }

    /// List all Git worktrees attached to `repo_root`.
    pub fn list_worktrees(repo_root: &Path) -> Result<Vec<WorktreeInfo>> {
        let output = Command::new("git")
            .current_dir(repo_root)
            .args(["worktree", "list", "--porcelain"])
            .output()
            .map_err(|e| {
                EnvError::Command(format!("failed to invoke `git worktree list`: {e}"))
            })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(EnvError::Command(format!(
                "git worktree list failed: {}",
                stderr.trim()
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut out = Vec::new();
        let mut cur_path: Option<PathBuf> = None;
        let mut cur_head: Option<String> = None;
        let mut cur_branch = String::new();

        let flush = |out: &mut Vec<WorktreeInfo>,
                     path: &mut Option<PathBuf>,
                     head: &mut Option<String>,
                     branch: &mut String| {
            if let Some(p) = path.take() {
                let is_main = out.is_empty();
                let b = if branch.is_empty() {
                    "detached".to_string()
                } else {
                    std::mem::take(branch)
                };
                let scope_id = b.strip_prefix("allyh/").map(str::to_string);
                out.push(WorktreeInfo {
                    path: canonicalize_normalized(&p),
                    branch: b,
                    head_commit: head.take(),
                    scope_id,
                    is_main,
                });
            }
        };

        for line in stdout.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                flush(&mut out, &mut cur_path, &mut cur_head, &mut cur_branch);
                continue;
            }
            if let Some(p) = trimmed.strip_prefix("worktree ") {
                flush(&mut out, &mut cur_path, &mut cur_head, &mut cur_branch);
                cur_path = Some(PathBuf::from(p.trim()));
            } else if let Some(h) = trimmed.strip_prefix("HEAD ") {
                cur_head = Some(h.trim().chars().take(12).collect());
            } else if let Some(b) = trimmed.strip_prefix("branch ") {
                cur_branch = b
                    .trim()
                    .strip_prefix("refs/heads/")
                    .unwrap_or(b.trim())
                    .to_string();
            }
        }
        flush(&mut out, &mut cur_path, &mut cur_head, &mut cur_branch);
        Ok(out)
    }

    /// Remove a linked worktree by path, branch, or `scope_id`, and prune stale git metadata.
    pub fn remove_worktree(repo_root: &Path, path_or_scope: &str, force: bool) -> Result<PathBuf> {
        let needle = path_or_scope.trim();
        let listed = Self::list_worktrees(repo_root)?;
        let target_path = listed
            .iter()
            .find(|w| {
                !w.is_main
                    && (w.scope_id.as_deref() == Some(needle)
                        || w.branch == needle
                        || w.branch == format!("allyh/{}", sanitize_scope_slug(needle))
                        || path_to_slash_string(&w.path) == needle.replace('\\', "/"))
            })
            .map(|w| w.path.clone())
            .unwrap_or_else(|| PathBuf::from(needle));

        let mut cmd = Command::new("git");
        cmd.current_dir(repo_root).arg("worktree").arg("remove");
        if force {
            cmd.arg("--force");
        }
        cmd.arg(&target_path);

        let output = cmd.output().map_err(|e| {
            EnvError::Command(format!("failed to invoke `git worktree remove`: {e}"))
        })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(EnvError::Command(format!(
                "git worktree remove failed: {}",
                stderr.trim()
            )));
        }

        let _ = Command::new("git")
            .current_dir(repo_root)
            .args(["worktree", "prune"])
            .output();
        Ok(target_path)
    }
}

fn default_worktree_dir(repo_root: &Path, scope_slug: &str) -> PathBuf {
    let canonical = canonicalize_normalized(repo_root);
    let repo_name = canonical
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("repo");
    let base = dirs::data_local_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("allyh")
        .join("worktrees")
        .join(repo_name);
    base.join(scope_slug)
}

fn sanitize_scope_slug(raw: &str) -> String {
    let mut out = String::new();
    for ch in raw.trim().chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}
