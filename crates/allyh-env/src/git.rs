use std::path::{Path, PathBuf};

use git2::{DiffOptions, Repository, StatusOptions};
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::path::{canonicalize_normalized, path_to_slash_string, strip_root_prefix, strip_verbatim_prefix};

/// Snapshot of a repository's current Git state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GitState {
    pub is_git_repo: bool,
    pub branch: String,
    pub head_commit: Option<String>,
    pub origin_url: Option<String>,
    pub dirty_files: Vec<String>,
    pub untracked_files: Vec<String>,
    pub diff_summary: String,
}

/// Reads Git branch, HEAD, working-tree dirtiness, and concise diffs via `git2`.
#[derive(Debug, Clone)]
pub struct GitTracker {
    repo_root: PathBuf,
}

impl GitTracker {
    pub fn new(repo_root: impl Into<PathBuf>) -> Self {
        Self {
            repo_root: strip_verbatim_prefix(&repo_root.into()),
        }
    }

    /// Return the canonical identifier used to derive `repo_id` (origin URL if present,
    /// otherwise canonicalized repository root path).
    pub fn canonical_repo_identifier(&self) -> String {
        if let Ok(repo) = Repository::discover(&self.repo_root)
            && let Ok(remote) = repo.find_remote("origin")
            && let Some(url) = remote.url()
            && !url.trim().is_empty()
        {
            return url.trim().to_string();
        }
        let canonical = canonicalize_normalized(&self.repo_root);
        path_to_slash_string(&canonical)
    }

    /// Inspect the current Git repository state. Non-git workspaces return a clean fallback.
    pub fn inspect(&self) -> Result<GitState> {
        let Ok(repo) = Repository::discover(&self.repo_root) else {
            return Ok(GitState {
                is_git_repo: false,
                branch: "no-git".to_string(),
                ..GitState::default()
            });
        };

        let branch = match repo.head() {
            Ok(head) => head
                .shorthand()
                .map(str::to_string)
                .unwrap_or_else(|| "detached".to_string()),
            Err(_) => "unborn".to_string(),
        };

        let head_commit = repo
            .head()
            .ok()
            .and_then(|h| h.peel_to_commit().ok())
            .map(|c| c.id().to_string().chars().take(12).collect());

        let origin_url = repo
            .find_remote("origin")
            .ok()
            .and_then(|r| r.url().map(str::to_string));

        let mut status_opts = StatusOptions::new();
        status_opts
            .include_untracked(true)
            .recurse_untracked_dirs(true)
            .exclude_submodules(true);

        let mut dirty_files = Vec::new();
        let mut untracked_files = Vec::new();

        if let Ok(statuses) = repo.statuses(Some(&mut status_opts)) {
            for entry in statuses.iter() {
                let Some(path) = entry.path() else {
                    continue;
                };
                if path.starts_with("target/") || path.starts_with(".git/") {
                    continue;
                }
                let st = entry.status();
                if st.is_wt_new() {
                    untracked_files.push(path.to_string());
                } else if st.is_wt_modified()
                    || st.is_wt_deleted()
                    || st.is_wt_renamed()
                    || st.is_index_modified()
                    || st.is_index_new()
                    || st.is_index_deleted()
                    || st.is_index_renamed()
                {
                    dirty_files.push(path.to_string());
                }
            }
        }
        dirty_files.sort();
        untracked_files.sort();

        let diff_summary = self.compute_diff_summary(&repo).unwrap_or_default();

        Ok(GitState {
            is_git_repo: true,
            branch,
            head_commit,
            origin_url,
            dirty_files,
            untracked_files,
            diff_summary,
        })
    }

    fn compute_diff_summary(&self, repo: &Repository) -> Option<String> {
        let mut opts = DiffOptions::new();
        opts.include_untracked(false).context_lines(1);
        let diff = repo.diff_index_to_workdir(None, Some(&mut opts)).ok()?;
        let stats = diff.stats().ok()?;
        if stats.files_changed() == 0 {
            return None;
        }

        let mut lines = Vec::new();
        lines.push(format!(
            "{} file(s) changed, +{} -{}",
            stats.files_changed(),
            stats.insertions(),
            stats.deletions()
        ));

        let mut patch_chars = 0usize;
        const MAX_PATCH_CHARS: usize = 1600;
        let _ = diff.print(git2::DiffFormat::Patch, |_delta, _hunk, line| {
            if patch_chars >= MAX_PATCH_CHARS {
                return false;
            }
            let origin = line.origin();
            if matches!(origin, '+' | '-' | 'H')
                && let Ok(text) = std::str::from_utf8(line.content())
            {
                let trimmed = text.trim_end();
                let formatted = if origin == 'H' {
                    trimmed.to_string()
                } else {
                    format!("{origin}{trimmed}")
                };
                patch_chars += formatted.len();
                lines.push(formatted);
            }
            true
        });

        Some(lines.join("\n"))
    }

    /// Normalize a path relative to the repository root using forward slashes.
    pub fn relative_path(&self, path: &Path) -> Option<String> {
        let canonical_root = canonicalize_normalized(&self.repo_root);
        strip_root_prefix(path, &self.repo_root, &canonical_root)
    }
}
