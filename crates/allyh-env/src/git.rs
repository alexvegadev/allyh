use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use git2::{BlameOptions, DiffOptions, Repository, Sort, StatusOptions};
use serde::{Deserialize, Serialize};

use crate::error::{EnvError, Result};
use crate::path::{
    canonicalize_normalized, path_to_slash_string, strip_root_prefix, strip_verbatim_prefix,
};

/// Single commit entry returned by [`GitTracker::log`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitLogEntry {
    pub commit: String,
    pub short_commit: String,
    pub author: String,
    pub timestamp: String,
    pub summary: String,
}

/// Structured working-tree / staged diff report returned by [`GitTracker::diff`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GitDiffReport {
    pub files_changed: usize,
    pub insertions: usize,
    pub deletions: usize,
    pub patch: String,
    pub truncated: bool,
}

/// Single line attribution returned by [`GitTracker::blame`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitBlameLine {
    pub line: usize,
    pub commit: String,
    pub author: String,
    pub content: String,
}

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

    /// Read recent Git commits (`git log`), optionally filtered to commits touching `path_filter`.
    pub fn log(&self, limit: usize, path_filter: Option<&str>) -> Result<Vec<GitLogEntry>> {
        let repo = Repository::discover(&self.repo_root)
            .map_err(|e| EnvError::Command(format!("not a git repository: {e}")))?;
        let mut revwalk = repo
            .revwalk()
            .map_err(|e| EnvError::Command(format!("git revwalk failed: {e}")))?;
        revwalk.set_sorting(Sort::TIME).ok();
        if revwalk.push_head().is_err() {
            return Ok(Vec::new());
        }

        let norm_filter = path_filter
            .map(|p| p.trim().replace('\\', "/").trim_matches('/').to_string())
            .filter(|p| !p.is_empty());

        let max_entries = limit.clamp(1, 50);
        let mut out = Vec::new();
        for oid_res in revwalk {
            let Ok(oid) = oid_res else { continue };
            let Ok(commit) = repo.find_commit(oid) else {
                continue;
            };
            if let Some(ref filter) = norm_filter
                && !commit_touches_path(&repo, &commit, filter)
            {
                continue;
            }
            let full_id = oid.to_string();
            let short_commit: String = full_id.chars().take(8).collect();
            let author = commit.author().name().unwrap_or("unknown").to_string();
            let secs = commit.time().seconds();
            let timestamp = DateTime::<Utc>::from_timestamp(secs, 0)
                .map(|dt| dt.format("%Y-%m-%d %H:%M UTC").to_string())
                .unwrap_or_else(|| secs.to_string());
            let summary = commit.summary().unwrap_or("").to_string();

            out.push(GitLogEntry {
                commit: full_id,
                short_commit,
                author,
                timestamp,
                summary,
            });
            if out.len() >= max_entries {
                break;
            }
        }
        Ok(out)
    }

    /// Compute structured unified diff for unstaged (`staged = false`) or staged (`staged = true`) changes.
    pub fn diff(
        &self,
        staged: bool,
        path_filter: Option<&str>,
        max_chars: usize,
    ) -> Result<GitDiffReport> {
        let repo = Repository::discover(&self.repo_root)
            .map_err(|e| EnvError::Command(format!("not a git repository: {e}")))?;
        let mut opts = DiffOptions::new();
        opts.context_lines(2);
        if let Some(p) = path_filter.map(str::trim).filter(|s| !s.is_empty()) {
            opts.pathspec(p.replace('\\', "/"));
        }

        let diff = if staged {
            let head_tree = repo
                .head()
                .ok()
                .and_then(|h| h.peel_to_tree().ok());
            repo.diff_tree_to_index(head_tree.as_ref(), None, Some(&mut opts))
        } else {
            repo.diff_index_to_workdir(None, Some(&mut opts))
        }
        .map_err(|e| EnvError::Command(format!("git diff failed: {e}")))?;

        let stats = diff
            .stats()
            .map_err(|e| EnvError::Command(format!("git diff stats failed: {e}")))?;
        let cap = max_chars.clamp(256, 32_000);
        let mut lines = Vec::new();
        let mut chars = 0usize;
        let mut truncated = false;

        let _ = diff.print(git2::DiffFormat::Patch, |_delta, _hunk, line| {
            if chars >= cap {
                truncated = true;
                return false;
            }
            if let Ok(text) = std::str::from_utf8(line.content()) {
                let origin = line.origin();
                let formatted = if matches!(origin, '+' | '-' | ' ') {
                    format!("{origin}{}", text.trim_end_matches(['\r', '\n']))
                } else {
                    text.trim_end_matches(['\r', '\n']).to_string()
                };
                chars += formatted.len() + 1;
                lines.push(formatted);
            }
            true
        });

        Ok(GitDiffReport {
            files_changed: stats.files_changed(),
            insertions: stats.insertions(),
            deletions: stats.deletions(),
            patch: lines.join("\n"),
            truncated,
        })
    }

    /// Read structured line-by-line Git blame attribution for `rel_path` in `[start_line..=end_line]`.
    pub fn blame(
        &self,
        rel_path: &str,
        start_line: usize,
        end_line: usize,
    ) -> Result<Vec<GitBlameLine>> {
        let repo = Repository::discover(&self.repo_root)
            .map_err(|e| EnvError::Command(format!("not a git repository: {e}")))?;
        let norm_rel = rel_path.trim().replace('\\', "/").trim_start_matches("./").to_string();
        let abs = self.repo_root.join(&norm_rel);
        let content = std::fs::read_to_string(&abs)?;
        let file_lines: Vec<&str> = content.lines().collect();
        if file_lines.is_empty() {
            return Ok(Vec::new());
        }

        let s = start_line.max(1).min(file_lines.len());
        let e = end_line.max(s).min(file_lines.len()).min(s + 120);

        let mut opts = BlameOptions::new();
        opts.min_line(s).max_line(e);
        let blame = repo
            .blame_file(Path::new(&norm_rel), Some(&mut opts))
            .map_err(|err| EnvError::Command(format!("git blame failed for `{norm_rel}`: {err}")))?;

        let mut out = Vec::new();
        for line_no in s..=e {
            let (commit, author) = match blame.get_line(line_no) {
                Some(hunk) => {
                    let cid: String = hunk.final_commit_id().to_string().chars().take(8).collect();
                    let auth = hunk
                        .final_signature()
                        .name()
                        .unwrap_or("uncommitted")
                        .to_string();
                    (cid, auth)
                }
                None => ("00000000".to_string(), "uncommitted".to_string()),
            };
            out.push(GitBlameLine {
                line: line_no,
                commit,
                author,
                content: file_lines[line_no - 1].to_string(),
            });
        }
        Ok(out)
    }
}

fn commit_touches_path(repo: &Repository, commit: &git2::Commit<'_>, path_prefix: &str) -> bool {
    let Ok(tree) = commit.tree() else {
        return false;
    };
    let parent_tree = commit.parent(0).ok().and_then(|p| p.tree().ok());
    let mut opts = DiffOptions::new();
    opts.pathspec(path_prefix);
    repo.diff_tree_to_tree(parent_tree.as_ref(), Some(&tree), Some(&mut opts))
        .ok()
        .is_some_and(|d| d.deltas().len() > 0)
}
