use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use chrono::{DateTime, Utc};
use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};
use similar::TextDiff;
use uuid::Uuid;

use crate::error::Result;
use crate::git::GitTracker;

/// Maximum single file size snapshotted into an atomic checkpoint (2 MiB).
const MAX_CHECKPOINT_FILE_BYTES: u64 = 2 * 1024 * 1024;

/// Out-of-tree pre-execution workspace checkpoint captured when a plan is approved.
///
/// Captures the exact bytes of all non-ignored workspace files without writing any
/// dotfiles, branches, or stashes into the target repository.
#[derive(Debug, Clone)]
pub struct PlanCheckpoint {
    pub id: String,
    pub label: String,
    pub created_at: DateTime<Utc>,
    pub head_commit: Option<String>,
    /// Repository-relative path -> original UTF-8/binary file bytes at checkpoint time.
    pub files: BTreeMap<String, Vec<u8>>,
}

/// Summary & unified diff computed between a [`PlanCheckpoint`] and the live workspace.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CheckpointDiffReport {
    pub checkpoint_id: String,
    pub checkpoint_label: String,
    pub modified_files: Vec<String>,
    pub created_files: Vec<String>,
    pub deleted_files: Vec<String>,
    pub insertions: usize,
    pub deletions: usize,
    pub unified_diff: String,
}

impl CheckpointDiffReport {
    pub fn has_changes(&self) -> bool {
        !self.modified_files.is_empty()
            || !self.created_files.is_empty()
            || !self.deleted_files.is_empty()
    }

    pub fn compact_summary(&self) -> String {
        format!(
            "{} modified, {} created, {} deleted (+{} -{})",
            self.modified_files.len(),
            self.created_files.len(),
            self.deleted_files.len(),
            self.insertions,
            self.deletions
        )
    }
}

/// Outcome of rolling back a workspace to a [`PlanCheckpoint`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RollbackReport {
    pub checkpoint_id: String,
    pub checkpoint_label: String,
    pub restored_files: Vec<String>,
    pub removed_files: Vec<String>,
}

impl PlanCheckpoint {
    /// Capture an out-of-tree checkpoint of all non-ignored files in `repo_root`.
    pub fn capture(repo_root: &Path, label: impl Into<String>) -> Result<Self> {
        let head_commit = GitTracker::new(repo_root)
            .inspect()
            .ok()
            .and_then(|g| g.head_commit);
        let files = snapshot_workspace_files(repo_root);
        Ok(Self {
            id: Uuid::new_v4().to_string(),
            label: label.into(),
            created_at: Utc::now(),
            head_commit,
            files,
        })
    }

    /// Compute a unified diff report between this checkpoint and the current files on disk.
    pub fn diff_against(&self, repo_root: &Path, max_diff_chars: usize) -> CheckpointDiffReport {
        let current = snapshot_workspace_files(repo_root);
        let all_paths: BTreeSet<&String> = self.files.keys().chain(current.keys()).collect();

        let mut modified_files = Vec::new();
        let mut created_files = Vec::new();
        let mut deleted_files = Vec::new();
        let mut insertions = 0usize;
        let mut deletions = 0usize;
        let mut diff_chunks = Vec::new();
        let mut total_chars = 0usize;

        for path in all_paths {
            match (self.files.get(path), current.get(path)) {
                (Some(old_bytes), Some(new_bytes)) if old_bytes != new_bytes => {
                    modified_files.push(path.clone());
                    let old_s = String::from_utf8_lossy(old_bytes);
                    let new_s = String::from_utf8_lossy(new_bytes);
                    let (ins, del, patch) = build_unified_patch(path, &old_s, &new_s);
                    insertions += ins;
                    deletions += del;
                    append_capped(&mut diff_chunks, &mut total_chars, max_diff_chars, patch);
                }
                (None, Some(new_bytes)) => {
                    created_files.push(path.clone());
                    let new_s = String::from_utf8_lossy(new_bytes);
                    let (ins, del, patch) = build_unified_patch(path, "", &new_s);
                    insertions += ins;
                    deletions += del;
                    append_capped(&mut diff_chunks, &mut total_chars, max_diff_chars, patch);
                }
                (Some(old_bytes), None) => {
                    deleted_files.push(path.clone());
                    let old_s = String::from_utf8_lossy(old_bytes);
                    let (ins, del, patch) = build_unified_patch(path, &old_s, "");
                    insertions += ins;
                    deletions += del;
                    append_capped(&mut diff_chunks, &mut total_chars, max_diff_chars, patch);
                }
                _ => {}
            }
        }

        CheckpointDiffReport {
            checkpoint_id: self.id.clone(),
            checkpoint_label: self.label.clone(),
            modified_files,
            created_files,
            deleted_files,
            insertions,
            deletions,
            unified_diff: diff_chunks.join("\n"),
        }
    }

    /// Atomically restore `repo_root` to the exact file state captured in this checkpoint.
    pub fn rollback(&self, repo_root: &Path) -> Result<RollbackReport> {
        let current = snapshot_workspace_files(repo_root);
        let mut restored_files = Vec::new();
        let mut removed_files = Vec::new();

        // 1. Restore modified or deleted files.
        for (rel_path, orig_bytes) in &self.files {
            let needs_restore = current
                .get(rel_path)
                .is_none_or(|cur_bytes| cur_bytes != orig_bytes);
            if needs_restore {
                let abs = repo_root.join(rel_path);
                if let Some(parent) = abs.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&abs, orig_bytes)?;
                restored_files.push(rel_path.clone());
            }
        }

        // 2. Delete files created after the checkpoint was captured.
        for rel_path in current.keys() {
            if !self.files.contains_key(rel_path) {
                let abs = repo_root.join(rel_path);
                if abs.is_file() {
                    std::fs::remove_file(&abs)?;
                    removed_files.push(rel_path.clone());
                }
            }
        }

        Ok(RollbackReport {
            checkpoint_id: self.id.clone(),
            checkpoint_label: self.label.clone(),
            restored_files,
            removed_files,
        })
    }
}

fn snapshot_workspace_files(repo_root: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut files = BTreeMap::new();
    let walker = WalkBuilder::new(repo_root)
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .ignore(true)
        .require_git(false)
        .filter_entry(|e| {
            e.depth() == 0 || !is_checkpoint_ignored_dir(e.file_name().to_str().unwrap_or(""))
        })
        .build();

    for entry in walker.flatten() {
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }
        if entry
            .metadata()
            .ok()
            .is_some_and(|m| m.len() > MAX_CHECKPOINT_FILE_BYTES)
        {
            continue;
        }
        let path = entry.path();
        if let Ok(rel) = path.strip_prefix(repo_root)
            && let Ok(bytes) = std::fs::read(path)
        {
            let rel_str = rel.to_string_lossy().replace('\\', "/");
            files.insert(rel_str, bytes);
        }
    }
    files
}

fn build_unified_patch(rel_path: &str, old: &str, new: &str) -> (usize, usize, String) {
    let diff = TextDiff::from_lines(old, new);
    let mut ins = 0usize;
    let mut del = 0usize;
    for change in diff.iter_all_changes() {
        match change.tag() {
            similar::ChangeTag::Insert => ins += 1,
            similar::ChangeTag::Delete => del += 1,
            similar::ChangeTag::Equal => {}
        }
    }
    let patch = diff
        .unified_diff()
        .context_radius(2)
        .header(&format!("a/{rel_path}"), &format!("b/{rel_path}"))
        .to_string();
    (ins, del, patch)
}

fn append_capped(chunks: &mut Vec<String>, total: &mut usize, max_chars: usize, patch: String) {
    if *total >= max_chars {
        return;
    }
    let remaining = max_chars - *total;
    if patch.len() <= remaining {
        *total += patch.len();
        chunks.push(patch);
    } else {
        let cut = patch
            .char_indices()
            .take_while(|(i, _)| *i < remaining)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        *total = max_chars;
        chunks.push(format!("{}\n... [diff truncated] ...", &patch[..cut]));
    }
}

fn is_checkpoint_ignored_dir(name: &str) -> bool {
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
            | "dist"
            | "build"
    )
}
