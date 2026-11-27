use std::path::Path;

use ignore::WalkBuilder;
use ignore::overrides::OverrideBuilder;
use regex::RegexBuilder;
use serde::{Deserialize, Serialize};

use crate::error::{EnvError, Result};

/// Maximum file size scanned by lexical/regex search (512 KiB) to avoid huge generated bundles.
const MAX_SEARCH_FILE_BYTES: u64 = 512 * 1024;

/// Parameters for fast `.gitignore`-aware repository text/regex search.
#[derive(Debug, Clone)]
pub struct CodeSearchQuery {
    pub pattern: String,
    pub is_regex: bool,
    pub case_sensitive: bool,
    pub path_prefix: Option<String>,
    pub file_glob: Option<String>,
    pub max_matches: usize,
}

/// Single line match returned by [`search_workspace_text`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeSearchMatch {
    pub file_path: String,
    pub line: usize,
    pub column: usize,
    pub snippet: String,
}

/// Aggregated result of a repository text/regex search.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TextSearchResult {
    pub matches: Vec<CodeSearchMatch>,
    pub files_scanned: usize,
    pub truncated: bool,
}

/// Execute a fast `.gitignore`-aware literal or regex search across `repo_root`.
pub fn search_workspace_text(
    repo_root: &Path,
    query: &CodeSearchQuery,
    extra_ignore_globs: &[String],
) -> Result<TextSearchResult> {
    let raw_pattern = query.pattern.trim();
    if raw_pattern.is_empty() {
        return Ok(TextSearchResult::default());
    }

    let regex_src = if query.is_regex {
        raw_pattern.to_string()
    } else {
        regex::escape(raw_pattern)
    };
    let matcher = RegexBuilder::new(&regex_src)
        .case_insensitive(!query.case_sensitive)
        .build()
        .map_err(|e| EnvError::Command(format!("invalid search regex `{raw_pattern}`: {e}")))?;

    let limit = query.max_matches.clamp(1, 200);
    let prefix_norm = query
        .path_prefix
        .as_deref()
        .map(|p| p.trim().replace('\\', "/").trim_matches('/').to_string())
        .filter(|p| !p.is_empty());

    let mut builder = WalkBuilder::new(repo_root);
    builder
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .ignore(true)
        .require_git(false)
        .filter_entry(|e| {
            e.depth() == 0 || !is_search_ignored_dir(e.file_name().to_str().unwrap_or(""))
        });

    let mut overrides = OverrideBuilder::new(repo_root);
    let mut has_overrides = false;
    if let Some(glob) = query.file_glob.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        let _ = overrides.add(glob);
        has_overrides = true;
    }
    for ig in extra_ignore_globs {
        let trimmed = ig.trim();
        if trimmed.is_empty() {
            continue;
        }
        let negated = if trimmed.starts_with('!') {
            trimmed.to_string()
        } else {
            format!("!{trimmed}")
        };
        let _ = overrides.add(&negated);
        has_overrides = true;
    }
    if has_overrides && let Ok(built) = overrides.build() {
        builder.overrides(built);
    }

    let mut matches = Vec::new();
    let mut files_scanned = 0usize;
    let mut truncated = false;

    for entry in builder.build().flatten() {
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }
        let path = entry.path();
        let Ok(rel) = path.strip_prefix(repo_root) else {
            continue;
        };
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if let Some(ref prefix) = prefix_norm
            && rel_str != *prefix
            && !rel_str.starts_with(&format!("{prefix}/"))
        {
            continue;
        }
        if entry
            .metadata()
            .ok()
            .is_some_and(|m| m.len() > MAX_SEARCH_FILE_BYTES)
        {
            continue;
        }
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        // Skip binary files containing NUL bytes.
        if bytes.iter().take(1024).any(|&b| b == 0) {
            continue;
        }
        let Ok(content) = String::from_utf8(bytes) else {
            continue;
        };
        files_scanned += 1;

        for (idx, line) in content.lines().enumerate() {
            if let Some(found) = matcher.find(line) {
                if matches.len() >= limit {
                    truncated = true;
                    break;
                }
                let col = line[..found.start()].chars().count() + 1;
                matches.push(CodeSearchMatch {
                    file_path: rel_str.clone(),
                    line: idx + 1,
                    column: col,
                    snippet: trim_snippet(line, 180),
                });
            }
        }
        if truncated {
            break;
        }
    }

    Ok(TextSearchResult {
        matches,
        files_scanned,
        truncated,
    })
}

fn trim_snippet(line: &str, max_chars: usize) -> String {
    let trimmed = line.trim();
    if trimmed.chars().count() <= max_chars {
        trimmed.to_string()
    } else {
        let taken: String = trimmed.chars().take(max_chars.saturating_sub(3)).collect();
        format!("{taken}...")
    }
}

fn is_search_ignored_dir(name: &str) -> bool {
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
