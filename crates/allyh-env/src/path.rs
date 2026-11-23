use std::path::{Component, Path, PathBuf};

/// Strip Windows extended-length verbatim path prefixes (`\\?\` and `\\?\UNC\`) so
/// paths remain portable across `git2`, `cargo`, `notify`, and `repo_id` hashing.
pub fn strip_verbatim_prefix(path: &Path) -> PathBuf {
    let raw = path.to_string_lossy();
    if let Some(unc) = raw
        .strip_prefix(r"\\?\UNC\")
        .or_else(|| raw.strip_prefix(r"\\?\unc\"))
    {
        return PathBuf::from(format!(r"\\{unc}"));
    }
    if let Some(stripped) = raw.strip_prefix(r"\\?\") {
        return PathBuf::from(stripped);
    }
    path.to_path_buf()
}

/// Canonicalize `path` when possible and remove Windows `\\?\` verbatim prefixes.
pub fn canonicalize_normalized(path: &Path) -> PathBuf {
    match std::fs::canonicalize(path) {
        Ok(canon) => strip_verbatim_prefix(&canon),
        Err(_) => strip_verbatim_prefix(path),
    }
}

/// Convert a path to a clean forward-slash string without Windows verbatim prefixes.
pub fn path_to_slash_string(path: &Path) -> String {
    strip_verbatim_prefix(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Returns `true` if `raw` begins with a Windows drive prefix such as `C:` or `d:`.
pub fn has_windows_drive_prefix(raw: &str) -> bool {
    let bytes = raw.as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

/// Strip `raw_root` or `canonical_root` from `path`, handling relative roots,
/// macOS `/var` -> `/private/var` symlinks, Windows `\\?\` prefixes, drive casing,
/// and removed files (whose parent directory still exists).
pub fn strip_root_prefix(path: &Path, raw_root: &Path, canonical_root: &Path) -> Option<String> {
    let normalized = strip_verbatim_prefix(path);

    if let Ok(rel) = normalized.strip_prefix(canonical_root) {
        return Some(format_relative(rel));
    }
    if let Ok(rel) = normalized.strip_prefix(raw_root) {
        return Some(format_relative(rel));
    }

    if let Ok(canon_path) = std::fs::canonicalize(&normalized) {
        let canon_clean = strip_verbatim_prefix(&canon_path);
        if let Ok(rel) = canon_clean.strip_prefix(canonical_root) {
            return Some(format_relative(rel));
        }
    } else if let (Some(parent), Some(name)) = (normalized.parent(), normalized.file_name())
        && let Ok(canon_parent) = std::fs::canonicalize(parent)
    {
        let rebuilt = strip_verbatim_prefix(&canon_parent).join(name);
        if let Ok(rel) = rebuilt.strip_prefix(canonical_root) {
            return Some(format_relative(rel));
        }
    }

    #[cfg(any(target_os = "windows", target_os = "macos"))]
    {
        let path_slash = path_to_slash_string(&normalized);
        for root in [canonical_root, raw_root] {
            let root_slash = path_to_slash_string(root);
            let root_trimmed = root_slash.trim_end_matches('/');
            if root_trimmed.is_empty() {
                continue;
            }
            if path_slash.len() > root_trimmed.len()
                && path_slash
                    .get(..root_trimmed.len())
                    .is_some_and(|prefix| prefix.eq_ignore_ascii_case(root_trimmed))
                && path_slash.as_bytes().get(root_trimmed.len()) == Some(&b'/')
            {
                return Some(
                    path_slash[root_trimmed.len() + 1..]
                        .trim_start_matches("./")
                        .to_string(),
                );
            }
        }
    }

    None
}

fn format_relative(rel: &Path) -> String {
    let mut parts = Vec::new();
    for comp in rel.components() {
        if let Component::Normal(seg) = comp {
            parts.push(seg.to_string_lossy().into_owned());
        }
    }
    parts.join("/")
}
