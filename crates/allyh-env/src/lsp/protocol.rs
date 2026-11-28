use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::path::strip_verbatim_prefix;
use crate::verify::CompilerDiagnostic;

/// Zero-based LSP position inside a text document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LspPosition {
    pub line: u32,
    pub character: u32,
}

/// Zero-based LSP range inside a text document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LspRange {
    pub start: LspPosition,
    pub end: LspPosition,
}

/// LSP location (`uri` + `range`) returned by `references`, `definition`, and `implementation`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LspLocation {
    pub uri: String,
    pub range: LspRange,
}

/// Call hierarchy item returned by `prepareCallHierarchy` / `incomingCalls` / `outgoingCalls`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LspCallHierarchyItem {
    pub name: String,
    #[serde(default)]
    pub kind: u32,
    pub uri: String,
    pub range: LspRange,
    pub selection_range: LspRange,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// Parsed LSP diagnostic (from `textDocument/publishDiagnostics` or `textDocument/diagnostic`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LspDiagnostic {
    pub range: LspRange,
    /// 1 = Error, 2 = Warning, 3 = Information, 4 = Hint.
    #[serde(default)]
    pub severity: Option<u32>,
    #[serde(default)]
    pub code: Option<Value>,
    #[serde(default)]
    pub source: Option<String>,
    pub message: String,
}

impl LspDiagnostic {
    /// Convert an LSP diagnostic into AllyH's unified [`CompilerDiagnostic`] (1-based line/col).
    pub fn to_compiler_diagnostic(&self, rel_file: Option<String>) -> Option<CompilerDiagnostic> {
        let level = match self.severity.unwrap_or(1) {
            1 => "error",
            2 => "warning",
            _ => return None,
        }
        .to_string();

        let code = self.code.as_ref().and_then(|v| match v {
            Value::String(s) => Some(s.clone()),
            Value::Number(n) => Some(n.to_string()),
            _ => None,
        });
        let suggestion = self
            .source
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(|src| format!("reported by LSP `{src}`"));

        Some(CompilerDiagnostic {
            level,
            code,
            message: self.message.clone(),
            file: rel_file,
            line: Some(self.range.start.line + 1),
            column: Some(self.range.start.character + 1),
            suggestion,
        })
    }
}

/// Convert a local filesystem path into a standards-compliant `file://` URI across Windows & Unix.
pub fn path_to_file_uri(path: &Path) -> String {
    let cleaned = strip_verbatim_prefix(path);
    let raw = cleaned.to_string_lossy().replace('\\', "/");
    let encoded = percent_encode_uri_path(&raw);
    if encoded.starts_with('/') {
        format!("file://{encoded}")
    } else {
        // Windows drive letter path `C:/...` -> `file:///C:/...`
        format!("file:///{encoded}")
    }
}

/// Convert a `file://` URI back into a normalized local [`PathBuf`].
pub fn file_uri_to_path(uri: &str) -> Option<PathBuf> {
    let rest = uri.strip_prefix("file://")?;
    let decoded = percent_decode_uri_path(rest);
    // On Windows `file:///C:/foo` has a leading slash before drive letter.
    let trimmed = if decoded.len() >= 3
        && decoded.starts_with('/')
        && decoded.as_bytes()[1].is_ascii_alphabetic()
        && decoded.as_bytes()[2] == b':'
    {
        &decoded[1..]
    } else {
        decoded.as_str()
    };
    Some(PathBuf::from(trimmed))
}

/// Convert a `file://` URI into a repo-relative forward-slash path if it resides inside `repo_root`.
pub fn file_uri_to_rel_path(uri: &str, repo_root: &Path) -> Option<String> {
    let path = file_uri_to_path(uri)?;
    let norm_root = strip_verbatim_prefix(repo_root)
        .to_string_lossy()
        .replace('\\', "/");
    let norm_path = strip_verbatim_prefix(&path)
        .to_string_lossy()
        .replace('\\', "/");

    let root_prefix = format!("{}/", norm_root.trim_end_matches('/'));
    if cfg!(windows) {
        if norm_path
            .to_ascii_lowercase()
            .starts_with(&root_prefix.to_ascii_lowercase())
        {
            return Some(norm_path[root_prefix.len()..].to_string());
        }
    } else if let Some(rel) = norm_path.strip_prefix(&root_prefix) {
        return Some(rel.to_string());
    }

    // Try canonical paths as fallback (handles symlinks / 8.3 short paths on Windows).
    if let (Ok(c_root), Ok(c_path)) = (std::fs::canonicalize(repo_root), std::fs::canonicalize(&path))
        && let Ok(rel) = strip_verbatim_prefix(&c_path).strip_prefix(strip_verbatim_prefix(&c_root))
    {
        return Some(rel.to_string_lossy().replace('\\', "/"));
    }
    None
}

/// Map a file extension or [`crate::lang::SourceLanguage`] string into an LSP `languageId`.
pub fn lsp_language_id_for_path(rel_path: &str) -> Option<&'static str> {
    let lower = rel_path.to_ascii_lowercase();
    if lower.ends_with(".rs") {
        Some("rust")
    } else if lower.ends_with(".tsx") {
        Some("typescriptreact")
    } else if lower.ends_with(".ts") || lower.ends_with(".mts") || lower.ends_with(".cts") {
        Some("typescript")
    } else if lower.ends_with(".jsx") {
        Some("javascriptreact")
    } else if lower.ends_with(".js") || lower.ends_with(".mjs") || lower.ends_with(".cjs") {
        Some("javascript")
    } else if lower.ends_with(".py") || lower.ends_with(".pyi") {
        Some("python")
    } else if lower.ends_with(".go") {
        Some("go")
    } else if lower.ends_with(".c") || lower.ends_with(".h") {
        Some("c")
    } else if lower.ends_with(".cpp")
        || lower.ends_with(".cc")
        || lower.ends_with(".cxx")
        || lower.ends_with(".hpp")
    {
        Some("cpp")
    } else {
        None
    }
}

/// Normalize an LSP `languageId` into a server-pool key (so `.ts`/`.tsx`/`.js`/`.jsx` share one TS server).
pub fn server_family_for_language_id(language_id: &str) -> Option<&'static str> {
    match language_id.trim().to_ascii_lowercase().as_str() {
        "rust" | "rs" => Some("rust"),
        "typescript" | "typescriptreact" | "tsx" | "ts" | "javascript" | "javascriptreact"
        | "jsx" | "js" => Some("typescript"),
        "python" | "py" => Some("python"),
        "go" | "golang" => Some("go"),
        "c" | "cpp" | "c++" => Some("cpp"),
        _ => None,
    }
}

/// Parse an LSP `Location | Location[] | LocationLink[] | null` response payload.
pub fn parse_locations_value(value: &Value) -> Vec<LspLocation> {
    if value.is_null() {
        return Vec::new();
    }
    if let Ok(single) = serde_json::from_value::<LspLocation>(value.clone()) {
        return vec![single];
    }
    let Some(arr) = value.as_array() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for item in arr {
        if let Ok(loc) = serde_json::from_value::<LspLocation>(item.clone()) {
            out.push(loc);
            continue;
        }
        // LocationLink format (`targetUri` + `targetSelectionRange` / `targetRange`).
        if let Some(uri) = item.get("targetUri").and_then(|v| v.as_str()) {
            let range_val = item
                .get("targetSelectionRange")
                .or_else(|| item.get("targetRange"));
            if let Some(rv) = range_val
                && let Ok(range) = serde_json::from_value::<LspRange>(rv.clone())
            {
                out.push(LspLocation {
                    uri: uri.to_string(),
                    range,
                });
            }
        }
    }
    out
}

/// Encode a JSON-RPC 2.0 LSP message with standard `Content-Length` header framing.
pub fn encode_lsp_frame(payload: &Value) -> Vec<u8> {
    let body = serde_json::to_string(payload).unwrap_or_else(|_| "{}".to_string());
    format!("Content-Length: {}\r\n\r\n{}", body.len(), body).into_bytes()
}

fn percent_encode_uri_path(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for byte in path.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' | b':' => {
                out.push(byte as char);
            }
            other => {
                out.push_str(&format!("%{:02X}", other));
            }
        }
    }
    out
}

fn percent_decode_uri_path(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let Ok(hex_str) = std::str::from_utf8(&bytes[i + 1..i + 3])
            && let Ok(val) = u8::from_str_radix(hex_str, 16)
        {
            out.push(val);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}
