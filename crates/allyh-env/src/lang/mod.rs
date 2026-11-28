use std::collections::BTreeSet;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tree_sitter::Node;

use crate::error::Result;
use crate::graph::{FileGraph, SymbolInfo, SymbolKind};

#[cfg(feature = "lang-go")]
pub mod go;
#[cfg(feature = "lang-javascript")]
pub mod javascript;
#[cfg(feature = "lang-python")]
pub mod python;
#[cfg(feature = "lang-rust")]
pub mod rust;
#[cfg(feature = "lang-typescript")]
pub mod typescript;

/// Source language recognized by AllyH's polyglot RepoGraph indexer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceLanguage {
    Rust,
    TypeScript,
    Tsx,
    JavaScript,
    Jsx,
    Python,
    Go,
    Config,
    Unknown,
}

impl SourceLanguage {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::TypeScript => "typescript",
            Self::Tsx => "tsx",
            Self::JavaScript => "javascript",
            Self::Jsx => "jsx",
            Self::Python => "python",
            Self::Go => "go",
            Self::Config => "config",
            Self::Unknown => "unknown",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Rust => "Rust",
            Self::TypeScript | Self::Tsx => "TypeScript",
            Self::JavaScript | Self::Jsx => "JavaScript",
            Self::Python => "Python",
            Self::Go => "Go",
            Self::Config => "Config",
            Self::Unknown => "Unknown",
        }
    }

    pub fn is_code(&self) -> bool {
        !matches!(self, Self::Config | Self::Unknown)
    }

    pub fn from_path(rel_path: &str) -> Self {
        let lower = rel_path.trim().to_ascii_lowercase();
        let file_name = lower.rsplit('/').next().unwrap_or(lower.as_str());
        if matches!(
            file_name,
            "cargo.toml"
                | "package.json"
                | "tsconfig.json"
                | "pyproject.toml"
                | "setup.cfg"
                | "requirements.txt"
                | "go.mod"
                | "makefile"
        ) {
            return Self::Config;
        }
        if lower.ends_with(".rs") {
            Self::Rust
        } else if lower.ends_with(".tsx") {
            Self::Tsx
        } else if lower.ends_with(".ts") || lower.ends_with(".mts") || lower.ends_with(".cts") {
            Self::TypeScript
        } else if lower.ends_with(".jsx") {
            Self::Jsx
        } else if lower.ends_with(".js") || lower.ends_with(".mjs") || lower.ends_with(".cjs") {
            Self::JavaScript
        } else if lower.ends_with(".py") || lower.ends_with(".pyi") {
            Self::Python
        } else if lower.ends_with(".go") {
            Self::Go
        } else if lower.ends_with(".toml") {
            Self::Config
        } else {
            Self::Unknown
        }
    }
}

impl std::fmt::Display for SourceLanguage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Pluggable `tree-sitter` language indexer producing a [`FileGraph`] for RepoGraph.
pub trait LanguageIndexer: Send + Sync {
    fn language(&self) -> SourceLanguage;
    fn extensions(&self) -> &'static [&'static str];
    fn index_source(&self, source: &str) -> Result<FileGraph>;
}

/// Registry dispatching source files by extension to feature-enabled [`LanguageIndexer`]s.
#[derive(Clone, Default)]
pub struct LanguageRegistry {
    indexers: Vec<Arc<dyn LanguageIndexer>>,
}

impl LanguageRegistry {
    /// Construct a registry containing all compile-time enabled Tier-1 language indexers.
    pub fn default_all() -> Self {
        let mut reg = Self {
            indexers: Vec::new(),
        };
        #[cfg(feature = "lang-rust")]
        reg.register(Arc::new(rust::RustLangIndexer));
        #[cfg(feature = "lang-typescript")]
        {
            reg.register(Arc::new(typescript::TypeScriptLangIndexer));
            reg.register(Arc::new(typescript::TsxLangIndexer));
        }
        #[cfg(feature = "lang-javascript")]
        reg.register(Arc::new(javascript::JavaScriptLangIndexer));
        #[cfg(feature = "lang-python")]
        reg.register(Arc::new(python::PythonLangIndexer));
        #[cfg(feature = "lang-go")]
        reg.register(Arc::new(go::GoLangIndexer));
        reg
    }

    pub fn register(&mut self, indexer: Arc<dyn LanguageIndexer>) {
        self.indexers.push(indexer);
    }

    pub fn detect_language(rel_path: &str) -> SourceLanguage {
        SourceLanguage::from_path(rel_path)
    }

    /// Return all file extensions handled by registered code indexers.
    pub fn supported_extensions(&self) -> Vec<&'static str> {
        let mut exts = BTreeSet::new();
        for idx in &self.indexers {
            for ext in idx.extensions() {
                exts.insert(*ext);
            }
        }
        exts.into_iter().collect()
    }

    /// Return whether a file path should be tracked by AllyH (either indexed source or manifest).
    pub fn is_tracked_file(&self, rel_path: &str) -> bool {
        let lower = rel_path.to_ascii_lowercase();
        let file_name = lower.rsplit('/').next().unwrap_or(lower.as_str());
        if matches!(
            file_name,
            "cargo.toml"
                | "package.json"
                | "tsconfig.json"
                | "pyproject.toml"
                | "go.mod"
                | "requirements.txt"
        ) {
            return true;
        }
        let Some(ext) = lower.rsplit('.').next() else {
            return false;
        };
        if ext == "toml" {
            return true;
        }
        self.indexers
            .iter()
            .any(|idx| idx.extensions().iter().any(|e| e.eq_ignore_ascii_case(ext)))
    }

    /// Index a repository file by path, returning `(detected_language, FileGraph)`.
    pub fn index_path(&self, rel_path: &str, source: &str) -> Result<(SourceLanguage, FileGraph)> {
        let lang = SourceLanguage::from_path(rel_path);
        let lower = rel_path.to_ascii_lowercase();
        let ext = lower.rsplit('.').next().unwrap_or("");
        for indexer in &self.indexers {
            if indexer
                .extensions()
                .iter()
                .any(|e| e.eq_ignore_ascii_case(ext))
            {
                let graph = indexer.index_source(source)?;
                return Ok((indexer.language(), graph));
            }
        }
        Ok((lang, FileGraph::default()))
    }
}

pub(crate) fn field_text(node: Node<'_>, field: &str, source: &[u8]) -> Option<String> {
    node.child_by_field_name(field)?
        .utf8_text(source)
        .ok()
        .map(|s| s.trim().to_string())
}

pub(crate) fn extract_signature(node: Node<'_>, source: &[u8], body_delim: char) -> String {
    let Ok(text) = node.utf8_text(source) else {
        return String::new();
    };
    let head = text.split(body_delim).next().unwrap_or(text);
    collapse_signature(head)
}

#[cfg_attr(not(feature = "lang-python"), allow(dead_code))]
pub(crate) fn extract_first_line_signature(node: Node<'_>, source: &[u8]) -> String {
    let Ok(text) = node.utf8_text(source) else {
        return String::new();
    };
    let head = text.lines().next().unwrap_or(text);
    collapse_signature(head)
}

pub(crate) fn collapse_signature(raw: &str) -> String {
    let collapsed = raw
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if collapsed.len() > 140 {
        let cut = collapsed
            .char_indices()
            .take_while(|(idx, _)| *idx <= 137)
            .last()
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        format!("{}...", &collapsed[..cut])
    } else {
        collapsed
    }
}

pub(crate) fn build_symbol_with_refs(
    node: Node<'_>,
    kind: SymbolKind,
    name: String,
    signature: String,
    mut refs: BTreeSet<String>,
) -> SymbolInfo {
    let start_line = node.start_position().row + 1;
    let end_line = node.end_position().row + 1;
    refs.remove(&name);
    if let Some((_, short)) = name.rsplit_once("::") {
        refs.remove(short);
    }
    if let Some((_, short)) = name.rsplit_once('.') {
        refs.remove(short);
    }

    SymbolInfo {
        name,
        kind,
        start_line,
        end_line,
        signature,
        references: refs.into_iter().take(40).collect(),
    }
}

pub(crate) fn is_common_noise_ident(ident: &str) -> bool {
    matches!(
        ident,
        "new"
            | "default"
            | "clone"
            | "into"
            | "from"
            | "unwrap"
            | "expect"
            | "to_string"
            | "as_str"
            | "iter"
            | "into_iter"
            | "collect"
            | "push"
            | "len"
            | "is_empty"
            | "map"
            | "filter"
            | "ok"
            | "err"
            | "log"
            | "warn"
            | "error"
            | "info"
            | "debug"
            | "print"
            | "println"
            | "format"
            | "get"
            | "set"
            | "add"
            | "append"
            | "keys"
            | "values"
            | "items"
            | "str"
            | "int"
            | "float"
            | "bool"
            | "dict"
            | "list"
            | "tuple"
            | "range"
            | "make"
            | "close"
            | "delete"
            | "copy"
            | "cap"
            | "panic"
            | "recover"
            | "forEach"
            | "reduce"
            | "slice"
            | "join"
            | "split"
            | "trim"
            | "toString"
            | "valueOf"
            | "then"
            | "catch"
            | "finally"
    )
}
