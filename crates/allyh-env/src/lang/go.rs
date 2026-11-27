use std::collections::BTreeSet;

use tree_sitter::{Node, Parser};

use crate::error::{EnvError, Result};
use crate::graph::{FileGraph, SymbolInfo, SymbolKind};
use crate::lang::{
    LanguageIndexer, SourceLanguage, build_symbol_with_refs, extract_signature, field_text,
    is_common_noise_ident,
};

/// `tree-sitter-go` adapter for `.go` source files.
pub struct GoLangIndexer;

impl LanguageIndexer for GoLangIndexer {
    fn language(&self) -> SourceLanguage {
        SourceLanguage::Go
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["go"]
    }

    fn index_source(&self, source: &str) -> Result<FileGraph> {
        let mut parser = Parser::new();
        let language: tree_sitter::Language = tree_sitter_go::LANGUAGE.into();
        parser
            .set_language(&language)
            .map_err(|e| EnvError::TreeSitter(e.to_string()))?;

        let tree = parser
            .parse(source, None)
            .ok_or_else(|| EnvError::TreeSitter("failed to parse Go source".to_string()))?;

        let bytes = source.as_bytes();
        let mut symbols = Vec::new();
        let mut deps = BTreeSet::new();

        walk_go_items(tree.root_node(), bytes, &mut symbols, &mut deps);

        Ok(FileGraph {
            symbols,
            dependencies: deps.into_iter().collect(),
        })
    }
}

fn walk_go_items(
    node: Node<'_>,
    source: &[u8],
    symbols: &mut Vec<SymbolInfo>,
    deps: &mut BTreeSet<String>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "import_declaration" => {
                collect_go_imports(child, source, deps);
            }
            "function_declaration" => {
                if let Some(name) = field_text(child, "name", source) {
                    symbols.push(build_go_symbol(child, SymbolKind::Function, name, source));
                }
            }
            "method_declaration" => {
                if let Some(name) = field_text(child, "name", source) {
                    let receiver = child
                        .child_by_field_name("receiver")
                        .and_then(|r| extract_go_receiver_type(r, source));
                    let full_name = match receiver {
                        Some(recv) => format!("{recv}.{name}"),
                        None => name,
                    };
                    symbols.push(build_go_symbol(child, SymbolKind::Method, full_name, source));
                }
            }
            "type_declaration" => {
                let mut spec_cursor = child.walk();
                for spec in child.children(&mut spec_cursor) {
                    if spec.kind() == "type_spec"
                        && let Some(name) = field_text(spec, "name", source)
                    {
                        let kind = match spec.child_by_field_name("type").map(|t| t.kind()) {
                            Some("struct_type") => SymbolKind::Struct,
                            Some("interface_type") => SymbolKind::Interface,
                            _ => SymbolKind::TypeAlias,
                        };
                        symbols.push(build_go_symbol(spec, kind, name, source));
                    }
                }
            }
            "const_declaration" => {
                let mut spec_cursor = child.walk();
                for spec in child.children(&mut spec_cursor) {
                    if spec.kind() == "const_spec"
                        && let Some(name) = field_text(spec, "name", source)
                    {
                        symbols.push(build_go_symbol(spec, SymbolKind::Const, name, source));
                    }
                }
            }
            "var_declaration" => {
                let mut spec_cursor = child.walk();
                for spec in child.children(&mut spec_cursor) {
                    if spec.kind() == "var_spec"
                        && let Some(name) = field_text(spec, "name", source)
                    {
                        symbols.push(build_go_symbol(spec, SymbolKind::Variable, name, source));
                    }
                }
            }
            _ => {}
        }
    }
}

fn collect_go_imports(node: Node<'_>, source: &[u8], deps: &mut BTreeSet<String>) {
    if node.kind() == "import_spec"
        && let Some(path) = field_text(node, "path", source)
    {
        let cleaned = path.trim_matches('"').trim().to_string();
        if !cleaned.is_empty() {
            deps.insert(cleaned);
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_go_imports(child, source, deps);
    }
}

fn extract_go_receiver_type(receiver: Node<'_>, source: &[u8]) -> Option<String> {
    let text = receiver.utf8_text(source).ok()?;
    // Receiver looks like `(s *Store)` or `(*Store)` -> extract `Store`.
    let inner = text.trim().trim_start_matches('(').trim_end_matches(')');
    let last_token = inner.split_whitespace().last()?;
    let cleaned = last_token.trim_start_matches('*');
    let base = cleaned.split('[').next().unwrap_or(cleaned).trim();
    (!base.is_empty()).then(|| base.to_string())
}

fn build_go_symbol(
    node: Node<'_>,
    kind: SymbolKind,
    name: String,
    source: &[u8],
) -> SymbolInfo {
    let signature = extract_signature(node, source, '{');
    let mut refs = BTreeSet::new();
    collect_go_references(node, source, &mut refs);
    build_symbol_with_refs(node, kind, name, signature, refs)
}

fn collect_go_references(node: Node<'_>, source: &[u8], refs: &mut BTreeSet<String>) {
    match node.kind() {
        "type_identifier" => {
            if let Ok(text) = node.utf8_text(source)
                && !is_go_builtin_type(text)
                && text.len() >= 2
            {
                refs.insert(text.to_string());
            }
        }
        "call_expression" => {
            if let Some(func) = node.child_by_field_name("function")
                && let Ok(text) = func.utf8_text(source)
            {
                let cleaned = text.trim();
                let short = cleaned.rsplit('.').next().unwrap_or(cleaned);
                if short.len() >= 3 && !is_common_noise_ident(short) {
                    refs.insert(short.to_string());
                }
                if cleaned.contains('.') && cleaned.len() <= 64 {
                    refs.insert(cleaned.to_string());
                }
            }
        }
        _ => {}
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_go_references(child, source, refs);
    }
}

fn is_go_builtin_type(t: &str) -> bool {
    matches!(
        t,
        "string"
            | "bool"
            | "byte"
            | "rune"
            | "error"
            | "any"
            | "int"
            | "int8"
            | "int16"
            | "int32"
            | "int64"
            | "uint"
            | "uint8"
            | "uint16"
            | "uint32"
            | "uint64"
            | "uintptr"
            | "float32"
            | "float64"
            | "complex64"
            | "complex128"
    )
}
