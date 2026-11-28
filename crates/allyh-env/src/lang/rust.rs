use std::collections::BTreeSet;

use tree_sitter::{Node, Parser};

use crate::error::{EnvError, Result};
use crate::graph::{FileGraph, SymbolInfo, SymbolKind};
use crate::lang::{
    LanguageIndexer, SourceLanguage, build_symbol_with_refs, extract_signature, field_text,
    is_common_noise_ident,
};

/// `tree-sitter-rust` adapter extracting Rust items, `impl` methods, `use` deps, and references.
pub struct RustLangIndexer;

impl LanguageIndexer for RustLangIndexer {
    fn language(&self) -> SourceLanguage {
        SourceLanguage::Rust
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["rs"]
    }

    fn index_source(&self, source: &str) -> Result<FileGraph> {
        let mut parser = Parser::new();
        let language: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
        parser
            .set_language(&language)
            .map_err(|e| EnvError::TreeSitter(e.to_string()))?;

        let tree = parser
            .parse(source, None)
            .ok_or_else(|| EnvError::TreeSitter("failed to parse Rust source".to_string()))?;

        let bytes = source.as_bytes();
        let mut symbols = Vec::new();
        let mut deps = BTreeSet::new();

        walk_items(tree.root_node(), bytes, None, &mut symbols, &mut deps);

        Ok(FileGraph {
            symbols,
            dependencies: deps.into_iter().collect(),
        })
    }
}

fn walk_items(
    node: Node<'_>,
    source: &[u8],
    impl_prefix: Option<&str>,
    symbols: &mut Vec<SymbolInfo>,
    deps: &mut BTreeSet<String>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "use_declaration" => {
                if let Ok(text) = child.utf8_text(source) {
                    let cleaned = text
                        .trim()
                        .trim_start_matches("pub ")
                        .trim_start_matches("use ")
                        .trim_end_matches(';')
                        .trim()
                        .to_string();
                    if !cleaned.is_empty() {
                        deps.insert(cleaned);
                    }
                }
            }
            "function_item" => {
                if let Some(name) = field_text(child, "name", source) {
                    let full_name = match impl_prefix {
                        Some(prefix) => format!("{prefix}::{name}"),
                        None => name,
                    };
                    symbols.push(build_rust_symbol(child, SymbolKind::Function, full_name, source));
                }
            }
            "struct_item" => {
                if let Some(name) = field_text(child, "name", source) {
                    symbols.push(build_rust_symbol(child, SymbolKind::Struct, name, source));
                }
            }
            "enum_item" => {
                if let Some(name) = field_text(child, "name", source) {
                    symbols.push(build_rust_symbol(child, SymbolKind::Enum, name, source));
                }
            }
            "trait_item" => {
                if let Some(name) = field_text(child, "name", source) {
                    symbols.push(build_rust_symbol(child, SymbolKind::Trait, name.clone(), source));
                    if let Some(body) = child.child_by_field_name("body") {
                        walk_items(body, source, Some(&name), symbols, deps);
                    }
                }
            }
            "impl_item" => {
                let type_name = field_text(child, "type", source)
                    .map(|t| simplify_type_name(&t))
                    .unwrap_or_else(|| "impl".to_string());
                let trait_name = field_text(child, "trait", source).map(|t| simplify_type_name(&t));
                let impl_label = match trait_name {
                    Some(tr) => format!("{tr} for {type_name}"),
                    None => type_name.clone(),
                };
                symbols.push(build_rust_symbol(child, SymbolKind::Impl, impl_label, source));
                if let Some(body) = child.child_by_field_name("body") {
                    walk_items(body, source, Some(&type_name), symbols, deps);
                }
            }
            "mod_item" => {
                if let Some(name) = field_text(child, "name", source) {
                    symbols.push(build_rust_symbol(child, SymbolKind::Module, name.clone(), source));
                    if let Some(body) = child.child_by_field_name("body") {
                        walk_items(body, source, Some(&name), symbols, deps);
                    }
                }
            }
            "const_item" => {
                if let Some(name) = field_text(child, "name", source) {
                    symbols.push(build_rust_symbol(child, SymbolKind::Const, name, source));
                }
            }
            "static_item" => {
                if let Some(name) = field_text(child, "name", source) {
                    symbols.push(build_rust_symbol(child, SymbolKind::Static, name, source));
                }
            }
            "type_item" => {
                if let Some(name) = field_text(child, "name", source) {
                    symbols.push(build_rust_symbol(child, SymbolKind::TypeAlias, name, source));
                }
            }
            "macro_definition" => {
                if let Some(name) = field_text(child, "name", source) {
                    symbols.push(build_rust_symbol(child, SymbolKind::Macro, name, source));
                }
            }
            _ => {}
        }
    }
}

fn build_rust_symbol(
    node: Node<'_>,
    kind: SymbolKind,
    name: String,
    source: &[u8],
) -> SymbolInfo {
    let signature = extract_signature(node, source, '{');
    let mut refs = BTreeSet::new();
    collect_references(node, source, &mut refs);
    build_symbol_with_refs(node, kind, name, signature, refs)
}

fn collect_references(node: Node<'_>, source: &[u8], refs: &mut BTreeSet<String>) {
    match node.kind() {
        "type_identifier" => {
            if let Ok(text) = node.utf8_text(source)
                && !is_rust_builtin_type(text)
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
                if let Some(ident) = cleaned.rsplit("::").next().or(Some(cleaned)) {
                    let ident = ident.rsplit('.').next().unwrap_or(ident);
                    if ident.len() >= 3 && !is_common_noise_ident(ident) {
                        refs.insert(ident.to_string());
                    }
                }
                if cleaned.contains("::") && cleaned.len() <= 64 {
                    refs.insert(cleaned.to_string());
                }
            }
        }
        _ => {}
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_references(child, source, refs);
    }
}

fn simplify_type_name(raw: &str) -> String {
    raw.split('<')
        .next()
        .unwrap_or(raw)
        .trim()
        .to_string()
}

fn is_rust_builtin_type(t: &str) -> bool {
    matches!(
        t,
        "String"
            | "Vec"
            | "Option"
            | "Result"
            | "Box"
            | "Arc"
            | "Rc"
            | "Self"
            | "HashMap"
            | "HashSet"
            | "BTreeMap"
            | "BTreeSet"
            | "Path"
            | "PathBuf"
    )
}
