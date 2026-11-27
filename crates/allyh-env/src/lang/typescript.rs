use std::collections::BTreeSet;

use tree_sitter::{Node, Parser};

use crate::error::{EnvError, Result};
use crate::graph::{FileGraph, SymbolInfo, SymbolKind};
use crate::lang::{
    LanguageIndexer, SourceLanguage, build_symbol_with_refs, extract_signature, field_text,
    is_common_noise_ident,
};

/// `tree-sitter-typescript` adapter for `.ts`, `.mts`, `.cts`.
pub struct TypeScriptLangIndexer;

/// `tree-sitter-typescript` TSX adapter for `.tsx`.
pub struct TsxLangIndexer;

impl LanguageIndexer for TypeScriptLangIndexer {
    fn language(&self) -> SourceLanguage {
        SourceLanguage::TypeScript
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["ts", "mts", "cts"]
    }

    fn index_source(&self, source: &str) -> Result<FileGraph> {
        index_ts_like(
            source,
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            false,
        )
    }
}

impl LanguageIndexer for TsxLangIndexer {
    fn language(&self) -> SourceLanguage {
        SourceLanguage::Tsx
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["tsx"]
    }

    fn index_source(&self, source: &str) -> Result<FileGraph> {
        index_ts_like(source, tree_sitter_typescript::LANGUAGE_TSX.into(), true)
    }
}

pub(crate) fn index_ts_like(
    source: &str,
    language: tree_sitter::Language,
    allow_components: bool,
) -> Result<FileGraph> {
    let mut parser = Parser::new();
    parser
        .set_language(&language)
        .map_err(|e| EnvError::TreeSitter(e.to_string()))?;

    let tree = parser
        .parse(source, None)
        .ok_or_else(|| EnvError::TreeSitter("failed to parse TypeScript/JS source".to_string()))?;

    let bytes = source.as_bytes();
    let mut symbols = Vec::new();
    let mut deps = BTreeSet::new();

    walk_ts_items(
        tree.root_node(),
        bytes,
        None,
        allow_components,
        &mut symbols,
        &mut deps,
    );

    Ok(FileGraph {
        symbols,
        dependencies: deps.into_iter().collect(),
    })
}

fn walk_ts_items(
    node: Node<'_>,
    source: &[u8],
    container: Option<&str>,
    allow_components: bool,
    symbols: &mut Vec<SymbolInfo>,
    deps: &mut BTreeSet<String>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "import_statement" => {
                if let Some(src) = field_text(child, "source", source) {
                    let cleaned = src.trim_matches(|c| c == '\'' || c == '"').trim().to_string();
                    if !cleaned.is_empty() {
                        deps.insert(cleaned);
                    }
                }
            }
            "export_statement" => {
                if let Some(src) = field_text(child, "source", source) {
                    let cleaned = src.trim_matches(|c| c == '\'' || c == '"').trim().to_string();
                    if !cleaned.is_empty() {
                        deps.insert(cleaned);
                    }
                }
                walk_ts_items(child, source, container, allow_components, symbols, deps);
            }
            "function_declaration" | "generator_function_declaration" => {
                if let Some(name) = field_text(child, "name", source) {
                    let kind = if allow_components && is_pascal_case(&name) && contains_jsx(child) {
                        SymbolKind::Component
                    } else {
                        SymbolKind::Function
                    };
                    let full_name = qualify(container, &name);
                    symbols.push(build_ts_symbol(child, kind, full_name, source));
                }
            }
            "class_declaration" | "abstract_class_declaration" => {
                if let Some(name) = field_text(child, "name", source) {
                    let full_name = qualify(container, &name);
                    symbols.push(build_ts_symbol(child, SymbolKind::Class, full_name.clone(), source));
                    if let Some(body) = child.child_by_field_name("body") {
                        walk_ts_items(body, source, Some(&full_name), allow_components, symbols, deps);
                    }
                }
            }
            "interface_declaration" => {
                if let Some(name) = field_text(child, "name", source) {
                    let full_name = qualify(container, &name);
                    symbols.push(build_ts_symbol(child, SymbolKind::Interface, full_name, source));
                }
            }
            "type_alias_declaration" => {
                if let Some(name) = field_text(child, "name", source) {
                    let full_name = qualify(container, &name);
                    symbols.push(build_ts_symbol(child, SymbolKind::TypeAlias, full_name, source));
                }
            }
            "enum_declaration" => {
                if let Some(name) = field_text(child, "name", source) {
                    let full_name = qualify(container, &name);
                    symbols.push(build_ts_symbol(child, SymbolKind::Enum, full_name, source));
                }
            }
            "internal_module" | "module" => {
                if let Some(name) = field_text(child, "name", source) {
                    let full_name = qualify(container, &name);
                    symbols.push(build_ts_symbol(
                        child,
                        SymbolKind::Namespace,
                        full_name.clone(),
                        source,
                    ));
                    if let Some(body) = child.child_by_field_name("body") {
                        walk_ts_items(body, source, Some(&full_name), allow_components, symbols, deps);
                    }
                }
            }
            "method_definition" | "public_field_definition" => {
                if let Some(name) = field_text(child, "name", source)
                    && name != "constructor"
                {
                    let is_method = child.kind() == "method_definition"
                        || child
                            .child_by_field_name("value")
                            .is_some_and(|v| matches!(v.kind(), "arrow_function" | "function_expression"));
                    if is_method {
                        let full_name = qualify(container, &name);
                        symbols.push(build_ts_symbol(child, SymbolKind::Method, full_name, source));
                    }
                }
            }
            "lexical_declaration" | "variable_declaration" => {
                let mut decl_cursor = child.walk();
                for decl in child.children(&mut decl_cursor) {
                    if decl.kind() != "variable_declarator" {
                        continue;
                    }
                    let Some(name) = field_text(decl, "name", source) else {
                        continue;
                    };
                    if name.starts_with('{') || name.starts_with('[') {
                        continue;
                    }
                    if let Some(value) = decl.child_by_field_name("value") {
                        // Track CommonJS require('...') dependencies
                        if value.kind() == "call_expression"
                            && let Some(req) = extract_require_dep(value, source)
                        {
                            deps.insert(req);
                            continue;
                        }
                        let kind = match value.kind() {
                            "arrow_function" | "function_expression" | "function" => {
                                if allow_components && is_pascal_case(&name) && contains_jsx(value) {
                                    SymbolKind::Component
                                } else {
                                    SymbolKind::Function
                                }
                            }
                            "class" => SymbolKind::Class,
                            _ => SymbolKind::Const,
                        };
                        let full_name = qualify(container, &name);
                        symbols.push(build_ts_symbol(decl, kind, full_name, source));
                    }
                }
            }
            "expression_statement" => {
                // Capture top-level require calls if any.
                if let Some(expr) = child.child(0)
                    && let Some(req) = extract_require_dep(expr, source)
                {
                    deps.insert(req);
                }
            }
            _ => {}
        }
    }
}

fn qualify(container: Option<&str>, name: &str) -> String {
    match container {
        Some(prefix) => format!("{prefix}.{name}"),
        None => name.to_string(),
    }
}

fn build_ts_symbol(
    node: Node<'_>,
    kind: SymbolKind,
    name: String,
    source: &[u8],
) -> SymbolInfo {
    let signature = extract_signature(node, source, '{');
    let mut refs = BTreeSet::new();
    collect_ts_references(node, source, &mut refs);
    build_symbol_with_refs(node, kind, name, signature, refs)
}

fn collect_ts_references(node: Node<'_>, source: &[u8], refs: &mut BTreeSet<String>) {
    match node.kind() {
        "type_identifier" => {
            if let Ok(text) = node.utf8_text(source)
                && !is_ts_builtin_type(text)
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
        "new_expression" => {
            if let Some(ctor) = node.child_by_field_name("constructor")
                && let Ok(text) = ctor.utf8_text(source)
            {
                let cleaned = text.trim();
                if cleaned.len() >= 2 && !is_ts_builtin_type(cleaned) {
                    refs.insert(cleaned.to_string());
                }
            }
        }
        "jsx_opening_element" | "jsx_self_closing_element" => {
            if let Some(name) = field_text(node, "name", source)
                && is_pascal_case(&name)
            {
                refs.insert(name);
            }
        }
        _ => {}
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_ts_references(child, source, refs);
    }
}

fn extract_require_dep(node: Node<'_>, source: &[u8]) -> Option<String> {
    if node.kind() != "call_expression" {
        return None;
    }
    let func = node.child_by_field_name("function")?.utf8_text(source).ok()?;
    if func.trim() != "require" {
        return None;
    }
    let args = node.child_by_field_name("arguments")?;
    let mut cursor = args.walk();
    for child in args.children(&mut cursor) {
        if child.kind() == "string"
            && let Ok(text) = child.utf8_text(source)
        {
            let cleaned = text.trim_matches(|c| c == '\'' || c == '"').trim().to_string();
            if !cleaned.is_empty() {
                return Some(cleaned);
            }
        }
    }
    None
}

fn contains_jsx(node: Node<'_>) -> bool {
    if matches!(
        node.kind(),
        "jsx_element" | "jsx_self_closing_element" | "jsx_fragment"
    ) {
        return true;
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if contains_jsx(child) {
            return true;
        }
    }
    false
}

fn is_pascal_case(name: &str) -> bool {
    name.chars().next().is_some_and(|c| c.is_ascii_uppercase())
}

fn is_ts_builtin_type(t: &str) -> bool {
    matches!(
        t,
        "Promise"
            | "Array"
            | "Record"
            | "Partial"
            | "Required"
            | "Readonly"
            | "Pick"
            | "Omit"
            | "Exclude"
            | "Extract"
            | "Map"
            | "Set"
            | "Date"
            | "Error"
            | "RegExp"
            | "Function"
            | "Object"
            | "String"
            | "Number"
            | "Boolean"
    )
}
