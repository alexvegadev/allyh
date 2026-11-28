use std::collections::BTreeSet;

use tree_sitter::{Node, Parser};

use crate::error::{EnvError, Result};
use crate::graph::{FileGraph, SymbolInfo, SymbolKind};
use crate::lang::{
    LanguageIndexer, SourceLanguage, build_symbol_with_refs, extract_first_line_signature,
    field_text, is_common_noise_ident,
};

/// `tree-sitter-python` adapter for `.py` and `.pyi` files.
pub struct PythonLangIndexer;

impl LanguageIndexer for PythonLangIndexer {
    fn language(&self) -> SourceLanguage {
        SourceLanguage::Python
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["py", "pyi"]
    }

    fn index_source(&self, source: &str) -> Result<FileGraph> {
        let mut parser = Parser::new();
        let language: tree_sitter::Language = tree_sitter_python::LANGUAGE.into();
        parser
            .set_language(&language)
            .map_err(|e| EnvError::TreeSitter(e.to_string()))?;

        let tree = parser
            .parse(source, None)
            .ok_or_else(|| EnvError::TreeSitter("failed to parse Python source".to_string()))?;

        let bytes = source.as_bytes();
        let mut symbols = Vec::new();
        let mut deps = BTreeSet::new();

        walk_py_items(tree.root_node(), bytes, None, &mut symbols, &mut deps);

        Ok(FileGraph {
            symbols,
            dependencies: deps.into_iter().collect(),
        })
    }
}

fn walk_py_items(
    node: Node<'_>,
    source: &[u8],
    class_prefix: Option<&str>,
    symbols: &mut Vec<SymbolInfo>,
    deps: &mut BTreeSet<String>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "import_statement" => {
                if let Ok(text) = child.utf8_text(source) {
                    let cleaned = text.trim().trim_start_matches("import ").trim().to_string();
                    if !cleaned.is_empty() {
                        deps.insert(cleaned);
                    }
                }
            }
            "import_from_statement" => {
                if let Some(module) = field_text(child, "module_name", source) {
                    deps.insert(module);
                } else if let Ok(text) = child.utf8_text(source) {
                    deps.insert(text.trim().to_string());
                }
            }
            "decorated_definition" => {
                if let Some(def) = child.child_by_field_name("definition") {
                    index_py_definition(def, child, source, class_prefix, symbols, deps);
                }
            }
            "function_definition" | "class_definition" => {
                index_py_definition(child, child, source, class_prefix, symbols, deps);
            }
            "expression_statement" if class_prefix.is_none() => {
                if let Some(assign) = child.child(0)
                    && assign.kind() == "assignment"
                    && let Some(left) = field_text(assign, "left", source)
                    && left
                        .chars()
                        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
                    && left.len() >= 2
                {
                    let sig = extract_first_line_signature(assign, source);
                    let mut refs = BTreeSet::new();
                    collect_py_references(assign, source, &mut refs);
                    symbols.push(build_symbol_with_refs(
                        assign,
                        SymbolKind::Const,
                        left,
                        sig,
                        refs,
                    ));
                }
            }
            _ => {}
        }
    }
}

fn index_py_definition(
    def_node: Node<'_>,
    span_node: Node<'_>,
    source: &[u8],
    class_prefix: Option<&str>,
    symbols: &mut Vec<SymbolInfo>,
    deps: &mut BTreeSet<String>,
) {
    match def_node.kind() {
        "function_definition" => {
            if let Some(name) = field_text(def_node, "name", source) {
                let (kind, full_name) = match class_prefix {
                    Some(cls) => (SymbolKind::Method, format!("{cls}.{name}")),
                    None => (SymbolKind::Function, name),
                };
                let sig = extract_first_line_signature(def_node, source);
                let mut refs = BTreeSet::new();
                collect_py_references(span_node, source, &mut refs);
                symbols.push(build_symbol_with_refs(span_node, kind, full_name, sig, refs));
            }
        }
        "class_definition" => {
            if let Some(name) = field_text(def_node, "name", source) {
                let full_name = match class_prefix {
                    Some(cls) => format!("{cls}.{name}"),
                    None => name,
                };
                let sig = extract_first_line_signature(def_node, source);
                let mut refs = BTreeSet::new();
                collect_py_references(span_node, source, &mut refs);
                symbols.push(build_symbol_with_refs(
                    span_node,
                    SymbolKind::Class,
                    full_name.clone(),
                    sig,
                    refs,
                ));
                if let Some(body) = def_node.child_by_field_name("body") {
                    walk_py_items(body, source, Some(&full_name), symbols, deps);
                }
            }
        }
        _ => {}
    }
}

fn collect_py_references(node: Node<'_>, source: &[u8], refs: &mut BTreeSet<String>) {
    match node.kind() {
        "call" => {
            if let Some(func) = node.child_by_field_name("function")
                && let Ok(text) = func.utf8_text(source)
            {
                let cleaned = text.trim();
                let short = cleaned.rsplit('.').next().unwrap_or(cleaned);
                if short.len() >= 3 && !is_common_noise_ident(short) {
                    refs.insert(short.to_string());
                }
                if cleaned.contains('.') && !cleaned.starts_with("self.") && cleaned.len() <= 64 {
                    refs.insert(cleaned.to_string());
                }
            }
        }
        "type" => {
            if let Ok(text) = node.utf8_text(source) {
                for token in text.split(|c: char| !c.is_alphanumeric() && c != '_') {
                    if token.len() >= 2
                        && token.chars().next().is_some_and(|c| c.is_ascii_uppercase())
                        && !is_py_builtin_type(token)
                    {
                        refs.insert(token.to_string());
                    }
                }
            }
        }
        _ => {}
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_py_references(child, source, refs);
    }
}

fn is_py_builtin_type(t: &str) -> bool {
    matches!(
        t,
        "Any"
            | "Optional"
            | "Union"
            | "List"
            | "Dict"
            | "Set"
            | "Tuple"
            | "Sequence"
            | "Mapping"
            | "Callable"
            | "Iterable"
            | "Iterator"
            | "Type"
            | "Final"
            | "ClassVar"
            | "Protocol"
            | "None"
    )
}
