use crate::error::Result;
use crate::graph::FileGraph;
use crate::lang::{LanguageIndexer, SourceLanguage};

/// `tree-sitter-javascript` adapter for `.js`, `.jsx`, `.mjs`, `.cjs`.
pub struct JavaScriptLangIndexer;

impl LanguageIndexer for JavaScriptLangIndexer {
    fn language(&self) -> SourceLanguage {
        SourceLanguage::JavaScript
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["js", "jsx", "mjs", "cjs"]
    }

    fn index_source(&self, source: &str) -> Result<FileGraph> {
        #[cfg(feature = "lang-typescript")]
        {
            crate::lang::typescript::index_ts_like(
                source,
                tree_sitter_javascript::LANGUAGE.into(),
                true,
            )
        }
        #[cfg(not(feature = "lang-typescript"))]
        {
            index_js_standalone(source)
        }
    }
}

#[cfg(not(feature = "lang-typescript"))]
fn index_js_standalone(source: &str) -> Result<FileGraph> {
    use std::collections::BTreeSet;
    use tree_sitter::{Node, Parser};

    use crate::error::EnvError;
    use crate::graph::{SymbolInfo, SymbolKind};
    use crate::lang::{build_symbol_with_refs, extract_signature, field_text};

    let mut parser = Parser::new();
    let language: tree_sitter::Language = tree_sitter_javascript::LANGUAGE.into();
    parser
        .set_language(&language)
        .map_err(|e| EnvError::TreeSitter(e.to_string()))?;
    let tree = parser
        .parse(source, None)
        .ok_or_else(|| EnvError::TreeSitter("failed to parse JavaScript source".to_string()))?;

    let bytes = source.as_bytes();
    let mut symbols = Vec::new();
    let mut deps = BTreeSet::new();

    fn walk(
        node: Node<'_>,
        source: &[u8],
        symbols: &mut Vec<SymbolInfo>,
        deps: &mut BTreeSet<String>,
    ) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "import_statement" => {
                    if let Some(src) = field_text(child, "source", source) {
                        deps.insert(src.trim_matches(|c| c == '\'' || c == '"').to_string());
                    }
                }
                "export_statement" => walk(child, source, symbols, deps),
                "function_declaration" => {
                    if let Some(name) = field_text(child, "name", source) {
                        symbols.push(build_symbol_with_refs(
                            child,
                            SymbolKind::Function,
                            name,
                            extract_signature(child, source, '{'),
                            BTreeSet::new(),
                        ));
                    }
                }
                "class_declaration" => {
                    if let Some(name) = field_text(child, "name", source) {
                        symbols.push(build_symbol_with_refs(
                            child,
                            SymbolKind::Class,
                            name,
                            extract_signature(child, source, '{'),
                            BTreeSet::new(),
                        ));
                    }
                }
                _ => {}
            }
        }
    }

    walk(tree.root_node(), bytes, &mut symbols, &mut deps);
    Ok(FileGraph {
        symbols,
        dependencies: deps.into_iter().collect(),
    })
}
