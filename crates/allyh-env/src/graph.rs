use std::collections::{HashMap, HashSet};

use allyh_memory::CodeGraphSnapshot;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::lang::LanguageRegistry;

/// Kind of source symbol extracted by AllyH's polyglot `tree-sitter` RepoGraph indexer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    Module,
    Const,
    Static,
    TypeAlias,
    Class,
    Interface,
    Method,
    Namespace,
    Variable,
    Component,
    Macro,
}

impl SymbolKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Function => "fn",
            Self::Struct => "struct",
            Self::Enum => "enum",
            Self::Trait => "trait",
            Self::Impl => "impl",
            Self::Module => "mod",
            Self::Const => "const",
            Self::Static => "static",
            Self::TypeAlias => "type",
            Self::Class => "class",
            Self::Interface => "interface",
            Self::Method => "method",
            Self::Namespace => "namespace",
            Self::Variable => "var",
            Self::Component => "component",
            Self::Macro => "macro",
        }
    }
}

/// Structured symbol definition stored in `code_graph_snapshots.symbols` JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolInfo {
    pub name: String,
    pub kind: SymbolKind,
    pub start_line: usize,
    pub end_line: usize,
    pub signature: String,
    pub references: Vec<String>,
}

/// Parsed symbol + dependency summary for a single source file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FileGraph {
    pub symbols: Vec<SymbolInfo>,
    pub dependencies: Vec<String>,
}

/// Compact RepoGraph neighborhood around a target symbol or file (1-hop / 2-hop).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EgoGraph {
    pub center: String,
    pub center_definitions: Vec<EgoNode>,
    pub hop1_neighbors: Vec<EgoEdge>,
    pub hop2_neighbors: Vec<EgoEdge>,
    /// Optional provenance tag when edges were semantically resolved via LSP (e.g. `"lsp:rust-analyzer"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_source: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EgoNode {
    pub file_path: String,
    pub symbol: SymbolInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EgoEdge {
    pub from_file: String,
    pub from_symbol: String,
    pub to_file: String,
    pub to_symbol: String,
    pub relation: String,
}

impl EgoGraph {
    /// Render a compact, token-efficient summary suitable for injecting into the dynamic preamble.
    pub fn render_compact(&self) -> String {
        if self.center_definitions.is_empty() && self.hop1_neighbors.is_empty() {
            return String::new();
        }
        let mut out = Vec::new();
        let source_tag = self
            .semantic_source
            .as_deref()
            .map(|src| format!(" [{src}]"))
            .unwrap_or_default();
        out.push(format!("### RepoGraph Ego-Graph: `{}`{source_tag}", self.center));
        for def in &self.center_definitions {
            out.push(format!(
                "- DEF `{}` ({}) in `{}:{}-{}` :: `{}`",
                def.symbol.name,
                def.symbol.kind.as_str(),
                def.file_path,
                def.symbol.start_line,
                def.symbol.end_line,
                def.symbol.signature
            ));
        }
        if !self.hop1_neighbors.is_empty() {
            out.push("- 1-Hop Neighbors:".to_string());
            for edge in self.hop1_neighbors.iter().take(12) {
                out.push(format!(
                    "  - `{}` (`{}`) --[{}]--> `{}` (`{}`)",
                    edge.from_symbol, edge.from_file, edge.relation, edge.to_symbol, edge.to_file
                ));
            }
        }
        if !self.hop2_neighbors.is_empty() {
            out.push("- 2-Hop Neighbors:".to_string());
            for edge in self.hop2_neighbors.iter().take(8) {
                out.push(format!(
                    "  - `{}` (`{}`) --[{}]--> `{}` (`{}`)",
                    edge.from_symbol, edge.from_file, edge.relation, edge.to_symbol, edge.to_file
                ));
            }
        }
        out.join("\n")
    }
}

/// Polyglot RepoGraph query engine operating over LanceDB [`CodeGraphSnapshot`]s.
pub struct RepoGraph;

impl RepoGraph {
    /// Build a 1-hop / 2-hop RepoGraph ego-graph around a symbol name or file path from LanceDB snapshots.
    pub fn build_ego_graph(
        snapshots: &[CodeGraphSnapshot],
        query_symbol: &str,
        max_hops: usize,
    ) -> EgoGraph {
        let query_lower = query_symbol.trim().to_ascii_lowercase();
        let mut parsed: Vec<(String, Vec<SymbolInfo>, Vec<String>)> = Vec::new();
        let mut symbol_index: HashMap<String, Vec<(String, SymbolInfo)>> = HashMap::new();

        for snap in snapshots {
            let syms: Vec<SymbolInfo> = serde_json::from_str(&snap.symbols_json).unwrap_or_default();
            for sym in &syms {
                let full_key = sym.name.to_ascii_lowercase();
                symbol_index
                    .entry(full_key.clone())
                    .or_default()
                    .push((snap.file_path.clone(), sym.clone()));

                // Also index unqualified member/method names (`Class.method` -> `method`, `Type::fn` -> `fn`)
                // so polyglot call-site references resolve cleanly across files.
                if let Some(short) = unqualified_symbol_key(&full_key)
                    && short != full_key
                    && short.len() >= 3
                {
                    symbol_index
                        .entry(short.to_string())
                        .or_default()
                        .push((snap.file_path.clone(), sym.clone()));
                }
            }
            parsed.push((snap.file_path.clone(), syms, snap.dependencies.clone()));
        }

        let mut center_defs: Vec<EgoNode> = Vec::new();
        let mut center_names: HashSet<String> = HashSet::new();

        for (file, syms, _) in &parsed {
            let file_matches = !query_lower.is_empty()
                && file.to_ascii_lowercase().contains(&query_lower);
            for sym in syms {
                let sym_lower = sym.name.to_ascii_lowercase();
                if (!query_lower.is_empty()
                    && (sym_lower == query_lower
                        || sym_lower.contains(&query_lower)
                        || (query_lower.contains(&sym_lower) && sym_lower.len() >= 4)))
                    || file_matches
                {
                    center_names.insert(sym_lower.clone());
                    if let Some(short) = unqualified_symbol_key(&sym_lower)
                        && short.len() >= 3
                    {
                        center_names.insert(short.to_string());
                    }
                    center_defs.push(EgoNode {
                        file_path: file.clone(),
                        symbol: sym.clone(),
                    });
                }
            }
        }
        center_defs.truncate(8);
        let center_def_keys: HashSet<(String, String)> = center_defs
            .iter()
            .map(|d| (d.file_path.clone(), d.symbol.name.clone()))
            .collect();

        let mut hop1_edges: Vec<EgoEdge> = Vec::new();
        let mut hop1_names: HashSet<String> = HashSet::new();
        let mut seen_edges: HashSet<(String, String, String, String)> = HashSet::new();

        if max_hops >= 1 {
            // Outgoing references from center definitions.
            for def in &center_defs {
                for ref_name in &def.symbol.references {
                    let key = ref_name.to_ascii_lowercase();
                    if center_names.contains(&key) {
                        continue;
                    }
                    if let Some(targets) = symbol_index.get(&key) {
                        for (target_file, target_sym) in targets {
                            if target_file == &def.file_path && target_sym.name == def.symbol.name {
                                continue;
                            }
                            let edge_key = (
                                def.file_path.clone(),
                                def.symbol.name.clone(),
                                target_file.clone(),
                                target_sym.name.clone(),
                            );
                            if seen_edges.insert(edge_key) {
                                hop1_names.insert(target_sym.name.to_ascii_lowercase());
                                hop1_edges.push(EgoEdge {
                                    from_file: def.file_path.clone(),
                                    from_symbol: def.symbol.name.clone(),
                                    to_file: target_file.clone(),
                                    to_symbol: target_sym.name.clone(),
                                    relation: "references".to_string(),
                                });
                            }
                        }
                    }
                }
            }

            // Incoming references pointing to center definitions.
            for (file, syms, _) in &parsed {
                for sym in syms {
                    let sym_key = sym.name.to_ascii_lowercase();
                    if center_names.contains(&sym_key) {
                        continue;
                    }
                    for ref_name in &sym.references {
                        let ref_key = ref_name.to_ascii_lowercase();
                        if center_names.contains(&ref_key)
                            && let Some(targets) = symbol_index.get(&ref_key)
                        {
                            for (target_file, target_sym) in targets {
                                if !center_def_keys.contains(&(target_file.clone(), target_sym.name.clone())) {
                                    continue;
                                }
                                if target_file == file && target_sym.name == sym.name {
                                    continue;
                                }
                                let edge_key = (
                                    file.clone(),
                                    sym.name.clone(),
                                    target_file.clone(),
                                    target_sym.name.clone(),
                                );
                                if seen_edges.insert(edge_key) {
                                    hop1_names.insert(sym_key.clone());
                                    hop1_edges.push(EgoEdge {
                                        from_file: file.clone(),
                                        from_symbol: sym.name.clone(),
                                        to_file: target_file.clone(),
                                        to_symbol: target_sym.name.clone(),
                                        relation: "calls_or_uses".to_string(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut hop2_edges: Vec<EgoEdge> = Vec::new();
        if max_hops >= 2 && !hop1_names.is_empty() {
            for hop1_name in &hop1_names {
                if let Some(nodes) = symbol_index.get(hop1_name) {
                    for (file, sym) in nodes {
                        for ref_name in &sym.references {
                            let ref_key = ref_name.to_ascii_lowercase();
                            if center_names.contains(&ref_key) || hop1_names.contains(&ref_key) {
                                continue;
                            }
                            if let Some(targets) = symbol_index.get(&ref_key) {
                                for (target_file, target_sym) in targets {
                                    if target_file == file && target_sym.name == sym.name {
                                        continue;
                                    }
                                    let edge_key = (
                                        file.clone(),
                                        sym.name.clone(),
                                        target_file.clone(),
                                        target_sym.name.clone(),
                                    );
                                    if seen_edges.insert(edge_key) {
                                        hop2_edges.push(EgoEdge {
                                            from_file: file.clone(),
                                            from_symbol: sym.name.clone(),
                                            to_file: target_file.clone(),
                                            to_symbol: target_sym.name.clone(),
                                            relation: "hop2_ref".to_string(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        EgoGraph {
            center: query_symbol.to_string(),
            center_definitions: center_defs,
            hop1_neighbors: hop1_edges,
            hop2_neighbors: hop2_edges,
            semantic_source: None,
        }
    }
}

fn unqualified_symbol_key(sym: &str) -> Option<&str> {
    sym.rsplit_once("::")
        .or_else(|| sym.rsplit_once('.'))
        .map(|(_, short)| short)
}

/// Backward-compatible wrapper preserving [`RustIndexer::index_source`] and [`RustIndexer::build_ego_graph`].
pub struct RustIndexer;

impl RustIndexer {
    /// Parse Rust source text and extract symbols, signatures, local references, and `use` dependencies.
    pub fn index_source(source: &str) -> Result<FileGraph> {
        let (_, graph) = LanguageRegistry::default_all().index_path("lib.rs", source)?;
        Ok(graph)
    }

    /// Build a 1-hop / 2-hop RepoGraph ego-graph around a symbol name or file path from LanceDB snapshots.
    pub fn build_ego_graph(
        snapshots: &[CodeGraphSnapshot],
        query_symbol: &str,
        max_hops: usize,
    ) -> EgoGraph {
        RepoGraph::build_ego_graph(snapshots, query_symbol, max_hops)
    }
}
