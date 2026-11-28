use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Duration;

use allyh_memory::CodeGraphSnapshot;
use sha2::{Digest, Sha256};

use crate::graph::{EgoEdge, EgoGraph, SymbolInfo};
use crate::verify::{CompilerDiagnostic, VerificationKind, VerificationReport};

pub mod client;
pub mod protocol;
pub mod registry;

pub use client::{LspClient, LspServerSpec};
pub use protocol::{
    LspCallHierarchyItem, LspDiagnostic, LspLocation, LspPosition, LspRange, encode_lsp_frame,
    file_uri_to_path, file_uri_to_rel_path, lsp_language_id_for_path, parse_locations_value,
    path_to_file_uri, server_family_for_language_id,
};
pub use registry::{LspManager, LspServerState, LspServerStatus, find_binary_on_path};

/// Default wait budget for sub-second post-edit LSP diagnostics before falling back to CLI.
pub const DEFAULT_LSP_DIAGNOSTIC_WAIT: Duration = Duration::from_millis(600);

/// Upgrade a syntactic `tree-sitter` [`EgoGraph`] with type-resolved semantic edges from active LSP
/// servers (`references`, `implementation`, `definition`, and `callHierarchy`).
///
/// When LSP returns semantic edges for a center definition, ambiguous name-matched `tree-sitter`
/// 1-hop edges for that center are replaced with exact LSP edges to eliminate cross-module name
/// collisions. Falls back transparently to the original `tree-sitter` graph when no LSP is available.
pub async fn enrich_ego_graph_with_lsp(
    mut ego: EgoGraph,
    snapshots: &[CodeGraphSnapshot],
    repo_root: &Path,
    lsp: &LspManager,
) -> EgoGraph {
    if ego.center_definitions.is_empty() {
        return ego;
    }

    let parsed_by_file: HashMap<String, Vec<SymbolInfo>> = snapshots
        .iter()
        .map(|s| {
            let syms: Vec<SymbolInfo> = serde_json::from_str(&s.symbols_json).unwrap_or_default();
            (s.file_path.clone(), syms)
        })
        .collect();

    let mut semantic_edges: Vec<EgoEdge> = Vec::new();
    let mut seen_edges: HashSet<(String, String, String, String, String)> = HashSet::new();
    let mut disambiguated_centers: HashSet<(String, String)> = HashSet::new();
    let mut used_servers: Vec<String> = Vec::new();

    for def in ego.center_definitions.iter().take(4) {
        let Some(lang_id) = lsp_language_id_for_path(&def.file_path) else {
            continue;
        };
        if !lsp.has_server_for(lang_id).await {
            continue;
        }
        let abs_path = repo_root.join(&def.file_path);
        let Ok(content) = std::fs::read_to_string(&abs_path) else {
            continue;
        };
        let Some(client) = lsp.get_or_start(lang_id).await else {
            continue;
        };

        let hash = sha256_hex(&content);
        let Ok((uri, _)) = client
            .sync_document(&abs_path, lang_id, &content, &hash, false)
            .await
        else {
            continue;
        };

        let position = locate_symbol_position(&content, &def.symbol);
        let mut center_edges_found = 0usize;

        // 1. Incoming references (`textDocument/references`)
        if let Ok(locs) = client.references(&uri, position).await {
            for loc in locs {
                if let Some((ref_file, ref_sym)) =
                    resolve_location_to_symbol(&loc, repo_root, &parsed_by_file)
                {
                    if ref_file == def.file_path && ref_sym == def.symbol.name {
                        continue;
                    }
                    let key = (
                        ref_file.clone(),
                        ref_sym.clone(),
                        def.file_path.clone(),
                        def.symbol.name.clone(),
                        "lsp_references".to_string(),
                    );
                    if seen_edges.insert(key) {
                        center_edges_found += 1;
                        semantic_edges.push(EgoEdge {
                            from_file: ref_file,
                            from_symbol: ref_sym,
                            to_file: def.file_path.clone(),
                            to_symbol: def.symbol.name.clone(),
                            relation: "lsp_references".to_string(),
                        });
                    }
                }
            }
        }

        // 2. Implementations (`textDocument/implementation`)
        if let Ok(impls) = client.implementation(&uri, position).await {
            for loc in impls {
                if let Some((impl_file, impl_sym)) =
                    resolve_location_to_symbol(&loc, repo_root, &parsed_by_file)
                {
                    if impl_file == def.file_path && impl_sym == def.symbol.name {
                        continue;
                    }
                    let key = (
                        impl_file.clone(),
                        impl_sym.clone(),
                        def.file_path.clone(),
                        def.symbol.name.clone(),
                        "lsp_implements".to_string(),
                    );
                    if seen_edges.insert(key) {
                        center_edges_found += 1;
                        semantic_edges.push(EgoEdge {
                            from_file: impl_file,
                            from_symbol: impl_sym,
                            to_file: def.file_path.clone(),
                            to_symbol: def.symbol.name.clone(),
                            relation: "lsp_implements".to_string(),
                        });
                    }
                }
            }
        }

        // 3. Call hierarchy (`incomingCalls` + `outgoingCalls`)
        if let Ok((incoming, outgoing)) = client.call_hierarchy(&uri, position).await {
            for item in incoming {
                let rel_file = file_uri_to_rel_path(&item.uri, repo_root)
                    .unwrap_or_else(|| item.uri.clone());
                let from_sym = resolve_symbol_at_line(
                    &rel_file,
                    item.selection_range.start.line as usize + 1,
                    &parsed_by_file,
                )
                .unwrap_or(item.name);
                if rel_file == def.file_path && from_sym == def.symbol.name {
                    continue;
                }
                let key = (
                    rel_file.clone(),
                    from_sym.clone(),
                    def.file_path.clone(),
                    def.symbol.name.clone(),
                    "lsp_called_by".to_string(),
                );
                if seen_edges.insert(key) {
                    center_edges_found += 1;
                    semantic_edges.push(EgoEdge {
                        from_file: rel_file,
                        from_symbol: from_sym,
                        to_file: def.file_path.clone(),
                        to_symbol: def.symbol.name.clone(),
                        relation: "lsp_called_by".to_string(),
                    });
                }
            }
            for item in outgoing {
                let rel_file = file_uri_to_rel_path(&item.uri, repo_root)
                    .unwrap_or_else(|| item.uri.clone());
                let to_sym = resolve_symbol_at_line(
                    &rel_file,
                    item.selection_range.start.line as usize + 1,
                    &parsed_by_file,
                )
                .unwrap_or(item.name);
                if rel_file == def.file_path && to_sym == def.symbol.name {
                    continue;
                }
                let key = (
                    def.file_path.clone(),
                    def.symbol.name.clone(),
                    rel_file.clone(),
                    to_sym.clone(),
                    "lsp_calls".to_string(),
                );
                if seen_edges.insert(key) {
                    center_edges_found += 1;
                    semantic_edges.push(EgoEdge {
                        from_file: def.file_path.clone(),
                        from_symbol: def.symbol.name.clone(),
                        to_file: rel_file,
                        to_symbol: to_sym,
                        relation: "lsp_calls".to_string(),
                    });
                }
            }
        }

        if center_edges_found > 0 {
            disambiguated_centers.insert((def.file_path.clone(), def.symbol.name.clone()));
            let srv = client.spec().name.clone();
            if !used_servers.contains(&srv) {
                used_servers.push(srv);
            }
        }
    }

    if !semantic_edges.is_empty() {
        let disambiguated_names: HashSet<String> = disambiguated_centers
            .iter()
            .map(|(_, sym)| sym.to_ascii_lowercase())
            .collect();
        // Keep syntactic edges only for centers that were NOT semantically disambiguated by LSP.
        let retained_syntactic: Vec<EgoEdge> = ego
            .hop1_neighbors
            .into_iter()
            .filter(|e| {
                !disambiguated_centers.contains(&(e.from_file.clone(), e.from_symbol.clone()))
                    && !disambiguated_centers.contains(&(e.to_file.clone(), e.to_symbol.clone()))
                    && !disambiguated_names.contains(&e.to_symbol.to_ascii_lowercase())
            })
            .collect();
        semantic_edges.extend(retained_syntactic);
        ego.hop1_neighbors = semantic_edges;
        ego.semantic_source = Some(format!("lsp:{}", used_servers.join(",")));
    }

    ego
}

/// Sync an edited file with its active LSP server and pull/await sub-second diagnostics.
/// Returns `Ok(None)` cleanly if no LSP server is available or if it did not return diagnostics
/// within `wait_budget`, signaling the caller to fall back to the CLI `ToolchainVerifier`.
pub async fn collect_lsp_diagnostics_report(
    rel_path: &str,
    abs_path: &Path,
    content: &str,
    content_hash: &str,
    repo_root: &Path,
    lsp: &LspManager,
    wait_budget: Duration,
) -> Option<VerificationReport> {
    let lang_id = lsp_language_id_for_path(rel_path)?;
    if !lsp.has_server_for(lang_id).await {
        return None;
    }
    let client = lsp.get_or_start(lang_id).await?;
    let (uri, pre_gen) = client
        .sync_document(abs_path, lang_id, content, content_hash, true)
        .await
        .ok()?;
    let raw_diags = client
        .fetch_diagnostics(&uri, pre_gen, wait_budget)
        .await
        .ok()??;

    let normalized_file = file_uri_to_rel_path(&uri, repo_root).unwrap_or_else(|| rel_path.to_string());
    let diagnostics: Vec<CompilerDiagnostic> = raw_diags
        .iter()
        .filter_map(|d| d.to_compiler_diagnostic(Some(normalized_file.clone())))
        .collect();
    let error_count = diagnostics.iter().filter(|d| d.level == "error").count();
    let warning_count = diagnostics.iter().filter(|d| d.level == "warning").count();
    let server_name = client.spec().name.clone();

    Some(VerificationReport {
        toolchain: format!("lsp:{server_name}"),
        working_dir: ".".to_string(),
        kind: VerificationKind::Check,
        command: format!("lsp:{server_name} diagnostics ({rel_path})"),
        success: error_count == 0,
        error_count,
        warning_count,
        diagnostics,
        raw_summary: String::new(),
    })
}

/// Locate the 0-based `(line, character)` of a symbol's identifier in `content`.
pub fn locate_symbol_position(content: &str, symbol: &SymbolInfo) -> LspPosition {
    let short_name = symbol
        .name
        .rsplit_once("::")
        .or_else(|| symbol.name.rsplit_once('.'))
        .map(|(_, s)| s)
        .unwrap_or(symbol.name.as_str());

    let start_idx = symbol.start_line.saturating_sub(1);
    let end_idx = symbol.end_line.saturating_sub(1).min(start_idx + 4);
    for (idx, line) in content.lines().enumerate().skip(start_idx).take(end_idx - start_idx + 1) {
        if let Some(col) = find_ident_column(line, short_name) {
            return LspPosition {
                line: idx as u32,
                character: col as u32,
            };
        }
    }

    LspPosition {
        line: start_idx as u32,
        character: 0,
    }
}

fn find_ident_column(line: &str, needle: &str) -> Option<usize> {
    if needle.is_empty() {
        return None;
    }
    let bytes = line.as_bytes();
    let mut search_from = 0usize;
    while let Some(rel) = line[search_from..].find(needle) {
        let abs = search_from + rel;
        let before_ok = abs == 0 || !is_ident_byte(bytes[abs - 1]);
        let after_pos = abs + needle.len();
        let after_ok = after_pos >= bytes.len() || !is_ident_byte(bytes[after_pos]);
        if before_ok && after_ok {
            return Some(line[..abs].encode_utf16().count());
        }
        search_from = abs + needle.len();
    }
    None
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn resolve_location_to_symbol(
    loc: &LspLocation,
    repo_root: &Path,
    parsed_by_file: &HashMap<String, Vec<SymbolInfo>>,
) -> Option<(String, String)> {
    let rel_file = file_uri_to_rel_path(&loc.uri, repo_root)?;
    let line_1based = loc.range.start.line as usize + 1;
    let sym_name = resolve_symbol_at_line(&rel_file, line_1based, parsed_by_file)
        .unwrap_or_else(|| format!("{rel_file}:{line_1based}"));
    Some((rel_file, sym_name))
}

fn resolve_symbol_at_line(
    rel_file: &str,
    line_1based: usize,
    parsed_by_file: &HashMap<String, Vec<SymbolInfo>>,
) -> Option<String> {
    let syms = parsed_by_file.get(rel_file)?;
    // Choose the innermost (smallest span) symbol enclosing `line_1based`.
    syms.iter()
        .filter(|s| s.start_line <= line_1based && line_1based <= s.end_line)
        .min_by_key(|s| s.end_line.saturating_sub(s.start_line))
        .map(|s| s.name.clone())
}

fn sha256_hex(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hex::encode(hasher.finalize())
}
