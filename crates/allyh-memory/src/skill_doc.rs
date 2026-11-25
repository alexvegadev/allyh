use std::path::Path;

use crate::error::{MemoryError, Result};
use crate::schema::{SkillRecord, normalize_skill_slug};

/// Portable `SKILL.md` representation (YAML frontmatter + Markdown instructions body).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillDocument {
    pub name: String,
    pub description: String,
    pub instructions: String,
    pub triggers: Vec<String>,
}

/// Load and parse a `SKILL.md` (or plain Markdown skill) file from disk.
pub fn load_skill_markdown_file(path: &Path) -> Result<SkillDocument> {
    let raw = std::fs::read_to_string(path)?;
    let fallback_name = infer_fallback_name(path);
    parse_skill_markdown(&raw, fallback_name.as_deref())
}

fn infer_fallback_name(path: &Path) -> Option<String> {
    let file_name = path.file_name()?.to_string_lossy();
    if file_name.eq_ignore_ascii_case("skill.md") {
        path.parent()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned())
    } else {
        path.file_stem().map(|s| s.to_string_lossy().into_owned())
    }
}

/// Parse a `SKILL.md` document with optional YAML frontmatter (`--- ... ---`).
pub fn parse_skill_markdown(raw: &str, fallback_name: Option<&str>) -> Result<SkillDocument> {
    let normalized = raw.replace("\r\n", "\n");
    let stripped = normalized.strip_prefix('\u{feff}').unwrap_or(&normalized);

    let (frontmatter, body) = split_frontmatter(stripped);

    let mut name: Option<String> = None;
    let mut description = String::new();
    let mut triggers: Vec<String> = Vec::new();

    if let Some(fm) = frontmatter {
        parse_frontmatter(fm, &mut name, &mut description, &mut triggers);
    }

    let instructions = body.trim().to_string();
    if instructions.is_empty() {
        return Err(MemoryError::InvalidSkill(
            "skill markdown body (instructions) must not be empty".to_string(),
        ));
    }

    let resolved_name = name
        .filter(|n| !n.trim().is_empty())
        .or_else(|| extract_heading_title(&instructions))
        .or_else(|| {
            fallback_name
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        })
        .ok_or_else(|| {
            MemoryError::InvalidSkill(
                "missing skill 'name' in YAML frontmatter or leading heading".to_string(),
            )
        })?;

    if normalize_skill_slug(&resolved_name).is_empty() {
        return Err(MemoryError::InvalidSkill(format!(
            "skill name '{resolved_name}' must contain at least one ASCII letter or digit"
        )));
    }

    if description.trim().is_empty() {
        description = first_summary_line(&instructions);
    }

    Ok(SkillDocument {
        name: resolved_name.trim().to_string(),
        description: description.trim().to_string(),
        instructions,
        triggers,
    })
}

/// Render a [`SkillRecord`] as a portable `SKILL.md` file with YAML frontmatter.
pub fn render_skill_markdown(skill: &SkillRecord) -> String {
    let mut out = String::from("---\n");
    out.push_str(&format!("name: {}\n", yaml_scalar(&skill.name)));
    out.push_str(&format!("description: {}\n", yaml_scalar(&skill.description)));
    if !skill.triggers.is_empty() {
        out.push_str("triggers:\n");
        for trigger in &skill.triggers {
            out.push_str(&format!("  - {}\n", yaml_scalar(trigger)));
        }
    }
    out.push_str("---\n\n");
    out.push_str(skill.instructions.trim());
    out.push('\n');
    out
}

fn split_frontmatter(content: &str) -> (Option<&str>, &str) {
    let Some(rest) = content.strip_prefix("---\n") else {
        return (None, content);
    };
    let mut offset = 0usize;
    for line in rest.split_inclusive('\n') {
        let trimmed = line.trim();
        if trimmed == "---" || trimmed == "..." {
            let fm = &rest[..offset];
            let body = &rest[offset + line.len()..];
            return (Some(fm), body);
        }
        offset += line.len();
    }
    (None, content)
}

fn parse_frontmatter(
    fm: &str,
    name: &mut Option<String>,
    description: &mut String,
    triggers: &mut Vec<String>,
) {
    let lines: Vec<&str> = fm.lines().collect();
    let mut idx = 0usize;
    while idx < lines.len() {
        let line = lines[idx];
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            idx += 1;
            continue;
        }
        let Some((raw_key, raw_val)) = trimmed.split_once(':') else {
            idx += 1;
            continue;
        };
        let key = raw_key.trim().to_ascii_lowercase();
        let val = raw_val.trim();

        match key.as_str() {
            "name" => {
                let parsed = unquote_yaml_scalar(val);
                if !parsed.is_empty() {
                    *name = Some(parsed);
                }
                idx += 1;
            }
            "description" => {
                if val == "|" || val == ">" || val == "|-" || val == ">-" {
                    let folded = val.starts_with('>');
                    let (block, next_idx) = collect_indented_block(&lines, idx + 1, folded);
                    *description = block;
                    idx = next_idx;
                } else {
                    *description = unquote_yaml_scalar(val);
                    idx += 1;
                }
            }
            "triggers" => {
                if !val.is_empty() {
                    triggers.extend(parse_inline_list(val));
                    idx += 1;
                } else {
                    let (items, next_idx) = collect_yaml_list(&lines, idx + 1);
                    triggers.extend(items);
                    idx = next_idx;
                }
            }
            _ => {
                idx += 1;
            }
        }
    }
}

fn collect_indented_block(lines: &[&str], start: usize, folded: bool) -> (String, usize) {
    let mut parts = Vec::new();
    let mut idx = start;
    while idx < lines.len() {
        let line = lines[idx];
        if line.trim().is_empty() {
            parts.push(String::new());
            idx += 1;
            continue;
        }
        if !line.starts_with(' ') && !line.starts_with('\t') {
            break;
        }
        parts.push(line.trim().to_string());
        idx += 1;
    }
    let joined = if folded {
        parts
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        parts.join("\n").trim().to_string()
    };
    (joined, idx)
}

fn collect_yaml_list(lines: &[&str], start: usize) -> (Vec<String>, usize) {
    let mut items = Vec::new();
    let mut idx = start;
    while idx < lines.len() {
        let line = lines[idx];
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            idx += 1;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("- ") {
            let item = unquote_yaml_scalar(rest.trim());
            if !item.is_empty() {
                items.push(item);
            }
            idx += 1;
        } else if line.starts_with(' ') || line.starts_with('\t') {
            idx += 1;
        } else {
            break;
        }
    }
    (items, idx)
}

fn parse_inline_list(val: &str) -> Vec<String> {
    let inner = val
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(val);
    inner
        .split(',')
        .map(|item| unquote_yaml_scalar(item.trim()))
        .filter(|s| !s.is_empty())
        .collect()
}

fn unquote_yaml_scalar(val: &str) -> String {
    let trimmed = val.trim();
    if trimmed.len() >= 2
        && ((trimmed.starts_with('"') && trimmed.ends_with('"'))
            || (trimmed.starts_with('\'') && trimmed.ends_with('\'')))
    {
        let inner = &trimmed[1..trimmed.len() - 1];
        if trimmed.starts_with('"') {
            inner
                .replace("\\\"", "\"")
                .replace("\\n", "\n")
                .replace("\\\\", "\\")
        } else {
            inner.replace("''", "'")
        }
    } else {
        trimmed.to_string()
    }
}

fn yaml_scalar(val: &str) -> String {
    let needs_quotes = val.is_empty()
        || val.contains(':')
        || val.contains('#')
        || val.contains('\n')
        || val.contains('"')
        || val.contains('\'')
        || val.starts_with('[')
        || val.starts_with('{')
        || val.starts_with('-')
        || val.starts_with('*')
        || val.starts_with('!')
        || val != val.trim();
    if needs_quotes {
        let escaped = val
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n");
        format!("\"{escaped}\"")
    } else {
        val.to_string()
    }
}

fn extract_heading_title(markdown: &str) -> Option<String> {
    for line in markdown.lines() {
        let trimmed = line.trim();
        if let Some(title) = trimmed.strip_prefix("# ") {
            let t = title.trim();
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
    }
    None
}

fn first_summary_line(markdown: &str) -> String {
    for line in markdown.lines() {
        let trimmed = line.trim().trim_start_matches('#').trim();
        if !trimmed.is_empty() {
            return trimmed.chars().take(120).collect();
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use chrono::Utc;

    use super::*;

    #[test]
    fn parses_frontmatter_with_list_triggers_and_roundtrips() {
        let raw = "\
---
name: add-lancedb-table
description: Add a new Apache Arrow table to LanceDB
triggers:
  - new table
  - lancedb schema
---

1. Add a struct and schema in schema.rs.
2. Register the table in store.rs.
";
        let doc = parse_skill_markdown(raw, None).expect("parse skill md");
        assert_eq!(doc.name, "add-lancedb-table");
        assert_eq!(doc.description, "Add a new Apache Arrow table to LanceDB");
        assert_eq!(doc.triggers, vec!["new table", "lancedb schema"]);
        assert!(doc.instructions.contains("schema.rs"));

        let record = SkillRecord {
            id: "r:s:add-lancedb-table".into(),
            repo_id: "r".into(),
            scope_id: "s".into(),
            name: doc.name,
            description: doc.description,
            instructions: doc.instructions,
            triggers: doc.triggers,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let rendered = render_skill_markdown(&record);
        let reparsed = parse_skill_markdown(&rendered, None).expect("reparse rendered skill md");
        assert_eq!(reparsed.name, record.name);
        assert_eq!(reparsed.description, record.description);
        assert_eq!(reparsed.triggers, record.triggers);
        assert_eq!(reparsed.instructions, record.instructions);
    }

    #[test]
    fn parses_inline_triggers_and_fallback_name() {
        let raw = "---\ndescription: \"Deploy: production\"\ntriggers: [deploy, release]\n---\nRun cargo build --release.";
        let doc = parse_skill_markdown(raw, Some("release-flow")).expect("parse");
        assert_eq!(doc.name, "release-flow");
        assert_eq!(doc.description, "Deploy: production");
        assert_eq!(doc.triggers, vec!["deploy", "release"]);
    }
}
