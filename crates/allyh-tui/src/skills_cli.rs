use std::path::Path;
use std::sync::Arc;

use allyh_env::GitTracker;
use allyh_memory::{
    DeterministicHashEmbedder, MemoryStore, StorageConfig, compute_repo_id,
    load_skill_markdown_file, render_skill_markdown,
};
use anyhow::{Context, Result, bail};

use crate::cli::{SkillsArgs, SkillsCommand};

pub async fn run_cli(
    args: SkillsArgs,
    repo: &Path,
    db_uri: Option<&str>,
    scope_id: &str,
) -> Result<()> {
    let storage = match db_uri {
        Some(uri) => StorageConfig::from_uri(uri.to_string()),
        None => StorageConfig::default_local(),
    };
    let store = MemoryStore::connect(storage, Arc::new(DeterministicHashEmbedder))
        .await
        .context("failed to connect to LanceDB memory store")?;

    let canonical_id = GitTracker::new(repo).canonical_repo_identifier();
    let repo_id = compute_repo_id(&canonical_id);

    match args.action {
        SkillsCommand::List { json } => {
            let skills = store.list_skills(&repo_id, scope_id).await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&skills)?);
                return Ok(());
            }
            if skills.is_empty() {
                println!(
                    "No skills registered for repo partition `{}` (scope `{scope_id}`).",
                    &repo_id[..repo_id.len().min(12)]
                );
                return Ok(());
            }
            println!(
                "Registered skills ({}) — repo `{}` | scope `{scope_id}`\n",
                skills.len(),
                &repo_id[..repo_id.len().min(12)]
            );
            for skill in skills {
                let desc = if skill.description.is_empty() {
                    "—"
                } else {
                    skill.description.as_str()
                };
                println!("  {}  —  {}", skill.name, desc);
                if !skill.triggers.is_empty() {
                    println!("    triggers: {}", skill.triggers.join(", "));
                }
            }
            Ok(())
        }
        SkillsCommand::Show { name, json } => {
            let Some(skill) = store.get_skill(&repo_id, scope_id, &name).await? else {
                bail!("skill '{name}' not found in this repository partition");
            };
            if json {
                println!("{}", serde_json::to_string_pretty(&skill)?);
                return Ok(());
            }
            println!("Name:        {}", skill.name);
            println!("ID:          {}", skill.id);
            println!(
                "Description: {}",
                if skill.description.is_empty() {
                    "—"
                } else {
                    skill.description.as_str()
                }
            );
            println!(
                "Triggers:    {}",
                if skill.triggers.is_empty() {
                    "none".to_string()
                } else {
                    skill.triggers.join(", ")
                }
            );
            println!(
                "Updated:     {}",
                skill.updated_at.format("%Y-%m-%d %H:%M:%S UTC")
            );
            println!("\n{}", skill.instructions);
            Ok(())
        }
        SkillsCommand::Add {
            name,
            description,
            instructions,
            file,
            triggers,
        } => {
            let body = match (instructions, file) {
                (Some(text), None) => text,
                (None, Some(path)) => std::fs::read_to_string(&path)
                    .with_context(|| format!("failed to read instructions from '{}'", path.display()))?,
                (None, None) => {
                    bail!("provide either --instructions <text> or --file <path>");
                }
                (Some(_), Some(_)) => {
                    bail!("--instructions and --file cannot be used together");
                }
            };
            let (record, created) = store
                .save_skill(&repo_id, scope_id, &name, &description, &body, triggers)
                .await?;
            let verb = if created { "Created" } else { "Updated" };
            println!(
                "{verb} skill `{}` ({} trigger(s)).",
                record.name,
                record.triggers.len()
            );
            Ok(())
        }
        SkillsCommand::Import { path, name } => {
            let doc = load_skill_markdown_file(&path)
                .with_context(|| format!("failed to load skill file '{}'", path.display()))?;
            let final_name = name.unwrap_or(doc.name);
            let (record, created) = store
                .save_skill(
                    &repo_id,
                    scope_id,
                    &final_name,
                    &doc.description,
                    &doc.instructions,
                    doc.triggers,
                )
                .await?;
            let verb = if created { "Imported" } else { "Updated" };
            println!(
                "{verb} skill `{}` from '{}' ({} trigger(s)).",
                record.name,
                path.display(),
                record.triggers.len()
            );
            Ok(())
        }
        SkillsCommand::Export { name, out } => {
            let Some(skill) = store.get_skill(&repo_id, scope_id, &name).await? else {
                bail!("skill '{name}' not found in this repository partition");
            };
            let rendered = render_skill_markdown(&skill);
            if let Some(path) = out {
                if let Some(parent) = path.parent()
                    && !parent.as_os_str().is_empty()
                {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&path, rendered)
                    .with_context(|| format!("failed to write '{}'", path.display()))?;
                println!("Exported skill `{}` to '{}'.", skill.name, path.display());
            } else {
                print!("{rendered}");
            }
            Ok(())
        }
        SkillsCommand::Remove { name } => {
            let Some(removed) = store.delete_skill(&repo_id, scope_id, &name).await? else {
                bail!("skill '{name}' not found in this repository partition");
            };
            println!("Removed skill `{}`.", removed.name);
            Ok(())
        }
    }
}
