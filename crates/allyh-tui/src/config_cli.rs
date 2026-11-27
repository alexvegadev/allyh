use std::path::Path;
use std::sync::Arc;

use allyh_env::{GitTracker, ToolchainVerifier};
use allyh_memory::{
    DeterministicHashEmbedder, MemoryStore, RepoProfile, StorageConfig, compute_repo_id,
};
use anyhow::{Context, Result};

use crate::cli::{ConfigArgs, ConfigCommand};

/// Execute `allyh config <subcommand>` against the out-of-tree LanceDB `repo_profiles` table.
pub async fn run_cli(
    args: ConfigArgs,
    repo: &Path,
    db_uri: Option<&str>,
    scope: &str,
) -> Result<()> {
    let storage = match db_uri {
        Some(uri) => StorageConfig::from_uri(uri.to_string()),
        None => StorageConfig::default_local(),
    };
    let store = MemoryStore::connect(storage, Arc::new(DeterministicHashEmbedder))
        .await
        .context("failed to connect to LanceDB")?;

    let git = GitTracker::new(repo);
    let repo_id = compute_repo_id(&git.canonical_repo_identifier());
    let verifier = ToolchainVerifier::new(repo);

    match args.action {
        ConfigCommand::Show { json } => {
            let profile = store.get_repo_profile(&repo_id, scope).await?;
            let detected_toolchains = verifier.detect_toolchains();
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "repo_id": repo_id,
                        "scope_id": scope,
                        "detected_toolchains": detected_toolchains,
                        "profile": profile,
                    }))?
                );
                return Ok(());
            }

            println!("Repository Partition: {} | Scope: {scope}", &repo_id[..repo_id.len().min(12)]);
            if detected_toolchains.is_empty() {
                println!("Detected Toolchains:  none");
            } else {
                let badges: Vec<String> = detected_toolchains
                    .iter()
                    .map(|t| t.display_badge())
                    .collect();
                println!("Detected Toolchains:  {}", badges.join(", "));
            }
            match profile {
                Some(p) => {
                    println!(
                        "Languages Override:   {}",
                        if p.languages_override.is_empty() {
                            "auto".to_string()
                        } else {
                            p.languages_override.join(", ")
                        }
                    );
                    println!(
                        "Check Command:        {}",
                        p.check_command.as_deref().unwrap_or("auto")
                    );
                    println!(
                        "Lint Command:         {}",
                        p.lint_command.as_deref().unwrap_or("auto")
                    );
                    println!(
                        "Test Command:         {}",
                        p.test_command.as_deref().unwrap_or("auto")
                    );
                    println!(
                        "Extra Ignore Globs:   {}",
                        if p.extra_ignore_globs.is_empty() {
                            "none".to_string()
                        } else {
                            p.extra_ignore_globs.join(", ")
                        }
                    );
                    println!(
                        "Auto-Verify Override: {}",
                        p.auto_verify_override
                            .map(|b| b.to_string())
                            .unwrap_or_else(|| "default".to_string())
                    );
                }
                None => {
                    println!("Profile Overrides:    none (using automatic polyglot detection)");
                }
            }
        }
        ConfigCommand::Set {
            check,
            lint,
            test,
            languages,
            ignore_globs,
            auto_verify,
        } => {
            let mut profile = store
                .get_repo_profile(&repo_id, scope)
                .await?
                .unwrap_or_else(|| RepoProfile::empty(&repo_id, scope));

            if let Some(cmd) = check {
                profile.check_command = normalize_optional_cmd(cmd);
            }
            if let Some(cmd) = lint {
                profile.lint_command = normalize_optional_cmd(cmd);
            }
            if let Some(cmd) = test {
                profile.test_command = normalize_optional_cmd(cmd);
            }
            if !languages.is_empty() {
                profile.languages_override = languages;
            }
            if !ignore_globs.is_empty() {
                profile.extra_ignore_globs = ignore_globs;
            }
            if let Some(enabled) = auto_verify {
                profile.auto_verify_override = Some(enabled);
            }

            store.upsert_repo_profile(&profile).await?;
            println!("Saved out-of-tree repository profile in LanceDB (scope: {scope}).");
        }
        ConfigCommand::Reset => {
            store.delete_repo_profile(&repo_id, scope).await?;
            println!("Reset out-of-tree repository profile; restored automatic detection.");
        }
    }

    Ok(())
}

fn normalize_optional_cmd(raw: String) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") || trimmed.eq_ignore_ascii_case("none") {
        None
    } else {
        Some(trimmed.to_string())
    }
}
