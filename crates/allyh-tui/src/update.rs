//! Self-update support for the `allyh` binary via GitHub Releases.

use std::fmt;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use self_update::VersionStatus;
use self_update::backends::github;
use self_update::cargo_crate_version;

use crate::cli::UpdateArgs;

/// Default GitHub repository owner for AllyH releases.
pub const DEFAULT_REPO_OWNER: &str = "alexvegadev";
/// Default GitHub repository name for AllyH releases.
pub const DEFAULT_REPO_NAME: &str = "allyh";
/// Installed executable name matched inside release archives.
pub const BIN_NAME: &str = "allyh";
/// Compile-time version of the running `allyh` binary.
pub const CURRENT_VERSION: &str = cargo_crate_version!();

const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const REQUEST_RETRIES: u32 = 2;

/// Validated configuration for checking or installing an AllyH self-update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateConfig {
    pub repo_owner: String,
    pub repo_name: String,
    pub bin_name: String,
    pub current_version: String,
    pub target_tag: Option<String>,
    pub auth_token: Option<String>,
    pub check_only: bool,
    pub no_confirm: bool,
    pub show_progress: bool,
}

impl UpdateConfig {
    /// Build a validated [`UpdateConfig`] from parsed CLI arguments.
    pub fn from_args(args: UpdateArgs) -> Result<Self> {
        let repo_owner = args.repo_owner.trim().to_owned();
        if repo_owner.is_empty() {
            bail!("update repository owner cannot be empty");
        }

        let repo_name = args.repo_name.trim().to_owned();
        if repo_name.is_empty() {
            bail!("update repository name cannot be empty");
        }

        let target_tag = match args.tag {
            Some(raw) => {
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    bail!("--tag cannot be empty");
                }
                Some(trimmed.to_owned())
            }
            None => None,
        };

        let auth_token = args
            .token
            .map(|t| t.trim().to_owned())
            .filter(|t| !t.is_empty());

        Ok(Self {
            repo_owner,
            repo_name,
            bin_name: BIN_NAME.to_owned(),
            current_version: CURRENT_VERSION.to_owned(),
            target_tag,
            auth_token,
            check_only: args.check,
            no_confirm: args.yes,
            show_progress: !args.check,
        })
    }
}

/// Outcome of an update check or installation attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpdateOutcome {
    /// The running binary is already at the newest available version.
    UpToDate { current: String },
    /// A newer release is available (returned in `--check` mode).
    UpdateAvailable { current: String, latest: String },
    /// The binary was successfully replaced with a newer release.
    Updated { from: String, to: String },
}

impl fmt::Display for UpdateOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UpToDate { current } => {
                write!(f, "{BIN_NAME} v{current} is already up to date.")
            }
            Self::UpdateAvailable { current, latest } => write!(
                f,
                "Update available for {BIN_NAME}: v{current} -> v{latest}. Run `allyh update` to install."
            ),
            Self::Updated { from, to } => {
                write!(f, "Updated {BIN_NAME}: v{from} -> v{to}.")
            }
        }
    }
}

/// Execute `allyh update` from the CLI and print a human-readable result.
pub async fn run_cli(args: UpdateArgs) -> Result<()> {
    let config = UpdateConfig::from_args(args)?;
    let outcome = if config.check_only {
        check_for_update(&config).await?
    } else {
        install_update(&config).await?
    };
    println!("{outcome}");
    Ok(())
}

/// Query GitHub Releases for a newer version without modifying the installed binary.
pub async fn check_for_update(config: &UpdateConfig) -> Result<UpdateOutcome> {
    let updater = build_updater(config, false)?;
    if let Some(tag) = &config.target_tag {
        let release = updater
            .get_release_version_async(tag)
            .await
            .with_context(|| {
                format!(
                    "failed to fetch release `{tag}` for {}/{}",
                    config.repo_owner, config.repo_name
                )
            })?;
        return Ok(if release.version() == config.current_version {
            UpdateOutcome::UpToDate {
                current: config.current_version.clone(),
            }
        } else {
            UpdateOutcome::UpdateAvailable {
                current: config.current_version.clone(),
                latest: release.version().to_owned(),
            }
        });
    }

    if let Some(latest) = updater
        .is_update_available_async()
        .await
        .with_context(|| {
            format!(
                "failed to check releases for {}/{}",
                config.repo_owner, config.repo_name
            )
        })?
    {
        return Ok(UpdateOutcome::UpdateAvailable {
            current: config.current_version.clone(),
            latest: latest.version().to_owned(),
        });
    }

    Ok(UpdateOutcome::UpToDate {
        current: config.current_version.clone(),
    })
}

/// Download and install the target (or latest) release from GitHub Releases.
pub async fn install_update(config: &UpdateConfig) -> Result<UpdateOutcome> {
    let updater = build_updater(config, true)?;
    let status = updater.update_async().await.with_context(|| {
        format!(
            "failed to update {BIN_NAME} from {}/{}",
            config.repo_owner, config.repo_name
        )
    })?;

    Ok(outcome_from_status(&config.current_version, status))
}

fn build_updater(config: &UpdateConfig, install_mode: bool) -> Result<github::AsyncUpdate> {
    let mut builder = github::Update::configure();
    builder
        .repo_owner(&config.repo_owner)
        .repo_name(&config.repo_name)
        .bin_name(&config.bin_name)
        .current_version(&config.current_version)
        .show_output(install_mode)
        .show_download_progress(install_mode && config.show_progress)
        .no_confirm(config.no_confirm)
        .check_install_path_writable(install_mode)
        .timeout(REQUEST_TIMEOUT)
        .retries(REQUEST_RETRIES);

    if let Some(tag) = &config.target_tag {
        builder.release_tag(tag);
    }
    if let Some(token) = &config.auth_token {
        builder.auth_token(token);
    } else {
        builder.auth_token_from_env();
    }

    builder
        .build_async()
        .context("failed to configure GitHub self-updater")
}

fn outcome_from_status(current: &str, status: VersionStatus) -> UpdateOutcome {
    match status {
        VersionStatus::UpToDate(version) => UpdateOutcome::UpToDate { current: version },
        VersionStatus::Updated(version) => UpdateOutcome::Updated {
            from: current.to_owned(),
            to: version,
        },
        other => UpdateOutcome::Updated {
            from: current.to_owned(),
            to: other.version().to_owned(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_args() -> UpdateArgs {
        UpdateArgs {
            check: false,
            yes: false,
            tag: None,
            repo_owner: DEFAULT_REPO_OWNER.to_owned(),
            repo_name: DEFAULT_REPO_NAME.to_owned(),
            token: None,
        }
    }

    #[test]
    fn config_validates_and_normalizes_inputs() {
        let mut args = sample_args();
        args.check = true;
        args.yes = true;
        args.tag = Some("  v0.2.0  ".to_owned());
        args.token = Some("   ".to_owned());

        let cfg = UpdateConfig::from_args(args).expect("valid config");
        assert_eq!(cfg.repo_owner, DEFAULT_REPO_OWNER);
        assert_eq!(cfg.repo_name, DEFAULT_REPO_NAME);
        assert_eq!(cfg.bin_name, BIN_NAME);
        assert_eq!(cfg.current_version, CURRENT_VERSION);
        assert_eq!(cfg.target_tag.as_deref(), Some("v0.2.0"));
        assert!(cfg.auth_token.is_none());
        assert!(cfg.check_only);
        assert!(cfg.no_confirm);
        assert!(!cfg.show_progress);
    }

    #[test]
    fn config_rejects_blank_fields() {
        let mut blank_owner = sample_args();
        blank_owner.repo_owner = "   ".to_owned();
        assert!(UpdateConfig::from_args(blank_owner).is_err());

        let mut blank_repo = sample_args();
        blank_repo.repo_name = String::new();
        assert!(UpdateConfig::from_args(blank_repo).is_err());

        let mut blank_tag = sample_args();
        blank_tag.tag = Some(" ".to_owned());
        assert!(UpdateConfig::from_args(blank_tag).is_err());
    }

    #[test]
    fn builder_constructs_async_updater_without_network() {
        let cfg = UpdateConfig::from_args(sample_args()).expect("valid config");
        let updater = build_updater(&cfg, false);
        assert!(updater.is_ok(), "unexpected error: {updater:?}");
    }

    #[test]
    fn formats_update_outcomes_cleanly() {
        assert_eq!(
            UpdateOutcome::UpToDate {
                current: "0.1.0".into(),
            }
            .to_string(),
            "allyh v0.1.0 is already up to date."
        );
        assert_eq!(
            UpdateOutcome::UpdateAvailable {
                current: "0.1.0".into(),
                latest: "0.2.0".into(),
            }
            .to_string(),
            "Update available for allyh: v0.1.0 -> v0.2.0. Run `allyh update` to install."
        );
        assert_eq!(
            UpdateOutcome::Updated {
                from: "0.1.0".into(),
                to: "0.2.0".into(),
            }
            .to_string(),
            "Updated allyh: v0.1.0 -> v0.2.0."
        );
    }
}
