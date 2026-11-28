mod app;
mod cli;
mod config_cli;
mod highlight;
mod lsp_cli;
mod mcp_cli;
mod sim_cli;
mod skills_cli;
mod theme;
mod ui;
mod update;

use std::io;

use allyh_core::{
    AllyHOrchestrator, HybridRouter, ModelProfile, OrchestratorConfig, ReasoningEffort, UserCommand,
};
use allyh_env::canonicalize_normalized;
use allyh_memory::{EmbeddingProfile, StorageConfig};
use anyhow::{Context, Result};
use clap::Parser;
use crossterm::cursor::Show;
use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use crate::app::{AppState, FocusPane};
use crate::cli::{CliArgs, CliCommand};
use crate::highlight::MarkdownHighlighter;

fn restore_terminal_state() {
    let _ = disable_raw_mode();
    let _ = execute!(io::stdout(), LeaveAlternateScreen, Show);
}

struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
}

impl TerminalGuard {
    fn enter() -> Result<Self> {
        enable_raw_mode().context("enable raw mode")?;
        let mut stdout = io::stdout();
        if let Err(err) = execute!(stdout, EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(err).context("enter alternate screen");
        }
        let backend = CrosstermBackend::new(stdout);
        match Terminal::new(backend) {
            Ok(terminal) => Ok(Self { terminal }),
            Err(err) => {
                restore_terminal_state();
                Err(err).context("create ratatui terminal")
            }
        }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal_state();
        previous(info);
    }));
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(io::stderr)
        .init();

    let args = CliArgs::parse();

    if let Some(command) = args.command {
        return match command {
            CliCommand::Update(update_args) => update::run_cli(update_args).await,
            CliCommand::Skills(skills_args) => {
                skills_cli::run_cli(
                    skills_args,
                    &args.repo,
                    args.db_uri.as_deref(),
                    &args.scope,
                )
                .await
            }
            CliCommand::Sim(sim_args) => {
                sim_cli::run_cli(
                    sim_args,
                    &args.repo,
                    args.db_uri.as_deref(),
                    &args.scope,
                    args.model.as_deref(),
                    args.worker_model.as_deref(),
                )
                .await
            }
            CliCommand::Config(config_args) => {
                config_cli::run_cli(
                    config_args,
                    &args.repo,
                    args.db_uri.as_deref(),
                    &args.scope,
                )
                .await
            }
            CliCommand::Lsp(lsp_args) => lsp_cli::run_cli(lsp_args, &args.repo).await,
            CliCommand::Mcp(mcp_args) => {
                mcp_cli::run_cli(
                    mcp_args,
                    &args.repo,
                    args.db_uri.as_deref(),
                    &args.scope,
                    args.embedding_model.as_deref(),
                )
                .await
            }
        };
    }

    let mut cfg = OrchestratorConfig::new(&args.repo);
    cfg.scope_id = args.scope.clone();
    cfg.auto_verify = !args.no_auto_check;
    if let Some(uri) = &args.db_uri {
        cfg.storage = StorageConfig::from_uri(uri.clone());
    }
    for spec in &args.mcp_servers {
        let parsed = allyh_tools::parse_mcp_server_spec(spec)
            .with_context(|| format!("invalid --mcp-server '{spec}'"))?;
        cfg.extra_mcp_servers.push(parsed);
    }

    let mut router = HybridRouter::from_env();
    if let Some(spec) = &args.model {
        router.set_primary(ModelProfile::parse_spec(spec).context("invalid --model spec")?);
    }
    if let Some(effort_spec) = &args.reasoning_effort {
        let effort = ReasoningEffort::parse_spec(effort_spec)
            .context("invalid --reasoning-effort value")?;
        router.set_reasoning_effort(Some(effort));
    }
    if let Some(spec) = &args.worker_model {
        router.set_local_worker(
            ModelProfile::parse_spec(spec).context("invalid --worker-model spec")?,
        );
    }

    let embedding_profile = match &args.embedding_model {
        Some(spec) => {
            EmbeddingProfile::parse_spec(spec).context("invalid --embedding-model spec")?
        }
        None => EmbeddingProfile::from_env(),
    };
    let embedder = embedding_profile.build_provider();

    let orchestrator = AllyHOrchestrator::initialize(cfg, router, Some(embedder))
        .await
        .context("failed to initialize AllyH orchestrator")?;
    let initial_snapshot = orchestrator
        .inspector_snapshot()
        .await
        .context("failed to build initial inspector snapshot")?;

    let (cmd_tx, mut ui_rx) = orchestrator.spawn_actor();
    let repo_path = canonicalize_normalized(&args.repo);
    let repo_name = repo_path
        .file_name()
        .unwrap_or(repo_path.as_os_str())
        .to_string_lossy();
    let mut app = AppState::new(initial_snapshot, repo_name.into_owned());
    app.repo_root = repo_path;
    let highlighter = MarkdownHighlighter::new();

    install_panic_hook();
    let mut guard = TerminalGuard::enter()?;
    run_tui_loop(
        &mut guard.terminal,
        &mut app,
        &highlighter,
        &cmd_tx,
        &mut ui_rx,
    )
    .await
}

async fn run_tui_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut AppState,
    highlighter: &MarkdownHighlighter,
    cmd_tx: &tokio::sync::mpsc::UnboundedSender<UserCommand>,
    ui_rx: &mut tokio::sync::mpsc::UnboundedReceiver<allyh_core::UiEvent>,
) -> Result<()> {
    let mut event_stream = EventStream::new();
    while !app.should_quit {
        terminal.draw(|frame| ui::render(frame, app, highlighter))?;
        tokio::select! {
            received = ui_rx.recv() => {
                let Some(event) = received else { break };
                app.apply_ui_event(event);
                // Tokens can arrive in bursts. Paint the batch together.
                for _ in 0..127 {
                    let Ok(event) = ui_rx.try_recv() else { break };
                    app.apply_ui_event(event);
                }
            }
            received = event_stream.next() => {
                let Some(event) = received else { break };
                handle_terminal_event(event?, app, cmd_tx);
            }
        }
    }

    Ok(())
}

fn handle_terminal_event(
    event: Event,
    app: &mut AppState,
    cmd_tx: &tokio::sync::mpsc::UnboundedSender<UserCommand>,
) {
    let Event::Key(key) = event else {
        return;
    };
    if key.kind == KeyEventKind::Release {
        return;
    }

    // Global quit shortcut
    if key.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(key.code, KeyCode::Char('c' | 'C'))
    {
        app.should_quit = true;
        let _ = cmd_tx.send(UserCommand::Shutdown);
        return;
    }

    // Interactive Command Approval Gate modal intercepts keys first
    if let Some(pending) = app.pending_command_approval.clone() {
        match key.code {
            KeyCode::Char('a') | KeyCode::Char('A') | KeyCode::Enter => {
                app.pending_command_approval = None;
                let _ = cmd_tx.send(UserCommand::ApproveCommandOnce(pending.command));
            }
            KeyCode::Char('p') | KeyCode::Char('P') => {
                app.pending_command_approval = None;
                let _ = cmd_tx.send(UserCommand::AllowCommandPattern(pending.suggested_pattern));
            }
            KeyCode::Char('d') | KeyCode::Char('D') | KeyCode::Esc => {
                app.pending_command_approval = None;
                let _ = cmd_tx.send(UserCommand::DenyCommand(pending.command));
            }
            _ => {}
        }
        return;
    }

    // Interactive Control Gate modal intercepts keys next
    if app.is_gate_modal_open() {
        match key.code {
            KeyCode::Char('a') | KeyCode::Char('A') | KeyCode::Enter => {
                let _ = cmd_tx.send(UserCommand::ApprovePlan);
            }
            KeyCode::Char('r') | KeyCode::Char('R') | KeyCode::Esc => {
                let _ = cmd_tx.send(UserCommand::RefinePlan(
                    "User requested refinement".to_string(),
                ));
            }
            other => app.plan_scroll.navigate(other),
        }
        return;
    }

    if (key.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(key.code, KeyCode::Char('t' | 'T')))
        || key.code == KeyCode::F(2)
    {
        app.toggle_tool_cards();
        return;
    }

    let commands = app.command_matches();
    if let Some(&choice) = commands.get(app.command_index) {
        match key.code {
            KeyCode::Up => app.command_index = app.command_index.saturating_sub(1),
            KeyCode::Down => app.command_index = (app.command_index + 1).min(commands.len() - 1),
            KeyCode::Tab if !key.modifiers.contains(KeyModifiers::SHIFT) => {
                app.complete_command(choice);
            }
            KeyCode::Enter
                if key.modifiers.is_empty()
                    && !app.input.lines()[0].eq_ignore_ascii_case(choice.name) =>
            {
                app.complete_command(choice);
            }
            KeyCode::Esc => app.commands_dismissed = true,
            _ => return handle_pane_key(key, app, cmd_tx),
        }
        return;
    }
    handle_pane_key(key, app, cmd_tx);
}

fn handle_pane_key(
    key: KeyEvent,
    app: &mut AppState,
    cmd_tx: &tokio::sync::mpsc::UnboundedSender<UserCommand>,
) {
    match key.code {
        KeyCode::Tab | KeyCode::BackTab => {
            app.cycle_focus(
                key.code == KeyCode::BackTab || key.modifiers.contains(KeyModifiers::SHIFT),
            );
            return;
        }
        KeyCode::Esc if app.focus != FocusPane::Input => {
            app.focus = FocusPane::Input;
            return;
        }
        _ => {}
    }

    match app.focus {
        FocusPane::Memories => match key.code {
            KeyCode::Up | KeyCode::Char('k') => app.select_prev_memory(),
            KeyCode::Down | KeyCode::Char('j') => app.select_next_memory(),
            KeyCode::Char('p') => {
                if let Some(id) = app.selected_memory_id() {
                    let _ = cmd_tx.send(UserCommand::PinMemory(id));
                }
            }
            KeyCode::Char('x') => {
                if let Some(id) = app.selected_memory_id() {
                    let _ = cmd_tx.send(UserCommand::InvalidateMemory {
                        decision_id: id,
                        reason: Some("invalidated from Coherence Inspector sidebar".to_string()),
                    });
                }
            }
            _ => {}
        },
        FocusPane::Chat => app.chat_scroll.navigate(key.code),
        FocusPane::Input => match key.code {
            KeyCode::Enter
                if !key.modifiers.contains(KeyModifiers::ALT)
                    && !key.modifiers.contains(KeyModifiers::SHIFT)
                    && !key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                let text = app.take_input_text();
                if let Some(cmd) = app.parse_submission(&text) {
                    let _ = cmd_tx.send(cmd);
                }
            }
            KeyCode::Enter => {
                app.input.insert_newline();
            }
            KeyCode::Char('j' | 'J') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.input.insert_newline();
            }
            _ => {
                if app.input.input(key) {
                    app.command_index = 0;
                    app.commands_dismissed = false;
                }
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use allyh_core::{ControlGateState, InspectorSnapshot};
    use allyh_memory::WorkerStatus;
    use ratatui::backend::TestBackend;

    fn sample_snapshot(gate_state: ControlGateState) -> InspectorSnapshot {
        InspectorSnapshot {
            repo_id: "abcdef1234567890".into(),
            scope_id: "default".into(),
            storage_uri: "~/.local/share/allyh/db".into(),
            storage_backend_label: "LanceDB: Local".into(),
            git_branch: "master".into(),
            dirty_files: vec!["src/lib.rs".into()],
            out_of_band_edits: Vec::new(),
            detected_languages: vec!["Rust".into()],
            detected_toolchains: vec!["Rust (cargo)".into()],
            active_lsp_servers: Vec::new(),
            active_checkpoint_label: None,
            mcp_servers_count: 0,
            build_status: "cargo check: PASS (0 warnings)".into(),
            gate_state,
            active_tasks: Vec::new(),
            activated_memories: Vec::new(),
            skills_count: 0,
            active_harness_label: "seed (default)".into(),
            sim_problems_count: 0,
            sim_run_id: None,
            active_model_label: "API: claude-sonnet-4-6".into(),
            worker_model_label: "Local 9B".into(),
            worker_status: WorkerStatus::Idle,
            tokens_used: 18_400,
            token_budget: 128_000,
            session_input_tokens: 0,
            session_output_tokens: 0,
            session_cached_tokens: 0,
            session_cost_usd: 0.0,
        }
    }

    fn screen(
        app: &mut AppState,
        highlighter: &MarkdownHighlighter,
        width: u16,
        height: u16,
    ) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal
            .draw(|frame| ui::render(frame, app, highlighter))
            .unwrap();
        terminal
            .backend()
            .buffer()
            .content()
            .chunks(width as usize)
            .map(|row| row.iter().map(|cell| cell.symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn press(
        app: &mut AppState,
        tx: &tokio::sync::mpsc::UnboundedSender<UserCommand>,
        code: KeyCode,
    ) {
        handle_terminal_event(Event::Key(KeyEvent::new(code, KeyModifiers::NONE)), app, tx);
    }

    #[test]
    fn test_slash_commands_and_tui_rendering() {
        let mut app = AppState::new(sample_snapshot(ControlGateState::Discovery), "allyh".into());

        assert!(matches!(
            app.parse_submission("/db-status"),
            Some(UserCommand::RequestDbStatus)
        ));
        assert!(matches!(
            app.parse_submission("/memories"),
            Some(UserCommand::RequestMemories)
        ));
        assert!(matches!(
            app.parse_submission("/model local:qwen-35b"),
            Some(UserCommand::SwitchModel(ref s)) if s == "local:qwen-35b"
        ));
        assert!(matches!(
            app.parse_submission("/plan Add RepoGraph"),
            Some(UserCommand::ProposePlan(_))
        ));

        let highlighter = MarkdownHighlighter::new();
        let base_screen = screen(&mut app, &highlighter, 120, 36);
        assert!(base_screen.contains("Workspace"));
        assert!(base_screen.contains("Tasks"));
        assert!(base_screen.contains("Memories"));
        assert!(base_screen.contains("LanceDB: Local"));

        app.inspector.gate_state = ControlGateState::PlanApproval {
            plan: AllyHOrchestrator::build_quick_plan("Add RepoGraph"),
        };
        let modal_screen = screen(&mut app, &highlighter, 120, 36);
        assert!(modal_screen.contains("Approve Plan"));
        assert!(modal_screen.contains("Refine"));
        assert!(!modal_screen.contains("Enter send"));
    }

    #[test]
    fn welcome_art_fits_the_pane_and_yields_to_the_conversation() {
        let mut app = AppState::new(sample_snapshot(ControlGateState::Discovery), "demo".into());
        let highlighter = MarkdownHighlighter::new();
        for (width, height, artwork) in [
            (160, 42, include_str!("../assets/logo-wide.txt")),
            (80, 24, include_str!("../assets/logo-standard.txt")),
            (100, 24, include_str!("../assets/logo-compact.txt")),
        ] {
            let welcome = screen(&mut app, &highlighter, width, height);
            assert!(
                artwork.lines().all(|row| welcome.contains(row)),
                "Incomplete logo at {width}×{height}"
            );
            assert!(welcome.contains("/help") && welcome.contains("Enter send"));
        }
        for (width, height) in [(40, 24), (80, 14)] {
            let welcome = screen(&mut app, &highlighter, width, height);
            assert!(welcome.contains("AllyH HARNESS"));
            assert!(welcome.contains("/help"));
        }

        app.parse_submission("Hello");
        let chat = screen(&mut app, &highlighter, 80, 24);
        assert!(chat.contains("Hello") && !chat.contains("H  A  R  N  E  S  S"));
    }

    #[test]
    fn compact_focus_keeps_the_inspector_reachable_and_the_draft_intact() {
        let mut app = AppState::new(sample_snapshot(ControlGateState::Discovery), "demo".into());
        let h = MarkdownHighlighter::new();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        app.input.insert_str("first line");
        handle_terminal_event(
            Event::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT)),
            &mut app,
            &tx,
        );
        app.input.insert_str("second line");

        let compact = screen(&mut app, &h, 80, 24);
        assert!(compact.contains("Conversation") && !compact.contains("Inspector"));
        assert!(compact.contains("first line") && compact.contains("second line"));
        press(&mut app, &tx, KeyCode::Tab);
        let inspector = screen(&mut app, &h, 80, 24);
        assert!(inspector.contains("Workspace") && inspector.contains("No active decisions"));
        press(&mut app, &tx, KeyCode::Esc);
        assert_eq!(app.focus, FocusPane::Input);
        press(&mut app, &tx, KeyCode::Enter);
        assert!(
            matches!(rx.try_recv(), Ok(UserCommand::SendPrompt(text)) if text == "first line\nsecond line")
        );
    }

    #[test]
    fn slash_completion_filters_navigates_and_waits_for_submission() {
        let mut app = AppState::new(sample_snapshot(ControlGateState::Discovery), "demo".into());
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        app.input.insert_str("/m");
        assert_eq!(app.command_matches().len(), 3);
        press(&mut app, &tx, KeyCode::Down);
        press(&mut app, &tx, KeyCode::Tab);
        assert_eq!(app.input.lines(), &["/memories"]);
        assert!(app.command_matches().is_empty() && rx.try_recv().is_err());
        press(&mut app, &tx, KeyCode::Enter);
        assert!(matches!(rx.try_recv(), Ok(UserCommand::RequestMemories)));

        app.input.insert_str("/pl");
        press(&mut app, &tx, KeyCode::Enter);
        assert_eq!(app.input.lines(), &["/plan "]);
        assert!(rx.try_recv().is_err());
        app.input.insert_str("navigation");
        press(&mut app, &tx, KeyCode::Enter);
        assert!(
            matches!(rx.try_recv(), Ok(UserCommand::ProposePlan(plan)) if plan.title == "navigation")
        );

        app.input.insert_str("/");
        press(&mut app, &tx, KeyCode::Esc);
        assert!(app.command_matches().is_empty());
        assert_eq!(app.input.lines(), &["/"]);
    }

    #[test]
    fn reading_wrapped_history_survives_stream_updates_and_end_resumes_follow() {
        let mut app = AppState::new(sample_snapshot(ControlGateState::Discovery), "demo".into());
        let h = MarkdownHighlighter::new();
        let (tx, _) = tokio::sync::mpsc::unbounded_channel();
        app.apply_ui_event(allyh_core::UiEvent::TokenDelta(format!(
            "START {} ENDPOINT",
            "wrapped words ".repeat(80)
        )));
        assert!(screen(&mut app, &h, 42, 14).contains("ENDPOINT"));
        press(&mut app, &tx, KeyCode::BackTab);
        press(&mut app, &tx, KeyCode::Home);
        let start = screen(&mut app, &h, 42, 14);
        assert!(start.contains("START") && !start.contains("ENDPOINT"));
        app.apply_ui_event(allyh_core::UiEvent::TokenDelta(" new tail".into()));
        assert_eq!(start, screen(&mut app, &h, 42, 14));
        press(&mut app, &tx, KeyCode::PageDown);
        assert!(app.chat_scroll.top > 1);
        press(&mut app, &tx, KeyCode::End);
        assert!(screen(&mut app, &h, 42, 14).contains("new tail"));
    }

    #[test]
    fn long_plan_keeps_actions_visible_and_scrolls_to_the_final_criterion() {
        let mut plan = AllyHOrchestrator::build_quick_plan("Navigation");
        plan.affected_files.push("src/lib.rs".into());
        plan.tasks[0].acceptance_criteria = (1..=30).map(|n| format!("Criterion {n}")).collect();
        let mut app = AppState::new(
            sample_snapshot(ControlGateState::PlanApproval { plan }),
            "demo".into(),
        );
        let h = MarkdownHighlighter::new();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let first = screen(&mut app, &h, 42, 18);
        assert!(
            first.contains("src/lib.rs")
                && first.contains("Enable writes")
                && first.contains("Refine")
        );
        press(&mut app, &tx, KeyCode::End);
        let last = screen(&mut app, &h, 42, 18);
        assert!(
            last.contains("Criterion 30")
                && last.contains("Enable writes")
                && last.contains("Refine")
        );
        assert!(rx.try_recv().is_err());
        press(&mut app, &tx, KeyCode::Esc);
        assert!(matches!(rx.try_recv(), Ok(UserCommand::RefinePlan(_))));
        press(&mut app, &tx, KeyCode::Char('a'));
        assert!(matches!(rx.try_recv(), Ok(UserCommand::ApprovePlan)));
    }

    #[test]
    fn cross_platform_key_events_support_repeat_ctrl_j_and_ignore_release() {
        let mut app = AppState::new(sample_snapshot(ControlGateState::Discovery), "demo".into());
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        // Key repeat (emitted on Windows & Kitty-protocol terminals) must work
        handle_terminal_event(
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Char('a'),
                KeyModifiers::NONE,
                KeyEventKind::Repeat,
            )),
            &mut app,
            &tx,
        );
        // Key release (emitted on Windows) must be ignored
        handle_terminal_event(
            Event::Key(KeyEvent::new_with_kind(
                KeyCode::Char('z'),
                KeyModifiers::NONE,
                KeyEventKind::Release,
            )),
            &mut app,
            &tx,
        );
        // Ctrl+J inserts a newline on terminals where Alt+Enter is intercepted
        handle_terminal_event(
            Event::Key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::CONTROL)),
            &mut app,
            &tx,
        );
        app.input.insert_str("b");
        press(&mut app, &tx, KeyCode::Enter);
        assert!(
            matches!(rx.try_recv(), Ok(UserCommand::SendPrompt(text)) if text == "a\nb")
        );

        // Uppercase Ctrl+C (e.g. with Caps Lock on Windows) still quits cleanly
        handle_terminal_event(
            Event::Key(KeyEvent::new(KeyCode::Char('C'), KeyModifiers::CONTROL)),
            &mut app,
            &tx,
        );
        assert!(app.should_quit);
        assert!(matches!(rx.try_recv(), Ok(UserCommand::Shutdown)));
    }

    #[test]
    fn test_skills_slash_commands_and_workspace_count() {
        use std::collections::HashSet;

        use allyh_core::UiEvent;
        use allyh_memory::SkillRecord;
        use chrono::Utc;

        let mut snap = sample_snapshot(ControlGateState::Discovery);
        snap.skills_count = 3;
        let mut app = AppState::new(snap, "allyh".into());

        // Slash command parsing for all /skills variants
        assert!(matches!(
            app.parse_submission("/skills"),
            Some(UserCommand::RequestSkills)
        ));
        assert!(matches!(
            app.parse_submission("/skills list"),
            Some(UserCommand::RequestSkills)
        ));
        assert!(matches!(
            app.parse_submission("/skills show release-flow"),
            Some(UserCommand::ShowSkill(ref s)) if s == "release-flow"
        ));
        assert!(matches!(
            app.parse_submission("/skill release-flow"),
            Some(UserCommand::ShowSkill(ref s)) if s == "release-flow"
        ));
        assert!(matches!(
            app.parse_submission("/skills pin release-flow"),
            Some(UserCommand::PinSkill(ref s)) if s == "release-flow"
        ));
        assert!(matches!(
            app.parse_submission("/skills remove release-flow"),
            Some(UserCommand::RemoveSkill(ref s)) if s == "release-flow"
        ));

        // Importing a SKILL.md from disk produces UserCommand::UpsertSkill
        let unique = format!(
            "allyh-skill-import-test-{}.md",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let temp_skill = std::env::temp_dir().join(unique);
        std::fs::write(
            &temp_skill,
            "---\nname: imported-skill\ndescription: Imported from file\ntriggers:\n  - import test\n---\nDo step one.\n",
        )
        .expect("write temp skill");
        let import_cmd = app.parse_submission(&format!("/skills import {}", temp_skill.display()));
        let _ = std::fs::remove_file(&temp_skill);
        assert!(matches!(
            import_cmd,
            Some(UserCommand::UpsertSkill { ref name, ref triggers, .. })
                if name == "imported-skill" && triggers == &vec!["import test".to_string()]
        ));

        // Render SkillsList + Workspace count in Inspector
        let skill = SkillRecord {
            id: "r:s:release-flow".into(),
            repo_id: "r".into(),
            scope_id: "s".into(),
            name: "release-flow".into(),
            description: "Release checklist".into(),
            instructions: "1. Run cargo test.".into(),
            triggers: vec!["release".into()],
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let mut pinned_ids = HashSet::new();
        pinned_ids.insert(skill.id.clone());
        app.apply_ui_event(UiEvent::SkillsList {
            skills: vec![skill],
            pinned_ids,
        });

        let highlighter = MarkdownHighlighter::new();
        let rendered = screen(&mut app, &highlighter, 120, 36);
        assert!(rendered.contains("Registered Skills (1)"));
        assert!(rendered.contains("release-flow"));
        assert!(rendered.contains("[PINNED]"));
        assert!(rendered.contains("Skills"));
        assert!(rendered.contains("3"));
    }

    #[test]
    fn test_sim_slash_commands_and_inspector_rendering() {
        use allyh_core::UiEvent;
        use allyh_memory::{SimAttemptInput, SimLedger, SimMemoryMode, SimProblemRef};

        let mut snap = sample_snapshot(ControlGateState::Discovery);
        snap.active_harness_label = "tb4-r2 (r2)".into();
        snap.sim_problems_count = 2;
        let mut app = AppState::new(snap, "allyh".into());

        assert!(matches!(
            app.parse_submission("/sim"),
            Some(UserCommand::RequestSimSummary)
        ));
        assert!(matches!(
            app.parse_submission("/sim list"),
            Some(UserCommand::RequestSimProblems)
        ));
        assert!(matches!(
            app.parse_submission("/sim show P3"),
            Some(UserCommand::ShowSimProblem(ref id)) if id == "P3"
        ));
        assert!(matches!(
            app.parse_submission("/sim P5"),
            Some(UserCommand::ShowSimProblem(ref id)) if id == "P5"
        ));
        assert!(matches!(
            app.parse_submission("/sim harness"),
            Some(UserCommand::ShowActiveHarness)
        ));
        assert!(matches!(
            app.parse_submission("/sim use hv-1234"),
            Some(UserCommand::UseHarness(ref id)) if id == "hv-1234"
        ));

        let mut ledger = SimLedger::new_source(
            "r",
            "s",
            "target-tb4",
            "tb4",
            "tb4",
            "anthropic:claude-sonnet-4-6",
            "anthropic:claude-sonnet-4-6",
            "hv-0",
            SimMemoryMode::FullSim,
        );
        ledger
            .record_accepted_attempt(SimAttemptInput {
                problem_ref: SimProblemRef::New("premature stop".into()),
                round: 1,
                sample: "s1".into(),
                accepted: true,
                backfilled: false,
                what: "Enable interlock".into(),
                files: vec!["interlock.rs".into()],
                predicted: vec!["t1".into()],
                fixed: vec!["t1".into()],
                missed: vec![],
                passed: 1,
                total: 1,
                score: 100.0,
                domain: "tb4".into(),
                benchmark: "tb4".into(),
                policy_model: "anthropic:claude-sonnet-4-6".into(),
                harness_version_id: "hv-1".into(),
            })
            .expect("record");

        app.apply_ui_event(UiEvent::SimProblems(Some(ledger)));
        let highlighter = MarkdownHighlighter::new();
        let rendered = screen(&mut app, &highlighter, 120, 36);
        assert!(rendered.contains("SIM Problem Ledger (1)"));
        assert!(rendered.contains("premature stop"));
        assert!(rendered.contains("tb4-r2 (r2)"));
        assert!(rendered.contains("2 problem(s)"));
    }

    #[test]
    fn test_config_slash_commands_and_polyglot_inspector_display() {
        let mut snap = sample_snapshot(ControlGateState::Discovery);
        snap.detected_languages = vec!["Rust".into(), "TypeScript".into()];
        snap.detected_toolchains = vec!["Rust (cargo)".into(), "TypeScript/JS (web)".into()];
        let mut app = AppState::new(snap, "monorepo".into());

        assert!(matches!(
            app.parse_submission("/config"),
            Some(UserCommand::RequestRepoProfile)
        ));
        assert!(matches!(
            app.parse_submission("/config set-check npx tsc --noEmit"),
            Some(UserCommand::SaveRepoProfile(ref p))
                if p.check_command.as_deref() == Some("npx tsc --noEmit")
        ));
        assert!(matches!(
            app.parse_submission("/config reset"),
            Some(UserCommand::ResetRepoProfile)
        ));

        let highlighter = MarkdownHighlighter::new();
        let rendered = screen(&mut app, &highlighter, 120, 36);
        assert!(rendered.contains("Stack"));
        assert!(rendered.contains("Rust, TypeScript"));
        assert!(rendered.contains("Tools"));
    }

    #[test]
    fn test_lsp_slash_commands_and_inspector_display() {
        use allyh_core::UiEvent;
        use allyh_env::{LspServerState, LspServerStatus};

        let mut snap = sample_snapshot(ControlGateState::Discovery);
        snap.active_lsp_servers = vec!["rust-analyzer (running)".into()];
        let mut app = AppState::new(snap, "allyh".into());

        assert!(matches!(
            app.parse_submission("/lsp"),
            Some(UserCommand::RequestLspStatus)
        ));
        assert!(matches!(
            app.parse_submission("/lsp status"),
            Some(UserCommand::RequestLspStatus)
        ));
        assert!(matches!(
            app.parse_submission("/lsp restart"),
            Some(UserCommand::RestartLsp(None))
        ));
        assert!(matches!(
            app.parse_submission("/lsp restart rust"),
            Some(UserCommand::RestartLsp(Some(ref s))) if s == "rust"
        ));

        app.apply_ui_event(UiEvent::LspStatus(vec![LspServerStatus {
            family: "rust".into(),
            server_name: "rust-analyzer".into(),
            command: "rust-analyzer".into(),
            state: LspServerState::Running,
            open_documents: 2,
            cached_diagnostic_files: 1,
            restart_count: 0,
            last_error: None,
        }]));

        let highlighter = MarkdownHighlighter::new();
        let rendered = screen(&mut app, &highlighter, 120, 36);
        assert!(rendered.contains("LSP Semantic Bridge (1)"));
        assert!(rendered.contains("rust-analyzer (running)"));
    }

    #[test]
    fn test_checkpoint_slash_commands_and_command_approval_modal() {
        use allyh_core::UiEvent;
        use allyh_env::CheckpointDiffReport;
        use allyh_tools::PendingCommandApproval;

        let mut snap = sample_snapshot(ControlGateState::Execution {
            active_task_id: None,
        });
        snap.active_checkpoint_label = Some("Refactor API".into());
        let mut app = AppState::new(snap, "allyh".into());
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let highlighter = MarkdownHighlighter::new();

        assert!(matches!(
            app.parse_submission("/diff"),
            Some(UserCommand::RequestCheckpointDiff)
        ));
        assert!(matches!(
            app.parse_submission("/rollback"),
            Some(UserCommand::RollbackCheckpoint)
        ));

        app.apply_ui_event(UiEvent::CheckpointDiff(Some(CheckpointDiffReport {
            checkpoint_id: "cp-1".into(),
            checkpoint_label: "Refactor API".into(),
            modified_files: vec!["src/lib.rs".into()],
            created_files: Vec::new(),
            deleted_files: Vec::new(),
            insertions: 1,
            deletions: 1,
            unified_diff: "-old\n+new\n".into(),
        })));
        let diff_screen = screen(&mut app, &highlighter, 120, 36);
        assert!(diff_screen.contains("Checkpoint Diff"));
        assert!(diff_screen.contains("Checkpt"));

        // Command Approval Modal renders & handles Allow Once / Allow Pattern / Deny
        app.apply_ui_event(UiEvent::CommandApprovalRequested(PendingCommandApproval {
            command: "cargo publish --dry-run".into(),
            working_dir: ".".into(),
            reason: "Mutating or network command".into(),
            suggested_pattern: "cargo publish".into(),
        }));
        assert!(app.is_gate_modal_open());
        let modal = screen(&mut app, &highlighter, 120, 36);
        assert!(modal.contains("Command Approval Gate"));
        assert!(modal.contains("cargo publish --dry-run"));
        assert!(modal.contains("Allow Pattern"));

        press(&mut app, &tx, KeyCode::Char('p'));
        assert!(!app.is_gate_modal_open());
        assert!(matches!(
            rx.try_recv(),
            Ok(UserCommand::AllowCommandPattern(ref pat)) if pat == "cargo publish"
        ));
    }

    #[test]
    fn test_mcp_slash_commands_and_inspector_display() {
        use allyh_core::UiEvent;
        use allyh_memory::McpServerTransport;
        use allyh_tools::McpServerStatus;

        let mut snap = sample_snapshot(ControlGateState::Discovery);
        snap.mcp_servers_count = 2;
        let mut app = AppState::new(snap, "allyh".into());

        assert!(matches!(
            app.parse_submission("/mcp"),
            Some(UserCommand::RequestMcpStatus)
        ));
        assert!(matches!(
            app.parse_submission("/mcp status"),
            Some(UserCommand::RequestMcpStatus)
        ));
        assert!(matches!(
            app.parse_submission("/mcp add docs https://mcp.example.com/rpc"),
            Some(UserCommand::AddMcpServer(ref cfg))
                if cfg.name == "docs"
                && matches!(cfg.transport, McpServerTransport::Http { ref url, .. } if url == "https://mcp.example.com/rpc")
        ));
        assert!(matches!(
            app.parse_submission("/mcp remove docs"),
            Some(UserCommand::RemoveMcpServer(ref name)) if name == "docs"
        ));

        app.apply_ui_event(UiEvent::McpStatus(vec![McpServerStatus {
            name: "docs".into(),
            transport: "http".into(),
            target: "https://mcp.example.com/rpc".into(),
            connected: true,
            tool_count: 2,
            tools: vec!["mcp__docs__search".into(), "mcp__docs__fetch".into()],
            last_error: None,
        }]));

        let highlighter = MarkdownHighlighter::new();
        let rendered = screen(&mut app, &highlighter, 120, 36);
        assert!(rendered.contains("2 connected"));
        assert!(rendered.contains("External MCP Servers (1)"));
        assert!(rendered.contains("mcp__docs__search"));
    }

    #[test]
    fn test_status_bar_displays_prompt_cache_and_cost_telemetry() {
        let mut snap = sample_snapshot(ControlGateState::Discovery);
        snap.session_input_tokens = 20_000;
        snap.session_output_tokens = 1_500;
        snap.session_cached_tokens = 15_000;
        snap.session_cost_usd = 0.034;

        let mut app = AppState::new(snap, "allyh".into());
        let highlighter = MarkdownHighlighter::new();
        let rendered = screen(&mut app, &highlighter, 120, 36);
        assert!(
            rendered.contains("Cache 75%"),
            "status bar should display prompt cache hit ratio, got:\n{rendered}"
        );
        assert!(
            rendered.contains("$0.034"),
            "status bar should display estimated session USD cost, got:\n{rendered}"
        );
    }
}
