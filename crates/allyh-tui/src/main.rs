mod app;
mod cli;
mod highlight;
mod theme;
mod ui;

use std::io;

use allyh_core::{AllyHOrchestrator, HybridRouter, ModelProfile, OrchestratorConfig, UserCommand};
use allyh_env::canonicalize_normalized;
use allyh_memory::StorageConfig;
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
use crate::cli::CliArgs;
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

    let mut cfg = OrchestratorConfig::new(&args.repo);
    cfg.scope_id = args.scope.clone();
    cfg.auto_cargo_check = !args.no_auto_check;
    if let Some(uri) = &args.db_uri {
        cfg.storage = StorageConfig::from_uri(uri.clone());
    }

    let mut router = HybridRouter::from_env();
    if let Some(spec) = &args.model {
        router.set_primary(ModelProfile::parse_spec(spec).context("invalid --model spec")?);
    }
    if let Some(spec) = &args.worker_model {
        router.set_local_worker(
            ModelProfile::parse_spec(spec).context("invalid --worker-model spec")?,
        );
    }

    let orchestrator = AllyHOrchestrator::initialize(cfg, router, None)
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

    // Interactive Control Gate modal intercepts keys first
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
            build_status: "cargo check: PASS (0 warnings)".into(),
            gate_state,
            active_tasks: Vec::new(),
            activated_memories: Vec::new(),
            active_model_label: "API: claude-sonnet-4-6".into(),
            worker_model_label: "Local 9B".into(),
            worker_status: WorkerStatus::Idle,
            tokens_used: 18_400,
            token_budget: 128_000,
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
        assert_eq!(app.command_matches().len(), 2);
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
}
