use allyh_core::{AllyHOrchestrator, ControlGateState, InspectorSnapshot, UiEvent, UserCommand};
use allyh_memory::WorkerStatus;
use allyh_tools::ToolExecutionRecord;
use crossterm::event::KeyCode;
use ratatui::style::Style;
use tui_textarea::TextArea;

use crate::theme::{MUTED, TEXT, ink};

pub struct SlashCommand {
    pub name: &'static str,
    pub argument: &'static str,
    pub description: &'static str,
}

pub const COMMANDS: &[SlashCommand] = &[
    SlashCommand::new("/plan", "<goal>", "Propose a change for approval"),
    SlashCommand::new("/model", "[spec]", "Show or switch the reasoning model"),
    SlashCommand::new("/memories", "", "List active project decisions"),
    SlashCommand::new("/pin", "[id]", "Pin or unpin a decision"),
    SlashCommand::new("/invalidate", "[id]", "Retire a decision"),
    SlashCommand::new("/db-status", "", "Inspect storage and table counts"),
    SlashCommand::new("/compact", "", "Compact conversation history"),
    SlashCommand::new("/refresh", "", "Rescan files and run cargo check"),
    SlashCommand::new("/help", "", "Show commands and shortcuts"),
    SlashCommand::new("/quit", "", "Close the session"),
];

impl SlashCommand {
    const fn new(name: &'static str, argument: &'static str, description: &'static str) -> Self {
        Self {
            name,
            argument,
            description,
        }
    }
}

/// Offset from the top, in wrapped terminal rows. Chat can also follow the live tail.
#[derive(Default)]
pub struct Scroll {
    pub top: u16,
    pub max: u16,
    height: u16,
    pub follow_end: bool,
}

impl Scroll {
    pub fn measure(&mut self, rows: usize, height: u16) {
        self.height = height;
        self.max = u16::try_from(rows.saturating_sub(usize::from(height))).unwrap_or(u16::MAX);
        self.top = if self.follow_end {
            self.max
        } else {
            self.top.min(self.max)
        };
    }

    pub fn navigate(&mut self, key: KeyCode) {
        let page = self.height.saturating_sub(1).max(1);
        self.top = match key {
            KeyCode::Up => self.top.saturating_sub(1),
            KeyCode::Down => self.top.saturating_add(1).min(self.max),
            KeyCode::PageUp => self.top.saturating_sub(page),
            KeyCode::PageDown => self.top.saturating_add(page).min(self.max),
            KeyCode::Home => 0,
            KeyCode::End => self.max,
            _ => return,
        };
        self.follow_end = self.top == self.max;
    }
}

fn composer() -> TextArea<'static> {
    let mut input = TextArea::default();
    input.set_placeholder_text("Ask about this repository, or type / for commands");
    input.set_placeholder_style(ink(MUTED));
    input.set_style(ink(TEXT));
    input.set_cursor_line_style(Style::default());
    input
}

/// Active UI focus region for keyboard navigation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusPane {
    Input,
    Chat,
    Memories,
}

/// Items rendered chronologically in the conversation.
#[derive(Debug, Clone)]
pub enum ChatEntry {
    User(String),
    Assistant {
        text: String,
        streaming: bool,
    },
    ToolCard {
        record: ToolExecutionRecord,
        expanded: bool,
    },
    System(String),
    Error(String),
}

/// Complete client-side TUI state driven by `UiEvent`s from `allyh-core`.
pub struct AppState {
    pub repo_name: String,
    pub entries: Vec<ChatEntry>,
    pub inspector: InspectorSnapshot,
    pub input: TextArea<'static>,
    pub focus: FocusPane,
    pub chat_scroll: Scroll,
    pub plan_scroll: Scroll,
    pub command_index: usize,
    pub commands_dismissed: bool,
    pub selected_memory_idx: usize,
    pub should_quit: bool,
    pub busy: bool,
}

impl AppState {
    pub fn new(initial_inspector: InspectorSnapshot, repo_name: String) -> Self {
        Self {
            repo_name,
            entries: Vec::new(),
            inspector: initial_inspector,
            input: composer(),
            focus: FocusPane::Input,
            chat_scroll: Scroll {
                follow_end: true,
                ..Scroll::default()
            },
            plan_scroll: Scroll::default(),
            command_index: 0,
            commands_dismissed: false,
            selected_memory_idx: 0,
            should_quit: false,
            busy: false,
        }
    }

    pub fn is_gate_modal_open(&self) -> bool {
        matches!(
            self.inspector.gate_state,
            ControlGateState::PlanApproval { .. }
        )
    }

    pub fn cycle_focus(&mut self, reverse: bool) {
        use FocusPane::*;
        self.focus = match (self.focus, reverse) {
            (Input, false) | (Chat, true) => Memories,
            (Memories, false) | (Input, true) => Chat,
            (Chat, false) | (Memories, true) => Input,
        };
    }

    pub fn command_matches(&self) -> Vec<&'static SlashCommand> {
        if self.focus != FocusPane::Input || self.commands_dismissed || self.is_gate_modal_open() {
            return Vec::new();
        }
        let [prefix] = self.input.lines() else {
            return Vec::new();
        };
        if !prefix.starts_with('/') || prefix.contains(char::is_whitespace) {
            return Vec::new();
        }
        let prefix = prefix.to_ascii_lowercase();
        COMMANDS
            .iter()
            .filter(|cmd| cmd.name.starts_with(&prefix))
            .collect()
    }

    pub fn complete_command(&mut self, command: &SlashCommand) {
        self.input.select_all();
        self.input.insert_str(command.name);
        if !command.argument.is_empty() {
            self.input.insert_char(' ');
        }
        self.commands_dismissed = true;
    }

    pub fn toggle_tool_cards(&mut self) {
        for entry in &mut self.entries {
            if let ChatEntry::ToolCard { expanded, .. } = entry {
                *expanded = !*expanded;
            }
        }
    }

    pub fn select_prev_memory(&mut self) {
        if self.selected_memory_idx > 0 {
            self.selected_memory_idx -= 1;
        }
    }

    pub fn select_next_memory(&mut self) {
        let max = self.inspector.activated_memories.len().saturating_sub(1);
        if self.selected_memory_idx < max {
            self.selected_memory_idx += 1;
        }
    }

    pub fn selected_memory_id(&self) -> Option<String> {
        self.inspector
            .activated_memories
            .get(self.selected_memory_idx)
            .map(|m| m.decision.id.clone())
    }

    /// Take the current multiline prompt text and reset the `tui-textarea` widget.
    pub fn take_input_text(&mut self) -> String {
        self.command_index = 0;
        self.commands_dismissed = false;
        std::mem::replace(&mut self.input, composer())
            .lines()
            .join("\n")
    }

    /// Parse user submission into either a slash command (`UserCommand`) or regular prompt.
    pub fn parse_submission(&mut self, raw: &str) -> Option<UserCommand> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        self.chat_scroll.follow_end = true;

        if let Some(cmd_body) = trimmed.strip_prefix('/') {
            let mut parts = cmd_body.splitn(2, ' ');
            let cmd = parts.next().unwrap_or("").to_ascii_lowercase();
            let arg = parts.next().unwrap_or("").trim();

            match cmd.as_str() {
                "model" => {
                    if arg.is_empty() {
                        self.entries.push(ChatEntry::System(format!(
                            "Active reasoning model: {}\nWorker model: {}\nUsage: /model anthropic:claude-sonnet-4-6 | openai:gpt-4o | local:qwen-35b@http://localhost:9931/v1 | offline",
                            self.inspector.active_model_label, self.inspector.worker_model_label
                        )));
                        None
                    } else {
                        Some(UserCommand::SwitchModel(arg.to_string()))
                    }
                }
                "plan" => {
                    let plan = AllyHOrchestrator::build_quick_plan(arg);
                    Some(UserCommand::ProposePlan(plan))
                }
                "memories" => Some(UserCommand::RequestMemories),
                "invalidate" => {
                    if arg.is_empty() {
                        if let Some(id) = self.selected_memory_id() {
                            Some(UserCommand::InvalidateMemory {
                                decision_id: id,
                                reason: Some("invalidated via /invalidate".to_string()),
                            })
                        } else {
                            self.entries.push(ChatEntry::System(
                                "Usage: /invalidate <decision-id-prefix> (or select a memory in the sidebar and press `x`)".to_string(),
                            ));
                            None
                        }
                    } else {
                        let resolved = self
                            .inspector
                            .activated_memories
                            .iter()
                            .find(|m| m.decision.id.starts_with(arg))
                            .map(|m| m.decision.id.clone())
                            .unwrap_or_else(|| arg.to_string());
                        Some(UserCommand::InvalidateMemory {
                            decision_id: resolved,
                            reason: Some("invalidated via /invalidate".to_string()),
                        })
                    }
                }
                "pin" => {
                    let resolved = if arg.is_empty() {
                        self.selected_memory_id()
                    } else {
                        Some(
                            self.inspector
                                .activated_memories
                                .iter()
                                .find(|m| m.decision.id.starts_with(arg))
                                .map(|m| m.decision.id.clone())
                                .unwrap_or_else(|| arg.to_string()),
                        )
                    };
                    if let Some(id) = resolved {
                        Some(UserCommand::PinMemory(id))
                    } else {
                        self.entries.push(ChatEntry::System(
                            "Usage: /pin <decision-id-prefix>".to_string(),
                        ));
                        None
                    }
                }
                "db-status" => Some(UserCommand::RequestDbStatus),
                "compact" => Some(UserCommand::CompactNow),
                "refresh" => Some(UserCommand::RefreshEnv {
                    run_build_check: true,
                }),
                "help" => {
                    let mut help = String::from("### Commands\n");
                    for cmd in COMMANDS {
                        let usage = format!("{} {}", cmd.name, cmd.argument);
                        help.push_str(&format!("- `{}`  {}\n", usage.trim_end(), cmd.description));
                    }
                    help.push_str(
                        "\n### Shortcuts\n\
                        - Enter sends. Alt+Enter, Shift+Enter, or Ctrl+J adds a line.\n\
                        - Tab / Shift+Tab change focus. Esc returns to the prompt.\n\
                        - In the command list, Up/Down select and Tab completes. Esc dismisses.\n\
                        - In chat or plan review, use arrows, PgUp/PgDn, Home or End.\n\
                        - In memories, Up/Down select, p pins and x invalidates.\n\
                        - Ctrl+T or F2 toggles tool output. Ctrl+C quits.",
                    );
                    self.entries.push(ChatEntry::System(help));
                    None
                }
                "quit" | "exit" => {
                    self.should_quit = true;
                    Some(UserCommand::Shutdown)
                }
                other => {
                    self.entries.push(ChatEntry::Error(format!(
                        "Unknown slash command '/{other}'. Type /help for available commands."
                    )));
                    None
                }
            }
        } else {
            self.entries.push(ChatEntry::User(trimmed.to_string()));
            self.entries.push(ChatEntry::Assistant {
                text: String::new(),
                streaming: true,
            });
            self.busy = true;
            Some(UserCommand::SendPrompt(trimmed.to_string()))
        }
    }

    /// Apply an asynchronous [`UiEvent`] emitted by the `allyh-core` actor.
    pub fn apply_ui_event(&mut self, event: UiEvent) {
        match event {
            UiEvent::TokenDelta(chunk) => {
                if let Some(ChatEntry::Assistant { text, streaming }) = self.entries.last_mut()
                    && *streaming
                {
                    text.push_str(&chunk);
                } else {
                    self.entries.push(ChatEntry::Assistant {
                        text: chunk,
                        streaming: true,
                    });
                }
            }
            UiEvent::AssistantMessageFinished { text, .. } => {
                self.busy = false;
                if let Some(ChatEntry::Assistant {
                    text: existing,
                    streaming,
                }) = self.entries.last_mut()
                    && *streaming
                {
                    if existing.is_empty() {
                        *existing = text;
                    }
                    *streaming = false;
                } else if !text.is_empty() {
                    self.entries.push(ChatEntry::Assistant {
                        text,
                        streaming: false,
                    });
                }
            }
            UiEvent::ToolCard(record) => {
                // Insert tool card ahead of the streaming assistant placeholder if empty.
                let pop_empty_placeholder = matches!(
                    self.entries.last(),
                    Some(ChatEntry::Assistant { text, streaming: true }) if text.is_empty()
                );
                let placeholder = if pop_empty_placeholder {
                    self.entries.pop()
                } else {
                    None
                };
                self.entries.push(ChatEntry::ToolCard {
                    record,
                    expanded: false,
                });
                if let Some(p) = placeholder {
                    self.entries.push(p);
                }
            }
            UiEvent::GateChanged(state) => {
                self.plan_scroll = Scroll::default();
                self.inspector.gate_state = state;
            }
            UiEvent::InspectorUpdated(snapshot) => {
                self.inspector = snapshot;
                if self.selected_memory_idx >= self.inspector.activated_memories.len() {
                    self.selected_memory_idx =
                        self.inspector.activated_memories.len().saturating_sub(1);
                }
            }
            UiEvent::WorkerStatusChanged(status) => {
                self.inspector.worker_status = status;
            }
            UiEvent::MemoriesList(list) => {
                if list.is_empty() {
                    self.entries.push(ChatEntry::System(
                        "No active architectural invariants stored for this repository partition."
                            .to_string(),
                    ));
                } else {
                    let rendered = list
                        .iter()
                        .map(|m| {
                            let pin = if m.pinned { " [PINNED]" } else { "" };
                            format!(
                                "- `{}` [{}]{pin} {}",
                                &m.decision.id[..m.decision.id.len().min(8)],
                                m.decision.topic,
                                m.decision.rule_statement
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    self.entries.push(ChatEntry::System(format!(
                        "### Active LanceDB Invariants ({})\n{rendered}",
                        list.len()
                    )));
                }
            }
            UiEvent::DbStatus(report) => {
                self.entries.push(ChatEntry::System(format!(
                    "### LanceDB Status ({})\n\
                     - URI: `{}`\n\
                     - Repo Partition: `{}` | Scope: `{}`\n\
                     - Sessions: {} | Active Rules: {} | Superseded/Invalidated: {}\n\
                     - Tasks: {} | Code Graph Snapshots: {}",
                    report.backend_label,
                    report.uri,
                    &report.repo_id[..report.repo_id.len().min(12)],
                    report.scope_id,
                    report.sessions_count,
                    report.active_decisions_count,
                    report.invalidated_decisions_count,
                    report.tasks_count,
                    report.code_snapshots_count,
                )));
            }
            UiEvent::SystemNotice(msg) => {
                self.entries.push(ChatEntry::System(msg));
            }
            UiEvent::Error(err) => {
                self.busy = false;
                if matches!(
                    self.entries.last(),
                    Some(ChatEntry::Assistant { text, streaming: true }) if text.is_empty()
                ) {
                    self.entries.pop();
                }
                self.entries.push(ChatEntry::Error(err));
            }
        }
    }

    pub fn worker_status_label(&self) -> &'static str {
        match self.inspector.worker_status {
            WorkerStatus::Idle => "Idle",
            WorkerStatus::Consolidating => "Consolidating",
            WorkerStatus::Optimizing => "Optimizing",
        }
    }
}
