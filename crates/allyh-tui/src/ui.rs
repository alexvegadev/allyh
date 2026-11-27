use allyh_core::{ControlGateState, ProposedPlan};
use allyh_memory::TaskStatus;
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Margin, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, BorderType, Clear, List, ListItem, ListState, Padding, Paragraph, Wrap,
};

use crate::app::{AppState, ChatEntry, FocusPane, Scroll};
use crate::highlight::MarkdownHighlighter;
use crate::theme::*;

pub fn render(frame: &mut Frame<'_>, app: &mut AppState, highlighter: &MarkdownHighlighter) {
    let area = frame.area();
    frame.render_widget(Block::default().style(ink(TEXT).bg(CANVAS)), area);
    let input_rows = app.input.lines().len().clamp(1, 5) as u16;
    let composer_height = (input_rows + 2).min((area.height / 3).max(3));
    let [header, body, composer, status] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(composer_height),
        Constraint::Length(1),
    ])
    .areas(area);
    render_header(frame, header, app);
    let wide = body.width >= 100;
    let body = body.inner(Margin::new(1, u16::from(area.height >= 27)));
    if wide {
        let [chat, inspector] = Layout::horizontal([
            Constraint::Fill(1),
            Constraint::Length((body.width / 3).clamp(34, 48)),
        ])
        .spacing(1)
        .areas(body);
        render_chat(frame, chat, app, highlighter);
        render_inspector(frame, inspector, app);
    } else if app.focus == FocusPane::Memories {
        render_inspector(frame, body, app);
    } else {
        render_chat(frame, body, app, highlighter);
    }
    if !app.is_gate_modal_open() {
        render_composer(frame, composer.inner(Margin::new(1, 0)), app);
    }
    render_status(frame, status, app);

    if let ControlGateState::PlanApproval { plan } = &app.inspector.gate_state {
        render_plan(frame, area, plan, &mut app.plan_scroll);
    } else {
        render_commands(frame, body, app);
    }
}

fn panel(title: &str, accent: Color) -> Block<'_> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(ink(accent))
        .title(Span::styled(format!(" {title} "), bold(TEXT)))
        .padding(Padding::horizontal(1))
}

fn render_header(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
    let (phase, short, color) = match app.inspector.gate_state {
        ControlGateState::Discovery => ("Discovery · Read-only", "Read-only", BLUE),
        ControlGateState::PlanApproval { .. } => ("Review · Writes locked", "Review", AMBER),
        ControlGateState::Execution { .. } => ("Execution · Writes enabled", "Execution", MINT),
    };
    let phase = if area.width >= 55 { phase } else { short };
    let mode = Line::from(Span::styled(format!(" ● {phase} "), bold(color))).right_aligned();
    let [left, right] =
        Layout::horizontal([Constraint::Min(0), Constraint::Length(mode.width() as u16)])
            .areas(area);
    frame.render_widget(Block::default().style(ink(TEXT).bg(PANEL)), area);
    frame.render_widget(
        Line::from(vec![
            Span::styled(" AllyH ", bold(CANVAS).bg(MINT)),
            Span::styled(format!("  {}", app.repo_name), bold(TEXT)),
        ]),
        left,
    );
    frame.render_widget(mode, right);
}

fn render_chat(
    frame: &mut Frame<'_>,
    area: Rect,
    app: &mut AppState,
    highlighter: &MarkdownHighlighter,
) {
    let edge = if app.focus == FocusPane::Chat {
        MINT
    } else {
        EDGE
    };
    let mut block = panel("Conversation", edge);
    let inner = block.inner(area);
    if app.entries.is_empty() {
        frame.render_widget(block, area);
        render_welcome(frame, inner);
        return;
    }

    let mut lines = Vec::new();
    for entry in &app.entries {
        if !lines.is_empty() {
            lines.push(Line::raw(""));
        }
        match entry {
            ChatEntry::User(text) => {
                lines.push(Line::styled("You", bold(BLUE)));
                lines.extend(text.lines().map(Line::raw));
            }
            ChatEntry::Assistant { text, streaming } => {
                let mut author = Line::styled("AllyH", bold(MINT));
                if *streaming {
                    author.push_span(Span::styled("  ·  responding", ink(MUTED)));
                }
                lines.push(author);
                lines.extend(highlighter.render_markdown(text, ink(TEXT)));
            }
            ChatEntry::ToolCard { record, expanded } => {
                let (label, color) = if record.succeeded {
                    ("done", MINT)
                } else {
                    ("failed", RED)
                };
                lines.push(Line::from(vec![
                    Span::styled(if *expanded { "▾ Tool" } else { "▸ Tool" }, bold(AMBER)),
                    Span::styled(format!("  {label}  "), ink(color)),
                    Span::styled(&record.compact_header, ink(TEXT)),
                ]));
                if *expanded {
                    lines.extend(
                        record
                            .output_text
                            .lines()
                            .map(|line| Line::styled(format!("  │ {line}"), ink(MUTED))),
                    );
                }
            }
            ChatEntry::System(msg) => lines.extend(highlighter.render_markdown(msg, ink(MUTED))),
            ChatEntry::Error(err) => {
                lines.push(Line::styled("Error", bold(RED)));
                lines.extend(err.lines().map(|line| Line::styled(line, ink(RED))));
            }
        }
    }
    let chat = Paragraph::new(lines).wrap(Wrap { trim: false });
    app.chat_scroll
        .measure(chat.line_count(inner.width), inner.height);
    if !app.chat_scroll.follow_end && app.chat_scroll.top < app.chat_scroll.max {
        block = block.title_bottom(Line::styled(" End  latest ", ink(MINT)).right_aligned());
    }
    frame.render_widget(block, area);
    frame.render_widget(chat.scroll((app.chat_scroll.top, 0)), inner);
}

fn render_welcome(frame: &mut Frame<'_>, area: Rect) {
    // Size against the conversation pane, which can be narrower than the terminal.
    let art = [
        include_str!("../assets/logo-wide.txt"),
        include_str!("../assets/logo-standard.txt"),
        include_str!("../assets/logo-compact.txt"),
    ]
    .into_iter()
    .find(|art| {
        art.lines().count() + 2 <= usize::from(area.height)
            && art
                .lines()
                .all(|row| Line::raw(row).width() <= usize::from(area.width))
    });
    let mut lines: Vec<Line> = match art {
        Some(art) => art
            .lines()
            .map(|row| Line::styled(row, ink(Color::White)))
            .collect(),
        None => vec![Line::styled("AllyH HARNESS", bold(TEXT))],
    };
    let available = usize::from(area.height).saturating_sub(lines.len());
    if available >= 6 {
        lines.extend([
            Line::raw(""),
            Line::styled("What are we working on?", bold(TEXT)),
            Line::raw(""),
        ]);
    } else if available >= 2 {
        lines.push(Line::raw(""));
    }
    let remaining = usize::from(area.height).saturating_sub(lines.len());
    if remaining < 4 {
        let shortcuts = if area.width >= 43 {
            "/plan <goal>   /memories   /skills   /help"
        } else if area.width >= 33 {
            "/plan <goal>   /memories   /help"
        } else {
            "/help"
        };
        lines.push(Line::styled(shortcuts, ink(MINT)));
    } else {
        for (command, description) in [
            ("/plan <goal>", "Propose a change"),
            ("/memories", "Review project decisions"),
            ("/skills", "Browse reusable skills"),
            ("/help", "Browse commands"),
        ] {
            lines.push(Line::from(vec![
                Span::styled(format!("{command:<15}"), ink(MINT)),
                Span::styled(description, ink(MUTED)),
            ]));
        }
    }
    let width = lines.iter().map(Line::width).max().unwrap_or_default() as u16;
    let region = centered(area, width, lines.len() as u16);
    frame.render_widget(Paragraph::new(lines), region);
}

fn field<'a>(label: &str, value: impl Into<std::borrow::Cow<'a, str>>) -> Line<'a> {
    Line::from(vec![
        Span::styled(format!("{label:<10}"), ink(MUTED)),
        Span::styled(value, ink(TEXT)),
    ])
}

fn render_inspector(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
    let selected = app.focus == FocusPane::Memories;
    let snapshot = &app.inspector;
    let block = panel("Inspector", if selected { MINT } else { EDGE }).style(ink(TEXT).bg(PANEL));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let changes = match snapshot.dirty_files.len() {
        0 => "Clean".to_owned(),
        1 => "1 changed file".to_owned(),
        n => format!("{n} changed files"),
    };
    let stack_label = if snapshot.detected_languages.is_empty() {
        "Auto".to_string()
    } else {
        snapshot.detected_languages.join(", ")
    };
    let mut workspace = vec![
        Line::styled("Workspace", bold(BLUE)),
        field("Branch", snapshot.git_branch.as_str()),
        field("Stack", stack_label),
        field("Changes", changes),
        field("Build", snapshot.build_status.as_str()),
        field("Storage", snapshot.storage_backend_label.as_str()),
        field("Scope", snapshot.scope_id.as_str()),
        field("Skills", snapshot.skills_count.to_string()),
        field("Harness", snapshot.active_harness_label.as_str()),
        field(
            "SIM",
            if snapshot.sim_problems_count == 0 {
                "0 problems".to_string()
            } else {
                format!("{} problem(s)", snapshot.sim_problems_count)
            },
        ),
        field(
            "Partition",
            snapshot.repo_id.chars().take(12).collect::<String>(),
        ),
        field(
            "Worker",
            format!(
                "{} · {}",
                snapshot.worker_model_label,
                app.worker_status_label()
            ),
        ),
    ];
    if !snapshot.detected_toolchains.is_empty() {
        workspace.insert(3, field("Tools", snapshot.detected_toolchains.join(", ")));
    }
    if !snapshot.active_lsp_servers.is_empty() {
        let insert_idx = if snapshot.detected_toolchains.is_empty() { 3 } else { 4 };
        workspace.insert(
            insert_idx,
            field("LSP", snapshot.active_lsp_servers.join(", ")),
        );
    }
    if !snapshot.out_of_band_edits.is_empty() {
        workspace.insert(
            3,
            field(
                "External",
                format!("{} edited", snapshot.out_of_band_edits.len()),
            ),
        );
    }
    let tasks = &snapshot.active_tasks;
    let memories = &snapshot.activated_memories;
    // Reserve space for the selected memory even when the terminal loses height.
    let details = Paragraph::new(workspace).wrap(Wrap { trim: true });
    let work_rows = (details.line_count(inner.width).min(u16::MAX as usize - 1) as u16 + 1)
        .min(inner.height.saturating_sub(6));
    let task_rows =
        (tasks.len().clamp(1, 8) as u16 + 2).min(inner.height.saturating_sub(work_rows + 2));
    let [work, queue, memory] = Layout::vertical([
        Constraint::Length(work_rows),
        Constraint::Length(task_rows),
        Constraint::Fill(1),
    ])
    .areas(inner);
    frame.render_widget(details, work);

    let mut queue_lines = vec![Line::styled(format!("Tasks  {}", tasks.len()), bold(BLUE))];
    if tasks.is_empty() {
        queue_lines.push(Line::styled("No active tasks", ink(MUTED)));
    } else {
        for task in tasks.iter().take(queue.height.saturating_sub(1) as usize) {
            let (label, color) = match task.status {
                TaskStatus::Pending => ("Pending", MUTED),
                TaskStatus::Approved => ("Approved", BLUE),
                TaskStatus::InProgress => ("Working", AMBER),
                TaskStatus::Verified => ("Verified", MINT),
            };
            queue_lines.push(Line::from(vec![
                Span::styled(format!("{label}  "), ink(color)),
                Span::raw(task.title.as_str()),
            ]));
        }
    }
    frame.render_widget(Paragraph::new(queue_lines), queue);

    let [title, list] =
        Layout::vertical([Constraint::Length(1), Constraint::Fill(1)]).areas(memory);
    frame.render_widget(
        Line::styled(
            format!("Memories  {}", memories.len()),
            bold(if selected { MINT } else { BLUE }),
        ),
        title,
    );
    if memories.is_empty() {
        frame.render_widget(Line::styled("No active decisions", ink(MUTED)), list);
    } else {
        let items = memories.iter().map(|m| {
            ListItem::new(Line::from(vec![
                Span::styled(
                    if m.pinned { "* " } else { "· " },
                    ink(if m.pinned { AMBER } else { MUTED }),
                ),
                Span::styled(format!("{}  ", m.decision.topic), bold(TEXT)),
                Span::raw(m.decision.rule_statement.as_str()),
            ]))
        });
        let mut state =
            ListState::default().with_selected(selected.then_some(app.selected_memory_idx));
        frame.render_stateful_widget(
            List::new(items)
                .highlight_symbol("› ")
                .highlight_style(ink(TEXT).bg(EDGE)),
            list,
            &mut state,
        );
    }
}

fn render_composer(frame: &mut Frame<'_>, area: Rect, app: &mut AppState) {
    let focused = app.focus == FocusPane::Input;
    let chooser = !app.command_matches().is_empty();
    let hints = match app.focus {
        FocusPane::Input if chooser => " ↑↓ choose   Tab complete   Esc close ",
        FocusPane::Input if area.width < 62 => " Enter send   Tab inspect ",
        FocusPane::Input => " Enter send   Alt+Enter newline   Tab inspect ",
        FocusPane::Chat if area.width < 55 => " ↑↓ scroll   F2 tools   Esc compose ",
        FocusPane::Chat => " ↑↓ scroll   PgUp/PgDn page   F2 tools   Esc compose ",
        FocusPane::Memories if area.width < 55 => " ↑↓ select   p pin   x retire   Esc back ",
        FocusPane::Memories => " ↑↓ select   p pin/unpin   x invalidate   Esc compose ",
    };
    let block = panel("Prompt", if focused { MINT } else { EDGE })
        .style(ink(TEXT).bg(PANEL))
        .title_bottom(Line::styled(hints, ink(if focused { MINT } else { MUTED })));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    app.input.set_cursor_style(if focused {
        ink(CANVAS).bg(MINT)
    } else {
        Style::default()
    });
    frame.render_widget(&app.input, inner);
}

fn tokens(n: u64) -> String {
    if n < 1_000 {
        n.to_string()
    } else {
        let k = n as f64 / 1000.0;
        if n.is_multiple_of(1000) {
            format!("{k:.0}k")
        } else {
            format!("{k:.1}k")
        }
    }
}

fn render_status(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
    let info = &app.inspector;
    let mut stats = Line::from(vec![
        Span::styled("Context ", ink(MUTED)),
        Span::styled(
            format!(
                "{}/{} ",
                tokens(info.tokens_used),
                tokens(info.token_budget)
            ),
            ink(TEXT),
        ),
    ]);
    if area.width >= 72 {
        stats.push_span(Span::styled("  Worker ", ink(MUTED)));
        stats.push_span(Span::styled(
            format!("{} ", app.worker_status_label()),
            ink(TEXT),
        ));
    }
    let [left, right] =
        Layout::horizontal([Constraint::Min(0), Constraint::Length(stats.width() as u16)])
            .areas(area);
    frame.render_widget(
        Line::from(vec![
            Span::styled(" Model ", ink(MUTED)),
            Span::styled(&info.active_model_label, ink(TEXT)),
        ]),
        left,
    );
    frame.render_widget(stats.right_aligned(), right);
}

fn render_commands(frame: &mut Frame<'_>, body: Rect, app: &AppState) {
    let commands = app.command_matches();
    if commands.is_empty() || body.height < 3 {
        return;
    }
    let height = (commands.len().min(5) as u16 + 2).min(body.height);
    let area = Rect::new(body.x, body.bottom() - height, body.width.min(64), height);
    frame.render_widget(Clear, area);
    let items = commands.iter().map(|cmd| {
        let usage = format!("{} {}", cmd.name, cmd.argument);
        ListItem::new(Line::from(vec![
            Span::styled(format!("{usage:<19}"), bold(MINT)),
            Span::styled(cmd.description, ink(TEXT)),
        ]))
    });
    let block = panel("Commands", MINT).style(ink(TEXT).bg(PANEL)).title(
        Line::styled(
            format!(" {}/{} ", app.command_index + 1, commands.len()),
            ink(MUTED),
        )
        .right_aligned(),
    );
    let mut state = ListState::default().with_selected(Some(app.command_index));
    frame.render_stateful_widget(
        List::new(items)
            .block(block)
            .highlight_symbol("› ")
            .highlight_style(ink(TEXT).bg(EDGE)),
        area,
        &mut state,
    );
}

fn render_plan(frame: &mut Frame<'_>, screen: Rect, plan: &ProposedPlan, scroll: &mut Scroll) {
    let mut lines = vec![
        Line::styled(&plan.title, bold(TEXT)),
        Line::raw(plan.summary.as_str()),
        Line::raw(""),
    ];
    if !plan.affected_files.is_empty() {
        lines.push(Line::styled("Affected files", bold(BLUE)));
        lines.extend(
            plan.affected_files
                .iter()
                .map(|path| Line::raw(path.as_str())),
        );
        lines.push(Line::raw(""));
    }
    lines.push(Line::styled("Planned tasks", bold(BLUE)));
    for (idx, task) in plan.tasks.iter().enumerate() {
        lines.push(Line::styled(
            format!("{}. {}", idx + 1, task.title),
            bold(TEXT),
        ));
        lines.extend(
            task.acceptance_criteria
                .iter()
                .map(|criterion| Line::styled(format!("  · {criterion}"), ink(MUTED))),
        );
    }
    let plan_text = Paragraph::new(lines).wrap(Wrap { trim: false });
    let width = if screen.width >= 72 {
        screen.width.min(108) * 4 / 5
    } else {
        screen.width.saturating_sub(2)
    };
    let block = panel("Review plan", AMBER).style(ink(TEXT).bg(PANEL));
    let max_area = centered(screen, width, screen.height.saturating_sub(4).clamp(8, 30));
    let rows = plan_text.line_count(block.inner(max_area).width);
    let area = centered(
        screen,
        width,
        rows.saturating_add(4).min(max_area.height as usize) as u16,
    );
    let inner = block.inner(area);
    let [content, hint, actions] = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .areas(inner);
    frame.buffer_mut().set_style(screen, ink(MUTED).bg(CANVAS));
    frame.render_widget(Clear, area);
    frame.render_widget(block, area);
    scroll.measure(rows, content.height);
    frame.render_widget(plan_text.scroll((scroll.top, 0)), content);
    if scroll.max > 0 {
        frame.render_widget(Line::styled("↑↓ scroll   PgUp/PgDn page", ink(MUTED)), hint);
    }
    let approve = if actions.width >= 54 {
        " A/Enter  Approve Plan & enable writes "
    } else {
        " A  Enable writes "
    };
    frame.render_widget(
        Line::from(vec![
            Span::styled(approve, bold(CANVAS).bg(AMBER)),
            Span::styled("  Esc  Refine", bold(TEXT)),
        ]),
        actions,
    );
}

fn centered(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    Rect::new(
        area.x + (area.width - width) / 2,
        area.y + (area.height - height) / 2,
        width,
        height,
    )
}
