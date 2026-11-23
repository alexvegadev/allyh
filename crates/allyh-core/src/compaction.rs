use rig_core::message::{AssistantContent, Message, UserContent};

use crate::preamble::estimate_tokens;
use crate::router::HybridRouter;

/// An atomic conversational unit that MUST NEVER be split across a compaction boundary
/// (preserving intact tool call / tool response pairs).
#[derive(Debug, Clone)]
pub struct HistoryUnit {
    pub messages: Vec<Message>,
    pub estimated_tokens: u64,
}

impl HistoryUnit {
    pub fn from_messages(messages: Vec<Message>) -> Self {
        let chars: usize = messages.iter().map(message_char_len).sum();
        Self {
            messages,
            estimated_tokens: ((chars as u64) + 3) / 4,
        }
    }
}

/// Non-destructive conversation history manager with local-model rolling compaction.
#[derive(Debug, Clone)]
pub struct CompactingHistory {
    units: Vec<HistoryUnit>,
    rolling_summary: String,
    keep_recent_units: usize,
    compaction_ratio: f64,
}

impl CompactingHistory {
    pub fn new(initial_summary: String) -> Self {
        Self {
            units: Vec::new(),
            rolling_summary: initial_summary,
            keep_recent_units: 6,
            compaction_ratio: 0.70,
        }
    }

    pub fn rolling_summary(&self) -> &str {
        &self.rolling_summary
    }

    pub fn set_rolling_summary(&mut self, summary: String) {
        self.rolling_summary = summary;
    }

    pub fn unit_count(&self) -> usize {
        self.units.len()
    }

    /// Push a single standalone message as an atomic unit.
    pub fn push_message(&mut self, message: Message) {
        self.units.push(HistoryUnit::from_messages(vec![message]));
    }

    /// Push a grouped sequence (e.g., assistant tool calls + matching user tool results)
    /// as one indivisible atomic unit so tool call/response pairs stay intact.
    pub fn push_atomic_group(&mut self, messages: Vec<Message>) {
        if !messages.is_empty() {
            self.units.push(HistoryUnit::from_messages(messages));
        }
    }

    /// Flatten retained units into chronological `rig-core` messages.
    pub fn messages(&self) -> Vec<Message> {
        self.units
            .iter()
            .flat_map(|u| u.messages.iter().cloned())
            .collect()
    }

    pub fn estimated_tokens(&self) -> u64 {
        let history_tokens: u64 = self.units.iter().map(|u| u.estimated_tokens).sum();
        history_tokens + estimate_tokens(&self.rolling_summary)
    }

    /// Compact older complete units into `rolling_summary` when context utilization crosses
    /// `compaction_ratio` of `context_window_tokens` (or when `force` is true).
    pub async fn compact_if_needed(
        &mut self,
        router: &HybridRouter,
        preamble_tokens: u64,
        context_window_tokens: u64,
        force: bool,
    ) -> bool {
        let total = preamble_tokens + self.estimated_tokens();
        let threshold = (context_window_tokens as f64 * self.compaction_ratio) as u64;
        if (!force && total < threshold) || self.units.len() <= self.keep_recent_units {
            return false;
        }

        let split_at = self.units.len().saturating_sub(self.keep_recent_units);
        if split_at == 0 {
            return false;
        }
        let evicted: Vec<HistoryUnit> = self.units.drain(0..split_at).collect();
        let transcript = render_units_for_summary(&evicted);

        let summarized = match router
            .run_local_worker(
                "You are AllyH's local memory compactor. Summarize the evicted conversation turns concisely, preserving architectural constraints, file paths, completed steps, and unresolved errors.",
                &format!(
                    "Existing rolling summary:\n{}\n\nNew turns to fold in:\n{}",
                    self.rolling_summary, transcript
                ),
            )
            .await
        {
            Ok(text) if !text.trim().is_empty() => text.trim().to_string(),
            _ => deterministic_rollup(&self.rolling_summary, &transcript),
        };

        self.rolling_summary = summarized;
        true
    }
}

fn render_units_for_summary(units: &[HistoryUnit]) -> String {
    let mut lines = Vec::new();
    for unit in units {
        for msg in &unit.messages {
            match msg {
                Message::System { content } => {
                    lines.push(format!("SYSTEM: {}", truncate_line(content, 200)));
                }
                Message::User { content } => {
                    for part in content {
                        match part {
                            UserContent::Text(t) => {
                                lines.push(format!("USER: {}", truncate_line(&t.text, 240)));
                            }
                            UserContent::ToolResult(res) => {
                                lines.push(format!(
                                    "TOOL_RESULT({}): {}",
                                    res.name,
                                    truncate_line(&format!("{:?}", res.content), 160)
                                ));
                            }
                            _ => {}
                        }
                    }
                }
                Message::Assistant(ast) => {
                    let text = ast.text();
                    if !text.trim().is_empty() {
                        lines.push(format!("ASSISTANT: {}", truncate_line(&text, 240)));
                    }
                    for call in ast.tool_calls() {
                        lines.push(format!("TOOL_CALL({})", call.function.name));
                    }
                }
            }
        }
    }
    lines.join("\n")
}

fn deterministic_rollup(existing: &str, newly_evicted: &str) -> String {
    let mut combined = String::new();
    if !existing.trim().is_empty() {
        combined.push_str(existing.trim());
        combined.push('\n');
    }
    for line in newly_evicted.lines().take(16) {
        combined.push_str("- ");
        combined.push_str(line);
        combined.push('\n');
    }
    if combined.len() > 2400 {
        let start = combined.len().saturating_sub(2400);
        let safe = combined
            .char_indices()
            .find(|(idx, _)| *idx >= start)
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        combined[safe..].to_string()
    } else {
        combined.trim().to_string()
    }
}

fn message_char_len(msg: &Message) -> usize {
    match msg {
        Message::System { content } => content.len(),
        Message::User { content } => content
            .iter()
            .map(|c| match c {
                UserContent::Text(t) => t.text.len(),
                UserContent::ToolResult(r) => format!("{:?}", r.content).len(),
                _ => 64,
            })
            .sum(),
        Message::Assistant(ast) => ast
            .content
            .iter()
            .map(|c| match c {
                AssistantContent::Text(t) => t.text.len(),
                AssistantContent::ToolCall(tc) => {
                    serde_json::to_string(&tc.function.arguments)
                        .map(|s| s.len())
                        .unwrap_or(0)
                        + 32
                }
                AssistantContent::Reasoning(r) => r.text.len(),
                _ => 32,
            })
            .sum(),
    }
}

fn truncate_line(text: &str, max_chars: usize) -> String {
    let single = text.replace('\n', " ");
    if single.len() <= max_chars {
        return single;
    }
    let cut = single
        .char_indices()
        .take_while(|(idx, _)| *idx < max_chars)
        .last()
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    format!("{}...", &single[..cut])
}
