use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use syntect::easy::HighlightLines;
use syntect::highlighting::ThemeSet;
use syntect::parsing::SyntaxSet;

use crate::theme::{AMBER, BLUE, MINT, MUTED, PANEL, bold, ink};

/// Renders streaming Markdown text and syntax-highlighted fenced code blocks via `syntect`.
pub struct MarkdownHighlighter {
    syntax_set: SyntaxSet,
    theme_set: ThemeSet,
}

impl Default for MarkdownHighlighter {
    fn default() -> Self {
        Self::new()
    }
}

impl MarkdownHighlighter {
    pub fn new() -> Self {
        Self {
            syntax_set: SyntaxSet::load_defaults_newlines(),
            theme_set: ThemeSet::load_defaults(),
        }
    }

    pub fn render_markdown(&self, text: &str, base_style: Style) -> Vec<Line<'static>> {
        let mut lines = Vec::new();
        let mut in_code_block = false;
        let mut code_lang = String::new();
        let mut code_buf: Vec<&str> = Vec::new();

        for raw_line in text.lines() {
            let trimmed = raw_line.trim_start();
            if let Some(rest) = trimmed.strip_prefix("```") {
                if !in_code_block {
                    in_code_block = true;
                    code_lang = rest.trim().to_string();
                    code_buf.clear();
                    let label = if code_lang.is_empty() {
                        "code"
                    } else {
                        code_lang.as_str()
                    };
                    lines.push(Line::from(Span::styled(
                        format!("  ┌─ {label} ─────────────────────"),
                        ink(MUTED),
                    )));
                } else {
                    self.flush_code_block(&code_lang, &code_buf, &mut lines);
                    lines.push(Line::from(Span::styled(
                        "  └───────────────────────────────",
                        ink(MUTED),
                    )));
                    in_code_block = false;
                    code_lang.clear();
                    code_buf.clear();
                }
                continue;
            }

            if in_code_block {
                code_buf.push(raw_line);
            } else if let Some(heading) = trimmed
                .strip_prefix("### ")
                .or_else(|| trimmed.strip_prefix("## "))
                .or_else(|| trimmed.strip_prefix("# "))
            {
                lines.push(Line::styled(heading.to_owned(), bold(BLUE)));
            } else if let Some(bullet) = trimmed
                .strip_prefix("- ")
                .or_else(|| trimmed.strip_prefix("* "))
            {
                let mut spans = vec![Span::styled("  • ", ink(MINT))];
                spans.extend(parse_inline_code(bullet, base_style));
                lines.push(Line::from(spans));
            } else {
                lines.push(Line::from(parse_inline_code(raw_line, base_style)));
            }
        }

        // Flush unclosed fenced code block while streaming.
        if in_code_block && !code_buf.is_empty() {
            self.flush_code_block(&code_lang, &code_buf, &mut lines);
        }

        lines
    }

    fn flush_code_block(&self, lang: &str, code_lines: &[&str], out: &mut Vec<Line<'static>>) {
        let syntax = self
            .syntax_set
            .find_syntax_by_token(lang)
            .or_else(|| self.syntax_set.find_syntax_by_extension(lang))
            .unwrap_or_else(|| self.syntax_set.find_syntax_plain_text());
        let theme = &self.theme_set.themes["base16-ocean.dark"];
        let mut highlighter = HighlightLines::new(syntax, theme);

        for code_line in code_lines {
            let with_nl = format!("{code_line}\n");
            let mut spans = vec![Span::styled("  │ ", ink(MUTED))];
            if let Ok(ranges) = highlighter.highlight_line(&with_nl, &self.syntax_set) {
                for (style, fragment) in ranges {
                    let clean = fragment.trim_end_matches(['\r', '\n']);
                    if clean.is_empty() {
                        continue;
                    }
                    let fg = Color::Rgb(style.foreground.r, style.foreground.g, style.foreground.b);
                    spans.push(Span::styled(clean.to_string(), Style::default().fg(fg)));
                }
            } else {
                spans.push(Span::raw((*code_line).to_string()));
            }
            out.push(Line::from(spans));
        }
    }
}

fn parse_inline_code(line: &str, base_style: Style) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut in_code = false;
    for part in line.split('`') {
        if !part.is_empty() {
            let style = if in_code {
                ink(AMBER).bg(PANEL)
            } else {
                base_style
            };
            spans.push(Span::styled(part.to_string(), style));
        }
        in_code = !in_code;
    }
    if spans.is_empty() {
        spans.push(Span::styled(String::new(), base_style));
    }
    spans
}
