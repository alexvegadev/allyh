use ratatui::style::{Color, Modifier, Style};

pub const CANVAS: Color = Color::Rgb(18, 23, 30);
pub const PANEL: Color = Color::Rgb(26, 33, 42);
pub const EDGE: Color = Color::Rgb(62, 75, 90);
pub const MUTED: Color = Color::Rgb(151, 166, 183);
pub const TEXT: Color = Color::Rgb(226, 233, 240);
pub const MINT: Color = Color::Rgb(128, 222, 192);
pub const BLUE: Color = Color::Rgb(158, 193, 248);
pub const AMBER: Color = Color::Rgb(244, 199, 125);
pub const RED: Color = Color::Rgb(255, 155, 148);

pub fn ink(color: Color) -> Style {
    Style::default().fg(color)
}

pub fn bold(color: Color) -> Style {
    ink(color).add_modifier(Modifier::BOLD)
}
