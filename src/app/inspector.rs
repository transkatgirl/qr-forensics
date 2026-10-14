//! Right panel: what the decode pipeline made of the grid, stage by stage.

use eframe::egui::{self, Color32, RichText};
use qr_forensics::decode::bch::InfoRead;
use qr_forensics::decode::bitstream::{Certainty, EndOfData, LostReason, SegmentContent};
use qr_forensics::decode::codewords::PinSource;
use qr_forensics::decode::known::{Place, TextPinError, ascii_pins, text_pins};
use qr_forensics::decode::padding::{Filler, padding_at};
use qr_forensics::decode::pipeline::{BlockOutcome, BlockReport, CodewordReport, CwStatus, SolveMethod};
use qr_forensics::decode::rs::RsFailure;
use qr_forensics::decode::sequence::ParityCheck;
use qr_forensics::decode::text::{Charset, TextReport, UNKNOWN_CHAR, control_picture};
use qr_forensics::symbol::{Family, SymbolVersion};
use qr_forensics::symbol::layout::{CodewordSpan, CwId};
use qr_forensics::symbol::tables::Mode;

use super::state::*;
use super::{Action, App, ExportFormat, Transform, ec_name};

const GOOD: Color32 = Color32::from_rgb(40, 170, 80);
/// How many of the readings of the format bits, and of the attempts of a
/// brute force, are listed until the rest are asked for.
const CANDIDATES_SHOWN: usize = 4;
const ATTEMPTS_SHOWN: usize = 6;

pub(super) fn show(ui: &mut egui::Ui, app: &mut App, actions: &mut Vec<Action>) {
    app.view.wheel.follow(ui.input(|i| i.pointer.hover_pos()));
    result_section(ui, app, actions);
    section(ui, "Format and version", true, |ui| format_section(ui, app, actions));
    section(ui, "Blocks", true, |ui| blocks_section(ui, app, actions));
    section(ui, "Codewords", true, |ui| codewords_section(ui, app, actions));
    section(ui, "Known text", true, |ui| known_text_section(ui, app, actions));
    section(ui, "Padding recovery and pins", true, |ui| padding_section(ui, app, actions));
    section(ui, "Structured append", !app.view.sequence.is_empty(), |ui| sequence_section(ui, app, actions));
    section(ui, "Segments", false, |ui| segments_section(ui, app));
    section(ui, "Data codewords", false, |ui| stream_section(ui, app));
}

fn section(ui: &mut egui::Ui, title: &str, open: bool, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::CollapsingHeader::new(RichText::new(title).strong()).default_open(open).show(ui, add_contents);
}

/// One of the lists that scroll inside the inspector, which takes the
/// wheel only while it is the list's: see [`WheelLatch`]. Its scroll bar
/// works either way.
fn latched(area: egui::ScrollArea, wheel: &WheelLatch, list: InnerList) -> egui::ScrollArea {
    area.scroll_source(egui::scroll_area::ScrollSource { mouse_wheel: wheel.owns(list), ..Default::default() })
}

/// Where a list was laid out, from what its scroll area hands back: the
/// room it was given, which a list shorter than that does not fill.
fn laid_out<R>(shown: &egui::scroll_area::ScrollAreaOutput<R>) -> egui::Rect {
    let room = shown.inner_rect;
    egui::Rect::from_min_size(room.min, egui::vec2(room.width(), room.height().min(shown.content_size.y)))
}

fn mono(text: impl Into<String>) -> RichText {
    RichText::new(text).monospace()
}

/// Characters as they should appear in the result box: control characters
/// as their control pictures, ␝ for the group separator and so on, as
/// zxing-cpp's ISO/IEC 15434 text mode shows them.
fn displayable(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            UNKNOWN_CHAR => '?',
            '\n' | '\t' => c,
            c => control_picture(c).unwrap_or(c),
        })
        .collect()
}

fn result_section(ui: &mut egui::Ui, app: &mut App, actions: &mut Vec<Action>) {
    let report = &app.derived.report;
    let visuals = ui.visuals().clone();
    let total = report.blocks.len();
    let (solved, verified) = (report.blocks_solved(), report.blocks_verified());

    ui.heading("Result");
    if report.is_trustworthy() {
        ui.colored_label(GOOD, RichText::new("Decoded and verified").strong());
    } else if report.is_complete() {
        ui.colored_label(visuals.warn_fg_color, RichText::new("Decoded, but not confirmed").strong());
        if verified < total {
            ui.label(format!(
                "{} of {total} blocks solved with too little left over to check the answer.",
                total - verified
            ));
        }
        if !report.score.stream_valid {
            ui.label("The data does not parse cleanly; the EC level or mask may be wrong.");
        }
    } else {
        ui.colored_label(visuals.error_fg_color, RichText::new(format!("{solved} of {total} blocks solved")).strong());
        ui.label("Text from unsolved blocks is shown as read. See Blocks for what each one needs.");
    }
    if app.derived.analysis.inverted {
        ui.colored_label(
            visuals.warn_fg_color,
            "The fixed patterns read inverted: this looks like a light-on-dark symbol taken as is.",
        );
        ui.horizontal(|ui| {
            if ui.button("Swap dark and light").on_hover_text("Swap the colour of every module in the grid (undoable)").clicked() {
                actions.push(Action::Transform(Transform::Invert));
            }
            if app.photo.is_some() {
                ui.weak("or tick Invert photo and sample again");
            }
        });
    } else if let Some(mirror) = app.derived.analysis.mirrored
        && !report.is_trustworthy()
    {
        ui.colored_label(
            visuals.warn_fg_color,
            "The format bits read as a mirror image: this looks like a symbol seen from the back.",
        );
        ui.horizontal(|ui| {
            if ui.button(mirror.label()).on_hover_text("Reflect the grid (undoable)").clicked() {
                actions.push(Action::Transform(Transform::Mirror(mirror)));
            }
            if app.photo.is_some() {
                ui.weak("or run Find in outline again, which places the corners in mirror order");
            }
        });
    } else if let Some(mirror) = app.derived.mirror_by_decode {
        ui.colored_label(
            visuals.warn_fg_color,
            "Nothing decodes as painted, but the grid's mirror image does: the symbol was seen from the back, and \
             its format bits could not say so.",
        );
        ui.horizontal(|ui| {
            if ui.button(mirror.label()).on_hover_text("Reflect the grid (undoable)").clicked() {
                actions.push(Action::Transform(Transform::Mirror(mirror)));
            }
            if app.photo.is_some() {
                ui.weak("or run Find in outline again, which places the corners in mirror order");
            }
        });
    } else if app.doc.format == FormatSel::Auto && app.derived.format_by_brute {
        ui.colored_label(
            visuals.warn_fg_color,
            "The format bits do not settle the EC level and mask; the best brute-force attempt is in use.",
        );
    }

    egui::Frame::new().fill(visuals.extreme_bg_color).inner_margin(8.0).corner_radius(4.0).show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        if report.text.is_empty() {
            ui.weak("No text yet.");
            return;
        }
        let area = egui::ScrollArea::vertical().id_salt("result_text").max_height(180.0);
        let picked = app.view.picked_char.filter(|&at| at < report.text.sources.len());
        let mut clicked = None;
        let shown = latched(area, &app.view.wheel, InnerList::ResultText).show(ui, |ui| match app.view.result_mode {
            ResultMode::Text => {
                // A label as egui draws one, with its text still to be
                // selected by dragging, and the character a click lands
                // on taken note of: the text is laid out a glyph to a
                // character, in the message's order.
                let label = egui::Label::new(text_job(ui, &report.text, picked)).sense(egui::Sense::click());
                let (at, galley, response) = label.layout_in_ui(ui);
                if ui.is_rect_visible(response.rect) {
                    let (color, underline) = (ui.visuals().text_color(), egui::Stroke::NONE);
                    egui::text_selection::LabelSelectionState::label_text_selection(ui, &response, at, galley.clone(), color, underline);
                }
                if response.clicked()
                    && let Some(pointer) = response.interact_pointer_pos()
                {
                    clicked = char_under(&galley, pointer - at);
                }
                response.on_hover_text(
                    "Click a character to outline the modules it is made of on the grid, and to start Known text there. \
                     Esc clears it.",
                );
            }
            other => {
                ui.label(mono(shown_as(&report.text, other)));
            }
        });
        app.view.wheel.place(InnerList::ResultText, laid_out(&shown));
        if let Some(at) = clicked.filter(|&at| at < report.text.sources.len()) {
            // A second click on the character lets go of it.
            app.view.picked_char = (picked != Some(at)).then_some(at);
            if app.view.picked_char.is_some() {
                (app.view.known_at, app.view.known_from_start) = (at, false);
                app.view.stage = Stage::Edit;
            }
        }
    });
    if let Some(at) = app.view.picked_char.filter(|&at| at < report.text.sources.len()) {
        ui.small(picked_line(app, at));
    }
    ui.horizontal(|ui| {
        ui.label("Show as");
        for mode in ResultMode::ALL {
            // GS1 element strings are offered only when FNC1 says the
            // message is one.
            if mode == ResultMode::Gs1 && report.text.gs1.is_none() {
                continue;
            }
            ui.selectable_value(&mut app.view.result_mode, mode, mode.label()).on_hover_text(match mode {
                ResultMode::Text => "The decoded text",
                ResultMode::Escaped => "Control characters spelt out, as <GS>",
                ResultMode::Hex => "The message's bytes in hexadecimal, before any character set",
                ResultMode::Eci => "ECI designators written in as \\000026, as the ECI protocol has them",
                ResultMode::Gs1 => {
                    "GS1 element strings, each application identifier in parentheses, as a label prints them"
                }
            });
        }
    });

    let (ok, unsure, missing) = report.text.counts();
    if unsure + missing > 0 {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            ui.small(format!("{ok} confirmed ·"));
            ui.small(RichText::new(format!("{unsure} unconfirmed")).color(visuals.warn_fg_color));
            ui.small("·");
            ui.small(RichText::new(format!("{missing} unreadable")).color(visuals.error_fg_color));
        });
    }
    ui.horizontal(|ui| {
        if ui.add_enabled(!report.text.is_empty(), egui::Button::new("Copy")).on_hover_text("Copy what is shown").clicked() {
            ui.ctx().copy_text(shown_as(&report.text, app.view.result_mode));
        }
        if ui
            .add_enabled(report.text.raw_bytes().is_some(), egui::Button::new("Save bytes…"))
            .on_hover_text("Write the message's bytes to a file, raw; needs every byte known")
            .clicked()
        {
            actions.push(Action::SaveBytes);
        }
        let mut charset = app.opts.charset;
        egui::ComboBox::from_id_salt("charset")
            .selected_text(charset.label())
            .show_ui(ui, |ui| {
                for candidate in Charset::ALL {
                    ui.selectable_value(&mut charset, candidate, candidate.label());
                }
            })
            .response
            .on_hover_text(
                "The character set byte data is read in where the symbol declares none; Auto guesses it from the bytes. \
                 A set the symbol declares with an ECI designator is followed whatever is chosen here",
            );
        if charset != app.opts.charset {
            actions.push(Action::SetCharset(charset));
        }
        if let Some(encoding) = &report.text.encoding {
            ui.weak(encoding);
        }
        if let Some(symbology) = &report.text.symbology {
            ui.weak(symbology).on_hover_text(
                "AIM symbology identifier: ]Q0 for a Model 1 symbol; for the others ]Q1 plain, ]Q3 with FNC1 in first \
                 position (GS1), ]Q5 with FNC1 in second position, and one more when an ECI is used",
            );
        }
    });
    for note in &report.text.notes {
        ui.small(note);
    }
    if let Some(gs1) = &report.text.gs1 {
        for problem in &gs1.problems {
            ui.small(RichText::new(format!("GS1: {problem}.")).color(visuals.warn_fg_color))
                .on_hover_text("The message says it is GS1 element strings and does not read as them here: a sign of a wrong read");
        }
    }
    if report.parse.segments.iter().any(|s| matches!(s.content, SegmentContent::StructuredAppend { .. }))
        && ui
            .button("Add to sequence")
            .on_hover_text("Collect this symbol's part of the message under Structured append, with the others")
            .clicked()
        {
            actions.push(Action::AddToSequence);
        }
    ui.horizontal_wrapped(|ui| {
        if ui
            .button("Apply corrections")
            .on_hover_text(
                "Write the corrected modules into the grid and redraw the fixed patterns and version information. \
                 The format information is redrawn too once a solved block or its own bits confirm it. Undoable.",
            )
            .clicked()
        {
            actions.push(Action::ApplyCorrections);
        }
        if ui
            .button("Export PNG…")
            .on_hover_text("Save the repaired symbol as a bitmap that an ordinary reader can scan")
            .clicked()
        {
            actions.push(Action::ExportImage(ExportFormat::Png));
        }
        if ui
            .button("Export SVG…")
            .on_hover_text("Save the repaired symbol as a drawing, which prints and scales at any size without blur")
            .clicked()
        {
            actions.push(Action::ExportImage(ExportFormat::Svg));
        }
    });
    ui.add_space(4.0);
}

/// The message laid out with its certainties coloured, and the character
/// `picked`, when there is one, marked as the grid marks its modules.
fn text_job(ui: &egui::Ui, text: &TextReport, picked: Option<usize>) -> egui::text::LayoutJob {
    let visuals = ui.visuals();
    let font = egui::FontId::monospace(14.0);
    let mut job = egui::text::LayoutJob::default();
    job.wrap.max_width = ui.available_width();
    let mut before = 0;
    for span in &text.spans {
        let mut format = egui::TextFormat::simple(font.clone(), visuals.text_color());
        match span.certainty {
            Certainty::Verified => {}
            Certainty::Unverified => format.color = visuals.warn_fg_color,
            Certainty::Unknown => {
                format.color = Color32::WHITE;
                format.background = visuals.error_fg_color;
            }
        }
        let shown = displayable(&span.text);
        let count = shown.chars().count();
        match picked.filter(|&at| (before..before + count).contains(&at)) {
            Some(at) => {
                // The span in three: up to the character, it, and the rest.
                let start = shown.char_indices().nth(at - before).map_or(shown.len(), |(i, _)| i);
                let end = shown[start..].chars().next().map_or(start, |ch| start + ch.len_utf8());
                let marked = egui::TextFormat { color: Color32::BLACK, background: super::canvas::PICKED, ..format.clone() };
                job.append(&shown[..start], 0.0, format.clone());
                job.append(&shown[start..end], 0.0, marked);
                job.append(&shown[end..], 0.0, format);
            }
            None => job.append(&shown, 0.0, format),
        }
        before += count;
    }
    job
}

/// The character of a laid-out text that lies under a point, which is
/// given from the text's own top left: the one whose glyph the point is
/// on, counted through the rows before it and the line breaks that end
/// them. `None` between rows and past a row's end.
fn char_under(galley: &egui::Galley, at: egui::Vec2) -> Option<usize> {
    let mut before = 0;
    for placed in &galley.rows {
        if (placed.min_y()..=placed.max_y()).contains(&at.y) {
            let x = at.x - placed.pos.x;
            let on = |glyph: &egui::epaint::text::Glyph| (glyph.logical_rect().left()..glyph.logical_rect().right()).contains(&x);
            return placed.row.glyphs.iter().position(on).map(|i| before + i);
        }
        before += placed.char_count_including_newline().0;
    }
    None
}

/// One line about the character picked in the result: what it is, the
/// bits of the data stream it is made of, and the codewords they lie in.
fn picked_line(app: &App, at: usize) -> String {
    let (report, layout) = (&app.derived.report, &app.derived.layout);
    let source = &report.text.sources[at];
    let shown: String = report.text.plain().chars().nth(at).map(|ch| displayable(&ch.to_string())).unwrap_or_default();
    let mut codewords: Vec<CwId> = source.bits.clone().filter_map(|bit| layout.stream_bit_location(bit).map(|(cw, _)| cw)).collect();
    codewords.dedup();
    let names: Vec<String> = codewords.iter().map(CwId::to_string).collect();
    let block = codewords.first().map_or(0, |&cw| layout.codewords[cw].block + 1);
    let mode = report.parse.segments.get(source.segment).map_or("", |segment| segment.mode.label());
    format!(
        "Character {} “{shown}” · {} · stream bits {} to {} · codeword{} {} of block {block}, outlined on the grid",
        at + 1,
        mode.to_lowercase(),
        source.bits.start,
        source.bits.end.saturating_sub(1),
        if names.len() == 1 { "" } else { "s" },
        names.join(", ")
    )
}

/// The known-text field's own id, so that a test can give it the keyboard.
pub(super) fn known_text_id() -> egui::Id {
    egui::Id::new("known_text")
}

/// Text the user knows the message to hold, and where: pinned, it costs
/// no error correction.
fn known_text_section(ui: &mut egui::Ui, app: &mut App, actions: &mut Vec<Action>) {
    let warn = ui.visuals().warn_fg_color;
    ui.small(
        "What you already know of the message costs no error correction once it is pinned: an address that begins \
         https://, a number printed beside the symbol. Type it, say where it starts, and pin it.",
    );
    ui.add(
        egui::TextEdit::multiline(&mut app.view.known_text)
            .id(known_text_id())
            .desired_rows(1)
            .desired_width(f32::INFINITY)
            .font(egui::TextStyle::Monospace)
            .hint_text("https://example.com/"),
    )
    .on_hover_text("Enter gives a line feed. Byte data is taken in the character set the result is shown in.");
    let length = app.derived.report.text.sources.len();
    let spec = app.derived.layout.stream;
    let view = &mut app.view;
    ui.horizontal_wrapped(|ui| {
        ui.label("Starts at");
        ui.radio_value(&mut view.known_from_start, false, "character")
            .on_hover_text("A character of the text as read, in the segment it is read in. Click one in the result to set it.");
        let mut number = view.known_at + 1;
        let field = egui::DragValue::new(&mut number).range(1..=length.max(1)).speed(0.2);
        if ui.add_enabled(!view.known_from_start, field).on_hover_text("The first character of the message is 1").changed() {
            view.known_at = number.saturating_sub(1);
        }
        ui.radio_value(&mut view.known_from_start, true, "the start, as").on_hover_text(
            "The very start of the message, whatever its header reads as: for a symbol whose first codewords are too \
             damaged to say how the message is packed. The mode is pinned with the text.",
        );
        ui.add_enabled_ui(view.known_from_start, |ui| {
            egui::ComboBox::from_id_salt("known_mode").selected_text(view.known_mode.label()).show_ui(ui, |ui| {
                for mode in [Mode::Byte, Mode::Alphanumeric, Mode::Numeric, Mode::Kanji, Mode::Hanzi] {
                    if spec.count_bits(mode).is_some() {
                        ui.selectable_value(&mut view.known_mode, mode, mode.label());
                    }
                }
            });
        });
    });
    // What pinning would do, said before it is done.
    let place = if view.known_from_start { Place::Start(view.known_mode) } else { Place::Char(view.known_at) };
    let found = text_pins(&app.derived.layout, &app.derived.report, place, &view.known_text, app.opts.charset);
    match &found {
        Ok(found) => {
            let mut text = format!("Pins {} characters, in {} codewords.", found.pinned, found.pins.len());
            if found.left_out > 0 {
                text.push_str(&format!(" {} at the ends share a group with characters not given, and are left out.", found.left_out));
            }
            ui.small(text);
            if found.contradicted > 0 {
                ui.colored_label(
                    warn,
                    format!(
                        "{} of its bits differ from bits error correction has confirmed: the text, or where it starts, \
                         looks wrong.",
                        found.contradicted
                    ),
                );
            }
        }
        Err(TextPinError::Empty) => {}
        Err(error) => {
            ui.colored_label(warn, format!("Cannot be pinned: {error}."));
        }
    }
    let text_pins = app.doc.pins.values().filter(|pin| pin.source == PinSource::Text).count();
    ui.horizontal_wrapped(|ui| {
        if ui
            .add_enabled(found.is_ok(), egui::Button::new("Pin text"))
            .on_hover_text("Fix these characters at their values, bit for bit, so that damage under them costs nothing. Undoable.")
            .clicked()
        {
            actions.push(Action::PinText);
        }
        if ui
            .add_enabled(!app.derived.report.is_trustworthy(), egui::Button::new("Try common beginnings"))
            .on_hover_text(
                "Try the ways messages commonly begin, https:// and WIFI: and BEGIN:VCARD among them, and pin the one \
                 that makes a block verify that did not. Error correction over the rest of the block is what bears a \
                 beginning out.",
            )
            .on_disabled_hover_text("The message is decoded and verified already")
            .clicked()
        {
            actions.push(Action::TryBeginnings);
        }
        if ui.add_enabled(text_pins > 0, egui::Button::new("Remove text pins")).clicked() {
            actions.push(Action::ClearTextPins);
        }
    });
    ascii_row(ui, app, actions);
}

/// The stretch of the message to assert is plain ASCII, and the button
/// that pins it: a stretch must be named, since a pin asserts, and an
/// address that is seven-bit throughout may be followed by a name with
/// an accent in it.
fn ascii_row(ui: &mut egui::Ui, app: &mut App, actions: &mut Vec<Action>) {
    let warn = ui.visuals().warn_fg_color;
    let length = app.derived.report.text.sources.len();
    let trustworthy = app.derived.report.is_trustworthy();
    let bytes = app.derived.report.parse.segments.iter().any(|s| matches!(s.content, SegmentContent::Bytes(_)));
    let (view, derived) = (&mut app.view, &app.derived);
    let found = view.ascii_range(length).map(|chars| ascii_pins(&derived.layout, &derived.report, chars));
    ui.horizontal_wrapped(|ui| {
        ui.label("ASCII from character");
        ui.add_enabled(length > 0, egui::DragValue::new(&mut view.ascii_from).range(0..=length).speed(0.2))
            .on_hover_text("The first character of the stretch you know to be plain ASCII; the first character of the message is 1. Click a character in the result to see its number.");
        ui.label("to");
        ui.add_enabled(length > 0, egui::DragValue::new(&mut view.ascii_to).range(0..=length).speed(0.2))
            .on_hover_text("The last character of the stretch, counted the same way");
        let ready = matches!(found, Some(Ok(_)));
        if ui
            .add_enabled(!trustworthy && ready, egui::Button::new("Pin as ASCII"))
            .on_hover_text(
                "Assert that these characters are plain ASCII text, as an address, a Wi-Fi label or a card is: the high \
                 bit of every byte of them is pinned at zero, one bit a byte that need not be read, which is what tips \
                 a block short of solving bit by bit. Name only the stretch you are sure of: a pin asserts, and an \
                 accented letter or a raw byte under it would be pinned wrong. Undoable.",
            )
            .on_disabled_hover_text(if trustworthy {
                "The message is decoded and verified already"
            } else if !bytes {
                "The message as read has no byte segment"
            } else if length == 0 {
                "No text was read to name a stretch of"
            } else {
                "Give the first and last character of the stretch that is ASCII, within the message as read"
            })
            .clicked()
        {
            actions.push(Action::PinAscii);
        }
    });
    match &found {
        Some(Ok(found)) => {
            ui.small(format!("Pins the high bit of {} bytes, in {} codewords.", found.pinned, found.pins.len()));
            if found.contradicted > 0 {
                ui.colored_label(
                    warn,
                    format!(
                        "{} of those bits differ from bits error correction has confirmed: the stretch holds a byte above \
                         127, so it is not all ASCII.",
                        found.contradicted
                    ),
                );
            }
        }
        Some(Err(error)) => {
            ui.colored_label(warn, format!("Cannot be pinned: {error}."));
        }
        None if length > 0 && bytes && !trustworthy => {
            ui.small("Give the first and last character of the stretch you know to be plain ASCII, and pin it.");
        }
        None => {}
    }
}

/// The message in one of the result modes, as a plain string.
fn shown_as(text: &TextReport, mode: ResultMode) -> String {
    match mode {
        ResultMode::Text => text.plain(),
        ResultMode::Escaped => text.escaped(),
        ResultMode::Hex => text.hex(),
        ResultMode::Eci => text.eci_text.clone(),
        // Plain text when the message is not GS1 after all.
        ResultMode::Gs1 => text.gs1.as_ref().map_or_else(|| text.plain(), |gs1| gs1.hri.clone()),
    }
}

fn sequence_section(ui: &mut egui::Ui, app: &mut App, actions: &mut Vec<Action>) {
    let sequence = &app.view.sequence;
    let visuals = ui.visuals().clone();
    if sequence.is_empty() {
        ui.weak(
            "A message split over several symbols is collected here: decode each symbol and press Add to \
             sequence under Result. The parity byte is checked once every part is in.",
        );
        return;
    }
    for part in sequence.parts() {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(format!("Part {} of {}", part.index, part.total)).strong());
            ui.weak(&part.label);
            let (ok, unsure, missing) = part.text.counts();
            let state = if missing > 0 {
                RichText::new(format!("{missing} unreadable")).color(visuals.error_fg_color)
            } else if unsure > 0 {
                RichText::new(format!("{unsure} unconfirmed")).color(visuals.warn_fg_color)
            } else {
                RichText::new(format!("{ok} characters confirmed")).color(GOOD)
            };
            ui.label(state);
            if ui.small_button("Remove").clicked() {
                actions.push(Action::RemoveFromSequence(part.index));
            }
        });
    }
    let missing = sequence.missing();
    if !missing.is_empty() {
        let list: Vec<String> = missing.iter().map(u8::to_string).collect();
        ui.weak(format!("Still missing: part {}.", list.join(", ")));
    }
    let check = sequence.check();
    match check {
        ParityCheck::Empty => {}
        ParityCheck::Inconsistent => {
            ui.colored_label(
                visuals.error_fg_color,
                "The parts disagree about how many there are or what the parity is: they are not all from one message.",
            );
        }
        ParityCheck::Incomplete { have, total } => {
            ui.label(format!("{have} of {total} parts collected; the parity is checked once all are in."));
        }
        ParityCheck::Unreadable => {
            ui.colored_label(visuals.warn_fg_color, "Every part is in, but some bytes are unreadable, so the parity cannot be checked.");
        }
        ParityCheck::Complete { computed, claimed } if computed == claimed => {
            ui.colored_label(GOOD, format!("Every part is in and the parity checks: the data XORs to {computed:#04X} as the headers say."));
        }
        ParityCheck::Complete { computed, claimed } => {
            ui.colored_label(
                visuals.error_fg_color,
                format!(
                    "Every part is in but the parity does not check: the data XORs to {computed:#04X}, the headers say \
                     {claimed:#04X}. A part was decoded wrongly or belongs to another message."
                ),
            );
        }
    }
    let assembled = sequence.assembled();
    egui::Frame::new().fill(visuals.extreme_bg_color).inner_margin(8.0).corner_radius(4.0).show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        let area = egui::ScrollArea::vertical().id_salt("sequence_text").max_height(180.0);
        let shown = latched(area, &app.view.wheel, InnerList::SequenceText).show(ui, |ui| {
            ui.label(text_job(ui, &assembled, None));
        });
        app.view.wheel.place(InnerList::SequenceText, laid_out(&shown));
    });
    ui.horizontal(|ui| {
        if ui.button("Copy assembled text").clicked() {
            ui.ctx().copy_text(assembled.plain());
        }
        if ui.button("Copy as hex").clicked() {
            ui.ctx().copy_text(sequence.hex());
        }
        if ui.button("Clear sequence").clicked() {
            actions.push(Action::ClearSequence);
        }
    });
}

/// Bits of an info copy, most significant first, `?` where unreadable.
fn info_bits(read: &InfoRead) -> String {
    (0..read.len as u32)
        .rev()
        .map(|i| match (read.known >> i & 1, read.bits >> i & 1) {
            (0, _) => '?',
            (_, 1) => '1',
            _ => '0',
        })
        .collect()
}

fn format_section(ui: &mut egui::Ui, app: &mut App, actions: &mut Vec<Action>) {
    let derived = &app.derived;
    let version = app.doc.version;
    let analysis = &derived.analysis.format;
    let warn = ui.visuals().warn_fg_color;

    for (i, copy) in analysis.copies.iter().enumerate() {
        ui.horizontal(|ui| {
            ui.label(format!("Copy {}", i + 1));
            ui.label(mono(info_bits(copy)));
            ui.weak(format!("{} of {} read", copy.known_count(), copy.len));
            if copy.is_blank() {
                ui.colored_label(warn, "one colour throughout").on_hover_text(
                    "Every module of this copy that was read is the same colour: ink or a stain over the copy, or \
                     bare paper where it should be. It is taken as no word at all, and settles no format.",
                );
            }
        });
    }

    ui.horizontal(|ui| {
        let mut auto = app.doc.format == FormatSel::Auto;
        if ui.radio_value(&mut auto, true, "From format bits").clicked() {
            actions.push(Action::SetFormat(FormatSel::Auto));
        }
        if ui.radio_value(&mut auto, false, "Set by hand").clicked() {
            actions.push(Action::SetFormat(FormatSel::Manual { ec: derived.ec, mask: derived.mask }));
        }
    });
    if let FormatSel::Manual { ec, mask } = app.doc.format {
        ui.horizontal(|ui| {
            ui.label("EC level");
            for &level in version.ec_levels() {
                if ui.selectable_label(level == ec, ec_name(version, level)).clicked() {
                    actions.push(Action::SetFormat(FormatSel::Manual { ec: level, mask }));
                }
            }
            if version.mask_count() > 1 {
                ui.separator();
                ui.label("Mask");
                for m in 0..version.mask_count() {
                    if ui.selectable_label(m == mask, m.to_string()).clicked() {
                        actions.push(Action::SetFormat(FormatSel::Manual { ec, mask: m }));
                    }
                }
            }
        });
    }
    let how = if derived.format_by_brute { " (chosen by brute force)" } else { "" };
    ui.label(format!("In effect: EC level {}, mask {}{how}", ec_name(version, derived.ec), derived.mask));
    if analysis.is_doubtful()
        && let Some(best) = analysis.best()
    {
        ui.colored_label(
            warn,
            format!(
                "The closest format word still differs in {} of {} bits, more than the code can correct, so the \
                 format bits are not trusted.",
                best.best_mismatches, best.best_compared
            ),
        );
    }
    if analysis.unmasked_in_use() {
        ui.colored_label(
            warn,
            "The format word was painted without the standard's XOR mask, a quirk of some encoders; it is read as \
             such.",
        );
    }
    if let Some(hint) = analysis.other_model_hint()
        && let Some(other) = version.other_model()
    {
        let (this_model, other_model) = match version.family() {
            Family::Model1 => ("Model 1", "Model 2"),
            _ => ("Model 2", "Model 1"),
        };
        ui.colored_label(
            warn,
            format!(
                "The format bits spell a QR Code {other_model} word exactly ({} / mask {}), and no {this_model} word \
                 is close. This looks like a {other_model} symbol, whose data is laid out differently.",
                ec_name(version, hint.fields.ec),
                hint.fields.mask
            ),
        );
        if ui
            .button(format!("Read the grid as {other_model}"))
            .on_hover_text(format!(
                "Keep every module as painted and read the grid as a {other_model} symbol of the same size, \
                 version {other}. Pins and anchors are dropped. Undoable."
            ))
            .clicked()
        {
            actions.push(Action::ReinterpretAs(other));
        }
    }

    if analysis.copies.iter().all(|c| c.known_count() == 0) {
        ui.weak("No format bits painted yet.");
    } else {
        egui::Grid::new("format_candidates").num_columns(3).striped(true).show(ui, |ui| {
            ui.weak("Reads as");
            ui.weak("Bits differing").on_hover_text(
                "Counted against the best word the copies make between them, each half taken from the copy that \
                 agrees better. Words are ranked likeliest first: by their differing bits over all copies, each \
                 weighted by its reliability. Hover a count to see each copy and the weight.",
            );
            ui.weak("");
            ui.end_row();
            let shown = if app.view.all_format_candidates { analysis.candidates.len() } else { CANDIDATES_SHOWN };
            for candidate in analysis.candidates.iter().take(shown) {
                let mut name = if candidate.fits_grid {
                    format!("{} / mask {}", ec_name(version, candidate.fields.ec), candidate.fields.mask)
                } else {
                    format!(
                        "{}-{} / mask {}",
                        candidate.fields.version,
                        ec_name(candidate.fields.version, candidate.fields.ec),
                        candidate.fields.mask
                    )
                };
                if candidate.unmasked {
                    name.push_str(" (unmasked)");
                }
                if candidate.other_model {
                    name.push_str(if version.family() == Family::Model1 { " (Model 2 mask)" } else { " (Model 1 mask)" });
                }
                ui.label(name);
                let mut per_copy: Vec<String> = candidate
                    .per_copy
                    .iter()
                    .enumerate()
                    .map(|(i, (wrong, compared))| format!("copy {}: {wrong} of {compared}", i + 1))
                    .collect();
                per_copy.push(format!("weight over all copies: {:.1} bits", f64::from(candidate.weight) / 8.0));
                ui.label(format!("{} of {}", candidate.best_mismatches, candidate.best_compared))
                    .on_hover_text(per_copy.join("\n"));
                if candidate.fits_grid {
                    if ui.small_button("Use").clicked() {
                        actions.push(Action::SetFormat(FormatSel::Manual {
                            ec: candidate.fields.ec,
                            mask: candidate.fields.mask,
                        }));
                    }
                } else {
                    ui.colored_label(warn, "other size");
                }
                ui.end_row();
            }
        });
        let hint = "Every format word the bits were set against, nearest first";
        super::show_all_button(ui, &mut app.view.all_format_candidates, analysis.candidates.len(), CANDIDATES_SHOWN, hint);
    }
    if let Some(stray) = analysis.candidates.iter().find(|c| !c.fits_grid && !c.unmasked && c.compared > 0)
        && analysis.best().is_none_or(|best| stray.best_mismatches + 2 < best.best_mismatches)
    {
        ui.colored_label(
            warn,
            format!("The format bits fit a {} symbol better than this grid. Check the version.", stray.fields.version),
        );
    }

    if let Some(info) = &derived.analysis.version {
        ui.add_space(4.0);
        let text = format!(
            "Version information reads {} ({} of {} bits differ)",
            info.best_version, info.mismatches, info.compared
        );
        if info.compared > 0 && SymbolVersion::Qr(info.best_version) != version && info.margin > 0 {
            ui.colored_label(warn, format!("{text}, but the grid is version {version}."));
        } else if info.compared > 0 {
            ui.label(text);
        } else {
            ui.weak("Version information not painted.");
        }
    }

    ui.add_space(4.0);
    ui.horizontal(|ui| {
        if ui
            .button("Brute force")
            .on_hover_text("Decode under every EC level and mask and rank the results")
            .clicked()
        {
            actions.push(Action::BruteForce);
        }
        if app.brute.as_ref().is_some_and(|b| b.stale) {
            ui.weak("results below are from before the last change");
        }
    });
    if let Some(brute) = &app.brute {
        let shown = if app.view.all_attempts { brute.attempts.len() } else { ATTEMPTS_SHOWN };
        egui::Grid::new("brute").num_columns(6).striped(true).show(ui, |ui| {
            for heading in ["EC", "Mask", "Blocks", "Format", "Text", ""] {
                ui.weak(heading);
            }
            ui.end_row();
            for attempt in brute.attempts.iter().take(shown) {
                let report = &attempt.report;
                let current = attempt.ec == derived.ec && attempt.mask == derived.mask;
                let color = if report.is_trustworthy() { GOOD } else { ui.visuals().text_color() };
                ui.colored_label(color, ec_name(version, attempt.ec));
                ui.colored_label(color, attempt.mask.to_string());
                ui.colored_label(color, format!("{}/{}", report.blocks_verified(), report.blocks.len()))
                    .on_hover_text("Blocks solved and confirmed");
                ui.label(format!("{} off", attempt.format_mismatches)).on_hover_text(
                    "Bits of the format word for this level and mask that differ from the format bits as read: between \
                     two attempts that decode alike, the one the format bits lie nearer is the likelier",
                );
                let preview: String = displayable(&report.text.plain()).chars().take(14).collect();
                let stream = if report.score.stream_valid { "" } else { " (does not parse)" };
                ui.label(mono(format!("{preview}{stream}")));
                if current {
                    ui.weak("in use");
                } else if ui.small_button("Use").clicked() {
                    actions.push(Action::SetFormat(FormatSel::Manual { ec: attempt.ec, mask: attempt.mask }));
                }
                ui.end_row();
            }
        });
        super::show_all_button(ui, &mut app.view.all_attempts, brute.attempts.len(), ATTEMPTS_SHOWN, "Every EC level and mask tried, best first");
    }
}

fn outcome_text(block: &BlockReport) -> (String, bool) {
    match block.outcome {
        BlockOutcome::Solved { method, errors, erasures, slack, confidence_bits, risk_bits, beyond_standard } => {
            let mut text = match method {
                SolveMethod::ReedSolomon if errors + erasures == 0 => "clean".to_owned(),
                SolveMethod::ReedSolomon => format!("solved: {errors} wrong, {erasures} unknown, {slack} to spare"),
                SolveMethod::BitLevel => format!("solved bit by bit ({} bits, {confidence_bits} checks)", block.unknown_bits),
                SolveMethod::Reliability { doubted } => format!(
                    "solved after doubting the {doubted} least reliable codewords: {errors} wrong, {erasures} unknown, {slack} to spare"
                ),
                SolveMethod::BitReliability { doubted_bits } => format!(
                    "solved bit by bit after doubting the {doubted_bits} least reliable bits: {errors} codewords changed, {confidence_bits} checks"
                ),
                SolveMethod::Chase { flipped_bits } => format!(
                    "solved after flipping the {flipped_bits} least reliable bits: {errors} wrong, {erasures} unknown, {slack} to spare"
                ),
                SolveMethod::BitFlip { doubted_bits: 0, flipped_bits } => format!(
                    "solved bit by bit after flipping {flipped_bits} bits read solidly wrong, which the equations named: {errors} \
                     codewords changed, {confidence_bits} checks"
                ),
                SolveMethod::BitFlip { doubted_bits, flipped_bits } => format!(
                    "solved bit by bit after flipping {flipped_bits} bits read solidly wrong, which the equations named, and doubting \
                     the {doubted_bits} least reliable bits: {errors} codewords changed, {confidence_bits} checks"
                ),
                SolveMethod::Trial { erased } => format!(
                    "solved after erasing {erased} codewords by trial, {} past what the code corrects: {errors} wrong, \
                     {erasures} unknown, {slack} to spare",
                    if erased > 2 { "two codewords" } else { "one codeword" }
                ),
                SolveMethod::BitTrial { doubted_bits: 0, erased } => format!(
                    "solved bit by bit after erasing {erased} codewords by trial: {errors} codewords changed, {confidence_bits} checks"
                ),
                SolveMethod::BitTrial { doubted_bits, erased } => format!(
                    "solved bit by bit after erasing {erased} codewords by trial and doubting the {doubted_bits} least reliable bits: \
                     {errors} codewords changed, {confidence_bits} checks"
                ),
                SolveMethod::List { multiplicity } => format!(
                    "solved by list decoding (multiplicity {multiplicity}): {errors} wrong, {erasures} unknown, {confidence_bits} checks"
                ),
                SolveMethod::BeliefPropagation { rounds, flipped_bits } => format!(
                    "solved by belief propagation ({rounds} rounds, {flipped_bits} bits flipped): {errors} wrong, \
                     {erasures} unknown, {confidence_bits} checks"
                ),
                SolveMethod::PinnedOnly => "solved from pins alone".to_owned(),
            };
            if risk_bits > 0 {
                text.push_str(&format!(" — at most a 1 in 2^{risk_bits} chance it's wrong"));
            } else {
                text.push_str(" — nothing confirms it");
            }
            if beyond_standard {
                text.push_str(&format!(
                    " — beyond the standard's limit: {} of the EC codewords are reserved for detection, not correction, \
                     and this solution spent them",
                    block.reserved
                ));
            } else if !block.is_verified() && risk_bits > 0 {
                text.push_str(" — unconfirmed");
            }
            if block.half_padded_with_ones {
                text.push_str(
                    " — with the final four-bit codeword's missing nibble taken as ones: the symbol's encoder computed its \
                     error correction over that, not over the standard's zeros",
                );
            }
            if block.rivals > 1 {
                text.push_str(&format!(
                    " — one of {} solutions the block as read lay as near to, told from the rest by the message: under it \
                     alone the stream parses",
                    block.rivals
                ));
            }
            if block.near_errors > 0 {
                text.push_str(&format!(
                    " — after doubting {} modules of the block beside errors other blocks located: damage is contiguous, \
                     and a module next to one the photo misread is one to doubt first",
                    block.near_errors
                ));
            }
            (text, true)
        }
        BlockOutcome::Failed(failure) => {
            // What there is to do about it: the ways the decoder has of
            // finding too much wrong all call for the same, and are said
            // apart by [`failure_detail`].
            let mut text = match failure {
                RsFailure::TooManyErasures { erasures, capacity } => {
                    format!("{erasures} unknown codewords, limit {capacity}: resolve {} more", erasures - capacity)
                }
                RsFailure::BudgetExceeded
                | RsFailure::LocatorDegree
                | RsFailure::RootCount { .. }
                | RsFailure::ZeroDerivative
                | RsFailure::ResidualSyndrome => "too many wrong modules, or the wrong EC level or mask".to_owned(),
                RsFailure::PinConflict { .. } => "the only solution contradicts a pin".to_owned(),
                RsFailure::KnownBitConflict { .. } => {
                    "the only solution overturns modules read as known, with too little to spare to bear it out".to_owned()
                }
            };
            if block.inferred_bits > 0 {
                text.push_str(&format!("; {} of {} unknown bits inferred", block.inferred_bits, block.unknown_bits));
            }
            (text, false)
        }
    }
}

/// How the decoder came to find too much wrong in a block, for the hover
/// of a row that says only that it did; nothing for a failure the row
/// says outright.
fn failure_detail(failure: RsFailure) -> Option<String> {
    match failure {
        RsFailure::BudgetExceeded | RsFailure::LocatorDegree => {
            Some("More codewords are wrong than its error correction has left for them after the unknown ones.".to_owned())
        }
        RsFailure::RootCount { want, got } => Some(format!(
            "Its error correction makes out {want} wrong codewords and can place only {got} of them: more are wrong than it takes."
        )),
        RsFailure::ZeroDerivative | RsFailure::ResidualSyndrome => Some("No solution holds together.".to_owned()),
        RsFailure::TooManyErasures { .. } | RsFailure::PinConflict { .. } | RsFailure::KnownBitConflict { .. } => None,
    }
}

fn blocks_section(ui: &mut egui::Ui, app: &mut App, actions: &mut Vec<Action>) {
    let report = &app.derived.report;
    let layout = &app.derived.layout;
    let has_photo = app.photo.is_some();
    let error = ui.visuals().error_fg_color;
    let warn = ui.visuals().warn_fg_color;
    // Wrapping rows rather than a table: the result text is long and the
    // panel may be narrow.
    // Large versions have dozens of blocks; keep the list from taking over.
    let area = egui::ScrollArea::vertical().id_salt("blocks").max_height(230.0);
    let shown = latched(area, &app.view.wheel, InnerList::Blocks).show(ui, |ui| {
        for (b, block) in report.blocks.iter().enumerate() {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(format!("{}", b + 1)).strong());
                ui.weak(format!("{} data + {} EC", block.data_len, block.ec_len))
                    .on_hover_text("Codewords in this block");
                let (text, solved) = outcome_text(block);
                let color = match (solved, block.is_verified()) {
                    (true, true) => GOOD,
                    (true, false) => warn,
                    _ => error,
                };
                let label = ui.colored_label(color, text);
                if let BlockOutcome::Failed(failure) = block.outcome
                    && let Some(detail) = failure_detail(failure)
                {
                    label.on_hover_text(detail);
                }
                // The codeword a failed solution contradicts, to go to.
                if let BlockOutcome::Failed(RsFailure::PinConflict { index } | RsFailure::KnownBitConflict { index }) = block.outcome
                    && let Some(&cw) = layout.blocks[b].codewords.get(index)
                    && ui
                        .small_button(format!("Go to codeword {cw}"))
                        .on_hover_text(
                            "Select this codeword, on the grid and in the table. Its pin, or the modules of it that were \
                             painted or sampled, are what the solution cannot be squared with: one of the two is wrong.",
                        )
                        .clicked()
                {
                    app.view.selected = Some(cw);
                    app.view.reveal_selected = true;
                    app.view.stage = Stage::Edit;
                }
                let mut shown = app.view.highlight_block == Some(b);
                let doubted = layout.blocks[b].codewords.iter().any(|&cw| report.codewords[cw].doubted != 0);
                let hint = if doubted {
                    "Outline on the grid the modules that were doubted to solve this block"
                } else {
                    "Outline this block's unknown codewords on the grid"
                };
                if ((block.erased_codewords > 0 && !solved) || doubted) && ui.toggle_value(&mut shown, "Show").on_hover_text(hint).changed() {
                    app.view.highlight_block = shown.then_some(b);
                    app.view.stage = Stage::Edit;
                }
                if has_photo
                    && !block.is_verified()
                    && ui
                        .small_button("Sample")
                        .on_hover_text(
                            "Read this block's modules from the photo again, with the outline and Sampling as they now \
                             stand, and leave the rest of the grid as it is (undoable).",
                        )
                        .clicked()
                {
                    actions.push(Action::Sample { only_unknown: false, scope: SampleScope::Block(b) });
                }
            });
        }
    });
    app.view.wheel.place(InnerList::Blocks, laid_out(&shown));
    if report.blocks.iter().any(|b| !b.is_solved()) {
        ui.small("A block is solved once its unknown codewords, plus twice its wrong ones, fit inside its EC count.");
    }
}

/// Codeword bits as read, pinned or inferred, most significant first, `?`
/// where unknown: the bits the codeword has modules for.
fn read_bits(report: &CodewordReport, span: CodewordSpan) -> String {
    let (value, known) = report.partial();
    (span.bit_offset..span.bit_offset + span.nbits)
        .map(|i| {
            let bit = 0x80u8 >> i;
            match (known & bit != 0, value & bit != 0) {
                (false, _) => '?',
                (true, true) => '1',
                (true, false) => '0',
            }
        })
        .collect()
}

fn status_label(status: CwStatus) -> &'static str {
    match status {
        CwStatus::Clean => "ok",
        CwStatus::Corrected => "corrected",
        CwStatus::Filled => "filled",
        CwStatus::Unverified => "unconfirmed",
        CwStatus::Inferred => "inferred",
        CwStatus::Unresolved => "unknown",
    }
}

fn codewords_section(ui: &mut egui::Ui, app: &mut App, actions: &mut Vec<Action>) {
    let layout = app.derived.layout.clone();
    let report = &app.derived.report;
    let view = &mut app.view;

    // Selecting on the canvas switches the table to that codeword's block.
    if view.reveal_selected
        && let Some(cw) = view.selected
        && view.table_block.is_some()
    {
        view.table_block = Some(layout.codewords[cw].block as usize);
    }

    ui.horizontal(|ui| {
        ui.label("Show");
        let current = view.table_block.map_or("All, as placed".to_owned(), |b| format!("Block {}", b + 1));
        egui::ComboBox::from_id_salt("table_block").selected_text(current).show_ui(ui, |ui| {
            ui.selectable_value(&mut view.table_block, None, "All, as placed");
            for b in 0..layout.blocks.len() {
                ui.selectable_value(&mut view.table_block, Some(b), format!("Block {}", b + 1));
            }
        });
    });
    ui.small("Click a row, or a module with the Inspect tool, to see where a codeword lives.");

    let rows: Vec<CwId> = match view.table_block {
        Some(b) => layout.blocks[b].codewords.clone(),
        None => (0..layout.codewords.len()).collect(),
    };
    let row_height = ui.text_style_height(&egui::TextStyle::Monospace) + 2.0;
    let area = egui::ScrollArea::both().id_salt("codewords").max_height(220.0).auto_shrink([false, true]);
    let mut area = latched(area, &view.wheel, InnerList::Codewords);
    if view.reveal_selected {
        if let Some(row) = view.selected.and_then(|cw| rows.iter().position(|&r| r == cw)) {
            let spacing = ui.spacing().item_spacing.y;
            area = area.vertical_scroll_offset((row as f32 * (row_height + spacing) - 80.0).max(0.0));
        }
        view.reveal_selected = false;
    }
    let warn = ui.visuals().warn_fg_color;
    let error = ui.visuals().error_fg_color;
    let shown = area.show_rows(ui, row_height, rows.len(), |ui, range| {
        for &cw in &rows[range] {
            let span = layout.codewords[cw];
            let item = &report.codewords[cw];
            let block = &layout.blocks[span.block as usize];
            let (kind, index) = if span.is_data {
                ('D', span.index_in_block as usize)
            } else {
                ('E', span.index_in_block as usize - block.data_len)
            };
            let value = item.final_value.map_or("--".to_owned(), |v| format!("{v:02X}"));
            let pin = match item.pin.map(|p| p.source) {
                Some(PinSource::Manual) => " pin",
                Some(PinSource::Text) => " txt",
                Some(PinSource::Padding) => " pad",
                Some(PinSource::Terminator) => " end",
                Some(PinSource::Header) => " hdr",
                None => "",
            };
            let doubted = if item.doubted != 0 { " doubted" } else { "" };
            let text = format!(
                "{cw:>4}  B{:<2} {kind}{index:<3}  {:<8}  {value}  {}{pin}{doubted}",
                span.block + 1,
                read_bits(item, span),
                status_label(item.status),
            );
            let color = match item.status {
                CwStatus::Clean => ui.visuals().text_color(),
                CwStatus::Corrected | CwStatus::Filled => GOOD,
                CwStatus::Unverified | CwStatus::Inferred => warn,
                CwStatus::Unresolved => error,
            };
            if ui.selectable_label(view.selected == Some(cw), mono(text).color(color)).clicked() {
                view.selected = if view.selected == Some(cw) { None } else { Some(cw) };
                view.pin_text.clear();
            }
        }
    });
    view.wheel.place(InnerList::Codewords, laid_out(&shown));

    let Some(cw) = view.selected else {
        return;
    };
    let span = layout.codewords[cw];
    let item = &report.codewords[cw];
    ui.separator();
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(format!("Codeword {cw}")).strong());
        ui.label(format!("block {}", span.block + 1));
        if let Some(stream) = layout.stream_index(cw) {
            ui.label(format!("· message byte {stream}"));
        }
        ui.label("· reads");
        ui.label(mono(read_bits(item, span)));
        if let Some(value) = item.final_value {
            ui.label("· is");
            ui.label(mono(format!("{value:02X} {value:08b}")));
        }
    });
    ui.horizontal(|ui| {
        ui.label("Pin to");
        ui.add(egui::TextEdit::singleline(&mut view.pin_text).desired_width(44.0).hint_text("hex").font(egui::TextStyle::Monospace));
        let parsed = u8::from_str_radix(view.pin_text.trim(), 16).ok();
        if ui
            .add_enabled(parsed.is_some(), egui::Button::new("Pin"))
            .on_hover_text("Assert this value. Pinned codewords cost no error-correction capacity.")
            .clicked()
            && let Some(value) = parsed
        {
            actions.push(Action::Pin(cw, value));
        }
        if item.pin.is_some() && ui.button("Unpin").clicked() {
            actions.push(Action::Unpin(cw));
        }
    });
    // Bit by bit: a pin holds any of a codeword's bits, and what is known
    // of a message often ends partway through one, as a mode indicator
    // does.
    ui.horizontal(|ui| {
        ui.label("Pin bits");
        ui.spacing_mut().item_spacing.x = 2.0;
        let (mask, value) = item.pin.map_or((0, 0), |pin| (pin.mask, pin.value));
        let exists = span.module_mask();
        for i in 0..8u8 {
            let bit = 0x80u8 >> i;
            let pinned = (mask & bit != 0).then_some(value & bit != 0);
            let label = match pinned {
                Some(true) => "1",
                Some(false) => "0",
                None => "·",
            };
            let button = egui::Button::selectable(pinned.is_some(), mono(label));
            let response = ui
                .add_enabled(exists & bit != 0, button)
                .on_hover_text(format!(
                    "Bit {} of the codeword, as it stands in the data once the mask is off. Click to pin it as it reads, \
                     again to pin it the other way, and again to let it go.",
                    7 - i
                ))
                .on_disabled_hover_text("The symbol has no module for this bit");
            if response.clicked() {
                // As it reads on the grid, whatever a pin has made of it
                // since, so that the three clicks come round.
                let read = item.known & bit != 0 && item.read & bit != 0;
                let (mask, value) = match pinned {
                    None => (mask | bit, if read { value | bit } else { value & !bit }),
                    Some(now) if now == read => (mask, value ^ bit),
                    Some(_) => (mask & !bit, value & !bit),
                };
                actions.push(Action::PinBits { cw, mask, value });
            }
        }
    });
}

fn padding_section(ui: &mut egui::Ui, app: &mut App, actions: &mut Vec<Action>) {
    let derived = &app.derived;
    let warn = ui.visuals().warn_fg_color;
    let pins = &app.doc.pins;
    let count = |source: PinSource| pins.values().filter(|p| p.source == source).count();
    let (manual, text, padding, terminator, header) = (
        count(PinSource::Manual),
        count(PinSource::Text),
        count(PinSource::Padding),
        count(PinSource::Terminator),
        count(PinSource::Header),
    );

    if !pins.is_empty() {
        ui.label(format!(
            "Pins: {manual} by hand, {text} known text, {padding} padding, {terminator} terminator, {header} header"
        ));
        if app.doc.pins_ec != Some(derived.ec) {
            ui.colored_label(
                warn,
                "These pins were made under a different EC level and are not applied. Remove them or switch back.",
            );
        }
    }
    let derived_pins = &derived.report.derived_pins;
    if !derived_pins.is_empty() {
        let of = |source: PinSource| derived_pins.values().filter(|p| p.source == source).count();
        ui.label(format!(
            "The decoder pinned {} codewords itself: {} padding, {} terminator, {} header.",
            derived_pins.len(),
            of(PinSource::Padding),
            of(PinSource::Terminator),
            of(PinSource::Header)
        ))
        .on_hover_text(
            "Set by Pin the padding and headers found, under Decoding. These pins belong to the decode, not the \
             document: Pin the padding adds them to the document, and unticking the option removes them.",
        );
    }

    match &derived.padding {
        Some(candidate) => {
            let source = match candidate.header_certainty {
                Some(Certainty::Verified) => "segment headers, confirmed",
                Some(_) => "segment headers, not yet confirmed",
                None => "the pad pattern itself",
            };
            ui.label(format!(
                "Padding starts at message byte {} and runs for {} codewords (located from {source}).",
                candidate.first_pad_codeword, candidate.pad_codewords
            ));
            if candidate.filler != Filler::Standard {
                ui.label(format!("It is not the standard's padding: {}.", candidate.filler.label()));
            }
            ui.label(format!(
                "{} of {} readable bits there match the pattern.",
                candidate.matches,
                candidate.matches + candidate.mismatches
            ));
        }
        None if derived.report.is_complete() => {
            ui.weak("Every block is solved; nothing to recover.");
        }
        None => {
            ui.weak("No padding found. The message may fill the symbol, or the encoder padded in a way that is not looked for.");
        }
    }
    ui.horizontal_wrapped(|ui| {
        if ui
            .add_enabled(derived.padding.is_some(), egui::Button::new("Pin the padding"))
            .on_hover_text("Fix the pad codewords at their known values so damage there costs nothing")
            .clicked()
        {
            actions.push(Action::ApplyPadding);
        }
        if ui.add_enabled(padding + terminator > 0, egui::Button::new("Remove padding pins")).clicked() {
            actions.push(Action::ClearPaddingPins);
        }
        if ui.add_enabled(!pins.is_empty(), egui::Button::new("Remove all pins")).clicked() {
            actions.push(Action::ClearAllPins);
        }
    });

    // By hand, for padding that is not found: too little of it readable,
    // or a filler that is looked for only where the headers say the data
    // ends, with headers too damaged to say.
    ui.add_space(4.0);
    let last = derived.layout.data_stream.len().saturating_sub(1);
    let found = derived.padding.map(|candidate| candidate.first_pad_codeword).or(derived.report.tail.map(|tail| tail.first_pad_codeword));
    let mut first = app.view.pad_from.or(found).unwrap_or(last).min(last);
    ui.horizontal_wrapped(|ui| {
        ui.label("By hand: padding from message byte");
        if ui
            .add(egui::DragValue::new(&mut first).range(0..=last).speed(0.2))
            .on_hover_text(
                "The first data codeword that is padding, counted from 0 as the Data codewords list counts them: the one \
                 after the terminator. It follows what the decode makes of the padding until it is set here, and \
                 again once the padding is pinned from it or the grid is another symbol's.",
            )
            .changed()
        {
            app.view.pad_from = Some(first);
        }
        ui.label("as");
        egui::ComboBox::from_id_salt("pad_filler").selected_text(app.view.pad_filler.label()).show_ui(ui, |ui| {
            for filler in Filler::ALL {
                ui.selectable_value(&mut app.view.pad_filler, filler, filler.label());
            }
        });
    });
    let filler = app.view.pad_filler;
    let fit = padding_at(&derived.layout, &derived.report, first, filler);
    let readable = fit.matches + fit.mismatches;
    let text = format!("{} codewords from there; {} of {readable} readable bits match that filler.", fit.pad_codewords, fit.matches);
    if fit.mismatches * 4 > fit.matches {
        ui.colored_label(warn, text).on_hover_text("Data agrees with a filler in half its bits by chance: this start, or this filler, looks wrong");
    } else {
        ui.small(text);
    }
    if ui
        .button("Pin from there")
        .on_hover_text(
            "Pin every data codeword from that byte on at the filler's value, in place of any padding pins there are. \
             Nothing checks a start given by hand but the blocks, which solve with it or do not. Undoable.",
        )
        .clicked()
    {
        actions.push(Action::PinPaddingAt { first, filler });
    }
}

fn segments_section(ui: &mut egui::Ui, app: &mut App) {
    let report = &app.derived.report;
    let warn = ui.visuals().warn_fg_color;
    if report.parse.segments.is_empty() {
        ui.weak("No segments read.");
    }
    let picked = app.view.picked_segment.filter(|&at| at < report.parse.segments.len());
    for (i, segment) in report.parse.segments.iter().enumerate() {
        let detail = match &segment.content {
            SegmentContent::Chars(cells) => {
                let text: String = cells.iter().map(|c| c.value.unwrap_or('?')).take(28).collect();
                format!("{} characters: {text}", segment.count.unwrap_or(cells.len()))
            }
            SegmentContent::Bytes(cells) => {
                let hex: Vec<String> =
                    cells.iter().take(10).map(|c| c.value.map_or("??".into(), |v| format!("{v:02X}"))).collect();
                format!("{} bytes: {}", segment.count.unwrap_or(cells.len()), hex.join(" "))
            }
            SegmentContent::Kanji(cells) | SegmentContent::Hanzi(cells) => {
                format!("{} characters", segment.count.unwrap_or(cells.len()))
            }
            SegmentContent::Eci(Some(value)) => format!("assignment {value}"),
            SegmentContent::Eci(None) => "unreadable".into(),
            SegmentContent::StructuredAppend { index, total, parity } => {
                format!("symbol {index} of {total}, parity {parity:02X}")
            }
            SegmentContent::Fnc1First => String::new(),
            SegmentContent::Fnc1Second(indicator) => format!("application indicator {indicator}"),
            SegmentContent::Unreadable => "unreadable".into(),
        };
        let inferred = if segment.header_inferred { " · header inferred" } else { "" };
        let text = format!(
            "{} · bits {} to {} · {detail}{inferred}",
            segment.mode.label(),
            segment.header.start,
            segment.payload.end
        );
        let doubtful = segment.damaged || segment.truncated || segment.header_certainty != Certainty::Verified;
        let text = if doubtful { RichText::new(text).color(warn) } else { RichText::new(text) };
        let why = if segment.header_inferred {
            "The header has unreadable bits, and this is the one reading of them under which the message parses: a guess \
             the rest of the message bears out, not a read. "
        } else if segment.truncated {
            "The header promises more data than the symbol holds. "
        } else if segment.damaged {
            "Some of its bits make no valid value: a wrong read, or the wrong EC level or mask. "
        } else if doubtful {
            "The header is not confirmed by error correction. "
        } else {
            ""
        };
        if ui
            .selectable_label(picked == Some(i), text)
            .on_hover_text(format!("{why}Click to outline its modules on the grid, the header in yellow; Esc clears it."))
            .clicked()
        {
            // A second click on the segment lets go of it.
            app.view.picked_segment = (picked != Some(i)).then_some(i);
            if app.view.picked_segment.is_some() {
                app.view.stage = Stage::Edit;
            }
        }
    }
    if report.parse.preamble_damaged {
        ui.colored_label(warn, "A bit of the zeros a Model 1 message begins with reads as one: a wrong read, or the wrong EC level or mask.");
    }
    // What the parser made of a header it could not read, which is
    // otherwise seen only in what Pin the padding then pins.
    if let Some(inference) = &report.parse.header_inference {
        let guessed = inference.guessed_bits;
        let bits = if guessed == 1 { "1 bit".to_owned() } else { format!("{guessed} bits") };
        let text = match (inference.complete, inference.candidates) {
            (false, found) => format!(
                "A segment header has unreadable bits, and the search for readings of them under which the message parses \
                 ran out of tries, at {bits} guessed, with {found} found: nothing is inferred. Paint or pin the header to \
                 settle it."
            ),
            (true, 0) => format!(
                "A segment header has unreadable bits, and no reading of them makes the message parse, with up to {bits} \
                 guessed: the stream is wrong somewhere besides."
            ),
            (true, 1) => format!(
                "A segment header has unreadable bits, and one reading of them alone makes the message parse, guessing \
                 {bits}: it is the one shown, and Pin the padding pins those bits."
            ),
            (true, found) => format!(
                "A segment header has unreadable bits, and {found} readings of them make the message parse, each guessing \
                 {bits}: they agree on {} of the unreadable bits, which Pin the padding pins, and the header stays unread.",
                inference.inferred.len()
            ),
        };
        ui.colored_label(warn, text);
    }
    match report.parse.end {
        EndOfData::At { bit, certainty } => {
            let sure = if certainty == Certainty::Verified { "" } else { " (unconfirmed)" };
            ui.label(format!("Data ends at bit {bit} of {}{sure}.", report.stream.len()));
            if let Some(tail) = &report.tail {
                if tail.zero_filled {
                    ui.weak("The rest is zero-filled rather than standard padding.");
                } else if tail.matches + tail.mismatches > 0 {
                    ui.weak(format!(
                        "Padding after it: {} of {} readable bits as expected.",
                        tail.matches,
                        tail.matches + tail.mismatches
                    ));
                }
            }
        }
        EndOfData::Lost { bit, reason } => {
            let why = match reason {
                LostReason::UnknownHeader => "a segment header there has unreadable bits".to_owned(),
                LostReason::InvalidMode(value) => format!("mode indicator {value:b} does not exist"),
                LostReason::InvalidEci => "an ECI designator starts with a lead byte of no defined length".to_owned(),
                LostReason::Overrun => "a segment claims more data than the symbol holds".to_owned(),
            };
            ui.colored_label(warn, format!("Parsing stops at bit {bit}: {why}."));
        }
    }
}

fn stream_section(ui: &mut egui::Ui, app: &mut App) {
    let derived = &app.derived;
    let visuals = ui.visuals().clone();
    let font = egui::FontId::monospace(12.0);
    let mut job = egui::text::LayoutJob::default();
    job.wrap.max_width = ui.available_width();
    for (j, &cw) in derived.layout.data_stream.iter().enumerate() {
        let item = &derived.report.codewords[cw];
        let exists = derived.layout.codewords[cw].module_mask();
        let (value, known) = item.best();
        let text = if known & exists == exists { format!("{value:02X}") } else { "??".to_owned() };
        let color = match item.status {
            CwStatus::Clean | CwStatus::Corrected | CwStatus::Filled => visuals.text_color(),
            CwStatus::Unverified | CwStatus::Inferred => visuals.warn_fg_color,
            CwStatus::Unresolved => visuals.error_fg_color,
        };
        let mut format = egui::TextFormat::simple(font.clone(), color);
        if app.view.selected == Some(cw) {
            format.background = visuals.selection.bg_fill;
        }
        job.append(&text, if j == 0 { 0.0 } else { 5.0 }, format);
    }
    ui.label(job);
    ui.small("Data codewords in message order, after error correction where it succeeded.");
}
