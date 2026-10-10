//! Right panel: what the decode pipeline made of the grid, stage by stage.

use eframe::egui::{self, Color32, RichText};
use qr_forensics::decode::bch::InfoRead;
use qr_forensics::decode::bitstream::{Certainty, EndOfData, LostReason, SegmentContent};
use qr_forensics::decode::codewords::PinSource;
use qr_forensics::decode::pipeline::{BlockOutcome, BlockReport, CodewordReport, CwStatus, SolveMethod};
use qr_forensics::decode::rs::RsFailure;
use qr_forensics::decode::sequence::ParityCheck;
use qr_forensics::decode::text::{Charset, TextReport, UNKNOWN_CHAR, control_picture};
use qr_forensics::symbol::{Family, SymbolVersion};
use qr_forensics::symbol::layout::{CodewordSpan, CwId};

use super::state::*;
use super::{Action, App, Transform, ec_name};

const GOOD: Color32 = Color32::from_rgb(40, 170, 80);

pub(super) fn show(ui: &mut egui::Ui, app: &mut App, actions: &mut Vec<Action>) {
    app.view.wheel.follow(ui.input(|i| i.pointer.hover_pos()));
    result_section(ui, app, actions);
    section(ui, "Format and version", true, |ui| format_section(ui, app, actions));
    section(ui, "Blocks", true, |ui| blocks_section(ui, app));
    section(ui, "Codewords", true, |ui| codewords_section(ui, app, actions));
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
                "{} of {total} blocks solved with nothing left over to check the answer.",
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
        let shown = latched(area, &app.view.wheel, InnerList::ResultText).show(ui, |ui| match app.view.result_mode {
            ResultMode::Text => ui.label(text_job(ui, &report.text)),
            other => ui.label(mono(shown_as(&report.text, other))),
        });
        app.view.wheel.place(InnerList::ResultText, laid_out(&shown));
    });
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
        egui::ComboBox::from_id_salt("charset").selected_text(charset.label()).show_ui(ui, |ui| {
            for candidate in Charset::ALL {
                ui.selectable_value(&mut charset, candidate, candidate.label());
            }
        });
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
    if report.parse.segments.iter().any(|s| matches!(s.content, SegmentContent::StructuredAppend { .. }))
        && ui
            .button("Add to sequence")
            .on_hover_text("Collect this symbol's part of the message under Structured append, with the others")
            .clicked()
        {
            actions.push(Action::AddToSequence);
        }
    ui.horizontal(|ui| {
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
            .button("Export image…")
            .on_hover_text("Save the repaired symbol as a PNG that an ordinary reader can scan")
            .clicked()
        {
            actions.push(Action::ExportImage);
        }
    });
    ui.add_space(4.0);
}

/// The message laid out with its certainties coloured.
fn text_job(ui: &egui::Ui, text: &TextReport) -> egui::text::LayoutJob {
    let visuals = ui.visuals();
    let font = egui::FontId::monospace(14.0);
    let mut job = egui::text::LayoutJob::default();
    job.wrap.max_width = ui.available_width();
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
        job.append(&displayable(&span.text), 0.0, format);
    }
    job
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
            ui.label(text_job(ui, &assembled));
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
                 agrees better. Hover a count to see each copy.",
            );
            ui.weak("");
            ui.end_row();
            for candidate in analysis.candidates.iter().take(4) {
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
                let per_copy: Vec<String> = candidate
                    .per_copy
                    .iter()
                    .enumerate()
                    .map(|(i, (wrong, compared))| format!("copy {}: {wrong} of {compared}", i + 1))
                    .collect();
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
        egui::Grid::new("brute").num_columns(5).striped(true).show(ui, |ui| {
            for heading in ["EC", "Mask", "Blocks", "Text", ""] {
                ui.weak(heading);
            }
            ui.end_row();
            for attempt in brute.attempts.iter().take(6) {
                let report = &attempt.report;
                let current = attempt.ec == derived.ec && attempt.mask == derived.mask;
                let color = if report.is_trustworthy() { GOOD } else { ui.visuals().text_color() };
                ui.colored_label(color, ec_name(version, attempt.ec));
                ui.colored_label(color, attempt.mask.to_string());
                ui.colored_label(color, format!("{}/{}", report.blocks_verified(), report.blocks.len()))
                    .on_hover_text("Blocks solved and confirmed");
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
    }
}

fn outcome_text(block: &BlockReport) -> (String, bool) {
    match block.outcome {
        BlockOutcome::Solved { method, errors, erasures, slack, confidence_bits, beyond_standard } => {
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
                SolveMethod::PinnedOnly => "solved from pins alone".to_owned(),
            };
            if beyond_standard {
                text.push_str(&format!(
                    " — beyond the standard's limit: {} of the EC codewords are reserved for detection, not correction, \
                     and this solution spent them",
                    block.reserved
                ));
            } else if confidence_bits < 8 {
                text.push_str(" — unconfirmed");
            }
            if block.half_padded_with_ones {
                text.push_str(
                    " — with the final four-bit codeword's missing nibble taken as ones: the symbol's encoder computed its \
                     error correction over that, not over the standard's zeros",
                );
            }
            (text, true)
        }
        BlockOutcome::Failed(failure) => {
            let mut text = match failure {
                RsFailure::TooManyErasures { erasures, capacity } => {
                    format!("{erasures} unknown codewords, limit {capacity}: resolve {} more", erasures - capacity)
                }
                RsFailure::PinConflict { .. } => "the only solution contradicts a pin".to_owned(),
                _ => "too many wrong modules, or the wrong EC level or mask".to_owned(),
            };
            if block.inferred_bits > 0 {
                text.push_str(&format!("; {} of {} unknown bits inferred", block.inferred_bits, block.unknown_bits));
            }
            (text, false)
        }
    }
}

fn blocks_section(ui: &mut egui::Ui, app: &mut App) {
    let report = &app.derived.report;
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
                ui.colored_label(color, text);
                let mut shown = app.view.highlight_block == Some(b);
                if block.erased_codewords > 0
                    && !solved
                    && ui
                        .toggle_value(&mut shown, "Show")
                        .on_hover_text("Outline this block's unknown codewords on the grid")
                        .changed()
                {
                    app.view.highlight_block = shown.then_some(b);
                    app.view.stage = Stage::Edit;
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
                Some(PinSource::Padding) => " pad",
                Some(PinSource::Terminator) => " end",
                Some(PinSource::Header) => " hdr",
                None => "",
            };
            let text = format!(
                "{cw:>4}  B{:<2} {kind}{index:<3}  {:<8}  {value}  {}{pin}",
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
}

fn padding_section(ui: &mut egui::Ui, app: &mut App, actions: &mut Vec<Action>) {
    let derived = &app.derived;
    let warn = ui.visuals().warn_fg_color;
    let pins = &app.doc.pins;
    let count = |source: PinSource| pins.values().filter(|p| p.source == source).count();
    let (manual, padding, terminator, header) =
        (count(PinSource::Manual), count(PinSource::Padding), count(PinSource::Terminator), count(PinSource::Header));

    if !pins.is_empty() {
        ui.label(format!("Pins: {manual} by hand, {padding} padding, {terminator} terminator, {header} header"));
        if app.doc.pins_ec != Some(derived.ec) {
            ui.colored_label(
                warn,
                "These pins were made under a different EC level and are not applied. Remove them or switch back.",
            );
        }
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
            ui.weak("No standard padding found. The message may fill the symbol, or the encoder padded differently.");
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
}

fn segments_section(ui: &mut egui::Ui, app: &mut App) {
    let report = &app.derived.report;
    let warn = ui.visuals().warn_fg_color;
    if report.parse.segments.is_empty() {
        ui.weak("No segments read.");
    }
    for segment in &report.parse.segments {
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
        let text = format!(
            "{} · bits {} to {} · {detail}",
            segment.mode.label(),
            segment.header.start,
            segment.payload.end
        );
        if segment.damaged || segment.truncated || segment.header_certainty != Certainty::Verified {
            ui.colored_label(warn, text).on_hover_text("Header not confirmed, or contents invalid");
        } else {
            ui.label(text);
        }
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
