//! Left panel: which symbol this is, the photo, and how to get modules in.

use eframe::egui;
use qr_forensics::imaging::adjust::{Adjust, Channel, Enlarge, Filter, LENS_MIN_PAIRS, MAX_ENLARGED_SIDE};
use qr_forensics::imaging::binarize::BinarizeParams;
use qr_forensics::imaging::detect::anchor_sites;
use qr_forensics::imaging::sample::Kernel;
use qr_forensics::symbol::function_map::looks_inverted;
use qr_forensics::symbol::grid::Module;
use qr_forensics::symbol::{Family, SymbolVersion};

use super::state::*;
use super::{Action, App, JobKind, Transform};

/// Why a button is disabled while a job runs.
const BUSY_HINT: &str = "Wait for the step under way to finish";

pub(super) fn show(ui: &mut egui::Ui, app: &mut App, actions: &mut Vec<Action>) {
    symbol_section(ui, app, actions);
    ui.separator();
    photo_section(ui, app, actions);
    if app.photo.is_some() {
        ui.separator();
        alignment_section(ui, app, actions);
        ui.separator();
        sampling_section(ui, app, actions);
    }
    ui.separator();
    edit_section(ui, app, actions);
}

fn percent(part: usize, whole: usize) -> String {
    if whole == 0 { "n/a".into() } else { format!("{:.1}%", 100.0 * part as f64 / whole as f64) }
}

fn symbol_section(ui: &mut egui::Ui, app: &mut App, actions: &mut Vec<Action>) {
    ui.heading("Symbol");
    let version = app.doc.version;
    let family = version.family();
    ui.horizontal(|ui| {
        for candidate in Family::ALL {
            let label = ui.selectable_label(family == candidate, candidate.label());
            let label = if candidate == Family::Model1 {
                label.on_hover_text(
                    "The original 1994 QR Code: no alignment patterns, and its own data layout. The photo search \
                     does not look for it; the inspector offers to read a grid as Model 1 when its format bits \
                     say so.",
                )
            } else {
                label
            };
            if label.clicked() && family != candidate {
                actions.push(Action::SetVersion(super::family_start(candidate)));
            }
        }
    });
    ui.horizontal(|ui| {
        ui.label("Version");
        match version {
            SymbolVersion::Qr(v) | SymbolVersion::Model1(v) => {
                let (make, last): (fn(u8) -> SymbolVersion, u8) = match version {
                    SymbolVersion::Model1(_) => (SymbolVersion::Model1, 14),
                    _ => (SymbolVersion::Qr, 40),
                };
                let mut value = v;
                if ui.add(egui::DragValue::new(&mut value).range(1..=last).speed(0.1)).changed() {
                    actions.push(Action::SetVersion(make(value)));
                }
                if ui.small_button("−").clicked() && v > 1 {
                    actions.push(Action::SetVersion(make(v - 1)));
                }
                if ui.small_button("+").clicked() && v < last {
                    actions.push(Action::SetVersion(make(v + 1)));
                }
            }
            SymbolVersion::Micro(v) => {
                for candidate in 1..=4 {
                    if ui.selectable_label(v == candidate, format!("M{candidate}")).clicked() {
                        actions.push(Action::SetVersion(SymbolVersion::Micro(candidate)));
                    }
                }
            }
            SymbolVersion::Rmqr(_) => {
                egui::ComboBox::from_id_salt("rmqr_version").selected_text(version.to_string()).show_ui(ui, |ui| {
                    for candidate in Family::Rmqr.versions() {
                        if ui.selectable_label(candidate == version, candidate.to_string()).clicked() {
                            actions.push(Action::SetVersion(candidate));
                        }
                    }
                });
            }
        }
    });
    ui.weak(format!("{} × {} modules", version.width(), version.height()));
    ui.weak("Changing the version starts a new grid (undoable).");
}

fn photo_section(ui: &mut egui::Ui, app: &mut App, actions: &mut Vec<Action>) {
    ui.heading("Photo");
    ui.horizontal(|ui| {
        if ui.button("Open…").clicked() {
            actions.push(Action::OpenImage);
        }
        let paste = format!("Use the image on the clipboard, or an image file copied there ({})", ui.ctx().format_shortcut(&super::PASTE));
        if ui.button("Paste").on_hover_text(paste).clicked() {
            actions.push(Action::PasteImage);
        }
        if app.photo.is_some() && ui.button("Close").clicked() {
            actions.push(Action::ClosePhoto);
        }
    });
    let Some(photo) = &app.photo else {
        ui.weak("Optional. Drop an image on the window, or paint the grid by hand.");
        return;
    };
    let mut about = format!("{} · {} × {}", photo.name, photo.width, photo.height);
    let (full_width, full_height) = photo.source.dimensions();
    if photo.crop.is_some() {
        about.push_str(&format!(" · cropped from {full_width} × {full_height}"));
    } else if photo.scale() > 1.0 {
        about.push_str(&format!(" · enlarged from {full_width} × {full_height}"));
    } else if photo.scale() < 1.0 {
        about.push_str(&format!(" · reduced from {full_width} × {full_height}"));
    }
    ui.weak(about);
    let (width, height) = (photo.width, photo.height);
    let cropped = photo.crop.is_some();
    // What the photo starts with, which Reset goes back to: no
    // adjustment but the enlargement a small photo is given.
    let fresh = Adjust::for_photo(photo.original.width(), photo.original.height());
    let enlargements = photo.enlargements.clone();
    ui.horizontal_wrapped(|ui| {
        if ui
            .button("Crop to outline")
            .on_hover_text(
                "Cut the photo down to the outline, with room about it. A search of the whole photo, which looks at no \
                 more than 4096 pixels on the longer side, then sees that part at its own size; every change of the \
                 adjustments works on that part alone; and the canvas draws it pixel for pixel, where a photo larger \
                 than a texture may be is drawn reduced. The part keeps the photo's adjustments, and is as much \
                 enlarged or reduced as the photo was. What was found in the photo is dropped; the outline and \
                 anchors stay on the symbol.",
            )
            .clicked()
        {
            actions.push(Action::CropToOutline);
        }
        if cropped && ui.button("Whole photo").on_hover_text("Show the whole photo again, with the outline where it was on it").clicked() {
            actions.push(Action::ShowWholePhoto);
        }
    });

    // The controls work on a copy; the document takes the change through
    // an action, after one undo point per gesture.
    let mut adjust = app.doc.adjust;
    let mut checkpoint = false;
    let pointer_down = ui.input(|i| i.pointer.any_down());
    let mut gesture = app.view.adjust_gesture && pointer_down;
    // A slider changes its value on every frame of a drag, and on the
    // press before egui calls it a drag: the first change of a press
    // takes the undo point, and a change with no button held, by key or
    // by a click's release, takes one of its own.
    let mut note = |changed: bool| {
        if changed && (!pointer_down || !gesture) {
            checkpoint = true;
            gesture = pointer_down;
        }
    };
    egui::Grid::new("adjust").num_columns(2).show(ui, |ui| {
        ui.label("Channel");
        let mut changed = false;
        egui::ComboBox::from_id_salt("channel").selected_text(adjust.channel.label()).show_ui(ui, |ui| {
            for channel in Channel::ALL {
                changed |= ui.selectable_value(&mut adjust.channel, channel, channel.label()).changed();
            }
        });
        note(changed);
        ui.end_row();
        ui.label("Brightness");
        note(ui.add(egui::Slider::new(&mut adjust.brightness, -1.0..=1.0)).changed());
        ui.end_row();
        ui.label("Contrast");
        note(ui.add(egui::Slider::new(&mut adjust.contrast, -1.0..=1.0)).changed());
        ui.end_row();
        ui.label("Sharpen");
        let sharpen = ui.add(egui::Slider::new(&mut adjust.sharpen, 0.0..=1.0)).on_hover_text(
            "Sharpens the photo with an unsharp mask, in its colours: for a symbol out of focus, or modules a pixel or two \
             across that the lens has run together, which read as grey as their neighbours. The mask is wider and \
             stronger the further this is set, a pixel wide at 0.5 and two at 1. Find in outline tries the photo \
             sharpened and enlarged of itself when it does not decode as it is, and sets this and Enlarge to what \
             decoded.",
        );
        note(sharpen.changed());
        ui.end_row();
        ui.label("Enlarge");
        // The slider goes from step to step of those on offer for this
        // photo: each twice the last, with what the photo started at,
        // and what a read set, among them.
        let last = enlargements.len().saturating_sub(1);
        let mut step = enlargements.iter().position(|&on_offer| on_offer == adjust.enlarge).unwrap_or(0);
        let slider = egui::Slider::new(&mut step, 0..=last)
            .custom_formatter(|step, _| enlargements.get(step.round() as usize).map_or_else(String::new, Enlarge::to_string))
            .custom_parser(|text| {
                let asked = Enlarge::parse(text)?.factor();
                let nearest = (0..enlargements.len()).min_by(|&a, &b| {
                    (enlargements[a].factor() - asked).abs().total_cmp(&(enlargements[b].factor() - asked).abs())
                });
                nearest.map(|step| step as f64)
            });
        let enlarge = ui.add_enabled(last > 0, slider).on_hover_text(format!(
            "The size the photo is worked on at, against its own: resized with Magic Kernel Sharp 2013, in its colours. \
             Enlarged, modules a pixel or two across become modules the search and the sampling can see, and a small \
             photo starts enlarged for that. Reduced, grain and the dots of a halftone screen run together, and a large \
             photo is quicker to work on. The outline and the anchors go with the photo. Any photo may be enlarged {} \
             times, and a small one further, as far as {MAX_ENLARGED_SIDE} pixels on its longer side. A large photo \
             enlarged takes time and memory to match: for a small symbol in one, cropping to its outline first is \
             quicker.",
            Enlarge::ALWAYS.factor(),
        ));
        if enlarge.changed()
            && let Some(&on_offer) = enlargements.get(step)
        {
            adjust.enlarge = on_offer;
        }
        note(enlarge.changed());
        ui.end_row();
        ui.label("Lens");
        ui.horizontal(|ui| {
            // The slider steps and clamps what is set through it, and
            // leaves a term set otherwise as it is: an estimate comes to
            // the thousandth, and the corners and anchors have been moved
            // for exactly that term, so it must not be rounded to the
            // slider's step behind them on the next frame.
            let slider = egui::Slider::new(&mut adjust.k1, -0.3..=0.3)
                .step_by(0.005)
                .fixed_decimals(3)
                .clamping(egui::SliderClamping::Edits);
            // A lens bends the photo about its centre, which a crop has
            // lost: the term is set on the whole photo, and applied
            // before the cut.
            const CROPPED_HINT: &str = "A lens bends the photo about its centre, which a cropped photo has lost: show the \
                                        whole photo to set it, and crop again";
            let lens = ui
                .add_enabled(!cropped, slider)
                .on_hover_text(
                    "Takes radial lens distortion out of the photo. Negative straightens edges that bow outwards \
                     (barrel), positive edges that bow inwards (pincushion).",
                )
                .on_disabled_hover_text(CROPPED_HINT);
            note(lens.changed());
            let enough = app.doc.anchors.len() + 4 >= LENS_MIN_PAIRS;
            if ui
                .add_enabled(enough && !cropped, egui::Button::new("Estimate"))
                .on_hover_text(
                    "Fit the lens setting to the corners and anchors, which lie on a plane once the distortion is \
                     gone. They move with the photo.",
                )
                .on_disabled_hover_text(if cropped {
                    CROPPED_HINT.to_owned()
                } else {
                    format!("Needs the four corners and at least {} anchors on the symbol first: press Find in outline", LENS_MIN_PAIRS - 4)
                })
                .clicked()
            {
                actions.push(Action::EstimateLens);
            }
        });
        ui.end_row();
        ui.label("Filter");
        let mut changed = false;
        egui::ComboBox::from_id_salt("filter").selected_text(adjust.filter.label()).show_ui(ui, |ui| {
            for filter in Filter::ALL {
                changed |= ui.selectable_value(&mut adjust.filter, filter, filter.label()).on_hover_text(filter.describe()).changed();
            }
        });
        note(changed);
        ui.end_row();
        if adjust.filter != Filter::None {
            ui.label("Radius");
            let radius = ui.add(egui::DragValue::new(&mut adjust.radius).range(1..=Filter::MAX_RADIUS).suffix(" px")).on_hover_text(
                "Half-width of the filter's neighbourhood, in photo pixels. With a filter on, the photo is shown \
                 as the filtered grey that sampling reads.",
            );
            note(radius.changed());
            ui.end_row();
        }
    });
    ui.horizontal_wrapped(|ui| {
        note(ui.checkbox(&mut adjust.invert, "Invert photo").on_hover_text("For light symbols on a dark background").changed());
        if ui.add_enabled(adjust != fresh, egui::Button::new("Reset adjustments")).clicked() {
            adjust = fresh;
            note(true);
        }
    });
    // A way of looking, like the binarised view: not an adjustment, and
    // not undone or reset with them.
    ui.checkbox(&mut app.view.greyscale, "Show in greyscale").on_hover_text(
        "Show the photo in grey, as the search and the sampling read it: the luminance of its colours under the \
         adjustments above. Only what is shown changes, and what is read is the same either way. With a single \
         channel chosen, or a filter on, the photo is shown so already.",
    );
    app.view.adjust_gesture = gesture;
    if checkpoint {
        app.history.checkpoint(&app.doc);
    }
    if adjust != app.doc.adjust {
        actions.push(Action::SetAdjust(adjust));
    }

    // The binarised view is a way of looking, not a reading: its settings
    // are not undoable, and are shown only while the view is, which is
    // the only time they do anything.
    if !(app.view.stage == Stage::Align && app.view.align_overlay == AlignOverlay::Binarised) {
        return;
    }
    ui.add_space(4.0);
    ui.weak("Binarised view").on_hover_text(
        "The photo in black and white, as the search for finder patterns sees it. The search reads it both ways, \
         local and global, and pools what it finds.",
    );
    let defaults = BinarizeParams::for_outline(app.doc.corners, width, height);
    if app.view.binarize_auto {
        app.view.binarize = defaults;
    }
    let mut changed = false;
    let params = &mut app.view.binarize;
    egui::Grid::new("binarize").num_columns(2).show(ui, |ui| {
        ui.label("Radius");
        changed |= ui
            .add(egui::DragValue::new(&mut params.radius).range(BinarizeParams::MIN_RADIUS..=BinarizeParams::MAX_RADIUS).speed(2.0).suffix(" px"))
            .on_hover_text(
                "How far around each pixel the local threshold looks, in photo pixels. Automatically a sixth of \
                 the outline's shorter side, wider than a finder pattern.",
            )
            .changed();
        ui.end_row();
        ui.label("Threshold");
        changed |= ui
            .checkbox(&mut params.global, "Global")
            .on_hover_text(
                "One threshold for the whole outline and its margin (Otsu's), instead of the local one",
            )
            .changed();
        ui.end_row();
    });
    if changed {
        app.view.binarize_auto = false;
    }
    if ui
        .add_enabled(!app.view.binarize_auto, egui::Button::new("Reset to automatic"))
        .on_hover_text("Set both from the outline again, as the search does")
        .clicked()
    {
        app.view.binarize = defaults;
        app.view.binarize_auto = true;
    }
}

fn alignment_section(ui: &mut egui::Ui, app: &mut App, actions: &mut Vec<Action>) {
    ui.heading("Alignment");
    ui.weak(
        "Drag TL, TR, BR and BL loosely around the symbol and press Find in outline, or put them on its outer \
         corners by hand. Arrow keys nudge the last corner touched; hold H to see the bare photo.",
    );
    ui.weak(
        "Anchors pin the fixed patterns to the photo, and the grid bends through them. Drag a hollow handle onto \
         its pattern to set one; right-click a handle to remove it.",
    );
    if let Some(preview) = &app.preview {
        let (matching, known) = preview.fixed;
        ui.label(format!("Fixed patterns match: {} ({matching} of {known})", percent(matching, known)))
            .on_hover_text("How many finder, timing and alignment modules the photo shows in the right colour at this version and alignment");
        if looks_inverted(preview.fixed) {
            let adjust = app.doc.adjust;
            let (why, label) = if adjust.invert {
                ("They read inverted: Invert photo looks wrong for this photo.", "Turn off Invert photo")
            } else {
                ("They read inverted: this looks like a light symbol on a dark background.", "Turn on Invert photo")
            };
            ui.colored_label(ui.visuals().warn_fg_color, why);
            if ui.button(label).on_hover_text("The same as the Invert photo checkbox under Photo").clicked() {
                app.history.checkpoint(&app.doc);
                actions.push(Action::SetAdjust(Adjust { invert: !adjust.invert, ..adjust }));
            }
        } else if app.doc.adjust.invert {
            ui.weak("Invert photo is on: the photo is read as a light symbol on a dark background.")
                .on_hover_text("Set by the checkbox under Photo, or by a detection or sampling that found the symbol light on dark");
        }
    } else {
        ui.colored_label(ui.visuals().warn_fg_color, "The corners do not form a usable outline.");
    }
    {
        let sites = anchor_sites(&app.derived.func).len();
        let detected = app.doc.anchors.iter().filter(|a| a.source == AnchorSource::Detected).count();
        let manual = app.doc.anchors.len() - detected;
        let partial = app.doc.anchors.iter().filter(|a| a.anchor.partial).count();
        let mut text = format!("Anchors: {} of {sites} sites set ({detected} found, {manual} by hand)", app.doc.anchors.len());
        if partial > 0 {
            text.push_str(&format!(", {partial} partly damaged"));
        }
        ui.label(text).on_hover_text(
            "A site is a finder pattern, an alignment pattern, the rMQR sub-pattern or a module of a timing pattern",
        );
        ui.horizontal(|ui| {
            ui.checkbox(&mut app.view.use_anchors, "Bend the grid through the anchors")
                .on_hover_text("Off: one flat perspective through the four corners, as if the symbol were on a plane");
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut app.view.show_anchors, "Show anchor handles").on_hover_text(
                "A handle on every fixed pattern and three along each edge. Drag one to where its pattern or edge \
                 really lies and the grid bends through it. Edge handles suit a symbol on a bottle or curled paper, \
                 whose sides bow.",
            );
            if ui.add_enabled(!app.doc.anchors.is_empty(), egui::Button::new("Clear anchors")).clicked() {
                actions.push(Action::ClearAnchors);
            }
        });
    }
    // The steps that take a second run on a thread of their own, one at a
    // time: while one runs, its button says so and the others wait.
    let running = app.running();
    let busy = running.is_some();
    ui.horizontal_wrapped(|ui| {
        if ui.button("Reset corners").clicked() {
            actions.push(Action::ResetCorners);
        }
        let label = if running == Some(JobKind::FindCorners) { "Finding…" } else { "Find in outline" };
        if ui
            .add_enabled(!busy, egui::Button::new(label))
            .on_hover_text(
                "Find the symbol inside the outline and move the corners and anchors onto it. The outline only has \
                 to be roughly around the symbol, and may be a quarter turn out. Also picks the version that fits.",
            )
            .on_disabled_hover_text(BUSY_HINT)
            .clicked()
        {
            actions.push(Action::FindCorners);
        }
        let label = if running == Some(JobKind::Refine) { "Refining…" } else { "Refine" };
        if ui
            .add_enabled(!busy, egui::Button::new(label))
            .on_hover_text(
                "Nudge the corners, then the anchors, to where the fixed patterns read best, to a fraction of a \
                 pixel. Place the corners within half a module first.",
            )
            .on_disabled_hover_text(BUSY_HINT)
            .clicked()
        {
            actions.push(Action::RefineCorners);
        }
        let label = if running == Some(JobKind::Suggest) { "Ranking…" } else { "Suggest version" };
        if ui
            .add_enabled(!busy, egui::Button::new(label))
            .on_hover_text("Try every version on these corners and rank them by fixed-pattern match")
            .on_disabled_hover_text(BUSY_HINT)
            .clicked()
        {
            actions.push(Action::SuggestVersions);
        }
        if app.search.is_some() {
            if ui.button("Stop searching").on_hover_text("Call the search of the photo off").clicked() {
                actions.push(Action::CancelSearch);
            }
        } else if ui
            .button("Search whole photo")
            .on_hover_text(
                "Search the whole photo for symbols and list what is found, those that decode as found first. Takes \
                 a few seconds, and longer on a damaged symbol, whose finder patterns are tried every way; it can be \
                 stopped. A photo over 4096 pixels on a side is searched in a copy reduced to that, and what is found \
                 is placed and read in the photo's own pixels; a symbol too small to show in the copy is found by \
                 Find in outline, which looks at the photo itself.",
            )
            .clicked()
        {
            actions.push(Action::FindSymbols);
        }
        if busy && ui.button("Stop").on_hover_text("Call off the step under way; nothing is changed").clicked() {
            actions.push(Action::CancelJob);
        }
    });
    let family = app.doc.version.family();
    ui.checkbox(&mut app.view.own_family_only, format!("Only look for {}", family.label())).on_hover_text(
        "Search whole photo and Find in outline look for symbols of the family chosen under Symbol, and no other. The \
         other families are otherwise looked for from every finder pattern the first search leaves over, which on a \
         damaged QR Code is most of the time a search takes.",
    );
    ui.checkbox(&mut app.view.search.coarse_to_fine, "Search a small copy first").on_hover_text(
        "Search a copy of the photo shrunk to a thousand pixels first, and the photo at the full size a search goes \
         to, 4096 pixels on its longer side, only if that shows nothing. Much faster on a large photo. Turn off for \
         a symbol only a few pixels a module, which the copy loses.",
    );
    if let Some(proposals) = &app.proposals {
        let in_use = super::proposal_in_use(proposals, &app.doc.corners);
        ui.label(format!("Found in the photo ({}):", proposals.found.len()));
        for (i, found) in proposals.found.iter().enumerate() {
            let decodes = if found.decodes == Some(true) { ", decodes" } else { "" };
            let confirmed = if found.version_confirmed() { ", version confirmed" } else { "" };
            let polarity = if found.inverted { ", light on dark" } else { "" };
            let retry = found.retry.map_or(String::new(), |r| format!(", with {r}"));
            let text = format!(
                "{}. {} {}  —  {:.0}% fixed{decodes}{confirmed}{polarity}{retry}",
                i + 1,
                found.version.family().label(),
                found.version,
                100.0 * found.score()
            );
            let placed = in_use == Some(i);
            if ui
                .add_enabled(!busy, egui::Button::selectable(placed, text))
                .on_hover_text(if placed {
                    "The outline is on this symbol. Click to place the corners and anchors on it again, as found.".to_owned()
                } else {
                    format!(
                        "Place the corners and anchors on this symbol; its outline on the photo can be clicked too.{}",
                        super::new_grid_note(found, app.doc.version)
                    )
                })
                .on_disabled_hover_text(BUSY_HINT)
                .clicked()
            {
                actions.push(Action::UseProposal(i));
            }
        }
        ui.horizontal(|ui| {
            let label = if running == Some(JobKind::ReadFound) { "Reading…" } else { "Read all" };
            if ui
                .add_enabled(!busy, egui::Button::new(label))
                .on_hover_text(
                    "Read every symbol found through to its message, as the command line reads a photo. The messages \
                     are listed under Symbols read, where each can be copied or loaded into the grid, and the parts \
                     of a structured append message go into the sequence.",
                )
                .on_disabled_hover_text(BUSY_HINT)
                .clicked()
            {
                actions.push(Action::ReadFound);
            }
            if ui.small_button("Clear the list").clicked() {
                actions.push(Action::ClearProposals);
            }
        });
    }
    if let Some(guesses) = &app.guesses {
        for guess in guesses.iter().take(4) {
            let polarity = if guess.inverted { ", inverted" } else { "" };
            let text = format!("{}  —  {:.0}%{polarity}", guess.version, 100.0 * guess.score());
            let response = ui.selectable_label(guess.version == app.doc.version, text);
            if response.clicked() {
                actions.push(Action::SetVersion(guess.version));
            }
            if guess.inverted {
                response.on_hover_text(
                    "The fixed patterns of this version match with dark and light swapped: the photo shows a \
                     light symbol on a dark background. Tick Invert photo before sampling.",
                );
            }
        }
    }
}

fn sampling_section(ui: &mut egui::Ui, app: &mut App, actions: &mut Vec<Action>) {
    ui.heading("Sampling");
    let params = &mut app.view.sample;
    let automatic = app.preview.as_ref().map(|p| p.classified.threshold);
    ui.checkbox(&mut params.local, "Local threshold")
        .on_hover_text(
            "Judge each module against the darkest and lightest of its neighbourhood, so shadows, glare and \
             stains do not tip it. Off, one threshold serves the whole symbol, which suits an evenly lit symbol \
             that is blurred. Find in outline tries both, and the threshold surface after them.",
        );
    if params.local {
        ui.horizontal(|ui| {
            ui.label("Neighbourhood");
            ui.add(egui::DragValue::new(&mut params.local_radius).range(2..=15).suffix(" modules"));
        });
    } else {
        let mut auto = params.threshold.is_none();
        if ui.checkbox(&mut auto, "Automatic threshold").changed() {
            params.threshold = if auto { None } else { Some(automatic.unwrap_or(0.5)) };
        }
        match &mut params.threshold {
            Some(threshold) => {
                ui.add(egui::Slider::new(threshold, 0.0..=1.0).text("threshold"));
            }
            None => {
                if let Some(value) = automatic {
                    ui.weak(format!("Threshold {value:.2}"));
                }
                ui.checkbox(&mut params.surface, "Threshold surface").on_hover_text(
                    "Let the one threshold lie as a smooth surface across the symbol, the thresholds of its four \
                     corners interpolated between them, as BoofCV reads a symbol: for a shadow across an evenly \
                     printed symbol, or light that fails toward one side, which no level threshold follows and \
                     which the local one follows along with every stain. Find in outline tries it of itself, \
                     after the local threshold and the level one, or first when it is ticked here.",
                );
            }
        }
    }
    ui.add(egui::Slider::new(&mut params.band, 0.0..=0.5).text("unsure band"))
        .on_hover_text(
            "Modules whose distance from the threshold is below this share of the local contrast are left unknown \
             instead of guessed",
        );
    ui.horizontal(|ui| {
        ui.label("Kernel");
        egui::ComboBox::from_id_salt("kernel").selected_text(params.kernel.label()).show_ui(ui, |ui| {
            for kernel in Kernel::ALL {
                ui.selectable_value(&mut params.kernel, kernel, kernel.label()).on_hover_text(kernel.describe());
            }
        });
    })
    .response
    .on_hover_text(
        "How each module's luminance is read. Find in outline tries every kernel, under every threshold rule, and \
         keeps what decodes.",
    );
    if let Some(preview) = &app.preview {
        let unknown = preview.classified.modules.iter().filter(|&&m| m == Module::Unknown).count();
        let flat = preview.classified.flat;
        let mut text = format!("{unknown} of {} modules would be left unknown", preview.classified.modules.len());
        if flat > 0 {
            text.push_str(&format!(", {flat} of them in patches with no contrast"));
        }
        ui.weak(text);
        if preview.classified.is_poorly_separated() {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                "The module luminances do not split into dark and light: the outline may be off the symbol, or \
                 the photo too washed out to read.",
            )
            .on_hover_text("A grid that straddles module boundaries averages dark with light into greys");
        }
    }
    ui.checkbox(&mut app.view.auto_invert, "Invert photo automatically when sampling")
        .on_hover_text(
            "Sampling first turns Invert photo on, or off, when the fixed patterns read inverted. Off, the photo \
             is sampled as it is and a notice says so.",
        );
    ui.horizontal(|ui| {
        if ui.button("Sample all").on_hover_text("Replace the grid with what the photo shows").clicked() {
            actions.push(Action::Sample { only_unknown: false });
        }
        if ui.button("Sample unknown only").on_hover_text("Keep everything already painted").clicked() {
            actions.push(Action::Sample { only_unknown: true });
        }
    });
}

fn edit_section(ui: &mut egui::Ui, app: &mut App, actions: &mut Vec<Action>) {
    ui.heading("Paint");
    ui.horizontal(|ui| {
        ui.selectable_value(&mut app.view.tool, Tool::Brush, "Brush").on_hover_text("B");
        ui.selectable_value(&mut app.view.tool, Tool::Rect, "Rectangle").on_hover_text("R");
        ui.selectable_value(&mut app.view.tool, Tool::Inspect, "Inspect")
            .on_hover_text("I — click a module to select its codeword");
    });
    ui.horizontal(|ui| {
        ui.selectable_value(&mut app.view.ink, Module::Dark, "Dark").on_hover_text("1");
        ui.selectable_value(&mut app.view.ink, Module::Light, "Light").on_hover_text("2");
        ui.selectable_value(&mut app.view.ink, Module::Unknown, "Unknown").on_hover_text("3");
    });
    ui.weak("Left button paints the ink; right button paints the other of dark and light; Shift paints unknown.");
    ui.checkbox(&mut app.view.lock_fixed, "Lock fixed patterns")
        .on_hover_text("Finders, timing and alignment patterns keep the colours the standard gives them");
    ui.horizontal(|ui| {
        if ui.button("Fill fixed patterns").clicked() {
            actions.push(Action::FillFixed);
        }
        if ui.button("Clear the rest").on_hover_text("Set every other module to unknown").clicked() {
            actions.push(Action::ClearData);
        }
    });
    let grid = &app.doc.grid;
    ui.weak(format!(
        "{} dark · {} light · {} unknown",
        grid.count(Module::Dark),
        grid.count(Module::Light),
        grid.count(Module::Unknown)
    ));
    let (matching, known) = app.derived.analysis.fixed;
    if app.derived.analysis.inverted {
        ui.colored_label(
            ui.visuals().warn_fg_color,
            "The fixed patterns are painted inverted: this looks like a light-on-dark symbol.",
        )
        .on_hover_text("Nothing decodes until dark and light are swapped");
        if ui.button("Swap dark and light").on_hover_text("Swap the colour of every module in the grid (undoable)").clicked() {
            actions.push(Action::Transform(Transform::Invert));
        }
    } else if let Some(mirror) = app.derived.analysis.mirrored
        && !app.derived.report.is_trustworthy()
    {
        ui.colored_label(
            ui.visuals().warn_fg_color,
            "The format bits read as a mirror image: the symbol was seen from the back.",
        )
        .on_hover_text("They spell nothing as painted, and a valid word once the grid is reflected");
        if ui.button(mirror.label()).on_hover_text("Reflect the grid (undoable)").clicked() {
            actions.push(Action::Transform(Transform::Mirror(mirror)));
        }
    } else if let Some(orientation) = app.derived.analysis.orientation {
        let (m, k) = orientation.agreement;
        let how = if orientation.inverted { " and with dark and light swapped" } else { "" };
        ui.colored_label(
            ui.visuals().warn_fg_color,
            format!(
                "The fixed patterns fit better with the grid turned{how}: {} of {k} read right that way, {matching} as painted.",
                m
            ),
        )
        .on_hover_text("The grid was painted the wrong way up, or from the back");
        let label = if orientation.inverted {
            format!("{} and swap dark and light", orientation.symmetry.label())
        } else {
            orientation.symmetry.label().to_owned()
        };
        if ui.button(label).on_hover_text("Transform the grid (undoable)").clicked() {
            actions.push(Action::Transform(if orientation.inverted {
                Transform::SymmetryInverted(orientation.symmetry)
            } else {
                Transform::Symmetry(orientation.symmetry)
            }));
        }
    } else if matching < known {
        ui.colored_label(
            ui.visuals().warn_fg_color,
            format!("{} fixed-pattern modules are painted the wrong colour.", known - matching),
        )
        .on_hover_text("Either the grid is misaligned or the wrong version is selected");
    }

    ui.separator();
    ui.heading("Decoding");
    let mut bit_level = app.opts.bit_level;
    if ui
        .checkbox(&mut bit_level, "Solve unknown bits individually")
        .on_hover_text(
            "When a block has more unknown codewords than Reed-Solomon can restore, solve for the unknown \
             modules one by one instead. Assumes every painted module is right.",
        )
        .changed()
    {
        actions.push(Action::SetBitLevel(bit_level));
    }
    let mut reliability = app.opts.reliability;
    if ui
        .checkbox(&mut reliability, "Doubt the least reliable modules")
        .on_hover_text(
            "When a block fails, treat its codewords with the faintest sampled modules as unknown, a few at a \
             time, until it solves with something to spare. Modules painted by hand are never doubted.",
        )
        .changed()
    {
        actions.push(Action::SetReliability(reliability));
    }
    if reliability && !app.doc.grid.has_reliability() {
        ui.weak("Takes effect once modules are sampled from a photo.");
    }
}
