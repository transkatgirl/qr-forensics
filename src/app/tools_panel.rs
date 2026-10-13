//! Left panel: which symbol this is, the photo, and how to get modules in.

use eframe::egui;
use qr_forensics::imaging::adjust::{Adjust, Channel, Enlarge, Filter, LENS_MIN_PAIRS, MAX_ENLARGED_SIDE};
use qr_forensics::imaging::binarize::{BinarizeParams, WINDOW_RADII};
use qr_forensics::imaging::detect::{AnchorKind, Detection, anchor_sites};
use qr_forensics::imaging::sample::{Kernel, POOR_SEPARABILITY, SHARPEN_BELOW_PX};
use qr_forensics::symbol::function_map::looks_inverted;
use qr_forensics::symbol::grid::Module;
use qr_forensics::symbol::{Family, SymbolVersion};

use super::state::*;
use super::{Action, App, JobKind, Transform};

/// Why a button is disabled while a job runs.
const BUSY_HINT: &str = "Wait for the step under way to finish";
/// How many of the versions Suggest version ranks are listed until the
/// rest are asked for.
const GUESSES_SHOWN: usize = 4;

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
    // the only time they do anything, unless they are set for the scan,
    // when they are what the scan reads by and stay on show.
    let shown = app.view.stage == Stage::Align && app.view.align_overlay == AlignOverlay::Binarised;
    if !(shown || app.view.binarize_for_scan) {
        return;
    }
    ui.add_space(4.0);
    ui.weak("Binarised view").on_hover_text(
        "The photo in black and white, as the search for finder patterns sees it. The search reads it three ways, \
         against the lighting around each pixel, against one threshold for the outline, and against the mean of a small \
         window about each pixel at each of a few widths, and pools what it finds.",
    );
    let defaults = BinarizeParams::for_outline(app.doc.corners, width, height);
    if app.view.binarize_auto {
        app.view.binarize = defaults;
    }
    let mut changed = false;
    let params = &mut app.view.binarize;
    egui::Grid::new("binarize").num_columns(2).show(ui, |ui| {
        ui.label("Threshold");
        ui.horizontal(|ui| {
            let local = !params.global && params.window.is_none();
            if ui
                .selectable_label(local, "Local")
                .on_hover_text("Each pixel against the lighting around it, the mean of the blocks of pixels within the radius")
                .clicked()
                && !local
            {
                (params.global, params.window) = (false, None);
                changed = true;
            }
            let global = params.global && params.window.is_none();
            if ui
                .selectable_label(global, "Global")
                .on_hover_text("One threshold for the whole outline and its margin (Otsu's), instead of the local one")
                .clicked()
                && !global
            {
                (params.global, params.window) = (true, None);
                changed = true;
            }
            if ui
                .selectable_label(params.window.is_some(), "Window")
                .on_hover_text(
                    "Each pixel against the mean of a window a few pixels about it: the third way the search reads the \
                     outline, which tells the rings of a finder pattern apart where blur has run them together until \
                     neither of the other two does.",
                )
                .clicked()
                && params.window.is_none()
            {
                params.window = Some(WINDOW_RADII[1]);
                changed = true;
            }
        });
        ui.end_row();
        match params.window {
            Some(window) => {
                ui.label("Half-width");
                ui.horizontal(|ui| {
                    for radius in WINDOW_RADII {
                        if ui.selectable_label(window == radius, format!("{radius} px")).clicked() && window != radius {
                            params.window = Some(radius);
                            changed = true;
                        }
                    }
                })
                .response
                .on_hover_text(
                    "How far either side of each pixel the window reaches. A width serves modules from a third of it to \
                     about the whole of it across; the search reads the outline at each of these that is no wider than \
                     its local radius.",
                );
                ui.end_row();
            }
            None if !params.global => {
                ui.label("Radius");
                changed |= ui
                    .add(egui::DragValue::new(&mut params.radius).range(BinarizeParams::MIN_RADIUS..=BinarizeParams::MAX_RADIUS).speed(2.0).suffix(" px"))
                    .on_hover_text(
                        "How far around each pixel the local threshold looks, in photo pixels. Automatically a sixth of \
                         the outline's shorter side, wider than a finder pattern.",
                    )
                    .changed();
                ui.end_row();
            }
            None => {}
        }
    });
    if changed {
        app.view.binarize_auto = false;
    }
    if ui
        .add_enabled(!app.view.binarize_auto, egui::Button::new("Reset to automatic"))
        .on_hover_text("The local threshold again, its radius set from the outline as the search sets it")
        .clicked()
    {
        app.view.binarize = defaults;
        app.view.binarize_auto = true;
    }
    ui.checkbox(&mut app.view.binarize_for_scan, "Read the photo as the binarised view shows it").on_hover_text(
        "Find in outline, Other placements, Search whole photo and the reads of what they find look for finder patterns \
         against this threshold, and nothing else, in place of the three readings they pool: the local threshold, one \
         threshold for the outline and the windows. These settings stay on show while this is ticked. What is scanned \
         sets a reading's level and floor: for an outline, the outline and its margin, as the view has it, and for the \
         whole photo, the whole of it. A copy of the photo made for a look, shrunk or enlarged, has the reach scaled \
         with it.",
    );
}

fn alignment_section(ui: &mut egui::Ui, app: &mut App, actions: &mut Vec<Action>) {
    ui.heading("Alignment");
    ui.weak(
        "Drag TL, TR, BR and BL loosely around the symbol and press Find in outline, or put them on its outer \
         corners by hand. Arrow keys nudge the last corner touched; hold H to see the bare photo.",
    );
    ui.weak(
        "Anchors pin the fixed patterns to the photo, and the grid bends through them. Drag a hollow handle onto \
         its pattern to set one, or drag any other module of the grid to set a custom anchor on its middle; \
         right-click a handle to remove it.",
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
        // Custom anchors stand on no site, and are counted apart.
        let custom = app.doc.anchors.iter().filter(|a| a.anchor.kind == AnchorKind::Custom).count();
        let sited = app.doc.anchors.len() - custom;
        let detected = app.doc.anchors.iter().filter(|a| a.source == AnchorSource::Detected).count();
        let manual = sited - detected;
        let partial = app.doc.anchors.iter().filter(|a| a.anchor.partial).count();
        let mut text = format!("Anchors: {sited} of {sites} sites set ({detected} found, {manual} by hand)");
        if partial > 0 {
            text.push_str(&format!(", {partial} partly damaged"));
        }
        if custom > 0 {
            text.push_str(&format!(", and {custom} custom"));
        }
        ui.label(text).on_hover_text(
            "A site is a finder pattern, an alignment pattern, the rMQR sub-pattern or a module of a timing pattern; a \
             custom anchor is one set by hand on any other module",
        );
        ui.horizontal(|ui| {
            ui.checkbox(&mut app.view.use_anchors, "Bend the grid through the anchors")
                .on_hover_text("Off: one flat perspective through the four corners, as if the symbol were on a plane");
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut app.view.show_anchors, "Show anchor handles").on_hover_text(
                "A handle on every fixed pattern and three along each edge. Drag one to where its pattern or edge \
                 really lies and the grid bends through it. Edge handles suit a symbol on a bottle or curled paper, \
                 whose sides bow. Drag any other module of the grid to set a custom anchor on its middle and bend \
                 the grid by it where it has no pattern.",
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
                 to be roughly around the symbol, and may be a quarter turn out. Also picks the version that fits. \
                 The symbol is read through to its message on the way, and a read that decodes is written into the \
                 grid while Load reads that decode is ticked.",
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
        let label = if running == Some(JobKind::Placements) { "Ranking…" } else { "Other placements" };
        if ui
            .add_enabled(!busy, egui::Button::new(label))
            .on_hover_text(
                "List every placement of the symbol inside the outline that the detector ranks, best first: the one Find \
                 in outline starts from, and the runners-up it falls back on, another version or another reading of the \
                 finder patterns. Pick one to put the corners and anchors on it. Looks for the family chosen under Symbol.",
            )
            .on_disabled_hover_text(BUSY_HINT)
            .clicked()
        {
            actions.push(Action::ListPlacements);
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
                 Find in outline, which looks at the photo itself. While Load reads that decode is ticked, a symbol \
                 that decodes as found is read through and written into the grid.",
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
    ui.checkbox(&mut app.view.load_reads, "Load reads that decode").on_hover_text(
        "When a symbol is read through to its message, write the grid as it was read, with its fixed patterns filled in \
         and the pins it was read under: a new grid of the symbol's version, in place of the one there is (undoable). \
         Find in outline loads what it reads inside the outline. Search whole photo and Read all load one of the \
         symbols found: the one the outline is on, when it is on one and that decodes, and otherwise the first that \
         does. A symbol or a placement picked from a list is read through and loaded the same way. Off, only the \
         outline and the anchors are placed, and the grid is left to be sampled. A read that does not decode never \
         replaces the grid.",
    );
    ui.checkbox(&mut app.view.search.coarse_to_fine, "Search a small copy first").on_hover_text(
        "Search a copy of the photo shrunk to a thousand pixels first, and the photo at the full size a search goes \
         to, 4096 pixels on its longer side, in full only if that shows nothing, and otherwise only for the finder \
         patterns its regions show beside the symbols found. Much faster on a large photo. Turn off for a symbol \
         only a few pixels a module, which the copy may lose.",
    );
    if app.view.binarize_for_scan {
        let (width, height) = app.photo.as_ref().map_or((0, 0), |photo| (photo.width, photo.height));
        let binarizer = app.view.binarize_for(app.doc.corners, width, height).binarizer();
        ui.weak(format!("The photo is read against {binarizer}, as the binarised view shows it.")).on_hover_text(
            "Find in outline, Other placements, Search whole photo and the reads of what they find look for finder \
             patterns against that threshold and nothing else. It is set under Photo, with the binarised view's \
             settings: untick Read the photo as the binarised view shows it there to pool the search's three readings \
             again.",
        );
    }
    app.view.hover_placement = None;
    if let Some(placements) = &app.placements {
        let in_use = super::placement_in_use(placements, &app.doc.corners);
        ui.label(format!("Placements inside the outline ({}):", placements.found.len()));
        for (i, found) in placements.found.iter().enumerate() {
            let placed = in_use == Some(i);
            let reads = if app.view.load_reads {
                " It is read through on the way, and a read that decodes is written into the grid (undoable)."
            } else {
                ""
            };
            let hint = format!(
                "Place the corners and anchors on the symbol as this placement has it: {} anchors, {} of {} fixed \
                 modules reading right. It is outlined on the photo while the pointer is here.{reads}{}",
                found.anchors.len(),
                found.agreement.0,
                found.agreement.1,
                super::new_grid_note(found, app.doc.version)
            );
            let response = detection_row(ui, i, found, None, placed, busy, hint);
            if response.hovered() {
                app.view.hover_placement = Some(i);
            }
            if response.clicked() {
                actions.push(Action::UsePlacement(i));
            }
        }
        if ui.small_button("Clear the list").clicked() {
            actions.push(Action::ClearPlacements);
        }
    }
    if let Some(proposals) = &app.proposals {
        let in_use = super::proposal_in_use(proposals, &app.doc.corners);
        ui.label(format!("Found in the photo ({}):", proposals.found.len()));
        for (i, found) in proposals.found.iter().enumerate() {
            let placed = in_use == Some(i);
            // Once Read all has run, how the symbol's read stands, and
            // whether picking it loads that read.
            let read = proposals.reads.as_ref().and_then(|reads| reads.results.get(i));
            let standing = read.map(|read| read.as_ref().map_or_else(|| "could not be sampled".to_owned(), crate::cli::standing));
            let loads = app.view.load_reads && read.and_then(Option::as_ref).is_some_and(|read| read.is_trustworthy());
            // Not read yet, it is read when it is picked, where reads
            // are loaded.
            let reads = app.view.load_reads && proposals.reads.is_none();
            let hint = match (loads, placed) {
                _ if reads => format!(
                    "Place the corners and anchors on this symbol and read it through to its message: a read that decodes \
                     is written into the grid, with the pins it was read under (undoable). Its outline on the photo can be \
                     clicked too.{}",
                    super::new_grid_note(found, app.doc.version)
                ),
                (true, true) => "The outline is on this symbol. Click to write the grid again as it was read, with the pins \
                                 it was read under (undoable)."
                    .to_owned(),
                (true, false) => "Put the outline and anchors on this symbol and write the grid as it was read, with the \
                                  pins it was read under: a new grid of the symbol's version (undoable). Its outline on \
                                  the photo can be clicked too."
                    .to_owned(),
                (false, true) => {
                    "The outline is on this symbol. Click to place the corners and anchors on it again, as found.".to_owned()
                }
                (false, false) => format!(
                    "Place the corners and anchors on this symbol; its outline on the photo can be clicked too.{}",
                    super::new_grid_note(found, app.doc.version)
                ),
            };
            if detection_row(ui, i, found, standing, placed, busy, hint).clicked() {
                actions.push(Action::UseProposal(i));
            }
        }
        ui.horizontal(|ui| {
            let label = if running == Some(JobKind::ReadFound) { "Reading…" } else { "Read all" };
            if ui
                .add_enabled(!busy, egui::Button::new(label))
                .on_hover_text(
                    "Read every symbol found through to its message, as the command line reads a photo. Each symbol \
                     in the list then says how its read stands, and the parts of a structured append message go into \
                     the sequence. While Load reads that decode is ticked, one that decoded is written into the grid, \
                     and picking another writes that one.",
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
    if !app.leads.is_empty() {
        ui.label(format!("Finder patterns with no symbol fitted ({}):", app.leads.len()));
        for (i, lead) in app.leads.iter().enumerate() {
            let text = format!("{}. {}, {} finder patterns", i + 1, lead.family.label(), lead.finders.len());
            if ui
                .add_enabled(!busy, egui::Button::new(text))
                .on_hover_text(
                    "The search found these finder patterns standing as one symbol's, with a format word reading outright \
                     beside them, and could fit no symbol to them. Put the outline about them, as the search proposed it; \
                     set it right by hand if it is not, and press Find in outline, which tries more than a search of the \
                     whole photo does.",
                )
                .on_disabled_hover_text(BUSY_HINT)
                .clicked()
            {
                actions.push(Action::UseLead(i));
            }
        }
        if app.proposals.is_none() && ui.small_button("Clear the list").clicked() {
            actions.push(Action::ClearProposals);
        }
    }
    if let Some(guesses) = &app.guesses {
        let shown = if app.view.all_guesses { guesses.len() } else { GUESSES_SHOWN };
        for guess in guesses.iter().take(shown) {
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
        let hint = "Every version of the family, as ranked on these corners";
        ui.horizontal(|ui| {
            super::show_all_button(ui, &mut app.view.all_guesses, guesses.len(), GUESSES_SHOWN, hint);
            if ui.small_button("Clear the list").clicked() {
                actions.push(Action::ClearGuesses);
            }
        });
    }
}

/// A symbol the detector placed, as a row of a list to pick it from: its
/// place in the list, what it is, how well it fits, and how a read of it
/// stands, as `standing` has it, once one has been made. `placed` marks
/// the one the outline is on, and `hint` says what picking it does.
fn detection_row(
    ui: &mut egui::Ui,
    i: usize,
    found: &Detection,
    standing: Option<String>,
    placed: bool,
    busy: bool,
    hint: String,
) -> egui::Response {
    let decodes = if found.decodes == Some(true) { ", decodes" } else { "" };
    let confirmed = if found.version_confirmed() { ", version confirmed" } else { "" };
    let polarity = if found.inverted { ", light on dark" } else { "" };
    let mirrored = if found.mirrored { ", mirrored" } else { "" };
    let retry = found.retry.map_or(String::new(), |r| format!(", with {r}"));
    let standing = standing.map_or(String::new(), |standing| format!("  ·  {standing}"));
    let text = format!(
        "{}. {} {}  —  {:.0}% fixed{decodes}{confirmed}{polarity}{mirrored}{retry}{standing}",
        i + 1,
        found.version.family().label(),
        found.version,
        100.0 * found.score()
    );
    ui.add_enabled(!busy, egui::Button::selectable(placed, text)).on_hover_text(hint).on_disabled_hover_text(BUSY_HINT)
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
    ui.checkbox(&mut params.sharpen_modules, "Sharpen small or blurred modules").on_hover_text(format!(
        "Once the modules are read, set each against its four neighbours and put back some of what blur took from it: \
         where a module is under {SHARPEN_BELOW_PX} pixels across, the more the smaller, and wherever the luminances \
         read do not separate into dark and light. The line below says when it has been done. Off, the modules are \
         taken as they were read, which suits a clean print with specks in it, each of which sharpening sets off.",
    ));
    if let Some(preview) = &app.preview {
        let sharpening = preview.sharpening;
        if sharpening.any() {
            let mut why = Vec::new();
            if sharpening.for_size > 0.0 {
                why.push(format!("they are {:.1} px across (by {:.2})", sharpening.module_px, sharpening.for_size));
            }
            if sharpening.for_separation > 0.0 {
                why.push(format!("they separated poorly (by {:.2})", sharpening.for_separation));
            }
            ui.weak(format!("The modules were sharpened against their neighbours: {}.", why.join(", and again because ")))
                .on_hover_text("The share of the difference between each module and the mean of its neighbours that was added back");
        }
        let unknown = preview.classified.modules.iter().filter(|&&m| m == Module::Unknown).count();
        let flat = preview.classified.flat;
        let mut text = format!("{unknown} of {} modules would be left unknown", preview.classified.modules.len());
        if flat > 0 {
            text.push_str(&format!(", {flat} of them in patches with no contrast"));
        }
        ui.weak(text);
        // One line, which turns to a warning where the two do not split.
        let separates = format!("Dark and light separate {:.2}", preview.classified.separability);
        let line = if preview.classified.is_poorly_separated() {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                format!("{separates}, too little to tell them apart: the outline may be off the symbol, or the photo too washed out to read."),
            )
        } else {
            ui.weak(separates)
        };
        line.on_hover_text(format!(
            "How clearly the module luminances fall into two groups, from 0 to 1: how empty the valley is between the \
             dark modules and the light. A well placed outline on a readable symbol scores near 1, and under \
             {POOR_SEPARABILITY} the two groups are not to be believed: a grid that straddles module boundaries averages \
             dark with light into greys. Watch it while moving a corner or a setting: it rises as the grid comes onto \
             the modules.",
        ));
    }
    ui.checkbox(&mut app.view.auto_invert, "Invert photo automatically when sampling")
        .on_hover_text(
            "Sampling first turns Invert photo on, or off, when the fixed patterns read inverted. Off, the photo \
             is sampled as it is and a notice says so.",
        );
    ui.horizontal(|ui| {
        if ui.button("Sample all").on_hover_text("Replace the grid with what the photo shows").clicked() {
            actions.push(Action::Sample { only_unknown: false, scope: SampleScope::All });
        }
        if ui.button("Sample unknown only").on_hover_text("Keep everything already painted").clicked() {
            actions.push(Action::Sample { only_unknown: true, scope: SampleScope::All });
        }
    });
    if let (Some(all), Some(unknown)) = (app.sample_would_change(false), app.sample_would_change(true)) {
        ui.weak(format!("Sample all would change {all} modules of the grid as it stands, and unknown only {unknown}."))
            .on_hover_text(
                "The Differs from photo overlay of the Edit stage shows which. To sample a part of the grid alone, paint \
                 it with the From photo ink under Paint, or press Sample on a block under Blocks.",
            );
    }
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
        ui.selectable_value(&mut app.view.ink, Ink::Module(Module::Dark), "Dark").on_hover_text("1");
        ui.selectable_value(&mut app.view.ink, Ink::Module(Module::Light), "Light").on_hover_text("2");
        ui.selectable_value(&mut app.view.ink, Ink::Module(Module::Unknown), "Unknown").on_hover_text("3");
        ui.add_enabled_ui(app.preview.is_some(), |ui| {
            ui.selectable_value(&mut app.view.ink, Ink::Photo, "From photo")
                .on_hover_text(
                    "4 — each module as sampling reads it off the photo, as Sample all reads the whole grid, but with no \
                     unsure band: a module near its threshold is painted the colour it falls on, at the reliability it \
                     read at, where Sample all would leave it unknown. Brush over a part of the grid, or drag a \
                     rectangle about it, to sample that part alone",
                )
                .on_disabled_hover_text("Load a photo and place the corners to paint modules from it");
        });
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
    let mut opts = app.opts;
    ui.checkbox(&mut opts.bit_level, "Solve unknown bits individually").on_hover_text(
        "When a block has more unknown codewords than Reed-Solomon can restore, solve for the unknown \
         modules one by one instead. Assumes every painted module is right.",
    );
    ui.checkbox(&mut opts.reliability, "Doubt the least reliable modules").on_hover_text(
        "When a block fails, treat its codewords with the faintest sampled modules as unknown, a few at a \
         time, until it solves with something to spare. Modules painted by hand are never doubted. The three \
         ways of doubting below are each tried, and the solution with the most to confirm it kept; Blocks says \
         which solved a block, and its Show outlines the modules that were doubted.",
    );
    ui.indent("doubting", |ui| {
        ui.add_enabled_ui(opts.reliability, |ui| {
            ui.checkbox(&mut opts.doubt_codewords, "Whole codewords").on_hover_text(
                "Take the least reliable codewords as unknown, one more at a time: each costs one error correction \
                 codeword, where a wrong one that is not doubted costs two.",
            );
            ui.add_enabled(opts.bit_level, egui::Checkbox::new(&mut opts.doubt_bits, "Single bits"))
                .on_hover_text(
                    "Take the least reliable modules as unknown one by one and solve bit by bit: reaches damage spread \
                     a faint module to a codeword over more codewords than doubting them whole can take.",
                )
                .on_disabled_hover_text("Needs Solve unknown bits individually");
            ui.checkbox(&mut opts.flip_bits, "Flip the faintest bits").on_hover_text(
                "Flip the few least reliable modules every way and decode each: reaches a block wrong in a little more \
                 than its error correction takes. Many tries are many chances of a wrong answer, so every pattern that \
                 solves must agree; turn this off where a solution it reached looks wrong.",
            );
        });
    });
    if opts != app.opts {
        actions.push(Action::SetDecodeOptions(opts));
    }
    if opts.reliability && !app.doc.grid.has_reliability() {
        ui.weak("Takes effect once modules are sampled from a photo.");
    }
}
