//! Left panel: which symbol this is, the photo, and how to get modules in.

use eframe::egui;
use qr_forensics::decode::abp::{ABP_ROUNDS_LEAST, ABP_ROUNDS_MAX, round_tries};
use qr_forensics::decode::list::{LIST_COST, LIST_COST_MAX, LIST_COST_MIN};
use qr_forensics::decode::pipeline::DecodeOptions;
use qr_forensics::decode::rs::{CHASE_MAX_BITS, TRIAL_SETS_MAX, TRIAL_SETS_MIN, tries_charge};
use qr_forensics::imaging::adjust::{Adjust, Backdrop, Channel, Enlarge, Filter, K1_RANGE, LENS_MIN_PAIRS, MAX_ENLARGED_SIDE, STRENGTH_RANGE};
use qr_forensics::imaging::deblur::{self, Spread};
use qr_forensics::imaging::binarize::{BinarizeParams, WINDOW_RADII};
use qr_forensics::imaging::detect::{AnchorKind, Detection, anchor_sites};
use qr_forensics::imaging::enhance::SHARPEN_RANGE;
use qr_forensics::imaging::sample::{
    BLUR_MIN_GAIN, DATA_GROSS, DATA_SPREAD, Kernel, MISREAD_FLOOR, POOR_SEPARABILITY, SAMPLED_MAX, SHARPEN_BELOW_PX, misread_odds,
    reliability_bits,
};
use qr_forensics::symbol::function_map::looks_inverted;
use qr_forensics::symbol::grid::{CERTAIN, Module};
use qr_forensics::symbol::{EcLevel, Family, SymbolVersion};

use super::canvas::one_in;
use super::state::*;
use super::{Action, App, JobKind, Reach, Transform, count_word, grouped, list_phrase, wrong_codewords};

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
    ui.separator();
    decoding_section(ui, app, actions);
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

/// The blur the Deblur filter undoes, before its strength: the kernel
/// the deconvolution divides by, drawn tap by tap, the heavier the
/// lighter, and how wide it is, which way, and where it was measured:
/// over the whole photo, or, `fixed`, set by Measure in outline or by a
/// read, which may have fixed the whole photo's blur as it then was.
fn kernel_view(ui: &mut egui::Ui, spread: Option<Spread>, fixed: bool) {
    let Some(spread) = spread else {
        ui.weak("No blur found: the photo is sharp or has too few edges to measure, so it's left as is.");
        return;
    };
    ui.horizontal_top(|ui| {
        let taps = deblur::kernel(&spread);
        let reach = taps.iter().map(|&(dx, dy, _)| dx.abs().max(dy.abs())).max().unwrap_or(0);
        let heaviest = taps.iter().map(|tap| tap.2).fold(f32::MIN_POSITIVE, f32::max);
        let side = 64.0;
        let (rect, response) = ui.allocate_exact_size(egui::vec2(side, side), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, egui::Color32::BLACK);
        let cell = side / (2 * reach + 1) as f32;
        for &(dx, dy, weight) in &taps {
            let at = rect.min + egui::vec2((dx + reach) as f32 * cell, (dy + reach) as f32 * cell);
            let grey = (255.0 * (weight / heaviest).sqrt()).round() as u8;
            painter.rect_filled(egui::Rect::from_min_size(at, egui::vec2(cell, cell)), 0.0, egui::Color32::from_gray(grey));
        }
        let (widest, narrowest) = spread.extremes();
        response.on_hover_text(format!(
            "The blur kernel being undone, before Strength: a Gaussian, σ {widest:.2} px on its widest axis and \
             {narrowest:.2} px on its narrowest, with {} taps out to 3σ. Brighter cells weigh more.",
            taps.len()
        ));
        let from = if fixed { "set by Measure in outline or a read" } else { "measured over the whole photo" };
        ui.weak(format!("σ {widest:.1} × {narrowest:.1} px at {:.0}°,\n{from}", spread.angle()));
    });
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
    let deblurred = photo.luma.deblurred;
    // What the photo starts with, which Reset goes back to: no
    // adjustment but the enlargement a small photo is given, its
    // transparent parts staying on the backdrop they are on.
    let fresh = Adjust { backdrop: app.doc.adjust.backdrop, ..Adjust::for_photo(photo.original.width(), photo.original.height()) };
    let transparent = photo.unflattened.is_some();
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
    // The adjustments in the order the photo is put through them
    // ([`Adjust::apply`]): what its transparent parts are filled with, its
    // size and shape, its sharpness, the one curve of its tone, and what
    // is read of it.
    egui::Grid::new("adjust").num_columns(2).show(ui, |ui| {
        if transparent {
            ui.label("Transparent parts");
            let mut changed = false;
            ui.horizontal(|ui| {
                for backdrop in Backdrop::ALL {
                    changed |= ui.selectable_value(&mut adjust.backdrop, backdrop, backdrop.label()).changed();
                }
            })
            .response
            .on_hover_text(
                "What transparent pixels are filled with before the photo is read: grey unless you choose another. \
                 A light symbol on a transparent background (made for dark pages) vanishes on white but reads on \
                 black; grey works for either.",
            );
            note(changed);
            ui.end_row();
        }
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
            let slider = egui::Slider::new(&mut adjust.k1, K1_RANGE)
                .step_by(0.005)
                .fixed_decimals(3)
                .clamping(egui::SliderClamping::Edits);
            // A lens bends the photo about its centre, which a crop has
            // lost: the term is set on the whole photo, and a part cut
            // under it is straightened as the whole is there.
            const CROPPED_HINT: &str = "A lens bends the photo about its centre, which a cropped photo has lost: show the \
                                        whole photo to set it, and crop again";
            let lens = ui
                .add_enabled(!cropped, slider)
                .on_hover_text(
                    "Corrects radial lens distortion around the photo's centre. Negative straightens edges that bow \
                     outwards (barrel), positive straightens edges that bow inwards (pincushion). Applied after \
                     Enlarge.",
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
        ui.label("Sharpen");
        let sharpen = ui.add(egui::Slider::new(&mut adjust.sharpen, SHARPEN_RANGE)).on_hover_text(
            "An unsharp mask, applied in colour after Enlarge and Lens, for a symbol out of focus or modules a pixel \
             or two across that have blurred together. It's 1 px wide at 0.5, 2 px at 1 and 4 px at 2, in the \
             photo's original pixels however far it's enlarged (or in the reduced pixels, if it's reduced). Below 0 \
             it blurs instead, to smooth out grain, halftone dots or in-camera sharpening halos: the blur fades in \
             down to -0.5 (1 px), then widens to 2 px at -1 and 4 px at -2. When a read fails, Find in outline \
             tries sharpened and enlarged copies, and sets this and Enlarge to whatever decoded.",
        );
        note(sharpen.changed());
        ui.end_row();
        ui.label("");
        note(ui.checkbox(&mut adjust.invert, "Invert photo").on_hover_text("For light symbols on a dark background").changed());
        ui.end_row();
        ui.label("Contrast");
        note(ui.add(egui::Slider::new(&mut adjust.contrast, -1.0..=1.0)).changed());
        ui.end_row();
        ui.label("Brightness");
        note(ui.add(egui::Slider::new(&mut adjust.brightness, -1.0..=1.0)).changed());
        ui.end_row();
        ui.label("Channel");
        let mut changed = false;
        egui::ComboBox::from_id_salt("channel").selected_text(adjust.channel.label()).show_ui(ui, |ui| {
            for channel in Channel::ALL {
                changed |= ui.selectable_value(&mut adjust.channel, channel, channel.label()).changed();
            }
        });
        note(changed);
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
        if adjust.filter == Filter::Deblur {
            ui.label("Kernel");
            ui.horizontal_wrapped(|ui| {
                let measure = ui.button("Measure in outline").on_hover_text(
                    "Measure the blur inside the outline only (after the adjustments above) and undo that instead of \
                     the whole photo's blur. Use it when only the symbol is out of focus, or when the rest of the photo \
                     throws off the measurement. It's rescaled if Enlarge changes.",
                );
                if measure.clicked() {
                    actions.push(Action::MeasureDeblur);
                }
                let whole = ui
                    .add_enabled(adjust.kernel.is_some(), egui::Button::new("Use whole photo"))
                    .on_hover_text("Go back to undoing the blur measured over the whole photo")
                    .on_disabled_hover_text("Already undoing the whole photo's blur");
                if whole.clicked() {
                    adjust.kernel = None;
                    note(true);
                }
            });
            ui.end_row();
            ui.label("");
            kernel_view(ui, deblurred, adjust.kernel.is_some());
            ui.end_row();
            ui.label("Strength");
            let strength = ui.add(egui::Slider::new(&mut adjust.strength, STRENGTH_RANGE)).on_hover_text(
                "Scales the kernel's width. 1 undoes the blur as measured (the measurement runs slightly low), \
                 higher undoes more but adds ringing at edges, and 0 leaves the photo unchanged.",
            );
            note(strength.changed());
            ui.end_row();
        }
        if adjust.filter.has_radius() {
            ui.label("Radius");
            let radius = ui.add(egui::DragValue::new(&mut adjust.radius).range(1..=Filter::MAX_RADIUS).suffix(" px")).on_hover_text(
                "Half-width of the filter's neighbourhood, in enlarged pixels (the filter runs last). With a filter \
                 on, the photo is shown in the filtered grey that sampling reads.",
            );
            note(radius.changed());
            ui.end_row();
        }
    });
    if ui.add_enabled(adjust != fresh, egui::Button::new("Reset adjustments")).clicked() {
        adjust = fresh;
        note(true);
    }
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
         its pattern to set one, drag the grid's outline to add an edge point, or drag any other module to set a \
         custom anchor on its middle. Right-click a handle to remove it.",
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
        // Edge points, data patches and custom anchors stand on no site,
        // and are counted apart.
        let of_kind = |kind: AnchorKind| app.doc.anchors.iter().filter(|a| a.anchor.kind == kind).count();
        let (edge, data, custom) = (of_kind(AnchorKind::Edge), of_kind(AnchorKind::Data), of_kind(AnchorKind::Custom));
        let on_site = |a: &&DocAnchor| !matches!(a.anchor.kind, AnchorKind::Edge | AnchorKind::Data | AnchorKind::Custom);
        let sited = app.doc.anchors.iter().filter(on_site).count();
        let detected = app.doc.anchors.iter().filter(on_site).filter(|a| a.source == AnchorSource::Detected).count();
        let manual = sited - detected;
        let partial = app.doc.anchors.iter().filter(|a| a.anchor.partial).count();
        let mut text = format!("Anchors: {sited} of {sites} sites set ({detected} found, {manual} by hand)");
        if partial > 0 {
            text.push_str(&format!(", {partial} partly damaged"));
        }
        let apart: Vec<String> = [(edge, "edge point", "edge points"), (data, "data patch", "data patches"), (custom, "custom", "custom")]
            .into_iter()
            .filter(|&(count, ..)| count > 0)
            .map(|(count, one, many)| format!("{count} {}", if count == 1 { one } else { many }))
            .collect();
        if !apart.is_empty() {
            text.push_str(&format!(", and {}", list_phrase(&apart)));
        }
        ui.label(text).on_hover_text(
            "A site is a finder pattern, an alignment pattern, the rMQR sub-pattern or a timing module. Edge points \
             bow the grid's edges to follow the symbol's, data patches are placed by a read where data modules read \
             sharpest or match the decode, and custom anchors are set by hand on any other module",
        );
        ui.horizontal(|ui| {
            ui.checkbox(&mut app.view.use_anchors, "Bend the grid through the anchors")
                .on_hover_text("Off: one flat perspective through the four corners, as if the symbol were on a plane");
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut app.view.show_anchors, "Show anchor handles").on_hover_text(
                "A handle on every fixed pattern and three along each edge. Drag one to where its pattern or edge \
                 really lies and the grid bends through it. Edge handles suit a symbol on a bottle or curled paper, \
                 whose sides bow; if three per edge aren't enough, drag the outline to add more. Drag any other \
                 module of the grid to set a custom anchor on its middle and bend the grid by it where it has no \
                 pattern.",
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
                "Search the whole photo for symbols and list them, those that decode first. Takes a few seconds, \
                 longer for a damaged symbol; it can be stopped. A photo over 4096 px on a side is searched at \
                 reduced size, but symbols are placed and read at full size; for a symbol too small to show at \
                 reduced size, use Find in outline. With Load reads that decode ticked, a symbol that decodes is \
                 read into the grid.",
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
        let read = if app.leads.iter().any(|lead| lead.fitted.is_some()) { "read" } else { "fitted" };
        ui.label(format!("Finder patterns with no symbol {read} ({}):", app.leads.len()));
        for (i, lead) in app.leads.iter().enumerate() {
            let (text, hint) = match lead.fitted {
                Some(version) => (
                    format!("{}. {} {version} fitted, not read", i + 1, lead.family.label()),
                    format!(
                        "The search fitted a symbol of this version here with its fixed patterns nearly right, but nothing \
                         else supported it and it didn't decode. Place the outline on it and press Find in outline, which \
                         tries more than a whole-photo search, or sample it and work on the grid.{}",
                        if version == app.doc.version {
                            String::new()
                        } else {
                            format!(" This starts a new {} {version} grid (undoable).", version.family().label())
                        }
                    ),
                ),
                None => (
                    format!("{}. {}, {} finder patterns", i + 1, lead.family.label(), lead.finders.len()),
                    "The search found these finder patterns arranged like one symbol's, with a clean format word beside \
                     them, but couldn't fit a symbol to them. Place the outline the search proposed, correct it by hand \
                     if needed, and press Find in outline, which tries more than a whole-photo search."
                        .to_owned(),
                ),
            };
            if ui.add_enabled(!busy, egui::Button::new(text)).on_hover_text(hint).on_disabled_hover_text(BUSY_HINT).clicked() {
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
    let retry = found.found_with().map_or(String::new(), |with| format!(", with {with}"));
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
        "Once the modules are read, take back out of each what blur put into it from its neighbours. How far the \
         photo blurs, and which way, is learnt from the fixed patterns, whose colours are known, and undone where \
         that gains enough to be worth it. Where it cannot be learnt, each module is set against its four neighbours \
         by a guess instead: where a module is under {SHARPEN_BELOW_PX} pixels across, the more the smaller, and \
         wherever the luminances read do not separate into dark and light. The lines below say what was done. Off, \
         the modules are taken as they were read, which suits a clean print with specks in it, each of which \
         sharpening sets off.",
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
        if let Some(blur) = preview.classified.blur {
            let what = if preview.classified.equalised {
                "and undone"
            } else if blur.is_worth_undoing() {
                "and left in, with sharpening off"
            } else {
                "and is too slight to be worth undoing"
            };
            ui.weak(format!(
                "The blur was learnt from {} fixed modules {what}: their neighbours put {:.0}% as much into each \
                 module's reading as its own colour does",
                blur.fitted,
                100.0 * blur.spill()
            ))
            .on_hover_text(format!(
                "Each module's own colour moves its reading by {:.2} of half the contrast, and each neighbour by {:.2} \
                 beside it in its row, {:.2} above and below it, and {:.2} and {:.2} at its corners down and up the \
                 diagonals: the more one way than another, the more the camera or the paper moved that way. Put through \
                 the filter that best takes that back out without raising the noise past what it gains, a module's own \
                 colour stands {:.1} times as clear in its reading, and the reading is so filtered when that is {:.1} \
                 times or more and the modules are to be sharpened.",
                blur.own,
                blur.across,
                blur.down,
                blur.falling,
                blur.rising,
                blur.gain,
                1.0 + BLUR_MIN_GAIN
            ));
        }
        if let Some(calibration) = preview.classified.calibration {
            ui.weak(format!(
                "Reliability calibrated on {} fixed modules: noise {:.2} of the contrast on the dark side and {:.2} on \
                 the light, {} of them read wrong",
                calibration.labelled, calibration.own_sigma[0], calibration.own_sigma[1], calibration.misread
            ))
            .on_hover_text(format!(
                "Reliability is calibrated on the fixed patterns, whose colours are known. Correctly read modules sit \
                 about {:.2} (dark) and {:.2} (light) of half the contrast from the threshold, spread by {:.2} and \
                 {:.2}; {:.1}% are wrong by more than that spread explains, however far from the threshold they read. \
                 A logo, a stain or a grid slipping on curled paper barely touches the fixed patterns, so for the data \
                 the spread is widened {:.0}%, to {:.2} and {:.2}, and the share of such misreads is multiplied by {}, \
                 to {:.1}%. Every reading is assumed wrong at least {} of the time, so none is trusted past {:.0} bits. \
                 When one colour spreads wider, a borderline reading is more likely that colour, so the threshold moves \
                 toward the other: here by {:+.2}. The decoder doubts the least reliable modules first, and weighs each \
                 solution by the reliability of what it overturns.",
                -calibration.dark.mean,
                calibration.light.mean,
                calibration.own_sigma[0],
                calibration.own_sigma[1],
                100.0 * calibration.grossly_misread as f32 / calibration.labelled.max(1) as f32,
                100.0 * (DATA_SPREAD - 1.0),
                calibration.dark.sigma,
                calibration.light.sigma,
                DATA_GROSS,
                100.0 * calibration.gross,
                one_in(1.0 / f64::from(MISREAD_FLOOR)),
                reliability_bits(SAMPLED_MAX),
                calibration.threshold
            ));
        }
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
    ui.horizontal_wrapped(|ui| {
        ui.selectable_value(&mut app.view.ink, Ink::Module(Module::Dark), "Dark").on_hover_text("1");
        ui.selectable_value(&mut app.view.ink, Ink::Module(Module::Light), "Light").on_hover_text("2");
        ui.selectable_value(&mut app.view.ink, Ink::Module(Module::Unknown), "Unknown").on_hover_text("3");
        ui.add_enabled_ui(app.preview.is_some(), |ui| {
            ui.selectable_value(&mut app.view.ink, Ink::Photo, "From photo")
                .on_hover_text(
                    "4 — each module as sampling reads it off the photo, as Sample all reads the whole grid, but with no \
                     unsure band: a module near its threshold is painted the colour it falls on, at the reliability it \
                     read at, where Sample all would leave it unknown. Brush over a part of the grid, or drag a \
                     rectangle about it, to sample that part alone. With the painted reliability below certain, that \
                     reliability is used instead",
                )
                .on_disabled_hover_text("Load a photo and place the corners to paint modules from it");
        });
        let doubtful = ui.selectable_label(app.view.ink == Ink::Doubt, "Doubtful").on_hover_text(
            "5 — keep each module's colour but set its reliability to the painted reliability below. For stains, \
             tears and folds: modules you can read but don't fully trust, which the decoder should doubt first when \
             a block fails. The right button makes modules certain again. Unknown modules are left alone.",
        );
        if doubtful.clicked() {
            app.view.ink = Ink::Doubt;
            if app.view.paint_reliability == CERTAIN {
                app.view.paint_reliability = DOUBT_RELIABILITY;
            }
        }
    });
    ui.weak(if app.view.ink == Ink::Doubt {
        "Left button sets the reliability; right button makes modules certain; Shift paints unknown."
    } else {
        "Left button paints the ink; right button paints the other of dark and light; Shift paints unknown."
    });
    // The slider runs from a guess to eight bits of likelihood, and one
    // step beyond that to certain.
    const CERTAIN_STEP: u8 = PAINT_RELIABILITY_MAX + 1;
    let mut step = if app.view.paint_reliability == CERTAIN { CERTAIN_STEP } else { app.view.paint_reliability.min(PAINT_RELIABILITY_MAX) };
    let painted = egui::Slider::new(&mut step, 0..=CERTAIN_STEP)
        .text("painted reliability")
        .custom_formatter(|value, _| {
            let step = value.round() as u8;
            if step >= CERTAIN_STEP {
                "certain".to_owned()
            } else if step == 0 {
                "a guess (1/2 misread)".to_owned()
            } else {
                format!("{:.1} bits ({} misread)", reliability_bits(step), one_in(misread_odds(step)))
            }
        })
        .custom_parser(|text| match text.trim() {
            "certain" => Some(f64::from(CERTAIN_STEP)),
            other => other
                .split(|c: char| !c.is_ascii_digit() && c != '.')
                .next()
                .and_then(|n| n.parse::<f64>().ok())
                .map(|bits| ((bits - 1.0) * 8.0).round().clamp(0.0, f64::from(PAINT_RELIABILITY_MAX))),
        });
    let response = ui.add(painted).on_hover_text(
        "Reliability of modules painted dark or light: from a guess (misread 1 in 2, like a module sampled right on \
         its threshold) up to 8 bits (1 in 256), then certain. Certain modules are never doubted. Below certain, a \
         painted module counts the same as one sampled at that reliability, so Doubt the least reliable modules, \
         under Decoding, may erase or flip it when a block fails; guesses go first. Paint over a module to change \
         its reliability. From photo also uses this reliability while it is below certain, and Doubtful applies it \
         without changing colours. Unknown is unaffected.",
    );
    if response.changed() {
        app.view.paint_reliability = if step >= CERTAIN_STEP { CERTAIN } else { step };
    }
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
}

/// A logarithmic slider over a count of tries, shown and typed with its
/// thousands set apart.
fn count_slider<'a>(value: &'a mut usize, range: std::ops::RangeInclusive<usize>, text: &str) -> egui::Slider<'a> {
    egui::Slider::new(value, range)
        .logarithmic(true)
        .text(text)
        .custom_formatter(|n, _| grouped(n.round() as usize))
        .custom_parser(|s| s.trim().replace(',', "").parse::<f64>().ok())
}

/// What a solver's tries cost `what` it finds, said below its slider:
/// each doubling of them is a bit more it needs to be confirmed
/// ([`tries_charge`]).
fn confirmation_cost(what: &str, tries: usize) -> String {
    match tries_charge(tries) {
        0 => format!("{what} needs no extra confirmation."),
        1 => format!("{what} needs up to 1 extra bit of confirmation."),
        bits => format!("{what} needs up to {bits} extra bits of confirmation."),
    }
}

/// The switch of a solver that is tried only while what it `needs` is
/// on: as set while it is, and otherwise off and greyed, with what it
/// needs said first in its hover, so that it shows what is tried. What it
/// was set to is kept for when it can be tried again. Whether it is.
fn needing(ui: &mut egui::Ui, value: &mut bool, needs: Option<&str>, text: &str, hover: &str) -> bool {
    match needs {
        None => {
            ui.checkbox(value, text).on_hover_text(hover);
            *value
        }
        Some(needs) => {
            ui.add_enabled(false, egui::Checkbox::new(&mut false, text)).on_disabled_hover_text(format!("{needs}\n\n{hover}"));
            false
        }
    }
}

/// What the list decoder's budget costs, said below its slider: time
/// alone, which grows with the square of the constraints, against the
/// default's, for the `used` of the `cost` set that this symbol's
/// largest block uses ([`Reach::list_used`]).
fn list_cost_time(cost: usize, used: usize) -> String {
    let ratio = (used as f64 / LIST_COST as f64).powi(2);
    let time = if ratio >= 1.5 {
        format!("Costs only time, not confirmation: about {:.0}× the default.", ratio)
    } else if ratio <= 1.0 / 1.5 {
        format!("Costs only time, not confirmation: about 1/{:.0} of the default.", 1.0 / ratio)
    } else {
        "Costs only time, not confirmation: about the same as the default.".to_owned()
    };
    if used < cost {
        format!("This symbol's blocks use at most {}; more won't reach further. {time}", grouped(used))
    } else {
        time
    }
}

/// How far list decoding reaches in this symbol's blocks at `cost`
/// constraints, said under its switch and in its hover ([`Reach::list`]).
fn list_note(reach: &Reach, version: SymbolVersion, level: EcLevel, cost: usize) -> String {
    let (blocks, standard) = (reach.blocks_phrase(), reach.standard);
    // What plain decoding reaches as far as list decoding does: past the
    // standard's limit, half the distance, spending the reserved
    // codewords as list decoding would.
    let plain = |list: usize| if list <= standard { standard } else { reach.half };
    if !reach.list_gains() && !reach.list_default_gains() {
        let level = match version {
            SymbolVersion::Micro(1) => String::new(),
            _ => format!(" at level {}", level.letter()),
        };
        let plain = plain(reach.list.max(reach.list_default));
        let mut text = format!(
            "No gain{level}: in this symbol's {blocks}, it's guaranteed to correct {}, the same as plain Reed-Solomon.",
            wrong_codewords(plain)
        );
        if plain > standard {
            text.push_str(&format!(" Past {standard}, both use the reserved error-detection codewords, so neither is confirmed."));
        }
        return text;
    }
    let mut text = if reach.list_gains() {
        format!("In this symbol's {blocks}: corrects at least {}, vs. {standard} for plain Reed-Solomon.", wrong_codewords(reach.list))
    } else {
        format!("In this symbol's {blocks}: at this budget, no gain over plain Reed-Solomon's {}.", wrong_codewords(plain(reach.list)))
    };
    if reach.list_default > reach.list {
        let which = if cost > LIST_COST { "Fewer" } else { "More" };
        text.push_str(&format!(
            " {which} constraints reach further here: the default, {}, corrects at least {}.",
            grouped(LIST_COST),
            reach.list_default
        ));
    }
    if reach.reserved > 0 && reach.list.max(reach.list_default) > standard {
        text.push_str(&format!(
            " With every module certain, anything past {standard} uses the reserved error-detection codewords and is \
             never confirmed."
        ));
    }
    text
}

/// A size of set the trials of erasures try, as their note names it.
fn every_set(size: usize) -> String {
    match size {
        1 => "every single codeword".to_owned(),
        2 => "every pair".to_owned(),
        3 => "every triple".to_owned(),
        n => format!("every set of {}", count_word(n)),
    }
}

/// Phrases joined as a list is said: "a", "a and b", "a, b and c".
fn joined(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [one] => one.clone(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

/// What the trials of erasures cover in this symbol's blocks at the
/// budget set, and how far past the code that reaches within the
/// standard's limit and past what plain decoding reaches, said below its
/// slider, and what the next size of set would take ([`Reach::trials`],
/// [`Reach::trials_within`]).
fn trial_note(reach: &Reach) -> String {
    let trials = reach.trials;
    let (blocks, standard) = (reach.blocks_phrase(), reach.standard);
    let next = every_set(trials.first_size + 2 * trials.past);
    let takes = |all: usize| {
        if all <= TRIAL_SETS_MAX { format!("a budget of {}", grouped(all)) } else { format!("more than the maximum, {}", grouped(TRIAL_SETS_MAX)) }
    };
    let mut text = match (trials.past, trials.further) {
        (0, None) => return format!("Nothing to try in this symbol's {blocks}: too little error correction."),
        (0, Some(all)) => return format!("In this symbol's {blocks}: the budget is too small for {next}, which would need {}.", takes(all)),
        (past, _) => {
            let sizes: Vec<String> = (0..past).map(|i| every_set(trials.first_size + 2 * i)).collect();
            let tried = format!("In this symbol's {blocks}: tries {} ({} sets)", joined(&sizes), grouped(trials.sets));
            // Where the standard keeps any codewords, what the sets reach
            // past it spends them, as the standard counts a solution's
            // errors; and where it keeps none, in a small block what the
            // sets reach may leave too little redundancy to confirm it,
            // even as the block itself judges it, uncharged for the sets.
            // It was once said to fall short for the sets' cost alone,
            // where the block's own count confirmed it.
            match reach.trials_within {
                0 if reach.reserved > 0 => format!(
                    "{tried}, but these only beat plain Reed-Solomon by using the reserved error-detection codewords, \
                     so they are never confirmed."
                ),
                0 => format!("{tried}, but anything they find past plain Reed-Solomon leaves too little redundancy to confirm it."),
                within => format!("{tried}, correcting up to {}, vs. {standard} for plain Reed-Solomon.", wrong_codewords(standard + within)),
            }
        }
    };
    if let Some(all) = trials.further {
        text.push_str(&format!(" Adding {next} would need {}.", takes(all)));
    }
    text
}

fn decoding_section(ui: &mut egui::Ui, app: &mut App, actions: &mut Vec<Action>) {
    ui.heading("Decoding");
    let mut opts = app.opts;
    let has_reliability = app.doc.grid.has_reliability();
    let (version, level) = (app.doc.version, app.derived.ec);
    let reach = app.reach(version, level);

    ui.checkbox(&mut opts.reliability, "Use module reliability").on_hover_text(
        "Let the solvers use how clearly each module was read. Leave on for a grid sampled from a photo. Turn off \
         to treat every module as certain, as if painted by hand: doubting, Flip the faintest bits, Belief \
         propagation and error clustering are skipped, and the message prior only fills in unknown modules. Modules \
         painted at certain are always certain.",
    );
    if opts.reliability && !has_reliability {
        ui.indent("reliability_note", |ui| {
            ui.weak("Takes effect once modules are sampled from a photo, or painted below certain.");
        });
    }
    let needs_reliability = (!opts.reliability).then_some("Turn on Use module reliability first.");

    ui.add_space(4.0);
    ui.weak("Tried on every unconfirmed block, best result kept:").on_hover_text(
        "These all run on every block plain Reed-Solomon doesn't confirm, and the best-confirmed solution is kept. \
         Blocks says which one solved each block; its Show outlines the doubted modules.",
    );
    ui.checkbox(&mut opts.bit_level, "Solve unknown bits individually").on_hover_text(
        "For unknown modules spread over more codewords than Reed-Solomon can restore: solves for the unknown bits \
         from the parity equations, rather than for whole codewords. Assumes the other modules are right, except for \
         a codeword or two that Erase codewords by trial may catch. Needed by Single bits, by the equation-guided \
         flips of Flip the faintest bits, and by the bit-by-bit trials of Erase codewords by trial.",
    );
    ui.label("Doubt the least reliable modules:").on_hover_text(
        "For a photo with faint, smudged or glared modules: marks the least clearly read codewords or bits unknown, \
         a few at a time, until the block solves with redundancy to spare. Hand-painted modules are doubted only if \
         painted below certain.",
    );
    ui.indent("doubting", |ui| {
        needing(
            ui,
            &mut opts.doubt_codewords,
            needs_reliability,
            "Whole codewords",
            "For faint modules concentrated in a few codewords: mark the least reliable codewords unknown, one more \
             at a time. An unknown codeword costs one error correction codeword; a wrong one left in costs two.",
        );
        let single_bits_needs = match (opts.reliability, opts.bit_level) {
            (true, true) => None,
            (false, false) => Some("Turn on Use module reliability and Solve unknown bits individually first."),
            (false, true) => needs_reliability,
            (true, false) => Some("Turn on Solve unknown bits individually first."),
        };
        needing(
            ui,
            &mut opts.doubt_bits,
            single_bits_needs,
            "Single bits",
            "For faint modules scattered about one per codeword, across more codewords than Whole codewords can \
             handle: marks the least reliable modules unknown one at a time and solves bit by bit.",
        );
    });

    ui.add_space(4.0);
    ui.weak("Then, one at a time, until the block is confirmed:").on_hover_text(
        "Each runs only if nothing before it has confirmed the block. List decoding and Erase codewords by trial run \
         only if plain Reed-Solomon found no solution at all; otherwise they'd find the same one. The \
         equation-guided flips and bit-by-bit trials run earlier, as part of Solve unknown bits individually.",
    );
    let flipping = needing(
        ui,
        &mut opts.flip_bits,
        needs_reliability,
        "Flip the faintest bits",
        "For a block with a few more errors than its error correction can fix, where the wrong modules read \
         faintly. Chase decoding: tries every combination of flips on the least reliable bits (how many is set \
         below), decoding each. Also flips up to three solidly misread bits that the parity equations point to, when \
         solving bit by bit fails. Every pattern that solves must agree, or the message decides between them. Used \
         only if the symbol doesn't read without it, so turning it on never loses a read. Turn it off if a solution \
         it found looks wrong.",
    );
    if flipping {
        ui.indent("chase_bits", |ui| {
            if !opts.bit_level {
                ui.weak("Solve unknown bits individually is off, so equation-guided flips are skipped.");
            }
            ui.add(egui::Slider::new(&mut opts.chase_bits, 1..=CHASE_MAX_BITS).text("bits flipped")).on_hover_text(
                "Raise it if a block still fails and has more faint modules than this. Each bit doubles the decodes: \
                 16 takes about a second on a large block. Default 10. When reading the photo, the budget is split \
                 across the placements, samplings and formats tried, down to 6.",
            );
            let patterns = (1usize << opts.chase_bits) - 1;
            ui.weak(format!("Up to {} decodes per block. {}", grouped(patterns), confirmation_cost("A solution", patterns)));
        });
    }

    let propagating = needing(
        ui,
        &mut opts.belief_propagation,
        needs_reliability,
        "Belief propagation",
        "For a noisy photo of a small symbol, with more faint modules than doubting can mark unknown or Flip the \
         faintest bits can try. Adaptive belief propagation (Jiang and Narayanan): each round, the parity checks \
         refine every bit's reliability, and the block is decoded again. A solution is confirmed by how rarely random \
         data lands that close to a codeword.",
    );
    if propagating {
        ui.indent("abp_rounds", |ui| {
            ui.add(egui::Slider::new(&mut opts.abp_rounds, ABP_ROUNDS_LEAST..=ABP_ROUNDS_MAX).text("rounds")).on_hover_text(
                "Raise it if a block is still changing when the rounds run out. A block stops early once it settles \
                 or solves, so more rounds can't lose a solution. A round takes a millisecond or two on the largest \
                 blocks. Default 20, and at least 3: the faintest bits are doubted again after the 3rd round, the \
                 6th, the 12th and so on. When reading the photo, the rounds are split across the placements, \
                 samplings and formats tried, down to 6.",
            );
            ui.weak(confirmation_cost("A solution", round_tries(opts.abp_rounds, flipping)));
        });
    }

    ui.checkbox(&mut opts.list_decode, "Use list decoding").on_hover_text(
        "For a block with more wrong codewords than its error correction can fix; helps more at higher levels. \
         Soft-decision list decoding (Koetter–Vardy): finds every valid codeword close enough to the block as read, \
         weighting each codeword by its modules' reliability. Faint modules count as both colours at once, so it \
         reaches further when they're faint than when every module is certain. If it finds two solutions, the \
         message decides between them.",
    );
    if opts.list_decode {
        ui.indent("list_cost", |ui| {
            if !opts.reliability {
                ui.weak("Use module reliability is off, so every module counts as certain.");
            }
            if let Some(reach) = &reach {
                ui.weak(list_note(reach, version, level, opts.list_cost));
            }
            ui.add(count_slider(&mut opts.list_cost, LIST_COST_MIN..=LIST_COST_MAX, "constraints")).on_hover_text(
                "Interpolation budget. Raise it for a block list decoding still can't reach, if you can wait: more \
                 constraints reach further, and run time grows with the square of the budget. The list is capped at \
                 32 codewords, so a small block at a high level reaches less far past a few thousand constraints and \
                 is given no more (see the note below). Default 2,048: about 0.1 s on the largest blocks; 32,768 takes \
                 seconds. When reading the photo, the budget is split across the placements, samplings and formats \
                 tried, down to 512.",
            );
            ui.weak(list_cost_time(opts.list_cost, reach.as_ref().map_or(opts.list_cost, |reach| reach.list_used)));
        });
    }

    ui.checkbox(&mut opts.erase_by_trial, "Erase codewords by trial").on_hover_text(
        "The last resort, and the most expensive. For a hand-painted grid (which the reliability-based solvers can't \
         help) with a codeword or two more wrong than its error correction can fix. Erases every set of codewords of \
         the smallest size that gets past plain Reed-Solomon, then the next size up, regardless of reliability, and \
         keeps a solution the rest of the block confirms. Reaches one wrong codeword past plain Reed-Solomon (two in \
         a small block), or a block whose unknown bits solve except for one to three wrong codewords. Works on \
         sampled grids too. If two solutions disagree, the message decides, or neither is kept. Used only if the \
         symbol doesn't read without it, so turning it on never loses a read.",
    );
    if opts.erase_by_trial {
        ui.indent("trial_sets", |ui| {
            if !opts.bit_level {
                ui.weak("Solve unknown bits individually is off, so bit-by-bit trials are skipped.");
            }
            ui.add(count_slider(&mut opts.trial_sets, TRIAL_SETS_MIN..=TRIAL_SETS_MAX, "sets tried")).on_hover_text(
                format!(
                    "Most codeword sets to erase by trial, across all sizes. Sets are tried smallest first, and each \
                     size reaches one more wrong codeword than the last; the note below says which sizes this symbol's \
                     blocks get. Raise it to reach further, at the confirmation cost shown below. Default 65,536. The \
                     minimum, {}, always covers the smallest size in any block. When reading the photo, the budget is \
                     split across the placements, samplings and formats tried, but never below the minimum.",
                    grouped(TRIAL_SETS_MIN)
                ),
            );
            if let Some(reach) = &reach {
                ui.weak(trial_note(reach));
            }
            // Each size of set tried has an equal share of the trials'
            // chances, and a solution of the largest the least of it.
            let tried = reach.as_ref().map_or(opts.trial_sets, |reach| reach.trials.most_tries());
            ui.weak(confirmation_cost("A solution", tried));
        });
    }

    ui.add_space(4.0);
    ui.weak("Using the rest of the message:").on_hover_text(
        "After each block has been tried on its own, the rest of the message is used to help the ones that \
         failed.",
    );
    ui.checkbox(&mut opts.derive_pins, "Pin the padding and headers found").on_hover_text(
        "For a message shorter than the symbol's capacity, as most are: the padding after it is a known pattern, \
         and a pinned codeword costs no error correction. When the decode doesn't verify, finds where the padding \
         starts (from the segment headers or the pad pattern), pins the pad codewords, the terminator and any header \
         bits with only one valid reading, then re-solves the blocks those pins touch until nothing changes. This is \
         Pin the padding, done automatically. Pins are kept only if they solve or verify another block, or the \
         segment headers confirm them. They show under Pins and in the Pins overlay; Pin the padding, under Padding \
         recovery, adds them to the document. Turn it off if a pin it made looks wrong.",
    );
    ui.checkbox(&mut opts.message_evidence, "Use the message as a prior").on_hover_text(
        "For a block that still fails, when the readable rest of the message shows what kind of data it holds, as \
         plain text or digits do: retries the block with each bit biased toward what the message's other readable \
         groups of the same kind (bytes, digit groups and so on) hold there. Bytes start out biased toward plain \
         ASCII, and a digit never exceeds nine. A module read against the prior is doubted sooner, and an unreadable \
         one is filled in from it. The prior counts toward confirming a solution only where it filled a module in: \
         for the solution if they agree, against it if not, by how sure the prior is. Blocks says when a block was \
         solved this way. Used only if the symbol doesn't read without it, so turning it on never loses a read.",
    );
    if opts.message_evidence && !opts.reliability {
        ui.indent("evidence_note", |ui| {
            ui.weak("Use module reliability is off, so the prior only fills in unknown modules.");
        });
    }

    ui.add_space(4.0);
    // The character set is chosen under Result, and is left as it is.
    let defaults = DecodeOptions { charset: opts.charset, ..super::default_decode_options() };
    // What Restore defaults replaced can be put back until the options
    // are changed some other way.
    if opts != defaults {
        app.view.before_defaults = None;
    }
    ui.horizontal(|ui| {
        if ui
            .add_enabled(opts != defaults, egui::Button::new("Restore defaults"))
            .on_hover_text("Reset every option and budget above to its default")
            .on_disabled_hover_text("Everything above is already at its default")
            .clicked()
        {
            app.view.before_defaults = Some(opts);
            opts = defaults;
        }
        if let Some(before) = app.view.before_defaults
            && ui.button("Put back").on_hover_text("Go back to the options and budgets you had before Restore defaults").clicked()
        {
            app.view.before_defaults = None;
            opts = DecodeOptions { charset: opts.charset, ..before };
        }
    });
    if opts != app.opts {
        actions.push(Action::SetDecodeOptions(opts));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_notes_on_reach_say_what_spends_the_codewords_the_standard_keeps() {
        let opts = super::super::default_decode_options();
        // 1-H keeps one codeword: list decoding and the trials reach past
        // the code only by spending it, the codewords the trials erase
        // being read and wrong as the standard counts a solution's errors.
        // The triples were once said to reach nine within it.
        let reach = Reach::of(SymbolVersion::Qr(1), EcLevel::H, &opts).unwrap();
        assert_eq!(
            list_note(&reach, SymbolVersion::Qr(1), EcLevel::H, opts.list_cost),
            "In this symbol's block of 26 codewords: corrects at least 11 wrong codewords, vs. 8 for plain \
             Reed-Solomon. With every module certain, anything past 8 uses the reserved error-detection codewords and \
             is never confirmed."
        );
        // 1-M keeps two: list decoding reaches six, a codeword past half
        // the distance, its codewords given a multiplicity of ten alike,
        // but past four spends them. It was once said to correct five
        // where plain decoding corrects four; and then, its codewords
        // given the budget a unit at a time, to reach no further than
        // half the distance, which plain decoding reaches too.
        let one_m = Reach::of(SymbolVersion::Qr(1), EcLevel::M, &opts).unwrap();
        assert_eq!(
            list_note(&one_m, SymbolVersion::Qr(1), EcLevel::M, opts.list_cost),
            "In this symbol's block of 26 codewords: corrects at least 6 wrong codewords, vs. 4 for plain Reed-Solomon. \
             With every module certain, anything past 4 uses the reserved error-detection codewords and is never \
             confirmed."
        );
        assert_eq!(
            trial_note(&reach),
            "In this symbol's block of 26 codewords: tries every single codeword and every triple (2,626 sets), but \
             these only beat plain Reed-Solomon by using the reserved error-detection codewords, so they are never \
             confirmed. Adding every set of five would need a budget of 68,406."
        );
        // 3-L at the least budget tries the single codewords alone.
        let least = DecodeOptions { trial_sets: TRIAL_SETS_MIN, ..opts };
        let reach = Reach::of(SymbolVersion::Qr(3), EcLevel::L, &least).unwrap();
        assert_eq!(
            trial_note(&reach),
            "In this symbol's block of 70 codewords: tries every single codeword (70 sets), but these only beat plain \
             Reed-Solomon by using the reserved error-detection codewords, so they are never confirmed. Adding every \
             triple would need a budget of 54,810."
        );
        // M2-L keeps three of five: even its triples beat plain
        // Reed-Solomon only by spending them.
        let reach = Reach::of(SymbolVersion::Micro(2), EcLevel::L, &opts).unwrap();
        assert!(trial_note(&reach).ends_with("so they are never confirmed."), "{}", trial_note(&reach));
        // The curve reaches three wrong, a codeword past half the
        // distance, with what the block itself must show of a codeword
        // to be verified, and a decode finds them there; but past the
        // one the standard corrects it spends the reserve. Said once
        // against three codewords' worth, it reached the one, as "1 wrong
        // codewords"; and then, held to what a fallback's own count must
        // show, the two plain decoding reaches.
        assert_eq!(
            list_note(&reach, SymbolVersion::Micro(2), EcLevel::L, opts.list_cost),
            "In this symbol's block of 10 codewords: corrects at least 3 wrong codewords, vs. 1 for plain \
             Reed-Solomon. With every module certain, anything past 1 uses the reserved error-detection codewords and \
             is never confirmed."
        );
        // 2-L keeps two of ten: the pairs reach five within the standard's
        // limit, where plain decoding goes first by spending them, and six
        // only by spending them too. It was once said to correct five.
        let reach = Reach::of(SymbolVersion::Qr(2), EcLevel::L, &opts).unwrap();
        assert_eq!(
            trial_note(&reach),
            "In this symbol's block of 44 codewords: tries every pair (946 sets), but these only beat plain \
             Reed-Solomon by using the reserved error-detection codewords, so they are never confirmed. Adding every \
             set of four would need a budget of 136,697."
        );
        // 1-M keeps two of ten too: the sets of four reach six, but past
        // the four the standard corrects, as it counts a solution's
        // errors, the codewords erased among them. It was once said to
        // correct up to six, and then to correct six.
        assert_eq!(
            trial_note(&one_m),
            "In this symbol's block of 26 codewords: tries every pair and every set of four (15,275 sets), but these \
             only beat plain Reed-Solomon by using the reserved error-detection codewords, so they are never \
             confirmed. Adding every set of six would need a budget of 245,505."
        );
    }
}
