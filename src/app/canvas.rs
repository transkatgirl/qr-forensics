//! The central canvas. In the Align stage it shows the photo with the grid
//! warped onto it; in the Edit stage it shows the grid square-on, with the
//! photo rectified underneath.

use eframe::egui::{self, Color32, Pos2, Rect, Shape, Stroke as Line, StrokeKind, Vec2, pos2, vec2};
use qr_forensics::decode::bitstream::EndOfData;
use qr_forensics::decode::codewords::PinSource;
use qr_forensics::decode::pipeline::CwStatus;
use qr_forensics::imaging::Luma;
use qr_forensics::imaging::binarize::binarize;
use qr_forensics::imaging::detect::{Anchor, AnchorKind, anchor_sites, edge_sites};
use qr_forensics::imaging::projection::{CORNER_CLEARANCE, Project, Projection};
use qr_forensics::symbol::function_map::Role;
use qr_forensics::symbol::grid::{CERTAIN, Module};
use qr_forensics::symbol::mask::mask_bit;

use super::state::*;
use super::{Action, App};

const HANDLE_NAMES: [&str; 4] = ["TL", "TR", "BR", "BL"];
pub(super) const ACCENT: Color32 = Color32::from_rgb(0, 150, 255);
pub(super) const SELECT: Color32 = Color32::from_rgb(255, 210, 0);
pub(super) const GOOD: Color32 = Color32::from_rgb(40, 190, 90);
const BAD: Color32 = Color32::from_rgb(235, 60, 60);
const UNKNOWN_TINT: Color32 = Color32::from_rgb(240, 150, 40);
const ALIGNMENT: Color32 = Color32::from_rgb(150, 90, 255);
/// Timing anchors, in the tint the Edit-stage role overlay gives timing
/// patterns, so that the two views agree on what is what.
const TIMING: Color32 = Color32::from_rgb(0, 200, 200);
/// Edge points traced along the symbol's edge.
const EDGE: Color32 = Color32::from_rgb(235, 150, 40);
/// Anchors set by hand on modules where no pattern is.
const CUSTOM: Color32 = Color32::from_rgb(240, 240, 240);
/// Outlines of symbols a search of the whole photo found.
const PROPOSAL: Color32 = Color32::from_rgb(255, 120, 200);
/// The modules of the character picked in the result, and that character
/// in the result's text.
pub(super) const PICKED: Color32 = Color32::from_rgb(160, 235, 0);
/// Modules whose bits a block's solution was reached by doubting.
const DOUBTED: Color32 = Color32::from_rgb(190, 110, 255);
/// Modules of photo shown around the symbol in the Edit stage.
const MARGIN: i32 = 2;
/// Half-size of an anchor handle's hit area, in screen points.
const ANCHOR_HIT: f32 = 9.0;
/// Half-size of a corner handle's hit area, in screen points.
const CORNER_HIT: f32 = 13.0;
/// Radius of a corner handle's ring, in screen points.
const CORNER_RING: f32 = 9.0;
/// Reliability (distance from the threshold, 0..=254) at which the
/// heatmap's tint has faded out: a module this far from the threshold is
/// not in doubt.
const RELIABILITY_FADE: u8 = 120;

pub(super) fn show(ui: &mut egui::Ui, app: &mut App, actions: &mut Vec<Action>) {
    toolbar(ui, app, actions);
    // Stay inside the visible part of the panel even if the toolbar above
    // did not fit and pushed the layout wider.
    let visible = ui.available_rect_before_wrap().intersect(ui.clip_rect());
    let (response, painter) = ui.allocate_painter(visible.size(), egui::Sense::click_and_drag());
    painter.rect_filled(response.rect, 0.0, ui.visuals().extreme_bg_color);
    if app.view.stage == Stage::Align && app.photo.is_some() {
        align_stage(ui, app, &response, &painter, actions);
    } else {
        edit_stage(ui, app, &response, &painter);
    }
}

fn toolbar(ui: &mut egui::Ui, app: &mut App, actions: &mut Vec<Action>) {
    ui.horizontal_wrapped(|ui| {
        let has_photo = app.photo.is_some();
        ui.add_enabled_ui(has_photo, |ui| {
            ui.selectable_value(&mut app.view.stage, Stage::Align, "Align")
                .on_hover_text("Place the grid on the photo")
                .on_disabled_hover_text("Load a photo to align a grid to it");
        });
        ui.selectable_value(&mut app.view.stage, Stage::Edit, "Edit").on_hover_text("Paint and inspect modules");
        ui.separator();
        if ui.button("Fit").on_hover_text("Fit to window (F)").clicked() {
            actions.push(Action::FitView);
        }
        ui.separator();
        match app.view.stage {
            Stage::Align if has_photo => {
                ui.label("Show");
                ui.selectable_value(&mut app.view.align_overlay, AlignOverlay::FixedPatterns, "Fixed patterns")
                    .on_hover_text("Where finders, timing and alignment patterns must fall");
                ui.selectable_value(&mut app.view.align_overlay, AlignOverlay::SamplePreview, "Sample preview")
                    .on_hover_text("What sampling would read from each module: a black dot for dark, white for light, orange for unknown");
                ui.selectable_value(&mut app.view.align_overlay, AlignOverlay::LinesOnly, "Lines only");
                ui.selectable_value(&mut app.view.align_overlay, AlignOverlay::Binarised, "Binarised")
                    .on_hover_text("The photo in black and white, as the search for finder patterns sees it; its settings appear under Photo while it is shown");
            }
            _ => {
                if has_photo {
                    ui.selectable_value(&mut app.view.layers, Layers::Both, "Both")
                        .on_hover_text("Painted modules blended over the photo");
                    ui.selectable_value(&mut app.view.layers, Layers::Photo, "Photo").on_hover_text("Hold H");
                    ui.selectable_value(&mut app.view.layers, Layers::Grid, "Grid");
                    if app.view.layers == Layers::Both {
                        ui.separator();
                        toolbar_slider(ui, &mut app.view.grid_opacity, 0.1..=1.0, "grid")
                            .on_hover_text("Opacity of painted modules over the photo");
                        toolbar_slider(ui, &mut app.view.photo_opacity, 0.0..=1.0, "photo")
                            .on_hover_text("Opacity of the photo under the grid");
                    }
                    ui.separator();
                }
                const COMBO_WIDTH: f32 = 130.0;
                let width = text_width(ui, "Overlay") + ui.spacing().item_spacing.x + COMBO_WIDTH;
                toolbar_item(ui, width, |ui| {
                    ui.label("Overlay");
                    egui::ComboBox::from_id_salt("overlay")
                        .width(COMBO_WIDTH)
                        .selected_text(app.view.overlay.label())
                        .show_ui(ui, |ui| {
                            for overlay in Overlay::ALL {
                                ui.selectable_value(&mut app.view.overlay, overlay, overlay.label());
                            }
                        });
                });
                if app.view.overlay == Overlay::Reliability {
                    ui.weak(format!(
                        "Red: painted as a guess, or sampled close to the threshold, fading out by a margin of {:.2}; other painted modules untinted",
                        RELIABILITY_FADE as f32 / (CERTAIN - 1) as f32
                    ));
                } else if app.view.overlay == Overlay::Photo && app.preview.is_none() {
                    ui.weak("Load a photo and place the corners to compare the grid with it");
                } else if app.view.overlay == Overlay::Photo
                    && let Some(note) = app.inverted_note()
                {
                    // Nothing inverts the photo for the comparison, as
                    // Sample all does for itself: every module differs.
                    ui.colored_label(ui.visuals().warn_fg_color, format!("The grid is compared with the photo as it is. {note}"));
                } else if !app.view.overlay.legend().is_empty() {
                    ui.weak(app.view.overlay.legend());
                }
            }
        }
    });
}

/// Width of `text` set in the body font, as a label would be.
fn text_width(ui: &egui::Ui, text: &str) -> f32 {
    let font = egui::TextStyle::Body.resolve(ui.style());
    ui.painter().layout_no_wrap(text.to_owned(), font, Color32::PLACEHOLDER).size().x
}

/// A toolbar control of a known width. The toolbar wraps, and a wrapping row
/// can only move a control to the next row in one piece if it knows the width
/// up front; egui's sliders and combo boxes lay themselves out in a nested
/// row that claims whatever is left, and run off the edge instead.
fn toolbar_item<R>(ui: &mut egui::Ui, width: f32, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let size = vec2(width, ui.spacing().interact_size.y);
    ui.allocate_ui_with_layout(size, egui::Layout::left_to_right(egui::Align::Center), add).inner
}

/// A compact labelled slider for the toolbar.
fn toolbar_slider(ui: &mut egui::Ui, value: &mut f32, range: std::ops::RangeInclusive<f32>, text: &str) -> egui::Response {
    let rail = 80.0;
    let (value_box, gap) = (ui.spacing().interact_size.x, ui.spacing().item_spacing.x);
    let width = rail + gap + value_box + gap + text_width(ui, text);
    toolbar_item(ui, width, |ui| {
        ui.spacing_mut().slider_width = rail;
        ui.add(egui::Slider::new(value, range).text(text))
    })
}

/// Pan and zoom handling shared by both stages. Returns true while the
/// pointer is being used to move the view.
fn navigate(ui: &egui::Ui, response: &egui::Response, view: &mut PanZoom, world: Rect) -> bool {
    let canvas = response.rect;
    if view.auto_fit {
        view.fit(canvas, world);
    }
    if response.contains_pointer() {
        let (zoom, scroll, pointer) = ui.input(|i| (i.zoom_delta(), i.smooth_scroll_delta(), i.pointer.hover_pos()));
        if let Some(pointer) = pointer
            && zoom != 1.0
        {
            view.zoom_about(canvas, pointer, zoom);
            view.auto_fit = false;
        }
        if scroll != Vec2::ZERO {
            view.center -= scroll / view.scale;
            view.auto_fit = false;
        }
    }
    let space = !ui.ctx().egui_wants_keyboard_input() && ui.input(|i| i.key_down(egui::Key::Space));
    let dragging = response.dragged_by(egui::PointerButton::Middle)
        || (space && response.dragged_by(egui::PointerButton::Primary));
    if dragging {
        view.center -= response.drag_delta() / view.scale;
        view.auto_fit = false;
    }
    dragging || space
}

fn quad_mesh(mesh: &mut egui::Mesh, corners: [Pos2; 4], color: Color32) {
    let base = mesh.vertices.len() as u32;
    for corner in corners {
        mesh.colored_vertex(corner, color);
    }
    mesh.add_triangle(base, base + 1, base + 2);
    mesh.add_triangle(base, base + 2, base + 3);
}

/// Bilinear point inside a module whose corners are `c00`, `c10`, `c11`,
/// `c01` (top-left, top-right, bottom-right, bottom-left), at (`u`, `v`)
/// across it.
fn within(c00: Pos2, c10: Pos2, c11: Pos2, c01: Pos2, u: f32, v: f32) -> Pos2 {
    let top = c00 + (c10 - c00) * u;
    let bottom = c01 + (c11 - c01) * u;
    top + (bottom - top) * v
}

fn align_stage(
    ui: &mut egui::Ui,
    app: &mut App,
    response: &egui::Response,
    painter: &egui::Painter,
    actions: &mut Vec<Action>,
) {
    let App { photo, view, derived, preview, doc, history, model, proposals, leads, placements, job, .. } = app;
    let photo = photo.as_mut().expect("align stage needs a photo");
    // Holding H shows the photo and nothing over it, as in the Edit
    // stage: no grid, no handles, no symbols found, and the photo itself
    // where the binarised view would be.
    let peek = !ui.ctx().egui_wants_keyboard_input() && ui.input(|i| i.key_down(egui::Key::H));
    // The binarised view's texture is made when first shown and kept
    // while the luminance and the settings stay the same.
    let texture = match view.align_overlay {
        AlignOverlay::Binarised if !peek => {
            view.binarize = view.binarize_for(doc.corners, photo.width, photo.height);
            if !photo.binarised.as_ref().is_some_and(|b| b.is_for(&photo.luma, &view.binarize)) {
                let bw = binarize(&photo.luma, &view.binarize);
                let gray: Vec<u8> = bw.data.iter().map(|&v| if v < 0.5 { 0 } else { 255 }).collect();
                // Within what a texture may measure, as the photo's own is.
                let image = super::texture_gray(gray, bw.width, bw.height, ui.input(|i| i.max_texture_side));
                let texture = ui.ctx().load_texture("binarised", image, egui::TextureOptions::LINEAR);
                photo.binarised = Some(Binarised { source: photo.luma.clone(), params: view.binarize, texture });
            }
            photo.binarised.as_ref().expect("just built").texture.id()
        }
        _ => photo.texture.id(),
    };
    let photo = &*photo;
    // While a step runs on its own thread, a symbol found cannot be
    // adopted: one job at a time.
    let busy = job.is_some();
    let canvas = response.rect;
    let (gw, gh) = (derived.func.width, derived.func.height);
    let world = Rect::from_min_size(Pos2::ZERO, vec2(photo.width as f32, photo.height as f32));
    let panning = navigate(ui, response, &mut view.align_view, world);
    let pz = view.align_view;
    let to_screen = |p: [f64; 2]| pz.to_screen(canvas, vec2(p[0] as f32, p[1] as f32));
    let pointer = response.hover_pos();

    painter.image(
        texture,
        Rect::from_min_max(to_screen([0.0, 0.0]), to_screen([photo.width as f64, photo.height as f64])),
        Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
        Color32::WHITE,
    );

    // Symbols a search of the whole photo found can be clicked to adopt,
    // inside their outlines. The one the pointer is on is the first in
    // the list whose outline the pointer is inside, so that where two
    // overlap the better claim takes the click. The one the outline has
    // been put on is not among them: a click inside it is meant for the
    // grid, and would put the outline back as found over whatever was
    // placed by hand since. The click is the canvas's own, and every
    // handle lies on top of the canvas, so a click or a right-click on a
    // handle inside a symbol found is the handle's. They are drawn
    // further down, over the grid.
    let in_use = proposals.as_ref().and_then(|proposals| super::proposal_in_use(proposals, &doc.corners));
    let hot = pointer.filter(|_| !peek && !panning).and_then(|pointer| {
        let at = pz.to_world(canvas, pointer);
        let at = [at.x as f64, at.y as f64];
        let found = &proposals.as_ref()?.found;
        (0..found.len()).find(|&i| in_use != Some(i) && super::quad_contains(&found[i].corners, at))
    });
    // The pointer is on a handle, whose business a press there is.
    let on = |center: Pos2, half: f32| pointer.is_some_and(|p| Rect::from_center_size(center, Vec2::splat(2.0 * half)).contains(p));
    let mut on_handle = doc.corners.iter().any(|&corner| on(to_screen(corner), CORNER_HIT));

    if let Some(model) = model.as_ref().filter(|m| !peek && m.width == gw && m.height == gh) {
        let corner = |x: usize, y: usize| to_screen(model.corner(x, y));
        let module_quad = |x: usize, y: usize, inset: f32| {
            let (c00, c10, c11, c01) = (corner(x, y), corner(x + 1, y), corner(x + 1, y + 1), corner(x, y + 1));
            let (a, b) = (inset, 1.0 - inset);
            [within(c00, c10, c11, c01, a, a), within(c00, c10, c11, c01, b, a), within(c00, c10, c11, c01, b, b), within(c00, c10, c11, c01, a, b)]
        };
        let mut mesh = egui::Mesh::default();
        match view.align_overlay {
            AlignOverlay::FixedPatterns => {
                for y in 0..gh {
                    for x in 0..gw {
                        match derived.func.expected(x, y) {
                            Some(true) => quad_mesh(&mut mesh, module_quad(x, y, 0.08), ACCENT.gamma_multiply(0.55)),
                            Some(false) => {
                                quad_mesh(&mut mesh, module_quad(x, y, 0.08), Color32::from_white_alpha(70));
                            }
                            None => {}
                        }
                    }
                }
            }
            AlignOverlay::SamplePreview => {
                // A dot of the colour each module would be painted, as
                // the Edit stage paints it, with a rim of the opposite
                // so that it shows on any photo: a dot that agrees with
                // the photo under it all but disappears, and one that
                // does not stands out.
                if let Some(preview) = preview {
                    for y in 0..gh {
                        for x in 0..gw {
                            let (dot, rim) = match preview.classified.modules[y * gw + x] {
                                Module::Dark => (Color32::BLACK, Color32::WHITE),
                                Module::Light => (Color32::WHITE, Color32::BLACK),
                                Module::Unknown => (UNKNOWN_TINT, Color32::BLACK),
                            };
                            quad_mesh(&mut mesh, module_quad(x, y, 0.27), rim);
                            quad_mesh(&mut mesh, module_quad(x, y, 0.34), dot);
                        }
                    }
                }
            }
            AlignOverlay::LinesOnly | AlignOverlay::Binarised => {}
        }
        painter.add(mesh);

        // Grid lines follow the projection, so a warped symbol shows its
        // warp. Skip lines when modules are too small on screen to tell
        // apart.
        let module_px = (corner(0, 0).distance(corner(gw, 0)) / gw as f32).min(corner(0, 0).distance(corner(0, gh)) / gh as f32);
        let step = if module_px >= 5.0 { 1 } else { (5.0 / module_px.max(0.01)).ceil() as usize };
        let thin = Line::new(1.0, ACCENT.gamma_multiply(0.6));
        for x in (step..gw).step_by(step) {
            let points: Vec<Pos2> = (0..=gh).step_by(step).chain(std::iter::once(gh)).map(|y| corner(x, y)).collect();
            painter.add(Shape::line(points, thin));
        }
        for y in (step..gh).step_by(step) {
            let points: Vec<Pos2> = (0..=gw).step_by(step).chain(std::iter::once(gw)).map(|x| corner(x, y)).collect();
            painter.add(Shape::line(points, thin));
        }
        let mut outline: Vec<Pos2> = (0..=gw).map(|x| corner(x, 0)).collect();
        outline.extend((1..=gh).map(|y| corner(gw, y)));
        outline.extend((0..gw).rev().map(|x| corner(x, gh)));
        outline.extend((1..gh).rev().map(|y| corner(0, y)));
        painter.add(Shape::closed_line(outline, Line::new(2.0, ACCENT)));

        // Anchor handles: one per site, filled where an anchor is set,
        // hollow where the projection merely predicts the pattern. Drag a
        // handle to set an anchor by hand; right-click to drop one.
        if view.show_anchors {
            // The grid itself may be taken hold of anywhere, to bend it
            // where it has no pattern to be anchored on: a drag begun on
            // a module of it, clear of every handle, sets a custom anchor
            // on that module, and the anchor goes as far as the pointer
            // does from there. The anchor stands for the middle of the
            // module whatever point of it was under the pointer, as
            // every other anchor stands for the middle of its pattern,
            // and starts where the grid has that middle, so that nothing
            // moves until the pointer does. A module a pattern is
            // centred on is that pattern's site, and an anchor set on
            // the module already is the one moved. The handles lie over
            // the canvas, so a drag begun on one is that handle's and
            // never comes here; with Space held the drag moves the view.
            // Within reach of a corner the map yields to the corner
            // ([`CORNER_CLEARANCE`]) and an anchor there would bend
            // nothing, so none is set.
            let sites = anchor_sites(&derived.func);
            let graspable = |px: [f64; 2]| -> Option<[f64; 2]> {
                let (w, h) = (gw as f64, gh as f64);
                let at = model.module_at(px)?;
                let at = [at[0].floor().min(w - 1.0) + 0.5, at[1].floor().min(h - 1.0) + 0.5];
                let near_corner = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]].iter().any(|c| (c[0] - at[0]).hypot(c[1] - at[1]) < CORNER_CLEARANCE);
                (!near_corner).then_some(at)
            };
            let world = |p: Pos2| {
                let at = pz.to_world(canvas, p);
                [at.x as f64, at.y as f64]
            };
            if view.use_anchors && !panning {
                if response.drag_started_by(egui::PointerButton::Primary)
                    && let Some(pressed) = ui.input(|i| i.pointer.press_origin()).or(response.interact_pointer_pos())
                    && let Some(at) = graspable(world(pressed))
                {
                    history.checkpoint(doc);
                    let existing = doc.anchor_at(at).map(|a| a.anchor);
                    let site = sites.iter().find(|s| (s.at[0] - at[0]).abs() < 1e-6 && (s.at[1] - at[1]).abs() < 1e-6);
                    let kind = site.map(|s| s.kind).or(existing.map(|a| a.kind)).unwrap_or(AnchorKind::Custom);
                    let from = existing.map_or_else(|| model.projection.apply(at[0], at[1]), |a| a.center);
                    let pressed = world(pressed);
                    let lead = [from[0] - pressed[0], from[1] - pressed[1]];
                    view.grasped = Some(Grasp { kind, at, partial: existing.is_some_and(|a| a.partial), lead });
                }
                if response.hovered() && pointer.is_some_and(|p| graspable(world(p)).is_some()) {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                }
            }
            match view.grasped {
                Some(Grasp { kind, at, partial, lead }) if response.dragged_by(egui::PointerButton::Primary) => {
                    if let Some(p) = response.interact_pointer_pos() {
                        let p = world(p);
                        let anchor = Anchor { kind, at, center: [p[0] + lead[0], p[1] + lead[1]], partial, inferred: false };
                        doc.set_anchor(anchor, AnchorSource::Manual);
                    }
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                }
                _ => view.grasped = None,
            }
            // The handle under the pointer, whose intensity profile is shown.
            let mut profile = None;
            // The sites the version has; the places along each edge where
            // an edge point may be set by hand, to bow the outline there,
            // less any an edge point already traced by a detection sits
            // on; and those traced edge points, which have no site of
            // their own.
            let mut handles: Vec<(AnchorKind, [f64; 2])> = sites.iter().map(|s| (s.kind, s.at)).collect();
            let traced: Vec<[f64; 2]> =
                doc.anchors.iter().filter(|a| a.anchor.kind == AnchorKind::Edge).map(|a| a.anchor.at).collect();
            let near_traced = |at: [f64; 2]| traced.iter().any(|t| (t[0] - at[0]).hypot(t[1] - at[1]) < 1.0);
            handles.extend(edge_sites(&derived.func).into_iter().filter(|s| doc.anchor_at(s.at).is_some() || !near_traced(s.at)).map(|s| (s.kind, s.at)));
            let placed: Vec<[f64; 2]> = handles.iter().map(|h| h.1).collect();
            handles.extend(traced.iter().filter(|&&at| !placed.contains(&at)).map(|&at| (AnchorKind::Edge, at)));
            // The custom anchors, each of which is a handle of its own.
            handles.extend(doc.anchors.iter().filter(|a| a.anchor.kind == AnchorKind::Custom).map(|a| (AnchorKind::Custom, a.anchor.at)));
            for (kind, at) in handles {
                let existing = doc.anchor_at(at).copied();
                let position = match &existing {
                    Some(a) => a.anchor.center,
                    None => model.projection.apply(at[0], at[1]),
                };
                let center = to_screen(position);
                // A handle is known by the module point it stands for,
                // which is its own for as long as it exists. Its place in
                // the list is not: a point traced along an edge has none
                // but its place among the anchors, and a drag that went
                // by the place would pass from one such point to the
                // next whenever the anchors were reordered under it.
                let id = response.id.with(("anchor", at[0].to_bits(), at[1].to_bits()));
                let handle = ui.interact(Rect::from_center_size(center, Vec2::splat(2.0 * ANCHOR_HIT)), id, egui::Sense::click_and_drag());
                on_handle |= on(center, ANCHOR_HIT);
                if handle.drag_started() {
                    history.checkpoint(doc);
                }
                let mut current = position;
                if handle.dragged() {
                    let delta = handle.drag_delta() / pz.scale;
                    current = [position[0] + delta.x as f64, position[1] + delta.y as f64];
                    let anchor = Anchor { kind, at, center: current, partial: existing.is_some_and(|a| a.anchor.partial), inferred: false };
                    doc.set_anchor(anchor, AnchorSource::Manual);
                }
                if handle.hovered() || handle.dragged() {
                    profile = Some(ProfileAt { kind, at, px: current });
                }
                if handle.secondary_clicked() && existing.is_some() {
                    history.checkpoint(doc);
                    doc.remove_anchor(at);
                }
                let color = match kind {
                    AnchorKind::Finder | AnchorKind::FinderSub => GOOD,
                    AnchorKind::Alignment => ALIGNMENT,
                    AnchorKind::Timing => TIMING,
                    AnchorKind::Edge => EDGE,
                    AnchorKind::Custom => CUSTOM,
                };
                let hovered = handle.hovered() || handle.dragged();
                // Timing sites are many, a dozen or more on a large
                // symbol, and each marks a single module: their handles
                // are drawn smaller so the patterns stay visible between
                // them, though they take the same hit area.
                let base = if matches!(kind, AnchorKind::Timing | AnchorKind::Edge | AnchorKind::Custom) { 4.0 } else { 5.5 };
                let radius = if hovered { base + 1.5 } else { base };
                painter.circle_stroke(center, radius, Line::new(3.0, Color32::from_black_alpha(140)));
                match existing {
                    Some(a) => {
                        if a.anchor.inferred {
                            // Not found but placed between its neighbours:
                            // hollow, as a prediction is, only heavier.
                            painter.circle_stroke(center, radius, Line::new(2.5, color));
                        } else {
                            painter.circle_filled(center, radius, color);
                        }
                        if a.source == AnchorSource::Manual {
                            painter.circle_stroke(center, radius + 2.0, Line::new(1.5, SELECT));
                        }
                        if a.anchor.partial {
                            painter.circle_stroke(center, radius + 2.0, Line::new(1.5, BAD));
                        }
                    }
                    None => {
                        painter.circle_stroke(center, radius, Line::new(1.5, color));
                    }
                }
                let what = match existing {
                    Some(a) if a.anchor.inferred => {
                        "not found; placed where the lines through its found neighbours cross; drag onto the pattern, right-click to drop"
                    }
                    Some(a) if a.source == AnchorSource::Manual => "placed by hand; drag to move, right-click to drop",
                    Some(_) if kind == AnchorKind::Edge => "traced along the symbol's edge in the photo; drag to move, right-click to drop",
                    Some(_) => "found in the photo; drag to move, right-click to drop",
                    None if kind == AnchorKind::Edge => {
                        "not set; drag to where the edge really runs and the grid bows through it, for a symbol on a bottle or \
                         curled paper"
                    }
                    None => "predicted; drag onto the pattern to anchor the grid there",
                };
                handle.on_hover_text(format!("{} at ({}, {}): {what}", kind.label(), at[0], at[1]));
            }
            if let Some(at) = profile {
                let threshold = preview.as_ref().map(|p| p.classified.threshold).or(view.sample.threshold).unwrap_or(0.5);
                profile_window(ui, &photo.luma, model.projection.as_ref(), at, threshold);
            }
        }
    }

    // Finder patterns the search could fit no symbol to: the outline it
    // proposed about them, thinner than a symbol's and numbered with a
    // question mark, as the list under Alignment numbers them.
    if !peek {
        for (i, lead) in leads.iter().enumerate() {
            let points: Vec<Pos2> = lead.outline.iter().map(|&c| to_screen(c)).collect();
            let middle = pos2(points.iter().map(|p| p.x).sum::<f32>() / 4.0, points.iter().map(|p| p.y).sum::<f32>() / 4.0);
            painter.add(Shape::closed_line(points, Line::new(1.0, PROPOSAL)));
            let text = format!("{}?", i + 1);
            let font = egui::FontId::proportional(14.0);
            painter.text(middle + vec2(1.0, 1.0), egui::Align2::CENTER_CENTER, &text, font.clone(), Color32::BLACK);
            painter.text(middle, egui::Align2::CENTER_CENTER, &text, font, PROPOSAL);
        }
    }
    // Symbols a search of the whole photo found: outlined and numbered as
    // the list under Alignment numbers them, but for the one the outline
    // has been put on, which the grid's own outline shows.
    if let Some(proposals) = proposals.as_ref().filter(|_| !peek) {
        let hot = hot.filter(|_| !on_handle);
        for (i, found) in proposals.found.iter().enumerate() {
            if in_use == Some(i) {
                continue;
            }
            let points: Vec<Pos2> = found.corners.iter().map(|&c| to_screen(c)).collect();
            let middle = pos2(points.iter().map(|p| p.x).sum::<f32>() / 4.0, points.iter().map(|p| p.y).sum::<f32>() / 4.0);
            let hovered = hot == Some(i);
            painter.add(Shape::closed_line(points, Line::new(if hovered { 3.5 } else { 2.0 }, PROPOSAL)));
            let text = (i + 1).to_string();
            let font = egui::FontId::proportional(if hovered { 20.0 } else { 16.0 });
            painter.text(middle + vec2(1.0, 1.0), egui::Align2::CENTER_CENTER, &text, font.clone(), Color32::BLACK);
            painter.text(middle, egui::Align2::CENTER_CENTER, &text, font, PROPOSAL);
        }
        if let Some(i) = hot {
            let found = &proposals.found[i];
            if response.clicked() && !busy {
                actions.push(Action::UseProposal(i));
            }
            let polarity = if found.inverted { ", light on dark" } else { "" };
            let decodes = if found.decodes == Some(true) { ", and it decodes as found" } else { "" };
            // Read through to a message by Read all, it is loaded as it
            // was read, where reads are; not read yet, it is read then.
            let read = proposals.reads.as_ref().and_then(|reads| reads.results.get(i)?.as_ref());
            let how = if busy {
                "Wait for the step under way to finish before placing the corners on it.".to_owned()
            } else if view.load_reads && read.is_some_and(|read| read.is_trustworthy()) {
                "Click to place the corners and anchors on it and write the grid as it was read (undoable).".to_owned()
            } else if view.load_reads && proposals.reads.is_none() {
                "Click to place the corners and anchors on it and read it through: a read that decodes is written into the \
                 grid (undoable)."
                    .to_owned()
            } else {
                format!("Click to place the corners and anchors on it.{}", super::new_grid_note(found, doc.version))
            };
            response.clone().on_hover_text_at_pointer(format!(
                "{}. {} {}: {:.0}% of fixed patterns read right{polarity}{decodes}. {how}",
                i + 1,
                found.version.family().label(),
                found.version,
                100.0 * found.score()
            ));
        }
    }

    // The placement the pointer is on in the list under Alignment: its
    // outline and where it has its anchors, so that one placement is told
    // from the next before either is taken.
    if let Some(found) = placements.as_ref().zip(view.hover_placement).and_then(|(placements, i)| placements.found.get(i)).filter(|_| !peek) {
        let points: Vec<Pos2> = found.corners.iter().map(|&c| to_screen(c)).collect();
        for (width, color) in [(5.0, Color32::from_black_alpha(160)), (2.5, PROPOSAL)] {
            painter.add(Shape::closed_line(points.clone(), Line::new(width, color)));
        }
        for anchor in &found.anchors {
            painter.circle_filled(to_screen(anchor.center), 3.5, PROPOSAL);
        }
        painter.text(points[0] + vec2(-8.0, -8.0), egui::Align2::RIGHT_BOTTOM, "TL", egui::FontId::proportional(13.0), PROPOSAL);
    }

    // The corner handles: a ring about the corner and four ticks towards
    // it, which leave the corner itself, and the photo under it, in view.
    let middle = doc.corners.iter().fold([0.0, 0.0], |a, c| [a[0] + c[0] / 4.0, a[1] + c[1] / 4.0]);
    let mut loupe = None;
    for (i, name) in HANDLE_NAMES.iter().enumerate() {
        let center = to_screen(doc.corners[i]);
        let handle =
            ui.interact(Rect::from_center_size(center, Vec2::splat(2.0 * CORNER_HIT)), response.id.with(("corner", i)), egui::Sense::drag());
        if handle.drag_started() {
            // One undo step per drag.
            history.checkpoint(doc);
        }
        if handle.dragged() {
            let delta = handle.drag_delta() / pz.scale;
            doc.corners[i][0] += delta.x as f64;
            doc.corners[i][1] += delta.y as f64;
            view.active_corner = i;
        }
        if handle.drag_started() || handle.clicked() {
            view.active_corner = i;
        }
        if handle.hovered() || handle.dragged() {
            loupe = Some(i);
        }
        if peek {
            continue;
        }
        let center = to_screen(doc.corners[i]);
        let active = view.active_corner == i;
        let fill = if active { SELECT } else { Color32::WHITE };
        let away = (center - to_screen(middle)).normalized();
        let away = if away.is_finite() { away } else { vec2(0.0, -1.0) };
        for (width, color) in [(4.0, Color32::BLACK), (2.0, fill)] {
            let line = Line::new(width, color);
            painter.circle_stroke(center, CORNER_RING, line);
            for tick in [vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(-1.0, 0.0), vec2(0.0, -1.0)] {
                painter.line_segment([center + tick * CORNER_RING, center + tick * (CORNER_RING - 5.0)], line);
            }
        }
        // Labels sit outside the symbol, away from the quad's centre.
        let label = center + away * (CORNER_RING + 13.0);
        painter.text(label + vec2(1.0, 1.0), egui::Align2::CENTER_CENTER, name, egui::FontId::proportional(13.0), Color32::BLACK);
        painter.text(label, egui::Align2::CENTER_CENTER, name, egui::FontId::proportional(13.0), fill);
    }
    if let Some(i) = loupe.filter(|_| !peek) {
        let corner = doc.corners[i];
        let away = vec2((corner[0] - middle[0]) as f32, (corner[1] - middle[1]) as f32).normalized();
        let at = LoupeAt { name: HANDLE_NAMES[i], px: corner, screen: to_screen(corner), away };
        // The two edges of the outline that meet at the corner, as the
        // grid has them, bent through the anchors, or straight to the
        // corners either side where the outline makes no grid.
        let edges: [Vec<[f64; 2]>; 2] = match model.as_ref().filter(|m| m.width == gw && m.height == gh) {
            Some(model) => {
                let (x, y) = [(0, 0), (gw, 0), (gw, gh), (0, gh)][i];
                [(0..=gw).map(|x| model.corner(x, y)).collect(), (0..=gh).map(|y| model.corner(x, y)).collect()]
            }
            None => [vec![doc.corners[i], doc.corners[(i + 1) % 4]], vec![doc.corners[i], doc.corners[(i + 3) % 4]]],
        };
        loupe_window(ui, texture, [photo.width, photo.height], pz.scale, at, &edges);
    }

    // Arrow keys nudge the last corner touched: one photo pixel, ten with
    // Shift, a tenth with Alt.
    if response.contains_pointer() && !ui.ctx().egui_wants_keyboard_input() {
        let nudge = ui.input(|i| {
            let step = if i.modifiers.shift { 10.0 } else if i.modifiers.alt { 0.1 } else { 1.0 };
            let mut delta = [0.0, 0.0];
            for (key, dx, dy) in [
                (egui::Key::ArrowLeft, -1.0, 0.0),
                (egui::Key::ArrowRight, 1.0, 0.0),
                (egui::Key::ArrowUp, 0.0, -1.0),
                (egui::Key::ArrowDown, 0.0, 1.0),
            ] {
                if i.key_pressed(key) {
                    delta[0] += dx * step;
                    delta[1] += dy * step;
                }
            }
            delta
        });
        if nudge != [0.0, 0.0] {
            history.checkpoint(doc);
            doc.corners[view.active_corner][0] += nudge[0];
            doc.corners[view.active_corner][1] += nudge[1];
        }
    }
    view.hover = None;
}

/// A corner handle under the pointer.
struct LoupeAt {
    name: &'static str,
    /// Where the corner is in the photo, and on the screen.
    px: [f64; 2],
    screen: Pos2,
    /// The way out of the outline from the corner, a unit vector.
    away: Vec2,
}

/// Side of the loupe's view, in screen points.
const LOUPE_SIDE: f32 = 176.0;
/// How many times the canvas's own scale the loupe shows the photo at,
/// and the screen points a photo pixel it keeps between.
const LOUPE_ZOOM: f32 = 4.0;
const LOUPE_SCALE: std::ops::RangeInclusive<f32> = 4.0..=32.0;

/// The photo about a corner handle, enlarged, in a small window beside
/// the handle: a corner is placed by where the outline's two edges meet
/// over the symbol's own corner, which on the canvas is a few pixels
/// under the pointer. The edges are drawn as the grid has them, and meet
/// at the middle of the view. `scale` is the canvas's, in screen points a
/// photo pixel; nothing is shown once the canvas is itself zoomed in
/// about as far as the loupe goes.
fn loupe_window(ui: &egui::Ui, texture: egui::TextureId, photo: [usize; 2], scale: f32, at: LoupeAt, edges: &[Vec<[f64; 2]>; 2]) {
    let zoom = (scale * LOUPE_ZOOM).clamp(*LOUPE_SCALE.start(), *LOUPE_SCALE.end());
    if zoom < 1.5 * scale {
        return;
    }
    // Outside the outline first, where it covers nothing of the symbol,
    // then to either side and last over it; never under the pointer,
    // where the window would take the hover from the handle and so close
    // itself.
    let size = vec2(LOUPE_SIDE, LOUPE_SIDE + 22.0);
    let outer = size + Vec2::splat(16.0);
    let screen = ui.ctx().content_rect();
    let pointer = ui.input(|i| i.pointer.latest_pos()).unwrap_or(at.screen);
    let away = if at.away.is_finite() && at.away != Vec2::ZERO { at.away } else { vec2(-1.0, -1.0).normalized() };
    let reach = outer.length() / 2.0 + CORNER_RING + 20.0;
    let place = |direction: Vec2| {
        let rect = Rect::from_center_size(at.screen + direction * reach, outer);
        let shift = vec2(
            (screen.left() - rect.left()).max(0.0) - (rect.right() - screen.right()).max(0.0),
            (screen.top() - rect.top()).max(0.0) - (rect.bottom() - screen.bottom()).max(0.0),
        );
        rect.translate(shift)
    };
    let rect = [away, away.rot90(), -away.rot90(), -away]
        .into_iter()
        .map(place)
        .find(|rect| !rect.expand(12.0).contains(pointer))
        .unwrap_or_else(|| place(away));
    let [px, py] = at.px;
    egui::Window::new("Loupe")
        .id(egui::Id::new("loupe"))
        .title_bar(false)
        .fixed_pos(rect.min)
        .fixed_size(size)
        .resizable(false)
        .collapsible(false)
        .interactable(false)
        .show(ui.ctx(), |ui| {
            ui.label(egui::RichText::new(format!("{} at {px:.1}, {py:.1} · {:.0}× the view", at.name, zoom / scale)).small().weak());
            let (view, _) = ui.allocate_exact_size(Vec2::splat(LOUPE_SIDE), egui::Sense::hover());
            let painter = ui.painter_at(view);
            painter.rect_filled(view, 2.0, ui.visuals().extreme_bg_color);
            let place = |p: [f64; 2]| view.center() + vec2((p[0] - px) as f32, (p[1] - py) as f32) * zoom;
            painter.image(
                texture,
                Rect::from_min_max(place([0.0, 0.0]), place([photo[0] as f64, photo[1] as f64])),
                Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                Color32::WHITE,
            );
            for (width, color) in [(3.5, Color32::from_black_alpha(160)), (1.5, ACCENT)] {
                for edge in edges {
                    painter.add(Shape::line(edge.iter().map(|&p| place(p)).collect(), Line::new(width, color)));
                }
            }
        });
}

/// An anchor handle under the pointer: its site and where it is now.
struct ProfileAt {
    kind: AnchorKind,
    at: [f64; 2],
    px: [f64; 2],
}

/// Samples along each profile trace.
const PROFILE_SAMPLES: usize = 64;
/// How far each trace reaches either side of the handle, in modules.
const PROFILE_SPAN: f64 = 3.0;

/// The luminance along the photo row and the photo column through an
/// anchor handle, three modules either side, in a small window by the
/// pointer: a reading aid like BoofCV's one-dimensional alignment
/// profile, which shows whether the handle sits on the centre of its
/// pattern. The threshold is drawn dashed and the handle as a tick.
fn profile_window(ui: &egui::Ui, luma: &Luma, projection: &Projection, at: ProfileAt, threshold: f32) {
    // The module size at the anchor, from the projection's local scale
    // along each axis, which a warped or foreshortened symbol varies.
    let [mx, my] = at.at;
    let length = |a: [f64; 2], b: [f64; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    let module_x = length(projection.apply(mx + 0.5, my), projection.apply(mx - 0.5, my));
    let module_y = length(projection.apply(mx, my + 0.5), projection.apply(mx, my - 0.5));
    let [px, py] = at.px;
    let trace = |along_row: bool| -> Vec<Option<f32>> {
        let half = PROFILE_SPAN * if along_row { module_x } else { module_y };
        (0..PROFILE_SAMPLES)
            .map(|i| {
                let t = i as f64 / (PROFILE_SAMPLES - 1) as f64 * 2.0 - 1.0;
                if along_row { luma.sample(px + t * half, py) } else { luma.sample(px, py + t * half) }
            })
            .collect()
    };
    let traces = [("Row", trace(true), module_x), ("Column", trace(false), module_y)];
    let here = luma.sample(px, py);

    // Beside the pointer, to its lower right or the other way where that
    // would leave the screen; never under it, where the window would take
    // the hover from the handle and so close itself.
    let size = vec2(272.0, 196.0);
    let gap = 24.0;
    let pointer = ui.input(|i| i.pointer.latest_pos()).unwrap_or(Pos2::ZERO);
    let screen = ui.ctx().content_rect();
    let x = if pointer.x + gap + size.x <= screen.right() { pointer.x + gap } else { pointer.x - gap - size.x };
    let y = if pointer.y + gap + size.y <= screen.bottom() { pointer.y + gap } else { pointer.y - gap - size.y };
    let pos = pos2(x.max(screen.left()), y.max(screen.top()));
    egui::Window::new("Profile")
        .fixed_pos(pos)
        .fixed_size(size)
        .resizable(false)
        .collapsible(false)
        .interactable(false)
        .show(ui.ctx(), |ui| {
            let value = here.map_or("off the photo".to_owned(), |v| format!("luma {v:.2}"));
            ui.weak(format!("{} at ({}, {}) · module ≈ {:.1} px · {value}", at.kind.label(), at.at[0], at.at[1], (module_x + module_y) / 2.0));
            for (name, trace, module) in &traces {
                ui.horizontal(|ui| {
                    ui.add_sized(vec2(48.0, 60.0), egui::Label::new(egui::RichText::new(*name).small()));
                    let (rect, _) = ui.allocate_exact_size(vec2(200.0, 60.0), egui::Sense::hover());
                    let painter = ui.painter_at(rect);
                    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
                    // One tick a module, the handle's own standing out.
                    let faint = Line::new(1.0, Color32::from_gray(128).gamma_multiply(0.5));
                    for m in -(PROFILE_SPAN as i32)..=PROFILE_SPAN as i32 {
                        let x = rect.left() + (m as f32 / PROFILE_SPAN as f32 + 1.0) / 2.0 * rect.width();
                        if m == 0 {
                            painter.line_segment([pos2(x, rect.top()), pos2(x, rect.bottom())], Line::new(1.5, SELECT));
                        } else {
                            painter.line_segment([pos2(x, rect.bottom() - 6.0), pos2(x, rect.bottom())], faint);
                        }
                    }
                    let level = |v: f32| rect.bottom() - v.clamp(0.0, 1.0) * rect.height();
                    painter.extend(Shape::dashed_line(
                        &[pos2(rect.left(), level(threshold)), pos2(rect.right(), level(threshold))],
                        Line::new(1.0, BAD),
                        4.0,
                        3.0,
                    ));
                    // The trace, broken where it leaves the photo.
                    let mut run: Vec<Pos2> = Vec::with_capacity(PROFILE_SAMPLES);
                    let stroke = Line::new(1.5, ACCENT);
                    for (i, sample) in trace.iter().enumerate() {
                        match sample {
                            Some(v) => {
                                let x = rect.left() + i as f32 / (PROFILE_SAMPLES - 1) as f32 * rect.width();
                                run.push(pos2(x, level(*v)));
                            }
                            None => {
                                if run.len() > 1 {
                                    painter.add(Shape::line(std::mem::take(&mut run), stroke));
                                } else {
                                    run.clear();
                                }
                            }
                        }
                    }
                    if run.len() > 1 {
                        painter.add(Shape::line(run, stroke));
                    }
                    ui.weak(format!("{module:.1} px"));
                });
            }
        });
}

fn role_tint(role: Role) -> Option<Color32> {
    Some(match role {
        Role::Finder | Role::FinderSub | Role::CornerFinder => Color32::from_rgb(0, 120, 255),
        Role::Separator => Color32::from_rgb(120, 180, 255),
        Role::Timing => Color32::from_rgb(0, 200, 200),
        Role::Alignment => Color32::from_rgb(150, 90, 255),
        Role::DarkModule => Color32::from_rgb(90, 90, 90),
        Role::Extension => Color32::from_rgb(0, 150, 120),
        Role::Format { .. } => Color32::from_rgb(255, 80, 80),
        Role::Version { .. } => Color32::from_rgb(255, 150, 0),
        Role::Remainder { .. } => Color32::from_rgb(140, 140, 140),
        Role::Data { .. } => return None,
    })
}

/// A distinct, evenly spread hue per block.
fn block_color(block: usize) -> Color32 {
    let hue = (block as f32 * 0.618_034).fract();
    egui::ecolor::Hsva::new(hue, 0.85, 0.95, 1.0).into()
}

fn edit_stage(ui: &mut egui::Ui, app: &mut App, response: &egui::Response, painter: &egui::Painter) {
    let canvas = response.rect;
    let (gw, gh) = (app.derived.func.width, app.derived.func.height);
    let world = Rect::from_min_size(Pos2::ZERO, vec2(gw as f32, gh as f32)).expand(1.5);
    let panning = navigate(ui, response, &mut app.view.edit_view, world);
    let pz = app.view.edit_view;
    let at = |x: f32, y: f32| pz.to_screen(canvas, vec2(x, y));
    let module_rect = |x: usize, y: usize| Rect::from_min_max(at(x as f32, y as f32), at(x as f32 + 1.0, y as f32 + 1.0));
    let px = pz.scale;

    // Only modules inside the canvas are drawn.
    let (low, high) = (pz.to_world(canvas, canvas.min), pz.to_world(canvas, canvas.max));
    let x_range = (low.x.floor().max(0.0) as usize).min(gw)..(high.x.ceil().max(0.0) as usize).min(gw);
    let y_range = (low.y.floor().max(0.0) as usize).min(gh)..(high.y.ceil().max(0.0) as usize).min(gh);

    let typing = ui.ctx().egui_wants_keyboard_input();
    let peek = !typing && ui.input(|i| i.key_down(egui::Key::H));
    let model = app.photo.as_ref().and(app.model.as_ref()).filter(|m| m.width == gw && m.height == gh);
    let layers = match (model, peek) {
        (None, _) => Layers::Grid,
        (Some(_), true) => Layers::Photo,
        (Some(_), false) => app.view.layers,
    };

    // Photo, rectified: one textured quad per module, with texture
    // coordinates taken through the projection at each corner, so a
    // warped symbol is drawn flat.
    if layers != Layers::Grid
        && let (Some(photo), Some(model)) = (&app.photo, model)
    {
        let x0 = (low.x.floor() as i32).max(-MARGIN);
        let x1 = (high.x.ceil() as i32).min(gw as i32 + MARGIN);
        let y0 = (low.y.floor() as i32).max(-MARGIN);
        let y1 = (high.y.ceil() as i32).min(gh as i32 + MARGIN);
        if x1 > x0 && y1 > y0 {
            let cols = (x1 - x0 + 1) as u32;
            let mut mesh = egui::Mesh::with_texture(photo.texture.id());
            let opacity = if layers == Layers::Both { app.view.photo_opacity } else { 1.0 };
            let tint = Color32::from_white_alpha((opacity * 255.0) as u8);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let [u, v] = if x >= 0 && y >= 0 && x as usize <= gw && y as usize <= gh {
                        model.corner(x as usize, y as usize)
                    } else {
                        model.projection.apply(x as f64, y as f64)
                    };
                    mesh.vertices.push(egui::epaint::Vertex {
                        pos: at(x as f32, y as f32),
                        uv: pos2(u as f32 / photo.width as f32, v as f32 / photo.height as f32),
                        color: tint,
                    });
                }
            }
            for row in 0..(y1 - y0) as u32 {
                for col in 0..cols - 1 {
                    let a = row * cols + col;
                    mesh.add_triangle(a, a + 1, a + cols + 1);
                    mesh.add_triangle(a, a + cols + 1, a + cols);
                }
            }
            painter.add(mesh);
        }
    }

    let grid = &app.doc.grid;
    if layers != Layers::Photo {
        let over_photo = layers == Layers::Both;
        let alpha = if over_photo { (app.view.grid_opacity * 255.0) as u8 } else { 255 };
        let unknown = if over_photo {
            UNKNOWN_TINT.gamma_multiply(0.30 * app.view.grid_opacity)
        } else {
            Color32::from_rgb(214, 170, 110)
        };
        let mut mesh = egui::Mesh::default();
        for y in y_range.clone() {
            for x in x_range.clone() {
                let color = match grid.get(x, y) {
                    Module::Dark => Color32::from_black_alpha(alpha),
                    Module::Light => Color32::from_white_alpha(alpha),
                    Module::Unknown => unknown,
                };
                mesh.add_colored_rect(module_rect(x, y), color);
            }
        }
        painter.add(mesh);
    }

    let derived = &app.derived;
    match app.view.overlay {
        Overlay::None => {}
        Overlay::Reliability => {
            // Red where a sampled module sat near its threshold, fading
            // with the margin, and where a module was painted as a guess,
            // which has no margin at all; other painted modules are
            // certain and unknown ones are shown as they are.
            if grid.has_reliability() {
                let mut mesh = egui::Mesh::default();
                for y in y_range.clone() {
                    for x in x_range.clone() {
                        let reliability = grid.reliability(x, y);
                        if reliability >= RELIABILITY_FADE || grid.get(x, y) == Module::Unknown {
                            continue;
                        }
                        let strength = 1.0 - reliability as f32 / RELIABILITY_FADE as f32;
                        mesh.add_colored_rect(module_rect(x, y), BAD.gamma_multiply(0.85 * strength));
                    }
                }
                painter.add(mesh);
            }
        }
        Overlay::Roles => {
            let mut mesh = egui::Mesh::default();
            for y in y_range.clone() {
                for x in x_range.clone() {
                    if let Some(color) = role_tint(derived.func.role(x, y)) {
                        mesh.add_colored_rect(module_rect(x, y), color.gamma_multiply(0.45));
                    }
                }
            }
            painter.add(mesh);
        }
        Overlay::Blocks => {
            let mut mesh = egui::Mesh::default();
            for y in y_range.clone() {
                for x in x_range.clone() {
                    if let Some((cw, _)) = derived.layout.module_codeword(x, y) {
                        let span = derived.layout.codewords[cw];
                        let solved = derived.report.blocks[span.block as usize].is_solved();
                        let strength = if span.is_data { 0.5 } else { 0.28 };
                        mesh.add_colored_rect(module_rect(x, y), block_color(span.block as usize).gamma_multiply(strength));
                        if !solved && px >= 5.0 {
                            let r = module_rect(x, y).shrink(px * 0.38);
                            mesh.add_colored_rect(r, BAD);
                        }
                    }
                }
            }
            painter.add(mesh);
        }
        Overlay::Corrections => {
            // What error correction says each module should have been.
            for (cw, report) in derived.report.codewords.iter().enumerate() {
                let Some(value) = report.final_value else { continue };
                if report.status == CwStatus::Clean {
                    continue;
                }
                let offset = derived.layout.codewords[cw].bit_offset as usize;
                for (bit, pos) in derived.layout.codeword_modules(cw).iter().enumerate() {
                    let (x, y) = (pos.x as usize, pos.y as usize);
                    if !x_range.contains(&x) || !y_range.contains(&y) {
                        continue;
                    }
                    let dark = (value & (0x80 >> (offset + bit)) != 0) != mask_bit(derived.func.version, derived.mask, x, y);
                    let painted = grid.get(x, y).as_dark();
                    if painted == Some(dark) {
                        continue;
                    }
                    let rect = module_rect(x, y);
                    let fill = if dark { Color32::BLACK } else { Color32::WHITE };
                    painter.rect_filled(rect.shrink(px * 0.22), 0.0, fill);
                    let color = if painted.is_none() { GOOD } else { BAD };
                    painter.rect_stroke(rect.shrink(1.0), 0.0, Line::new((px * 0.12).clamp(1.0, 3.0), color), StrokeKind::Inside);
                }
            }
        }
        Overlay::Mask => {
            let mut mesh = egui::Mesh::default();
            for y in y_range.clone() {
                for x in x_range.clone() {
                    let masked = matches!(derived.func.role(x, y), Role::Data { .. } | Role::Remainder { .. });
                    if masked && mask_bit(derived.func.version, derived.mask, x, y) {
                        mesh.add_colored_rect(module_rect(x, y), ACCENT.gamma_multiply(0.5));
                    }
                }
            }
            painter.add(mesh);
        }
        Overlay::Unmasked => {
            // Each data module as the bit it holds: what the codewords
            // are read from, where the grid shows the bit under the mask.
            let mut mesh = egui::Mesh::default();
            for y in y_range.clone() {
                for x in x_range.clone() {
                    if !matches!(derived.func.role(x, y), Role::Data { .. } | Role::Remainder { .. }) {
                        continue;
                    }
                    let Some(dark) = grid.get(x, y).as_dark() else { continue };
                    let one = dark != mask_bit(derived.func.version, derived.mask, x, y);
                    mesh.add_colored_rect(module_rect(x, y), if one { Color32::BLACK } else { Color32::WHITE });
                }
            }
            painter.add(mesh);
        }
        Overlay::Status => {
            let mut mesh = egui::Mesh::default();
            let mut doubted = Vec::new();
            for y in y_range.clone() {
                for x in x_range.clone() {
                    let Some((cw, bit)) = derived.layout.module_codeword(x, y) else { continue };
                    let report = &derived.report.codewords[cw];
                    let tint = match report.status {
                        CwStatus::Clean => None,
                        CwStatus::Corrected | CwStatus::Filled => Some(GOOD),
                        CwStatus::Unverified | CwStatus::Inferred => Some(UNKNOWN_TINT),
                        CwStatus::Unresolved => Some(BAD),
                    };
                    if let Some(tint) = tint {
                        mesh.add_colored_rect(module_rect(x, y), tint.gamma_multiply(0.5));
                    }
                    if report.doubted & (0x80 >> bit) != 0 {
                        doubted.push(module_rect(x, y));
                    }
                }
            }
            painter.add(mesh);
            for rect in doubted {
                painter.rect_stroke(rect, 0.0, Line::new((px * 0.12).clamp(1.5, 3.0), DOUBTED), StrokeKind::Inside);
            }
        }
        Overlay::Segments => {
            // The stream cut up as it parsed: each segment's header and
            // payload, and from the end of the data on, the terminator
            // and the padding. The parts lie in the stream's order.
            let parse = &derived.report.parse;
            let mut parts: Vec<(std::ops::Range<usize>, Color32)> = Vec::new();
            for (i, segment) in parse.segments.iter().enumerate() {
                parts.push((segment.header.clone(), block_color(i).gamma_multiply(0.8)));
                parts.push((segment.payload.clone(), block_color(i).gamma_multiply(0.4)));
            }
            if let EndOfData::At { bit, .. } = parse.end {
                parts.push((bit..derived.layout.data_bits, Color32::from_gray(128).gamma_multiply(0.6)));
            }
            parts.retain(|(bits, _)| !bits.is_empty());
            let mut mesh = egui::Mesh::default();
            for y in y_range.clone() {
                for x in x_range.clone() {
                    let Some((cw, place)) = derived.layout.module_codeword(x, y) else { continue };
                    let Some(index) = derived.layout.stream_index(cw) else { continue };
                    let bit = 8 * index + usize::from(place);
                    let part = parts.partition_point(|(bits, _)| bits.end <= bit);
                    if let Some((_, color)) = parts.get(part).filter(|(bits, _)| bits.contains(&bit)) {
                        mesh.add_colored_rect(module_rect(x, y), *color);
                    }
                }
            }
            painter.add(mesh);
        }
        Overlay::Pins => {
            let mut mesh = egui::Mesh::default();
            for (cw, report) in derived.report.codewords.iter().enumerate() {
                let Some(pin) = report.pin else { continue };
                let color = match pin.source {
                    PinSource::Manual => SELECT,
                    PinSource::Text => GOOD,
                    PinSource::Padding => Color32::from_gray(128),
                    PinSource::Terminator => ACCENT,
                    PinSource::Header => DOUBTED,
                };
                let offset = derived.layout.codewords[cw].bit_offset as usize;
                for (i, pos) in derived.layout.codeword_modules(cw).iter().enumerate() {
                    let (x, y) = (pos.x as usize, pos.y as usize);
                    if pin.mask & (0x80 >> (offset + i)) != 0 && x_range.contains(&x) && y_range.contains(&y) {
                        mesh.add_colored_rect(module_rect(x, y), color.gamma_multiply(0.6));
                    }
                }
            }
            painter.add(mesh);
        }
        Overlay::Photo => {
            // What sampling reads off the photo as things stand, where
            // the grid has it otherwise: what Sample all would change,
            // but that the fixed patterns are shown as the photo has
            // them whether or not they are locked.
            if let Some(preview) = app.preview.as_ref().filter(|p| p.classified.modules.len() == gw * gh) {
                for y in y_range.clone() {
                    for x in x_range.clone() {
                        let (sampled, painted) = (preview.classified.modules[y * gw + x], grid.get(x, y));
                        let Some(dark) = sampled.as_dark().filter(|_| sampled != painted) else { continue };
                        let rect = module_rect(x, y);
                        painter.rect_filled(rect.shrink(px * 0.22), 0.0, if dark { Color32::BLACK } else { Color32::WHITE });
                        let color = if painted == Module::Unknown { GOOD } else { BAD };
                        painter.rect_stroke(rect.shrink(1.0), 0.0, Line::new((px * 0.12).clamp(1.0, 3.0), color), StrokeKind::Inside);
                    }
                }
            }
        }
    }

    if app.view.grid_lines && px >= 6.0 {
        let line = Line::new(1.0, Color32::from_gray(128).gamma_multiply(0.5));
        for x in x_range.start..=x_range.end {
            painter.line_segment([at(x as f32, y_range.start as f32), at(x as f32, y_range.end as f32)], line);
        }
        for y in y_range.start..=y_range.end {
            painter.line_segment([at(x_range.start as f32, y as f32), at(x_range.end as f32, y as f32)], line);
        }
    }
    painter.rect_stroke(Rect::from_min_max(at(0.0, 0.0), at(gw as f32, gh as f32)), 0.0, Line::new(1.5, ACCENT), StrokeKind::Outside);

    // Unresolved codewords of the block the inspector asked about, and
    // the bits its solution was reached by doubting.
    if let Some(block) = app.view.highlight_block {
        for &cw in &derived.layout.blocks[block].codewords {
            let report = &derived.report.codewords[cw];
            let offset = derived.layout.codewords[cw].bit_offset as usize;
            for (i, pos) in derived.layout.codeword_modules(cw).iter().enumerate() {
                let rect = module_rect(pos.x as usize, pos.y as usize);
                if matches!(report.status, CwStatus::Unresolved) {
                    painter.rect_stroke(rect, 0.0, Line::new(2.0, UNKNOWN_TINT), StrokeKind::Inside);
                } else if report.doubted & (0x80 >> (offset + i)) != 0 {
                    painter.rect_stroke(rect, 0.0, Line::new(2.0, DOUBTED), StrokeKind::Inside);
                }
            }
        }
    }

    if let Some(cw) = app.view.selected {
        let offset = derived.layout.codewords[cw].bit_offset as usize;
        for (bit, pos) in derived.layout.codeword_modules(cw).iter().enumerate() {
            let rect = module_rect(pos.x as usize, pos.y as usize);
            painter.rect_stroke(rect, 0.0, Line::new(2.5, SELECT), StrokeKind::Inside);
            if px >= 16.0 {
                let label = (7 - offset - bit).to_string();
                let font = egui::FontId::monospace((px * 0.45).min(14.0));
                painter.text(rect.center() + vec2(1.0, 1.0), egui::Align2::CENTER_CENTER, &label, font.clone(), Color32::BLACK);
                painter.text(rect.center(), egui::Align2::CENTER_CENTER, &label, font, SELECT);
            }
        }
    }

    // The modules of the character picked in the result: one for each bit
    // of the data stream the character is made of.
    if let Some(source) = app.view.picked_char.and_then(|at| derived.report.text.sources.get(at)) {
        for bit in source.bits.clone() {
            let Some((cw, place)) = derived.layout.stream_bit_location(bit) else { continue };
            let span = derived.layout.codewords[cw];
            // A bit the codeword has no module for is fixed by the layout.
            let Some(i) = usize::from(place).checked_sub(usize::from(span.bit_offset)).filter(|&i| i < usize::from(span.nbits)) else {
                continue;
            };
            let pos = derived.layout.codeword_modules(cw)[i];
            painter.rect_stroke(module_rect(pos.x as usize, pos.y as usize), 0.0, Line::new(2.5, PICKED), StrokeKind::Inside);
        }
    }

    // The modules of the segment picked under Segments: its header in the
    // colour of a selection, and what it carries in the colour a picked
    // character's modules have.
    if let Some(segment) = app.view.picked_segment.and_then(|at| derived.report.parse.segments.get(at)) {
        for (bits, color) in [(segment.payload.clone(), PICKED), (segment.header.clone(), SELECT)] {
            for bit in bits {
                let Some((cw, place)) = derived.layout.stream_bit_location(bit) else { continue };
                let span = derived.layout.codewords[cw];
                let Some(i) = usize::from(place).checked_sub(usize::from(span.bit_offset)).filter(|&i| i < usize::from(span.nbits)) else {
                    continue;
                };
                let pos = derived.layout.codeword_modules(cw)[i];
                painter.rect_stroke(module_rect(pos.x as usize, pos.y as usize), 0.0, Line::new(2.0, color), StrokeKind::Inside);
            }
        }
    }

    // Pointer.
    let cell_at = |pos: Pos2| {
        let w = pz.to_world(canvas, pos);
        (w.x.floor() as i32, w.y.floor() as i32)
    };
    let in_grid = |(x, y): (i32, i32)| x >= 0 && y >= 0 && (x as usize) < gw && (y as usize) < gh;
    let hover = response.hover_pos().map(cell_at).filter(|&c| in_grid(c));
    app.view.hover = hover.map(|(x, y)| (x as usize, y as usize));
    if let Some((x, y)) = app.view.hover {
        painter.rect_stroke(module_rect(x, y), 0.0, Line::new(1.5, ACCENT), StrokeKind::Outside);
    }

    let (primary, secondary, shift, pointer) =
        ui.input(|i| (i.pointer.primary_down(), i.pointer.secondary_down(), i.modifiers.shift, i.pointer.interact_pos()));
    let pressed = !panning && response.is_pointer_button_down_on() && (primary || secondary);
    let cell = pointer.map(cell_at);
    let ink = if shift {
        Ink::Module(Module::Unknown)
    } else if primary {
        app.view.ink
    } else {
        Ink::Module(match app.view.ink {
            Ink::Module(Module::Dark) => Module::Light,
            _ => Module::Dark,
        })
    };

    match app.view.tool {
        Tool::Brush => match (pressed, cell) {
            (true, Some(cell)) => {
                let stroke = match app.view.stroke {
                    Some(stroke) => stroke,
                    None => {
                        app.history.checkpoint(&app.doc);
                        if ink == Ink::Photo {
                            app.warn_if_painted_inverted();
                        }
                        Stroke { ink, start: cell, last: cell }
                    }
                };
                // Fill in between frames so fast strokes leave no gaps.
                let (dx, dy) = (cell.0 - stroke.last.0, cell.1 - stroke.last.1);
                let steps = dx.abs().max(dy.abs()).max(1);
                for s in 0..=steps {
                    let x = stroke.last.0 + (dx as f32 * s as f32 / steps as f32).round() as i32;
                    let y = stroke.last.1 + (dy as f32 * s as f32 / steps as f32).round() as i32;
                    app.paint(x, y, stroke.ink);
                }
                app.view.stroke = Some(Stroke { last: cell, ..stroke });
            }
            _ => app.view.stroke = None,
        },
        Tool::Rect => match (pressed, cell) {
            (true, Some(cell)) => {
                let stroke = app.view.stroke.unwrap_or(Stroke { ink, start: cell, last: cell });
                app.view.stroke = Some(Stroke { last: cell, ..stroke });
                let (a, b) = (stroke.start, cell);
                let rect = Rect::from_min_max(
                    at(a.0.min(b.0) as f32, a.1.min(b.1) as f32),
                    at(a.0.max(b.0) as f32 + 1.0, a.1.max(b.1) as f32 + 1.0),
                );
                let fill = match stroke.ink {
                    Ink::Module(Module::Dark) => Color32::from_black_alpha(110),
                    Ink::Module(Module::Light) => Color32::from_white_alpha(110),
                    Ink::Module(Module::Unknown) => UNKNOWN_TINT.gamma_multiply(0.45),
                    Ink::Photo => ACCENT.gamma_multiply(0.25),
                };
                painter.rect_filled(rect, 0.0, fill);
                painter.rect_stroke(rect, 0.0, Line::new(1.5, SELECT), StrokeKind::Inside);
            }
            _ => {
                if let Some(stroke) = app.view.stroke.take() {
                    app.history.checkpoint(&app.doc);
                    let (a, b) = (stroke.start, stroke.last);
                    for y in a.1.min(b.1)..=a.1.max(b.1) {
                        for x in a.0.min(b.0)..=a.0.max(b.0) {
                            app.paint(x, y, stroke.ink);
                        }
                    }
                    if stroke.ink == Ink::Photo {
                        app.warn_if_painted_inverted();
                    }
                }
            }
        },
        Tool::Inspect => {
            app.view.stroke = None;
            if response.clicked()
                && let Some((x, y)) = app.view.hover
            {
                app.view.selected = app.derived.layout.module_codeword(x, y).map(|(cw, _)| cw);
                app.view.reveal_selected = app.view.selected.is_some();
            }
        }
    }
}

/// One line about a module for the status bar.
pub(super) fn describe_module(derived: &Derived, doc: &Document, x: usize, y: usize) -> String {
    let role = derived.func.role(x, y);
    let painted = match doc.grid.get(x, y) {
        Module::Dark => "dark",
        Module::Light => "light",
        Module::Unknown => "unknown",
    };
    let mut text = format!("({x}, {y})  {}  ·  {painted}", role.label());
    let reliability = doc.grid.reliability(x, y);
    if reliability < CERTAIN && doc.grid.get(x, y) != Module::Unknown {
        match reliability {
            // Nothing tells a guess from a module read right on its threshold.
            0 => text.push_str("  ·  a guess, or sampled on its threshold"),
            _ => text.push_str(&format!("  ·  sampled, margin {:.2}", reliability as f32 / (CERTAIN - 1) as f32)),
        }
    }
    match role {
        Role::Format { copy, bit } => text.push_str(&format!("  ·  copy {} bit {bit}", copy + 1)),
        Role::Version { copy, bit } => text.push_str(&format!("  ·  copy {} bit {bit}", copy + 1)),
        Role::Data { .. } => {
            if let Some((cw, bit)) = derived.layout.module_codeword(x, y) {
                let span = derived.layout.codewords[cw];
                let kind = if span.is_data { "data" } else { "EC" };
                let index = if span.is_data {
                    span.index_in_block as usize
                } else {
                    span.index_in_block as usize - derived.layout.blocks[span.block as usize].data_len
                };
                text.push_str(&format!(
                    "  ·  codeword {cw} bit {}  ·  block {} {kind} {index}",
                    7 - bit,
                    span.block + 1
                ));
                if let Some(dark) = doc.grid.get(x, y).as_dark() {
                    let masked = mask_bit(derived.func.version, derived.mask, x, y);
                    text.push_str(&format!("  ·  reads {}", u8::from(dark != masked)));
                }
                if let Some(value) = derived.report.codewords[cw].final_value {
                    text.push_str(&format!("  ·  corrected {}", value >> (7 - bit) & 1));
                }
                if derived.report.codewords[cw].doubted & (0x80 >> bit) != 0 {
                    text.push_str("  ·  doubted to solve the block");
                }
            }
        }
        _ => {
            if let Some(dark) = derived.func.expected(x, y) {
                text.push_str(if dark { "  ·  must be dark" } else { "  ·  must be light" });
            }
        }
    }
    text
}
