//! What the app keeps between frames.

use std::sync::Arc;

use eframe::egui;
use qr_forensics::decode::codewords::Pins;
use qr_forensics::decode::padding::{Filler, PadCandidate};
use qr_forensics::decode::pipeline::{DecodeReport, GridAnalysis, Orientation};
use qr_forensics::decode::sequence::Sequence;
use qr_forensics::imaging::Luma;
use qr_forensics::imaging::adjust::{Adjust, Enlarge};
use qr_forensics::imaging::binarize::BinarizeParams;
use qr_forensics::imaging::detect::{Anchor, AnchorKind, Detection};
use qr_forensics::imaging::projection::{ControlPoint, Projection};
use qr_forensics::imaging::detect::SearchParams;
use qr_forensics::imaging::read::ReadResult;
use qr_forensics::imaging::sample::{Classified, SampleParams, Sharpening};
use qr_forensics::symbol::function_map::FunctionMap;
use qr_forensics::symbol::grid::{CERTAIN, Mirror, Module, ModuleGrid};
use qr_forensics::symbol::layout::{CwId, SymbolLayout};
use qr_forensics::symbol::tables::Mode;
use qr_forensics::symbol::{EcLevel, SymbolVersion};

/// Where an anchor's photo position came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnchorSource {
    /// Found in the photo by the corner search.
    Detected,
    /// Placed by hand.
    Manual,
}

/// A fixed pattern whose place in the photo is known, by detection or by
/// hand, and which the projection passes through.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DocAnchor {
    pub anchor: Anchor,
    pub source: AnchorSource,
}

impl DocAnchor {
    pub fn control_point(&self) -> ControlPoint {
        self.anchor.control_point()
    }
}

/// A hold taken of the grid itself in the Align stage, on a module of it
/// and not on a handle: the anchor that module's middle is, or is to be,
/// and how far it lies from the pointer in the photo, which it keeps
/// while the pointer moves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grasp {
    pub kind: AnchorKind,
    /// The middle of the module, in module coordinates.
    pub at: [f64; 2],
    /// The anchor was set already, and partly damaged.
    pub partial: bool,
    /// The anchor's place in the photo less the pointer's.
    pub lead: [f64; 2],
}

/// Symbols a search of the whole photo found, on offer until the list is
/// cleared, the photo changes, or its lens setting does, which moves the
/// photo under them.
pub struct Proposals {
    pub found: Vec<Detection>,
    /// The photo's Invert setting when the search ran. A detection's
    /// `inverted` says how the symbol read against the photo as it was
    /// then, so adopting it later, after the setting has changed, must
    /// reckon from here.
    pub photo_inverted: bool,
    /// The one of them last picked to put the outline on. While the
    /// outline still lies on it, it is not offered again on the canvas,
    /// where a click meant for the grid would put the outline back as
    /// found, over whatever was placed by hand since.
    pub adopted: Option<usize>,
    /// What Read all made of them, once it has run: each symbol in the
    /// list then says how its read stands, and one that decoded is
    /// loaded into the grid when it is picked, while reads are loaded.
    pub reads: Option<FoundReads>,
}

/// What reading every symbol a search found made of them.
pub struct FoundReads {
    /// A read for each symbol found, in the list's order: `None` for one
    /// that could not be read at all.
    pub results: Vec<Option<ReadResult>>,
    /// The photo's Invert setting when they were read, from which a
    /// read's own polarity is reckoned, as a detection's is from the
    /// setting it was found under.
    pub photo_inverted: bool,
    /// The photo's enlargement when they were read: a read made through
    /// a copy of the photo enlarged again was made at so many times
    /// this, whatever the photo has been set to since.
    pub enlarge: Enlarge,
}

/// The placements of the symbol inside the outline the detector ranks,
/// best first, on offer as the symbols a search found are: the first is
/// what Find in outline tries first, and the others are what it falls
/// back on with nothing said of them.
pub struct Placements {
    pub found: Vec<Detection>,
    /// The photo's Invert setting when they were made, as
    /// [`Proposals::photo_inverted`].
    pub photo_inverted: bool,
    /// The one of them last picked to put the outline on, as
    /// [`Proposals::adopted`]: the placements lie over one another, so
    /// where the outline lies does not say which of them it was put on.
    pub adopted: Option<usize>,
}

/// How the EC level and mask are chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FormatSel {
    /// Whatever the format bits match best.
    Auto,
    Manual { ec: EcLevel, mask: u8 },
}

/// Everything undo and redo restore.
#[derive(Clone, PartialEq)]
pub struct Document {
    pub version: SymbolVersion,
    pub grid: ModuleGrid,
    pub pins: Pins,
    /// The EC level the pins were made under; codeword numbering depends on it.
    pub pins_ec: Option<EcLevel>,
    pub format: FormatSel,
    /// Symbol outline on the photo: top-left, top-right, bottom-right,
    /// bottom-left, in photo pixels. Kept here so that moving a corner is
    /// undoable; meaningless while no photo is loaded.
    pub corners: [[f64; 2]; 4],
    /// Fixed patterns pinned to their places in the photo, which bend the
    /// projection between the corners. Their module coordinates belong to
    /// `version`.
    pub anchors: Vec<DocAnchor>,
    /// How the photo is read: kept here so that a change of adjustments
    /// is undoable. The photo's own copy says what its luminance and
    /// texture were last built under; meaningless while no photo is
    /// loaded.
    pub adjust: Adjust,
}

impl Document {
    pub fn blank(version: SymbolVersion) -> Self {
        Self {
            version,
            grid: FunctionMap::new(version).blank_grid(),
            pins: Pins::new(),
            pins_ec: None,
            format: FormatSel::Auto,
            corners: [[0.0; 2]; 4],
            anchors: Vec::new(),
            adjust: Adjust::default(),
        }
    }

    /// A fresh grid at `version` that keeps the photo outline and
    /// adjustments. Anchors name modules of the old version and do not
    /// carry over.
    pub fn blank_like(&self, version: SymbolVersion) -> Self {
        Self { corners: self.corners, adjust: self.adjust, ..Self::blank(version) }
    }

    /// The anchors as control points of the projection.
    pub fn control_points(&self) -> Vec<ControlPoint> {
        self.anchors.iter().map(DocAnchor::control_point).collect()
    }

    /// The anchor at a site, if one is set.
    pub fn anchor_at(&self, at: [f64; 2]) -> Option<&DocAnchor> {
        self.anchors.iter().find(|a| (a.anchor.at[0] - at[0]).abs() < 1e-6 && (a.anchor.at[1] - at[1]).abs() < 1e-6)
    }

    /// Set or replace the anchor at a site. One replaced keeps its place
    /// among the others, so that nothing which goes by their order, as
    /// the handles of the Align stage are drawn, sees it move.
    pub fn set_anchor(&mut self, anchor: Anchor, source: AnchorSource) {
        let at = anchor.at;
        let set = DocAnchor { anchor, source };
        match self.anchors.iter().position(|a| (a.anchor.at[0] - at[0]).abs() < 1e-6 && (a.anchor.at[1] - at[1]).abs() < 1e-6) {
            Some(i) => self.anchors[i] = set,
            None => self.anchors.push(set),
        }
    }

    pub fn remove_anchor(&mut self, at: [f64; 2]) {
        self.anchors.retain(|a| !((a.anchor.at[0] - at[0]).abs() < 1e-6 && (a.anchor.at[1] - at[1]).abs() < 1e-6));
    }

    /// Whether the two hold the same symbol, whatever photo each lies on:
    /// the version, the modules, the pins and the format.
    fn same_symbol_as(&self, other: &Document) -> bool {
        self.version == other.version
            && self.grid == other.grid
            && self.pins == other.pins
            && self.pins_ec == other.pins_ec
            && self.format == other.format
    }

    /// This document's symbol laid on the photo `other` lies on: with
    /// `other`'s outline and adjustments, and its anchors where they name
    /// modules of the same version.
    fn on_photo_of(self, other: &Document) -> Document {
        let anchors = if self.version == other.version { other.anchors.clone() } else { Vec::new() };
        Document { corners: other.corners, anchors, adjust: other.adjust, ..self }
    }

    /// Module coordinates to photo pixels: the corner homography bent
    /// through the anchors, or the homography alone when `use_anchors` is
    /// off or there are none.
    pub fn projection(&self, width: usize, height: usize, use_anchors: bool) -> Option<Projection> {
        let controls = if use_anchors { self.control_points() } else { Vec::new() };
        Projection::new(width, height, self.corners, &controls)
    }
}

/// Everything the projection depends on.
#[derive(Clone, PartialEq)]
pub struct ModelKey {
    pub version: SymbolVersion,
    /// Width and height of the function map the projection was laid out
    /// for. They follow from the version, and are kept beside it so that
    /// a projection made for a grid of another size can never pass for
    /// the one wanted, whatever the version says.
    pub size: (usize, usize),
    pub corners: [[f64; 2]; 4],
    pub anchors: Vec<DocAnchor>,
    pub use_anchors: bool,
}

/// The projection for the current outline, with the photo position of
/// every module corner worked out once, for drawing and sampling.
pub struct Model {
    pub key: ModelKey,
    pub projection: Arc<Projection>,
    /// `(width + 1) * (height + 1)` points, row-major.
    pub lattice: Arc<Vec<[f64; 2]>>,
    pub width: usize,
    pub height: usize,
}

impl Model {
    /// Photo position of module corner (`x`, `y`), 0..=width, 0..=height.
    pub fn corner(&self, x: usize, y: usize) -> [f64; 2] {
        self.lattice[y * (self.width + 1) + x]
    }

    /// The point of the grid, in modules, drawn at `px` in the photo:
    /// the module whose four corners the point lies among, and how far
    /// across and down it. The grid is drawn straight from one module
    /// corner to the next, and this is the grid as drawn, so that what is
    /// grasped is what was under the pointer. `None` off the grid.
    pub fn module_at(&self, px: [f64; 2]) -> Option<[f64; 2]> {
        for y in 0..self.height {
            for x in 0..self.width {
                let (c00, c10, c11, c01) = (self.corner(x, y), self.corner(x + 1, y), self.corner(x + 1, y + 1), self.corner(x, y + 1));
                if !super::quad_contains(&[c00, c10, c11, c01], px) {
                    continue;
                }
                // Across and down the module by Newton's method on the
                // blend of its corners, which a module near enough a
                // parallelogram settles in a step or two.
                let (mut u, mut v) = (0.5, 0.5);
                for _ in 0..8 {
                    let at = |i: usize| (1.0 - u) * (1.0 - v) * c00[i] + u * (1.0 - v) * c10[i] + u * v * c11[i] + (1.0 - u) * v * c01[i];
                    let du = |i: usize| (1.0 - v) * (c10[i] - c00[i]) + v * (c11[i] - c01[i]);
                    let dv = |i: usize| (1.0 - u) * (c01[i] - c00[i]) + u * (c11[i] - c10[i]);
                    let (ex, ey) = (px[0] - at(0), px[1] - at(1));
                    let det = du(0) * dv(1) - du(1) * dv(0);
                    if det.abs() < 1e-12 {
                        break;
                    }
                    u += (ex * dv(1) - ey * dv(0)) / det;
                    v += (du(0) * ey - du(1) * ex) / det;
                }
                return Some([x as f64 + u.clamp(0.0, 1.0), y as f64 + v.clamp(0.0, 1.0)]);
            }
        }
        None
    }
}

/// One undo point: the document, and which photo it lay on.
struct Snapshot {
    doc: Document,
    photo: u64,
}

impl Snapshot {
    /// Whether going back to this point would change `doc`, which lies
    /// on photo `photo`. A point taken on another photo brings back the
    /// symbol alone, so it changes nothing unless the symbol differs.
    fn changes(&self, doc: &Document, photo: u64) -> bool {
        if self.photo == photo { self.doc != *doc } else { !self.doc.same_symbol_as(doc) }
    }

    /// The document this point brings back in place of `doc`, which lies
    /// on photo `photo`: all of it when it was taken on that photo, and
    /// otherwise its symbol on `doc`'s photo. An outline, anchors and
    /// lens setting are of the photo they were placed on, and mean
    /// nothing on another.
    fn onto(self, doc: &Document, photo: u64) -> Document {
        if self.photo == photo { self.doc } else { self.doc.on_photo_of(doc) }
    }
}

#[derive(Default)]
pub struct History {
    undo: Vec<Snapshot>,
    redo: Vec<Snapshot>,
    /// Counts the photos the document has lain on: see [`Snapshot`].
    photo: u64,
    /// A checkpoint has been taken, and whether the change it was taken
    /// for then happened has not been looked at yet.
    open: bool,
}

impl History {
    const LIMIT: usize = 200;

    /// Record the state before a change. Whether the change then happens
    /// is seen at the next call, of this or of undo or redo: a press that
    /// turns out to change nothing, on a module already that colour or
    /// beside the grid, costs neither the states that could be redone nor
    /// a step of undo.
    pub fn checkpoint(&mut self, doc: &Document) {
        self.close(doc);
        if self.undo.last().is_none_or(|last| last.photo != self.photo || last.doc != *doc) {
            self.undo.push(Snapshot { doc: doc.clone(), photo: self.photo });
            if self.undo.len() > Self::LIMIT {
                self.undo.remove(0);
            }
        }
        self.open = true;
    }

    /// Reckon with the last checkpoint: when the document has changed
    /// since it was taken, the states undone before it are a road not
    /// taken, and can no longer be redone.
    fn close(&mut self, doc: &Document) {
        if std::mem::take(&mut self.open) && self.changed_since_checkpoint(doc) {
            self.redo.clear();
        }
    }

    fn changed_since_checkpoint(&self, doc: &Document) -> bool {
        self.undo.last().is_some_and(|last| last.doc != *doc)
    }

    /// The document is about to be laid on another photo, or on none.
    /// Undo and redo from here on bring back the symbol of an earlier
    /// state, not the outline, anchors and adjustments it had on the
    /// photo it was on.
    pub fn photo_changed(&mut self, doc: &Document) {
        self.close(doc);
        self.photo += 1;
    }

    pub fn can_undo(&self, doc: &Document) -> bool {
        self.undo.iter().rev().any(|point| point.changes(doc, self.photo))
    }

    pub fn can_redo(&self, doc: &Document) -> bool {
        !(self.open && self.changed_since_checkpoint(doc)) && self.redo.iter().rev().any(|point| point.changes(doc, self.photo))
    }

    pub fn undo(&mut self, doc: &mut Document) -> bool {
        self.close(doc);
        // A checkpoint taken for a change that never happened is skipped.
        while let Some(previous) = self.undo.pop() {
            if previous.changes(doc, self.photo) {
                let restored = previous.onto(doc, self.photo);
                self.redo.push(Snapshot { doc: std::mem::replace(doc, restored), photo: self.photo });
                return true;
            }
        }
        false
    }

    pub fn redo(&mut self, doc: &mut Document) -> bool {
        self.close(doc);
        while let Some(next) = self.redo.pop() {
            if next.changes(doc, self.photo) {
                let restored = next.onto(doc, self.photo);
                let current = std::mem::replace(doc, restored);
                if self.undo.last().is_none_or(|last| last.photo != self.photo || last.doc != current) {
                    self.undo.push(Snapshot { doc: current, photo: self.photo });
                }
                return true;
            }
        }
        false
    }
}

/// The part of a photo's pixels as loaded that is on show.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Crop {
    /// Left, top, width and height, in the pixels of the photo as loaded.
    pub rect: [u32; 4],
    /// The lens term the photo was straightened by before it was cut:
    /// the cut is of the straightened photo, whose lens term is then 0,
    /// and the whole photo takes this one back.
    pub k1: f32,
    /// The enlargement the whole photo had when it was cut, which it
    /// takes back. The part starts with the same, and may be taken
    /// further.
    pub enlarge: Enlarge,
}

pub struct Photo {
    pub name: String,
    pub width: usize,
    pub height: usize,
    /// The photo as loaded, with its transparency laid on white: what a
    /// crop is cut from.
    pub source: Arc<image::RgbaImage>,
    /// The part of `source` on show, when it is not the whole.
    pub crop: Option<Crop>,
    /// RGBA of the part on show, at its own size: the photo as loaded
    /// itself, when the whole of it is on show. What the adjustments
    /// are made to, the enlargement among them.
    pub original: Arc<image::RgbaImage>,
    pub adjust: Adjust,
    /// The enlargements on offer for the part on show, least first: the
    /// steps its size allows ([`Enlarge::steps_for`]), and any other it
    /// has been set to since, by a read that decoded only through a
    /// copy of it so enlarged.
    pub enlargements: Vec<Enlarge>,
    /// What sampling reads: the adjusted image as luminance. Shared, so
    /// that a search or a job on its own thread starts without copying
    /// the photo; a change of adjustments makes a new one.
    pub luma: Arc<Luma>,
    pub texture: egui::TextureHandle,
    /// Whether the texture shows the photo in grey, as the luminance
    /// has it, or in its colours: what the view asked for when it was
    /// last made.
    pub greyscale: bool,
    /// The binarised view, built when first shown and kept until the
    /// luminance or the settings change.
    pub binarised: Option<Binarised>,
}

/// The adaptive threshold of the photo as a texture, with what it was
/// made from. It holds the luminance it was built from, so that a new one
/// cannot take the old one's address and pass for it.
pub struct Binarised {
    pub source: Arc<Luma>,
    pub params: BinarizeParams,
    pub texture: egui::TextureHandle,
}

impl Binarised {
    /// Whether this is the threshold of `luma` under `params`.
    pub fn is_for(&self, luma: &Arc<Luma>, params: &BinarizeParams) -> bool {
        Arc::ptr_eq(&self.source, luma) && self.params == *params
    }
}

impl Photo {
    /// Put `enlarge` among the enlargements on offer, where it is not
    /// one of them already.
    pub fn offer(&mut self, enlarge: Enlarge) {
        if let Err(at) = self.enlargements.binary_search(&enlarge) {
            self.enlargements.insert(at, enlarge);
        }
    }

    /// Where a point of the photo on show, in its working pixels, lies in
    /// the pixels of the photo as loaded.
    pub fn to_source(&self, point: [f64; 2]) -> [f64; 2] {
        let [x, y, w, h] = self.shown();
        [x + point[0] * w / self.width as f64, y + point[1] * h / self.height as f64]
    }

    /// The other way about from [`Self::to_source`]: where a point of the
    /// photo as loaded lies in the working pixels of the photo on show.
    pub fn to_working(&self, point: [f64; 2]) -> [f64; 2] {
        let [x, y, w, h] = self.shown();
        [(point[0] - x) * self.width as f64 / w, (point[1] - y) * self.height as f64 / h]
    }

    /// Pixels of the photo on show to a pixel of the photo as loaded:
    /// what the Enlarge adjustment makes of it, one for a photo at its
    /// own size.
    pub fn scale(&self) -> f64 {
        self.width as f64 / self.shown()[2]
    }

    /// The part of the photo as loaded that is on show: left, top, width
    /// and height.
    fn shown(&self) -> [f64; 4] {
        match self.crop {
            Some(crop) => crop.rect.map(f64::from),
            None => [0.0, 0.0, f64::from(self.source.width()), f64::from(self.source.height())],
        }
    }

    /// A square outline in the middle of the photo, to start from.
    pub fn default_corners(&self) -> [[f64; 2]; 4] {
        let (w, h) = (self.width as f64, self.height as f64);
        let side = 0.7 * w.min(h);
        let (x0, y0) = ((w - side) / 2.0, (h - side) / 2.0);
        [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]]
    }
}

/// Result of sampling the photo with the current corners and settings.
pub struct SamplePreview {
    pub key: SampleKey,
    pub classified: Classified,
    /// (matching, known) against the version's fixed patterns.
    pub fixed: (usize, usize),
    /// How far the modules were sharpened against their neighbours once
    /// they were read.
    pub sharpening: Sharpening,
    /// The turn or reflection under which the fixed patterns, as sampled,
    /// fit the grid, when they fit it poorly as it lies: a sampling of
    /// the whole grid then leaves them as sampled, locked or not, so that
    /// the grid shows it.
    pub turned: Option<Orientation>,
}

/// Everything the sample preview depends on.
#[derive(Clone, PartialEq)]
pub struct SampleKey {
    pub model: ModelKey,
    pub adjust: Adjust,
    pub params: SampleParams,
}

/// Values computed from the document, refreshed whenever it changes.
pub struct Derived {
    pub func: Arc<FunctionMap>,
    pub analysis: GridAnalysis,
    /// The EC level and mask in effect.
    pub ec: EcLevel,
    pub mask: u8,
    pub layout: Arc<SymbolLayout>,
    pub report: DecodeReport,
    pub padding: Option<PadCandidate>,
    /// The EC level and mask came from brute force because the format bits
    /// could not settle them.
    pub format_by_brute: bool,
    /// Nothing decoded as painted, but the grid's mirror image did: the
    /// symbol was seen from the back and its format bits could not say so.
    pub mirror_by_decode: Option<Mirror>,
}

/// A line for the status bar.
pub struct Notice {
    pub text: String,
    pub is_error: bool,
    pub at: std::time::Instant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Photo coordinates: place the corners.
    Align,
    /// Module coordinates: paint and inspect.
    Edit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    Brush,
    Rect,
    Inspect,
}

/// What the Brush and the Rectangle put into a module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ink {
    /// A colour, or unknown.
    Module(Module),
    /// What sampling reads off the photo there, as things stand, with no
    /// unsure band: the ink is put where a colour is wanted, so a module
    /// near its threshold is the colour it falls on, at the reliability
    /// it read at, where a sampling of the grid would leave it unknown.
    Photo,
    /// The colour left as it is, and how far it is to be believed set to
    /// the painted reliability ([`ViewState::paint_reliability`]): for a
    /// region a stain or a tear runs over, whose modules can be read but
    /// not trusted as far as the rest, so that the decoder doubts them
    /// first. Nothing on an unknown module.
    Doubt,
    /// The colour left as it is and made certain again: what the other
    /// button paints while the ink is Doubtful.
    Sure,
}

/// The reliability the Doubtful ink is given when it is picked with the
/// painted reliability at certain: three bits of likelihood, one module
/// in eight misread, as a module beside a located error is held to.
pub const DOUBT_RELIABILITY: u8 = 16;

/// The most a module painted by hand may be believed short of certain:
/// eight bits of likelihood, one module in two hundred and fifty-six
/// misread. The painted reliability slider runs from a guess to here,
/// and one step beyond to certain; a module believed further than this
/// and still open to doubt is a nicety the decoder never needs, since
/// nothing it doubts by reliability reaches so far down the order.
pub const PAINT_RELIABILITY_MAX: u8 = 56;

/// Which modules a sampling of the photo writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleScope {
    All,
    /// The modules of one block's codewords.
    Block(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Overlay {
    None,
    Roles,
    Blocks,
    Corrections,
    /// How far each sampled module was from its threshold.
    Reliability,
    /// Where the mask in effect swaps dark and light.
    Mask,
    /// The data modules with the mask taken off: the bits themselves.
    Unmasked,
    /// How each codeword stands after error correction.
    Status,
    /// Which segment of the message each data module belongs to.
    Segments,
    /// The modules of pinned bits, by where the pin came from.
    Pins,
    /// Where the grid differs from what sampling reads off the photo.
    Photo,
}

impl Overlay {
    pub const ALL: [Overlay; 11] = [
        Overlay::None,
        Overlay::Roles,
        Overlay::Blocks,
        Overlay::Corrections,
        Overlay::Reliability,
        Overlay::Mask,
        Overlay::Unmasked,
        Overlay::Status,
        Overlay::Segments,
        Overlay::Pins,
        Overlay::Photo,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Overlay::None => "None",
            Overlay::Roles => "Module roles",
            Overlay::Blocks => "RS blocks",
            Overlay::Corrections => "Corrections",
            Overlay::Reliability => "Reliability",
            Overlay::Mask => "Mask pattern",
            Overlay::Unmasked => "Unmasked data",
            Overlay::Status => "Codeword status",
            Overlay::Segments => "Segments",
            Overlay::Pins => "Pins",
            Overlay::Photo => "Differs from photo",
        }
    }

    /// What the overlay's colours mean, for the toolbar; nothing where
    /// the name says it all or the colours are too many to list.
    pub fn legend(self) -> &'static str {
        match self {
            Overlay::None | Overlay::Roles | Overlay::Blocks | Overlay::Reliability => "",
            Overlay::Corrections => "Each module as error correction has it: green rim where it was unknown, red where it was painted otherwise",
            Overlay::Mask => "Blue: the mask swaps dark and light there",
            Overlay::Unmasked => "Data modules as their bits read once the mask is taken off: dark for 1, light for 0",
            Overlay::Status => "Green: corrected or filled in · amber: read, unconfirmed · red: unknown · violet rim: doubted to solve the block",
            Overlay::Segments => "A hue to each segment, its header the stronger · grey: terminator and padding",
            Overlay::Pins => "Yellow: by hand · green: known text · grey: padding · blue: terminator · violet: header",
            Overlay::Photo => "Red rim: the photo reads the other colour · green rim: the photo reads a module left unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlignOverlay {
    FixedPatterns,
    SamplePreview,
    LinesOnly,
    /// The photo thresholded as a finder scan sees it, under the lines.
    Binarised,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layers {
    Both,
    Photo,
    Grid,
}

/// How the decoded message is shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResultMode {
    /// Decoded text, with unreadable characters marked.
    Text,
    /// Text with control characters spelt out.
    Escaped,
    /// The message's bytes in hexadecimal.
    Hex,
    /// Text with ECI designators written in.
    Eci,
    /// GS1 element strings, each application identifier in parentheses.
    Gs1,
}

impl ResultMode {
    pub const ALL: [ResultMode; 5] =
        [ResultMode::Text, ResultMode::Escaped, ResultMode::Hex, ResultMode::Eci, ResultMode::Gs1];

    pub fn label(self) -> &'static str {
        match self {
            ResultMode::Text => "Text",
            ResultMode::Escaped => "Escaped",
            ResultMode::Hex => "Hex",
            ResultMode::Eci => "ECI",
            ResultMode::Gs1 => "GS1",
        }
    }
}

/// Pan and zoom of one canvas stage.
#[derive(Clone, Copy, Debug)]
pub struct PanZoom {
    /// World point shown at the centre of the canvas.
    pub center: egui::Vec2,
    /// Screen points per world unit.
    pub scale: f32,
    /// Keep the content fitted to the canvas, following window resizes,
    /// until the user pans or zooms.
    pub auto_fit: bool,
}

impl Default for PanZoom {
    fn default() -> Self {
        Self { center: egui::Vec2::ZERO, scale: 1.0, auto_fit: true }
    }
}

impl PanZoom {
    pub fn to_screen(self, canvas: egui::Rect, world: egui::Vec2) -> egui::Pos2 {
        canvas.center() + (world - self.center) * self.scale
    }

    pub fn to_world(self, canvas: egui::Rect, screen: egui::Pos2) -> egui::Vec2 {
        self.center + (screen - canvas.center()) / self.scale
    }

    pub fn fit(&mut self, canvas: egui::Rect, world: egui::Rect) {
        let scale = (canvas.width() / world.width()).min(canvas.height() / world.height()) * 0.92;
        if scale.is_finite() && scale > 0.0 {
            self.scale = scale;
            self.center = world.center().to_vec2();
        }
    }

    /// Zoom by `factor`, keeping the world point under `anchor` in place.
    pub fn zoom_about(&mut self, canvas: egui::Rect, anchor: egui::Pos2, factor: f32) {
        let before = self.to_world(canvas, anchor);
        self.scale = (self.scale * factor).clamp(0.01, 400.0);
        self.center += before - self.to_world(canvas, anchor);
    }
}

/// The lists that scroll inside the inspector, which scrolls itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InnerList {
    ResultText,
    Blocks,
    Codewords,
    SequenceText,
}

impl InnerList {
    const ALL: [InnerList; 4] = [InnerList::ResultText, InnerList::Blocks, InnerList::Codewords, InnerList::SequenceText];
}

/// Which of the inspector's lists the wheel belongs to, if any. A list
/// takes the wheel once the pointer has been moved onto it. One that
/// slides under a pointer at rest, as the inspector scrolls, does not:
/// it would stop the inspector there, with every turn of the wheel going
/// to the list until its end, and the rest of the inspector out of reach.
#[derive(Default)]
pub struct WheelLatch {
    /// Where each list lay on the last frame, by [`InnerList`]'s order.
    rects: [Option<egui::Rect>; 4],
    owner: Option<InnerList>,
    /// Where the pointer was when `owner` was last settled.
    settled_at: Option<egui::Pos2>,
}

impl WheelLatch {
    /// How far the pointer must move, in points, to count as moved: more
    /// than turning a mouse's wheel jogs it.
    const MOVED: f32 = 6.0;

    /// Take in where the pointer is, at the start of a frame, before the
    /// lists are laid out again: one that is not, its section closed, is
    /// no longer anywhere.
    pub fn follow(&mut self, pointer: Option<egui::Pos2>) {
        let rects = std::mem::take(&mut self.rects);
        let Some(pointer) = pointer else { return };
        if self.settled_at.is_some_and(|at| at.distance(pointer) <= Self::MOVED) {
            return;
        }
        self.settled_at = Some(pointer);
        self.owner = InnerList::ALL.into_iter().find(|&list| rects[list as usize].is_some_and(|rect| rect.contains(pointer)));
    }

    /// Whether `list` has the wheel.
    pub fn owns(&self, list: InnerList) -> bool {
        self.owner == Some(list)
    }

    /// Record where `list` was laid out this frame.
    pub fn place(&mut self, list: InnerList, rect: egui::Rect) {
        self.rects[list as usize] = Some(rect);
    }

    /// Where `list` was laid out this frame, if it was.
    #[cfg(test)]
    pub fn placed(&self, list: InnerList) -> Option<egui::Rect> {
        self.rects[list as usize]
    }
}

/// A brush stroke or rectangle drag in progress.
#[derive(Clone, Copy, Debug)]
pub struct Stroke {
    pub ink: Ink,
    pub start: (i32, i32),
    pub last: (i32, i32),
}

pub struct ViewState {
    pub stage: Stage,
    pub tool: Tool,
    pub ink: Ink,
    pub overlay: Overlay,
    pub align_overlay: AlignOverlay,
    pub layers: Layers,
    pub photo_opacity: f32,
    pub grid_opacity: f32,
    pub grid_lines: bool,
    /// Keep fixed patterns at their specified colours.
    pub lock_fixed: bool,
    /// How far a module painted dark or light is to be believed, on the
    /// decoder's scale: [`CERTAIN`] for a module that is never doubted,
    /// nought for a guess no surer than a module sampled on its
    /// threshold, and anything between, up to [`PAINT_RELIABILITY_MAX`]
    /// from the slider. What the From photo ink reads goes in at this
    /// reliability too, below certain, in place of the one it read at;
    /// and the Doubtful ink sets it on a module without changing its
    /// colour.
    pub paint_reliability: u8,
    pub align_view: PanZoom,
    pub edit_view: PanZoom,
    pub selected: Option<CwId>,
    /// Scroll the codeword table to the selection on the next frame.
    pub reveal_selected: bool,
    /// Block whose unresolved codewords, or the bits doubted to solve it,
    /// are outlined on the canvas.
    pub highlight_block: Option<usize>,
    pub hover: Option<(usize, usize)>,
    pub active_corner: usize,
    pub sample: SampleParams,
    /// How Search whole photo searches the photo.
    pub search: SearchParams,
    pub stroke: Option<Stroke>,
    /// Codeword table filter: a block index, or everything.
    pub table_block: Option<usize>,
    pub pin_text: String,
    /// Bend the projection through the anchors.
    pub use_anchors: bool,
    /// When the fixed patterns read inverted at sampling time, invert the
    /// photo before writing the grid, as a reader retries inverted of its
    /// own accord.
    pub auto_invert: bool,
    /// Draw anchor handles in the Align stage.
    pub show_anchors: bool,
    /// The anchor a drag begun on the grid itself is pulling about,
    /// while the drag lasts.
    pub grasped: Option<Grasp>,
    pub result_mode: ResultMode,
    /// Parts of a structured append message collected so far.
    pub sequence: Sequence,
    /// Settings of the binarised view: the scan's own rule for the
    /// outline while `binarize_auto` holds, and whatever was set since,
    /// over the outline's region either way ([`ViewState::binarize_for`]).
    pub binarize: BinarizeParams,
    /// The binarised view follows the outline, as the scan does.
    pub binarize_auto: bool,
    /// The scan reads the photo against the binarised view's threshold
    /// and nothing else, in place of the three readings it pools: Find
    /// in outline, Other placements, Search whole photo and the reads of
    /// what they find take it from the view's settings as they stand
    /// ([`ViewState::binarize_for`]) when they start. Set with those
    /// settings, which stay on show while it holds.
    pub binarize_for_scan: bool,
    /// The pointer press on a photo adjustment control under way has
    /// taken its undo point: a slider drag changes the value every frame
    /// and must make one step of undo, not one a frame.
    pub adjust_gesture: bool,
    /// Show the photo in grey, as the luminance the pipeline reads, in
    /// place of its colours. A way of looking: what is read is the same
    /// either way.
    pub greyscale: bool,
    /// Which of the inspector's lists the wheel scrolls.
    pub wheel: WheelLatch,
    /// Search the photo, and read an outline, for symbols of the family
    /// in use alone. The other families are searched from the finder
    /// patterns the first leaves over, and on a damaged QR Code that is
    /// most of the time a search takes.
    pub own_family_only: bool,
    /// The character of the message, counted from 0, whose modules are
    /// outlined on the grid: the one last clicked in the result.
    pub picked_char: Option<usize>,
    /// The segment of the message whose modules are outlined on the grid:
    /// the one last clicked under Segments.
    pub picked_segment: Option<usize>,
    /// Lists the inspector and the tools cut short, shown whole.
    pub all_format_candidates: bool,
    pub all_attempts: bool,
    pub all_guesses: bool,
    /// Where padding set by hand starts, a message byte counted from 0,
    /// and what it is. The start follows what the decode makes of the
    /// padding until it is set here, and again once it has been pinned
    /// from, or the grid is another symbol's: a start given for one
    /// symbol says nothing of the next.
    pub pad_from: Option<usize>,
    pub pad_filler: Filler,
    /// The placement in the list under Alignment the pointer is on, whose
    /// outline the canvas shows.
    pub hover_placement: Option<usize>,
    /// Load a read that decodes into the grid of itself: what Find in
    /// outline reads inside the outline, what is read of a symbol found
    /// or of a placement when it is picked, and of the symbol a search
    /// of the whole photo or Read all picks of itself. A read that does
    /// not decode is never loaded.
    pub load_reads: bool,
    /// Text the user knows the message to hold, to be pinned.
    pub known_text: String,
    /// Where it stands: at the start of the message under a header of
    /// `known_mode`, or from character `known_at` of the text as read.
    pub known_from_start: bool,
    pub known_at: usize,
    pub known_mode: Mode,
    /// The stretch of the message as read that Pin as ASCII asserts is
    /// seven-bit: its first and last characters, counted from one, and
    /// nought until they are given. A pin asserts, so the stretch is
    /// named rather than taken to be the whole message.
    pub ascii_from: usize,
    pub ascii_to: usize,
    /// A press of the paste shortcut has already been answered: the
    /// release that follows it is not a second request.
    pub pasted_on_press: bool,
}

impl ViewState {
    /// The characters Pin as ASCII would assert, counted from nought, in
    /// a message of `length` characters: `None` until a stretch within
    /// the message is given.
    pub fn ascii_range(&self, length: usize) -> Option<std::ops::Range<usize>> {
        (self.ascii_from >= 1 && self.ascii_from <= self.ascii_to && self.ascii_to <= length).then(|| self.ascii_from - 1..self.ascii_to)
    }
}

impl ViewState {
    /// The binarised view's settings as they stand for an outline
    /// `corners` in a photo `width` by `height`: the scan's own for that
    /// outline while they are automatic, and whatever was set otherwise,
    /// over that outline's region all the same: the scan takes the level
    /// and the floor of a reading from the outline it is given, and
    /// settings made by hand once kept the region of the outline they
    /// were made on, so that the view showed a level the scan did not
    /// read by once the outline had moved.
    pub fn binarize_for(&self, corners: [[f64; 2]; 4], width: usize, height: usize) -> BinarizeParams {
        let auto = BinarizeParams::for_outline(corners, width, height);
        if self.binarize_auto { auto } else { BinarizeParams { region: auto.region, ..self.binarize } }
    }
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            stage: Stage::Edit,
            tool: Tool::Brush,
            ink: Ink::Module(Module::Dark),
            overlay: Overlay::None,
            align_overlay: AlignOverlay::FixedPatterns,
            layers: Layers::Both,
            photo_opacity: 1.0,
            grid_opacity: 0.6,
            grid_lines: true,
            lock_fixed: true,
            paint_reliability: CERTAIN,
            align_view: PanZoom::default(),
            edit_view: PanZoom::default(),
            selected: None,
            reveal_selected: false,
            highlight_block: None,
            hover: None,
            active_corner: 0,
            sample: SampleParams::default(),
            search: SearchParams::default(),
            stroke: None,
            table_block: None,
            pin_text: String::new(),
            use_anchors: true,
            auto_invert: true,
            show_anchors: true,
            grasped: None,
            result_mode: ResultMode::Text,
            sequence: Sequence::default(),
            binarize: BinarizeParams::default(),
            binarize_auto: true,
            binarize_for_scan: false,
            adjust_gesture: false,
            greyscale: false,
            wheel: WheelLatch::default(),
            own_family_only: false,
            picked_char: None,
            picked_segment: None,
            all_format_candidates: false,
            all_attempts: false,
            all_guesses: false,
            pad_from: None,
            pad_filler: Filler::Standard,
            hover_placement: None,
            load_reads: true,
            known_text: String::new(),
            known_from_start: false,
            known_at: 0,
            known_mode: Mode::Byte,
            ascii_from: 0,
            ascii_to: 0,
            pasted_on_press: false,
        }
    }
}
