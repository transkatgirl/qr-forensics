//! The application window: owns the document, applies edits, and lays out
//! the panels. Panels describe what the user asked for as [`Action`]s, which
//! are applied after the frame's widgets have been drawn.

mod canvas;
mod inspector;
mod screen;
#[cfg(feature = "selftest")]
mod selftest;
mod state;
mod tools_panel;

use std::cell::RefCell;
use std::path::Path;
use std::sync::Arc;

use eframe::egui;
use qr_forensics::decode::codewords::{Pin, PinSource, Pins};
use qr_forensics::decode::known::{Place, lay_over, text_pins, try_beginnings};
use qr_forensics::decode::list::{LIST_COST, hard_reach, usable_cost};
use qr_forensics::decode::padding::{Filler, clear_padding_pins, detect_padding, pin_padding_at, recover_padding};
use qr_forensics::decode::pipeline::{
    Attempt, BlockOutcome, BlockReport, DecodeOptions, Harder, analyze, brute_force_mirrored_whole, brute_force_whole, decode, decode_keeping,
    decode_trying_harder, formats, judged_verifying_bits, orientation_of, verifying_bits,
};
use qr_forensics::decode::repair::{quiet_zone, render_rgba, render_svg, repaired_grid};
use qr_forensics::decode::rs::{TrialReach, trial_reach};
use qr_forensics::decode::sequence::Part;
use qr_forensics::decode::text::Charset;
use qr_forensics::imaging::adjust::{Adjust, Backdrop, Enlarge, Filter, LENS_MIN_PAIRS, estimate_k1, remap_point, straightened};
use qr_forensics::imaging::photo::{crop_box, flatten_alpha_onto, has_transparency};
use qr_forensics::imaging::resize::resize;
use qr_forensics::imaging::detect::{
    Anchor, AnchorKind, AnchorSite, Detection, Lead, SEARCH_SIDE, SearchParams, anchor_sites, decodes_as_found,
    bled_outline, bloom_of, detect_candidates, refine_anchors, refine_anchors_with, search_for_families, settle_loose_corners,
};
use qr_forensics::imaging::read::{ReadResult, read_detection, read_outline, read_outline_one_of, settle_unvouched, standing_of};
use qr_forensics::imaging::refine::{Refinement, placement_agreement, refine_corners, refine_corners_expecting};
use qr_forensics::imaging::gpu::Shown;
use qr_forensics::imaging::{Luma, SharedRgba};
use qr_forensics::imaging::deblur;
use qr_forensics::imaging::enhance::{Enhance, Enhanced, Sharpen};
use qr_forensics::imaging::sample::{
    Classified, Kernel, Rule, SampleParams, VersionGuess, read_modules, suggest_versions,
};
use qr_forensics::symbol::function_map::{FunctionMap, looks_inverted};
use qr_forensics::symbol::grid::{CERTAIN, Mirror, Module, ModuleGrid, Symmetry};
use qr_forensics::symbol::layout::{CwId, SymbolLayout};
use qr_forensics::symbol::tables::{block_shape, misdecode_protection};
use qr_forensics::symbol::{EcLevel, Family, SymbolVersion};
use qr_forensics::work::{self, Cancel};
use state::*;

/// Pixels per module in exported images.
const EXPORT_SCALE: usize = 10;
/// How long a notice that is not an error stays in the status bar.
const NOTICE_SECONDS: f32 = 12.0;

const UNDO: egui::KeyboardShortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Z);
const REDO: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND.plus(egui::Modifiers::SHIFT), egui::Key::Z);
const OPEN: egui::KeyboardShortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::O);
/// Paste, as the menu shows it. The key itself never reaches the window
/// as a shortcut: see [`App::shortcuts`].
const PASTE: egui::KeyboardShortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::V);

/// What the repaired symbol is saved as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExportFormat {
    /// A bitmap, [`EXPORT_SCALE`] pixels a module.
    Png,
    /// A drawing, which scales to any size without blur.
    Svg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Transform {
    RotateCw,
    Rotate180,
    Mirror(Mirror),
    Invert,
    /// Any turn or reflection of the grid's rectangle.
    Symmetry(Symmetry),
    /// A turn or reflection with dark and light swapped as well, as one
    /// edit, so that one Undo takes both back.
    SymmetryInverted(Symmetry),
}

impl Transform {
    /// The turn or reflection of the grid's rectangle the transform
    /// makes, if it makes one: what the outline and the anchors must
    /// follow, since they say where the grid's modules lie in the photo.
    fn symmetry(self) -> Option<Symmetry> {
        match self {
            Transform::RotateCw => Some(Symmetry::RotateCw),
            Transform::Rotate180 => Some(Symmetry::Rotate180),
            Transform::Mirror(Mirror::Transpose) => Some(Symmetry::Transpose),
            Transform::Mirror(Mirror::Horizontal) => Some(Symmetry::FlipHorizontal),
            Transform::Mirror(Mirror::Vertical) => Some(Symmetry::FlipVertical),
            Transform::Invert => None,
            Transform::Symmetry(symmetry) | Transform::SymmetryInverted(symmetry) => Some(symmetry),
        }
    }
}

/// Something the user asked for during a frame.
pub(crate) enum Action {
    OpenImage,
    PasteImage,
    ClosePhoto,
    NewBlank,
    SetVersion(SymbolVersion),
    /// Read the grid as painted as a symbol of `version`, the same size
    /// as the current one: a QR Code grid as Model 1, or the reverse.
    ReinterpretAs(SymbolVersion),
    Undo,
    Redo,
    FillFixed,
    ClearData,
    Transform(Transform),
    /// Write what the photo shows into the grid: the whole of it, or the
    /// modules of `scope` alone.
    Sample { only_unknown: bool, scope: SampleScope },
    SuggestVersions,
    /// Take the versions Suggest version ranked off the list.
    ClearGuesses,
    ResetCorners,
    /// Change how the photo is read. The control that asks records the
    /// undo point first, once per gesture, since a slider drag asks on
    /// every frame.
    SetAdjust(Adjust),
    /// Estimate the lens term from the corners and anchors and apply it,
    /// moving them with the photo. Only ever at the user's asking.
    EstimateLens,
    /// Measure the blur inside the outline for the Deblur filter to undo.
    MeasureDeblur,
    SetFormat(FormatSel),
    BruteForce,
    /// Search a block again with every dear solver's budget at its most,
    /// on a thread of its own, and keep what it finds for as long as the
    /// block reads the same ([`KeptHarder`]).
    TryHarder(usize),
    Pin(CwId, u8),
    /// Pin the bits of a codeword that `mask` names at their values in
    /// `value`, in place of whatever pin it had; no bits, and it has none.
    PinBits { cw: CwId, mask: u8, value: u8 },
    Unpin(CwId),
    ApplyPadding,
    /// Pin the padding from the message byte the user gives, as the
    /// filler the user names.
    PinPaddingAt { first: usize, filler: Filler },
    ClearPaddingPins,
    ClearAllPins,
    SetCharset(Charset),
    /// Change which solvers a block that fails is tried with.
    SetDecodeOptions(DecodeOptions),
    /// Find the finder patterns inside the outline and move the corners
    /// onto the symbol.
    FindCorners,
    /// Search the whole photo for symbols of every family, with no
    /// outline drawn, and list what is found.
    FindSymbols,
    /// Adopt one of the symbols found: its version, outline and anchors,
    /// and, where Read all read it through to a message and reads are
    /// loaded, its grid as read with the pins it was read under.
    UseProposal(usize),
    /// List every placement of the symbol inside the outline the detector
    /// ranks, the runners-up with the best.
    ListPlacements,
    /// Adopt one of those placements, as a symbol found is adopted.
    UsePlacement(usize),
    ClearPlacements,
    /// Put the outline about finder patterns the search could fit no
    /// symbol to.
    UseLead(usize),
    ClearProposals,
    /// Call off the search of the whole photo that is under way.
    CancelSearch,
    /// Call off the step of the alignment that is under way.
    CancelJob,
    /// Read every symbol the search found through to its message.
    ReadFound,
    /// Pin the text the user knows, at the place the inspector gives.
    PinText,
    /// Try the ways messages commonly begin and pin the one that error
    /// correction bears out.
    TryBeginnings,
    /// Remove the pins made from known text.
    ClearTextPins,
    /// Cut the photo down to the outline, from its pixels as loaded.
    CropToOutline,
    /// Show the whole photo again after a crop.
    ShowWholePhoto,
    /// Refine the corners and every anchor against the photo.
    RefineCorners,
    ClearAnchors,
    /// Add the current decode to the structured append sequence.
    AddToSequence,
    RemoveFromSequence(u8),
    ClearSequence,
    /// Save the message's bytes, raw, to a file.
    SaveBytes,
    /// Write every corrected module and regenerated function pattern into
    /// the grid.
    ApplyCorrections,
    /// Save the repaired symbol as an image.
    ExportImage(ExportFormat),
    DismissNotice,
    FitView,
    Quit,
}

pub(crate) struct BruteResults {
    pub attempts: Vec<Attempt>,
    /// The document has changed since these were computed.
    pub stale: bool,
}

/// Blocks searched harder at the user's asking ([`Harder`]), with the
/// version, level, mask and decode options they were searched under:
/// a decode under those takes them, where a block reads as it did, and
/// a decode under any other does not ([`Self::holds`]).
struct KeptHarder {
    version: SymbolVersion,
    ec: EcLevel,
    mask: u8,
    opts: DecodeOptions,
    searched: Harder,
}

impl KeptHarder {
    /// The searches hold under these, the character set aside: a search
    /// of a block as read never asks it, and one guided by the message
    /// holds only for the evidence it was guided by, whatever set read
    /// that. Choosing the set under Result once dropped every search.
    fn holds(&self, version: SymbolVersion, ec: EcLevel, mask: u8, opts: &DecodeOptions) -> bool {
        (self.version, self.ec, self.mask) == (version, ec, mask) && self.opts == DecodeOptions { charset: self.opts.charset, ..*opts }
    }
}

/// What a block of the layout in effect is read from in the document:
/// the modules of its codewords, with their reliability, and the pins on
/// its codewords. A search of the block is of the block as it is while
/// these stay as they were ([`App::finish_try_harder`]).
#[derive(PartialEq)]
struct BlockRead {
    modules: Vec<(Module, u8)>,
    pins: Vec<Option<Pin>>,
}

/// A search of the whole photo for symbols, running on a thread of its
/// own: it takes seconds a photo, a few per family, and the window must
/// keep answering while it runs. The result arrives through the channel.
struct PendingSearch {
    /// The photo, at the lens setting, the search was started for.
    generation: u64,
    receiver: std::sync::mpsc::Receiver<Sought>,
    /// The photo's Invert setting when the search began.
    photo_inverted: bool,
    /// The photo's adjustments when the search began, which what it read
    /// was read under ([`App::poll_search`]).
    adjust: Option<Adjust>,
    /// The one family searched for, when the search is not of every
    /// family.
    family: Option<Family>,
    started: std::time::Instant,
    /// Calls the search off: asked when the search is dropped, whether
    /// by the user or because its photo has gone, so that a thread no
    /// one is waiting for does not go on searching.
    cancel: Cancel,
}

impl Drop for PendingSearch {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// A slow step of the alignment running on a thread of its own: finding
/// the corners takes a second a photo and refining them most of another,
/// and the window must keep answering meanwhile, as it does during a
/// search. The result arrives through the channel and is taken into the
/// document on the UI thread, with the undo point and the notice the step
/// would have left had it run in the frame. One job runs at a time; a
/// search may run beside it, since the two touch nothing in common.
struct PendingJob {
    kind: JobKind,
    /// The photo, at the lens setting, the job was started for.
    generation: u64,
    /// The document's version when the job began. A result that means
    /// something only at that version is dropped if the version changed.
    version: SymbolVersion,
    /// The document's outline and anchors when the job began. A result
    /// that would place the outline is of the outline the job set out
    /// from: one the user has moved on meanwhile is not overwritten
    /// with it.
    corners: [[f64; 2]; 4],
    anchors: Vec<Anchor>,
    /// The photo's adjustments when the job began. A read the job makes
    /// is of the photo so adjusted, and one that decoded only through a
    /// copy of it sets the photo's adjustments from those
    /// ([`App::adjust_as_read`]): its result is dropped if they changed.
    adjust: Option<Adjust>,
    receiver: std::sync::mpsc::Receiver<JobResult>,
    started: std::time::Instant,
    /// Calls the job off, as a search's does: see [`PendingSearch`].
    cancel: Cancel,
}

impl Drop for PendingJob {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// Which step a job is running, for the status bar and the buttons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JobKind {
    FindCorners,
    /// Placing the corners and anchors on a symbol the search found.
    Adopt,
    Refine,
    /// Ranking the versions on the corners.
    Suggest,
    /// Reading every symbol the search found.
    ReadFound,
    /// Ranking the placements of the symbol inside the outline.
    Placements,
    /// Searching the block with every budget at its most. Of the grid,
    /// not the photo: its result stands whatever becomes of the photo.
    TryHarder(usize),
}

impl JobKind {
    /// What the step is doing, as the status bar puts it.
    fn label(self) -> &'static str {
        match self {
            JobKind::FindCorners => "Finding the symbol inside the outline",
            JobKind::Adopt => "Placing the corners on the symbol found",
            JobKind::Refine => "Refining the corners and anchors",
            JobKind::Suggest => "Ranking the versions on these corners",
            JobKind::ReadFound => "Reading the symbols found",
            JobKind::Placements => "Ranking the placements inside the outline",
            JobKind::TryHarder(_) => "Searching a block harder",
        }
    }

    /// The step works on the photo, and is dropped with it: a step on the
    /// grid alone runs on whatever becomes of the photo.
    fn of_photo(self) -> bool {
        !matches!(self, JobKind::TryHarder(_))
    }
}

/// A row of a list under Alignment that a step was begun from, to put
/// the outline on what it holds: marked as the one in use
/// ([`Proposals::adopted`], [`Placements::adopted`]) once the outline
/// is put there, and not as the step begins, since what the step hands
/// back may be dropped. Marked as the step began, a row stayed marked
/// after a drag of the outline had the result dropped, and the symbol
/// in it was listed as in use and kept off the canvas, where the notice
/// said to click it again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Picked {
    /// The `i`-th symbol the search found.
    Proposal(usize),
    /// The `i`-th placement listed inside the outline.
    Placement(usize),
}

/// What a job worked out, to be taken into the document on the UI thread.
enum JobResult {
    /// Find in outline: the symbol found inside the outline, made ready to
    /// adopt, with the family it was found under when that is not the one
    /// asked about; or nothing.
    Corners(Option<(Prepared, Option<Family>)>),
    /// A symbol the search found, or a placement the detector ranked,
    /// made ready to adopt, what to call it in a notice, and the row of
    /// a list under Alignment it was picked from, when it was.
    Adopt(Prepared, String, Option<Picked>),
    /// Refine: the corners and anchors it set out from, and what it made
    /// of them, `None` when the corners form no usable outline. A result
    /// is for the outline it set out from: one the user has moved on
    /// meanwhile is not overwritten with it.
    Refine { corners: [[f64; 2]; 4], anchors: Vec<Anchor>, refined: Option<Refined> },
    Suggest(Vec<VersionGuess>),
    /// Read all: what each symbol found reads as, in the order of the
    /// outlines the symbols had when the job began, and the photo's
    /// Invert setting they were read under.
    ReadFound { outlines: Vec<[[f64; 2]; 4]>, reads: Vec<Option<ReadResult>>, photo_inverted: bool },
    /// List placements: every placement the detector ranks inside the
    /// outline, best first, and the photo's Invert setting they were
    /// made under.
    Placements { found: Vec<Detection>, photo_inverted: bool },
    /// Try harder: the block searched, what it was read from when the
    /// search began, and every block searched harder so far, with what
    /// they were searched under ([`KeptHarder`]).
    Harder { block: usize, read: BlockRead, kept: KeptHarder },
}

/// A detection refined against the photo, with everything else that
/// adopting it needs and takes time to work out.
struct Prepared {
    /// The detection, its corners and anchors refined.
    found: Detection,
    /// (matching, fixed) at the refined corners.
    agreement: (usize, usize),
    /// Every version of the family in use when the job began, ranked on
    /// the corners found; made when the version found would not be
    /// adopted outright, so that it can be suggested instead.
    guesses: Option<(Family, Vec<VersionGuess>)>,
    /// The photo's Invert setting the detection was made under.
    searched_inverted: bool,
    /// What the detection decoded to, where Find in outline read it through
    /// to text to choose among the placements.
    read: Option<ReadNote>,
    /// That read whole, its grid and pins with it: loaded in place of
    /// adopting the detection, which takes the outline and the anchors
    /// and no more, when it decoded and reads are loaded.
    result: Option<Box<ReadResult>>,
}

/// What a read of a detection made of it, in brief.
struct ReadNote {
    /// The sampling kernel the read that decoded best used.
    kernel: Kernel,
    /// The threshold rule it used: each module against its neighbourhood,
    /// or every module against one split of the symbol, or against a
    /// threshold that lies as a surface across it.
    rule: Rule,
    /// What it read the photo through: a copy of it about the symbol,
    /// enlarged and sharpened, or the photo as it is.
    enhance: Enhance,
    /// The blur that copy undid, when it was deblurred, in the photo's
    /// pixels ([`ReadResult::deblurred`]).
    deblurred: Option<deblur::Spread>,
    /// Whether it sharpened the modules against their neighbours, as
    /// Sampling asked when the read began.
    sharpen_modules: bool,
    /// Whether it kept the modules the unsure band would leave unknown
    /// at the colour they fell on.
    unsure_kept: bool,
    /// Blocks verified, of the blocks in the symbol.
    verified: usize,
    blocks: usize,
    trustworthy: bool,
    /// The read stands only with the symbol's padding pinned.
    padding_pinned: bool,
    /// The grid read as the other QR Code model, which its format word
    /// named and which decoded.
    reinterpreted: Option<SymbolVersion>,
}

/// What Refine worked out: the corners refined, and every anchor moved
/// onto its pattern, beside the anchors as they were.
struct Refined {
    refinement: Refinement,
    before: Vec<Anchor>,
    after: Vec<Anchor>,
    /// The bleed the symbol was placed expecting ([`Detection::bloom`]):
    /// nought for a symbol placed as printed.
    bloom: f64,
}

pub struct App {
    doc: Document,
    history: History,
    photo: Option<Photo>,
    view: ViewState,
    derived: Derived,
    dirty: bool,
    opts: DecodeOptions,
    /// How long a notice that is no error stays in the status bar, in
    /// seconds: [`NOTICE_SECONDS`], or for ever where no one is watching
    /// the clock, as in a test.
    notice_seconds: f32,
    brute: Option<BruteResults>,
    /// Blocks searched harder, and what they were searched under.
    harder: Option<KeptHarder>,
    /// How far the solvers reach under each version and level asked
    /// about lately, at the budgets they were asked about at ([`Reach`]):
    /// the panels ask every frame, and it takes a moment to work out.
    reaches: RefCell<Vec<(ReachKey, Option<Reach>)>>,
    guesses: Option<Vec<VersionGuess>>,
    /// The projection for the current outline and anchors.
    model: Option<Model>,
    preview: Option<SamplePreview>,
    /// What the last corner search found, with the corners it left; shown
    /// on the canvas while the outline stays there.
    detection: Option<Detection>,
    /// Symbols a search of the whole photo found, on offer until the list
    /// is cleared or the photo, or its lens setting, changes.
    proposals: Option<Proposals>,
    /// Finder patterns the last search of the whole photo could fit no
    /// symbol to, on offer for as long as its list of symbols is.
    leads: Vec<Lead>,
    /// The placements the detector ranks inside the outline, when they
    /// were asked for, on offer for as long.
    placements: Option<Placements>,
    /// A search of the whole photo still running on its own thread.
    search: Option<PendingSearch>,
    /// A step of the alignment still running on its own thread.
    job: Option<PendingJob>,
    /// Counts photos loaded and changes of the lens setting, which moves
    /// every pixel of a photo and so makes another of it, so that a search
    /// or a job finishing after its photo was replaced or straightened
    /// anew is told apart from one for the photo on show.
    photo_generation: u64,
    /// Last notice for the status bar.
    message: Option<Notice>,
    /// The EC level and mask were carried over during a stroke; settle them
    /// by brute force once it ends.
    settle_pending: bool,
    /// The most a texture may measure on a side, as the window last said:
    /// a photo larger than that is drawn through a texture shrunk to fit,
    /// stretched over the photo's own pixels, which everything else
    /// works in.
    texture_limit: usize,
    /// The window's drawing, where the graphics card the photo is
    /// adjusted on draws it: the photo's textures are then laid there.
    screen: Option<screen::Screen>,
    #[cfg(feature = "selftest")]
    selftest: Option<selftest::SelfTest>,
}

/// Recompute everything that follows from the document. When the format
/// bits cannot settle the EC level and mask, brute force chooses them and
/// its attempts are returned as well; while `settle` is false the previous
/// choice is carried over instead, which keeps a brush stroke responsive.
/// The blocks searched harder are taken from `harder` where it holds for
/// the level and mask chosen. Every decode is charged as one of those the
/// document's grid was picked among ([`Document::picked`]), and for what
/// each pin it is made under was picked among where the pin reaches
/// ([`qr_forensics::decode::codewords::Pin::among`]), and the decode
/// under the level and mask a brute force chose as one of those it
/// tried ([`decoding_options`]): it once went uncharged for them, and
/// stood on more than the attempt it was the pick of.
fn derive(
    doc: &Document,
    previous: Option<&Derived>,
    opts: &DecodeOptions,
    harder: Option<&KeptHarder>,
    settle: bool,
) -> (Derived, Option<Vec<Attempt>>) {
    let picked = doc.picked(opts);
    let func = match previous {
        Some(old) if old.func.version == doc.version => old.func.clone(),
        _ => Arc::new(FunctionMap::new(doc.version)),
    };
    let analysis = analyze(&doc.grid, &func);
    let levels = doc.version.ec_levels();
    let mut attempts = None;
    let (ec, mask, format_by_brute) = match doc.format {
        FormatSel::Manual { ec, mask } if levels.contains(&ec) && mask < doc.version.mask_count() => {
            (ec, mask, false)
        }
        _ if analysis.format.is_reliable() => {
            let best = analysis.format.best().expect("a reliable analysis has a best candidate");
            (best.fields.ec, best.fields.mask, false)
        }
        _ => match previous {
            Some(old) if !settle && old.format_by_brute && old.func.version == doc.version => {
                (old.ec, old.mask, true)
            }
            _ => {
                let pins = doc.pins_ec.map(|level| (level, &doc.pins));
                let tried = brute_force_whole(&doc.grid, &func, pins, &picked);
                let (ec, mask) = tried.first().map_or((levels[0], 0), |a| (a.ec, a.mask));
                attempts = Some(tried);
                (ec, mask, true)
            }
        },
    };
    let layout = match previous {
        Some(old) if Arc::ptr_eq(&old.layout.func, &func) && old.ec == ec => old.layout.clone(),
        _ => Arc::new(SymbolLayout::new(func.clone(), ec).expect("EC level is valid for the version")),
    };
    // Pins number codewords under one EC level's block structure; under any
    // other they would land on the wrong codewords.
    let no_pins = Pins::new();
    let pins = if doc.pins_ec == Some(ec) { &doc.pins } else { &no_pins };
    let decoding = decoding_options(&picked, format_by_brute, &func);
    let no_harder = Harder::default();
    let harder = harder.filter(|kept| kept.holds(doc.version, ec, mask, &decoding)).map_or(&no_harder, |kept| &kept.searched);
    let report = decode_keeping(&doc.grid, &layout, mask, pins, &decoding, harder);
    let padding = if report.is_complete() { None } else { detect_padding(&layout, &report) };
    // When nothing decodes and the format bits could not say the grid is
    // a mirror image, try its mirror images outright, as ZXing and ZBar
    // retry a failed decode transposed, with the budgets whole, as the
    // decode they are set beside has them. Only once a stroke has ended:
    // it is a brute force of its own.
    let mirror_by_decode = (settle
        && format_by_brute
        && !report.is_trustworthy()
        && !analysis.inverted
        && analysis.mirrored.is_none()
        && doc.grid.count(Module::Unknown) < doc.grid.cells().len())
    .then(|| {
        let best = brute_force_mirrored_whole(&doc.grid, &func, &picked).into_iter().next()?;
        let better = best.report.blocks_verified() > report.blocks_verified()
            && (best.report.is_trustworthy() || best.report.blocks_verified() * 2 > best.report.blocks.len());
        better.then_some(best.mirror).flatten()
    })
    .flatten();
    (Derived { func, analysis, ec, mask, layout, report, padding, format_by_brute, mirror_by_decode }, attempts)
}

/// The options a document is decoded under, from `opts` with the
/// decodes it was picked among counted already ([`derive`]): counted too
/// as one of the levels and masks a brute force tried, where that is
/// what settled the format ([`formats`]), since the decode is the pick
/// of them.
fn decoding_options(opts: &DecodeOptions, format_by_brute: bool, func: &FunctionMap) -> DecodeOptions {
    if format_by_brute { opts.one_of(formats(func)) } else { *opts }
}

impl App {
    /// `image` is a photo to open straight away.
    pub fn new(cc: &eframe::CreationContext<'_>, image: Option<std::path::PathBuf>) -> Self {
        let mut app = Self::blank();
        app.screen = screen::Screen::of(cc);
        if let Some(path) = image {
            app.load_path(&cc.egui_ctx, &path);
        }
        app
    }

    /// The app as it starts: a blank grid and no photo, decoded under
    /// [`default_decode_options`].
    fn blank() -> Self {
        let doc = Document::blank(SymbolVersion::Qr(2));
        let opts = default_decode_options();
        let (derived, _) = derive(&doc, None, &opts, None, true);
        Self {
            doc,
            history: History::default(),
            photo: None,
            view: ViewState::default(),
            derived,
            dirty: false,
            opts,
            notice_seconds: NOTICE_SECONDS,
            brute: None,
            harder: None,
            reaches: RefCell::default(),
            guesses: None,
            model: None,
            preview: None,
            detection: None,
            proposals: None,
            leads: Vec::new(),
            placements: None,
            search: None,
            job: None,
            photo_generation: 0,
            message: None,
            settle_pending: false,
            // What egui allows until a window says what its own limit is.
            texture_limit: 2048,
            screen: None,
            #[cfg(feature = "selftest")]
            selftest: selftest::SelfTest::from_env(),
        }
    }

    /// Bring everything that follows from the document up to it, when
    /// the document has changed since. Called before anything reads the
    /// derived values against the document: at the start of a frame,
    /// again once the results of the worker threads have been taken in,
    /// which may switch the version, and before each action is applied,
    /// since an action before it in the same frame may have done so. The
    /// function map, the layout and the report are then always those of
    /// the document's version, and nothing indexes a grid of one size by
    /// the map of another.
    fn catch_up(&mut self) {
        if self.dirty || self.derived.func.version != self.doc.version {
            self.refresh();
        }
    }

    fn refresh(&mut self) {
        let settle = self.view.stroke.is_none();
        let (derived, attempts) = derive(&self.doc, Some(&self.derived), &self.opts, self.harder.as_ref(), settle);
        // A start of the padding given by hand counts codewords of the
        // layout it was given under.
        if (derived.func.version, derived.ec) != (self.derived.func.version, self.derived.ec) {
            self.view.pad_from = None;
        }
        self.derived = derived;
        self.settle_pending = !settle && self.derived.format_by_brute;
        match attempts {
            Some(attempts) => self.brute = Some(BruteResults { attempts, stale: false }),
            None => {
                if let Some(brute) = &mut self.brute {
                    brute.stale = true;
                }
            }
        }
        if self.view.selected.is_some_and(|cw| cw >= self.derived.layout.codewords.len()) {
            self.view.selected = None;
        }
        if self.view.highlight_block.is_some_and(|b| b >= self.derived.layout.blocks.len()) {
            self.view.highlight_block = None;
        }
        if self.view.table_block.is_some_and(|b| b >= self.derived.layout.blocks.len()) {
            self.view.table_block = None;
        }
        let (width, height) = (self.derived.func.width, self.derived.func.height);
        if self.view.hover.is_some_and(|(x, y)| x >= width || y >= height) {
            self.view.hover = None;
        }
        self.dirty = false;
    }

    fn notify(&mut self, text: impl Into<String>) {
        self.message = Some(Notice { text: text.into(), is_error: false, at: std::time::Instant::now() });
    }

    fn fail(&mut self, text: impl Into<String>) {
        self.message = Some(Notice { text: text.into(), is_error: true, at: std::time::Instant::now() });
    }

    /// Pins in effect: those made under the current EC level.
    fn active_pins(&self) -> Pins {
        if self.doc.pins_ec == Some(self.derived.ec) { self.doc.pins.clone() } else { Pins::new() }
    }

    /// Record an undo point. A notice about an earlier state is dropped.
    fn checkpoint(&mut self) {
        self.history.checkpoint(&self.doc);
        self.message = None;
    }

    /// Record an undo point and mark the document changed.
    fn begin_edit(&mut self) {
        self.checkpoint();
        self.dirty = true;
    }

    /// Set one module, leaving fixed patterns alone when they are locked:
    /// to a colour, or to what sampling reads off the photo there, with
    /// the reliability it reads it at and with no unsure band, whatever
    /// Sampling has for one: the colour the module falls on, however
    /// near its threshold. Nothing is read where there is no sample of
    /// the photo to go by. A colour goes in at the painted reliability,
    /// certain at the top of the scale and a guess at the bottom, and a
    /// module of that colour already is painted all the same: how sure
    /// it is goes with the paint, so that a guess is made certain, or a
    /// module sampled or certain made a guess, by painting over it. What
    /// is read off the photo goes in at the painted reliability too while
    /// that is below certain, whatever reliability it read at. Unknown is
    /// neither, and on a module unknown already it paints nothing: there
    /// is no colour there to be sure of, and the press is no edit. The
    /// Doubtful ink leaves the colour and sets the reliability alone, and
    /// Sure, its other button, makes the module certain again; neither
    /// touches an unknown module.
    fn paint(&mut self, x: i32, y: i32, ink: Ink) {
        let func = &self.derived.func;
        if x < 0 || y < 0 || x as usize >= func.width || y as usize >= func.height {
            return;
        }
        let (x, y) = (x as usize, y as usize);
        if self.view.lock_fixed && func.expected(x, y).is_some() {
            return;
        }
        let reliability = self.view.paint_reliability;
        let grid = &mut self.doc.grid;
        let before = (grid.get(x, y), grid.reliability(x, y));
        match ink {
            Ink::Module(Module::Unknown) if before.0 == Module::Unknown => {}
            Ink::Module(Module::Unknown) => grid.set(x, y, Module::Unknown),
            Ink::Module(module) => grid.set_painted(x, y, module, reliability),
            Ink::Photo => {
                let sampled = self.preview.as_ref().map(|preview| &preview.classified);
                let Some(sampled) = sampled.filter(|sampled| sampled.modules.len() == func.width * func.height) else { return };
                let i = y * func.width + x;
                let read = if reliability < CERTAIN { reliability } else { sampled.reliability[i] };
                grid.set_sampled(x, y, sampled.decided[i], read);
            }
            Ink::Doubt => grid.set_reliability(x, y, reliability.min(CERTAIN - 1)),
            Ink::Sure => grid.set_reliability(x, y, CERTAIN),
        }
        self.dirty |= before != (grid.get(x, y), grid.reliability(x, y));
    }

    fn set_version(&mut self, version: SymbolVersion) {
        if version == self.doc.version {
            return;
        }
        self.begin_edit();
        // The versions suggested were ranked for the family in use: under
        // another family the list would offer them still, and a click on
        // one would switch the family back.
        if version.family() != self.doc.version.family() {
            self.guesses = None;
        }
        let (old, anchors) = (self.doc.version, self.doc.anchors.len());
        self.doc = self.doc.blank_like(version);
        self.view.selected = None;
        self.view.highlight_block = None;
        self.view.table_block = None;
        self.view.edit_view.auto_fit = true;
        self.brute = None;
        self.model = None;
        self.preview = None;
        // The anchors named modules of the old version, and went with
        // its grid: said, since nothing else shows that they did.
        if anchors > 0 {
            let what = if anchors == 1 { "1 anchor was".to_owned() } else { format!("{anchors} anchors were") };
            self.notify(format!(
                "Started a new {} {version} grid (undoable). {what} removed: they named modules of version {old}.",
                version.family().label()
            ));
        }
    }

    /// Keep the modules as painted and read them as a symbol of `version`,
    /// which must be the same size: the two QR Code models share their
    /// sizes and fixed patterns, and only the format word and the data
    /// layout tell them apart. Pins name codewords of the old layout and
    /// anchors name its alignment patterns, so neither carries over.
    fn reinterpret_as(&mut self, version: SymbolVersion) {
        if version == self.doc.version {
            return;
        }
        if (version.width(), version.height()) != (self.doc.version.width(), self.doc.version.height()) {
            self.fail(format!("A {} {} symbol is not the size of this grid.", version.family().label(), version));
            return;
        }
        self.begin_edit();
        let grid = self.doc.grid.clone();
        self.doc = Document { grid, ..self.doc.blank_like(version) };
        self.view.selected = None;
        self.view.highlight_block = None;
        self.view.table_block = None;
        self.brute = None;
        self.model = None;
        self.preview = None;
        self.notify(format!("Reading the grid as {} version {version}.", version.family().label()));
    }

    fn set_photo(&mut self, ctx: &egui::Context, name: String, mut image: image::RgbaImage) {
        // On grey where it is transparent, before anything is cut or
        // resized from it; and kept as it was as well, for Transparent
        // parts to lay on another. A photo with no transparency is laid on
        // white, which only what the lens term draws in from beyond its
        // edges shows ([`Adjust::backdrop`]), as it always has.
        let unflattened = has_transparency(&image).then(|| Arc::new(image.clone()));
        let backdrop = if unflattened.is_some() { Backdrop::default() } else { Backdrop::White };
        flatten_alpha_onto(&mut image, backdrop);
        // The outline, the anchors and the adjustments are of the photo
        // they were placed on: the new photo starts without them, and no
        // undo or redo brings another photo's back onto it. The symbol,
        // the grid and its pins, stays as it is, with its own history.
        // A small photo starts enlarged, so that its modules can be seen.
        let adjust = Adjust { backdrop, ..Adjust::for_photo(image.width(), image.height()) };
        self.show_photo(ctx, name, Arc::new(image), unflattened, None, adjust);
        let Some(photo) = &self.photo else { return };
        self.doc.corners = photo.default_corners();
        self.doc.anchors.clear();
        let (width, height) = (photo.width, photo.height);
        let mut loaded = format!("Loaded {} ({width} × {height}", photo.name);
        if photo.scale() > 1.0 {
            loaded.push_str(&format!(
                ", enlarged {:.0} times from {} × {} so that its modules can be seen: Enlarge, under Photo, sets how far",
                photo.scale(),
                photo.source.width(),
                photo.source.height()
            ));
        }
        loaded.push_str(
            "). Press Search whole photo to look for the symbols in it, or drag the corners loosely around the symbol \
             and press Find in outline.",
        );
        if photo.unflattened.is_some() {
            let others: Vec<String> = Backdrop::ALL.into_iter().filter(|&other| other != backdrop).map(|other| other.label().to_lowercase()).collect();
            loaded.push_str(&format!(
                " Its transparent parts are filled with {}; Transparent parts, under Photo, switches to {}.",
                backdrop.label().to_lowercase(),
                others.join(" or ")
            ));
        }
        if width.max(height) > SEARCH_SIDE {
            loaded.push_str(&format!(
                " A photo this large is searched whole in a copy reduced to {SEARCH_SIDE} pixels on its longer side; an \
                 outline is read in the photo's own pixels. If the search misses a small symbol, draw the outline \
                 round it and press Find in outline, or Crop to outline and search again.",
            ));
        }
        // The photo is searched for symbols only when the user asks: a
        // search takes seconds, and the outline may as well be drawn by
        // hand.
        self.notify(loaded);
    }

    /// Put a photo as loaded, or the part `crop` of it, on show under
    /// `adjust`, enlarged no further than its size allows: its
    /// luminance and texture made, and whatever was found or under way
    /// on the photo shown before dropped. Where the outline and anchors
    /// go is the caller's to say.
    fn show_photo(
        &mut self,
        ctx: &egui::Context,
        name: String,
        source: Arc<image::RgbaImage>,
        unflattened: Option<Arc<image::RgbaImage>>,
        crop: Option<Crop>,
        adjust: Adjust,
    ) {
        let original = part_on_show(&source, crop, adjust.backdrop);
        let (own_width, own_height) = original.dimensions();
        let adjust = Adjust { enlarge: adjust.enlarge.within(own_width, own_height), ..adjust };
        let greyscale = self.view.greyscale;
        let (display, luma) = adjusted(&adjust, &source, &original, crop, self.screen.is_some(), greyscale);
        let (width, height) = (luma.width, luma.height);
        self.texture_limit = ctx.input(|i| i.max_texture_side);
        let texture = photo_texture(self.screen.as_ref(), ctx, "photo", display, (width, height), self.texture_limit);
        self.history.photo_changed(&self.doc);
        self.doc.adjust = adjust;
        let enlargements = Enlarge::steps_for(own_width, own_height);
        let mut photo =
            Photo { name, width, height, source, unflattened, crop, original, adjust, enlargements, luma: Arc::new(luma), texture, greyscale, binarised: None };
        photo.offer(adjust.enlarge);
        self.photo = Some(photo);
        self.view.stage = Stage::Align;
        self.view.align_view.auto_fit = true;
        self.view.binarize_auto = true;
        self.model = None;
        self.preview = None;
        self.guesses = None;
        self.detection = None;
        self.proposals = None;
        self.leads.clear();
        self.placements = None;
        self.search = None;
        self.drop_photo_job();
        self.photo_generation += 1;
    }

    /// Show another part of the photo as loaded, `rect` of it or the
    /// whole, with the outline and the anchors carried to where they lie
    /// in it. A lens term in force is of the whole photo, about its
    /// centre, and means nothing of a part: the part is of the photo
    /// straightened by it, after it is enlarged as the whole is, and
    /// takes 0 for its own, and the whole photo takes the term back. The enlargement is of the pixels as
    /// loaded, and a part keeps it whatever its size, as it keeps the
    /// other adjustments: the symbol is as large after the cut as it
    /// was before it, and Enlarge is the user's to set. The whole photo
    /// takes back the enlargement it had, which a part may since have
    /// been taken past what the whole of a large photo is enlarged to
    /// unasked. False with no photo.
    fn reframe(&mut self, ctx: &egui::Context, rect: Option<[u32; 4]>) -> bool {
        let Some(old) = self.photo.as_ref() else { return false };
        let lens = old.crop.map_or(self.doc.adjust.k1, |crop| crop.k1);
        let whole = old.crop.map_or(self.doc.adjust.enlarge, |crop| crop.enlarge);
        let crop = rect.map(|rect| Crop { rect, k1: lens, enlarge: whole });
        let enlarge = if crop.is_some() { self.doc.adjust.enlarge } else { whole };
        let adjust = Adjust { k1: if crop.is_some() { 0.0 } else { lens }, enlarge, ..self.doc.adjust };
        let corners = self.doc.corners.map(|c| old.to_source(c));
        let mut anchors = self.doc.anchors.clone();
        for anchor in &mut anchors {
            anchor.anchor.center = old.to_source(anchor.anchor.center);
        }
        let (name, source, unflattened) = (old.name.clone(), old.source.clone(), old.unflattened.clone());
        self.show_photo(ctx, name, source, unflattened, crop, adjust);
        let Some(new) = self.photo.as_ref() else { return false };
        self.doc.corners = corners.map(|c| new.to_working(c));
        for anchor in &mut anchors {
            anchor.anchor.center = new.to_working(anchor.anchor.center);
        }
        self.doc.anchors = anchors;
        true
    }

    /// Cut the photo down to the outline, with room about it. A search of
    /// the whole photo looks at a copy no larger than [`SEARCH_SIDE`],
    /// and then sees that part at its own size; every change of the
    /// adjustments works on that part alone. The part is enlarged, or
    /// reduced, as the photo was: a small one is not enlarged for being
    /// small, as a small photo is when it is loaded.
    fn crop_to_outline(&mut self, ctx: &egui::Context) {
        let Some(photo) = &self.photo else {
            self.fail("Load a photo first.");
            return;
        };
        let (width, height) = photo.source.dimensions();
        let rect = crop_box(self.doc.corners.map(|c| photo.to_source(c)), width, height);
        let Some(rect) = rect.filter(|&rect| rect != photo.crop.map_or([0, 0, width, height], |crop| crop.rect)) else {
            self.fail(
                "There is nothing to crop to: drag the corners around the symbol first, so that the outline takes in \
                 less than the photo on show and lies on it.",
            );
            return;
        };
        let lens = photo.crop.is_none() && self.doc.adjust.k1 != 0.0;
        if !self.reframe(ctx, Some(rect)) {
            return;
        }
        let mut text = format!("Cropped to the outline: {} × {} pixels of the photo as loaded", rect[2], rect[3]);
        let enlarge = self.doc.adjust.enlarge;
        if enlarge != Enlarge::NONE {
            text.push_str(&format!(", at {enlarge} as the photo was"));
        }
        text.push_str(". What was found in the photo was dropped with the rest of it: search, or Find in outline, again.");
        if lens {
            text.push_str(" The cropped part keeps the whole photo's lens correction, and Lens reads 0 until the whole photo is shown again.");
        }
        self.notify(text);
    }

    /// Put the whole photo back on show after a crop.
    fn show_whole_photo(&mut self, ctx: &egui::Context) {
        if self.photo.as_ref().is_none_or(|photo| photo.crop.is_none()) {
            return;
        }
        if self.reframe(ctx, None)
            && let Some(photo) = &self.photo
        {
            self.notify(format!(
                "Showing the whole photo again ({} × {}). The outline and anchors are where they were on it.",
                photo.width, photo.height
            ));
        }
    }

    /// Recompute the photo's luminance and texture after a change to its
    /// adjustments, at the size its enlargement now makes of it.
    fn reapply_adjust(&mut self) {
        if let Some(photo) = &mut self.photo {
            photo.greyscale = self.view.greyscale;
            let (display, luma) = adjusted(&photo.adjust, &photo.source, &photo.original, photo.crop, self.screen.is_some(), photo.greyscale);
            (photo.width, photo.height) = (luma.width, luma.height);
            photo.luma = Arc::new(luma);
            lay_photo_texture(self.screen.as_ref(), &mut photo.texture, display, (photo.width, photo.height), self.texture_limit);
            photo.binarised = None;
            photo.offer(photo.adjust.enlarge);
        }
    }

    /// Draw the photo again as the view asks for it, in grey or in its
    /// colours. What is read off it is as it was.
    fn redraw_photo(&mut self) {
        let Some(photo) = &mut self.photo else { return };
        photo.greyscale = self.view.greyscale;
        let display = if photo.greyscale {
            Shown::Bytes(photo.luma.to_rgba())
        } else {
            adjusted(&photo.adjust, &photo.source, &photo.original, photo.crop, self.screen.is_some(), false).0
        };
        lay_photo_texture(self.screen.as_ref(), &mut photo.texture, display, (photo.width, photo.height), self.texture_limit);
    }

    /// Change how the photo is read, and bring the photo up to it. An
    /// enlargement moves every point of the photo with it, and the
    /// outline and the anchors go with the points they were on. The
    /// caller has taken the undo point. Returns what to tell the user,
    /// when what was found or under way in the photo as it was had to
    /// be dropped.
    fn set_adjust(&mut self, mut adjust: Adjust) -> Option<String> {
        if let Some(photo) = &self.photo {
            let (own_width, own_height) = photo.original.dimensions();
            adjust.enlarge = adjust.enlarge.within(own_width, own_height);
            let (width, height) = adjust.enlarge.size(own_width as usize, own_height as usize);
            let scale = [width as f64 / photo.width as f64, height as f64 / photo.height as f64];
            if scale != [1.0, 1.0] {
                self.doc.corners = self.doc.corners.map(|c| [c[0] * scale[0], c[1] * scale[1]]);
                for anchor in &mut self.doc.anchors {
                    let center = anchor.anchor.center;
                    anchor.anchor.center = [center[0] * scale[0], center[1] * scale[1]];
                }
            }
        }
        self.doc.adjust = adjust;
        self.sync_adjust()
    }

    /// Bring the photo up to the document's adjustments, after a change
    /// of them or an undo that took one back. A change of the
    /// enlargement among them carries what was found in the photo to
    /// where it lies at its new size, and a change of the lens term
    /// drops it: see [`Self::photo_resized`] and [`Self::lens_changed`],
    /// whose notices come back when they have any.
    fn sync_adjust(&mut self) -> Option<String> {
        let photo = self.photo.as_mut()?;
        if photo.adjust == self.doc.adjust {
            return None;
        }
        let lens = photo.adjust.k1 != self.doc.adjust.k1;
        let before = (photo.width, photo.height);
        // The photo as loaded is laid on the new backdrop, and the part
        // on show is cut again.
        if photo.adjust.backdrop != self.doc.adjust.backdrop
            && let Some(unflattened) = &photo.unflattened
        {
            let mut source = (**unflattened).clone();
            flatten_alpha_onto(&mut source, self.doc.adjust.backdrop);
            photo.source = Arc::new(source);
            photo.original = part_on_show(&photo.source, photo.crop, self.doc.adjust.backdrop);
        }
        photo.adjust = self.doc.adjust;
        self.reapply_adjust();
        let resized = self.photo.as_ref().is_some_and(|photo| (photo.width, photo.height) != before);
        let resized = if resized { self.photo_resized(before) } else { None };
        let lens = if lens { self.lens_changed() } else { None };
        match (resized, lens) {
            (Some(resized), Some(lens)) => Some(format!("{resized} {lens}")),
            (resized, lens) => resized.or(lens),
        }
    }

    /// The photo is worked on at another size than the `width` by
    /// `height` it was: every point of it lies elsewhere in its new
    /// pixels, by the change of size and no more. So what was found in
    /// it is carried to where it lies now, with the view of it, where a
    /// change of lens leaves nothing to carry; the outline and the
    /// anchors, which are the document's, have gone there already. A
    /// search or a step still under way is of the photo as it was, and
    /// is abandoned, the photo counting as a new one. Returns what to
    /// tell the user, when anything was.
    fn photo_resized(&mut self, (width, height): (usize, usize)) -> Option<String> {
        let photo = self.photo.as_ref()?;
        let scale = [photo.width as f64 / width as f64, photo.height as f64 / height as f64];
        let carry = |found: &mut Detection| {
            found.corners = found.corners.map(|c| [c[0] * scale[0], c[1] * scale[1]]);
            for anchor in &mut found.anchors {
                anchor.center = [anchor.center[0] * scale[0], anchor.center[1] * scale[1]];
            }
        };
        if let Some(found) = &mut self.detection {
            carry(found);
        }
        if let Some(placements) = &mut self.placements {
            placements.found.iter_mut().for_each(carry);
        }
        if let Some(proposals) = &mut self.proposals {
            proposals.found.iter_mut().for_each(carry);
            if let Some(reads) = &mut proposals.reads {
                reads.results.iter_mut().flatten().for_each(|read| carry(&mut read.detection));
            }
            // What the search's reads were charged is found by the corners
            // of the symbol read ([`Proposals::among_of`]), which go with
            // it. Left where they were, every symbol of the list was read
            // as one asked for once the photo was enlarged, as it is when a
            // read made through a copy enlarged is loaded.
            for charged in &mut proposals.charged {
                charged.corners = charged.corners.map(|c| [c[0] * scale[0], c[1] * scale[1]]);
            }
        }
        for lead in &mut self.leads {
            lead.outline = lead.outline.map(|c| [c[0] * scale[0], c[1] * scale[1]]);
            for finder in &mut lead.finders {
                *finder = [finder[0] * scale[0], finder[1] * scale[1]];
            }
        }
        // The photo stays where it was on the canvas, and as large.
        let view = &mut self.view.align_view;
        view.center = egui::vec2(view.center.x * scale[0] as f32, view.center.y * scale[1] as f32);
        view.scale /= ((scale[0] + scale[1]) / 2.0) as f32;
        // The binarised view's radius and window were in the old pixels.
        self.view.binarize_auto = true;
        self.photo_generation += 1;
        let dropped = self.abandon_work();
        (!dropped.is_empty()).then(|| {
            format!(
                "The photo changed size, so {}. Run the search or the step again: the outline and what was found went \
                 with the photo.",
                list_phrase(&dropped)
            )
        })
    }

    /// Abandon a search or a step still under way, which is of the photo
    /// as it was: what each was, for telling the user.
    fn abandon_work(&mut self) -> Vec<String> {
        let mut dropped = Vec::new();
        if self.search.take().is_some() {
            dropped.push("the search under way was stopped".to_owned());
        }
        if let Some(job) = self.drop_photo_job() {
            dropped.push(format!("{} was stopped", job.kind.label().to_lowercase()));
        }
        dropped
    }

    /// Call off the step under way when it works on the photo
    /// ([`JobKind::of_photo`]), and say which it was. A step on the grid
    /// alone is left to run.
    fn drop_photo_job(&mut self) -> Option<PendingJob> {
        if self.job.as_ref().is_some_and(|job| job.kind.of_photo()) { self.job.take() } else { None }
    }

    /// The lens term has changed, which moves every pixel of the photo:
    /// whatever was worked out on the photo as it was lies somewhere else
    /// now. The symbols a search listed are taken off offer, a search or
    /// a step still under way is abandoned, and the photo counts as a new
    /// one, so that nothing which finishes later is taken in at the old
    /// coordinates. Returns what to tell the user, when anything was
    /// dropped.
    fn lens_changed(&mut self) -> Option<String> {
        self.photo_generation += 1;
        self.detection = None;
        self.placements = None;
        let mut dropped = Vec::new();
        // The finder patterns no symbol was fitted to go with the symbols
        // found, or alone where none was.
        let leads = !std::mem::take(&mut self.leads).is_empty();
        if self.proposals.take().is_some() {
            dropped.push("the symbols found were taken off the list".to_owned());
        } else if leads {
            dropped.push("the finder patterns found were taken off the list".to_owned());
        }
        dropped.extend(self.abandon_work());
        (!dropped.is_empty()).then(|| {
            format!(
                "The lens setting changed, which moves every part of the photo, so {}. Run the search or the step \
                 again.",
                list_phrase(&dropped)
            )
        })
    }

    /// Set the photo's Enlarge and Sharpen to what a read decoded
    /// through: a copy of the photo about the symbol, enlarged so many
    /// times again from the photo as it was adjusted when the read was
    /// made, `under`, and sharpened ([`qr_forensics::imaging::enhance`]).
    /// The whole photo so adjusted shows what the copy showed, and
    /// Sample reads the symbol as the read did. The outline, the anchors
    /// and whatever was found go with the photo to its new size. A
    /// large photo is not enlarged for it, which would make it many
    /// times the pixels to hold with no one having asked: that is left
    /// to the user, and said. A photo blurred or sharpened under Sharpen
    /// stays so through a read that only enlarged its copy, which was
    /// made of the photo so; one that sharpened its copy has Sharpen set
    /// to that, in place of what it was, the copy having been made of the
    /// photo without it ([`qr_forensics::imaging::Luma::as_shot`]). One
    /// that deblurred its copy has the Deblur filter undo what the copy
    /// undid, `deblurred` in the pixels of the photo as it was enlarged
    /// then: measured about the symbol, which the blur measured over the
    /// whole photo may not be. One that enlarged or sharpened a copy of a
    /// photo under the Deblur filter with no blur set for it holds the
    /// filter to the blur it measured over the whole photo, which the copy
    /// undid: measured again over the photo so enlarged and sharpened, it
    /// comes out otherwise. Where it measured none, and the copy was left
    /// unfiltered, the filter is set to None.
    ///
    /// Sharpen and the filter are set from what they were when the read
    /// was made, as the copy was made under them: a read loaded from the
    /// list Read all made, after another of them set the photo otherwise,
    /// or after the user did, has them set back for it, and said. The
    /// tone is left as it is. Returns what to tell the user; `None` for a
    /// read of the photo as it is, or when the photo is so adjusted
    /// already. The caller has taken the undo point.
    fn adjust_as_read(&mut self, enhance: Enhance, deblurred: Option<deblur::Spread>, under: ReadUnder) -> Option<String> {
        if enhance.is_none() {
            return None;
        }
        let photo = self.photo.as_ref()?;
        let (own_width, own_height) = photo.original.dimensions();
        // The photo as it is now, and as the read had it.
        let (before, was) = (self.doc.adjust, under.adjust);
        let wanted = was.enlarge.by(enhance.enlarge);
        // Left as it is, too small, where the photo is too large to be
        // enlarged unasked.
        let left = before.enlarge < wanted && !wanted.fits_unasked(own_width, own_height);
        // A deblurred copy lays no mask: the photo is deblurred as the
        // copy was, by its Deblur filter at full strength undoing the blur
        // the copy undid, carried to the photo's own pixels, with Sharpen
        // as it was.
        let deblur = enhance.sharpen == Sharpen::Deblur;
        let kernel = deblurred.filter(|_| deblur).map(|spread| spread.scaled(1.0 / was.enlarge.factor()));
        let enlarge = if left { before.enlarge } else { before.enlarge.max(wanted) };
        let sharpen = if matches!(enhance.sharpen, Sharpen::Off | Sharpen::Deblur) { was.sharpen } else { enhance.sharpen.amount() };
        // A copy made under the photo's own Deblur filter, with no blur set
        // for it, undid the blur the filter measured over the whole photo,
        // carried to the copy's pixels, and not one measured again at the
        // copy's size and sharpness ([`Enhanced::window`]). So the photo,
        // enlarged or sharpened as the copy was, is held to that blur, in
        // its own pixels: `Some(None)` where the filter measured none and
        // the copy was left unfiltered.
        let held = (!deblur && was.filter == Filter::Deblur && was.kernel.is_none() && (enlarge != was.enlarge || sharpen != was.sharpen))
            .then(|| under.measured.map(|spread| spread.scaled(1.0 / was.enlarge.factor())));
        let adjust = Adjust {
            enlarge,
            sharpen,
            filter: match held {
                _ if deblur => Filter::Deblur,
                Some(None) => Filter::None,
                _ => was.filter,
            },
            radius: was.radius,
            strength: if deblur { 1.0 } else { was.strength },
            kernel: kernel.or(held.flatten()).or(was.kernel),
            ..before
        };
        // What was set, the first said as set and the rest after it.
        let mut set = String::new();
        let mut say = |first: String, later: String, joined: &str| {
            if set.is_empty() {
                set = first;
            } else {
                set.push_str(joined);
                set.push_str(&later);
            }
        };
        if adjust.enlarge != before.enlarge {
            say(format!("Enlarge was set to {}", adjust.enlarge), format!("Enlarge to {}", adjust.enlarge), " and ");
        }
        if adjust.sharpen != before.sharpen {
            say(format!("Sharpen was set to {:.2}", adjust.sharpen), format!("Sharpen to {:.2}", adjust.sharpen), " and ");
        }
        // The filter and what of its settings it heeds.
        let filtered = |a: &Adjust| (a.filter, a.filter.has_radius().then_some(a.radius), (a.filter == Filter::Deblur).then_some((a.strength, a.kernel)));
        if held == Some(None) && before.filter != Filter::None {
            let none = "None, as Deblur found no blur to undo";
            say(format!("Filter was set to {none}"), format!("Filter to {none}"), ", and ");
        } else if held.is_some() && filtered(&before) == filtered(&was) && filtered(&adjust) != filtered(&before) {
            let kept = "kept at the blur it had measured over the whole photo";
            say(format!("Deblur was {kept}"), format!("Deblur {kept}"), ", and ");
        } else if held != Some(None) && filtered(&adjust) != filtered(&before) {
            // Deblurred as the copy was, or set back to what the copy was
            // made under.
            let mut filter = adjust.filter.label().to_owned();
            if adjust.filter.has_radius() && adjust.radius != before.radius {
                filter.push_str(&format!(" {} px either side", adjust.radius));
            }
            if adjust.filter == Filter::Deblur {
                if adjust.strength != before.strength {
                    filter.push_str(&if deblur { " at full strength".to_owned() } else { format!(" at strength {:.2}", adjust.strength) });
                }
                if adjust.kernel != before.kernel {
                    filter.push_str(match adjust.kernel {
                        _ if deblur => ", undoing the blur measured around the symbol",
                        Some(_) if held.is_some() => ", undoing the blur it had measured over the whole photo",
                        Some(_) => ", undoing the blur it was set to undo",
                        None => ", undoing the blur it measures over the whole photo",
                    });
                }
            }
            if !deblur {
                filter.push_str(", as it was when the symbol was read");
            }
            say(format!("Filter was set to {filter}"), format!("Filter to {filter}"), ", and ");
        }
        if set.is_empty() && !left {
            return None;
        }
        let mut text = format!("It decoded only with the photo {enhance}");
        text.push_str(&if set.is_empty() { ".".to_owned() } else { format!(", so under Photo {set}, and Sample reads it that way.") });
        if left {
            text.push_str(&format!(
                " A photo this large is not enlarged to {wanted} unasked: set Enlarge to that by hand, which takes time and \
                 memory, or crop to the outline and find the symbol again, to sample it so.",
            ));
        }
        if let Some(dropped) = self.set_adjust(adjust) {
            text.push(' ');
            text.push_str(&dropped);
        }
        Some(text)
    }

    /// Set Transparent parts, for a photo that has any; true when it
    /// changed. The caller has taken the undo point.
    fn set_photo_backdrop(&mut self, backdrop: Backdrop) -> bool {
        if self.photo.as_ref().is_none_or(|photo| photo.unflattened.is_none()) || self.doc.adjust.backdrop == backdrop {
            return false;
        }
        self.doc.adjust.backdrop = backdrop;
        // No lens term changes here, so there is nothing to report.
        let _ = self.sync_adjust();
        true
    }

    /// Set the Invert adjustment; true when it changed. The caller has
    /// taken the undo point.
    fn set_photo_invert(&mut self, invert: bool) -> bool {
        if self.photo.is_none() || self.doc.adjust.invert == invert {
            return false;
        }
        self.doc.adjust.invert = invert;
        // No lens term changes here, so there is nothing to report.
        let _ = self.sync_adjust();
        true
    }

    fn load_path(&mut self, ctx: &egui::Context, path: &Path) {
        match qr_forensics::imaging::photo::open(path) {
            Ok(image) => {
                let name = path.file_name().map_or_else(|| "image".into(), |n| n.to_string_lossy().into_owned());
                self.set_photo(ctx, name, image);
            }
            Err(error) => self.fail(format!("Could not open {}: {error}", path.display())),
        }
    }

    fn paste_image(&mut self, ctx: &egui::Context) {
        let mut clipboard = match arboard::Clipboard::new() {
            Ok(clipboard) => clipboard,
            Err(error) => {
                self.fail(format!("The clipboard could not be read ({error})."));
                return;
            }
        };
        match clipboard.get_image() {
            Ok(data) => {
                let (w, h) = (data.width as u32, data.height as u32);
                match image::RgbaImage::from_raw(w, h, data.bytes.into_owned()) {
                    Some(image) => self.set_photo(ctx, "clipboard image".into(), image),
                    None => self.fail("The clipboard image has an unexpected layout."),
                }
            }
            // A file copied in a file manager is on the clipboard by its
            // path, not its pixels: it is opened as one dropped on the
            // window is.
            Err(error) => match clipboard.get().file_list() {
                Ok(files) if !files.is_empty() => self.load_path(ctx, &files[0]),
                _ => self.fail(format!("No image on the clipboard ({error}).")),
            },
        }
    }

    /// Refresh the projection and the cached sampling of the photo if
    /// anything they depend on moved.
    fn update_preview(&mut self) {
        // The projection is laid out for the function map in use, which
        // must be the document's own.
        self.catch_up();
        let Some(photo) = &self.photo else {
            self.model = None;
            self.preview = None;
            return;
        };
        let func = &self.derived.func;
        let model_key = ModelKey {
            version: self.doc.version,
            size: (func.width, func.height),
            corners: self.doc.corners,
            anchors: self.doc.anchors.clone(),
            use_anchors: self.view.use_anchors,
        };
        if !self.model.as_ref().is_some_and(|m| m.key == model_key) {
            self.model = self.doc.projection(func.width, func.height, self.view.use_anchors).map(|projection| {
                let lattice = Arc::new(projection.lattice(func.width, func.height));
                Model { key: model_key.clone(), projection: Arc::new(projection), lattice, width: func.width, height: func.height }
            });
        }
        let key = SampleKey { model: model_key, adjust: photo.adjust, params: self.view.sample };
        if self.preview.as_ref().is_some_and(|p| p.key == key) {
            return;
        }
        self.preview = self.model.as_ref().map(|model| {
            // The photo as it lies, adjusted as the user has it: what a read
            // made through a copy enlarged or sharpened shows here once the
            // photo is adjusted to match.
            let params = SampleParams { enhance: Enhance::NONE, ..self.view.sample };
            let (classified, sharpening) =
                read_modules(&photo.luma, model.projection.as_ref(), func.width, func.height, &params, func.expected_modules(), false);
            let grid = classified.to_grid(func.width, func.height);
            let fixed = func.fixed_agreement(&grid);
            let turned = orientation_of(&grid, func);
            SamplePreview { key, classified, fixed, sharpening, turned }
        });
    }

    /// Find the symbol inside the outline, put the corners and anchors on
    /// it, refine them, and read it through to text, on a thread of its
    /// own: the decode chooses among the placements, kernels and turns
    /// the fit alone cannot. The current family is tried first; when its
    /// read does not decode, the other families are, as a reader tries
    /// every symbology it knows.
    fn find_corners(&mut self, ctx: &egui::Context) {
        if self.job_busy() {
            return;
        }
        let Some(photo) = &self.photo else {
            self.fail("Load a photo first.");
            return;
        };
        let luma = photo.luma.clone();
        let searched_inverted = photo.adjust.invert;
        let (version, corners, params, opts) = (self.doc.version, self.doc.corners, self.scan_params(), self.opts);
        let own_family_only = self.view.own_family_only;
        self.start_job(ctx, JobKind::FindCorners, move || {
            JobResult::Corners(read_inside(&luma, version, corners, &params, &opts, searched_inverted, own_family_only))
        });
    }

    /// Measure the blur the photo shows inside the outline, off the photo
    /// as its adjustments make it there but for its filter, and have the
    /// Deblur filter undo that, in the photo's own pixels, in place of
    /// the blur the whole photo shows. Never of its own accord: the user
    /// asks, with the outline on the symbol.
    fn measure_deblur(&mut self) {
        let Some(photo) = &self.photo else {
            self.fail("Load a photo first.");
            return;
        };
        let corners = self.doc.corners;
        let (x0, x1) = corners.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), c| (lo.min(c[0]), hi.max(c[0])));
        let (y0, y1) = corners.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), c| (lo.min(c[1]), hi.max(c[1])));
        if !(x1 > x0 && y1 > y0) {
            self.fail("The outline has no area: drag its corners apart, around the symbol, first.");
            return;
        }
        let window = [x0.floor().max(0.0) as usize, y0.floor().max(0.0) as usize, x1.ceil().max(0.0) as usize, y1.ceil().max(0.0) as usize];
        let Some(view) = Enhanced::unfiltered(&photo.luma, window) else {
            self.fail("The outline is off the photo: drag its corners onto the symbol first.");
            return;
        };
        let Some(spread) = deblur::spread_of(&view.luma.data, view.luma.width, view.luma.height) else {
            self.fail(
                "No blur found inside the outline: the symbol is sharp or has too few edges to measure. If it looks \
                 blurred, check that the outline is around the symbol.",
            );
            return;
        };
        let (widest, narrowest) = spread.extremes();
        let own = spread.scaled(1.0 / photo.scale());
        self.checkpoint();
        self.doc.adjust.kernel = Some(own);
        let mut text = format!(
            "Measured the blur inside the outline: σ {widest:.1} × {narrowest:.1} px at {:.0}°. Deblur now undoes this instead \
             of the whole photo's blur.",
            spread.angle()
        );
        if let Some(dropped) = self.sync_adjust() {
            text.push(' ');
            text.push_str(&dropped);
        }
        self.notify(text);
    }

    /// Estimate the radial lens term from the corners and anchors, and
    /// apply it, moving the corners and anchors with the photo so that
    /// they stay on the symbol. Never run of its own accord: the user
    /// asks, with the corners and anchors placed.
    fn estimate_lens(&mut self) {
        let Some(photo) = &self.photo else {
            self.fail("Load a photo first.");
            return;
        };
        if photo.crop.is_some() {
            self.fail(
                "A lens bends the photo about its centre, which a cropped photo has lost: show the whole photo, fit \
                 the lens there, and crop again.",
            );
            return;
        }
        let (width, height) = (photo.width, photo.height);
        let (w, h) = (self.derived.func.width as f64, self.derived.func.height as f64);
        let mut pairs: Vec<([f64; 2], [f64; 2])> =
            [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]].into_iter().zip(self.doc.corners).collect();
        pairs.extend(self.doc.anchors.iter().map(|a| (a.anchor.at, a.anchor.center)));
        if pairs.len() < LENS_MIN_PAIRS {
            self.fail(format!(
                "Estimating the lens needs the four corners and at least {} anchors on the symbol; press Find in outline \
                 to place them, or drag the hollow handles onto their patterns.",
                LENS_MIN_PAIRS - 4
            ));
            return;
        }
        let from = self.doc.adjust.k1;
        match estimate_k1(width, height, from, &pairs) {
            Some(estimate) => {
                self.checkpoint();
                self.doc.corners = self.doc.corners.map(|c| remap_point(c, width, height, from, estimate.k1));
                for anchor in &mut self.doc.anchors {
                    anchor.anchor.center = remap_point(anchor.anchor.center, width, height, from, estimate.k1);
                }
                self.doc.adjust.k1 = estimate.k1;
                // Whatever else was found in the photo as it was is dropped.
                let dropped = self.sync_adjust();
                let mut text = format!(
                    "Lens set to {:.3}: the corners and anchors lie {:.2} px from a plane, down from {:.2}, and were moved \
                     with the photo. Press Refine to settle them on it.",
                    estimate.k1, estimate.rms_after, estimate.rms_before
                );
                if let Some(dropped) = dropped {
                    text.push(' ');
                    text.push_str(&dropped);
                }
                self.notify(text);
            }
            None => self.notify(format!(
                "No lens setting fits the corners and anchors better than the current {from:.3}, so it was left as it is."
            )),
        }
    }

    /// Take in a detection a job made ready, and return the notice to
    /// show. A read of it that decoded is loaded whole, its grid and pins
    /// with the outline, where reads are loaded; otherwise it is adopted,
    /// which takes the outline and the anchors and leaves the grid to be
    /// sampled. `what` names the symbol in the notice of a read loaded.
    fn take_in(&mut self, ctx: &egui::Context, mut prepared: Prepared, what: &str) -> String {
        let read = prepared.result.take().filter(|read| self.view.load_reads && read.is_trustworthy());
        // The photo is still adjusted as it was when the read was made: a
        // job outlives no change of that ([`Self::poll_job`]).
        let under = self.photo.as_ref().map(ReadUnder::of);
        match (read, under) {
            (Some(read), Some(under)) => self.load_read_result(*read, prepared.searched_inverted, under, what),
            _ => self.adopt_prepared(ctx, prepared),
        }
    }

    /// Take in what Find in outline found, loading its read or adopting
    /// it ([`Self::take_in`]), or say that nothing was found. `searched`
    /// is the family the outline was searched for first.
    fn finish_find_corners(&mut self, ctx: &egui::Context, found: Option<(Prepared, Option<Family>)>, searched: Family) {
        match found {
            Some((prepared, other_family)) => {
                let text = self.take_in(ctx, prepared, "the symbol inside the outline");
                match other_family {
                    Some(family) => self.notify(format!(
                        "What is inside the outline reads as a {} symbol, not a {} one. {text}",
                        family.label(),
                        searched.label()
                    )),
                    None => self.notify(text),
                }
            }
            None if self.view.own_family_only => self.fail(format!(
                "No {} finder pattern was found inside the outline. Draw the outline loosely around the whole symbol, \
                 so that its finder patterns are inside, and try again; or untick Only look for {} to try the other \
                 families.",
                searched.label(),
                searched.label()
            )),
            None => self.fail(
                "No finder pattern of any family was found inside the outline. Draw the outline loosely around the \
                 whole symbol, so that its finder patterns are inside, and try again; or press Search whole \
                 photo.",
            ),
        }
    }

    /// Place the corners and anchors on a symbol the search found, or on
    /// a placement the detector ranked, refining them on a thread of its
    /// own. `searched_inverted` is the Invert setting it was found under,
    /// and `what` names it for a notice. With `read`, it is read through
    /// to its message first, as Find in outline reads what it finds,
    /// charged as one of as many as `read` says ([`Proposals::among_of`]),
    /// and a read that decodes comes back with it, to be loaded.
    fn adopt_proposal(
        &mut self,
        ctx: &egui::Context,
        found: Detection,
        searched_inverted: bool,
        what: String,
        read: Option<usize>,
        picked: Option<Picked>,
    ) {
        if self.job_busy() {
            return;
        }
        let Some(photo) = &self.photo else {
            self.fail("Load a photo first.");
            return;
        };
        let luma = photo.luma.clone();
        let photo_inverted = photo.adjust.invert;
        let (version, params, opts, func) = (self.doc.version, self.scan_params(), self.opts, self.derived.func.clone());
        self.start_job(ctx, JobKind::Adopt, move || {
            // A detection says how its symbol read against the photo as
            // it was searched: with Invert photo changed since, the
            // symbol reads the other way against the photo as it is.
            let as_it_is = Detection { inverted: found.inverted ^ (searched_inverted != photo_inverted), ..found.clone() };
            let prepared = match read.and_then(|among| read_detection(&luma, as_it_is, &params, &opts.one_of(among))) {
                Some(result) if result.is_trustworthy() => prepared_from_read(&luma, version, result, photo_inverted, &params),
                // A read that came to no message may have ended on
                // another placement than the one picked: the symbol is
                // placed as it was picked, and the read only told of.
                result => Prepared {
                    read: result.as_ref().map(read_note),
                    ..prepare_adoption(&luma, &func, version, found, searched_inverted, photo_inverted, &params)
                },
            };
            JobResult::Adopt(prepared, what, picked)
        });
    }

    /// Refine the corners and every anchor against the photo, on a thread
    /// of its own.
    fn refine(&mut self, ctx: &egui::Context) {
        if self.job_busy() {
            return;
        }
        let Some(photo) = &self.photo else {
            self.fail("Load a photo first.");
            return;
        };
        let luma = photo.luma.clone();
        let func = self.derived.func.clone();
        let (corners, params) = (self.doc.corners, self.view.sample);
        let anchors: Vec<Anchor> = self.doc.anchors.iter().map(|a| a.anchor).collect();
        let already: Vec<f64> = self.doc.anchors.iter().map(|a| a.bled).collect();
        self.start_job(ctx, JobKind::Refine, move || {
            let refined = refine_outline(&luma, &func, corners, anchors.clone(), &already, &params);
            JobResult::Refine { corners, anchors, refined }
        });
    }

    /// Take in what Refine worked out: the corners move, and every anchor
    /// that found its pattern moves onto it, an edge point moved out for
    /// the bleed the symbol was placed expecting remembered as moved so
    /// ([`DocAnchor::bled`]).
    fn finish_refine(&mut self, refined: Option<Refined>) {
        let Some(Refined { refinement: refined, before, after, bloom }) = refined else {
            self.fail("The corners do not form a usable outline.");
            return;
        };
        self.checkpoint();
        self.doc.corners = refined.corners;
        // An edge point is never looked for, only moved for the bleed, and
        // is counted apart from the anchors that moved onto a pattern.
        let (mut moved, mut shifted) = (0, 0);
        for (old, new) in before.iter().zip(&after) {
            if old.center != new.center {
                if new.kind == AnchorKind::Edge {
                    shifted += 1;
                } else {
                    moved += 1;
                }
                let source = self.doc.anchor_at(old.at).map_or(AnchorSource::Detected, |a| a.source);
                let bled = if new.kind == AnchorKind::Edge { bloom } else { 0.0 };
                self.doc.set_anchor_bled(*new, source, bled);
            }
        }
        let (after_count, fixed, before_count) = (refined.after.0, refined.after.1, refined.before.0);
        let change = if after_count >= before_count { "up from" } else { "down from" };
        let mut text = if refined.inverted {
            format!(
                "Refined the corners: with dark and light swapped the fixed patterns match {after_count} of \
                 {fixed}, {change} {before_count}."
            )
        } else {
            format!("Refined the corners: fixed patterns match {after_count} of {fixed}, {change} {before_count}.")
        };
        if moved > 0 {
            text.push_str(&format!(" {moved} of {} anchors moved onto their patterns.", after.len()));
        }
        if bloom != 0.0 {
            // The bleed is into the symbol's dark modules, which on a
            // light-on-dark symbol are the photo's light.
            let what = if (bloom > 0.0) != refined.inverted { "dark thinned" } else { "ink spread" };
            let points = match shifted {
                0 => String::new(),
                1 => ", 1 edge point moved from the edge of the print to the symbol's".into(),
                n => format!(", {n} edge points moved from the edge of the print to the symbol's"),
            };
            text.push_str(&format!(
                " The finder patterns show the {what} by {:.2} of a module at every boundary, and the symbol was placed expecting it{points}.",
                bloom.abs()
            ));
        }
        if refined.inverted {
            text.push_str(if self.photo.as_ref().is_some_and(|p| p.adjust.invert) {
                " Invert photo looks wrong for this photo."
            } else {
                " This looks like a light-on-dark symbol: tick Invert photo before sampling."
            });
        }
        self.notify(text);
    }

    /// Search `block` again with every dear solver's budget at its most
    /// ([`decode_trying_harder`]), on a thread of its own: on a large
    /// block that can take a minute.
    fn try_harder(&mut self, ctx: &egui::Context, block: usize) {
        if self.job_busy() || block >= self.derived.report.blocks.len() {
            return;
        }
        let Some(read) = self.block_read(block) else { return };
        let grid = self.doc.grid.clone();
        let layout = self.derived.layout.clone();
        let pins = self.active_pins();
        let (version, ec, mask, opts) = (self.doc.version, self.derived.ec, self.derived.mask, self.decoding_options());
        let kept = self.harder.as_ref().filter(|kept| kept.holds(version, ec, mask, &opts)).map(|kept| kept.searched.clone()).unwrap_or_default();
        self.start_job(ctx, JobKind::TryHarder(block), move || {
            let (_, searched) = decode_trying_harder(&grid, &layout, mask, &pins, &opts, &kept, &[block]);
            JobResult::Harder { block, read, kept: KeptHarder { version, ec, mask, opts, searched } }
        });
    }

    /// The options the document is decoded under as derived: the decode
    /// options, counted as one of the decodes the document was picked
    /// among and of those that settled its format ([`decoding_options`]).
    /// What a decode made beside the derived one, of a block searched
    /// harder or of pins tried, is made under, so that it is charged as
    /// the derived one is.
    fn decoding_options(&self) -> DecodeOptions {
        decoding_options(&self.doc.picked(&self.opts), self.derived.format_by_brute, &self.derived.func)
    }

    /// The options the document would be decoded under at `level`: as
    /// derived at the level in effect ([`Self::decoding_options`]), and at
    /// another, which only a format set by hand reaches, the decode
    /// options counted as one of the decodes the document was picked
    /// among.
    fn options_at(&self, level: EcLevel) -> DecodeOptions {
        if level == self.derived.ec { self.decoding_options() } else { self.doc.picked(&self.opts) }
    }

    /// What `block` of the layout in effect is read from in the document
    /// now ([`BlockRead`]).
    fn block_read(&self, block: usize) -> Option<BlockRead> {
        let layout = &self.derived.layout;
        let spec = layout.blocks.get(block)?;
        let grid = &self.doc.grid;
        let modules = spec
            .codewords
            .iter()
            .flat_map(|&cw| layout.codeword_modules(cw))
            .map(|p| (grid.get(p.x as usize, p.y as usize), grid.reliability(p.x as usize, p.y as usize)))
            .collect();
        let pins = self.active_pins();
        Some(BlockRead { modules, pins: spec.codewords.iter().map(|cw| pins.get(cw).copied()).collect() })
    }

    /// Take in a block searched harder, and say what came of it: what it
    /// found holds only under the version, level, mask and options it was
    /// searched under, and only while the block reads as it did. Whether
    /// it does is asked of the block's own modules and pins, as `read`
    /// holds them from when the search began: a block the search solved
    /// may be confirmed in the end under padding pins found once it
    /// read, with no search of it as it reads under them, and is not
    /// marked [`BlockReport::tried_harder`]. That was once taken for the
    /// block having changed.
    fn finish_try_harder(&mut self, block: usize, read: BlockRead, kept: KeptHarder) {
        let n = block + 1;
        // The derived values of the document as it is now: it may have
        // changed while the block was searched.
        self.catch_up();
        if !kept.holds(self.doc.version, self.derived.ec, self.derived.mask, &self.decoding_options()) {
            self.notify(format!(
                "The version, EC level, mask or decoding options changed while block {n} was being searched, so the \
                 result was dropped. Press Try harder again."
            ));
            return;
        }
        let changed = self.block_read(block).as_ref() != Some(&read);
        let before = self.derived.report.blocks.get(block).map(|b| b.outcome);
        self.harder = Some(kept);
        self.refresh();
        let Some(after) = self.derived.report.blocks.get(block) else { return };
        let text = try_harder_notice(n, changed, before, after);
        self.notify(text);
    }

    /// Rank every version of the family on the corners, on a thread of
    /// its own. Nothing to rank against without a photo.
    fn rank_versions(&mut self, ctx: &egui::Context) {
        if self.job_busy() {
            return;
        }
        let Some(photo) = &self.photo else { return };
        let luma = photo.luma.clone();
        let (corners, family, params) = (self.doc.corners, self.doc.version.family(), self.view.sample);
        self.start_job(ctx, JobKind::Suggest, move || {
            JobResult::Suggest(suggest_versions(&luma, corners, family, &params))
        });
    }

    /// List the placements of the symbol inside the outline the detector
    /// ranks, on a thread of its own: the first is the one Find in outline
    /// starts from, and the others are what it tries, unseen, when that
    /// one does not decode. Of the family in use, as the detector takes
    /// its family from the version.
    fn list_placements(&mut self, ctx: &egui::Context) {
        if self.job_busy() {
            return;
        }
        let Some(photo) = &self.photo else {
            self.fail("Load a photo first.");
            return;
        };
        let luma = photo.luma.clone();
        let photo_inverted = photo.adjust.invert;
        let (version, corners, params) = (self.doc.version, self.doc.corners, self.scan_params());
        self.start_job(ctx, JobKind::Placements, move || {
            JobResult::Placements { found: detect_candidates(&luma, version, corners, &params), photo_inverted }
        });
    }

    /// Offer the placements the detector ranked. `family` is the one it
    /// looked for.
    fn finish_placements(&mut self, found: Vec<Detection>, photo_inverted: bool, family: Family) {
        if found.is_empty() {
            self.placements = None;
            self.fail(format!(
                "No {} finder pattern was found inside the outline, so there is no placement to list. Draw the outline                  loosely around the whole symbol, so that its finder patterns are inside, and try again.",
                family.label()
            ));
            return;
        }
        let count = found.len();
        self.placements = Some(Placements { found, photo_inverted, adopted: None });
        self.view.stage = Stage::Align;
        self.notify(if count == 1 {
            "The detector makes one placement of the symbol inside the outline, and it is listed under Alignment: pick it              to put the corners and anchors on it."
                .to_owned()
        } else {
            format!(
                "The detector ranks {count} placements of the symbol inside the outline, listed under Alignment best                  first: point at one to see it on the photo, and pick it to put the corners and anchors on it."
            )
        });
    }

    /// Whether a job is still running, in which case say so. The buttons
    /// that start one are disabled meanwhile, so this is only a safeguard.
    fn job_busy(&mut self) -> bool {
        let Some(job) = &self.job else { return false };
        let text = format!("{} is still under way.", job.kind.label());
        self.notify(text);
        true
    }

    /// The step a job is running, if one is.
    fn running(&self) -> Option<JobKind> {
        self.job.as_ref().map(|job| job.kind)
    }

    /// The sampling settings a step that looks for the symbol in the
    /// photo works with: the view's, and, where the binarised view's
    /// threshold is set for the scan, that threshold in place of the
    /// scan's three readings, as the view would show it now, following
    /// the outline while it is automatic. The settings the sample
    /// preview is keyed by are left as they are: the preview reads no
    /// finder patterns.
    fn scan_params(&self) -> SampleParams {
        let mut params = self.view.sample;
        if self.view.binarize_for_scan
            && let Some(photo) = &self.photo
        {
            params.binarizer = self.view.binarize_for(self.doc.corners, photo.width, photo.height).binarizer();
        }
        params
    }

    /// Run `work` on a thread of its own; [`Self::poll_job`] takes its
    /// result in when it arrives. The caller has checked that no job is
    /// running.
    fn start_job(&mut self, ctx: &egui::Context, kind: JobKind, work: impl FnOnce() -> JobResult + Send + 'static) {
        let (sender, receiver) = std::sync::mpsc::channel();
        let cancel = Cancel::new();
        let (ctx, request) = (ctx.clone(), cancel.clone());
        std::thread::spawn(move || {
            let result = request.scope(work);
            // The window may have closed meanwhile, or the job been
            // called off; nothing to do then.
            let _ = sender.send(result);
            ctx.request_repaint();
        });
        self.job = Some(PendingJob {
            kind,
            generation: self.photo_generation,
            version: self.doc.version,
            corners: self.doc.corners,
            anchors: self.doc.anchors.iter().map(|a| a.anchor).collect(),
            adjust: self.photo.as_ref().map(|photo| photo.adjust),
            receiver,
            started: std::time::Instant::now(),
            cancel,
        });
    }

    /// Take in a finished job, if one has finished. A result for a photo
    /// since replaced is dropped; so is one that means something only at
    /// the version the job began under, when the version changed
    /// meanwhile, and one made of the outline the job set out from, when
    /// the outline or the anchors moved meanwhile. A detection carries
    /// its own version, and adopting it reckons with the document as it
    /// is now.
    fn poll_job(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.job else { return };
        let result = match job.receiver.try_recv() {
            Ok(result) => result,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                // The thread ended without a result, which only a panic
                // in it does: say so, rather than let the step vanish.
                let text = format!(
                    "{} stopped on an internal error and changed nothing. That is a fault in this program, worth \
                     reporting with the photo; the outline, the anchors and the grid are as they were.",
                    job.kind.label()
                );
                self.job = None;
                self.fail(text);
                return;
            }
        };
        let job = self.job.take().expect("a job was pending");
        if let JobResult::Harder { block, read, kept } = result {
            self.finish_try_harder(block, read, kept);
            return;
        }
        if job.generation != self.photo_generation || self.photo.is_none() {
            return;
        }
        let version_changed = self.doc.version != job.version;
        // A corner dragged or an anchor moved or dropped while the step
        // ran: a result that would place the outline is of the outline
        // before that, and taking it in would undo the hand's work; and
        // a ranking made inside the outline is of the outline before
        // that, and would be shown as one made on these corners.
        let outline_moved =
            job.corners != self.doc.corners || self.doc.anchors.iter().map(|a| a.anchor).ne(job.anchors.iter().copied());
        // The photo adjusted otherwise while the step ran, by the user or a
        // read taken in meanwhile: what the step read was read off the
        // photo as it was, and a read that decoded only through a copy of
        // it would set the photo's filter and sharpening from what they
        // are now. A change of size or lens has dropped it already.
        let adjusted = self.photo.as_ref().map(|photo| photo.adjust) != job.adjust;
        match result {
            JobResult::Corners(_) | JobResult::Adopt(..) | JobResult::ReadFound { .. } if adjusted => self.notify(format!(
                "{} was dropped: the photo's adjustments changed while it ran, and what it read was read off the photo \
                 as it was. Run it again.",
                job.kind.label()
            )),
            JobResult::Corners(_) if outline_moved => self.notify(
                "The outline or the anchors changed while the symbol inside the outline was being found, so the \
                 result was dropped; press Find in outline again.",
            ),
            JobResult::Corners(found) => self.finish_find_corners(ctx, found, job.version.family()),
            JobResult::Adopt(..) if outline_moved => self.notify(
                "The outline or the anchors changed while the corners were being placed on the symbol found, so the \
                 result was dropped; click the symbol again.",
            ),
            JobResult::Adopt(prepared, what, picked) => {
                let text = self.take_in(ctx, prepared, &what);
                self.mark_picked(picked);
                self.notify(text);
            }
            JobResult::Refine { .. } if version_changed => self.notify(
                "The version changed while the corners were being refined, so the result was dropped; press \
                 Refine again.",
            ),
            // A corner dragged or an anchor moved or dropped while the
            // step ran: the result is of the outline before that, and
            // taking it in would undo the hand's work.
            JobResult::Refine { corners, anchors, .. }
                if corners != self.doc.corners || self.doc.anchors.iter().map(|a| a.anchor).ne(anchors.iter().copied()) =>
            {
                self.notify(
                    "The outline or the anchors changed while they were being refined, so the result was dropped; \
                     press Refine again.",
                );
            }
            JobResult::Refine { refined, .. } => self.finish_refine(refined),
            JobResult::Suggest(_) if version_changed => self.notify(
                "The version changed while the versions were being ranked, so the ranking was dropped; press \
                 Suggest version again.",
            ),
            JobResult::Suggest(_) if outline_moved => self.notify(
                "The outline or the anchors changed while the versions were being ranked, so the ranking was \
                 dropped; press Suggest version again.",
            ),
            JobResult::Suggest(guesses) => self.guesses = Some(guesses),
            JobResult::ReadFound { outlines, reads, photo_inverted } => self.finish_read_found(&outlines, reads, photo_inverted),
            JobResult::Placements { .. } if outline_moved => self.notify(
                "The outline or the anchors changed while the placements inside the outline were being ranked, so \
                 the list was dropped; press Other placements again.",
            ),
            JobResult::Placements { found, photo_inverted } => self.finish_placements(found, photo_inverted, job.version.family()),
            JobResult::Harder { .. } => unreachable!("taken in above, photo or none"),
        }
    }

    /// Search the whole photo for symbols of every family and offer what
    /// is found, as a reader tries every symbology it knows; or for the
    /// family in use alone, when that is asked. Only ever at the user's
    /// asking.
    fn search_symbols(&mut self, ctx: &egui::Context) {
        let Some(photo) = &self.photo else {
            self.fail("Load a photo first.");
            return;
        };
        if self.search.is_some() {
            self.notify("The photo is still being searched.");
            return;
        }
        let photo_inverted = photo.adjust.invert;
        let luma = photo.luma.clone();
        let (params, opts) = (self.scan_params(), self.opts);
        let search = self.view.search;
        let family = self.view.own_family_only.then(|| self.doc.version.family());
        self.start_search(ctx, photo_inverted, family, move || match family {
            Some(family) => search_for_family(&luma, family, &params, &search, &opts),
            None => search_for_every_family(&luma, &params, &search, &opts),
        });
    }

    /// Run `work`, a search of the photo, on a thread of its own;
    /// [`Self::poll_search`] takes in what it finds when it arrives. The
    /// caller has checked that no search is running.
    fn start_search<S: Into<Sought> + Send + 'static>(
        &mut self,
        ctx: &egui::Context,
        photo_inverted: bool,
        family: Option<Family>,
        work: impl FnOnce() -> S + Send + 'static,
    ) {
        let (sender, receiver) = std::sync::mpsc::channel();
        let cancel = Cancel::new();
        let (ctx, request) = (ctx.clone(), cancel.clone());
        std::thread::spawn(move || {
            let found: Sought = request.scope(work).into();
            // The window may have closed meanwhile, or the search been
            // called off; nothing to do then.
            let _ = sender.send(found);
            ctx.request_repaint();
        });
        self.search = Some(PendingSearch {
            generation: self.photo_generation,
            receiver,
            photo_inverted,
            adjust: self.photo.as_ref().map(|photo| photo.adjust),
            family,
            started: std::time::Instant::now(),
            cancel,
        });
    }

    /// Take in a finished search, if one has finished. A result for a
    /// photo since replaced is dropped.
    fn poll_search(&mut self, ctx: &egui::Context) {
        let Some(search) = &self.search else { return };
        let found = match search.receiver.try_recv() {
            Ok(found) => found,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                // The thread ended without a result, which only a panic
                // in it does: say so, rather than let the search vanish.
                self.search = None;
                self.fail(
                    "The search of the photo for symbols stopped on an internal error and found nothing. That is a \
                     fault in this program, worth reporting with the photo; draw an outline around the symbol by \
                     hand and press Find in outline.",
                );
                return;
            }
        };
        let search = self.search.take().expect("a search was pending");
        if search.generation != self.photo_generation || self.photo.is_none() {
            return;
        }
        // What the search read, it read off the photo as it began.
        // Inverted since, a symbol reads the other way, as is reckoned
        // from the photo's polarity then ([`Proposals::photo_inverted`]);
        // adjusted otherwise, whether a symbol decoded says nothing of how
        // it reads now, and none is picked of itself for it. Picked so, a
        // symbol was said to decode and then read off the photo as it is,
        // where it might not.
        let adjusted = self
            .photo
            .as_ref()
            .zip(search.adjust)
            .is_some_and(|(photo, then)| Adjust { invert: then.invert, ..photo.adjust } != then);
        let charged = found.charged();
        self.leads = found.leads;
        let mut found = found.found;
        if adjusted {
            found.iter_mut().for_each(|found| found.decodes = None);
        }
        self.finish_search(ctx, found, charged, search.photo_inverted, search.family, adjusted);
    }

    /// Offer what a search of the whole photo found. `family` is the one
    /// family it looked for, when it did not look for every one. Where
    /// reads are loaded, a symbol that decodes as found is picked of
    /// itself ([`symbol_to_load`]), as a click on it in the list picks
    /// it: read through to its message, and loaded. `charged` says what
    /// the reads the search made of some of them were charged as one of
    /// ([`Proposals::charged`]). `adjusted` says the photo's adjustments
    /// changed while the search ran, and none was asked whether it
    /// decodes as the photo is now.
    fn finish_search(
        &mut self,
        ctx: &egui::Context,
        found: Vec<Detection>,
        charged: Vec<Charged>,
        photo_inverted: bool,
        family: Option<Family>,
        adjusted: bool,
    ) {
        if found.is_empty() && !self.leads.is_empty() {
            // Nothing fitted, and finders to say so of: listed, with no
            // symbols above them.
            self.proposals = None;
            self.view.stage = Stage::Align;
            let places = if self.leads.len() == 1 { "1 place".to_owned() } else { format!("{} places", self.leads.len()) };
            let came_to = leads_came_to(&self.leads);
            self.fail(format!(
                "No symbol was found in the photo, but finder patterns arranged like a symbol's were found in {places}, \
                 and {came_to}. They're listed under Alignment: pick one to place the \
                 outline around it, correct the outline by hand if needed, and press Find in outline."
            ));
            return;
        }
        if found.is_empty() {
            self.proposals = None;
            self.fail(match family {
                Some(family) => format!(
                    "No {} symbol was found in the photo. Drag the corners loosely around the symbol and press Find in \
                     outline, or untick Only look for {} to search for the other families.",
                    family.label(),
                    family.label()
                ),
                None => "No symbol of any family was found in the photo. Drag the corners loosely around the symbol and \
                         press Find in outline."
                    .to_owned(),
            });
            return;
        }
        let count = found.len();
        // A symbol the outline was put on from the list before is still
        // the one it lies on in the new list.
        let adopted = self
            .proposals
            .as_ref()
            .and_then(|before| proposal_in_use(before, &self.doc.corners))
            .and_then(|_| found.iter().position(|f| same_outline(&f.corners, &self.doc.corners)));
        // No step is under way to wait for but one begun during the
        // search, which is left to finish.
        let pick = (self.view.load_reads && self.job.is_none())
            .then(|| symbol_to_load(&found, adopted, &self.doc.corners, |i| found[i].decodes == Some(true)))
            .flatten();
        self.proposals = Some(Proposals { found, photo_inverted, adopted, reads: None, charged });
        self.view.stage = Stage::Align;
        let what = if count == 1 { "Found 1 symbol".to_owned() } else { format!("Found {count} symbols") };
        let how = match pick {
            Some(i) => format!(
                "outlined on the photo. Symbol {} {} and is being read into the grid; to use another, pick it under \
                 Alignment or click its outline.",
                i + 1,
                // A placement read through during the search decoded so,
                // through whatever retry it took, and not as it was found.
                if self.proposals.as_ref().is_some_and(|proposals| proposals.read_in_search(&proposals.found[i])) {
                    "decoded during the search"
                } else {
                    "decodes as found"
                }
            ),
            None => "outlined on the photo. Pick one under Alignment, or click its outline, to place the corners on it; \
                     or press Read all to read every one through to its message."
                .to_owned(),
        };
        let came_to = leads_came_to(&self.leads);
        let beside = match self.leads.len() {
            0 => String::new(),
            1 => format!(" Finder patterns in 1 other place look like a symbol's, but {came_to}; they're listed below the symbols."),
            n => format!(" Finder patterns in {n} other places look like a symbol's, but {came_to}; they're listed below the symbols."),
        };
        let stale = if adjusted && self.view.load_reads {
            " The photo's adjustments changed while the search ran, so none was read into the grid of itself."
        } else {
            ""
        };
        self.notify(format!("{what}, {how}{stale}{beside}"));
        if let Some(i) = pick {
            self.use_proposal(ctx, i);
        }
    }

    /// Read every symbol the search found through to its message, each
    /// charged as the command line charges it ([`Proposals::among_of`]),
    /// on a thread of their own and side by side.
    fn read_found(&mut self, ctx: &egui::Context) {
        if self.job_busy() {
            return;
        }
        let (Some(photo), Some(proposals)) = (&self.photo, &self.proposals) else {
            self.fail("There are no symbols found to read: press Search whole photo first.");
            return;
        };
        let luma = photo.luma.clone();
        let photo_inverted = photo.adjust.invert;
        // A detection says how its symbol read against the photo as it
        // was searched: with Invert photo changed since, the symbol reads
        // the other way against the photo as it is.
        let flipped = proposals.photo_inverted != photo_inverted;
        let found: Vec<(Detection, usize)> = proposals
            .found
            .iter()
            .map(|found| (Detection { inverted: found.inverted ^ flipped, ..found.clone() }, proposals.among_of(found)))
            .collect();
        let (params, opts) = (self.scan_params(), self.opts);
        self.start_job(ctx, JobKind::ReadFound, move || {
            let outlines = found.iter().map(|(found, _)| found.corners).collect();
            let reads = work::map(&found, |(found, among)| read_detection(&luma, found.clone(), &params, &opts.one_of(*among)));
            JobResult::ReadFound { outlines, reads, photo_inverted }
        });
    }

    /// Take in what Read all read: each symbol in the list then says how
    /// its read stands, the parts of a structured append message among
    /// those that decoded are put in the sequence, and, where reads are
    /// loaded, one that decoded is ([`symbol_to_load`]).
    fn finish_read_found(&mut self, outlines: &[[[f64; 2]; 4]], reads: Vec<Option<ReadResult>>, photo_inverted: bool) {
        let name = self.photo.as_ref().map_or_else(String::new, |photo| photo.name.clone());
        let same_list = |proposals: &&mut Proposals| proposals.found.iter().map(|found| &found.corners).eq(outlines);
        let Some(proposals) = self.proposals.as_mut().filter(same_list) else {
            self.notify("The list of symbols found changed while they were being read, so the reads were dropped; press Read all again.");
            return;
        };
        let decoded = reads.iter().flatten().filter(|read| read.is_trustworthy()).count();
        // The parts of a structured append message among the reads go
        // into the sequence: of the message it holds, or, holding none,
        // of the one most of them are of ([`Sequence::add_all`]). Parts
        // of another message read from the same photo are no parts of
        // it, and are said instead: they once went in with the rest, a
        // part of one message taking the place of the other's by its
        // index.
        let found: Vec<(usize, Part)> = reads
            .iter()
            .enumerate()
            .filter_map(|(i, read)| {
                let read = read.as_ref().filter(|read| read.is_trustworthy())?;
                Part::from_report(&read.report, format!("{name}, symbol {}", i + 1)).map(|part| (i, part))
            })
            .collect();
        let left_out = self.view.sequence.add_all(found.iter().map(|(_, part)| part.clone()).collect());
        let parts = found.len() - left_out.len();
        let left_out: Vec<usize> = found.iter().filter(|(_, part)| left_out.contains(part)).map(|(i, _)| i + 1).collect();
        let total = reads.len();
        // As the photo is adjusted now, which is as it was when the reads
        // were made: a change meanwhile dropped them ([`Self::poll_job`]).
        let Some(under) = self.photo.as_ref().map(ReadUnder::of) else { return };
        let decodes = |i: usize| reads.get(i).and_then(Option::as_ref).is_some_and(ReadResult::is_trustworthy);
        let load = symbol_to_load(&proposals.found, proposals.adopted, &self.doc.corners, decodes).filter(|_| self.view.load_reads);
        proposals.reads = Some(FoundReads { results: reads, photo_inverted, under });
        let mut text =
            format!("Read {decoded} of {total} symbols found through to a message: the list under Alignment says how each read stands.");
        if parts > 0 {
            let what = if parts == 1 { "1 part".to_owned() } else { format!("{parts} parts") };
            text.push_str(&format!(" {what} of a structured append message went into the sequence."));
        }
        if !left_out.is_empty() {
            let symbols = left_out.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ");
            text.push_str(&if left_out.len() == 1 {
                format!(
                    " Symbol {symbols} carries a part of another structured append message, by its count or parity, and it \
                     stayed out of the sequence: to collect that message, clear the sequence, then pick each of its symbols \
                     and press Add to sequence."
                )
            } else {
                format!(
                    " Symbols {symbols} carry parts of another structured append message or more, by their count or parity, \
                     and they stayed out of the sequence: to collect those, clear the sequence, then pick each symbol and \
                     press Add to sequence."
                )
            });
        }
        match load.and_then(|i| self.load_read(i)) {
            Some(loaded) => {
                text.push(' ');
                text.push_str(&loaded);
                if decoded > 1 {
                    text.push_str(" Pick another in the list to load that one instead.");
                }
            }
            None if decoded > 0 && self.view.load_reads => text.push_str(
                " The symbol the outline is on did not decode, so its grid is left as it is: pick one that did in the \
                 list to load that instead.",
            ),
            None if decoded > 0 => text.push_str(
                " Tick Load reads that decode, and pick one in the list, to have its grid written as it was read.",
            ),
            None => {}
        }
        self.notify(text);
    }

    /// Put the outline on the `i`-th symbol the search found, as picking
    /// it in the list or on the photo asks. Where reads are loaded, the
    /// symbol is read through to its message on the way, when Read all
    /// has not read it already, and a read that decodes is loaded: the
    /// outline, the anchors and the grid at once.
    fn use_proposal(&mut self, ctx: &egui::Context, i: usize) {
        let proposal = self.proposals.as_ref().and_then(|p| p.found.get(i).map(|f| (f.clone(), p.photo_inverted, p.reads.is_none(), p.among_of(f))));
        let Some((found, photo_inverted, unread, among)) = proposal else {
            self.fail("That symbol is no longer in the list; press Search whole photo again.");
            return;
        };
        // Read through to a message by Read all, it is loaded as that
        // read it, with nothing left to refine.
        if self.view.load_reads && self.read_decoded(i) {
            if !self.job_busy()
                && let Some(text) = self.load_read(i)
            {
                self.notify(text);
            }
            return;
        }
        // One Read all read to no message is not read again for it. The
        // symbol is marked as the one in use once the outline is put on
        // it ([`Picked`]).
        self.adopt_proposal(
            ctx,
            found,
            photo_inverted,
            format!("symbol {}", i + 1),
            (self.view.load_reads && unread).then_some(among),
            Some(Picked::Proposal(i)),
        );
    }

    /// Mark the row of a list under Alignment that a step has just put
    /// the outline on as the one in use ([`Picked`]). The list may have
    /// been cleared meanwhile, and then there is nothing to mark.
    fn mark_picked(&mut self, picked: Option<Picked>) {
        match picked {
            Some(Picked::Proposal(i)) => {
                if let Some(proposals) = self.proposals.as_mut().filter(|proposals| i < proposals.found.len()) {
                    proposals.adopted = Some(i);
                }
            }
            Some(Picked::Placement(i)) => {
                if let Some(placements) = self.placements.as_mut().filter(|placements| i < placements.found.len()) {
                    placements.adopted = Some(i);
                }
            }
            None => {}
        }
    }

    /// Whether Read all read the `i`-th symbol found through to a message.
    fn read_decoded(&self, i: usize) -> bool {
        let reads = self.proposals.as_ref().and_then(|proposals| proposals.reads.as_ref());
        reads.and_then(|reads| reads.results.get(i)?.as_ref()).is_some_and(ReadResult::is_trustworthy)
    }

    /// Put the outline on a symbol Read all read, and make the document
    /// what was read there: the symbol's version, the modules as they
    /// were sampled, and the pins the read was made under. Returns the
    /// notice to show; `None` when there is no read of it to load. The
    /// caller has seen that no step is under way, which would lay what
    /// it works out over what is loaded here.
    fn load_read(&mut self, i: usize) -> Option<String> {
        self.photo.as_ref()?;
        let proposals = self.proposals.as_mut()?;
        let reads = proposals.reads.as_ref()?;
        let read = reads.results.get(i)?.clone()?;
        let (read_inverted, under) = (reads.photo_inverted, reads.under);
        proposals.adopted = Some(i);
        Some(self.load_read_result(read, read_inverted, under, &format!("symbol {}", i + 1)))
    }

    /// Make the document what `read` read: the outline and anchors on its
    /// symbol, the symbol's version, the modules as they were sampled,
    /// but for the fixed patterns, which are filled in as the standard
    /// has them, and the pins the read was made under. `read_inverted` and
    /// `under` are the photo's Invert setting and adjustments when the
    /// read was made, and `what` names the symbol for the notice, which
    /// comes back.
    fn load_read_result(&mut self, read: ReadResult, read_inverted: bool, under: ReadUnder, what: &str) -> String {
        self.checkpoint();
        let (version, decoded, padding_pinned) = (read.version(), read.is_trustworthy(), read.padding_pinned());
        let (verified, blocks) = (read.report.blocks_verified(), read.report.blocks.len());
        let mut found = read.detection;
        if read.reinterpreted.is_some() {
            // Read as the other QR Code model, whose anchors are its
            // finders alone.
            found.version = version;
            found.anchors.retain(|a| a.kind != AnchorKind::Alignment);
        }
        let pins_ec = (!read.pins.is_empty()).then_some(read.report.ec);
        // A read made under another level and mask than the format bits
        // settle, the blocks having borne out a word next nearest them,
        // keeps that level and mask: left to its bits, the grid would be
        // decoded here under the word they spell, and not read.
        let format = match read.analysis.format.best() {
            Some(best)
                if read.reinterpreted.is_none()
                    && read.analysis.format.is_reliable()
                    && (best.fields.ec, best.fields.mask) != (read.report.ec, read.report.mask) =>
            {
                FormatSel::Manual { ec: read.report.ec, mask: read.report.mask }
            }
            _ => FormatSel::Auto,
        };
        // The fixed patterns are written as the standard has them, as a
        // blank grid has them, whatever the photo showed of them: the
        // read placed the symbol and decoded it, so they have no more to
        // say, and one of their modules misread or left unknown would be
        // carried into the grid for nothing.
        let func = if self.derived.func.version == version { self.derived.func.clone() } else { Arc::new(FunctionMap::new(version)) };
        let mut grid = read.grid;
        func.fill_fixed(&mut grid);
        // The read is the pick of every read its search made, and the
        // grid and pins with it: a decode of them is one of as many. A
        // brute force of the format among them is made again by the
        // decode of the document where the format bits cannot settle it,
        // and counted there. The padding pins it took carry what they
        // were picked among besides, the passes of their recovery.
        let brute = format == FormatSel::Auto && !analyze(&grid, &func).format.is_reliable();
        let grid_among = read.report.tries.div_ceil(if brute { formats(&func) } else { 1 }).max(1);
        self.doc = Document { grid, pins: read.pins, pins_ec, grid_among, format, ..self.doc.blank_like(version) };
        self.view.selected = None;
        self.view.highlight_block = None;
        self.view.table_block = None;
        self.view.picked_char = None;
        self.view.picked_segment = None;
        self.view.pad_from = None;
        self.view.edit_view.auto_fit = true;
        self.brute = None;
        self.guesses = None;
        self.model = None;
        self.preview = None;
        self.dirty = true;
        self.set_photo_invert(read_inverted ^ found.inverted);
        // Laid on the backdrop it was read on, as it is inverted as it
        // read: Read all's reads outlast a change to Transparent parts,
        // and a symbol in light ink on transparency that read on grey is
        // gone on white, for Sample and the eye alike. It was once left
        // as it was set.
        let backdrop = under.adjust.backdrop;
        let backdrop_set = self.set_photo_backdrop(backdrop);
        self.doc.corners = found.corners;
        self.doc.anchors = doc_anchors(&found);
        self.detection = Some(found);
        // What read this symbol is what Sample would have to use to read
        // it the same again: the kernel and the threshold rule, the
        // modules sharpened or left as read, and the photo enlarged and
        // sharpened as the copy of it was that the read was made through.
        let mut through = None;
        if decoded {
            let sharpen_modules = read.sharpen_modules;
            let band = if read.unsure_kept { 0.0 } else { self.view.sample.band };
            self.view.sample = SampleParams { kernel: read.kernel, sharpen_modules, band, ..self.view.sample.under(read.rule) };
            through = self.adjust_as_read(read.enhance, read.deblurred, under);
        }
        // The stage is left as it is: a read loaded from the Align stage
        // is looked over there, on the photo, and the grid is a click away.
        let how = match (decoded, padding_pinned) {
            (true, true) => "it decodes with its padding pinned, and the pins are in place".to_owned(),
            (true, false) => "it decodes".to_owned(),
            (false, _) => format!("{verified} of {blocks} blocks verified"),
        };
        let mut text = format!(
            "Loaded {what} as it was read, {} {version}: {how}. The outline and anchors are on it and the grid holds \
             its modules as sampled, with its fixed patterns filled in (undoable).",
            version.family().label()
        );
        if backdrop_set {
            text.push_str(&format!(
                " Under Photo, Transparent parts was set to {} to match the read.",
                backdrop.label().to_lowercase()
            ));
        }
        if let Some(through) = through {
            text.push(' ');
            text.push_str(&through);
        }
        text
    }

    /// Take a detection a job refined and made ready as the document's
    /// outline and anchors, and adopt its version when the symbol's own
    /// bits or a clean fit vouch for it, or when it is of another family.
    /// A symbol that read light on dark has the photo inverted for it, as
    /// a reader retries inverted of its own accord. Returns the notice to
    /// show. `ctx` is for a step this starts on a thread of its own,
    /// where the versions are to be ranked again.
    fn adopt_prepared(&mut self, ctx: &egui::Context, prepared: Prepared) -> String {
        let Prepared { found, agreement: (matching, fixed), guesses, searched_inverted, read, result: _ } = prepared;
        let Some(photo) = &self.photo else { return String::new() };
        // Whether the version found is adopted is judged against the
        // document as it is now, which may have changed while the job
        // ran. The ranking the job made serves when it was made for the
        // family in use; otherwise, which takes a change of family during
        // the job, the versions are ranked again, on a thread of their
        // own as Suggest version ranks them, once the outline is placed:
        // ranked here, they would hold the window for as long as that
        // step takes.
        let decoded = read.as_ref().is_some_and(|r| r.trustworthy);
        // What a read that decoded was made through, of the photo as it
        // is adjusted now.
        let through = read.as_ref().filter(|read| read.trustworthy).map(|read| (read.enhance, read.deblurred, ReadUnder::of(photo)));
        let adopt = adopts_version(&found, self.doc.version, decoded);
        let suggest = found.version != self.doc.version && !adopt;
        let family = self.doc.version.family();
        let guesses = guesses.filter(|_| suggest).and_then(|(ranked, guesses)| (ranked == family).then_some(guesses));
        let rank_again = suggest && guesses.is_none();
        self.checkpoint();
        if adopt {
            self.doc = self.doc.blank_like(found.version);
            self.view.selected = None;
            self.view.highlight_block = None;
            self.view.table_block = None;
            self.view.edit_view.auto_fit = true;
            self.brute = None;
            self.guesses = None;
            self.model = None;
            self.preview = None;
            self.dirty = true;
        }
        // The symbol read light on dark against the photo as it was
        // searched: invert the photo for it, so that the sample preview and
        // Sample read it the right way round.
        let want_inverted = searched_inverted ^ found.inverted;
        let photo_flipped = self.set_photo_invert(want_inverted);
        self.doc.corners = found.corners;
        self.doc.anchors = if adopt || found.version == self.doc.version {
            doc_anchors(&found)
        } else {
            Vec::new()
        };

        let plural = |n: usize, what: &str| if n == 1 { format!("1 {what}") } else { format!("{n} {what}s") };
        let mut parts = vec![plural(found.count(AnchorKind::Finder), "finder pattern")];
        let alignments = found.count(AnchorKind::Alignment);
        if alignments > 0 {
            parts.push(plural(alignments, "alignment pattern"));
        }
        if found.count(AnchorKind::FinderSub) > 0 {
            parts.push("the finder sub-pattern".into());
        }
        let timings = found.count(AnchorKind::Timing);
        if timings > 0 {
            parts.push(plural(timings, "timing module"));
        }
        let edges = found.count(AnchorKind::Edge);
        if edges > 0 {
            parts.push(plural(edges, "edge point"));
        }
        let what = list_phrase(&parts);
        let polarity = if found.inverted { " with dark and light swapped" } else { "" };
        let mut text = format!("Found {what}; fixed patterns match {matching} of {fixed}{polarity}.");
        if let Some(with) = found.found_with() {
            text.push_str(&format!(" Found only with {with}."));
        }
        if found.partial() > 0 {
            text.push_str(&format!(" {} partly damaged, placed from the undamaged part.", plural(found.partial(), "pattern")));
        }
        let inferred = found.anchors.iter().filter(|a| a.inferred).count();
        if inferred > 0 {
            text.push_str(&format!(" {} not found and placed between neighbours.", plural(inferred, "alignment pattern")));
        }
        if !found.edged_corners.is_empty() {
            text.push_str(&format!(" {} placed by tracing the symbol's edge.", plural(found.edged_corners.len(), "corner")));
        }
        match found.info.named_version {
            // A word read outright confirms; one that took correcting
            // only names, as it would beside print that is no symbol.
            Some(named) if named == found.version => {
                let what = if found.version.family() == Family::Qr { "version" } else { "format" };
                let says = if found.version_confirmed() { "confirms" } else { "names" };
                text.push_str(&format!(" The {what} information {says} the version."));
            }
            Some(named) => text.push_str(&format!(
                " The symbol's own information names version {named}, but its fixed patterns did not fit there."
            )),
            None => {}
        }
        if adopt {
            text.push_str(&format!(" Switched to version {}, which the photo fits.", found.version));
        }
        if suggest {
            text.push_str(&format!(
                " Looks like version {} rather than {}: {}.",
                found.version,
                self.doc.version,
                if rank_again {
                    "the versions are being ranked on these corners, to pick from under Suggest version"
                } else {
                    "pick it under Suggest version"
                }
            ));
        }
        if let Some(guesses) = guesses {
            self.guesses = Some(guesses);
        }
        if found.turns != 0 {
            text.push_str(" The outline was turned so that TL is the symbol's own top left.");
        }
        if let Some(read) = read {
            text.push_str(&format!(
                " Read through: {} of {} blocks verified{}.",
                read.verified,
                read.blocks,
                match (read.trustworthy, read.padding_pinned) {
                    (true, true) => {
                        ", the message decodes once its padding is pinned: after sampling, press Pin the padding \
                         under Padding recovery and pins"
                    }
                    (true, false) => ", the message decodes",
                    (false, _) => "",
                }
            ));
            if let Some(other) = read.reinterpreted {
                text.push_str(&format!(" The format bits name {}, and the grid reads as one.", other.family().label()));
            }
            // The kernel and threshold rule of a read that decoded are
            // the ones that read this photo, and Sampling takes them
            // over. Those of a read that did not are only the best of
            // attempts that all failed, and say nothing: the settings
            // stay as the user has them.
            if read.trustworthy && read.kernel != self.view.sample.kernel {
                self.view.sample.kernel = read.kernel;
                text.push_str(&format!(
                    " Only the {} sampling kernel decoded it, so Sampling now uses that.",
                    read.kernel.label().to_lowercase()
                ));
            }
            if read.trustworthy && read.rule != self.view.sample.rule() {
                self.view.sample = self.view.sample.under(read.rule);
                text.push_str(match read.rule {
                    Rule::Neighbourhood => " Only the local threshold decoded it, so Sampling now uses that.",
                    Rule::Level => " Only one threshold for the whole symbol decoded it, so Sampling now uses that.",
                    Rule::Surface => {
                        " Only a threshold that lies as a surface across the symbol decoded it, so Sampling now \
                         uses that."
                    }
                });
            }
            // Sharpening of the modules is as Sampling had it when the
            // read began, and is put back where it was changed since.
            if read.trustworthy && read.sharpen_modules != self.view.sample.sharpen_modules {
                self.view.sample.sharpen_modules = read.sharpen_modules;
                text.push_str(if read.sharpen_modules {
                    " It was read with its modules sharpened, so Sampling sharpens them again."
                } else {
                    " It was read with its modules left as they read, so Sampling leaves them so again."
                });
            }
            // A read that stands only with its unsure modules kept is
            // sampled again with no band at all.
            if read.trustworthy && read.unsure_kept && self.view.sample.band > 0.0 {
                self.view.sample.band = 0.0;
                text.push_str(
                    " Only with the modules its unsure band would leave unknown kept at the colour they fell on did it \
                     decode, so Sampling's unsure band is now nought: those modules are guesses the decoder doubts first.",
                );
            }
        }
        if found.mirrored {
            text.push_str(if found.mirrored_by_decode {
                " The photo shows the symbol mirrored, as from the back: its format bits were unreadable, but \
                 only the mirror image decodes, so the corners were placed in mirror order."
            } else {
                " The photo shows the symbol mirrored, as from the back: the corners were placed in mirror order, so \
                 the grid reads it the right way round."
            });
        }
        if photo_flipped {
            text.push_str(if want_inverted {
                " This is a light-on-dark symbol, so Invert photo was turned on."
            } else {
                " Invert photo looked wrong for this photo and was turned off."
            });
        } else if found.inverted && want_inverted {
            text.push_str(" This is a light-on-dark symbol, read with Invert photo on.");
        }
        self.detection = Some(found);
        // And so with the photo it was read through: a symbol that
        // decoded only through a copy of the photo enlarged or sharpened
        // has the photo so adjusted, and samples the same way. Last,
        // since the outline, the anchors and the detection, placed
        // above, go with the photo to its new size.
        if let Some((enhance, deblurred, under)) = through
            && let Some(note) = self.adjust_as_read(enhance, deblurred, under)
        {
            text.push(' ');
            text.push_str(&note);
        }
        // Last of all, once the outline lies where the ranking is to be
        // made and the photo is at the size it is read at: the step
        // takes the corners as they stand when it starts, and a change
        // of the photo's size would call it off.
        if rank_again {
            self.rank_versions(ctx);
        }
        text
    }

    fn apply(&mut self, ctx: &egui::Context, action: Action) {
        // An edit made on the canvas this frame, or an action before
        // this one, may have left the document ahead of what follows
        // from it.
        self.catch_up();
        match action {
            Action::OpenImage => {
                let picked = rfd::FileDialog::new()
                    .add_filter("Images", qr_forensics::imaging::photo::EXTENSIONS)
                    .pick_file();
                if let Some(path) = picked {
                    self.load_path(ctx, &path);
                }
            }
            Action::PasteImage => self.paste_image(ctx),
            Action::ClosePhoto => {
                self.history.photo_changed(&self.doc);
                self.photo = None;
                self.model = None;
                self.preview = None;
                self.guesses = None;
                self.detection = None;
                self.proposals = None;
                self.leads.clear();
                self.placements = None;
                self.search = None;
                self.drop_photo_job();
                self.photo_generation += 1;
                self.view.stage = Stage::Edit;
                // The photo's ink goes with the photo.
                if self.view.ink == Ink::Photo {
                    self.view.ink = Ink::Module(Module::Dark);
                }
            }
            Action::NewBlank => {
                self.begin_edit();
                self.doc = self.doc.blank_like(self.doc.version);
                self.brute = None;
                self.view.pad_from = None;
            }
            Action::SetVersion(version) => self.set_version(version),
            Action::ReinterpretAs(version) => self.reinterpret_as(version),
            Action::Undo => {
                // A stroke under way took its undo point as it began, and
                // that is the point taken back here: the stroke ends, so
                // that what the pointer paints from here on, still held
                // down, is a stroke of its own with a point of its own.
                // Painted on as part of the old stroke, it had none, and
                // could neither be undone nor survive a redo.
                self.view.stroke = None;
                if self.history.undo(&mut self.doc) {
                    self.dirty = true;
                    self.view.edit_view.auto_fit |= self.doc.version != self.derived.func.version;
                    if let Some(text) = self.sync_adjust() {
                        self.notify(text);
                    }
                }
            }
            Action::Redo => {
                // As for Undo: the stroke under way ends here.
                self.view.stroke = None;
                if self.history.redo(&mut self.doc) {
                    self.dirty = true;
                    self.view.edit_view.auto_fit |= self.doc.version != self.derived.func.version;
                    if let Some(text) = self.sync_adjust() {
                        self.notify(text);
                    }
                }
            }
            Action::FillFixed => {
                self.begin_edit();
                self.derived.func.fill_fixed(&mut self.doc.grid);
            }
            Action::ClearData => {
                self.begin_edit();
                self.doc.grid = self.derived.func.blank_grid();
                self.view.pad_from = None;
            }
            Action::Transform(transform) => {
                self.begin_edit();
                let (width, height) = (self.doc.grid.width(), self.doc.grid.height());
                self.doc.grid = match transform {
                    Transform::RotateCw => self.doc.grid.rotated_cw(),
                    Transform::Rotate180 => self.doc.grid.rotated_180(),
                    Transform::Mirror(mirror) => self.doc.grid.mirrored(mirror),
                    Transform::Invert => self.doc.grid.inverted(),
                    Transform::Symmetry(symmetry) => self.doc.grid.symmetric(symmetry),
                    Transform::SymmetryInverted(symmetry) => self.doc.grid.symmetric(symmetry).inverted(),
                };
                let mut notes = Vec::new();
                // The outline and the anchors say where the grid's
                // modules lie in the photo, and the modules have just
                // moved: they move with them, so that the photo under
                // the grid, and whatever is sampled from it next, is
                // still the photo of these modules. The corner of the
                // photo's symbol that was the grid's top left becomes
                // whichever corner the turn takes it to, and an anchor
                // names the module its pattern has gone to. One that
                // lands where the version has no pattern of its kind, as
                // a finder pattern turned into the corner without one
                // does, has no handle there and nothing to be refined
                // against, and goes.
                if let Some(symmetry) = transform.symmetry() {
                    self.doc.corners = symmetry.map_corners(self.doc.corners);
                    let sites = anchor_sites(&self.derived.func);
                    let before = self.doc.anchors.len();
                    self.doc.anchors = std::mem::take(&mut self.doc.anchors)
                        .into_iter()
                        .filter_map(|mut anchor| {
                            let at = symmetry.map_point(width, height, anchor.anchor.at);
                            anchor.anchor.at = at;
                            let same_place = |site: &&AnchorSite| (site.at[0] - at[0]).abs() < 1e-6 && (site.at[1] - at[1]).abs() < 1e-6;
                            // A point traced along the edge is on the
                            // edge wherever the turn takes it, and a
                            // custom anchor on its module. One the turn
                            // sets on a module that has a pattern of its
                            // own becomes that pattern's anchor, as a
                            // grasp on the pattern makes one, so that the
                            // module has one handle and not two.
                            let sited = match anchor.anchor.kind {
                                AnchorKind::Edge => true,
                                AnchorKind::Custom | AnchorKind::Data => {
                                    if let Some(site) = sites.iter().find(same_place) {
                                        anchor.anchor.kind = site.kind;
                                    }
                                    true
                                }
                                kind => sites.iter().filter(same_place).any(|site| site.kind == kind),
                            };
                            sited.then_some(anchor)
                        })
                        .collect();
                    let dropped = before - self.doc.anchors.len();
                    if dropped > 0 {
                        let what = if dropped == 1 { "1 anchor was".to_owned() } else { format!("{dropped} anchors were") };
                        notes.push(format!(
                            "The outline and anchors were turned with the grid. {what} removed: the turn left no \
                             pattern of the same kind underneath."
                        ));
                    }
                }
                // Pins assert values of codewords whose modules have just
                // moved or changed colour, so they no longer mean anything.
                if !self.doc.pins.is_empty() {
                    let count = self.doc.pins.len();
                    self.doc.pins.clear();
                    self.doc.pins_ec = None;
                    notes.push(format!("Removed {count} pins: they referred to the grid before the transform."));
                }
                if !notes.is_empty() {
                    self.notify(notes.join(" "));
                }
            }
            Action::Sample { only_unknown, scope } => {
                self.update_preview();
                if self.preview.is_none() {
                    self.fail("Load a photo and place the corners first.");
                    return;
                }
                // One undo point takes back the modules and any inversion
                // of the photo made for them.
                self.begin_edit();
                let whole = scope == SampleScope::All;
                // The fixed patterns reading inverted means a light-on-dark
                // symbol, or an Invert setting that is wrong for the photo:
                // either way the photo is inverted before sampling, as a
                // reader retries inverted of its own accord. Only for a
                // sampling of the whole grid: a part of it is sampled to
                // go with the rest as it stands, and is told of below.
                let mut flipped = None;
                if whole && self.view.auto_invert && self.preview.as_ref().is_some_and(|p| looks_inverted(p.fixed)) {
                    let invert = !self.doc.adjust.invert;
                    if self.set_photo_invert(invert) {
                        self.update_preview();
                        flipped = Some(invert);
                    }
                }
                let Some(preview) = &self.preview else {
                    self.fail("Load a photo and place the corners first.");
                    return;
                };
                let classified = preview.classified.clone();
                let sampled_turned = preview.turned;
                let inverted = self.inverted_note();
                let func = self.derived.func.clone();
                // The preview is keyed on the function map's size, so
                // this cannot differ; were it ever to, the modules read
                // would be those of another grid, and none is taken.
                if classified.modules.len() != func.width * func.height {
                    self.fail("The sample of the photo is not of this grid's size; move a corner and sample again.");
                    return;
                }
                // The fixed patterns as the photo shows them are the one
                // thing that says whether the outline lies on the symbol
                // the right way round. Locked, they would be painted as
                // specified, and a grid sampled a quarter turn out, or
                // from the back, would look sound and decode to nothing
                // with no word of why. So they are judged as sampled
                // first, and when they fit the grid turned or reflected
                // they are left as sampled: the grid then shows it, and
                // the Paint section offers the transform that puts it
                // right, which takes the outline with it.
                let turned = sampled_turned.filter(|_| whole);
                let lock_fixed = self.view.lock_fixed && turned.is_none();
                let within = self.scope_mask(scope);
                let (writes, changed) = sample_writes(&self.doc.grid, &func, &classified, only_unknown, lock_fixed, within.as_deref());
                for (x, y, module, reliability) in writes {
                    match reliability {
                        Some(reliability) => self.doc.grid.set_sampled(x, y, module, reliability),
                        None => self.doc.grid.set(x, y, module),
                    }
                }
                // Sampled whole, the grid is the sampling asked for, and
                // no longer the pick of a search.
                if whole && !only_unknown {
                    self.doc.grid_among = 1;
                }
                // Inverting the photo above refreshed what follows from
                // the document, and cleared the mark `begin_edit` made:
                // the grid as sampled is what is to be derived from now.
                self.dirty = true;
                self.view.stage = Stage::Edit;
                let mut text = match scope {
                    SampleScope::All => format!("Sampled the photo: {changed} modules changed. Check them against the photo."),
                    SampleScope::Block(block) => format!("Sampled block {} from the photo: {changed} modules changed.", block + 1),
                };
                match flipped {
                    Some(true) => text.push_str(
                        " This is a light-on-dark symbol: its fixed patterns read inverted, so Invert photo was turned \
                         on first. Untick it to override.",
                    ),
                    Some(false) => text.push_str(
                        " The fixed patterns read inverted with Invert photo on, so it was turned off first. Tick it \
                         again to override.",
                    ),
                    None => {}
                }
                if let Some(inverted) = inverted {
                    text.push(' ');
                    text.push_str(inverted);
                }
                if let Some(orientation) = turned {
                    let (matching, known) = orientation.agreement;
                    let how = if orientation.inverted { ", with dark and light swapped" } else { "" };
                    text.push_str(&format!(
                        " The outline is turned on the symbol: the fixed patterns fit after \"{}\"{how} ({matching} of \
                         {known} read right).",
                        orientation.symmetry.label()
                    ));
                    text.push_str(if only_unknown {
                        " Only unknown modules were sampled, so the grid may not show it. Move the corners round by \
                         hand, or use Sample all and then the button under Paint."
                    } else if self.view.lock_fixed {
                        " The fixed patterns were left as sampled, not locked, so that the grid shows it. Use the \
                         button under Paint to turn the grid and the outline together."
                    } else {
                        " Use the button under Paint to turn the grid and the outline together."
                    });
                }
                self.notify(text);
            }
            Action::SuggestVersions => self.rank_versions(ctx),
            Action::ClearGuesses => self.guesses = None,
            Action::ResetCorners => {
                if let Some(photo) = &self.photo {
                    let corners = photo.default_corners();
                    self.checkpoint();
                    self.doc.corners = corners;
                    self.doc.anchors.clear();
                    self.detection = None;
                }
            }
            Action::ClearAnchors => {
                if !self.doc.anchors.is_empty() {
                    self.checkpoint();
                    self.doc.anchors.clear();
                }
            }
            Action::FindCorners => self.find_corners(ctx),
            Action::FindSymbols => self.search_symbols(ctx),
            Action::UseProposal(i) => self.use_proposal(ctx, i),
            Action::ListPlacements => self.list_placements(ctx),
            Action::UsePlacement(i) => {
                let placement = self.placements.as_ref().and_then(|p| p.found.get(i).map(|f| (f.clone(), p.photo_inverted)));
                let Some((found, photo_inverted)) = placement else {
                    self.fail("That placement is no longer in the list; press Other placements again.");
                    return;
                };
                self.adopt_proposal(
                    ctx,
                    found,
                    photo_inverted,
                    format!("placement {}", i + 1),
                    self.view.load_reads.then_some(1),
                    Some(Picked::Placement(i)),
                );
            }
            Action::ClearPlacements => self.placements = None,
            Action::UseLead(i) => {
                let Some(lead) = self.leads.get(i).cloned() else {
                    self.fail("Those finder patterns are no longer in the list; press Search whole photo again.");
                    return;
                };
                self.checkpoint();
                // A placement fitted and not read is put down as fitted, at
                // its version, to be sampled as it is or searched about: the
                // grid was once left at whatever version it had. Another
                // version is a new grid, which is said, as a proposal taken
                // says it ([`new_grid_note`]): it was once started with
                // nothing said of what went with the grid before.
                let new_grid = lead.fitted.filter(|&version| version != self.doc.version);
                if let Some(version) = lead.fitted {
                    self.set_version(version);
                }
                self.doc.corners = lead.outline;
                self.doc.anchors.clear();
                self.detection = None;
                self.view.stage = Stage::Align;
                self.notify(match lead.fitted {
                    Some(version) => format!(
                        "The outline is where the search fitted a {} {version} symbol to {} finder patterns, which nothing \
                         else bore out and which did not read. Set it right by hand if it is not, and press Find in outline, \
                         or sample it and work on the grid.{}",
                        version.family().label(),
                        lead.finders.len(),
                        new_grid.map_or(String::new(), |version| format!(
                            " This started a new {} {version} grid (undoable).",
                            version.family().label()
                        ))
                    ),
                    None => format!(
                        "The outline is about {} finder patterns the search took for a {} symbol's and could fit none to. \
                         Set it right by hand if it is not, and press Find in outline.",
                        lead.finders.len(),
                        lead.family.label()
                    ),
                });
            }
            Action::ClearProposals => {
                self.proposals = None;
                self.leads.clear();
            }
            Action::CancelSearch => {
                // Dropping the search calls it off: its thread stops at
                // the next turn of its loop.
                if self.search.take().is_some() {
                    self.notify("The search of the photo was called off.");
                }
            }
            Action::CancelJob => {
                if let Some(job) = self.job.take() {
                    self.notify(format!("{} was called off; nothing was changed.", job.kind.label()));
                }
            }
            Action::ReadFound => self.read_found(ctx),
            Action::CropToOutline => self.crop_to_outline(ctx),
            Action::ShowWholePhoto => self.show_whole_photo(ctx),
            Action::PinText => self.pin_text(),
            Action::TryBeginnings => self.pin_a_beginning(),
            Action::ClearTextPins => {
                self.begin_edit();
                self.doc.pins.retain(|_, pin| pin.source != PinSource::Text);
            }
            Action::RefineCorners => self.refine(ctx),
            Action::SetAdjust(adjust) => {
                if self.doc.adjust != adjust
                    && let Some(text) = self.set_adjust(adjust)
                {
                    self.notify(text);
                }
            }
            Action::EstimateLens => self.estimate_lens(),
            Action::MeasureDeblur => self.measure_deblur(),
            Action::SetFormat(format) => {
                if self.doc.format != format {
                    self.begin_edit();
                    self.doc.format = format;
                }
            }
            Action::BruteForce => {
                let pins = self.active_pins();
                let attempts = brute_force_whole(
                    &self.doc.grid,
                    &self.derived.func,
                    Some((self.derived.ec, &pins)),
                    &self.doc.picked(&self.opts),
                );
                self.brute = Some(BruteResults { attempts, stale: false });
            }
            Action::TryHarder(block) => self.try_harder(ctx, block),
            Action::Pin(cw, value) => {
                self.begin_edit();
                if self.doc.pins_ec != Some(self.derived.ec) {
                    self.doc.pins.clear();
                    self.doc.pins_ec = Some(self.derived.ec);
                }
                self.doc.pins.insert(cw, Pin::full(value, PinSource::Manual));
            }
            Action::PinBits { cw, mask, value } => {
                self.begin_edit();
                if self.doc.pins_ec != Some(self.derived.ec) {
                    self.doc.pins.clear();
                    self.doc.pins_ec = Some(self.derived.ec);
                }
                if mask == 0 {
                    self.doc.pins.remove(&cw);
                } else {
                    self.doc.pins.insert(cw, Pin::new(mask, value & mask, PinSource::Manual));
                }
            }
            Action::Unpin(cw) => {
                // Only a pin in the document is the document's to take
                // away, whatever its source: Pin the padding takes the
                // decoder's pins in with the sources they came with. One
                // the decoder placed by itself is placed again by the
                // next decode, and is dropped by turning that off.
                if self.doc.pins_ec != Some(self.derived.ec) || !self.doc.pins.contains_key(&cw) {
                    self.fail(
                        "That pin is the decoder's, not the document's: untick \"Pin the padding and headers found\" \
                         under Decoding to drop it, or press Pin the padding to take the decoder's pins into the \
                         document, where each can be unpinned.",
                    );
                    return;
                }
                self.begin_edit();
                self.doc.pins.remove(&cw);
            }
            Action::PinPaddingAt { first, filler } => {
                let (pins, added) = pin_padding_at(&self.derived.layout, first, filler, &self.active_pins());
                if added == 0 {
                    self.fail(
                        "No padding was pinned there: the start lies past the last data codeword, or every codeword from \
                         it on is pinned by hand already.",
                    );
                    return;
                }
                let before = self.derived.report.blocks_solved();
                self.begin_edit();
                // Pinned by hand over the pins in force, or, where those
                // were made under another level, in place of them.
                self.doc.pins = pins;
                self.doc.pins_ec = Some(self.derived.ec);
                // Pinned, the start is the decode's to say again.
                self.view.pad_from = None;
                // Under the options the document is derived under, so
                // that the count is the derived decode's.
                let opts = self.decoding_options();
                let after = decode(&self.doc.grid, &self.derived.layout, self.derived.mask, &self.doc.pins, &opts).blocks_solved();
                let total = self.derived.layout.blocks.len();
                let progress = if after > before {
                    format!("{after} of {total} blocks solved, up from {before}")
                } else {
                    format!("still {after} of {total} blocks solved")
                };
                self.notify(format!(
                    "Pinned {added} codewords of padding from message byte {first}, as {}: {progress}. The start and the \
                     filler are as given by hand, and nothing but the blocks checks them.",
                    filler.label()
                ));
            }
            Action::ApplyPadding => {
                let pins = self.active_pins();
                let opts = self.decoding_options();
                let recovery = recover_padding(&self.doc.grid, &self.derived.layout, self.derived.mask, &pins, &opts);
                match recovery {
                    Some(recovery) => {
                        self.begin_edit();
                        self.doc.pins = recovery.pins;
                        self.doc.pins_ec = Some(self.derived.ec);
                        let check = if recovery.confirmed {
                            "segment headers confirm where the data ends"
                        } else {
                            "not confirmed by segment headers"
                        };
                        let total = self.derived.layout.blocks.len();
                        let progress = if recovery.solved_after > recovery.solved_before {
                            format!("{} of {total} blocks solved, up from {}", recovery.solved_after, recovery.solved_before)
                        } else {
                            format!("still {} of {total} blocks solved", recovery.solved_after)
                        };
                        // The standard's filler goes without saying.
                        let filler = recovery.candidate.filler;
                        let what = if filler == Filler::Standard { String::new() } else { format!(" as {}", filler.label()) };
                        self.notify(format!("Padding recovery pinned {} codewords{what}: {progress}; {check}.", recovery.added));
                    }
                    None => self.fail("No padding could be located in this symbol."),
                }
            }
            Action::ClearPaddingPins => {
                self.begin_edit();
                clear_padding_pins(&mut self.doc.pins);
            }
            Action::ClearAllPins => {
                self.begin_edit();
                self.doc.pins.clear();
            }
            Action::SetCharset(charset) => {
                self.opts.charset = charset;
                self.dirty = true;
            }
            Action::SetDecodeOptions(opts) => {
                self.opts = opts;
                self.dirty = true;
            }
            Action::ApplyCorrections => {
                let repaired =
                    repaired_grid(&self.doc.grid, &self.derived.layout, self.derived.mask, &self.derived.report);
                let changed = repaired.cells().iter().zip(self.doc.grid.cells()).filter(|(a, b)| a != b).count();
                self.begin_edit();
                self.doc.grid = repaired;
                let unsolved = self.derived.report.blocks.len() - self.derived.report.blocks_solved();
                let rest = if unsolved > 0 {
                    format!(" The modules of {unsolved} unsolved blocks are as they were.")
                } else {
                    String::new()
                };
                self.notify(format!("Applied corrections: {changed} modules changed.{rest}"));
            }
            Action::ExportImage(format) => {
                let (what, extension) = match format {
                    ExportFormat::Png => ("PNG image", "png"),
                    ExportFormat::Svg => ("SVG image", "svg"),
                };
                let picked = rfd::FileDialog::new()
                    .add_filter(what, &[extension])
                    .set_file_name(format!("repaired.{extension}"))
                    .save_file();
                let Some(path) = picked else { return };
                self.export_image(&path, format);
            }
            Action::AddToSequence => {
                let label = self.photo.as_ref().map_or_else(|| "grid".to_owned(), |p| p.name.clone());
                match Part::from_report(&self.derived.report, label) {
                    Some(part) => {
                        let (index, total) = (part.index, part.total);
                        let replaced = self.view.sequence.add(part);
                        let what = if replaced { "replaced" } else { "added" };
                        self.notify(format!(
                            "Part {index} of {total} {what} to the sequence; {} of {total} parts collected.",
                            self.view.sequence.len()
                        ));
                    }
                    None => self.fail("This symbol carries no structured append header."),
                }
            }
            Action::RemoveFromSequence(index) => self.view.sequence.remove(index),
            Action::ClearSequence => self.view.sequence.clear(),
            Action::SaveBytes => {
                let Some(bytes) = self.derived.report.text.raw_bytes() else {
                    self.fail("Some bytes of the message are unreadable; nothing to save yet.");
                    return;
                };
                let picked = rfd::FileDialog::new().set_file_name("message.bin").save_file();
                let Some(path) = picked else { return };
                match std::fs::write(&path, &bytes) {
                    Ok(()) => self.notify(format!("Saved {} bytes to {}.", bytes.len(), path.display())),
                    Err(error) => self.fail(format!("Could not save {}: {error}", path.display())),
                }
            }
            Action::DismissNotice => self.message = None,
            Action::FitView => match self.view.stage {
                Stage::Align => self.view.align_view.auto_fit = true,
                Stage::Edit => self.view.edit_view.auto_fit = true,
            },
            Action::Quit => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
        }
    }

    /// The modules `scope` takes in, a flag for each in the grid's order;
    /// `None` for the whole grid.
    fn scope_mask(&self, scope: SampleScope) -> Option<Vec<bool>> {
        let (width, height) = (self.derived.func.width, self.derived.func.height);
        let mut mask = vec![false; width * height];
        match scope {
            SampleScope::All => return None,
            SampleScope::Block(block) => {
                let layout = &self.derived.layout;
                for &cw in layout.blocks.get(block).map_or(&[][..], |block| &block.codewords) {
                    for pos in layout.codeword_modules(cw) {
                        mask[pos.y as usize * width + pos.x as usize] = true;
                    }
                }
            }
        }
        Some(mask)
    }

    /// How many modules Sample all, or Sample unknown only, would change
    /// as things stand, for saying so beside the buttons. `None` with no
    /// sample to go by, or when the photo would be inverted first, which
    /// makes another sample of it. Fixed patterns that fit the grid
    /// turned are counted as sampled, as they are then written.
    pub(crate) fn sample_would_change(&self, only_unknown: bool) -> Option<usize> {
        let preview = self.preview.as_ref()?;
        let func = &self.derived.func;
        if preview.classified.modules.len() != func.width * func.height
            || (self.view.auto_invert && looks_inverted(preview.fixed))
        {
            return None;
        }
        let lock_fixed = self.view.lock_fixed && preview.turned.is_none();
        Some(sample_writes(&self.doc.grid, func, &preview.classified, only_unknown, lock_fixed, None).1)
    }

    /// What to say of modules taken off the photo while its fixed
    /// patterns read inverted, as it is sampled now: they are the wrong
    /// way round, and which way Invert photo should go. `None` while the
    /// fixed patterns read as they should, or with no sample to go by.
    pub(crate) fn inverted_note(&self) -> Option<&'static str> {
        self.preview.as_ref().filter(|preview| looks_inverted(preview.fixed))?;
        Some(if self.doc.adjust.invert {
            "The fixed patterns read inverted: Invert photo looks wrong for this photo. Untick it and sample again."
        } else {
            "The fixed patterns read inverted: this looks like a light-on-dark symbol. Tick Invert photo and sample again."
        })
    }

    /// Say so when modules have just been painted with the From photo
    /// ink off a photo whose fixed patterns read inverted: a part of the
    /// grid is sampled to go with the rest as it stands, so the photo is
    /// not inverted for it as it is for Sample all, and the modules
    /// painted are the wrong way round.
    pub(crate) fn warn_if_painted_inverted(&mut self) {
        if let Some(note) = self.inverted_note() {
            self.notify(format!("The modules painted were read off the photo as it is. {note}"));
        }
    }

    /// Pin the text the inspector's Known text holds, at the place it
    /// gives, and say what that did.
    fn pin_text(&mut self) {
        let place = if self.view.known_from_start { Place::Start(self.view.known_mode) } else { Place::Char(self.view.known_at) };
        let found = text_pins(&self.derived.layout, &self.derived.report, place, &self.view.known_text, self.opts.charset);
        let found = match found {
            Ok(found) => found,
            Err(error) => {
                self.fail(format!("The text was not pinned: {error}."));
                return;
            }
        };
        let before = self.derived.report.blocks_verified();
        self.begin_edit();
        // Pins made under another EC level name other codewords.
        if self.doc.pins_ec != Some(self.derived.ec) {
            self.doc.pins.clear();
            self.doc.pins_ec = Some(self.derived.ec);
        }
        lay_over(&mut self.doc.pins, &found.pins);
        self.refresh();
        let (after, total) = (self.derived.report.blocks_verified(), self.derived.report.blocks.len());
        let progress = if after > before {
            format!("{after} of {total} blocks now verify, up from {before}")
        } else {
            format!("{after} of {total} blocks verify, as before")
        };
        let mut text = format!("Pinned {} characters in {} codewords: {progress}.", found.pinned, found.pins.len());
        if found.left_out > 0 {
            text.push_str(&format!(
                " {} at the ends were left out: they are packed in a group with characters that were not given.",
                found.left_out
            ));
        }
        if found.contradicted > 0 {
            text.push_str(&format!(
                " {} of the bits pinned differ from bits error correction had confirmed: check the text and where it \
                 starts, or undo.",
                found.contradicted
            ));
        }
        self.notify(text);
    }

    /// Try the ways messages commonly begin and pin the one that error
    /// correction bears out, when one is.
    fn pin_a_beginning(&mut self) {
        let pins = self.active_pins();
        let (before, total) = (self.derived.report.blocks_verified(), self.derived.report.blocks.len());
        let opts = self.decoding_options();
        let found = try_beginnings(&self.doc.grid, &self.derived.layout, self.derived.mask, &pins, &opts, &self.derived.report);
        let Some(found) = found else {
            self.fail(
                "None of the common beginnings made a block verify that had not: either the message begins some other \
                 way, or its first block is too damaged for a beginning alone to solve. Type what you know of it under \
                 Known text.",
            );
            return;
        };
        self.begin_edit();
        self.doc.pins = found.pins;
        self.doc.pins_ec = Some(self.derived.ec);
        self.view.known_text = found.text.to_owned();
        match found.place {
            Place::Char(at) => (self.view.known_from_start, self.view.known_at) = (false, at),
            Place::Start(mode) => (self.view.known_from_start, self.view.known_mode) = (true, mode),
        }
        self.notify(format!(
            "The message begins {:?}: with that pinned, {} of {total} blocks verify, up from {before}.",
            found.text,
            found.report.blocks_verified()
        ));
    }

    /// Save the grid as error correction leaves it as an image at `path`,
    /// and say what was saved: a symbol whose every block solved is the
    /// repaired one, which an ordinary reader can scan, and anything
    /// short of that is said to be.
    fn export_image(&mut self, path: &Path, format: ExportFormat) {
        let repaired = repaired_grid(&self.doc.grid, &self.derived.layout, self.derived.mask, &self.derived.report);
        let quiet = quiet_zone(self.doc.version);
        let (w, h, saved) = match format {
            ExportFormat::Png => {
                let (w, h, pixels) = render_rgba(&repaired, EXPORT_SCALE, quiet);
                let saved = image::save_buffer(path, &pixels, w as u32, h as u32, image::ExtendedColorType::Rgba8);
                (w, h, saved.map_err(|error| error.to_string()))
            }
            ExportFormat::Svg => {
                let (w, h) = ((repaired.width() + 2 * quiet) * EXPORT_SCALE, (repaired.height() + 2 * quiet) * EXPORT_SCALE);
                let saved = std::fs::write(path, render_svg(&repaired, EXPORT_SCALE, quiet));
                (w, h, saved.map_err(|error| error.to_string()))
            }
        };
        match saved {
            Ok(()) => {
                let unknown = repaired.count(Module::Unknown);
                let mut note = if unknown > 0 { format!(" {unknown} unknown modules are grey.") } else { String::new() };
                let (solved, total) = (self.derived.report.blocks_solved(), self.derived.report.blocks.len());
                if solved < total {
                    note.push_str(&format!(
                        " Not every block is solved ({solved} of {total}), and the modules of the rest are as painted: \
                         this is not yet the repaired symbol, and a reader may well not scan it."
                    ));
                }
                self.notify(format!("Saved {} ({w} × {h}).{note}", path.display()));
            }
            Err(error) => self.fail(format!("Could not save {}: {error}", path.display())),
        }
    }

    fn shortcuts(&mut self, ui: &egui::Ui, actions: &mut Vec<Action>) {
        // A text field that has the keyboard keeps every shortcut: its
        // own editing undoes a typo with the undo keys, which it would
        // never see once they were consumed here, and would have the
        // document's last edit undone under it instead.
        let typing = ui.ctx().egui_wants_keyboard_input();
        // Redo first: the undo shortcut also matches with Shift held.
        let (redo, undo, open) = if typing {
            (false, false, false)
        } else {
            ui.input_mut(|i| (i.consume_shortcut(&REDO), i.consume_shortcut(&UNDO), i.consume_shortcut(&OPEN)))
        };
        if redo {
            actions.push(Action::Redo);
        } else if undo {
            actions.push(Action::Undo);
        }
        if open {
            actions.push(Action::OpenImage);
        }
        let (pasted, released) = ui.input(|i| {
            let released = |event: &egui::Event| {
                matches!(event, egui::Event::Key { key: egui::Key::V, pressed: false, modifiers, .. } if modifiers.command)
            };
            (i.events.iter().any(|event| matches!(event, egui::Event::Paste(_))), i.events.iter().any(released))
        });
        if paste_asked(pasted, released, typing, &mut self.view.pasted_on_press) {
            actions.push(Action::PasteImage);
        }
        if typing {
            return;
        }
        ui.input(|i| {
            if i.modifiers.command || i.modifiers.ctrl || i.modifiers.alt {
                return;
            }
            let inks = [
                (egui::Key::Num1, Ink::Module(Module::Dark)),
                (egui::Key::Num2, Ink::Module(Module::Light)),
                (egui::Key::Num3, Ink::Module(Module::Unknown)),
                (egui::Key::Num4, Ink::Photo),
                (egui::Key::Num5, Ink::Doubt),
            ];
            for (key, ink) in inks {
                // The photo's ink only while there is a sample of it.
                if i.key_pressed(key) && (ink != Ink::Photo || self.preview.is_some()) {
                    self.view.ink = ink;
                    if ink == Ink::Doubt && self.view.paint_reliability == CERTAIN {
                        self.view.paint_reliability = DOUBT_RELIABILITY;
                    }
                }
            }
            for (key, tool) in [(egui::Key::B, Tool::Brush), (egui::Key::R, Tool::Rect), (egui::Key::I, Tool::Inspect)] {
                if i.key_pressed(key) {
                    self.view.tool = tool;
                }
            }
            if i.key_pressed(egui::Key::X)
                && let Ink::Module(module) = self.view.ink
            {
                self.view.ink = Ink::Module(module.inverted());
            }
            if i.key_pressed(egui::Key::F) {
                actions.push(Action::FitView);
            }
            if i.key_pressed(egui::Key::Escape) {
                self.view.selected = None;
                self.view.highlight_block = None;
                self.view.picked_char = None;
                self.view.picked_segment = None;
            }
        });
    }

    fn menu_bar(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        let shortcut = |ui: &egui::Ui, text: &str, shortcut: &egui::KeyboardShortcut| {
            egui::Button::new(text.to_owned()).shortcut_text(ui.ctx().format_shortcut(shortcut))
        };
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.add(shortcut(ui, "Open Image…", &OPEN)).clicked() {
                    actions.push(Action::OpenImage);
                }
                if ui.add(shortcut(ui, "Paste Image from Clipboard", &PASTE)).clicked() {
                    actions.push(Action::PasteImage);
                }
                if ui.add_enabled(self.photo.is_some(), egui::Button::new("Close Image")).clicked() {
                    actions.push(Action::ClosePhoto);
                }
                ui.separator();
                if ui.button("New Blank Grid").clicked() {
                    actions.push(Action::NewBlank);
                }
                ui.separator();
                if ui.button("Export Repaired Image…").clicked() {
                    actions.push(Action::ExportImage(ExportFormat::Png));
                }
                if ui
                    .button("Export Repaired Image as SVG…")
                    .on_hover_text("A drawing, which prints and scales at any size without blur")
                    .clicked()
                {
                    actions.push(Action::ExportImage(ExportFormat::Svg));
                }
                ui.separator();
                if ui.button("Quit").clicked() {
                    actions.push(Action::Quit);
                }
            });
            ui.menu_button("Edit", |ui| {
                if ui.add_enabled(self.history.can_undo(&self.doc), shortcut(ui, "Undo", &UNDO)).clicked() {
                    actions.push(Action::Undo);
                }
                if ui.add_enabled(self.history.can_redo(&self.doc), shortcut(ui, "Redo", &REDO)).clicked() {
                    actions.push(Action::Redo);
                }
                ui.separator();
                if ui.button("Fill Fixed Patterns").clicked() {
                    actions.push(Action::FillFixed);
                }
                if ui.button("Clear Everything Else to Unknown").clicked() {
                    actions.push(Action::ClearData);
                }
                if ui.button("Apply Corrections").clicked() {
                    actions.push(Action::ApplyCorrections);
                }
                ui.separator();
                let square = self.doc.version.family() != Family::Rmqr;
                if ui.add_enabled(square, egui::Button::new("Rotate 90° Clockwise")).clicked() {
                    actions.push(Action::Transform(Transform::RotateCw));
                }
                if ui.button("Rotate 180°").clicked() {
                    actions.push(Action::Transform(Transform::Rotate180));
                }
                for &mirror in self.doc.version.mirrors() {
                    if ui.button(mirror.label()).on_hover_text("For a symbol seen from the back").clicked() {
                        actions.push(Action::Transform(Transform::Mirror(mirror)));
                    }
                }
                if square && ui.button(Symmetry::RotateCcw.label()).clicked() {
                    actions.push(Action::Transform(Transform::Symmetry(Symmetry::RotateCcw)));
                }
                if ui.button("Swap Dark and Light").clicked() {
                    actions.push(Action::Transform(Transform::Invert));
                }
                ui.separator();
                if ui.add_enabled(!self.doc.pins.is_empty(), egui::Button::new("Remove All Pins")).clicked() {
                    actions.push(Action::ClearAllPins);
                }
            });
            ui.menu_button("View", |ui| {
                if ui.button("Fit to Window    F").clicked() {
                    actions.push(Action::FitView);
                }
                ui.checkbox(&mut self.view.grid_lines, "Grid Lines");
                ui.separator();
                ui.label("Overlay");
                for overlay in Overlay::ALL {
                    ui.radio_value(&mut self.view.overlay, overlay, overlay.label());
                }
            });
        });
    }

    fn status_bar(&self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        ui.horizontal(|ui| {
            // What runs on other threads, with how long it has run; a job
            // and a search may run side by side.
            let mut running = Vec::new();
            if let Some(job) = &self.job {
                running.push((format!("{}… {:.0} s", job.kind.label(), job.started.elapsed().as_secs_f32()), Action::CancelJob));
            }
            if let Some(search) = &self.search {
                let what = search.family.map_or("symbols of every family".to_owned(), |family| format!("{} symbols", family.label()));
                let text = format!("Searching the photo for {what}… {:.0} s", search.started.elapsed().as_secs_f32());
                running.push((text, Action::CancelSearch));
            }
            if !running.is_empty() {
                ui.spinner();
                for (i, (text, stop)) in running.into_iter().enumerate() {
                    if i > 0 {
                        ui.label("·");
                    }
                    ui.label(text);
                    if ui.small_button("Stop").on_hover_text("Call it off. Nothing it has worked out so far is taken in.").clicked() {
                        actions.push(stop);
                    }
                }
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(250));
            } else if let Some((x, y)) = self.view.hover
                && self.view.stage == Stage::Edit
            {
                ui.monospace(canvas::describe_module(&self.derived, &self.doc, x, y));
            } else {
                ui.weak("Scroll to pan · pinch or ⌘-scroll to zoom · 1/2/3/4 ink · B/R/I tool · hold H for the photo");
            }
            let Some(notice) = &self.message else { return };
            // Errors stay until dismissed or the next edit; other notices fade.
            let age = notice.at.elapsed().as_secs_f32();
            if !notice.is_error && self.notice_seconds.is_finite() {
                if age >= self.notice_seconds {
                    actions.push(Action::DismissNotice);
                    return;
                }
                ui.ctx().request_repaint_after(std::time::Duration::from_secs_f32(self.notice_seconds - age));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("×").on_hover_text("Dismiss").clicked() {
                    actions.push(Action::DismissNotice);
                }
                let color = if notice.is_error { ui.visuals().error_fg_color } else { ui.visuals().text_color() };
                ui.add(egui::Label::new(egui::RichText::new(&notice.text).color(color)).truncate());
            });
        });
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.show(ui);
    }
}

impl App {
    /// One frame: take in what happened since the last, lay out the
    /// panels, and apply what the user asked for in them.
    fn show(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let mut actions = Vec::new();
        // The last frame is drawn: the textures on the card it may have
        // drawn through and has since dropped are let go of.
        if let Some(screen) = &self.screen {
            screen.let_go();
        }

        for file in ui.input(|i| i.raw.dropped_files.clone()) {
            let path = file.path().to_owned();
            self.load_path(&ctx, &path);
        }
        self.shortcuts(ui, &mut actions);
        #[cfg(feature = "selftest")]
        self.selftest_tick(ui, &mut actions);
        if self.settle_pending && self.view.stroke.is_none() {
            self.dirty = true;
        }
        // A photo opened before the window had said how large a texture
        // may be, as one named on the command line is, was drawn through
        // a texture no larger than egui allows unasked: it is made again
        // when the limit turns out to be another.
        let limit = ui.input(|i| i.max_texture_side);
        if limit != self.texture_limit {
            let before = std::mem::replace(&mut self.texture_limit, limit);
            if self.photo.as_ref().is_some_and(|photo| photo.width.max(photo.height) > before.min(limit)) {
                self.reapply_adjust();
            }
        }
        self.catch_up();
        self.poll_search(&ctx);
        self.poll_job(&ctx);
        // A result just taken in may have switched the version: what
        // follows from the document is brought up to it before anything
        // is laid out or drawn from the two together.
        self.catch_up();
        self.update_preview();

        egui::Panel::top("menu").show(ui, |ui| self.menu_bar(ui, &mut actions));
        egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui, &mut actions));
        egui::Panel::left("tools").default_size(250.0).size_range(210.0..=360.0).show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
                tools_panel::show(ui, self, &mut actions);
            });
        });
        egui::Panel::right("inspector").default_size(400.0).size_range(300.0..=640.0).show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
                inspector::show(ui, self, &mut actions);
            });
        });
        egui::CentralPanel::default().show(ui, |ui| canvas::show(ui, self, &mut actions));

        for action in actions {
            self.apply(&ctx, action);
        }
        // The photo in grey or in its colours, as the view now asks.
        if self.photo.as_ref().is_some_and(|photo| photo.greyscale != self.view.greyscale) {
            self.redraw_photo();
        }
        if self.dirty {
            ctx.request_repaint();
        }
    }
}

/// One module a sampling writes: where it is, its colour, and the
/// reliability it was read at, or none for a fixed module written as the
/// standard has it.
type SampleWrite = (usize, usize, Module, Option<u8>);

/// What a sampling of the photo writes into `grid`, and how many modules
/// change colour by it: each module `within` takes in, or every one when
/// it is `None`, and with `only_unknown` those the grid has as unknown
/// alone. A fixed module is written as the standard has it while
/// `lock_fixed` holds, with no reliability, and every other as
/// `classified` read it, with the reliability it was read at. That is
/// written where the colour agrees too: the reliability is new
/// information.
pub(crate) fn sample_writes(
    grid: &ModuleGrid,
    func: &FunctionMap,
    classified: &Classified,
    only_unknown: bool,
    lock_fixed: bool,
    within: Option<&[bool]>,
) -> (Vec<SampleWrite>, usize) {
    let mut writes = Vec::new();
    let mut changed = 0;
    for y in 0..func.height {
        for x in 0..func.width {
            let i = y * func.width + x;
            let current = grid.get(x, y);
            if within.is_some_and(|within| !within[i]) || (only_unknown && current != Module::Unknown) {
                continue;
            }
            match func.expected(x, y) {
                Some(dark) if lock_fixed => {
                    if current != Module::from_dark(dark) {
                        writes.push((x, y, Module::from_dark(dark), None));
                        changed += 1;
                    }
                }
                _ => {
                    changed += usize::from(classified.modules[i] != current);
                    writes.push((x, y, classified.modules[i], Some(classified.reliability[i])));
                }
            }
        }
    }
    (writes, changed)
}

/// Whether a frame's input asks for the image on the clipboard to be
/// pasted. Paste never arrives as a shortcut: the window's backend turns
/// the press into a paste of the clipboard's text, and when the
/// clipboard holds none, as with a screenshot on it, sends nothing for
/// the press at all; only the key's release comes through. So a paste of
/// text is taken as the request, and so is the release of the key when
/// its press brought none, which `pasted_on_press` remembers between the
/// two. While a text field has the keyboard, `typing`, a paste is the
/// field's and asks nothing here.
fn paste_asked(pasted: bool, released: bool, typing: bool, pasted_on_press: &mut bool) -> bool {
    *pasted_on_press |= pasted;
    let asked = !typing && (pasted || (released && !*pasted_on_press));
    if released {
        *pasted_on_press = false;
    }
    asked
}

/// The decode options the app starts with, and Restore defaults goes
/// back to: the library's, but with the decoder pinning the padding and
/// headers it finds of its own accord, which the library leaves to its
/// callers.
pub(crate) fn default_decode_options() -> DecodeOptions {
    DecodeOptions { derive_pins: true, ..DecodeOptions::default() }
}

/// What a search of block `n`, counted from one, with every budget at its
/// most came to, said once it is taken in: whether the block's own
/// modules or pins `changed` while it was searched, and its outcome
/// `before` the search was taken in and `after`. Where they are as they
/// were and the block is neither confirmed nor marked as searched as it
/// reads now, it reads otherwise all the same: under the padding and
/// header pins found anew, since other blocks changed.
fn try_harder_notice(n: usize, changed: bool, before: Option<BlockOutcome>, after: &BlockReport) -> String {
    // Said where Try harder is offered for the block as it reads now.
    let again = if after.is_verified() || after.tried_harder { "" } else { " Press Try harder again to search it as it is now." };
    let moved = before != Some(after.outcome);
    if changed {
        format!("Block {n} changed while it was being searched.{again}")
    } else if after.is_verified() && moved {
        format!("Block {n} is solved and confirmed after searching harder.")
    } else if !after.is_verified() && !after.tried_harder {
        format!("Other blocks changed while block {n} was being searched, so it is now decoded under different padding and header pins.{again}")
    } else if after.is_solved() && moved {
        format!("Searching block {n} harder found a solution, but it is not confirmed.")
    } else {
        format!("Searching block {n} harder found nothing better.")
    }
}

/// Short name for an EC level in the context of a version.
pub(crate) fn ec_name(version: SymbolVersion, ec: EcLevel) -> &'static str {
    version.ec_label(ec)
}

/// How far error correction reaches in the blocks of a version at a
/// level, in wrong codewords of a block read whole, for the notes that
/// say so: by the standard, and by list decoding and the trials of
/// erasures at the budgets the decode options give them. Where the
/// blocks differ in length, list decoding is of the one it reaches
/// least far in, and the trials of the longest, which takes the most
/// sets.
#[derive(Clone, Copy)]
pub(crate) struct Reach {
    pub blocks: usize,
    /// Codewords of the shortest block and of the longest.
    pub lengths: (usize, usize),
    /// What the standard corrects, keeping the codewords it reserves for
    /// detecting misreads ([`misdecode_protection`]).
    pub standard: usize,
    pub reserved: usize,
    /// What plain Reed-Solomon reaches spending the reserved codewords
    /// too, half the code's distance: past [`Self::standard`], never
    /// confirmed.
    pub half: usize,
    /// What list decoding is sure to reach at the options' constraints
    /// ([`hard_reach`]), and at the default's.
    pub list: usize,
    pub list_default: usize,
    /// Constraints the largest block uses of the options' ([`usable_cost`]).
    pub list_used: usize,
    pub trials: TrialReach,
    /// Codewords past [`Self::standard`] the trials reach and confirm,
    /// with what their solution must show to be verified to spare
    /// ([`TrialReach::within_standard`]): nought where any codewords are
    /// reserved, since a solution past what the standard corrects spends
    /// them, and nought unless that is past [`Self::half`], where plain
    /// Reed-Solomon finds a solution first and the trials are not run.
    pub trials_within: usize,
}

/// What [`Reach::of`] depends on: the version, the level, the budgets
/// of list decoding and of the trials, and what a fallback's solution
/// must show to be verified, as its solver counts it
/// ([`verifying_bits`]), in the first stage and in the trials' own
/// ([`Reach::trial_verifying_bits`]), and as the block judges it
/// ([`judged_verifying_bits`]), which the solvers switched on and the
/// decodes the document was picked among set. The trials' own was once
/// left out, and solving bit by bit switched on or off, which moves it
/// a bit where the first stage's stays, left the trials' reach as it was.
type ReachKey = (SymbolVersion, EcLevel, usize, usize, u32, u32, u32);

/// Reaches [`App::reach`] keeps, at the most, the oldest forgotten first:
/// a level for each of a few versions, and the budgets a slider passes
/// through as it is dragged.
const REACHES_KEPT: usize = 16;

impl App {
    /// [`Reach::of`] under the options the document would be decoded
    /// under at `level` ([`Self::options_at`]), worked out once for each
    /// version, level, budgets and verification asked about. It was once
    /// worked out under the decode options alone, uncounted for the
    /// decodes a grid read from a photo was picked among, and kept while
    /// a solver was switched on or off, which moves what a fallback's
    /// solution must show.
    pub(crate) fn reach(&self, version: SymbolVersion, level: EcLevel) -> Option<Reach> {
        let opts = self.options_at(level);
        let key = (
            version,
            level,
            opts.list_cost,
            opts.trial_sets,
            verifying_bits(version, &opts.first_stage(), false),
            Reach::trial_verifying_bits(version, &opts),
            judged_verifying_bits(&opts.first_stage()),
        );
        let mut reaches = self.reaches.borrow_mut();
        if let Some((_, reach)) = reaches.iter().find(|(k, _)| *k == key) {
            return *reach;
        }
        let reach = Reach::of(version, level, &opts);
        if reaches.len() == REACHES_KEPT {
            reaches.remove(0);
        }
        reaches.push((key, reach));
        reach
    }
}

impl Reach {
    pub fn of(version: SymbolVersion, level: EcLevel, opts: &DecodeOptions) -> Option<Self> {
        let shape = block_shape(version, level)?;
        let ec = shape.ec_per_block;
        let lengths = (shape.data_len(0) + ec, shape.data_len(shape.blocks - 1) + ec);
        // A block read certain, as one painted by hand is, verified by
        // what it must show charged for what found it, in the first stage
        // of the decode, which list decoding is first run in and which
        // makes the fewest decodes and searches besides it
        // ([`DecodeOptions::stages`]).
        let needed = verifying_bits(version, &opts.first_stage(), false);
        // Or what the block itself must show of the codeword, whatever
        // reached it, which asks nothing of the solver's tries
        // ([`judged_verifying_bits`]); it was once left out, and the
        // curve said to reach a codeword less at M3-M than it confirms.
        let judged = judged_verifying_bits(&opts.first_stage());
        let list = |cost| hard_reach(lengths.0, ec, cost, needed.min(judged)).min(hard_reach(lengths.1, ec, cost, needed.min(judged)));
        let reserved = misdecode_protection(version, level);
        let trials = trial_reach(lengths.1, ec, opts.trial_sets);
        let trial_needed = Self::trial_verifying_bits(version, opts);
        Some(Self {
            blocks: shape.blocks,
            lengths,
            standard: ec.saturating_sub(reserved) / 2,
            reserved,
            half: ec / 2,
            list: list(opts.list_cost),
            list_default: list(LIST_COST),
            list_used: usable_cost(lengths.1, ec, opts.list_cost),
            trials,
            trials_within: trials.within_standard(lengths.1, ec, reserved, trial_needed, judged),
        })
    }

    /// What a solution the trials of erasures found must show to be
    /// verified, as its solver counts it ([`verifying_bits`]): the trials
    /// run in a stage of their own, which makes the decodes of the first
    /// and the trials' besides.
    fn trial_verifying_bits(version: SymbolVersion, opts: &DecodeOptions) -> u32 {
        verifying_bits(version, &DecodeOptions { erase_by_trial: true, ..opts.first_stage() }, false)
    }

    /// The blocks, as the notes name them: "block of 26 codewords", or
    /// "blocks of 44 and 45 codewords".
    pub fn blocks_phrase(&self) -> String {
        let (short, long) = self.lengths;
        let lengths = if short == long { long.to_string() } else { format!("{short} and {long}") };
        if self.blocks == 1 { format!("block of {lengths} codewords") } else { format!("blocks of {lengths} codewords") }
    }

    /// List decoding is sure to reach further than the code alone: past
    /// half its distance, where plain Reed-Solomon goes too, spending
    /// the reserved codewords. It was once held to the standard's limit,
    /// and a 1-M block, whose list reaches half the distance and no
    /// further, was said to gain a codeword that plain decoding gets as
    /// well, and that list decoding, run only where plain decoding finds
    /// nothing, never tries for.
    pub fn list_gains(&self) -> bool {
        self.list > self.half
    }

    /// [`Self::list_gains`] at the default's constraints.
    pub fn list_default_gains(&self) -> bool {
        self.list_default > self.half
    }

    /// List decoding is sure to reach further than the code alone with
    /// the reserved codewords unused, so that what it reaches may be
    /// confirmed: only where none are, since a block read certain spends
    /// them past half the distance ([`hard_reach`]).
    pub fn list_gains_within_standard(&self) -> bool {
        self.list_gains() && self.reserved == 0
    }
}

/// A count with its thousands set apart by commas, as the hover texts
/// write them.
pub(crate) fn grouped(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// A count in words, as the notes on reach say it: "one", "two", and
/// so on to ten, and digits past that.
pub(crate) fn count_word(n: usize) -> String {
    const WORDS: [&str; 11] = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
    WORDS.get(n).map_or_else(|| n.to_string(), |word| (*word).to_owned())
}

/// A count of wrong codewords, as the notes on reach say it: "1 wrong
/// codeword", "2 wrong codewords".
pub(crate) fn wrong_codewords(n: usize) -> String {
    if n == 1 { "1 wrong codeword".to_owned() } else { format!("{n} wrong codewords") }
}

/// Share of fixed modules a detection of another family must read right
/// for Find in outline to take it when the current family finds nothing, and
/// the share of the outline's area it must fill.
const OTHER_FAMILY_SCORE: f64 = 0.8;
const OTHER_FAMILY_FILL: f64 = 0.25;

/// The version a search of a family starts from: a middling one, from
/// which the detector tries the versions the finder spacing suggests.
pub(crate) fn family_start(family: Family) -> SymbolVersion {
    match family {
        Family::Qr => SymbolVersion::Qr(2),
        Family::Micro => SymbolVersion::Micro(2),
        Family::Rmqr => SymbolVersion::Rmqr(11),
        Family::Model1 => SymbolVersion::Model1(2),
    }
}

/// The slow part of Find in outline: the symbol inside the outline read
/// through to text, as [`read_outline_of_any_family`] reads it, and made
/// ready to adopt. The family it was found under comes back with it when
/// that is not the one asked about.
fn read_inside(
    luma: &Luma,
    version: SymbolVersion,
    corners: [[f64; 2]; 4],
    params: &SampleParams,
    opts: &DecodeOptions,
    searched_inverted: bool,
    own_family_only: bool,
) -> Option<(Prepared, Option<Family>)> {
    let (read, other) = if own_family_only {
        (read_outline(luma, version, corners, params, opts)?, None)
    } else {
        read_outline_of_any_family(luma, version, corners, params, opts)?
    };
    let asked = other.map_or(version, family_start);
    Some((prepared_from_read(luma, asked, read, searched_inverted, params), other))
}

/// Read the symbol inside an outline drawn loosely around it, as one of
/// `version`'s family first and, when that read does not decode, as one
/// of the other families too, as a reader tries every symbology it knows;
/// the window's Find in outline and the command line's `--outline` both read
/// an outline this way. The family the read was made under comes back
/// with it when that is not `version`'s. `None` when no family finds a
/// symbol there at all.
///
/// A read of `version`'s family that decodes is the answer, and nothing
/// else is tried. Otherwise another family is only taken on good
/// evidence: a message that decodes, or its own bits naming the version
/// and a clean fit, over a fair share of the outline. Every finder
/// pattern of a QR Code passes for a Micro QR symbol of its own, and
/// reads well as one, but fills only a corner of an outline drawn around
/// the whole symbol. Of the reads that did not decode, the one asked
/// about and any other that passed that bar, the one with the most to
/// show for itself is kept: see [`standing`].
///
/// Each family's read is one of the families' ([`read_outline_one_of`]):
/// the first of `version`'s goes uncounted, as the read asked for, and
/// every other is charged as one of all the families make.
pub(crate) fn read_outline_of_any_family(
    luma: &Luma,
    version: SymbolVersion,
    corners: [[f64; 2]; 4],
    params: &SampleParams,
    opts: &DecodeOptions,
) -> Option<(ReadResult, Option<Family>)> {
    let current = version.family();
    let others: Vec<Family> = Family::SEARCHED.into_iter().filter(|&f| f != current).collect();
    let ways = 1 + others.len();
    let own = read_outline_one_of(luma, version, corners, params, opts, ways, true);
    if own.as_ref().is_some_and(ReadResult::is_trustworthy) {
        return own.map(|read| (read, None));
    }
    let mut best: Option<(ReadResult, Option<Family>)> = own.map(|read| (read, None));
    // The other families side by side, the read asked for having come to
    // nothing, and each taken in the families' order.
    let mut decoded = None;
    work::in_turn(
        &others,
        |&family| {
            let read = if work::cancelled() { None } else { read_outline_one_of(luma, family_start(family), corners, params, opts, ways, false) };
            (family, read)
        },
        |(family, read)| {
            let Some(read) = read else { return false };
            let vouched = read.is_trustworthy()
                || (read.detection.info.named_version == Some(read.detection.version)
                    && read.detection.score() >= OTHER_FAMILY_SCORE);
            if !vouched || quad_area(&read.detection.corners) < OTHER_FAMILY_FILL * quad_area(&corners) {
                return false;
            }
            if read.is_trustworthy() {
                decoded = Some((read, Some(family)));
                return true;
            }
            // Neither decoded: ranked by what each has to show besides its
            // decode ([`standing_of`]).
            let outranks = |held: &ReadResult| {
                standing_of(read.report.score.blocks_verified, read.agreement) > standing_of(held.report.score.blocks_verified, held.agreement)
            };
            if best.as_ref().is_none_or(|(held, _)| outranks(held)) {
                best = Some((read, Some(family)));
            }
            false
        },
    );
    decoded.or(best)
}

/// What a read made of its detection, in brief.
fn read_note(result: &ReadResult) -> ReadNote {
    ReadNote {
        kernel: result.kernel,
        rule: result.rule,
        enhance: result.enhance,
        deblurred: result.deblurred,
        sharpen_modules: result.sharpen_modules,
        unsure_kept: result.unsure_kept,
        verified: result.report.blocks_verified(),
        blocks: result.report.blocks.len(),
        trustworthy: result.is_trustworthy(),
        padding_pinned: result.padding_pinned(),
        reinterpreted: result.reinterpreted,
    }
}

/// What Find in outline read, made ready to adopt: the detection, already
/// refined, with what it decoded to, and the versions ranked on its
/// corners when the version found would not be adopted outright. A grid
/// that decoded only as the other QR Code model is adopted as that
/// model's symbol, whose anchors are its finders alone.
fn prepared_from_read(
    luma: &Luma,
    version: SymbolVersion,
    result: ReadResult,
    searched_inverted: bool,
    params: &SampleParams,
) -> Prepared {
    let trustworthy = result.is_trustworthy();
    let whole = Box::new(result.clone());
    let read = read_note(&whole);
    let mut found = result.detection;
    if let Some(other) = result.reinterpreted
        && trustworthy
    {
        found.version = other;
        found.anchors.retain(|a| a.kind != AnchorKind::Alignment);
    }
    let guesses = (found.version != version && !adopts_version(&found, version, trustworthy))
        .then(|| (version.family(), suggest_versions(luma, found.corners, version.family(), params)));
    Prepared { found, agreement: result.agreement, guesses, searched_inverted, read: Some(read), result: Some(whole) }
}

/// The slow part of adopting a detection: its corners refined against
/// the photo with its anchors held, then every anchor against its own
/// pattern, each expecting the bleed the detection was placed for
/// ([`Detection::bloom`]) at the polarity it was measured at
/// ([`Detection::held_polarity`]), and the versions ranked on the corners
/// when the version found would not be adopted outright, so that it can
/// be suggested instead. `func` is the function map in use, kept when the
/// detection is of its version, and `version` the document's when the
/// job began. `searched_inverted` is the Invert setting the detection was
/// found under, which its `inverted` is reckoned from, and
/// `photo_inverted` the one `luma` is under.
fn prepare_adoption(
    luma: &Luma,
    func: &Arc<FunctionMap>,
    version: SymbolVersion,
    mut found: Detection,
    searched_inverted: bool,
    photo_inverted: bool,
    params: &SampleParams,
) -> Prepared {
    let func = if func.version == found.version { func.clone() } else { Arc::new(FunctionMap::new(found.version)) };
    let controls: Vec<_> = found.anchors.iter().map(Anchor::control_point).collect();
    // The polarity held is the detection's against the photo as it was
    // searched; with Invert photo changed since, the symbol reads the
    // other way against `luma`, and the bleed held at the polarity found
    // would be the opposite bleed.
    let polarity = found.held_polarity().map(|inverted| inverted ^ (searched_inverted != photo_inverted));
    let refined = refine_corners_expecting(luma, &func, found.corners, &controls, &found.held_corners(), found.bloom, polarity, params);
    let agreement = refined.as_ref().map_or(found.agreement, |r| r.after);
    found.corners = refined.map_or(found.corners, |r| r.corners);
    found.anchors = refine_anchors_with(luma, &func, found.corners, &found.anchors, found.bloom, polarity, params);
    let guesses = (found.version != version && !adopts_version(&found, version, false))
        .then(|| (version.family(), suggest_versions(luma, found.corners, version.family(), params)));
    Prepared { found, agreement, guesses, searched_inverted, read: None, result: None }
}

/// The anchors of `found` as the document holds them: detected, its edge
/// points moved out for the bleed it was placed expecting
/// ([`Detection::bloom`], [`DocAnchor::bled`]), as a read moves them.
fn doc_anchors(found: &Detection) -> Vec<DocAnchor> {
    let bled = |anchor: &Anchor| if anchor.kind == AnchorKind::Edge { found.bloom } else { 0.0 };
    found.anchors.iter().map(|&anchor| DocAnchor { anchor, source: AnchorSource::Detected, bled: bled(&anchor) }).collect()
}

/// Whether adopting `found` switches a document at `version` to the
/// version found: when the symbol's own bits, a clean fit or a message
/// that `decoded` vouch for it, or when it is of another family, which
/// the user chose by picking it. Otherwise the version found is only
/// suggested.
fn adopts_version(found: &Detection, version: SymbolVersion, decoded: bool) -> bool {
    found.version != version
        && (found.version.family() != version.family()
            || found.info.named_version == Some(found.version)
            || found.score() > 0.9
            || decoded)
}

/// What to add to the offer of a symbol found when taking it would
/// switch a document at `version` to the symbol's own, which starts the
/// grid afresh; nothing otherwise.
fn new_grid_note(found: &Detection, version: SymbolVersion) -> String {
    if !adopts_version(found, version, false) {
        return String::new();
    }
    format!(" This starts a new {} {} grid (undoable).", found.version.family().label(), found.version)
}

/// The slow part of Refine: corners nothing holds are settled by the
/// symbol's edge first, then every corner is refined with the anchors
/// held, then every anchor against its own pattern; and all of it again
/// expecting the bleed the finders show, where that reads no worse
/// ([`refined_for_bleed`]), `already[k]` saying how far `anchors[k]` was
/// moved out for a bleed before ([`DocAnchor::bled`]). `None` when the
/// corners form no usable outline.
fn refine_outline(
    luma: &Luma,
    func: &FunctionMap,
    corners: [[f64; 2]; 4],
    anchors: Vec<Anchor>,
    already: &[f64],
    params: &SampleParams,
) -> Option<Refined> {
    let settled = settle_loose_corners(luma, func, corners, &anchors, params);
    let controls: Vec<_> = anchors.iter().map(Anchor::control_point).collect();
    let refinement = refine_corners(luma, func, settled, &controls, params)?;
    let after = refine_anchors(luma, func, refinement.corners, &anchors, 0.0, params);
    let printed = Refined { refinement, before: anchors, after, bloom: 0.0 };
    Some(refined_for_bleed(luma, func, &printed, already, params).unwrap_or(printed))
}

/// Refine as `printed` refined, again expecting the bleed the finders
/// show ([`bloom_of`]), as a read places a symbol again expecting it
/// ([`qr_forensics::imaging::read`]): the anchors refined expecting it,
/// the edge points moved out by however much of it `already` says they
/// have yet to be, and the corners placed afresh by the finders, the
/// symbol's edges and the plane through the anchors ([`bled_outline`]),
/// the corners refined expecting it from there with those the finders or
/// the edges placed held ([`refine_corners_expecting`]), and every anchor
/// once more. Kept unless the fixed patterns read worse for it, each
/// placement read through the anchors refined on it
/// ([`placement_agreement`]): a bleed is measured as one number, and ink
/// that spread further on one side of a finder than the other was placed
/// better as printed, which is why a read places a symbol expecting one
/// only when no other read of it is beyond doubt. Where the two read
/// alike the fixed patterns cannot tell them apart, and the finders,
/// which read more crisply expecting the bleed, can: as printed, every
/// anchor lies as far as the bleed off its pattern. `None` when the
/// finders show no bleed, or the symbol reads worse for it.
fn refined_for_bleed(luma: &Luma, func: &FunctionMap, printed: &Refined, already: &[f64], params: &SampleParams) -> Option<Refined> {
    // Light on dark or not as the refinement as printed found it, for the
    // bleed, every pattern looked for again, and the corners and anchors
    // refined again alike: the bleed is into the symbol's dark, and
    // refined at the other polarity it would be the opposite bleed.
    let (corners, inverted) = (printed.refinement.corners, printed.refinement.inverted);
    let bloom = bloom_of(luma, func, corners, &printed.before, inverted, params);
    if bloom == 0.0 {
        return None;
    }
    let bled = bled_outline(luma, func, corners, &printed.before, bloom, inverted, already, params)?;
    let controls: Vec<_> = bled.anchors.iter().map(Anchor::control_point).collect();
    let again = refine_corners_expecting(luma, func, bled.corners, &controls, &bled.held_corners(), bloom, Some(inverted), params)?;
    let after = refine_anchors_with(luma, func, again.corners, &bled.anchors, bloom, Some(inverted), params);
    // Each placement read as it would be taken in, through the anchors
    // refined on it. A refinement counts its fixed modules through the
    // anchors it was given: for the placement as printed those Refine
    // began from, for this one those already refined expecting the bleed,
    // and an anchor dropped by hand off its pattern would count against
    // the first alone.
    let matching = |corners: [[f64; 2]; 4], anchors: &[Anchor]| {
        let controls: Vec<_> = anchors.iter().map(Anchor::control_point).collect();
        placement_agreement(luma, func, corners, &controls, inverted, params).map(|(matching, _)| matching)
    };
    if matching(again.corners, &after)? < matching(corners, &printed.after)? {
        return None;
    }
    let refinement = Refinement { before: printed.refinement.before, evaluations: printed.refinement.evaluations + again.evaluations, ..again };
    Some(Refined { refinement, before: printed.before.clone(), after, bloom })
}

/// Items as a sentence lists them: "a", "a and b", "a, b and c".
fn list_phrase(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [only] => only.clone(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

/// What came of the finder patterns a search lists as leads, for a
/// notice to say after "but": no symbol fitted to them, or one fitted
/// that didn't read ([`Lead::fitted`]), or some of each.
fn leads_came_to(leads: &[Lead]) -> &'static str {
    let fitted = leads.iter().filter(|lead| lead.fitted.is_some()).count();
    if fitted == 0 {
        "no symbol could be fitted to them"
    } else if fitted == leads.len() {
        "the symbols fitted to them didn't read"
    } else {
        "no symbol could be fitted to some of them, and those fitted to the rest didn't read"
    }
}

/// The button under a list cut short at `limit` of its `total` rows, which
/// shows the list whole and cuts it short again; nothing under a list no
/// longer than that. `hint` says what the whole list holds.
fn show_all_button(ui: &mut egui::Ui, all: &mut bool, total: usize, limit: usize, hint: &str) {
    if total <= limit {
        return;
    }
    let label = if *all { format!("Show the first {limit} alone") } else { format!("Show all {total}") };
    if ui.small_button(label).on_hover_text(hint).clicked() {
        *all = !*all;
    }
}

/// Share of fixed modules a detection that does not decode as found must
/// read right to count as a credible symbol, inside which a smaller
/// detection is only a part of it, and whose finder patterns are its own.
const CREDIBLE_SCORE: f64 = 0.75;

/// What a search of the whole photo finds, over every family, as ZXing's
/// MultiFormatReader and zxing-cpp's ReadBarcodes try every symbology in
/// one pass: what decodes as it was found first, then the best fit,
/// whatever its family, and among equal fits the one whose own bits name
/// its version, in each family's own order.
///
/// Every finder pattern of a QR Code passes for the finder of a Micro QR
/// or rMQR symbol of its own, so QR Code is searched first, and the
/// finders of each credible QR Code found are withheld from the searches
/// after it. For what gets by all the same, a detection whose centre lies
/// inside a larger, credible one is dropped as a part of it, unless it
/// decodes as found, as a symbol set within another does; and two
/// detections of about the same size and place are the same symbol,
/// listed once under the better claim.
///
/// Beside what it lists come the finder patterns the search could fit no
/// symbol to ([`Lead`]). The placements the search fitted and nothing
/// vouched for are read through first, under `opts` as a read of what is
/// found is made, and listed when they read ([`settle_unvouched`]).
pub(crate) fn search_for_every_family(luma: &Luma, params: &SampleParams, search: &SearchParams, opts: &DecodeOptions) -> Sought {
    let (sought, read) = settle_unvouched(luma, search_for_families(luma, &Family::SEARCHED, params, search, &credible), params, opts);
    Sought { found: listed(luma, sought.detections, params), leads: sought.leads, read }
}

/// What a search of the whole photo has to offer: the symbols it found,
/// listed, and the finder patterns it could fit no symbol to
/// ([`Lead`]).
pub(crate) struct Sought {
    pub found: Vec<Detection>,
    pub leads: Vec<Lead>,
    /// The reads made of placements nothing vouched for that decoded,
    /// each of a symbol among those found, at its corners, with what it
    /// was charged as one of for the placements read beside it
    /// ([`settle_unvouched`]): for the command line, which reads every
    /// symbol found, to print as they are rather than read them again,
    /// uncharged for the placements read beside them; and for the window,
    /// which reads them again, to charge them the same
    /// ([`Proposals::charged`]).
    pub read: Vec<(ReadResult, usize)>,
}

impl Sought {
    /// The read made already of `found`, one of the symbols found, where
    /// the search made one ([`Self::read`]).
    pub(crate) fn read_of(&self, found: &Detection) -> Option<&ReadResult> {
        self.read.iter().map(|(read, _)| read).find(|read| read.detection.corners == found.corners && read.detection.version == found.version)
    }

    /// What each read the search made is charged as one of, by the
    /// corners and version of the symbol it is of ([`Proposals::charged`]).
    fn charged(&self) -> Vec<Charged> {
        self.read.iter().map(|(read, among)| Charged { corners: read.detection.corners, version: read.detection.version, among: *among }).collect()
    }
}

impl From<Vec<Detection>> for Sought {
    fn from(found: Vec<Detection>) -> Self {
        Self { found, leads: Vec::new(), read: Vec::new() }
    }
}

/// What a search of the whole photo finds of one family alone, listed as
/// [`search_for_every_family`] lists what it finds. The families
/// with one finder pattern are searched from every finder the photo
/// shows, a QR Code's among them: it is for the user who knows which
/// family the symbol is of, and above all for a damaged QR Code, whose
/// leftover finders the search of every family spends most of its time
/// taking for Micro QR and rMQR symbols.
pub(crate) fn search_for_family(luma: &Luma, family: Family, params: &SampleParams, search: &SearchParams, opts: &DecodeOptions) -> Sought {
    let (sought, read) = settle_unvouched(luma, search_for_families(luma, &[family], params, search, &|_| true), params, opts);
    Sought { found: listed(luma, sought.detections, params), leads: sought.leads, read }
}

/// Detections as a search lists them: the decode asked of each as found,
/// on the photo itself, which says which of two claims on one place is
/// the symbol and puts what needs no work at the head of the list. One
/// read through already, which decodes, is not asked again.
fn listed(luma: &Luma, mut all: Vec<Detection>, params: &SampleParams) -> Vec<Detection> {
    let decodes = work::map(&all, |found| found.decodes == Some(true) || (!work::cancelled() && decodes_as_found(luma, found, params)));
    for (found, decodes) in all.iter_mut().zip(decodes) {
        found.decodes = Some(decodes);
    }
    merge_detections(all)
}

/// The part on show at its own size: the photo as loaded, shared, or
/// the part `crop` of it, straightened by the lens term the crop was
/// made under, as the whole photo straightened has it there, on
/// `backdrop` where it draws on none of the photo. What the adjustments
/// are made to, but for a part so straightened, which they make from the
/// photo as loaded about it ([`adjusted`]).
fn part_on_show(source: &Arc<image::RgbaImage>, crop: Option<Crop>, backdrop: Backdrop) -> Arc<image::RgbaImage> {
    let Some(crop) = crop else { return source.clone() };
    let [x, y, w, h] = crop.rect;
    if crop.k1 == 0.0 {
        return Arc::new(image::imageops::crop_imm(&**source, x, y, w, h).to_image());
    }
    // The outline the crop was made to lay on the photo straightened.
    let (width, height) = source.dimensions();
    let part = straightened(source.as_raw(), width as usize, height as usize, crop.k1, [x, y, w, h].map(|side| side as usize), backdrop.level());
    Arc::new(image::RgbaImage::from_raw(w, h, part).expect("the part straightened is the part's size"))
}

/// The part on show under `adjust`: the RGBA image to display and the
/// luminance to sample, made of the pixels on show ([`part_on_show`]),
/// or, for a part cut under a lens term, of the photo as loaded about
/// it, which is straightened about the whole photo's centre after it is
/// enlarged, as the whole photo is ([`Adjust::apply_to`]). The pixels are
/// handed over shared, so that the luminance keeps the photo as loaded
/// beside it as these same bytes, and not as a copy of them
/// ([`Adjust::apply_to_shared`]). What is displayed is in the photo's
/// colours, or where `grey`, in the grey of the luminance, as the view in
/// grey shows the photo; and left on the graphics card where `on_card`
/// and it was made there whole. The colours are made only where they are
/// shown.
fn adjusted(adjust: &Adjust, source: &Arc<image::RgbaImage>, original: &Arc<image::RgbaImage>, crop: Option<Crop>, on_card: bool, grey: bool) -> (Shown, Luma) {
    let (pixels, adjust, part) = match crop {
        Some(crop) if crop.k1 != 0.0 => (source, Adjust { k1: crop.k1, ..*adjust }, crop.rect.map(|side| side as usize)),
        _ => (original, *adjust, [0, 0, original.width() as usize, original.height() as usize]),
    };
    let shared: SharedRgba = pixels.clone();
    let (width, height) = (pixels.width() as usize, pixels.height() as usize);
    match (on_card, grey) {
        (true, false) => adjust.apply_to_shared_shown(&shared, width, height, part),
        (true, true) => adjust.apply_to_shared_grey(&shared, width, height, part),
        (false, false) => {
            let (display, luma) = adjust.apply_to_shared(&shared, width, height, part);
            (Shown::Bytes(display), luma)
        }
        (false, true) => {
            let luma = adjust.luma_of_shared(&shared, width, height, part);
            (Shown::Bytes(luma.to_rgba()), luma)
        }
    }
}

/// A texture of the photo's pixels `shown`, `width` by `height`: laid on
/// the graphics card where the window is drawn by it and they fit within
/// `limit`, the most a texture may measure on a side; and otherwise
/// egui's own, shrunk to fit ([`texture_image`]).
fn photo_texture(screen: Option<&screen::Screen>, ctx: &egui::Context, name: &str, shown: Shown, (width, height): (usize, usize), limit: usize) -> screen::Texture {
    match screen {
        Some(screen) if width.max(height) <= limit => screen::Texture::Card(screen.laid((&shown).into(), (width, height))),
        _ => screen::Texture::Egui(ctx.load_texture(name, texture_image(shown.into_bytes(), width, height, limit), egui::TextureOptions::LINEAR)),
    }
}

/// [`photo_texture`], laid into `texture` where it serves.
fn lay_photo_texture(screen: Option<&screen::Screen>, texture: &mut screen::Texture, shown: Shown, (width, height): (usize, usize), limit: usize) {
    match (screen, &mut *texture) {
        (Some(screen), _) if width.max(height) <= limit => screen.lay(texture, (&shown).into(), (width, height)),
        (_, screen::Texture::Egui(handle)) => handle.set(texture_image(shown.into_bytes(), width, height, limit), egui::TextureOptions::LINEAR),
        (Some(screen), screen::Texture::Card(_)) => *texture = photo_texture(Some(screen), screen.ctx(), "photo", shown, (width, height), limit),
        (None, screen::Texture::Card(_)) => unreachable!("a texture on the card is the screen's"),
    }
}

/// Pixels as a texture holds them: as they are when they fit within
/// `limit` on each side, the most a texture may measure on this machine,
/// and otherwise shrunk to fit. Only what is drawn is shrunk. The
/// outline, the sampling and the search are in the photo's own pixels,
/// and the texture is stretched over them; zoomed far into a photo
/// larger than the limit, the canvas shows less than the sampling reads,
/// and a crop, which is smaller, is drawn in full.
pub(crate) fn texture_image(rgba: Vec<u8>, width: usize, height: usize, limit: usize) -> egui::ColorImage {
    let (shown_width, shown_height) = texture_size(width, height, limit);
    if (shown_width, shown_height) == (width, height) {
        return egui::ColorImage::from_rgba_unmultiplied([width, height], &rgba);
    }
    let shown = resize(&rgba, width, height, 4, shown_width, shown_height);
    egui::ColorImage::from_rgba_unmultiplied([shown_width, shown_height], &shown)
}

/// [`texture_image`] for one channel, a byte a pixel.
pub(crate) fn texture_gray(gray: Vec<u8>, width: usize, height: usize, limit: usize) -> egui::ColorImage {
    let (shown_width, shown_height) = texture_size(width, height, limit);
    if (shown_width, shown_height) == (width, height) {
        return egui::ColorImage::from_gray([width, height], &gray);
    }
    let shown = resize(&gray, width, height, 1, shown_width, shown_height);
    egui::ColorImage::from_gray([shown_width, shown_height], &shown)
}

/// The size an image `width` by `height` is drawn through a texture at:
/// its own, or the largest of its shape that fits within `limit`.
fn texture_size(width: usize, height: usize, limit: usize) -> (usize, usize) {
    let longest = width.max(height);
    if longest <= limit.max(1) {
        return (width, height);
    }
    let scale = limit.max(1) as f64 / longest as f64;
    (((width as f64 * scale) as usize).clamp(1, limit.max(1)), ((height as f64 * scale) as usize).clamp(1, limit.max(1)))
}

/// The detections of every family as one list, what decodes first and
/// then the best fit: parts of a larger, credible detection dropped, and
/// two detections of one symbol listed once, as
/// [`search_for_every_family`] describes.
fn merge_detections(mut all: Vec<Detection>) -> Vec<Detection> {
    let quality =
        |found: &Detection| (found.decodes == Some(true), found.score(), found.info.named_version == Some(found.version));
    all.sort_by(|a, b| {
        let (qa, qb) = (quality(a), quality(b));
        qb.0.cmp(&qa.0).then(qb.1.total_cmp(&qa.1)).then(qb.2.cmp(&qa.2))
    });
    let mut merged: Vec<Detection> = Vec::new();
    for found in all {
        if merged.iter().any(|taken| part_of(&found, taken) || same_symbol(taken, &found)) {
            continue;
        }
        // A finder pattern often reads better as a Micro QR symbol than
        // the damaged symbol it belongs to reads as a whole, and so comes
        // before it: the parts already taken go when their whole arrives.
        merged.retain(|taken| !part_of(taken, &found));
        merged.push(found);
    }
    merged
}

/// Whether `small` is a part of `big`: its centre lies inside `big`'s
/// outline, which is larger and credible, and it does not decode as
/// found. One that does is a symbol set within another, where a logo
/// might be, and no part of it: a finder pattern and the data beside it
/// fit as a symbol often enough, and do not decode as one.
fn part_of(small: &Detection, big: &Detection) -> bool {
    credible(big)
        && small.decodes != Some(true)
        && quad_area(&big.corners) > quad_area(&small.corners)
        && quad_contains(&big.corners, quad_centre(&small.corners))
}

/// Whether a detection is to be believed a symbol: it decodes as found,
/// or reads [`CREDIBLE_SCORE`] of its fixed modules right.
fn credible(found: &Detection) -> bool {
    found.decodes == Some(true) || found.score() >= CREDIBLE_SCORE
}

/// Whether two detections outline the same symbol.
fn same_symbol(a: &Detection, b: &Detection) -> bool {
    same_outline(&a.corners, &b.corners)
}

/// Whether two outlines lie on the same symbol: one's centre lies inside
/// the other, and their areas are within a factor of two.
fn same_outline(a: &[[f64; 2]; 4], b: &[[f64; 2]; 4]) -> bool {
    let (area_a, area_b) = (quad_area(a), quad_area(b));
    if !(area_a > 0.0 && area_b > 0.0) || area_a > 2.0 * area_b || area_b > 2.0 * area_a {
        return false;
    }
    quad_contains(a, quad_centre(b)) || quad_contains(b, quad_centre(a))
}

/// The symbol found that the outline was put on and still lies on, if
/// there is one: listed as in use, and not offered again on the canvas.
fn proposal_in_use(proposals: &Proposals, outline: &[[f64; 2]; 4]) -> Option<usize> {
    proposals.adopted.filter(|&i| proposals.found.get(i).is_some_and(|found| same_outline(&found.corners, outline)))
}

/// The symbol found to load of itself, where reads are loaded, among those
/// `decodes` says read through to a message: the one the outline lies
/// on, when it lies on one, and otherwise the first in the list. With
/// the outline on a symbol that does not decode, none: the grid is then
/// that symbol's, being worked on, and is not swapped for another's
/// unasked. `adopted` is the symbol the outline was last put on, which is
/// the one it lies on where two found lie over one another.
fn symbol_to_load(found: &[Detection], adopted: Option<usize>, outline: &[[f64; 2]; 4], decodes: impl Fn(usize) -> bool) -> Option<usize> {
    let lies_on = |i: &usize| found.get(*i).is_some_and(|found| same_outline(&found.corners, outline));
    match adopted.filter(lies_on).or_else(|| (0..found.len()).find(lies_on)) {
        Some(i) => decodes(i).then_some(i),
        None => (0..found.len()).find(|&i| decodes(i)),
    }
}

/// The placement the outline was put on and still lies on, if there is
/// one, as [`proposal_in_use`] has it of a symbol found. The placements
/// are of one symbol and lie over one another, so where the outline lies
/// names none of them: only the one that was picked can be in use.
fn placement_in_use(placements: &Placements, outline: &[[f64; 2]; 4]) -> Option<usize> {
    placements.adopted.filter(|&i| placements.found.get(i).is_some_and(|found| same_outline(&found.corners, outline)))
}

fn quad_centre(quad: &[[f64; 2]; 4]) -> [f64; 2] {
    [quad.iter().map(|p| p[0]).sum::<f64>() / 4.0, quad.iter().map(|p| p[1]).sum::<f64>() / 4.0]
}

/// Area of a quadrilateral by the shoelace formula.
fn quad_area(quad: &[[f64; 2]; 4]) -> f64 {
    let mut twice = 0.0;
    for i in 0..4 {
        let (p, q) = (quad[i], quad[(i + 1) % 4]);
        twice += p[0] * q[1] - q[0] * p[1];
    }
    twice.abs() / 2.0
}

/// Whether a point lies inside a convex quadrilateral: it is on the same
/// side of every edge.
fn quad_contains(quad: &[[f64; 2]; 4], point: [f64; 2]) -> bool {
    let mut sign = 0.0f64;
    for i in 0..4 {
        let (p, q) = (quad[i], quad[(i + 1) % 4]);
        let cross = (q[0] - p[0]) * (point[1] - p[1]) - (q[1] - p[1]) * (point[0] - p[0]);
        if cross == 0.0 {
            continue;
        }
        if sign == 0.0 {
            sign = cross.signum();
        } else if cross.signum() != sign {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use qr_forensics::decode::list::LIST_COST_MAX;
    use qr_forensics::decode::pipeline::{BlockOutcome, BlockReport};
    use qr_forensics::decode::rs::RsFailure;
    use qr_forensics::imaging::detect::InfoEvidence;
    use qr_forensics::imaging::enhance::Sharpen;
    use qr_forensics::imaging::projection::Project;
    use qr_forensics::symbol::grid::{CERTAIN, ModuleGrid};

    use super::*;

    fn square(x: f64, y: f64, side: f64) -> [[f64; 2]; 4] {
        [[x, y], [x + side, y], [x + side, y + side], [x, y + side]]
    }

    #[test]
    fn quad_geometry() {
        let big = square(0.0, 0.0, 100.0);
        assert_eq!(quad_area(&big), 10_000.0);
        assert_eq!(quad_centre(&big), [50.0, 50.0]);
        assert!(quad_contains(&big, [50.0, 50.0]));
        assert!(quad_contains(&big, [1.0, 99.0]));
        assert!(!quad_contains(&big, [101.0, 50.0]));
        // Corner order the other way round (as a mirrored photo gives it)
        // still contains its inside.
        let reversed = [big[0], big[3], big[2], big[1]];
        assert!(quad_contains(&reversed, [50.0, 50.0]));
        assert!(!quad_contains(&reversed, [-1.0, 50.0]));
    }

    /// A photo as the app loads it, for the search helpers.
    fn load(name: &str) -> Luma {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/photos").join(name);
        qr_forensics::imaging::photo::luma_as_loaded(image::open(&path).expect("photo opens").to_rgba8())
    }

    #[test]
    fn every_family_is_searched_and_the_best_fit_leads() {
        let params = SampleParams::default();
        // A QR Code, whose finder patterns each pass for the finder of a
        // Micro QR or rMQR symbol when those families are searched: they
        // are the QR Code's, and nothing is made of them, neither inside
        // it nor hung off its corners over the quiet zone.
        let found = search_for_every_family(&load("zxing_qrcode-3_01.png"), &params, &SearchParams::default(), &DecodeOptions::default()).found;
        let first = found.first().expect("the QR Code is found");
        assert_eq!(first.version.family(), Family::Qr, "{:?}", found.iter().map(|f| f.version).collect::<Vec<_>>());
        assert!(first.score() > 0.95);
        assert_eq!(first.decodes, Some(true));
        assert_eq!(found.len(), 1, "{:?}", found.iter().map(|f| (f.version, f.score())).collect::<Vec<_>>());
        // A Micro QR, with QR Code selected as it would be by default.
        let found = search_for_every_family(&load("cpp_micro_1.png"), &params, &SearchParams::default(), &DecodeOptions::default()).found;
        let first = found.first().expect("the Micro QR is found");
        assert_eq!(first.version, SymbolVersion::Micro(3));
        assert!(!first.inverted);
        // A light-on-dark rMQR: found, and known to be inverted.
        let found = search_for_every_family(&load("cpp_rmqr_R7x43-H_inverted.png"), &params, &SearchParams::default(), &DecodeOptions::default()).found;
        let first = found.first().expect("the rMQR is found");
        assert_eq!(first.version.family(), Family::Rmqr);
        assert!(first.inverted);
    }

    #[test]
    fn detections_of_one_symbol_are_listed_once() {
        let params = SampleParams::default();
        let found = search_for_every_family(&load("cpp_micro_1.png"), &params, &SearchParams::default(), &DecodeOptions::default()).found;
        for (i, a) in found.iter().enumerate() {
            for b in &found[i + 1..] {
                assert!(!same_symbol(a, b), "{} and {} outline the same symbol", a.version, b.version);
            }
        }
    }

    #[test]
    fn a_part_is_not_listed_with_its_whole_whichever_fits_better() {
        let scored = |version, corners, agreement| Detection { agreement, ..detection(version, corners) };
        // A finder pattern read as a Micro QR symbol, fitting a little
        // better than the QR Code it is a corner of, and another Micro QR
        // symbol beside them.
        let whole = scored(SymbolVersion::Qr(2), square(0.0, 0.0, 250.0), (93, 100));
        let part = scored(SymbolVersion::Micro(3), square(0.0, 0.0, 110.0), (94, 100));
        let beside = scored(SymbolVersion::Micro(3), square(400.0, 0.0, 110.0), (90, 100));
        assert!(part_of(&part, &whole) && !part_of(&beside, &whole));
        for found in [vec![part.clone(), whole.clone(), beside.clone()], vec![beside.clone(), whole.clone(), part.clone()]] {
            let merged = merge_detections(found);
            assert_eq!(merged.iter().map(|d| d.version).collect::<Vec<_>>(), [SymbolVersion::Qr(2), SymbolVersion::Micro(3)]);
            assert_eq!(merged[1].corners, beside.corners);
        }
        // The same the other way about, the whole fitting better.
        let better = scored(SymbolVersion::Qr(2), square(0.0, 0.0, 250.0), (99, 100));
        assert_eq!(merge_detections(vec![part.clone(), better]).len(), 1);
        // A whole that fits too poorly to be believed takes no parts with it.
        let poor = scored(SymbolVersion::Qr(2), square(0.0, 0.0, 250.0), (50, 100));
        assert_eq!(merge_detections(vec![part, poor]).len(), 2);
    }

    #[test]
    fn what_decodes_as_found_leads_whatever_fits_better() {
        let claim = |version, corners, agreement, decodes| Detection { agreement, decodes: Some(decodes), ..detection(version, corners) };
        // Two claims on one place: the one that decodes is the symbol,
        // though the other's fixed patterns fit better. A symbol
        // elsewhere that fits best of all and does not decode comes after.
        let decoded = claim(SymbolVersion::Qr(2), square(0.0, 0.0, 250.0), (85, 100), true);
        let fitted = claim(SymbolVersion::Qr(3), square(5.0, 5.0, 250.0), (95, 100), false);
        let elsewhere = claim(SymbolVersion::Micro(3), square(400.0, 0.0, 110.0), (99, 100), false);
        let merged = merge_detections(vec![fitted, elsewhere, decoded]);
        assert_eq!(merged.iter().map(|d| d.version).collect::<Vec<_>>(), [SymbolVersion::Qr(2), SymbolVersion::Micro(3)]);
        // A whole that decodes is believed at a fit that would not be
        // otherwise, and takes its parts with it.
        let poor = claim(SymbolVersion::Qr(2), square(0.0, 0.0, 250.0), (70, 100), true);
        let part = claim(SymbolVersion::Micro(3), square(0.0, 0.0, 110.0), (94, 100), false);
        assert!(!part_of(&part, &Detection { decodes: Some(false), ..poor.clone() }));
        assert_eq!(merge_detections(vec![part, poor.clone()]).len(), 1);
        // A symbol set within another decodes, which no part of one
        // does, and is listed with it whichever comes first.
        let within = claim(SymbolVersion::Qr(1), square(100.0, 100.0, 40.0), (99, 100), true);
        assert!(!part_of(&within, &poor));
        for found in [vec![within.clone(), poor.clone()], vec![poor, within]] {
            assert_eq!(merge_detections(found).len(), 2);
        }
    }

    #[test]
    fn a_read_that_did_not_decode_is_ranked_by_what_it_has_to_show() {
        // A QR Code laid on a Micro QR symbol, as `cpp_micro_9.webp` reads
        // when asked about as one: seven in ten of its many fixed modules
        // right, which is more modules than the Micro QR symbol has, and
        // nothing to set against the Micro QR symbol's own clean fit.
        assert!(standing_of(0, (78, 78)) > standing_of(0, (144, 203)));
        // A finder pattern of a damaged QR Code fits well as a Micro QR
        // symbol, and has less to show than the whole it is a corner of.
        assert!(standing_of(0, (195, 230)) > standing_of(0, (77, 82)));
        // A block that verified outweighs any fit.
        assert!(standing_of(1, (150, 230)) > standing_of(0, (85, 85)));
    }

    /// A version 2 symbol photographed over-exposed, as BoofCV's
    /// photograph of one is (`brightness/image012.jpg`): its dark thinned
    /// by a share of a module at every boundary.
    struct Thinned {
        luma: Luma,
        /// Module coordinates onto the photo.
        truth: qr_forensics::imaging::homography::Homography,
        module_px: f64,
    }

    impl Thinned {
        const VERSION: SymbolVersion = SymbolVersion::Qr(2);
        /// An outline drawn loosely round the symbol.
        const AROUND: [[f64; 2]; 4] = [[120.0, 40.0], [690.0, 90.0], [640.0, 595.0], [65.0, 535.0]];

        /// The symbol with its dark thinned by `r` of a module at every
        /// boundary.
        fn new(r: f64) -> Self {
            use qr_forensics::imaging::homography::Homography;
            use qrcode_generator::{Segment, qr};
            let symbol = qr::Encoder::new(qr::ErrorCorrection::Low)
                .version(qr::Version::new(2).unwrap())
                .boost_error_correction(false)
                .mask(qr::Mask::new(0).unwrap())
                .encode_segments(&[Segment::bytes("the dark thinned at every edge")])
                .unwrap();
            let grid = ModuleGrid::from_bools(symbol.width(), symbol.height(), symbol.modules());
            let quad = [[150.5, 70.25], [655.0, 120.5], [610.75, 560.0], [95.25, 500.5]];
            let (gw, gh) = (grid.width() as f64, grid.height() as f64);
            let truth = Homography::rect_to_quad(gw, gh, quad).unwrap();
            let dark_at = |mx: f64, my: f64| mx >= 0.0 && my >= 0.0 && mx < gw && my < gh && grid.get(mx as usize, my as usize) == Module::Dark;
            let to_grid = truth.inverse().unwrap();
            let (width, height) = (760, 640);
            let mut data = Vec::with_capacity(width * height);
            for py in 0..height {
                for px in 0..width {
                    let mut sum = 0.0;
                    for (sx, sy) in (0..3).flat_map(|sy| (0..3).map(move |sx| (sx, sy))) {
                        let [mx, my] = to_grid.apply(px as f64 + (sx as f64 + 0.5) / 3.0, py as f64 + (sy as f64 + 0.5) / 3.0);
                        let dark = [[0.0, 0.0], [r, 0.0], [-r, 0.0], [0.0, r], [0.0, -r], [r, r], [-r, -r], [r, -r], [-r, r]]
                            .iter()
                            .all(|[dx, dy]| dark_at(mx + dx, my + dy));
                        sum += if dark { 0.08f32 } else { 0.92 };
                    }
                    data.push(sum / 9.0);
                }
            }
            Self { luma: Luma::new(width, height, data), truth, module_px: (quad[1][0] - quad[0][0]) / gw }
        }

        /// How far, in modules, `px` lies from where the photo has the
        /// module point `at`.
        fn off(&self, at: [f64; 2], [x, y]: [f64; 2]) -> f64 {
            let [tx, ty] = self.truth.apply(at[0], at[1]);
            (x - tx).hypot(y - ty) / self.module_px
        }
    }

    #[test]
    fn refine_places_a_symbol_whose_dark_has_thinned_expecting_the_bleed() {
        // Over-exposed as BoofCV's photograph of a version 2 symbol is
        // (`brightness/image012.jpg`): the dark thinned by three tenths
        // of a module at every boundary. Found as printed, a finder may
        // sit as far off as the bleed to one side of itself, and a corner
        // traced against the quiet zone lies as far in. Expecting the
        // bleed, the anchors come back onto their patterns, and the
        // corners must come with them, or the map, held at the anchors
        // and stretched to the corners, runs off between the two.
        let (func, params) = (FunctionMap::new(Thinned::VERSION), SampleParams::default());
        let (gw, gh) = (func.width as f64, func.height as f64);
        let refine = |luma: &Luma| {
            let found = detect_candidates(luma, Thinned::VERSION, Thinned::AROUND, &params).into_iter().next().expect("the symbol is found");
            refine_outline(luma, &func, found.corners, found.anchors, &[], &params).expect("the corners refine")
        };
        let thinned = Thinned::new(0.3);
        let off = |at, px| thinned.off(at, px);
        let refined = refine(&thinned.luma);
        assert!(refined.bloom > 0.2, "placed expecting a bleed of {:.2} modules", refined.bloom);
        let (after, before) = (refined.refinement.after, refined.refinement.before);
        assert!(after.0 > before.0, "fixed patterns match {after:?}, from {before:?}");
        for (at, px) in [[0.0, 0.0], [gw, 0.0], [gw, gh], [0.0, gh]].into_iter().zip(refined.refinement.corners) {
            assert!(off(at, px) < 0.15, "corner {at:?} placed {:.2} modules off", off(at, px));
        }
        for anchor in refined.after.iter().filter(|a| matches!(a.kind, AnchorKind::Finder | AnchorKind::Alignment)) {
            assert!(off(anchor.at, anchor.center) < 0.1, "{:?} at {:?} placed {:.2} modules off", anchor.kind, anchor.at, off(anchor.at, anchor.center));
        }
        // A symbol that has not bled is refined as printed.
        assert_eq!(refine(&Thinned::new(0.0).luma).bloom, 0.0);
    }

    #[test]
    fn a_symbol_whose_dark_has_thinned_stays_placed_as_refine_is_pressed_again() {
        use qr_forensics::imaging::detect::edge_sites;
        // The symbol whose dark has thinned by three tenths of a module,
        // with an edge point dropped by hand on the edge of its ink at
        // every place the window offers one, Refine pressed again and
        // again as the window takes each result in. An edge point is
        // never looked for again, only moved out for the bleed, and once
        // was moved out by the whole bleed every time: the second time
        // the placement through edge points moved twice read worse than
        // the fit as printed, which was kept with the finders four tenths
        // of a module off, and the third time the edge points came out a
        // quarter of a module beyond the symbol's edge. And the top-left
        // finder, refitted through corners fitted as printed, was now and
        // then settled squeezed, and the corner its fit placed came out as
        // much as half a module off, with five of the fixed modules wrong.
        let (func, params) = (FunctionMap::new(Thinned::VERSION), SampleParams::default());
        let (gw, gh) = (func.width as f64, func.height as f64);
        let r = 0.3;
        let thinned = Thinned::new(r);
        let found = detect_candidates(&thinned.luma, Thinned::VERSION, Thinned::AROUND, &params).into_iter().next().expect("the symbol is found");
        let mut doc = Document::blank(Thinned::VERSION);
        doc.corners = found.corners;
        for &anchor in found.anchors.iter().filter(|a| a.kind != AnchorKind::Edge) {
            doc.set_anchor(anchor, AnchorSource::Detected);
        }
        for site in edge_sites(&func) {
            let [x, y] = site.at;
            let inward = |v: f64, end: f64| if v <= 0.0 { r } else if v >= end { -r } else { 0.0 };
            let center = thinned.truth.apply(x + inward(x, gw), y + inward(y, gh));
            doc.set_anchor(Anchor { kind: AnchorKind::Edge, at: site.at, center, partial: false, inferred: false }, AnchorSource::Manual);
        }
        for press in 1..=6 {
            let anchors: Vec<Anchor> = doc.anchors.iter().map(|a| a.anchor).collect();
            let already: Vec<f64> = doc.anchors.iter().map(|a| a.bled).collect();
            let refined = refine_outline(&thinned.luma, &func, doc.corners, anchors, &already, &params).expect("the corners refine");
            assert!(refined.bloom > 0.2, "press {press}: placed expecting a bleed of {:.2} modules", refined.bloom);
            let (matching, fixed) = refined.refinement.after;
            assert_eq!(matching, fixed, "press {press}: fixed patterns match {matching} of {fixed}");
            // Taken in as the window takes it in ([`App::finish_refine`]).
            doc.corners = refined.refinement.corners;
            for (old, new) in refined.before.iter().zip(&refined.after) {
                if old.center != new.center {
                    let source = doc.anchor_at(old.at).map_or(AnchorSource::Detected, |a| a.source);
                    doc.set_anchor_bled(*new, source, if new.kind == AnchorKind::Edge { refined.bloom } else { 0.0 });
                }
            }
            for anchor in doc.anchors.iter().map(|a| a.anchor).filter(|a| matches!(a.kind, AnchorKind::Edge | AnchorKind::Finder | AnchorKind::Alignment)) {
                let off = thinned.off(anchor.at, anchor.center);
                assert!(off < 0.1, "press {press}: {:?} at {:?} placed {off:.2} modules off", anchor.kind, anchor.at);
            }
            for (at, px) in [[0.0, 0.0], [gw, 0.0], [gw, gh], [0.0, gh]].into_iter().zip(doc.corners) {
                let off = thinned.off(at, px);
                assert!(off < 0.15, "press {press}: corner {at:?} placed {off:.2} modules off");
            }
        }
    }

    #[test]
    fn a_placement_made_expecting_a_bleed_is_adopted_after_invert_photo_is_changed() {
        // Placed expecting the bleed with Invert photo off, and adopted
        // once it is on: the detection says the symbol reads dark on
        // light, against the photo as it was searched, and the photo is
        // now the other way about. Refined at the polarity found, every
        // pattern was looked for the wrong way round, and the bleed with
        // it.
        let (func, params) = (Arc::new(FunctionMap::new(Thinned::VERSION)), SampleParams::default());
        let (gw, gh) = (func.width as f64, func.height as f64);
        let thinned = Thinned::new(0.3);
        let read = read_outline(&thinned.luma, Thinned::VERSION, Thinned::AROUND, &params, &DecodeOptions::default()).expect("the symbol is read");
        let found = read.detection;
        assert!(found.bloom > 0.2 && !found.inverted, "read expecting a bleed of {:.2} modules, inverted {}", found.bloom, found.inverted);
        let inverted = Luma::new(thinned.luma.width, thinned.luma.height, thinned.luma.data.iter().map(|v| 1.0 - v).collect());
        let prepared = prepare_adoption(&inverted, &func, Thinned::VERSION, found, false, true, &params);
        let (matching, fixed) = prepared.agreement;
        assert_eq!(matching, fixed, "fixed patterns match {matching} of {fixed}");
        for anchor in prepared.found.anchors.iter().filter(|a| matches!(a.kind, AnchorKind::Finder | AnchorKind::Alignment)) {
            let off = thinned.off(anchor.at, anchor.center);
            assert!(off < 0.1, "{:?} at {:?} placed {off:.2} modules off", anchor.kind, anchor.at);
        }
        for (at, px) in [[0.0, 0.0], [gw, 0.0], [gw, gh], [0.0, gh]].into_iter().zip(prepared.found.corners) {
            let off = thinned.off(at, px);
            assert!(off < 0.15, "corner {at:?} placed {off:.2} modules off");
        }
    }

    #[test]
    fn an_outline_is_read_as_another_family_when_its_own_does_not_decode() {
        // A Micro QR symbol inside the outline, asked about as a QR Code:
        // a QR Code fits it at the level of chance and decodes nothing.
        let (params, opts) = (SampleParams::default(), DecodeOptions::default());
        let luma = load("cpp_micro_1.png");
        let outline = [[210.3, 27.8], [423.2, 28.0], [423.2, 240.7], [210.3, 240.9]];
        let (read, other) = read_outline_of_any_family(&luma, SymbolVersion::Qr(2), outline, &params, &opts).expect("a symbol");
        assert_eq!(other, Some(Family::Micro));
        assert!(read.is_trustworthy());
        assert_eq!(read.text(), "Wikipedia");
        // Asked about as what it is, it is read as that, and no other
        // family comes into it.
        let (read, other) = read_outline_of_any_family(&luma, SymbolVersion::Micro(2), outline, &params, &opts).expect("a symbol");
        assert_eq!(other, None);
        assert_eq!(read.text(), "Wikipedia");
    }

    #[test]
    fn the_binarised_views_threshold_can_be_set_for_the_scan() {
        use qr_forensics::imaging::binarize::{BinarizeParams, Binarizer};
        let mut bench = Bench::new();
        // The scan pools its three readings until the view's is set for it.
        assert_eq!(bench.app.scan_params(), bench.app.view.sample);
        assert_eq!(bench.app.scan_params().binarizer, Binarizer::Pooled);
        bench.app.view.binarize_for_scan = true;
        // Automatic, the view shows the scan's own local reading at the
        // reach the outline sets, and the scan reads by that alone.
        let photo = bench.app.photo.as_ref().expect("a photo is loaded");
        let (width, height) = (photo.width, photo.height);
        let expected = BinarizeParams::for_outline(bench.app.doc.corners, width, height);
        assert_eq!(bench.app.scan_params().binarizer, Binarizer::Local { radius: expected.radius as f64 });
        // It follows the outline as it stands when the step starts.
        bench.app.doc.corners = square(10.0, 10.0, 30.0);
        assert_eq!(bench.app.scan_params().binarizer, Binarizer::Local { radius: BinarizeParams::MIN_RADIUS as f64 });
        // Set by hand, it is what was set.
        bench.app.view.binarize_auto = false;
        bench.app.view.binarize = BinarizeParams { window: Some(8), ..expected };
        assert_eq!(bench.app.scan_params().binarizer, Binarizer::Window { radius: 8.0 });
        bench.app.view.binarize = BinarizeParams { global: true, ..expected };
        assert_eq!(bench.app.scan_params().binarizer, Binarizer::Global);
        // The settings the sample preview is keyed by are untouched.
        assert_eq!(bench.app.view.sample.binarizer, Binarizer::Pooled);
        // The panels show the setting with the view off and on, and the
        // view, once shown, is of the threshold set.
        bench.app.view.align_overlay = AlignOverlay::FixedPatterns;
        bench.step();
        bench.app.view.align_overlay = AlignOverlay::Binarised;
        bench.step();
        let binarised = bench.app.photo.as_ref().and_then(|photo| photo.binarised.as_ref()).expect("the view was made");
        assert_eq!(binarised.params.binarizer(), Binarizer::Global);
        // And over the outline as it stands, not as it stood when the
        // threshold was set by hand: the scan takes its level from there.
        assert_ne!(binarised.params.region, expected.region);
        assert_eq!(binarised.params.region, BinarizeParams::for_outline(bench.app.doc.corners, width, height).region);
        // Unticked, the scan pools again.
        bench.app.view.binarize_for_scan = false;
        assert_eq!(bench.app.scan_params().binarizer, Binarizer::Pooled);
    }

    #[test]
    fn an_anchor_replaced_keeps_its_place() {
        let mut doc = Document::blank(SymbolVersion::Qr(2));
        let traced = |x: f64, px: f64| anchor(AnchorKind::Edge, [x, 0.0], [px, 0.0]);
        for x in [5.0, 9.0, 14.0] {
            doc.set_anchor(traced(x, x), AnchorSource::Detected);
        }
        doc.set_anchor(traced(5.0, 99.0), AnchorSource::Manual);
        assert_eq!(doc.anchors.iter().map(|a| a.anchor.at[0]).collect::<Vec<_>>(), [5.0, 9.0, 14.0]);
        assert_eq!((doc.anchors[0].anchor.center, doc.anchors[0].source), ([99.0, 0.0], AnchorSource::Manual));
    }

    #[test]
    fn a_checkpoint_for_a_change_that_never_came_costs_nothing() {
        let mut history = History::default();
        let mut doc = Document::blank(SymbolVersion::Qr(2));
        let blank = doc.clone();
        history.checkpoint(&doc);
        doc.grid.set(10, 10, Module::Dark);
        let painted = doc.clone();
        assert!(history.undo(&mut doc) && doc == blank);
        // A press that paints nothing, on a module already that colour or
        // beside the grid: the checkpoint is taken, and nothing follows.
        history.checkpoint(&doc);
        assert!(history.can_redo(&doc));
        assert!(!history.can_undo(&doc), "the press left no step to undo");
        assert!(history.redo(&mut doc) && doc == painted);
        assert!(history.undo(&mut doc) && doc == blank);
        assert!(!history.undo(&mut doc));
        // A press that does paint leaves the undone state behind.
        history.checkpoint(&doc);
        doc.grid.set(11, 11, Module::Light);
        assert!(!history.can_redo(&doc));
        assert!(!history.redo(&mut doc));
        assert!(history.undo(&mut doc) && doc == blank);
    }

    fn anchor(kind: AnchorKind, at: [f64; 2], center: [f64; 2]) -> Anchor {
        Anchor { kind, at, center, partial: false, inferred: false }
    }

    /// An anchor on the finder pattern at the top left.
    fn finder_anchor(center: [f64; 2]) -> Anchor {
        anchor(AnchorKind::Finder, [3.5, 3.5], center)
    }

    #[test]
    fn undo_on_another_photo_brings_back_the_symbol_and_not_the_outline() {
        let mut history = History::default();
        let mut doc = Document::blank(SymbolVersion::Qr(2));
        // On the first photo: an outline, an anchor and a lens term, and
        // then a module painted.
        doc.corners = square(10.0, 10.0, 100.0);
        doc.set_anchor(finder_anchor([24.0, 24.0]), AnchorSource::Detected);
        doc.adjust.k1 = 0.05;
        history.checkpoint(&doc);
        doc.grid.set(10, 10, Module::Dark);
        // The second photo, as loading it leaves the document.
        history.photo_changed(&doc);
        doc.corners = square(50.0, 50.0, 300.0);
        doc.anchors.clear();
        doc.adjust = Adjust::default();
        let on_second = doc.clone();
        assert!(history.undo(&mut doc));
        assert_eq!(doc.grid.get(10, 10), Module::Unknown, "the module painted is taken back");
        assert_eq!(doc.corners, on_second.corners, "the outline is still the second photo's");
        assert!(doc.anchors.is_empty() && doc.adjust == Adjust::default());
        assert!(history.redo(&mut doc) && doc == on_second);
        // Loading a photo is no step of its own: once the symbol is as it
        // was, there is nothing further to undo.
        assert!(history.undo(&mut doc));
        assert!(!history.can_undo(&doc) && !history.undo(&mut doc));
    }

    #[test]
    fn an_export_of_an_unsolved_symbol_says_so() {
        let mut app = App::blank();
        let path = std::env::temp_dir().join(format!("qr-forensics-export-test-{}.png", std::process::id()));
        app.export_image(&path, ExportFormat::Png);
        let _ = std::fs::remove_file(&path);
        let notice = &app.message.as_ref().expect("a notice").text;
        assert!(notice.starts_with("Saved "), "{notice}");
        assert!(notice.contains("Not every block is solved (0 of 1)"), "{notice}");
        // As a drawing, the same symbol at the same size, and the same
        // word about it.
        let path = path.with_extension("svg");
        app.export_image(&path, ExportFormat::Svg);
        let drawing = std::fs::read_to_string(&path).expect("the drawing was written");
        let _ = std::fs::remove_file(&path);
        assert!(drawing.starts_with("<svg ") && drawing.contains("viewBox=\"0 0 33 33\" width=\"330\" height=\"330\""), "{drawing}");
        let notice = &app.message.as_ref().expect("a notice").text;
        assert!(notice.contains("(330 × 330)") && notice.contains("Not every block is solved (0 of 1)"), "{notice}");
    }

    // The tests from here on show the window, frame by frame, to an egui
    // context with no display behind it, as egui's own tests run a UI:
    // the frame's order of work, and what a click lands on, are the
    // things under test.

    /// Corners of the symbol in the photo [`Bench::new`] loads.
    const STRAIGHT: [[f64; 2]; 4] = [[40.0, 40.0], [290.0, 40.0], [290.0, 290.0], [40.0, 290.0]];

    /// A version 2 QR Code grid with every module painted: the fixed
    /// patterns, and data that is no message.
    fn painted_grid() -> ModuleGrid {
        let func = FunctionMap::new(SymbolVersion::Qr(2));
        let mut grid = func.blank_grid();
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        for y in 0..func.height {
            for x in 0..func.width {
                if func.expected(x, y).is_none() {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    grid.set(x, y, Module::from_dark(seed & 1 == 1));
                }
            }
        }
        grid
    }

    /// A detection of `version` at `corners`, with a fit good enough to
    /// be adopted, as a job hands one back.
    fn detection(version: SymbolVersion, corners: [[f64; 2]; 4]) -> Detection {
        Detection {
            corners,
            version,
            inverted: false,
            mirrored: false,
            mirrored_by_decode: false,
            turns: 0,
            anchors: Vec::new(),
            agreement: (100, 100),
            info: InfoEvidence::default(),
            edged_corners: Vec::new(),
            measured_corners: Vec::new(),
            retry: None,
            turned: 0,
            decodes: None,
            bloom: 0.0,
            decoded_at: None,
        }
    }

    fn prepared(found: Detection) -> Prepared {
        Prepared { agreement: found.agreement, found, guesses: None, searched_inverted: false, read: None, result: None }
    }

    /// End a worker thread without a result, as a panic in it does, but
    /// without the panic hook's message in the test's output.
    fn die<T>() -> T {
        std::panic::resume_unwind(Box::new("a worker dying, for the test"))
    }

    struct Bench {
        ctx: egui::Context,
        app: App,
        /// The window's size, in points.
        size: egui::Vec2,
        /// The most a texture may measure on a side, when the window
        /// says: `None` leaves it at what egui allows unasked, 2048.
        texture_side: Option<usize>,
        time: f64,
        /// What the last frame painted.
        shapes: Vec<egui::epaint::ClippedShape>,
        /// What each texture was last set to, whole.
        textures: std::collections::HashMap<egui::TextureId, Arc<egui::ColorImage>>,
    }

    impl Bench {
        /// A fresh app with a render of [`painted_grid`] as its photo,
        /// ten pixels a module with its quiet zone, in the Align stage.
        /// Loading searches nothing, so no symbol is on offer.
        fn new() -> Self {
            let ctx = egui::Context::default();
            let mut app = App::blank();
            // A notice is read by the test, not by someone watching the
            // clock, and a frame of decoding on a hopeless grid takes
            // seconds in a debug build: notices stay until the next edit.
            app.notice_seconds = f32::INFINITY;
            let (w, h, pixels) = render_rgba(&painted_grid(), 10, 4);
            let image = image::RgbaImage::from_raw(w as u32, h as u32, pixels).expect("a render is an image");
            app.set_photo(&ctx, "render".into(), image);
            let mut bench =
                Self { ctx, app, size: egui::vec2(1440.0, 900.0), texture_side: None, time: 0.0, shapes: Vec::new(), textures: Default::default() };
            bench.settle();
            bench
        }

        fn frame(&mut self, events: Vec<egui::Event>) {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, self.size)),
                max_texture_side: self.texture_side,
                time: Some(self.time),
                events,
                ..Default::default()
            };
            self.time += 1.0 / 60.0;
            let app = &mut self.app;
            let mut output = self.ctx.run_ui(input, |ui| app.show(ui));
            for (id, deltas) in output.textures_delta.set.drain() {
                for delta in deltas.into_iter().filter(|delta| delta.pos.is_none()) {
                    let egui::ImageData::Color(image) = delta.image;
                    self.textures.insert(id, image);
                }
            }
            output.textures_delta.clear();
            self.shapes = output.shapes;
        }

        fn step(&mut self) {
            self.frame(Vec::new());
        }

        /// The textures the last frame drew through.
        fn drawn_through(&self) -> Vec<egui::TextureId> {
            fn of(shape: &egui::Shape, ids: &mut Vec<egui::TextureId>) {
                match shape {
                    egui::Shape::Vec(shapes) => shapes.iter().for_each(|shape| of(shape, ids)),
                    shape => ids.push(shape.texture_id()),
                }
            }
            let mut ids = Vec::new();
            self.shapes.iter().for_each(|clipped| of(&clipped.shape, &mut ids));
            ids
        }

        /// Frames until no search and no job is under way, and one more.
        fn settle(&mut self) {
            let started = std::time::Instant::now();
            loop {
                self.step();
                if self.app.search.is_none() && self.app.job.is_none() {
                    break;
                }
                assert!(started.elapsed().as_secs() < 120, "a worker did not finish");
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            self.step();
        }

        /// Apply an action as the end of a frame applies it.
        fn act(&mut self, action: Action) {
            let ctx = self.ctx.clone();
            self.app.apply(&ctx, action);
        }

        fn button(&mut self, pos: egui::Pos2, button: egui::PointerButton, pressed: bool) {
            self.frame(vec![egui::Event::PointerButton { pos, button, pressed, modifiers: egui::Modifiers::NONE }]);
        }

        fn click(&mut self, pos: egui::Pos2, button: egui::PointerButton) {
            self.frame(vec![egui::Event::PointerMoved(pos)]);
            self.step();
            self.button(pos, button, true);
            self.button(pos, button, false);
            self.step();
        }

        /// Turn the wheel by `points`, negative to bring what is below
        /// into view, and let the scroll run its course.
        fn wheel(&mut self, points: f32) {
            self.frame(vec![egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: egui::vec2(0.0, points),
                phase: egui::TouchPhase::Move,
                modifiers: egui::Modifiers::NONE,
            }]);
            for _ in 0..30 {
                self.step();
            }
        }

        /// Where a point of the photo is on the canvas, by way of an
        /// anchor's handle, which is drawn where its anchor is.
        fn locate(&mut self) -> impl Fn([f64; 2]) -> egui::Pos2 + use<> {
            let saved = self.app.doc.anchors.clone();
            self.app.doc.anchors = vec![DocAnchor { anchor: finder_anchor([75.0, 75.0]), source: AnchorSource::Detected, bled: 0.0 }];
            self.step();
            self.step();
            let handles = self.handles(5.5);
            assert_eq!(handles.len(), 1, "{handles:?}");
            self.app.doc.anchors = saved;
            self.step();
            let (origin, scale) = (handles[0], self.app.view.align_view.scale);
            move |px| origin + egui::vec2(px[0] as f32 - 75.0, px[1] as f32 - 75.0) * scale
        }

        /// Centres of the filled circles of `radius` the last frame
        /// painted: the handles of anchors set are such, at 5.5 or, for
        /// timing modules and edge points, at 4.
        fn handles(&self, radius: f32) -> Vec<egui::Pos2> {
            fn walk(shape: &egui::Shape, radius: f32, out: &mut Vec<egui::Pos2>) {
                match shape {
                    egui::Shape::Circle(circle) if circle.radius == radius && circle.fill != egui::Color32::TRANSPARENT => {
                        out.push(circle.center);
                    }
                    egui::Shape::Vec(shapes) => shapes.iter().for_each(|shape| walk(shape, radius, out)),
                    _ => {}
                }
            }
            let mut out = Vec::new();
            self.shapes.iter().for_each(|clipped| walk(&clipped.shape, radius, &mut out));
            out
        }

        /// Every colour the meshes the last frame painted have a vertex
        /// of: the photo is one, in white, and the overlays on it others.
        fn mesh_colours(&self) -> Vec<egui::Color32> {
            fn walk(shape: &egui::Shape, out: &mut Vec<egui::Color32>) {
                match shape {
                    egui::Shape::Mesh(mesh) => {
                        for vertex in &mesh.vertices {
                            if !out.contains(&vertex.color) {
                                out.push(vertex.color);
                            }
                        }
                    }
                    egui::Shape::Vec(shapes) => shapes.iter().for_each(|shape| walk(shape, out)),
                    _ => {}
                }
            }
            let mut out = Vec::new();
            self.shapes.iter().for_each(|clipped| walk(&clipped.shape, &mut out));
            out
        }

        /// Every piece of text the last frame painted.
        fn texts(&self) -> Vec<String> {
            fn walk(shape: &egui::Shape, out: &mut Vec<String>) {
                match shape {
                    egui::Shape::Text(text) => out.push(text.galley.text().to_owned()),
                    egui::Shape::Vec(shapes) => shapes.iter().for_each(|shape| walk(shape, out)),
                    _ => {}
                }
            }
            let mut out = Vec::new();
            self.shapes.iter().for_each(|clipped| walk(&clipped.shape, &mut out));
            out
        }

        fn notice(&self) -> &Notice {
            self.app.message.as_ref().expect("a notice")
        }

        /// The photo as its texture last had it: the red, green and
        /// blue of each pixel.
        fn drawn(&self) -> Vec<[u8; 3]> {
            let id = self.app.photo.as_ref().expect("a photo is on show").texture.id();
            self.textures.get(&id).expect("the photo's texture was set").pixels.iter().map(|c| [c.r(), c.g(), c.b()]).collect()
        }

        /// The middle of where the last frame painted `text`, which it
        /// painted once.
        fn text_at(&self, text: &str) -> egui::Pos2 {
            fn walk(shape: &egui::Shape, text: &str, out: &mut Vec<egui::Pos2>) {
                match shape {
                    egui::Shape::Text(shape) if shape.galley.text() == text => out.push(shape.visual_bounding_rect().center()),
                    egui::Shape::Vec(shapes) => shapes.iter().for_each(|shape| walk(shape, text, out)),
                    _ => {}
                }
            }
            let mut out = Vec::new();
            self.shapes.iter().for_each(|clipped| walk(&clipped.shape, text, &mut out));
            assert_eq!(out.len(), 1, "{text:?} was painted at {out:?}");
            out[0]
        }

        /// The rail of the slider on the row labelled `label`, as the
        /// last frame painted it: a flat bar beside the label.
        fn rail_beside(&self, label: &str) -> egui::Rect {
            fn walk(shape: &egui::Shape, at: egui::Pos2, out: &mut Vec<egui::Rect>) {
                match shape {
                    egui::Shape::Rect(bar) => {
                        let rect = bar.rect;
                        if (rect.center().y - at.y).abs() < 6.0 && rect.left() > at.x && rect.width() > 60.0 && rect.height() < 12.0 {
                            out.push(rect);
                        }
                    }
                    egui::Shape::Vec(shapes) => shapes.iter().for_each(|shape| walk(shape, at, out)),
                    _ => {}
                }
            }
            let at = self.text_at(label);
            let mut out = Vec::new();
            self.shapes.iter().for_each(|clipped| walk(&clipped.shape, at, &mut out));
            assert_eq!(out.len(), 1, "the rails beside {label:?}: {out:?}");
            out[0]
        }

        /// The document, what follows from it, and what is cached for
        /// the photo are all of `version`.
        fn assert_all_of(&self, version: SymbolVersion) {
            let app = &self.app;
            let (width, height) = (version.width(), version.height());
            assert_eq!(app.doc.version, version);
            assert_eq!(app.derived.func.version, version);
            assert_eq!((app.doc.grid.width(), app.doc.grid.height()), (width, height));
            assert_eq!(app.model.as_ref().map(|m| (m.width, m.height)), Some((width, height)));
            assert_eq!(app.preview.as_ref().map(|p| p.classified.modules.len()), Some(width * height));
        }
    }

    #[test]
    fn a_version_adopted_from_a_job_is_the_one_drawn_and_sampled() {
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        // A larger symbol than the grid in use, taken in while the Align
        // stage shows. Sampling then read the old version's preview by
        // the new version's indices, and ran off its end.
        let found = detection(SymbolVersion::Qr(6), STRAIGHT);
        bench.app.start_job(&ctx, JobKind::Adopt, move || JobResult::Adopt(prepared(found), String::new(), None));
        bench.settle();
        bench.assert_all_of(SymbolVersion::Qr(6));
        bench.act(Action::Sample { only_unknown: false, scope: SampleScope::All });
        bench.step();
        bench.assert_all_of(SymbolVersion::Qr(6));
        // A smaller one, taken in while the Edit stage shows the larger
        // grid, which drew the new grid by the old version's map.
        bench.app.view.stage = Stage::Edit;
        bench.step();
        let found = detection(SymbolVersion::Micro(2), STRAIGHT);
        bench.app.start_job(&ctx, JobKind::Adopt, move || JobResult::Adopt(prepared(found), String::new(), None));
        bench.settle();
        bench.assert_all_of(SymbolVersion::Micro(2));
    }

    #[test]
    fn an_action_in_the_frame_of_a_version_change_works_on_the_new_version() {
        let mut bench = Bench::new();
        // No frame comes between the two: the function map in hand when
        // the second is applied was made for the version before.
        bench.act(Action::SetVersion(SymbolVersion::Qr(5)));
        bench.act(Action::ClearData);
        assert_eq!((bench.app.doc.grid.width(), bench.app.doc.grid.height()), (37, 37));
        bench.act(Action::SetVersion(SymbolVersion::Micro(1)));
        bench.act(Action::FillFixed);
        assert_eq!((bench.app.doc.grid.width(), bench.app.doc.grid.height()), (11, 11));
        bench.act(Action::Undo);
        bench.act(Action::Sample { only_unknown: false, scope: SampleScope::All });
        bench.step();
        bench.assert_all_of(SymbolVersion::Qr(5));
    }

    #[test]
    fn sampling_that_turns_the_photo_s_inversion_round_derives_the_grid_as_sampled() {
        let mut bench = Bench::new();
        bench.app.doc.corners = STRAIGHT;
        bench.step();
        // Invert photo ticked wrongly for a dark-on-light symbol: Sample
        // all turns it off first, and samples. Turning it off refreshed
        // what follows from the document before the modules were
        // written, and the decode shown stayed that of the blank grid
        // until the next edit.
        assert!(bench.app.view.auto_invert);
        bench.act(Action::SetAdjust(Adjust { invert: true, ..bench.app.doc.adjust }));
        bench.step();
        assert!(bench.app.derived.report.codewords.iter().all(|c| c.known == 0));
        let unknown_before = bench.app.doc.grid.count(Module::Unknown);
        bench.act(Action::Sample { only_unknown: false, scope: SampleScope::All });
        assert!(!bench.app.doc.adjust.invert, "the inversion was turned off first");
        assert_eq!(bench.app.doc.grid.count(Module::Unknown), 0);
        assert!(bench.app.dirty, "the grid as sampled is still to be derived from");
        bench.step();
        assert!(bench.app.derived.report.codewords.iter().all(|c| c.known == 0xFF), "the decode is of the grid as sampled");
        assert!(!bench.app.dirty);
        // One undo point takes back the modules and the inversion alike.
        bench.act(Action::Undo);
        assert!(bench.app.doc.adjust.invert);
        assert_eq!(bench.app.doc.grid.count(Module::Unknown), unknown_before);
    }

    /// [`App::adjust_as_read`] of a read made through a copy of the photo
    /// as the bench has it adjusted now, but enlarged `base` times.
    fn adjust_as_read(app: &mut App, enhance: Enhance, deblurred: Option<deblur::Spread>, base: Enlarge) -> Option<String> {
        let photo = app.photo.as_ref().expect("the bench has a photo");
        let under = ReadUnder { adjust: Adjust { enlarge: base, ..photo.adjust }, measured: photo.luma.deblurred };
        app.adjust_as_read(enhance, deblurred, under)
    }

    /// The bench's render with its ink in red, so that a photo shown in
    /// its colours can be told from one shown in grey.
    fn red_render() -> image::RgbaImage {
        let (w, h, pixels) = render_rgba(&painted_grid(), 10, 4);
        let mut image = image::RgbaImage::from_raw(w as u32, h as u32, pixels).expect("a render is an image");
        for pixel in image.pixels_mut().filter(|pixel| pixel[0] < 128) {
            pixel.0 = [200, 30, 40, 255];
        }
        image
    }

    #[test]
    fn a_photo_with_transparent_parts_is_laid_on_grey_unless_another_is_chosen() {
        use qr_forensics::imaging::adjust::Backdrop;
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        // The bench's render with its ink drawn in white on transparency,
        // for a dark page: on white it is white on white.
        let (w, h, pixels) = render_rgba(&painted_grid(), 10, 4);
        let mut image = image::RgbaImage::from_raw(w as u32, h as u32, pixels).expect("a render is an image");
        for pixel in image.pixels_mut() {
            pixel.0 = if pixel[0] < 128 { [255, 255, 255, 255] } else { [0, 0, 0, 0] };
        }
        bench.app.set_photo(&ctx, "white on nothing".into(), image);
        assert_eq!(bench.app.doc.adjust.backdrop, Backdrop::Grey);
        assert!(bench.notice().text.contains("Its transparent parts are filled with grey"), "{}", bench.notice().text);
        let range = |bench: &Bench| {
            let luma = &bench.app.photo.as_ref().expect("a photo is on show").luma;
            luma.data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)))
        };
        // On grey, #808080, the ink is light on mid grey: the quiet zone
        // in the corner is the grey.
        let corner = bench.app.photo.as_ref().expect("a photo is on show").luma.data[0];
        assert!((corner - 128.0 / 255.0).abs() < 0.01 && range(&bench).1 > 0.95, "{corner}, {:?}", range(&bench));
        bench.step();
        assert!(bench.texts().iter().any(|text| text == "Transparent parts"), "{:?}", bench.texts());
        // Laid on black, it is light on dark.
        bench.act(Action::SetAdjust(Adjust { backdrop: Backdrop::Black, ..bench.app.doc.adjust }));
        let (dark, light) = range(&bench);
        assert!(dark < 0.05 && light > 0.95, "{dark} to {light}");
        // What a lens term draws in from beyond the photo's edges is the
        // backdrop too, and not white wedges at its corners.
        bench.act(Action::SetAdjust(Adjust { k1: 0.1, ..bench.app.doc.adjust }));
        let corner = bench.app.photo.as_ref().expect("a photo is on show").luma.data[0];
        assert!(corner < 0.05, "{corner}");
        bench.act(Action::SetAdjust(Adjust { k1: 0.0, ..bench.app.doc.adjust }));
        // Laid on white, there is nothing to read; undone, it is back.
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { backdrop: Backdrop::White, ..bench.app.doc.adjust }));
        assert!(range(&bench).0 > 0.95, "{:?}", range(&bench));
        bench.act(Action::Undo);
        assert_eq!(bench.app.doc.adjust.backdrop, Backdrop::Black);
        assert!(range(&bench).0 < 0.05, "{:?}", range(&bench));
        // A photo with no transparency is the same on any, and offers no
        // choice: it lies on white, which only what a lens term draws in
        // from beyond its edges shows, as it always was.
        bench.app.set_photo(&ctx, "red".into(), red_render());
        assert!(bench.app.photo.as_ref().is_some_and(|photo| photo.unflattened.is_none()));
        assert_eq!(bench.app.doc.adjust.backdrop, Backdrop::White);
        bench.step();
        assert!(!bench.texts().iter().any(|text| text == "Transparent parts"));
    }

    #[test]
    fn a_read_loaded_lays_the_photo_on_the_backdrop_it_was_read_on() {
        use qr_forensics::imaging::adjust::Backdrop;
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        // A symbol in white ink on transparency, read by Read all on grey;
        // then laid on white, where there is nothing to read, and the read
        // loaded.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/photos");
        let mut image = image::open(dir.join("zxing_qrcode-3_01.png")).expect("photo opens").to_rgba8();
        for pixel in image.pixels_mut() {
            pixel.0 = if pixel[0] < 128 { [255, 255, 255, 255] } else { [0, 0, 0, 0] };
        }
        bench.app.set_photo(&ctx, "white on nothing".into(), image);
        bench.app.view.load_reads = false;
        bench.act(Action::FindSymbols);
        bench.settle();
        bench.act(Action::ReadFound);
        bench.settle();
        assert!(bench.app.read_decoded(0), "{}", bench.notice().text);
        bench.act(Action::SetAdjust(Adjust { backdrop: Backdrop::White, ..bench.app.doc.adjust }));
        bench.app.view.load_reads = true;
        bench.act(Action::UseProposal(0));
        assert_eq!(bench.app.doc.adjust.backdrop, Backdrop::Grey);
        assert_eq!(bench.app.photo.as_ref().map(|photo| photo.adjust.backdrop), Some(Backdrop::Grey));
        assert!(bench.notice().text.contains("Under Photo, Transparent parts was set to grey"), "{}", bench.notice().text);
        // One undo puts back the white it was on, and the grid.
        bench.act(Action::Undo);
        assert_eq!(bench.app.doc.adjust.backdrop, Backdrop::White);
    }

    #[test]
    fn the_photo_and_its_binarised_view_are_laid_on_the_card_that_draws_the_window() {
        use qr_forensics::imaging::binarize::binarize;
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        let screen = screen::Screen::on_card(&ctx);
        bench.app.screen = Some(screen.clone());
        // Opened with the window drawn by the card, the photo is laid there
        // in its colours, as adjusted.
        bench.app.set_photo(&ctx, "red".into(), red_render());
        bench.app.doc.corners = square(30.0, 30.0, 270.0);
        bench.step();
        fn photo(bench: &Bench) -> &Photo {
            bench.app.photo.as_ref().expect("a photo is on show")
        }
        let colours = |photo: &Photo| adjusted(&photo.adjust, &photo.source, &photo.original, photo.crop, false, false).0.into_bytes();
        assert!(matches!(photo(&bench).texture, screen::Texture::Card(_)));
        assert_eq!(screen.read(&photo(&bench).texture), colours(photo(&bench)));
        let id = photo(&bench).texture.id();
        // In grey, the grey of what is read, laid into the same texture:
        // drawn again, and made again with the adjustments.
        bench.app.view.greyscale = true;
        bench.app.redraw_photo();
        assert_eq!((photo(&bench).texture.id(), screen.read(&photo(&bench).texture)), (id, photo(&bench).luma.to_rgba()));
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { contrast: 0.2, ..bench.app.doc.adjust }));
        assert_eq!((photo(&bench).texture.id(), screen.read(&photo(&bench).texture)), (id, photo(&bench).luma.to_rgba()));
        bench.app.view.greyscale = false;
        bench.app.redraw_photo();
        assert_eq!(screen.read(&photo(&bench).texture), colours(photo(&bench)));
        // The binarised view, laid there too, and laid into its own
        // texture again as the outline moves.
        let binarised = |photo: &Photo| {
            let view = photo.binarised.as_ref().expect("the view was made");
            let grey: Vec<u8> = binarize(&photo.luma, &view.params).data.iter().flat_map(|&v| if v < 0.5 { [0, 0, 0, 255] } else { [255; 4] }).collect();
            (view.texture.id(), screen.read(&view.texture), grey)
        };
        bench.app.view.align_overlay = AlignOverlay::Binarised;
        bench.step();
        let (view_id, shown, made) = binarised(photo(&bench));
        assert_eq!(shown, made);
        let params = photo(&bench).binarised.as_ref().expect("the view was made").params;
        bench.app.doc.corners = square(60.0, 60.0, 210.0);
        bench.step();
        assert_ne!(photo(&bench).binarised.as_ref().expect("the view was made").params, params, "made again for the outline moved");
        let (moved_id, shown, made) = binarised(photo(&bench));
        assert_eq!((moved_id, shown), (view_id, made));
        // Enlarged, the photo is laid into a texture of its new size, and
        // the old one and the old view's are let go of.
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { enlarge: Enlarge::times(2), ..bench.app.doc.adjust }));
        bench.step();
        assert_eq!(photo(&bench).texture.size(), [660, 660]);
        assert!(!screen.registered(id) && !screen.registered(view_id));
        assert_eq!(screen.read(&photo(&bench).texture), colours(photo(&bench)));
        let (_, shown, made) = binarised(photo(&bench));
        assert_eq!(shown, made);
        // Larger than a texture may be, each is egui's own, shrunk to fit,
        // and the card's are let go of; laid on the card again where it
        // fits once more.
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { enlarge: Enlarge::times(4), ..bench.app.doc.adjust }));
        bench.step();
        assert!(matches!(photo(&bench).texture, screen::Texture::Card(_)) && photo(&bench).texture.size() == [1320, 1320]);
        let ids = [photo(&bench).texture.id(), photo(&bench).binarised.as_ref().expect("the view was made").texture.id()];
        bench.texture_side = Some(1024);
        bench.step();
        assert!(matches!(photo(&bench).texture, screen::Texture::Egui(_)) && photo(&bench).texture.size() == [1024, 1024]);
        assert!(photo(&bench).binarised.as_ref().is_some_and(|view| matches!(view.texture, screen::Texture::Egui(_))));
        // Dropped in that frame, they are let go of once it is drawn.
        assert!(ids.iter().all(|&id| screen.registered(id)));
        bench.step();
        assert!(!ids.iter().any(|&id| screen.registered(id)));
        bench.texture_side = Some(8192);
        bench.step();
        assert!(matches!(photo(&bench).texture, screen::Texture::Card(_)));
        assert_eq!(screen.read(&photo(&bench).texture), colours(photo(&bench)));
        let (_, shown, made) = binarised(photo(&bench));
        assert_eq!(shown, made);
        // Undone by a key, as the end of a frame that drew the view in the
        // photo's place applies it, the photo is made again at its size
        // before and its view is made anew; the textures let go of are let
        // go of once that frame is drawn. Let go of as they were dropped,
        // it drew without them.
        let key = |modifiers| egui::Event::Key { key: egui::Key::Z, physical_key: None, pressed: true, repeat: false, modifiers };
        let (undo, redo) = (key(egui::Modifiers::COMMAND), key(egui::Modifiers::COMMAND.plus(egui::Modifiers::SHIFT)));
        let ids = [photo(&bench).texture.id(), photo(&bench).binarised.as_ref().expect("the view was made").texture.id()];
        bench.frame(vec![undo]);
        assert!(photo(&bench).texture.size() == [660, 660] && photo(&bench).binarised.is_none());
        assert!(bench.drawn_through().contains(&ids[1]), "the frame drew the view");
        assert!(ids.iter().all(|&id| screen.registered(id)), "what the frame drew through is there to be drawn");
        bench.step();
        assert!(!ids.iter().any(|&id| screen.registered(id)));
        // And the photo itself, redone by a key in a frame that drew it.
        bench.app.view.align_overlay = AlignOverlay::LinesOnly;
        bench.step();
        let id = photo(&bench).texture.id();
        bench.frame(vec![redo]);
        assert_eq!(photo(&bench).texture.size(), [1320, 1320]);
        assert!(bench.drawn_through().contains(&id) && screen.registered(id), "the frame drew the photo, which is there to be drawn");
        bench.step();
        assert!(!screen.registered(id));
    }

    #[test]
    fn enlarging_the_photo_takes_the_outline_and_what_was_found_with_it() {
        fn slowly<T>(result: T) -> T {
            std::thread::sleep(std::time::Duration::from_millis(300));
            result
        }
        let twice = |points: [[f64; 2]; 4]| points.map(|[x, y]| [2.0 * x, 2.0 * y]);
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        bench.app.doc.corners = STRAIGHT;
        bench.app.doc.set_anchor(finder_anchor([75.0, 75.0]), AnchorSource::Manual);
        bench.app.detection = Some(detection(SymbolVersion::Qr(2), STRAIGHT));
        bench.app.proposals = Some(Proposals {
            found: vec![detection(SymbolVersion::Qr(2), STRAIGHT)],
            photo_inverted: false,
            adopted: Some(0),
            reads: None,
            charged: vec![Charged { corners: STRAIGHT, version: SymbolVersion::Qr(2), among: 3 }],
        });
        bench.step();
        // The render is large enough to start at its own size, with the
        // steps its size allows on offer.
        let photo = bench.app.photo.as_ref().expect("the bench has a photo");
        assert_eq!((photo.width, photo.height, photo.adjust.enlarge), (330, 330, Enlarge::NONE));
        assert_eq!(photo.enlargements.iter().map(Enlarge::to_string).collect::<Vec<_>>(), ["0.25×", "0.5×", "1×", "2×", "4×", "8×"]);
        assert!(bench.texts().iter().any(|text| text == "1×"), "{:?}", bench.texts());
        // The view left where the user put it, to see that it follows.
        bench.app.view.align_view.auto_fit = false;
        let on_screen = bench.locate()([75.0, 75.0]);
        // Twice the size: the photo, its luminance and its texture, with
        // the pixels it had kept beside the luminance.
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { enlarge: Enlarge::times(2), ..bench.app.doc.adjust }));
        bench.step();
        let photo = bench.app.photo.as_ref().expect("the bench has a photo");
        assert_eq!((photo.width, photo.height, photo.scale()), (660, 660, 2.0));
        assert_eq!((photo.luma.width, photo.luma.height), (660, 660));
        assert_eq!(photo.luma.unenlarged.as_ref().map(|own| (own.width, own.height)), Some((330, 330)));
        assert_eq!(photo.texture.size(), [660, 660]);
        assert!(Arc::ptr_eq(&photo.source, &photo.original), "the photo as loaded is what is enlarged");
        // The outline, the anchor, and what was found went with it, and
        // the symbol reads there as it did.
        assert_eq!(bench.app.doc.corners, twice(STRAIGHT));
        assert_eq!(bench.app.doc.anchors[0].anchor.center, [150.0, 150.0]);
        assert_eq!(bench.app.detection.as_ref().map(|found| found.corners), Some(twice(STRAIGHT)));
        let proposals = bench.app.proposals.as_ref().expect("the symbols found stay on offer");
        assert_eq!(proposals.found[0].corners, twice(STRAIGHT));
        assert_eq!(proposal_in_use(proposals, &bench.app.doc.corners), Some(0));
        // And what its read in the search was charged, with it.
        assert_eq!(proposals.among_of(&proposals.found[0]), 3);
        let (matching, fixed) = bench.app.preview.as_ref().expect("the outline makes a grid").fixed;
        assert_eq!(matching, fixed);
        assert!(bench.notice().text.starts_with("Loaded render"), "nothing was under way, so nothing to say: {}", bench.notice().text);
        let texts = bench.texts();
        assert!(texts.iter().any(|text| text == "render · 660 × 660 · enlarged from 330 × 330"), "{texts:?}");
        assert!(texts.iter().any(|text| text == "2×"), "{texts:?}");
        // And the canvas shows the same part of the photo, as large.
        let moved = bench.locate()([150.0, 150.0]);
        assert!(moved.distance(on_screen) < 0.5, "{on_screen:?} to {moved:?}");
        // Undo takes the size back, and the outline and what was found
        // with it.
        bench.act(Action::Undo);
        bench.step();
        let photo = bench.app.photo.as_ref().expect("the bench has a photo");
        assert_eq!((photo.width, photo.height, photo.luma.unenlarged.is_none()), (330, 330, true));
        assert_eq!(bench.app.doc.corners, STRAIGHT);
        assert_eq!(bench.app.detection.as_ref().map(|found| found.corners), Some(STRAIGHT));
        let proposals = bench.app.proposals.as_ref().expect("the symbols found stay on offer");
        assert_eq!(proposals.among_of(&proposals.found[0]), 3);
        assert!(bench.locate()([75.0, 75.0]).distance(on_screen) < 0.5);
        // Reduced, it keeps nothing beside it, and says so.
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { enlarge: Enlarge::HALF, ..bench.app.doc.adjust }));
        bench.step();
        let photo = bench.app.photo.as_ref().expect("the bench has a photo");
        assert_eq!((photo.width, photo.height, photo.scale(), photo.luma.unenlarged.is_none()), (165, 165, 0.5, true));
        assert_eq!(bench.app.doc.corners, STRAIGHT.map(|[x, y]| [x / 2.0, y / 2.0]));
        let (matching, fixed) = bench.app.preview.as_ref().expect("the outline makes a grid").fixed;
        assert_eq!(matching, fixed, "five pixels a module still read");
        assert!(bench.texts().iter().any(|text| text == "render · 165 × 165 · reduced from 330 × 330"), "{:?}", bench.texts());
        // More than the photo's size allows is held to what it does.
        bench.act(Action::SetAdjust(Adjust { enlarge: Enlarge::times(64), ..bench.app.doc.adjust }));
        assert_eq!(bench.app.doc.adjust.enlarge, Enlarge::times(12));
        bench.act(Action::Undo);
        assert_eq!(bench.app.photo.as_ref().map(|photo| (photo.width, photo.adjust.enlarge)), Some((330, Enlarge::NONE)));
        // A search and a step under way are of the photo as it was: they
        // are abandoned, and what they hand back is not taken in. What
        // was found before stays, carried to the new size.
        let elsewhere = || detection(SymbolVersion::Qr(2), square(0.0, 0.0, 10.0));
        bench.app.start_search(&ctx, false, None, move || slowly(vec![elsewhere()]));
        bench.app.start_job(&ctx, JobKind::Adopt, move || slowly(JobResult::Adopt(prepared(elsewhere()), String::new(), None)));
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { enlarge: Enlarge::times(2), ..bench.app.doc.adjust }));
        assert!(bench.app.search.is_none() && bench.app.job.is_none());
        assert!(bench.notice().text.starts_with("The photo changed size, so the search under way was stopped and"), "{}", bench.notice().text);
        std::thread::sleep(std::time::Duration::from_millis(500));
        bench.settle();
        assert_eq!(bench.app.doc.corners, twice(STRAIGHT));
        let carried = bench.app.proposals.as_ref().map(|proposals| proposals.found[0].corners).expect("the symbols found stay on offer");
        for (corner, on_symbol) in carried.iter().zip(twice(STRAIGHT)) {
            assert!((corner[0] - on_symbol[0]).hypot(corner[1] - on_symbol[1]) < 1e-9, "{carried:?}");
        }
    }

    /// `adjust` as a photo with no transparency is loaded under it: on
    /// white, which only what a lens term draws in from beyond its edges
    /// shows ([`App::set_photo`]).
    fn opaque(adjust: Adjust) -> Adjust {
        Adjust { backdrop: Backdrop::White, ..adjust }
    }

    #[test]
    fn the_enlarge_slider_goes_from_step_to_step_of_those_on_offer() {
        let mut bench = Bench::new();
        bench.app.doc.corners = STRAIGHT;
        bench.step();
        bench.step();
        // Enlarge and Sharpen are adjustments of the photo, each one
        // slider under Photo, and Sampling has neither.
        for label in ["Enlarge", "Sharpen"] {
            bench.text_at(label);
        }
        // A press at the far end of the rail is the largest step on
        // offer, eight times the photo's size, and one near its start
        // the least, a quarter of it: one step of undo each.
        let rail = bench.rail_beside("Enlarge");
        bench.click(egui::pos2(rail.left() + 3.0, rail.center().y), egui::PointerButton::Primary);
        let photo = bench.app.photo.as_ref().expect("the bench has a photo");
        assert_eq!((bench.app.doc.adjust.enlarge, photo.width, photo.height), (Enlarge::QUARTER, 83, 83));
        assert!(bench.texts().iter().any(|text| text == "0.25×"), "{:?}", bench.texts());
        let scale = 83.0 / 330.0;
        let reduced = STRAIGHT.map(|[x, y]| [x * scale, y * scale]);
        assert_eq!(bench.app.doc.corners, reduced);
        // The fourth of the six, by where the rail has it: the handle
        // runs the rail's length less its own width, seven points in
        // from either end.
        let rail = bench.rail_beside("Enlarge");
        let fourth = rail.left() + 7.0 + (rail.width() - 14.0) * 3.0 / 5.0;
        bench.click(egui::pos2(fourth, rail.center().y), egui::PointerButton::Primary);
        assert_eq!(bench.app.doc.adjust.enlarge, Enlarge::times(2));
        assert_eq!(bench.app.photo.as_ref().map(|photo| (photo.width, photo.height)), Some((660, 660)));
        bench.act(Action::Undo);
        assert_eq!((bench.app.doc.adjust.enlarge, bench.app.doc.corners), (Enlarge::QUARTER, reduced));
        bench.act(Action::Undo);
        assert_eq!((bench.app.doc.adjust.enlarge, bench.app.doc.corners), (Enlarge::NONE, STRAIGHT));
        assert!(!bench.app.history.can_undo(&bench.app.doc));
        // Reset goes back to what the photo started with: for a small
        // one, the enlargement it was given.
        let ctx = bench.ctx.clone();
        let small = image::imageops::crop_imm(&red_render(), 0, 0, 110, 110).to_image();
        bench.app.set_photo(&ctx, "small".into(), small);
        bench.step();
        bench.step();
        let photo = bench.app.photo.as_ref().expect("the photo is loaded");
        assert_eq!(photo.enlargements.iter().map(Enlarge::to_string).collect::<Vec<_>>(), ["0.5×", "1×", "2×", "3×", "4×", "8×", "16×", "32×"]);
        assert_eq!((photo.adjust, bench.app.doc.adjust), (opaque(Adjust::for_photo(110, 110)), opaque(Adjust::for_photo(110, 110))));
        assert!(bench.texts().iter().any(|text| text == "3×"), "{:?}", bench.texts());
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { enlarge: Enlarge::NONE, sharpen: 0.4, ..bench.app.doc.adjust }));
        bench.step();
        bench.step();
        assert_eq!(bench.app.photo.as_ref().map(|photo| photo.width), Some(110));
        let reset = bench.text_at("Reset adjustments");
        bench.click(reset, egui::PointerButton::Primary);
        assert_eq!(bench.app.doc.adjust, opaque(Adjust::for_photo(110, 110)));
        assert_eq!(bench.app.photo.as_ref().map(|photo| photo.width), Some(330));
    }

    #[test]
    fn the_photo_is_sharpened_whole_and_in_its_colours() {
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        bench.app.set_photo(&ctx, "red".into(), red_render());
        bench.app.doc.corners = STRAIGHT;
        bench.app.proposals = Some(Proposals { found: vec![detection(SymbolVersion::Qr(2), STRAIGHT)], photo_inverted: false, adopted: None, reads: None, charged: Vec::new() });
        bench.step();
        // Ink in the middle of the first finder's stone, and paper in
        // the quiet zone.
        let (ink, paper) = (75 * 330 + 75, 20 * 330 + 20);
        let plain = bench.drawn();
        assert_eq!((plain[ink], plain[paper]), ([200, 30, 40], [255; 3]));
        let before = bench.app.photo.as_ref().map(|photo| photo.luma.clone()).expect("the photo is loaded");
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { sharpen: 0.5, ..bench.app.doc.adjust }));
        bench.step();
        // The photo is the size it was, and still in its colours: flat
        // ink and flat paper as they were, and the ink darker along its
        // edges, where the mask steepens them.
        let photo = bench.app.photo.as_ref().expect("the photo is loaded");
        assert_eq!((photo.width, photo.height, photo.adjust.sharpen), (330, 330, 0.5));
        let sharp = bench.drawn();
        assert_eq!((sharp[ink], sharp[paper]), ([200, 30, 40], [255; 3]));
        let edge = 75 * 330 + 60;
        assert_eq!(plain[edge], [200, 30, 40]);
        assert!(sharp[edge][0] < 190 && sharp[edge][1] < 30 && sharp[edge][0] > sharp[edge][1] + 100, "{:?}", sharp[edge]);
        // What is read is the luminance of what is shown.
        assert!(!Arc::ptr_eq(&photo.luma, &before));
        let shown = |[r, g, b]: [u8; 3]| (0.299 * f32::from(r) + 0.587 * f32::from(g) + 0.114 * f32::from(b)) / 255.0;
        assert!(sharp.iter().zip(&photo.luma.data).all(|(&pixel, &luma)| (shown(pixel) - luma).abs() < 1e-4));
        // Nothing moved: what was found stays, and the symbol reads.
        assert!(bench.app.proposals.is_some() && bench.notice().text.starts_with("Loaded red"), "{}", bench.notice().text);
        let (matching, fixed) = bench.app.preview.as_ref().expect("the outline makes a grid").fixed;
        assert_eq!(matching, fixed);
        bench.act(Action::Undo);
        bench.step();
        assert_eq!(bench.drawn(), plain);
    }

    #[test]
    fn the_blur_the_deblur_filter_undoes_is_measured_inside_the_outline_when_asked() {
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        // The red render blurred, with the outline on its symbol.
        let photo = image::imageops::blur(&red_render(), 1.6);
        bench.app.set_photo(&ctx, "blurred".into(), photo);
        bench.app.doc.corners = STRAIGHT;
        bench.step();
        bench.step();
        // With Deblur on, the panel offers the kernel and its strength,
        // and shows the blur measured over the whole photo.
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { filter: Filter::Deblur, ..bench.app.doc.adjust }));
        bench.step();
        for text in ["Kernel", "Measure in outline", "Use whole photo", "Strength"] {
            bench.text_at(text);
        }
        let photo = bench.app.photo.as_ref().expect("the photo is loaded");
        let whole = photo.luma.deblurred.expect("the whole photo's blur is measured");
        assert!(bench.texts().iter().any(|text| text.ends_with("measured over the whole photo")), "{:?}", bench.texts());
        // Measured inside the outline, the blur is kept in the photo's own
        // pixels, undone in place of the whole photo's, and said; an undo
        // takes it back.
        bench.act(Action::MeasureDeblur);
        bench.step();
        assert!(bench.notice().text.starts_with("Measured the blur inside the outline"), "{}", bench.notice().text);
        let kernel = bench.app.doc.adjust.kernel.expect("the blur is set");
        let photo = bench.app.photo.as_ref().expect("the photo is loaded");
        assert_eq!(photo.luma.deblurred, Some(kernel.scaled(photo.scale())));
        assert_ne!(Some(kernel), Some(whole));
        assert!(bench.texts().iter().any(|text| text.ends_with("set by Measure in outline or a read")), "{:?}", bench.texts());
        bench.act(Action::Undo);
        bench.step();
        assert_eq!(bench.app.doc.adjust.kernel, None);
        // An outline off the photo has nothing to measure.
        bench.app.doc.corners = [[-50.0, -50.0], [-10.0, -50.0], [-10.0, -10.0], [-50.0, -10.0]];
        bench.act(Action::MeasureDeblur);
        assert!(bench.notice().is_error && bench.app.doc.adjust.kernel.is_none(), "{}", bench.notice().text);
        assert!(bench.notice().text.starts_with("The outline is off the photo"), "{}", bench.notice().text);
        // Nor has an outline drawn to a line on it, which is not said to be
        // off the photo.
        bench.app.doc.corners = [[60.0, 60.0], [200.0, 60.0], [200.0, 60.0], [60.0, 60.0]];
        bench.act(Action::MeasureDeblur);
        assert!(bench.notice().is_error && bench.app.doc.adjust.kernel.is_none(), "{}", bench.notice().text);
        assert!(bench.notice().text.starts_with("The outline has no area"), "{}", bench.notice().text);
        // A read that decoded only through a copy deblurred at full
        // strength sets the filter's strength back to that, and has it
        // undo what the copy undid, measured about the symbol, in place
        // of the blur the whole photo shows.
        bench.act(Action::SetAdjust(Adjust { strength: 0.4, ..bench.app.doc.adjust }));
        let undid = deblur::Spread { xx: 2.0, xy: 0.3, yy: 1.2 };
        let deblurred = Enhance { enlarge: 1, sharpen: Sharpen::Deblur };
        let text = adjust_as_read(&mut bench.app, deblurred, Some(undid), Enlarge::NONE).expect("the strength is set");
        assert!(text.contains("Filter was set to Deblur at full strength, undoing the blur measured around the symbol"), "{text}");
        let adjust = bench.app.doc.adjust;
        assert_eq!((adjust.filter, adjust.strength, adjust.kernel), (Filter::Deblur, 1.0, Some(undid)));
        // Read in the photo enlarged, the blur is carried back to the
        // photo's own pixels.
        bench.act(Action::SetAdjust(Adjust { enlarge: Enlarge::times(2), ..bench.app.doc.adjust }));
        let text = adjust_as_read(&mut bench.app, deblurred, Some(undid), Enlarge::times(2)).expect("the kernel is set");
        assert!(text.contains("Filter was set to Deblur, undoing the blur measured around the symbol"), "{text}");
        assert_eq!(bench.app.doc.adjust.kernel, Some(undid.scaled(0.5)));
    }

    #[test]
    fn a_read_is_set_against_the_photo_as_it_was_adjusted_when_it_was_made() {
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        bench.app.set_photo(&ctx, "blurred".into(), image::imageops::blur(&red_render(), 1.6));
        bench.app.doc.corners = STRAIGHT;
        bench.step();
        // A step under way while the user changes the photo's filter: what
        // it read was read off the photo filtered as it was, and its
        // result is dropped, the version and the outline as they were.
        // Taken in, a read that decoded only through a copy of the photo
        // made under the Deblur filter set the photo up as though the copy
        // had been made with none.
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { filter: Filter::Deblur, ..bench.app.doc.adjust }));
        let version = bench.app.doc.version;
        let found = detection(SymbolVersion::Qr(6), STRAIGHT);
        bench.app.start_job(&ctx, JobKind::Adopt, move || JobResult::Adopt(prepared(found), String::new(), None));
        bench.act(Action::SetAdjust(Adjust { filter: Filter::None, ..bench.app.doc.adjust }));
        bench.settle();
        let notice = bench.notice();
        assert!(notice.text.starts_with("Placing the corners on the symbol found was dropped: the photo's adjustments changed"), "{}", notice.text);
        assert_eq!((bench.app.doc.version, bench.app.doc.adjust.filter), (version, Filter::None));
        // A read Read all made under a median filter, loaded after the user
        // took the filter off: its copy was made of the photo filtered, and
        // the filter is set back, and said, with Sharpen as the copy had it.
        let under = ReadUnder { adjust: Adjust { filter: Filter::Median, radius: 2, ..bench.app.doc.adjust }, measured: None };
        let text = bench.app.adjust_as_read(Enhance { enlarge: 1, sharpen: Sharpen::Light }, None, under).expect("the photo is set");
        assert!(text.contains("Sharpen was set to 0.50, and Filter to Median 2 px either side, as it was when the symbol was read"), "{text}");
        assert_eq!((bench.app.doc.adjust.filter, bench.app.doc.adjust.radius, bench.app.doc.adjust.sharpen), (Filter::Median, 2, 0.5));
        // And one read under the Deblur filter with no blur set for it,
        // through a copy enlarged twice: the filter is set back, held to
        // the blur it measured over the whole photo then, carried to the
        // photo's own pixels, whatever the photo shows now.
        let measured = deblur::Spread::round(1.4);
        let under = ReadUnder { adjust: Adjust { filter: Filter::Deblur, sharpen: 0.0, ..Adjust::default() }, measured: Some(measured) };
        let text = bench.app.adjust_as_read(Enhance { enlarge: 2, sharpen: Sharpen::Off }, None, under).expect("the photo is set");
        assert!(text.contains("Enlarge was set to 2× and Sharpen to 0.00, and Filter to Deblur, undoing the blur it had measured"), "{text}");
        let adjust = bench.app.doc.adjust;
        assert_eq!((adjust.enlarge, adjust.filter, adjust.kernel), (Enlarge::times(2), Filter::Deblur, Some(measured)));
    }

    #[test]
    fn anchors_on_no_site_are_counted_apart_from_the_sites_set() {
        // A finder pattern found, two patches of data a read placed, a point
        // traced along an edge and a custom anchor set by hand: one site
        // set, and the rest on none. The patches and the edge point were
        // once counted among the sites set, which a read's hundred patches
        // took far past the sites there are.
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        bench.app.set_photo(&ctx, "red".into(), red_render());
        bench.app.doc.corners = STRAIGHT;
        bench.app.doc.set_anchor(finder_anchor([75.0, 75.0]), AnchorSource::Detected);
        bench.app.doc.set_anchor(anchor(AnchorKind::Data, [12.5, 12.5], [165.0, 165.0]), AnchorSource::Detected);
        bench.app.doc.set_anchor(anchor(AnchorKind::Data, [16.5, 12.5], [205.0, 165.0]), AnchorSource::Detected);
        bench.app.doc.set_anchor(anchor(AnchorKind::Edge, [12.0, 0.0], [160.0, 40.0]), AnchorSource::Detected);
        bench.app.doc.set_anchor(anchor(AnchorKind::Custom, [14.5, 18.5], [185.0, 225.0]), AnchorSource::Manual);
        bench.step();
        bench.step();
        let sites = anchor_sites(&bench.app.derived.func).len();
        let want = format!("Anchors: 1 of {sites} sites set (1 found, 0 by hand), and 1 edge point, 2 data patches and 1 custom");
        assert!(bench.texts().contains(&want), "{:?}", bench.texts().iter().filter(|text| text.starts_with("Anchors")).collect::<Vec<_>>());
    }

    #[test]
    fn a_read_through_a_copy_enlarged_and_sharpened_fixes_the_deblur_filter_to_the_blur_it_undid() {
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        // The red render blurred, under the Deblur filter with no blur set
        // for it: it undoes what it measures over the whole photo.
        bench.app.set_photo(&ctx, "blurred".into(), image::imageops::blur(&red_render(), 1.6));
        bench.app.doc.corners = STRAIGHT;
        bench.step();
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { filter: Filter::Deblur, ..bench.app.doc.adjust }));
        bench.step();
        let photo = bench.app.photo.as_ref().expect("the photo is loaded");
        let (before, base) = (photo.luma.clone(), photo.adjust.enlarge);
        let measured = before.deblurred.expect("the whole photo's blur is measured");
        // A read that decoded only through a copy enlarged and sharpened,
        // made under the filter, which undid the blur measured at the
        // photo's size and not the one the photo enlarged and sharpened
        // shows: the photo so adjusted holds the filter to that blur, and
        // shows the copy again, pixel for pixel. Measured again, it once
        // came out narrower, and the photo differed from the copy by most
        // of the contrast along its edges.
        let enhance = Enhance { enlarge: 2, sharpen: Sharpen::Strong };
        let copy = Enhanced::window(&before, [0, 0, before.width, before.height], enhance).expect("a copy");
        let text = adjust_as_read(&mut bench.app, enhance, None, base).expect("the photo is enlarged and sharpened");
        assert!(text.contains("and Deblur kept at the blur it had measured over the whole photo"), "{text}");
        let adjust = bench.app.doc.adjust;
        assert_eq!((adjust.filter, adjust.kernel), (Filter::Deblur, Some(measured.scaled(1.0 / base.factor()))));
        let photo = bench.app.photo.as_ref().expect("the photo is loaded");
        assert_eq!((photo.luma.width, photo.luma.height), (copy.luma.width, copy.luma.height));
        assert!(photo.luma.data == copy.luma.data, "the photo shows the copy again");
        // Where the filter measured no blur, the copy was left unfiltered,
        // and so is the photo.
        let mut bench = Bench::new();
        bench.app.set_photo(&ctx, "flat".into(), image::RgbaImage::from_pixel(330, 330, image::Rgba([128, 128, 128, 255])));
        bench.step();
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { filter: Filter::Deblur, ..bench.app.doc.adjust }));
        bench.step();
        let photo = bench.app.photo.as_ref().expect("the photo is loaded");
        assert_eq!(photo.luma.deblurred, None);
        let base = photo.adjust.enlarge;
        let text = adjust_as_read(&mut bench.app, Enhance { enlarge: 2, sharpen: Sharpen::Light }, None, base).expect("the photo is enlarged");
        assert!(text.contains("and Filter to None, as Deblur found no blur to undo"), "{text}");
        assert_eq!((bench.app.doc.adjust.filter, bench.app.doc.adjust.kernel), (Filter::None, None));
    }

    #[test]
    fn the_sharpen_slider_blurs_the_photo_below_nought() {
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        bench.app.set_photo(&ctx, "red".into(), red_render());
        bench.app.doc.corners = STRAIGHT;
        bench.step();
        bench.step();
        // Ink in the middle of the first finder's stone, paper in the
        // quiet zone, and the first pixel of ink along the stone's left
        // edge, with the ring's light module beside it.
        let (ink, paper, edge) = (75 * 330 + 75, 20 * 330 + 20, 75 * 330 + 60);
        let plain = bench.drawn();
        assert_eq!((plain[ink], plain[paper], plain[edge]), ([200, 30, 40], [255; 3], [200, 30, 40]));
        // A press at the start of the rail is the furthest below nought
        // the slider goes: one step of undo.
        let rail = bench.rail_beside("Sharpen");
        bench.click(egui::pos2(rail.left() + 3.0, rail.center().y), egui::PointerButton::Primary);
        let photo = bench.app.photo.as_ref().expect("the photo is loaded");
        assert_eq!((bench.app.doc.adjust.sharpen, photo.adjust.sharpen, photo.width, photo.height), (-2.0, -2.0, 330, 330));
        // The photo is the size it was, and still in its colours: flat
        // ink and flat paper as they were, and the edge between them run
        // together, part ink and part paper.
        let blurred = bench.drawn();
        assert_eq!((blurred[ink], blurred[paper]), ([200, 30, 40], [255; 3]));
        assert!(blurred[edge][0] > 210 && blurred[edge][1] > 100 && blurred[edge][1] < 200, "{:?}", blurred[edge]);
        assert!(blurred[edge][0] > blurred[edge][1] + 50, "still red: {:?}", blurred[edge]);
        // What is read is the luminance of what is shown.
        let shown = |[r, g, b]: [u8; 3]| (0.299 * f32::from(r) + 0.587 * f32::from(g) + 0.114 * f32::from(b)) / 255.0;
        assert!(blurred.iter().zip(&photo.luma.data).all(|(&pixel, &luma)| (shown(pixel) - luma).abs() < 1e-4));
        // Nothing moved, and the symbol still reads through the blur.
        let (matching, fixed) = bench.app.preview.as_ref().expect("the outline makes a grid").fixed;
        assert_eq!(matching, fixed);
        bench.act(Action::Undo);
        bench.step();
        assert_eq!((bench.app.doc.adjust.sharpen, bench.drawn()), (0.0, plain));
        assert!(!bench.app.history.can_undo(&bench.app.doc));
    }

    #[test]
    fn a_blurred_photo_stays_blurred_through_a_read_that_only_enlarged_its_copy() {
        let mut bench = Bench::new();
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { sharpen: -0.5, ..bench.app.doc.adjust }));
        // The copy was made of the photo blurred, and enlarged: the
        // photo is enlarged with the blur it had, and nothing is said of
        // Sharpen.
        let text = adjust_as_read(&mut bench.app, Enhance { enlarge: 2, sharpen: Sharpen::Off }, None, Enlarge::NONE).expect("the photo is enlarged");
        assert!(text.contains("so under Photo Enlarge was set to 2×, and Sample reads it that way.") && !text.contains("Sharpen"), "{text}");
        assert_eq!(bench.app.doc.adjust, Adjust { enlarge: Enlarge::times(2), sharpen: -0.5, ..opaque(Adjust::default()) });
        assert_eq!(bench.app.photo.as_ref().map(|photo| (photo.width, photo.adjust.sharpen)), Some((660, -0.5)));
        // A copy sharpened decoded where the photo blurred did not: the
        // sharpening takes the blur's place.
        let text = adjust_as_read(&mut bench.app, Enhance { enlarge: 1, sharpen: Sharpen::Light }, None, Enlarge::times(2)).expect("the photo is sharpened");
        assert!(text.contains("so under Photo Sharpen was set to 0.50, and Sample reads it that way."), "{text}");
        assert_eq!(bench.app.doc.adjust, Adjust { enlarge: Enlarge::times(2), sharpen: 0.5, ..opaque(Adjust::default()) });
        // Set so, a read through the same copy has nothing to add.
        assert_eq!(adjust_as_read(&mut bench.app, Enhance { enlarge: 1, sharpen: Sharpen::Light }, None, Enlarge::times(2)), None);
        // Nor is a sharpening the photo had kept past one a read made in
        // its place, weaker though that is: the copy was made of the
        // photo without it.
        bench.act(Action::SetAdjust(Adjust { sharpen: 1.5, ..bench.app.doc.adjust }));
        let text = adjust_as_read(&mut bench.app, Enhance { enlarge: 1, sharpen: Sharpen::Light }, None, Enlarge::times(2)).expect("the photo is sharpened");
        assert!(text.contains("so under Photo Sharpen was set to 0.50, and Sample reads it that way."), "{text}");
        assert_eq!(bench.app.doc.adjust.sharpen, 0.5);
    }

    #[test]
    fn the_photo_is_shown_in_grey_when_asked_and_read_the_same() {
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        bench.app.set_photo(&ctx, "red".into(), red_render());
        bench.app.doc.corners = STRAIGHT;
        bench.step();
        bench.step();
        let (ink, paper) = (75 * 330 + 75, 20 * 330 + 20);
        // In its colours to begin with.
        assert!(!bench.app.view.greyscale);
        assert_eq!((bench.drawn()[ink], bench.drawn()[paper]), ([200, 30, 40], [255; 3]));
        let read = bench.app.photo.as_ref().map(|photo| photo.luma.clone()).expect("the photo is loaded");
        let grey = |luma: &Luma| luma.data.iter().map(|&v| [(v * 255.0).round() as u8; 3]).collect::<Vec<_>>();
        // The checkbox under Photo shows it as the luminance that is
        // read, which is the same luminance as before.
        let checkbox = bench.text_at("Show in greyscale");
        bench.click(checkbox, egui::PointerButton::Primary);
        assert!(bench.app.view.greyscale);
        let photo = bench.app.photo.as_ref().expect("the photo is loaded");
        assert!(photo.greyscale && Arc::ptr_eq(&photo.luma, &read), "only what is shown changed");
        assert_eq!(bench.drawn(), grey(&read));
        assert_eq!(bench.drawn()[ink], [82; 3]);
        assert!(!bench.app.history.can_undo(&bench.app.doc), "a way of looking, not an edit");
        // It stays in grey through a change of the adjustments, an undo
        // of it, and another photo.
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { enlarge: Enlarge::times(2), sharpen: 0.5, ..bench.app.doc.adjust }));
        bench.step();
        let photo = bench.app.photo.as_ref().expect("the photo is loaded");
        assert_eq!((photo.texture.size(), bench.drawn()), ([660, 660], grey(&photo.luma)));
        bench.act(Action::Undo);
        bench.step();
        assert_eq!(bench.drawn(), grey(&read));
        bench.app.set_photo(&ctx, "red again".into(), red_render());
        bench.step();
        assert_eq!(bench.drawn(), grey(&read));
        // Unticked, the colours are back, and what is read is as it was.
        bench.step();
        bench.click(bench.text_at("Show in greyscale"), egui::PointerButton::Primary);
        assert!(!bench.app.view.greyscale);
        assert_eq!((bench.drawn()[ink], bench.drawn()[paper]), ([200, 30, 40], [255; 3]));
        assert_eq!(bench.app.photo.as_ref().map(|photo| photo.luma.data.clone()), Some(read.data.clone()));
    }

    #[test]
    fn only_a_read_that_decoded_sets_the_sampling() {
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        let before = bench.app.view.sample;
        assert_ne!(before.kernel, Kernel::Centre);
        assert_ne!(before.rule(), Rule::Surface);
        let read = |trustworthy: bool, enhance: Enhance| {
            let (kernel, rule, verified) = (Kernel::Centre, Rule::Surface, usize::from(trustworthy));
            let note = ReadNote {
                kernel,
                rule,
                enhance,
                deblurred: None,
                sharpen_modules: true,
                unsure_kept: false,
                verified,
                blocks: 1,
                trustworthy,
                padding_pinned: false,
                reinterpreted: None,
            };
            Prepared { read: Some(note), ..prepared(detection(SymbolVersion::Qr(2), STRAIGHT)) }
        };
        let copy = Enhance { enlarge: 2, sharpen: Sharpen::Light };
        // The best of the attempts at a read that decoded nothing says
        // nothing about how this photo wants sampling, or adjusting.
        let text = bench.app.adopt_prepared(&ctx, read(false, copy));
        assert!(bench.app.view.sample == before);
        assert_eq!(bench.app.doc.adjust, opaque(Adjust::default()));
        assert!(!text.contains("decoded it"), "{text}");
        let text = bench.app.adopt_prepared(&ctx, read(true, Enhance::NONE));
        // The rule goes with the kernel, to the checkbox that asks for it.
        let sample = bench.app.view.sample;
        assert_eq!((sample.kernel, sample.rule(), sample.local, sample.surface), (Kernel::Centre, Rule::Surface, false, true));
        assert!(text.contains("sampling kernel decoded it"), "{text}");
        assert!(text.contains("a surface across the symbol decoded it"), "{text}");
        assert_eq!(bench.app.doc.adjust, opaque(Adjust::default()), "read off the photo as it is");
        assert!(!text.contains("only with the photo"), "{text}");
        // A read made through a copy of the photo about the symbol,
        // enlarged and sharpened, has the whole photo so adjusted: the
        // outline and the detection go with it to its new size, and
        // Sampling reads it there as ever, through no copy of its own.
        let text = bench.app.adopt_prepared(&ctx, read(true, copy));
        assert!(text.contains("It decoded only with the photo enlarged 2 times and sharpened lightly, so under Photo Enlarge was set to 2× and Sharpen to 0.50"), "{text}");
        assert_eq!(bench.app.doc.adjust, Adjust { enlarge: Enlarge::times(2), sharpen: 0.5, ..opaque(Adjust::default()) });
        let photo = bench.app.photo.as_ref().expect("the bench has a photo");
        assert_eq!((photo.width, photo.height, photo.adjust), (660, 660, bench.app.doc.adjust));
        let doubled = STRAIGHT.map(|[x, y]| [2.0 * x, 2.0 * y]);
        assert_eq!(bench.app.doc.corners, doubled);
        assert_eq!(bench.app.detection.as_ref().map(|found| found.corners), Some(doubled));
        assert_eq!(bench.app.view.sample.enhance, Enhance::NONE);
        bench.step();
        let (matching, fixed) = bench.app.preview.as_ref().expect("the outline makes a grid").fixed;
        assert_eq!(matching, fixed);
        // One undo takes the adoption back, the photo's size with it.
        bench.act(Action::Undo);
        assert_eq!((bench.app.doc.adjust, bench.app.doc.corners), (opaque(Adjust::default()), STRAIGHT));
        assert_eq!(bench.app.photo.as_ref().map(|photo| photo.width), Some(330));
        // A copy enlarged by a number of times that is no step of the
        // slider's is offered as one from then on.
        let text = bench.app.adopt_prepared(&ctx, read(true, Enhance { enlarge: 3, sharpen: Sharpen::Off }));
        assert!(text.contains("only with the photo enlarged 3 times, so under Photo Enlarge was set to 3×, and Sample reads it that way."), "{text}");
        let photo = bench.app.photo.as_ref().expect("the bench has a photo");
        assert_eq!((photo.width, photo.adjust.sharpen), (990, 0.0));
        assert_eq!(photo.enlargements.iter().map(Enlarge::to_string).collect::<Vec<_>>(), ["0.25×", "0.5×", "1×", "2×", "3×", "4×", "8×"]);
        bench.step();
        bench.step();
        assert!(bench.texts().iter().any(|text| text == "3×"), "{:?}", bench.texts());
        // So adjusted already, a read through the same copy of the photo
        // as it was has nothing to add.
        assert_eq!(adjust_as_read(&mut bench.app, Enhance { enlarge: 3, sharpen: Sharpen::Off }, None, Enlarge::NONE), None);
        // A photo too large to be enlarged so far without its being
        // asked for is sharpened, and says what to do about the rest:
        // the slider offers it.
        bench.app.set_photo(&ctx, "wide".into(), image::RgbaImage::from_pixel(1500, 300, image::Rgba([255; 4])));
        let text = bench.app.adopt_prepared(&ctx, read(true, Enhance { enlarge: 4, sharpen: Sharpen::Strong }));
        assert!(text.contains("so under Photo Sharpen was set to 1.00, and Sample reads it that way. A photo this large is not enlarged to 4× unasked"), "{text}");
        assert_eq!(bench.app.doc.adjust, Adjust { sharpen: 1.0, ..opaque(Adjust::default()) });
        assert_eq!(bench.app.doc.corners, STRAIGHT);
        let photo = bench.app.photo.as_ref().expect("the photo is loaded");
        assert_eq!(photo.enlargements.iter().map(Enlarge::to_string).collect::<Vec<_>>(), ["0.25×", "0.5×", "1×", "2×", "4×"]);
        // Set so by hand, the photo is as the read had it, and the same
        // read has nothing more to ask.
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { enlarge: Enlarge::times(4), ..bench.app.doc.adjust }));
        assert_eq!(bench.app.photo.as_ref().map(|photo| (photo.width, photo.height)), Some((6000, 1200)));
        assert_eq!(bench.app.doc.corners, STRAIGHT.map(|[x, y]| [4.0 * x, 4.0 * y]));
        assert_eq!(adjust_as_read(&mut bench.app, Enhance { enlarge: 4, sharpen: Sharpen::Strong }, None, Enlarge::NONE), None);
    }

    #[test]
    fn a_symbol_read_only_through_a_sharpened_copy_samples_the_same_once_the_photo_is_so_adjusted() {
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        // A version 3 symbol three pixels a module in a photo out of
        // focus by eight tenths of a module: grey where a dark module
        // has light neighbours, and its finders' rings run into their
        // stones. (Out of focus by seven tenths it read through the
        // module centres once reliability was calibrated. By eight it
        // read through the copy only with its padding pinned, until the
        // blur learnt from its fixed patterns was undone. Blurred by 2.35
        // pixels, its finders read whole against the dark of their rings
        // in a copy sharpened only lightly, and the read through that copy
        // wants its padding pinned. Blurred by 2.4 the copy sharpened
        // strongly read it outright, until a solution that spends the
        // codeword a 3-L block keeps for detecting misreads was confirmed
        // no longer; it then wanted its padding pinned too, as at 2.25
        // and 2.5, and at 2.3 the photo as it was read it. Blurred by
        // 2.45 the copy sharpened strongly reads it outright.)
        let message = "https://example.com/out-of-focus";
        let (w, h, pixels) = render_rgba(&symbol_of(message), 3, 4);
        let render = image::RgbaImage::from_raw(w as u32, h as u32, pixels).expect("a render is an image");
        let mut photo = image::RgbaImage::from_pixel(400, 360, image::Rgba([255; 4]));
        image::imageops::replace(&mut photo, &render, 150, 120);
        bench.app.set_photo(&ctx, "blurred".into(), image::imageops::blur(&photo, 2.45));
        bench.app.doc.corners = square(145.0, 115.0, 121.0);
        // Find in outline reads it through a copy of the photo sharpened
        // and enlarged, and the photo is adjusted to match. The read is
        // not loaded, so that the grid is left for Sample to read.
        bench.app.view.load_reads = false;
        bench.act(Action::FindCorners);
        bench.settle();
        let notice = bench.notice().text.clone();
        assert!(notice.contains("the message decodes.") && notice.contains("It decoded only with the photo"), "{notice}");
        let adjust = bench.app.doc.adjust;
        assert!(adjust.sharpen >= 0.5 && adjust.enlarge > Enlarge::NONE, "{adjust:?}");
        let photo = bench.app.photo.as_ref().expect("the photo is loaded");
        assert_eq!((photo.width, photo.height), adjust.enlarge.size(400, 360));
        assert_eq!(bench.app.doc.version, SymbolVersion::Qr(3));
        // Sample then reads off the photo so adjusted what the read read
        // through its copy: the message.
        bench.act(Action::Sample { only_unknown: false, scope: SampleScope::All });
        bench.step();
        assert!(bench.app.derived.report.is_trustworthy(), "{}: {:?}", bench.notice().text, bench.app.derived.report.text.plain());
        assert_eq!(bench.app.derived.report.text.plain(), message);
        // The same outline on the photo as it was, neither enlarged nor
        // sharpened, does not come to the message that way.
        let (outline, enlarge) = (bench.app.doc.corners, adjust.enlarge.factor());
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust::default()));
        assert_eq!(bench.app.doc.corners.map(|c| c.map(f64::round)), outline.map(|c| c.map(|v| (v / enlarge).round())));
        bench.act(Action::ClearAllPins);
        bench.act(Action::Sample { only_unknown: false, scope: SampleScope::All });
        bench.step();
        bench.act(Action::ApplyPadding);
        bench.step();
        assert!(!bench.app.derived.report.is_trustworthy());
    }

    #[test]
    fn dragging_a_traced_edge_point_moves_that_point_alone() {
        let mut bench = Bench::new();
        bench.app.doc.corners = STRAIGHT;
        // Three points traced along the top edge, as the detection of a
        // bowed symbol leaves them: they stand on no site of the version,
        // and are known among the handles only by where they are.
        for x in [6.3, 12.1, 18.4] {
            let anchor = anchor(AnchorKind::Edge, [x, 0.0], [40.0 + 10.0 * x, 36.0]);
            bench.app.doc.anchors.push(DocAnchor { anchor, source: AnchorSource::Detected, bled: 0.0 });
        }
        bench.step();
        bench.step();
        let handles = bench.handles(4.0);
        assert_eq!(handles.len(), 3, "{handles:?}");
        let mut pointer = handles.into_iter().min_by(|a, b| a.x.total_cmp(&b.x)).expect("three handles");
        bench.frame(vec![egui::Event::PointerMoved(pointer)]);
        bench.step();
        bench.button(pointer, egui::PointerButton::Primary, true);
        for _ in 0..6 {
            pointer.y -= 10.0;
            bench.frame(vec![egui::Event::PointerMoved(pointer)]);
        }
        bench.button(pointer, egui::PointerButton::Primary, false);
        bench.step();
        let heights: Vec<f64> = bench.app.doc.anchors.iter().map(|a| a.anchor.center[1]).collect();
        let dragged = 40.0 / f64::from(bench.app.view.align_view.scale);
        assert!(heights[0] < 36.0 - dragged, "the point under the pointer went with it: {heights:?}");
        assert_eq!(heights[1..], [36.0, 36.0], "and the others stayed");
        assert_eq!(bench.app.doc.anchors.iter().map(|a| a.anchor.at[0]).collect::<Vec<_>>(), [6.3, 12.1, 18.4]);
    }

    #[test]
    fn a_point_of_the_photo_is_found_on_the_grid_drawn_there() {
        let mut bench = Bench::new();
        // An outline out of square, and an anchor that bends the grid in
        // it: the grid is asked where it was drawn.
        bench.app.doc.corners = [[40.0, 40.0], [290.0, 50.0], [280.0, 290.0], [45.0, 285.0]];
        bench.app.doc.set_anchor(anchor(AnchorKind::Custom, [12.5, 12.5], [175.0, 163.0]), AnchorSource::Manual);
        bench.step();
        let model = bench.app.model.as_ref().expect("the outline is usable");
        let apart = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
        assert!(apart(model.projection.apply(12.5, 12.5), [175.0, 163.0]) < 2.0, "the grid is not bent through the anchor");
        for at in [[0.5, 0.5], [12.25, 3.75], [12.5, 12.5], [7.0, 19.0], [24.9, 24.9]] {
            let px = model.projection.apply(at[0], at[1]);
            let found = model.module_at(px).expect("the point is on the grid");
            assert!(apart(found, at) < 0.02, "{at:?} was found at {found:?}");
        }
        // Off the grid there is nothing to find.
        assert_eq!(model.module_at([10.0, 10.0]), None);
        assert_eq!(model.module_at([300.0, 160.0]), None);
    }

    #[test]
    fn the_grid_is_taken_hold_of_where_it_has_no_pattern() {
        let mut bench = Bench::new();
        bench.app.doc.corners = STRAIGHT;
        bench.step();
        // The view a little closer than it fits the photo at, so that a
        // module is wider than the handle on it.
        bench.app.view.align_view.auto_fit = false;
        bench.app.view.align_view.scale = 2.5;
        bench.step();
        let locate = bench.locate();
        let scale = f64::from(bench.app.view.align_view.scale);
        let apart = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
        // A drag of thirty points across and eighteen up from a place of
        // the photo: how far that is in the photo.
        let dragged = [30.0 / scale, -18.0 / scale];
        let drag = |bench: &mut Bench, from: [f64; 2]| {
            let mut pointer = locate(from);
            bench.frame(vec![egui::Event::PointerMoved(pointer)]);
            bench.step();
            bench.button(pointer, egui::PointerButton::Primary, true);
            for _ in 0..6 {
                pointer += egui::vec2(5.0, -3.0);
                bench.frame(vec![egui::Event::PointerMoved(pointer)]);
            }
            bench.button(pointer, egui::PointerButton::Primary, false);
            bench.step();
        };
        // The pointer taken off the grid, since a handle under it is
        // drawn larger.
        let aside = |bench: &mut Bench| {
            bench.frame(vec![egui::Event::PointerMoved(locate([140.0, 34.0]))]);
            bench.step();
        };
        // In the symbol's data, clear of every handle: the grid lies ten
        // pixels a module from (40, 40), so this is a point of the
        // module twelve across and fourteen down, towards its corner.
        // The anchor is set on the middle of that module, and goes as
        // far from where the grid had the middle as the pointer went.
        let (middle, start) = ([165.0, 185.0], [163.0, 187.0]);
        drag(&mut bench, start);
        let anchors = bench.app.doc.anchors.clone();
        assert_eq!(anchors.len(), 1, "{anchors:?}");
        let custom = anchors[0];
        assert_eq!((custom.anchor.kind, custom.anchor.at, custom.source), (AnchorKind::Custom, [12.5, 14.5], AnchorSource::Manual));
        let end = [middle[0] + dragged[0], middle[1] + dragged[1]];
        assert!(apart(custom.anchor.center, end) < 1e-3, "it went to {:?}, not {end:?}", custom.anchor.center);
        assert_eq!(bench.app.view.grasped, None, "the drag is over");
        // The grid goes with it there, most of the way: the map may
        // miss an anchor a little to save bending.
        let model = bench.app.model.as_ref().expect("the outline is usable");
        let bent = model.projection.apply(12.5, 14.5);
        assert!(apart(bent, end) < 0.3 * apart(middle, end), "the grid went from {middle:?} to {bent:?}, the anchor to {end:?}");
        // It has a handle of its own, drawn where it is, by which it is
        // moved again: the same anchor, and no second one beside it.
        aside(&mut bench);
        let handles = bench.handles(4.0);
        assert_eq!(handles.len(), 1, "{handles:?}");
        assert!((handles[0] - locate(custom.anchor.center)).length() < 1.0);
        drag(&mut bench, custom.anchor.center);
        let anchors = bench.app.doc.anchors.clone();
        assert_eq!(anchors.len(), 1, "{anchors:?}");
        assert_eq!((anchors[0].anchor.kind, anchors[0].anchor.at), (AnchorKind::Custom, [12.5, 14.5]));
        let moved = [anchors[0].anchor.center[0] - custom.anchor.center[0], anchors[0].anchor.center[1] - custom.anchor.center[1]];
        assert!(moved[0] > dragged[0] / 2.0 && moved[1] < dragged[1] / 2.0, "the handle took it {moved:?}");
        // So it is by another drag begun on its module, clear of its
        // handle: the anchor the module has is the one moved, from
        // where it stands.
        let stood = anchors[0].anchor.center;
        bench.app.doc.set_anchor(Anchor { center: [165.0, 185.0], ..anchors[0].anchor }, AnchorSource::Manual);
        bench.step();
        assert!(4.8 * scale > 9.5, "a module is no wider than the handle on it");
        drag(&mut bench, [169.8, 189.8]);
        let anchors = bench.app.doc.anchors.clone();
        assert_eq!(anchors.len(), 1, "{anchors:?}");
        assert!(apart(anchors[0].anchor.center, end) < 1e-3, "{:?} for {end:?}", anchors[0].anchor.center);
        // Each drag is a step to be undone.
        bench.act(Action::Undo);
        assert_eq!(bench.app.doc.anchors[0].anchor.center, [165.0, 185.0]);
        bench.app.doc.set_anchor(Anchor { center: stood, ..anchors[0].anchor }, AnchorSource::Manual);
        bench.act(Action::Undo);
        assert_eq!(bench.app.doc.anchors, [custom]);
        bench.act(Action::Undo);
        assert!(bench.app.doc.anchors.is_empty());
        // A right-click on its handle drops it.
        drag(&mut bench, start);
        aside(&mut bench);
        let handles = bench.handles(4.0);
        assert_eq!(handles.len(), 1, "{handles:?}");
        bench.click(handles[0], egui::PointerButton::Secondary);
        assert!(bench.app.doc.anchors.is_empty());
        // A module a pattern is centred on is that pattern's site: taken
        // hold of clear of its handle, it is the pattern's anchor that
        // is set.
        let timing = anchor_sites(&bench.app.derived.func).into_iter().find(|site| site.kind == AnchorKind::Timing).expect("a timing site").at;
        drag(&mut bench, [40.0 + 10.0 * timing[0] + 4.8, 40.0 + 10.0 * timing[1] + 4.8]);
        let anchors = bench.app.doc.anchors.clone();
        assert_eq!(anchors.len(), 1, "{anchors:?}");
        assert_eq!((anchors[0].anchor.kind, anchors[0].anchor.at, anchors[0].source), (AnchorKind::Timing, timing, AnchorSource::Manual));
        bench.act(Action::ClearAnchors);
        // In the module at a corner the map yields to the corner, and no
        // anchor is set; nor off the grid, nor with the grid held flat.
        drag(&mut bench, [48.0, 48.0]);
        assert_eq!(bench.app.doc.corners, STRAIGHT, "the press was on the corner's handle");
        drag(&mut bench, [140.0, 34.0]);
        bench.app.view.use_anchors = false;
        drag(&mut bench, start);
        assert!(bench.app.doc.anchors.is_empty(), "{:?}", bench.app.doc.anchors);
        // With Space held the drag moves the view.
        bench.app.view.use_anchors = true;
        let before = bench.app.view.align_view.center;
        let space = |pressed| egui::Event::Key { key: egui::Key::Space, physical_key: None, pressed, repeat: false, modifiers: egui::Modifiers::NONE };
        bench.frame(vec![space(true)]);
        drag(&mut bench, start);
        bench.frame(vec![space(false)]);
        assert!(bench.app.doc.anchors.is_empty(), "{:?}", bench.app.doc.anchors);
        assert_ne!(bench.app.view.align_view.center, before);
    }

    #[test]
    fn an_edge_point_is_set_anywhere_along_the_outline() {
        let mut bench = Bench::new();
        bench.app.doc.corners = STRAIGHT;
        bench.step();
        bench.app.view.align_view.auto_fit = false;
        bench.app.view.align_view.scale = 2.5;
        bench.step();
        let locate = bench.locate();
        let scale = f64::from(bench.app.view.align_view.scale);
        let apart = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
        let dragged = [30.0 / scale, -18.0 / scale];
        let drag = |bench: &mut Bench, from: [f64; 2]| {
            let mut pointer = locate(from);
            bench.frame(vec![egui::Event::PointerMoved(pointer)]);
            bench.step();
            bench.button(pointer, egui::PointerButton::Primary, true);
            for _ in 0..6 {
                pointer += egui::vec2(5.0, -3.0);
                bench.frame(vec![egui::Event::PointerMoved(pointer)]);
            }
            bench.button(pointer, egui::PointerButton::Primary, false);
            bench.step();
        };
        // On the top edge between the handles along it, a photo pixel
        // outside it: the grid lies ten pixels a module from (40, 40), so
        // this is the outer side of the tenth border module. The edge
        // point is set on the middle of that side, and goes as far from
        // where the grid had it as the pointer went.
        let sited: Vec<[f64; 2]> = qr_forensics::imaging::detect::edge_sites(&bench.app.derived.func).into_iter().map(|s| s.at).collect();
        assert!(sited.iter().all(|at| apart(*at, [9.5, 0.0]) > 2.0), "{sited:?}");
        let (middle, start) = ([135.0, 40.0], [133.0, 39.0]);
        drag(&mut bench, start);
        let anchors = bench.app.doc.anchors.clone();
        assert_eq!(anchors.len(), 1, "{anchors:?}");
        let edge = anchors[0];
        assert_eq!((edge.anchor.kind, edge.anchor.at, edge.source), (AnchorKind::Edge, [9.5, 0.0], AnchorSource::Manual));
        let end = [middle[0] + dragged[0], middle[1] + dragged[1]];
        assert!(apart(edge.anchor.center, end) < 1e-3, "it went to {:?}, not {end:?}", edge.anchor.center);
        // The edge bows out across itself through it, and slides along
        // itself as it pleases.
        let model = bench.app.model.as_ref().expect("the outline is usable");
        let bowed = model.projection.apply(9.5, 0.0)[1];
        assert!(bowed - 40.0 < 0.5 * dragged[1], "the edge went to {bowed}, the point to {}", end[1]);
        // It has a handle of its own, by which it is dropped again.
        bench.frame(vec![egui::Event::PointerMoved(locate([140.0, 34.0]))]);
        bench.step();
        let handles = bench.handles(4.0);
        let own = handles.iter().find(|h| (**h - locate(edge.anchor.center)).length() < 1.0).copied().expect("a handle on it");
        bench.click(own, egui::PointerButton::Secondary);
        assert!(bench.app.doc.anchors.is_empty(), "{:?}", bench.app.doc.anchors);
        // Further inside than the outline's reach, the border module
        // itself is taken hold of, for a custom anchor on its middle.
        drag(&mut bench, [133.0, 44.0]);
        let anchors = bench.app.doc.anchors.clone();
        assert_eq!(anchors.len(), 1, "{anchors:?}");
        assert_eq!((anchors[0].anchor.kind, anchors[0].anchor.at), (AnchorKind::Custom, [9.5, 0.5]));
        bench.act(Action::ClearAnchors);
        // Down the left edge as well; but not beside a corner, where the
        // map yields to the corner, nor with the grid held flat.
        drag(&mut bench, [40.5, 197.0]);
        let anchors = bench.app.doc.anchors.clone();
        assert_eq!(anchors.len(), 1, "{anchors:?}");
        assert_eq!((anchors[0].anchor.kind, anchors[0].anchor.at), (AnchorKind::Edge, [0.0, 15.5]));
        bench.act(Action::ClearAnchors);
        drag(&mut bench, [282.0, 290.5]);
        assert_eq!(bench.app.doc.corners, STRAIGHT, "the press was on the corner's handle");
        assert!(bench.app.doc.anchors.is_empty(), "{:?}", bench.app.doc.anchors);
        bench.app.view.use_anchors = false;
        drag(&mut bench, start);
        assert!(bench.app.doc.anchors.is_empty(), "{:?}", bench.app.doc.anchors);
    }

    #[test]
    fn a_handle_inside_a_symbol_found_takes_its_own_clicks() {
        let mut bench = Bench::new();
        bench.app.doc.corners = STRAIGHT;
        // A symbol found that the outline happens to lie on, and has not
        // been put on from the list.
        bench.app.proposals = Some(Proposals { found: vec![detection(SymbolVersion::Qr(2), STRAIGHT)], photo_inverted: false, adopted: None, reads: None, charged: Vec::new() });
        bench.app.doc.set_anchor(finder_anchor([75.0, 75.0]), AnchorSource::Detected);
        bench.step();
        bench.step();
        let handles = bench.handles(5.5);
        assert_eq!(handles.len(), 1, "{handles:?}");
        // A click on the handle is not a click on the symbol found
        // beneath it, which would adopt it again over what was placed.
        bench.click(handles[0], egui::PointerButton::Primary);
        assert_eq!(bench.app.running(), None);
        // A right-click drops the anchor.
        bench.click(handles[0], egui::PointerButton::Secondary);
        assert!(bench.app.doc.anchors.is_empty());
        // Clear of the handles, in the middle of the symbol, ninety photo
        // pixels right of the anchor and below it, the symbol found is
        // there to be clicked.
        let middle = handles[0] + egui::vec2(90.0, 90.0) * bench.app.view.align_view.scale;
        bench.click(middle, egui::PointerButton::Primary);
        assert_eq!(bench.app.running(), Some(JobKind::Adopt));
        bench.settle();
    }

    /// An outline off in a corner of the photo, on no symbol found.
    const ASIDE: [[f64; 2]; 4] = [[4.0, 4.0], [24.0, 4.0], [24.0, 24.0], [4.0, 24.0]];

    #[test]
    fn a_symbol_found_is_clicked_inside_its_outline_and_the_better_of_two_where_they_overlap() {
        let mut bench = Bench::new();
        bench.app.doc.corners = ASIDE;
        // A square standing on its corner, whose box takes in the whole
        // symbol and whose outline leaves the symbol's corners out; and
        // after it in the list, a claim on the symbol's lower right
        // quarter, which the two share about (200, 200).
        let diamond = [[165.0, 40.0], [290.0, 165.0], [165.0, 290.0], [40.0, 165.0]];
        let found = vec![detection(SymbolVersion::Qr(2), diamond), detection(SymbolVersion::Micro(3), square(170.0, 170.0, 140.0))];
        bench.app.proposals = Some(Proposals { found, photo_inverted: false, adopted: None, reads: None, charged: Vec::new() });
        let at = bench.locate();
        // Inside the box about the first and outside its outline, there
        // is nothing to click.
        bench.click(at([60.0, 60.0]), egui::PointerButton::Primary);
        assert_eq!(bench.app.running(), None);
        // Where the two overlap, the click is the first one's, marked as
        // the one in use once the outline is put on it.
        bench.click(at([200.0, 200.0]), egui::PointerButton::Primary);
        assert_eq!(bench.app.running(), Some(JobKind::Adopt));
        assert_eq!(bench.app.proposals.as_ref().and_then(|p| p.adopted), None);
        bench.settle();
        assert_eq!(bench.app.proposals.as_ref().and_then(|p| p.adopted), Some(0));
        // Inside the second alone, it is the second's.
        bench.app.doc.corners = ASIDE;
        bench.step();
        bench.click(at([300.0, 300.0]), egui::PointerButton::Primary);
        assert_eq!(bench.app.running(), Some(JobKind::Adopt));
        bench.settle();
        assert_eq!(bench.app.proposals.as_ref().and_then(|p| p.adopted), Some(1));
    }

    #[test]
    fn the_symbol_the_outline_was_put_on_is_not_offered_again_on_the_canvas() {
        let mut bench = Bench::new();
        bench.app.doc.corners = STRAIGHT;
        bench.app.proposals = Some(Proposals { found: vec![detection(SymbolVersion::Qr(2), STRAIGHT)], photo_inverted: false, adopted: Some(0), reads: None, charged: Vec::new() });
        let at = bench.locate();
        // A click inside the symbol, clear of the handles, is not a click
        // on the symbol found: it would put the outline back as found,
        // over whatever was placed by hand since.
        bench.click(at([165.0, 165.0]), egui::PointerButton::Primary);
        assert_eq!(bench.app.running(), None);
        assert_eq!(proposal_in_use(bench.app.proposals.as_ref().expect("the list"), &bench.app.doc.corners), Some(0));
        // Once the outline is somewhere else, by hand or by an undo, the
        // symbol found is on offer again.
        bench.app.doc.corners = ASIDE;
        bench.step();
        assert_eq!(proposal_in_use(bench.app.proposals.as_ref().expect("the list"), &bench.app.doc.corners), None);
        bench.click(at([165.0, 165.0]), egui::PointerButton::Primary);
        assert_eq!(bench.app.running(), Some(JobKind::Adopt));
        bench.settle();
        // A search run again lists the symbol the outline lies on as the
        // one in use still.
        let again = vec![detection(SymbolVersion::Micro(3), square(170.0, 170.0, 140.0)), detection(SymbolVersion::Qr(2), STRAIGHT)];
        bench.app.finish_search(&bench.ctx.clone(), again, Vec::new(), false, None, false);
        assert_eq!(bench.app.proposals.as_ref().and_then(|p| p.adopted), Some(1));
    }

    #[test]
    fn a_corner_under_the_pointer_is_shown_enlarged_and_h_shows_the_bare_photo() {
        let mut bench = Bench::new();
        bench.app.doc.corners = STRAIGHT;
        bench.app.doc.set_anchor(finder_anchor([75.0, 75.0]), AnchorSource::Detected);
        let at = bench.locate();
        bench.app.doc.set_anchor(finder_anchor([75.0, 75.0]), AnchorSource::Detected);
        bench.step();
        assert_eq!(bench.handles(5.5).len(), 1);
        assert!(bench.texts().iter().any(|text| text == "TL"));
        assert!(!bench.texts().iter().any(|text| text.starts_with("TL at")));
        // The pointer on the top-left corner's handle: the photo about
        // the corner is shown enlarged, beside the handle and not under
        // the pointer.
        let corner = at(STRAIGHT[0]);
        bench.frame(vec![egui::Event::PointerMoved(corner)]);
        bench.step();
        bench.step();
        assert!(bench.texts().iter().any(|text| text.starts_with("TL at 40.0, 40.0")), "{:?}", bench.texts());
        let loupe = bench.ctx.memory(|memory| memory.area_rect(egui::Id::new("loupe"))).expect("the loupe is a window");
        assert!(!loupe.contains(corner), "{loupe:?} is under the pointer at {corner:?}");
        // With H held, the photo is shown with nothing over it: no
        // handles, no labels and no loupe.
        let key = |pressed| egui::Event::Key { key: egui::Key::H, physical_key: None, pressed, repeat: false, modifiers: egui::Modifiers::NONE };
        bench.frame(vec![key(true)]);
        bench.step();
        assert!(bench.handles(5.5).is_empty());
        assert!(!bench.texts().iter().any(|text| text.starts_with("TL")), "{:?}", bench.texts());
        bench.frame(vec![key(false)]);
        bench.step();
        assert_eq!(bench.handles(5.5).len(), 1);
    }

    #[test]
    fn the_sample_preview_shows_each_module_in_the_colour_it_would_be_painted() {
        let mut bench = Bench::new();
        bench.app.doc.corners = STRAIGHT;
        bench.app.view.show_anchors = false;
        bench.step();
        let (black, red, green) = (egui::Color32::BLACK, egui::Color32::from_rgb(235, 60, 60), egui::Color32::from_rgb(40, 190, 90));
        assert!(!bench.mesh_colours().contains(&black), "nothing black is drawn over the photo to start with");
        // Dark is a black dot and light a white one, as the grid paints
        // them: not red and green, which read as wrong and right and which
        // many cannot tell apart.
        bench.app.view.align_overlay = AlignOverlay::SamplePreview;
        bench.step();
        let colours = bench.mesh_colours();
        assert!(colours.contains(&black) && colours.contains(&egui::Color32::WHITE));
        assert!(!colours.contains(&red) && !colours.contains(&green));
    }

    #[test]
    fn a_list_takes_the_wheel_only_once_the_pointer_is_moved_onto_it() {
        let mut wheel = WheelLatch::default();
        let list = egui::Rect::from_min_size(egui::pos2(100.0, 300.0), egui::vec2(200.0, 100.0));
        // The pointer at rest above the list.
        let rest = egui::pos2(150.0, 250.0);
        wheel.follow(Some(rest));
        wheel.place(InnerList::Codewords, list);
        wheel.follow(Some(rest));
        assert!(!wheel.owns(InnerList::Codewords));
        // The inspector scrolls, and the list slides under the pointer:
        // the wheel is not the list's, however long the pointer rests
        // there, and a jog of the mouse as its wheel turns is no move.
        let list = list.translate(egui::vec2(0.0, -80.0));
        for jog in [0.0, 0.0, 3.0] {
            wheel.place(InnerList::Codewords, list);
            wheel.follow(Some(rest + egui::vec2(jog, 0.0)));
            assert!(!wheel.owns(InnerList::Codewords));
        }
        // Moved, within the list: the wheel is the list's, and stays so
        // while the pointer rests.
        wheel.place(InnerList::Codewords, list);
        wheel.follow(Some(rest + egui::vec2(20.0, 0.0)));
        assert!(wheel.owns(InnerList::Codewords) && !wheel.owns(InnerList::Blocks));
        wheel.place(InnerList::Codewords, list);
        wheel.follow(Some(rest + egui::vec2(20.0, 0.0)));
        assert!(wheel.owns(InnerList::Codewords));
        // Moved off it, or onto where a list was before its section was
        // closed: the wheel is the inspector's.
        wheel.follow(Some(rest + egui::vec2(40.0, 0.0)));
        assert!(!wheel.owns(InnerList::Codewords));
    }

    #[test]
    fn the_wheel_scrolls_the_inspector_past_a_list_that_slides_under_the_pointer() {
        let mut bench = Bench::new();
        // A window short enough that the inspector has far to scroll.
        bench.size = egui::vec2(1440.0, 600.0);
        for _ in 0..3 {
            bench.step();
        }
        let list = |bench: &Bench| bench.app.view.wheel.placed(InnerList::Codewords).expect("the codeword table is laid out");
        // The pointer at rest a little above the codeword table, and a
        // turn of the wheel that brings the table up under it.
        let before = list(&bench);
        let pointer = egui::pos2(before.center().x, before.top() - 14.0);
        bench.frame(vec![egui::Event::PointerMoved(pointer)]);
        bench.step();
        bench.wheel(-40.0);
        let under = list(&bench);
        assert!(under.top() < before.top() - 20.0, "the inspector scrolled: {before:?} to {under:?}");
        assert!(under.contains(pointer), "and the table is under the pointer now: {under:?}");
        // The next turn is the inspector's still, not the table's.
        bench.wheel(-40.0);
        let past = list(&bench);
        assert!(past.top() < under.top() - 20.0, "the inspector scrolled on: {under:?} to {past:?}");
        // Moved onto the table, the pointer gives it the wheel: the table
        // scrolls, and the inspector stays where it is.
        let pointer = egui::pos2(past.center().x + 20.0, past.center().y);
        bench.frame(vec![egui::Event::PointerMoved(pointer)]);
        bench.step();
        assert!(bench.app.view.wheel.owns(InnerList::Codewords));
        bench.wheel(-40.0);
        assert_eq!(list(&bench).top(), past.top());
    }

    #[test]
    fn a_lens_term_between_the_sliders_steps_is_left_as_it_is() {
        let mut bench = Bench::new();
        // What Estimate sets: a term to the thousandth, where the slider
        // steps by five of them. The corners and anchors have been moved
        // for exactly this term.
        bench.app.doc.adjust.k1 = -0.062;
        let _ = bench.app.sync_adjust();
        for _ in 0..3 {
            bench.step();
        }
        assert_eq!(bench.app.doc.adjust.k1, -0.062);
        assert_eq!(bench.app.photo.as_ref().map(|p| p.adjust.k1), Some(-0.062));
        assert!(!bench.app.history.can_undo(&bench.app.doc), "nothing changed, and nothing was recorded");
    }

    #[test]
    fn a_new_photo_starts_without_the_last_photos_outline_and_anchors() {
        let mut bench = Bench::new();
        bench.app.checkpoint();
        bench.app.doc.corners = STRAIGHT;
        bench.app.doc.set_anchor(finder_anchor([75.0, 75.0]), AnchorSource::Detected);
        bench.app.doc.grid.set(10, 10, Module::Dark);
        let ctx = bench.ctx.clone();
        bench.app.set_photo(&ctx, "another".into(), image::RgbaImage::from_pixel(400, 300, image::Rgba([255; 4])));
        assert!(bench.app.doc.anchors.is_empty());
        let outline = bench.app.doc.corners;
        assert_eq!(Some(outline), bench.app.photo.as_ref().map(Photo::default_corners));
        // Undo takes the painted module back, and leaves the new photo
        // its own outline.
        bench.act(Action::Undo);
        assert_eq!(bench.app.doc.grid.get(10, 10), Module::Unknown);
        assert_eq!(bench.app.doc.corners, outline);
        assert!(bench.app.doc.anchors.is_empty());
        bench.settle();
    }

    #[test]
    fn a_change_of_lens_drops_what_was_found_in_the_photo_as_it_was() {
        fn slowly<T>(result: T) -> T {
            std::thread::sleep(std::time::Duration::from_millis(300));
            result
        }
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        bench.app.doc.corners = STRAIGHT;
        bench.app.proposals = Some(Proposals { found: vec![detection(SymbolVersion::Qr(2), STRAIGHT)], photo_inverted: false, adopted: None, reads: None, charged: Vec::new() });
        // A search and a step that are still under way when the lens
        // changes, and would move the outline were they taken in.
        let elsewhere = || detection(SymbolVersion::Qr(2), square(0.0, 0.0, 10.0));
        bench.app.start_search(&ctx, false, None, move || slowly(vec![elsewhere()]));
        bench.app.start_job(&ctx, JobKind::Adopt, move || slowly(JobResult::Adopt(prepared(elsewhere()), String::new(), None)));
        bench.app.history.checkpoint(&bench.app.doc);
        let adjust = Adjust { k1: -0.05, ..bench.app.doc.adjust };
        bench.act(Action::SetAdjust(adjust));
        assert!(bench.app.proposals.is_none() && bench.app.search.is_none() && bench.app.job.is_none());
        assert!(bench.notice().text.starts_with("The lens setting changed"), "{}", bench.notice().text);
        // What the abandoned threads hand back is not taken in.
        std::thread::sleep(std::time::Duration::from_millis(500));
        bench.settle();
        assert_eq!(bench.app.doc.corners, STRAIGHT);
        assert!(bench.app.proposals.is_none());
        bench.act(Action::UseProposal(0));
        assert_eq!(bench.app.running(), None);
        // Undo takes the lens back, and with it whatever was found since.
        bench.app.proposals = Some(Proposals { found: vec![detection(SymbolVersion::Qr(2), STRAIGHT)], photo_inverted: false, adopted: None, reads: None, charged: Vec::new() });
        bench.act(Action::Undo);
        assert_eq!(bench.app.doc.adjust.k1, 0.0);
        assert!(bench.app.proposals.is_none());
    }

    #[test]
    fn a_symbol_the_search_found_decoding_is_not_picked_once_the_photo_is_adjusted_meanwhile() {
        // Whether a symbol decoded, the search asked of the photo as it
        // began. Adjusted while it ran, a symbol said to decode was once
        // picked and read off the photo as it is, where it might not.
        fn slowly<T>(result: T) -> T {
            std::thread::sleep(std::time::Duration::from_millis(300));
            result
        }
        fn taken_in(bench: &mut Bench) {
            while bench.app.search.is_some() {
                bench.step();
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
        let decoding = || Detection { decodes: Some(true), ..detection(SymbolVersion::Qr(2), STRAIGHT) };
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        assert!(bench.app.view.load_reads);
        // Left as it was, the symbol is picked of itself and read.
        bench.app.start_search(&ctx, false, None, move || slowly(vec![decoding()]));
        taken_in(&mut bench);
        assert_eq!(bench.app.running(), Some(JobKind::Adopt));
        bench.settle();
        // Inverted meanwhile, the symbol reads the other way, which the
        // search's polarity accounts for: picked all the same.
        bench.app.start_search(&ctx, false, None, move || slowly(vec![decoding()]));
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { invert: true, ..bench.app.doc.adjust }));
        taken_in(&mut bench);
        assert_eq!(bench.app.running(), Some(JobKind::Adopt));
        bench.settle();
        // Its contrast changed meanwhile: offered, but not picked, and
        // not said to decode.
        bench.app.start_search(&ctx, false, None, move || slowly(vec![decoding()]));
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { contrast: bench.app.doc.adjust.contrast + 0.5, ..bench.app.doc.adjust }));
        assert!(bench.app.search.is_some(), "a change of tone leaves the search running");
        taken_in(&mut bench);
        assert_eq!(bench.app.running(), None);
        let proposals = bench.app.proposals.as_ref().expect("what was found is offered");
        assert_eq!(proposals.found.iter().map(|found| found.decodes).collect::<Vec<_>>(), [None]);
        assert!(bench.notice().text.contains("adjustments changed while the search ran"), "{}", bench.notice().text);
    }

    #[test]
    fn a_grid_sampled_a_quarter_turn_out_shows_it_and_turns_with_its_outline() {
        let mut bench = Bench::new();
        // The outline with its top-left handle on the symbol's top-right
        // corner, and the fixed patterns locked, as they are to start
        // with: painted as specified, they would hide the turn.
        let [tl, tr, br, bl] = STRAIGHT;
        bench.app.doc.corners = [tr, br, bl, tl];
        assert!(bench.app.view.lock_fixed);
        bench.step();
        // What sampling would change is counted with the fixed patterns
        // as sampled, as they are then written, and not as locked.
        let said = bench.app.sample_would_change(false).expect("a sample to go by");
        let func = bench.app.derived.func.clone();
        let sampled = bench.app.preview.as_ref().map(|preview| preview.classified.clone()).expect("a sample to go by");
        assert_ne!(said, sample_writes(&bench.app.doc.grid, &func, &sampled, false, true, None).1);
        bench.act(Action::Sample { only_unknown: false, scope: SampleScope::All });
        assert!(bench.notice().text.starts_with(&format!("Sampled the photo: {said} modules changed.")), "{}", bench.notice().text);
        bench.step();
        let turned = bench.app.derived.analysis.orientation.expect("the fixed patterns, left as sampled, show the turn");
        assert!(bench.notice().text.contains("outline is turned on the symbol"), "{}", bench.notice().text);
        // Two anchors on the grid as it lies: on the finder pattern at
        // its top left, and on a module of its timing pattern.
        let projection = bench.app.model.as_ref().expect("the outline is usable").projection.clone();
        let sites = anchor_sites(&bench.app.derived.func);
        let timing = sites.into_iter().find(|site| site.kind == AnchorKind::Timing).expect("a timing site").at;
        let finder_px = projection.apply(3.5, 3.5);
        let timing_px = projection.apply(timing[0], timing[1]);
        bench.app.doc.set_anchor(finder_anchor(finder_px), AnchorSource::Detected);
        bench.app.doc.set_anchor(anchor(AnchorKind::Timing, timing, timing_px), AnchorSource::Detected);
        // And one set by hand on a module of the data, where no pattern
        // is.
        let custom = anchor(AnchorKind::Custom, [12.5, 14.5], projection.apply(12.5, 14.5));
        bench.app.doc.set_anchor(custom, AnchorSource::Manual);
        // The transform the Paint section offers turns the grid, and the
        // outline and the anchors with it.
        bench.act(Action::Transform(Transform::Symmetry(turned.symmetry)));
        bench.step();
        assert!(bench.app.doc.grid.cells() == painted_grid().cells(), "the grid now reads as the symbol");
        assert!(bench.app.derived.analysis.orientation.is_none());
        assert_eq!(bench.app.doc.corners, STRAIGHT);
        // The finder pattern's anchor went with its pattern to the corner
        // the turn took it to, where the version has a finder pattern;
        // the timing module landed on no timing pattern, and went; the
        // custom anchor stands for the module it went to with the rest.
        let anchors: Vec<Anchor> = bench.app.doc.anchors.iter().map(|a| a.anchor).collect();
        assert_eq!(
            anchors,
            [
                Anchor { at: turned.symmetry.map_point(25, 25, [3.5, 3.5]), ..finder_anchor(finder_px) },
                Anchor { at: turned.symmetry.map_point(25, 25, [12.5, 14.5]), ..custom },
            ]
        );
        assert!(bench.notice().text.contains("1 anchor was removed"), "{}", bench.notice().text);
        // What is sampled next is the photo of these modules.
        for y in 9..15 {
            for x in 9..15 {
                bench.app.doc.grid.set(x, y, Module::Unknown);
            }
        }
        bench.act(Action::Sample { only_unknown: true, scope: SampleScope::All });
        assert!(bench.app.doc.grid.cells() == painted_grid().cells());
    }

    #[test]
    fn a_worker_that_dies_is_reported() {
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        bench.app.start_job(&ctx, JobKind::Refine, die);
        bench.settle();
        let notice = bench.notice();
        assert!(notice.is_error, "{}", notice.text);
        assert!(notice.text.starts_with("Refining the corners and anchors stopped on an internal error"), "{}", notice.text);
        bench.app.start_search(&ctx, false, None, die::<Vec<Detection>>);
        bench.settle();
        let notice = bench.notice();
        assert!(notice.is_error, "{}", notice.text);
        assert!(notice.text.starts_with("The search of the photo for symbols stopped on an internal error"), "{}", notice.text);
    }

    #[test]
    fn a_text_field_that_has_the_keyboard_keeps_the_undo_keys() {
        let mut bench = Bench::new();
        let key = |modifiers| egui::Event::Key { key: egui::Key::Z, physical_key: None, pressed: true, repeat: false, modifiers };
        let (undo, redo) = (key(egui::Modifiers::COMMAND), key(egui::Modifiers::COMMAND.plus(egui::Modifiers::SHIFT)));
        // The document's grid starts blank: an edit paints a module.
        assert_eq!(bench.app.doc.grid.get(10, 10), Module::Unknown);
        // An edit to the grid, undone and redone by the keys while
        // nothing has the keyboard.
        bench.app.checkpoint();
        bench.app.doc.grid.set(10, 10, Module::Dark);
        bench.frame(vec![undo.clone()]);
        bench.step();
        assert_eq!(bench.app.doc.grid.get(10, 10), Module::Unknown);
        bench.frame(vec![redo]);
        bench.step();
        assert_eq!(bench.app.doc.grid.get(10, 10), Module::Dark);
        // With the known-text field holding the keyboard the keys are
        // the field's, which undoes a typo with them: the grid's edit
        // stands, where it was undone under the typist before.
        bench.ctx.memory_mut(|memory| memory.request_focus(inspector::known_text_id()));
        bench.step();
        assert!(bench.ctx.egui_wants_keyboard_input(), "the field has the keyboard");
        bench.frame(vec![undo.clone()]);
        bench.step();
        assert!(bench.ctx.egui_wants_keyboard_input(), "and keeps it");
        assert_eq!(bench.app.doc.grid.get(10, 10), Module::Dark, "the edit stands");
        // The field given up, the keys are the document's again.
        bench.ctx.memory_mut(|memory| memory.surrender_focus(inspector::known_text_id()));
        bench.step();
        assert!(!bench.ctx.egui_wants_keyboard_input());
        bench.frame(vec![undo]);
        bench.step();
        assert_eq!(bench.app.doc.grid.get(10, 10), Module::Unknown);
    }

    #[test]
    fn a_refine_result_is_dropped_when_the_outline_moved_meanwhile() {
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        bench.app.doc.corners = STRAIGHT;
        bench.app.doc.set_anchor(finder_anchor([75.0, 75.0]), AnchorSource::Manual);
        let anchors: Vec<Anchor> = bench.app.doc.anchors.iter().map(|a| a.anchor).collect();
        let job = |corners: [[f64; 2]; 4], anchors: Vec<Anchor>| move || JobResult::Refine { corners, anchors, refined: None };
        // Of the outline as it stands, the result is taken in: here,
        // that the corners are unusable.
        bench.app.start_job(&ctx, JobKind::Refine, job(STRAIGHT, anchors.clone()));
        bench.settle();
        let notice = bench.notice();
        assert!(notice.is_error && notice.text.starts_with("The corners do not form"), "{}", notice.text);
        // A corner dragged while the step ran: the result is of the
        // outline before the drag, and is dropped, with the drag kept.
        bench.app.start_job(&ctx, JobKind::Refine, job(STRAIGHT, anchors.clone()));
        bench.app.doc.corners[2][0] += 3.0;
        bench.settle();
        let notice = bench.notice();
        assert!(!notice.is_error && notice.text.contains("changed while they were being refined"), "{}", notice.text);
        assert_eq!(bench.app.doc.corners[2][0], STRAIGHT[2][0] + 3.0);
        // So is the result of a step whose anchors were dropped meanwhile.
        let corners = bench.app.doc.corners;
        bench.app.start_job(&ctx, JobKind::Refine, job(corners, anchors));
        bench.app.doc.anchors.clear();
        bench.settle();
        assert!(bench.notice().text.contains("changed while they were being refined"), "{}", bench.notice().text);
        assert!(bench.app.doc.anchors.is_empty(), "the anchor dropped stays dropped");
    }

    #[test]
    fn a_symbol_found_inside_an_outline_moved_meanwhile_is_dropped() {
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        bench.app.doc.corners = STRAIGHT;
        // Of the outline as it stands, the result is taken in: here,
        // that nothing was found inside it.
        bench.app.start_job(&ctx, JobKind::FindCorners, || JobResult::Corners(None));
        bench.settle();
        let notice = bench.notice();
        assert!(notice.is_error && notice.text.starts_with("No finder pattern"), "{}", notice.text);
        // A corner dragged while the step ran: the result is of the
        // outline before the drag, and is dropped, with the drag kept.
        bench.app.start_job(&ctx, JobKind::FindCorners, || JobResult::Corners(None));
        bench.app.doc.corners[1][1] += 2.0;
        bench.settle();
        let notice = bench.notice();
        assert!(!notice.is_error && notice.text.contains("changed while the symbol inside the outline was being found"), "{}", notice.text);
        assert_eq!(bench.app.doc.corners[1][1], STRAIGHT[1][1] + 2.0);
        // So is a symbol the search found being placed, when an anchor
        // was dropped meanwhile: the version and the outline stay, and
        // the symbol is not marked as the one in use, which it is not.
        // Marked as the step began, it stayed marked with the result
        // dropped, listed as in use and kept off the canvas, where the
        // notice said to click it again.
        let (version, corners) = (bench.app.doc.version, bench.app.doc.corners);
        bench.app.doc.set_anchor(finder_anchor([75.0, 75.0]), AnchorSource::Manual);
        let found = detection(SymbolVersion::Qr(6), STRAIGHT);
        bench.app.proposals = Some(Proposals { found: vec![found.clone()], photo_inverted: false, adopted: None, reads: None, charged: Vec::new() });
        bench.app.start_job(&ctx, JobKind::Adopt, move || JobResult::Adopt(prepared(found), String::new(), Some(Picked::Proposal(0))));
        bench.app.doc.anchors.clear();
        bench.settle();
        assert!(bench.notice().text.contains("changed while the corners were being placed"), "{}", bench.notice().text);
        assert_eq!((bench.app.doc.version, bench.app.doc.corners), (version, corners));
        assert!(bench.app.doc.anchors.is_empty(), "the anchor dropped stays dropped");
        assert_eq!(bench.app.proposals.as_ref().and_then(|p| p.adopted), None);
        // Left alone, the symbol found is placed, and marked as in use.
        let found = detection(SymbolVersion::Qr(6), STRAIGHT);
        bench.app.start_job(&ctx, JobKind::Adopt, move || JobResult::Adopt(prepared(found), String::new(), Some(Picked::Proposal(0))));
        bench.settle();
        assert_eq!(bench.app.doc.version, SymbolVersion::Qr(6));
        assert_eq!(bench.app.proposals.as_ref().and_then(|p| p.adopted), Some(0));
    }

    #[test]
    fn a_ranking_made_inside_an_outline_moved_meanwhile_is_dropped() {
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        bench.app.doc.corners = STRAIGHT;
        let guess = || VersionGuess { version: SymbolVersion::Qr(2), matching: 90, fixed: 100, inverted: false };
        // Of the outline as it stands, the versions ranked are listed.
        bench.app.start_job(&ctx, JobKind::Suggest, move || JobResult::Suggest(vec![guess()]));
        bench.settle();
        assert_eq!(bench.app.guesses.as_ref().map(Vec::len), Some(1));
        bench.app.guesses = None;
        // A corner dragged while they were being ranked: the ranking was
        // of the outline before the drag, and would be shown as made on
        // these corners. It is dropped, with the drag kept.
        bench.app.start_job(&ctx, JobKind::Suggest, move || JobResult::Suggest(vec![guess()]));
        bench.app.doc.corners[1][1] += 2.0;
        bench.settle();
        assert!(bench.app.guesses.is_none());
        let notice = bench.notice();
        assert!(!notice.is_error && notice.text.contains("changed while the versions were being ranked"), "{}", notice.text);
        assert_eq!(bench.app.doc.corners[1][1], STRAIGHT[1][1] + 2.0);
        // So are the placements inside the outline, when an anchor was
        // dropped meanwhile.
        bench.app.doc.set_anchor(finder_anchor([75.0, 75.0]), AnchorSource::Manual);
        let found = detection(SymbolVersion::Qr(2), STRAIGHT);
        bench.app.start_job(&ctx, JobKind::Placements, move || JobResult::Placements { found: vec![found], photo_inverted: false });
        bench.app.doc.anchors.clear();
        bench.settle();
        assert!(bench.app.placements.is_none());
        assert!(bench.notice().text.contains("changed while the placements inside the outline were being ranked"), "{}", bench.notice().text);
        assert!(bench.app.doc.anchors.is_empty(), "the anchor dropped stays dropped");
        // Left alone, they are listed.
        let found = detection(SymbolVersion::Qr(2), STRAIGHT);
        bench.app.start_job(&ctx, JobKind::Placements, move || JobResult::Placements { found: vec![found], photo_inverted: false });
        bench.settle();
        assert_eq!(bench.app.placements.as_ref().map(|placements| placements.found.len()), Some(1));
    }

    #[test]
    fn a_pin_taken_into_the_document_from_padding_recovery_can_be_unpinned() {
        let mut bench = Bench::new();
        bench.act(Action::ClosePhoto);
        // Two characters in a version 3 symbol, and sixteen codewords of
        // the padding unknown, one more than the block's error correction
        // restores; the rest of the padding shows the pattern.
        let mut grid = symbol_of("hi");
        let layout = SymbolLayout::new(Arc::new(FunctionMap::new(SymbolVersion::Qr(3))), EcLevel::L).expect("a level of the version");
        for &cw in &layout.data_stream[20..36] {
            for pos in layout.codeword_modules(cw) {
                grid.set(pos.x as usize, pos.y as usize, Module::Unknown);
            }
        }
        bench.hold(grid);
        // The decoder pins the padding itself, and the block solves on
        // those pins, which are the decoder's: in the report and not in
        // the document, so there is nothing to unpin by hand.
        assert!(bench.app.opts.derive_pins);
        assert_eq!(bench.app.derived.report.text.plain(), "hi");
        assert!(bench.app.doc.pins.is_empty());
        let pad = layout.data_stream[20];
        assert!(bench.app.derived.report.codewords[pad].pin.is_some_and(|pin| pin.source == PinSource::Padding));
        bench.act(Action::Unpin(pad));
        assert!(bench.notice().is_error && bench.notice().text.starts_with("That pin is the decoder's"), "{}", bench.notice().text);
        assert!(bench.app.doc.pins.is_empty());
        // Pin the padding takes them into the document with the source
        // each came with, and each is then the user's to take away by
        // the codeword, as a pin typed in is.
        bench.act(Action::ApplyPadding);
        assert!(!bench.notice().is_error, "{}", bench.notice().text);
        let pinned = bench.app.doc.pins.len();
        assert!(bench.app.doc.pins.get(&pad).is_some_and(|pin| pin.source == PinSource::Padding));
        bench.act(Action::Unpin(pad));
        assert_eq!(bench.app.doc.pins.len(), pinned - 1);
        assert!(!bench.app.doc.pins.contains_key(&pad));
        // The inspector offers Unpin for a pin in the document, and for
        // one of the decoder's own says whose it is.
        bench.size = egui::vec2(1440.0, 3000.0);
        bench.app.view.selected = Some(layout.data_stream[21]);
        bench.step();
        bench.step();
        let texts = bench.texts();
        assert!(texts.iter().any(|text| text == "Unpin"), "{texts:?}");
        bench.act(Action::ClearAllPins);
        bench.step();
        bench.step();
        let texts = bench.texts();
        assert!(texts.iter().any(|text| text == "pinned by padding recovery"), "{texts:?}");
        assert!(!texts.iter().any(|text| text == "Unpin"), "{texts:?}");
    }

    #[test]
    fn the_versions_suggested_go_with_their_family() {
        let mut bench = Bench::new();
        let guess = |version| VersionGuess { version, matching: 90, fixed: 100, inverted: false };
        bench.app.guesses = Some(vec![guess(SymbolVersion::Qr(3)), guess(SymbolVersion::Qr(2))]);
        // Another version of the family keeps the list; another family
        // drops it, since a click on one of its versions would switch
        // the family back.
        bench.act(Action::SetVersion(SymbolVersion::Qr(3)));
        assert!(bench.app.guesses.is_some());
        bench.act(Action::SetVersion(SymbolVersion::Micro(2)));
        assert!(bench.app.guesses.is_none());
    }

    #[test]
    fn the_versions_are_ranked_again_on_their_own_thread_when_the_family_changed_during_the_step() {
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        let guess = |version| VersionGuess { version, matching: 90, fixed: 100, inverted: false };
        // A symbol found at another version of the family, fitting too
        // poorly to be adopted outright: the version is suggested, from
        // the ranking the step made while it ran, when that was made for
        // the family in use.
        let found = || Detection { agreement: (80, 100), ..detection(SymbolVersion::Qr(6), STRAIGHT) };
        let ranked = Prepared { guesses: Some((Family::Qr, vec![guess(SymbolVersion::Qr(6))])), ..prepared(found()) };
        let text = bench.app.adopt_prepared(&ctx, ranked);
        assert!(text.contains("Looks like version 6 rather than 2: pick it under Suggest version."), "{text}");
        assert_eq!((bench.app.doc.version, bench.app.doc.corners), (SymbolVersion::Qr(2), STRAIGHT));
        assert_eq!(bench.app.guesses.as_ref().map(|guesses| guesses[0].version), Some(SymbolVersion::Qr(6)));
        assert!(bench.app.running().is_none(), "nothing more to rank");
        bench.app.guesses = None;
        // Made for another family, as it was when the family changed
        // during the step, the ranking is made again: on a thread of its
        // own, as Suggest version makes one, not in the frame.
        let elsewhere = Prepared { guesses: Some((Family::Micro, vec![guess(SymbolVersion::Micro(3))])), ..prepared(found()) };
        let text = bench.app.adopt_prepared(&ctx, elsewhere);
        assert!(text.contains("Looks like version 6 rather than 2: the versions are being ranked on these corners"), "{text}");
        assert_eq!(bench.app.running(), Some(JobKind::Suggest));
        assert!(bench.app.guesses.is_none(), "not until the step is done");
        bench.settle();
        let guesses = bench.app.guesses.as_ref().expect("the ranking was taken in");
        assert!(!guesses.is_empty() && guesses.iter().all(|guess| guess.version.family() == Family::Qr), "{guesses:?}");
        assert_eq!((bench.app.doc.version, bench.app.doc.corners), (SymbolVersion::Qr(2), STRAIGHT));
    }

    #[test]
    fn a_new_blank_grid_keeps_the_outline_and_the_anchors() {
        let mut bench = Bench::new();
        bench.app.doc.corners = STRAIGHT;
        bench.app.doc.set_anchor(finder_anchor([75.0, 75.0]), AnchorSource::Manual);
        bench.act(Action::Sample { only_unknown: false, scope: SampleScope::All });
        bench.act(Action::Pin(0, 0x40));
        let before = bench.app.doc.clone();
        assert_eq!(before.grid.count(Module::Unknown), 0);
        assert!(!before.pins.is_empty());
        // The grid starts afresh at the same version: the pins and the
        // format go with the modules, and the outline and the anchors
        // stay, since the fixed patterns they pin to the photo are where
        // they were.
        bench.act(Action::NewBlank);
        assert_eq!(bench.app.doc.grid, bench.app.derived.func.blank_grid());
        assert!(bench.app.doc.pins.is_empty() && bench.app.doc.pins_ec.is_none());
        assert_eq!((bench.app.doc.corners, &bench.app.doc.anchors), (before.corners, &before.anchors));
        bench.act(Action::Undo);
        assert!(bench.app.doc == before);
        // Another version is another grid, whose modules the anchors do
        // not name: they go, and the notice says so.
        bench.act(Action::SetVersion(SymbolVersion::Qr(3)));
        assert!(bench.app.doc.anchors.is_empty());
        assert_eq!(bench.app.doc.corners, before.corners);
        let notice = bench.notice();
        assert!(!notice.is_error && notice.text.contains("1 anchor was removed: they named modules of version 2"), "{}", notice.text);
        bench.act(Action::Undo);
        assert!(bench.app.doc == before);
    }

    #[test]
    fn a_custom_anchor_turned_onto_a_pattern_becomes_that_patterns_anchor() {
        let mut bench = Bench::new();
        bench.app.doc.corners = STRAIGHT;
        // An anchor set by hand on the corner module of the top-left
        // finder pattern, which is no site: a half turn takes it onto the
        // alignment pattern, which is one, and would otherwise leave the
        // module with two handles, the pattern's and the anchor's own.
        let custom = anchor(AnchorKind::Custom, [6.5, 6.5], [105.0, 105.0]);
        bench.app.doc.set_anchor(custom, AnchorSource::Manual);
        bench.act(Action::Transform(Transform::Symmetry(Symmetry::Rotate180)));
        bench.step();
        let anchors: Vec<Anchor> = bench.app.doc.anchors.iter().map(|a| a.anchor).collect();
        assert_eq!(anchors, [Anchor { kind: AnchorKind::Alignment, at: [18.5, 18.5], ..custom }]);
        assert_eq!(bench.app.doc.anchors[0].source, AnchorSource::Manual);
        assert!(bench.app.message.as_ref().is_none_or(|notice| !notice.text.contains("removed")), "{:?}", bench.app.message.as_ref().map(|n| &n.text));
        // The pattern's anchor now, it goes as any pattern's anchor goes
        // when a turn leaves no pattern of its kind under it.
        bench.act(Action::Transform(Transform::Symmetry(Symmetry::Rotate180)));
        bench.step();
        assert!(bench.app.doc.anchors.is_empty());
        assert!(bench.notice().text.contains("1 anchor was removed"), "{}", bench.notice().text);
    }

    #[test]
    fn a_photo_is_searched_for_symbols_only_when_asked() {
        let mut bench = Bench::new();
        // Loading the photo searched nothing: the notice says it is
        // loaded, and how to go on.
        assert!(bench.app.search.is_none() && bench.app.proposals.is_none());
        let notice = bench.notice();
        assert!(!notice.is_error && notice.text.starts_with("Loaded render"), "{}", notice.text);
        assert!(notice.text.contains("Search whole photo"), "{}", notice.text);
        // Search whole photo is what searches it.
        bench.act(Action::FindSymbols);
        assert!(bench.app.search.is_some());
        bench.settle();
        let proposals = bench.app.proposals.as_ref().expect("the symbol is found");
        assert_eq!(proposals.found.first().map(|found| found.version), Some(SymbolVersion::Qr(2)));
    }

    #[test]
    fn a_paste_is_asked_for_once_whether_or_not_its_press_brought_text() {
        let mut held = false;
        // A screenshot on the clipboard: the press brings nothing, and
        // the release asks.
        assert!(paste_asked(false, true, false, &mut held));
        // Text on the clipboard: the press brings a paste, which asks,
        // and the release that follows does not ask again.
        assert!(paste_asked(true, false, false, &mut held));
        assert!(!paste_asked(false, true, false, &mut held));
        assert!(paste_asked(false, true, false, &mut held), "the next press is its own");
        // Press and release in one frame are one request.
        assert!(paste_asked(true, true, false, &mut held));
        assert!(!held);
        // A paste into a text field is the field's, whatever it brought.
        assert!(!paste_asked(true, false, true, &mut held));
        assert!(!paste_asked(false, true, true, &mut held));
        assert!(!paste_asked(false, true, true, &mut held));
    }

    #[test]
    fn a_search_or_a_step_under_way_is_called_off() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        // Work that goes on until it is called off, and counts itself
        // out when it stops.
        let stopped = Arc::new(AtomicUsize::new(0));
        let until_called_off = || {
            let stopped = stopped.clone();
            move || {
                let started = std::time::Instant::now();
                while !work::cancelled() && started.elapsed().as_secs() < 60 {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
                stopped.fetch_add(1, Ordering::SeqCst);
            }
        };
        let wait_for = |count: usize| {
            let started = std::time::Instant::now();
            while stopped.load(Ordering::SeqCst) < count {
                assert!(started.elapsed().as_secs() < 20, "the work was not called off");
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        };
        let (searching, stepping) = (until_called_off(), until_called_off());
        bench.app.start_search(&ctx, false, None, move || {
            searching();
            vec![detection(SymbolVersion::Qr(2), square(0.0, 0.0, 10.0))]
        });
        bench.app.start_job(&ctx, JobKind::Adopt, move || {
            stepping();
            JobResult::Adopt(prepared(detection(SymbolVersion::Qr(6), STRAIGHT)), String::new(), None)
        });
        bench.step();
        // Each is on show with a way to stop it, in the status bar and
        // beside the buttons that started them.
        let texts = bench.texts();
        assert!(texts.iter().filter(|text| *text == "Stop").count() >= 3, "{texts:?}");
        assert!(texts.iter().any(|text| text == "Stop searching"));
        bench.act(Action::CancelSearch);
        assert!(bench.app.search.is_none());
        assert_eq!(bench.notice().text, "The search of the photo was called off.");
        bench.act(Action::CancelJob);
        assert!(bench.app.job.is_none());
        assert!(bench.notice().text.starts_with("Placing the corners on the symbol found was called off"), "{}", bench.notice().text);
        // Their threads stop, and what they hand back is not taken in.
        wait_for(2);
        bench.settle();
        assert!(bench.app.proposals.is_none());
        assert_eq!(bench.app.doc.version, SymbolVersion::Qr(2));
        // With nothing under way there is nothing to call off, and no
        // notice of it.
        bench.app.message = None;
        bench.act(Action::CancelSearch);
        bench.act(Action::CancelJob);
        assert!(bench.app.message.is_none());
        // A search and a step whose photo goes are called off with it.
        let (searching, stepping) = (until_called_off(), until_called_off());
        bench.app.start_search(&ctx, false, None, move || {
            searching();
            Vec::new()
        });
        bench.app.start_job(&ctx, JobKind::Refine, move || {
            stepping();
            JobResult::Refine { corners: [[0.0; 2]; 4], anchors: Vec::new(), refined: None }
        });
        bench.act(Action::ClosePhoto);
        wait_for(4);
    }

    #[test]
    fn a_search_looks_for_the_family_in_use_alone_when_asked() {
        let mut bench = Bench::new();
        // Every family, as a search goes unless told otherwise.
        bench.act(Action::FindSymbols);
        assert_eq!(bench.app.search.as_ref().map(|search| search.family), Some(None));
        bench.step();
        assert!(bench.texts().iter().any(|text| text.starts_with("Searching the photo for symbols of every family")));
        bench.act(Action::CancelSearch);
        // The family chosen under Symbol alone.
        bench.app.view.own_family_only = true;
        bench.act(Action::SetVersion(SymbolVersion::Micro(2)));
        bench.act(Action::FindSymbols);
        assert_eq!(bench.app.search.as_ref().map(|search| search.family), Some(Some(Family::Micro)));
        bench.step();
        let texts = bench.texts();
        assert!(texts.iter().any(|text| text.starts_with("Searching the photo for Micro QR symbols")), "{texts:?}");
        assert!(texts.iter().any(|text| text == "Only look for Micro QR"));
        bench.act(Action::CancelSearch);
        // Finding none of it says which family, and how to look wider.
        bench.app.finish_search(&bench.ctx.clone(), Vec::new(), Vec::new(), false, Some(Family::Micro), false);
        let notice = bench.notice();
        assert!(notice.is_error && notice.text.starts_with("No Micro QR symbol was found"), "{}", notice.text);
        assert!(notice.text.contains("untick Only look for Micro QR"), "{}", notice.text);
        // Finding none, and finder patterns no symbol could be fitted
        // to, says so and lists them; picking one puts the outline about
        // it, to be undone like any move of the corners.
        let outline = [[30.0, 40.0], [130.0, 42.0], [128.0, 141.0], [29.0, 139.0]];
        bench.app.leads = vec![Lead { family: Family::Qr, outline, finders: vec![[50.0, 60.0], [110.0, 61.0]], fitted: None }];
        bench.app.finish_search(&bench.ctx.clone(), Vec::new(), Vec::new(), false, None, false);
        let notice = bench.notice();
        assert!(notice.is_error && notice.text.starts_with("No symbol was found in the photo, but finder patterns"), "{}", notice.text);
        bench.step();
        assert!(bench.texts().iter().any(|text| text == "Finder patterns with no symbol fitted (1):"), "{:?}", bench.texts());
        let before = bench.app.doc.corners;
        assert_ne!(before, outline);
        bench.act(Action::UseLead(0));
        assert_eq!(bench.app.doc.corners, outline);
        assert!(bench.notice().text.starts_with("The outline is about 2 finder patterns"), "{}", bench.notice().text);
        bench.act(Action::Undo);
        assert_eq!(bench.app.doc.corners, before);
        bench.act(Action::UseLead(3));
        assert!(bench.notice().is_error, "{}", bench.notice().text);
        // A placement fitted and not read is put down at its own corners
        // and version, undone in one step with the move of the corners.
        let version = bench.app.doc.version;
        let placed = [[31.0, 41.0], [129.0, 43.0], [127.0, 140.0], [30.0, 138.0]];
        let fitted = SymbolVersion::Qr(if version == SymbolVersion::Qr(3) { 4 } else { 3 });
        bench.app.leads = vec![Lead { family: Family::Qr, outline: placed, finders: vec![[50.0, 60.0], [110.0, 61.0], [50.0, 120.0]], fitted: Some(fitted) }];
        bench.act(Action::UseLead(0));
        assert_eq!((bench.app.doc.corners, bench.app.doc.version), (placed, fitted));
        assert!(bench.notice().text.starts_with(&format!("The outline is where the search fitted a QR Code {fitted} symbol")), "{}", bench.notice().text);
        // The grid it starts afresh is said, as a proposal taken says it.
        assert!(bench.notice().text.ends_with(&format!("This started a new QR Code {fitted} grid (undoable).")), "{}", bench.notice().text);
        bench.act(Action::Undo);
        assert_eq!((bench.app.doc.corners, bench.app.doc.version), (before, version));
        // At the version the grid has already, it stays, and nothing of a
        // new grid is said. The grid's version is of the family it was
        // last set to, and the lead's is the same: a lead of one family
        // fitted at a version of another was once put to it here, and
        // said as a QR Code M2 symbol.
        let finders = vec![[50.0, 60.0], [110.0, 61.0], [50.0, 120.0]];
        bench.app.leads = vec![Lead { family: version.family(), outline: placed, finders, fitted: Some(version) }];
        bench.act(Action::UseLead(0));
        assert_eq!((bench.app.doc.corners, bench.app.doc.version), (placed, version));
        let said = format!("The outline is where the search fitted a {} {version} symbol", version.family().label());
        assert!(bench.notice().text.starts_with(&said), "{}", bench.notice().text);
        assert!(!bench.notice().text.contains("new"), "{}", bench.notice().text);
        bench.act(Action::Undo);
        assert_eq!(bench.app.doc.corners, before);
        bench.act(Action::ClearProposals);
        assert!(bench.app.leads.is_empty());
        // The render's QR Code, looked for as that and nothing else.
        bench.act(Action::SetVersion(SymbolVersion::Qr(2)));
        bench.act(Action::FindSymbols);
        assert_eq!(bench.app.search.as_ref().map(|search| search.family), Some(Some(Family::Qr)));
        bench.settle();
        let found = &bench.app.proposals.as_ref().expect("the symbol is found").found;
        assert_eq!(found.iter().map(|found| found.version).collect::<Vec<_>>(), [SymbolVersion::Qr(2)]);
    }

    #[test]
    fn a_symbol_the_search_read_aside_is_read_again_charged_as_it_was() {
        // A placement nothing vouched for, read during the search as one
        // of several, is charged so ([`Proposals::charged`]); Read all
        // and a pick from the list read it again, and charge it the same,
        // where a symbol the search vouched for is a read asked for.
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/photos");
        let photo = image::open(dir.join("zxing_qrcode-3_01.png")).expect("photo opens").to_rgba8();
        bench.app.set_photo(&ctx, "photo".into(), photo);
        bench.app.view.load_reads = false;
        bench.act(Action::FindSymbols);
        bench.settle();
        let tries = |bench: &Bench| {
            let reads = bench.app.proposals.as_ref().and_then(|proposals| proposals.reads.as_ref()).expect("the symbols were read");
            reads.results[0].as_ref().expect("the symbol could be sampled").report.tries
        };
        bench.act(Action::ReadFound);
        bench.settle();
        let asked = tries(&bench);
        let proposals = bench.app.proposals.as_mut().expect("the symbol is found");
        let found = &proposals.found[0];
        proposals.charged = vec![Charged { corners: found.corners, version: found.version, among: 3 }];
        assert_eq!(proposals.among_of(&proposals.found[0]), 3);
        bench.act(Action::ReadFound);
        bench.settle();
        assert_eq!(tries(&bench), 3 * asked);
        // Picked from the list with reads loaded, the same.
        bench.app.view.load_reads = true;
        if let Some(proposals) = bench.app.proposals.as_mut() {
            proposals.reads = None;
        }
        bench.act(Action::UseProposal(0));
        bench.settle();
        assert!(bench.notice().text.starts_with("Loaded symbol 1 as it was read"), "{}", bench.notice().text);
        assert_eq!(bench.app.derived.report.tries, 3 * asked);
    }

    #[test]
    fn every_symbol_found_is_read_and_a_read_that_decodes_is_loaded_into_the_grid() {
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/photos");
        let photo = image::open(dir.join("zxing_qrcode-3_01.png")).expect("photo opens").to_rgba8();
        let message = std::fs::read_to_string(dir.join("zxing_qrcode-3_01.txt")).expect("the photo's text");
        bench.app.set_photo(&ctx, "photo".into(), photo);
        // Before a search has found anything there is nothing to read.
        bench.act(Action::ReadFound);
        assert!(bench.notice().is_error && bench.app.job.is_none(), "{}", bench.notice().text);
        // Searched with reads loaded, as they are unless that is turned
        // off, the symbol found is picked of itself, since it decodes as
        // found: read through to its message, and loaded.
        assert!(bench.app.view.load_reads, "reads are loaded unless that is turned off");
        bench.act(Action::FindSymbols);
        bench.settle();
        assert!(bench.notice().text.starts_with("Loaded symbol 1 as it was read"), "{}", bench.notice().text);
        // Loading a read leaves the stage as it is.
        assert_eq!(bench.app.view.stage, Stage::Align);
        assert!(bench.app.derived.report.is_trustworthy());
        assert_eq!(bench.app.derived.report.text.plain(), message);
        let in_use = bench.app.proposals.as_ref().and_then(|proposals| proposal_in_use(proposals, &bench.app.doc.corners));
        assert_eq!(in_use, Some(0), "the outline lies on the symbol loaded");
        bench.act(Action::Undo);
        assert_eq!(bench.app.doc.version, SymbolVersion::Qr(2));
        // With reads not loaded, the search lists what it found and no
        // more.
        bench.app.view.load_reads = false;
        bench.act(Action::FindSymbols);
        bench.settle();
        assert!(bench.notice().text.contains("press Read all"), "{}", bench.notice().text);
        assert!(bench.app.running().is_none() && !bench.app.derived.report.is_trustworthy());
        // Picked from the list with reads loaded, it is read through on
        // the way, with no Read all asked for, and loaded.
        bench.app.view.load_reads = true;
        bench.act(Action::UseProposal(0));
        assert_eq!(bench.app.running(), Some(JobKind::Adopt));
        bench.settle();
        assert!(bench.notice().text.starts_with("Loaded symbol 1 as it was read"), "{}", bench.notice().text);
        assert!(bench.app.derived.report.is_trustworthy());
        bench.act(Action::Undo);
        assert_eq!(bench.app.doc.version, SymbolVersion::Qr(2));
        // With reads not loaded, reading the symbols leaves the grid as
        // it is, and says how to have one loaded.
        bench.app.view.load_reads = false;
        bench.act(Action::ReadFound);
        assert_eq!(bench.app.running(), Some(JobKind::ReadFound));
        bench.settle();
        let reads = bench.app.proposals.as_ref().and_then(|proposals| proposals.reads.as_ref()).expect("the symbols were read");
        assert_eq!(reads.results.len(), 1);
        let read = reads.results[0].as_ref().expect("the symbol could be sampled");
        assert!(read.is_trustworthy() && !read.padding_pinned() && read.sharpen_modules);
        assert_eq!(read.text(), message);
        let version = read.version();
        assert!(bench.notice().text.starts_with("Read 1 of 1 symbols found through to a message"), "{}", bench.notice().text);
        assert!(bench.notice().text.contains("Tick Load reads that decode"), "{}", bench.notice().text);
        assert_ne!(bench.app.doc.version, version);
        assert!(!bench.app.derived.report.is_trustworthy());
        // The symbol says how its read stands in the list it is picked
        // from, and nothing lists the reads apart.
        bench.size = egui::vec2(1440.0, 3000.0);
        bench.step();
        bench.step();
        let texts = bench.texts();
        assert!(texts.iter().any(|text| text.starts_with(&format!("1. QR Code {version}")) && text.ends_with("  ·  read")), "{texts:?}");
        assert!(texts.iter().any(|text| text == "Load reads that decode"), "{texts:?}");
        assert!(!texts.iter().any(|text| text == "Symbols read"), "{texts:?}");
        // Picked so, it is adopted as any symbol found is: its outline
        // and anchors, refined on a thread of their own.
        bench.act(Action::UseProposal(0));
        assert_eq!(bench.app.running(), Some(JobKind::Adopt));
        bench.settle();
        assert!(!bench.app.derived.report.is_trustworthy());
        bench.act(Action::Undo);
        // Drawn again before the checkbox is looked for: the anchors went
        // with the undo, and the lines that count them with them.
        bench.step();
        assert_eq!(bench.app.doc.version, SymbolVersion::Qr(2));
        // With reads loaded, picking it makes the document what was
        // read: the symbol's version and modules, under the outline the
        // read left on it, and Sampling as the read had it.
        bench.click(bench.text_at("Load reads that decode"), egui::PointerButton::Primary);
        assert!(bench.app.view.load_reads);
        bench.app.view.sample.sharpen_modules = false;
        // Two modules of a finder pattern as a photo may show them, one
        // the wrong colour and one not at all.
        let held = bench.app.proposals.as_mut().and_then(|proposals| proposals.reads.as_mut()).and_then(|reads| reads.results[0].as_mut());
        let held = held.expect("the read is on hand");
        held.grid.set(0, 0, Module::Light);
        held.grid.set_sampled(3, 3, Module::Unknown, 0);
        bench.act(Action::UseProposal(0));
        assert_eq!(bench.app.running(), None);
        bench.step();
        assert_eq!(bench.app.view.stage, Stage::Align);
        bench.assert_all_of(version);
        assert!(bench.app.derived.report.is_trustworthy());
        assert_eq!(bench.app.derived.report.text.plain(), message);
        assert!(bench.notice().text.starts_with("Loaded symbol 1 as it was read"), "{}", bench.notice().text);
        // The fixed patterns are filled in as the standard has them,
        // every module of them, whatever the read had there.
        let (func, grid) = (bench.app.derived.func.clone(), &bench.app.doc.grid);
        assert_eq!((grid.get(0, 0), grid.get(3, 3), grid.reliability(3, 3)), (Module::Dark, Module::Dark, CERTAIN));
        let fixed = (0..func.height).flat_map(|y| (0..func.width).map(move |x| (x, y))).filter(|&(x, y)| func.expected(x, y).is_some()).count();
        assert_eq!(func.fixed_agreement(grid), (fixed, fixed));
        assert!(bench.app.view.sample.sharpen_modules, "the read sharpened its modules, and so does Sampling again");
        let in_use = bench.app.proposals.as_ref().and_then(|proposals| proposal_in_use(proposals, &bench.app.doc.corners));
        assert_eq!(in_use, Some(0), "the outline lies on the symbol loaded");
        // One undo takes the whole of it back.
        bench.act(Action::Undo);
        assert_eq!(bench.app.doc.version, SymbolVersion::Qr(2));
        // Read again with reads loaded, the one that decoded is loaded
        // with no picking.
        bench.act(Action::ReadFound);
        bench.settle();
        let notice = bench.notice().text.clone();
        assert!(notice.starts_with("Read 1 of 1 symbols found through to a message") && notice.contains("Loaded symbol 1 as it was read"), "{notice}");
        assert!(!notice.contains("Pick another"), "{notice}");
        bench.assert_all_of(version);
        assert!(bench.app.derived.report.is_trustworthy());
        assert_eq!(bench.app.derived.report.text.plain(), message);
        bench.act(Action::Undo);
        assert_eq!(bench.app.doc.version, SymbolVersion::Qr(2));
        // A list that has changed since the reads were asked for does
        // not take them.
        let stale = bench.app.proposals.as_ref().map(|proposals| vec![proposals.found[0].corners, STRAIGHT]).unwrap();
        bench.app.finish_read_found(&stale, vec![None, None], false);
        assert!(bench.notice().text.starts_with("The list of symbols found changed"), "{}", bench.notice().text);
        // Reads none of which decoded load nothing, and a symbol that
        // has gone from the list cannot be picked.
        let outlines = bench.app.proposals.as_ref().map(|proposals| vec![proposals.found[0].corners]).unwrap();
        bench.app.finish_read_found(&outlines, vec![None], false);
        assert_eq!(bench.notice().text, "Read 0 of 1 symbols found through to a message: the list under Alignment says how each read stands.");
        assert_eq!(bench.app.doc.version, SymbolVersion::Qr(2));
        bench.act(Action::ClearProposals);
        bench.act(Action::UseProposal(0));
        assert!(bench.notice().is_error);
    }

    /// A version 3-L QR Code that holds `text` as byte data.
    fn symbol_of(text: &str) -> ModuleGrid {
        use qrcode_generator::{Segment, qr};
        let symbol = qr::Encoder::new(qr::ErrorCorrection::Low)
            .version(qr::Version::new(3).unwrap())
            .boost_error_correction(false)
            .encode_segments(&[Segment::bytes(text)])
            .unwrap();
        ModuleGrid::from_bools(symbol.width(), symbol.height(), symbol.modules())
    }

    /// `grid` with one module wrong in each of the data codewords
    /// `wrong`, by their places in the message.
    fn with_wrong(grid: &ModuleGrid, layout: &SymbolLayout, wrong: impl IntoIterator<Item = usize>) -> ModuleGrid {
        let mut grid = grid.clone();
        for i in wrong {
            let module = layout.codeword_modules(layout.data_stream[i])[3];
            let (x, y) = (module.x as usize, module.y as usize);
            grid.set(x, y, grid.get(x, y).inverted());
        }
        grid
    }

    impl Bench {
        /// Put `grid` in the document, as a version 3 QR Code, and bring
        /// what follows from it up to date.
        fn hold(&mut self, grid: ModuleGrid) {
            self.app.doc = Document { grid, ..Document::blank(SymbolVersion::Qr(3)) };
            self.app.dirty = true;
            self.step();
            self.step();
        }

        /// Where the last frame painted `text`, and how it was laid out.
        fn text_shape(&self, text: &str) -> Option<(egui::Pos2, Arc<egui::Galley>)> {
            fn walk(shape: &egui::Shape, text: &str, out: &mut Option<(egui::Pos2, Arc<egui::Galley>)>) {
                match shape {
                    egui::Shape::Text(shape) if shape.galley.text() == text => *out = Some((shape.pos, shape.galley.clone())),
                    egui::Shape::Vec(shapes) => shapes.iter().for_each(|shape| walk(shape, text, out)),
                    _ => {}
                }
            }
            let mut out = None;
            self.shapes.iter().for_each(|clipped| walk(&clipped.shape, text, &mut out));
            out
        }

        /// How many rectangles the last frame outlined in `color`.
        fn outlined(&self, color: egui::Color32) -> usize {
            fn walk(shape: &egui::Shape, color: egui::Color32, count: &mut usize) {
                match shape {
                    egui::Shape::Rect(rect) if rect.stroke.color == color => *count += 1,
                    egui::Shape::Vec(shapes) => shapes.iter().for_each(|shape| walk(shape, color, count)),
                    _ => {}
                }
            }
            let mut count = 0;
            self.shapes.iter().for_each(|clipped| walk(&clipped.shape, color, &mut count));
            count
        }
    }

    #[test]
    fn a_character_clicked_in_the_result_shows_its_modules_on_the_grid() {
        let mut bench = Bench::new();
        bench.act(Action::ClosePhoto);
        let text = "https://example.com/q?id=7Q2M-XK4P-99AD";
        bench.hold(symbol_of(text));
        assert!(bench.app.derived.report.is_trustworthy());
        assert_eq!(bench.outlined(canvas::PICKED), 0);
        // The x of example, the tenth character: its glyph in the result.
        let (at, galley) = bench.text_shape(text).expect("the result shows the message");
        let row = &galley.rows[0];
        assert!(row.row.glyphs.len() > 9, "the message's first row holds its tenth character");
        let glyph = row.row.glyphs[9].logical_rect().center();
        bench.click(at + row.pos.to_vec2() + glyph.to_vec2(), egui::PointerButton::Primary);
        assert_eq!(bench.app.view.picked_char, Some(9));
        // Known text starts there, and the grid outlines the eight
        // modules the character is made of, over two codewords.
        assert_eq!((bench.app.view.known_at, bench.app.view.known_from_start), (9, false));
        assert_eq!(bench.outlined(canvas::PICKED), 8);
        let texts = bench.texts();
        let line = texts.iter().find(|text| text.starts_with("Character 10 “x”")).unwrap_or_else(|| panic!("{texts:?}"));
        assert!(line.contains("byte") && line.contains("stream bits 84 to 91") && line.contains("codewords "), "{line}");
        // The codewords named are those the bits lie in.
        let layout = bench.app.derived.layout.clone();
        assert!(line.contains(&format!("codewords {}, {} of block 1", layout.data_stream[10], layout.data_stream[11])), "{line}");
        // A second click on it lets go, and so does Esc.
        let (at, galley) = bench.text_shape(text).expect("the result shows the message");
        let row = &galley.rows[0];
        let glyph = row.row.glyphs[9].logical_rect().center();
        bench.click(at + row.pos.to_vec2() + glyph.to_vec2(), egui::PointerButton::Primary);
        assert_eq!(bench.app.view.picked_char, None);
        assert_eq!(bench.outlined(canvas::PICKED), 0);
        bench.app.view.picked_char = Some(3);
        bench.frame(vec![egui::Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }]);
        assert_eq!(bench.app.view.picked_char, None);
        // A character that is no longer there outlines nothing.
        bench.app.view.picked_char = Some(500);
        bench.step();
        assert_eq!(bench.outlined(canvas::PICKED), 0);
    }

    #[test]
    fn known_text_is_pinned_from_the_inspector() {
        let mut bench = Bench::new();
        // A window tall enough to show the whole of the inspector, whose
        // lower sections are otherwise scrolled out of what is painted.
        bench.size = egui::vec2(1440.0, 3000.0);
        bench.act(Action::ClosePhoto);
        // Without the trials of erasures and list decoding, which reach a
        // wrong codeword or two past the code by themselves: what pins
        // and doubting do is the point here. List decoding once reached
        // no further than the code here, its codewords given their
        // multiplicities a unit at a time.
        bench.app.opts.erase_by_trial = false;
        bench.app.opts.list_decode = false;
        let text = "https://example.com/q?id=7Q2M-XK4P-99AD";
        let clean = symbol_of(text);
        bench.hold(clean.clone());
        let layout = bench.app.derived.layout.clone();
        // Fifteen codewords wrong, twelve of them under the address's
        // beginning: more than twice what the block corrects.
        bench.hold(with_wrong(&clean, &layout, (2..=13).chain([26, 31, 37])));
        assert_eq!(bench.app.derived.report.blocks_solved(), 0);
        bench.app.view.known_text = "https://example.com/".to_owned();
        bench.step();
        // The inspector says what pinning would do before it is done.
        let texts = bench.texts();
        assert!(texts.iter().any(|text| text == "Pins 20 characters, in 21 codewords."), "{texts:?}");
        bench.act(Action::PinText);
        assert!(bench.app.derived.report.is_trustworthy());
        assert_eq!(bench.app.derived.report.text.plain(), text);
        assert_eq!(bench.notice().text, "Pinned 20 characters in 21 codewords: 1 of 1 blocks now verify, up from 0.");
        assert_eq!(bench.app.doc.pins.len(), 21);
        assert!(bench.app.doc.pins.values().all(|pin| pin.source == PinSource::Text));
        assert_eq!(bench.app.doc.pins_ec, Some(bench.app.derived.ec));
        bench.step();
        let texts = bench.texts();
        assert!(texts.iter().any(|text| text.starts_with("Pins: 0 by hand, 21 known text")), "{texts:?}");
        // One undo takes the pins back, and Remove text pins the same.
        bench.act(Action::Undo);
        assert!(bench.app.doc.pins.is_empty());
        bench.act(Action::Redo);
        bench.act(Action::ClearTextPins);
        assert!(bench.app.doc.pins.is_empty());
        bench.step();
        assert_eq!(bench.app.derived.report.blocks_solved(), 0);
        // Text that cannot be pinned where it is put is refused, with
        // the reason, and nothing changes.
        bench.app.view.known_at = 500;
        bench.act(Action::PinText);
        let notice = bench.notice();
        assert!(notice.is_error && notice.text.contains("has 39 characters, and none at that place"), "{}", notice.text);
        assert!(bench.app.doc.pins.is_empty());
        // A common beginning, tried and borne out: eight codewords wrong
        // under `https://`, one more than the block corrects.
        bench.hold(with_wrong(&clean, &layout, 2..=9));
        assert_eq!(bench.app.derived.report.blocks_solved(), 0);
        bench.act(Action::TryBeginnings);
        assert!(bench.notice().text.starts_with("The message begins \"https://\": with that pinned, 1 of 1 blocks verify, up from 0."), "{}", bench.notice().text);
        bench.step();
        assert!(bench.app.derived.report.is_trustworthy());
        assert_eq!(bench.app.derived.report.text.plain(), text);
        assert_eq!((bench.app.view.known_text.as_str(), bench.app.view.known_at, bench.app.view.known_from_start), ("https://", 0, false));
        // A message that begins no common way is left as it is, and
        // told so.
        bench.hold(with_wrong(&symbol_of("Invoice 2026-1010 for the lathe, due on receipt"), &layout, 2..=9));
        bench.act(Action::TryBeginnings);
        assert!(bench.notice().is_error && bench.app.doc.pins.is_empty(), "{}", bench.notice().text);
    }

    #[test]
    fn a_large_photo_keeps_its_pixels_and_is_drawn_through_a_texture_that_fits() {
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        // A photo wider than a texture may be here, the bench's render,
        // 330 pixels square, small in it at (3000, 600).
        let (w, h, pixels) = render_rgba(&painted_grid(), 10, 4);
        let render = image::RgbaImage::from_raw(w as u32, h as u32, pixels).expect("a render is an image");
        let mut large = image::RgbaImage::from_pixel(5000, 1500, image::Rgba([255, 255, 255, 255]));
        image::imageops::replace(&mut large, &render, 3000, 600);
        bench.app.set_photo(&ctx, "large".into(), large);
        // Loaded at its own size, pixel for pixel, and told of what its
        // size costs.
        let photo = bench.app.photo.as_ref().expect("the photo is loaded");
        assert_eq!((photo.width, photo.height, photo.scale()), (5000, 1500, 1.0));
        assert_eq!((photo.luma.width, photo.luma.height), (5000, 1500));
        assert!(Arc::ptr_eq(&photo.source, &photo.original), "the photo as loaded is held once");
        assert!(photo.adjust.enlarge == Enlarge::NONE && photo.luma.unenlarged.is_none());
        let on_offer = photo.enlargements.iter().map(Enlarge::to_string).collect::<Vec<_>>();
        assert_eq!(on_offer, ["0.25×", "0.5×", "1×", "2×", "4×"], "enlarged four times at the most, when that is asked for");
        let notice = &bench.notice().text;
        assert!(notice.starts_with("Loaded large (5000 × 1500)") && notice.contains("Crop to outline"), "{notice}");
        // What draws it is no larger than a texture may be, and is
        // stretched over the photo's own pixels.
        assert_eq!(photo.texture.size(), [2048, 614]);
        // The symbol is read where it lies, at its ten pixels a module:
        // an outline in the photo's own pixels reads every fixed module.
        bench.app.doc.corners = square(3040.0, 640.0, 250.0);
        bench.step();
        let (matching, fixed) = bench.app.preview.as_ref().expect("the outline makes a grid").fixed;
        assert_eq!(matching, fixed);
        // A search of the whole of it looks at a copy reduced to the size
        // a search goes to, and places what it finds in the photo's own
        // pixels.
        bench.act(Action::FindSymbols);
        bench.settle();
        let found = &bench.app.proposals.as_ref().expect("the symbol is found").found;
        assert_eq!(found.iter().map(|found| found.version).collect::<Vec<_>>(), [SymbolVersion::Qr(2)]);
        for (corner, on_symbol) in found[0].corners.iter().zip(square(3040.0, 640.0, 250.0)) {
            assert!((corner[0] - on_symbol[0]).hypot(corner[1] - on_symbol[1]) < 2.0, "{:?}", found[0].corners);
        }
        // So is the black and white view of it.
        bench.app.view.align_overlay = AlignOverlay::Binarised;
        bench.step();
        let binarised = bench.app.photo.as_ref().and_then(|photo| photo.binarised.as_ref()).expect("the view was made");
        assert_eq!(binarised.texture.size(), [2048, 614]);
        // A change of the adjustments keeps the texture within the limit.
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { contrast: 0.2, ..bench.app.doc.adjust }));
        assert_eq!(bench.app.photo.as_ref().map(|photo| photo.texture.size()), Some([2048, 614]));
        // A photo opened before the window has said what its limit is,
        // as one named on the command line is, is drawn in full once it
        // says so.
        bench.texture_side = Some(8192);
        bench.step();
        let photo = bench.app.photo.as_ref().expect("the photo is on show");
        assert_eq!(photo.texture.size(), [5000, 1500]);
        // The black and white view, on show, was made again with it.
        assert_eq!(photo.binarised.as_ref().map(|view| view.texture.size()), Some([5000, 1500]));
        // A small photo starts enlarged, and says so.
        let small = resize(render.as_raw(), w, h, 4, 110, 110);
        let small = image::RgbaImage::from_raw(110, 110, small).expect("the render at a third of its size");
        bench.app.set_photo(&ctx, "small".into(), small);
        let photo = bench.app.photo.as_ref().expect("the photo is loaded");
        assert_eq!((photo.width, photo.height, photo.scale()), (330, 330, 3.0));
        assert_eq!((photo.adjust.enlarge, bench.app.doc.adjust.enlarge), (Enlarge::times(3), Enlarge::times(3)));
        assert!(bench.notice().text.starts_with("Loaded small (330 × 330, enlarged 3 times from 110 × 110"), "{}", bench.notice().text);
        // Its own pixels are kept beside the enlarged, as loaded and as
        // luminance, through a change of the adjustments.
        let own = |bench: &Bench| {
            let photo = bench.app.photo.as_ref().expect("the photo is loaded");
            (photo.original.dimensions(), photo.luma.unenlarged.as_ref().map(|own| (own.width, own.height)))
        };
        assert!(Arc::ptr_eq(&photo.original, &photo.source), "the photo as loaded, held once");
        assert_eq!(own(&bench), ((110, 110), Some((110, 110))));
        bench.act(Action::SetAdjust(Adjust { contrast: 0.3, ..bench.app.doc.adjust }));
        assert_eq!(own(&bench), ((110, 110), Some((110, 110))));
        bench.step();
        assert!(bench.texts().iter().any(|text| text == "small · 330 × 330 · enlarged from 110 × 110"), "{:?}", bench.texts());
    }

    #[test]
    fn texture_sizes_fit_the_limit_and_keep_the_shape() {
        assert_eq!(texture_size(330, 330, 2048), (330, 330));
        assert_eq!(texture_size(2048, 100, 2048), (2048, 100));
        assert_eq!(texture_size(8000, 6000, 2048), (2048, 1536));
        assert_eq!(texture_size(6000, 8000, 4096), (3072, 4096));
        assert_eq!(texture_size(100_000, 3, 2048), (2048, 1), "no side comes to nothing");
        // The pixels are those given when they fit, and fewer of the
        // same when they do not.
        let pixels = |w: usize, h: usize| (0..w * h).flat_map(|i| [(i % 251) as u8, 0, 0, 255]).collect::<Vec<u8>>();
        let whole = texture_image(pixels(40, 30), 40, 30, 64);
        assert_eq!(whole.size, [40, 30]);
        assert_eq!(whole.pixels[41], egui::Color32::from_rgb(41, 0, 0));
        assert_eq!(texture_image(pixels(400, 300), 400, 300, 64).size, [64, 48]);
        assert_eq!(texture_gray(vec![255; 400 * 300], 400, 300, 64).size, [64, 48]);
        assert_eq!(texture_gray(vec![0; 40 * 30], 40, 30, 64).size, [40, 30]);
    }

    #[test]
    fn a_photo_is_cropped_to_its_outline_from_its_pixels_as_loaded() {
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        // The bench's render, 330 pixels square, at (3000, 600) of a
        // large photo.
        let (w, h, pixels) = render_rgba(&painted_grid(), 10, 4);
        let render = image::RgbaImage::from_raw(w as u32, h as u32, pixels).expect("a render is an image");
        let mut large = image::RgbaImage::from_pixel(5000, 1500, image::Rgba([255, 255, 255, 255]));
        image::imageops::replace(&mut large, &render, 3000, 600);
        bench.app.set_photo(&ctx, "large".into(), large);
        // The outline on the symbol and an anchor on its first finder.
        let on_symbol = square(3040.0, 640.0, 250.0);
        bench.app.doc.corners = on_symbol;
        bench.app.doc.set_anchor(finder_anchor([3075.0, 675.0]), AnchorSource::Manual);
        bench.act(Action::CropToOutline);
        bench.step();
        // Cut a fifth of the outline's size about it, and small enough
        // now to be drawn pixel for pixel.
        let photo = bench.app.photo.as_ref().expect("the photo is on show");
        assert_eq!(photo.crop, Some(Crop { rect: [2990, 590, 350, 350], k1: 0.0, enlarge: Enlarge::NONE }));
        assert_eq!((photo.width, photo.height, photo.scale()), (350, 350, 1.0));
        assert_eq!(photo.texture.size(), [350, 350]);
        assert!(bench.notice().text.starts_with("Cropped to the outline: 350 × 350 pixels of the photo as loaded. What was found"), "{}", bench.notice().text);
        // The outline and the anchor went with the symbol, which reads
        // there as it did.
        assert_eq!(bench.app.doc.corners, square(50.0, 50.0, 250.0));
        assert_eq!(bench.app.doc.anchors.iter().map(|a| (a.anchor.center, a.source)).collect::<Vec<_>>(), [([85.0, 85.0], AnchorSource::Manual)]);
        let (matching, fixed) = bench.app.preview.as_ref().expect("the outline makes a grid").fixed;
        assert_eq!(matching, fixed);
        assert!(bench.texts().iter().any(|text| text == "large · 350 × 350 · cropped from 5000 × 1500"), "{:?}", bench.texts());
        // A lens bends the photo about its centre, which the crop lost.
        bench.act(Action::EstimateLens);
        assert!(bench.notice().is_error && bench.notice().text.contains("show the whole photo"), "{}", bench.notice().text);
        // The whole photo again, with the outline where it was on it.
        bench.act(Action::ShowWholePhoto);
        bench.step();
        let photo = bench.app.photo.as_ref().expect("the photo is on show");
        assert_eq!((photo.width, photo.height, photo.crop), (5000, 1500, None));
        assert_eq!(bench.app.doc.corners, on_symbol);
        assert_eq!(bench.app.doc.anchors[0].anchor.center, [3075.0, 675.0]);
        assert!(bench.notice().text.starts_with("Showing the whole photo again (5000 × 1500)"), "{}", bench.notice().text);
        // With a lens term in force, the crop is of the photo
        // straightened by it, pixel for pixel what the whole photo shows
        // there, and takes none of its own; the whole photo takes the
        // term back.
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { k1: -0.05, ..bench.app.doc.adjust }));
        bench.act(Action::CropToOutline);
        let photo = bench.app.photo.as_ref().expect("the photo is on show");
        assert_eq!(photo.crop.map(|crop| crop.k1), Some(-0.05));
        assert_eq!((bench.app.doc.adjust.k1, photo.adjust.k1), (0.0, 0.0));
        assert!(bench.notice().text.contains("The cropped part keeps the whole photo's lens correction"), "{}", bench.notice().text);
        let [x, y, w, h] = photo.crop.expect("cropped").rect.map(|side| side as usize);
        let whole = Adjust { k1: -0.05, ..photo.adjust }.apply(photo.source.as_raw(), 5000, 1500).0;
        let there: Vec<u8> = (y..y + h).flat_map(|row| whole[(row * 5000 + x) * 4..(row * 5000 + x + w) * 4].to_vec()).collect();
        assert!(adjusted(&photo.adjust, &photo.source, &photo.original, photo.crop, false, false).0.into_bytes() == there, "the part is the whole straightened");
        assert!(photo.original.as_raw() == &there, "and so are its own pixels");
        bench.act(Action::ShowWholePhoto);
        assert_eq!(bench.app.doc.adjust.k1, -0.05);
        assert!(bench.app.photo.as_ref().is_some_and(|photo| photo.crop.is_none() && photo.adjust.k1 == -0.05));
        // A part keeps the enlargement the photo has, whatever its size:
        // its own size here, though a photo as small would start
        // enlarged were it loaded. Enlarge is the user's to set.
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { k1: 0.0, ..bench.app.doc.adjust }));
        bench.app.doc.corners = square(3040.0, 640.0, 100.0);
        bench.act(Action::CropToOutline);
        let photo = bench.app.photo.as_ref().expect("the photo is on show");
        assert_eq!(photo.crop, Some(Crop { rect: [3020, 620, 140, 140], k1: 0.0, enlarge: Enlarge::NONE }));
        assert_eq!((photo.width, photo.height, photo.scale(), photo.luma.unenlarged.is_none()), (140, 140, 1.0, true));
        assert_eq!((photo.adjust.enlarge, bench.app.doc.adjust.enlarge), (Enlarge::NONE, Enlarge::NONE));
        assert_eq!(photo.enlargements.iter().map(Enlarge::to_string).collect::<Vec<_>>(), ["0.25×", "0.5×", "1×", "2×", "4×", "8×", "16×"]);
        assert_eq!(bench.app.doc.corners, square(20.0, 20.0, 100.0));
        assert!(bench.notice().text.starts_with("Cropped to the outline: 140 × 140 pixels of the photo as loaded. What was found"), "{}", bench.notice().text);
        // Enlarged by hand, a part of that part is as much enlarged, and
        // the whole photo takes back the size it had.
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { enlarge: Enlarge::times(4), ..bench.app.doc.adjust }));
        assert_eq!(bench.app.photo.as_ref().map(|photo| (photo.width, photo.height)), Some((560, 560)));
        assert_eq!(bench.app.doc.corners, square(80.0, 80.0, 400.0));
        bench.app.doc.corners = square(120.0, 120.0, 320.0);
        bench.act(Action::CropToOutline);
        let photo = bench.app.photo.as_ref().expect("the photo is on show");
        assert_eq!(photo.crop, Some(Crop { rect: [3034, 634, 112, 112], k1: 0.0, enlarge: Enlarge::NONE }));
        assert_eq!((photo.width, photo.height, photo.adjust.enlarge), (448, 448, Enlarge::times(4)));
        assert_eq!(bench.app.doc.corners, square(64.0, 64.0, 320.0));
        assert!(bench.notice().text.contains("112 × 112 pixels of the photo as loaded, at 4× as the photo was."), "{}", bench.notice().text);
        bench.act(Action::ShowWholePhoto);
        let photo = bench.app.photo.as_ref().expect("the photo is on show");
        assert_eq!((photo.width, photo.height, photo.adjust.enlarge, bench.app.doc.adjust.enlarge), (5000, 1500, Enlarge::NONE, Enlarge::NONE));
        assert_eq!(bench.app.doc.corners, square(3050.0, 650.0, 80.0));
        // A photo reduced is cropped from its pixels as loaded all the
        // same, by where the outline lies on them, and the part keeps
        // the reduction, as it keeps the other adjustments.
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { enlarge: Enlarge::HALF, contrast: 0.25, ..bench.app.doc.adjust }));
        assert_eq!(bench.app.photo.as_ref().map(|photo| (photo.width, photo.height)), Some((2500, 750)));
        assert_eq!(bench.app.doc.corners, square(1525.0, 325.0, 40.0));
        bench.app.doc.corners = square(1500.0, 300.0, 400.0);
        bench.act(Action::CropToOutline);
        bench.step();
        let photo = bench.app.photo.as_ref().expect("the photo is on show");
        assert_eq!(photo.crop, Some(Crop { rect: [2840, 440, 1120, 1060], k1: 0.0, enlarge: Enlarge::HALF }));
        assert_eq!((photo.width, photo.height, photo.scale()), (560, 530, 0.5));
        assert_eq!(bench.app.doc.adjust, Adjust { enlarge: Enlarge::HALF, contrast: 0.25, ..opaque(Adjust::default()) });
        assert_eq!(bench.app.doc.corners, square(80.0, 80.0, 400.0));
        assert!(bench.notice().text.contains("1120 × 1060 pixels of the photo as loaded, at 0.5× as the photo was."), "{}", bench.notice().text);
        assert!(bench.texts().iter().any(|text| text == "large · 560 × 530 · cropped from 5000 × 1500"), "{:?}", bench.texts());
        assert!(bench.texts().iter().any(|text| text == "0.5×"), "the slider shows it: {:?}", bench.texts());
        // The symbol in it reads as it did, at five pixels a module.
        bench.app.doc.corners = square(100.0, 100.0, 125.0);
        bench.step();
        let (matching, fixed) = bench.app.preview.as_ref().expect("the outline makes a grid").fixed;
        assert_eq!(matching, fixed);
        // A smaller part of it is reduced as much.
        bench.act(Action::CropToOutline);
        let photo = bench.app.photo.as_ref().expect("the photo is on show");
        assert_eq!(photo.crop, Some(Crop { rect: [2990, 590, 350, 350], k1: 0.0, enlarge: Enlarge::HALF }));
        assert_eq!((photo.width, photo.height, photo.adjust.enlarge), (175, 175, Enlarge::HALF));
        assert_eq!(bench.app.doc.corners, square(25.0, 25.0, 125.0));
        assert!(bench.notice().text.contains("350 × 350 pixels of the photo as loaded, at 0.5× as the photo was."), "{}", bench.notice().text);
        // And so is a part so small that the slider would not offer the
        // reduction for it: it is offered what it has, and keeps it
        // through a change of another adjustment.
        bench.app.doc.corners = square(45.0, 45.0, 20.0);
        bench.act(Action::CropToOutline);
        bench.step();
        let photo = bench.app.photo.as_ref().expect("the photo is on show");
        assert_eq!(photo.crop, Some(Crop { rect: [3072, 672, 56, 56], k1: 0.0, enlarge: Enlarge::HALF }));
        assert_eq!((photo.width, photo.height, photo.adjust.enlarge), (28, 28, Enlarge::HALF));
        assert_eq!(bench.app.doc.corners, square(4.0, 4.0, 20.0));
        assert_eq!(photo.enlargements.iter().map(Enlarge::to_string).collect::<Vec<_>>(), ["0.5×", "1×", "2×", "4×", "5×", "8×", "16×", "32×", "64×"]);
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { contrast: 0.1, ..bench.app.doc.adjust }));
        bench.step();
        let photo = bench.app.photo.as_ref().expect("the photo is on show");
        assert_eq!((photo.width, photo.height, bench.app.doc.adjust.enlarge), (28, 28, Enlarge::HALF));
        // Reset, cropped, goes to what a photo of the part's size starts
        // with, and the whole photo takes its reduction back.
        bench.step();
        bench.click(bench.text_at("Reset adjustments"), egui::PointerButton::Primary);
        assert_eq!(bench.app.doc.adjust, opaque(Adjust::for_photo(56, 56)));
        assert_eq!(bench.app.photo.as_ref().map(|photo| (photo.width, photo.adjust.enlarge)), Some((280, Enlarge::times(5))));
        bench.act(Action::ShowWholePhoto);
        let photo = bench.app.photo.as_ref().expect("the photo is on show");
        assert_eq!((photo.width, photo.height, photo.adjust.enlarge, photo.crop), (2500, 750, Enlarge::HALF, None));
        assert_eq!(bench.app.doc.corners, square(1540.0, 340.0, 20.0));
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust::default()));
        // An outline that takes in the whole photo leaves nothing to
        // crop to, and one off the photo nothing to cut.
        bench.app.doc.corners = [[0.0, 0.0], [5000.0, 0.0], [5000.0, 1500.0], [0.0, 1500.0]];
        bench.act(Action::CropToOutline);
        assert!(bench.notice().is_error && bench.app.photo.as_ref().is_some_and(|photo| photo.crop.is_none()));
        bench.app.doc.corners = square(9000.0, 9000.0, 100.0);
        bench.act(Action::CropToOutline);
        assert!(bench.notice().is_error && bench.app.photo.as_ref().is_some_and(|photo| photo.crop.is_none()));
    }

    impl Bench {
        /// Where the Edit stage drew each module's middle, by way of the
        /// grid's outline, the largest rectangle it strokes in its accent.
        fn cells(&self) -> impl Fn(usize, usize) -> egui::Pos2 + use<> {
            fn walk(shape: &egui::Shape, out: &mut Option<egui::Rect>) {
                match shape {
                    egui::Shape::Rect(rect) if rect.stroke.color == canvas::ACCENT && out.is_none_or(|held| rect.rect.area() > held.area()) => {
                        *out = Some(rect.rect);
                    }
                    egui::Shape::Vec(shapes) => shapes.iter().for_each(|shape| walk(shape, out)),
                    _ => {}
                }
            }
            let mut grid = None;
            self.shapes.iter().for_each(|clipped| walk(&clipped.shape, &mut grid));
            let grid = grid.expect("the Edit stage outlined the grid");
            let module = grid.width() / self.app.derived.func.width as f32;
            move |x, y| grid.min + egui::vec2(x as f32 + 0.5, y as f32 + 0.5) * module
        }
    }

    #[test]
    fn what_find_in_outline_reads_is_loaded_into_the_grid_unless_that_is_turned_off() {
        let mut bench = Bench::new();
        let ctx = bench.ctx.clone();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/photos");
        let photo = image::open(dir.join("zxing_qrcode-3_01.png")).expect("photo opens").to_rgba8();
        let message = std::fs::read_to_string(dir.join("zxing_qrcode-3_01.txt")).expect("the photo's text");
        bench.app.set_photo(&ctx, "photo".into(), photo);
        // An outline loosely about the whole photo.
        let (w, h) = bench.app.photo.as_ref().map(|photo| (photo.width as f64, photo.height as f64)).expect("a photo");
        let loose = [[0.03 * w, 0.03 * h], [0.99 * w, 0.03 * h], [0.99 * w, 0.99 * h], [0.03 * w, 0.99 * h]];
        bench.app.doc.corners = loose;
        // With reads not loaded, the symbol found and read inside it is
        // adopted and no more: the outline and the version, with the
        // grid still to be sampled.
        bench.app.view.load_reads = false;
        bench.act(Action::FindCorners);
        bench.settle();
        let version = bench.app.doc.version;
        assert_ne!(version, SymbolVersion::Qr(2));
        bench.assert_all_of(version);
        let notice = bench.notice().text.clone();
        assert!(notice.contains("blocks verified, the message decodes.") && !notice.contains("Loaded"), "{notice}");
        assert!(!bench.app.derived.report.is_trustworthy());
        assert!(bench.app.doc.grid.count(Module::Unknown) > 0);
        let outline = bench.app.doc.corners;
        bench.act(Action::Undo);
        bench.step();
        assert_eq!((bench.app.doc.version, bench.app.doc.corners), (SymbolVersion::Qr(2), loose));
        // With reads loaded, as they are unless that is turned off, the
        // document is what was read, under the outline the read left on
        // the symbol.
        bench.app.view.load_reads = true;
        bench.act(Action::FindCorners);
        bench.settle();
        bench.step();
        assert_eq!(bench.app.view.stage, Stage::Align);
        bench.assert_all_of(version);
        assert!(bench.app.derived.report.is_trustworthy());
        assert_eq!(bench.app.derived.report.text.plain(), message);
        assert!(same_outline(&bench.app.doc.corners, &outline));
        assert!(bench.notice().text.starts_with("Loaded the symbol inside the outline as it was read"), "{}", bench.notice().text);
        // Nothing lists the read apart: the grid is where it went.
        bench.size = egui::vec2(1440.0, 3000.0);
        bench.step();
        bench.step();
        let texts = bench.texts();
        assert!(!texts.iter().any(|text| text == "Symbols read" || text == "Load"), "{texts:?}");
        // One undo takes the whole of it back.
        bench.act(Action::Undo);
        bench.step();
        assert!(!bench.app.derived.report.is_trustworthy());
        assert_eq!((bench.app.doc.version, bench.app.doc.corners), (SymbolVersion::Qr(2), loose));
        // A placement picked from those the detector ranks inside the
        // outline is read through and loaded the same way.
        bench.act(Action::ListPlacements);
        bench.settle();
        bench.act(Action::UsePlacement(0));
        assert_eq!(bench.app.running(), Some(JobKind::Adopt));
        bench.settle();
        assert!(bench.notice().text.starts_with("Loaded placement 1 as it was read"), "{}", bench.notice().text);
        assert!(bench.app.derived.report.is_trustworthy());
        assert_eq!(bench.app.derived.report.text.plain(), message);
    }

    #[test]
    fn the_symbol_loaded_of_itself_is_the_outlines_own_or_the_first_that_decodes() {
        // A claim on the symbol's lower right quarter, and the symbol.
        let found = vec![detection(SymbolVersion::Micro(3), square(170.0, 170.0, 140.0)), detection(SymbolVersion::Qr(2), STRAIGHT)];
        // With the outline on neither, the first that decodes.
        assert_eq!(symbol_to_load(&found, None, &ASIDE, |_| true), Some(0));
        assert_eq!(symbol_to_load(&found, None, &ASIDE, |i| i == 1), Some(1));
        assert_eq!(symbol_to_load(&found, None, &ASIDE, |_| false), None);
        // With the outline on one, that one or none: a grid being worked
        // on is not swapped for another symbol's.
        assert_eq!(symbol_to_load(&found, None, &STRAIGHT, |_| true), Some(1));
        assert_eq!(symbol_to_load(&found, None, &STRAIGHT, |i| i == 0), None);
        // Where two lie over one another, the one the outline was put on.
        let over = vec![detection(SymbolVersion::Qr(2), STRAIGHT), detection(SymbolVersion::Qr(3), STRAIGHT)];
        assert_eq!(symbol_to_load(&over, Some(1), &STRAIGHT, |_| true), Some(1));
        assert_eq!(symbol_to_load(&over, Some(1), &STRAIGHT, |i| i == 0), None);
        assert_eq!(symbol_to_load(&over, None, &STRAIGHT, |_| true), Some(0));
        // A symbol since gone from the list was the outline's no longer.
        assert_eq!(symbol_to_load(&over, Some(7), &ASIDE, |i| i == 1), Some(1));
    }

    #[test]
    fn a_search_loads_no_symbol_over_the_one_being_worked_on() {
        let mut bench = Bench::new();
        let decoding = |version, corners| Detection { decodes: Some(true), ..detection(version, corners) };
        // The outline on a symbol found that does not decode as found,
        // and another in the list that does: nothing is picked, and the
        // list says how to go on.
        bench.app.doc.corners = STRAIGHT;
        let found = vec![decoding(SymbolVersion::Micro(3), square(170.0, 170.0, 140.0)), detection(SymbolVersion::Qr(2), STRAIGHT)];
        bench.app.finish_search(&bench.ctx.clone(), found.clone(), Vec::new(), false, None, false);
        assert_eq!(bench.app.running(), None);
        assert!(bench.notice().text.contains("Pick one under Alignment"), "{}", bench.notice().text);
        // With the outline on neither, the one that decodes as found is
        // picked, and read through.
        bench.app.doc.corners = ASIDE;
        bench.app.finish_search(&bench.ctx.clone(), found.clone(), Vec::new(), false, None, false);
        assert_eq!(bench.app.running(), Some(JobKind::Adopt));
        assert!(bench.notice().text.contains("Symbol 1 decodes as found and is being read into the grid"), "{}", bench.notice().text);
        bench.settle();
        assert_eq!(bench.app.proposals.as_ref().and_then(|proposals| proposals.adopted), Some(0));
        // Not with reads not loaded, nor while another step is under way.
        bench.app.doc.corners = ASIDE;
        bench.app.view.load_reads = false;
        bench.app.finish_search(&bench.ctx.clone(), found, Vec::new(), false, None, false);
        assert_eq!(bench.app.running(), None);
    }

    #[test]
    fn the_versions_suggested_are_taken_off_the_list_when_asked() {
        let mut bench = Bench::new();
        bench.app.doc.corners = STRAIGHT;
        bench.act(Action::SuggestVersions);
        bench.settle();
        assert!(bench.app.guesses.as_ref().is_some_and(|guesses| !guesses.is_empty()));
        bench.size = egui::vec2(1440.0, 3000.0);
        bench.step();
        bench.step();
        bench.click(bench.text_at("Clear the list"), egui::PointerButton::Primary);
        assert!(bench.app.guesses.is_none());
        bench.step();
        assert!(!bench.texts().iter().any(|text| text == "Clear the list"), "{:?}", bench.texts());
    }

    #[test]
    fn the_photos_ink_paints_a_colour_where_the_unsure_band_leaves_a_module_unknown() {
        let mut bench = Bench::new();
        bench.app.doc.corners = STRAIGHT;
        // An unsure band wider than any margin, set past what the slider
        // allows, with no frame drawn to bring it back: a sampling of
        // the grid leaves every module unknown.
        bench.app.view.sample.band = 100.0;
        bench.app.update_preview();
        let truth = painted_grid();
        let sampled = &bench.app.preview.as_ref().expect("a sample to go by").classified;
        assert!(sampled.modules.iter().all(|&module| module == Module::Unknown));
        // The ink has no band: each module is the colour it falls on,
        // at the reliability it read at.
        for (x, y) in [(10, 10), (11, 10), (12, 11)] {
            bench.app.paint(x, y, Ink::Photo);
            assert_eq!(bench.app.doc.grid.get(x as usize, y as usize), truth.get(x as usize, y as usize), "({x}, {y})");
            assert!(bench.app.doc.grid.reliability(x as usize, y as usize) < CERTAIN);
        }
        // Sample all keeps the band, and leaves them unknown.
        bench.act(Action::Sample { only_unknown: false, scope: SampleScope::All });
        assert_eq!(bench.app.doc.grid.get(10, 10), Module::Unknown);
    }

    #[test]
    fn the_photos_ink_paints_a_guess_while_paint_as_a_guess_holds() {
        let mut bench = Bench::new();
        bench.app.doc.corners = STRAIGHT;
        bench.app.update_preview();
        let truth = painted_grid();
        let read: Vec<u8> = {
            let sampled = &bench.app.preview.as_ref().expect("a sample to go by").classified;
            [(10, 10), (11, 10), (12, 11)].iter().map(|&(x, y): &(usize, usize)| sampled.reliability[y * truth.width() + x]).collect()
        };
        assert!(read.iter().all(|&reliability| reliability > 0), "{read:?}: a clean photo reads well clear of its threshold");
        // As a guess, each module is the colour it falls on still, and no
        // surer than one read right on its threshold; at a reliability
        // set by hand, as sure as that.
        bench.app.view.paint_reliability = 0;
        for (x, y) in [(10, 10), (11, 10), (12, 11)] {
            bench.app.paint(x, y, Ink::Photo);
            assert_eq!(bench.app.doc.grid.get(x as usize, y as usize), truth.get(x as usize, y as usize), "({x}, {y})");
            assert_eq!(bench.app.doc.grid.reliability(x as usize, y as usize), 0, "({x}, {y})");
        }
        bench.app.view.paint_reliability = 40;
        bench.app.paint(10, 10, Ink::Photo);
        assert_eq!(bench.app.doc.grid.reliability(10, 10), 40);
        // The Doubtful ink lowers a module without repainting it, and its
        // other button makes it certain again.
        bench.app.view.paint_reliability = 16;
        bench.app.paint(11, 10, Ink::Doubt);
        assert_eq!((bench.app.doc.grid.get(11, 10), bench.app.doc.grid.reliability(11, 10)), (truth.get(11, 10), 16));
        bench.app.paint(11, 10, Ink::Sure);
        assert_eq!(bench.app.doc.grid.reliability(11, 10), CERTAIN);
        // Painted over at certain, each has the reliability it read at
        // again.
        bench.app.view.paint_reliability = CERTAIN;
        for (&(x, y), &reliability) in [(10, 10), (11, 10), (12, 11)].iter().zip(&read) {
            bench.app.paint(x, y, Ink::Photo);
            assert_eq!(bench.app.doc.grid.reliability(x as usize, y as usize), reliability.min(CERTAIN - 1), "({x}, {y})");
        }
    }

    #[test]
    fn a_part_of_the_grid_is_sampled_and_the_rest_left_as_it_is() {
        let mut bench = Bench::new();
        bench.app.doc.corners = STRAIGHT;
        bench.step();
        let (truth, blank) = (painted_grid(), bench.app.doc.grid.clone());
        let (width, height) = (blank.width(), blank.height());
        let in_rect = |x: usize, y: usize| (10..=12).contains(&x) && (10..=11).contains(&y);
        let check = |grid: &ModuleGrid, sampled: &dyn Fn(usize, usize) -> bool| {
            for y in 0..height {
                for x in 0..width {
                    assert_eq!(grid.get(x, y), if sampled(x, y) { truth.get(x, y) } else { blank.get(x, y) }, "({x}, {y})");
                }
            }
        };
        // The photo's ink, picked by its key, and a rectangle of six data
        // modules dragged with it, from either corner: each is read off
        // the photo, and none beside.
        let key = |key| egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers: egui::Modifiers::NONE };
        bench.app.view.stage = Stage::Edit;
        bench.app.view.tool = Tool::Rect;
        bench.frame(vec![key(egui::Key::Num4)]);
        assert_eq!(bench.app.view.ink, Ink::Photo);
        bench.step();
        let cell = bench.cells();
        bench.frame(vec![egui::Event::PointerMoved(cell(12, 10))]);
        bench.step();
        bench.button(cell(12, 10), egui::PointerButton::Primary, true);
        bench.frame(vec![egui::Event::PointerMoved(cell(10, 11))]);
        bench.step();
        assert!(bench.app.view.stroke.is_some(), "the drag is under way");
        bench.button(cell(10, 11), egui::PointerButton::Primary, false);
        bench.step();
        check(&bench.app.doc.grid, &in_rect);
        assert!(bench.app.doc.grid.reliability(10, 10) < CERTAIN, "sampled, with how surely it read");
        assert_eq!(bench.app.doc.grid.reliability(13, 10), CERTAIN);
        // One undo takes the rectangle back.
        bench.act(Action::Undo);
        bench.step();
        check(&bench.app.doc.grid, &|_, _| false);
        // The Brush reads each module it passes over the same way, and
        // with the right button paints a colour, as under any ink.
        bench.app.view.tool = Tool::Brush;
        bench.step();
        let cell = bench.cells();
        bench.frame(vec![egui::Event::PointerMoved(cell(10, 10))]);
        bench.step();
        bench.button(cell(10, 10), egui::PointerButton::Primary, true);
        bench.frame(vec![egui::Event::PointerMoved(cell(12, 10))]);
        bench.step();
        bench.button(cell(12, 10), egui::PointerButton::Primary, false);
        bench.step();
        check(&bench.app.doc.grid, &|x, y| (10..=12).contains(&x) && y == 10);
        bench.click(cell(13, 11), egui::PointerButton::Secondary);
        assert_eq!(bench.app.doc.grid.get(13, 11), Module::Dark);
        // A fixed pattern stays as the standard has it while the fixed
        // patterns are locked, whatever the photo shows there.
        assert!(bench.app.view.lock_fixed);
        bench.click(cell(3, 3), egui::PointerButton::Primary);
        assert_eq!(bench.app.doc.grid.reliability(3, 3), CERTAIN);
        // A block alone: every module of its codewords, and none beside.
        bench.app.doc.grid = blank.clone();
        bench.app.dirty = true;
        bench.step();
        bench.act(Action::Sample { only_unknown: false, scope: SampleScope::Block(0) });
        bench.step();
        let layout = bench.app.derived.layout.clone();
        let mut of_block = vec![false; width * height];
        for &cw in &layout.blocks[0].codewords {
            for pos in layout.codeword_modules(cw) {
                of_block[pos.y as usize * width + pos.x as usize] = true;
            }
        }
        assert_eq!(of_block.iter().filter(|&&of| of).count(), 8 * layout.blocks[0].codewords.len());
        check(&bench.app.doc.grid, &|x, y| of_block[y * width + x]);
        assert!(bench.notice().text.starts_with("Sampled block 1 from the photo:"), "{}", bench.notice().text);
        // What sampling the whole grid would change is said before it is
        // done, and is what it then changes.
        let all = bench.app.sample_would_change(false).expect("a sample to go by");
        let unknown = bench.app.sample_would_change(true).expect("a sample to go by");
        assert!(all > 0 && unknown <= all, "{all} and {unknown}");
        bench.act(Action::Sample { only_unknown: false, scope: SampleScope::All });
        assert!(bench.notice().text.starts_with(&format!("Sampled the photo: {all} modules changed.")), "{}", bench.notice().text);
        bench.step();
        assert_eq!(bench.app.sample_would_change(false), Some(0));
        // Nothing inverts the photo for a part of the grid, as Sample all
        // inverts it for the whole: off a photo whose fixed patterns read
        // inverted the ink paints modules the wrong way round, and says
        // so, as the overlay that sets the grid against the photo does.
        assert_eq!(bench.app.inverted_note(), None);
        bench.app.history.checkpoint(&bench.app.doc);
        bench.act(Action::SetAdjust(Adjust { invert: true, ..bench.app.doc.adjust }));
        bench.step();
        assert!(bench.app.inverted_note().is_some_and(|note| note.contains("Untick it")));
        assert_eq!((bench.app.view.tool, bench.app.view.ink), (Tool::Brush, Ink::Photo));
        let cell = bench.cells();
        bench.click(cell(12, 11), egui::PointerButton::Primary);
        assert_eq!(bench.app.doc.grid.get(12, 11), truth.get(12, 11).inverted());
        let notice = bench.notice().text.clone();
        assert!(notice.starts_with("The modules painted were read off the photo as it is. The fixed patterns read inverted"), "{notice}");
        bench.app.view.overlay = Overlay::Photo;
        bench.step();
        let texts = bench.texts();
        assert!(texts.iter().any(|text| text.starts_with("The grid is compared with the photo as it is.")), "{texts:?}");
        // The photo's ink goes with the photo.
        bench.act(Action::ClosePhoto);
        assert_eq!(bench.app.view.ink, Ink::Module(Module::Dark));
    }

    #[test]
    fn the_placements_inside_an_outline_are_listed_and_one_is_taken() {
        let mut bench = Bench::new();
        // An outline loosely about the symbol, as one is drawn by hand.
        bench.app.doc.corners = square(25.0, 25.0, 280.0);
        bench.act(Action::ListPlacements);
        assert_eq!(bench.app.running(), Some(JobKind::Placements));
        bench.settle();
        let placements = bench.app.placements.as_ref().expect("the placements were ranked");
        let count = placements.found.len();
        assert!((1..=qr_forensics::imaging::detect::MAX_CANDIDATE_DETECTIONS).contains(&count), "{count}");
        // The first is the symbol as it lies, and the outline has not
        // moved for the asking.
        assert_eq!(placements.found[0].version, SymbolVersion::Qr(2));
        assert!(same_outline(&placements.found[0].corners, &STRAIGHT));
        assert_eq!(bench.app.doc.corners, square(25.0, 25.0, 280.0));
        // The outline lies about every one of them, and is on none until
        // one is picked.
        assert!(same_outline(&placements.found[0].corners, &bench.app.doc.corners));
        assert_eq!(placement_in_use(placements, &bench.app.doc.corners), None);
        bench.size = egui::vec2(1440.0, 3000.0);
        bench.step();
        bench.step();
        let texts = bench.texts();
        assert!(texts.iter().any(|text| *text == format!("Placements inside the outline ({count}):")), "{texts:?}");
        // Taken, it is refined and adopted as a symbol a search found is.
        bench.act(Action::UsePlacement(0));
        bench.settle();
        assert!(bench.notice().text.contains("fixed patterns match 236 of 236"), "{}", bench.notice().text);
        for (corner, want) in bench.app.doc.corners.iter().zip(STRAIGHT) {
            assert!((corner[0] - want[0]).abs() < 2.0 && (corner[1] - want[1]).abs() < 2.0, "{corner:?} for {want:?}");
        }
        let placements = bench.app.placements.as_ref().expect("the list is still on offer");
        assert_eq!(placement_in_use(placements, &bench.app.doc.corners), Some(0));
        // One that has gone cannot be taken, and the list goes with the
        // photo it was made on.
        bench.act(Action::ClearPlacements);
        bench.act(Action::UsePlacement(0));
        assert!(bench.notice().is_error && bench.app.placements.is_none());
        bench.act(Action::ListPlacements);
        bench.settle();
        assert!(bench.app.placements.is_some());
        bench.act(Action::ClosePhoto);
        assert!(bench.app.placements.is_none());
    }

    #[test]
    fn bits_of_a_codeword_are_pinned_by_themselves() {
        let mut bench = Bench::new();
        bench.act(Action::ClosePhoto);
        bench.hold(symbol_of("https://example.com/q?id=7Q2M-XK4P-99AD"));
        assert!(bench.app.derived.report.is_trustworthy());
        let cw = bench.app.derived.layout.data_stream[5];
        let value = bench.app.derived.report.codewords[cw].final_value.expect("the block solved");
        // The high four bits, as they are: a pin of those bits and no
        // others, which changes nothing of the decode.
        bench.act(Action::PinBits { cw, mask: 0xF0, value });
        bench.step();
        assert_eq!(bench.app.doc.pins.get(&cw), Some(&Pin::new(0xF0, value & 0xF0, PinSource::Manual)));
        assert_eq!(bench.app.doc.pins_ec, Some(bench.app.derived.ec));
        assert!(bench.app.derived.report.is_trustworthy());
        // One of them pinned the other way contradicts the only
        // solution, and the block says of which codeword, with a button
        // that goes to it.
        bench.act(Action::PinBits { cw, mask: 0xF0, value: value ^ 0x80 });
        bench.size = egui::vec2(1440.0, 3000.0);
        bench.step();
        bench.step();
        let index = bench.app.derived.layout.blocks[0].codewords.iter().position(|&of| of == cw).expect("a codeword of the block");
        assert_eq!(bench.app.derived.report.blocks[0].outcome, BlockOutcome::Failed(RsFailure::PinConflict { index }));
        let texts = bench.texts();
        assert!(texts.iter().any(|text| text.ends_with("the only solution contradicts a pin")), "{texts:?}");
        assert_eq!(bench.app.view.selected, None);
        bench.click(bench.text_at(&format!("Go to codeword {cw}")), egui::PointerButton::Primary);
        assert_eq!(bench.app.view.selected, Some(cw));
        // No bits, and the pin is gone.
        bench.act(Action::PinBits { cw, mask: 0, value: 0 });
        bench.step();
        assert!(bench.app.doc.pins.is_empty() && bench.app.derived.report.is_trustworthy());
        // One undo for each.
        bench.act(Action::Undo);
        assert_eq!(bench.app.doc.pins.get(&cw).map(|pin| pin.value), Some((value ^ 0x80) & 0xF0));
    }

    #[test]
    fn each_pin_is_charged_for_what_it_was_picked_among_until_it_goes() {
        let mut bench = Bench::new();
        bench.act(Action::ClosePhoto);
        bench.hold(symbol_of("hi"));
        assert!(bench.app.derived.report.is_trustworthy());
        let ec = bench.app.derived.ec;
        let value = |bench: &Bench, cw: usize| bench.app.derived.report.codewords[cw].final_value.expect("a clean symbol");
        let (first, last) = (bench.app.derived.layout.data_stream[0], bench.app.derived.layout.data_stream[50]);
        let picked = |bench: &mut Bench| {
            bench.app.refresh();
            bench.app.derived.report.pin_tries
        };
        // A beginning tried and pinned as text, the pick of a thousand
        // ways, beside a pin typed by hand, and then cleared: the pin
        // typed by hand that is left was picked among nothing. It once
        // kept the beginning's count.
        let text = Pin { among: 1000, ..Pin::full(value(&bench, first), PinSource::Text) };
        bench.app.doc.pins = [(first, text), (last, Pin::full(value(&bench, last), PinSource::Manual))].into();
        bench.app.doc.pins_ec = Some(ec);
        assert_eq!(picked(&mut bench), 1000);
        bench.act(Action::ClearTextPins);
        assert_eq!((bench.app.doc.pins.len(), picked(&mut bench)), (1, 1));
        // Padding recovered under the beginning was the pick of it too,
        // and keeps its count with the beginning gone.
        let padding = Pin { among: 1000 * 6, ..Pin::full(value(&bench, last), PinSource::Padding) };
        bench.app.doc.pins = [(first, text), (last, padding)].into();
        bench.act(Action::ClearTextPins);
        assert_eq!((bench.app.doc.pins.len(), picked(&mut bench)), (1, 6000));
        // With the padding cleared, a codeword pinned by hand is picked
        // among nothing. It once took the padding's count.
        bench.act(Action::ClearPaddingPins);
        assert!(bench.app.doc.pins.is_empty());
        bench.act(Action::Pin(first, value(&bench, first)));
        assert_eq!((bench.app.doc.pins.len(), picked(&mut bench)), (1, 1));
        // Pins made under another level give way to padding pinned by
        // hand, which was picked among nothing.
        let other = *bench.app.derived.func.version.ec_levels().iter().find(|&&level| level != ec).expect("another level");
        bench.app.doc.pins = [(first, text)].into();
        bench.app.doc.pins_ec = Some(other);
        bench.act(Action::PinPaddingAt { first: 4, filler: Filler::Standard });
        assert_eq!(bench.app.doc.pins_ec, Some(ec));
        assert!(bench.app.doc.pins.values().all(|pin| pin.among == 1));
        assert_eq!(picked(&mut bench), 1);
    }

    #[test]
    fn padding_is_pinned_from_a_start_given_by_hand() {
        let mut bench = Bench::new();
        bench.act(Action::ClosePhoto);
        // Two characters in a version 3 symbol, and every codeword of
        // the padding after them unknown: more than error correction
        // restores, and nothing of the padding left to be found by.
        let mut grid = symbol_of("hi");
        let layout = SymbolLayout::new(Arc::new(FunctionMap::new(SymbolVersion::Qr(3))), EcLevel::L).expect("a level of the version");
        for &cw in &layout.data_stream[4..] {
            for pos in layout.codeword_modules(cw) {
                grid.set(pos.x as usize, pos.y as usize, Module::Unknown);
            }
        }
        bench.hold(grid);
        assert_eq!(bench.app.derived.ec, EcLevel::L);
        assert!(!bench.app.derived.report.is_complete() && bench.app.derived.padding.is_none());
        bench.act(Action::ApplyPadding);
        assert!(bench.notice().is_error, "{}", bench.notice().text);
        // Said by hand, it is pinned and the block solves, and the start
        // is the decode's to say again.
        bench.app.view.pad_from = Some(4);
        bench.act(Action::PinPaddingAt { first: 4, filler: Filler::Standard });
        assert_eq!(bench.app.view.pad_from, None);
        bench.step();
        assert!(bench.notice().text.starts_with("Pinned 51 codewords of padding from message byte 4, as 0xEC and 0x11 in turn: 1 of 1 blocks solved, up from 0."), "{}", bench.notice().text);
        assert_eq!(bench.app.doc.pins.values().filter(|pin| pin.source == PinSource::Padding).count(), 51);
        assert!(bench.app.derived.report.is_trustworthy());
        assert_eq!(bench.app.derived.report.text.plain(), "hi");
        // The inspector offers it, with how the start and filler fit.
        bench.size = egui::vec2(1440.0, 3000.0);
        bench.step();
        bench.step();
        let texts = bench.texts();
        assert!(texts.iter().any(|text| text == "By hand: padding from message byte"), "{texts:?}");
        assert!(texts.iter().any(|text| text.starts_with("51 codewords from there; ")), "{texts:?}");
        // A start past the data pins nothing, and says so.
        bench.act(Action::PinPaddingAt { first: 99, filler: Filler::Standard });
        assert!(bench.notice().is_error);
        // One undo takes the pins back.
        bench.act(Action::Undo);
        assert!(bench.app.doc.pins.is_empty());
        // A start given for this symbol is not kept for another: not
        // for one of another layout, nor for a grid begun afresh.
        bench.app.view.pad_from = Some(9);
        bench.step();
        assert_eq!(bench.app.view.pad_from, Some(9), "it stays while the symbol does");
        bench.act(Action::SetVersion(SymbolVersion::Qr(4)));
        bench.step();
        assert_eq!(bench.app.view.pad_from, None);
        bench.app.view.pad_from = Some(9);
        bench.act(Action::NewBlank);
        assert_eq!(bench.app.view.pad_from, None);
    }

    #[test]
    fn every_overlay_draws_and_a_segment_picked_is_outlined() {
        let mut bench = Bench::new();
        bench.app.doc.corners = STRAIGHT;
        bench.app.view.stage = Stage::Edit;
        bench.step();
        // Against the photo, a grid with nothing sampled differs in
        // every module the photo reads and the grid leaves unknown, and
        // one sampled from it in none.
        bench.app.view.overlay = Overlay::Photo;
        bench.step();
        let unknown = bench.app.doc.grid.count(Module::Unknown);
        assert!(unknown > 300 && bench.outlined(canvas::GOOD) == unknown, "{} of {unknown}", bench.outlined(canvas::GOOD));
        bench.act(Action::Sample { only_unknown: false, scope: SampleScope::All });
        bench.step();
        bench.step();
        assert_eq!(bench.outlined(canvas::GOOD), 0);
        // A symbol with a message, a pin and a module misread: every
        // overlay has something to draw, and draws it.
        bench.act(Action::ClosePhoto);
        let text = "https://example.com/q?id=7Q2M-XK4P-99AD";
        bench.hold(symbol_of(text));
        let cw = bench.app.derived.layout.data_stream[7];
        let value = bench.app.derived.report.codewords[cw].final_value.expect("the block solved");
        bench.act(Action::PinBits { cw, mask: 0x0F, value });
        for overlay in Overlay::ALL {
            bench.app.view.overlay = overlay;
            bench.step();
            assert!(bench.texts().iter().any(|text| text == overlay.legend()) || overlay.legend().is_empty() || overlay == Overlay::Photo, "{overlay:?}");
        }
        // The pinned bits are tinted as pinned by hand, and no others.
        bench.app.view.overlay = Overlay::Pins;
        bench.step();
        assert!(bench.mesh_colours().contains(&canvas::SELECT.gamma_multiply(0.6)));
        bench.app.view.overlay = Overlay::None;
        // The one segment of the message, picked: its eight bits a
        // character outlined as a character's are.
        assert_eq!(bench.outlined(canvas::PICKED), 0);
        bench.app.view.picked_segment = Some(0);
        bench.step();
        assert_eq!(bench.outlined(canvas::PICKED), 8 * text.len());
        assert_eq!(bench.outlined(canvas::SELECT), 4 + 8, "the mode indicator and the count");
        bench.frame(vec![egui::Event::Key { key: egui::Key::Escape, physical_key: None, pressed: true, repeat: false, modifiers: egui::Modifiers::NONE }]);
        assert_eq!(bench.app.view.picked_segment, None);
    }

    #[test]
    fn each_way_of_doubting_is_switched_by_itself() {
        let mut bench = Bench::new();
        bench.size = egui::vec2(1440.0, 3000.0);
        bench.step();
        let texts = bench.texts();
        for label in [
            "Use module reliability",
            "Tried on every unconfirmed block, best result kept:",
            "Doubt the least reliable modules:",
            "Whole codewords",
            "Single bits",
            "Then, one at a time, until the block is confirmed:",
            "Flip the faintest bits",
            "bits flipped",
            "Belief propagation",
            "rounds",
            "Restore defaults",
        ] {
            assert!(texts.iter().any(|text| text == label), "{label}: {texts:?}");
        }
        // The budgets are shown with their thousands set apart.
        for value in ["65,536", "2,048"] {
            assert!(texts.iter().any(|text| text == value), "{value}: {texts:?}");
        }
        let base = default_decode_options();
        assert_eq!(bench.app.opts, base);
        bench.click(bench.text_at("Flip the faintest bits"), egui::PointerButton::Primary);
        assert_eq!(bench.app.opts, DecodeOptions { flip_bits: false, ..base });
        bench.click(bench.text_at("Whole codewords"), egui::PointerButton::Primary);
        assert_eq!(bench.app.opts, DecodeOptions { flip_bits: false, doubt_codewords: false, ..base });
        for label in ["Use list decoding", "Pin the padding and headers found", "Erase codewords by trial"] {
            assert!(bench.texts().iter().any(|text| text == label), "{label}");
        }
        bench.click(bench.text_at("Pin the padding and headers found"), egui::PointerButton::Primary);
        assert!(!bench.app.opts.derive_pins);
        bench.click(bench.text_at("Use list decoding"), egui::PointerButton::Primary);
        assert!(!bench.app.opts.list_decode);
        // With module reliability off the ways that work from it have
        // nothing to say, and keep what they were set to.
        bench.click(bench.text_at("Use module reliability"), egui::PointerButton::Primary);
        assert!(!bench.app.opts.reliability && !bench.app.opts.flip_bits && bench.app.opts.doubt_bits);
        bench.click(bench.text_at("Single bits"), egui::PointerButton::Primary);
        assert!(bench.app.opts.doubt_bits, "a switch that is greyed out takes no click");
        bench.click(bench.text_at("Belief propagation"), egui::PointerButton::Primary);
        assert!(bench.app.opts.belief_propagation, "a switch that is greyed out takes no click");
        // Restore defaults puts every switch back, and leaves the
        // character set, which is chosen under Result, as it is.
        bench.act(Action::SetCharset(Charset::ShiftJis));
        bench.step();
        let tuned = bench.app.opts;
        assert!(!bench.texts().iter().any(|text| text == "Put back"), "nothing was restored yet");
        bench.click(bench.text_at("Restore defaults"), egui::PointerButton::Primary);
        assert_eq!(bench.app.opts, DecodeOptions { charset: Charset::ShiftJis, ..base });
        // Put back takes the restore back, with the character set as it
        // is now.
        bench.act(Action::SetCharset(Charset::Utf8));
        bench.step();
        bench.click(bench.text_at("Put back"), egui::PointerButton::Primary);
        assert_eq!(bench.app.opts, DecodeOptions { charset: Charset::Utf8, ..tuned });
        assert!(!bench.texts().iter().any(|text| text == "Put back"));
        // And is offered only until the options are changed otherwise.
        bench.click(bench.text_at("Restore defaults"), egui::PointerButton::Primary);
        assert!(bench.texts().iter().any(|text| text == "Put back"));
        bench.click(bench.text_at("Use list decoding"), egui::PointerButton::Primary);
        assert!(!bench.texts().iter().any(|text| text == "Put back"));
    }

    #[test]
    fn the_reach_said_is_charged_as_the_document_is_decoded() {
        // A version 1 grid at level H, a block of 26 codewords with 17 of
        // error correction, painted by hand: a solution the curve finds
        // eleven codewords out, confirmed by twenty-five bits, has the
        // eleven a fallback's must show in the search of the block as
        // read, and more. What it must show is the less of what its
        // solver's count must, and what the block's own count must, the
        // nine of a bit for the choice and no share of the search's
        // solvers ([`judged_verifying_bits`]).
        let (version, level) = (SymbolVersion::Qr(1), EcLevel::H);
        let mut app = App::blank();
        app.doc = Document { format: FormatSel::Manual { ec: level, mask: 0 }, ..Document::blank(version) };
        app.refresh();
        let reach = |app: &App| app.reach(version, level).expect("a level of the version").list;
        let fresh = |app: &App| Reach::of(version, level, &app.options_at(level)).expect("a level of the version").list;
        let painted = reach(&app);
        assert!(painted >= 11, "{painted}");
        // Read from a photo as the pick of a hundred and thirty thousand
        // decodes: a solution must show seventeen bits more, and eleven
        // out no longer has them. The reach was once said of the grid as
        // though painted. (At sixty-five thousand the block's own count
        // still verifies eleven out, to the bit; it was once said not
        // to, held to what the solver's count must show.)
        app.doc.grid_among = 1 << 17;
        assert_eq!((reach(&app), fresh(&app)), (10, 10));
        // The solvers that go by reliability find nothing in a block
        // painted by hand, and switched off they charge a solution nothing
        // less; nor does the message prior, whose searches follow the
        // block as read and take none of its share
        // ([`qr_forensics::decode::pipeline::verifying_bits`]).
        app.opts = DecodeOptions { doubt_codewords: false, doubt_bits: false, flip_bits: false, belief_propagation: false, ..app.opts };
        assert_eq!((reach(&app), fresh(&app)), (10, 10));
        app.opts = DecodeOptions { message_evidence: false, ..app.opts };
        assert_eq!((reach(&app), fresh(&app)), (10, 10));
        // The pick of a quarter as many, a solution is charged two bits
        // less: eleven out is in reach again. The reach was once kept as
        // it was worked out under what was charged before.
        app.doc.grid_among = 1 << 15;
        assert_eq!((reach(&app), fresh(&app)), (11, 11));
    }

    #[test]
    fn the_notes_on_reach_are_of_the_symbol_in_hand() {
        let mut bench = Bench::new();
        bench.size = egui::vec2(1440.0, 3000.0);
        let painted = |bench: &Bench, text: &str| bench.texts().iter().any(|painted| painted == text);
        let starting = |bench: &Bench, start: &str| bench.texts().into_iter().find(|text| text.starts_with(start));
        // Version 2 at level L, a block of 44 codewords with 10 of error
        // correction, 2 of them kept for detecting misreads.
        bench.act(Action::SetFormat(FormatSel::Manual { ec: EcLevel::L, mask: 0 }));
        bench.step();
        // Under the format, what the level in effect corrects, and nothing
        // past that: the pairs erased by trial reach five within the
        // standard's limit, which plain decoding reaches first by spending
        // the two, and the trials never see such a block. It was once said
        // that they reach five. The curve reaches five as plain decoding
        // does, spending the two as well.
        assert!(painted(&bench, "Corrects up to 4 wrong codewords or 8 unknown."), "{:?}", bench.texts());
        assert!(
            painted(
                &bench,
                "No gain at level L: in this symbol's block of 44 codewords, it's guaranteed to correct 5 wrong \
                 codewords, the same as plain Reed-Solomon. Past 4, both use the reserved error-detection codewords, \
                 so neither is confirmed."
            ),
            "{:?}",
            bench.texts()
        );
        assert!(
            painted(
                &bench,
                "In this symbol's block of 44 codewords: tries every pair (946 sets), but these only beat plain \
                 Reed-Solomon by using the reserved error-detection codewords, so they are never confirmed. Adding every \
                 set of four would need a budget of 136,697."
            ),
            "{:?}",
            bench.texts()
        );
        // At level H, with 28 of error correction, list decoding reaches
        // past the code's 14.
        bench.act(Action::SetFormat(FormatSel::Manual { ec: EcLevel::H, mask: 0 }));
        bench.step();
        let note = starting(&bench, "In this symbol's block of 44 codewords: corrects at least ").expect("a note on list decoding");
        assert!(note.ends_with("wrong codewords, vs. 14 for plain Reed-Solomon."), "{note}");
        // A block so small has no use for the most constraints: it uses
        // what its list lets it, and the note under the slider says how
        // many it uses. Its codewords given more alike, it reaches a
        // codeword further than at the default, eighteen to seventeen; a
        // unit at a time, it once reached no further.
        bench.act(Action::SetDecodeOptions(DecodeOptions { list_cost: LIST_COST_MAX, ..bench.app.opts }));
        bench.step();
        assert_eq!(note, "In this symbol's block of 44 codewords: corrects at least 17 wrong codewords, vs. 14 for plain Reed-Solomon.");
        assert_eq!(
            starting(&bench, "In this symbol's block of 44 codewords: corrects at least ").as_deref(),
            Some("In this symbol's block of 44 codewords: corrects at least 18 wrong codewords, vs. 14 for plain Reed-Solomon.")
        );
        let used = starting(&bench, "This symbol's blocks use ").expect("a note on the constraints used");
        assert!(used.contains("; more won't reach further."), "{used}");
    }

    #[test]
    fn a_switch_says_where_it_is_off_what_it_needs() {
        let mut bench = Bench::new();
        bench.size = egui::vec2(1440.0, 3000.0);
        bench.step();
        let painted = |bench: &Bench, text: &str| bench.texts().iter().any(|painted| painted == text);
        let notes = [
            "Solve unknown bits individually is off, so equation-guided flips are skipped.",
            "Solve unknown bits individually is off, so bit-by-bit trials are skipped.",
            "Use module reliability is off, so every module counts as certain.",
            "Use module reliability is off, so the prior only fills in unknown modules.",
        ];
        assert!(notes.iter().all(|note| !painted(&bench, note)), "everything is on");
        bench.click(bench.text_at("Solve unknown bits individually"), egui::PointerButton::Primary);
        assert!(painted(&bench, notes[0]) && painted(&bench, notes[1]));
        assert!(!painted(&bench, notes[2]) && !painted(&bench, notes[3]));
        // Without module reliability, the flips and the propagation are not
        // tried, and their budgets are not offered.
        bench.click(bench.text_at("Use module reliability"), egui::PointerButton::Primary);
        assert!(painted(&bench, notes[2]) && painted(&bench, notes[3]));
        assert!(!painted(&bench, notes[0]), "nothing is flipped at all");
        assert!(!painted(&bench, "bits flipped") && !painted(&bench, "rounds"));
        assert!(bench.app.opts.flip_bits && bench.app.opts.belief_propagation, "what they were set to is kept");
        bench.click(bench.text_at("Use module reliability"), egui::PointerButton::Primary);
        assert!(painted(&bench, "bits flipped") && painted(&bench, "rounds"));
    }

    #[test]
    fn a_block_tried_harder_is_kept_while_it_reads_the_same_under_the_same_options() {
        use qrcode_generator::{Segment, qr};
        let mut bench = Bench::new();
        bench.size = egui::vec2(1440.0, 3000.0);
        bench.act(Action::ClosePhoto);
        // An 11-Q symbol painted by hand with its first block seventeen
        // codewords wrong of fifty: past what list decoding reaches at its
        // usual budget, and within what it reaches at its most. An 8-Q
        // block thirteen wrong of forty once was, its codewords given the
        // budget a unit at a time; given it alike, the usual budget
        // reaches it.
        let text = "it was the best of times, it was the worst of times, it was the age of wisdom, it was the age";
        let symbol = qr::Encoder::new(qr::ErrorCorrection::Quartile)
            .version(qr::Version::new(11).unwrap())
            .boost_error_correction(false)
            .encode_segments(&[Segment::bytes(text)])
            .unwrap();
        let clean = ModuleGrid::from_bools(symbol.width(), symbol.height(), symbol.modules());
        bench.app.doc = Document { grid: clean.clone(), ..Document::blank(SymbolVersion::Qr(11)) };
        bench.app.dirty = true;
        bench.step();
        assert!(bench.app.derived.report.is_trustworthy());
        let layout = bench.app.derived.layout.clone();
        let wrong = |grid: &mut ModuleGrid, cw: CwId, bit: usize| {
            let module = layout.codeword_modules(cw)[bit];
            let (x, y) = (module.x as usize, module.y as usize);
            grid.set(x, y, grid.get(x, y).inverted());
        };
        let mut grid = clean;
        let block = &layout.blocks[0];
        for i in 0..block.ec_len / 2 + 3 {
            wrong(&mut grid, block.codewords[i], i % 8);
        }
        bench.app.doc.grid = grid.clone();
        bench.app.dirty = true;
        bench.step();
        bench.step();
        let report = &bench.app.derived.report;
        assert!(!report.blocks[0].is_verified() && report.blocks[1..].iter().all(BlockReport::is_verified), "{:?}", report.blocks[0].outcome);
        bench.click(bench.text_at("Try harder"), egui::PointerButton::Primary);
        assert_eq!(bench.app.job.as_ref().map(|job| job.kind), Some(JobKind::TryHarder(0)));
        // The search is of the grid: a photo loaded and closed meanwhile
        // leaves it running. Both once called it off, and said nothing.
        let ctx = bench.ctx.clone();
        bench.app.set_photo(&ctx, "another".into(), image::RgbaImage::from_pixel(400, 300, image::Rgba([255; 4])));
        assert_eq!(bench.app.running(), Some(JobKind::TryHarder(0)));
        bench.act(Action::ClosePhoto);
        assert_eq!(bench.app.running(), Some(JobKind::TryHarder(0)));
        bench.settle();
        assert_eq!(bench.notice().text, "Block 1 is solved and confirmed after searching harder.");
        let report = &bench.app.derived.report;
        assert!(report.blocks[0].is_verified() && report.blocks[0].tried_harder, "{:?}", report.blocks[0].outcome);
        assert_eq!(report.text.plain(), text);
        assert!(!bench.texts().iter().any(|text| text == "Try harder"));
        // Its row says it was searched harder, confirmed though it is: it is
        // confirmed only while the search holds. It once said nothing.
        assert!(bench.texts().iter().any(|text| text == "searched harder"));
        // Painting another block leaves the search standing.
        let mut elsewhere = grid.clone();
        wrong(&mut elsewhere, layout.blocks[2].codewords[0], 0);
        bench.app.doc.grid = elsewhere;
        bench.app.dirty = true;
        bench.step();
        assert!(bench.app.derived.report.blocks[0].is_verified() && bench.app.derived.report.text.plain() == text);
        // So does reading the message in another character set, which no
        // search of a block asks: it was once dropped for it.
        let charset = bench.app.opts.charset;
        bench.act(Action::SetCharset(Charset::Latin1));
        bench.step();
        let first = &bench.app.derived.report.blocks[0];
        assert!(first.is_verified() && first.tried_harder && bench.app.derived.report.text.plain() == text);
        bench.act(Action::SetCharset(charset));
        bench.step();
        // Under other options it does not hold, and is offered again; back
        // under these, it holds again.
        let opts = bench.app.opts;
        bench.act(Action::SetDecodeOptions(DecodeOptions { chase_bits: 12, ..opts }));
        bench.step();
        assert!(!bench.app.derived.report.blocks[0].is_verified() && !bench.app.derived.report.blocks[0].tried_harder);
        assert!(bench.texts().iter().any(|text| text == "Try harder"));
        assert!(!bench.texts().iter().any(|text| text == "searched harder"));
        bench.act(Action::SetDecodeOptions(opts));
        bench.step();
        assert!(bench.app.derived.report.blocks[0].is_verified());
        // Once the block reads otherwise, it is no longer searched harder.
        wrong(&mut grid, block.codewords[block.codewords.len() - 1], 0);
        bench.app.doc.grid = grid.clone();
        bench.app.dirty = true;
        bench.step();
        assert!(!bench.app.derived.report.blocks[0].tried_harder);
        assert!(bench.texts().iter().any(|text| text == "Try harder"));
        // Painted while it is searched, it has changed, and is offered
        // again as it is now.
        bench.click(bench.text_at("Try harder"), egui::PointerButton::Primary);
        assert_eq!(bench.app.running(), Some(JobKind::TryHarder(0)));
        wrong(&mut grid, block.codewords[block.codewords.len() - 2], 0);
        bench.app.doc.grid = grid.clone();
        bench.app.dirty = true;
        bench.settle();
        assert_eq!(
            bench.notice().text,
            "Block 1 changed while it was being searched. Press Try harder again to search it as it is now."
        );
        assert!(!bench.app.derived.report.blocks[0].tried_harder);
        // Another block painted meanwhile leaves it as it was.
        bench.click(bench.text_at("Try harder"), egui::PointerButton::Primary);
        assert_eq!(bench.app.running(), Some(JobKind::TryHarder(0)));
        let mut elsewhere = grid.clone();
        wrong(&mut elsewhere, layout.blocks[2].codewords[0], 0);
        bench.app.doc.grid = elsewhere;
        bench.app.dirty = true;
        bench.settle();
        assert_eq!(bench.notice().text, "Searching block 1 harder found nothing better.");
        assert!(bench.app.derived.report.blocks[0].tried_harder);
    }

    #[test]
    fn a_block_the_search_solved_is_said_to_be_solved_however_it_came_to_be_confirmed() {
        use qrcode_generator::{Segment, qr};
        // A block the search solves may be confirmed in the end under the
        // padding pins the decode finds once it reads, with no search of
        // it as it reads under them, and is not marked as searched harder.
        // It was once said to have changed while it was searched.
        let symbol = qr::Encoder::new(qr::ErrorCorrection::High)
            .version(qr::Version::new(1).unwrap())
            .mask(qr::Mask::new(2).unwrap())
            .boost_error_correction(false)
            .encode_segments(&[Segment::bytes("seen")])
            .unwrap();
        let grid = ModuleGrid::from_bools(symbol.width(), symbol.height(), symbol.modules());
        let layout = SymbolLayout::new(Arc::new(FunctionMap::new(SymbolVersion::Qr(1))), EcLevel::H).unwrap();
        let mut block = decode(&grid, &layout, 2, &Pins::new(), &default_decode_options()).blocks[0];
        assert!(block.is_verified() && !block.tried_harder);
        let failed = BlockOutcome::Failed(RsFailure::BudgetExceeded);
        assert_eq!(try_harder_notice(1, false, Some(failed), &block), "Block 1 is solved and confirmed after searching harder.");
        // What is offered again is said only where it is.
        assert_eq!(try_harder_notice(1, true, Some(failed), &block), "Block 1 changed while it was being searched.");
        // Neither confirmed nor searched as it reads now, though its own
        // modules and pins are as they were: other blocks moved its
        // padding pins.
        block.outcome = failed;
        assert_eq!(
            try_harder_notice(1, false, Some(failed), &block),
            "Other blocks changed while block 1 was being searched, so it is now decoded under different padding and header \
             pins. Press Try harder again to search it as it is now."
        );
        block.tried_harder = true;
        assert_eq!(try_harder_notice(1, false, Some(failed), &block), "Searching block 1 harder found nothing better.");
    }

    #[test]
    fn pins_made_under_another_level_are_a_click_from_applying() {
        let mut bench = Bench::new();
        bench.size = egui::vec2(1440.0, 3000.0);
        bench.act(Action::SetFormat(FormatSel::Manual { ec: EcLevel::M, mask: 3 }));
        bench.act(Action::Pin(0, 0x40));
        assert_eq!(bench.app.doc.pins_ec, Some(EcLevel::M));
        bench.act(Action::SetFormat(FormatSel::Manual { ec: EcLevel::Q, mask: 3 }));
        bench.step();
        assert!(bench.texts().iter().any(|text| text.starts_with("Pins on 1 codeword were made under level M")));
        bench.click(bench.text_at("Use level M"), egui::PointerButton::Primary);
        assert_eq!(bench.app.doc.format, FormatSel::Manual { ec: EcLevel::M, mask: 3 });
        assert_eq!(bench.app.derived.ec, EcLevel::M);
        assert!(!bench.texts().iter().any(|text| text == "Use level M"), "the pins apply, and the warning is gone");
        bench.act(Action::Undo);
        assert_eq!(bench.app.doc.format, FormatSel::Manual { ec: EcLevel::Q, mask: 3 });
    }

    #[test]
    fn a_module_painted_as_a_guess_is_doubted_and_one_painted_certain_is_not() {
        let mut bench = Bench::new();
        bench.size = egui::vec2(1440.0, 3000.0);
        bench.act(Action::ClosePhoto);
        // Without the trials of erasures and list decoding, which reach a
        // wrong codeword or two past the code by themselves: what pins
        // and doubting do is the point here. List decoding once reached
        // no further than the code here, its codewords given their
        // multiplicities a unit at a time.
        bench.app.opts.erase_by_trial = false;
        bench.app.opts.list_decode = false;
        let text = "https://example.com/q?id=7Q2M-XK4P-99AD";
        let clean = symbol_of(text);
        bench.hold(clean.clone());
        let layout = bench.app.derived.layout.clone();
        // Eight codewords wrong, one more than the block corrects, each
        // by a module painted as certain: nothing is there to doubt.
        bench.hold(with_wrong(&clean, &layout, 2..=9));
        assert_eq!(bench.app.derived.report.blocks_solved(), 0);
        assert!(!bench.app.doc.grid.has_reliability());
        bench.app.view.stage = Stage::Edit;
        bench.step();
        let texts = bench.texts();
        assert!(texts.iter().any(|text| text == "Takes effect once modules are sampled from a photo, or painted below certain."), "{texts:?}");
        let wrong: Vec<(usize, usize)> = (2..=9)
            .map(|i| layout.codeword_modules(layout.data_stream[i])[3])
            .map(|module| (module.x as usize, module.y as usize))
            .collect();
        // Each painted over in the colour it has, the wrong one still,
        // with either button.
        let repaint = |bench: &mut Bench| {
            let cell = bench.cells();
            for (n, &(x, y)) in wrong.iter().enumerate() {
                let module = bench.app.doc.grid.get(x, y);
                let (ink, button) = if n % 2 == 0 {
                    (module, egui::PointerButton::Primary)
                } else {
                    (module.inverted(), egui::PointerButton::Secondary)
                };
                bench.app.view.ink = Ink::Module(ink);
                bench.click(cell(x, y), button);
                assert_eq!(bench.app.doc.grid.get(x, y), module, "({x}, {y})");
            }
            bench.step();
        };
        // As guesses, they are the first modules doubted, and as unknown
        // the eight codewords are well within what the block takes.
        bench.app.view.paint_reliability = 0;
        bench.step();
        assert!(bench.texts().iter().any(|text| text.contains("a guess")), "the slider says what it paints");
        repaint(&mut bench);
        assert!(wrong.iter().all(|&(x, y)| bench.app.doc.grid.reliability(x, y) == 0));
        assert!(bench.app.derived.report.is_trustworthy());
        assert_eq!(bench.app.derived.report.text.plain(), text);
        let texts = bench.texts();
        assert!(!texts.iter().any(|text| text.starts_with("Takes effect once")), "{texts:?}");
        // Unknown is no guess: it has nothing to be unsure of.
        let (x, y) = wrong[0];
        let cell = bench.cells();
        bench.app.view.ink = Ink::Module(Module::Unknown);
        bench.click(cell(x, y), egui::PointerButton::Primary);
        assert_eq!((bench.app.doc.grid.get(x, y), bench.app.doc.grid.reliability(x, y)), (Module::Unknown, CERTAIN));
        bench.act(Action::Undo);
        assert_eq!(bench.app.doc.grid.reliability(x, y), 0, "one undo takes the stroke back, and the guess is a guess again");
        // Painted over once more at certain they are certain, and the
        // block is as far from solving as it was.
        bench.app.view.paint_reliability = CERTAIN;
        bench.step();
        assert!(bench.texts().iter().any(|text| text.contains("certain")), "the slider says what it paints");
        repaint(&mut bench);
        assert!(wrong.iter().all(|&(x, y)| bench.app.doc.grid.reliability(x, y) == CERTAIN));
        assert_eq!(bench.app.derived.report.blocks_solved(), 0);
    }

    #[test]
    fn unknown_painted_on_a_module_unknown_already_is_no_edit() {
        let mut bench = Bench::new();
        bench.size = egui::vec2(1440.0, 3000.0);
        bench.act(Action::ClosePhoto);
        let clean = symbol_of("https://example.com/q?id=7Q2M-XK4P-99AD");
        bench.hold(clean.clone());
        let layout = bench.app.derived.layout.clone();
        let at = |i: usize| {
            let module = layout.codeword_modules(layout.data_stream[i])[3];
            (module.x as usize, module.y as usize)
        };
        // A module sampling left unknown, inside its unsure band: it has
        // the reliability it was read at all the same.
        let (x, y) = at(2);
        let mut grid = clean;
        grid.set_sampled(x, y, Module::Unknown, 40);
        bench.hold(grid);
        bench.app.view.stage = Stage::Edit;
        bench.step();
        // Another module painted and the stroke undone, so that there is
        // a state to redo and none to undo.
        let cell = bench.cells();
        let (ox, oy) = at(5);
        bench.app.view.ink = Ink::Module(bench.app.doc.grid.get(ox, oy).inverted());
        bench.click(cell(ox, oy), egui::PointerButton::Primary);
        bench.act(Action::Undo);
        bench.step();
        let before = bench.app.doc.clone();
        assert!(bench.app.history.can_redo(&before) && !bench.app.history.can_undo(&before));
        // Unknown on the unknown module, certain and as a guess: nothing
        // of it is changed, and the press costs neither.
        for guess in [false, true] {
            bench.app.view.paint_reliability = if guess { 0 } else { CERTAIN };
            bench.app.view.ink = Ink::Module(Module::Unknown);
            bench.click(cell(x, y), egui::PointerButton::Primary);
            assert!(bench.app.doc == before, "guess {guess}");
            assert_eq!(bench.app.doc.grid.reliability(x, y), 40, "guess {guess}");
            assert!(bench.app.history.can_redo(&bench.app.doc), "guess {guess}: the state undone is still there to redo");
            assert!(!bench.app.history.can_undo(&bench.app.doc), "guess {guess}: the press left no step to undo");
        }
    }

    #[test]
    fn a_stroke_that_goes_on_after_an_undo_is_a_step_of_its_own() {
        let mut bench = Bench::new();
        bench.act(Action::ClosePhoto);
        bench.app.view.stage = Stage::Edit;
        bench.app.view.ink = Ink::Module(Module::Dark);
        bench.step();
        bench.step();
        let cell = bench.cells();
        let dark = |bench: &Bench| (8..17).filter(|&x| bench.app.doc.grid.get(x, 10) == Module::Dark).collect::<Vec<usize>>();
        // A stroke along row 10, from module 10 to 12, with the button
        // still held.
        bench.frame(vec![egui::Event::PointerMoved(cell(10, 10))]);
        bench.button(cell(10, 10), egui::PointerButton::Primary, true);
        bench.frame(vec![egui::Event::PointerMoved(cell(12, 10))]);
        assert_eq!(dark(&bench), [10, 11, 12]);
        // Undo, by its shortcut, while the button is held: the stroke is
        // taken back whole, and ends there. What the pointer paints from
        // here on is a stroke of its own, with an undo point of its own.
        // Painted on as part of the old stroke, whose point the undo had
        // taken, it could neither be undone nor survive a redo.
        bench.act(Action::Undo);
        assert!(bench.app.view.stroke.is_none());
        assert_eq!(dark(&bench), Vec::<usize>::new());
        bench.frame(vec![egui::Event::PointerMoved(cell(14, 10))]);
        assert_eq!(dark(&bench), [14]);
        bench.button(cell(14, 10), egui::PointerButton::Primary, false);
        bench.step();
        assert!(bench.app.history.can_undo(&bench.app.doc));
        bench.act(Action::Undo);
        assert_eq!(dark(&bench), Vec::<usize>::new());
        bench.act(Action::Redo);
        assert_eq!(dark(&bench), [14]);
    }
}
