//! Reading a symbol out of a photo, outline to text: the detection, the
//! refinement of its corners and anchors, the sampling, and the decode,
//! with the decode deciding what the fit of the fixed patterns cannot.
//!
//! The fixed patterns fit well enough on most photos that the first
//! placement the detector ranks decodes outright, and that is the end of
//! it. When it does not, the decode is the one judge the detector lacks:
//! a wrong version, a grid a fraction of a module off, or a sampling that
//! reads a smeared print as grey all leave blocks unsolved, where the
//! right reading solves them with redundancy to spare. So, as ZXing's and
//! zxing-cpp's readers try the alternatives their detectors cannot choose
//! between and keep what decodes, the retries here are judged by the
//! decode and the best is kept:
//!
//! - the runners-up among the detector's placements, which lets the
//!   decode arbitrate the version where the fit misjudged it;
//! - the other threshold rules, of the three a module may be read by
//!   ([`super::sample::Rule`]): every module against one split of the
//!   symbol where each against its neighbourhood was asked for, or the
//!   reverse, and then every module against a threshold that lies as a
//!   surface across the symbol. The neighbourhood follows shadows and
//!   glare, but on an evenly lit symbol blurred until its modules run
//!   together it reads more of them as unsure than one split does, which
//!   costs the blocks their erasures: over BoofCV's benchmark photos the
//!   split read 35 of the 138 symbols the neighbourhood did not, and
//!   lost none as a retry. BoofCV itself reads every module against a
//!   surface, the thresholds of its position patterns carried across the
//!   symbol, which follows light that fails toward one side as no split
//!   does and reads a blurred module against none of its neighbours.
//!   Of the thirty labelled symbols of that benchmark which were found
//!   and not read under the first two rules, each read again through
//!   its own outline, the surface read one, on a crumpled receipt, and
//!   it is tried last;
//! - the other sampling kernels, the module's centre alone where the
//!   area average was asked for, or the vote of five points about it,
//!   since a smeared print or a symbol drawn as dots reads at the centre
//!   where the average takes in the neighbours, a noisy one the other way
//!   about, and a specked one by the vote;
//! - the placement taken in mirror order, as from the back of the paper,
//!   when its format bits read as a word and it does not decode: ZXing
//!   and wechat_qrcode read a symbol mirrored whenever it has failed
//!   plain, and a mirrored symbol's format word, a bit or two misread,
//!   passes for a plain one's more often than not;
//! - the placement without the anchor that lies farthest from where the
//!   others put it, when not one block has verified: see below;
//! - the photo turned a quarter at a time, as zxing-cpp's `tryRotate`
//!   does: the fits walk the photo's axes, and a symbol that sits a
//!   fraction of a module off at one turn reads at another.
//! - the photo about the outline enlarged and sharpened
//!   ([`super::enhance`]), and the whole read made again in that copy,
//!   from the scan for finder patterns to the decode, when the symbol's
//!   modules are small enough for it to matter: wechat_qrcode reads a
//!   small crop at its own size, then at twice it through a network
//!   that sharpens as it enlarges, and keeps what decodes. A symbol
//!   blurred by half a module shows the scan no finder pattern and the
//!   sampler modules as grey as their neighbours, and neither a kernel
//!   nor a threshold rule nor a turn mends that; the copy does, and
//!   what was read through it says so ([`ReadResult::enhance`]), so
//!   that it can be read the same again.
//!
//! Before any of those, a decode that solved some blocks but not all is
//! turned back on the placement. Every module of a verified block is now
//! known, and with the fixed patterns they make a template of hundreds
//! of modules where the fixed patterns alone gave tens: the corners are
//! refined against it, each anchor against the known modules around it,
//! and the symbol is sampled and decoded again. A grid a fraction of a
//! module off in one quarter, which costs that quarter's block at a few
//! pixels a module, is pulled straight by the three that read. An
//! anchor the known modules contradict, fitted to a stain or to data
//! that happened to look like its pattern, is dropped: the modules about
//! it are read through the map with it and through the map without it,
//! and when they read better without it the anchor misleads. BoofCV,
//! when a symbol's bits fail to correct, refits its grid-to-pixel
//! transform without the located pattern of largest residual and reads
//! the bits again, up to six reads in all; the residual against a plane
//! only says which pattern lies farthest from one, which on curled paper
//! the right patterns do, where the decoded modules say outright which
//! anchor reads wrong. None of the readers this crate follows refits to a
//! partial decode; they have none to learn from.
//!
//! A decode that solved nothing has nothing to say of the anchors, and
//! an anchor fitted to a logo printed over the middle of the symbol,
//! where an alignment pattern should be, a module off its place, bends
//! the grid about it until no block solves. There BoofCV's way is all
//! there is: the anchor farthest from where the corners and the other
//! anchors put it is dropped, and the symbol read again, a few times at
//! the most. It is tried after the plain reads and judged by the decode
//! like them, so that an anchor curled paper has carried off its plane
//! costs a read that comes to nothing, and no more.
//!
//! Each retry is dearer than the last and runs only while the decode is
//! not yet trustworthy, so a photo that reads at once costs one pass.
//! Once the first try of a kind has failed, the tries after it are made
//! side by side on the cores to spare, as [`crate::work`] describes, and
//! taken in the order given here: what is read is what one core read.
//!
//! One retry costs next to nothing and is made of every decode that is
//! no read: the padding. A short message in a large symbol leaves most
//! of the data codewords as filler whose values the standard fixes, and
//! damage that falls on them is damage to what is already known. So when
//! the padding can be located, by the segment headers or by its own
//! pattern, its codewords are pinned at their values and the blocks
//! solved again ([`crate::decode::padding`]); and when every block then
//! verifies and the headers bear the padding out, that is the read, and
//! it says that it was made so. A strip of tape across a third of a
//! version 10 symbol that holds a short address leaves no block that
//! error correction alone can solve, and all five solve this way.

use std::sync::Arc;

use super::detect::{
    Anchor, AnchorKind, Climb, Detection, Retry, detect_candidates_by, outline_around, refine_anchors, seen_from_behind,
    turns_between,
};
use super::enhance::{Enhance, Enhanced};
use super::projection::{ControlPoint, Project, Projection};
use super::refine::{MOVES, Scorer, Shifted, Template, Warp, descend_by, fixed_threshold, refine_corners_holding};
use super::sample::{Kernel, Rule, SampleParams, classify, sample_modules_with};
use super::{Luma, rotate_outline_cw, unrotate_point_cw};
use crate::decode::codewords::Pins;
use crate::decode::padding::recover_padding_after;
use crate::decode::pipeline::{DecodeOptions, DecodeReport, GridAnalysis, analyze, brute_force, decode};
use crate::decode::repair::known_modules;
use crate::symbol::function_map::FunctionMap;
use crate::symbol::{Family, SymbolVersion};
use crate::symbol::grid::ModuleGrid;
use crate::symbol::layout::SymbolLayout;
use crate::work;

/// What one read of a photo made of it.
#[derive(Clone, Debug)]
pub struct ReadResult {
    /// The symbol as placed, its corners and anchors refined, in the
    /// photo's own pixels whatever turn it was read at.
    pub detection: Detection,
    /// (matching, fixed) over the version's fixed modules at the refined
    /// corners.
    pub agreement: (usize, usize),
    /// The sampling kernel the grid was read with: the one asked for, or
    /// another when only that decoded.
    pub kernel: Kernel,
    /// The threshold rule the grid was read with, each module against its
    /// neighbourhood, or every module against one split of the symbol or
    /// against a threshold that lies as a surface across it: the rule
    /// asked for, or another when only that decoded.
    pub rule: Rule,
    pub grid: ModuleGrid,
    pub analysis: GridAnalysis,
    pub report: DecodeReport,
    /// The pins the report was decoded under: those of the padding, its
    /// terminator and any header it settled, when the grid reads only
    /// with its padding taken as known, and none otherwise.
    pub pins: Pins,
    /// The grid was read as the other QR Code model, of this version,
    /// because its format bits said so and that read ranked better.
    pub reinterpreted: Option<SymbolVersion>,
    /// What was done to the photo about the symbol for this read: what
    /// was asked for, or the enlargement and sharpening only which
    /// decoded. The detection is in the photo's own pixels either way,
    /// and sampled through the same ([`super::sample::sample_modules_as`])
    /// gives this grid again.
    pub enhance: Enhance,
}

impl ReadResult {
    /// Every block solved with confirmation and the stream parsing: a read
    /// that needs no second look.
    pub fn is_trustworthy(&self) -> bool {
        self.report.is_trustworthy()
    }

    /// The message as text.
    pub fn text(&self) -> String {
        self.report.text.plain()
    }

    /// The version the grid was read under.
    pub fn version(&self) -> SymbolVersion {
        self.reinterpreted.unwrap_or(self.detection.version)
    }

    /// The read stands only with the padding pinned: error correction
    /// alone did not solve every block.
    pub fn padding_pinned(&self) -> bool {
        !self.pins.is_empty()
    }

    fn outranks(&self, other: &ReadResult) -> bool {
        self.report.score > other.report.score
    }
}

/// Read the symbol inside `region`, an outline drawn loosely around it,
/// as [`super::detect::detect_outline`] takes one, and decode it: the
/// detection refined and sampled with `params`, then decoded under the
/// format its bits settle or, failing that, the best attempt of a
/// brute-force search.
/// When the decode is not trustworthy the retries the module describes
/// are tried in turn, dearest last, and the best read is returned: the
/// first that is trustworthy, or else the one whose attempt ranks
/// highest. `None` when no symbol can be found at all.
///
/// The photo is read as `params` ask for it, enlarged or sharpened or as
/// it is ([`SampleParams::enhance`]). When that does not decode, and the
/// symbol's modules are small enough for it to matter or no symbol was
/// found at all, the whole read is made again through the photo
/// enlarged and sharpened, lightly and then strongly ([`Enhance::rungs`]).
pub fn read_outline(
    image: &Luma,
    version: SymbolVersion,
    region: [[f64; 2]; 4],
    params: &SampleParams,
    opts: &DecodeOptions,
) -> Option<ReadResult> {
    let asked = params.enhance;
    let mut best = read_outline_through(image, version, region, params, opts, asked);
    if best.as_ref().is_some_and(ReadResult::is_trustworthy) || work::cancelled() {
        return best;
    }
    // The photo as it is, when it was asked for otherwise and did not
    // read so; then the rungs, side by side on the cores to spare and
    // taken in their order.
    let mut rungs: Vec<Enhance> = Vec::new();
    if !asked.is_none() {
        rungs.push(Enhance::NONE);
    }
    if worth_enhancing(image, best.as_ref()) {
        rungs.extend(Enhance::rungs(region));
    }
    rungs.retain(|rung| *rung != asked);
    work::in_turn(
        &rungs,
        |&rung| if work::cancelled() { None } else { read_outline_through(image, version, region, params, opts, rung) },
        |result| result.is_some_and(|result| consider(&mut best, result)),
    );
    best
}

/// Modules of fewer pixels than this, in the pixels the photo had
/// before it was enlarged, are small enough for a read that did not
/// decode to be made again through the photo enlarged and sharpened.
/// The masks reach
/// a pixel or two, which is what a lens takes from modules of this size
/// and makes no odds to larger ones: a symbol of large modules that
/// does not decode is damaged, which is what this crate is for, and
/// reading it twice more would treble the wait for the same answer.
const ENHANCE_BELOW_PX: f64 = 6.0;

/// Whether a read that did not decode is worth making again through the
/// photo enlarged and sharpened: nothing was found at all, which is what
/// a blur makes of finder patterns, or what was found has modules under
/// [`ENHANCE_BELOW_PX`].
fn worth_enhancing(image: &Luma, best: Option<&ReadResult>) -> bool {
    let Some(read) = best else { return true };
    let (c, version) = (read.detection.corners, read.detection.version);
    let across = (distance(c[0], c[1]) + distance(c[3], c[2])) / 2.0 / version.width().max(1) as f64;
    let down = (distance(c[0], c[3]) + distance(c[1], c[2])) / 2.0 / version.height().max(1) as f64;
    // A photo enlarged has its modules enlarged with it.
    let times = image.unenlarged.as_ref().map_or(1, |own| image.width / own.width.max(1)).max(1) as f64;
    across.min(down) / times < ENHANCE_BELOW_PX
}

/// [`read_outline`] through the photo as `enhance` makes it, and no
/// other way: as it is, or in a copy of it about the outline, enlarged
/// and sharpened, with what is read there brought back to the photo's
/// own pixels.
fn read_outline_through(
    image: &Luma,
    version: SymbolVersion,
    region: [[f64; 2]; 4],
    params: &SampleParams,
    opts: &DecodeOptions,
    enhance: Enhance,
) -> Option<ReadResult> {
    // Whatever is read, it is read off the pixels handed on: the copy's,
    // which are enlarged and sharpened already, or the photo's.
    let plain = SampleParams { enhance: Enhance::NONE, ..*params };
    if enhance.is_none() {
        return read_outline_plain(image, version, region, &plain, opts, false);
    }
    let view = Enhanced::around(image, region, enhance)?;
    let mut result = read_outline_plain(&view.luma, version, region.map(|p| view.to_view(p)), &plain, opts, true)?;
    bring_back(&mut result, &view, enhance);
    Some(result)
}

/// A read made in an enlarged and sharpened copy of the photo, with its
/// detection brought back to the photo's own pixels and the copy noted.
fn bring_back(result: &mut ReadResult, view: &Enhanced, enhance: Enhance) {
    let found = &mut result.detection;
    found.corners = found.corners.map(|c| view.to_photo(c));
    for anchor in &mut found.anchors {
        anchor.center = view.to_photo(anchor.center);
    }
    found.retry = Some(Retry::Enhanced(enhance));
    result.enhance = enhance;
}

/// [`read_outline`] of the pixels given, as they are. `copy` says that
/// they are a copy of the photo made for this read, enlarged and
/// sharpened.
fn read_outline_plain(
    image: &Luma,
    version: SymbolVersion,
    region: [[f64; 2]; 4],
    params: &SampleParams,
    opts: &DecodeOptions,
    copy: bool,
) -> Option<ReadResult> {
    let variants = variants_of(params);
    let mut best: Option<ReadResult> = None;
    // In a copy made for this read the detector climbs none of its
    // ladder: over sixty photographs that read only through a copy, or
    // not at all, the ladder climbed inside the copies read not one
    // more. The copy's other three turns, below, read three.
    let climb = if copy { Climb::Nothing } else { Climb::Unsharpened };
    // The detector's placements, best fit first, each with every way of
    // sampling it: the reads of each, up to the one that settles it.
    let candidates = detect_candidates_by(image, version, region, params, climb);
    let reads_of = |found: &Detection| {
        let mut reads = Vec::new();
        if work::cancelled() {
            return reads;
        }
        if let Some(prepared) = prepare(image, found.clone(), params) {
            read_placement(image, &prepared, &variants, params, opts, &mut |result| {
                let settled = result.is_trustworthy();
                reads.push(result);
                settled
            });
        }
        reads
    };
    if work::first_then_in_turn(&candidates, reads_of, |reads| reads.into_iter().any(|result| consider(&mut best, result))) {
        return best;
    }
    // Nothing found in the photo as it is: the detector's own ladder has
    // already been through the transformed copies, and a turn would find
    // no more.
    best.as_ref()?;
    if work::cancelled() {
        return best;
    }
    // The photo turned a quarter at a time, read in full each way and the
    // result brought back to the photo's own pixels.
    let reads_turned = |&quarters: &u8| {
        let mut reads = Vec::new();
        if work::cancelled() {
            return reads;
        }
        let turned = image.rotated_cw(quarters);
        let turned_region = rotate_outline_cw(region, quarters, image.width, image.height);
        let Some(found) = detect_candidates_by(&turned, version, turned_region, params, climb).into_iter().next() else { return reads };
        let Some(prepared) = prepare(&turned, found, params) else { return reads };
        work::first_then_in_turn(
            &variants,
            |&(kernel, rule)| if work::cancelled() { None } else { read_prepared(&turned, &prepared, kernel, rule, params, opts) },
            |result| {
                let Some(mut result) = result else { return false };
                unturn(&mut result.detection, quarters, image.width, image.height, region);
                let settled = result.is_trustworthy();
                reads.push(result);
                settled
            },
        );
        reads
    };
    // Side by side, as far as the photo's size leaves room for turned
    // copies of it.
    work::in_turn_by(image.copies_at_once(), &[1, 2, 3], reads_turned, |reads| reads.into_iter().any(|result| consider(&mut best, result)));
    best
}

/// The ways of sampling a placement a read tries, in order: the kernel
/// and threshold rule asked for, then the other rules with that kernel
/// ([`SampleParams::rules`]), then each other kernel under every rule.
/// The rule comes before the kernels because it decides more often: see
/// the module's account.
fn variants_of(params: &SampleParams) -> Vec<(Kernel, Rule)> {
    params.kernel.retries().flat_map(|kernel| params.rules().map(move |rule| (kernel, rule))).collect()
}

/// The margin, as a share of its size, of the outline a symbol the
/// search placed is read through again when it does not decode as placed:
/// wechat_qrcode crops a tenth past the box its detector gave before it
/// reads one.
const AROUND_FOUND: f64 = 0.1;

/// Read a symbol the detector has already placed, as [`read_outline`]
/// reads one it places itself: the placement refined, sampled every way
/// in turn, kernel by kernel under every threshold rule, decoded, and
/// refitted to what decodes. `None` when the placement cannot be sampled
/// and nothing is found where it lies.
///
/// A placement that does not decode is no more than an outline drawn
/// round a symbol, and is read again as one, a little wider than it was
/// placed: a search of the whole photo keeps one placement of each symbol
/// it proposes, where a read of an outline has the detector's runners-up
/// to try, the enlargement a symbol of small modules is found again
/// through, and the photo's other three turns. Without that a symbol that
/// read at once in an outline drawn by hand was found in the photo
/// around it and left unread; wechat_qrcode gives every box its detector
/// finds the same reader an image of one symbol gets, padded.
pub fn read_detection(image: &Luma, found: Detection, params: &SampleParams, opts: &DecodeOptions) -> Option<ReadResult> {
    let (version, around) = (found.version, outline_around(found.corners, AROUND_FOUND));
    // A symbol the search found only in the photo sharpened or enlarged
    // is read as placed through that copy, where its modules read as the
    // search saw them, unless the photo was asked for some other way.
    let through = match found.retry {
        Some(Retry::Enhanced(enhance)) if params.enhance.is_none() => enhance,
        _ => params.enhance,
    };
    let mut best: Option<ReadResult> = None;
    if read_found_through(image, found, around, params, opts, through, &mut best) || work::cancelled() {
        return best;
    }
    if let Some(again) = read_outline(image, version, around, params, opts)
        && best.as_ref().is_none_or(|held| again.outranks(held))
    {
        best = Some(again);
    }
    best
}

/// A symbol as the detector placed it, read through the photo as
/// `enhance` makes it about `around`, every way a placement is read
/// ([`read_placement`]), the best read kept in `best`; true when a read
/// settles it.
fn read_found_through(
    image: &Luma,
    mut found: Detection,
    around: [[f64; 2]; 4],
    params: &SampleParams,
    opts: &DecodeOptions,
    enhance: Enhance,
    best: &mut Option<ReadResult>,
) -> bool {
    let plain = SampleParams { enhance: Enhance::NONE, ..*params };
    let variants = variants_of(&plain);
    if enhance.is_none() {
        return prepare(image, found, &plain)
            .is_some_and(|prepared| read_placement(image, &prepared, &variants, &plain, opts, &mut |result| consider(best, result)));
    }
    let Some(view) = Enhanced::around(image, around, enhance) else { return false };
    found.corners = found.corners.map(|c| view.to_view(c));
    for anchor in &mut found.anchors {
        anchor.center = view.to_view(anchor.center);
    }
    prepare(&view.luma, found, &plain).is_some_and(|prepared| {
        read_placement(&view.luma, &prepared, &variants, &plain, opts, &mut |mut result| {
            bring_back(&mut result, &view, enhance);
            consider(best, result)
        })
    })
}

/// Anchors dropped at most from one placement that solves no block, each
/// followed by a read: BoofCV reads a symbol that fails up to six times,
/// the first with every pattern it located and each of the five after
/// without the pattern then farthest from its fit, while one lies more
/// than two pixels off; here a read is dearer and an anchor is dropped
/// only when it stands apart.
const DOUBT_LIMIT: usize = 3;
/// How far, in modules, an anchor must lie from where the corners and
/// the other anchors put its pattern to be doubted when nothing decodes.
/// The detector drops an anchor its neighbours cannot explain, and allows
/// more the farther the nearest of them lies, since on curled paper the
/// prediction drifts with distance: the alignment pattern in the middle
/// of a version 10 symbol is twenty-two modules from the next anchor and
/// was allowed 1.16, and the logo printed over it was fitted 1.08 from
/// where the pattern belonged. Here the decode has already failed, and
/// what is asked is only which anchor to try the symbol without.
const DOUBT_MODULES: f64 = 0.35;

/// Read one placement every way a read tries, handing each read to
/// `keep`, which says whether it settles the search; true when one did.
/// The kernels and threshold rules in `variants` first; then, for a
/// square symbol whose format bits read as a word, which the detector
/// therefore never tried the other way round, the placement in mirror
/// order, kept when it verifies more blocks than any read so far; then,
/// when no read has verified a block, the placement with its most
/// doubtful anchors dropped, one at a time. The later two are read under
/// the kernel asked for alone, with every threshold rule: what they
/// settle does not hang on the kernel.
fn read_placement(
    image: &Luma,
    prepared: &Prepared,
    variants: &[(Kernel, Rule)],
    params: &SampleParams,
    opts: &DecodeOptions,
    keep: &mut dyn FnMut(ReadResult) -> bool,
) -> bool {
    let mut verified = 0;
    let settled = work::first_then_in_turn(
        variants,
        |&(kernel, rule)| if work::cancelled() { None } else { read_prepared(image, prepared, kernel, rule, params, opts) },
        |result| {
            result.is_some_and(|result| {
                verified = verified.max(result.report.blocks_verified());
                keep(result)
            })
        },
    );
    if settled {
        return true;
    }
    if work::cancelled() {
        return false;
    }
    // The variants lead with the kernel asked for under each rule.
    let few = &variants[..variants.len().min(params.rules().count())];
    let found = &prepared.detection;
    if found.version.family() != Family::Rmqr
        && found.info.format.valid == Some(true)
        && let Some(detection) = seen_from_behind(image, found, params)
    {
        let behind = Prepared { detection, agreement: prepared.agreement };
        for &(kernel, rule) in few {
            if let Some(result) = read_prepared(image, &behind, kernel, rule, params, opts)
                && result.report.blocks_verified() > verified
            {
                verified = result.report.blocks_verified();
                if keep(result) {
                    return true;
                }
            }
        }
    }
    let mut doubted = Prepared { detection: found.clone(), agreement: prepared.agreement };
    for _ in 0..DOUBT_LIMIT {
        // A block that verified knows which anchors mislead, and the
        // refit to it has already dropped them.
        if verified > 0 || work::cancelled() {
            break;
        }
        let Some(k) = doubtful_anchor(&doubted.detection) else { break };
        doubted.detection.anchors.remove(k);
        for &(kernel, rule) in few {
            if let Some(result) = read_prepared(image, &doubted, kernel, rule, params, opts) {
                verified = verified.max(result.report.blocks_verified());
                if keep(result) {
                    return true;
                }
            }
        }
    }
    false
}

/// The anchor, no finder's, that lies farthest from where the corners
/// and every other anchor put its pattern, when that is
/// [`DOUBT_MODULES`] or more: the one most likely fitted to something
/// that is not its pattern.
fn doubtful_anchor(found: &Detection) -> Option<usize> {
    let (w, h) = (found.version.width(), found.version.height());
    let controls: Vec<ControlPoint> = found.anchors.iter().map(Anchor::control_point).collect();
    let mut worst: Option<(usize, f64)> = None;
    for k in 0..controls.len() {
        if found.anchors[k].kind == AnchorKind::Finder {
            continue;
        }
        let others: Vec<ControlPoint> = controls.iter().enumerate().filter(|(i, _)| *i != k).map(|(_, c)| *c).collect();
        let Some(without) = Projection::new(w, h, found.corners, &others) else { continue };
        let [x, y] = found.anchors[k].at;
        let expected = without.apply(x, y);
        let module = distance(expected, without.apply(x + 1.0, y)).max(distance(expected, without.apply(x, y + 1.0))).max(1e-9);
        let off = distance(expected, found.anchors[k].center) / module;
        if off >= DOUBT_MODULES && worst.is_none_or(|(_, o)| off > o) {
            worst = Some((k, off));
        }
    }
    worst.map(|(k, _)| k)
}

/// The decoder's verdict on a read: it is kept when it is the best so
/// far, and the search stops, true, once a read is beyond doubt.
fn consider(best: &mut Option<ReadResult>, result: ReadResult) -> bool {
    let settled = result.is_trustworthy();
    if best.as_ref().is_none_or(|b| result.outranks(b)) {
        *best = Some(result);
    }
    settled
}

/// A detection with its corners and anchors refined against the photo,
/// ready to sample.
struct Prepared {
    detection: Detection,
    agreement: (usize, usize),
}

/// Refine a detection's corners, with the anchors held and the corners a
/// finder pattern measured kept, and then every anchor against its own
/// pattern. `None` when the corners form no usable outline.
fn prepare(image: &Luma, mut found: Detection, params: &SampleParams) -> Option<Prepared> {
    let func = FunctionMap::new(found.version);
    let controls: Vec<ControlPoint> = found.anchors.iter().map(|a| a.control_point()).collect();
    let refined = refine_corners_holding(image, &func, found.corners, &controls, &found.held_corners(), params);
    let agreement = refined.as_ref().map_or(found.agreement, |r| r.after);
    found.corners = refined.map_or(found.corners, |r| r.corners);
    found.anchors = refine_anchors(image, &func, found.corners, &found.anchors, params);
    found.projection()?;
    Some(Prepared { detection: found, agreement })
}

/// Sample a prepared detection with `kernel`, classify it under the
/// threshold rule `rule`, and decode what it reads.
fn read_prepared(
    image: &Luma,
    prepared: &Prepared,
    kernel: Kernel,
    rule: Rule,
    params: &SampleParams,
    opts: &DecodeOptions,
) -> Option<ReadResult> {
    let found = &prepared.detection;
    let func = Arc::new(FunctionMap::new(found.version));
    let projection = found.projection()?;
    let sampling = SampleParams { kernel, ..params.under(rule) };
    let sampled = sample_modules_with(image, &projection, func.width, func.height, kernel);
    let mut grid = classify(&sampled, &sampling).to_grid(func.width, func.height);
    if found.inverted {
        grid = grid.inverted();
    }
    let analysis = analyze(&grid, &func);
    let mut report = decode_as(&grid, &func, &analysis, opts)?;
    // The format bits may spell a word of the other QR Code model where
    // none of this one is near: a Model 1 symbol found as Model 2, or the
    // reverse. The two lay their data out differently, so the grid is read
    // again as the other model, and the better-ranked read is kept.
    let mut reinterpreted = None;
    if !report.is_trustworthy()
        && analysis.format.other_model_hint().is_some()
        && let Some(other) = found.version.other_model()
    {
        let other_func = Arc::new(FunctionMap::new(other));
        if let Some(again) = decode_as(&grid, &other_func, &analyze(&grid, &other_func), opts)
            && again.score > report.score
        {
            report = again;
            reinterpreted = Some(other);
        }
    }
    let read_as = Arc::new(FunctionMap::new(reinterpreted.unwrap_or(found.version)));
    let (report, pins) = match SymbolLayout::new(read_as, report.ec) {
        Some(layout) => with_padding(&grid, &layout, report, opts),
        None => (report, Pins::new()),
    };
    let result = ReadResult {
        detection: found.clone(),
        agreement: prepared.agreement,
        kernel,
        rule,
        grid,
        analysis,
        report,
        pins,
        reinterpreted,
        enhance: Enhance::NONE,
    };
    Some(if result.is_trustworthy() || work::cancelled() { result } else { refit_by_decode(image, result, &sampling, opts) })
}

/// A decode that is no read, made again with the padding pinned, when
/// that makes a read of it: every block verified, the stream parsing,
/// and the segment headers, once they can be read, bearing out where the
/// padding was placed. The report comes back with the pins it was then
/// decoded under, or as it was with none. A decode the pins leave short
/// of a read is left as it was: what they would add to it rests on the
/// padding alone, with nothing to confirm it.
fn with_padding(grid: &ModuleGrid, layout: &SymbolLayout, report: DecodeReport, opts: &DecodeOptions) -> (DecodeReport, Pins) {
    if report.is_trustworthy() {
        return (report, Pins::new());
    }
    match recover_padding_after(grid, layout, report.mask, &Pins::new(), opts, &report) {
        Some(recovery) if recovery.confirmed && recovery.report.is_trustworthy() => (recovery.report, recovery.pins),
        _ => (report, Pins::new()),
    }
}

/// Known modules a decode must add to the fixed patterns before the
/// placement is refitted to them.
const FEEDBACK_MIN_KNOWN: usize = 64;
/// How far the refit may move a corner, or an anchor, in modules: it
/// corrects a fraction of a module, and further is another placement.
const FEEDBACK_TRAVEL: f64 = 0.75;
/// Rounds of refit and decode: the second reads through a placement the
/// first improved, and a third rarely adds to it.
const FEEDBACK_ROUNDS: usize = 2;
/// Most boundaries the corner refit scores: a large symbol has tens of
/// thousands between its known modules, and every tenth places the
/// corners as well.
const FEEDBACK_MAX_EDGES: usize = 2000;
/// Half-width, in modules, of the window of known modules an anchor is
/// refitted against.
const FEEDBACK_WINDOW: f64 = 5.0;

/// Turn a partial decode back on the placement, as the module describes,
/// for as long as each round improves the decode's rank. `params` carry
/// the kernel and threshold rule the read was made with.
fn refit_by_decode(image: &Luma, mut current: ReadResult, params: &SampleParams, opts: &DecodeOptions) -> ReadResult {
    for _ in 0..FEEDBACK_ROUNDS {
        if work::cancelled() {
            break;
        }
        let Some(next) = feedback_round(image, &current, params, opts) else { break };
        if next.report.score <= current.report.score {
            break;
        }
        let settled = next.is_trustworthy();
        current = next;
        if settled {
            break;
        }
    }
    current
}

/// One round: the template of known modules, the corners and the anchors
/// refitted to it, and the symbol sampled and decoded again under the
/// format the blocks that solved settled. `None` when the decode knows
/// too little, or the refit cannot be scored.
fn feedback_round(image: &Luma, current: &ReadResult, params: &SampleParams, opts: &DecodeOptions) -> Option<ReadResult> {
    // Nothing solved, nothing learnt: the format and version bits alone
    // add a few dozen modules, which is no template.
    if current.report.blocks_verified() == 0 {
        return None;
    }
    let found = &current.detection;
    let func = Arc::new(FunctionMap::new(current.version()));
    let layout = SymbolLayout::new(func.clone(), current.report.ec)?;
    let expected = known_modules(&layout, current.report.mask, &current.report);
    let fixed = (0..func.height).flat_map(|y| (0..func.width).map(move |x| (x, y))).filter(|&(x, y)| func.expected(x, y).is_some()).count();
    if expected.iter().flatten().count() < fixed + FEEDBACK_MIN_KNOWN {
        return None;
    }
    let template = Template { width: func.width, height: func.height, expected };
    let sign = if found.inverted { -1.0 } else { 1.0 };
    let threshold = fixed_threshold(params);

    // The corners, against every known module, with the anchors held.
    let controls: Vec<ControlPoint> = found.anchors.iter().map(|a| a.control_point()).collect();
    let warp = Warp::new(func.width, func.height, &controls);
    let scorer = Scorer::new(image, &template, found.corners, threshold).with_spline(warp.spline()).with_max_edges(FEEDBACK_MAX_EDGES);
    let module_px = scorer.module_px();
    let mut corners = found.corners;
    let mut evaluations = 0;
    let score = |trial: [[f64; 2]; 4]| scorer.contrast_projection(&warp.place(trial)?, sign);
    if let Some((moved, _)) = descend_by(found.corners, &MOVES, module_px * 0.25, &mut evaluations, score)
        && (0..4).all(|i| distance(moved[i], found.corners[i]) <= FEEDBACK_TRAVEL * module_px)
    {
        // Believed only while it reads at least as many known modules
        // right as the corners it started from.
        let before = scorer.agreement_known(&warp.place(found.corners)?, found.inverted)?.0;
        let after = scorer.agreement_known(&warp.place(moved)?, found.inverted)?.0;
        if after >= before {
            corners = moved;
        }
    }

    // Each anchor, against the known modules around it, through the map
    // the corners and the other anchors make.
    let mut anchors = found.anchors.clone();
    for k in 0..anchors.len() {
        // An edge point was traced where the paper turns to ink, with
        // half its window off the symbol: the known modules do not
        // place it better.
        if anchors[k].kind == AnchorKind::Edge {
            continue;
        }
        let Some((window, [x0, y0, x1, y1])) = window_of(&template, anchors[k].at) else { continue };
        let controls: Vec<ControlPoint> = anchors.iter().map(|a| a.control_point()).collect();
        let placed = |center: [f64; 2]| {
            let mut moved = controls.clone();
            moved[k].px = center;
            Projection::new(func.width, func.height, corners, &moved)
        };
        let Some(initial) = placed(anchors[k].center) else { continue };
        let frame = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(|[x, y]| initial.apply(x as f64, y as f64));
        let local = Scorer::new(image, &window, frame, threshold).with_along(6);
        let local_px = local.module_px();
        let (dx, dy) = (x0 as f64, y0 as f64);
        let score = |trial: [[f64; 2]; 4]| local.contrast_with(&Shifted { map: &placed(trial[0])?, dx, dy }, sign);
        let Some(start) = score([anchors[k].center; 4]) else { continue };
        let mut evaluations = 0;
        if let Some((moved, best)) = descend_by([anchors[k].center; 4], &[[1.0, 0.0, 0.0, 0.0]], local_px * 0.25, &mut evaluations, score)
            && best > start
            && distance(moved[0], anchors[k].center) <= FEEDBACK_TRAVEL * local_px
        {
            anchors[k].center = moved[0];
        }
    }

    // Then the anchors the known modules contradict are dropped.
    let anchors = drop_contradicted_anchors(image, &func, &template, corners, &anchors, found.inverted, params);

    // Sampled and decoded again, under the format the blocks that solved
    // settled.
    let controls: Vec<ControlPoint> = anchors.iter().map(|a| a.control_point()).collect();
    let projection = Projection::new(func.width, func.height, corners, &controls)?;
    let sampled = sample_modules_with(image, &projection, func.width, func.height, current.kernel);
    let mut grid = classify(&sampled, params).to_grid(func.width, func.height);
    if found.inverted {
        grid = grid.inverted();
    }
    let own = Arc::new(FunctionMap::new(found.version));
    let analysis = analyze(&grid, &own);
    let report = decode(&grid, &layout, current.report.mask, &Pins::new(), opts);
    let (report, pins) = with_padding(&grid, &layout, report, opts);
    let whole = Scorer::new(image, &Template::whole(&own), corners, threshold);
    let agreement = (whole.agreement_known(&projection, found.inverted)?.0, whole.fixed());
    let mut detection = found.clone();
    detection.corners = corners;
    detection.anchors = anchors;
    detection.agreement = agreement;
    Some(ReadResult {
        detection,
        agreement,
        kernel: current.kernel,
        rule: current.rule,
        grid,
        analysis,
        report,
        pins,
        reinterpreted: current.reinterpreted,
        enhance: current.enhance,
    })
}

/// The window of `template` [`FEEDBACK_WINDOW`] modules either way about
/// `at`, clipped to the symbol, with its bounds `[x0, y0, x1, y1]`;
/// `None` when it is too small or holds fewer than eight known modules.
fn window_of(template: &Template, at: [f64; 2]) -> Option<(Template, [usize; 4])> {
    let x0 = (at[0] - FEEDBACK_WINDOW).floor().max(0.0) as usize;
    let y0 = (at[1] - FEEDBACK_WINDOW).floor().max(0.0) as usize;
    let x1 = ((at[0] + FEEDBACK_WINDOW).ceil().max(0.0) as usize).min(template.width);
    let y1 = ((at[1] + FEEDBACK_WINDOW).ceil().max(0.0) as usize).min(template.height);
    if x1 <= x0 + 2 || y1 <= y0 + 2 {
        return None;
    }
    let window = Template {
        width: x1 - x0,
        height: y1 - y0,
        expected: (y0..y1).flat_map(|y| (x0..x1).map(move |x| (x, y))).map(|(x, y)| template.expected(x, y)).collect(),
    };
    (window.fixed() >= 8).then_some((window, [x0, y0, x1, y1]))
}

/// Known modules about an anchor that must read right through the map
/// without the anchor and wrong through the map with it, over and above
/// the other way about, for the anchor to be dropped: one is the noise of
/// a module on the edge of the threshold.
const DROP_MARGIN: usize = 2;
/// Anchors dropped at most in one pass: as many as BoofCV reads a symbol
/// that fails, which drops one fewer.
const DROP_LIMIT: usize = 6;

/// The anchors among `anchors` that the known modules of `template` bear
/// out, in their order: the known modules within [`FEEDBACK_WINDOW`] of
/// each anchor are read through the map the corners and every anchor
/// make and through the map without that anchor, and the anchor whose
/// window reads better without it by the most, [`DROP_MARGIN`] modules
/// at least, is dropped, the rest judged again through the map without
/// it, up to [`DROP_LIMIT`] times, as BoofCV refits without its worst
/// located pattern and reads again. Finder anchors are never dropped:
/// the placement stands on them, and their corners are held besides.
/// `inverted` reads the symbol light on dark; `params` carry the
/// threshold the modules are read against.
pub fn drop_contradicted_anchors(
    image: &Luma,
    func: &FunctionMap,
    template: &Template,
    corners: [[f64; 2]; 4],
    anchors: &[Anchor],
    inverted: bool,
    params: &SampleParams,
) -> Vec<Anchor> {
    let threshold = fixed_threshold(params);
    let mut kept: Vec<Anchor> = anchors.to_vec();
    for _ in 0..DROP_LIMIT {
        let controls: Vec<ControlPoint> = kept.iter().map(Anchor::control_point).collect();
        let Some(with_all) = Projection::new(func.width, func.height, corners, &controls) else { break };
        let mut worst: Option<(usize, usize)> = None;
        for k in 0..kept.len() {
            if kept[k].kind == AnchorKind::Finder {
                continue;
            }
            let Some((window, [x0, y0, x1, y1])) = window_of(template, kept[k].at) else { continue };
            let others: Vec<ControlPoint> = controls.iter().enumerate().filter(|(i, _)| *i != k).map(|(_, c)| *c).collect();
            let Some(without) = Projection::new(func.width, func.height, corners, &others) else { continue };
            let frame = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(|[x, y]| with_all.apply(x as f64, y as f64));
            let local = Scorer::new(image, &window, frame, threshold).with_along(6);
            let (dx, dy) = (x0 as f64, y0 as f64);
            let Some((with_it, _)) = local.agreement_known(&Shifted { map: &with_all, dx, dy }, inverted) else { continue };
            let Some((without_it, _)) = local.agreement_known(&Shifted { map: &without, dx, dy }, inverted) else { continue };
            if without_it >= with_it + DROP_MARGIN && worst.is_none_or(|(_, gain)| without_it - with_it > gain) {
                worst = Some((k, without_it - with_it));
            }
        }
        match worst {
            Some((k, _)) => {
                kept.remove(k);
            }
            None => break,
        }
    }
    kept
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Decode `grid` as the symbol `func` maps: under the EC level and mask
/// the format bits settle, or, when they settle nothing, the best attempt
/// of a brute-force search. `None` when the version has no attempt at all.
fn decode_as(grid: &ModuleGrid, func: &Arc<FunctionMap>, analysis: &GridAnalysis, opts: &DecodeOptions) -> Option<DecodeReport> {
    if analysis.format.is_reliable() {
        let best = analysis.format.best().expect("a reliable format has a best candidate");
        let layout = SymbolLayout::new(func.clone(), best.fields.ec)?;
        Some(decode(grid, &layout, best.fields.mask, &Pins::new(), opts))
    } else {
        Some(brute_force(grid, func, None, opts).into_iter().next()?.report)
    }
}

/// Bring a detection made in the photo turned `quarters` quarter turns
/// back to the photo's own pixels, `width` by `height`, with its turn
/// reckoned against `region`, the outline in the photo as it is.
fn unturn(found: &mut Detection, quarters: u8, width: usize, height: usize, region: [[f64; 2]; 4]) {
    found.corners = found.corners.map(|c| unrotate_point_cw(c, quarters, width, height));
    for anchor in &mut found.anchors {
        anchor.center = unrotate_point_cw(anchor.center, quarters, width, height);
    }
    found.turns = turns_between(region, found.corners);
    if found.retry.is_none() {
        found.retry = Some(Retry::Turned { quarters });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imaging::detect::InfoEvidence;

    /// A version 10 symbol lying square to the photo at eight pixels a
    /// module, with a finder or alignment anchor at each of its patterns,
    /// every one where the plane has it.
    fn placed() -> Detection {
        let version = SymbolVersion::Qr(10);
        let (origin, module) = ([40.0, 60.0], 8.0);
        let at_px = |[x, y]: [f64; 2]| [origin[0] + x * module, origin[1] + y * module];
        let n = version.width() as f64;
        let finders = [[3.5, 3.5], [n - 3.5, 3.5], [3.5, n - 3.5]];
        let alignments = [[28.5, 6.5], [6.5, 28.5], [28.5, 28.5], [50.5, 28.5], [28.5, 50.5], [50.5, 50.5]];
        let anchor = |kind: AnchorKind, at: [f64; 2]| Anchor { kind, at, center: at_px(at), partial: false, inferred: false };
        let anchors: Vec<Anchor> = finders
            .into_iter()
            .map(|at| anchor(AnchorKind::Finder, at))
            .chain(alignments.into_iter().map(|at| anchor(AnchorKind::Alignment, at)))
            .collect();
        Detection {
            corners: [at_px([0.0, 0.0]), at_px([n, 0.0]), at_px([n, n]), at_px([0.0, n])],
            version,
            inverted: false,
            mirrored: false,
            mirrored_by_decode: false,
            turns: 0,
            anchors,
            agreement: (0, 0),
            info: InfoEvidence::default(),
            edged_corners: Vec::new(),
            measured_corners: vec![0, 1, 3],
            retry: None,
            decodes: None,
        }
    }

    #[test]
    fn the_anchor_farthest_from_its_place_is_the_one_doubted() {
        let mut found = placed();
        assert_eq!(found.count(AnchorKind::Finder), 3);
        assert_eq!(found.count(AnchorKind::Alignment), 6);
        // Every anchor where the others put it: none to doubt.
        assert_eq!(doubtful_anchor(&found), None);
        // The pattern in the middle fitted a module off, as one fitted
        // to a logo printed over it is: twenty-two modules from the
        // nearest anchor, where the detector allows it more than that.
        let middle = found.anchors.iter().position(|a| a.at == [28.5, 28.5]).unwrap();
        found.anchors[middle].center[0] += 8.0;
        assert_eq!(doubtful_anchor(&found), Some(middle));
        // Another, less far out: the worse of the two is still the one.
        let side = found.anchors.iter().position(|a| a.at == [50.5, 28.5]).unwrap();
        found.anchors[side].center[1] -= 5.0;
        assert_eq!(doubtful_anchor(&found), Some(middle));
        found.anchors.remove(middle);
        let side = found.anchors.iter().position(|a| a.at == [50.5, 28.5]).unwrap();
        assert_eq!(doubtful_anchor(&found), Some(side));
        // A quarter of a module is how well a blurred pattern is placed.
        found.anchors[side].center[1] += 5.0 - 2.0;
        assert_eq!(doubtful_anchor(&found), None);
        // A finder is never doubted, wherever it stands: the placement
        // stands on it.
        let finder = found.anchors.iter().position(|a| a.kind == AnchorKind::Finder).unwrap();
        found.anchors[finder].center[0] += 20.0;
        assert_ne!(doubtful_anchor(&found), Some(finder));
    }
}
