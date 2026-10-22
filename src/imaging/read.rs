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
//! - the format words next nearest the format bits, when the word they
//!   settle verifies no block: a word misread in four bits or five lies
//!   within the three the code corrects of another two times in five,
//!   and then reads as that other with nothing to say it is not. ZXing's
//!   decoder, and OpenCV's, take the word their format bits correct to
//!   and fail with it; here the blocks are asked, and a level and mask
//!   they bear out stand in its place;
//! - the placement turned on its symbol, before any of the above, when
//!   its fixed patterns fit that way and not as placed: a finder pattern
//!   in the corner that should have none is as plain as anything a photo
//!   says, and BoofCV's Micro QR reader, which has one position pattern
//!   to go by, tries all four turns and keeps the one that decodes;
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
    turned_on_symbol,
};
use super::enhance::{Enhance, Enhanced};
use super::projection::{ControlPoint, Project, Projection};
use super::refine::{MOVES, Scorer, Shifted, Template, Warp, descend_by, fixed_threshold, refine_corners_holding};
use super::sample::{Kernel, Rule, SampleParams, classify, read_modules, sample_modules_noting};
use super::{Luma, rotate_outline_cw, unrotate_point_cw};
use crate::decode::bch::{FormatAnalysis, FormatCandidate};
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
    /// Whether the modules were sharpened against their neighbours as
    /// they were read ([`SampleParams::sharpen_modules`]), as was asked
    /// for: with the kernel and the rule, what a sampling must be made
    /// with to give this grid again.
    pub sharpen_modules: bool,
    /// The modules the unsure band would have left unknown were kept at
    /// the colour they fell on, at their reliability, for the decoder to
    /// doubt rather than erase ([`Classified::to_grid_as_decided`]): a
    /// sampling must have no band to give this grid again.
    pub unsure_kept: bool,
}

/// One way of reading a placement: the kernel its modules are sampled
/// with, the threshold rule they are judged by, and whether the modules
/// the unsure band would leave unknown are kept at the colour they fell
/// on instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Variant {
    kernel: Kernel,
    rule: Rule,
    unsure_kept: bool,
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
    read_outline_as(image, version, region, params, Reads::new(opts))
}

/// [`read_outline`] as one of `ways` the caller reads the outline, the
/// best of which is kept, as a reader tries every family it knows: the
/// read asked for is the first of the way `asked` for, and goes
/// uncounted as [`read_outline`]'s does, and every other read of every
/// way is charged as one of all those the ways may make ([`Reads`]).
/// The other ways' reads were once each a read of their own, the first
/// of each uncounted.
pub fn read_outline_one_of(
    image: &Luma,
    version: SymbolVersion,
    region: [[f64; 2]; 4],
    params: &SampleParams,
    opts: &DecodeOptions,
    ways: usize,
    asked: bool,
) -> Option<ReadResult> {
    let reads = Reads::new(opts).one_of(ways);
    read_outline_as(image, version, region, params, if asked { reads } else { reads.past() })
}

/// What a read of a photo is charged for the others its search may
/// make, as the options it decodes under: the read asked for, and every
/// other. The read asked for is the first at every turn of the search,
/// the detector's first placement sampled with the kernel and rule
/// asked for, through the photo as asked, as the format bits settle it,
/// and nothing refitted: the read every reader makes, charged for
/// nothing but the decodes its caller counts, as the plain decode of a
/// block as read stands on its own count ([`crate::decode::pipeline`]).
/// Every other read, the runners-up and the other samplings, the
/// photo's other turns and copies, the reads in mirror order and
/// without anchors, as the other QR Code model and refitted, and with
/// the padding pinned, is a fresh chance for a block of noise to verify,
/// and the best read is kept: each is charged as one of every read the
/// search may make, and a bit more for the share the read asked for
/// leaves the rest. Those reads were once charged in part, the
/// placements and samplings counted, the read asked for among them, and
/// the rest uncounted; and then, past the read asked for, the first way
/// tried at each turn beneath went uncounted for that turn's others, as
/// though it were the read asked for, and every sampling of a turned
/// photo was counted as the only one.
#[derive(Clone, Copy, Debug)]
struct Reads {
    first: DecodeOptions,
    again: DecodeOptions,
    /// `first` is the read asked for, and goes uncounted at each turn.
    /// Past it, `first` is one of the others, counted as `again` is.
    asked: bool,
}

impl Reads {
    fn new(opts: &DecodeOptions) -> Self {
        Self { first: *opts, again: opts.one_of(2), asked: true }
    }

    /// These reads at a turn of the search that tries `n` ways, the first
    /// of them going on as asked where these are the read asked for:
    /// every other read counted as one of them, and the budgets of the
    /// dear solvers shared over them all ([`DecodeOptions::among`]).
    fn among(self, n: usize) -> Self {
        let again = self.again.among(n);
        Self { first: if self.asked { self.first.sharing(n) } else { again }, again, ..self }
    }

    /// As [`Self::among`], with the budgets whole.
    fn one_of(self, n: usize) -> Self {
        let again = self.again.one_of(n);
        Self { first: if self.asked { self.first } else { again }, again, ..self }
    }

    /// The reads past the first way tried at a turn: none of them asked
    /// for, at this turn or any beneath it.
    fn past(self) -> Self {
        Self { first: self.again, asked: false, ..self }
    }

    /// These reads for each way tried at a turn, in the order tried: the
    /// first as these are, and every other past it ([`Self::past`]).
    fn first_then_past(self) -> impl Iterator<Item = Self> {
        std::iter::once(self).chain(std::iter::repeat(self.past()))
    }
}

/// [`read_outline`], charging its reads as `reads` says.
fn read_outline_as(image: &Luma, version: SymbolVersion, region: [[f64; 2]; 4], params: &SampleParams, reads: Reads) -> Option<ReadResult> {
    let asked = params.enhance;
    // Every read is one of those through each way of seeing the photo
    // that may be tried, the best of which is kept: the way asked for,
    // the photo as it is, and the rungs, each once. The way asked for was
    // once counted again where it is a rung.
    let mut ways = vec![asked];
    for way in std::iter::once(Enhance::NONE).chain(Enhance::rungs(region)) {
        if !ways.contains(&way) {
            ways.push(way);
        }
    }
    let reads = reads.one_of(ways.len());
    let mut best = read_outline_through(image, version, region, params, reads, asked);
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
        |&rung| if work::cancelled() { None } else { read_outline_through(image, version, region, params, reads.past(), rung) },
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
    reads: Reads,
    enhance: Enhance,
) -> Option<ReadResult> {
    // Whatever is read, it is read off the pixels handed on: the copy's,
    // which are enlarged and sharpened already, or the photo's.
    let plain = SampleParams { enhance: Enhance::NONE, ..*params };
    if enhance.is_none() {
        return read_outline_plain(image, version, region, &plain, reads, false);
    }
    let view = Enhanced::around(image, region, enhance)?;
    // A reading set for the scan reaches as far across the copy as it
    // was set to reach across the photo.
    let plain = SampleParams { binarizer: plain.binarizer.scaled(view.factor), ..plain };
    let mut result = read_outline_plain(&view.luma, version, region.map(|p| view.to_view(p)), &plain, reads, true)?;
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
    reads: Reads,
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
    let candidates = seated(image, detect_candidates_by(image, version, region, params, climb), params);
    // Each read is one of about as many as there are placements and
    // turns of the photo, times the ways of sampling each.
    let all = reads.among(candidates.len() + 3);
    let placements: Vec<(&Detection, Reads)> = candidates.iter().zip(all.first_then_past()).collect();
    let reads_of = |&(found, counted): &(&Detection, Reads)| {
        let mut reads = Vec::new();
        if work::cancelled() {
            return reads;
        }
        if let Some(prepared) = prepare(image, found.clone(), params) {
            read_placement(image, &prepared, &variants, params, counted, &mut |result| {
                let settled = result.is_trustworthy();
                reads.push(result);
                settled
            });
        }
        reads
    };
    if work::first_then_in_turn(&placements, reads_of, |reads| reads.into_iter().any(|result| consider(&mut best, result))) {
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
        let Some(prepared) = prepare(&turned, seat(&turned, found, params), params) else { return reads };
        let turned_reads = all.past().among(variants.len());
        work::first_then_in_turn(
            &variants,
            |&variant| if work::cancelled() { None } else { read_prepared(&turned, &prepared, variant, params, turned_reads) },
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
/// ([`SampleParams::rules`]); then, when the unsure band would leave
/// any module unknown, the kernel asked for under every rule again with
/// those modules kept at the colour they fell on
/// ([`Classified::to_grid_as_decided`]); then each other kernel under
/// every rule. The rule comes before the kernels because it decides more
/// often: see the module's account. The unsure modules come between
/// because keeping them changes the least of what is read: a module in
/// the band is unknown to the exact decoder either way, as a codeword it
/// erases or as a bit the fallbacks doubt before any other, and what
/// differs is that the erasure spends a unit of redundancy on the whole
/// codeword where the doubt spends an equation on the bit. Over blocks
/// of simulated noise the band read no more than the colours kept, and
/// with a stain beside the noise far less; on the photographs it is
/// tried second, since what the band leaves unknown on them is often a
/// module the grid straddles, which reads as nothing in either colour.
fn variants_of(params: &SampleParams) -> Vec<Variant> {
    let under = |kernel: Kernel, unsure_kept: bool| params.rules().map(move |rule| Variant { kernel, rule, unsure_kept });
    let mut variants: Vec<Variant> = under(params.kernel, false).collect();
    if params.band > 0.0 {
        variants.extend(under(params.kernel, true));
    }
    variants.extend(params.kernel.retries().skip(1).flat_map(|kernel| under(kernel, false)));
    variants
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
    // The placement found, read every way, and then the search about it,
    // each read one of those of both ([`Reads`]).
    let reads = Reads::new(opts).one_of(2);
    if read_found_through(image, found, around, params, reads, through, &mut best) || work::cancelled() {
        return best;
    }
    if let Some(again) = read_outline_as(image, version, around, params, reads.past())
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
    reads: Reads,
    enhance: Enhance,
    best: &mut Option<ReadResult>,
) -> bool {
    let plain = SampleParams { enhance: Enhance::NONE, ..*params };
    let variants = variants_of(&plain);
    if enhance.is_none() {
        return prepare(image, seat(image, found, &plain), &plain)
            .is_some_and(|prepared| read_placement(image, &prepared, &variants, &plain, reads, &mut |result| consider(best, result)));
    }
    let Some(view) = Enhanced::around(image, around, enhance) else { return false };
    found.corners = found.corners.map(|c| view.to_view(c));
    for anchor in &mut found.anchors {
        anchor.center = view.to_view(anchor.center);
    }
    let plain = SampleParams { binarizer: plain.binarizer.scaled(view.factor), ..plain };
    prepare(&view.luma, seat(&view.luma, found, &plain), &plain).is_some_and(|prepared| {
        read_placement(&view.luma, &prepared, &variants, &plain, reads, &mut |mut result| {
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
/// settle does not hang on the kernel. The first sampling goes on as
/// the read asked for, and every other read is charged as one of all
/// these that may be made, the best of which is kept ([`Reads`]): the
/// reads in mirror order and without anchors once went uncounted.
fn read_placement(
    image: &Luma,
    prepared: &Prepared,
    variants: &[Variant],
    params: &SampleParams,
    reads: Reads,
    keep: &mut dyn FnMut(ReadResult) -> bool,
) -> bool {
    let few = variants.len().min(params.rules().count());
    let reads = reads.among(variants.len() + few + DOUBT_LIMIT * few);
    let mut verified = 0;
    let samplings: Vec<(Variant, Reads)> = variants.iter().copied().zip(reads.first_then_past()).collect();
    let settled = work::first_then_in_turn(
        &samplings,
        |&(variant, reads)| if work::cancelled() { None } else { read_prepared(image, prepared, variant, params, reads) },
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
    let few = &variants[..few];
    let found = &prepared.detection;
    if found.version.family() != Family::Rmqr
        && found.info.format.valid == Some(true)
        && let Some(detection) = seen_from_behind(image, found, params)
    {
        let behind = Prepared { detection, agreement: prepared.agreement };
        for &variant in few {
            if let Some(result) = read_prepared(image, &behind, variant, params, reads.past())
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
        for &variant in few {
            if let Some(result) = read_prepared(image, &doubted, variant, params, reads.past()) {
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

/// A placement the right way up on its symbol: as given, unless the fixed
/// patterns sampled through it fit the grid turned or reflected and not
/// as it lies, when it is the placement turned so
/// ([`turned_on_symbol`]), with what the turned grid's fixed modules
/// read as its agreement.
///
/// The fit that ranks the placements of an outline is no judge of the
/// turn where light falls unevenly across a symbol or a finder is
/// damaged, and a square a quarter turn out on two of a symbol's three
/// finders has been the best fit, and the only placement of that square
/// offered. Sampled, it shows a finder pattern where the version has
/// none, and every read of it came back with that noted in its analysis
/// and no block solved, to be turned by hand in the window. A grid's
/// fixed patterns fit so much better turned
/// ([`crate::decode::pipeline::GridAnalysis::orientation`]) only when
/// the placement is turned, so the turn is made here, before the
/// corners are refined against patterns that are not where the
/// placement has them. A turn that would also swap dark and light is
/// left to the detector, which settles which is ink.
fn seat(image: &Luma, found: Detection, params: &SampleParams) -> Detection {
    let func = FunctionMap::new(found.version);
    let Some(projection) = found.projection() else { return found };
    let sampled = sample_modules_noting(image, &projection, func.width, func.height, params.kernel, params.sharpen_modules).0;
    let mut grid = classify(&sampled, params).to_grid(func.width, func.height);
    if found.inverted {
        grid = grid.inverted();
    }
    match analyze(&grid, &func).orientation {
        Some(turn) if !turn.inverted => {
            let mut seated = turned_on_symbol(image, &found, turn.symmetry, params);
            seated.agreement = turn.agreement;
            seated
        }
        _ => found,
    }
}

/// The detector's placements each the right way up on its symbol
/// ([`seat`]), and each once: a placement that was a turn out comes to
/// lie where the same square the right way up may already stand among
/// them, and a symbol too damaged to read is not read there twice.
fn seated(image: &Luma, candidates: Vec<Detection>, params: &SampleParams) -> Vec<Detection> {
    let mut out: Vec<Detection> = Vec::new();
    for found in candidates {
        let found = seat(image, found, params);
        if !out.iter().any(|kept| kept.placed_as(&found)) {
            out.push(found);
        }
    }
    out
}

/// A detection with its corners and anchors refined against the photo,
/// ready to sample.
struct Prepared {
    detection: Detection,
    agreement: (usize, usize),
}

/// Refine a detection's corners, with the anchors held and the corners a
/// finder pattern measured kept, and then every anchor against its own
/// pattern. `None` when the corners form no usable outline. The
/// detection is taken to lie the right way up on its symbol ([`seat`]).
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

/// Sample a prepared detection with the variant's kernel, classify it
/// under its threshold rule, with the unsure modules kept at their
/// colour when it says so, and decode what it reads. `None` for a
/// variant that keeps the unsure modules when the band left none
/// unknown: the read would be the one already made. The decode as the
/// version goes on as the read asked for, and those made after it, as
/// the other QR Code model, with the padding pinned, and refitted to
/// the decode as many rounds as that takes, the best of which is kept,
/// are charged as reads after it ([`Reads`]): they once went uncounted.
/// Those made under the level and mask it settled are charged besides
/// for what it picked them among, every level and mask of a brute force
/// or the rivals of the word the format bits read ([`decode_as`]).
fn read_prepared(image: &Luma, prepared: &Prepared, variant: Variant, params: &SampleParams, reads: Reads) -> Option<ReadResult> {
    let found = &prepared.detection;
    let models = 1 + usize::from(found.version.other_model().is_some());
    // Pinned with the padding too, each of these.
    let reads = reads.one_of(2 * models * (1 + FEEDBACK_ROUNDS));
    let (opts, retry) = (&reads.first, &reads.again);
    let func = Arc::new(FunctionMap::new(found.version));
    let projection = found.projection()?;
    let Variant { kernel, rule, unsure_kept } = variant;
    let banded = SampleParams { kernel, ..params.under(rule) };
    let sampling = SampleParams { band: if unsure_kept { 0.0 } else { params.band }, ..banded };
    // How the photo blurs, and how far a reading is to be believed, are
    // learnt from the fixed patterns, whose colours are known before
    // anything is decoded.
    let (classified, _) = read_modules(image, &projection, func.width, func.height, &banded, func.expected_modules(), found.inverted);
    if unsure_kept && classified.unsure() == 0 {
        return None;
    }
    let mut grid =
        if unsure_kept { classified.to_grid_as_decided(func.width, func.height) } else { classified.to_grid(func.width, func.height) };
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
        if let Some(again) = decode_as(&grid, &other_func, &analyze(&grid, &other_func), retry)
            && again.score > report.score
        {
            report = again;
            reinterpreted = Some(other);
        }
    }
    // The level and mask the report was decoded under may be the pick of
    // many, and the decodes it was charged for say how many: every decode
    // made under them after it, with the padding pinned or refitted, is a
    // pick of as many, besides a read after the first. Those decodes were
    // once charged as reads after the first alone, and stood on as much
    // as five bits more than the brute force they were made under.
    let decoded_under = if reinterpreted.is_some() { retry.tries } else { opts.tries };
    let retry = &retry.one_of(report.tries.div_ceil(decoded_under.max(1)));
    let read_as = Arc::new(FunctionMap::new(reinterpreted.unwrap_or(found.version)));
    let (report, pins) = match SymbolLayout::new(read_as, report.ec) {
        Some(layout) => with_padding(&grid, &layout, report, retry),
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
        sharpen_modules: params.sharpen_modules,
        unsure_kept,
    };
    Some(if result.is_trustworthy() || work::cancelled() { result } else { refit_by_decode(image, result, &sampling, retry) })
}

/// A decode that is no read, made again with the padding pinned, when
/// that makes a read of it: every block verified, the stream parsing,
/// and the segment headers, once they can be read, bearing out where the
/// padding was placed. The report comes back with the pins it was then
/// decoded under, or as it was with none. A decode the pins leave short
/// of a read is left as it was: what they would add to it rests on the
/// padding alone, with nothing to confirm it. A decode that is a read
/// already comes back with the pins it derived for itself, when it was
/// asked to derive any ([`DecodeOptions::derive_pins`]), which are the
/// same pins found the same way.
fn with_padding(grid: &ModuleGrid, layout: &SymbolLayout, report: DecodeReport, opts: &DecodeOptions) -> (DecodeReport, Pins) {
    if report.is_trustworthy() {
        let pins = report.derived_pins.clone();
        return (report, pins);
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
        // place it better. An anchor set by hand where no pattern is stays
        // where it was set.
        if matches!(anchors[k].kind, AnchorKind::Edge | AnchorKind::Custom) {
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
    // The template has every module of the blocks that verified besides
    // the fixed patterns: hundreds of labelled modules to learn the
    // photo's blur and misreads from, where the fixed patterns gave tens.
    let how = SampleParams { kernel: current.kernel, ..*params };
    let (classified, _) = read_modules(image, &projection, func.width, func.height, &how, &template.expected, found.inverted);
    let mut grid = classified.to_grid(func.width, func.height);
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
        sharpen_modules: current.sharpen_modules,
        unsure_kept: current.unsure_kept,
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

/// Bits a format word may lie from the format bits as read and still be
/// tried when the word they settle verifies no block: two past the three
/// the code corrects.
const RIVAL_REACH: u32 = 5;
/// Format words so tried at the most, nearest first.
const RIVALS: usize = 3;

/// Decode `grid` as the symbol `func` maps: under the EC level and mask
/// the format bits settle, or, when they settle nothing, the best attempt
/// of a brute-force search. `None` when the version has no attempt at all.
///
/// A word the bits settle beyond doubt is still not always the symbol's.
/// The words of the format code differ in seven bits or eight, so one
/// misread in four lies as often as not three from another, which is
/// within what the code corrects: of the patterns of four wrong bits, and
/// of five, two in five read so, as that other word with nothing to say
/// it is not. Every block is then unmasked with the wrong mask or cut to
/// the wrong level's blocks, and none solves. So when the word settled
/// verifies no block, the words next nearest the bits are tried
/// ([`rival_decode`]), and one the blocks bear out is the format.
fn decode_as(grid: &ModuleGrid, func: &Arc<FunctionMap>, analysis: &GridAnalysis, opts: &DecodeOptions) -> Option<DecodeReport> {
    if !analysis.format.is_reliable() {
        return Some(brute_force(grid, func, None, opts).into_iter().next()?.report);
    }
    let best = analysis.format.best().expect("a reliable format has a best candidate");
    let layout = SymbolLayout::new(func.clone(), best.fields.ec)?;
    let report = decode(grid, &layout, best.fields.mask, &Pins::new(), opts);
    if report.blocks_verified() > 0 {
        return Some(report);
    }
    Some(rival_decode(grid, func, &analysis.format, opts).unwrap_or(report))
}

/// The best decode of `grid` under the format words nearest its format
/// bits after the one they settle, when the blocks bear one out: the
/// words of the grid's own size, painted as the settled one is, within
/// [`RIVAL_REACH`] bits of the bits read, the [`RIVALS`] nearest of them.
/// A word wrong in four bits or five that reads as another is always
/// among the three nearest of the rest, and a word read clean, or with
/// one bit wrong, has no other within reach, so a symbol whose format
/// reads well and whose blocks are past correcting costs nothing here.
///
/// A rival stands on its blocks alone, the bits having spoken for
/// another: on a block it verifies, every decode charged as one of the
/// rivals and the word the bits settled ([`DecodeOptions::among`]), so
/// that each word tried is the fresh chance for noise to pass that it
/// is. The blocks verified were once held to three codewords' worth
/// between them, whatever was tried, which left out the sixteen bits of
/// an M1 symbol's one block. As [`brute_force`] does, each is decoded
/// without doubting first, which tells the right mask from the rest
/// when any block of it reads nearly clean, and only one borne out so
/// is decoded again with it.
fn rival_decode(grid: &ModuleGrid, func: &Arc<FunctionMap>, format: &FormatAnalysis, opts: &DecodeOptions) -> Option<DecodeReport> {
    let best = format.best()?;
    let mut rivals: Vec<&FormatCandidate> = format
        .candidates
        .iter()
        .filter(|c| c.fits_grid && (c.unmasked, c.other_model) == (best.unmasked, best.other_model))
        .filter(|c| c.fields != best.fields && c.best_mismatches <= RIVAL_REACH)
        .collect();
    rivals.sort_by_key(|c| (c.best_mismatches, c.best_weight));
    rivals.truncate(RIVALS);
    let borne_out = |report: &DecodeReport| report.blocks_verified() > 0;
    let opts = &opts.among(rivals.len() + 1);
    let two_passes = (opts.reliability && grid.has_reliability()) || opts.erase_by_trial;
    let first_pass = DecodeOptions { reliability: opts.reliability && !two_passes, erase_by_trial: false, ..*opts };
    let mut found: Option<DecodeReport> = None;
    for rival in rivals {
        let Some(layout) = SymbolLayout::new(func.clone(), rival.fields.ec) else { continue };
        let mut tried = decode(grid, &layout, rival.fields.mask, &Pins::new(), &first_pass);
        if !borne_out(&tried) {
            continue;
        }
        if two_passes && !tried.is_trustworthy() {
            tried = decode(grid, &layout, rival.fields.mask, &Pins::new(), opts);
        }
        if found.as_ref().is_none_or(|held| tried.score > held.score) {
            found = Some(tried);
        }
    }
    found
}

/// Bring a detection made in the photo turned `quarters` quarter turns
/// back to the photo's own pixels, `width` by `height`, with its turn
/// reckoned against `region`, the outline in the photo as it is.
fn unturn(found: &mut Detection, quarters: u8, width: usize, height: usize, region: [[f64; 2]; 4]) {
    found.corners = found.corners.map(|c| unrotate_point_cw(c, quarters, width, height));
    for anchor in &mut found.anchors {
        anchor.center = unrotate_point_cw(anchor.center, quarters, width, height);
    }
    found.turns = found.turns_in(region);
    // The turn is noted beside whatever else the photo was made into:
    // the step of the ladder the detector climbed to, or the copy a
    // read in one brings the detection back from, stays as it is.
    found.turned = quarters % 4;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::rs::tries_charge;
    use crate::imaging::detect::InfoEvidence;

    #[test]
    fn every_read_but_the_one_asked_for_is_counted_at_every_turn() {
        // The read asked for goes uncounted at every turn of the search,
        // and every other read is one of all those of every turn above it,
        // a bit more for the share the read asked for leaves the rest.
        let opts = DecodeOptions::default();
        let asked = Reads::new(&opts).among(5).one_of(3);
        assert_eq!((asked.first.tries, asked.again.tries), (1, 2 * 5 * 3));
        // Past it, the first way tried at each turn beneath is counted as
        // the rest are: it once went uncounted for its turn, as though it
        // were the read asked for. So, with every sampling of a turned
        // photo, which were each counted as the only one.
        let past = Reads::new(&opts).among(5).past().among(4).one_of(3);
        assert_eq!((past.first.tries, past.again.tries), (2 * 5 * 4 * 3, 2 * 5 * 4 * 3));
        // The budgets are shared over every turn alike.
        assert_eq!(past.first.trial_sets, Reads::new(&opts).among(5).among(4).first.trial_sets);
        // An outline read as one of three ways, as the families are
        // ([`read_outline_one_of`]): the first read of the way asked for
        // goes uncounted, and the first of every other way is counted as
        // the rest are. Each way's first read once went uncounted.
        let asked = Reads::new(&opts).one_of(3);
        let other = asked.past();
        assert_eq!((asked.first.tries, asked.again.tries, other.first.tries), (1, 2 * 3, 2 * 3));
    }

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
            turned: 0,
            decodes: None,
        }
    }

    #[test]
    fn a_symbol_brought_back_from_a_turned_photo_is_turned_as_seen_from_the_front() {
        use crate::imaging::rotate_point_cw;
        let (width, height) = (500, 600);
        let square = placed().corners;
        let turned_in_place = [square[1], square[2], square[3], square[0]];
        // Found in the photo as it lies, on the outline's own corners:
        // no turn, from the front or from behind. The corners of a
        // mirrored symbol run in mirror order, and reckoned on them as
        // they ran, the turn came to a quarter.
        let mut upright = placed();
        unturn(&mut upright, 0, width, height, square);
        assert_eq!(upright.turns, 0);
        let mut mirrored = Detection { corners: [square[0], square[3], square[2], square[1]], mirrored: true, ..placed() };
        unturn(&mut mirrored, 0, width, height, square);
        assert_eq!(mirrored.turns, 0);
        // A quarter turn clockwise in the outline, with the symbol's
        // top-left corner at the outline's top right, is a quarter from
        // the front and from behind alike.
        let mut turned = Detection { corners: turned_in_place, ..placed() };
        unturn(&mut turned, 0, width, height, square);
        assert_eq!(turned.turns, 1);
        let mut turned_behind = Detection { corners: [turned_in_place[0], turned_in_place[3], turned_in_place[2], turned_in_place[1]], mirrored: true, ..placed() };
        unturn(&mut turned_behind, 0, width, height, square);
        assert_eq!(turned_behind.turns, 1);
        // Found in the photo turned a quarter, where an upright symbol
        // lies turned: brought back, it is upright, mirrored or not, and
        // the copy it was found in is noted.
        let in_turned_photo = square.map(|p| rotate_point_cw(p, 1, width, height));
        let mut brought_back = Detection { corners: in_turned_photo, ..placed() };
        unturn(&mut brought_back, 1, width, height, square);
        assert_eq!((brought_back.turns, brought_back.turned), (0, 1));
        for (i, corner) in brought_back.corners.iter().enumerate() {
            assert!((corner[0] - square[i][0]).abs() < 1e-9 && (corner[1] - square[i][1]).abs() < 1e-9, "{i}: {corner:?}");
        }
        let behind = [in_turned_photo[0], in_turned_photo[3], in_turned_photo[2], in_turned_photo[1]];
        let mut brought_back = Detection { corners: behind, mirrored: true, ..placed() };
        unturn(&mut brought_back, 1, width, height, square);
        assert_eq!(brought_back.turns, 0);
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

    /// A version `v` symbol at `level` holding `text`, as painted.
    fn symbol_of(v: u8, level: qrcode_generator::qr::ErrorCorrection, text: &str) -> ModuleGrid {
        use qrcode_generator::{Segment, qr};
        let symbol = qr::Encoder::new(level)
            .version(qr::Version::new(v).unwrap())
            .boost_error_correction(false)
            .encode_segments(&[Segment::bytes(text)])
            .unwrap();
        ModuleGrid::from_bools(symbol.width(), symbol.height(), symbol.modules())
    }

    /// A version 2 symbol holding `text`, eight pixels a module with four
    /// modules of paper round it, and its corners in the image.
    fn drawn(text: &str) -> (Luma, [[f64; 2]; 4]) {
        drawn_grid(&symbol_of(2, qrcode_generator::qr::ErrorCorrection::Low, text))
    }

    /// `grid` drawn as [`drawn`] draws a symbol, an unknown module the
    /// grey between ink and paper, and its corners in the image.
    fn drawn_grid(grid: &ModuleGrid) -> (Luma, [[f64; 2]; 4]) {
        use crate::symbol::grid::Module;
        let (n, px, margin) = (grid.width(), 8usize, 4usize);
        let side = (n + 2 * margin) * px;
        let mut data = vec![0.92f32; side * side];
        for y in 0..side {
            for x in 0..side {
                let (mx, my) = ((x / px).wrapping_sub(margin), (y / px).wrapping_sub(margin));
                if mx < n && my < n {
                    match grid.get(mx, my) {
                        Module::Dark => data[y * side + x] = 0.08,
                        Module::Unknown => data[y * side + x] = 0.5,
                        _ => {}
                    }
                }
            }
        }
        let (near, far) = ((margin * px) as f64, ((margin + n) * px) as f64);
        (Luma::new(side, side, data), [[near, near], [far, near], [far, far], [near, far]])
    }

    /// A version 2 placement at `corners`, standing on finder patterns
    /// at the photo points `finders`, each named as the finder of the
    /// corner it lies nearest and that corner taken as measured by it.
    fn on_finders(corners: [[f64; 2]; 4], finders: &[[f64; 2]]) -> Detection {
        let sites = [[3.5, 3.5], [21.5, 3.5], [21.5, 21.5], [3.5, 21.5]];
        let nearest = |p: [f64; 2]| (0..4).min_by(|&i, &j| distance(corners[i], p).total_cmp(&distance(corners[j], p))).unwrap();
        Detection {
            corners,
            version: SymbolVersion::Qr(2),
            anchors: finders.iter().map(|&center| Anchor { kind: AnchorKind::Finder, at: sites[nearest(center)], center, partial: false, inferred: false }).collect(),
            measured_corners: finders.iter().map(|&center| nearest(center)).collect(),
            ..placed()
        }
    }

    #[test]
    fn a_placement_turned_on_its_symbol_is_seated_the_right_way_up_and_reads() {
        let text = "the corner without a finder";
        let (image, square) = drawn(text);
        let (params, opts) = (SampleParams::default(), DecodeOptions::default());
        let [tl, tr, br, bl] = square;
        // The centres of the symbol's three finder patterns, 3.5 modules
        // in from its corners.
        let centre = |corner: [f64; 2], [dx, dy]: [f64; 2]| [corner[0] + 28.0 * dx, corner[1] + 28.0 * dy];
        let (top_left, top_right, bottom_left) = (centre(tl, [1.0, 1.0]), centre(tr, [-1.0, 1.0]), centre(bl, [1.0, -1.0]));
        let read = |found: Detection| {
            let prepared = prepare(&image, found, &params).expect("an outline");
            let variant = Variant { kernel: params.kernel, rule: params.rule(), unsure_kept: false };
            read_prepared(&image, &prepared, variant, &params, Reads::new(&opts)).expect("a decode")
        };

        // The right way up, a placement is left as it is, and reads.
        let upright = on_finders(square, &[top_left, top_right, bottom_left]);
        assert_eq!(seat(&image, upright.clone(), &params), upright);
        assert_eq!(read(upright.clone()).text(), text);

        // The same square a quarter turn out, standing on two of the
        // three finders as the detector stands one: its top-left corner
        // at the symbol's top right, a finder expected at the corner
        // that has none, and the third finder lying where no pattern of
        // the placement is. As placed it solves nothing.
        let turned = Detection { turns: 1, ..on_finders([tr, br, bl, tl], &[top_right, top_left]) };
        assert_eq!(turned.anchors.iter().map(|a| a.at).collect::<Vec<_>>(), [[3.5, 3.5], [3.5, 21.5]]);
        let as_placed = read(turned.clone());
        assert!(!as_placed.is_trustworthy());
        assert!(as_placed.analysis.orientation.is_some(), "the grid shows the turn");

        // Seated, it is the symbol's own outline: each finder named as
        // the finder it is, the corners they measured still theirs, the
        // turn undone, and nothing mirrored.
        let upright_again = seat(&image, turned.clone(), &params);
        assert_eq!(upright_again.corners, square);
        assert_eq!((upright_again.turns, upright_again.mirrored, upright_again.mirrored_by_decode), (0, false, false));
        let named: Vec<([f64; 2], [f64; 2])> = upright_again.anchors.iter().map(|a| (a.at, a.center)).collect();
        assert_eq!(named, [([21.5, 3.5], top_right), ([3.5, 3.5], top_left)]);
        assert_eq!(upright_again.measured_corners, [1, 0]);
        assert_eq!(upright_again.agreement.0, upright_again.agreement.1, "{:?}", upright_again.agreement);
        assert_eq!(upright_again.info.format.valid, Some(true));
        let result = read(upright_again.clone());
        assert!(result.is_trustworthy());
        assert_eq!(result.text(), text);
        assert_eq!(seat(&image, upright_again.clone(), &params), upright_again, "seated once is seated");

        // The square flipped left to right, as a placement taken for the
        // symbol seen from behind has it, fits as a reflection and as a
        // turn alike; the format bits read from the front, and it is
        // seated from the front.
        let flipped = Detection { mirrored: true, ..on_finders([tr, tl, bl, br], &[top_right, top_left]) };
        let from_the_front = seat(&image, flipped, &params);
        assert_eq!((from_the_front.corners, from_the_front.mirrored), (square, false));

        // An anchor that lands where the version has no pattern of its
        // kind has nothing to be refined against, and goes: here a
        // finder the turned placement stood at the corner without one.
        let mut stray = turned.clone();
        stray.anchors.push(Anchor { kind: AnchorKind::Finder, at: [21.5, 3.5], center: centre(br, [-1.0, -1.0]), partial: false, inferred: false });
        assert_eq!(seat(&image, stray, &params).anchors.len(), 2);

        // Offered beside the upright placement, the turned one comes to
        // lie on it and is read once.
        let offered = seated(&image, vec![upright.clone(), turned], &params);
        assert_eq!(offered, [upright]);
    }

    #[test]
    fn a_decode_under_the_format_a_brute_force_picked_is_charged_for_the_pick() {
        // A version 3-Q symbol with both copies of its format bits
        // unreadable, and every padding codeword of the second of its two
        // blocks and four of that block's error correction codewords too:
        // its level and mask are the pick of a brute force of every one,
        // which the first block bears out, and the second, with more
        // codewords unknown than its eighteen of error correction, solves
        // only with the padding pinned. That decode is made under the
        // level and mask the brute force picked, and is the pick of as
        // many as the decode it follows, besides a read after the first.
        // It was once charged as a read after the first alone, and stood
        // five bits above the brute force it was made under.
        use crate::decode::pipeline::formats;
        use crate::symbol::EcLevel;
        use crate::symbol::grid::Module;
        let text = "hi";
        let version = SymbolVersion::Qr(3);
        let func = Arc::new(FunctionMap::new(version));
        let layout = SymbolLayout::new(func.clone(), EcLevel::Q).unwrap();
        assert_eq!(layout.blocks.len(), 2);
        let mut grid = symbol_of(3, qrcode_generator::qr::ErrorCorrection::Quartile, text);
        let unread = |grid: &mut ModuleGrid, at: &[crate::symbol::Pos]| {
            for at in at {
                grid.set(at.x as usize, at.y as usize, Module::Unknown);
            }
        };
        for copy in &func.format {
            unread(&mut grid, &copy.bits);
        }
        // The block's data after the first two codewords, "hi" lying in
        // the stream's first four, is padding.
        let second = &layout.blocks[1];
        for &cw in second.codewords[2..second.data_len].iter().chain(&second.codewords[second.data_len..second.data_len + 4]) {
            unread(&mut grid, layout.codeword_modules(cw));
        }
        let (image, square) = drawn_grid(&grid);
        let n = version.width() as f64;
        let at_px = |[x, y]: [f64; 2]| [square[0][0] + 8.0 * x, square[0][1] + 8.0 * y];
        let anchor = |kind: AnchorKind, at: [f64; 2]| Anchor { kind, at, center: at_px(at), partial: false, inferred: false };
        let found = Detection {
            corners: square,
            version,
            anchors: vec![
                anchor(AnchorKind::Finder, [3.5, 3.5]),
                anchor(AnchorKind::Finder, [n - 3.5, 3.5]),
                anchor(AnchorKind::Finder, [3.5, n - 3.5]),
                anchor(AnchorKind::Alignment, [n - 6.5, n - 6.5]),
            ],
            ..placed()
        };
        // Read against one threshold, and unsharpened, a grey module is
        // unknown whatever lies beside it.
        let params = SampleParams { threshold: Some(0.5), local: false, sharpen_modules: false, ..SampleParams::default() };
        let opts = DecodeOptions::default();
        let prepared = prepare(&image, found, &params).expect("an outline");
        let variant = Variant { kernel: params.kernel, rule: params.rule(), unsure_kept: false };
        let read = read_prepared(&image, &prepared, variant, &params, Reads::new(&opts)).expect("a decode");
        assert!(!read.analysis.format.is_reliable());
        assert!(read.is_trustworthy() && !read.pins.is_empty(), "{:?}", read.report.blocks);
        assert_eq!(read.text(), text);
        let models = 1 + usize::from(version.other_model().is_some());
        let after = Reads::new(&opts).one_of(2 * models * (1 + FEEDBACK_ROUNDS)).again;
        assert_eq!(read.report.tries, after.tries * formats(&func));
    }

    #[test]
    fn a_format_word_read_as_its_neighbour_is_settled_by_the_blocks() {
        use crate::decode::bch::{format_word_for, format_words};
        use crate::symbol::EcLevel;
        use qrcode_generator::{Segment, qr};
        // The words of the format code lie seven bits apart, or eight.
        // With all but three of the bits between the symbol's word and
        // another misread, the same in both copies, what is read lies
        // three from the other word, which the code corrects, and four
        // or five from the symbol's own: the bits settle the other
        // beyond doubt, and under its level and mask no block solves.
        // The symbol's own word is among the nearest of the rest, and
        // its blocks bear it out.
        let (version, ec, mask) = (SymbolVersion::Qr(2), EcLevel::L, 3);
        let text = "format words lie seven bits off";
        let symbol = qr::Encoder::new(qr::ErrorCorrection::Low)
            .version(qr::Version::new(2).unwrap())
            .mask(qr::Mask::new(mask).unwrap())
            .boost_error_correction(false)
            .encode_segments(&[Segment::bytes(text)])
            .unwrap();
        let clean = ModuleGrid::from_bools(symbol.width(), symbol.height(), symbol.modules());
        let func = Arc::new(FunctionMap::new(version));
        let opts = DecodeOptions::default();
        let misread = |flips: &[usize]| {
            let mut grid = clean.clone();
            for copy in &func.format {
                for &b in flips {
                    let (x, y) = (copy.bits[b].x as usize, copy.bits[b].y as usize);
                    grid.set(x, y, grid.get(x, y).inverted());
                }
            }
            grid
        };
        let own = format_word_for(version, ec, mask).unwrap();
        let mut neighbours = 0;
        for (fields, word) in format_words(version) {
            let differing: Vec<usize> = (0..15).filter(|b| (own ^ word) >> b & 1 != 0).collect();
            if !matches!(differing.len(), 7 | 8) {
                continue;
            }
            neighbours += 1;
            // Each run of three of the differing bits left as they
            // were, and the rest misread.
            for kept in 0..differing.len() - 2 {
                let flips: Vec<usize> = (0..differing.len()).filter(|i| !(kept..kept + 3).contains(i)).map(|i| differing[i]).collect();
                let grid = misread(&flips);
                let analysis = analyze(&grid, &func);
                assert!(analysis.format.is_reliable(), "{fields:?} {flips:?}");
                assert_eq!(analysis.format.best().map(|best| best.fields), Some(fields), "{flips:?}");
                let settled = decode(&grid, &SymbolLayout::new(func.clone(), fields.ec).unwrap(), fields.mask, &Pins::new(), &opts);
                assert_eq!(settled.blocks_verified(), 0, "{fields:?} {flips:?}");
                let report = decode_as(&grid, &func, &analysis, &opts).unwrap();
                assert!(report.is_trustworthy(), "{fields:?} {flips:?}");
                assert_eq!((report.ec, report.mask, report.text.plain().as_str()), (ec, mask, text), "{fields:?} {flips:?}");
            }
        }
        assert_eq!(neighbours, 30);

        // A word read clean has no other within reach, and a symbol whose
        // blocks are past correcting is decoded under it once and left.
        let mut torn = clean.clone();
        for y in 9..func.height {
            for x in 9..func.width {
                if func.expected(x, y).is_none() && (x + y) % 2 == 0 {
                    torn.set(x, y, torn.get(x, y).inverted());
                }
            }
        }
        let analysis = analyze(&torn, &func);
        assert!(analysis.format.is_reliable());
        assert!(rival_decode(&torn, &func, &analysis.format, &opts).is_none());
        let report = decode_as(&torn, &func, &analysis, &opts).unwrap();
        assert_eq!((report.ec, report.mask, report.blocks_verified()), (ec, mask, 0));

        // An M1 symbol's two check codewords are sixteen bits whatever
        // the mask. With the bits misread toward another mask's word,
        // its own mask is among the rivals, and its block bears it out
        // by those sixteen bits less a bit for each doubling of the
        // words tried, each a chance of noise passing. It was once left
        // unread, the blocks of a rival held to three codewords' worth.
        use qrcode_generator::micro;
        let m1 = micro::Encoder::new(micro::ErrorCorrection::DetectionOnly)
            .version(micro::Version::M1)
            .mask(micro::Mask::new(1).unwrap())
            .boost_error_correction(false)
            .encode_segments(&[Segment::numeric("12345").unwrap()])
            .unwrap();
        let clean = ModuleGrid::from_bools(m1.width(), m1.height(), m1.modules());
        let version = SymbolVersion::Micro(1);
        let func = Arc::new(FunctionMap::new(version));
        let analysis = analyze(&clean, &func);
        let read = decode_as(&clean, &func, &analysis, &opts).unwrap();
        assert!(read.is_trustworthy() && read.text.plain() == "12345" && (read.ec, read.mask) == (EcLevel::L, 1));
        let own = format_word_for(version, EcLevel::L, 1).unwrap();
        let mut others = 0;
        for (fields, word) in format_words(version).into_iter().filter(|(fields, _)| fields.version == version && fields.mask != 1) {
            let differing: Vec<usize> = (0..15).filter(|b| (own ^ word) >> b & 1 != 0).collect();
            let mut grid = clean.clone();
            for &b in &differing[3..] {
                let (x, y) = (func.format[0].bits[b].x as usize, func.format[0].bits[b].y as usize);
                grid.set(x, y, grid.get(x, y).inverted());
            }
            let analysis = analyze(&grid, &func);
            assert!(analysis.format.is_reliable(), "{fields:?}");
            assert_eq!(analysis.format.best().map(|best| best.fields.mask), Some(fields.mask));
            let rival = rival_decode(&grid, &func, &analysis.format, &opts).unwrap_or_else(|| panic!("{fields:?}"));
            assert!(rival.is_trustworthy() && rival.text.plain() == "12345" && (rival.ec, rival.mask) == (EcLevel::L, 1), "{fields:?}");
            let risk = rival.blocks[0].outcome.risk_bits();
            assert!((16 - tries_charge(RIVALS + 1)..16).contains(&risk), "{fields:?}: {risk}");
            assert!(decode_as(&grid, &func, &analysis, &opts).unwrap().is_trustworthy(), "{fields:?}");
            others += 1;
        }
        assert_eq!(others, 3);
    }
}
