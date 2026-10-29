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
//! - the placement again expecting the bleed its finders show, when no
//!   other read of it is beyond doubt ([`super::detect::bled_placement`]):
//!   on an over-exposed photo whose dark has thinned, or a print whose
//!   ink has spread, the fixed patterns are fitted as printed to one side
//!   of themselves, by as much as the bleed, and the grid with them;
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
//! - the photo about the outline deblurred ([`super::deblur`]): the blur
//!   it shows measured off its own edges, how far and which way, and
//!   undone, when it spreads an edge by a fifth of a module or more,
//!   whatever the modules' size. The masks of the copy above reach a
//!   pixel or two, and are tried only for small modules; a photograph
//!   out of focus by eight pixels blurs modules of seventeen by half of
//!   one, and one shaken smears them along the way the hand moved.
//!
//! Every placement is read anchored besides on patches of its data
//! ([`super::patches`]), the first as much as any retry: far from every
//! fixed pattern, the grid is moved a little each way to where the data
//! modules about each point of a lattice read most crisply, once the
//! corners and the patterns' anchors are refined. On a large symbol on
//! curled paper the cells between the fixed patterns read a fraction of a
//! module off, which no pattern says. The patches were once placed only
//! when every other read had failed, as the last retry of all; placed on
//! every read, they read one more symbol of BoofCV's benchmark, which
//! had been found and not read, and lost none. What the reads make of
//! their modules moved both ways: on curled paper fewer read wrong, and
//! on a few large symbols, which the patterns place well enough, more,
//! the patches of damaged data settling for a read the search had made
//! before with fewer modules wrong. On a photo blurred or dim enough that
//! its data reads crisply nowhere, the patches are placed by noise, and
//! two symbols of the benchmark that read through the corners marked for
//! them did not read through their patches; so a placement whose patched
//! reads settle nothing is read without its patches too, before any
//! other retry ([`read_placement`]).
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
//! and when they read better without it the anchor misleads. Where no
//! anchor is, the known modules place new ones: the blocks that verified
//! lie all over the symbol, interleaved, and a patch of their modules a
//! few modules across says where the grid runs there as an alignment
//! pattern would. BoofCV,
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
//! Each is charged by its place in that order, the first the least
//! ([`Reads`]).
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

use std::sync::{Arc, OnceLock};

use super::detect::{
    Anchor, AnchorKind, Climb, Detection, Lead, Retry, Search, bled_placement, detect_candidates_by, first_finder_bleed, leads_left,
    may_bleed, clockwise, outline_around, placed_by_hand, refine_anchors_with, seen_from_behind, turned_on_symbol,
};
use super::deblur::Spread;
use super::enhance::{Enhance, Enhanced, Sharpen};
use super::patches::with_patches;
use super::projection::{ControlPoint, Project, Projection};
use super::refine::{MOVES, Moved, Scorer, Shifted, Template, Warp, descend_by, fixed_threshold, refine_corners_expecting};
use super::sample::{Kernel, Rule, SampleParams, classify_in, read_modules, sample_modules_noting, suggest_versions};
use super::{Luma, rotate_outline_cw, unrotate_point_cw};
use crate::decode::bch::{FormatAnalysis, FormatCandidate, QUIRK_REACH};
use crate::decode::codewords::Pins;
use crate::decode::padding::recover_padding_after;
use crate::decode::likeness::{likeness_bits, stream_as_read};
use crate::decode::pipeline::{
    AttemptScore, DecodeOptions, DecodeReport, GridAnalysis, LIKENESS_FLOOR, analyze, brute_force, brute_force_preferring, decode,
};
use crate::decode::repair::known_modules;
use crate::symbol::function_map::FunctionMap;
use crate::symbol::{EcLevel, Family, SymbolVersion};
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
    /// The blur the copy undid, when it was deblurred
    /// ([`Sharpen::Deblur`]): in the pixels of the photo the read was
    /// made in, and what the photo's Deblur filter undoes to show the
    /// copy again ([`super::adjust::Adjust::kernel`]). Measured over the
    /// copy, about the symbol, it may differ from the blur the whole
    /// photo shows.
    pub deblurred: Option<Spread>,
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
/// on instead. The light they are read in is the image's
/// ([`Luma::light`]).
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

    /// Whether this read is the better of two, by where each stands
    /// ([`Self::standing`]).
    fn outranks(&self, other: &ReadResult) -> bool {
        self.standing() > other.standing()
    }

    /// Where this read stands among others: first what it has to show
    /// besides its decode ([`standing_of`]), whether a block of it is
    /// confirmed and, for a read with none, how far its fixed patterns
    /// read right beyond what chance gives; and then the decode, as the
    /// module describes. What a decode with not a block confirmed has to
    /// show is nothing a block of noise does not show as often. A version
    /// two modules smaller, laid over a symbol, reads its codewords from
    /// modules the grid straddles, and fills those it erases with nothing
    /// left over to check them by: a block "solved" so has no confirming
    /// bit, and any input would have solved. Its stream, read through bits
    /// no block confirmed, parses as often as a short stream of noise
    /// does. Of eighty version 2 symbols of random data, each read through
    /// an outline round it, sixty-five came back as version 1 that way
    /// with the decode judging, most with a fit of 120 to 140 of 203 fixed
    /// modules beside version 2's 236 of 236.
    ///
    /// One key for every read, and not a rule for each pair of them: with
    /// the fit judging between reads of two versions and the decode
    /// between reads of one, three reads can each outrank the next round
    /// in a ring, and which is kept then hangs on the order they come in.
    /// A read of version 2 that fits outright gives way to a worse-fitting
    /// one of version 2 whose stream parses, and that to a read of version
    /// 1 that fits better than it, though not as well as the first. A
    /// block confirmed outweighs any fit, as it does where the app ranks
    /// the reads of each family that did not decode.
    fn standing(&self) -> (bool, i64, AttemptScore) {
        standing_by(&self.report.score, self.agreement)
    }
}

/// Where a read whose decode scored `score`, of a placement whose fixed
/// modules read `agreement`, stands ([`ReadResult::standing`]): for a
/// decode to be weighed against another of the same placement before
/// either is made a read.
fn standing_by(score: &AttemptScore, agreement: (usize, usize)) -> (bool, i64, AttemptScore) {
    let (confirmed, fit) = standing_of(score.blocks_verified, agreement);
    (confirmed, if confirmed { 0 } else { fit }, *score)
}

/// What a read has to show for itself besides its decode, from the
/// blocks of it that verified and its (matching, fixed) agreement:
/// whether any block verified, and then how far its fixed modules read
/// right beyond what chance gives ([`fit_beyond_chance`]), which is how a
/// read of one version is told from a read of another when neither has a
/// block confirmed. The blocks are counted as the decode's score counts
/// them ([`AttemptScore::blocks_verified`]): none where every block
/// "verified" and not one segment header parsed, the false solve of a
/// wrong placement. A read ranks its own by it ([`ReadResult::standing`]),
/// and the app the reads of each family that did not decode.
pub fn standing_of(blocks_verified: usize, agreement: (usize, usize)) -> (bool, i64) {
    (blocks_verified > 0, fit_beyond_chance(agreement))
}

/// How far a placement's (matching, fixed) agreement over its version's
/// fixed modules reads right beyond what chance gives, for choosing
/// between placements of different sizes. Chance is no coin toss here:
/// the placement of a symbol of the wrong family or version has been
/// pushed to the best fit the outline allows, and over the test
/// photographs a QR Code laid on a Micro QR symbol reads as many as seven
/// in ten of its fixed modules right, though never three in four. So a
/// fixed module that reads wrong is set against two that read right: a
/// fit of two in three nets nothing, however many fixed modules the
/// version has, where a small symbol that fits nets nearly all of its
/// own, and a larger one that fits, damaged as it may be, more than any
/// part of it read as a symbol of its own.
fn fit_beyond_chance((matching, fixed): (usize, usize)) -> i64 {
    3 * matching as i64 - 2 * fixed as i64
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
/// and the best read is kept:
/// each is charged for its share of the chances of every read the
/// search may make, and a bit more for the share the read asked for
/// leaves the rest.
///
/// The share goes by place. The ways each turn of the search tries,
/// its placements, the samplings and retries of a placement, the ways
/// of seeing the photo, are taken in an order fixed before any of them
/// is read, the likeliest first, and half the turn's chances go to them
/// by their place, the `k`th, from one, having 1/(k(k+1)), as padding
/// recovery shares its passes ([`crate::decode::padding::pass_tries`]),
/// and the last what the others leave, 1/n of `n`; the other half goes
/// to them evenly, 1/n each ([`by_place`]). The shares of a turn's ways
/// add to one, and a read's share is the product of its ways' down the
/// turns above it, so the shares of every read the search may make add
/// to one too, and to a half past the read asked for: the chance that
/// noise passes in any of those is no more than half the chance in the
/// one read that is charged nothing. That holds whichever of them are
/// made, though what a way showed decides which turns are taken beneath
/// it, since the ways a turn takes never have more than its whole; and a
/// read that settles the search leaves the rest unmade, which spends
/// less. Charged evenly, every read past the first was one of every
/// read the search might make, some 2^15 of them past a placement's
/// samplings and retries and the decodes of each, and the threshold
/// rule tried second, which reads more symbols than any other retry,
/// paid as much as the fortieth: a block needed about two codewords
/// more to spare than at the first read. By place the ways tried first
/// pay a few bits less, and none more than a bit over its even share at
/// any turn. Within one sampling the decodes it makes, as the other
/// model, refitted and with the padding pinned, follow from one another
/// rather than stand in turn, and stay even.
///
/// Those reads were once charged in part, the placements and samplings
/// counted, the read asked for among them, and the rest uncounted; and
/// then, past the read asked for, the first way tried at each turn
/// beneath went uncounted for that turn's others, as though it were the
/// read asked for, and every sampling of a turned photo was counted as
/// the only one.
#[derive(Clone, Copy, Debug)]
struct Reads {
    first: DecodeOptions,
    again: DecodeOptions,
    /// `first` is the read asked for, and goes uncounted at each turn.
    /// Past it, `first` is one of the others, counted as `again` is.
    asked: bool,
}

/// What the way `k` places, counting from nought, of the `n` a turn of
/// the search tries in its fixed order is counted as, as
/// [`DecodeOptions::one_of`] counts it: one over its share of the turn's
/// chances, rounded up, the share half by place and half even
/// ([`Reads`]). Four at most for the first, two bits; less than twice
/// `n` for any, at most a bit over its even share; and `n` for the last.
fn by_place(k: usize, n: usize) -> usize {
    let n = n.max(1);
    debug_assert!(k < n, "way {k} of {n}");
    if k + 1 >= n {
        return n;
    }
    let by_place = (k + 1) * (k + 2);
    (2 * n * by_place).div_ceil(n + by_place)
}

impl Reads {
    fn new(opts: &DecodeOptions) -> Self {
        Self { first: *opts, again: opts.one_of(2), asked: true }
    }

    /// These reads at a turn of the search that tries `n` ways, with the
    /// budgets of the dear solvers shared over them all
    /// ([`DecodeOptions::sharing`]), and nothing counted: each way is
    /// counted by its place ([`Self::nth_of`]).
    fn sharing(self, n: usize) -> Self {
        Self { first: self.first.sharing(n), again: self.again.sharing(n), ..self }
    }

    /// The reads of the way `k` places, counting from nought, of the `n`
    /// a turn of the search tries in an order fixed before any of them is
    /// read: the first read of the first going on as asked where these
    /// are the read asked for, and every other counted as one of as many
    /// as its place's share of the turn's chances is one of
    /// ([`by_place`]).
    fn nth_of(self, k: usize, n: usize) -> Self {
        let again = self.again.one_of(by_place(k, n));
        let asked = self.asked && k == 0;
        Self { first: if asked { self.first } else { again }, again, asked }
    }

    /// These reads at a turn of the search that tries `n` ways, every way
    /// alike, the first of them going on as asked where these are the
    /// read asked for: every other read counted as one of them, and the
    /// budgets whole.
    fn one_of(self, n: usize) -> Self {
        let again = self.again.one_of(n);
        Self { first: if self.asked { self.first } else { again }, again, ..self }
    }

    /// These reads of a placement that was itself the pick of `n` decodes,
    /// the first read too: a decode chose it, and noise that passed that
    /// decode passes the read of the same pixels as often as not.
    fn picked_among(self, n: usize) -> Self {
        Self { first: self.first.one_of(n), again: self.again.one_of(n), ..self }
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
    // the photo as it is, and the rungs, each once, and each charged by
    // its place among them ([`Reads::nth_of`]). The way asked for was
    // once counted again where it is a rung.
    let mut ways = vec![asked];
    for way in std::iter::once(Enhance::NONE).chain(Enhance::rungs(region)).chain([Enhance::deblurred(region)]) {
        if !ways.contains(&way) {
            ways.push(way);
        }
    }
    let way = |enhance: Enhance| reads.nth_of(ways.iter().position(|&way| way == enhance).unwrap_or(ways.len() - 1), ways.len());
    let mut best = read_outline_through(image, version, region, params, way(asked), asked);
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
    if worth_deblurring(image, region, best.as_ref()) {
        rungs.push(Enhance::deblurred(region));
    }
    rungs.retain(|rung| *rung != asked);
    work::in_turn(
        &rungs,
        |&rung| if work::cancelled() { None } else { read_outline_through(image, version, region, params, way(rung), rung) },
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

/// Whether a read of `region` that did not decode is worth making again
/// through the photo deblurred ([`Enhance::deblurred`]): the photo about
/// it spreads an edge by more than [`super::deblur::LEAST_SPREAD`]
/// pixels and by a fifth of a module or more
/// ([`super::deblur::WORTH_UNDOING`]), the modules as `best`
/// places them, or, with nothing found, as a symbol of twenty-five
/// across the outline would have them. Unlike the masks, this does not
/// wait on the modules being small: a photograph out of focus by eight
/// pixels blurs modules of seventeen by half of one, and its finder
/// patterns' rings run into their stones whatever their size. The blur
/// is measured as the copy deblurred measures it
/// ([`super::deblur::spread_of`]): over a window of more than
/// [`super::deblur::MOST_PIXELS`], in the parts of it with the most
/// edges, where measured whole the grain of the paper about a symbol
/// sets what an edge is judged against.
fn worth_deblurring(image: &Luma, region: [[f64; 2]; 4], best: Option<&ReadResult>) -> bool {
    let Some([x0, y0, x1, y1]) = Enhanced::window_about(image, region) else { return false };
    let (w, h) = (x1.saturating_sub(x0), y1.saturating_sub(y0));
    let part = super::adjust::cut(&image.data, image.width, 1, [x0, y0, x1, y1]);
    let Some(spread) = super::deblur::spread_of(&part, w, h, image.light) else { return false };
    let (widest, _) = spread.extremes();
    let module = match best {
        Some(read) => {
            let (c, version) = (read.detection.corners, read.detection.version);
            ((distance(c[0], c[1]) + distance(c[3], c[2])) / 2.0 / version.width().max(1) as f64)
                .min((distance(c[0], c[3]) + distance(c[1], c[2])) / 2.0 / version.height().max(1) as f64)
        }
        None => distance(region[0], region[1]).min(distance(region[0], region[3])) / 25.0,
    };
    // Under a fifth of a module the masks of the other rungs reach what
    // there is, where the modules are small enough for them, and a symbol
    // of large modules so little blurred that does not decode is damaged.
    widest >= super::deblur::LEAST_SPREAD && widest >= super::deblur::WORTH_UNDOING * module
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
    let plain = SampleParams { binarizer: plain.binarizer.scaled(view.mean_factor()), ..plain };
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
    result.deblurred = view.luma.deblurred.map(|spread| spread.stretched(view.factor.map(|factor| 1.0 / factor)));
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
    // Each read is one of the placements and turns of the photo, each
    // charged by its place among them, best fit first and the turns
    // last, times the ways of sampling each.
    let placings = candidates.len() + 3;
    let all = reads.sharing(placings);
    let placements: Vec<(&Detection, Reads)> = candidates.iter().enumerate().map(|(k, found)| (found, all.nth_of(k, placings))).collect();
    let reads_of = |&(found, counted): &(&Detection, Reads)| {
        let mut reads = Vec::new();
        if work::cancelled() {
            return reads;
        }
        // A render read off its pixels on the second of its readings, the
        // first decoded and failed, is the pick of both decodes
        // ([`Detection::decoded_charge`]).
        let counted = counted.picked_among(found.decoded_charge());
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
        // Charged for its pick as a placement in the photo as it is is.
        let charge = found.decoded_charge();
        let Some(prepared) = prepare(&turned, seat(&turned, found, params), params) else { return reads };
        // A mirror image a decode chose is the pick of two decodes, as
        // [`read_placement`] charges it; read through the turn it was once
        // charged as the photo's own pick alone, and passed noise twice as
        // often as the same placement in the photo as it is.
        let charge = if prepared.detection.mirrored_by_decode { charge.saturating_mul(2) } else { charge };
        // Read as a placement in the photo as it is is read, and, short of
        // a read that settles the search, without its patches of data the
        // ways that lead with the kernel asked for ([`read_placement`]):
        // each way charged by its place among them.
        let patched = prepared.detection.anchors.len() != prepared.plain.anchors.len();
        let few = &variants[..variants.len().min(params.rules().count())];
        let ways = variants.len() + usize::from(patched) * few.len();
        let turned_reads = all.nth_of(candidates.len() + usize::from(quarters) - 1, placings).picked_among(charge).sharing(ways);
        let mut keep = |result: Option<ReadResult>| {
            let Some(mut result) = result else { return false };
            unturn(&mut result.detection, quarters, image.width, image.height, region);
            let settled = result.is_trustworthy();
            reads.push(result);
            settled
        };
        let samplings: Vec<(usize, Variant)> = variants.iter().copied().enumerate().collect();
        let read = |prepared: &Prepared, variant: Variant, k: usize| {
            if work::cancelled() { None } else { read_prepared(&turned, prepared, variant, params, turned_reads.nth_of(k, ways)) }
        };
        let settled = work::first_then_in_turn(&samplings, |&(k, variant)| read(&prepared, variant, k), &mut keep);
        if patched && !settled && !work::cancelled() {
            let without = Prepared::new(prepared.plain.clone(), prepared.agreement);
            for (j, &variant) in few.iter().enumerate() {
                if keep(read(&without, variant, variants.len() + j)) {
                    break;
                }
            }
        }
        reads
    };
    // Side by side, as far as the photo's size leaves room for turned
    // copies of it.
    work::in_turn_by(image.copies_at_once(), &[1, 2, 3], reads_turned, |reads| reads.into_iter().any(|result| consider(&mut best, result)));
    best
}

/// The ways of sampling a placement a read tries, in order: the kernel
/// and threshold rule asked for, then the other rules with that kernel
/// ([`SampleParams::rules`]); then, when the unsure band would leave any
/// module unknown, the kernel asked for under every rule again with those
/// modules kept at the colour they fell on
/// ([`Classified::to_grid_as_decided`]); then each other kernel under
/// every rule. Every way is in the light the photo was worked in
/// ([`super::sample::Light`]), and none in the other, which a read made
/// so would not be the photo read again. The rule comes before the
/// kernels because it decides more often: see the module's account. The
/// unsure modules come between because keeping them changes the least of
/// what is read: a module in the band is unknown to the exact decoder
/// either way, as a codeword it erases or as a bit the fallbacks doubt
/// before any other, and what differs is that the erasure spends a unit
/// of redundancy on the whole codeword where the doubt spends an equation
/// on the bit. Over blocks of simulated noise the band read no more than
/// the colours kept, and with a stain beside the noise far less; on the
/// photographs it is tried second, since what the band leaves unknown on
/// them is often a module the grid straddles, which reads as nothing in
/// either colour.
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
///
/// A placement nothing but a trial decode of the search vouched for is
/// read charged as one of what its place among the search's trial decodes
/// says ([`Detection::decoded_charge`]), whoever reads it: the search, a
/// read of every symbol found, or a pick from the list.
pub fn read_detection(image: &Luma, found: Detection, params: &SampleParams, opts: &DecodeOptions) -> Option<ReadResult> {
    let opts = &opts.one_of(found.decoded_charge());
    let (version, around) = (found.version, outline_around(found.corners, AROUND_FOUND));
    // A symbol the search found only in the photo sharpened or enlarged
    // is read as placed through that copy, where its modules read as the
    // search saw them, unless the photo was asked for some other way.
    let through = match found.retry {
        Some(Retry::Enhanced(enhance)) if params.enhance.is_none() => enhance,
        _ => params.enhance,
    };
    let mut best: Option<ReadResult> = None;
    // The placement found, read every way, then the search about it, each
    // read one of those of both by its place ([`Reads`]): of two, half
    // the chances each.
    let reads = Reads::new(opts);
    if read_found_through(image, found, around, params, reads.nth_of(0, 2), through, &mut best) || work::cancelled() {
        return best;
    }
    if let Some(again) = read_outline_as(image, version, around, params, reads.nth_of(1, 2))
        && best.as_ref().is_none_or(|held| again.outranks(held))
    {
        best = Some(again);
    }
    best
}

/// A search of the whole photo with the placements it kept aside for a
/// read ([`Search::unvouched`]) read through, side by side, as a symbol
/// the search found is ([`read_detection`]): each whose read is to be
/// trusted is a symbol found, placed as its read placed it and marked as
/// decoding, and each other is said as a lead, with the version it was
/// fitted as ([`Lead::fitted`]), ahead of the leads the search had, each
/// place once. Nothing vouched for such a placement but its fit, and a
/// read that does not decode leaves it no better a symbol than finders
/// with nothing fitted to them; one that decodes is the last word on
/// it, as on any. Returned beside the search are the reads that decoded,
/// each of a detection it lists and with what it was charged as one of
/// ([`DecodeOptions::one_of`]), for a caller that reads every symbol
/// found to take as they are rather than make again, or to read again
/// charged the same.
///
/// The placements are read in the order the search kept them, best fit
/// first, and none of them is a read asked for: each is a chance the
/// search took for noise to pass, and is charged by its place among them
/// as the ways a read tries are ([`by_place`]), so that together they
/// pass no oftener than one read would. They were once read uncharged,
/// each as a symbol the search found is, eight chances for one.
pub fn settle_unvouched(image: &Luma, search: Search, params: &SampleParams, opts: &DecodeOptions) -> (Search, Vec<(ReadResult, usize)>) {
    let Search { mut detections, leads, unvouched } = search;
    if unvouched.is_empty() {
        return (Search { detections, leads, unvouched }, Vec::new());
    }
    let placed: Vec<(usize, &Detection)> = unvouched.iter().enumerate().collect();
    let n = placed.len();
    let reads = work::map(&placed, |&(k, placement)| {
        if work::cancelled() { None } else { read_detection(image, placement.clone(), params, &opts.one_of(by_place(k, n))) }
    });
    let mut unread: Vec<Lead> = Vec::new();
    let mut settled: Vec<(ReadResult, usize)> = Vec::new();
    for (k, (placement, read)) in unvouched.into_iter().zip(reads).enumerate() {
        match read.filter(ReadResult::is_trustworthy) {
            Some(read) => {
                let detection = Detection { decodes: Some(true), ..read.detection.clone() };
                detections.push(detection.clone());
                settled.push((ReadResult { detection, ..read }, by_place(k, n)));
            }
            // The placement's own corners, as a lead's outline is the one
            // its finders propose: to sample through as they are, at the
            // version fitted. Widened as a read searches about a placement,
            // they sampled a tenth off at every side.
            None => unread.push(Lead {
                family: placement.version.family(),
                outline: placement.corners,
                finders: placement.anchors.iter().filter(|a| a.kind == AnchorKind::Finder).map(|a| a.center).collect(),
                fitted: Some(placement.version),
            }),
        }
    }
    unread.extend(leads);
    (Search { leads: leads_left(unread, &detections), detections, unvouched: Vec::new() }, settled)
}

/// Placements by hand a read tries: the versions and turns of the
/// corners that fit best ([`suggest_versions`]). A version a module or
/// two off its symbol's reads half its fixed modules right, as does the
/// right version a quarter turn out, and the symbol's own, the right way
/// up, is among the first three wherever the corners lie within a
/// module of it.
const PLACED_VERSIONS: usize = 3;

/// Read a symbol through its own four corners, given as a hand places
/// them in the Align stage, with no search: at each of the versions and
/// turns of the corners that fit best, the version the symbol's own
/// version information names first, where it names one
/// ([`super::detect::version_named_beside_finders`]), the placement
/// refined, sampled every way, decoded and refitted as a placement the
/// search found is ([`read_found_through`]), and read again through the
/// photo enlarged and sharpened when its modules are small, as an
/// outline is. Where it does not decode it is not looked for afresh, as
/// a placement the search found is: what it reads is what the corners
/// give. `None` when the corners make no outline at any version.
///
/// The corners may start at any corner of the symbol and run round it
/// either way: they are put clockwise first, as the symbol's own run,
/// so that every version is fitted to the symbol seen from the front and
/// not to its transpose, which for a rectangular symbol lays its long
/// side along its short one. A placement a turn out is seated on its
/// symbol when its fixed patterns read so ([`seat`]), but not when a blur
/// leaves them reading no better turned: a symbol smeared by a hand that
/// moved, placed from its top-right corner, was read as it lay and solved
/// nothing, and read through the same corners from its top-left at once.
/// So every turn of the corners is fitted, a rectangular symbol's
/// quarter turns among them, which start the corners on a short side,
/// and the turns are ranked with the versions.
///
/// Through the corners a benchmark marks by hand, this tells a symbol the
/// search lost from one its reading lost ([`super::detect::placed_by_hand`]).
pub fn read_placed(image: &Luma, family: Family, corners: [[f64; 2]; 4], params: &SampleParams, opts: &DecodeOptions) -> Option<ReadResult> {
    let corners = clockwise(corners);
    let mut fits: Vec<(f32, SymbolVersion, [[f64; 2]; 4])> = Vec::new();
    let mut named: Vec<(f32, SymbolVersion, [[f64; 2]; 4])> = Vec::new();
    let mut formatted: Vec<(f32, SymbolVersion, [[f64; 2]; 4])> = Vec::new();
    for turn in 0..4 {
        let mut turned = corners;
        turned.rotate_left(turn);
        let guesses = suggest_versions(image, turned, family, params);
        // The version information is read through the few versions that
        // fit the turn best: any version near the symbol's own finds its
        // finders, and the word names the symbol's.
        let word = guesses
            .iter()
            .take(PLACED_VERSIONS)
            .find_map(|guess| super::detect::version_named_beside_finders(image, guess.version, turned, params));
        if let Some(version) = word
            && let Some(guess) = guesses.iter().find(|guess| guess.version == version)
        {
            named.push((guess.score(), version, turned));
        }
        // And the format information, which every version has: the best
        // fitting version of the few at a turn it spells a word at.
        if let Some(guess) = guesses.iter().take(PLACED_VERSIONS).find(|guess| super::detect::format_reads_through(image, guess.version, turned, params)) {
            formatted.push((guess.score(), guess.version, turned));
        }
        fits.extend(guesses.into_iter().map(|guess| (guess.score(), guess.version, turned)));
    }
    // The versions and turns the symbol's own version information names
    // come first, best fitting first; then the best fitting of those its
    // format information reads at; then the rest, by their fit. By the fit
    // alone, the right version a quarter turn out, which reads three
    // quarters of its fixed modules right, came before the symbol as it
    // lay where glare left it reading barely more: under glare, a symbol
    // of BoofCV's benchmark read through the corners marked for it, and
    // with the marks a twentieth of a pixel off, a quarter turn out. One
    // and no more: the format information lies about the first finder at
    // every version, and a turn whose modules there happen to spell a
    // word, blurred out of telling, spells it at every version, which
    // taken all together put the turn the fit chose out of reach.
    named.sort_by(|a, b| b.0.total_cmp(&a.0));
    formatted.sort_by(|a, b| b.0.total_cmp(&a.0));
    fits.sort_by(|a, b| b.0.total_cmp(&a.0));
    let listed = |list: &[(f32, SymbolVersion, [[f64; 2]; 4])], fit: &(f32, SymbolVersion, [[f64; 2]; 4])| list.iter().any(|n| n.1 == fit.1 && n.2 == fit.2);
    named.truncate(PLACED_VERSIONS);
    if let Some(first) = formatted.into_iter().find(|fit| !listed(&named, fit)) {
        named.push(first);
    }
    fits.retain(|fit| !listed(&named, fit));
    fits.truncate(PLACED_VERSIONS.saturating_sub(named.len()));
    named.truncate(PLACED_VERSIONS);
    named.extend(fits);
    let fits = named;
    let around = outline_around(corners, AROUND_FOUND);
    let ways: Vec<Enhance> = std::iter::once(params.enhance)
        .chain(Enhance::rungs(around).into_iter().chain([Enhance::deblurred(around)]).filter(|&rung| rung != params.enhance))
        .collect();
    // Every way at every fit.
    let reads = Reads::new(opts).one_of(fits.len() * ways.len());
    let mut best: Option<ReadResult> = None;
    let mut counted = reads.first_then_past();
    for (_, version, corners) in fits {
        let Some(found) = placed_by_hand(image, version, corners, params) else { continue };
        for (k, &way) in ways.iter().enumerate() {
            let reads = counted.next().expect("the reads go on");
            let worth = if way.sharpen == Sharpen::Deblur { worth_deblurring(image, around, best.as_ref()) } else { worth_enhancing(image, best.as_ref()) };
            if work::cancelled() || (k > 0 && !worth) {
                continue;
            }
            if read_found_through(image, found.clone(), around, params, reads, way, &mut best) {
                return best;
            }
        }
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
    let Some(view) = Enhanced::around(image, around, enhance) else {
        // Deblurred where the photo about the symbol shows no blur worth
        // undoing, the copy is the photo enlarged as `enhance` says and
        // no more, as the photo's Deblur filter leaves a photo it measures
        // no blur in: read through that. A search that found the symbol
        // in the photo deblurred measured the blur over the whole of it,
        // which about the symbol may be sharp; the placement was once not
        // read at all there, and the symbol looked for again about it.
        return enhance.sharpen == Sharpen::Deblur
            && read_found_through(image, found, around, params, reads, Enhance { sharpen: Sharpen::Off, ..enhance }, best);
    };
    found.corners = found.corners.map(|c| view.to_view(c));
    for anchor in &mut found.anchors {
        anchor.center = view.to_view(anchor.center);
    }
    let plain = SampleParams { binarizer: plain.binarizer.scaled(view.mean_factor()), ..plain };
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
/// The kernels and threshold rules in `variants` first, through the
/// placement anchored on patches of its data; then, where patches were
/// placed, the placement without them; then, for a
/// square symbol whose format bits read as a word, which the detector
/// therefore never tried the other way round, the placement in mirror
/// order, kept when it verifies more blocks than any read so far; then,
/// when no read has verified a block, the placement with its most
/// doubtful anchors dropped, one at a time, and its patches of data
/// placed again through the map without each. Those three are read under
/// the kernel asked for alone, with every threshold rule: what they
/// settle does not hang on the kernel. Then, when no read has settled the
/// search, the placement again expecting the bleed its finders show,
/// every way the placement was read, and without its patches of data the
/// ways it was read without them. The first sampling goes on as the
/// read asked for, and every other read is charged by its place among
/// all these that may be made, in this order, the best of which is kept
/// ([`Reads`]): the reads in mirror order and without anchors once went
/// uncounted, and then every read was charged as one of all of them
/// alike.
fn read_placement(
    image: &Luma,
    prepared: &Prepared,
    variants: &[Variant],
    params: &SampleParams,
    reads: Reads,
    keep: &mut dyn FnMut(ReadResult) -> bool,
) -> bool {
    let (found, plain) = (&prepared.detection, &prepared.plain);
    let few = variants.len().min(params.rules().count());
    // Whether the placement is read expecting a bleed is known only once
    // every other read has failed, and every read is counted before the
    // first is made. Measuring every finder's bleed costs more than a
    // read that decodes at once, 125 ms against 35 for a version 2 symbol;
    // but a symbol bleeds only where every finder does ([`symbol_bleed`]),
    // and one finder took 29. So those reads are counted where the first
    // finder shows a bleed, and the others are measured only once the
    // reads are reached ([`bled_placement`]). Counted wherever the
    // placement had finders enough, they charged every other read of it
    // more than half a bit, for a bleed few photographs show.
    let first_bleed = if may_bleed(plain) { first_finder_bleed(image, plain, params) } else { 0.0 };
    let patched = found.anchors.len() != plain.anchors.len();
    // The placement expecting the bleed has its patches of data placed
    // afresh, and whether any are is known only once it is made: its
    // reads without them are counted whether or not this placement's
    // were, as reads that may be made. Counted as this one's were, they
    // went uncounted where its patches placed none and the bled one's
    // placed some.
    let bled_ways = if first_bleed != 0.0 { variants.len() + few } else { 0 };
    // Where each way stands among those that may be tried, in the order
    // they are tried: the samplings, then without the patches of data, in
    // mirror order, without each doubted anchor, and expecting the bleed.
    let without_at = variants.len();
    let behind_at = without_at + usize::from(patched) * few;
    let doubted_at = behind_at + few;
    let bled_at = doubted_at + DOUBT_LIMIT * few;
    let ways = bled_at + bled_ways;
    // A placement whose mirror image a decode of it both ways round chose,
    // its format bits saying nothing ([`super::detect`]'s
    // `settle_mirror_by_decode`), is the pick of two decodes, and a block
    // of noise that verified mirrored chose it and verifies again in the
    // read. Read as asked, uncharged, it passed noise twice as often as a
    // read is let; its own retry in mirror order, below, is charged.
    let reads = if found.mirrored_by_decode { reads.picked_among(2) } else { reads };
    let reads = reads.sharing(ways);
    let at = |k: usize| reads.nth_of(k, ways);
    let mut verified = 0;
    let samplings: Vec<(usize, Variant)> = variants.iter().copied().enumerate().collect();
    let settled = work::first_then_in_turn(
        &samplings,
        |&(k, variant)| if work::cancelled() { None } else { read_prepared(image, prepared, variant, params, at(k)) },
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
    // The placement without its patches of data, which on a photo blurred
    // or dim enough that its data reads crisply nowhere are placed by
    // what noise there is, and bend the grid off modules the patterns
    // placed right: a version 7 symbol blurred by most of a module, and
    // a dim version 1, each read through the corners marked for it by
    // hand as the patterns placed it, and through its patches not at all.
    if patched {
        let without = Prepared::new(plain.clone(), prepared.agreement);
        for (j, &variant) in few.iter().enumerate() {
            if let Some(result) = read_prepared(image, &without, variant, params, at(without_at + j)) {
                verified = verified.max(result.report.blocks_verified());
                if keep(result) {
                    return true;
                }
            }
        }
    }
    if found.version.family() != Family::Rmqr
        && found.info.format.valid == Some(true)
        && let Some(detection) = seen_from_behind(image, found, params)
    {
        let behind = Prepared::new(detection, prepared.agreement);
        for (j, &variant) in few.iter().enumerate() {
            if let Some(result) = read_prepared(image, &behind, variant, params, at(behind_at + j))
                && result.report.blocks_verified() > verified
            {
                verified = result.report.blocks_verified();
                if keep(result) {
                    return true;
                }
            }
        }
    }
    // The anchors doubted are the patterns', and the patches of data,
    // which were placed through the map a doubtful anchor bent, are placed
    // again through the map without it.
    let mut doubted = plain.clone();
    for round in 0..DOUBT_LIMIT {
        // A block that verified knows which anchors mislead, and the
        // refit to it has already dropped them.
        if verified > 0 || work::cancelled() {
            break;
        }
        let Some(k) = doubtful_anchor(&doubted) else { break };
        doubted.anchors.remove(k);
        let without = Prepared::patched(image, doubted.clone(), prepared.agreement);
        for (j, &variant) in few.iter().enumerate() {
            if let Some(result) = read_prepared(image, &without, variant, params, at(doubted_at + round * few.len() + j)) {
                verified = verified.max(result.report.blocks_verified());
                if keep(result) {
                    return true;
                }
            }
        }
    }
    // Last, the placement expecting the bleed its finders show
    // ([`bled_placement`]), every way a placement is read. Over-exposed, the
    // dark of BoofCV's photograph of a version 2 symbol had thinned by a
    // third of a module at every boundary, and the patterns were fitted
    // a third of a module to one side of themselves, where every light
    // module between two dark ones read dark: it read nothing, and
    // zxing-cpp read it outright. It is read so only when no read before
    // it settled the search, though one may have confirmed a block or
    // more: the symbol is not placed expecting the bleed as it is found,
    // since a bleed is measured as one number, and ink that has spread
    // further on one side of a finder than the other, as on a creased
    // thermal receipt in the same benchmark, was placed better by the
    // fit as printed, and its symbol, which read so, no longer did.
    if bled_ways == 0 || work::cancelled() {
        return false;
    }
    let Some(bled) = bled_placement(image, plain, first_bleed, params).and_then(|bled| prepare(image, bled, params)) else { return false };
    // The first way alone before the rest, as the placement was read, so
    // that the levels and masks it ranks for them do not hang on which of
    // them came first ([`decode_as_ranked`]).
    if work::first_then_in_turn(
        &samplings,
        |&(k, variant)| if work::cancelled() { None } else { read_prepared(image, &bled, variant, params, at(bled_at + k)) },
        |result| result.is_some_and(&mut *keep),
    ) {
        return true;
    }
    // And without its patches of data, as the placement was read.
    if bled.detection.anchors.len() == bled.plain.anchors.len() || work::cancelled() {
        return false;
    }
    let without = Prepared::new(bled.plain.clone(), bled.agreement);
    let place = |j: usize| bled_at + variants.len() + j;
    few.iter().enumerate().any(|(j, &variant)| read_prepared(image, &without, variant, params, at(place(j))).is_some_and(&mut *keep))
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
    let mut grid = classify_in(&sampled, params, image.light).to_grid(func.width, func.height);
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
/// and anchored besides on patches of its data, ready to sample.
struct Prepared {
    /// The detection sampled: `plain` with the anchors the patches of its
    /// data place ([`with_patches`]).
    detection: Detection,
    /// The detection as refined, anchored on its patterns and edges and on
    /// no patch of data: what the anchors doubted are taken from, and the
    /// bleed is measured through ([`read_placement`]).
    plain: Detection,
    agreement: (usize, usize),
    /// The levels and masks that ranked best when the first grid sampled
    /// through it was brute forced, for the grids sampled after it
    /// ([`decode_as_ranked`]). Empty until a grid of it is.
    formats: OnceLock<Vec<(EcLevel, u8)>>,
}

impl Prepared {
    /// `detection` as it is, its patches of data placed already or none
    /// to be placed.
    fn new(detection: Detection, agreement: (usize, usize)) -> Self {
        Self { plain: detection.clone(), detection, agreement, formats: OnceLock::new() }
    }

    /// `plain` anchored besides on patches of its data, far from every
    /// anchor it has ([`with_patches`]), or as it is where no patch places
    /// one.
    fn patched(image: &Luma, plain: Detection, agreement: (usize, usize)) -> Self {
        let detection = with_patches(image, &plain).unwrap_or_else(|| plain.clone());
        Self { detection, plain, agreement, formats: OnceLock::new() }
    }
}

/// Refine a detection's corners, with the anchors held and the corners a
/// finder pattern measured kept, and then every anchor against its own
/// pattern, each expecting the bleed the detection was placed for
/// ([`Detection::bloom`]) at the polarity it was measured at
/// ([`Detection::held_polarity`]); and last anchor it besides on patches
/// of its data far from every anchor, through the map the refined ones
/// make ([`with_patches`]). `None` when the corners form no usable
/// outline. The detection is taken to lie the right way up on its symbol
/// ([`seat`]).
fn prepare(image: &Luma, mut found: Detection, params: &SampleParams) -> Option<Prepared> {
    let func = FunctionMap::new(found.version);
    let controls: Vec<ControlPoint> = found.anchors.iter().map(|a| a.control_point()).collect();
    let polarity = found.held_polarity();
    let refined = refine_corners_expecting(image, &func, found.corners, &controls, &found.held_corners(), found.bloom, polarity, params);
    let agreement = refined.as_ref().map_or(found.agreement, |r| r.after);
    found.corners = refined.map_or(found.corners, |r| r.corners);
    found.anchors = refine_anchors_with(image, &func, found.corners, &found.anchors, found.bloom, polarity, params);
    found.projection()?;
    Some(Prepared::patched(image, found, agreement))
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
    // Only the first way of sampling a placement, which is read alone
    // before the others, ranks the levels and masks for them.
    let first = variants_of(params).first() == Some(&variant);
    let mut report = decode_as_ranked(&grid, &func, &analysis, opts, &prepared.formats, first)?;
    // The format bits may spell a word of the other QR Code model where
    // none of this one is near: a Model 1 symbol found as Model 2, or the
    // reverse. The two lay their data out differently, so the grid is read
    // again as the other model, and the better-ranked read is kept, ranked
    // as reads are ([`ReadResult::standing`]): a block confirmed first.
    // By the decode's score alone, a stream of noise that parsed under the
    // other model outranked blocks confirmed under this one.
    let mut reinterpreted = None;
    if !report.is_trustworthy()
        && analysis.format.other_model_hint().is_some()
        && let Some(other) = found.version.other_model()
    {
        let other_func = Arc::new(FunctionMap::new(other));
        if let Some(again) = decode_as(&grid, &other_func, &analyze(&grid, &other_func), retry)
            && standing_by(&again.score, prepared.agreement) > standing_by(&report.score, prepared.agreement)
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
        deblurred: None,
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
/// for as long as each round outranks the read before it, ranked as
/// reads are ([`ReadResult::outranks`]). `params` carry the kernel and
/// threshold rule the read was made with. Ranked by the decode's score
/// alone, which puts a stream that parses before the blocks confirmed, a
/// refit that lost every block its read had confirmed, and whose stream
/// of noise parsed, took that read's place, and the read it came from
/// was not seen again.
fn refit_by_decode(image: &Luma, mut current: ReadResult, params: &SampleParams, opts: &DecodeOptions) -> ReadResult {
    for _ in 0..FEEDBACK_ROUNDS {
        if work::cancelled() {
            break;
        }
        let Some(next) = feedback_round(image, &current, params, opts) else { break };
        if !next.outranks(&current) {
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
    // Expecting the bleed the placement was made for, as it was refined
    // ([`Detection::bloom`]): fitted as printed, the known modules of a
    // photo that has bled pull the placement back to one side of them.
    let scorer =
        Scorer::new(image, &template, found.corners, threshold).with_spline(warp.spline()).with_max_edges(FEEDBACK_MAX_EDGES).with_bloom(found.bloom);
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
        // A patch of data was placed by the data already, and there are
        // many of them: each refit here solves the map afresh for every
        // place tried, and a hundred patches made a round take minutes.
        if matches!(anchors[k].kind, AnchorKind::Edge | AnchorKind::Custom | AnchorKind::Data) {
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
        let local = Scorer::new(image, &window, frame, threshold).with_along(6).with_bloom(found.bloom);
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

    // Anchors where none is, on patches of the known modules: the blocks
    // that verified lie all over the symbol, interleaved, and the cells
    // between the fixed patterns, which the corners and the spline alone
    // placed, have their modules among the known ones as much as any.
    // Each is fitted as the anchors above were, but as a move of the
    // window through the one map the anchors make, which is not solved
    // afresh for every place tried, and kept where the known modules
    // about it read better for the move and it moved less than
    // [`FEEDBACK_TRAVEL`].
    let points = feedback_points(&func, &anchors);
    let controls: Vec<ControlPoint> = anchors.iter().map(|a| a.control_point()).collect();
    if let Some(map) = (!points.is_empty()).then(|| Projection::new(func.width, func.height, corners, &controls)).flatten() {
        let patched: Vec<Anchor> = work::map(&points, |&at| {
            let (window, [x0, y0, x1, y1]) = window_of(&template, at).filter(|(window, _)| window.fixed() >= FEEDBACK_LEAST_KNOWN)?;
            let frame = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(|[x, y]| map.apply(x as f64, y as f64));
            let local = Scorer::new(image, &window, frame, threshold).with_along(6).with_bloom(found.bloom);
            let local_px = local.module_px();
            let (dx, dy) = (x0 as f64, y0 as f64);
            let score = |trial: [[f64; 2]; 4]| local.contrast_with(&Shifted { map: &Moved { map: &map, by: trial[0] }, dx, dy }, sign);
            let start = score([[0.0, 0.0]; 4])?;
            let mut evaluations = 0;
            let (moved, best) = descend_by([[0.0, 0.0]; 4], &[[1.0, 0.0, 0.0, 0.0]], local_px * 0.25, &mut evaluations, score)?;
            let [mx, my] = moved[0];
            let travel = mx.hypot(my);
            (best > start && travel <= FEEDBACK_TRAVEL * local_px).then(|| {
                let [px, py] = map.apply(at[0], at[1]);
                Anchor { kind: AnchorKind::Data, at, center: [px + mx, py + my], partial: false, inferred: false }
            })
        })
        .into_iter()
        .flatten()
        .collect();
        anchors.extend(patched);
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
        deblurred: current.deblurred,
        sharpen_modules: current.sharpen_modules,
        unsure_kept: current.unsure_kept,
    })
}

/// Modules between the points a refit to a partial decode anchors where
/// no anchor is, and how far each must lie from every anchor and corner:
/// a point nearer than that is held by the anchor already.
const FEEDBACK_STEP: usize = 8;
const FEEDBACK_CLEAR: f64 = 5.0;
/// Fewest known modules the window about such a point must hold: half
/// as many as a finder pattern with its separator has, so that the fit
/// is to modules and not to a stray few.
const FEEDBACK_LEAST_KNOWN: usize = 32;

/// The points of a lattice [`FEEDBACK_STEP`] modules apart, each in the
/// middle of a module, that lie [`FEEDBACK_CLEAR`] modules or more from
/// every one of `anchors` and from the symbol's corners.
fn feedback_points(func: &FunctionMap, anchors: &[Anchor]) -> Vec<[f64; 2]> {
    let (w, h) = (func.width as f64, func.height as f64);
    let held: Vec<[f64; 2]> = anchors.iter().map(|a| a.at).chain([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]]).collect();
    let half = (FEEDBACK_STEP / 2) as f64;
    (0..func.height)
        .step_by(FEEDBACK_STEP)
        .flat_map(|y| (0..func.width).step_by(FEEDBACK_STEP).map(move |x| [x as f64 + 0.5 + half, y as f64 + 0.5 + half]))
        .filter(|at| at[0] < w && at[1] < h)
        .filter(|&at| held.iter().all(|p| distance(*p, at) >= FEEDBACK_CLEAR))
        .collect()
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
    decode_as_ranked(grid, func, analysis, opts, &OnceLock::new(), false)
}

/// Levels and masks a brute force of one grid of a placement hands on to
/// those of the grids sampled after it ([`brute_force_preferring`]).
const RANKED_FORMATS: usize = 2;

/// [`decode_as`] of one of the grids a placement is sampled into: the
/// first, which `ranks`, where it is brute forced leaves in `formats`
/// the levels and masks that ranked best ([`RANKED_FORMATS`]), and the
/// rest are searched in stages under those alone and under the one each
/// ranks first itself ([`brute_force_preferring`]). Where the first is
/// not brute forced, its format bits having read, the rest are brute
/// forced whole. The level and mask are the symbol's, the
/// same however its modules are sampled, and the grids of one placement
/// are each brute forced where its format bits do not read: over the
/// photographs of BoofCV's benchmark that took longest to search, the
/// most of the time went on the stages of every level and mask of each
/// sampling, decoding noise. The first grid is sampled alone before the
/// others ([`work::first_then_in_turn`]), so that what the others are
/// handed does not hang on which of them came first.
fn decode_as_ranked(
    grid: &ModuleGrid,
    func: &Arc<FunctionMap>,
    analysis: &GridAnalysis,
    opts: &DecodeOptions,
    formats: &OnceLock<Vec<(EcLevel, u8)>>,
    ranks: bool,
) -> Option<DecodeReport> {
    if !analysis.format.is_reliable() {
        let attempts = match formats.get() {
            Some(preferred) if !ranks => brute_force_preferring(grid, func, None, opts, preferred),
            _ => {
                let attempts = brute_force(grid, func, None, opts);
                if ranks {
                    let _ = formats.set(attempts.iter().take(RANKED_FORMATS).map(|a| (a.ec, a.mask)).collect());
                }
                attempts
            }
        };
        return Some(attempts.into_iter().next()?.report);
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
/// bits after the one they settle, when the blocks bear one out
/// ([`nearest_rivals`]).
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
///
/// Beside the nearest words, the word the data and the format bits
/// favour together is a rival, wherever it lies, when the data reads as
/// a message under it and it is likelier so than the word settled
/// ([`likeliest_word`]): a word misread in more bits than the code
/// corrects lies beyond reach of the nearest, and the data says which
/// it was.
fn rival_decode(grid: &ModuleGrid, func: &Arc<FunctionMap>, format: &FormatAnalysis, opts: &DecodeOptions) -> Option<DecodeReport> {
    let best = format.best()?;
    let mut rivals = nearest_rivals(format, best);
    if let Some(likeliest) = likeliest_word(grid, func, format, best)
        && !rivals.iter().any(|c| c.fields == likeliest.fields)
    {
        rivals.push(likeliest);
    }
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

/// The format words [`rival_decode`] tries beside `best`, the word the
/// format bits settle: the words of the grid's own size within
/// [`RIVAL_REACH`] bits of the bits read, or, painted without the mask,
/// within [`QUIRK_REACH`], as far as such a word is ever taken for the
/// format; the [`RIVALS`] nearest of them, each level and mask once,
/// those painted as `best` is before those painted the other way the
/// grid's model paints, with or without the mask. A word wrong in four
/// bits or five that reads as another is always among the three nearest
/// of the rest, and a word painted the standard way read clean, or with
/// one bit wrong, has no other within reach, so a symbol whose format
/// reads well and whose blocks are past correcting costs nothing here.
/// The words painted otherwise than the settled one were once never
/// tried, and a word settled on as painted without the mask had for
/// rivals only the words so painted, every one of them six bits from
/// the read and more. Those painted without the mask are held to the
/// nearer reach: five of them lie just five bits from a QR Code's word
/// read clean, and some four from one read a bit wrong, which says
/// nothing for any of them, and within the reach of a standard word
/// three of them would be decoded each time such a symbol's blocks were
/// past correcting.
fn nearest_rivals<'a>(format: &'a FormatAnalysis, best: &FormatCandidate) -> Vec<&'a FormatCandidate> {
    let reach = |c: &FormatCandidate| if c.unmasked { QUIRK_REACH } else { RIVAL_REACH };
    let mut rivals: Vec<&FormatCandidate> = format
        .candidates
        .iter()
        .filter(|c| c.fits_grid && c.other_model == best.other_model)
        .filter(|c| c.fields != best.fields && c.best_mismatches <= reach(c))
        .collect();
    rivals.sort_by_key(|c| (c.unmasked != best.unmasked, c.best_mismatches, c.best_weight));
    // Each level and mask once, at the painting that comes first.
    let mut named = Vec::new();
    rivals.retain(|c| !named.contains(&c.fields) && {
        named.push(c.fields);
        true
    });
    rivals.truncate(RIVALS);
    rivals
}

/// The word of the grid's size, painted either way the grid's model
/// paints as `settled` is, with or without the mask, that the data read
/// under it and the format bits together favour most
/// ([`crate::decode::pipeline::Attempt::odds_bits`]), each word weighed
/// with what its painting weighs before any bit is read
/// ([`FormatCandidate::prior_weight`]), when the data reads as a message
/// under it ([`LIKENESS_FLOOR`]) and the two favour it over `settled`.
/// `None` when they favour `settled`, or the data reads as a message
/// under no word.
fn likeliest_word<'a>(grid: &ModuleGrid, func: &Arc<FunctionMap>, format: &'a FormatAnalysis, settled: &FormatCandidate) -> Option<&'a FormatCandidate> {
    let layouts: Vec<(EcLevel, SymbolLayout)> =
        func.version.ec_levels().iter().filter_map(|&ec| Some((ec, SymbolLayout::new(func.clone(), ec)?))).collect();
    let odds = |c: &FormatCandidate| -> Option<(f64, f64)> {
        let (_, layout) = layouts.iter().find(|(ec, _)| *ec == c.fields.ec)?;
        let likeness = likeness_bits(&stream_as_read(grid, layout, c.fields.mask));
        Some((likeness - f64::from(c.prior_weight()) / 8.0, likeness))
    };
    let (settled_odds, _) = odds(settled)?;
    format
        .candidates
        .iter()
        .filter(|c| c.fits_grid && c.other_model == settled.other_model && c.fields != settled.fields)
        .filter_map(|c| odds(c).map(|o| (c, o)))
        .filter(|(_, (odds, likeness))| *likeness >= LIKENESS_FLOOR && *odds > settled_odds)
        .max_by(|a, b| a.1.0.total_cmp(&b.1.0))
        .map(|(c, _)| c)
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
        // and every other read is counted by its way's place at every turn
        // above it, a bit more for the share the read asked for leaves the
        // rest: the first of five ways at a turn as one of three, and the
        // turn beneath it, of three ways alike, as one of three.
        let opts = DecodeOptions::default();
        let asked = Reads::new(&opts).sharing(5).nth_of(0, 5).one_of(3);
        assert_eq!((asked.first.tries, asked.again.tries), (1, 2 * 3 * 3));
        // Past it, the first way tried at each turn beneath is counted as
        // the rest are: it once went uncounted for its turn, as though it
        // were the read asked for. So, with every sampling of a turned
        // photo, which were each counted as the only one.
        let past = Reads::new(&opts).sharing(5).nth_of(1, 5).sharing(4).nth_of(0, 4).one_of(3);
        let counted = 2 * by_place(1, 5) * by_place(0, 4) * 3;
        assert_eq!((past.first.tries, past.again.tries), (counted, counted));
        // The budgets are shared over every turn alike, whatever the place.
        assert_eq!(past.first.trial_sets, Reads::new(&opts).sharing(5).sharing(4).first.trial_sets);
        assert_eq!(past.first.trial_sets, Reads::new(&opts).sharing(5).nth_of(4, 5).sharing(4).nth_of(3, 4).first.trial_sets);
        // An outline read as one of three ways, as the families are
        // ([`read_outline_one_of`]): the first read of the way asked for
        // goes uncounted, and the first of every other way is counted as
        // the rest are. Each way's first read once went uncounted.
        let asked = Reads::new(&opts).one_of(3);
        let other = asked.past();
        assert_eq!((asked.first.tries, asked.again.tries, other.first.tries), (1, 2 * 3, 2 * 3));
    }

    #[test]
    fn the_ways_of_a_turn_take_no_more_than_its_whole_by_their_places() {
        // However many ways a turn tries, their shares add to no more than
        // one, so that the reads past the one asked for, charged by their
        // places down every turn, are as likely to let noise through as
        // ever. The first of many pays two bits at most, the last its even
        // share, and none a bit more than that.
        for n in 1..=200 {
            let shares: f64 = (0..n).map(|k| 1.0 / by_place(k, n) as f64).sum();
            assert!(shares <= 1.0 + 1e-12, "{n} ways take {shares}");
            assert_eq!(by_place(n - 1, n), n);
            for k in 0..n {
                assert!(by_place(k, n) < 2 * n, "way {k} of {n}: {}", by_place(k, n));
                assert!(k == 0 || k + 1 == n || by_place(k, n) >= by_place(k - 1, n), "way {k} of {n}");
            }
            assert!(by_place(0, n) <= 4, "{n}");
        }
        // Of two, half each, as before.
        assert_eq!((by_place(0, 2), by_place(1, 2)), (2, 2));
        // The second sampling of a placement with forty-two ways to try,
        // the threshold rule that reads more than any retry, pays three and
        // a half bits where it paid more than five; the last but one six
        // and a third, and the last as ever.
        assert_eq!((by_place(1, 42), by_place(40, 42), by_place(41, 42)), (11, 82, 42));
    }

    #[test]
    fn a_decode_that_lost_every_confirmed_block_ranks_below_its_read_whatever_parses() {
        // The decode ranks its attempts at one grid with a stream that
        // parses first; a read, and so a refit of one or a read of it as
        // the other model, ranks a block confirmed first. A refit whose
        // every block failed, and whose stream of noise parsed, once took
        // the place of the read it came from by the decode's order.
        let confirmed = AttemptScore {
            stream_valid: false,
            all_verified: false,
            tail_plausible: false,
            blocks_verified: 2,
            blocks_solved: 2,
            confidence_bits: 64,
            neg_errata: -3,
        };
        let parsed = AttemptScore {
            stream_valid: true,
            tail_plausible: true,
            blocks_verified: 0,
            blocks_solved: 3,
            confidence_bits: 0,
            ..confirmed
        };
        assert!(parsed > confirmed);
        let agreement = (230, 236);
        assert!(standing_by(&confirmed, agreement) > standing_by(&parsed, agreement));
        // Between two that each confirm a block, the decode's own order.
        let more = AttemptScore { blocks_verified: 3, ..confirmed };
        assert!(standing_by(&more, agreement) > standing_by(&confirmed, agreement));
        assert!(standing_by(&AttemptScore { stream_valid: true, ..confirmed }, agreement) > standing_by(&more, agreement));
    }

    #[test]
    fn an_outline_off_the_photo_reads_nothing() {
        // Past the right edge of the photo, or below it: the box about the
        // outline, with its margin, holds none of the photo. Past the
        // right, the measure of the blur there once cut the photo's rows
        // backwards and panicked.
        let image = Luma::new(200, 200, vec![0.9; 200 * 200]);
        let opts = DecodeOptions::default();
        for region in [[[400.0, 50.0], [500.0, 50.0], [500.0, 150.0], [400.0, 150.0]], [[50.0, 400.0], [150.0, 400.0], [150.0, 500.0], [50.0, 500.0]]] {
            assert!(read_outline(&image, SymbolVersion::Qr(1), region, &SampleParams::default(), &opts).is_none(), "{region:?}");
            assert!(!worth_deblurring(&image, region, None), "{region:?}");
        }
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
            bloom: 0.0,
            decoded_at: None,
            picked_among: 1,
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
    fn placements_kept_aside_are_symbols_where_they_read_and_leads_where_they_do_not() {
        // The symbol on the left of a sheet, and bare paper on the right
        // as wide again, each with a placement of version 2 nothing
        // vouched for.
        let (drawn, corners) = drawn("kept aside for a read");
        let (width, height) = (2 * drawn.width, drawn.height);
        let mut data = vec![0.92f32; width * height];
        for y in 0..height {
            data[y * width..y * width + drawn.width].copy_from_slice(&drawn.data[y * drawn.width..(y + 1) * drawn.width]);
        }
        let image = Luma::new(width, height, data);
        let params = SampleParams::default();
        let beside = corners.map(|[x, y]| [x + drawn.width as f64, y]);
        let symbol = placed_by_hand(&image, SymbolVersion::Qr(2), corners, &params).unwrap();
        let paper = placed_by_hand(&image, SymbolVersion::Qr(2), beside, &params).unwrap();
        let alone = read_detection(&image, symbol.clone(), &params, &DecodeOptions::default()).expect("the symbol reads");
        let search = Search { detections: Vec::new(), leads: Vec::new(), unvouched: vec![symbol, paper] };
        let (settled, reads) = settle_unvouched(&image, search, &params, &DecodeOptions::default());
        // The symbol reads, and is found, as decoding; the paper is a lead
        // with the version it was fitted as, about where it lay.
        assert!(settled.unvouched.is_empty());
        // Its read comes with it, of the detection listed, charged by its
        // place among the two read, the first of two, as one of two, where
        // read alone it is a read asked for.
        assert_eq!(reads.len(), 1);
        let (read, among) = &reads[0];
        assert_eq!((read.detection.corners, read.detection.decodes), (settled.detections[0].corners, Some(true)));
        assert_eq!(read.text(), "kept aside for a read");
        assert_eq!(*among, by_place(0, 2));
        assert_eq!(read.report.tries, among * alone.report.tries, "{} against {}", read.report.tries, alone.report.tries);
        assert_eq!(settled.detections.iter().map(|d| (d.version, d.decodes)).collect::<Vec<_>>(), [(SymbolVersion::Qr(2), Some(true))]);
        assert_eq!(settled.leads.iter().map(|lead| (lead.family, lead.fitted)).collect::<Vec<_>>(), [(Family::Qr, Some(SymbolVersion::Qr(2)))]);
        let middle = |quad: [[f64; 2]; 4]| [quad.iter().map(|p| p[0]).sum::<f64>() / 4.0, quad.iter().map(|p| p[1]).sum::<f64>() / 4.0];
        assert!(distance(middle(settled.leads[0].outline), middle(beside)) < 1.0);
    }

    #[test]
    fn a_placement_a_trial_decode_vouched_for_is_read_charged_by_its_place() {
        // The fifth trial decode of the search vouched for it: read, it is
        // one of thirty ([`Detection::decoded_charge`]), wherever it is
        // read from, where read as one anything else vouched for it is a
        // read asked for.
        let (image, corners) = drawn("vouched by a decode");
        let params = SampleParams::default();
        let symbol = placed_by_hand(&image, SymbolVersion::Qr(2), corners, &params).unwrap();
        let alone = read_detection(&image, symbol.clone(), &params, &DecodeOptions::default()).expect("the symbol reads");
        let decoded = Detection { decoded_at: Some(4), ..symbol };
        let read = read_detection(&image, decoded, &params, &DecodeOptions::default()).expect("the symbol reads");
        assert_eq!(read.text(), "vouched by a decode");
        assert_eq!(read.report.tries, 30 * alone.report.tries, "{} against {}", read.report.tries, alone.report.tries);
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
    fn a_placement_found_deblurred_is_read_as_it_is_where_the_photo_about_it_is_sharp() {
        // Found in the photo deblurred, as a search finds a symbol where
        // the whole photo is blurred, and sharp about the symbol itself:
        // no copy deblurred is made there, and the placement is read
        // through the photo enlarged as the copy would have been, and
        // sharpened no more. It was once not read at all.
        let text = "sharp where it was found";
        let (image, square) = drawn(text);
        let (params, opts) = (SampleParams::default(), DecodeOptions::default());
        let [tl, tr, _, bl] = square;
        let centre = |corner: [f64; 2], [dx, dy]: [f64; 2]| [corner[0] + 28.0 * dx, corner[1] + 28.0 * dy];
        let found = on_finders(square, &[centre(tl, [1.0, 1.0]), centre(tr, [-1.0, 1.0]), centre(bl, [1.0, -1.0])]);
        let around = outline_around(square, AROUND_FOUND);
        for enlarge in [1, 2] {
            let deblurred = Enhance { enlarge, sharpen: Sharpen::Deblur };
            assert!(Enhanced::around(&image, around, deblurred).is_none(), "{enlarge}: no blur to undo");
            let mut best = None;
            assert!(read_found_through(&image, found.clone(), around, &params, Reads::new(&opts), deblurred, &mut best), "{enlarge}: settled");
            let read = best.expect("a read");
            assert_eq!(read.text(), text);
            assert_eq!((read.enhance, read.deblurred), (Enhance { enlarge, sharpen: Sharpen::Off }, None), "{enlarge}");
        }
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

        // A word read clean, or a bit wrong, has no other within reach,
        // painted either way: every word painted without the mask lies
        // four bits from it or more. And a symbol whose blocks are past
        // correcting is decoded under it once and left.
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
        assert!(nearest_rivals(&analysis.format, analysis.format.best().unwrap()).is_empty());
        for b in 0..15 {
            let format = analyze(&misread(&[b]), &func).format;
            assert!(format.is_reliable() && nearest_rivals(&format, format.best().unwrap()).is_empty(), "bit {b}");
        }
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
