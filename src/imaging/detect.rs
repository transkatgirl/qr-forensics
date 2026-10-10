//! Coarse alignment: find the finder patterns inside a loosely drawn
//! outline, then grow a fit of the symbol's fixed patterns outward from them
//! until the four corners are close enough for the refinement to take over.
//!
//! The outline says roughly where the symbol is, how big it is and which way
//! up. The photo is first rectified through that outline, with a margin all
//! round, so that finder patterns come out close to square whatever the
//! perspective. Each row of the rectified image is then scanned for the
//! 1:1:3:1:1 run of dark and light that crosses the centre of every finder,
//! hits are cross-checked down the column and clustered; the connected
//! regions of the binarised image propose finders too, each fitted a
//! quadrilateral and read along two lines through it, as BoofCV finds
//! position patterns, which a scratch through both axes or a steep view
//! does not break. Each surviving candidate is refined in the photo itself
//! by fitting a finder-with-ring template to it with the contrast score
//! the corner refinement uses. A candidate whose template reads right is a
//! capstone. One with a quarter torn away still counts, as long as the
//! other three quarters read right. A candidate no template fits is
//! asked once more along the lines through its centre alone, as
//! zxing-cpp asks a concentric pattern: the column, the row and both
//! diagonals must each read 1:1:3:1:1, and nothing is asked of the ring
//! between them, which a finder drawn as a large dot ringed by small
//! ones does not have.
//!
//! The size of the symbol is taken two ways, as OpenCV takes it: from the
//! spacing of the capstones against their module size, and from a count
//! of the modules along the timing pattern between two of them, which
//! does not depend on the module size at all; a lone finder walks its
//! symbol's timing patterns outward instead. The count also tells a true
//! pair of finders from two patterns in other print, which have no timing
//! pattern between them.
//!
//! Capstones are matched to the finder positions of a version, trying each
//! orientation of a square symbol, so an outline drawn a quarter turn out
//! still works. The matched capstones give a first map from module
//! coordinates to photo pixels. Every other distinctive fixed pattern (a
//! finder the scan missed, alignment patterns, the rMQR finder sub-pattern,
//! and a few dark modules along each timing pattern, which on a symbol
//! without alignment patterns are the only fixed points between the
//! finders) is then looked for in a window around where the anchors found
//! so far predict it, nearest first. The prediction is the global map bent through
//! the anchors already placed, so on paper that is not flat the search
//! follows the curve, as ZBar extrapolates each alignment pattern from its
//! neighbours. An anchor is doubted only against what its neighbours
//! predict for it, never against a plane the whole symbol is assumed to
//! lie in. The map is upgraded from a similarity through an affine
//! transform to a full perspective one as the anchors spread out.
//!
//! A corner a finder stands at is the capstone's own, measured to a
//! fraction of a pixel. A corner left far from any anchor gets a search of
//! its own. Where the symbol has a quiet zone, the edge of the symbol is
//! traced there, as ZBar traces it: profiles across the predicted edge find
//! where the paper turns to ink, a line is fitted to those crossings, or a
//! bend where the paper has curled, as OpenCV fits a spline to each side
//! of a curved symbol, and the corner is where two such lines meet.
//! Without a quiet zone, or where the trace fails, the corner is placed
//! where the fixed modules of its own quarter read best and the data
//! modules there read crisply dark or light rather than grey.
//!
//! The version is not taken on trust. Before anything is fitted, the
//! symbol's own information bits are read beside the finders alone,
//! through a map fitted to each capstone's own corners, as zxing-cpp reads
//! the format word of a Micro QR or rMQR symbol through its finder and
//! takes the version from it: the format word of those families names the
//! version, and the version information of a QR symbol of version 7 or
//! more sits beside two of its finders. A version so named is fitted
//! first, and believed while its fixed patterns fit tolerably. Then the
//! capstone spacing and the outline size give a size estimate, the
//! nearest versions and the current one are each fitted, and the one
//! whose fixed patterns read best wins, unless the information bits read
//! through the fitted outline name a version. A version they name that
//! was not tried is fitted too.
//!
//! The same bits tell a mirrored photo apart. A square symbol seen from
//! the back has its finder patterns in the same three corners, and every
//! fixed pattern reads the same, so the fit cannot know; the format word
//! only reads as a word with the corners taken in mirror order. When the
//! format bits cannot be read at all, both orders are tried by decoding,
//! as ZXing and ZBar retry a failed decode mirrored, and the order whose
//! blocks verify wins. An rMQR symbol's finder and sub-pattern are
//! asymmetric, and the matching of capstones to finder positions settles
//! its mirror image directly.
//!
//! The outline is a search region, not a frame. A hand draws a box square
//! to the photo around a symbol that sits at an angle, or a box twice the
//! symbol's size, and in either every finder lies far from where the
//! version would have it across the box. ZXing and ZBar place a symbol
//! from its finder centres alone, with no outline at all, and so does the
//! fit here once the capstones are found: two or three capstones that lie
//! as a symbol's finders do propose an outline of their own, square to
//! the symbol, and the fit is made in that frame, with the outline given
//! kept as the last frame tried.
//!
//! When no finder pattern can be found as the photo is, the search is run
//! again through a ladder of transformed copies, as zxing-cpp retries with
//! `tryDenoise` and `tryDownscale`: the dark closed over small gaps for a
//! print made of dots, the photo shrunk for a halftone screen, and the
//! photo enlarged for a symbol a pixel or two a module, each copy of
//! another size made with Magic Kernel Sharp 2013 ([`super::resize`]).
//! One rung transforms nothing but the scan's own threshold: the
//! neighbourhood each pixel is judged against, a sixth of the outline
//! wide, is narrowed to a few tens of pixels, so that a finder in a
//! shadow or a glare narrower than the neighbourhood is judged against
//! its own light rather than the paper beyond it, as OpenCV's ArUco-based
//! QR detector thresholds at several window sizes and keeps what any of
//! them finds. The scan itself reads the outline against the mean of a
//! window a few pixels about each pixel as well, from the start
//! ([`super::binarize::window_dark`]): a finder pattern blurred by half
//! a module keeps its rings only against what lies right beside them.
//! And three rungs sharpen the photo, at its size and enlarged
//! ([`super::enhance`]), for a symbol so far out of focus that no
//! threshold tells its finders' rings from their stones.
//!
//! The fit of the fixed patterns ranks the placements, but it is not the
//! last word. [`detect_candidates`] keeps the best few, across the
//! versions and frames tried, and when every version fits poorly the
//! next nearest versions are tried as well; the decode then arbitrates,
//! in [`super::read`], as ZXing's and zxing-cpp's readers keep what
//! decodes. The few kept are not simply the best fits: every way the
//! capstones were matched to the finders is represented before any is
//! represented twice, since a blurred finder left unmatched leaves the
//! symbol placed turned on the other two, and such a placement can fit
//! better than the right one at three versions over. A symbol found with modules too small for the scan to read
//! well is looked for a second time in the photo enlarged by what the
//! version found says its modules need, where the first pass enlarges
//! only by what the family's smallest symbol would need.
//!
//! Where an edge of the symbol bows, a few points along it anchor the
//! map as well, each where the paper turns to ink: on a bottle or a
//! curled receipt the side of the symbol farthest from any fixed pattern
//! is otherwise placed by the corners and the spline's guess alone, and
//! the bowed edge says where its rows really run.
//!
//! [`find_symbols`] runs the same machinery with no outline at all: the
//! whole photo is scanned for finder patterns, which are grouped into
//! symbols by their geometry, as ZXing and ZBar group them, and each
//! group's outline is then detected in turn. With no outline to say that
//! a symbol is there, a detection is offered only when more than the
//! finders that proposed it vouches for it: its timing patterns must
//! read as well, or a word of its own information bits. A photo may hold
//! a sheet of symbols, and the search goes on for as long as its
//! proposals find them: every finder pattern the scan shows is kept, a
//! proposal that finds a symbol costs the search none of its patience,
//! and the finders left over when the triples are done propose in
//! pairs, for the symbol with a finder torn off or cut by the frame.
//!
//! A large photo is searched first in a copy shrunk to about a thousand
//! pixels on its longer side, and each symbol the copy shows is found
//! again at full size through its own outline; the photo is searched at
//! full size only when no copy shows a symbol, or when a copy shows the
//! finders of one it could not fit. BoofCV's detector is
//! quick because its costly refinement touches only the shapes a cheap
//! pass found, and the copy serves the same end: a twelve-megapixel
//! photo that took two minutes to search at full size takes twenty
//! seconds.
//!
//! A photo keeps the size it has: it is not made smaller when it is
//! loaded, and what is found in it is fitted, refined and sampled in its
//! own pixels, however many they are. Only the search of the whole
//! photo looks at less. It is made in a copy of the photo reduced to
//! [`SEARCH_SIDE`] pixels on its longer side, as every photo once was
//! when it was loaded, and each symbol the copy shows is then found
//! again in the photo itself through its outline, as a symbol shown by
//! a shrunk copy is. A symbol too small to show in the copy is not
//! found by the search: an outline drawn round it finds it.

use std::borrow::Cow;
use std::f64::consts::SQRT_2;
use std::fmt;
use std::sync::{Arc, OnceLock};

use super::Luma;
use super::adjust::Filter;
use super::binarize::{BlockMean, WINDOW_BIAS, WINDOW_RADII, window_dark};
use super::enhance::{Enhance, Enhanced, MAX_VIEW_SIDE, Sharpen};
use super::homography::{Homography, Model};
use super::projection::{ControlPoint, Project, Projection};
use super::refine::{MOVES, SHIFT, Scorer, Template, descend_by, fixed_threshold};
use super::sample::{
    INNER_FRACTION, Kernel, NOISE_RANGE, POINTS, SampleParams, classify, otsu, otsu_split, robust_spread, sample_modules_as,
    sample_positions,
};
use crate::decode::bch;
use crate::decode::pipeline::{AttemptScore, DecodeOptions, brute_force};
use super::regions;
use crate::symbol::function_map::{FunctionMap, InfoCopy, Role};
use crate::symbol::grid::ModuleGrid;
use crate::symbol::{Family, Pos, SymbolVersion, tables};
use crate::work;

/// How far beyond the outline the search reaches, as a fraction of its size.
pub(crate) const MARGIN: f64 = 0.15;
/// Longest side of the image rectified for the scan of an outline for
/// finder patterns, in pixels: what is inside a larger outline is first
/// scanned reduced to this. The scan only proposes. Every finder it
/// proposes is fitted, and every corner, anchor and module after it is
/// placed and read, in the photo's own pixels, and an outline may be no
/// more than four times its symbol across, so that a module of the
/// largest symbol still has two pixels in the reduced scan. Scanning a
/// large outline unreduced from the start was tried when photos stopped
/// being shrunk at loading: on a forty-eight megapixel photo it found
/// nothing the reduced scan did not, and with the grain such a photo
/// has it took eight times as long. So the unreduced scan is the last
/// resort of a search that has found nothing: see [`detect_candidates`].
const MAX_RECTIFIED: f64 = 2048.0;
/// No limit on the rectified image: the photo's own pixels.
const UNREDUCED: f64 = f64::INFINITY;
/// The longest side, in pixels, of the photo a search of the whole of it
/// looks at: a larger photo is searched in a copy reduced to this.
pub const SEARCH_SIDE: usize = 4096;
/// Smallest module worth looking for, in rectified pixels.
const MIN_MODULE: f64 = 1.5;
/// A module is at most this many times the outline's shorter side divided
/// by the fewest modules a symbol of the family has along its shorter
/// side: the outline is drawn loosely, and the symbol may overshoot it by
/// this much. An rMQR symbol is only seven modules tall where a Micro QR
/// is eleven across and a QR Code twenty-one, so each family has its own
/// cap; one cap for all would either lose a short rMQR symbol or admit
/// blobs far too big to be a QR Code's finder.
const MODULE_CAP_SLACK: f64 = 1.375;
/// Scan clusters refined, strongest first: as many as ZXing keeps. A busy
/// frame of text and logos throws up many false candidates, and a small
/// true finder must not be crowded out by them.
const MAX_CANDIDATES: usize = 25;
/// A capstone accepted on three quarters must read at least this share of
/// the contrast of the strongest whole capstone. A pattern fitted to data
/// or noise on three quarters never reads as crisply as a true finder,
/// torn or not, and a false one misleads the size estimate and the
/// matching of capstones to finders.
const PARTIAL_CONTRAST_SHARE: f64 = 0.75;
/// Share of a template's modules that must read the right colour.
const MIN_AGREEMENT: f64 = 0.88;
/// A pattern with one quarter damaged is accepted when the other three
/// quarters each read at least this well and the whole at least
/// [`MIN_PARTIAL_AGREEMENT`].
const MIN_QUARTER_AGREEMENT: f64 = 0.85;
const MIN_PARTIAL_AGREEMENT: f64 = 0.65;
/// A capstone counts as a version's finder when it lies within this
/// fraction of the outline's size of where that finder should be.
const MATCH_TOLERANCE: f64 = 0.2;
/// Assignments of capstones to finders costing within this much of the
/// cheapest, in the squared units of the outline, are each fitted and the
/// best fit kept. Leaving a finder unmatched costs the tolerance squared,
/// about what matching a false capstone that lies near a finder's place
/// costs, and only the fit can tell a torn-away finder from a wrong turn.
const ASSIGNMENT_SLACK: f64 = 0.05;
/// An assignment standing more capstones as finders than the cheapest
/// does is fitted too, within this much of it: the size term that leaves
/// a blurred finder out costs about this at a module ratio of one and a
/// half, and the fit and the decode judge the two placements better than
/// the cost can.
const ASSIGNMENT_FULLER_SLACK: f64 = 0.2;
/// Spacing of search grids, in modules: inside the contrast score's basin.
const SEARCH_STEP: f64 = 0.3;
/// Widest search window for a pattern, in modules, while the map is still
/// affine, and once it is a full perspective one. A wide window is needed
/// while perspective is unaccounted for; once it is, a far-off match is
/// more likely data that happens to look like an alignment pattern.
const WINDOW_AFFINE: f64 = 4.0;
const WINDOW_PROJECTIVE: f64 = 2.5;
/// An anchor this far, in modules, from where its neighbours predict it is
/// a false match, and is dropped from the fit. The allowance grows with
/// the anchor's distance from its nearest neighbour, as the prediction
/// drifts with distance from what is known.
const OUTLIER_MODULES: f64 = 0.6;
const OUTLIER_BASE: f64 = 0.5;
const OUTLIER_RATE: f64 = 0.03;
/// How far a timing anchor may stray, in modules, from the bend its
/// line-mates trace: paper curls smoothly along a timing pattern, so the
/// anchors along one lie on a gentle curve, and one off it has read a
/// row of data or a smudge instead.
const TIMING_LINE_TOLERANCE: f64 = 0.25;
/// Fewest anchors on a timing line for the check to say anything.
const TIMING_LINE_MIN: usize = 6;
/// Anchors needed before one may be dropped as an outlier.
const OUTLIER_MIN_ANCHORS: usize = 4;
/// Half-width of the final search of a corner no anchor is near, in
/// modules, when fixed modules constrain it, when only the data modules'
/// crispness does, and when the symbol's edge has been traced.
const CORNER_SEARCH: f64 = 2.0;
const CORNER_SEARCH_CRISP: f64 = 0.6;
const CORNER_SEARCH_EDGED: f64 = 0.5;
/// Each run of a finder-shaped 1:1:3:1:1 crossing may be off its ideal
/// length by this fraction of a module, along a row or column, and along
/// the diagonal, where a turned outline stretches the runs unevenly.
const RATIO_TOLERANCE: f64 = 0.5;
const DIAGONAL_TOLERANCE: f64 = 0.75;
/// How far the module a cross-check measures may stray from the one
/// expected, as shares of it: along a row or column, and along the
/// diagonal, which crosses the rings of a finder that sits straight in as
/// many steps as a row does and the rings of one turned forty-five
/// degrees, whose own axis it then runs along, in half as many.
const MODULE_FIT: (f64, f64) = (0.6, 1.4);
const DIAGONAL_FIT: (f64, f64) = (0.45, 1.05);
/// Half candidates, which passed the cross-check on one side only, are
/// fitted after the whole ones and at most this many: data throws up
/// many, and a torn finder's ranks among the strongest of them.
const MAX_HALF_CANDIDATES: usize = 16;
/// Weight of the difference in module size between two capstones matched
/// to one symbol: at a ratio of 1.4, as ZXing allows, it costs as much as
/// leaving a finder unmatched.
const SIZE_WEIGHT: f64 = 0.25;
/// A corner within this many modules of an anchor needs no search of its own.
const ANCHORED_WITHIN: f64 = 12.0;
/// A corner search must gain this much score to move the corner, or noise
/// would walk a corner that nothing constrains.
const CORNER_GAIN: f64 = 0.01;
/// Weight of the data modules' crispness against the fixed modules'
/// contrast in the corner search.
const CRISP_WEIGHT: f64 = 1.0;
/// Versions tried around the size estimate when it rests on one capstone
/// and the outline, or on the spacing of two or more capstones. The rMQR
/// sizes lie close together, and its one finder gives only the outline to
/// go on, so more of them are tried.
const VERSIONS_LOOSE: usize = 3;
const VERSIONS_TIGHT: usize = 2;
const VERSIONS_LOOSE_RMQR: usize = 6;
const VERSIONS_TIGHT_RMQR: usize = 3;
/// Anchors this far from the line through the two farthest apart make a
/// perspective fit well conditioned.
const SPREAD_MODULES: f64 = 3.0;
/// Edge tracing: how far outside and inside the predicted edge a profile
/// reaches, in modules, and its sampling step.
const PROFILE_REACH: f64 = 1.5;
const PROFILE_STEP: f64 = 0.1;
/// The part of a profile that must read as paper for the edge to count as
/// having a quiet zone there, in modules out from the edge.
const QUIET_FROM: f64 = 0.7;
const QUIET_TO: f64 = 1.2;
/// Share of profiles along an edge that must show quiet zone outside it.
const QUIET_SHARE: f64 = 0.7;
/// Least contrast between the paper and a border module for a crossing to
/// be located in a profile, as a share of the symbol's own contrast, the
/// gap between the means of its dark and light fixed modules: a fifth of
/// the scale on a well exposed photo, and in proportion on a dim one.
const PROFILE_SHARE: f32 = 0.3;
/// Most profiles traced along one edge.
const MAX_PROFILES: usize = 30;
/// Fewest crossings to fit an edge line, and the worst fit accepted, as a
/// root-mean-square distance in modules.
const MIN_CROSSINGS: usize = 3;
const MAX_EDGE_RMS: f64 = 0.2;
/// Fewest crossings to fit a bend to an edge instead, and the most an
/// edge may bow from straight, in modules, for the bend to be believed:
/// paper curls gently, and a sharper bend is the crossings of other print.
const MIN_CURVE_CROSSINGS: usize = 6;
const MAX_EDGE_SAGITTA: f64 = 3.0;
/// What a bend must show before it is preferred to the line: the edge
/// bows by at least this many modules; the bend fits the crossings within
/// this root-mean-square distance in modules; and it fits them at most
/// this share as badly as the line does. A bend fitted to the jagged
/// crossings of a straight edge, or to crossings scattered by other
/// print, is not wanted: its tangent at a corner beyond the crossings
/// swings with it. Real curls bow a tenth of a module and more and fit
/// to a fiftieth; noise bows less, fits worse, and the line explains it
/// nearly as well.
const MIN_EDGE_SAGITTA: f64 = 0.1;
const CURVE_RMS: f64 = 0.05;
const CURVE_GAIN: f64 = 0.2;
/// An anchor refined by hand or by the photo may not wander further than
/// this from where it started, in modules; a timing anchor, which the
/// search placed through the map already and which a blurred photo reads
/// to a few tenths of a module either way, only this far.
const REFINE_LIMIT: f64 = 1.0;
const TIMING_REFINE_LIMIT: f64 = 0.15;
/// Boundary moves tried, in modules, for a finder pattern that does not
/// fit as printed: into the dark for an over-exposed photo, into the light
/// for an under-exposed one.
const BLOOMS: [f64; 3] = [0.2, 0.35, -0.2];
/// Contrast forgone per square module of distance from the predicted
/// place of a pattern, and how many of the best placements are read
/// before giving up on a site.
const DISTANCE_PENALTY: f64 = 0.03;
const PLACEMENTS_TRIED: usize = 3;
/// The same for a timing site, whose five modules in a line are imitated
/// by print and data too easily for a far-off match to be believed: its
/// prediction comes from an anchor a few modules away and is good to a
/// fraction of a module, so a module off costs a third of the contrast.
const TIMING_DISTANCE_PENALTY: f64 = 0.3;
/// Share of the ring of modules around a finder that must read as paper
/// for the ring to be used in placing it.
const RING_PAPER: f64 = 0.9;
/// Lowest share of fixed and quiet-zone modules that a version the
/// symbol's own bits name must still fit with; below it the fits of every
/// version decide among themselves, as they do when the bits say nothing.
const MIN_NAMED_FIT: f64 = 0.7;
/// Most modules a QR version named by its version information may differ
/// from what the finder spacing says, as zxing-cpp allows.
const NAMED_SIZE_SLACK: f64 = 8.0;
/// Placements of a lone finder within this much of the best, across the
/// outline, are all tried for the pre-read: the plain and the mirrored
/// placement land it on the same corner, and the position cannot tell
/// them apart.
const PLACEMENT_SLACK: f64 = 0.05;
/// Outline proposals that come to nothing before a search of one copy of
/// the photo gives a family up. It was the most proposals a search
/// detected at all, a dozen whatever became of them, and on a sheet of
/// symbols the false triples that three finders of three neighbours make
/// spent the dozen before every symbol had its turn: of eight symbols in
/// one photo seven were found. A proposal that finds a symbol costs the
/// search nothing it would not spend anyway, and is not counted.
const MAX_PROPOSALS: usize = 12;
/// The same for the outlines two leftover capstones propose, which are
/// tried after every triple and have a budget of their own, so that false
/// triples cannot spend it for them.
const MAX_PAIR_PROPOSALS: usize = 6;
/// Proposals fitted side by side, at most, once the best of them has come
/// to nothing: every fit of such a batch is waited for, and a fit made
/// for a proposal that another of the batch disposes of is wasted, so
/// the batch is kept well under the dozen that may fail.
const FITS_AT_ONCE: usize = 8;
/// Scan clusters a search of the whole photo refines, strongest first.
/// [`MAX_CANDIDATES`] is as many as one symbol in an outline needs, and a
/// sheet of twenty symbols threw up ninety-one clusters, sixty of them
/// finders, of which the twenty-five strongest made six whole triples:
/// OpenCV's ArUco-based detector keeps every quadrilateral its
/// thresholds find, and BoofCV every square.
const MAX_SEARCH_CANDIDATES: usize = 128;
/// How far from the diagonal of its own square, in degrees, the line
/// between two capstones may lie at each for the pair to be proposed as
/// the finders at the ends of a symbol's diagonal, in a search of the
/// whole photo: no timing pattern runs between such a pair to vouch for
/// it, as one does along a side, and a pair lying any way at all across
/// a sheet proposes a square.
const DIAGONAL_DEGREES: f64 = 20.0;
/// Share of fixed modules a symbol found by a search of the whole photo
/// must read right to be offered.
const MIN_FOUND_SCORE: f64 = 0.8;
/// The share that will do when the symbol's own bits name its version as
/// well. A name alone vouches for nothing: the version fitted to a Micro
/// QR or rMQR finder is the very one the bits beside it name, and bits
/// read off anything at all name a version more often than not, so that
/// with no fit asked of a name a symbol was offered at one module in
/// five right.
const MIN_NAMED_SCORE: f64 = 0.7;
/// Share of the modules of a pattern that is no finder's, a timing
/// pattern or a QR Code's alignment patterns, that must read right for
/// the pattern to show that a symbol lies between and beyond the finders.
const MIN_BEYOND_FINDERS: f64 = 0.75;
/// The two legs from the top-left finder to the other two may differ in
/// length by this ratio at most, and meet this far from a right angle, in
/// degrees, for three capstones to be one QR symbol's finders. Perspective
/// foreshortens one leg against the other and leans the angle: a symbol
/// photographed from well below has its top finders three fifths as far
/// apart as its left ones. ZXing's right isosceles measure, which it
/// ranks by and does not cut on, ranks the proposals here too; the cut is
/// where OpenCV's ArUco-based detector makes its own, a side mismatch of
/// 1.75 (`maxModuleSizeMismatch`) and an angle within sixty degrees of
/// square (`checkTriangleAngle`). zxing-cpp cuts at a mismatch of two and
/// at thirty degrees, and thirty stood here, taken for OpenCV's figure
/// too; one photograph of the corpus shows a symbol leaning past forty.
/// The measure ranks a triple that leans so far below every squarer one,
/// and the fit and the decode have the last word on it either way. A cut
/// on the measure itself, at half the longest side, lost every such
/// photograph before any fit.
const MAX_LEG_RATIO: f64 = 1.75;
const MAX_CORNER_TILT: f64 = 60.0;
/// Capstones of one symbol may differ in module size by this ratio, as
/// OpenCV's ArUco-based detector allows: ZXing's 1.4 loses the symbol seen
/// from below, whose top finders are drawn three fifths the size of its
/// bottom one.
const SIZE_RATIO: f64 = 1.75;
/// How far beyond the finder centres a proposed outline reaches, in
/// modules: the symbol's edge lies three and a half modules past them, and
/// two more take in the quiet zone, which the detector wants a loose
/// outline to hold. It was a share of the finder spacing, three tenths,
/// which put the finder centres of a large symbol so far across the
/// outline that the assignment of capstones to finders, which allows
/// [`MATCH_TOLERANCE`] of the outline, matched none of them from version
/// 15 up, and the search could not fit a symbol whose finders it had found
/// exactly. A reach in modules keeps every version's finders where its
/// assignment expects them.
const PROPOSAL_REACH: f64 = 5.5;
/// Longest side of the rectified image for a search of the whole photo:
/// the photo's own size, so that a symbol a few pixels a module in a large
/// photo is not shrunk below what the scan can see.
const MAX_RECTIFIED_WHOLE: f64 = 4096.0;
/// Outlines the capstones propose for the symbol that are fitted, best
/// first, after the outline given: three for a symbol with three
/// finders, whose capstones propose many; five for a symbol with one,
/// the box its format word proposes and the four boxes of the family's
/// largest size with the finder in each corner, which are all a lone
/// finder has to propose, and whose versions are few to fit.
fn max_frames(family: Family) -> usize {
    if family.has_three_finders() { 3 } else { 5 }
}
/// What a proposal costs for the weakest capstone in it reading at a
/// share of the strongest capstone's contrast: a true finder reads about
/// as crisply as the others of its symbol, and a pattern in the data or
/// the print beside it, which may lie as a finder would, reads fainter.
/// Perspective bends a true triple out of its right isosceles shape, and
/// without this a false one that happens to lie true outranks it.
const CONTRAST_WEIGHT: f64 = 1.0;
/// A fit of the fixed patterns and quiet zone this good in one frame
/// settles a symbol with three finders, and the other frames are not
/// fitted. A symbol with one finder is settled by its own word alone:
/// see [`detect_candidates_in`].
const SETTLED_FIT: f64 = 0.9;
/// What an outline proposed from two capstones costs beyond one proposed
/// from three: a pair says how big the symbol is but not which side of
/// the pair it lies on, so three outlines are proposed for every pair,
/// after any proposed from three capstones.
const PAIR_BADNESS: f64 = 0.5;
/// Two outlines whose corners each lie within this share of their size of
/// a corner of the other are the same outline.
const FRAME_LIKENESS: f64 = 0.05;
/// Where the line from one finder's centre to another's may cross the
/// side of each that faces the other, as a share of the side either way
/// from its middle, and how far from square to that line the side may
/// lie, in degrees, for the two to stand as finders of one symbol. In a
/// symbol the line between two of its finders runs along a row or a
/// column of modules, through the middle of the sides that face and
/// square to them; perspective bends that a little, and a finder fitted
/// to a blurred print turns a little. BoofCV connects two position
/// patterns only when the line between their centres crosses a side of
/// each within 0.35 of its middle and the two sides lie nearly parallel,
/// and OpenCV's ArUco-based detector asks that two finders of one symbol
/// face each other, and each the line between them, to within fifteen
/// degrees (`maxRotation`); the thirty here leave room for the turn a
/// blurred finder's fit takes.
const FACING_MIDPOINT: f64 = 0.3;
const FACING_DEGREES: f64 = 30.0;
/// An outline the capstones propose is fitted only when its centre lies
/// within the outline given, with the search margin, and its size is
/// between these shares of the outline given: the outline says where the
/// symbol is and about how big, if not which way up.
const PROPOSAL_SIZE_MIN: f64 = 0.25;
const PROPOSAL_SIZE_MAX: f64 = 2.0;
/// Pixels a module the family's smallest symbol would have across the
/// outline given, below which the photo around the outline is enlarged
/// before the search, by a whole factor of up to [`MAX_UPSCALE`]: the
/// scan looks for modules of a pixel and a half or more.
const TARGET_MODULE_PX: f64 = 3.0;
const MAX_UPSCALE: u8 = 4;
/// The neighbourhood the ladder narrows the local threshold to, as a
/// half-width in pixels: two of the threshold's blocks, so that a shadow
/// or a glare forty pixels across is judged within itself, while a
/// finder a module and a half wide, the smallest the scan looks for, is
/// ten pixels across and keeps its neighbourhood about it.
const NARROW_RADIUS: u8 = 16;
/// Reading a timing pattern: samples taken per module along it; the
/// shortest and longest run, against the module the runs agree on, that
/// still counts as one module; the share of runs that must; and the
/// light run, in modules, that is the quiet zone rather than a module.
const TIMING_SAMPLES: f64 = 4.0;
const TIMING_RUN_MIN: f64 = 0.5;
const TIMING_RUN_MAX: f64 = 1.6;
const TIMING_REGULAR: f64 = 0.75;
const PAPER_RUN: f64 = 1.6;
/// What an outline proposed from capstones costs for each side between
/// two of them along which no timing pattern reads: two finders of one
/// symbol have one between them, and two patterns in other print do not,
/// as zxing-cpp reads the finder pattern along the line between two
/// before it samples the symbol they would make.
const TIMING_PENALTY: f64 = 1.0;
/// What it earns for a side along which one does.
const TIMING_BONUS: f64 = 0.5;
/// The locator of an alignment pattern the template cannot be fitted to
/// (see [`locate_alignment_by_meanshift`]): the half-width of the window
/// its mean shift samples, in modules, and the samples across it, BoofCV's
/// eight across three modules; the rounds of each of its two searches,
/// and how the shift's step shrinks each round.
const LOCATOR_REACH: f64 = 1.5;
const LOCATOR_SAMPLES: usize = 8;
const LOCATOR_ROUNDS: usize = 10;
const LOCATOR_DECAY: f64 = 0.7;
/// Within this many modules of the pattern's centre a sample should read
/// dark, the centre module, and beyond it light, the ring about it.
const LOCATOR_CORE: f64 = 0.5;
/// The step, in modules, at which the rings are read across the window
/// for the shift's seed.
const LOCATOR_SEED_STEP: f64 = 0.25;
/// The least a sample may weigh against the shift, as a share of the
/// symbol's contrast: BoofCV's ten of 255.
const LOCATOR_CLAMP: f32 = 0.05;
/// What a located pattern must still show, as a share of the symbol's
/// contrast: its centre darker than the ring a module out, and that ring
/// lighter than the ring two modules out. Blurred to a couple of pixels a
/// module a pattern keeps that much of the contrast between its rings
/// where it keeps none of its modules' edges.
const LOCATOR_MIN_RING_CONTRAST: f32 = 0.15;

/// Largest module worth looking for, as a fraction of the outline's
/// shorter side, for a family.
fn max_module_fraction(family: Family) -> f64 {
    let fewest = match family {
        Family::Qr | Family::Model1 => 21.0,
        Family::Micro => 11.0,
        Family::Rmqr => 7.0,
    };
    MODULE_CAP_SLACK / fewest
}

/// A point in module coordinates and where it lies in the photo.
type Pair = ([f64; 2], [f64; 2]);

/// A fixed pattern distinctive enough to be found on its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AnchorKind {
    Finder,
    Alignment,
    /// The rMQR finder sub-pattern, bottom right.
    FinderSub,
    /// One dark module of a timing pattern: a known point on the symbol
    /// where no larger pattern is, which lets the map follow paper that
    /// curls between the finders, as OpenCV straightens a curved symbol
    /// along its sides and zxing-cpp refines a Micro QR symbol's transform
    /// by the ends of its timing patterns.
    Timing,
    /// A point of the symbol's edge, where the paper turns to ink along
    /// an edge that bows. On paper that curls, a side of the symbol with
    /// no fixed pattern of its own is placed by the corners and the
    /// spline's guess at what lies between them; the bowed edge says
    /// where its rows really run, as OpenCV fits a curve to each side of
    /// a bent symbol and straightens it by them.
    Edge,
}

impl AnchorKind {
    pub fn label(self) -> &'static str {
        match self {
            AnchorKind::Finder => "finder pattern",
            AnchorKind::Alignment => "alignment pattern",
            AnchorKind::FinderSub => "finder sub-pattern",
            AnchorKind::Timing => "timing module",
            AnchorKind::Edge => "edge point",
        }
    }
}

/// A transform of the photo through which a symbol was found that the
/// photo as it is did not give up: a step of the retry ladder, run when
/// the finder scan comes up empty, as zxing-cpp retries with `tryDenoise`
/// (a morphological close) and `tryDownscale` (a pyramid of smaller
/// copies). Enlarging is the step those lack, for a symbol rendered a
/// pixel or two a module, which zxing-cpp reads by a path of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Retry {
    /// The dark closed over gaps of up to `radius` pixels, which joins
    /// the dots of a dot-matrix, inkjet or halftone print into solid
    /// modules.
    Closed { radius: u8 },
    /// Shrunk by a whole factor, which averages a screen or grain away.
    Downscaled { factor: u8 },
    /// Enlarged by a whole factor, which gives a symbol of a pixel a
    /// module runs the scan can see.
    Upscaled { factor: u8 },
    /// Turned clockwise by so many quarter turns, as zxing-cpp retries a
    /// failed read with `tryRotate`: the fits that place the corners and
    /// anchors walk the photo's axes, and a symbol that reads at one turn
    /// of the photo can sit a fraction of a module off at another.
    Turned { quarters: u8 },
    /// The scan's local threshold narrowed to the blocks within `radius`
    /// pixels of each pixel, where the outline sets a neighbourhood a
    /// sixth of its size: a finder pattern in a shadow or a glare
    /// narrower than that neighbourhood is otherwise judged against
    /// paper and ink that lie in another light, and its light runs read
    /// dark or its dark ones light. The photo itself is left as it is.
    /// OpenCV's ArUco-based QR detector thresholds at window sizes of
    /// three, thirteen and twenty-three pixels and merges what any size
    /// finds.
    Thresholded { radius: u8 },
    /// Sharpened, and enlarged or not, as [`super::enhance`] makes a
    /// copy of the photo: a symbol blurred by half a module and more
    /// has finder patterns whose rings have run into their stones, and
    /// no threshold reads them apart again; an unsharp mask does, as
    /// wechat_qrcode reads a small crop a second time through a network
    /// that enlarges and sharpens it.
    Enhanced(Enhance),
}

impl Retry {
    /// The ladder, cheapest step first: a smaller copy costs less than a
    /// filtered one, and that less than a larger one. The narrowed
    /// threshold costs a pass over the photo as it is, between the two,
    /// and so does the photo sharpened at its size.
    pub const LADDER: [Retry; 10] = [
        Retry::Downscaled { factor: 2 },
        Retry::Thresholded { radius: NARROW_RADIUS },
        Retry::Enhanced(Enhance { enlarge: 1, sharpen: Sharpen::Light }),
        Retry::Enhanced(Enhance { enlarge: 1, sharpen: Sharpen::Strong }),
        Retry::Closed { radius: 1 },
        Retry::Upscaled { factor: 2 },
        Retry::Enhanced(Enhance { enlarge: 2, sharpen: Sharpen::Light }),
        Retry::Downscaled { factor: 3 },
        Retry::Closed { radius: 2 },
        Retry::Upscaled { factor: 3 },
    ];
}

impl fmt::Display for Retry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Retry::Closed { radius } => write!(f, "the dark closed over gaps of {radius} px"),
            Retry::Downscaled { factor } => write!(f, "the photo shrunk {factor} times"),
            Retry::Upscaled { factor } => write!(f, "the photo enlarged {factor} times"),
            Retry::Turned { quarters } => write!(f, "the photo turned {}°", 90 * u16::from(quarters % 4)),
            Retry::Thresholded { radius } => write!(f, "the threshold narrowed to {radius} px about each pixel"),
            Retry::Enhanced(enhance) => write!(f, "the photo {enhance}"),
        }
    }
}

/// A fixed pattern found in the photo.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Anchor {
    pub kind: AnchorKind,
    /// Its centre in module coordinates.
    pub at: [f64; 2],
    /// Its centre in photo pixels.
    pub center: [f64; 2],
    /// One quarter of the pattern did not read right: torn, stained or
    /// covered. The other three placed it.
    pub partial: bool,
    /// The pattern was not found; the anchor stands where the lines
    /// through its found neighbours cross, which on flat or gently curled
    /// paper is near enough, and is better than nothing between them.
    pub inferred: bool,
}

impl Anchor {
    pub fn control_point(&self) -> ControlPoint {
        ControlPoint { at: self.at, px: self.center }
    }
}

/// A place in a version where a pattern worth anchoring on sits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnchorSite {
    pub kind: AnchorKind,
    /// The pattern's centre in module coordinates.
    pub at: [f64; 2],
}

/// Every pattern of a version that can serve as an anchor, finders first.
pub fn anchor_sites(func: &FunctionMap) -> Vec<AnchorSite> {
    sites(func).iter().map(|s| AnchorSite { kind: s.kind, at: s.at }).collect()
}

/// Places along the edges of a version's outline where an edge point may
/// be set by hand, to bow the outline there: a quarter, a half and three
/// quarters of the way along each edge, top, right, bottom and left in
/// turn, less any too near a corner to say anything the corner does not.
/// Dragged to where the edge really runs, each bends the map through it
/// as the edge points the detector traces do, for a symbol on a bottle
/// or on curled paper, whose sides bow and whose modules crowd together
/// towards its sides.
pub fn edge_sites(func: &FunctionMap) -> Vec<AnchorSite> {
    let (w, h) = (func.width as f64, func.height as f64);
    let mut out = Vec::new();
    for (from, along, length) in [
        ([0.0, 0.0], [1.0, 0.0], w),
        ([w, 0.0], [0.0, 1.0], h),
        ([w, h], [-1.0, 0.0], w),
        ([0.0, h], [0.0, -1.0], h),
    ] {
        for share in [0.25, 0.5, 0.75] {
            let s = share * length;
            if s < EDGE_ANCHOR_CLEARANCE || length - s < EDGE_ANCHOR_CLEARANCE {
                continue;
            }
            out.push(AnchorSite { kind: AnchorKind::Edge, at: [from[0] + along[0] * s, from[1] + along[1] * s] });
        }
    }
    out
}

/// What the symbol's own information bits, read through the outline found,
/// said about it. Bits off the photo or too faint to call are left
/// unknown, and too few known bits make both answers `None`. The version
/// is the one exception to "through the outline": where the read beside
/// the finders alone, made before any outline was fitted, names one the
/// fit bears out and names it more cleanly, its name stands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InfoEvidence {
    /// Whether the format bits spell a valid word (`Some(false)` when they
    /// read as noise, `None` when too few could be read to tell), and how
    /// far the nearest word is.
    pub format: bch::FormatVerdict,
    /// The version the bits name, where they carry one: the format word of
    /// a Micro QR or rMQR symbol, or the version information of a QR
    /// symbol of version 7 or more. `None` when there is none or it could
    /// not be read.
    pub named_version: Option<SymbolVersion>,
    /// Bits of the word naming the version that had to be corrected for
    /// it to read as that word: none when it read outright. `None` when
    /// nothing names a version.
    pub named_mismatches: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    /// Top-left, top-right, bottom-right, bottom-left, in photo pixels, as
    /// the symbol has them: a turned outline comes back turned straight,
    /// and on a mirrored photo the corners run the other way round.
    pub corners: [[f64; 2]; 4],
    /// The version the corners were placed for: the one whose fixed
    /// patterns read best, or the one the symbol's own bits name, which
    /// need not be the one asked about.
    pub version: SymbolVersion,
    /// The capstones read light on dark.
    pub inverted: bool,
    /// The photo shows the symbol mirrored, as from the back of the paper.
    /// The corners follow the symbol's own order, so a grid sampled
    /// through them reads the right way round without further ado.
    pub mirrored: bool,
    /// The mirror image was told not by the format bits but by decoding the
    /// symbol both ways round.
    pub mirrored_by_decode: bool,
    /// Quarter turns clockwise the symbol was found to sit in the outline
    /// given: how far the corners were turned to put the finder patterns
    /// where the version has them.
    pub turns: u8,
    pub anchors: Vec<Anchor>,
    /// (matching, fixed) over the version's fixed modules at these corners.
    pub agreement: (usize, usize),
    pub info: InfoEvidence,
    /// Corners placed by tracing the symbol's edge against its quiet zone.
    pub edged_corners: Vec<usize>,
    /// Corners placed by a finder pattern's own fit, to a fraction of a
    /// pixel: the refinement holds them, where the map through every
    /// anchor is a plane and on curled paper puts them a fraction of a
    /// module off.
    pub measured_corners: Vec<usize>,
    /// The photo was transformed this way to find the symbol: a step of
    /// the retry ladder, run when the photo as it is showed no finder
    /// pattern, or the enlargement a small outline is given from the
    /// start. `None` when the photo was searched as it is.
    pub retry: Option<Retry>,
    /// Whether the symbol decodes as found (see [`decodes_as_found`]).
    /// Asked by whoever lists detections for choosing among, which lists
    /// what decodes first; `None` when it was not asked.
    pub decodes: Option<bool>,
}

impl Detection {
    pub fn count(&self, kind: AnchorKind) -> usize {
        self.anchors.iter().filter(|a| a.kind == kind).count()
    }

    /// Anchors with a damaged quarter.
    pub fn partial(&self) -> usize {
        self.anchors.iter().filter(|a| a.partial).count()
    }

    /// Share of fixed modules that read right.
    pub fn score(&self) -> f64 {
        self.agreement.0 as f64 / self.agreement.1.max(1) as f64
    }

    /// The symbol's own bits name the version the corners were placed
    /// for, and read as its word outright, with no bit corrected, through
    /// the outline or beside the finders. A word within correction range
    /// names a version too, and the fit takes the hint, but it confirms
    /// little: more than half of all strings of fifteen bits lie within
    /// three of a Micro QR format word.
    pub fn version_confirmed(&self) -> bool {
        self.info.named_version == Some(self.version) && self.info.named_mismatches == Some(0)
    }

    /// The format word read through the outline is a word outright. The
    /// thirty-two words of a QR Code or Micro QR symbol are a thousandth
    /// of all strings of fifteen bits, so print that is no symbol spells
    /// one about that often, where it lies within correction range of
    /// one more often than not.
    fn format_reads_outright(&self) -> bool {
        self.info.format.valid == Some(true) && self.info.format.mismatches == Some(0)
    }

    /// The corners the refinement should hold where the detection put
    /// them: those a finder pattern's own fit measured, and those placed
    /// where two traced edges meet, which on curled paper the contrast of
    /// the whole outline, read through a plane, would walk off.
    pub fn held_corners(&self) -> Vec<usize> {
        let mut held: Vec<usize> = self.measured_corners.iter().chain(&self.edged_corners).copied().collect();
        held.sort_unstable();
        held.dedup();
        held
    }

    /// A square symbol's placement taken in mirror order: its corners
    /// round the other way and its anchors at their transposed modules,
    /// so that a grid sampled through it is the transpose of the grid
    /// sampled through this one. The information bits are as they were
    /// read here, for the caller to read again.
    fn transposed(&self) -> Detection {
        let mut swapped = self.clone();
        swapped.corners = [self.corners[0], self.corners[3], self.corners[2], self.corners[1]];
        swapped.edged_corners = self.edged_corners.iter().map(|&i| [0, 3, 2, 1][i]).collect();
        swapped.measured_corners = self.measured_corners.iter().map(|&i| [0, 3, 2, 1][i]).collect();
        for anchor in &mut swapped.anchors {
            anchor.at = [anchor.at[1], anchor.at[0]];
        }
        swapped.mirrored = !self.mirrored;
        swapped
    }

    /// The map from module coordinates to the photo through the corners
    /// and anchors found.
    pub fn projection(&self) -> Option<Projection> {
        let (w, h) = (self.version.width(), self.version.height());
        let controls: Vec<ControlPoint> = self.anchors.iter().map(Anchor::control_point).collect();
        Projection::new(w, h, self.corners, &controls)
    }
}

/// Most placements [`detect_candidates`] offers: room for the versions
/// nearest the size estimate and for another assignment of the capstones.
pub const MAX_CANDIDATE_DETECTIONS: usize = 4;

/// Find the symbol inside `region`, an outline drawn loosely around it as
/// top-left, top-right, bottom-right, bottom-left in photo pixels, and
/// return corners close to its true outline. `version` is the one in use:
/// it fixes the family, and is tried alongside the versions the finder
/// spacing suggests. `None` when no finder pattern can be found.
pub fn detect_outline(
    image: &Luma,
    version: SymbolVersion,
    region: [[f64; 2]; 4],
    params: &SampleParams,
) -> Option<Detection> {
    detect_candidates(image, version, region, params).into_iter().next()
}

/// The placements of the symbol inside `region` worth reading, best
/// first, at most [`MAX_CANDIDATE_DETECTIONS`]: the best fit of each
/// version and frame tried. The first is what [`detect_outline`] returns.
/// The fit of the fixed patterns ranks them, and where it ranks a wrong
/// placement first, as it may when the symbol is drawn as dots or the
/// finder spacing was misjudged, the runners-up are there for the decode
/// to arbitrate, as ZXing's and zxing-cpp's readers let the decode decide
/// what the fit cannot; [`super::read::read_outline`] does so.
pub fn detect_candidates(
    image: &Luma,
    version: SymbolVersion,
    region: [[f64; 2]; 4],
    params: &SampleParams,
) -> Vec<Detection> {
    detect_candidates_by(image, version, region, params, Climb::All)
}

/// How much of the retry ladder a detection climbs when the photo as it
/// is shows no symbol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Climb {
    /// Every rung.
    All,
    /// Every rung but the sharpened ones: a read that does not decode
    /// goes through the photo sharpened itself, the whole of it from the
    /// scan to the decode ([`super::read::read_outline`]), and the
    /// placements those rungs would hand it, to be read in the photo as
    /// it lies, it would only make again.
    Unsharpened,
    /// None: the pixels given are themselves a copy made for the look,
    /// enlarged and sharpened, and a copy of the copy shows little the
    /// copy does not, at several times the cost.
    Nothing,
}

/// [`detect_candidates`], climbing as much of the ladder as `climb` says.
pub(crate) fn detect_candidates_by(
    image: &Luma,
    version: SymbolVersion,
    region: [[f64; 2]; 4],
    params: &SampleParams,
    climb: Climb,
) -> Vec<Detection> {
    // A symbol rendered pixel for pixel is read off its pixels, exactly,
    // before any fitting.
    if let Some(found) = detect_pure(image, version.family(), region, params) {
        return vec![found];
    }
    // A small outline holds small modules, which the scan cannot see
    // below a pixel and a half: the photo around it is enlarged first.
    let enlarge = needed_upscale(region, version.family());
    if enlarge > 1 {
        let found = detect_candidates_through(image, version, region, params, Retry::Upscaled { factor: enlarge });
        if !found.is_empty() {
            return found;
        }
    }
    let found = detect_candidates_in(image, version, region, params, true, None, MAX_RECTIFIED);
    if found.is_empty() {
        // Nothing read as a symbol: the ladder, cheapest step first.
        let found = Retry::LADDER
            .into_iter()
            .filter(|step| !matches!(step, Retry::Upscaled { factor } if *factor <= enlarge))
            .filter(|step| match climb {
                Climb::All => true,
                Climb::Unsharpened => !matches!(step, Retry::Enhanced(_)),
                Climb::Nothing => false,
            })
            .take_while(|_| !work::cancelled())
            .map(|step| detect_candidates_through(image, version, region, params, step))
            .find(|found| !found.is_empty())
            .unwrap_or_default();
        // And last, for an outline so large that all of those scanned it
        // reduced, the scan in the photo's own pixels: no outline goes
        // unfound for want of pixels the photo has. Last, since it is
        // the dearest look there is, and on a photo with grain the one
        // most likely to take the grain for finders.
        if found.is_empty() && scanned_reduced(region) && !work::cancelled() {
            return detect_candidates_in(image, version, region, params, true, None, UNREDUCED);
        }
        return found;
    }
    // A second pass for a symbol whose modules are too small for the scan
    // to read well. The first pass enlarges the photo by what the family's
    // smallest symbol would need across the outline, and a large symbol
    // in the same outline needs several times that; the version found
    // says how small its modules really are, so the photo is enlarged by
    // that and searched again, and the better fit is kept.
    if let Some(factor) = upscale_for(&found[0]) {
        let bigger = detect_candidates_through(image, found[0].version, region, params, Retry::Upscaled { factor });
        if bigger.first().is_some_and(|b| b.score() > found[0].score()) {
            return bigger;
        }
    }
    found
}

/// The enlargement a symbol found in the photo as it is needs for its
/// modules to reach [`TARGET_MODULE_PX`], when they fall short of that;
/// `None` otherwise, or when the symbol was found through a transformed
/// copy already.
fn upscale_for(found: &Detection) -> Option<u8> {
    if found.retry.is_some() {
        return None;
    }
    let [c0, c1, c2, c3] = found.corners;
    let across = (distance(c0, c1) + distance(c3, c2)) / 2.0 / found.version.width() as f64;
    let down = (distance(c0, c3) + distance(c1, c2)) / 2.0 / found.version.height() as f64;
    let module = across.min(down);
    if !(module > 0.0 && module.is_finite()) || module >= TARGET_MODULE_PX {
        return None;
    }
    Some(((TARGET_MODULE_PX / module).ceil() as u8).clamp(2, MAX_UPSCALE))
}

/// [`detect_candidates`] through a transformed copy of the photo around
/// the outline, with the results brought back to photo pixels.
fn detect_candidates_through(
    image: &Luma,
    version: SymbolVersion,
    region: [[f64; 2]; 4],
    params: &SampleParams,
    step: Retry,
) -> Vec<Detection> {
    let Some(view) = View::around(image, region, step) else { return Vec::new() };
    let mut found = detect_candidates_in(&view.luma, version, region.map(|p| view.to_view(p)), params, true, view.local_radius, MAX_RECTIFIED);
    for detection in &mut found {
        view.unscale(detection);
        detection.retry = Some(step);
    }
    found
}

/// The whole factor the photo around `region` must be enlarged by for
/// the family's smallest symbol, drawn across the region's shorter side,
/// to have [`TARGET_MODULE_PX`] a module; one when it needs none.
fn needed_upscale(region: [[f64; 2]; 4], family: Family) -> u8 {
    let w = distance(region[0], region[1]).min(distance(region[3], region[2]));
    let h = distance(region[0], region[3]).min(distance(region[1], region[2]));
    let shorter = w.min(h);
    if !(shorter > 0.0 && shorter.is_finite()) {
        return 1;
    }
    let fewest = match family {
        Family::Qr | Family::Model1 => 21.0,
        Family::Micro => 11.0,
        Family::Rmqr => 7.0,
    };
    ((TARGET_MODULE_PX * fewest / shorter).ceil() as u8).clamp(1, MAX_UPSCALE)
}

/// Whether the scan of `region` for finder patterns looks at it reduced:
/// the outline, with the scan's margin about it, is longer than
/// [`MAX_RECTIFIED`] on a side.
fn scanned_reduced(region: [[f64; 2]; 4]) -> bool {
    let across = distance(region[0], region[1]).max(distance(region[3], region[2]));
    let down = distance(region[0], region[3]).max(distance(region[1], region[2]));
    (1.0 + 2.0 * MARGIN) * across.max(down) > MAX_RECTIFIED
}

/// [`detect_candidates`] on the photo as it is: the finder patterns inside
/// the outline, then a fit in each frame they propose, when `with_frames`,
/// and in the outline's own frame last; the placements found, best first.
/// `local_radius` narrows the scan's local threshold to that many pixels
/// about each pixel, as a rung of the ladder asks. `longest` is the most
/// the image rectified for the scan may measure on a side:
/// [`MAX_RECTIFIED`], or [`UNREDUCED`] for a scan in the photo's own
/// pixels.
fn detect_candidates_in(
    image: &Luma,
    version: SymbolVersion,
    region: [[f64; 2]; 4],
    params: &SampleParams,
    with_frames: bool,
    local_radius: Option<f64>,
    longest: f64,
) -> Vec<Detection> {
    let Some(rect) = Rectified::with_longest(image, region, longest).map(|rect| rect.narrowed(local_radius)) else { return Vec::new() };
    let threshold = fixed_threshold(params);
    let family = version.family();
    let all = collect_capstones(image, &rect, threshold, family);
    if all.is_empty() {
        return Vec::new();
    }
    // The outline given says where the symbol is and about how big, but
    // not which way up nor just where: drawn loosely, or square to the
    // photo around a symbol at an angle, it puts every finder far from
    // where the version expects one across it. The capstones themselves
    // say which way up, as ZXing and ZBar place a symbol from its finder
    // centres alone, so the outlines they propose are fitted too, each
    // in its own frame. The outline given is fitted first: drawn well, it
    // settles the symbol at once, and drawn badly it fails at once, with
    // no capstone near any finder's place in it.
    let mut frames: Vec<[[f64; 2]; 4]> = vec![region];
    if with_frames {
        frames.extend(
            capstone_frames(image, &rect, &all, family, threshold, params.band)
                .into_iter()
                .filter(|frame| !frames_alike(*frame, region))
                .take(max_frames(family)),
        );
    }
    let mut found: Vec<(Detection, Rank)> = Vec::new();
    for frame in frames {
        if work::cancelled() {
            break;
        }
        let Some(geometry) = Frame::new(frame, longest) else { continue };
        let capstones: Vec<Capstone> = all.iter().map(|c| c.reframed(&geometry)).collect();
        let mut settled = false;
        for (mut fit, ranked) in detect_in_frame(image, version, &geometry, &capstones, threshold, params) {
            fit.turns = turns_between(region, fit.corners);
            // The symbol's own bits name this version, or, for a symbol
            // with three finders, its fixed patterns and quiet zone read
            // nearly all right: no other frame will do better. A symbol
            // with one finder is settled by its word alone, and only by
            // one read outright. In an outline drawn loosely round a
            // Micro QR symbol the finder lies where a smaller version's
            // would lie across the outline, and that version's fixed
            // patterns, which are a subset of the symbol's own, read all
            // right there, with only the quiet zone it expects over the
            // symbol's data to say otherwise; and a word within
            // correction range is read in the symbol's data through some
            // wrong placement often enough, since more than half of all
            // strings of fifteen bits lie within three of one. The boxes
            // the finder proposes are the frames the right version fits
            // in, and for one finder they are few.
            settled |= if family.has_three_finders() {
                ranked.0 >= 1 || (ranked.0 >= 0 && ranked.1 >= SETTLED_FIT)
            } else {
                ranked.0 >= 1 && fit.info.named_mismatches == Some(0)
            };
            found.push((fit, ranked));
        }
        if settled {
            break;
        }
    }
    found.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = pick_candidates(found);
    for fit in &mut out {
        settle_mirror_by_decode(image, fit, threshold, params);
    }
    out
}

/// How a placement is told from another for the decode's sake: the
/// finder patterns it stands on, to the pixel, and the turn and
/// reflection it gives the symbol over them. Two versions placed on the
/// same finders the same way share a signature.
fn placement_signature(found: &Detection) -> (Vec<[i64; 2]>, u8, bool) {
    let mut finders: Vec<[i64; 2]> = found
        .anchors
        .iter()
        .filter(|a| a.kind == AnchorKind::Finder)
        .map(|a| [a.center[0].round() as i64, a.center[1].round() as i64])
        .collect();
    finders.sort_unstable();
    (finders, found.turns, found.mirrored)
}

/// The placements offered to the decode, from the fits ranked best first:
/// at most [`MAX_CANDIDATE_DETECTIONS`], the best of every signature
/// before any signature's second, and the same placement reached from
/// two frames once. A fit that stands two capstones as finders and leaves
/// a third out, turned to suit, can outrank the fit that stands all three
/// at every version near the size estimate; the decode must see both, as
/// zxing-cpp tries every set of finder patterns it makes until one
/// decodes.
fn pick_candidates(ranked: Vec<(Detection, Rank)>) -> Vec<Detection> {
    let alike = |kept: &[Detection], fit: &Detection| kept.iter().any(|o| o.version == fit.version && placements_alike(o.corners, fit.corners));
    let mut out: Vec<Detection> = Vec::new();
    let mut later: Vec<Detection> = Vec::new();
    let mut seen: Vec<(Vec<[i64; 2]>, u8, bool)> = Vec::new();
    for (fit, _) in ranked {
        if alike(&out, &fit) || alike(&later, &fit) {
            continue;
        }
        let signature = placement_signature(&fit);
        if out.len() < MAX_CANDIDATE_DETECTIONS && !seen.contains(&signature) {
            seen.push(signature);
            out.push(fit);
        } else {
            later.push(fit);
        }
    }
    for fit in later {
        if out.len() >= MAX_CANDIDATE_DETECTIONS {
            break;
        }
        out.push(fit);
    }
    out
}

/// A square symbol whose format bits say nothing: decide the mirror image
/// by decoding both ways round, as ZXing and ZBar retry a failed decode
/// mirrored.
fn settle_mirror_by_decode(image: &Luma, found: &mut Detection, threshold: Option<f32>, params: &SampleParams) {
    if found.version.family() == Family::Rmqr || found.info.format.valid == Some(true) {
        return;
    }
    let func = Arc::new(FunctionMap::new(found.version));
    let plain = trial_decode(image, &func, found, params);
    let mut swapped = found.transposed();
    let mirrored = trial_decode(image, &func, &swapped, params);
    if let (Some(plain), Some(mirrored)) = (plain, mirrored)
        && mirrored.blocks_verified > plain.blocks_verified
    {
        swapped.mirrored_by_decode = true;
        swapped.info = read_info(image, &func, swapped.corners, found.inverted, threshold, params.band);
        *found = swapped;
    }
}

/// A square symbol's placement as it would be were the photo of the back
/// of the paper, for a read to try when the placement as found does not
/// decode: the same outline and anchors taken in mirror order, and the
/// symbol's information bits read again through them. `None` for an
/// rMQR symbol, whose finder and sub-pattern settle its mirror image.
///
/// [`settle_mirror_by_decode`] asks the decode only when the format bits
/// say nothing, and the format bits of a mirrored symbol say something
/// more often than not: twenty-six of a QR Code's thirty-two format words
/// read in mirror order lie exactly three bits from some other word, so
/// a mirrored symbol with a bit or two of its format word misread passes
/// for a plain one with a word within correction, and was decoded as
/// that and no other way. ZXing and wechat_qrcode read a symbol mirrored
/// whenever it has failed to decode plain, whatever its format bits.
pub(crate) fn seen_from_behind(image: &Luma, found: &Detection, params: &SampleParams) -> Option<Detection> {
    if found.version.family() == Family::Rmqr {
        return None;
    }
    let mut swapped = found.transposed();
    swapped.mirrored_by_decode = true;
    swapped.info = read_info(image, &FunctionMap::new(found.version), swapped.corners, found.inverted, fixed_threshold(params), params.band);
    Some(swapped)
}

/// Least share of fixed modules that must read right for a symbol read
/// straight off the pixels to be believed: such a read is exact or wrong.
const PURE_MIN_FIT: f64 = 0.97;

/// A symbol rendered pixel for pixel inside `region`, read off the pixels
/// themselves (see [`super::pure`]), with its corners on the pixel edges
/// and no anchors, since a map through four exact corners is exact.
/// `None` when the region holds no such symbol, and the photo pipeline
/// takes over.
fn detect_pure(image: &Luma, family: Family, region: [[f64; 2]; 4], params: &SampleParams) -> Option<Detection> {
    // Dark on light, then light on dark: the first reading whose fixed
    // patterns fit.
    super::pure::find_each(image, family, region).into_iter().find_map(|pure| pure_detection(image, family, region, pure, params))
}

/// The detection of a symbol read straight off the pixels, when its
/// fixed patterns bear the reading out.
fn pure_detection(
    image: &Luma,
    family: Family,
    region: [[f64; 2]; 4],
    pure: super::pure::PureSymbol,
    params: &SampleParams,
) -> Option<Detection> {
    let func = FunctionMap::new(pure.version);
    let threshold = fixed_threshold(params);
    let projection = Projection::new(func.width, func.height, pure.corners, &[])?;
    let scorer = Scorer::new(image, &Template::whole(&func), pure.corners, threshold);
    let (matching, _) = scorer.agreement_known(&projection, pure.inverted)?;
    if (matching as f64) < PURE_MIN_FIT * scorer.fixed() as f64 {
        return None;
    }
    let info = read_info(image, &func, pure.corners, pure.inverted, threshold, params.band);
    let mut found = Detection {
        corners: pure.corners,
        version: pure.version,
        inverted: pure.inverted,
        mirrored: false,
        mirrored_by_decode: false,
        turns: turns_between(region, pure.corners),
        anchors: Vec::new(),
        agreement: (matching, scorer.fixed()),
        info,
        edged_corners: Vec::new(),
        measured_corners: vec![0, 1, 2, 3],
        retry: None,
        decodes: None,
    };
    // A square symbol seen from the back reads as its transpose; the
    // format bits decide, as they do for a fitted symbol.
    if family != Family::Rmqr {
        let swapped = [pure.corners[0], pure.corners[3], pure.corners[2], pure.corners[1]];
        let swapped_info = read_info(image, &func, swapped, pure.inverted, threshold, params.band);
        if bch::mirror_wins(found.info.format, swapped_info.format) {
            found.corners = swapped;
            found.mirrored = true;
            found.info = swapped_info;
        }
    }
    settle_mirror_by_decode(image, &mut found, threshold, params);
    Some(found)
}

/// Quarter turns clockwise a symbol with `corners` sits in `outline`, to
/// the nearest quarter: for telling the user, not for the fit.
pub(crate) fn turns_between(outline: [[f64; 2]; 4], corners: [[f64; 2]; 4]) -> u8 {
    let o = [outline[1][0] - outline[0][0], outline[1][1] - outline[0][1]];
    let s = [corners[1][0] - corners[0][0], corners[1][1] - corners[0][1]];
    let angle = (o[0] * s[1] - o[1] * s[0]).atan2(o[0] * s[0] + o[1] * s[1]);
    ((angle / std::f64::consts::FRAC_PI_2).round() as i64).rem_euclid(4) as u8
}

/// Versions tried beyond [`candidate_versions`] when every version tried
/// fits poorly: the size estimate may then be off by more than the
/// versions around it cover, as it is for a symbol drawn as dots, whose
/// finder rings measure wrong.
const WIDEN_VERSIONS: usize = 2;

/// The placements of the symbol in one frame, each with its rank: the
/// capstones, already seen in that frame, matched to the finders of each
/// version worth trying and the fit grown from them.
fn detect_in_frame(
    image: &Luma,
    version: SymbolVersion,
    frame: &Frame,
    all: &[Capstone],
    threshold: Option<f32>,
    params: &SampleParams,
) -> Vec<(Detection, Rank)> {
    // Capstones of both polarities may turn up: an over-exposed symbol's
    // data reads, at half scale, as light-centred finders. Whichever
    // polarity's capstones fit a version's fixed patterns better is the
    // photo's; where only one polarity has capstones there is no choice.
    let mut out: Vec<(Detection, Rank)> = Vec::new();
    for inverted in [false, true] {
        // The light-centred reading of the finders is the right one only
        // when the dark-centred reading fits nothing well.
        if inverted && out.iter().any(|(_, r)| r.0 >= 1 || r.1 >= SETTLED_FIT) {
            break;
        }
        let capstones: Vec<Capstone> = all.iter().copied().filter(|c| c.inverted == inverted).collect();
        if capstones.is_empty() {
            continue;
        }
        let (hint, tight) = size_hint(image, frame, &capstones, version.family());
        // The symbol's own bits, read beside the finders alone, may name
        // the version before anything is fitted. That version is fitted
        // first; the others are fitted after it, as the fallback.
        let prename = prename_version(image, &capstones, version.family(), inverted, threshold, params.band, hint, tight);
        let mut tried = candidate_versions(version, hint, tight);
        if let Some(p) = prename {
            tried.retain(|&v| v != p.version);
            tried.insert(0, p.version);
        }
        let mut fits: Vec<(Detection, f64)> = Vec::new();
        let mut next = 0;
        let mut widened = false;
        while next < tried.len() && !work::cancelled() {
            let func = FunctionMap::new(tried[next]);
            next += 1;
            for found in align_version(image, &func, &capstones, inverted, threshold, params.band, prename) {
                // The symbol may name a version the size estimate did not reach.
                if let Some(named) = found.0.info.named_version
                    && named.family() == version.family()
                    && !tried.contains(&named)
                {
                    tried.push(named);
                }
                fits.push(found);
            }
            // Every version tried fits poorly: the next nearest are tried
            // too, once, and the decode may yet tell them apart.
            if next == tried.len() && !widened && fits.iter().all(|(_, fit)| *fit < MIN_FOUND_SCORE) {
                widened = true;
                tried.extend(further_versions(version.family(), hint, &tried));
            }
        }
        // The pre-read is believed only while the version it names fits
        // tolerably. A name the fit contradicts is treated as no name at
        // all, and the fits decide among themselves, as they do for a
        // symbol whose bits say nothing.
        let trusted = prename.filter(|p| fits.iter().any(|(d, fit)| d.version == p.version && *fit >= MIN_NAMED_FIT));
        for (mut found, fit) in fits {
            if let Some(p) = trusted
                && found.info.named_version.is_none()
            {
                found.info.named_version = Some(p.version);
                found.info.named_mismatches = Some(p.mismatches);
            }
            let ranked = rank(&found, fit, trusted);
            // Once ranked, on what each read said for itself, the cleaner
            // of two names stands. The read beside the finders was made
            // where the geometry is surest; the read through the outline
            // is a bit or two out wherever the outline is, as it is at
            // the far corners of a symbol whose timing patterns are gone.
            if let Some(p) = trusted
                && found.info.named_mismatches.is_some_and(|m| p.mismatches < m)
            {
                found.info.named_version = Some(p.version);
                found.info.named_mismatches = Some(p.mismatches);
            }
            out.push((found, ranked));
        }
    }
    out
}

/// The versions of `family` nearest the size estimate that are not among
/// `tried`, at most [`WIDEN_VERSIONS`] of them.
fn further_versions(family: Family, hint: [f64; 2], tried: &[SymbolVersion]) -> Vec<SymbolVersion> {
    let error = |v: SymbolVersion| size_error(v, hint);
    let mut versions: Vec<SymbolVersion> = family.versions().into_iter().filter(|v| !tried.contains(v)).collect();
    versions.sort_by(|a, b| error(*a).total_cmp(&error(*b)));
    versions.truncate(WIDEN_VERSIONS);
    versions
}

/// Sample the symbol through a detection and decode it under every format,
/// returning how the best attempt ranked. `None` when the outline cannot
/// be sampled.
fn trial_decode(image: &Luma, func: &Arc<FunctionMap>, found: &Detection, params: &SampleParams) -> Option<AttemptScore> {
    let grid = grid_as_found(image, func, found, params)?;
    let opts = DecodeOptions { reliability: false, ..DecodeOptions::default() };
    brute_force(&grid, func, None, &opts).first().map(|a| a.report.score)
}

/// The symbol sampled once through a detection's corners and anchors,
/// with nothing refined, dark for dark whichever way the photo has it.
/// `None` when the outline cannot be sampled. A symbol found only in
/// the photo sharpened or enlarged is sampled through that copy, where
/// its modules read as the search saw them, and any other off the
/// pixels given as they are, whatever `params` ask of a read: a
/// detection still being made in a copy of the photo is placed in that
/// copy's pixels, which are what it was found in.
fn grid_as_found(image: &Luma, func: &FunctionMap, found: &Detection, params: &SampleParams) -> Option<ModuleGrid> {
    let map = found.projection()?;
    let enhance = match found.retry {
        Some(Retry::Enhanced(enhance)) => enhance,
        _ => Enhance::NONE,
    };
    let sampled = sample_modules_as(image, &map, func.width, func.height, &SampleParams { kernel: Kernel::Area, enhance, ..*params });
    let grid = classify(&sampled, params).to_grid(func.width, func.height);
    Some(if found.inverted { grid.inverted() } else { grid })
}

/// Whether the symbol decodes as a detection places it: sampled once
/// through the corners and anchors found and decoded under every format,
/// every block solving with confirmation and the stream parsing. Print
/// that is no symbol does not, which is every reader's last word on a
/// candidate; but neither does a damaged symbol, nor every sound one
/// before its corners are refined, so this ranks what a search of the
/// whole photo lists and rules nothing out.
pub fn decodes_as_found(image: &Luma, found: &Detection, params: &SampleParams) -> bool {
    let func = Arc::new(FunctionMap::new(found.version));
    trial_decode(image, &func, found, params).is_some_and(|score| score.all_verified && score.stream_valid)
}

/// How a placement ranks against others: first by what the symbol's own
/// bits say of its version (named beyond doubt beside the finders, named
/// through the fitted outline, nothing, or another version), then by how
/// well its fixed patterns and quiet zone read. A name read through an
/// outline that fits worse than [`MIN_NAMED_FIT`] counts as nothing, as
/// a pre-read the fit contradicts does: more than half of all strings of
/// fifteen bits lie within three of a Micro QR format word, so the bits
/// beside a false finder name a version more often than not, and that
/// version is the one then fitted there.
type Rank = (i8, f64);

fn rank(found: &Detection, fit: f64, trusted: Option<Prename>) -> Rank {
    let verdict = match (trusted, found.info.named_version) {
        (Some(p), _) if p.strong && p.version == found.version => 2,
        (_, Some(named)) if named == found.version && fit >= MIN_NAMED_FIT => 1,
        (_, Some(named)) if named == found.version => 0,
        (_, Some(_)) => -1,
        (_, None) => 0,
    };
    (verdict, fit)
}

/// The finder patterns of `family` verified against the photo inside
/// `rect`: the scan's candidates, strongest first, each fitted and read,
/// with repeats merged.
fn collect_capstones(image: &Luma, rect: &Rectified, threshold: Option<f32>, family: Family) -> Vec<Capstone> {
    let max_module = rect.frame.size[0].min(rect.frame.size[1]) * max_module_fraction(family);
    let candidates = scan_for_finders(rect, threshold, max_module);
    let regions = region_candidates(rect, threshold, max_module, &candidates);
    let mut all: Vec<Capstone> = Vec::new();
    // Whole candidates first, then the half ones, which passed the
    // cross-check on one side only and come from the data as often as
    // from a torn finder, then the regions the scan did not propose.
    let whole = candidates.iter().filter(|c| !c.half).take(MAX_CANDIDATES);
    let half = candidates.iter().filter(|c| c.half).take(MAX_HALF_CANDIDATES);
    for candidate in whole.chain(half).chain(regions.iter()) {
        let Some(capstone) = refine_capstone(image, rect, candidate, threshold) else { continue };
        merge_capstone(&mut all, rect, capstone);
    }
    drop_faint_partials(&mut all);
    all
}

/// Add a capstone to those gathered so far. The same finder found again,
/// from another candidate of the scan, perhaps at another size: the
/// stronger reading wins. A reading of the opposite polarity is kept
/// apart, since a finder's rings also read as a finder of the other
/// polarity at another scale, and which polarity is the photo's is
/// settled by the fit.
fn merge_capstone(all: &mut Vec<Capstone>, rect: &Rectified, capstone: Capstone) {
    let same = all.iter().position(|c| {
        let apart = distance(rect.to_rectified(c.center), rect.to_rectified(capstone.center));
        c.inverted == capstone.inverted && apart < 5.0 * c.module_mean().max(capstone.module_mean())
    });
    match same {
        Some(i) if capstone.contrast > all[i].contrast => all[i] = capstone,
        Some(_) => {}
        None => all.push(capstone),
    }
}

/// With more candidates refined, more false patterns reach this far, and
/// they are the partial ones: three quarters of a finder read in the
/// data, at a fraction of a true finder's contrast. Those fainter than
/// [`PARTIAL_CONTRAST_SHARE`] of the strongest whole capstone go.
fn drop_faint_partials(all: &mut Vec<Capstone>) {
    let strongest = all.iter().filter(|c| !c.partial).map(|c| c.contrast).fold(f64::NEG_INFINITY, f64::max);
    if strongest.is_finite() {
        all.retain(|c| !c.partial || c.contrast >= PARTIAL_CONTRAST_SHARE * strongest);
    }
}

/// Most candidates the regions of the binarised image add to the scan's.
const MAX_REGION_CANDIDATES: usize = 16;

/// Finder candidates from the connected regions of the binarised image,
/// as BoofCV finds position patterns: a quadrilateral fitted to each
/// region and the 1:1:3:1:1 proportion read along two lines through it.
/// The scan wants the proportion intact along two rows or columns through
/// the centre, which a finder scratched through both ways no longer has,
/// and reads the runs square to the frame, which a finder seen steeply
/// through a frame no outline rectified does not keep; a region keeps its
/// shape through either. Regions within a module of a candidate the scan
/// already made are left to it, and the template fit verifies the rest as
/// it verifies the scan's.
fn region_candidates(rect: &Rectified, threshold: Option<f32>, max_module: f64, scanned: &[Candidate]) -> Vec<Candidate> {
    let global = threshold.unwrap_or_else(|| otsu(rect.data.iter().copied().filter(|v| v.is_finite())));
    let global_dark = |x: i64, y: i64| rect.at(x, y).map(|v| v < global);
    let local = LocalMean::new(rect);
    let local_dark = |x: i64, y: i64| local.dark(rect, x, y);
    let binarisers: [&dyn Fn(i64, i64) -> Option<bool>; 2] = [&global_dark, &local_dark];
    let mut out: Vec<Candidate> = Vec::new();
    for dark in binarisers {
        for region in regions::finder_regions(rect.frame.width, rect.frame.height, dark, MIN_MODULE, max_module) {
            let module = (region.module[0] + region.module[1]) / 2.0;
            let near = |c: &Candidate| c.inverted == region.inverted && distance([c.x, c.y], [region.x, region.y]) < module.max(c.module_mean());
            if scanned.iter().any(near) || out.iter().any(near) {
                continue;
            }
            let quad = region.corners.map(|[x, y]| rect.frame.to_image.apply(x, y));
            out.push(Candidate {
                x: region.x,
                y: region.y,
                module: region.module,
                inverted: region.inverted,
                votes: MIN_VOTES,
                half: false,
                quad: Some(quad),
            });
        }
    }
    out.truncate(MAX_REGION_CANDIDATES);
    out
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Where the line through the two alignment anchors nearest `site` along
/// its row crosses the line through the two nearest down its column, in
/// photo pixels. `None` without two of each, or with the lines parallel.
fn neighbours_crossing(site: &Site, anchors: &[Anchor]) -> Option<[f64; 2]> {
    let mates = |axis: usize| -> Option<([f64; 2], [f64; 2])> {
        // Anchors on the same row (axis 0) or column (axis 1), nearest first.
        let mut along: Vec<&Anchor> = anchors
            .iter()
            .filter(|a| a.kind == AnchorKind::Alignment && (a.at[1 - axis] - site.at[1 - axis]).abs() < 1e-6 && (a.at[axis] - site.at[axis]).abs() > 1e-6)
            .collect();
        along.sort_by(|a, b| (a.at[axis] - site.at[axis]).abs().total_cmp(&(b.at[axis] - site.at[axis]).abs()));
        match along[..] {
            [a, b, ..] => Some((a.center, b.center)),
            _ => None,
        }
    };
    let (p1, p2) = mates(0)?;
    let (q1, q2) = mates(1)?;
    let (dp, dq) = ([p2[0] - p1[0], p2[1] - p1[1]], [q2[0] - q1[0], q2[1] - q1[1]]);
    let cross = dp[0] * dq[1] - dp[1] * dq[0];
    if cross.abs() < 1e-9 {
        return None;
    }
    let t = ((q1[0] - p1[0]) * dq[1] - (q1[1] - p1[1]) * dq[0]) / cross;
    Some([p1[0] + t * dp[0], p1[1] + t * dp[1]])
}

/// The geometry of the photo seen through an outline, with a margin
/// around it, at about one rectified pixel per photo pixel: the map
/// between rectified pixels and photo pixels, and where the outline lies
/// in the rectified image.
#[derive(Clone, Copy, Debug)]
struct Frame {
    /// Size of the rectified image in pixels.
    width: usize,
    height: usize,
    /// Rectified pixel coordinates to photo pixels.
    to_image: Homography,
    from_image: Homography,
    /// Rectified pixel of the outline's top-left corner.
    origin: [f64; 2],
    /// The outline's size in rectified pixels.
    size: [f64; 2],
}

impl Frame {
    /// The frame through `region`, with the longest side of the rectified
    /// image capped at `longest` pixels.
    fn new(region: [[f64; 2]; 4], longest: f64) -> Option<Self> {
        let w = distance(region[0], region[1]).max(distance(region[3], region[2]));
        let h = distance(region[0], region[3]).max(distance(region[1], region[2]));
        if !(w >= 8.0 && h >= 8.0 && w.is_finite() && h.is_finite()) {
            return None;
        }
        let scale = (longest / ((1.0 + 2.0 * MARGIN) * w.max(h))).min(1.0);
        let size = [w * scale, h * scale];
        let origin = [MARGIN * size[0], MARGIN * size[1]];
        let width = ((1.0 + 2.0 * MARGIN) * size[0]).ceil() as usize;
        let height = ((1.0 + 2.0 * MARGIN) * size[1]).ceil() as usize;
        let unit_to_image = Homography::rect_to_quad(1.0, 1.0, region)?;
        let pixel_to_unit = Homography::from_matrix([
            1.0 / size[0],
            0.0,
            -origin[0] / size[0],
            0.0,
            1.0 / size[1],
            -origin[1] / size[1],
            0.0,
            0.0,
            1.0,
        ])?;
        let to_image = unit_to_image.compose(&pixel_to_unit);
        let from_image = to_image.inverse()?;
        Some(Self { width, height, to_image, from_image, origin, size })
    }

    fn to_rectified(self, point: [f64; 2]) -> [f64; 2] {
        self.from_image.apply(point[0], point[1])
    }

    /// Where a photo point lies across the outline, 0 to 1 along each side.
    fn unit(self, point: [f64; 2]) -> [f64; 2] {
        let [px, py] = self.to_rectified(point);
        [(px - self.origin[0]) / self.size[0], (py - self.origin[1]) / self.size[1]]
    }
}

/// The photo resampled through a [`Frame`].
struct Rectified {
    frame: Frame,
    /// Row-major luminance; NaN where the photo has no pixel.
    data: Vec<f32>,
    /// How far the luminance reaches from the darkest to the lightest
    /// hundredth of the pixels: the scale the contrast floors are set on.
    spread: f32,
    /// How far around each pixel the scan's local threshold reaches, in
    /// rectified pixels: what the outline's size sets, unless a rung of
    /// the ladder narrowed it.
    local_radius: f64,
}

impl Rectified {
    /// The photo inside `region`, reduced as an outline the search of
    /// the whole photo proposes is.
    #[cfg(test)]
    fn new(image: &Luma, region: [[f64; 2]; 4]) -> Option<Self> {
        Self::with_longest(image, region, MAX_RECTIFIED)
    }

    /// The photo inside `region`, and a margin about it, resampled square
    /// to the region, with the longest side of the result capped at
    /// `longest` pixels: a larger region comes out reduced.
    fn with_longest(image: &Luma, region: [[f64; 2]; 4], longest: f64) -> Option<Self> {
        let frame = Frame::new(region, longest)?;
        let mut data = Vec::with_capacity(frame.width * frame.height);
        for py in 0..frame.height {
            for px in 0..frame.width {
                let [x, y] = frame.to_image.apply(px as f64 + 0.5, py as f64 + 0.5);
                data.push(image.sample(x, y).unwrap_or(f32::NAN));
            }
        }
        let spread = robust_spread(data.iter().copied().filter(|v| v.is_finite()));
        let local_radius = BlockMean::radius_for(frame.size[0].min(frame.size[1]));
        Some(Self { frame, data, spread, local_radius })
    }

    /// The same, with the scan's local threshold reaching `radius`
    /// pixels about each pixel when one is given, instead of what the
    /// outline's size sets.
    fn narrowed(mut self, radius: Option<f64>) -> Self {
        if let Some(radius) = radius {
            self.local_radius = radius;
        }
        self
    }

    fn at(&self, x: i64, y: i64) -> Option<f32> {
        if x < 0 || y < 0 || x as usize >= self.frame.width || y as usize >= self.frame.height {
            return None;
        }
        let value = self.data[y as usize * self.frame.width + x as usize];
        value.is_finite().then_some(value)
    }

    fn to_rectified(&self, point: [f64; 2]) -> [f64; 2] {
        self.frame.to_rectified(point)
    }

    /// Where a photo point lies across the outline, 0 to 1 along each side.
    fn unit(&self, point: [f64; 2]) -> [f64; 2] {
        self.frame.unit(point)
    }
}

/// The photo, or a window of it, transformed for one attempt at finding
/// a symbol: closed, shrunk or enlarged. What is found in the view maps
/// back to photo pixels through its origin and scale.
struct View {
    luma: Luma,
    /// The photo pixel at the view's origin.
    origin: [f64; 2],
    /// Photo pixels per view pixel.
    scale: f64,
    /// The reach of the scan's local threshold in view pixels, when the
    /// step narrows it; `None` leaves it to the outline's size.
    local_radius: Option<f64>,
}

impl View {
    /// The photo around `region`, with the search's margin, transformed by
    /// `step`.
    fn around(image: &Luma, region: [[f64; 2]; 4], step: Retry) -> Option<Self> {
        let (x0, x1) = region.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p[0]), a.1.max(p[0])));
        let (y0, y1) = region.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p[1]), a.1.max(p[1])));
        if !(x0.is_finite() && x1.is_finite() && y0.is_finite() && y1.is_finite()) {
            return None;
        }
        let (mx, my) = (MARGIN * (x1 - x0) + 2.0, MARGIN * (y1 - y0) + 2.0);
        let window = [
            (x0 - mx).floor().max(0.0) as usize,
            (y0 - my).floor().max(0.0) as usize,
            (x1 + mx).ceil().clamp(0.0, image.width as f64) as usize,
            (y1 + my).ceil().clamp(0.0, image.height as f64) as usize,
        ];
        Self::window(image, window, step)
    }

    /// The whole photo transformed by `step`.
    fn whole(image: &Luma, step: Retry) -> Option<Self> {
        Self::window(image, [0, 0, image.width, image.height], step)
    }

    /// The window `[x0, y0, x1, y1]` of the photo transformed by `step`.
    /// `None` when the window is empty, or the view would be too small to
    /// hold a symbol or too large to hold at all.
    fn window(image: &Luma, [x0, y0, x1, y1]: [usize; 4], step: Retry) -> Option<Self> {
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        let (w, h) = (x1 - x0, y1 - y0);
        let at = |x: usize, y: usize| image.data[y * image.width + x];
        let crop = || -> Vec<f32> { (y0..y1).flat_map(|y| (x0..x1).map(move |x| at(x, y))).collect() };
        let mut local_radius = None;
        let (luma, scale) = match step {
            Retry::Closed { radius } => (Luma::new(w, h, Filter::Close.apply(&crop(), w, h, radius as usize)), 1.0),
            // The photo as it is: only the scan's threshold changes.
            Retry::Thresholded { radius } => {
                local_radius = Some(f64::from(radius));
                (Luma::new(w, h, crop()), 1.0)
            }
            Retry::Downscaled { factor } => {
                let k = factor as usize;
                let (vw, vh) = (w / k, h / k);
                if vw < 8 || vh < 8 {
                    return None;
                }
                // The whole pixels of the view, so that each is `k` of
                // the photo's on a side: what is left over at the far
                // edges is left out.
                let whole: Vec<f32> = (y0..y0 + vh * k).flat_map(|y| (x0..x0 + vw * k).map(move |x| at(x, y))).collect();
                (Luma::new(vw * k, vh * k, whole).resized(vw, vh), k as f64)
            }
            Retry::Upscaled { factor } => {
                let f = factor as usize;
                let (vw, vh) = (w * f, h * f);
                if vw.max(vh) > MAX_VIEW_SIDE {
                    return None;
                }
                (Luma::new(w, h, crop()).resized(vw, vh), 1.0 / f as f64)
            }
            // Sharpened and enlarged from the pixels the photo had, and
            // begun where a whole one of those begins.
            Retry::Enhanced(enhance) => {
                let view = Enhanced::window(image, [x0, y0, x1, y1], enhance)?;
                return Some(Self { luma: view.luma, origin: view.origin, scale: 1.0 / view.factor, local_radius: None });
            }
            // A turn is not a view of a window: the whole photo is turned
            // and read in full by `read`, which brings the result back.
            Retry::Turned { .. } => return None,
        };
        Some(Self { luma, origin: [x0 as f64, y0 as f64], scale, local_radius })
    }

    fn to_view(&self, p: [f64; 2]) -> [f64; 2] {
        [(p[0] - self.origin[0]) / self.scale, (p[1] - self.origin[1]) / self.scale]
    }

    fn to_photo(&self, p: [f64; 2]) -> [f64; 2] {
        [self.origin[0] + p[0] * self.scale, self.origin[1] + p[1] * self.scale]
    }

    /// A detection made in the view, brought back to photo pixels.
    fn unscale(&self, found: &mut Detection) {
        found.corners = found.corners.map(|c| self.to_photo(c));
        for anchor in &mut found.anchors {
            anchor.center = self.to_photo(anchor.center);
        }
    }
}

/// The scan's local threshold over the rectified image: the block mean of
/// [`super::binarize`], with the radius the outline's size sets, or the
/// one a rung of the ladder narrowed it to.
struct LocalMean {
    blocks: BlockMean,
}

impl LocalMean {
    fn new(rect: &Rectified) -> Self {
        Self { blocks: BlockMean::new(&rect.data, rect.frame.width, rect.frame.height, rect.local_radius) }
    }

    /// Whether the pixel is dark against its surroundings; `None` where
    /// there is no pixel.
    fn dark(&self, rect: &Rectified, x: i64, y: i64) -> Option<bool> {
        let value = rect.at(x, y)?;
        Some(self.blocks.is_dark(value, x as usize, y as usize))
    }
}

/// A place where rows and a column cross a finder-shaped run of dark and
/// light, in rectified pixels.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    x: f64,
    y: f64,
    /// Module size along each axis.
    module: [f64; 2],
    inverted: bool,
    /// Rows that agreed on it.
    votes: usize,
    /// Only one side of the column through the centre read as a finder:
    /// the other may be torn away. Such a candidate needs more rows behind
    /// it, and the template fit has the last word.
    half: bool,
    /// The outer quadrilateral of the pattern in photo pixels, top-left
    /// first and clockwise, where a region proposed the candidate: it
    /// carries the turn and the perspective the scan's runs do not.
    quad: Option<[[f64; 2]; 4]>,
}

impl Candidate {
    fn module_mean(&self) -> f64 {
        (self.module[0] + self.module[1]) / 2.0
    }
}

/// Rows a candidate needs before it is worth fitting a template to, when
/// its column cross-checked fully and when only half of it did.
const MIN_VOTES: usize = 2;
const MIN_VOTES_HALF: usize = 3;

/// The module size five run lengths imply when they are in the finder's
/// 1:1:3:1:1 proportion, and `None` when they are not. Each run may be off
/// by `tolerance` of a module; or, as ZBar tests it, each pair of adjacent
/// runs may be, which a photo whose dark has bled into its light, or the
/// reverse, still passes: what one run loses the next gains. Or the outer
/// two runs may be any length from a module up: a symbol printed hard
/// against other print has its finder's outer ring run into it, and the
/// inner three runs fix the size alone.
fn finder_module(counts: [f64; 5], tolerance: f64) -> Option<f64> {
    if counts.iter().any(|&c| c <= 0.0) {
        return None;
    }
    let module = counts.iter().sum::<f64>() / 7.0;
    let variance = module * tolerance;
    let runs = [1.0, 1.0, 3.0, 1.0, 1.0].iter().zip(&counts).all(|(k, c)| (c - k * module).abs() < variance);
    if runs {
        return Some(module);
    }
    // Bleeding takes the same from every dark run and gives the same to
    // every light one, so the runs on either side of the centre must still
    // match each other.
    let pairs = [2.0, 4.0, 4.0, 2.0];
    let centre_widest = counts[2] > counts[1] && counts[2] > counts[3];
    let symmetric = (counts[0] - counts[4]).abs() < variance && (counts[1] - counts[3]).abs() < variance;
    if centre_widest && symmetric && pairs.iter().enumerate().all(|(i, k)| (counts[i] + counts[i + 1] - k * module).abs() < variance) {
        return Some(module);
    }
    let inner = (counts[1] + counts[2] + counts[3]) / 5.0;
    let variance = inner * tolerance;
    let inner_ok = (counts[1] - inner).abs() < variance && (counts[3] - inner).abs() < variance && (counts[2] - 3.0 * inner).abs() < variance;
    let outer_ok = counts[0] >= inner * (1.0 - tolerance) && counts[4] >= inner * (1.0 - tolerance);
    (inner_ok && outer_ok).then_some(inner)
}

/// True when five run lengths are in the finder's proportion.
#[cfg(test)]
fn finder_ratio(counts: [f64; 5], tolerance: f64) -> bool {
    finder_module(counts, tolerance).is_some()
}

/// Walk out from `start` along one line through the middle run and the two
/// on either side, and check that they are finder-shaped, within
/// `tolerance` of a module each, and about `expected_module` in size.
/// Returns the centre of the middle run and the module size. The outer
/// runs are followed for two modules at most: past that they may be other
/// print the finder stands against.
fn cross_check(
    dark_at: impl Fn(i64) -> Option<bool>,
    start: i64,
    centre_dark: bool,
    expected_module: f64,
    tolerance: f64,
    fit: (f64, f64),
) -> Option<(f64, f64)> {
    let max_light = (2.0 * expected_module).ceil() as i64;
    let max_outer = (2.0 * expected_module).ceil() as i64;
    let max_centre = (5.0 * expected_module).ceil() as i64;
    let mut counts = [0i64; 5];
    let mut i = start;
    while dark_at(i) == Some(centre_dark) && counts[2] <= max_centre {
        counts[2] += 1;
        i -= 1;
    }
    if counts[2] == 0 || counts[2] > max_centre {
        return None;
    }
    let up = counts[2] - 1;
    while dark_at(i) == Some(!centre_dark) && counts[1] <= max_light {
        counts[1] += 1;
        i -= 1;
    }
    if counts[1] == 0 || counts[1] > max_light {
        return None;
    }
    while dark_at(i) == Some(centre_dark) && counts[0] < max_outer {
        counts[0] += 1;
        i -= 1;
    }
    if counts[0] == 0 {
        return None;
    }
    i = start + 1;
    let mut down = 0;
    while dark_at(i) == Some(centre_dark) && down <= max_centre {
        down += 1;
        i += 1;
    }
    counts[2] += down;
    if counts[2] > max_centre {
        return None;
    }
    while dark_at(i) == Some(!centre_dark) && counts[3] <= max_light {
        counts[3] += 1;
        i += 1;
    }
    if counts[3] == 0 || counts[3] > max_light {
        return None;
    }
    while dark_at(i) == Some(centre_dark) && counts[4] < max_outer {
        counts[4] += 1;
        i += 1;
    }
    if counts[4] == 0 {
        return None;
    }
    let module = finder_module(counts.map(|c| c as f64), tolerance)?;
    if module < fit.0 * expected_module || module > fit.1 * expected_module {
        return None;
    }
    // The middle run covers [start - up, start + down + 1).
    Some((start as f64 + (down as f64 - up as f64 + 1.0) / 2.0, module))
}

/// One side of a cross-check, for a finder pattern whose other side may be
/// torn away: from `start` along one direction, the rest of the middle run,
/// then a light run and a dark run of about a module each. Returns the
/// centre of the middle run on the assumption that it is three modules
/// long and ends where it was seen to end.
fn half_cross_check(
    dark_at: impl Fn(i64) -> Option<bool>,
    start: i64,
    centre_dark: bool,
    module: f64,
    tolerance: f64,
) -> Option<f64> {
    let variance = tolerance * module;
    for direction in [1i64, -1] {
        let max_centre = (1.5 + tolerance) * module;
        let mut i = start;
        let mut centre = 0i64;
        while dark_at(i) == Some(centre_dark) && (centre as f64) <= max_centre + 1.0 {
            centre += 1;
            i += direction;
        }
        if centre == 0 || centre as f64 > max_centre {
            continue;
        }
        let mut light = 0i64;
        while dark_at(i) == Some(!centre_dark) && (light as f64) <= 2.0 * module {
            light += 1;
            i += direction;
        }
        if (light as f64 - module).abs() >= variance {
            continue;
        }
        let mut dark = 0i64;
        while dark_at(i) == Some(centre_dark) && (dark as f64) < 2.0 * module {
            dark += 1;
            i += direction;
        }
        if (dark as f64) < module * (1.0 - tolerance) {
            continue;
        }
        // Pixel i spans [i, i + 1); the middle run's far boundary is where
        // the light run began.
        let boundary = if direction > 0 { (start + centre) as f64 } else { (start - centre + 1) as f64 };
        return Some(boundary - direction as f64 * 1.5 * module);
    }
    None
}

/// Merge a hit into the cluster it belongs to, or start a new one.
fn merge(clusters: &mut Vec<Candidate>, hit: Candidate) {
    let size = (hit.module[0] + hit.module[1]) / 2.0;
    for c in clusters.iter_mut() {
        let own = (c.module[0] + c.module[1]) / 2.0;
        if c.inverted == hit.inverted
            && (c.x - hit.x).abs() <= own
            && (c.y - hit.y).abs() <= own
            && (own - size).abs() <= 0.5 * own.max(size)
        {
            let n = c.votes as f64;
            c.x = (c.x * n + hit.x) / (n + 1.0);
            c.y = (c.y * n + hit.y) / (n + 1.0);
            c.module = [(c.module[0] * n + hit.module[0]) / (n + 1.0), (c.module[1] * n + hit.module[1]) / (n + 1.0)];
            c.votes += 1;
            c.half &= hit.half;
            return;
        }
    }
    clusters.push(hit);
}

/// Scan every line of a `width` × `height` image along its first axis,
/// binarised by `dark`, for finder-shaped runs, and cross-check each
/// along the other axis and back along its own. With `transposed` the
/// image is the rectified image on its side, and each candidate is turned
/// back before it joins the clusters: the columns are scanned as well as
/// the rows, as ZBar's scanner runs both ways, so that a finder whose
/// rows a vertical scratch breaks is still seen down its columns.
#[allow(clippy::too_many_arguments)]
fn scan(
    width: usize,
    height: usize,
    dark: &dyn Fn(i64, i64) -> Option<bool>,
    min_module: f64,
    max_module: f64,
    clusters: &mut Vec<Candidate>,
    transposed: bool,
) {
    let push = |clusters: &mut Vec<Candidate>, hit: Candidate| {
        let hit = if transposed {
            Candidate { x: hit.y, y: hit.x, module: [hit.module[1], hit.module[0]], ..hit }
        } else {
            hit
        };
        merge(clusters, hit);
    };
    // (start, length, kind): 0 light, 1 dark, 2 no pixel.
    let mut runs: Vec<(usize, usize, u8)> = Vec::new();
    for y in 0..height {
        runs.clear();
        for x in 0..width {
            let kind = match dark(x as i64, y as i64) {
                Some(true) => 1,
                Some(false) => 0,
                None => 2,
            };
            match runs.last_mut() {
                Some(run) if run.2 == kind => run.1 += 1,
                _ => runs.push((x, 1, kind)),
            }
        }
        for window in runs.windows(5) {
            let kind = window[0].2;
            if kind > 1 || window.iter().enumerate().any(|(i, run)| run.2 != if i % 2 == 0 { kind } else { 1 - kind }) {
                continue;
            }
            let counts: [f64; 5] = std::array::from_fn(|i| window[i].1 as f64);
            let Some(row_module) = finder_module(counts, RATIO_TOLERANCE) else { continue };
            if !(min_module..=max_module).contains(&row_module) {
                continue;
            }
            let centre_dark = kind == 1;
            let cx = window[2].0 as f64 + window[2].1 as f64 / 2.0;
            let column = (cx.floor() as i64).clamp(0, width as i64 - 1);
            let Some((cy, v_module)) =
                cross_check(|i| dark(column, i), y as i64, centre_dark, row_module, RATIO_TOLERANCE, MODULE_FIT)
            else {
                // A finder with a corner torn through its centre column
                // still shows the rest of the column on the other side.
                // The row and diagonal through the centre cross the tear
                // as likely as not, so they are not asked; the template
                // fit has the last word.
                if let Some(cy) = half_cross_check(|i| dark(column, i), y as i64, centre_dark, row_module, RATIO_TOLERANCE) {
                    push(clusters, Candidate { x: cx, y: cy, module: [row_module, row_module], inverted: !centre_dark, votes: 1, half: true, quad: None });
                }
                continue;
            };
            let row = (cy.floor() as i64).clamp(0, height as i64 - 1);
            let Some((cx, h_module)) = cross_check(|i| dark(i, row), column, centre_dark, v_module, RATIO_TOLERANCE, MODULE_FIT)
            else {
                continue;
            };
            // As ZXing does, ask the diagonal through the centre for the
            // same proportions: a stripe or a logo that passes the row and
            // column rarely passes this too. A diagonal crosses the rings
            // of a square that sits straight in as many steps as a row
            // does, and the rings of one turned forty-five degrees, whose
            // own axis the diagonal then runs along, in half as many; the
            // size asked of it is the middle of that range, with the
            // allowance reaching both ends.
            let centre_column = (cx.floor() as i64).clamp(0, width as i64 - 1);
            let diagonal = |i: i64| dark(centre_column + (i - row), i);
            let other = |i: i64| dark(centre_column - (i - row), i);
            // Either diagonal: asking one alone would read a finder
            // differently as the photo is turned, and a scratch along
            // one leaves the other.
            if cross_check(diagonal, row, centre_dark, h_module, DIAGONAL_TOLERANCE, DIAGONAL_FIT).is_none()
                && cross_check(other, row, centre_dark, h_module, DIAGONAL_TOLERANCE, DIAGONAL_FIT).is_none()
            {
                continue;
            }
            let module = [h_module, v_module];
            if !module.iter().all(|m| (min_module..=max_module).contains(m)) {
                continue;
            }
            push(clusters, Candidate { x: cx, y: cy, module, inverted: !centre_dark, votes: 1, half: false, quad: None });
        }
    }
}

/// Finder candidates, strongest first. Rows and columns are each scanned
/// against one threshold for the whole outline, against a threshold that
/// follows the local lighting, and against the mean of a window a few
/// pixels about each pixel at each of [`WINDOW_RADII`], and the hits of
/// every scan are pooled. Modules larger than `max_module` rectified
/// pixels are not looked for.
///
/// The windows were a rung of the ladder alone, the neighbourhood
/// narrowed once every other look had come to nothing, and a scan that
/// found anything at all never reached it: over symbols blurred by half
/// to six tenths of a module, which read every one when their true
/// placement was handed to the reader, the scan made no whole candidate
/// at a true finder, and often enough a pattern in the blurred data
/// passed for one and stood a symbol of the wrong size on it, which
/// ended the search. With the windows scanned from the start, a search
/// read 97 of 144 photographs of symbols so blurred, or of a pixel or
/// so a module, where it had read 83.
///
/// Candidates come whole ones first and then those of which only half
/// cross-checked, and within each the two thresholds' before the
/// windows', each strongest first.
///
/// Candidates rank by the share of the lines across their centre run that
/// voted for them, not by the count of votes: a vote is one line, so the
/// count grows with the finder's size in pixels, and in a busy frame the
/// print and logos around a small symbol would outrank its real finder.
/// ZXing likewise asks only for a quorum of two rows and then judges the
/// candidates by their geometry.
fn scan_for_finders(rect: &Rectified, threshold: Option<f32>, max_module: f64) -> Vec<Candidate> {
    let mut clusters = scan_thresholds(rect, threshold, max_module);
    // The windows' hits are gathered apart, and stand only where the
    // two thresholds made no whole candidate: pooled into one gathering
    // they shifted the centre and the module of a candidate those two
    // had already made, by the little the windows read it differently,
    // and a symbol on curled paper that had read outright was then
    // placed a few pixels off at its far corner and read only with its
    // padding pinned.
    let mut windowed = scan_windows(rect, max_module);
    windowed.retain(|c| {
        let near = |o: &Candidate| o.inverted == c.inverted && distance([o.x, o.y], [c.x, c.y]) < c.module_mean().max(o.module_mean());
        !clusters.iter().any(|o| !o.half && near(o))
    });
    // The centre run is three modules across, so that many lines could
    // vote, along each axis.
    let rate = |c: &Candidate| c.votes as f64 / (3.0 * (c.module[0] + c.module[1])).max(1.0);
    let strongest_first = |a: &Candidate, b: &Candidate| {
        a.half.cmp(&b.half).then(rate(b).total_cmp(&rate(a))).then((b.module[0] + b.module[1]).total_cmp(&(a.module[0] + a.module[1])))
    };
    clusters.sort_by(strongest_first);
    windowed.sort_by(strongest_first);
    // Whole candidates before half ones, and among each the two
    // thresholds' before the windows': what the scan made before the
    // windows were added keeps its place in the count of candidates
    // refined.
    let (whole, half): (Vec<Candidate>, Vec<Candidate>) = clusters.into_iter().partition(|c| !c.half);
    let (windowed_whole, windowed_half): (Vec<Candidate>, Vec<Candidate>) = windowed.into_iter().partition(|c| !c.half);
    whole.into_iter().chain(windowed_whole).chain(half).chain(windowed_half).collect()
}

/// Whether enough lines agreed on a candidate for it to be worth
/// fitting a template to.
fn quorate(c: &Candidate) -> bool {
    c.votes >= if c.half { MIN_VOTES_HALF } else { MIN_VOTES }
}

/// The candidates of the scans against one threshold for the whole
/// outline and against the threshold that follows the local lighting,
/// rows and columns each, pooled, in no order.
fn scan_thresholds(rect: &Rectified, threshold: Option<f32>, max_module: f64) -> Vec<Candidate> {
    let global = threshold.unwrap_or_else(|| otsu(rect.data.iter().copied().filter(|v| v.is_finite())));
    let (w, h) = (rect.frame.width, rect.frame.height);
    let mut clusters = Vec::new();
    let global_dark = |x: i64, y: i64| rect.at(x, y).map(|v| v < global);
    scan(w, h, &global_dark, MIN_MODULE, max_module, &mut clusters, false);
    scan(h, w, &|x, y| global_dark(y, x), MIN_MODULE, max_module, &mut clusters, true);
    let local = LocalMean::new(rect);
    let local_dark = |x: i64, y: i64| local.dark(rect, x, y);
    scan(w, h, &local_dark, MIN_MODULE, max_module, &mut clusters, false);
    scan(h, w, &|x, y| local_dark(y, x), MIN_MODULE, max_module, &mut clusters, true);
    clusters.retain(quorate);
    clusters
}

/// The candidates of the scans against the mean of a window a few
/// pixels about each pixel, at each width narrower than the
/// neighbourhood the local threshold has, pooled, in no order: a finder
/// blurred until its rings are ripples reads as rings only against what
/// lies right beside them. See [`super::binarize`].
fn scan_windows(rect: &Rectified, max_module: f64) -> Vec<Candidate> {
    let (w, h) = (rect.frame.width, rect.frame.height);
    let mut clusters = Vec::new();
    let bias = WINDOW_BIAS * rect.spread;
    for radius in WINDOW_RADII.into_iter().filter(|&radius| radius as f64 <= rect.local_radius) {
        let mask = window_dark(&rect.data, w, h, radius, bias);
        let window = |x: i64, y: i64| {
            if x < 0 || y < 0 || x as usize >= w || y as usize >= h {
                return None;
            }
            match mask[y as usize * w + x as usize] {
                2 => None,
                kind => Some(kind == 1),
            }
        };
        scan(w, h, &window, MIN_MODULE, max_module, &mut clusters, false);
        scan(h, w, &|x, y| window(y, x), MIN_MODULE, max_module, &mut clusters, true);
    }
    clusters.retain(quorate);
    clusters
}

/// A finder pattern verified against the photo.
#[derive(Clone, Copy, Debug)]
struct Capstone {
    /// Corners of the finder pattern, seven modules square, in photo
    /// pixels: top-left, top-right, bottom-right, bottom-left as the
    /// outline has them.
    corners: [[f64; 2]; 4],
    /// The finder's centre in photo pixels.
    center: [f64; 2],
    /// The centre across the outline, 0 to 1 along each side.
    unit: [f64; 2],
    /// Module size along each axis, in rectified pixels.
    module: [f64; 2],
    contrast: f64,
    inverted: bool,
    /// One quarter of the pattern did not read right.
    partial: bool,
    /// Luminance between the finder's own dark and light, for reading
    /// the modules beside it.
    threshold: f32,
    /// Accepted from the lines through its centre alone: its corners are
    /// put square to the frame, not measured, and only its centre goes
    /// into the fit.
    line_only: bool,
    /// The corners were fitted to the pattern and carry its turn. False
    /// where they were put square to the frame about a centre read along
    /// lines or rays, and say nothing of the turn.
    measured: bool,
}

impl Capstone {
    fn module_mean(&self) -> f64 {
        (self.module[0] + self.module[1]) / 2.0
    }

    /// The same finder pattern as seen in another frame: its corners in
    /// that frame's order, where its centre lies across that outline, and
    /// how big its modules are in that frame's pixels. The fit in the
    /// photo is unchanged.
    fn reframed(&self, frame: &Frame) -> Self {
        let corners = order_in_frame(self.corners, frame);
        let r = corners.map(|c| frame.to_rectified(c));
        let module = [
            (distance(r[0], r[1]) + distance(r[3], r[2])) / 14.0,
            (distance(r[0], r[3]) + distance(r[1], r[2])) / 14.0,
        ];
        Self { corners, unit: frame.unit(self.center), module, ..*self }
    }
}

/// Four corners of a square patch in the order a frame has them: the one
/// nearest the frame's top left first, then clockwise about their centre.
fn order_in_frame(corners: [[f64; 2]; 4], frame: &Frame) -> [[f64; 2]; 4] {
    let r = corners.map(|c| frame.to_rectified(c));
    let centre = quad_centre(r);
    let angle = |i: usize| (r[i][1] - centre[1]).atan2(r[i][0] - centre[0]);
    let mut order = [0, 1, 2, 3];
    order.sort_by(|&a, &b| angle(a).total_cmp(&angle(b)));
    let first = (0..4).min_by(|&a, &b| (r[order[a]][0] + r[order[a]][1]).total_cmp(&(r[order[b]][0] + r[order[b]][1]))).expect("four corners");
    order.rotate_left(first);
    order.map(|i| corners[i])
}

/// Whether a template placed at `corners` reads as its pattern: all of it,
/// or three quarters of it with the fourth damaged. `Some(partial)` when
/// it does.
fn accept(scorer: &Scorer, corners: [[f64; 2]; 4], inverted: bool) -> Option<bool> {
    judge(scorer.agreement_quarters(corners, inverted)?, None)
}

/// The verdict on per-quarter agreement: `Some(false)` when the whole
/// pattern reads right, `Some(true)` when three quarters do and the
/// fourth is damaged, and `None` otherwise. With `excluded`, that quarter
/// is known to be damaged and only the other three are asked.
fn judge(quarters: [(usize, usize); 4], excluded: Option<usize>) -> Option<bool> {
    let (matching, known) = quarters.iter().fold((0, 0), |a, q| (a.0 + q.0, a.1 + q.1));
    if known == 0 {
        return None;
    }
    if excluded.is_none() && matching as f64 >= MIN_AGREEMENT * known as f64 {
        return Some(false);
    }
    if (matching as f64) < MIN_PARTIAL_AGREEMENT * known as f64 || quarters.iter().any(|q| q.1 == 0) {
        return None;
    }
    // One quarter clearly torn and the other three, taken together,
    // reading right: a tear spills a little past its own quarter, so the
    // three are not asked one by one. A quarter that merely reads poorly
    // is noise, and no excuse.
    let bad = excluded.or_else(|| worst_quarter(quarters))?;
    let (m, k) = (0..4).filter(|&i| i != bad).fold((0, 0), |a, i| (a.0 + quarters[i].0, a.1 + quarters[i].1));
    (k > 0 && m as f64 >= MIN_QUARTER_AGREEMENT * k as f64).then_some(true)
}

/// Share of a quarter's modules reading right below which the quarter is
/// torn rather than merely noisy.
const TORN_QUARTER: f64 = 0.6;

/// The quarter of a pattern that reads worst, when it alone reads badly:
/// the mark of a tear, as against noise everywhere.
fn worst_quarter(quarters: [(usize, usize); 4]) -> Option<usize> {
    let shares: Vec<f64> = quarters.iter().map(|q| if q.1 > 0 { q.0 as f64 / q.1 as f64 } else { 1.0 }).collect();
    let bad: Vec<usize> = (0..4).filter(|&i| shares[i] < TORN_QUARTER).collect();
    (bad.len() == 1).then(|| bad[0])
}

/// Whether the ring of modules around a finder placed at the nine-module
/// `corners` reads as paper: light, or dark for a light-on-dark symbol.
fn ring_is_paper(image: &Luma, corners: [[f64; 2]; 4], inverted: bool, threshold: Option<f32>) -> bool {
    let Some(h) = Homography::rect_to_quad(9.0, 9.0, corners) else { return false };
    let ring: Vec<Pos> = (0..9)
        .flat_map(|y| (0..9).map(move |x| (x, y)))
        .filter(|&(x, y)| x == 0 || y == 0 || x == 8 || y == 8)
        .map(|(x, y)| Pos::new(x, y))
        .collect();
    let inner: Vec<Pos> = (1..8).flat_map(|y| (1..8).map(move |x| Pos::new(x, y))).collect();
    let ring_luma = sample_positions(image, &h, &ring, INNER_FRACTION, POINTS);
    let split = threshold.unwrap_or_else(|| {
        let mut all: Vec<f32> = sample_positions(image, &h, &inner, INNER_FRACTION, POINTS).into_iter().flatten().collect();
        all.extend(ring_luma.iter().flatten());
        otsu(all.into_iter())
    });
    let known = ring_luma.iter().flatten().count();
    let paper = ring_luma.iter().flatten().filter(|&&v| (v >= split) != inverted).count();
    known * 2 >= ring.len() && paper as f64 >= RING_PAPER * known as f64
}

/// The dark and light levels of a finder pattern placed at the seven-module
/// `corners`: the threshold halfway between its own dark centre and its
/// own light ring, which a histogram split cannot be trusted to find when
/// over-exposure has thinned the dark modules, and half the contrast
/// between the two.
fn finder_levels(image: &Luma, corners: [[f64; 2]; 4]) -> Option<(f32, f32)> {
    let h = Homography::rect_to_quad(7.0, 7.0, corners)?;
    let ring = |x: usize, y: usize| x.abs_diff(3).max(y.abs_diff(3));
    let mean = |pick: &dyn Fn(usize) -> bool| {
        let positions: Vec<Pos> =
            (0..7).flat_map(|y| (0..7).map(move |x| (x, y))).filter(|&(x, y)| pick(ring(x, y))).map(|(x, y)| Pos::new(x, y)).collect();
        let values: Vec<f32> = sample_positions(image, &h, &positions, INNER_FRACTION, POINTS).into_iter().flatten().collect();
        (values.len() * 2 > positions.len()).then(|| values.iter().sum::<f32>() / values.len() as f32)
    };
    let centre = mean(&|r| r <= 1)?;
    let light = mean(&|r| r == 2)?;
    Some(((centre + light) / 2.0, ((light - centre) / 2.0).abs()))
}

/// The threshold between dark and light for a finder pattern placed at the
/// seven-module `corners`.
fn finder_threshold(image: &Luma, corners: [[f64; 2]; 4]) -> Option<f32> {
    finder_levels(image, corners).map(|(threshold, _)| threshold)
}

/// Fit a finder pattern to a candidate. `None` when it does not read as
/// one. The fit is of the seven-module finder alone: the light ring around
/// it is the quiet zone on two sides, which a symbol printed hard against
/// other print does not have. A finder with one quarter torn away is fitted
/// again on its other three.
fn refine_capstone(image: &Luma, rect: &Rectified, candidate: &Candidate, threshold: Option<f32>) -> Option<Capstone> {
    let (hx, hy) = (3.5 * candidate.module[0], 3.5 * candidate.module[1]);
    // The template starts square to the frame, from where the descent
    // reaches a finder turned up to about thirty degrees. One turned
    // further is as near one fit as another, and the descent settles on
    // neither; so a start turned forty-five degrees is tried next. The
    // scan lines cross such a finder at an angle and its runs read a
    // square root of two too long, which puts its corners that far out
    // along the axes.
    let offsets = [[[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], [[0.0, -hy], [hx, 0.0], [0.0, hy], [-hx, 0.0]]];
    let mut starts: Vec<([[f64; 2]; 4], bool)> = offsets
        .into_iter()
        .enumerate()
        // A finder that is turned, and bloomed or torn besides, is not
        // looked for: the turned start takes the plain fit alone.
        .map(|(i, offsets)| (offsets.map(|[dx, dy]| rect.frame.to_image.apply(candidate.x + dx, candidate.y + dy)), i == 0))
        .collect();
    // A region's own quadrilateral already has the turn and the
    // perspective of a finder seen steeply, as BoofCV fits its position
    // patterns from the polygon it found: it is tried first.
    if let Some(quad) = candidate.quad {
        starts.insert(0, (quad, true));
    }
    starts
        .into_iter()
        .find_map(|(start, thorough)| refine_capstone_from(image, rect, candidate, threshold, start, thorough))
        .or_else(|| concentric_capstone(image, rect, candidate, threshold))
        .or_else(|| line_capstone(image, rect, candidate, threshold))
}

/// What a capstone read along lines alone is worth against a fitted one,
/// as a share of its contrast: the lines say where the finder is and how
/// big, not its outline, and a template that fits is the surer reading of
/// the same place.
const LINE_CONTRAST_SHARE: f64 = 0.75;
/// The four lines through a finder's centre may differ in the module they
/// measure by at most this factor: zxing-cpp's bound on the spread of a
/// concentric pattern.
const LINE_SPREAD_RATIO: f64 = 5.0;
/// Least contrast between a finder's dark centre and its light ring for a
/// capstone read along lines alone, as a share of the spread of the
/// rectified outline from its darkest to its lightest hundredth: a tenth
/// of the scale on a well exposed photo, and in proportion on a dim one,
/// which a fixed floor would read as having no finder at all.
const LINE_CONTRAST_FLOOR: f64 = 0.12;

/// A finder accepted from the lines through its centre alone, as
/// zxing-cpp's `LocateConcentricPattern` accepts one: the column, the row
/// and both diagonals through the centre must each read as 1:1:3:1:1,
/// with the runs tested pair by pair as the scan tests them, and the
/// modules they measure must agree within [`LINE_SPREAD_RATIO`]. Nothing
/// is asked of the ring between the lines, so a finder drawn as a large
/// dot ringed by small ones, or as rings with gaps in them, passes here
/// where every template above wants a solid ring. The capstone's corners
/// are put square to the frame about the centre and are not measured:
/// only the centre goes into the fit. For a candidate that no template
/// fitted.
fn line_capstone(image: &Luma, rect: &Rectified, candidate: &Candidate, threshold: Option<f32>) -> Option<Capstone> {
    let module = candidate.module_mean();
    let reach = (4.0 * module).ceil() as i64;
    let (cx, cy) = (candidate.x.floor() as i64, candidate.y.floor() as i64);
    let split = threshold.unwrap_or_else(|| {
        otsu((cy - reach..=cy + reach).flat_map(|y| (cx - reach..=cx + reach).map(move |x| (x, y))).filter_map(|(x, y)| rect.at(x, y)))
    });
    let dark = |x: i64, y: i64| rect.at(x, y).map(|v| v < split);
    let centre_dark = !candidate.inverted;
    // The column through the candidate, then the row through the centre
    // it gives, as the scan cross-checks a hit; each settles the centre
    // along its own axis.
    let (y, v_module) = cross_check(|i| dark(cx, i), cy, centre_dark, module, RATIO_TOLERANCE, MODULE_FIT)?;
    let row = (y.floor() as i64).clamp(0, rect.frame.height as i64 - 1);
    let (x, h_module) = cross_check(|i| dark(i, row), cx, centre_dark, v_module, RATIO_TOLERANCE, MODULE_FIT)?;
    let column = (x.floor() as i64).clamp(0, rect.frame.width as i64 - 1);
    let axis_module = (h_module + v_module) / 2.0;
    // Both diagonals through that centre, as zxing-cpp asks of every
    // direction: a stripe or a logo that passes the row and the column
    // rarely passes both.
    let (_, d1) = cross_check(|i| dark(column + (i - row), i), row, centre_dark, axis_module, DIAGONAL_TOLERANCE, DIAGONAL_FIT)?;
    let (_, d2) = cross_check(|i| dark(column - (i - row), i), row, centre_dark, axis_module, DIAGONAL_TOLERANCE, DIAGONAL_FIT)?;
    let (least, most) = [h_module, v_module, d1, d2].iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), &m| (lo.min(m), hi.max(m)));
    if !(least >= MIN_MODULE) || most > LINE_SPREAD_RATIO * least {
        return None;
    }
    let module = [h_module, v_module];
    let centre = [x, y];
    let half = [3.5 * h_module, 3.5 * v_module];
    let square = [[-half[0], -half[1]], [half[0], -half[1]], [half[0], half[1]], [-half[0], half[1]]]
        .map(|[dx, dy]| rect.frame.to_image.apply(centre[0] + dx, centre[1] + dy));
    let corners = order_in_frame(square, &rect.frame);
    let (own, half_contrast) = finder_levels(image, corners)?;
    let contrast = 2.0 * f64::from(half_contrast);
    if contrast < LINE_CONTRAST_FLOOR * f64::from(rect.spread) {
        return None;
    }
    let center = rect.frame.to_image.apply(centre[0], centre[1]);
    Some(Capstone {
        corners,
        center,
        unit: rect.unit(center),
        module,
        contrast: LINE_CONTRAST_SHARE * contrast,
        inverted: candidate.inverted,
        partial: false,
        threshold: threshold.unwrap_or(own),
        line_only: true,
        measured: false,
    })
}

/// Rays along which a finder's rings are read for their centre.
const RING_RAYS: usize = 16;
/// Rays that must read the rings for the finder to be taken as round.
const RING_RAYS_MIN: usize = 13;
/// Where the rings end along a ray, in modules from the centre: the inner
/// disc or square, the light ring, the dark ring.
const RING_EDGES: [f64; 3] = [1.5, 2.5, 3.5];
/// How far an edge may be from its place along a ray, in modules.
const RING_TOLERANCE: f64 = 0.5;

/// A finder read along rays from its centre rather than fitted as a
/// square: the rings of a round finder, as some decorative symbols draw
/// them, cross every ray through the centre in the 1:1:3:1:1 proportion,
/// which zxing-cpp's concentric-ring finder reads the same way, and the
/// edges found along the rays place the centre and measure the module.
/// The square the capstone reports is put square to the frame, as a
/// symbol with round finders is drawn, not photographed. For a candidate
/// that no square template fitted.
fn concentric_capstone(image: &Luma, rect: &Rectified, candidate: &Candidate, threshold: Option<f32>) -> Option<Capstone> {
    let mut centre = [candidate.x, candidate.y];
    let mut module = candidate.module_mean();
    let at = |x: f64, y: f64| rect.at(x.floor() as i64, y.floor() as i64).filter(|v| v.is_finite());
    let mut contrast = 0.0;
    for _ in 0..3 {
        let reach = (4.0 * module).ceil() as i64;
        let (cx, cy) = (centre[0].floor() as i64, centre[1].floor() as i64);
        let split = threshold.unwrap_or_else(|| {
            otsu((cy - reach..=cy + reach).flat_map(|y| (cx - reach..=cx + reach).map(move |x| (x, y))).filter_map(|(x, y)| rect.at(x, y)).filter(|v| v.is_finite()))
        });
        let step = (0.25 * module).max(0.5);
        let tolerance = RING_TOLERANCE * module;
        // Along each ray, where the colour changes the first three times,
        // and the edge contrast there.
        let mut outer: Vec<[f64; 2]> = Vec::new();
        let mut modules = 0.0;
        let mut edge_contrast = 0.0;
        for k in 0..RING_RAYS {
            let theta = k as f64 * std::f64::consts::TAU / RING_RAYS as f64;
            let (dx, dy) = (theta.cos(), theta.sin());
            let mut changes: Vec<f64> = Vec::new();
            let mut last: Option<bool> = None;
            let mut r = 0.0;
            while r <= 4.5 * module && changes.len() < 3 {
                let Some(v) = at(centre[0] + r * dx, centre[1] + r * dy) else { break };
                let dark = (v < split) != candidate.inverted;
                match last {
                    None if !dark => break,
                    Some(prev) if prev != dark => changes.push(r - step / 2.0),
                    _ => {}
                }
                last = Some(dark);
                r += step;
            }
            if changes.len() == 3 && (0..3).all(|i| (changes[i] - RING_EDGES[i] * module).abs() <= tolerance) {
                outer.push([centre[0] + changes[2] * dx, centre[1] + changes[2] * dy]);
                modules += changes[2] / RING_EDGES[2];
                let sign = if candidate.inverted { -1.0 } else { 1.0 };
                for (i, &edge) in changes.iter().enumerate() {
                    let (inside, outside) = (edge - 0.35 * module, edge + 0.35 * module);
                    if let (Some(a), Some(b)) = (at(centre[0] + inside * dx, centre[1] + inside * dy), at(centre[0] + outside * dx, centre[1] + outside * dy)) {
                        // Dark inside the first and third edges, light inside the second.
                        let toward_light = if i == 1 { -1.0 } else { 1.0 };
                        edge_contrast += sign * toward_light * f64::from(b - a);
                    }
                }
            }
        }
        if outer.len() < RING_RAYS_MIN {
            return None;
        }
        let n = outer.len() as f64;
        centre = [outer.iter().map(|p| p[0]).sum::<f64>() / n, outer.iter().map(|p| p[1]).sum::<f64>() / n];
        module = modules / n;
        contrast = edge_contrast / (3.0 * n);
    }
    if !(contrast > 0.0 && module >= MIN_MODULE) {
        return None;
    }
    let half = 3.5 * module;
    let square = [[-half, -half], [half, -half], [half, half], [-half, half]].map(|[dx, dy]| rect.frame.to_image.apply(centre[0] + dx, centre[1] + dy));
    let corners = order_in_frame(square, &rect.frame);
    let center = rect.frame.to_image.apply(centre[0], centre[1]);
    let threshold = threshold.or_else(|| finder_threshold(image, corners)).unwrap_or(0.5);
    Some(Capstone {
        corners,
        center,
        unit: rect.unit(center),
        module: [module, module],
        contrast,
        inverted: candidate.inverted,
        partial: false,
        threshold,
        line_only: false,
        measured: false,
    })
}

/// A finder pattern whose rings are round rather than square, as some
/// decorative symbols draw them: seven modules square, with the corner
/// modules of the outer ring and of the inner square left unfixed, since
/// a circle has paper where a square has ink there. zxing-cpp's
/// concentric-ring finder asks nothing of the corners either.
fn round_finder() -> Template {
    let mut template = Template::finder_core();
    for (x, y) in [(0, 0), (6, 0), (0, 6), (6, 6), (2, 2), (4, 2), (2, 4), (4, 4)] {
        template.expected[y * 7 + x] = None;
    }
    template
}

/// Share of a template's modules that must read right after the first
/// descent for the dearer fits to be tried: the template expecting bloom,
/// and the one with a torn quarter left out. A finder that has bloomed or
/// lost a corner still reads three quarters right at the first fit; most
/// candidates are not finders at all, and read half, and the retries
/// would cost more than the first fit for nothing.
const RETRY_MIN_SHARE: f64 = 0.7;

/// [`refine_capstone`] from one placement of the template; `thorough` tries
/// the dearer fits, for bloom and a torn quarter, when the plain one fails.
fn refine_capstone_from(
    image: &Luma,
    rect: &Rectified,
    candidate: &Candidate,
    threshold: Option<f32>,
    start: [[f64; 2]; 4],
    thorough: bool,
) -> Option<Capstone> {
    let template = Template::finder_core();
    let corners = start;
    let scorer = Scorer::new(image, &template, corners, threshold).with_along(8);
    let sign = if candidate.inverted { -1.0 } else { 1.0 };
    let mut evaluations = 0;
    // The scan places the centre within a fraction of a module and the size
    // within a few tenths, so a corner may start most of a module off.
    let (mut corners, mut contrast) = scorer.descend(corners, sign, &MOVES, scorer.module_px(), &mut evaluations)?;
    if contrast <= 0.0 {
        return None;
    }
    let read = |corners: [[f64; 2]; 4]| {
        let split = threshold.or_else(|| finder_threshold(image, corners));
        scorer.agreement_quarters_at(corners, candidate.inverted, split)
    };
    let mut quarters = read(corners)?;
    let mut partial = judge(quarters, None);
    let (matching, known) = quarters.iter().fold((0, 0), |a, q| (a.0 + q.0, a.1 + q.1));
    let worth_retrying = thorough && matching as f64 >= RETRY_MIN_SHARE * known.max(1) as f64;
    if partial.is_none() && !worth_retrying {
        return None;
    }
    if partial.is_none() {
        // An over-exposed photo has every boundary moved into the dark,
        // and an under-exposed one into the light; a template that expects
        // the move fits where the plain one could not.
        for bloom in BLOOMS {
            let bloomed = Scorer::new(image, &template, start, threshold).with_along(8).with_bloom(bloom);
            if let Some((again, score)) = bloomed.descend(start, sign, &MOVES, bloomed.module_px(), &mut evaluations)
                && score > 0.0
                && let Some(q) = read(again)
                && let Some(verdict) = judge(q, None)
            {
                corners = again;
                contrast = score;
                quarters = q;
                partial = Some(verdict);
                break;
            }
        }
    }
    if partial.is_none() && thorough {
        // A round finder reads as a square one along its rings' edges but
        // not at the corners, every quarter alike, which no tear explains:
        // the template that asks nothing of the corners fits it.
        let round_template = round_finder();
        let round = Scorer::new(image, &round_template, start, threshold).with_along(8);
        if let Some((again, score)) = round.descend(start, sign, &MOVES, round.module_px(), &mut evaluations)
            && score > 0.0
            && let Some(q) = round.agreement_quarters_at(again, candidate.inverted, threshold.or_else(|| finder_threshold(image, again)))
            && let Some(verdict) = judge(q, None)
        {
            corners = again;
            contrast = score;
            quarters = q;
            partial = Some(verdict);
        }
    }
    if partial.is_none()
        && let Some(bad) = worst_quarter(quarters)
    {
        // Fit again on the three quarters that read, from where the first
        // fit left off: the tear dragged it, and without the tear's
        // boundaries in the score it settles on the pattern.
        let torn = Scorer::new(image, &template.without_quarter(bad), corners, threshold).with_along(8);
        if let Some((again, score)) = torn.descend(corners, sign, &MOVES, scorer.module_px() * 0.5, &mut evaluations)
            && score > 0.0
        {
            corners = again;
            contrast = score;
            partial = judge(read(corners)?, Some(bad));
        }
    }
    let partial = partial?;
    // Where the ring of light around the finder is really there, the
    // separator and quiet zone, it places the finder more exactly than
    // the finder alone: more boundaries, further apart.
    let patch = Homography::rect_to_quad(7.0, 7.0, corners)?;
    let ringed = [[-1.0, -1.0], [8.0, -1.0], [8.0, 8.0], [-1.0, 8.0]].map(|[x, y]| patch.apply(x, y));
    if ring_is_paper(image, ringed, candidate.inverted, threshold) {
        let ring = Scorer::new(image, &Template::finder_with_ring(), ringed, threshold).with_along(8);
        if let Some((again, score)) = ring.descend(ringed, sign, &MOVES, ring.module_px() * 0.25, &mut evaluations)
            && score > 0.0
            && let Some(patch9) = Homography::rect_to_quad(9.0, 9.0, again)
        {
            let inner = [[1.0, 1.0], [8.0, 1.0], [8.0, 8.0], [1.0, 8.0]].map(|[x, y]| patch9.apply(x, y));
            if read(inner).and_then(|q| judge(q, None)).is_some() {
                corners = inner;
                contrast = score;
            }
        }
    }
    let patch = Homography::rect_to_quad(7.0, 7.0, corners)?;
    let center = patch.apply(3.5, 3.5);
    let corners = order_in_frame(corners, &rect.frame);
    let r = corners.map(|c| rect.to_rectified(c));
    let module = [
        (distance(r[0], r[1]) + distance(r[3], r[2])) / 14.0,
        (distance(r[0], r[3]) + distance(r[1], r[2])) / 14.0,
    ];
    let threshold = threshold.or_else(|| finder_threshold(image, corners)).unwrap_or(0.5);
    Some(Capstone { corners, center, unit: rect.unit(center), module, contrast, inverted: candidate.inverted, partial, threshold, line_only: false, measured: true })
}

/// The symbol's extent in modules along the frame's two axes, and
/// whether the estimate is tight: its (width, height) when it lies in
/// the frame as its version has it, and the other way round when it
/// stands on end there, which the versions it is matched against allow
/// for (see [`size_error`]). OpenCV's `versionDefinition` takes the size two ways, from
/// the module size and the finder spacing and from a count of the timing
/// pattern's transitions, and checks one against the other; so here.
/// The spacing of the capstones that stand as finders, divided by their
/// module size, is ZXing's estimate, and a tenth of error in the module
/// is three modules of size on a version 6 symbol: the count along the
/// timing pattern between two finders does not depend on the module size
/// at all, and when it reads regularly and gives a size a version has it
/// is believed instead. A lone finder walks its symbol's timing patterns
/// outward: on a Micro QR symbol they run along the edges to the far
/// corners and count its modules, and on any symbol the walk reaches the
/// quiet zone, which gives the extent in modules. Failing all that, the
/// outline and the module size of the one capstone give a loose estimate.
/// Capstones that stand as no finder, print beside the symbol or a
/// pattern in its data, are left out, as ZXing estimates the dimension
/// from the three patterns it has chosen alone.
fn size_hint(image: &Luma, frame: &Frame, capstones: &[Capstone], family: Family) -> ([f64; 2], bool) {
    let roles = finder_roles(capstones, family);
    let matched: Vec<&Capstone> = roles
        .as_ref()
        .map(|(_, _, assignment)| assignment.iter().flatten().map(|&c| &capstones[c]).collect())
        .unwrap_or_default();
    let pool: Vec<&Capstone> = if matched.is_empty() { capstones.iter().collect() } else { matched };
    // Whole capstones place their centres more surely than partial ones,
    // and are less often false: the spacing is taken from them alone when
    // two or more exist.
    let whole: Vec<&Capstone> = pool.iter().copied().filter(|c| !c.partial).collect();
    let pool: Vec<&Capstone> = if whole.len() >= 2 { whole } else { pool };
    let p: Vec<[f64; 2]> = pool.iter().map(|c| frame.to_rectified(c.center)).collect();
    // Three capstones: the longest side between them is the diagonal and
    // the top-left finder is the vertex opposite it, as ZXing has it.
    // Two: the pair runs along a side or across the diagonal, which their
    // directions in the frame can only guess at.
    let longest = (pool.len() == 3).then(|| {
        (0..3).max_by(|&i, &j| distance(p[(i + 1) % 3], p[(i + 2) % 3]).total_cmp(&distance(p[(j + 1) % 3], p[(j + 2) % 3])))
    }).flatten();
    let mut estimates = Vec::new();
    for i in 0..pool.len() {
        for j in i + 1..pool.len() {
            let (pa, pb) = (p[i], p[j]);
            let (du, dv) = ((pa[0] - pb[0]).abs(), (pa[1] - pb[1]).abs());
            let module = (pool[i].module_mean() + pool[j].module_mean()) / 2.0;
            // Finders along one side are n - 7 modules apart; across the
            // diagonal, √2 times that.
            let diagonal = match longest {
                Some(tl) => i != tl && j != tl,
                None => du.min(dv) * 2.0 > du.max(dv),
            };
            let apart = distance(pa, pb) / module / if diagonal { SQRT_2 } else { 1.0 };
            estimates.push(7.0 + apart);
        }
    }
    let spacing = (!estimates.is_empty()).then(|| estimates.iter().sum::<f64>() / estimates.len() as f64);

    // The timing pattern between the top-left finder and each of the
    // other two. With two capstones only, the pattern may lie on either
    // side of the pair: a regular one on either says the pair runs along
    // a side, and which side the symbol is on.
    let mut counts: Vec<TimingCount> = Vec::new();
    if matches!(family, Family::Qr | Family::Model1) {
        match longest {
            Some(tl) => {
                let others: Vec<usize> = (0..3).filter(|&v| v != tl).collect();
                for (o, third) in [(others[0], others[1]), (others[1], others[0])] {
                    if let Some(count) = timing_between(image, frame, pool[tl], pool[o], p[third]) {
                        counts.push(count);
                    }
                }
            }
            None if pool.len() == 2 => {
                let mid = [(p[0][0] + p[1][0]) / 2.0, (p[0][1] + p[1][1]) / 2.0];
                let across = [p[0][1] - p[1][1], p[1][0] - p[0][0]];
                for side in [1.0, -1.0] {
                    let inside = [mid[0] + side * across[0], mid[1] + side * across[1]];
                    if let Some(count) = timing_between(image, frame, pool[0], pool[1], inside) {
                        counts.push(count);
                        break;
                    }
                }
            }
            None => {}
        }
    }
    let agrees = |n: f64| spacing.is_none_or(|s| (n - s).abs() <= NAMED_SIZE_SLACK);
    let exact: Vec<f64> = counts.iter().filter(|c| c.exact && agrees(c.size)).map(|c| c.size).collect();
    if let Some(&n) = exact.first() {
        // Two exact counts that differ: the one nearer the spacing.
        let n = exact.iter().copied().min_by(|a, b| spacing.map_or(0.0, |s| (a - s).abs()).total_cmp(&spacing.map_or(0.0, |s| (b - s).abs()))).unwrap_or(n);
        return ([n, n], true);
    }
    let rough: Vec<f64> = counts.iter().filter(|c| agrees(c.size)).map(|c| c.size).collect();
    if let Some(s) = spacing {
        let n = match rough.first() {
            Some(&r) => (r + s) / 2.0,
            None => s,
        };
        return ([n, n], true);
    }
    if let Some(&r) = rough.first() {
        return ([r, r], false);
    }
    if let [one] = pool.as_slice()
        && let Some((size, exact)) = lone_finder_size(image, one, family)
    {
        return (size, exact);
    }
    let mut sizes: Vec<[f64; 2]> = pool.iter().map(|c| c.module).collect();
    sizes.sort_by(|a, b| (a[0] + a[1]).total_cmp(&(b[0] + b[1])));
    let module = sizes[sizes.len() / 2];
    ([frame.size[0] / module[0], frame.size[1] / module[1]], false)
}

/// Runs of dark and light along the segment from `a` to `b` in photo
/// pixels, read every `step` pixels against `threshold`: each run's
/// colour, dark as the symbol prints it (on a light-on-dark photo the
/// light runs count as dark), and its length in pixels. The runs stop
/// where the photo does.
fn runs_along(image: &Luma, a: [f64; 2], b: [f64; 2], step: f64, threshold: f32, inverted: bool) -> Vec<(bool, f64)> {
    let length = distance(a, b);
    let n = (length / step.max(1e-3)).ceil().max(1.0) as usize;
    let each = length / n as f64;
    let mut runs: Vec<(bool, f64)> = Vec::new();
    for i in 0..=n {
        let t = i as f64 / n as f64;
        let Some(v) = image.sample(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t) else { break };
        let dark = (v < threshold) != inverted;
        match runs.last_mut() {
            Some((d, len)) if *d == dark => *len += each,
            _ => runs.push((dark, each)),
        }
    }
    runs
}

/// The module length a timing pattern's runs agree on, in pixels: their
/// median, when enough of them lie within a module's tolerance of it.
fn regular_pitch(runs: &[f64]) -> Option<f64> {
    if runs.len() < 2 {
        return None;
    }
    let mut sorted = runs.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    let agree = runs.iter().filter(|&&l| (TIMING_RUN_MIN * median..=TIMING_RUN_MAX * median).contains(&l)).count();
    (agree as f64 >= TIMING_REGULAR * runs.len() as f64).then_some(median)
}

/// What a timing pattern says the symbol's size is along it.
#[derive(Clone, Copy, Debug)]
struct TimingCount {
    size: f64,
    /// Every run read as a module, and the size is one a version has.
    exact: bool,
}

/// A capstone's own module as two vectors in photo pixels, from the
/// quadrilateral fitted to it: one module along the axis of the finder
/// that points most nearly at `toward`, and one along its other axis, to
/// the side `within` lies on. Seen at an angle a finder is foreshortened
/// along one axis and not the other, and its two modules differ.
fn module_axes(c: &Capstone, toward: [f64; 2], within: [f64; 2]) -> ([f64; 2], [f64; 2]) {
    let q = c.corners;
    let mid = |i: usize| [(q[i][0] + q[(i + 1) % 4][0]) / 2.0, (q[i][1] + q[(i + 1) % 4][1]) / 2.0];
    // From the middle of one side to the middle of the side opposite,
    // seven modules: across the finder, and down it.
    let axes = [[(mid(1)[0] - mid(3)[0]) / 7.0, (mid(1)[1] - mid(3)[1]) / 7.0], [(mid(2)[0] - mid(0)[0]) / 7.0, (mid(2)[1] - mid(0)[1]) / 7.0]];
    let dot = |u: [f64; 2], v: [f64; 2]| u[0] * v[0] + u[1] * v[1];
    let cos_to = |axis: [f64; 2], v: [f64; 2]| dot(axis, v) / dot(axis, axis).sqrt().max(1e-12);
    let to = [toward[0] - c.center[0], toward[1] - c.center[1]];
    let (mut along, mut across) = if cos_to(axes[0], to).abs() >= cos_to(axes[1], to).abs() { (axes[0], axes[1]) } else { (axes[1], axes[0]) };
    if dot(along, to) < 0.0 {
        along = [-along[0], -along[1]];
    }
    if dot(across, [within[0] - c.center[0], within[1] - c.center[1]]) < 0.0 {
        across = [-across[0], -across[1]];
    }
    (along, across)
}

/// The timing pattern between two capstones that stand as the finders
/// along one side of a QR symbol, read from the centre of its first
/// module to the centre of its last: five modules along from each finder's
/// centre toward the other and three in toward the symbol, which `inside`
/// (a rectified point) says the side of. One run for each module, and the
/// symbol is sixteen modules wider than their count. `None` when the runs
/// do not read as a timing pattern.
///
/// The row is placed by each finder's own modules, along and across (see
/// [`module_axes`]), as OpenCV's ArUco-based detector starts its timing
/// line from the corners of the finder's own quadrilateral
/// (`getTimingStart`). It was placed by one module for both finders and
/// both directions, the mean of the four: a symbol photographed from
/// below has finders seven pixels a module across and five down, and
/// three mean modules in from the centres of the top two lay more than
/// half a module past the timing row, in the data. No pattern read there,
/// nearly every true triple in a photograph taken at an angle was
/// charged for both its sides, the count of modules was lost with it,
/// and the size fell to the spacing over the same mean module, which
/// named a version or two too many. Capstones whose corners were never
/// measured have no module of their own, and the mean still places the
/// row for them.
fn timing_between(image: &Luma, frame: &Frame, a: &Capstone, b: &Capstone, inside: [f64; 2]) -> Option<TimingCount> {
    let (pa, pb) = (frame.to_rectified(a.center), frame.to_rectified(b.center));
    let apart = distance(pa, pb);
    if apart < 1e-6 {
        return None;
    }
    // Where the row starts and ends in the photo, and the pitches between
    // the two timing module centres, as the spacing has it.
    let (start, end, span) = if a.measured && b.measured {
        let within = frame.to_image.apply(inside[0], inside[1]);
        let (along_a, across_a) = module_axes(a, b.center, within);
        let (along_b, across_b) = module_axes(b, a.center, within);
        let from = |c: &Capstone, along: [f64; 2], across: [f64; 2]| {
            [c.center[0] + 5.0 * along[0] + 3.0 * across[0], c.center[1] + 5.0 * along[1] + 3.0 * across[1]]
        };
        let length = |v: [f64; 2]| (v[0] * v[0] + v[1] * v[1]).sqrt();
        let module = (length(along_a) + length(along_b)) / 2.0;
        (from(a, along_a, across_a), from(b, along_b, across_b), distance(a.center, b.center) / module.max(1e-9) - 10.0)
    } else {
        let module = (a.module_mean() + b.module_mean()) / 2.0;
        let u = [(pb[0] - pa[0]) / apart, (pb[1] - pa[1]) / apart];
        let mid = [(pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0];
        let mut n = [-u[1], u[0]];
        if (inside[0] - mid[0]) * n[0] + (inside[1] - mid[1]) * n[1] < 0.0 {
            n = [-n[0], -n[1]];
        }
        let point = |p: [f64; 2], along: f64| {
            frame.to_image.apply(p[0] + (u[0] * along + n[0] * 3.0) * module, p[1] + (u[1] * along + n[1] * 3.0) * module)
        };
        (point(pa, 5.0), point(pb, -5.0), apart / module - 10.0)
    };
    if span < 2.0 {
        return None;
    }
    let pitch = distance(start, end) / span;
    let threshold = (a.threshold + b.threshold) / 2.0;
    let runs = runs_along(image, start, end, pitch / TIMING_SAMPLES, threshold, a.inverted);
    let (&(first, first_len), &(last, last_len)) = runs.first().zip(runs.last())?;
    if runs.len() < 3 || !first || !last {
        return None;
    }
    let interior: Vec<f64> = runs[1..runs.len() - 1].iter().map(|r| r.1).collect();
    let median = regular_pitch(&interior)?;
    // The count must agree roughly with what the spacing says lies
    // between the two: data across the diagonal of a symbol can
    // alternate regularly for a while, but not for the right distance.
    let modules = runs.len();
    if (modules as f64 - (span + 1.0)).abs() > 0.25 * span + 1.5 {
        return None;
    }
    // The end runs begin and end at module centres: half a module each.
    let ends = first_len <= TIMING_RUN_MAX * median && last_len <= TIMING_RUN_MAX * median;
    Some(TimingCount { size: modules as f64 + 16.0, exact: ends && (modules + 16) % 4 == 1 })
}

/// What a walk out from a lone finder along one of its symbol's timing
/// patterns says of the symbol's size along that axis.
#[derive(Clone, Copy, Debug, Default)]
struct Walk {
    /// Modules, by counting the timing pattern's modules to where it ends.
    by_count: Option<f64>,
    /// Modules, by how far the symbol extends before the quiet zone.
    by_extent: Option<f64>,
    /// Modules, when the quiet zone begins at the finder's own edge with
    /// no separator between: seven, the height of an R7 symbol, whose
    /// finder fills it. A walk away from any symbol reads the same.
    at_edge: Option<f64>,
}

/// Walk from a capstone's centre along `along`, one module vector into
/// the symbol in photo pixels, on the line `across` module vectors over
/// from the centre, for `reach` modules, reading the runs: the finder's
/// own dark, the separator, then the timing pattern. Its modules are
/// counted until a run that is not one, and the extent is measured to
/// the first light run wide enough to be the quiet zone.
fn walk_from_finder(image: &Luma, c: &Capstone, along: [f64; 2], across: [f64; 2], reach: f64) -> Walk {
    let module = distance([0.0, 0.0], along);
    if module < 1e-6 {
        return Walk::default();
    }
    let start = [c.center[0] + across[0], c.center[1] + across[1]];
    let end = [start[0] + along[0] * reach, start[1] + along[1] * reach];
    let runs = runs_along(image, start, end, module / TIMING_SAMPLES, c.threshold, c.inverted);
    // The finder's own dark, from its centre to its edge, then the
    // separator, light and one module wide.
    let [(true, ring), (false, separator), ..] = runs.as_slice() else { return Walk::default() };
    if *ring < 2.5 * module {
        return Walk::default();
    }
    if !(TIMING_RUN_MIN * module..=TIMING_RUN_MAX * module).contains(separator) {
        // Paper where the separator would be: the symbol ends with the
        // finder, if it lies this way at all.
        let at_edge = (*separator >= PAPER_RUN * module).then(|| 3.5 + ring / module);
        return Walk { at_edge, ..Walk::default() };
    }
    let one = |len: f64| (TIMING_RUN_MIN * module..=TIMING_RUN_MAX * module).contains(&len);
    let mut counted = 0usize;
    let mut last_light = false;
    let mut by_count = None;
    for &(dark, len) in &runs[2..] {
        if one(len) {
            counted += 1;
            last_light = !dark;
            continue;
        }
        // The pattern ends at a run that is not a module: the far
        // finder, dark, or the paper, light. A run too short is damage.
        // A timing pattern ends on a dark module, so a light one counted
        // just before the far finder was that finder's separator.
        if len > TIMING_RUN_MAX * module && (!dark || counted > 0) {
            by_count = Some((counted - usize::from(dark && last_light)) as f64);
        }
        break;
    }
    // The extent from the finder's centre, three and a half modules
    // behind its edge, to the quiet zone.
    let mut extent = 0.0;
    let mut by_extent = None;
    for &(dark, len) in &runs {
        if !dark && len >= PAPER_RUN * module && extent > *ring {
            by_extent = Some(3.5 + extent / module);
            break;
        }
        extent += len;
    }
    Walk { by_count, by_extent, at_edge: None }
}

/// The size of a lone finder's symbol from walks along its timing
/// patterns, tried with the symbol on each side of the finder in turn,
/// and whether the size found is one the family has, to the module. A
/// Micro QR symbol's patterns run along its top and left edges from the
/// finder to the far corners, so their modules count the size; a QR
/// symbol's run three modules in to the far finders; an rMQR symbol's
/// are broken by its alignment patterns, and only the extent counts. The
/// size is along the capstone's own axes, which are the frame's: an rMQR
/// symbol on end in the frame comes out the other way round, and the
/// versions are matched both ways.
fn lone_finder_size(image: &Luma, c: &Capstone, family: Family) -> Option<([f64; 2], bool)> {
    let [c0, c1, _, c3] = c.corners;
    let mx = [(c1[0] - c0[0]) / 7.0, (c1[1] - c0[1]) / 7.0];
    let my = [(c3[0] - c0[0]) / 7.0, (c3[1] - c0[1]) / 7.0];
    let versions = family.versions();
    let reach = versions.iter().map(|v| v.width().max(v.height())).max()? as f64 + 2.0;
    let scale = |v: [f64; 2], k: f64| [v[0] * k, v[1] * k];
    // The timing patterns lie on the symbol's edges, three modules out
    // from the finder's centre, or three modules in on row and column six.
    let over = if matches!(family, Family::Micro | Family::Rmqr) { -3.0 } else { 3.0 };
    const SIDES: [(f64, f64); 4] = [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)];
    let walks = SIDES.map(|(sx, sy)| {
        let (inward_x, inward_y) = (scale(mx, sx), scale(my, sy));
        let row = walk_from_finder(image, c, inward_x, scale(inward_y, over), reach);
        let column = walk_from_finder(image, c, inward_y, scale(inward_x, over), reach);
        (row, column)
    });
    let nearest = |w: f64, h: f64| {
        let off = |v: &SymbolVersion, along: f64, down: f64| (v.width() as f64 - along).abs() + (v.height() as f64 - down).abs();
        versions.iter().map(|v| off(v, w, h).min(off(v, h, w))).fold(f64::INFINITY, f64::min)
    };
    // The extent along each axis, with whether it was counted to the
    // module, from one side of the finder; `None` where a walk failed.
    type Extents = Option<((f64, bool), (f64, bool))>;
    let best_of = |sizes: &[Extents]| {
        let mut best: Option<(f64, [f64; 2], bool)> = None;
        for &((w, w_counted), (h, h_counted)) in sizes.iter().flatten() {
            let error = nearest(w, h);
            let exact = w_counted && h_counted && error <= 1.0;
            if error <= 3.0 && best.is_none_or(|b| error < b.0) {
                best = Some((error, [w, h], exact));
            }
        }
        best.map(|(_, size, exact)| (size, exact))
    };
    let size_of = |walk: Walk| match family {
        Family::Micro => walk.by_count.map(|n| (n + 8.0, true)).or(walk.by_extent.map(|e| (e, false))),
        Family::Qr | Family::Model1 => walk.by_count.map(|n| (n + 16.0, true)).or(walk.by_extent.map(|e| (e, false))),
        Family::Rmqr => walk.by_extent.map(|e| (e, false)),
    };
    let found = best_of(&walks.map(|(row, column)| size_of(row).zip(size_of(column))));
    if found.is_some() || family != Family::Rmqr {
        return found;
    }
    // An R7 symbol is as tall as its finder: there is no separator below
    // it and no timing pattern beside it to walk, only paper both ways.
    // That is believed of an axis only when the walks along it from both
    // sides of the finder say so, and no reading with a separator was
    // found, since a walk the wrong way out of a taller symbol meets
    // paper at once as well.
    let bare = |axis: usize, i: usize| {
        // The side across the finder along `axis`: the other sign of x
        // for a row, of y for a column.
        let j = if axis == 0 { i ^ 1 } else { i ^ 2 };
        let pick = |k: usize| if axis == 0 { walks[k].0 } else { walks[k].1 };
        pick(i).at_edge.zip(pick(j).at_edge).map(|(a, b)| ((a + b) / 2.0, false))
    };
    let sizes: Vec<Extents> = (0..4)
        .map(|i| {
            let (row, column) = walks[i];
            size_of(row).or_else(|| bare(0, i)).zip(size_of(column).or_else(|| bare(1, i)))
        })
        .collect();
    best_of(&sizes)
}

/// How far a version's size lies from a size estimate, as shares of the
/// estimate along each axis, summed. The estimate is along the frame's
/// axes, and a symbol may stand on end in the frame, so the version is
/// tried both ways round and the nearer taken; only an rMQR symbol, which
/// is not square, can tell the two apart.
fn size_error(version: SymbolVersion, hint: [f64; 2]) -> f64 {
    let (w, h) = (version.width() as f64, version.height() as f64);
    let lying = |along: f64, down: f64| ((w - along) / along).abs() + ((h - down) / down).abs();
    lying(hint[0], hint[1]).min(lying(hint[1], hint[0]))
}

/// The versions of `current`'s family nearest the size estimate, and
/// `current` itself when it is not far off, first.
fn candidate_versions(current: SymbolVersion, hint: [f64; 2], tight: bool) -> Vec<SymbolVersion> {
    let error = |v: SymbolVersion| size_error(v, hint);
    let mut versions = current.family().versions();
    versions.sort_by(|a, b| error(*a).total_cmp(&error(*b)));
    let keep = match (current.family(), tight) {
        (Family::Rmqr, true) => VERSIONS_TIGHT_RMQR,
        (Family::Rmqr, false) => VERSIONS_LOOSE_RMQR,
        (_, true) => VERSIONS_TIGHT,
        (_, false) => VERSIONS_LOOSE,
    };
    versions.truncate(keep);
    versions.retain(|&v| v != current);
    if error(current) < 0.5 {
        versions.insert(0, current);
    }
    versions
}

/// A fixed pattern of a version worth looking for on its own: the modules
/// in a rectangle of the symbol.
#[derive(Clone, Copy, Debug)]
struct Site {
    kind: AnchorKind,
    x0: i64,
    y0: i64,
    width: usize,
    height: usize,
    /// The pattern's centre in module coordinates.
    at: [f64; 2],
    /// The window reaches outside the symbol, where the quiet zone must
    /// read light: a timing pattern along the edge of a Micro QR symbol
    /// is five modules in a line, which print beside the symbol imitates
    /// too easily, and the paper beside it is what tells them apart.
    paper: bool,
}

impl Site {
    fn finder(x0: usize, y0: usize, height: usize) -> Self {
        Self { kind: AnchorKind::Finder, x0: x0 as i64, y0: y0 as i64, width: 8, height, at: [x0 as f64 + 3.5, y0 as f64 + 3.5], paper: false }
    }

    /// The site's template: its fixed modules, and the quiet zone where
    /// the window reaches outside the symbol.
    fn template(&self, func: &FunctionMap) -> Template {
        let mut template = Template::window(func, self.x0, self.y0, self.width, self.height);
        if self.paper {
            for dy in 0..self.height {
                for dx in 0..self.width {
                    let (x, y) = (self.x0 + dx as i64, self.y0 + dy as i64);
                    if x < 0 || y < 0 || x >= func.width as i64 || y >= func.height as i64 {
                        template.expected[dy * self.width + dx] = Some(false);
                    }
                }
            }
        }
        template
    }

    fn rect_corners(&self) -> [[f64; 2]; 4] {
        let (x0, y0) = (self.x0 as f64, self.y0 as f64);
        let (x1, y1) = (x0 + self.width as f64, y0 + self.height as f64);
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    }
}

/// Finders first, in the order top-left, top-right, bottom-left.
fn sites(func: &FunctionMap) -> Vec<Site> {
    let (w, h) = (func.width, func.height);
    let mut out = Vec::new();
    match func.version {
        SymbolVersion::Qr(_) | SymbolVersion::Model1(_) => {
            // The finder with its separator: eight modules square.
            out.push(Site::finder(0, 0, 8));
            out.push(Site { x0: w as i64 - 8, ..Site::finder(w - 7, 0, 8) });
            out.push(Site { y0: h as i64 - 8, ..Site::finder(0, h - 7, 8) });
            // Model 1 has no alignment patterns.
            let centers = match func.version {
                SymbolVersion::Qr(v) => tables::qr_alignment_centers(v),
                _ => Vec::new(),
            };
            let last = centers.len().saturating_sub(1);
            for (i, &cx) in centers.iter().enumerate() {
                for (j, &cy) in centers.iter().enumerate() {
                    if (i == 0 && j == 0) || (i == 0 && j == last) || (i == last && j == 0) {
                        continue;
                    }
                    out.push(Site {
                        kind: AnchorKind::Alignment,
                        x0: cx as i64 - 2,
                        y0: cy as i64 - 2,
                        width: 5,
                        height: 5,
                        at: [cx as f64 + 0.5, cy as f64 + 0.5],
                        paper: false,
                    });
                }
            }
        }
        SymbolVersion::Micro(_) => out.push(Site::finder(0, 0, 8)),
        SymbolVersion::Rmqr(_) => {
            out.push(Site::finder(0, 0, h.min(8)));
            out.push(Site {
                kind: AnchorKind::FinderSub,
                x0: w as i64 - 5,
                y0: h as i64 - 5,
                width: 5,
                height: 5,
                at: [w as f64 - 2.5, h as f64 - 2.5],
                paper: false,
            });
            // The three-by-three alignment patterns with the timing modules
            // beside and below them, or above them at the bottom edge. The
            // site's place is the pattern's own centre, a module and a
            // half in from the edge it lies on.
            for &cx in tables::rmqr_alignment_centers(w) {
                for (y0, centre) in [(0, 1.5), (h - 4, h as f64 - 1.5)] {
                    out.push(Site {
                        kind: AnchorKind::Alignment,
                        x0: cx as i64 - 2,
                        y0: y0 as i64,
                        width: 5,
                        height: 4,
                        at: [cx as f64 + 0.5, centre],
                        paper: false,
                    });
                }
            }
        }
    }
    timing_sites(func, &mut out);
    out
}

/// Most dark timing modules anchored on along one timing pattern. A few,
/// spread along it, are enough to follow the bend of curled paper; every
/// one would cost a search each on a large symbol and crowd the display.
const MAX_TIMING_SITES: usize = 8;
/// Modules of the timing pattern in a site's template, on a QR symbol,
/// whose pattern has data on both sides: a row of data reads as five
/// alternating modules once in thirty-two, and as seven once in a
/// hundred and twenty-eight. A Micro QR symbol's pattern has the quiet
/// zone beside it, which tells it from data, and five suffice.
const QR_TIMING_SPAN: usize = 7;
const MICRO_TIMING_SPAN: usize = 5;

/// The dark modules of a timing pattern between `first` and `last`, both
/// dark and two apart, thinned to at most [`MAX_TIMING_SITES`] with both
/// ends kept: the ends are where a lone finder's symbol has nothing else
/// to hold its far corners by.
fn timing_picks(first: usize, last: usize) -> Vec<usize> {
    let all: Vec<usize> = (first..=last).step_by(2).collect();
    if all.len() <= MAX_TIMING_SITES {
        return all;
    }
    (0..MAX_TIMING_SITES)
        .map(|k| all[((k * (all.len() - 1)) as f64 / (MAX_TIMING_SITES - 1) as f64).round() as usize])
        .collect()
}

/// Sites on the timing patterns: five modules of a pattern, dark at both
/// ends, with the dark one in the middle as the anchor. A QR symbol's run
/// along row and column six between the finders; a Micro QR symbol's along
/// its top and left edges to the far corners; an rMQR symbol's are already
/// in its alignment sites, which take the timing modules beside them. The
/// window is pulled inside the symbol at the far end, since beyond it is
/// quiet zone, not pattern.
fn timing_sites(func: &FunctionMap, out: &mut Vec<Site>) {
    let (w, h) = (func.width, func.height);
    // A Micro QR symbol's patterns lie on its edges, with the quiet zone
    // beside them, which the window takes in.
    let (line, first, last, paper, span) = match func.version {
        SymbolVersion::Qr(_) | SymbolVersion::Model1(_) => (6usize, 8usize, [w - 9, h - 9], false, QR_TIMING_SPAN),
        SymbolVersion::Micro(_) => (0, 8, [w - 1, h - 1], true, MICRO_TIMING_SPAN),
        SymbolVersion::Rmqr(_) => return,
    };
    let reach = i64::from(paper);
    let half = span / 2;
    // Only modules of the timing pattern itself, or of the finder and
    // separator it runs into: an alignment pattern that sits on the
    // timing line has an anchor of its own, and a second one there would
    // only quarrel with it.
    let timing = |x: usize, y: usize| matches!(func.role(x, y), Role::Timing | Role::Finder | Role::Separator);
    for x in timing_picks(first, last[0]) {
        let x0 = (x + half).min(w - 1) + 1 - span;
        if func.role(x, line) == Role::Timing && (x0..x0 + span).all(|xx| timing(xx, line)) {
            out.push(Site {
                kind: AnchorKind::Timing,
                x0: x0 as i64,
                y0: line as i64 - reach,
                width: span,
                height: 1 + reach as usize,
                at: [x as f64 + 0.5, line as f64 + 0.5],
                paper,
            });
        }
    }
    for y in timing_picks(first, last[1]) {
        let y0 = (y + half).min(h - 1) + 1 - span;
        if func.role(line, y) == Role::Timing && (y0..y0 + span).all(|yy| timing(line, yy)) {
            out.push(Site {
                kind: AnchorKind::Timing,
                x0: line as i64 - reach,
                y0: y0 as i64,
                width: 1 + reach as usize,
                height: span,
                at: [line as f64 + 0.5, y as f64 + 0.5],
                paper,
            });
        }
    }
}

/// The site of an anchor, by kind and centre.
fn site_of(func: &FunctionMap, anchor: &Anchor) -> Option<Site> {
    sites(func)
        .into_iter()
        .find(|s| s.kind == anchor.kind && (s.at[0] - anchor.at[0]).abs() < 1e-6 && (s.at[1] - anchor.at[1]).abs() < 1e-6)
}

/// Where a module coordinate of a `width` × `height` symbol lands across
/// the outline, 0 to 1 along each side, when the symbol sits in the outline
/// turned `turns` quarter turns clockwise, after being reflected left to
/// right when `mirrored`.
fn to_unit(turns: u8, mirrored: bool, width: f64, height: f64, [x, y]: [f64; 2]) -> [f64; 2] {
    let x = if mirrored { width - x } else { x };
    match turns % 4 {
        0 => [x / width, y / height],
        1 => [(height - y) / height, x / width],
        2 => [(width - x) / width, (height - y) / height],
        _ => [y / height, (width - x) / width],
    }
}

/// The capstone for each finder position, or none, minimising the summed
/// squared distance with an unmatched finder costing the tolerance squared,
/// and the capstones of one symbol expected to share a module size: a
/// logo or a neighbouring symbol's finder is usually a different size.
fn assign(capstones: &[Capstone], finders: &[[f64; 2]]) -> (f64, Vec<Option<usize>>) {
    fn go(
        capstones: &[Capstone],
        finders: &[[f64; 2]],
        f: usize,
        cost: f64,
        used: &mut Vec<bool>,
        current: &mut Vec<Option<usize>>,
        best: &mut (f64, Vec<Option<usize>>),
    ) {
        if cost >= best.0 {
            return;
        }
        if f == finders.len() {
            *best = (cost, current.clone());
            return;
        }
        current.push(None);
        go(capstones, finders, f + 1, cost + MATCH_TOLERANCE * MATCH_TOLERANCE, used, current, best);
        current.pop();
        for (c, capstone) in capstones.iter().enumerate() {
            if used[c] {
                continue;
            }
            let d = distance(capstone.unit, finders[f]);
            let size: f64 = current
                .iter()
                .flatten()
                .map(|&other| {
                    let (a, b) = (capstone.module_mean(), capstones[other].module_mean());
                    SIZE_WEIGHT * (a.max(b) / a.min(b) - 1.0).powi(2)
                })
                .sum();
            used[c] = true;
            current.push(Some(c));
            go(capstones, finders, f + 1, cost + d * d + size, used, current, best);
            current.pop();
            used[c] = false;
        }
    }
    let mut best = (f64::INFINITY, Vec::new());
    go(capstones, finders, 0, 0.0, &mut vec![false; capstones.len()], &mut Vec::new(), &mut best);
    best
}

/// Correspondences from a capstone matched to a finder: the four corners of
/// its seven-module patch, in symbol module coordinates and photo pixels.
fn capstone_points(capstone: &Capstone, site: &Site, turns: u8, mirrored: bool, width: f64, height: f64) -> Vec<Pair> {
    let (x0, y0) = (site.at[0] - 3.5, site.at[1] - 3.5);
    let patch = [[x0, y0], [x0 + 7.0, y0], [x0 + 7.0, y0 + 7.0], [x0, y0 + 7.0]];
    // The capstone's corners are in the outline's order; find which patch
    // corner lands at each of the outline's corners.
    let units = patch.map(|p| to_unit(turns, mirrored, width, height, p));
    let pick = |key: &dyn Fn([f64; 2]) -> f64| {
        (0..4).min_by(|&a, &b| key(units[a]).total_cmp(&key(units[b]))).expect("four corners")
    };
    let order = [
        pick(&|[u, v]| u + v),
        pick(&|[u, v]| v - u),
        pick(&|[u, v]| -u - v),
        pick(&|[u, v]| u - v),
    ];
    (0..4).map(|i| (patch[order[i]], capstone.corners[i])).collect()
}

/// Greatest distance of an anchor from the line through the two farthest
/// apart, in modules.
fn spread(anchors: &[Anchor]) -> f64 {
    let points: Vec<[f64; 2]> = anchors.iter().map(|a| a.at).collect();
    super::projection::spread(&points)
}

/// The richest transform the anchors found so far can support. One
/// capstone already gives an affine map: its four corners are placed to a
/// fraction of a pixel, and a photo taken at an angle foreshortens one axis
/// in a way a similarity cannot follow.
fn choose_model(anchors: &[Anchor], points: usize) -> Model {
    if anchors.len() >= 3 && points >= 8 && spread(anchors) >= SPREAD_MODULES {
        Model::Projective
    } else if points >= 3 {
        Model::Affine
    } else {
        Model::Similarity
    }
}

/// Look for a site's pattern within `window` modules of where `predicted`
/// puts its centre, with the shape of the patch taken from `map`. Returns
/// the anchor and its correspondences: the four corners of a finder,
/// whose shape the template pins down, or the centre alone of a smaller
/// pattern.
#[allow(clippy::too_many_arguments)]
fn search_site(
    image: &Luma,
    func: &FunctionMap,
    site: &Site,
    map: &impl Project,
    window: f64,
    inverted: bool,
    threshold: Option<f32>,
) -> Option<(Anchor, Vec<Pair>)> {
    let template = site.template(func);
    if template.fixed() < 4 {
        return None;
    }
    let rect = site.rect_corners();
    let placed = |dx: f64, dy: f64| rect.map(|[x, y]| map.apply(x + dx, y + dy));
    let scorer = Scorer::new(image, &template, placed(0.0, 0.0), threshold).with_along(6);
    let sign = if inverted { -1.0 } else { 1.0 };
    let steps = (window / SEARCH_STEP).ceil() as i32;
    let timing = site.kind == AnchorKind::Timing;
    let penalty = if timing { TIMING_DISTANCE_PENALTY } else { DISTANCE_PENALTY };
    // Every placement in the window, scored by contrast less a little for
    // its distance from the prediction: a timing pattern repeats every two
    // modules, and a match two modules off is more likely a false one, as
    // ZBar keeps its alignment searches close for the same reason.
    let mut placements: Vec<([[f64; 2]; 4], f64)> = Vec::new();
    for iy in -steps..=steps {
        for ix in -steps..=steps {
            let (dx, dy) = (ix as f64 * SEARCH_STEP, iy as f64 * SEARCH_STEP);
            let corners = placed(dx, dy);
            if let Some(score) = scorer.contrast_signed(corners, sign) {
                placements.push((corners, score - penalty * (dx * dx + dy * dy)));
            }
        }
    }
    placements.sort_by(|a, b| b.1.total_cmp(&a.1));
    let moves: &[[f64; 4]] = if site.kind == AnchorKind::Finder { &MOVES } else { &SHIFT };
    let mut evaluations = 0;
    // The best few placements are each settled and read; the first that
    // reads as the pattern wins.
    let mut found = None;
    for (start, _) in placements.iter().copied().take(PLACEMENTS_TRIED) {
        let Some((corners, score)) = scorer.descend(start, sign, moves, scorer.module_px() * SEARCH_STEP, &mut evaluations) else { continue };
        if score <= 0.0 {
            continue;
        }
        // A timing site has too few modules to carry a damaged one: every
        // one of them must read right. Where the quiet zone is in the
        // window it is the paper that tells the pattern from print, and
        // must read clean to the module; the pattern itself, blurred on
        // a small photo, may then have one module read wrong.
        if timing {
            let q = scorer.agreement_quarters(corners, inverted)?;
            let (matching, known) = q.iter().fold((0, 0), |a, q| (a.0 + q.0, a.1 + q.1));
            let whole = known == template.fixed() && matching == known;
            let passable = site.paper && {
                // The quarters split the window across and along; the
                // paper is the first row of a row site, the first column
                // of a column site.
                let (paper, pattern) = if site.width > site.height { ([q[0], q[1]], [q[2], q[3]]) } else { ([q[0], q[2]], [q[1], q[3]]) };
                let (pm, pk) = (paper[0].0 + paper[1].0, paper[0].1 + paper[1].1);
                let (tm, tk) = (pattern[0].0 + pattern[1].0, pattern[0].1 + pattern[1].1);
                pm == pk && pk + tk == template.fixed() && tm + 1 >= tk
            };
            if whole || passable {
                found = Some((corners, false));
                break;
            }
            continue;
        }
        if let Some(partial) = accept(&scorer, corners, inverted) {
            found = Some((corners, partial));
            break;
        }
    }
    let (corners, partial) = found?;
    let patch = Homography::rect_to_quad(site.width as f64, site.height as f64, corners)?;
    let center = patch.apply(site.at[0] - site.x0 as f64, site.at[1] - site.y0 as f64);
    let points = if site.kind == AnchorKind::Finder {
        rect.iter().copied().zip(corners).collect()
    } else {
        vec![(site.at, center)]
    };
    Some((Anchor { kind: site.kind, at: site.at, center, partial, inferred: false }, points))
}

/// An anchor with the correspondences it contributes to the fit.
struct Placed {
    anchor: Anchor,
    pairs: Vec<Pair>,
}

fn anchors_of(placed: &[Placed]) -> Vec<Anchor> {
    placed.iter().map(|p| p.anchor).collect()
}

fn pairs_of(placed: &[Placed]) -> Vec<Pair> {
    placed.iter().flat_map(|p| p.pairs.iter().copied()).collect()
}

/// The global fit corrected by the anchors near a point: each anchor's
/// residual against the fit, weighted by the inverse square of its
/// distance, so that a prediction beside an anchor follows that anchor
/// and one far from every anchor follows the fit. Bounded by the residuals
/// themselves, which a spline is not where it extrapolates, and so safe
/// for guessing where the next pattern is and for judging one against
/// its neighbours, as ZBar extrapolates each alignment pattern from the
/// cells beside it.
struct LocalPrediction {
    base: Homography,
    controls: Vec<ControlPoint>,
    residuals: Vec<[f64; 2]>,
}

impl LocalPrediction {
    fn new(base: Homography, placed: &[Placed]) -> Self {
        let controls: Vec<ControlPoint> = placed.iter().map(|p| p.anchor.control_point()).collect();
        let residuals = controls
            .iter()
            .map(|c| {
                let [bx, by] = base.apply(c.at[0], c.at[1]);
                [c.px[0] - bx, c.px[1] - by]
            })
            .collect();
        Self { base, controls, residuals }
    }
}

impl Project for LocalPrediction {
    fn base(&self, x: f64, y: f64) -> [f64; 2] {
        self.base.apply(x, y)
    }

    fn displacement(&self, x: f64, y: f64) -> [f64; 2] {
        let (mut sum, mut weight) = ([0.0, 0.0], 0.0);
        for (c, r) in self.controls.iter().zip(&self.residuals) {
            let d2 = (c.at[0] - x).powi(2) + (c.at[1] - y).powi(2);
            let w = 1.0 / (d2 + 0.25);
            sum[0] += w * r[0];
            sum[1] += w * r[1];
            weight += w;
        }
        if weight > 0.0 { [sum[0] / weight, sum[1] / weight] } else { [0.0, 0.0] }
    }
}

fn local_map(map: Homography, placed: &[Placed]) -> LocalPrediction {
    LocalPrediction::new(map, placed)
}

/// Size of a module in photo pixels at `at`, under `map`.
fn module_size_at(map: &impl Project, at: [f64; 2]) -> f64 {
    let here = map.apply(at[0], at[1]);
    (distance(here, map.apply(at[0] + 1.0, at[1])) + distance(here, map.apply(at[0], at[1] + 1.0))) / 2.0
}

/// The anchor its neighbours explain worst, when it lies further from
/// where they predict it than a true pattern could. Each anchor is judged
/// against the fit of all the others, bent through them; the allowance
/// grows with its distance from its nearest neighbour. Finder capstones
/// are never doubted: a nine-module template pinned them down, and they
/// carry the geometry. Under an affine fit the far side of a tilted
/// symbol is off for want of perspective, not for being wrong, so only a
/// perspective fit of the others may condemn an anchor.
fn worst_outlier(placed: &[Placed]) -> Option<usize> {
    if placed.len() < OUTLIER_MIN_ANCHORS {
        return None;
    }
    let mut worst: Option<(usize, f64)> = None;
    for k in 0..placed.len() {
        if placed[k].anchor.kind == AnchorKind::Finder {
            continue;
        }
        let others: Vec<&Placed> = placed.iter().enumerate().filter(|(i, _)| *i != k).map(|(_, p)| p).collect();
        let anchors: Vec<Anchor> = others.iter().map(|p| p.anchor).collect();
        let pairs: Vec<Pair> = others.iter().flat_map(|p| p.pairs.iter().copied()).collect();
        let model = choose_model(&anchors, pairs.len());
        if model != Model::Projective {
            continue;
        }
        let Some(base) = Homography::fit(model, &pairs) else { continue };
        let others_placed: Vec<Placed> = others.iter().map(|p| Placed { anchor: p.anchor, pairs: Vec::new() }).collect();
        let map = LocalPrediction::new(base, &others_placed);
        let at = placed[k].anchor.at;
        let predicted = map.apply(at[0], at[1]);
        let module = module_size_at(&map, at).max(1e-9);
        let residual = distance(predicted, placed[k].anchor.center) / module;
        let far = anchors.iter().map(|a| distance(a.at, at)).fold(f64::INFINITY, f64::min);
        let allowance = OUTLIER_MODULES.max(OUTLIER_BASE + OUTLIER_RATE * far);
        if residual > allowance && worst.is_none_or(|(_, r)| residual / allowance > r) {
            worst = Some((k, residual / allowance));
        }
    }
    worst.map(|(k, _)| k)
}

/// The map the anchors support, with anchors their neighbours cannot
/// explain dropped one at a time, worst first, and the map refitted
/// without them.
fn fit_anchors(placed: &mut Vec<Placed>) -> Option<(Homography, Model)> {
    loop {
        let anchors = anchors_of(placed);
        let pairs = pairs_of(placed);
        let model = choose_model(&anchors, pairs.len());
        let map = Homography::fit(model, &pairs)?;
        match worst_outlier(placed) {
            Some(k) => {
                placed.remove(k);
            }
            None => return Some((map, model)),
        }
    }
}

/// Timing anchors that break the smooth trend of their line: along each
/// timing pattern the anchors' residuals against `base`, the plane fitted
/// to every anchor, are fitted with a gentle curve (a line for three
/// anchors, a bend for more), and any anchor further than
/// [`TIMING_LINE_TOLERANCE`] from it is returned with the position the
/// curve predicts for it, worst first, the curve refitted without it.
fn timing_line_strays(anchors: &[Anchor], base: &Homography) -> Vec<(usize, [f64; 2])> {
    let mut strays = Vec::new();
    for axis in 0..2 {
        // Anchors on one line share the other coordinate.
        let mut lines: Vec<f64> = anchors.iter().filter(|a| a.kind == AnchorKind::Timing).map(|a| a.at[1 - axis]).collect();
        lines.sort_by(f64::total_cmp);
        lines.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        for line in lines {
            let mut members: Vec<usize> = (0..anchors.len())
                .filter(|&i| anchors[i].kind == AnchorKind::Timing && (anchors[i].at[1 - axis] - line).abs() < 1e-6)
                .collect();
            // Each anchor is judged against the curve through the others:
            // a curve through all of them bends to meet a stray at the
            // end of the line, and judges it fair. Fewer than six on a
            // line, and the curve through the rest predicts an end anchor
            // no better than the anchor reads itself.
            loop {
                if members.len() < TIMING_LINE_MIN {
                    break;
                }
                let samples: Vec<([f64; 2], f64)> = members
                    .iter()
                    .map(|&i| {
                        let a = &anchors[i];
                        let [bx, by] = base.apply(a.at[0], a.at[1]);
                        ([a.center[0] - bx, a.center[1] - by], a.at[axis])
                    })
                    .collect();
                let worst = (0..members.len())
                    .filter_map(|k| {
                        let others: Vec<([f64; 2], f64)> = samples.iter().enumerate().filter(|(j, _)| *j != k).map(|(_, s)| *s).collect();
                        // Four others fit a bend; three, a line.
                        let degree = if others.len() >= 4 { 2 } else { 1 };
                        let fit = fit_polynomial(&others, degree)?;
                        let predicted = fit(samples[k].1);
                        let deviation = distance(samples[k].0, predicted) / module_size_at(base, anchors[members[k]].at).max(1e-9);
                        Some((k, deviation, predicted))
                    })
                    .max_by(|a, b| a.1.total_cmp(&b.1));
                let Some((k, deviation, predicted)) = worst else { break };
                if deviation <= TIMING_LINE_TOLERANCE {
                    break;
                }
                let i = members[k];
                let [bx, by] = base.apply(anchors[i].at[0], anchors[i].at[1]);
                strays.push((i, [bx + predicted[0], by + predicted[1]]));
                members.remove(k);
            }
        }
    }
    strays
}

/// Least-squares polynomial of `degree` (one or two) through `samples`,
/// each a two-dimensional value at a position, as a function of position.
fn fit_polynomial(samples: &[([f64; 2], f64)], degree: usize) -> Option<impl Fn(f64) -> [f64; 2]> {
    let n = degree + 1;
    let mut a = [[0.0f64; 3]; 3];
    let mut b = [[0.0f64; 3]; 2];
    for &(value, s) in samples {
        let basis = [1.0, s, s * s];
        for i in 0..n {
            for j in 0..n {
                a[i][j] += basis[i] * basis[j];
            }
            b[0][i] += basis[i] * value[0];
            b[1][i] += basis[i] * value[1];
        }
    }
    if n < 3 {
        // Pad the system to three unknowns with a trivial equation.
        a[2][2] = 1.0;
    }
    let cx = solve3(a, b[0])?;
    let cy = solve3(a, b[1])?;
    Some(move |s: f64| {
        let basis = [1.0, s, if n == 3 { s * s } else { 0.0 }];
        [cx[0] * basis[0] + cx[1] * basis[1] + cx[2] * basis[2], cy[0] * basis[0] + cy[1] * basis[1] + cy[2] * basis[2]]
    })
}

/// The symbol's format and version bits, read through `corners` the way
/// the sampler reads modules: dark or light against the threshold the
/// fixed modules give, unknown off the photo or within `band` of it. What
/// they say comes back as evidence; bits that cannot be read make the
/// evidence neutral, never wrong.
fn read_info(
    image: &Luma,
    func: &FunctionMap,
    corners: [[f64; 2]; 4],
    inverted: bool,
    threshold: Option<f32>,
    band: f32,
) -> InfoEvidence {
    let Some(map) = Homography::rect_to_quad(func.width as f64, func.height as f64, corners) else {
        return InfoEvidence::default();
    };
    let fixed: Vec<Pos> = (0..func.height)
        .flat_map(|y| (0..func.width).map(move |x| (x, y)))
        .filter(|&(x, y)| func.expected(x, y).is_some())
        .map(|(x, y)| Pos::new(x, y))
        .collect();
    let fixed_luma = sample_positions(image, &map, &fixed, INNER_FRACTION, POINTS);
    let threshold = threshold.unwrap_or_else(|| otsu(fixed_luma.iter().flatten().copied()));
    // The band is a share of the contrast between dark and light modules.
    let (dark_sum, light_sum, dark_n, light_n) =
        fixed_luma.iter().flatten().fold((0.0f32, 0.0f32, 0u32, 0u32), |(d, l, nd, nl), &v| {
            if v < threshold { (d + v, l, nd + 1, nl) } else { (d, l + v, nd, nl + 1) }
        });
    let half = if dark_n > 0 && light_n > 0 {
        (light_sum / light_n as f32 - dark_sum / dark_n as f32) / 2.0
    } else {
        0.25
    };
    let band_abs = band * half.max(1e-3);
    let read = |copy: &InfoCopy| read_copy_through(image, &map, copy, inverted, threshold, band_abs);
    let format = bch::analyze_format_reads(func.version, func.format.iter().map(read).collect());
    let version = (!func.version_info.is_empty())
        .then(|| bch::analyze_version_reads(func.version_info.iter().map(read).collect()));
    // The version named, and how many bits its word was read off by.
    let named = match func.version {
        SymbolVersion::Qr(_) => {
            version.as_ref().filter(|v| v.is_reliable()).map(|v| (SymbolVersion::Qr(v.best_version), v.mismatches))
        }
        // Nothing in a Model 1 symbol names its version: no version
        // information, and a format word that says only level and mask.
        SymbolVersion::Model1(_) => None,
        _ => format.closest().filter(|_| format.verdict() == Some(true)).map(|c| (c.fields.version, c.best_mismatches)),
    };
    InfoEvidence { format: format.summary(), named_version: named.map(|n| n.0), named_mismatches: named.map(|n| n.1) }
}

/// One info copy read through `map`, each bit dark or light against
/// `threshold`, and unknown within `band_abs` of it or off the photo.
fn read_copy_through(
    image: &Luma,
    map: &impl Project,
    copy: &InfoCopy,
    inverted: bool,
    threshold: f32,
    band_abs: f32,
) -> bch::InfoRead {
    let luma = sample_positions(image, map, &copy.bits, INNER_FRACTION, POINTS);
    bch::info_read_from(copy, |i| {
        luma[i].filter(|v| (v - threshold).abs() >= band_abs).map(|v| (v < threshold) != inverted)
    })
}

/// What the symbol's own information bits say when read beside a finder
/// pattern alone, before any version is fitted. zxing-cpp reads the format
/// word of a Micro QR or rMQR symbol through a transform fitted to the
/// finder's own corners and takes the version from it, so that the size
/// of the outline never enters; the version information of a QR symbol
/// sits beside two of its finders and is read the same way.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Prename {
    version: SymbolVersion,
    /// The read lies within one bit of a codeword over enough bits, which
    /// noise manages a few times in a hundred at most. A read merely
    /// within correction range is weaker evidence: more than half of all
    /// fifteen-bit strings lie within three bits of some format word.
    strong: bool,
    /// Bits the read was off its word by.
    mismatches: u32,
    /// The clean read came through the mirror image of the finder's
    /// placement: the symbol is seen from the back.
    mirrored: bool,
}

/// The map from the module coordinates of `func` to the photo fitted to one
/// capstone standing as the finder at `site`, with the symbol turned
/// `turns` quarter turns in the outline and reflected when `mirrored`.
/// Exact at the finder and good for a few modules around it, as far as
/// the information bits beside it.
fn finder_map(capstone: &Capstone, site: &Site, turns: u8, mirrored: bool, func: &FunctionMap) -> Option<Homography> {
    let pairs = capstone_points(capstone, site, turns, mirrored, func.width as f64, func.height as f64);
    let from: [[f64; 2]; 4] = std::array::from_fn(|i| pairs[i].0);
    let to: [[f64; 2]; 4] = std::array::from_fn(|i| pairs[i].1);
    Homography::from_points(from, to)
}

/// A reference version of a family. The information bits beside a finder
/// sit at the same offsets from it in every version of the family, so one
/// version's map serves to read them through any finder.
fn reference_version(family: Family) -> SymbolVersion {
    match family {
        Family::Qr => SymbolVersion::Qr(7),
        Family::Micro => SymbolVersion::Micro(4),
        Family::Rmqr => SymbolVersion::Rmqr(11),
        Family::Model1 => SymbolVersion::Model1(7),
    }
}

/// Which capstones stand as a family's finders, by their places in the
/// outline: the turn and reflection under which the reference version's
/// finder positions match them best, as the fit will assign them, and the
/// capstone standing at each finder. Capstones left out are print beside
/// the symbol or patterns in its data.
fn finder_roles(capstones: &[Capstone], family: Family) -> Option<(u8, bool, Vec<Option<usize>>)> {
    let func = FunctionMap::new(reference_version(family));
    let (w, h) = (func.width as f64, func.height as f64);
    let sites = sites(&func);
    let centres_at = |turns: u8, mirrored: bool| -> Vec<[f64; 2]> {
        sites.iter().filter(|s| s.kind == AnchorKind::Finder).map(|s| to_unit(turns, mirrored, w, h, s.at)).collect()
    };
    let rmqr = family == Family::Rmqr;
    let turn_options: &[u8] = &[0, 1, 2, 3];
    let mirror_options: &[bool] = if rmqr { &[false, true] } else { &[false] };
    let mut best: Option<(f64, u8, bool, Vec<Option<usize>>)> = None;
    for &mirrored in mirror_options {
        for &turns in turn_options {
            let (cost, assignment) = assign(capstones, &centres_at(turns, mirrored));
            if best.as_ref().is_none_or(|b| cost < b.0) {
                best = Some((cost, turns, mirrored, assignment));
            }
        }
    }
    best.map(|(_, turns, mirrored, assignment)| (turns, mirrored, assignment))
}

/// The threshold and the band, in luminance, for reading bits beside a
/// capstone: the capstone's own dark and light levels, unless a threshold
/// is fixed.
fn capstone_levels(image: &Luma, capstone: &Capstone, threshold: Option<f32>, band: f32) -> Option<(f32, f32)> {
    let (own, half) = finder_levels(image, capstone.corners)?;
    Some((threshold.unwrap_or(own), band * half.max(1e-3)))
}

/// One way a lone finder's symbol may sit in the outline, with the version
/// its format word names when read through that placement.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Placement {
    turns: u8,
    mirrored: bool,
    version: SymbolVersion,
    mismatches: u32,
    compared: u32,
}

/// For a symbol with one finder, a Micro QR or rMQR, every placement of
/// `capstone` as that finder under which the format word beside it reads
/// as a word, cleanest first. The word names the version.
fn format_placements(image: &Luma, capstone: &Capstone, family: Family, threshold: Option<f32>, band: f32) -> Vec<Placement> {
    // A capstone read along lines alone has no measured corners to read
    // the word through.
    if capstone.line_only {
        return Vec::new();
    }
    let func = FunctionMap::new(reference_version(family));
    let sites = sites(&func);
    let Some(site) = sites.iter().find(|s| s.kind == AnchorKind::Finder) else { return Vec::new() };
    // The finder-side copy: the only one of a Micro QR, the first of an rMQR.
    let copy = &func.format[0];
    let Some((threshold, band_abs)) = capstone_levels(image, capstone, threshold, band) else { return Vec::new() };
    let mut out = Vec::new();
    for mirrored in [false, true] {
        for turns in 0..4u8 {
            let Some(map) = finder_map(capstone, site, turns, mirrored, &func) else { continue };
            let read = read_copy_through(image, &map, copy, capstone.inverted, threshold, band_abs);
            let analysis = bch::analyze_format_reads(func.version, vec![read]);
            if analysis.verdict() != Some(true) {
                continue;
            }
            let Some(closest) = analysis.closest() else { continue };
            out.push(Placement {
                turns,
                mirrored,
                version: closest.fields.version,
                mismatches: closest.best_mismatches,
                compared: closest.best_compared,
            });
        }
    }
    out.sort_by_key(|p| (p.mismatches, std::cmp::Reverse(p.compared)));
    out
}

/// The version the symbol's own bits name when read beside the finder
/// patterns alone, if any, and how far it can be trusted. For QR Code the
/// version information beside the top-right and bottom-left finders is
/// read, each copy through its own finder; a symbol below version 7 has
/// none, and the finder spacing decides as before. For Micro QR and rMQR
/// the format word beside the one finder names the version, read through
/// the placements that put the finder where the capstone lies.
#[allow(clippy::too_many_arguments)]
fn prename_version(
    image: &Luma,
    capstones: &[Capstone],
    family: Family,
    inverted: bool,
    threshold: Option<f32>,
    band: f32,
    hint: [f64; 2],
    tight: bool,
) -> Option<Prename> {
    let func = FunctionMap::new(reference_version(family));
    let sites = sites(&func);
    let finders: Vec<&Site> = sites.iter().filter(|s| s.kind == AnchorKind::Finder).collect();
    match family {
        // Nothing in a Model 1 symbol names its version.
        Family::Model1 => None,
        Family::Qr => {
            // The finders' roles from their places in the outline, as the
            // fit will assign them.
            let (turns, _, assignment) = finder_roles(capstones, family)?;
            // The copy beside each finder: copy 0 lies by the top-right
            // finder, copy 1 by the bottom-left.
            let mut reads: Vec<(usize, bch::VersionAnalysis)> = Vec::new();
            for (f, site) in finders.iter().enumerate().skip(1) {
                let Some(c) = assignment[f] else { continue };
                let capstone = &capstones[c];
                if capstone.line_only {
                    continue;
                }
                let Some((threshold, band_abs)) = capstone_levels(image, capstone, threshold, band) else { continue };
                let Some(map) = finder_map(capstone, site, turns, false, &func) else { continue };
                let read = read_copy_through(image, &map, &func.version_info[f - 1], inverted, threshold, band_abs);
                reads.push((c, bch::analyze_version_reads(vec![read])));
            }
            // One read within a bit of a word, or two reads beside
            // different finders agreeing within correction range. A symbol
            // below version 7 has data where the version bits would be,
            // and data manages neither often.
            let clean = reads
                .iter()
                .filter(|(_, a)| a.compared >= bch::MIN_INFO_BITS && a.mismatches <= 1 && a.margin > 0)
                .min_by_key(|(_, a)| a.mismatches);
            let agreed = reads.iter().find(|(c, a)| {
                a.is_reliable() && reads.iter().any(|(d, b)| d != c && b.is_reliable() && b.best_version == a.best_version)
            });
            let read = &clean.or(agreed)?.1;
            let version = SymbolVersion::Qr(read.best_version);
            // The finder spacing, where there is one, must roughly agree,
            // as zxing-cpp requires of a version it reads.
            if tight && (version.width() as f64 - hint[0]).abs() > NAMED_SIZE_SLACK {
                return None;
            }
            Some(Prename { version, strong: true, mismatches: read.mismatches, mirrored: false })
        }
        Family::Micro | Family::Rmqr => {
            let site = finders.first()?;
            let turn_options: &[u8] = &[0, 1, 2, 3];
            let mut best: Option<Placement> = None;
            for capstone in capstones.iter().filter(|c| !c.line_only) {
                for placement in format_placements(image, capstone, family, threshold, band) {
                    // Only the placements that put the finder about where
                    // the capstone lies in the outline, as the fit will:
                    // in the version the word names, not the reference
                    // one. The finder lies at the same offset from the
                    // corner in every version, but not at the same share
                    // of the outline: at mid-height of an R7 symbol, where
                    // the plain placement and the one flipped top to
                    // bottom are equally near, and the word must choose.
                    let (w, h) = (placement.version.width() as f64, placement.version.height() as f64);
                    let cost = |turns: u8, mirrored: bool| distance(capstone.unit, to_unit(turns, mirrored, w, h, site.at));
                    let least =
                        turn_options.iter().flat_map(|&t| [cost(t, false), cost(t, true)]).fold(f64::INFINITY, f64::min);
                    if cost(placement.turns, placement.mirrored) > least + PLACEMENT_SLACK {
                        continue;
                    }
                    let key = (placement.mismatches, std::cmp::Reverse(placement.compared));
                    if best.is_none_or(|b| key < (b.mismatches, std::cmp::Reverse(b.compared))) {
                        best = Some(placement);
                    }
                }
            }
            best.map(|p| Prename { version: p.version, strong: p.mismatches <= 1, mismatches: p.mismatches, mirrored: p.mirrored })
        }
    }
}

/// Symbols found anywhere in the photo, with no outline drawn, best first.
///
/// The whole frame is scanned for finder patterns. For QR Code, every
/// three capstones of alike module size whose centres make near enough a
/// right isosceles triangle, as ZXing's `selectBestPatterns` judges them
/// and perspective allows, propose an outline around themselves, when the two along each side
/// face each other as finders do and no finder lies inside the symbol
/// they would make, which three finders of a sheet of symbols as readily
/// do. A Micro QR or rMQR finder stands
/// alone, so the format word beside it is read to learn which way the
/// symbol extends and how far, and a box is proposed from that, or four
/// boxes when the word cannot be read. Each proposal is then detected as
/// a hand-drawn outline would be, and the capstones inside a symbol found
/// are not offered again, as ZXing and ZBar mark the finders a symbol has
/// used.
///
/// What is detected is offered only when more than its finders vouches
/// for it: its fixed patterns fit, and its timing patterns read as well,
/// or a word of its own information bits does outright, since the
/// finders are what proposed it and say nothing more by fitting.
pub fn find_symbols(image: &Luma, family: Family, params: &SampleParams) -> Vec<Detection> {
    find_symbols_with(image, family, params, &SearchParams::default())
}

/// How a search of the whole photo goes about it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchParams {
    /// Search a copy of the photo shrunk to [`COARSE_SIDE`] pixels on its
    /// longer side first, and find each symbol the copy shows again at
    /// full size, through the outline the copy gave; search the photo at
    /// full size only when the copy shows nothing, or shows the finders
    /// of a symbol it could not fit. BoofCV's detector
    /// costs little because its contour pass is cheap and its grey
    /// refinement touches only the shapes that pass it; here the search
    /// at full size fits the whole placement machinery to every outline
    /// the capstones propose, and a twelve-megapixel photo took a hundred
    /// seconds where its quarter-size copy took one and found the same
    /// symbol. A symbol a few pixels a module in a large photo vanishes
    /// in the copy, and is found at full size as before.
    pub coarse_to_fine: bool,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self { coarse_to_fine: true }
    }
}

/// Longest side, in pixels, of the first copy the coarse search looks at.
pub const COARSE_SIDE: f64 = 1024.0;
/// Least module size, in pixels of a shrunk copy, for the symbols found
/// in it to be taken as all the copy can show: smaller, and symbols the
/// scan could not make out at that size may be hiding beside them, so
/// the next larger copy is searched as well. The scan makes out a finder
/// pattern down to about two pixels a module, and a symbol found with
/// less was found by luck.
pub const COARSE_MIN_MODULE: f64 = 1.5;

/// [`find_symbols`] as `search` says to go about it.
pub fn find_symbols_with(image: &Luma, family: Family, params: &SampleParams, search: &SearchParams) -> Vec<Detection> {
    find_symbols_of_families(image, &[family], params, search, &|_| true)
}

/// Symbols of every family in `families` found anywhere in the photo,
/// in the families' order, as ZXing's MultiFormatReader and zxing-cpp's
/// ReadBarcodes try every symbology in one pass. Every finder pattern of
/// a QR Code passes for the finder of a Micro QR or rMQR symbol of its
/// own, so the finders of each QR Code found that `credible` vouches for
/// are withheld from the families searched after it, which leave them
/// alone: a search of each family knowing nothing of the others proposes
/// a Micro QR and an rMQR symbol from every finder of a QR Code and
/// spends on fitting them most of the time the whole search takes.
///
/// The families search each copy of the photo together before the next
/// copy is made, the copy scanned for finder patterns once for all of
/// them, and the search ends at the first copy that shows a symbol of
/// any family and no finders of a symbol it could not fit: the renders
/// read off their pixels, then the shrunk copies when `search` asks for
/// them, then the photo as it is and the ladder of transformed copies,
/// each copy leaving alone the symbols the copies before it found. A
/// search of each family on its own scanned every copy once a family
/// and, finding nothing of one family, climbed the whole ladder for it:
/// a photo of a QR Code took four seconds to search for QR Codes and a
/// hundred and twelve more to search for Micro QR symbols, on the same
/// pixels.
///
/// A photo longer than [`SEARCH_SIDE`] on a side is searched in a copy
/// reduced to that, and each symbol the copy shows is found again in the
/// photo's own pixels through its outline, when the search would vouch
/// for the placement there, or else stands as the copy placed it,
/// scaled up: either way it comes back in the photo's own pixels, where
/// whatever is done with it next, a refinement or a read, is done.
pub fn find_symbols_of_families(
    image: &Luma,
    families: &[Family],
    params: &SampleParams,
    search: &SearchParams,
    credible: &(dyn Fn(&Detection) -> bool + Sync),
) -> Vec<Detection> {
    let Some(copy) = reduced_for_search(image) else {
        return search_whole(image, families, params, search, credible);
    };
    let (across, down) = (image.width as f64 / copy.width as f64, image.height as f64 / copy.height as f64);
    let found = search_whole(&copy, families, params, search, credible);
    work::map(&found, |found| {
        let mut found = found.clone();
        found.corners = found.corners.map(|[x, y]| [x * across, y * down]);
        for anchor in &mut found.anchors {
            anchor.center = [anchor.center[0] * across, anchor.center[1] * down];
        }
        if work::cancelled() { found } else { refine_found(image, found, params) }
    })
}

/// The copy of a photo that a search of the whole of it looks at, when
/// that is not the photo itself: the photo reduced, smoothly, so that
/// its longer side is [`SEARCH_SIDE`]. `None` for a photo no larger.
fn reduced_for_search(image: &Luma) -> Option<Luma> {
    let longest = image.width.max(image.height);
    if longest <= SEARCH_SIDE {
        return None;
    }
    let scale = SEARCH_SIDE as f32 / longest as f32;
    let (width, height) = (((image.width as f32 * scale) as usize).max(1), ((image.height as f32 * scale) as usize).max(1));
    Some(image.resized(width, height))
}

/// [`find_symbols_of_families`] of a photo, or a copy of one, that is no
/// larger than the search looks at.
fn search_whole(
    image: &Luma,
    families: &[Family],
    params: &SampleParams,
    search: &SearchParams,
    credible: &(dyn Fn(&Detection) -> bool + Sync),
) -> Vec<Detection> {
    let whole = [[0.0, 0.0], [image.width as f64, 0.0], [image.width as f64, image.height as f64], [0.0, image.height as f64]];
    let mut all: Vec<Detection> = Vec::new();
    let mut taken: Vec<Detection> = Vec::new();
    // A screenshot or a file an encoder wrote holds one symbol, pixel for
    // pixel, and is read off its pixels; the families it does not read
    // as search the photo.
    let mut pending: Vec<Family> = Vec::new();
    for &family in families {
        match detect_pure(image, family, whole, params) {
            Some(found) => note(&mut all, &mut taken, family, vec![found], credible),
            None => pending.push(family),
        }
    }
    if pending.is_empty() {
        return all;
    }
    let mut known: Vec<Detection> = Vec::new();
    if search.coarse_to_fine {
        let coarse = search_coarse(image, &pending, params, &mut taken, credible);
        // Only when no family shows anywhere, in a render or a shrunk
        // copy, is the photo searched at full size for every family; and
        // when the last copy shows finders of a symbol it could not fit,
        // for the families whose finders they are.
        if !coarse.detections.is_empty() || !all.is_empty() {
            if coarse.more.is_empty() {
                all.extend(coarse.detections);
                return all;
            }
            pending = coarse.more;
        }
        known = coarse.detections;
    }
    let fine = search_fine(image, &pending, params, &mut taken, &known, credible);
    all.extend(known);
    all.extend(fine);
    all
}

/// Add what one family's search of a copy found to `all`, and its
/// credible QR Codes to `taken`.
fn note(all: &mut Vec<Detection>, taken: &mut Vec<Detection>, family: Family, found: Vec<Detection>, credible: &(dyn Fn(&Detection) -> bool + Sync)) {
    if family == Family::Qr {
        taken.extend(found.iter().filter(|d| credible(d)).cloned());
    }
    all.extend(found);
}

/// Every family's search of one copy of the photo, in the families'
/// order, each handed the credible QR Codes found before it and the
/// symbols `known` from the copies searched before this one.
fn search_view(
    view: &ScannedView,
    families: &[Family],
    params: &SampleParams,
    taken: &mut Vec<Detection>,
    known: &[Detection],
    credible: &(dyn Fn(&Detection) -> bool + Sync),
) -> Searched {
    let mut all = Searched::default();
    // The finders of the QR Codes found are withheld from the families
    // after, so QR Code is searched for on its own. The families that
    // follow it take nothing from one another, and are searched for side
    // by side.
    let mut rest = families;
    while let Some((&first, after)) = rest.split_first() {
        let together = if first == Family::Qr { 1 } else { 1 + after.iter().take_while(|&&family| family != Family::Qr).count() };
        let (batch, later) = rest.split_at(together);
        let withheld: &[Detection] = taken;
        let found = work::map(batch, |&family| view.find(family, params, withheld, known));
        for (&family, found) in batch.iter().zip(found) {
            if found.more {
                all.more.push(family);
            }
            note(&mut all.detections, taken, family, found.detections, credible);
        }
        rest = later;
    }
    all
}

/// The shrink factors of the copies the coarse search looks at, in order:
/// to [`COARSE_SIDE`] on the longer side, then half as much each time,
/// down to half size. None for a photo too small to shrink.
fn coarse_factors(image: &Luma) -> Vec<u8> {
    let longest = image.width.max(image.height) as f64;
    let mut factor = (longest / COARSE_SIDE).ceil().clamp(1.0, 255.0) as u8;
    let mut out = Vec::new();
    while factor >= 2 {
        out.push(factor);
        factor /= 2;
    }
    out
}

/// The coarse search of `families`: the photo searched in copies shrunk
/// by [`coarse_factors`], each symbol a copy shows found again at full
/// size through its own outline, and left alone by the copies after it.
/// The first copy whose symbols are all [`COARSE_MIN_MODULE`] a module or
/// more, and which shows no finders of a symbol it could not fit, settles
/// it; once a copy has shown a symbol, the copies after it are searched
/// only for the families that have such finders left. No detections when
/// no copy shows a symbol, or the photo is too small to shrink; `more`
/// is the last copy's, the families to search the photo at full size
/// for.
fn search_coarse(
    image: &Luma,
    families: &[Family],
    params: &SampleParams,
    taken: &mut Vec<Detection>,
    credible: &(dyn Fn(&Detection) -> bool + Sync),
) -> Searched {
    let mut all = Searched::default();
    let mut wanted: Vec<Family> = families.to_vec();
    for factor in coarse_factors(image) {
        if work::cancelled() {
            break;
        }
        let Some(view) = ScannedView::through(image, Retry::Downscaled { factor }, params, &wanted) else { continue };
        let found = search_view(&view, &wanted, params, taken, &all.detections, credible);
        // The modules in the copy's own pixels.
        let smallest = found.detections.iter().map(|d| module_size_of(d) / view.scale).fold(f64::INFINITY, f64::min);
        all.detections.extend(found.detections.into_iter().map(|found| refine_found(image, found, params)));
        all.more = found.more;
        if all.detections.is_empty() {
            continue;
        }
        if smallest >= COARSE_MIN_MODULE && all.more.is_empty() {
            break;
        }
        // Symbols too small for the copy to show well may have others
        // beside them it did not show at all, of any family; finders left
        // over are their own family's business.
        if smallest >= COARSE_MIN_MODULE {
            wanted = all.more.clone();
        }
    }
    all
}

/// The fine search of `families`: the photo as it is, then the ladder of
/// transformed copies, cheapest step first, until a copy shows a symbol
/// and no finders of another it could not fit. The symbols `known`
/// already, from the shrunk copies, are left alone, and so are those each
/// copy finds by the copies after it: what comes back is what is new.
/// Once any symbol is known, a copy is searched only for the families
/// with such finders left: the others have had the look they always had.
///
/// The search ended at the first copy that showed any symbol, so that of
/// two symbols in a photo, one in shadow, the copy that showed the one
/// was the last looked at, and the rung that would have shown the other
/// was never climbed to. OpenCV's ArUco-based detector thresholds at
/// every window size before it groups anything, and pools the lot.
fn search_fine(
    image: &Luma,
    families: &[Family],
    params: &SampleParams,
    taken: &mut Vec<Detection>,
    known: &[Detection],
    credible: &(dyn Fn(&Detection) -> bool + Sync),
) -> Vec<Detection> {
    let mut all: Vec<Detection> = known.to_vec();
    let mut wanted: Vec<Family> = families.to_vec();
    let steps: Vec<Option<Retry>> = std::iter::once(None).chain(Retry::LADDER.iter().copied().map(Some)).collect();
    let view_of = |step: Option<Retry>, wanted: &[Family]| match step {
        None => ScannedView::of(image, params, wanted),
        Some(step) => ScannedView::through(image, step, params, wanted),
    };
    let mut climbed = 0;
    if all.is_empty() {
        // While no symbol is known, a rung that shows none leaves the
        // next the same search to make, of the same families with the
        // same finders withheld. So the rungs after the first, which on
        // a sound photo shows the symbol, are climbed side by side and
        // taken in their order up to the first that shows one: a photo
        // of a symbol too damaged to find has every rung climbed, and
        // climbed one after another that took minutes.
        // A large photo has fewer of its rungs at once: each is a copy
        // of it.
        let withheld: &[Detection] = taken;
        let mut shown = None;
        work::first_then_in_turn_by(
            image.copies_at_once(),
            &steps,
            |&step| {
                let mut withheld = withheld.to_vec();
                let found = match (!work::cancelled()).then(|| view_of(step, &wanted)).flatten() {
                    Some(view) => search_view(&view, &wanted, params, &mut withheld, &[], credible),
                    None => Searched::default(),
                };
                (found, withheld)
            },
            |(found, withheld)| {
                climbed += 1;
                let any = !found.detections.is_empty();
                if any {
                    shown = Some((found, withheld));
                }
                any
            },
        );
        let Some((found, withheld)) = shown else { return Vec::new() };
        *taken = withheld;
        all.extend(found.detections);
        if found.more.is_empty() {
            return all;
        }
        wanted = found.more;
    }
    for &step in &steps[climbed..] {
        if work::cancelled() {
            break;
        }
        let Some(view) = view_of(step, &wanted) else { continue };
        let found = search_view(&view, &wanted, params, taken, &all, credible);
        all.extend(found.detections);
        if all.is_empty() {
            continue;
        }
        if found.more.is_empty() {
            break;
        }
        wanted = found.more;
    }
    all.split_off(known.len())
}

/// The coarse half of [`find_symbols_with`] for one family: a render read
/// off its pixels, or else the shrunk copies searched as
/// [`find_symbols_of_families`] searches them. `None` when no copy shows a
/// symbol, or the photo is too small to shrink. `taken` are symbols
/// already found in the photo, of other families: their finder patterns
/// are theirs, and propose nothing here.
pub fn find_symbols_coarse(image: &Luma, family: Family, params: &SampleParams, taken: &[Detection]) -> Option<Vec<Detection>> {
    let whole = [[0.0, 0.0], [image.width as f64, 0.0], [image.width as f64, image.height as f64], [0.0, image.height as f64]];
    if let Some(found) = detect_pure(image, family, whole, params) {
        return Some(vec![found]);
    }
    let found = search_coarse(image, &[family], params, &mut taken.to_vec(), &|_| true).detections;
    (!found.is_empty()).then_some(found)
}

/// The fine half of [`find_symbols_with`] for one family, and the whole
/// of it without the coarse search: a render read off its pixels, or else
/// the photo searched at full size, and then through the ladder of
/// transformed copies when it shows nothing. `taken` as for
/// [`find_symbols_coarse`].
pub fn find_symbols_fine(image: &Luma, family: Family, params: &SampleParams, taken: &[Detection]) -> Vec<Detection> {
    let whole = [[0.0, 0.0], [image.width as f64, 0.0], [image.width as f64, image.height as f64], [0.0, image.height as f64]];
    if let Some(found) = detect_pure(image, family, whole, params) {
        return vec![found];
    }
    search_fine(image, &[family], params, &mut taken.to_vec(), &[], &|_| true)
}

/// The size of a symbol's modules in the pixels its corners are in,
/// along the top and bottom edges.
fn module_size_of(found: &Detection) -> f64 {
    let c = found.corners;
    (distance(c[0], c[1]) + distance(c[3], c[2])) / 2.0 / found.version.width().max(1) as f64
}

/// A symbol found in a shrunk copy of the photo, found again at full
/// size through the outline the copy gave, when the search would vouch
/// for that placement; otherwise as the copy placed it, scaled up, which
/// the refinement of its corners and anchors takes the rest of the way.
fn refine_found(image: &Luma, coarse: Detection, params: &SampleParams) -> Detection {
    // A mirrored symbol's corners run anticlockwise round the photo. An
    // outline runs clockwise, and one given the other way round would
    // have the search look at the photo as from the back, find the symbol
    // unmirrored there, and report it so.
    // The outline the copy gave sets the threshold's reach at full size,
    // as any outline does: the coarse search runs no rung of the ladder.
    match detect_candidates_in(image, coarse.version, clockwise(coarse.corners), params, false, None, MAX_RECTIFIED).into_iter().next() {
        Some(fine) if vouched(image, &fine, params) => fine,
        _ => coarse,
    }
}

/// Whether a search of the whole photo offers a detection: its fixed
/// patterns fit, with a little less asked of the fit when the symbol's
/// own bits name the version fitted, and something of the symbol besides
/// its finders reads.
///
/// The fit alone says little of a small symbol. Sixty-four of the
/// eighty-two fixed modules of a Micro QR M4 symbol are its finder and
/// separator, and all but eleven of the two hundred and three of a
/// version 1 QR Code are its finders and their separators, so that
/// whatever the scan took for finders fits four parts in five or better
/// with nothing else there: every finder of a QR Code was offered as a
/// Micro QR symbol hanging off it, and three finder-like marks at the
/// right spacing as a QR Code. So either the symbol's own bits read as
/// a word outright, the word that names its version or its format word,
/// or the patterns beyond the finders read. zxing-cpp asks as much of a
/// Micro QR finder before it samples anything, the timing modules beside
/// it and paper past the symbol's far edges, to "check that we are in
/// fact not looking at a corner of a non-micro QRCode symbol", and of an
/// rMQR finder the timing pattern along every edge.
fn vouched(image: &Luma, found: &Detection, params: &SampleParams) -> bool {
    let named = found.info.named_version == Some(found.version);
    found.score() >= if named { MIN_NAMED_SCORE } else { MIN_FOUND_SCORE }
        && (found.version_confirmed() || found.format_reads_outright() || reads_beyond_finders(image, found, params))
}

/// Whether the patterns of a symbol that are none of its finders read as
/// a detection places them, sampled as the symbol would be:
/// [`MIN_BEYOND_FINDERS`] of their modules the right colour, a module too
/// faint to call counting against, as blank paper reads. Both timing
/// patterns of a Micro QR symbol are asked, each on its own: hung off a
/// finder of a QR Code and turned to lie along that symbol's edge, one
/// of the two lies on the QR Code's own timing pattern and reads without
/// a fault. An rMQR symbol's timing patterns are taken together, and so
/// are a QR Code's, whose alignment patterns will do in their place: a
/// scratch along a timing pattern takes all of it, and the three finders
/// of a QR Code already say more than the one of the other families.
fn reads_beyond_finders(image: &Luma, found: &Detection, params: &SampleParams) -> bool {
    let func = FunctionMap::new(found.version);
    let Some(grid) = grid_as_found(image, &func, found, params) else { return false };
    let family = found.version.family();
    // (matching, all) of the timing modules, a Micro QR symbol's second
    // pattern apart from its first, and of the alignment patterns.
    let mut tally = [(0usize, 0usize); 3];
    for y in 0..func.height {
        for x in 0..func.width {
            let slot = match func.role(x, y) {
                Role::Timing if family == Family::Micro && y != 0 => 1,
                Role::Timing => 0,
                Role::Alignment => 2,
                _ => continue,
            };
            let Some(want) = func.expected(x, y) else { continue };
            tally[slot].1 += 1;
            if grid.get(x, y).as_dark() == Some(want) {
                tally[slot].0 += 1;
            }
        }
    }
    let reads = |(matching, all): (usize, usize)| all > 0 && matching as f64 >= MIN_BEYOND_FINDERS * all as f64;
    match family {
        Family::Micro => reads(tally[0]) && reads(tally[1]),
        Family::Rmqr => reads(tally[0]),
        Family::Qr | Family::Model1 => reads(tally[0]) || reads(tally[2]),
    }
}

/// One copy of the photo, as it is or transformed by a rung of the
/// ladder, scanned once for the finder patterns every family's search
/// works from. The scan of the rows and columns, the labelling of the
/// regions and the fitting of a template to each candidate are the same
/// whatever the family, and a search of each family on its own repeated
/// them on the same pixels: here each candidate is refined once, when a
/// family first asks for it, and each family takes from the candidates
/// those within its own cap on the module size, the copy having been
/// scanned with the widest cap of the families that search it.
struct ScannedView<'a> {
    /// The copy's pixels: the photo's own, or those of a transformed copy.
    luma: Cow<'a, Luma>,
    /// The photo pixel at the copy's origin, and photo pixels per pixel
    /// of the copy.
    origin: [f64; 2],
    scale: f64,
    /// The rung of the ladder the copy was made by; `None` for the photo
    /// as it is.
    step: Option<Retry>,
    /// The reach of the scan's local threshold, where the rung narrows it.
    local_radius: Option<f64>,
    rect: Rectified,
    threshold: Option<f32>,
    /// The scan's candidates, strongest first, and the regions', best first.
    candidates: Vec<Candidate>,
    regions: Vec<Candidate>,
    /// Each candidate's refinement once made: the scan's, then the
    /// regions'. The families' searches of the copy run side by side and
    /// ask for the same candidates.
    refined: [Vec<OnceLock<Option<Capstone>>>; 2],
}

impl<'a> ScannedView<'a> {
    /// The photo as it is, scanned for `families`.
    fn of(image: &'a Luma, params: &SampleParams, families: &[Family]) -> Option<Self> {
        Self::new(Cow::Borrowed(image), [0.0, 0.0], 1.0, None, None, params, families)
    }

    /// The photo transformed by `step`, scanned for `families`; `None`
    /// when the copy cannot be made.
    fn through(image: &'a Luma, step: Retry, params: &SampleParams, families: &[Family]) -> Option<Self> {
        let view = View::whole(image, step)?;
        Self::new(Cow::Owned(view.luma), view.origin, view.scale, Some(step), view.local_radius, params, families)
    }

    fn new(
        luma: Cow<'a, Luma>,
        origin: [f64; 2],
        scale: f64,
        step: Option<Retry>,
        local_radius: Option<f64>,
        params: &SampleParams,
        families: &[Family],
    ) -> Option<Self> {
        let (w, h) = (luma.width as f64, luma.height as f64);
        let region = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
        let rect = Rectified::with_longest(&luma, region, MAX_RECTIFIED_WHOLE)?.narrowed(local_radius);
        let threshold = fixed_threshold(params);
        let widest = families.iter().map(|&f| max_module_fraction(f)).fold(0.0, f64::max);
        let max_module = rect.frame.size[0].min(rect.frame.size[1]) * widest;
        let candidates = scan_for_finders(&rect, threshold, max_module);
        let regions = region_candidates(&rect, threshold, max_module, &candidates);
        let refined = [vec![OnceLock::new(); candidates.len()], vec![OnceLock::new(); regions.len()]];
        Some(Self { luma, origin, scale, step, local_radius, rect, threshold, candidates, regions, refined })
    }

    fn to_photo(&self, p: [f64; 2]) -> [f64; 2] {
        [self.origin[0] + p[0] * self.scale, self.origin[1] + p[1] * self.scale]
    }

    /// A candidate's capstone, refined when first asked for: `source` 0
    /// is the scan's candidates, 1 the regions'.
    fn refined(&self, source: usize, i: usize) -> Option<Capstone> {
        *self.refined[source][i].get_or_init(|| {
            let candidate = if source == 0 { &self.candidates[i] } else { &self.regions[i] };
            refine_capstone(&self.luma, &self.rect, candidate, self.threshold)
        })
    }

    /// The finder patterns of `family` verified against the copy, as
    /// [`collect_capstones`] gathers them: the scan's whole candidates
    /// first, then its half ones, then the regions', each within the
    /// family's cap on the module size, with repeats merged.
    fn capstones(&self, family: Family) -> Vec<Capstone> {
        let max_module = self.rect.frame.size[0].min(self.rect.frame.size[1]) * max_module_fraction(family);
        let within = |c: &Candidate| c.module.iter().all(|&m| m <= max_module);
        let scan = |half: bool, most: usize| {
            (0..self.candidates.len()).filter(move |&i| self.candidates[i].half == half && within(&self.candidates[i])).take(most).map(|i| (0, i))
        };
        let regions = (0..self.regions.len()).filter(|&i| within(&self.regions[i])).take(MAX_REGION_CANDIDATES).map(|i| (1, i));
        let mut all: Vec<Capstone> = Vec::new();
        for (source, i) in scan(false, MAX_SEARCH_CANDIDATES).chain(scan(true, MAX_HALF_CANDIDATES)).chain(regions) {
            let Some(capstone) = self.refined(source, i) else { continue };
            merge_capstone(&mut all, &self.rect, capstone);
        }
        drop_faint_partials(&mut all);
        all
    }

    /// The symbols of `family` in the copy, in the photo's pixels, with
    /// the rung the copy was made by noted on each. The copy's capstones,
    /// less those of the symbols `taken` and of the symbols `known`,
    /// which earlier copies found, propose outlines, and each proposal
    /// is detected in turn as a hand-drawn outline would be, the
    /// detections the search vouches for kept, as [`find_symbols`]
    /// describes.
    ///
    /// Three capstones propose a QR Code first, best triple first, and a
    /// proposal that stands on a finder of a symbol found since, or lies
    /// over one, is passed by: on a sheet of symbols most triples are
    /// three finders of three neighbours, and finding the neighbours
    /// disposes of them. Then the capstones no symbol was found on
    /// propose in pairs, as they do inside an outline, for the symbol
    /// with a finder torn off, cut by the edge of the photo, or missed:
    /// two along a side with the timing pattern between them, or two
    /// across the diagonal that lie turned as a symbol's would. OpenCV's
    /// detectors and BoofCV's want all three finders; the pairs are the
    /// outline search's own, and without them a symbol that read in a
    /// loose outline was not found in the photo around it.
    fn find(&self, family: Family, params: &SampleParams, taken: &[Detection], known: &[Detection]) -> Found {
        let image: &Luma = &self.luma;
        let threshold = self.threshold;
        let mut capstones = self.capstones(family);
        // The finders of a symbol already found are not offered again, as
        // those of a symbol found here are not below. A symbol with one
        // finder may have been claimed off a finder of a QR Code no copy
        // has yet shown whole, so a search for QR Codes leaves alone only
        // the finders of the QR Codes known.
        let theirs = |d: &&Detection| !family.has_three_finders() || d.version.family().has_three_finders();
        capstones.retain(|c| {
            let at = self.to_photo(c.center);
            !taken.iter().chain(known.iter().filter(theirs)).any(|d| covers(d, at))
        });
        /// An outline to detect, the capstones it stands on, and whether
        /// a timing pattern reads between them.
        struct Proposal {
            badness: f64,
            outline: [[f64; 2]; 4],
            members: Vec<usize>,
            timed: bool,
        }
        let mut proposals: Vec<Proposal> = match family {
            Family::Qr | Family::Model1 => qr_triples(image, &self.rect.frame, &capstones)
                .into_iter()
                .map(|t| Proposal { badness: t.badness, outline: t.outline, members: t.members.to_vec(), timed: t.timed > 0 })
                .collect(),
            Family::Micro | Family::Rmqr => capstones
                .iter()
                .enumerate()
                .flat_map(|(i, c)| {
                    single_finder_proposals(image, c, family, threshold, params.band)
                        .into_iter()
                        .map(move |(badness, outline)| Proposal { badness, outline, members: vec![i], timed: false })
                })
                .collect(),
        };
        proposals.sort_by(|a, b| a.badness.total_cmp(&b.badness));
        let start = match family {
            Family::Qr => SymbolVersion::Qr(2),
            Family::Micro => SymbolVersion::Micro(2),
            Family::Rmqr => SymbolVersion::Rmqr(11),
            Family::Model1 => SymbolVersion::Model1(2),
        };
        // Whether a proposal is still worth detecting: no symbol found
        // lies where it does, or stands on a capstone it stands on.
        let stands = |found: &[Detection], p: &Proposal| {
            !found.iter().any(|d| covers(d, quad_centre(p.outline)) || p.members.iter().any(|&m| covers(d, capstones[m].center)))
        };
        // The proposal is already square to the symbol: no other frame
        // is asked for, and the ladder is the caller's. The placement the
        // fit ranks best is the one judged. Its runners-up were judged
        // too, for a while, and a Micro QR symbol of version M2, hung
        // off a finder of a QR Code, was vouched for by the few modules
        // of its timing patterns reading right by chance where the best
        // fit, of another version, was not.
        let fit = |p: &Proposal| {
            detect_candidates_in(image, start, p.outline, params, false, self.local_radius, MAX_RECTIFIED)
                .into_iter()
                .next()
                .filter(|d| vouched(image, d, params))
        };
        // The proposals in their turn, until `most` have come to nothing,
        // each detected unless a symbol found by then lies where it does.
        // The fit is nearly all the time a search takes, and says the
        // same whatever was found before it. The first is made alone: on
        // a sound photo the best proposal is the symbol, and disposes of
        // the rest. Once one has come to nothing, the proposals next in
        // line are fitted side by side on the cores to spare, and still
        // taken one at a time: a fit made for a proposal that a symbol
        // found meanwhile disposes of is thrown away, as on one core it
        // would never have been made.
        let detect_each = |proposals: &[Proposal], most: usize, found: &mut Vec<Detection>, unfitted: &mut Vec<Vec<usize>>| {
            let mut fits: Vec<Option<Option<Detection>>> = vec![None; proposals.len()];
            let mut failed = 0;
            for (i, p) in proposals.iter().enumerate() {
                if failed >= most || work::cancelled() {
                    break;
                }
                if !stands(found.as_slice(), p) {
                    continue;
                }
                if fits[i].is_none() {
                    let crew = work::Crew::gather(if failed == 0 { 0 } else { FITS_AT_ONCE.min(most - failed) - 1 });
                    let ahead: Vec<usize> = (i..proposals.len())
                        .filter(|&j| fits[j].is_none() && stands(found.as_slice(), &proposals[j]))
                        .take(crew.width())
                        .collect();
                    let fitted = crew.map(&ahead, |&j| if work::cancelled() { None } else { fit(&proposals[j]) });
                    for (&j, fitted) in ahead.iter().zip(fitted) {
                        fits[j] = Some(fitted);
                    }
                }
                let detection = fits[i].take().flatten().filter(|d| !found.iter().any(|f| covers(f, quad_centre(d.corners))));
                match detection {
                    Some(detection) => found.push(detection),
                    None => {
                        failed += 1;
                        if p.timed {
                            unfitted.push(p.members.clone());
                        }
                    }
                }
            }
        };
        let mut found: Vec<Detection> = Vec::new();
        // The capstones of proposals with a timing pattern between them
        // that came to nothing.
        let mut unfitted: Vec<Vec<usize>> = Vec::new();
        detect_each(&proposals, MAX_PROPOSALS, &mut found, &mut unfitted);
        if family.has_three_finders() {
            // The capstones no symbol was found on, in pairs. A partial
            // capstone's centre is placed by three quarters of it, and a
            // pair's outline hangs on both centres.
            let left: Vec<usize> = (0..capstones.len())
                .filter(|&i| !capstones[i].partial && capstones[i].measured && !found.iter().any(|d| covers(d, capstones[i].center)))
                .collect();
            let spare: Vec<Capstone> = left.iter().map(|&i| capstones[i]).collect();
            let mut pairs: Vec<Proposal> = qr_pairs(image, &self.rect.frame, &spare)
                .into_iter()
                .filter(|p| match p.kind {
                    PairKind::Side { timed } => timed,
                    PairKind::Diagonal => finders_lie_diagonal(&spare[p.members[0]], &spare[p.members[1]]),
                })
                .map(|p| Proposal {
                    badness: p.badness,
                    outline: p.outline,
                    members: p.members.iter().map(|&m| left[m]).collect(),
                    timed: matches!(p.kind, PairKind::Side { .. }),
                })
                .collect();
            pairs.sort_by(|a, b| a.badness.total_cmp(&b.badness));
            detect_each(&pairs, MAX_PAIR_PROPOSALS, &mut found, &mut unfitted);
        }
        // Capstones with a timing pattern between them and no symbol
        // found on any of them: a symbol this copy shows and could not
        // fit, which another copy may.
        let more = unfitted.iter().any(|members| members.iter().all(|&m| !found.iter().any(|d| covers(d, capstones[m].center))));
        found.sort_by(|a, b| {
            let named = |d: &Detection| d.info.named_version == Some(d.version);
            named(b).cmp(&named(a)).then(b.score().total_cmp(&a.score()))
        });
        for detection in &mut found {
            detection.corners = detection.corners.map(|c| self.to_photo(c));
            for anchor in &mut detection.anchors {
                anchor.center = self.to_photo(anchor.center);
            }
            if self.step.is_some() {
                detection.retry = self.step;
            }
        }
        Found { detections: found, more }
    }
}

/// What one family's search of one copy of the photo came to.
struct Found {
    /// The symbols found, in the photo's pixels.
    detections: Vec<Detection>,
    /// The copy shows finders that stand as a symbol's, with a timing
    /// pattern between them, on which no symbol was fitted: there is
    /// more in the photo than was found, which another copy may show
    /// better.
    more: bool,
}

/// What every family's search of one copy of the photo came to.
#[derive(Default)]
struct Searched {
    /// The symbols found, in the photo's pixels, in the families' order.
    detections: Vec<Detection>,
    /// The families whose search left [more](Found::more) to find.
    more: Vec<Family>,
}

fn quad_centre(quad: [[f64; 2]; 4]) -> [f64; 2] {
    [quad.iter().map(|c| c[0]).sum::<f64>() / 4.0, quad.iter().map(|c| c[1]).sum::<f64>() / 4.0]
}

/// Whether a photo point lies on, or just around, a symbol found.
fn covers(found: &Detection, point: [f64; 2]) -> bool {
    let (w, h) = (found.version.width() as f64, found.version.height() as f64);
    let Some(inverse) = Homography::rect_to_quad(w, h, found.corners).and_then(|m| m.inverse()) else { return false };
    let [x, y] = inverse.apply(point[0], point[1]);
    (-2.0..=w + 2.0).contains(&x) && (-2.0..=h + 2.0).contains(&y)
}

/// An outline three capstones propose for a QR symbol.
#[derive(Clone, Copy, Debug)]
struct TripleProposal {
    /// How far the three depart from the shape of a symbol's finders.
    badness: f64,
    outline: [[f64; 2]; 4],
    /// The three, as indices into the capstones they were taken from.
    members: [usize; 3],
    /// Sides from the top-left finder along which a timing pattern reads.
    timed: usize,
}

/// Outline proposals for QR symbols from triples of capstones: module sizes
/// alike, centres a right isosceles triangle, spacing one a version could
/// have. Each comes with how far it departs from that shape.
fn qr_proposals(image: &Luma, frame: &Frame, capstones: &[Capstone]) -> Vec<(f64, [[f64; 2]; 4])> {
    qr_triples(image, frame, capstones).into_iter().map(|t| (t.badness, t.outline)).collect()
}

/// [`qr_proposals`] with the capstones each proposal stands on.
fn qr_triples(image: &Luma, frame: &Frame, capstones: &[Capstone]) -> Vec<TripleProposal> {
    let mut out = Vec::new();
    let n = capstones.len();
    let strongest = strongest_contrast(capstones);
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                let trio = [&capstones[i], &capstones[j], &capstones[k]];
                if trio[1].inverted != trio[0].inverted || trio[2].inverted != trio[0].inverted {
                    continue;
                }
                let weakest = trio.iter().map(|c| c.contrast).fold(f64::INFINITY, f64::min);
                let faintness = CONTRAST_WEIGHT * (1.0 - weakest / strongest).max(0.0);
                let sizes = trio.map(Capstone::module_mean);
                let lo = sizes.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = sizes.iter().copied().fold(0.0, f64::max);
                if hi > SIZE_RATIO * lo {
                    continue;
                }
                // Squared side lengths in rectified pixels, each with the
                // vertex it lies opposite.
                let p = trio.map(|c| frame.to_rectified(c.center));
                let d2 = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
                let mut sides = [(d2(p[1], p[2]), 0usize), (d2(p[0], p[2]), 1), (d2(p[0], p[1]), 2)];
                sides.sort_by(|a, b| a.0.total_cmp(&b.0));
                let [(a, _), (b, _), (c, tl)] = sides;
                if a <= 0.0 || c <= 0.0 {
                    continue;
                }
                // The legs from the top-left finder, which lies opposite
                // the longest side: alike in length and near square to
                // each other, within what perspective does to them. How
                // far they are from a right isosceles pair ranks the
                // proposal below, as ZXing ranks by it.
                if (b / a).sqrt() > MAX_LEG_RATIO {
                    continue;
                }
                let cos = (a + b - c) / (2.0 * (a * b).sqrt());
                if cos.abs() > MAX_CORNER_TILT.to_radians().sin() {
                    continue;
                }
                let distortion = ((c - 2.0 * b).abs() + (c - 2.0 * a).abs()) / c;
                // Finders along a side lie n - 7 modules apart in a symbol
                // of n modules.
                let module = sizes.iter().sum::<f64>() / 3.0;
                let spacing = (a.sqrt() + b.sqrt()) / 2.0 / module;
                if !(12.0..=176.0).contains(&spacing) {
                    continue;
                }
                // The top-left finder is the vertex opposite the longest
                // side; the other two are told apart by the handedness of
                // the photo's axes.
                let others: Vec<usize> = (0..3).filter(|&v| v != tl).collect();
                let (mut tr, mut bl) = (others[0], others[1]);
                let t = trio[tl].center;
                let cross = |r: [f64; 2], l: [f64; 2]| (r[0] - t[0]) * (l[1] - t[1]) - (r[1] - t[1]) * (l[0] - t[0]);
                if cross(trio[tr].center, trio[bl].center) < 0.0 {
                    std::mem::swap(&mut tr, &mut bl);
                }
                let (r, l) = (trio[tr].center, trio[bl].center);
                // The finders along each side must face each other as a
                // symbol's do, and no finder that could be another
                // symbol's may lie within the symbol proposed: three
                // finders of a sheet of symbols stand as readily at the
                // corners of a square as three of one symbol do, and
                // OpenCV's multi-symbol detector likewise rejects a triple
                // with a fourth finder inside it.
                if !finders_face(trio[tl], trio[tr], frame) || !finders_face(trio[tl], trio[bl], frame) {
                    continue;
                }
                let (u_r, v_r) = ([p[tr][0] - p[tl][0], p[tr][1] - p[tl][1]], [p[bl][0] - p[tl][0], p[bl][1] - p[tl][1]]);
                if encloses_a_finder(frame, capstones, [i, j, k], p[tl], u_r, v_r, spacing, (lo, hi)) {
                    continue;
                }
                // Along each side from the top-left finder a timing
                // pattern should read, with the third finder on its side:
                // asked last, of the triples the shape has let through,
                // since it is the one test that reads the photo.
                let timed = [(tr, bl), (bl, tr)]
                    .iter()
                    .filter(|&&(other, third)| {
                        timing_between(image, frame, trio[tl], trio[other], frame.to_rectified(trio[third].center)).is_some()
                    })
                    .count();
                let unread = (2 - timed) as f64;
                let (u, v) = ([r[0] - t[0], r[1] - t[1]], [l[0] - t[0], l[1] - t[1]]);
                // The reach, as a share of the finder spacing along a side.
                let m = PROPOSAL_REACH / spacing;
                let outline = [
                    [t[0] - m * (u[0] + v[0]), t[1] - m * (u[1] + v[1])],
                    [t[0] + u[0] + m * (u[0] - v[0]), t[1] + u[1] + m * (u[1] - v[1])],
                    [t[0] + u[0] + v[0] + m * (u[0] + v[0]), t[1] + u[1] + v[1] + m * (u[1] + v[1])],
                    [t[0] + v[0] + m * (v[0] - u[0]), t[1] + v[1] + m * (v[1] - u[1])],
                ];
                let badness = distortion + (hi / lo - 1.0) + faintness + TIMING_PENALTY * unread;
                out.push(TripleProposal { badness, outline, members: [i, j, k], timed });
            }
        }
    }
    out
}

/// Whether two capstones face each other as the finders along one side
/// of a symbol do: the line between their centres crosses a side of each
/// within [`FACING_MIDPOINT`] of that side's middle, and within
/// [`FACING_DEGREES`] of square to it, in the frame's pixels. A capstone
/// whose corners were not measured faces anything: its square was put
/// square to the frame and says nothing of its turn.
fn finders_face(a: &Capstone, b: &Capstone, frame: &Frame) -> bool {
    if !a.measured || !b.measured {
        return true;
    }
    let (pa, pb) = (frame.to_rectified(a.center), frame.to_rectified(b.center));
    let faces = |c: &Capstone, from: [f64; 2], to: [f64; 2]| {
        let quad = c.corners.map(|p| frame.to_rectified(p));
        let line = [to[0] - from[0], to[1] - from[1]];
        (0..4).any(|i| {
            let (p, q) = (quad[i], quad[(i + 1) % 4]);
            let side = [q[0] - p[0], q[1] - p[1]];
            // from + s·line = p + t·side, with both on their segments.
            let det = line[0] * side[1] - line[1] * side[0];
            if det.abs() < 1e-12 {
                return false;
            }
            let (dx, dy) = (p[0] - from[0], p[1] - from[1]);
            let s = (dx * side[1] - dy * side[0]) / det;
            let t = (dx * line[1] - dy * line[0]) / det;
            if !(0.0..=1.0).contains(&s) || !(0.0..=1.0).contains(&t) {
                return false;
            }
            let lengths = (line[0] * line[0] + line[1] * line[1]).sqrt() * (side[0] * side[0] + side[1] * side[1]).sqrt();
            let cos = (line[0] * side[0] + line[1] * side[1]) / lengths.max(1e-12);
            (t - 0.5).abs() <= FACING_MIDPOINT && cos.abs() <= FACING_DEGREES.to_radians().sin()
        })
    };
    faces(a, pa, pb) && faces(b, pb, pa)
}

/// Whether a capstone that could be the finder of another symbol lies
/// inside the symbol three capstones propose: one not among `members`,
/// of their polarity, whole, with measured corners and a module within
/// the range `modules`, whose centre lies within the symbol that spans
/// `spacing` modules from `origin` along `u` and along `v`, in the
/// frame's pixels, finder centres being three and a half modules in from
/// the symbol's edges.
fn encloses_a_finder(
    frame: &Frame,
    capstones: &[Capstone],
    members: [usize; 3],
    origin: [f64; 2],
    u: [f64; 2],
    v: [f64; 2],
    spacing: f64,
    modules: (f64, f64),
) -> bool {
    let det = u[0] * v[1] - u[1] * v[0];
    if det.abs() < 1e-12 {
        return false;
    }
    let reach = 3.5 / spacing;
    let polarity = capstones[members[0]].inverted;
    capstones.iter().enumerate().any(|(c, capstone)| {
        if members.contains(&c) || capstone.inverted != polarity || capstone.partial || !capstone.measured {
            return false;
        }
        let module = capstone.module_mean();
        if module < modules.0 / SIZE_RATIO || module > modules.1 * SIZE_RATIO {
            return false;
        }
        let r = frame.to_rectified(capstone.center);
        let (dx, dy) = (r[0] - origin[0], r[1] - origin[1]);
        let alpha = (dx * v[1] - dy * v[0]) / det;
        let beta = (u[0] * dy - u[1] * dx) / det;
        (-reach..=1.0 + reach).contains(&alpha) && (-reach..=1.0 + reach).contains(&beta)
    })
}

/// The contrast of the capstone that reads most crisply, for judging the
/// others by.
fn strongest_contrast(capstones: &[Capstone]) -> f64 {
    capstones.iter().map(|c| c.contrast).fold(0.0, f64::max).max(1e-9)
}

/// Outline proposals for a symbol with one finder pattern, from the
/// capstone alone. The format word beside the finder says which way the
/// symbol extends and which version it is, so a box of that size is
/// proposed first; then four boxes of the family's largest size, with
/// the finder in each corner in turn, which are all there is when the
/// word cannot be read, and which stand behind the word's box when it
/// can. The word is read through every placement of the finder, and
/// more than half of all strings of fifteen bits lie within correction
/// range of some Micro QR format word: a symbol whose own word is
/// damaged reads one in its data through a wrong placement often
/// enough, and were that box the only one proposed, the symbol's own
/// frame would never be fitted and the decode never get to arbitrate.
fn single_finder_proposals(
    image: &Luma,
    capstone: &Capstone,
    family: Family,
    threshold: Option<f32>,
    band: f32,
) -> Vec<(f64, [[f64; 2]; 4])> {
    let Some(patch) = Homography::rect_to_quad(7.0, 7.0, capstone.corners) else { return Vec::new() };
    let reference = FunctionMap::new(reference_version(family));
    let (w, h) = (reference.width as f64, reference.height as f64);
    // The outline corner the finder sits in under a placement.
    let corner_of = |turns: u8, mirrored: bool| {
        let [u, v] = to_unit(turns, mirrored, w, h, [3.5, 3.5]);
        match (u < 0.5, v < 0.5) {
            (true, true) => 0usize,
            (false, true) => 1,
            (false, false) => 2,
            (true, false) => 3,
        }
    };
    let largest = family.versions().into_iter().max_by_key(|v| v.width() * v.height()).expect("a family has versions");
    let mut boxes: Vec<(f64, usize, u8, SymbolVersion)> = format_placements(image, capstone, family, threshold, band)
        .first()
        .map(|p| (f64::from(p.mismatches) - capstone.contrast, corner_of(p.turns, p.mirrored), p.turns, p.version))
        .into_iter()
        .collect();
    // A square symbol stands the same way whichever quarter turn it has;
    // an rMQR on end is as long one way as it is wide the other.
    let turn_options: &[u8] = if family == Family::Rmqr { &[0, 1] } else { &[0] };
    boxes.extend(
        turn_options
            .iter()
            .flat_map(|&turns| (0..4).map(move |corner| (10.0 + corner as f64 - capstone.contrast, corner, turns, largest))),
    );
    boxes
        .into_iter()
        .map(|(badness, corner, turns, version)| {
            // In the finder's own module coordinates the finder is the
            // square [0, 7]²; the symbol is `sw` by `sh` modules, with a
            // quiet zone of two and a module of slack around it. Turned a
            // quarter, its width runs up the finder's own y-axis.
            let (sw, sh) = if turns % 2 == 1 {
                (version.height() as f64, version.width() as f64)
            } else {
                (version.width() as f64, version.height() as f64)
            };
            let reach = 3.0;
            let (x0, x1) = if corner == 0 || corner == 3 { (-reach, sw + reach) } else { (7.0 - sw - reach, 7.0 + reach) };
            let (y0, y1) = if corner == 0 || corner == 1 { (-reach, sh + reach) } else { (7.0 - sh - reach, 7.0 + reach) };
            (badness, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(|[x, y]| patch.apply(x, y)))
        })
        .collect()
}

/// Outlines the capstones themselves propose for the symbol, square to
/// it, best first: from three capstones that lie as a QR symbol's finders
/// do, from two that may, or from a lone finder and the format word
/// beside it. Only outlines that lie about where the outline given was
/// drawn are kept, since that is what it says, and the same outline
/// proposed twice is kept once.
fn capstone_frames(
    image: &Luma,
    rect: &Rectified,
    capstones: &[Capstone],
    family: Family,
    threshold: Option<f32>,
    band: f32,
) -> Vec<[[f64; 2]; 4]> {
    let mut proposals: Vec<(f64, [[f64; 2]; 4])> = match family {
        Family::Qr | Family::Model1 => {
            let mut out = qr_proposals(image, &rect.frame, capstones);
            out.extend(qr_pair_proposals(image, &rect.frame, capstones));
            out
        }
        Family::Micro | Family::Rmqr => {
            capstones.iter().flat_map(|c| single_finder_proposals(image, c, family, threshold, band)).collect()
        }
    };
    proposals.retain(|(_, p)| proposal_fits_outline(&rect.frame, *p));
    proposals.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out: Vec<[[f64; 2]; 4]> = Vec::new();
    for (_, proposal) in proposals {
        if !out.iter().any(|o| frames_alike(*o, proposal)) {
            out.push(proposal);
        }
    }
    out
}

/// Whether a proposed outline lies about where the outline given was
/// drawn: its centre inside that outline, with the search margin, and its
/// size within range of it.
fn proposal_fits_outline(frame: &Frame, proposal: [[f64; 2]; 4]) -> bool {
    let [u, v] = frame.unit(quad_centre(proposal));
    let within = -MARGIN..=1.0 + MARGIN;
    if !(within.contains(&u) && within.contains(&v)) {
        return false;
    }
    let r = proposal.map(|c| frame.to_rectified(c));
    let side = (distance(r[0], r[1]) + distance(r[1], r[2]) + distance(r[2], r[3]) + distance(r[3], r[0])) / 4.0;
    let given = (frame.size[0] + frame.size[1]) / 2.0;
    (PROPOSAL_SIZE_MIN..=PROPOSAL_SIZE_MAX).contains(&(side / given))
}

/// Whether two outlines are the same outline, whichever corner each
/// starts from: every corner of each lies within [`FRAME_LIKENESS`] of
/// their size of a corner of the other.
fn frames_alike(a: [[f64; 2]; 4], b: [[f64; 2]; 4]) -> bool {
    let size = |q: [[f64; 2]; 4]| (distance(q[0], q[1]) + distance(q[0], q[3])) / 2.0;
    let tolerance = FRAME_LIKENESS * size(a).max(size(b));
    a.iter().all(|p| b.iter().any(|q| distance(*p, *q) <= tolerance))
        && b.iter().all(|q| a.iter().any(|p| distance(*p, *q) <= tolerance))
}

/// Whether two placements of a symbol are the same placement: the same
/// outline with each corner of the symbol at the same corner of it. The
/// corners of a placement are in the symbol's own order, so one turned or
/// reflected over the very square another stands on is another reading
/// of the photo, which [`frames_alike`] would take for the same.
fn placements_alike(a: [[f64; 2]; 4], b: [[f64; 2]; 4]) -> bool {
    let size = |q: [[f64; 2]; 4]| (distance(q[0], q[1]) + distance(q[0], q[3])) / 2.0;
    let tolerance = FRAME_LIKENESS * size(a).max(size(b));
    a.iter().zip(&b).all(|(p, q)| distance(*p, *q) <= tolerance)
}

/// The quadrilateral with its corners in clockwise order on the photo,
/// as an outline has them, reversed if they were not.
fn clockwise(quad: [[f64; 2]; 4]) -> [[f64; 2]; 4] {
    let area: f64 = (0..4).map(|i| {
        let (p, q) = (quad[i], quad[(i + 1) % 4]);
        p[0] * q[1] - q[0] * p[1]
    }).sum();
    if area < 0.0 { [quad[0], quad[3], quad[2], quad[1]] } else { quad }
}

/// A loose outline around a symbol placed at `corners`, as a hand would
/// draw one: clockwise round the photo whichever way the symbol's own
/// corners run, and `margin` of its size wider all round.
pub(crate) fn outline_around(corners: [[f64; 2]; 4], margin: f64) -> [[f64; 2]; 4] {
    widened(clockwise(corners), margin)
}

/// The quadrilateral pushed outward from its centre by `margin` of itself.
fn widened(quad: [[f64; 2]; 4], margin: f64) -> [[f64; 2]; 4] {
    let c = quad_centre(quad);
    quad.map(|p| [c[0] + (p[0] - c[0]) * (1.0 + 2.0 * margin), c[1] + (p[1] - c[1]) * (1.0 + 2.0 * margin)])
}

/// How two capstones stand in the outline they propose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PairKind {
    /// The finders along one side, with the symbol to one side of them,
    /// and whether the timing pattern between them reads on that side.
    Side { timed: bool },
    /// The finders at the ends of the diagonal.
    Diagonal,
}

/// An outline two capstones propose for a QR symbol.
#[derive(Clone, Copy, Debug)]
struct PairProposal {
    badness: f64,
    outline: [[f64; 2]; 4],
    /// The two, as indices into the capstones they were taken from.
    members: [usize; 2],
    kind: PairKind,
}

/// Outline proposals for QR symbols from pairs of capstones, for when a
/// third finder is torn away or was missed: module sizes alike, spacing
/// one a version could have. The two may be the finders along one side,
/// with the symbol on either side of them, or those at the ends of the
/// diagonal; all three outlines are proposed, and the fit tells them
/// apart, after any outline proposed from three capstones.
fn qr_pair_proposals(image: &Luma, frame: &Frame, capstones: &[Capstone]) -> Vec<(f64, [[f64; 2]; 4])> {
    qr_pairs(image, frame, capstones).into_iter().map(|p| (p.badness, p.outline)).collect()
}

/// [`qr_pair_proposals`] with the capstones each proposal stands on, and
/// how they stand in it.
fn qr_pairs(image: &Luma, frame: &Frame, capstones: &[Capstone]) -> Vec<PairProposal> {
    let mut out = Vec::new();
    let n = capstones.len();
    let strongest = strongest_contrast(capstones);
    for i in 0..n {
        for j in i + 1..n {
            let (a, b) = (&capstones[i], &capstones[j]);
            if a.inverted != b.inverted {
                continue;
            }
            let (lo, hi) = (a.module_mean().min(b.module_mean()), a.module_mean().max(b.module_mean()));
            if hi > SIZE_RATIO * lo {
                continue;
            }
            let (pa, pb) = (frame.to_rectified(a.center), frame.to_rectified(b.center));
            let spacing = distance(pa, pb) / ((lo + hi) / 2.0);
            // Along a side, finders lie n - 7 modules apart; across the
            // diagonal, √2 times that.
            if !(12.0..=176.0 * SQRT_2).contains(&spacing) {
                continue;
            }
            let faintness = CONTRAST_WEIGHT * (1.0 - a.contrast.min(b.contrast) / strongest).max(0.0);
            let badness = PAIR_BADNESS + (hi / lo - 1.0) + faintness;
            let (t, r) = (a.center, b.center);
            let u = [r[0] - t[0], r[1] - t[1]];
            let v = [-u[1], u[0]];
            // The pair along a side: the square on one side of it, and
            // the same square shifted across to the other side. The
            // timing pattern between the two lies on the symbol's side,
            // and says which that is.
            for shift in [[0.0, 0.0], [u[1], -u[0]]] {
                let tl = [t[0] + shift[0], t[1] + shift[1]];
                let square = [tl, [tl[0] + u[0], tl[1] + u[1]], [tl[0] + u[0] + v[0], tl[1] + u[1] + v[1]], [tl[0] + v[0], tl[1] + v[1]]];
                let inside = frame.to_rectified(quad_centre(square));
                let timed = timing_between(image, frame, a, b, inside).is_some();
                let timing = if timed { -TIMING_BONUS } else { TIMING_PENALTY };
                let outline = widened(clockwise(square), PROPOSAL_REACH / spacing);
                out.push(PairProposal { badness: badness + timing, outline, members: [i, j], kind: PairKind::Side { timed } });
            }
            // The pair across the diagonal: the other two corners lie
            // across its middle.
            if spacing >= 12.0 * SQRT_2 {
                let c = [(t[0] + r[0]) / 2.0, (t[1] + r[1]) / 2.0];
                let p = [-(r[1] - t[1]) / 2.0, (r[0] - t[0]) / 2.0];
                let square = [t, [c[0] + p[0], c[1] + p[1]], r, [c[0] - p[0], c[1] - p[1]]];
                // The square's side is the pair's spacing over root two.
                let outline = widened(clockwise(square), PROPOSAL_REACH * SQRT_2 / spacing);
                out.push(PairProposal { badness, outline, members: [i, j], kind: PairKind::Diagonal });
            }
        }
    }
    out
}

/// Whether the line between two capstones leaves each along the diagonal
/// of its own square, to within [`DIAGONAL_DEGREES`], as the line between
/// the finders at the ends of a symbol's diagonal does. A capstone whose
/// corners were not measured says nothing of its turn, and so not.
fn finders_lie_diagonal(a: &Capstone, b: &Capstone) -> bool {
    if !a.measured || !b.measured {
        return false;
    }
    let line = [b.center[0] - a.center[0], b.center[1] - a.center[1]];
    let off_diagonal = |c: &Capstone| {
        let side = [c.corners[1][0] - c.corners[0][0], c.corners[1][1] - c.corners[0][1]];
        let (cross, dot) = (line[0] * side[1] - line[1] * side[0], line[0] * side[0] + line[1] * side[1]);
        // The angle between the line and the side, folded into a quarter
        // turn: forty-five degrees for the diagonal.
        (cross.abs().atan2(dot.abs()).to_degrees() - 45.0).abs()
    };
    off_diagonal(a) <= DIAGONAL_DEGREES && off_diagonal(b) <= DIAGONAL_DEGREES
}

/// A line in photo pixels: a point on it and a unit direction.
#[derive(Clone, Copy, Debug)]
struct Line {
    point: [f64; 2],
    dir: [f64; 2],
}

impl Line {
    /// Total-least-squares line through `points`; `None` for fewer than two.
    fn fit(points: &[[f64; 2]]) -> Option<Self> {
        if points.len() < 2 {
            return None;
        }
        let n = points.len() as f64;
        let cx = points.iter().map(|p| p[0]).sum::<f64>() / n;
        let cy = points.iter().map(|p| p[1]).sum::<f64>() / n;
        let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
        for p in points {
            let (dx, dy) = (p[0] - cx, p[1] - cy);
            sxx += dx * dx;
            sxy += dx * dy;
            syy += dy * dy;
        }
        // Principal direction of the scatter.
        let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
        let dir = [theta.cos(), theta.sin()];
        Some(Self { point: [cx, cy], dir })
    }

    fn distance(&self, p: [f64; 2]) -> f64 {
        ((p[0] - self.point[0]) * self.dir[1] - (p[1] - self.point[1]) * self.dir[0]).abs()
    }

    fn intersect(&self, other: &Line) -> Option<[f64; 2]> {
        let cross = self.dir[0] * other.dir[1] - self.dir[1] * other.dir[0];
        if cross.abs() < 1e-9 {
            return None;
        }
        let dx = other.point[0] - self.point[0];
        let dy = other.point[1] - self.point[1];
        let t = (dx * other.dir[1] - dy * other.dir[0]) / cross;
        Some([self.point[0] + t * self.dir[0], self.point[1] + t * self.dir[1]])
    }

    /// The point on the line nearest `p`.
    fn project(&self, p: [f64; 2]) -> [f64; 2] {
        let t = (p[0] - self.point[0]) * self.dir[0] + (p[1] - self.point[1]) * self.dir[1];
        [self.point[0] + t * self.dir[0], self.point[1] + t * self.dir[1]]
    }
}

/// Fit a line to `points`, dropping the quarter that fit worst and fitting
/// again, as ZBar keeps only the inliers of its edges. Returns the line
/// and the root-mean-square distance of the kept points, in pixels.
fn robust_line(points: &[[f64; 2]]) -> Option<(Line, f64, usize)> {
    let first = Line::fit(points)?;
    let mut distances: Vec<(f64, [f64; 2])> = points.iter().map(|&p| (first.distance(p), p)).collect();
    distances.sort_by(|a, b| a.0.total_cmp(&b.0));
    let keep = (points.len() - points.len() / 4).max(2);
    let kept: Vec<[f64; 2]> = distances[..keep].iter().map(|d| d.1).collect();
    let line = Line::fit(&kept)?;
    let rms = (kept.iter().map(|&p| line.distance(p).powi(2)).sum::<f64>() / kept.len() as f64).sqrt();
    Some((line, rms, kept.len()))
}

/// One outer edge of the symbol as seen from a corner: where it starts, in
/// module coordinates, which way it runs away from the corner, and which
/// way is into the symbol.
struct EdgeFrame {
    corner: [f64; 2],
    along: [f64; 2],
    inward: [f64; 2],
    /// Modules along the edge.
    length: f64,
}

/// The two edges meeting at corner `i` of a `w` × `h` symbol.
fn edges_at(i: usize, w: f64, h: f64) -> [EdgeFrame; 2] {
    let corner = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]][i];
    let horizontal = EdgeFrame {
        corner,
        along: [if i == 0 || i == 3 { 1.0 } else { -1.0 }, 0.0],
        inward: [0.0, if i == 0 || i == 1 { 1.0 } else { -1.0 }],
        length: w,
    };
    let vertical = EdgeFrame {
        corner,
        along: [0.0, if i == 0 || i == 1 { 1.0 } else { -1.0 }],
        inward: [if i == 0 || i == 3 { 1.0 } else { -1.0 }, 0.0],
        length: h,
    };
    [horizontal, vertical]
}

/// What tracing one edge found: the line through the crossings, or, on
/// an edge that bows, the tangent at the corner to the bend fitted to
/// them; whether it was a bend; and the crossings the fit kept, each as
/// the point of the symbol's edge it lies on, in module coordinates, with
/// where the paper turned to ink there in the photo.
struct Traced {
    line: Line,
    bent: bool,
    crossings: Vec<ControlPoint>,
}

/// Trace the symbol's edge against its quiet zone along one outer edge from
/// a corner, as ZBar walks the edge of a code looking for the border
/// modules. Profiles across the predicted edge, at module centres along
/// the edge, find where the paper turns to ink. Returns a line through
/// those crossings when enough of them agree, and `None` when the edge has
/// no quiet zone, too few border modules are dark, or the crossings do not
/// line up. `contrast` is the symbol's own, which the floor on a border
/// module's contrast is a share of.
fn trace_edge(
    image: &Luma,
    map: &impl Project,
    edge: &EdgeFrame,
    sign: f64,
    threshold: f32,
    contrast: f32,
) -> Option<Traced> {
    // Every border module but the corner ones, thinned to the limit for a
    // long edge. Only about half the border modules are dark, so a short
    // edge needs all of them to have a line to fit.
    let along_modules = (edge.length as usize).saturating_sub(2).max(2);
    let stride = along_modules.div_ceil(MAX_PROFILES).max(1);
    let positions: Vec<usize> = (1..edge.length as usize - 1).step_by(stride).collect();
    let probes = positions.len();
    let steps = (2.0 * PROFILE_REACH / PROFILE_STEP).round() as usize;
    let sample = |s: f64, t: f64| {
        let [x, y] = [
            edge.corner[0] + edge.along[0] * s + edge.inward[0] * t,
            edge.corner[1] + edge.along[1] * s + edge.inward[1] * t,
        ];
        let [px, py] = map.apply(x, y);
        image.sample(px, py)
    };
    // Paper is light, or dark for a light-on-dark symbol.
    let is_quiet = |v: f32| (sign > 0.0) == (v >= threshold);
    let min_contrast = (PROFILE_SHARE * contrast).max(NOISE_RANGE);
    let mut quiet = 0usize;
    let mut crossings: Vec<[f64; 2]> = Vec::new();
    // The point of the symbol's edge each crossing lies on, in module
    // coordinates: where the map should put the crossing.
    let mut sites: Vec<[f64; 2]> = Vec::new();
    for &k in &positions {
        let s = k as f64 + 0.5;
        let profile: Vec<Option<f32>> = (0..=steps).map(|j| sample(s, -PROFILE_REACH + j as f64 * PROFILE_STEP)).collect();
        let t_of = |j: usize| -PROFILE_REACH + j as f64 * PROFILE_STEP;
        let outside: Vec<f32> = (0..=steps)
            .filter(|&j| (-QUIET_TO..=-QUIET_FROM).contains(&t_of(j)))
            .filter_map(|j| profile[j])
            .collect();
        if outside.is_empty() {
            continue;
        }
        let paper = outside.iter().sum::<f32>() / outside.len() as f32;
        if is_quiet(paper) {
            quiet += 1;
        } else {
            continue;
        }
        // The border module, half a module in.
        let border_j = ((0.5 + PROFILE_REACH) / PROFILE_STEP).round() as usize;
        let Some(border) = profile.get(border_j).copied().flatten() else { continue };
        if (sign as f32) * (paper - border) < min_contrast {
            // A light border module: no edge to see here.
            continue;
        }
        let level = (paper + border) / 2.0;
        // Walk inward from the quiet zone to the first crossing of the level.
        let start_j = ((-QUIET_FROM + PROFILE_REACH) / PROFILE_STEP).round() as usize;
        let mut found = None;
        for j in start_j..border_j {
            let (Some(a), Some(b)) = (profile[j], profile[j + 1]) else { continue };
            let (fa, fb) = ((sign as f32) * (level - a), (sign as f32) * (level - b));
            if fa <= 0.0 && fb > 0.0 {
                let frac = if (fb - fa).abs() > 1e-6 { -fa / (fb - fa) } else { 0.5 };
                found = Some(t_of(j) + frac as f64 * PROFILE_STEP);
                break;
            }
        }
        let Some(t) = found else { continue };
        let [x, y] = [
            edge.corner[0] + edge.along[0] * s + edge.inward[0] * t,
            edge.corner[1] + edge.along[1] * s + edge.inward[1] * t,
        ];
        crossings.push(map.apply(x, y));
        sites.push([edge.corner[0] + edge.along[0] * s, edge.corner[1] + edge.along[1] * s]);
    }
    if (quiet as f64) < QUIET_SHARE * probes as f64 || crossings.len() < MIN_CROSSINGS {
        return None;
    }
    let module = module_size_at(map, [edge.corner[0] + edge.along[0] * 2.0, edge.corner[1] + edge.along[1] * 2.0]);
    let limit = MAX_EDGE_RMS * module;
    // The crossings within reach of what was fitted through them, as
    // control points; the rest are stray print.
    let kept = |near: &dyn Fn([f64; 2]) -> f64| -> Vec<ControlPoint> {
        crossings
            .iter()
            .zip(&sites)
            .filter(|(p, _)| near(**p) <= 2.0 * limit)
            .map(|(p, at)| ControlPoint { at: *at, px: *p })
            .collect()
    };
    // A straight edge: every crossing lies on one line.
    let straight = Line::fit(&crossings)?;
    let rms_all = (crossings.iter().map(|&p| straight.distance(p).powi(2)).sum::<f64>() / crossings.len() as f64).sqrt();
    // Paper that curls bows the edge. A gentle bow still passes for a
    // line across the crossings, which lie along the middle of the edge,
    // but the edge bends most towards its ends, and the line misses the
    // corner by a module. OpenCV fits a cubic spline to each side of a
    // curved symbol and straightens it in strips; here a bend fitted to
    // the crossings gives the edge's tangent at the corner, which is all
    // the corner needs, and the anchors bend the map inside the outline.
    // The bend is preferred when it fits the crossings clearly better
    // than the line and bows by more than noise would.
    let corner = map.apply(edge.corner[0], edge.corner[1]);
    if let Some((tangent, rms_curve, bow, curve)) = robust_curve(&crossings, corner, limit, MAX_EDGE_SAGITTA * module)
        && rms_curve <= CURVE_RMS * module
        && rms_curve < CURVE_GAIN * rms_all
        && bow >= MIN_EDGE_SAGITTA * module
    {
        return Some(Traced { line: tangent, bent: true, crossings: kept(&|p| curve.residual(p).abs()) });
    }
    if rms_all <= limit {
        return Some(Traced { line: straight, bent: false, crossings: kept(&|p| straight.distance(p)) });
    }
    // Otherwise the worst crossings are stray print, and are left out.
    robust_line(&crossings)
        .filter(|(_, rms, kept)| *kept >= MIN_CROSSINGS && *rms <= limit)
        .map(|(line, _, _)| Traced { line, bent: false, crossings: kept(&|p| line.distance(p)) })
}

/// A gentle bend fitted to points in photo pixels: a quadratic across the
/// line through them.
struct Curve {
    axis: Line,
    normal: [f64; 2],
    c: [f64; 3],
}

impl Curve {
    /// How far `p` lies across the bend from it, in pixels, signed.
    fn residual(&self, p: [f64; 2]) -> f64 {
        let s = (p[0] - self.axis.point[0]) * self.axis.dir[0] + (p[1] - self.axis.point[1]) * self.axis.dir[1];
        let t = (p[0] - self.axis.point[0]) * self.normal[0] + (p[1] - self.axis.point[1]) * self.normal[1];
        t - (self.c[0] + self.c[1] * s + self.c[2] * s * s)
    }
}

/// Fit a gentle bend to `points`, a quadratic across the line through
/// them, dropping the quarter that fit worst and fitting again, and
/// return the tangent to it at the point along it nearest `at`, with the
/// root-mean-square distance of the kept points from it and how far it
/// bows from straight over them, both in pixels, and the bend itself.
/// `None` for too few points, a fit worse than `limit`, or a bow deeper
/// than `sagitta`.
fn robust_curve(points: &[[f64; 2]], at: [f64; 2], limit: f64, sagitta: f64) -> Option<(Line, f64, f64, Curve)> {
    if points.len() < MIN_CURVE_CROSSINGS {
        return None;
    }
    let axis = Line::fit(points)?;
    let normal = [-axis.dir[1], axis.dir[0]];
    let along = |p: [f64; 2]| (p[0] - axis.point[0]) * axis.dir[0] + (p[1] - axis.point[1]) * axis.dir[1];
    let across = |p: [f64; 2]| (p[0] - axis.point[0]) * normal[0] + (p[1] - axis.point[1]) * normal[1];
    let st: Vec<[f64; 2]> = points.iter().map(|&p| [along(p), across(p)]).collect();
    let fit = |st: &[[f64; 2]]| -> Option<[f64; 3]> {
        // Normal equations of t = c0 + c1 s + c2 s².
        let mut a = [[0.0f64; 3]; 3];
        let mut b = [0.0f64; 3];
        for &[s, t] in st {
            let basis = [1.0, s, s * s];
            for i in 0..3 {
                b[i] += basis[i] * t;
                for j in 0..3 {
                    a[i][j] += basis[i] * basis[j];
                }
            }
        }
        solve3(a, b)
    };
    let residual = |c: [f64; 3], [s, t]: [f64; 2]| t - (c[0] + c[1] * s + c[2] * s * s);
    let first = fit(&st)?;
    let mut ranked: Vec<[f64; 2]> = st.clone();
    ranked.sort_by(|p, q| residual(first, *p).abs().total_cmp(&residual(first, *q).abs()));
    ranked.truncate((st.len() - st.len() / 4).max(MIN_CURVE_CROSSINGS - 1));
    let c = fit(&ranked)?;
    let rms = (ranked.iter().map(|&p| residual(c, p).powi(2)).sum::<f64>() / ranked.len() as f64).sqrt();
    let (s_min, s_max) = ranked.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |m, p| (m.0.min(p[0]), m.1.max(p[0])));
    let bow = c[2].abs() * ((s_max - s_min) / 2.0).powi(2);
    if rms > limit || bow > sagitta {
        return None;
    }
    let s = along(at);
    let t = c[0] + c[1] * s + c[2] * s * s;
    let slope = c[1] + 2.0 * c[2] * s;
    let point = [axis.point[0] + axis.dir[0] * s + normal[0] * t, axis.point[1] + axis.dir[1] * s + normal[1] * t];
    let dir = [axis.dir[0] + normal[0] * slope, axis.dir[1] + normal[1] * slope];
    let len = distance([0.0, 0.0], dir);
    Some((Line { point, dir: [dir[0] / len, dir[1] / len] }, rms, bow, Curve { axis, normal, c }))
}

/// Solve the 3 × 3 system `a · x = b` by elimination with pivoting;
/// `None` when it is singular.
fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let pivot = (col..3).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in 0..3 {
            if row != col {
                let f = a[row][col] / a[col][col];
                let pivot_row = a[col];
                for (entry, &p) in a[row].iter_mut().zip(&pivot_row).skip(col) {
                    *entry -= f * p;
                }
                b[row] -= f * b[col];
            }
        }
    }
    Some([b[0] / a[0][0], b[1] / a[1][1], b[2] / a[2][2]])
}

/// Scores placements of one corner's quarter of the symbol: the contrast
/// of the fixed modules there, where there are any, and how crisply the
/// data modules read dark or light rather than grey, which a grid off by
/// part of a module cannot manage.
struct CornerScorer<'a> {
    image: &'a Luma,
    width: f64,
    height: f64,
    quarter: [[f64; 2]; 4],
    fixed: Option<Scorer<'a>>,
    data: Vec<Pos>,
    threshold: Option<f32>,
}

impl<'a> CornerScorer<'a> {
    fn new(image: &'a Luma, func: &FunctionMap, corners: [[f64; 2]; 4], i: usize, threshold: Option<f32>) -> Option<Self> {
        let (w, h) = (func.width, func.height);
        let (qx0, qx1) = if i == 0 || i == 3 { (0, w.div_ceil(2)) } else { (w / 2, w) };
        let (qy0, qy1) = if i == 0 || i == 1 { (0, h.div_ceil(2)) } else { (h / 2, h) };
        let template = Template::window(func, qx0 as i64, qy0 as i64, qx1 - qx0, qy1 - qy0);
        let quarter = [[qx0, qy0], [qx1, qy0], [qx1, qy1], [qx0, qy1]].map(|[x, y]| [x as f64, y as f64]);
        let map = Homography::rect_to_quad(w as f64, h as f64, corners)?;
        let base = quarter.map(|[x, y]| map.apply(x, y));
        let fixed = (template.fixed() >= 4).then(|| Scorer::new(image, &template, base, threshold).with_along(6));
        // Every module of the quarter that is not fixed, thinned to a
        // manageable number.
        let mut data: Vec<Pos> = (qy0..qy1)
            .flat_map(|y| (qx0..qx1).map(move |x| (x, y)))
            .filter(|&(x, y)| func.expected(x, y).is_none())
            .map(|(x, y)| Pos::new(x, y))
            .collect();
        let stride = data.len().div_ceil(200).max(1);
        data = data.into_iter().step_by(stride).collect();
        Some(Self { image, width: w as f64, height: h as f64, quarter, fixed, data, threshold })
    }

    fn has_fixed(&self) -> bool {
        self.fixed.is_some()
    }

    fn score(&self, outline: [[f64; 2]; 4], sign: f64) -> Option<f64> {
        let map = Homography::rect_to_quad(self.width, self.height, outline)?;
        let fixed = match &self.fixed {
            Some(scorer) => {
                let placed = self.quarter.map(|[x, y]| map.apply(x, y));
                let h = Homography::rect_to_quad(scorer_width(scorer), scorer_height(scorer), placed)?;
                scorer.contrast_with(&h, sign)?
            }
            None => 0.0,
        };
        let luma = sample_positions(self.image, &map, &self.data, INNER_FRACTION, 3);
        let values: Vec<f32> = luma.iter().flatten().copied().collect();
        let crisp = if values.len() >= 8 {
            let t = self.threshold.unwrap_or_else(|| otsu(values.iter().copied()));
            values.iter().map(|v| (v - t).abs() as f64).sum::<f64>() / values.len() as f64
        } else {
            0.0
        };
        Some(fixed + CRISP_WEIGHT * crisp)
    }
}

fn scorer_width(scorer: &Scorer) -> f64 {
    scorer.template_width()
}

fn scorer_height(scorer: &Scorer) -> f64 {
    scorer.template_height()
}

/// The threshold the edge trace reads against, the one given or the split
/// of the fixed modules, and the symbol's contrast: the gap between the
/// means of its dark and light fixed modules, which the floors of the
/// trace are shares of.
fn edge_levels(image: &Luma, func: &FunctionMap, map: &impl Project, threshold: Option<f32>) -> (f32, f32) {
    let fixed: Vec<Pos> = (0..func.height)
        .flat_map(|y| (0..func.width).map(move |x| (x, y)))
        .filter(|&(x, y)| func.expected(x, y).is_some())
        .map(|(x, y)| Pos::new(x, y))
        .collect();
    let (split, means, _) = otsu_split(sample_positions(image, map, &fixed, INNER_FRACTION, POINTS).into_iter().flatten());
    (threshold.unwrap_or(split), (means[1] - means[0]).max(0.0))
}

/// Most crossings of one edge kept as anchors: enough to draw its bow.
const EDGE_ANCHORS_PER_EDGE: usize = 5;
/// How far from a corner, in modules, an edge anchor must lie: nearer, it
/// says nothing the corner does not, and the projection drops it anyway.
const EDGE_ANCHOR_CLEARANCE: f64 = 2.5;

/// Anchors along the edges of the symbol that bow, for paper that curls:
/// each of the four edges is traced against the quiet zone through the
/// map the corners and `anchors` make, and where the crossings fit a
/// bend rather than a line, a few of them, spread along the edge, become
/// anchors of their own, the crossing in the photo standing for the
/// point of the symbol's edge it was traced from. A straight edge adds
/// nothing the corners do not say, and an edge without a quiet zone
/// cannot be traced.
fn edge_anchors(
    image: &Luma,
    func: &FunctionMap,
    corners: [[f64; 2]; 4],
    anchors: &[Anchor],
    sign: f64,
    threshold: Option<f32>,
) -> Vec<Anchor> {
    let (w, h) = (func.width as f64, func.height as f64);
    let controls: Vec<ControlPoint> = anchors.iter().map(Anchor::control_point).collect();
    let Some(map) = Projection::new(func.width, func.height, corners, &controls) else { return Vec::new() };
    let (level, contrast) = edge_levels(image, func, &map, threshold);
    let outline = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
    let mut out = Vec::new();
    // The top and left edges from the top-left corner, the bottom and
    // right from the bottom-right.
    for edge in edges_at(0, w, h).iter().chain(edges_at(2, w, h).iter()) {
        let Some(traced) = trace_edge(image, &map, edge, sign, level, contrast) else { continue };
        if !traced.bent {
            continue;
        }
        let clear: Vec<&ControlPoint> =
            traced.crossings.iter().filter(|c| outline.iter().all(|&o| distance(o, c.at) >= EDGE_ANCHOR_CLEARANCE)).collect();
        if clear.is_empty() {
            continue;
        }
        let stride = clear.len().div_ceil(EDGE_ANCHORS_PER_EDGE).max(1);
        for c in clear.iter().step_by(stride) {
            out.push(Anchor { kind: AnchorKind::Edge, at: c.at, center: c.px, partial: false, inferred: false });
        }
    }
    out
}

/// Move corner `i` of an outline that an anchor holds, an alignment
/// pattern or a timing module some modules in from it, to where its two
/// edges meet, when both trace against the quiet zone. The map through
/// the anchor runs on to the corner as the paper lies between them,
/// which on curled paper is not where the edge goes, and on a blurred
/// photo the anchor itself is read to a few tenths of a module; the
/// edges, traced over many modules, place the corner better either way.
/// Returns whether the corner was moved.
fn settle_traced_corner(image: &Luma, func: &FunctionMap, corners: &mut [[f64; 2]; 4], i: usize, sign: f64, threshold: Option<f32>) -> bool {
    let (w, h) = (func.width as f64, func.height as f64);
    let Some(map) = Homography::rect_to_quad(w, h, *corners) else { return false };
    let (edge_threshold, contrast) = edge_levels(image, func, &map, threshold);
    let traced: Vec<Line> =
        edges_at(i, w, h).iter().filter_map(|e| trace_edge(image, &map, e, sign, edge_threshold, contrast).map(|t| t.line)).collect();
    let [a, b] = traced.as_slice() else { return false };
    let Some(p) = a.intersect(b) else { return false };
    let [cx, cy] = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]][i];
    // No further than a bend could carry it: beyond that the trace has
    // found other print.
    if distance(p, corners[i]) > MAX_EDGE_SAGITTA * module_size_at(&map, [cx, cy]) {
        return false;
    }
    corners[i] = p;
    polish_corner_by_data(image, func, corners, i, sign, threshold);
    if distance(corners[i], p) > POLISH_LIMIT * module_size_at(&map, [cx, cy]) {
        corners[i] = p;
    }
    true
}

/// Move corner `i` of the outline to where its quarter of the symbol reads
/// best, by the symbol's edge where there is a quiet zone to trace it
/// against, and otherwise by the fixed and data modules of the quarter
/// alone. Returns whether the edge was traced on both sides.
fn settle_corner(image: &Luma, func: &FunctionMap, corners: &mut [[f64; 2]; 4], i: usize, sign: f64, threshold: Option<f32>) -> bool {
    let (w, h) = (func.width as f64, func.height as f64);
    let Some(map) = Homography::rect_to_quad(w, h, *corners) else { return false };
    let Some(scorer) = CornerScorer::new(image, func, *corners, i, threshold) else { return false };
    let [cx, cy] = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]][i];

    let (edge_threshold, contrast) = edge_levels(image, func, &map, threshold);
    let lines: Vec<Line> =
        edges_at(i, w, h).iter().filter_map(|e| trace_edge(image, &map, e, sign, edge_threshold, contrast).map(|t| t.line)).collect();
    if let [a, b] = lines.as_slice()
        && let Some(p) = a.intersect(b)
    {
        // Two traced edges fix the corner outright: their crossing is a
        // measurement, where the search below is only a preference. The
        // profiles that found the edges stepped a tenth of a module, and
        // the polish takes the corner the last fraction of a pixel. The
        // polish reads the quarter through a plane, and on curled paper
        // would walk the corner off its edges to where a plane reads the
        // quarter best, so it may take the corner only so far.
        corners[i] = p;
        polish_corner_by_data(image, func, corners, i, sign, threshold);
        if distance(corners[i], p) > POLISH_LIMIT * module_size_at(&map, [cx, cy]) {
            corners[i] = p;
        }
        return true;
    }
    let (start, half_width) = match lines.as_slice() {
        [a] => (a.project(corners[i]), CORNER_SEARCH_EDGED),
        _ => (corners[i], if scorer.has_fixed() { CORNER_SEARCH } else { CORNER_SEARCH_CRISP }),
    };

    let mut best_outline = *corners;
    best_outline[i] = start;
    let Some(mut best) = scorer.score(best_outline, sign) else { return false };
    let Some(mut baseline) = scorer.score(*corners, sign) else { return false };
    baseline += CORNER_GAIN;
    let steps = (half_width / SEARCH_STEP).ceil() as i32;
    // Offsets in module units around the start, through the map in use.
    let Some(start_module) = map.inverse().map(|inv| inv.apply(start[0], start[1])) else { return false };
    for iy in -steps..=steps {
        for ix in -steps..=steps {
            let mut trial = *corners;
            trial[i] = map.apply(start_module[0] + ix as f64 * SEARCH_STEP, start_module[1] + iy as f64 * SEARCH_STEP);
            if let [a] = lines.as_slice() {
                trial[i] = a.project(trial[i]);
            }
            if let Some(score) = scorer.score(trial, sign)
                && score > best
            {
                best = score;
                best_outline = trial;
            }
        }
    }
    // Then settle to a fraction of the step.
    let module_px = module_size_at(&map, [cx, cy]);
    let mut pattern = [0.0; 4];
    pattern[i] = 1.0;
    let mut evaluations = 0;
    if let Some((refined, score)) = descend_by(best_outline, &[pattern], module_px * SEARCH_STEP / 2.0, &mut evaluations, |trial| {
        let mut t = trial;
        if let [a] = lines.as_slice() {
            t[i] = a.project(t[i]);
        }
        scorer.score(t, sign)
    }) {
        best = score;
        best_outline = refined;
        // The score was of the corner on the traced edge's line, and
        // that corner is the one to keep: the descent's own lies wherever
        // its steps across the line left it.
        if let [a] = lines.as_slice() {
            best_outline[i] = a.project(best_outline[i]);
        }
    }
    if best > baseline {
        *corners = best_outline;
    }
    polish_corner_by_data(image, func, corners, i, sign, threshold);
    false
}

/// Least margin from the threshold, as a share of the contrast between
/// dark and light modules, for a data module to be trusted in the polish.
const POLISH_MARGIN: f32 = 0.3;
/// Fewest trusted data modules for the polish to have anything to fit.
const POLISH_MIN_MODULES: usize = 8;
/// The step the polish starts at, in modules: it refines a corner the
/// edge trace or the crispness search has placed, and does not place it.
const POLISH_STEP: f64 = 0.15;
/// The most the polish may move a corner that two traced edges placed,
/// in modules: further than this is the plane it reads through disagreeing
/// with curled paper, not the trace being off.
const POLISH_LIMIT: f64 = 0.3;

/// Refine corner `i` by the data modules of its quarter, read through the
/// outline as it stands: the modules that read clearly dark or light
/// become a template of their own, with the fixed modules of the quarter,
/// and the corner moves to where the boundaries between them read
/// sharpest, as the fixed patterns place the other corners. The edge
/// trace and the crispness search place the corner to within a fraction
/// of a module; the boundaries, once there are modules to see them
/// between, place it to a fraction of a pixel. The corner moves only
/// when the boundaries read sharper for it.
fn polish_corner_by_data(
    image: &Luma,
    func: &FunctionMap,
    corners: &mut [[f64; 2]; 4],
    i: usize,
    sign: f64,
    threshold: Option<f32>,
) {
    let (w, h) = (func.width, func.height);
    let (qx0, qx1) = if i == 0 || i == 3 { (0, w.div_ceil(2)) } else { (w / 2, w) };
    let (qy0, qy1) = if i == 0 || i == 1 { (0, h.div_ceil(2)) } else { (h / 2, h) };
    let Some(map) = Homography::rect_to_quad(w as f64, h as f64, *corners) else { return };
    let positions: Vec<Pos> = (qy0..qy1).flat_map(|y| (qx0..qx1).map(move |x| Pos::new(x, y))).collect();
    let luma = sample_positions(image, &map, &positions, INNER_FRACTION, 3);
    let values: Vec<f32> = luma.iter().flatten().copied().collect();
    if values.len() < POLISH_MIN_MODULES {
        return;
    }
    let (split, means, _) = otsu_split(values.iter().copied());
    let split = threshold.unwrap_or(split);
    let half = ((means[1] - means[0]) / 2.0).max(1e-3);
    let inverted = sign < 0.0;
    let mut trusted = 0;
    let expected: Vec<Option<bool>> = positions
        .iter()
        .zip(&luma)
        .map(|(pos, value)| {
            let fixed = func.expected(pos.x as usize, pos.y as usize);
            // The colour as specified: on a light-on-dark photo a dark
            // module reads light.
            let read = value.filter(|v| (v - split).abs() >= POLISH_MARGIN * half).map(|v| (v < split) != inverted);
            if fixed.is_none() && read.is_some() {
                trusted += 1;
            }
            fixed.or(read)
        })
        .collect();
    if trusted < POLISH_MIN_MODULES {
        return;
    }
    let template = Template { width: qx1 - qx0, height: qy1 - qy0, expected };
    let quarter = [[qx0, qy0], [qx1, qy0], [qx1, qy1], [qx0, qy1]].map(|[x, y]| [x as f64, y as f64]);
    let scorer = Scorer::new(image, &template, quarter.map(|[x, y]| map.apply(x, y)), threshold).with_along(6);
    let score = |outline: [[f64; 2]; 4]| {
        let m = Homography::rect_to_quad(w as f64, h as f64, outline)?;
        let placed = quarter.map(|[x, y]| m.apply(x, y));
        let patch = Homography::rect_to_quad(template.width as f64, template.height as f64, placed)?;
        scorer.contrast_with(&patch, sign)
    };
    let Some(before) = score(*corners) else { return };
    let mut pattern = [0.0; 4];
    pattern[i] = 1.0;
    let corner = [[0.0, 0.0], [w as f64, 0.0], [w as f64, h as f64], [0.0, h as f64]][i];
    let step = module_size_at(&map, corner) * POLISH_STEP;
    let mut evaluations = 0;
    if let Some((polished, after)) = descend_by(*corners, &[pattern], step, &mut evaluations, score)
        && after > before
    {
        *corners = polished;
    }
}

/// The threshold between dark and light and the contrast between them of
/// the capstones standing as finders under `assignment`: the mean of their
/// own levels, or the fixed threshold with their contrast. `None` when no
/// capstone's levels can be read.
fn finder_capstone_levels(
    image: &Luma,
    capstones: &[Capstone],
    assignment: &[Option<usize>],
    threshold: Option<f32>,
) -> Option<(f32, f32)> {
    let levels: Vec<(f32, f32)> = assignment.iter().flatten().filter_map(|&c| finder_levels(image, capstones[c].corners)).collect();
    if levels.is_empty() {
        return None;
    }
    let n = levels.len() as f32;
    let split = threshold.unwrap_or_else(|| levels.iter().map(|l| l.0).sum::<f32>() / n);
    let contrast = 2.0 * levels.iter().map(|l| l.1).sum::<f32>() / n;
    Some((split, contrast))
}

/// Where a QR Code's alignment pattern lies when the template cannot be
/// fitted to it, by the mean shift BoofCV's `QrCodeAlignmentPatternLocator`
/// places every one of its alignment patterns with. The pattern is looked
/// for about where `map`, the fit through the anchors found so far bent
/// through the nearest of them, predicts it, as BoofCV starts from each
/// pattern's place corrected by the offsets at which its neighbours were
/// found. The shift works in module coordinates through the map, so that
/// perspective and scale cost nothing: over eight samples by eight across
/// three modules, a sample within half a module of the centre weighs by
/// how dark it reads and one beyond by how light, the guess moves to their
/// weighted centre, and the step shrinks each round.
///
/// BoofCV first walks along the gradient of three samples by three to
/// where it vanishes, which about the pattern's centre is the light ring
/// all round, and takes what the shift then gives, leaving a pattern
/// placed wrong to the residuals of its grid fit. A start a module off
/// walks as readily to the middle of a flat light patch, where the
/// gradient vanishes too, and the shift from there settles on the
/// nearest light modules of the data; BoofCV's own source calls the
/// search brittle. So the shift is seeded instead by the pattern's rings,
/// dark at the centre, light a module out and dark two modules out, read
/// as three levels of grey at every quarter module across `window`
/// modules of the prediction, as zxing-cpp seeds its locator by a ring
/// test over a wide window; the shift refines the best seed, and the
/// centre it settles on is taken only while its rings still read by
/// [`LOCATOR_MIN_RING_CONTRAST`] of the symbol's contrast and it lies
/// within the window. The anchor it makes is still doubted against its
/// neighbours and against the decode. At a couple of pixels a module,
/// blurred, the template's twenty-five modules cannot be told apart and
/// its fit fails, where the rings still place the pattern: on such a
/// photo the template found four of a version 34 symbol's thirty-six
/// alignment patterns, all beside its finders, and the symbol could not
/// be read. `split` and `contrast` are the finders' own levels; `inverted`
/// reads the symbol light on dark.
fn locate_alignment_by_meanshift(
    image: &Luma,
    site: &Site,
    map: &impl Project,
    window: f64,
    inverted: bool,
    split: f32,
    contrast: f32,
) -> Option<[f64; 2]> {
    // Luminance as the symbol prints it, dark low, whichever way the
    // photo has it; `None` off the photo.
    let read = |at: [f64; 2]| -> Option<f32> {
        let [px, py] = map.apply(at[0], at[1]);
        image.sample(px, py).map(|v| if inverted { 1.0 - v } else { v })
    };
    let split = if inverted { 1.0 - split } else { split };
    // How well the rings read about a centre: the lesser of the two
    // contrasts, from the centre to the ring a module out and from that
    // ring to the ring two modules out.
    let rings = |at: [f64; 2]| -> Option<f32> {
        let centre = read(at)?;
        let ring = |d: f64| -> Option<f32> {
            let mut sum = 0.0;
            for [dx, dy] in [[d, 0.0], [-d, 0.0], [0.0, d], [0.0, -d]] {
                sum += read([at[0] + dx, at[1] + dy])?;
            }
            Some(sum / 4.0)
        };
        let (light, dark) = (ring(1.0)?, ring(2.0)?);
        Some((light - centre).min(light - dark))
    };
    // The seed: where the rings read best across the window.
    let steps = (window / LOCATOR_SEED_STEP).ceil() as i32;
    let mut seed: Option<([f64; 2], f32)> = None;
    for iy in -steps..=steps {
        for ix in -steps..=steps {
            let at = [site.at[0] + f64::from(ix) * LOCATOR_SEED_STEP, site.at[1] + f64::from(iy) * LOCATOR_SEED_STEP];
            if distance(at, site.at) > window {
                continue;
            }
            if let Some(score) = rings(at)
                && seed.is_none_or(|(_, best)| score > best)
            {
                seed = Some((at, score));
            }
        }
    }
    let (mut best, mut best_score) = seed?;
    // The mean shift from it.
    let clamp = -(LOCATOR_CLAMP * contrast);
    let mut at = best;
    let mut step = 1.0;
    for _ in 0..LOCATOR_ROUNDS {
        let (mut total, mut sx, mut sy) = (0.0f32, 0.0f32, 0.0f32);
        for y in 0..LOCATOR_SAMPLES {
            for x in 0..LOCATOR_SAMPLES {
                let offset = |k: usize| -LOCATOR_REACH + 2.0 * LOCATOR_REACH * k as f64 / (LOCATOR_SAMPLES - 1) as f64;
                let (dx, dy) = (offset(x), offset(y));
                let Some(v) = read([at[0] + dx, at[1] + dy]) else { break };
                let r = (dx * dx + dy * dy).sqrt();
                let w = (if r > LOCATOR_CORE { v - split } else { split - v }).max(clamp);
                total += w.abs();
                sx += w * dx as f32;
                sy += w * dy as f32;
            }
        }
        if total <= 0.0 {
            break;
        }
        at = [at[0] + step * f64::from(sx / total), at[1] + step * f64::from(sy / total)];
        step *= LOCATOR_DECAY;
        // The shift is believed only while the rings read better where it goes.
        if distance(at, site.at) <= window
            && let Some(score) = rings(at)
            && score > best_score
        {
            best = at;
            best_score = score;
        }
    }
    (best_score >= LOCATOR_MIN_RING_CONTRAST * contrast).then(|| map.apply(best[0], best[1]))
}

/// Place one version's outline from the capstones: match them to its
/// finders, grow the fit through its other fixed patterns, and settle any
/// corner left loose. Returns the result and its fit: the share of its
/// fixed modules, and of the first module of quiet zone around it, that
/// read right. The quiet zone tells apart versions whose fixed patterns
/// are alike, as the Micro QR versions' are: an outline drawn too small
/// puts data where the paper should be bare.
#[allow(clippy::too_many_arguments)]
fn align_version(
    image: &Luma,
    func: &FunctionMap,
    capstones: &[Capstone],
    inverted: bool,
    threshold: Option<f32>,
    band: f32,
    prename: Option<Prename>,
) -> Vec<(Detection, f64)> {
    let (w, h) = (func.width as f64, func.height as f64);
    let sites = sites(func);
    let finders: Vec<usize> = (0..sites.len()).filter(|&i| sites[i].kind == AnchorKind::Finder).collect();
    // Any symbol may sit in the outline any way up, an rMQR on end as
    // much as a square one a quarter turn round, as zxing-cpp tries its
    // finder at all four turns. Only an rMQR symbol's fixed patterns
    // tell its mirror image apart; a square symbol's read the same both
    // ways, and its information bits decide below.
    let rmqr = func.version.family() == Family::Rmqr;
    let turn_options: &[u8] = &[0, 1, 2, 3];
    let mirror_options: &[bool] = if rmqr { &[false, true] } else { &[false] };
    let mut matches: Vec<(f64, u8, bool, Vec<Option<usize>>)> = Vec::new();
    for &mirrored in mirror_options {
        for &turns in turn_options {
            let centres: Vec<[f64; 2]> = finders.iter().map(|&i| to_unit(turns, mirrored, w, h, sites[i].at)).collect();
            let (cost, assignment) = assign(capstones, &centres);
            matches.push((cost, turns, mirrored, assignment));
        }
    }
    matches.sort_by(|a, b| a.0.total_cmp(&b.0));
    let Some(cheapest) = matches.first().map(|m| m.0) else { return Vec::new() };
    // Every assignment near the cheapest is fitted, and so is the cheapest
    // assignment that stands more capstones as finders than the cheapest
    // does, within a wider margin: a finder whose module reads a third
    // larger than its fellows', as a blurred one does, costs more to match
    // than to leave out, and left out the symbol is placed turned on the
    // two that remain. Every fit comes back, ranked by the caller, for
    // the decode to arbitrate.
    let matched = |assignment: &[Option<usize>]| assignment.iter().flatten().count();
    let base = matched(&matches[0].3);
    let mut chosen: Vec<usize> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    for (i, m) in matches.iter().enumerate() {
        let count = matched(&m.3);
        let near = m.0 <= cheapest + ASSIGNMENT_SLACK;
        let fuller = count > base && !counts.contains(&count) && m.0 <= cheapest + ASSIGNMENT_FULLER_SLACK;
        if fuller {
            counts.push(count);
        }
        if near || fuller {
            chosen.push(i);
        }
    }
    let mut out = Vec::new();
    for i in chosen {
        let (_, turns, mirrored, assignment) = &matches[i];
        let standing = Standing { sites: &sites, finders: &finders, rmqr, turns: *turns, mirrored: *mirrored, assignment };
        if let Some(found) = align_assignment(image, func, capstones, inverted, threshold, band, prename, standing) {
            out.push(found);
        }
    }
    out
}

/// How the capstones stand for one version's finders: the version's sites,
/// which of them are finders, and the turn, reflection and assignment of
/// capstones to finders under test.
#[derive(Clone, Copy)]
struct Standing<'a> {
    sites: &'a [Site],
    finders: &'a [usize],
    rmqr: bool,
    turns: u8,
    mirrored: bool,
    assignment: &'a [Option<usize>],
}

/// Place one version's outline under one assignment of the capstones to
/// its finders, as [`align_version`] describes.
#[allow(clippy::too_many_arguments)]
fn align_assignment(
    image: &Luma,
    func: &FunctionMap,
    capstones: &[Capstone],
    inverted: bool,
    threshold: Option<f32>,
    band: f32,
    prename: Option<Prename>,
    standing: Standing,
) -> Option<(Detection, f64)> {
    let (w, h) = (func.width as f64, func.height as f64);
    let Standing { sites, finders, rmqr, turns, mirrored, assignment } = standing;
    let mut placed: Vec<Placed> = Vec::new();
    let mut todo: Vec<usize> = (0..sites.len()).collect();
    for (f, &site_index) in finders.iter().enumerate() {
        if let Some(c) = assignment[f] {
            let site = &sites[site_index];
            // A capstone read along lines alone has no measured corners:
            // its centre is the one point it vouches for.
            let pairs = if capstones[c].line_only {
                vec![(site.at, capstones[c].center)]
            } else {
                capstone_points(&capstones[c], site, turns, mirrored, w, h)
            };
            placed.push(Placed {
                anchor: Anchor { kind: AnchorKind::Finder, at: site.at, center: capstones[c].center, partial: capstones[c].partial, inferred: false },
                pairs,
            });
            todo.retain(|&i| i != site_index);
        }
    }
    if placed.is_empty() {
        return None;
    }

    let nearest = |anchors: &[Anchor], at: [f64; 2]| anchors.iter().map(|a| distance(a.at, at)).fold(f64::INFINITY, f64::min);
    // The finders' own dark and light, for the locator of an alignment
    // pattern the template cannot be fitted to: BoofCV reads its
    // alignment patterns against the corner position pattern's threshold.
    let levels = match func.version {
        SymbolVersion::Qr(_) => finder_capstone_levels(image, capstones, assignment, threshold),
        _ => None,
    };
    let mut missed: Vec<Site> = Vec::new();
    while !todo.is_empty() {
        let (map, model) = fit_anchors(&mut placed)?;
        let anchors = anchors_of(&placed);
        let (k, far) = todo
            .iter()
            .enumerate()
            .map(|(k, &i)| (k, nearest(&anchors, sites[i].at)))
            .min_by(|a, b| a.1.total_cmp(&b.1))?;
        let site = sites[todo.swap_remove(k)];
        // The prediction drifts with distance from what is known, less so
        // once perspective is accounted for.
        let (rate, widest) = if model == Model::Projective { (0.03, WINDOW_PROJECTIVE) } else { (0.08, WINDOW_AFFINE) };
        let window = (0.5 + rate * far).clamp(0.75, widest);
        let local = local_map(map, &placed);
        if let Some((anchor, pairs)) = search_site(image, func, &site, &local, window, inverted, threshold) {
            placed.push(Placed { anchor, pairs });
        } else if site.kind == AnchorKind::Alignment {
            // A QR Code's alignment pattern the template does not fit is
            // looked for once more by its rings alone, as BoofCV looks
            // for every one of them.
            let located = levels.and_then(|(split, contrast)| {
                locate_alignment_by_meanshift(image, &site, &local, window, inverted, split, contrast)
            });
            match located {
                Some(center) => placed.push(Placed {
                    anchor: Anchor { kind: AnchorKind::Alignment, at: site.at, center, partial: false, inferred: false },
                    pairs: vec![(site.at, center)],
                }),
                None => missed.push(site),
            }
        }
    }
    // An alignment pattern the search did not find, blurred or stained
    // away, is placed where the lines through its found neighbours along
    // its row and down its column cross, as zxing-cpp's sampler places
    // one on its second pass. On flat paper that is where it is; on
    // curled paper it is nearer than the plane through every anchor, and
    // the map between the anchors would otherwise bend there through
    // nothing at all.
    for site in missed {
        if let Some(center) = neighbours_crossing(&site, &anchors_of(&placed)) {
            let anchor = Anchor { kind: AnchorKind::Alignment, at: site.at, center, partial: false, inferred: true };
            placed.push(Placed { anchor, pairs: vec![(site.at, center)] });
        }
    }

    let (mut map, _) = fit_anchors(&mut placed)?;
    // A timing anchor off the bend its line-mates trace has read a row of
    // data or a smudge: it is dropped, and the map refitted without it.
    let strays = timing_line_strays(&anchors_of(&placed), &map);
    if !strays.is_empty() {
        let mut drop: Vec<usize> = strays.iter().map(|(i, _)| *i).collect();
        drop.sort_unstable_by(|a, b| b.cmp(a));
        for i in drop {
            placed.remove(i);
        }
        map = fit_anchors(&mut placed)?.0;
    }
    let mut anchors = anchors_of(&placed);
    let outline = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
    // The corners come from the global fit, bent through the anchors
    // nearest them: a corner beside the end of a timing pattern follows
    // that end, where the plane through every anchor puts it a fraction
    // of a module off on curled paper, and a corner far from every anchor
    // follows the plane, since there is nothing else for it to follow.
    // The anchors themselves go into the detection, and the projection
    // built from them later bends the map between them while the corners
    // hold its edges.
    let local = local_map(map, &placed);
    let mut corners = outline.map(|[x, y]| local.apply(x, y));
    // Where a finder capstone stands at a corner of the symbol, its own
    // fit places that corner to a fraction of a pixel; the map through
    // every anchor is a plane, and on curled paper its corner is a
    // fraction of a module off. The capstone's corners went into the fit
    // as pairs, so the one at the symbol's corner is to hand.
    let mut measured_corners = Vec::new();
    for (i, corner) in outline.iter().enumerate() {
        let at_corner = placed
            .iter()
            .filter(|p| p.anchor.kind == AnchorKind::Finder)
            .flat_map(|p| p.pairs.iter())
            .find(|(at, _)| distance(*at, *corner) < 1e-9);
        if let Some(&(_, px)) = at_corner {
            corners[i] = px;
            measured_corners.push(i);
        }
    }

    let sign = if inverted { -1.0 } else { 1.0 };
    let mut loose: Vec<usize> = (0..4).filter(|&i| nearest(&anchors, outline[i]) > ANCHORED_WITHIN).collect();
    loose.sort_by(|&a, &b| nearest(&anchors, outline[a]).total_cmp(&nearest(&anchors, outline[b])));
    let mut edged_corners = Vec::new();
    // A corner no finder measured but an anchor holds follows the map
    // between that anchor and the edge; where both its edges trace, they
    // place it instead. First, so that a loose corner beside it starts
    // from the better place.
    for i in (0..4).filter(|i| !measured_corners.contains(i) && !loose.contains(i)) {
        if settle_traced_corner(image, func, &mut corners, i, sign, threshold) {
            edged_corners.push(i);
        }
    }
    // With one finder and nothing else but the timing modules beside it,
    // the map's perspective rests on anchors that all lie along two
    // edges from that finder, and runs wild at the far corner.
    let lone_finder = anchors.iter().filter(|a| a.kind != AnchorKind::Timing).count() == 1;
    // Twice over: each corner's search is truer once its neighbours have
    // moved.
    for pass in 0..2 {
        for (n, &i) in loose.iter().enumerate() {
            let before = corners;
            let mut edged = settle_corner(image, func, &mut corners, i, sign, threshold);
            if pass == 0 && !edged && lone_finder && n + 1 == loose.len() {
                // The edges were not found from where the map put the
                // corner: the corner opposite the one finder is tried
                // again from where the three corners now placed complete
                // the parallelogram, which on flat paper is near enough
                // for the edges to be traced from.
                let mut again = before;
                again[i] = std::array::from_fn(|axis| before[(i + 1) % 4][axis] + before[(i + 3) % 4][axis] - before[(i + 2) % 4][axis]);
                if settle_corner(image, func, &mut again, i, sign, threshold) {
                    corners = again;
                    edged = true;
                }
            }
            if edged && pass == 1 {
                edged_corners.push(i);
            }
        }
    }

    // Where an edge of the symbol bows, the paper curls, and between the
    // anchors and that edge the map is the spline's guess: points along
    // the bowed edge anchor it, so that the rows beside the edge follow
    // the paper and not the plane. A version 1 symbol on a bottle has
    // anchors on its timing patterns alone, and its far quadrant is
    // placed by nothing else.
    anchors.extend(edge_anchors(image, func, corners, &anchors, sign, threshold));
    let controls: Vec<ControlPoint> = anchors.iter().map(Anchor::control_point).collect();
    let projection = Projection::new(func.width, func.height, corners, &controls)?;
    let scorer = Scorer::new(image, &Template::whole(func), corners, threshold);
    let agreement = scorer.agreement_known(&projection, inverted)?.0;
    let around = [[-1.0, -1.0], [w + 1.0, -1.0], [w + 1.0, h + 1.0], [-1.0, h + 1.0]].map(|[x, y]| projection.apply(x, y));
    let quiet = Scorer::new(image, &Template::with_quiet_zone(func), around, threshold);
    let (matching, _) = quiet.agreement_known(&super::refine::Shifted { map: &projection, dx: -1.0, dy: -1.0 }, inverted)?;
    // Over every module of the pattern, not only those the photo covers:
    // a placement that runs off the photo is judged on the few modules
    // left, and reads as well as it pleases there.
    let fit = matching as f64 / quiet.fixed().max(1) as f64;

    let info = read_info(image, func, corners, inverted, threshold, band);
    let mut found = Detection {
        corners,
        version: func.version,
        inverted,
        mirrored,
        mirrored_by_decode: false,
        turns,
        anchors,
        agreement: (agreement, scorer.fixed()),
        info,
        edged_corners,
        measured_corners,
        retry: None,
        decodes: None,
    };
    // A square symbol seen from the back reads as its transpose, which
    // swaps the top-right and bottom-left corners. Its fixed patterns fit
    // either way (the version information, too, is its own transpose), so
    // the format bits decide, and only when they spell a word more
    // clearly through the mirror order than as is.
    if !rmqr {
        let swapped = [corners[0], corners[3], corners[2], corners[1]];
        let info = read_info(image, func, swapped, inverted, threshold, band);
        // The pre-read beside the finder, where the geometry is surest,
        // may already have read the word cleanly in mirror order.
        let mirrored_by_preread =
            prename.is_some_and(|p| p.strong && p.mirrored) && found.info.format.valid != Some(true);
        if bch::mirror_wins(found.info.format, info.format) || mirrored_by_preread {
            found.corners = swapped;
            found.mirrored = true;
            found.info = info;
            for anchor in &mut found.anchors {
                anchor.at = [anchor.at[1], anchor.at[0]];
            }
            found.edged_corners = found.edged_corners.iter().map(|&i| [0, 3, 2, 1][i]).collect();
            found.measured_corners = found.measured_corners.iter().map(|&i| [0, 3, 2, 1][i]).collect();
        }
    }
    Some((found, fit))
}

/// Settle every corner of an outline that no anchor or fixed pattern is
/// near, by the symbol's edge where it has a quiet zone and by its own
/// quarter's modules otherwise, with `anchors` holding the rest of the
/// outline in place. For an outline placed by hand, before refinement.
pub fn settle_loose_corners(
    image: &Luma,
    func: &FunctionMap,
    corners: [[f64; 2]; 4],
    anchors: &[Anchor],
    params: &SampleParams,
) -> [[f64; 2]; 4] {
    let (w, h) = (func.width as f64, func.height as f64);
    let threshold = fixed_threshold(params);
    // Both polarities, as the refinement tries them: the one with the
    // better fixed-pattern agreement is the photo's.
    let sign = {
        let scorer = Scorer::new(image, &Template::whole(func), corners, threshold);
        match Homography::rect_to_quad(w, h, corners) {
            Some(map) => {
                let (plain, flipped) = (scorer.agreement_known(&map, false), scorer.agreement_known(&map, true));
                if flipped.map(|a| a.0) > plain.map(|a| a.0) { -1.0 } else { 1.0 }
            }
            None => return corners,
        }
    };
    let anchored: Vec<[f64; 2]> = anchors.iter().map(|a| a.at).collect();
    settle_loose_corners_with(image, func, corners, &anchored, sign, params)
}

/// The corners of a `func` symbol that nothing holds: neither an anchor at
/// `anchored` (module coordinates) nor a fixed module lies within
/// [`ANCHORED_WITHIN`] modules. Fixed modules count as anchors here: a
/// corner with a finder, an alignment pattern or a timing pattern near it
/// is held by the refinement. The top-left finder and the separator and
/// format bits around it hold only their own corner, however near the far
/// corner of a small symbol they come, so they are left out.
pub(crate) fn loose_corners(func: &FunctionMap, anchored: &[[f64; 2]]) -> Vec<usize> {
    let (w, h) = (func.width as f64, func.height as f64);
    let outline = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
    let fixed_near = |corner: [f64; 2]| {
        let mut nearest = anchored.iter().map(|a| distance(*a, corner)).fold(f64::INFINITY, f64::min);
        for y in 0..func.height {
            for x in 0..func.width {
                if func.expected(x, y).is_some() && !(x < 9 && y < 9) {
                    nearest = nearest.min(distance([x as f64 + 0.5, y as f64 + 0.5], corner));
                }
            }
        }
        nearest
    };
    (0..4).filter(|&i| fixed_near(outline[i]) > ANCHORED_WITHIN).collect()
}

/// How far a fixed module may lie from a control point, in modules, and
/// still belong to the pattern the point holds: the ends of a strip of
/// timing modules lie three from its centre, the corner modules of an
/// alignment pattern a little under.
const CONTROL_REACH: f64 = 3.6;

/// The corners of a `func` symbol that the contrast of the fixed patterns
/// cannot place once the map passes through `controls`: every fixed
/// module within [`ANCHORED_WITHIN`] of the corner belongs to a pattern a
/// control point holds, which the map passes through wherever the corner
/// goes, so the corner moves the score not at all, and a descent of it
/// walks by noise. Such a corner keeps the place the detection gave it,
/// from the map bent through those very anchors.
pub(crate) fn corners_held_by_controls(func: &FunctionMap, controls: &[ControlPoint]) -> Vec<usize> {
    if controls.is_empty() {
        return Vec::new();
    }
    let (w, h) = (func.width as f64, func.height as f64);
    let outline = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
    let free_near = |corner: [f64; 2]| {
        (0..func.height).flat_map(|y| (0..func.width).map(move |x| (x, y))).any(|(x, y)| {
            let at = [x as f64 + 0.5, y as f64 + 0.5];
            func.expected(x, y).is_some()
                && !(x < 9 && y < 9)
                && distance(at, corner) <= ANCHORED_WITHIN
                && !controls.iter().any(|c| distance(c.at, at) <= CONTROL_REACH)
        })
    };
    (0..4).filter(|&i| !free_near(outline[i])).collect()
}

/// [`settle_loose_corners`] for a symbol of known polarity (`sign` is -1
/// for light on dark), with the anchors given by their module coordinates.
pub(crate) fn settle_loose_corners_with(
    image: &Luma,
    func: &FunctionMap,
    corners: [[f64; 2]; 4],
    anchored: &[[f64; 2]],
    sign: f64,
    params: &SampleParams,
) -> [[f64; 2]; 4] {
    let loose = loose_corners(func, anchored);
    if loose.is_empty() {
        return corners;
    }
    let threshold = fixed_threshold(params);
    let mut corners = corners;
    for _ in 0..2 {
        for &i in &loose {
            settle_corner(image, func, &mut corners, i, sign, threshold);
        }
    }
    corners
}

/// Move each anchor to where its own pattern reads best in the photo,
/// through the outline given by `corners` and the other anchors, by at
/// most [`REFINE_LIMIT`] modules. Anchors whose pattern cannot be found
/// are returned as they were.
pub fn refine_anchors(
    image: &Luma,
    func: &FunctionMap,
    corners: [[f64; 2]; 4],
    anchors: &[Anchor],
    params: &SampleParams,
) -> Vec<Anchor> {
    let threshold = fixed_threshold(params);
    let controls: Vec<ControlPoint> = anchors.iter().map(Anchor::control_point).collect();
    let Some(map) = Projection::new(func.width, func.height, corners, &controls) else { return anchors.to_vec() };
    let inverted = {
        let scorer = Scorer::new(image, &Template::whole(func), corners, threshold);
        let (plain, flipped) = (scorer.agreement_known(&map, false), scorer.agreement_known(&map, true));
        flipped.map(|a| a.0) > plain.map(|a| a.0)
    };
    let mut refined: Vec<Anchor> = anchors
        .iter()
        .enumerate()
        .map(|(k, anchor)| {
            let Some(site) = site_of(func, anchor) else { return *anchor };
            // Search a small window around where the anchor already is,
            // through the map the other anchors make: the map through
            // this one too passes through wherever it was put, and an
            // anchor dropped by hand half a module off bends it there,
            // which an anchor two modules away then holds against.
            let others: Vec<ControlPoint> = controls.iter().enumerate().filter(|(i, _)| *i != k).map(|(_, c)| *c).collect();
            let Some(map) = Projection::new(func.width, func.height, corners, &others) else { return *anchor };
            let shifted = |x: f64, y: f64| {
                let [px, py] = map.apply(x, y);
                let [ax, ay] = map.apply(anchor.at[0], anchor.at[1]);
                [px + anchor.center[0] - ax, py + anchor.center[1] - ay]
            };
            let local = ShiftedMap { f: &shifted };
            let limit = if anchor.kind == AnchorKind::Timing { TIMING_REFINE_LIMIT } else { REFINE_LIMIT };
            match search_site(image, func, &site, &local, SEARCH_STEP, inverted, threshold) {
                Some((found, _)) if distance(found.center, anchor.center) <= limit * module_size_at(&map, anchor.at) => found,
                _ => *anchor,
            }
        })
        .collect();
    // A timing anchor that still breaks the trend of its line is put
    // where the trend has it: that is what its neighbours say of the
    // paper there, and its own reading has gone astray.
    let pairs: Vec<Pair> = refined.iter().map(|a| (a.at, a.center)).collect();
    let corner_pairs: Vec<Pair> = [[0.0, 0.0], [func.width as f64, 0.0], [func.width as f64, func.height as f64], [0.0, func.height as f64]]
        .iter()
        .zip(corners)
        .map(|(at, px)| (*at, px))
        .collect();
    if let Some(base) = Homography::fit(Model::Projective, &pairs.iter().chain(&corner_pairs).copied().collect::<Vec<_>>()) {
        for (i, position) in timing_line_strays(&refined, &base) {
            refined[i].center = position;
        }
    }
    refined
}

/// A map given as a closure.
struct ShiftedMap<'a> {
    f: &'a dyn Fn(f64, f64) -> [f64; 2],
}

impl Project for ShiftedMap<'_> {
    fn base(&self, x: f64, y: f64) -> [f64; 2] {
        (self.f)(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finder_ratio_allows_half_a_module() {
        assert!(finder_ratio([10.0, 10.0, 30.0, 10.0, 10.0], RATIO_TOLERANCE));
        assert!(finder_ratio([8.0, 12.0, 32.0, 9.0, 11.0], RATIO_TOLERANCE));
        assert!(!finder_ratio([10.0, 10.0, 10.0, 10.0, 10.0], RATIO_TOLERANCE));
        assert!(!finder_ratio([10.0, 0.0, 30.0, 10.0, 10.0], RATIO_TOLERANCE));
        // A long outer run is other print the finder stands against, and
        // is allowed; a short one is not a finder.
        assert!(finder_ratio([10.0, 10.0, 30.0, 10.0, 17.0], RATIO_TOLERANCE));
        assert!(!finder_ratio([10.0, 10.0, 30.0, 10.0, 4.0], RATIO_TOLERANCE));
        // The diagonal is allowed more.
        assert!(!finder_ratio([10.0, 19.0, 30.0, 10.0, 10.0], RATIO_TOLERANCE));
        assert!(finder_ratio([10.0, 19.0, 30.0, 10.0, 10.0], DIAGONAL_TOLERANCE));
    }

    #[test]
    fn bloomed_runs_pass_by_their_pairs() {
        // Dark has bled 0.6 of a module into light: every dark run is 0.6
        // module short and every light run 0.6 long. Run by run that is
        // beyond the tolerance; pair by pair nothing has changed.
        let bloomed = [4.0, 16.0, 24.0, 16.0, 4.0];
        assert!(finder_ratio(bloomed, RATIO_TOLERANCE));
        // Light bled into dark passes the same way.
        assert!(finder_ratio([16.0, 4.0, 36.0, 4.0, 16.0], RATIO_TOLERANCE));
        // Not just anything with the right pair sums: the runs either side
        // of the centre must match, and the centre must be the widest.
        assert!(!finder_ratio([10.0, 19.0, 30.0, 10.0, 10.0], RATIO_TOLERANCE));
        assert!(!finder_ratio([10.0, 30.0, 10.0, 30.0, 10.0], RATIO_TOLERANCE));
    }

    #[test]
    fn cross_check_centres_the_middle_run() {
        // A finder column: light, then 1:1:3:1:1 at four pixels a module.
        let column: Vec<bool> = [false; 6]
            .into_iter()
            .chain([true; 4])
            .chain([false; 4])
            .chain([true; 12])
            .chain([false; 4])
            .chain([true; 4])
            .chain([false; 6])
            .collect();
        let at = |i: i64| column.get(usize::try_from(i).ok()?).copied();
        let (centre, module) = cross_check(at, 18, true, 4.0, RATIO_TOLERANCE, MODULE_FIT).unwrap();
        assert_eq!(module, 4.0);
        assert!((centre - 20.0).abs() < 1e-9, "{centre}");
        assert_eq!(cross_check(at, 18, true, 9.0, RATIO_TOLERANCE, MODULE_FIT), None, "far from the expected size");
        assert_eq!(cross_check(at, 2, true, 4.0, RATIO_TOLERANCE, MODULE_FIT), None, "not on the middle run");
        assert_eq!(cross_check(at, 18, false, 4.0, RATIO_TOLERANCE, MODULE_FIT), None, "wrong polarity");
        // Hard against other print the outer runs run on; the inner three
        // still say what the module is.
        let hemmed: Vec<bool> = [true; 9].into_iter().chain([false; 4]).chain([true; 12]).chain([false; 4]).chain([true; 9]).collect();
        let at = |i: i64| hemmed.get(usize::try_from(i).ok()?).copied();
        let (centre, module) = cross_check(at, 19, true, 4.0, RATIO_TOLERANCE, MODULE_FIT).unwrap();
        assert!((module - 4.0).abs() < 1e-9 && (centre - 19.0).abs() < 1e-9, "{centre} {module}");
        assert_eq!(finder_module([9.0, 4.0, 12.0, 4.0, 9.0], RATIO_TOLERANCE), Some(4.0));
        assert_eq!(finder_module([9.0, 4.0, 12.0, 9.0, 9.0], RATIO_TOLERANCE), None, "inner runs must still fit");
    }

    #[test]
    fn a_cross_passes_the_row_and_column_but_not_the_diagonal() {
        // A dark seven-module square with a light cross cut through it two
        // modules out from the centre, 4 px a module, on bare paper: the
        // row and the column through the centre both read 1:1:3:1:1, but
        // the diagonal meets solid dark out to the edge.
        // Module index 0 to 6 along an axis, and the ring it lies on.
        let ring = |d: i64| ((d + 14) / 4 - 3).abs();
        let cross = |x: i64, y: i64| {
            let (dx, dy) = (x - 20, y - 20);
            let inside = (-14..14).contains(&dx) && (-14..14).contains(&dy);
            let cut = (ring(dx) == 2 && ring(dy) == 0) || (ring(dy) == 2 && ring(dx) == 0);
            (x >= 0 && y >= 0 && x < 40 && y < 40).then_some(inside && !cut)
        };
        assert!(cross_check(|i| cross(20, i), 20, true, 4.0, RATIO_TOLERANCE, MODULE_FIT).is_some());
        assert!(cross_check(|i| cross(i, 20), 20, true, 4.0, RATIO_TOLERANCE, MODULE_FIT).is_some());
        let diagonal = |i: i64| cross(20 + (i - 20), i);
        assert_eq!(cross_check(diagonal, 20, true, 4.0, DIAGONAL_TOLERANCE, DIAGONAL_FIT), None);
        // A real finder passes the diagonal: modules 0 to 6 along each
        // axis, dark except on the ring one in from the edge.
        let module_ring = |v: i64| (v / 4 - 3).abs();
        let finder = |x: i64, y: i64| {
            (x >= 0 && y >= 0 && x < 28 && y < 28).then(|| module_ring(x).max(module_ring(y)) != 2)
        };
        let diagonal = |i: i64| finder(14 + (i - 14), i);
        let (centre, module) = cross_check(diagonal, 14, true, 4.0, DIAGONAL_TOLERANCE, DIAGONAL_FIT).unwrap();
        assert!((centre - 14.0).abs() < 1.0 && (module - 4.0).abs() < 0.5, "{centre} {module}");
    }

    #[test]
    fn units_turn_with_the_symbol() {
        let close = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9;
        // The symbol's top-left corner, seen in an outline it is turned in.
        assert!(close(to_unit(0, false, 21.0, 21.0, [0.0, 0.0]), [0.0, 0.0]));
        assert!(close(to_unit(1, false, 21.0, 21.0, [0.0, 0.0]), [1.0, 0.0]));
        assert!(close(to_unit(2, false, 21.0, 21.0, [0.0, 0.0]), [1.0, 1.0]));
        assert!(close(to_unit(3, false, 21.0, 21.0, [0.0, 0.0]), [0.0, 1.0]));
        // Its top-right corner follows a quarter turn behind.
        assert!(close(to_unit(1, false, 21.0, 21.0, [21.0, 0.0]), [1.0, 1.0]));
        // Mirrored, the top-left corner is seen at the top right, and the
        // turns go on from there.
        assert!(close(to_unit(0, true, 43.0, 13.0, [0.0, 0.0]), [1.0, 0.0]));
        assert!(close(to_unit(2, true, 43.0, 13.0, [0.0, 0.0]), [0.0, 1.0]));
        assert!(close(to_unit(0, true, 43.0, 13.0, [43.0, 13.0]), [0.0, 1.0]));
    }

    #[test]
    fn capstones_of_different_sizes_are_kept_apart() {
        let capstone = |unit: [f64; 2], module: f64| Capstone {
            measured: true,
            corners: [[0.0; 2]; 4],
            center: [0.0; 2],
            unit,
            module: [module, module],
            contrast: 1.0,
            inverted: false,
            partial: false,
            threshold: 0.5,
            line_only: false,
        };
        let finders = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]];
        // Three alike: all matched.
        let alike = [capstone([0.12, 0.1], 8.0), capstone([0.9, 0.08], 8.2), capstone([0.1, 0.92], 7.9)];
        let (cost, assignment) = assign(&alike, &finders);
        assert_eq!(assignment, [Some(0), Some(1), Some(2)]);
        assert!(cost < 0.01, "{cost}");
        // One twice the size, where a finder should be: dearer to take
        // than to leave the finder unmatched.
        let stranger = [capstone([0.12, 0.1], 8.0), capstone([0.9, 0.08], 16.5), capstone([0.1, 0.92], 7.9)];
        let (_, assignment) = assign(&stranger, &finders);
        assert_eq!(assignment, [Some(0), None, Some(2)]);
    }

    #[test]
    fn versions_near_the_estimate_are_tried_with_the_current_one_first() {
        let tried = candidate_versions(SymbolVersion::Qr(2), [58.0, 58.0], true);
        assert_eq!(tried, [SymbolVersion::Qr(10), SymbolVersion::Qr(11)]);
        let tried = candidate_versions(SymbolVersion::Qr(10), [60.0, 60.0], false);
        assert_eq!(tried[0], SymbolVersion::Qr(10));
        assert_eq!(tried.len(), 3);
        let tried = candidate_versions(SymbolVersion::Rmqr(0), [80.0, 13.5], false);
        assert_eq!(tried[0], SymbolVersion::Rmqr(19), "R13x77 is nearest");
        assert_eq!(tried.len(), VERSIONS_LOOSE_RMQR);
        assert!(tried.iter().all(|v| v.family() == Family::Rmqr));
        assert!(!tried.contains(&SymbolVersion::Rmqr(0)), "R7x43 is far off");
        // The estimate is along the frame's axes: a symbol on end in the
        // frame is the same symbol, and the same versions are tried.
        let on_end = candidate_versions(SymbolVersion::Rmqr(0), [13.5, 80.0], false);
        assert_eq!(on_end, tried);
        assert_eq!(candidate_versions(SymbolVersion::Rmqr(19), [13.2, 77.4], true)[0], SymbolVersion::Rmqr(19), "the version in use, when near");
        assert_eq!(size_error(SymbolVersion::Rmqr(19), [77.0, 13.0]), 0.0);
        assert_eq!(size_error(SymbolVersion::Rmqr(19), [13.0, 77.0]), 0.0);
        let more = further_versions(Family::Rmqr, [13.5, 80.0], &tried);
        assert!(!more.is_empty() && more.iter().all(|v| !tried.contains(v)));
        assert!(size_error(more[0], [80.0, 13.5]) >= size_error(*tried.last().unwrap(), [80.0, 13.5]));
    }

    #[test]
    fn model_grows_with_the_anchors() {
        let anchor = |x: f64, y: f64| Anchor { kind: AnchorKind::Alignment, at: [x, y], center: [0.0, 0.0], partial: false, inferred: false };
        assert_eq!(choose_model(&[anchor(3.5, 3.5)], 1), Model::Similarity);
        assert_eq!(choose_model(&[anchor(3.5, 3.5)], 4), Model::Affine);
        assert_eq!(choose_model(&[anchor(3.5, 3.5), anchor(50.0, 3.5)], 8), Model::Affine);
        let line = [anchor(3.5, 3.5), anchor(20.0, 3.5), anchor(50.0, 3.5)];
        assert_eq!(choose_model(&line, 12), Model::Affine);
        let triangle = [anchor(3.5, 3.5), anchor(50.0, 3.5), anchor(3.5, 50.0)];
        assert_eq!(choose_model(&triangle, 12), Model::Projective);
        assert_eq!(choose_model(&triangle, 6), Model::Affine, "too few points");
    }

    #[test]
    fn sites_cover_every_family() {
        let qr = anchor_sites(&FunctionMap::new(SymbolVersion::Qr(7)));
        assert_eq!(qr.iter().filter(|s| s.kind == AnchorKind::Finder).count(), 3);
        assert_eq!(qr.iter().filter(|s| s.kind == AnchorKind::Alignment).count(), 6);
        assert_eq!(qr[0].at, [3.5, 3.5]);
        // Timing sites: a few along each pattern, with both ends kept, and
        // none whose window reaches into the alignment patterns that sit
        // on the timing line of this version.
        let func7 = FunctionMap::new(SymbolVersion::Qr(7));
        let timing: Vec<[f64; 2]> = qr.iter().filter(|s| s.kind == AnchorKind::Timing).map(|s| s.at).collect();
        assert!((8..=16).contains(&timing.len()), "{timing:?}");
        for at in [[8.5, 6.5], [36.5, 6.5], [6.5, 8.5], [6.5, 36.5]] {
            assert!(timing.contains(&at), "{at:?} not among {timing:?}");
        }
        for at in &timing {
            assert_eq!(func7.role(at[0] as usize, at[1] as usize), Role::Timing, "{at:?}");
            assert_eq!(func7.expected(at[0] as usize, at[1] as usize), Some(true), "{at:?}");
        }
        let micro = anchor_sites(&FunctionMap::new(SymbolVersion::Micro(2)));
        assert_eq!(micro.iter().filter(|s| s.kind == AnchorKind::Finder).count(), 1);
        let timing: Vec<[f64; 2]> = micro.iter().filter(|s| s.kind == AnchorKind::Timing).map(|s| s.at).collect();
        assert_eq!(timing, [[8.5, 0.5], [10.5, 0.5], [12.5, 0.5], [0.5, 8.5], [0.5, 10.5], [0.5, 12.5]]);
        let model1 = anchor_sites(&FunctionMap::new(SymbolVersion::Model1(2)));
        assert!(model1.iter().filter(|s| s.kind == AnchorKind::Timing).count() >= 6);
        // Edge points by hand: three along each edge of a square symbol,
        // on its outline, clear of the corners; a short rMQR edge keeps its
        // midpoint alone.
        let edges = edge_sites(&FunctionMap::new(SymbolVersion::Qr(1)));
        assert_eq!(edges.len(), 12);
        assert!(edges.iter().all(|s| s.kind == AnchorKind::Edge));
        assert_eq!(edges[1].at, [10.5, 0.0]);
        assert!(edges.iter().all(|s| s.at[0] == 0.0 || s.at[0] == 21.0 || s.at[1] == 0.0 || s.at[1] == 21.0));
        let short = edge_sites(&FunctionMap::new(SymbolVersion::Rmqr(0)));
        assert_eq!(short.len(), 8, "{short:?}");
        assert!(short.iter().filter(|s| s.at[0] == 0.0 || s.at[0] == 43.0).all(|s| s.at[1] == 3.5));
        let rmqr = anchor_sites(&FunctionMap::new(SymbolVersion::Rmqr(11)));
        assert_eq!(rmqr.iter().filter(|s| s.kind == AnchorKind::FinderSub).count(), 1);
        assert_eq!(rmqr.iter().filter(|s| s.kind == AnchorKind::Alignment).count(), 2);
    }

    #[test]
    fn a_bent_edge_gives_its_tangent_at_the_corner() {
        // Crossings along a bowed edge y = x²/4000 from x = 0 to 200, as
        // a curled label bows, with one stray: the tangent at the origin
        // is the x axis, and at the far end it climbs at a slope of a
        // tenth.
        let mut points: Vec<[f64; 2]> = (0..=20).map(|i| [i as f64 * 10.0, (i as f64 * 10.0).powi(2) / 4000.0]).collect();
        points.push([100.0, 8.0]);
        let (tangent, rms, bow, _) = robust_curve(&points, [0.0, 0.0], 0.3, 20.0).unwrap();
        assert!(distance(tangent.point, [0.0, 0.0]) < 0.3, "{tangent:?}");
        assert!(tangent.dir[1].abs() < 0.01, "{tangent:?}");
        assert!(rms < 0.3 && (bow - 2.5).abs() < 0.3, "rms {rms} bow {bow}");
        let (far, _, _, _) = robust_curve(&points, [200.0, 10.0], 0.3, 20.0).unwrap();
        assert!((far.dir[1] / far.dir[0] - 0.1).abs() < 0.01, "{far:?}");
        // A bow deeper than allowed, a fit worse than allowed, or too few
        // points are refused.
        assert!(robust_curve(&points, [0.0, 0.0], 0.3, 1.0).is_none());
        assert!(robust_curve(&points, [0.0, 0.0], 0.0001, 20.0).is_none());
        assert!(robust_curve(&points[..4], [0.0, 0.0], 0.3, 20.0).is_none());
        // A gentle bow, as paper curls: a hundredth of the length.
        let gentle: Vec<[f64; 2]> = (0..=30).map(|i| [i as f64 * 10.0, (i as f64 * 10.0 - 150.0).powi(2) / 7500.0]).collect();
        let (t, _, _, _) = robust_curve(&gentle, [0.0, 3.0], 0.2, 100.0).unwrap();
        assert!((t.dir[1] / t.dir[0] + 0.04).abs() < 0.005, "{t:?}");
    }

    #[test]
    fn lines_fit_intersect_and_project() {
        let points = [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [1.5, 6.0]];
        let (line, rms, kept) = robust_line(&points).unwrap();
        assert_eq!(kept, 4, "the stray point is dropped");
        assert!(rms < 1e-9, "{rms}");
        assert!(line.distance([5.0, 6.0]) < 1e-9);
        let other = Line { point: [0.0, 4.0], dir: [1.0, 0.0] };
        let p = line.intersect(&other).unwrap();
        assert!((p[0] - 3.0).abs() < 1e-9 && (p[1] - 4.0).abs() < 1e-9, "{p:?}");
        let q = other.project([7.0, 9.0]);
        assert_eq!(q, [7.0, 4.0]);
        assert!(Line::fit(&points[..1]).is_none());
    }

    #[test]
    fn edges_run_away_from_their_corner() {
        let [h, v] = edges_at(2, 21.0, 25.0);
        assert_eq!(h.corner, [21.0, 25.0]);
        assert_eq!(h.along, [-1.0, 0.0]);
        assert_eq!(h.inward, [0.0, -1.0]);
        assert_eq!(v.along, [0.0, -1.0]);
        assert_eq!(v.inward, [-1.0, 0.0]);
        let [h, _] = edges_at(0, 21.0, 25.0);
        assert_eq!((h.along, h.inward), ([1.0, 0.0], [0.0, 1.0]));
    }

    /// A symbol's fixed patterns drawn turned `degrees` clockwise about
    /// `centre` at `module_px` pixels a module, data modules mid grey.
    fn draw_turned(version: SymbolVersion, centre: [f64; 2], module_px: f64, degrees: f64, size: (usize, usize)) -> Luma {
        let func = FunctionMap::new(version);
        let (s, c) = degrees.to_radians().sin_cos();
        let (w, h) = (func.width as f64, func.height as f64);
        let mut data = Vec::with_capacity(size.0 * size.1);
        for py in 0..size.1 {
            for px in 0..size.0 {
                let mut sum = 0.0;
                for sy in 0..3 {
                    for sx in 0..3 {
                        let (x, y) = (px as f64 + (sx as f64 + 0.5) / 3.0 - centre[0], py as f64 + (sy as f64 + 0.5) / 3.0 - centre[1]);
                        let (ux, uy) = (x * c + y * s, -x * s + y * c);
                        let (mx, my) = (ux / module_px + w / 2.0, uy / module_px + h / 2.0);
                        let inside = mx >= 0.0 && my >= 0.0 && mx < w && my < h;
                        sum += match inside.then(|| func.expected(mx as usize, my as usize)).flatten() {
                            Some(true) => 0.08f32,
                            Some(false) => 0.92,
                            None if inside => 0.5,
                            None => 0.92,
                        };
                    }
                }
                data.push(sum / 9.0);
            }
        }
        Luma::new(size.0, size.1, data)
    }

    /// A box around a symbol drawn by `draw_turned`.
    fn box_around(centre: [f64; 2], modules: f64, module_px: f64, degrees: f64) -> [[f64; 2]; 4] {
        let half = modules * module_px / 2.0;
        let (s, c) = degrees.to_radians().sin_cos();
        let reach = half * (c.abs() + s.abs()) * 1.15;
        [
            [centre[0] - reach, centre[1] - reach],
            [centre[0] + reach, centre[1] - reach],
            [centre[0] + reach, centre[1] + reach],
            [centre[0] - reach, centre[1] + reach],
        ]
    }

    #[test]
    fn timing_pattern_between_finders_counts_the_size() {
        // The count along the timing pattern names the size to the
        // module, straight or turned, where the spacing estimate rests on
        // the module size the capstone fits measured.
        let centre = [380.0, 320.0];
        for (version, module_px) in [(SymbolVersion::Qr(1), 14.0), (SymbolVersion::Qr(3), 10.0), (SymbolVersion::Qr(6), 7.0)] {
            for degrees in [0.0, 20.0] {
                let image = draw_turned(version, centre, module_px, degrees, (760, 640));
                let region = box_around(centre, version.width() as f64, module_px, degrees);
                let rect = Rectified::new(&image, region).unwrap();
                let caps: Vec<Capstone> = collect_capstones(&image, &rect, None, Family::Qr).into_iter().filter(|c| !c.inverted).collect();
                let (hint, tight) = size_hint(&image, &rect.frame, &caps, Family::Qr);
                assert!(tight, "{version} at {degrees}°");
                assert_eq!(hint, [version.width() as f64; 2], "{version} at {degrees}°: {caps:?}");
            }
        }
    }

    #[test]
    fn the_timing_row_is_placed_by_each_finders_own_modules() {
        // A symbol seen from below is foreshortened down the photo and
        // not across it: seven pixels a module one way and five the
        // other, as the far finders of a sheet of paper on a desk are
        // drawn. Three modules in from the line between the top finders
        // is fifteen pixels, and by the mean of the two modules eighteen,
        // which is half a module past the timing row and in the data.
        let version = SymbolVersion::Qr(4);
        let func = FunctionMap::new(version);
        let (module, origin) = ([7.3, 5.1], [60.4, 50.3]);
        let (width, height) = (360, 270);
        let mut data = Vec::with_capacity(width * height);
        for py in 0..height {
            for px in 0..width {
                let mut sum = 0.0;
                for sy in 0..3 {
                    for sx in 0..3 {
                        let mx = (px as f64 + (sx as f64 + 0.5) / 3.0 - origin[0]) / module[0];
                        let my = (py as f64 + (sy as f64 + 0.5) / 3.0 - origin[1]) / module[1];
                        let inside = mx >= 0.0 && my >= 0.0 && mx < func.width as f64 && my < func.height as f64;
                        sum += match inside.then(|| func.expected(mx as usize, my as usize)).flatten() {
                            Some(true) => 0.08f32,
                            Some(false) => 0.92,
                            None if inside => 0.5,
                            None => 0.92,
                        };
                    }
                }
                data.push(sum / 9.0);
            }
        }
        let image = Luma::new(width, height, data);
        let whole = [[0.0, 0.0], [width as f64, 0.0], [width as f64, height as f64], [0.0, height as f64]];
        let rect = Rectified::with_longest(&image, whole, MAX_RECTIFIED_WHOLE).unwrap();
        let caps: Vec<Capstone> = collect_capstones(&image, &rect, None, Family::Qr).into_iter().filter(|c| !c.inverted).collect();
        assert_eq!(caps.len(), 3, "{caps:?}");
        assert!(caps.iter().all(|c| c.measured && (c.module[0] / c.module[1] - 7.3 / 5.1).abs() < 0.15), "{caps:?}");
        let at = |x: f64, y: f64| {
            let want = [origin[0] + x * module[0], origin[1] + y * module[1]];
            caps.iter().find(|c| distance(c.center, want) < 3.0).unwrap_or_else(|| panic!("no capstone at {want:?}: {caps:?}"))
        };
        let (tl, tr, bl) = (at(3.5, 3.5), at(29.5, 3.5), at(3.5, 29.5));
        for (other, third) in [(tr, bl), (bl, tr)] {
            let count = timing_between(&image, &rect.frame, tl, other, rect.frame.to_rectified(third.center)).expect("the timing pattern reads");
            assert!(count.exact && count.size == 33.0, "{count:?}");
        }
        // Neither side is charged for a pattern that did not read, and
        // the size is the count's.
        let proposals = qr_proposals(&image, &rect.frame, &caps);
        assert_eq!(proposals.len(), 1);
        assert!(proposals[0].0 < TIMING_PENALTY, "{}", proposals[0].0);
        assert_eq!(size_hint(&image, &rect.frame, &caps, Family::Qr), ([33.0, 33.0], true));
    }

    #[test]
    fn a_lone_finder_walks_its_timing_patterns() {
        // A Micro QR symbol's timing patterns run along its edges to the
        // far corners: counting them names the version. An rMQR symbol's
        // are broken by alignment patterns, and the walk to the quiet zone
        // gives its width and height to a module or so.
        let centre = [380.0, 320.0];
        for (version, module_px, degrees) in [(SymbolVersion::Micro(2), 12.0, 0.0), (SymbolVersion::Micro(4), 10.0, 25.0), (SymbolVersion::Micro(3), 12.0, 0.0)] {
            let image = draw_turned(version, centre, module_px, degrees, (760, 640));
            let region = box_around(centre, version.width() as f64, module_px, degrees);
            let rect = Rectified::new(&image, region).unwrap();
            // The finder's rings read as a light-centred finder too, at
            // another scale; the fit sorts the polarities out later.
            let caps: Vec<Capstone> = collect_capstones(&image, &rect, None, Family::Micro).into_iter().filter(|c| !c.inverted).collect();
            assert_eq!(caps.len(), 1, "{version}: {caps:?}");
            let (hint, tight) = size_hint(&image, &rect.frame, &caps, Family::Micro);
            assert!(tight, "{version} at {degrees}°");
            assert_eq!(hint, [version.width() as f64; 2], "{version} at {degrees}°");
        }
        let version = SymbolVersion::Rmqr(10);
        assert_eq!((version.width(), version.height()), (27, 11));
        let image = draw_turned(version, centre, 12.0, 0.0, (760, 640));
        let half = [27.0 * 12.0 / 2.0 * 1.15, 11.0 * 12.0 / 2.0 * 1.4];
        let region = [
            [centre[0] - half[0], centre[1] - half[1]],
            [centre[0] + half[0], centre[1] - half[1]],
            [centre[0] + half[0], centre[1] + half[1]],
            [centre[0] - half[0], centre[1] + half[1]],
        ];
        let rect = Rectified::new(&image, region).unwrap();
        let caps: Vec<Capstone> = collect_capstones(&image, &rect, None, Family::Rmqr).into_iter().filter(|c| !c.inverted).collect();
        let (hint, _) = size_hint(&image, &rect.frame, &caps, Family::Rmqr);
        assert!((hint[0] - 27.0).abs() <= 1.0 && (hint[1] - 11.0).abs() <= 1.0, "{hint:?} from {caps:?}");
    }

    #[test]
    fn a_lone_finder_sizes_a_symbol_on_end_or_no_taller_than_itself() {
        let centre = [380.0, 320.0];
        let box_of = |along: f64, down: f64, module_px: f64| {
            let half = [along * module_px / 2.0 * 1.15, down * module_px / 2.0 * 1.4];
            [
                [centre[0] - half[0], centre[1] - half[1]],
                [centre[0] + half[0], centre[1] - half[1]],
                [centre[0] + half[0], centre[1] + half[1]],
                [centre[0] - half[0], centre[1] + half[1]],
            ]
        };
        let hint_of = |version: SymbolVersion, module_px: f64, degrees: f64, along: f64, down: f64| {
            let image = draw_turned(version, centre, module_px, degrees, (760, 640));
            let rect = Rectified::new(&image, box_of(along, down, module_px)).unwrap();
            let caps: Vec<Capstone> = collect_capstones(&image, &rect, None, Family::Rmqr).into_iter().filter(|c| !c.inverted).collect();
            assert_eq!(caps.len(), 1, "{version} at {degrees}°: {caps:?}");
            size_hint(&image, &rect.frame, &caps, Family::Rmqr).0
        };
        let near = |hint: [f64; 2], want: [f64; 2]| (hint[0] - want[0]).abs() <= 1.0 && (hint[1] - want[1]).abs() <= 1.0;
        // An R7 symbol is as tall as its finder: no separator lies below
        // it and no timing pattern runs down beside it, and the paper on
        // both sides says the symbol is seven modules that way.
        let r7 = SymbolVersion::Rmqr(0);
        assert_eq!((r7.width(), r7.height()), (43, 7));
        let hint = hint_of(r7, 12.0, 0.0, 43.0, 7.0);
        assert!(near(hint, [43.0, 7.0]), "{hint:?}");
        assert_eq!(candidate_versions(SymbolVersion::Rmqr(11), hint, false)[0], r7);
        // On end in the frame the extents come out the other way round,
        // and the versions are matched to them either way.
        let r11 = SymbolVersion::Rmqr(10);
        let hint = hint_of(r11, 12.0, 90.0, 11.0, 27.0);
        assert!(near(hint, [11.0, 27.0]), "{hint:?}");
        assert_eq!(candidate_versions(SymbolVersion::Rmqr(0), hint, false)[0], r11);
        let hint = hint_of(r7, 12.0, 270.0, 7.0, 43.0);
        assert!(near(hint, [7.0, 43.0]), "{hint:?}");
        assert_eq!(candidate_versions(SymbolVersion::Rmqr(11), hint, false)[0], r7);
    }

    #[test]
    fn a_lone_qr_finder_counts_its_timing_pattern_up_to_the_far_finder() {
        // The walk along row and column six from the top-left finder ends
        // at the far finders, whose separators are a module wide, as the
        // timing modules are, and are not to be counted with them.
        let centre = [380.0, 320.0];
        for (version, module_px) in [(SymbolVersion::Qr(1), 14.0), (SymbolVersion::Qr(5), 9.0), (SymbolVersion::Model1(2), 12.0)] {
            let image = draw_turned(version, centre, module_px, 0.0, (760, 640));
            let region = box_around(centre, version.width() as f64, module_px, 0.0);
            let rect = Rectified::new(&image, region).unwrap();
            let caps: Vec<Capstone> = collect_capstones(&image, &rect, None, version.family()).into_iter().filter(|c| !c.inverted).collect();
            let top_left = caps.iter().min_by(|a, b| (a.center[0] + a.center[1]).total_cmp(&(b.center[0] + b.center[1]))).unwrap();
            let n = version.width() as f64;
            assert_eq!(lone_finder_size(&image, top_left, version.family()), Some(([n, n], true)), "{version}");
        }
    }

    #[test]
    fn capstones_are_fitted_at_any_turn() {
        // A symbol at any angle in a box square to the photo: every finder
        // is fitted, to a fraction of a pixel, with its modules measured
        // right, including at forty-five degrees, where the template
        // square to the box is as near one fit as another.
        let version = SymbolVersion::Qr(4);
        let centre = [380.0, 320.0];
        let module_px = 12.0;
        for degrees in [0.0, 15.0, 22.5, 30.0, 45.0, 60.0, 75.0] {
            let image = draw_turned(version, centre, module_px, degrees, (760, 640));
            let half = 33.0 * module_px / 2.0;
            let (s, c) = degrees.to_radians().sin_cos();
            let reach = half * (c.abs() + s.abs()) * 1.1;
            let region = [
                [centre[0] - reach, centre[1] - reach],
                [centre[0] + reach, centre[1] - reach],
                [centre[0] + reach, centre[1] + reach],
                [centre[0] - reach, centre[1] + reach],
            ];
            let rect = Rectified::new(&image, region).unwrap();
            let caps = collect_capstones(&image, &rect, None, Family::Qr);
            let truth = [[3.5, 3.5], [29.5, 3.5], [3.5, 29.5]].map(|[mx, my]| {
                let (ux, uy) = ((mx - 16.5) * module_px, (my - 16.5) * module_px);
                [centre[0] + ux * c - uy * s, centre[1] + ux * s + uy * c]
            });
            for t in truth {
                let found = caps
                    .iter()
                    .filter(|cap| !cap.inverted && !cap.partial)
                    .find(|cap| distance(cap.center, t) < 0.5)
                    .unwrap_or_else(|| panic!("{degrees}°: no capstone at {t:?} among {caps:?}"));
                assert!((found.module_mean() - module_px).abs() < 0.3, "{degrees}°: module {:?}", found.module);
                // The corners come in the frame's order: nearest the top
                // left first, then clockwise.
                let r = found.corners.map(|p| rect.to_rectified(p));
                let sums: Vec<f64> = r.iter().map(|p| p[0] + p[1]).collect();
                assert!(sums[0] <= sums[1] + 1e-6 && sums[0] <= sums[3] + 1e-6, "{degrees}°: {r:?}");
                let cross = (r[1][0] - r[0][0]) * (r[2][1] - r[1][1]) - (r[1][1] - r[0][1]) * (r[2][0] - r[1][0]);
                assert!(cross > 0.0, "{degrees}°: corners not clockwise: {r:?}");
            }
        }
    }

    /// A capstone with measured corners: a square `module` pixels a
    /// module about `center`, turned clockwise by `degrees`.
    fn square_capstone(center: [f64; 2], module: f64, degrees: f64, frame: &Frame) -> Capstone {
        let (s, c) = degrees.to_radians().sin_cos();
        let half = 3.5 * module;
        let corners = [[-half, -half], [half, -half], [half, half], [-half, half]]
            .map(|[dx, dy]| [center[0] + dx * c - dy * s, center[1] + dx * s + dy * c]);
        Capstone {
            corners: order_in_frame(corners, frame),
            center,
            unit: frame.unit(center),
            module: [module, module],
            contrast: 1.0,
            inverted: false,
            partial: false,
            threshold: 0.5,
            line_only: false,
            measured: true,
        }
    }

    #[test]
    fn finders_face_as_a_symbols_finders_do() {
        let frame = Frame::new([[0.0, 0.0], [800.0, 0.0], [800.0, 800.0], [0.0, 800.0]], MAX_RECTIFIED).unwrap();
        let a = square_capstone([100.0, 100.0], 8.0, 0.0, &frame);
        // Along a row, and along a column.
        assert!(finders_face(&a, &square_capstone([300.0, 100.0], 8.0, 0.0, &frame), &frame));
        assert!(finders_face(&a, &square_capstone([100.0, 300.0], 8.0, 0.0, &frame), &frame));
        // A gentle perspective tilts the line a little off the middle of
        // the sides, and the sides a little off square to it.
        assert!(finders_face(&a, &square_capstone([300.0, 160.0], 8.0, 12.0, &frame), &frame));
        // Turned thirty-five degrees against its partner, past the thirty
        // allowed: not one symbol's finders.
        assert!(!finders_face(&a, &square_capstone([300.0, 100.0], 8.0, 35.0, &frame), &frame));
        // Across the diagonal the line leaves each square by a corner.
        assert!(!finders_face(&a, &square_capstone([300.0, 300.0], 8.0, 0.0, &frame), &frame));
        // Far enough off the row that the line crosses near a corner.
        assert!(!finders_face(&a, &square_capstone([300.0, 250.0], 8.0, 0.0, &frame), &frame));
        // Corners never measured say nothing against a partner.
        let unmeasured = Capstone { measured: false, ..square_capstone([300.0, 100.0], 8.0, 35.0, &frame) };
        assert!(finders_face(&a, &unmeasured, &frame));
    }

    #[test]
    fn a_triple_with_a_finder_inside_proposes_nothing() {
        let frame = Frame::new([[0.0, 0.0], [800.0, 0.0], [800.0, 800.0], [0.0, 800.0]], MAX_RECTIFIED).unwrap();
        let image = Luma::new(800, 800, vec![0.92; 800 * 800]);
        // Three finders at the spacing of a version 4 symbol, ten pixels
        // a module: one proposal, centred on the symbol.
        let trio = |frame: &Frame| {
            vec![square_capstone([200.0, 200.0], 10.0, 0.0, frame), square_capstone([460.0, 200.0], 10.0, 0.0, frame), square_capstone([200.0, 460.0], 10.0, 0.0, frame)]
        };
        let proposals = qr_proposals(&image, &frame, &trio(&frame));
        assert_eq!(proposals.len(), 1);
        assert!(distance(quad_centre(proposals[0].1), [330.0, 330.0]) < 1e-6, "{proposals:?}");
        // A fourth finder of the same size inside the symbol they would
        // make: it is another symbol's, and nothing is proposed.
        let mut inside = trio(&frame);
        inside.push(square_capstone([330.0, 330.0], 10.0, 0.0, &frame));
        assert!(qr_proposals(&image, &frame, &inside).is_empty());
        // A partial pattern inside, three quarters of a finder read in the
        // data, counts for nothing against the triple.
        let mut torn = trio(&frame);
        torn.push(Capstone { partial: true, ..square_capstone([330.0, 330.0], 10.0, 0.0, &frame) });
        assert_eq!(qr_proposals(&image, &frame, &torn).len(), 1);
        // A finder beyond the symbol leaves the proposal alone.
        let mut beyond = trio(&frame);
        beyond.push(square_capstone([640.0, 200.0], 10.0, 0.0, &frame));
        assert!(qr_proposals(&image, &frame, &beyond).iter().any(|(_, p)| distance(quad_centre(*p), [330.0, 330.0]) < 1e-6));
        // One of the three turned thirty-five degrees against the others:
        // no symbol's finders stand so, and nothing is proposed.
        let mut turned = trio(&frame);
        turned[1] = square_capstone([460.0, 200.0], 10.0, 35.0, &frame);
        assert!(qr_proposals(&image, &frame, &turned).is_empty());
    }

    #[test]
    fn a_sheet_of_symbols_proposes_each_alone() {
        // Four version 2 symbols in a square, eight pixels a module and
        // ten modules apart: the top-left finder of one, the top-right of
        // the next and the bottom-left of the one below make as good a
        // right isosceles triangle as any symbol's finders, around a
        // square with the other finders inside it. Each proposal is a
        // symbol's own.
        let version = SymbolVersion::Qr(2);
        let centres = [[140.0, 140.0], [420.0, 140.0], [140.0, 420.0], [420.0, 420.0]];
        let mut image = Luma::new(560, 560, vec![0.92; 560 * 560]);
        for centre in centres {
            let one = draw_turned(version, centre, 8.0, 0.0, (560, 560));
            for (a, b) in image.data.iter_mut().zip(&one.data) {
                *a = a.min(*b);
            }
        }
        let region = [[0.0, 0.0], [560.0, 0.0], [560.0, 560.0], [0.0, 560.0]];
        let rect = Rectified::with_longest(&image, region, MAX_RECTIFIED_WHOLE).unwrap();
        let capstones: Vec<Capstone> = collect_capstones(&image, &rect, None, Family::Qr).into_iter().filter(|c| !c.inverted).collect();
        assert!(capstones.len() >= 12, "{capstones:?}");
        let proposals = qr_proposals(&image, &rect.frame, &capstones);
        // A symbol's own outline is its finder spacing of 18 modules,
        // with the proposal's reach beyond the finders either side: 29
        // modules, 232 pixels a side.
        for (_, outline) in &proposals {
            let side = distance(outline[0], outline[1]).max(distance(outline[0], outline[3]));
            assert!(side < 300.0, "{outline:?} spans more than one symbol");
        }
        for centre in centres {
            assert!(proposals.iter().any(|(_, p)| distance(quad_centre(*p), centre) < 8.0), "no proposal at {centre:?}: {proposals:?}");
        }
    }

    #[test]
    fn pair_proposals_cover_both_sides_and_the_diagonal() {
        let frame = Frame::new([[0.0, 0.0], [800.0, 0.0], [800.0, 800.0], [0.0, 800.0]], MAX_RECTIFIED).unwrap();
        // Blank paper: no timing pattern reads anywhere, so every side
        // costs the same and the geometry alone is under test.
        let image = Luma::new(800, 800, vec![0.92; 800 * 800]);
        let capstone = |center: [f64; 2]| Capstone {
            measured: true,
            corners: [[0.0; 2]; 4],
            center,
            unit: frame.unit(center),
            module: [10.0, 10.0],
            contrast: 1.0,
            inverted: false,
            partial: false,
            threshold: 0.5,
            line_only: false,
        };
        // Two finders 260 px apart along a side, at ten pixels a module:
        // 26 modules, as a version 4 symbol has them.
        let pair = [capstone([200.0, 200.0]), capstone([460.0, 200.0])];
        let proposals = qr_pair_proposals(&image, &frame, &pair);
        assert_eq!(proposals.len(), 3);
        let centres: Vec<[f64; 2]> = proposals.iter().map(|(_, p)| quad_centre(*p)).collect();
        // The squares either side of the pair, and the one with the pair
        // as its diagonal, centred on the pair's middle.
        let close = |a: [f64; 2], b: [f64; 2]| distance(a, b) < 1e-6;
        assert!(centres.iter().any(|c| close(*c, [330.0, 330.0])), "{centres:?}");
        assert!(centres.iter().any(|c| close(*c, [330.0, 70.0])), "{centres:?}");
        assert!(centres.iter().any(|c| close(*c, [330.0, 200.0])), "{centres:?}");
        for (_, p) in &proposals {
            assert_eq!(clockwise(*p), *p, "{p:?}");
        }
        // The one with the symbol below the pair lies within an outline
        // drawn around the symbol; the one above does not.
        let drawn = Frame::new([[150.0, 150.0], [520.0, 150.0], [520.0, 520.0], [150.0, 520.0]], MAX_RECTIFIED).unwrap();
        let fitting: Vec<[f64; 2]> =
            proposals.iter().filter(|(_, p)| proposal_fits_outline(&drawn, *p)).map(|(_, p)| quad_centre(*p)).collect();
        assert!(fitting.iter().any(|c| close(*c, [330.0, 330.0])), "{fitting:?}");
        assert!(!fitting.iter().any(|c| close(*c, [330.0, 70.0])), "{fitting:?}");
        // Too far apart for any version, or too different in size: nothing.
        let far = [capstone([0.0, 0.0]), capstone([3000.0, 0.0])];
        assert!(qr_pair_proposals(&image, &frame, &far).is_empty());
        let mut odd = pair;
        odd[1].module = [20.0, 20.0];
        assert!(qr_pair_proposals(&image, &frame, &odd).is_empty());
    }

    #[test]
    fn a_turned_placement_over_the_same_square_is_offered_too() {
        // Two finders on a diagonal stand a version 1 symbol either way
        // up over the same square. The decode must be offered both: the
        // fits are near enough alike that either may rank first.
        let square = [[100.0, 100.0], [500.0, 100.0], [500.0, 500.0], [100.0, 500.0]];
        let finder = |center: [f64; 2]| Anchor { kind: AnchorKind::Finder, at: [3.5, 3.5], center, partial: false, inferred: false };
        let placement = |corners: [[f64; 2]; 4], turns: u8| Detection {
            corners,
            version: SymbolVersion::Qr(1),
            inverted: false,
            mirrored: false,
            mirrored_by_decode: false,
            turns,
            anchors: vec![finder([433.0, 167.0]), finder([167.0, 433.0])],
            agreement: (170, 203),
            info: InfoEvidence::default(),
            edged_corners: Vec::new(),
            measured_corners: Vec::new(),
            retry: None,
            decodes: None,
        };
        let upright = placement(square, 0);
        let turned = placement([square[2], square[3], square[0], square[1]], 2);
        let again = placement(square.map(|[x, y]| [x + 2.0, y - 1.0]), 0);
        let offered = pick_candidates(vec![(turned.clone(), (0, 0.84)), (upright.clone(), (0, 0.837)), (again, (0, 0.83))]);
        assert_eq!(offered.len(), 2, "the same placement from another frame is offered once");
        assert_eq!((offered[0].turns, offered[1].turns), (2, 0));
        assert_eq!(offered[1].corners, upright.corners);
    }

    #[test]
    fn a_version_is_confirmed_only_by_a_word_read_outright() {
        let named = |version: SymbolVersion, mismatches: Option<u32>| Detection {
            corners: [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
            version: SymbolVersion::Micro(4),
            inverted: false,
            mirrored: false,
            mirrored_by_decode: false,
            turns: 0,
            anchors: Vec::new(),
            agreement: (75, 82),
            info: InfoEvidence { named_version: Some(version), named_mismatches: mismatches, ..InfoEvidence::default() },
            edged_corners: Vec::new(),
            measured_corners: Vec::new(),
            retry: None,
            decodes: None,
        };
        assert!(named(SymbolVersion::Micro(4), Some(0)).version_confirmed());
        // Within correction range of the word, as the bits beside most
        // things are of some word: named, not confirmed.
        assert!(!named(SymbolVersion::Micro(4), Some(2)).version_confirmed());
        assert!(!named(SymbolVersion::Micro(4), None).version_confirmed());
        // Another version's word, however cleanly read.
        assert!(!named(SymbolVersion::Micro(3), Some(0)).version_confirmed());
    }

    #[test]
    fn frames_are_alike_whichever_corner_they_start_from() {
        let a = [[100.0, 100.0], [500.0, 100.0], [500.0, 500.0], [100.0, 500.0]];
        let turned = [a[3], a[0], a[1], a[2]];
        assert!(frames_alike(a, turned));
        let nudged = a.map(|[x, y]| [x + 15.0, y - 10.0]);
        assert!(frames_alike(a, nudged), "within a twentieth of the size");
        let shifted = a.map(|[x, y]| [x + 30.0, y]);
        assert!(!frames_alike(a, shifted));
        // A placement is its outline and which corner of the symbol
        // stands where: turned or reflected over the same square it is
        // another placement.
        assert!(placements_alike(a, a) && placements_alike(a, nudged));
        assert!(!placements_alike(a, turned));
        assert!(!placements_alike(a, [a[2], a[3], a[0], a[1]]));
        assert!(!placements_alike(a, [a[0], a[3], a[2], a[1]]));
        assert!(!placements_alike(a, shifted));
        assert_eq!(turns_between(a, a), 0);
        assert_eq!(turns_between(turned, a), 1);
        assert_eq!(turns_between([a[2], a[3], a[0], a[1]], a), 2);
        assert_eq!(turns_between([a[1], a[2], a[3], a[0]], a), 3);
    }

    /// `image` lit unevenly: each pixel's luminance times `light(x, y)`.
    fn lit(image: &Luma, light: impl Fn(usize, usize) -> f32) -> Luma {
        let data = image.data.iter().enumerate().map(|(i, &v)| v * light(i % image.width, i / image.width)).collect();
        Luma::new(image.width, image.height, data)
    }

    #[test]
    fn a_narrowed_threshold_reads_a_finder_in_a_shadow() {
        // A finder, eight pixels a module, with lit paper to its left and
        // above and a shadow two fifths as bright from twenty pixels
        // before its edges on: the neighbourhood the whole image sets,
        // some fifty pixels, reaches from the finder's first light run
        // into the lit paper, whose brightness pulls the mean above the
        // shadowed paper, and the run reads dark along the row and the
        // column alike, so no line through the centre reads 1:1:3:1:1 and
        // the two thresholds propose nothing; narrowed to sixteen pixels,
        // the neighbourhood lies within the shadow and the finder reads.
        // The windows, the widest of them thirty-two pixels either side,
        // read it in the image as it is, and the scan has it from the
        // start.
        let finder = draw_turned(SymbolVersion::Micro(1), [128.0, 128.0], 8.0, 0.0, (300, 300));
        let image = lit(&finder, |x, y| if x >= 64 && y >= 64 { 0.4 } else { 1.0 });
        let region = [[0.0, 0.0], [300.0, 0.0], [300.0, 300.0], [0.0, 300.0]];
        let rect = Rectified::new(&image, region).unwrap();
        assert!(rect.local_radius > 40.0, "{}", rect.local_radius);
        let max_module = 300.0 * max_module_fraction(Family::Qr);
        // The finder's centre, in rectified pixels: the Micro QR symbol's
        // is 3.5 modules in from its top-left corner, 2 from its centre.
        let centre = rect.to_rectified([128.0 - 2.0 * 8.0, 128.0 - 2.0 * 8.0]);
        let near = |c: &Candidate| distance([c.x, c.y], centre) < 8.0 && !c.inverted;
        let wide = scan_thresholds(&rect, None, max_module);
        assert!(!wide.iter().any(near), "{wide:?}");
        let windowed = scan_for_finders(&rect, None, max_module);
        assert!(windowed.iter().any(|c| near(c) && !c.half), "{windowed:?}");
        let narrowed = rect.narrowed(Some(f64::from(NARROW_RADIUS)));
        let found = scan_thresholds(&narrowed, None, max_module);
        assert!(found.iter().any(|c| near(c) && !c.half), "{found:?}");
    }

    /// A grid drawn `module_px` pixels a module about `centre`, turned
    /// clockwise by `degrees`: dark modules as ink, light ones as paper,
    /// unknown ones as the mid grey no threshold can call, on paper.
    fn draw_grid(grid: &ModuleGrid, centre: [f64; 2], module_px: f64, degrees: f64, size: (usize, usize)) -> Luma {
        use crate::symbol::grid::Module;
        let (s, c) = degrees.to_radians().sin_cos();
        let (w, h) = (grid.width() as f64, grid.height() as f64);
        let mut data = Vec::with_capacity(size.0 * size.1);
        for py in 0..size.1 {
            for px in 0..size.0 {
                let mut sum = 0.0;
                for sy in 0..3 {
                    for sx in 0..3 {
                        let (x, y) = (px as f64 + (sx as f64 + 0.5) / 3.0 - centre[0], py as f64 + (sy as f64 + 0.5) / 3.0 - centre[1]);
                        let (ux, uy) = (x * c + y * s, -x * s + y * c);
                        let (mx, my) = (ux / module_px + w / 2.0, uy / module_px + h / 2.0);
                        let inside = mx >= 0.0 && my >= 0.0 && mx < w && my < h;
                        sum += match inside.then(|| grid.get(mx as usize, my as usize)) {
                            Some(Module::Dark) => 0.08f32,
                            Some(Module::Unknown) => 0.5,
                            _ => 0.92,
                        };
                    }
                }
                data.push(sum / 9.0);
            }
        }
        Luma::new(size.0, size.1, data)
    }

    /// An encoded Micro QR M2 symbol.
    fn micro_m2() -> ModuleGrid {
        let encoded = qrcode_generator::micro::Encoder::new(qrcode_generator::micro::ErrorCorrection::Low)
            .version(qrcode_generator::micro::Version::M2)
            .encode_text("12345")
            .expect("an M2 symbol");
        let grid = ModuleGrid::from_bools(encoded.width(), encoded.height(), encoded.modules());
        assert_eq!((grid.width(), grid.height()), (13, 13));
        grid
    }

    #[test]
    fn a_micro_qr_symbol_in_a_loose_outline_is_sized_right() {
        // An M2 symbol drawn small in the middle of the photo, turned a
        // little so that it is no render to be read off its pixels, with
        // the whole photo as its outline. Across that outline its one
        // finder lies nearer where an M1 symbol's would, and M1's fixed
        // patterns, a subset of the symbol's own, read all right there;
        // the format word names M2, and the box the word proposes is the
        // frame M2 fits in. With the word unreadable, the frames the
        // finder proposes are fitted all the same, and M2, which reads
        // clean in its own box where M1 has the symbol's data over its
        // quiet zone, is among the placements offered; the data read as
        // a word through some turned placement may put another version
        // first, and the decode arbitrates, as it does in a read.
        let version = SymbolVersion::Micro(2);
        let region = [[0.0, 0.0], [400.0, 0.0], [400.0, 400.0], [0.0, 400.0]];
        let params = SampleParams::default();
        let grid = micro_m2();
        let worded = draw_grid(&grid, [200.0, 200.0], 8.0, 4.0, (400, 400));
        let found = detect_outline(&worded, version, region, &params).expect("found");
        assert_eq!((found.version, found.retry), (version, None), "{found:?}");
        assert_eq!(found.info.named_version, Some(version));
        let func = FunctionMap::new(version);
        let mut wordless = grid.clone();
        for y in 0..func.height {
            for x in 0..func.width {
                if matches!(func.role(x, y), Role::Format { .. }) {
                    wordless.set(x, y, crate::symbol::grid::Module::Unknown);
                }
            }
        }
        let unworded = draw_grid(&wordless, [200.0, 200.0], 8.0, 4.0, (400, 400));
        let offered = detect_candidates(&unworded, version, region, &params);
        assert!(offered.iter().any(|d| d.version == version && d.retry.is_none() && d.score() > 0.99), "{offered:?}");
        let read = crate::imaging::read::read_outline(&unworded, version, region, &params, &DecodeOptions::default()).expect("read");
        assert_eq!(read.version(), version, "{:?}", read.detection);
        assert!(read.is_trustworthy());
        assert_eq!(read.text(), "12345");
    }

    #[test]
    fn a_symbol_in_a_shadow_is_found_in_the_photo_as_it_is() {
        // The same shadow over a whole Micro QR symbol, whose one finder is
        // all the scan has to go on, in an outline drawn loosely round
        // the lit paper and the shadow both. The two thresholds show no
        // finder there, and the symbol was found only on the rung of the
        // ladder that narrows the neighbourhood; the windows show it in
        // the photo as it is, with no rung climbed.
        let version = SymbolVersion::Micro(2);
        let symbol = draw_grid(&micro_m2(), [200.0, 200.0], 8.0, 0.0, (400, 400));
        let image = lit(&symbol, |x, y| if x >= 120 && y >= 120 { 0.4 } else { 1.0 });
        let region = [[0.0, 0.0], [400.0, 0.0], [400.0, 400.0], [0.0, 400.0]];
        let rect = Rectified::new(&image, region).unwrap();
        let wide = scan_thresholds(&rect, None, 400.0 * max_module_fraction(Family::Micro));
        assert!(!wide.iter().any(|c| !c.half && !c.inverted && distance(rect.frame.to_image.apply(c.x, c.y), [176.0, 176.0]) < 8.0), "{wide:?}");
        let found = detect_outline(&image, version, region, &SampleParams::default()).expect("found as it is");
        assert_eq!(found.retry, None, "{found:?}");
        assert_eq!(found.version, version);
        let truth = [[148.0, 148.0], [252.0, 148.0], [252.0, 252.0], [148.0, 252.0]];
        for (c, t) in found.corners.iter().zip(truth) {
            assert!(distance(*c, t) < 4.0, "{:?} against {truth:?}", found.corners);
        }
    }

    #[test]
    fn views_map_back_to_the_photo() {
        // Ink and paper: a render, enlarged.
        let image = Luma::new(40, 30, (0..1200).map(|i| if i % 7 < 3 { 0.0 } else { 1.0 }).collect());
        let region = [[10.0, 8.0], [30.0, 8.0], [30.0, 24.0], [10.0, 24.0]];
        let up = View::around(&image, region, Retry::Upscaled { factor: 3 }).unwrap();
        assert_eq!(up.scale, 1.0 / 3.0);
        let p = up.to_view([12.5, 9.0]);
        let back = up.to_photo(p);
        assert!(distance(back, [12.5, 9.0]) < 1e-9);
        // The view's pixel (3i + 1, 3j + 1) lies on the middle of the
        // photo's (x0 + i, y0 + j), and holds its ink or its paper.
        let (x0, y0) = (up.origin[0] as usize, up.origin[1] as usize);
        for (i, j) in [(2, 2), (3, 2), (5, 4), (6, 7)] {
            let value = up.luma.data[(3 * j + 1) * up.luma.width + 3 * i + 1];
            assert!((value - image.data[(y0 + j) * 40 + x0 + i]).abs() < 0.01, "{i}, {j}: {value}");
        }
        // Shrunk, the whole photo is the photo resized, and what its
        // far edges have over a whole number of the view's pixels is
        // left out.
        let down = View::whole(&image, Retry::Downscaled { factor: 2 }).unwrap();
        assert_eq!((down.luma.width, down.luma.height), (20, 15));
        assert_eq!(down.scale, 2.0);
        assert_eq!(down.luma, image.resized(20, 15));
        let thirds = View::whole(&image, Retry::Downscaled { factor: 3 }).unwrap();
        assert_eq!((thirds.luma.width, thirds.luma.height, thirds.scale), (13, 10, 3.0));
        let whole = Luma::new(39, 30, (0..1200).filter(|i| i % 40 < 39).map(|i| image.data[i]).collect());
        assert_eq!(thirds.luma, whole.resized(13, 10));
        assert!(View::whole(&image, Retry::Downscaled { factor: 5 }).is_none(), "too small to hold a symbol");
        let closed = View::whole(&image, Retry::Closed { radius: 1 }).unwrap();
        assert_eq!((closed.luma.width, closed.luma.height), (40, 30));
        assert_eq!(closed.scale, 1.0);
        assert_eq!(needed_upscale([[0.0, 0.0], [25.0, 0.0], [25.0, 25.0], [0.0, 25.0]], Family::Qr), 3);
        assert_eq!(needed_upscale([[0.0, 0.0], [200.0, 0.0], [200.0, 200.0], [0.0, 200.0]], Family::Qr), 1);
        assert_eq!(needed_upscale([[0.0, 0.0], [100.0, 0.0], [100.0, 10.0], [0.0, 10.0]], Family::Rmqr), 3);
    }

    #[test]
    fn a_small_render_enlarged_is_read_off_the_pixels_it_had() {
        // Renders of a pixel or two a module, as an encoder writes them,
        // each small enough to be enlarged when it is loaded.
        for (version, px, quiet) in
            [(SymbolVersion::Qr(1), 1, 2), (SymbolVersion::Qr(12), 1, 0), (SymbolVersion::Micro(3), 2, 2), (SymbolVersion::Rmqr(0), 2, 1)]
        {
            let func = FunctionMap::new(version);
            let side = |modules: usize| ((modules + 2 * quiet) * px) as u32;
            let render = image::RgbaImage::from_fn(side(func.width), side(func.height), |x, y| {
                let (mx, my) = ((x as usize / px).wrapping_sub(quiet), (y as usize / px).wrapping_sub(quiet));
                let dark = mx < func.width && my < func.height && func.expected(mx, my) == Some(true);
                image::Rgba(if dark { [0, 0, 0, 255] } else { [255, 255, 255, 255] })
            });
            let image = crate::imaging::photo::luma_as_loaded(render.clone());
            let own = image.unenlarged.as_ref().unwrap_or_else(|| panic!("{version} at {px} px is enlarged"));
            assert_eq!((own.width, own.height), (render.width() as usize, render.height() as usize));
            let times = image.width / own.width;
            assert!(times > 1 && (image.width, image.height) == (own.width * times, own.height * times));
            // Read off its own pixels, it is placed in the enlarged ones,
            // its corners on the edges its pixels had.
            let whole = [[0.0, 0.0], [image.width as f64, 0.0], [image.width as f64, image.height as f64], [0.0, image.height as f64]];
            let found = detect_pure(&image, version.family(), whole, &SampleParams::default()).unwrap_or_else(|| panic!("{version} at {px} px"));
            assert_eq!(found.version, version);
            let at = |modules: usize| ((quiet + modules) * px * times) as f64;
            let truth = [[at(0), at(0)], [at(func.width), at(0)], [at(func.width), at(func.height)], [at(0), at(func.height)]];
            assert_eq!(found.corners, truth, "{version} at {px} px");
            // An outline in the enlarged pixels says where to look, as
            // the whole photo does, and the photo turned reads as well.
            let outline = [[at(0) - 3.0, at(0) - 2.0], [at(func.width) + 4.0, at(0) - 1.0], truth[2], truth[3]];
            let inside = detect_pure(&image, version.family(), outline, &SampleParams::default());
            assert_eq!(inside.map(|found| found.corners), Some(truth), "{version} at {px} px, in an outline");
            let turned = image.rotated_cw(1);
            let whole = [[0.0, 0.0], [turned.width as f64, 0.0], [turned.width as f64, turned.height as f64], [0.0, turned.height as f64]];
            let found = detect_pure(&turned, version.family(), whole, &SampleParams::default()).unwrap_or_else(|| panic!("{version}, turned"));
            assert_eq!((found.version, found.corners[0]), (version, [turned.width as f64 - at(0), at(0)]), "{version}, turned");
        }
    }

    /// Paper `width` by `height` with a Micro QR M2 symbol on it at each
    /// of `places`, `module_px` pixels a module and a little turned.
    fn paper_with_symbols(width: usize, height: usize, module_px: f64, places: &[[usize; 2]]) -> Luma {
        let side = (13.0 * module_px * 1.6) as usize;
        let symbol = draw_grid(&micro_m2(), [side as f64 / 2.0, side as f64 / 2.0], module_px, 7.0, (side, side));
        let mut data = vec![0.92f32; width * height];
        for &[x, y] in places {
            for row in 0..side {
                let at = (y - side / 2 + row) * width + x - side / 2;
                data[at..at + side].copy_from_slice(&symbol.data[row * side..(row + 1) * side]);
            }
        }
        Luma::new(width, height, data)
    }

    #[test]
    fn the_search_of_a_large_photo_looks_at_a_copy_and_answers_in_the_photos_own_pixels() {
        // The copy is the photo reduced to the search's side on its
        // longer one, whichever that is, and a photo no larger is
        // searched as it is.
        let flat = |width: usize, height: usize| Luma::new(width, height, vec![0.5; width * height]);
        assert!(reduced_for_search(&flat(SEARCH_SIDE, 300)).is_none());
        let copy = reduced_for_search(&flat(5000, 1000)).expect("a photo over the search's side");
        assert_eq!((copy.width, copy.height), (4096, 819));
        assert!(copy.data.iter().all(|v| (v - 0.5).abs() < 1e-4), "reducing keeps the tone");
        assert_eq!(reduced_for_search(&flat(300, 8192)).map(|copy| (copy.width, copy.height)), Some((150, 4096)));
        // A symbol of eight pixels a module on paper larger than that:
        // found in the copy, where it has under seven, and placed where
        // it lies in the photo, at the size it has there.
        let image = paper_with_symbols(4800, 3600, 8.0, &[[3300, 2500]]);
        let found = find_symbols(&image, Family::Micro, &SampleParams::default());
        assert_eq!(found.iter().map(|d| d.version).collect::<Vec<_>>(), [SymbolVersion::Micro(2)]);
        assert!(distance(quad_centre(found[0].corners), [3300.0, 2500.0]) < 1.5, "{:?}", found[0].corners);
        let side = distance(found[0].corners[0], found[0].corners[1]);
        assert!((side - 104.0).abs() < 2.0, "thirteen modules of eight pixels, not {side}");
    }

    #[test]
    fn an_outline_too_large_to_scan_whole_is_scanned_reduced_and_placed_in_the_photos_own_pixels() {
        // Up to the scan's size with its margin, an outline is scanned
        // as it is; beyond, reduced.
        let outline = |side: f64| [[100.0, 100.0], [100.0 + side, 100.0], [100.0 + side, 100.0 + side], [100.0, 100.0 + side]];
        assert!(!scanned_reduced(outline(1575.0)));
        assert!(scanned_reduced(outline(1576.0)));
        assert!(scanned_reduced([[0.0, 0.0], [4000.0, 0.0], [4000.0, 300.0], [0.0, 300.0]]), "by its longer side");
        // A symbol of fifty pixels a module in an outline two and a half
        // thousand across, where the scan sees it at a little over
        // thirty: the finder it proposes is fitted in the photo itself,
        // and the symbol placed to the pixel there.
        let image = paper_with_symbols(3000, 3000, 50.0, &[[1500, 1500]]);
        let (version, params) = (SymbolVersion::Micro(2), SampleParams::default());
        let region = outline(2500.0);
        assert!(scanned_reduced(region));
        let found = detect_outline(&image, version, region, &params).expect("the symbol is found");
        assert_eq!((found.version, found.retry), (version, None));
        assert!(distance(quad_centre(found.corners), [1500.0, 1500.0]) < 1.5, "{:?}", found.corners);
        let side = distance(found.corners[0], found.corners[1]);
        assert!((side - 650.0).abs() < 2.0, "thirteen modules of fifty pixels, not {side}");
        // An outline that holds nothing has been scanned every way,
        // its own pixels last, and holds nothing still.
        let blank = Luma::new(2400, 2400, vec![0.92; 2400 * 2400]);
        assert!(detect_candidates(&blank, version, outline(2000.0), &params).is_empty());
    }
}
