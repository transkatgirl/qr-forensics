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
//! does not break, and a region that lies on a candidate of the scan's
//! lends it its four corners. Each surviving candidate is refined in the
//! photo itself by fitting a finder-with-ring template to it with the
//! contrast score the corner refinement uses, from the scan's own
//! reckoning of where it lies and from the region's corners where it has
//! them. A candidate whose template reads right is a capstone. One with a
//! quarter torn away still counts, as long as the other three quarters
//! read right. A candidate no template fits is
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
//! pattern between them. Where a blur has run the timing pattern's
//! modules together and no runs can be counted, the pattern is read by
//! its phase instead ([`timing_by_phase`]): each size puts the timing
//! modules at a pitch and a phase the finders' centres fix, and the row
//! follows the alternation of the right size however shallow the blur
//! leaves it. The same alternation, read through a placement, vouches
//! for a symbol the search found whose fixed modules a blur reads wrong
//! ([`timing_follows`]).
//!
//! Capstones are matched to the finder positions of a version, trying each
//! orientation of a square symbol, so an outline drawn a quarter turn out
//! still works. Where they stand as a QR Code's finders more ways than
//! one, as two of three do, the format word read beside them, each bit
//! through its own finder's corners as BoofCV reads it, says which way up
//! the symbol is. The matched capstones give a first map from module
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
//! lie in. A QR Code's alignment patterns make a grid, and with all three
//! finders standing each is looked for first where the plane puts it
//! moved by the offsets at which its neighbours in the grid were found,
//! as BoofCV carries each pattern's offset on to the next; one whose
//! template reads whole stands as a finder does, and is doubted by no
//! fit, and one that does not is looked for again once more of its
//! neighbours and the corner beside it are placed. The map is upgraded from a similarity through an affine
//! transform to a full perspective one as the anchors spread out.
//!
//! A corner a finder stands at is the capstone's own, measured to a
//! fraction of a pixel. A corner left far from any anchor gets a search of
//! its own. Where the symbol has a quiet zone, the edge of the symbol is
//! traced there, as ZBar traces it: profiles across the predicted edge find
//! where the paper turns to ink, a line is fitted to those crossings, or a
//! bend where the paper has curled, as OpenCV fits a spline to each side
//! of a curved symbol, and the corner is where two such lines meet. The
//! trace sees an edge only within half a module of where it is looked
//! for, so where the edges do not trace from where the fit puts the
//! corner, or trace with no paper beyond them, as a row of modules a
//! step inside the symbol has none, they are traced from other places:
//! where the outer sides of the finders beside the corner cross, as
//! BoofCV and OpenCV's `QRCodeDetectorAruco` place it, where the other
//! three corners complete a parallelogram, as wechat_qrcode guesses it,
//! and a module further out.
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
//! One of the three readings may be set for the scan in place of the
//! three ([`SampleParams::binarizer`]), reaching as far as it is set
//! to, as the app's binarised view shows one: the scan then reads the
//! outline against that and nothing else, so that what the view shows
//! is what the scan sees, and the rung that narrows the scan's own
//! threshold is not climbed.
//! And three rungs sharpen the photo, at its size and enlarged
//! ([`super::enhance`]), for a symbol so far out of focus that no
//! threshold tells its finders' rings from their stones, and one
//! deblurs it by the blur it shows ([`super::deblur`]), for a photo out
//! of focus by more than a mask reaches, or shaken.
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
//! Where an edge of the symbol bows, or the anchors inside the symbol
//! bend the map at a straight one, a few points along it anchor the map
//! as well, each where the paper turns to ink: on a bottle or a curled
//! receipt the side of the symbol farthest from any fixed pattern is
//! otherwise placed by the corners and the spline's guess alone, and
//! the edge says where its rows really run. They hold the map across the
//! edge and leave it free along it, where the trace has nothing to say.
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
//! Fitting a proposal is nearly all the search costs, so the proposals
//! take their turn by what the photo shows for them before any fit: a
//! timing pattern between the capstones, a format word that reads
//! outright through the capstones' own squares, as BoofCV reads one
//! before it looks for anything else, or, of a lone finder, the symbol
//! decoding through that finder's square alone, as BoofCV's Micro QR
//! reader decodes one. Those are fitted first and as far as the
//! proposals go; of the rest only the best few are
//! (`MAX_UNSPOKEN_PROPOSALS`). And where finders with a format word
//! outright beside them had no symbol fitted to them in any copy of the
//! photo, the search says so beside what it found ([`Lead`]), as BoofCV
//! hands back what it could not decode.
//!
//! A large photo is searched first in a copy shrunk to about a thousand
//! pixels on its longer side, and each symbol the copy shows is found
//! again at full size through its own outline; the photo is searched at
//! full size only when no copy shows a symbol, when a copy shows the
//! finders of one it could not fit, or when the last copy shows a symbol
//! too small for it to show well. BoofCV's detector is
//! quick because its costly refinement touches only the shapes a cheap
//! pass found, and the copy serves the same end: a twelve-megapixel
//! photo that took two minutes to search at full size takes twenty
//! seconds. What a copy does not show it says nothing of, so where the
//! copies settle the search the photo at its own size is asked one thing
//! more, cheaply, as BoofCV asks it everything: what finder patterns its
//! regions show on none of the symbols found, and what symbols those
//! make.
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
use std::sync::{Arc, Mutex, OnceLock};

use super::Luma;
use super::adjust::Filter;
use super::binarize::{Binarizer, DARK, NONE, WINDOW_BIAS, WINDOW_RADII, local_radius_for, local_rule};
use super::gpu::{self, scan::{Reading, Rule, Values}};
use super::enhance::{Enhance, Enhanced, Sharpen};
use super::homography::{Homography, Model};
use super::projection::{ControlPoint, Project, Projection};
use super::refine::{MOVES, SHIFT, Scorer, Template, descend_by, fixed_threshold};
use super::sample::{
    BINS, INNER_FRACTION, Kernel, NOISE_RANGE, POINTS, SampleParams, classify, histogram_of, otsu, otsu_of, otsu_split, robust_spread_of, sample_modules_as,
    sample_positions,
};
use crate::decode::bch;
use crate::decode::pipeline::{AttemptScore, DecodeOptions, brute_force};
use super::regions;
use crate::symbol::function_map::{FunctionMap, InfoCopy, Role};
use crate::symbol::grid::{ModuleGrid, Symmetry};
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
/// Rows of the rectified image resampled as one piece on the cores to
/// spare.
const RECTIFIED_BAND: usize = 32;
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
/// degrees, whose own axis it then runs along, in half as many. Along a
/// column the module is asked to be about what the row made it, so a
/// finder foreshortened past seven to five is no candidate of the scan's:
/// that is the regions' to find ([`region_candidates`]), which take one
/// at four to one. With the band widened to take them here as well,
/// from two fifths to two and a half, the photographs of symbols seen
/// steeply read no sooner, having read through their regions, and three
/// ordinary ones took a tenth to a third longer for the candidates it
/// let in.
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
/// How far from where a profile was first taken, in modules, the
/// crossings found along the rest of the edge must put the edge for the
/// profile to be taken again about it, and the most it is moved: nearer,
/// the first profile looked there already, and further, a profile about
/// it reads a row of modules for the edge as readily as the edge.
const RECENTRE_LEAST: f64 = 0.25;
const RECENTRE_MOST: f64 = 0.8;
/// What must read as paper beyond the line or bend fitted to an edge, for
/// the fit to be clear of the symbol and not a row of modules inside it:
/// the strip between these many modules out from the fit, read against
/// the paper a little further out ([`BEYOND_PAPER`], the nearer and the
/// further of it). A sample of the strip darker than that paper by
/// [`BEYOND_DROP`] of the symbol's contrast is ink: the fit lies where a
/// blurred edge is halfway from paper to ink, so half a module beyond a
/// true edge no blur takes a sample half the contrast down, and a dark
/// module of the symbol's own border, when the fit runs a row inside it,
/// is all of it down. The strip is read at the profiles that found no
/// crossing of their own, within [`BEYOND_NEAR`] modules of the corner
/// the edge is traced from or a third of the edge if that is more: a
/// profile with a crossing started from paper, and it is at the corner
/// that a fit run on from the far end of the edge goes astray. At most
/// [`BEYOND_INKED`] of those profiles may show ink, which of fewer than
/// ten is none of them: a line a row inside the symbol shows it at every
/// dark module of the border.
const BEYOND_STRIP: [f64; 2] = [0.5, 1.0];
const BEYOND_PAPER: [f64; 2] = [1.2, 1.6];
const BEYOND_DROP: f32 = 0.5;
const BEYOND_NEAR: f64 = 8.0;
const BEYOND_INKED: f64 = 0.1;
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
/// How far, in modules, a timing module of a symbol placed again
/// expecting its bleed ([`bled_placement`]) is looked for from where the
/// finders and alignment patterns placed so put it: the narrowest window
/// the detector looks in. [`search_site`] rounds it up to whole
/// [`SEARCH_STEP`]s, as it does every window, so it reaches nine tenths
/// of a module either way, still inside the two modules a timing pattern
/// repeats at.
const BLED_TIMING_WINDOW: f64 = 0.75;
/// Boundary moves tried, in modules, for a finder pattern that does not
/// fit as printed: into the dark for an over-exposed photo, into the light
/// for an under-exposed one.
const BLOOMS: [f64; 3] = [0.2, 0.35, -0.2];
/// How much more crisply a finder must read fitted expecting a bleed than
/// fitted as printed for the bleed to be taken as the photo's
/// ([`measured_bleed`]). On BoofCV's over-exposed photograph of a version
/// 2 symbol, `brightness/image012.jpg`, whose dark had thinned by a third
/// of a module at every boundary, the fit expecting a third read a
/// quarter more crisply than the plain one; on a photo that has not bled
/// the plain fit reads the more crisply, and a bleed it barely gains on
/// is too slight to move a pattern far.
const BLOOM_GAIN: f64 = 1.1;
/// How far either side of the crispest of [`BLOOMS`] a finder's bleed
/// is measured again, in modules: those lie fifteen hundredths apart and
/// more. Over renders that had bled 0.15, 0.25 and 0.3 of a module at
/// every boundary it measured 0.125, 0.275 and 0.275.
const BLEED_STEP: f64 = 0.075;
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
/// Of those, the proposals that come to nothing with nothing in the
/// photo to speak for them but the capstones they stand on: three with
/// no timing pattern between any two and no format word beside them, two
/// that lie as the ends of a symbol's diagonal or face each other along
/// a side with none between them, a lone finder with neither a word nor
/// a timing pattern running from it. After so many, no more of their
/// kind are fitted in that copy of the photo.
///
/// BoofCV fits nothing to such. It reads a candidate's format word
/// through the corner position pattern's own square, then its version
/// off the patterns' spacing, and only then looks for an alignment
/// pattern or a module of data: of the three thousand candidates the
/// photographs of its benchmark throw up that do not decode, four in
/// five end at one of the first two, some dozens of samples in. Here
/// the fit is nearly all a search costs and every proposal was fitted
/// until a dozen had failed, whatever spoke for it. A photograph of a
/// juice bottle, half a megapixel with one clean symbol on its label,
/// took three seconds to find the symbol and twenty-four more to fit
/// twenty-six outlines to the lettering round it; with the symbol
/// painted out it took ten minutes of processor time to fit two hundred
/// and eighty-six up the ladder and find nothing, where BoofCV made no
/// candidate of any of it. Over eighty photographs of the benchmark,
/// three fits in five came to nothing and took three quarters of the
/// time fitting took.
///
/// What speaks for a proposal is what the fit would have to find to be
/// vouched for ([`vouched`]), read beforehand where the capstones' own
/// squares put it: all three hundred and twenty-two triples of those
/// photographs with a timing pattern between two of their capstones and
/// a format word outright beside them were fitted and vouched for. It
/// ranks and budgets, and vetoes nothing: twenty of the seventy-four
/// triples nothing spoke for were vouched for too, symbols blurred past
/// reading a timing pattern across, and so were six of the forty such
/// pairs. Each was the best or the next best of its kind in its copy,
/// none with more than two failures before it, so the best few are
/// still fitted and the rest wait for a copy that shows more of them.
const MAX_UNSPOKEN_PROPOSALS: usize = 4;
/// The same for two leftover capstones, and for the boxes about a lone
/// finder: of five hundred and seventy-seven such boxes fitted over
/// those photographs, which hold no Micro QR or rMQR symbol, three were
/// vouched for, by timing patterns the lettering made.
const MAX_UNSPOKEN_PAIR_PROPOSALS: usize = 3;
const MAX_UNSPOKEN_LONE_PROPOSALS: usize = 2;
/// Timing modules read out from a lone finder, along each timing pattern
/// that starts at it, to see whether one does: as far as every version
/// of the family has them, the quiet zone past a Micro QR symbol of
/// version M1 reading as its fourth would.
const MICRO_STUB: usize = 4;
const RMQR_STUB: usize = 7;
/// What a lone finder's box with no format word to name it stands behind
/// one whose word reads: a bit more than a word may be off by.
const UNNAMED_DOUBT: f64 = 4.0;
/// Proposals fitted side by side, at most, once the best of them has come
/// to nothing: every fit of such a batch is waited for, and a fit made
/// for a proposal that another of the batch disposes of is wasted, so
/// the batch is kept well under the dozen that may fail.
const FITS_AT_ONCE: usize = 8;
/// Scan clusters a search of the whole photo refines, strongest first,
/// before those a region bears out, which are all refined
/// ([`within_cap`]). [`MAX_CANDIDATES`] is as many as one symbol in an
/// outline needs, and a sheet of twenty symbols threw up ninety-one
/// clusters, sixty of them finders, of which the twenty-five strongest
/// made six whole triples: OpenCV's ArUco-based detector keeps every
/// quadrilateral its thresholds find, and BoofCV every square. Those
/// keep everything because the test their shapes have passed is one
/// worth passing, and a cluster of the scan's has passed none: a photo
/// of one symbol on a desk throws up five hundred, and two thousand at
/// full size, nearly all of them grain. So the cap stays on the clusters
/// that are the scan's alone, and what a region has read as a finder
/// along two lines of its own, the test BoofCV's squares pass, is kept
/// however many there are: a sheet of sixty symbols, twenty of them
/// turned, read forty-three under one cap for all and fifty-nine so.
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
/// photograph before any fit. Three capstones that fail the cut in the
/// photo's pixels are asked once more by their own squares
/// ([`stands_by_squares`]), where a symbol seen from far to one side,
/// one leg a third of the other, still measures as square as it is.
const MAX_LEG_RATIO: f64 = 1.75;
const MAX_CORNER_TILT: f64 = 60.0;
/// Capstones of one symbol may differ in module size by this ratio, as
/// OpenCV's ArUco-based detector allows: ZXing's 1.4 loses the symbol seen
/// from below, whose top finders are drawn three fifths the size of its
/// bottom one.
const SIZE_RATIO: f64 = 1.75;
/// And by this ratio where their own squares say they are one symbol's
/// ([`stands_by_squares`]), seen so steeply that its near corner is
/// drawn at more than [`SIZE_RATIO`] times the size of its far ones.
/// The ratio alone once ended it. A symbol seen from over its top-left
/// corner, the far corner drawn in to seven tenths of the way to the
/// middle, shows that finder at thirteen pixels a module and the other
/// two at six: all three were found, whole, with their corners
/// measured, and proposed nothing, no triple and no pair, so that every
/// rung of the ladder was climbed and a version 1 and a version 5 symbol
/// that both read through an outline came back as no symbol found.
/// What such a symbol's finders have in common is not their size in the
/// photo's pixels but the modules each leg measures in the squares at
/// its two ends, which is what the squares are asked.
const STEEP_SIZE_RATIO: f64 = 3.0;
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
/// photo is not shrunk below what the scan can see, and the margin the
/// frame leaves about it ([`MARGIN`]) besides. It stood at the longest a
/// photo is searched at ([`SEARCH_SIDE`]) and no more, and the margin
/// came out of that: a photo over some 3150 pixels on its longer side
/// was scanned at as little as ten thirteenths of its size, by point
/// samples, and a module of under two pixels in a twelve-megapixel
/// photograph came to less than the scan looks for ([`MIN_MODULE`]).
const MAX_RECTIFIED_WHOLE: f64 = (1.0 + 2.0 * MARGIN) * SEARCH_SIDE as f64;
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
/// from its middle, and how far from parallel to each other the two
/// sides crossed may lie, in degrees, for the two to stand as finders of
/// one symbol. In a symbol the line between two of its finders runs
/// along a row or a column of modules, through the middle of the sides
/// that face; perspective bends that a little, and a finder fitted to a
/// blurred print turns a little. Both are BoofCV's tests, which connects
/// two position patterns only when the line between their centres
/// crosses a side of each within 0.35 of its middle and the two sides
/// crossed lie within forty-five degrees of parallel
/// (`QrCodePositionPatternGraphGenerator`, `SquareGraph::almostParallel`).
///
/// The angle was once asked of each side against the line, after
/// OpenCV's ArUco-based detector, which asks that two finders of one
/// symbol face each other, and each the line between them, to within
/// fifteen degrees (`maxRotation`): thirty here, to leave room for the
/// turn a blurred finder's fit takes, and the crossing within three
/// tenths. The sides that face lie parallel however steeply the symbol
/// is seen, where the line between two finders of a symbol foreshortened
/// past about two to one along its diagonal meets them more than thirty
/// degrees off square. Three such finders proposed nothing together and
/// were left to propose in pairs, and the pairs found nothing either: a
/// symbol sheared to a rhombus, which read at every turn through an
/// outline drawn round it, was not found in the photo around it, with
/// all three of its finders seen and their corners measured.
const FACING_MIDPOINT: f64 = 0.35;
const FACING_PARALLEL: f64 = 45.0;
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
    /// A point of the symbol's edge, where the paper turns to ink. On
    /// paper that curls, a side of the symbol with no fixed pattern of
    /// its own is placed by the corners and the spline's guess at what
    /// lies between them; the edge says where its rows really run, as
    /// OpenCV fits a curve to each side of a bent symbol and straightens
    /// it by them. It holds the map across the edge only
    /// ([`super::projection`]).
    Edge,
    /// The middle of a module where no pattern is, set by hand: whoever
    /// sees in the photo where a module of the data lies may say so, and
    /// the map bends through it as through any anchor. No search places
    /// one and no refinement moves one, there being nothing at it to
    /// look for.
    Custom,
    /// The middle of a patch of data modules, placed where the modules
    /// about it read most crisply dark or light ([`super::patches`]), or
    /// where those a decode came to know read right ([`super::read`]):
    /// on a large symbol on curled paper, the map between the fixed
    /// patterns follows the data.
    Data,
}

impl AnchorKind {
    pub fn label(self) -> &'static str {
        match self {
            AnchorKind::Finder => "finder pattern",
            AnchorKind::Alignment => "alignment pattern",
            AnchorKind::FinderSub => "finder sub-pattern",
            AnchorKind::Timing => "timing module",
            AnchorKind::Edge => "edge point",
            AnchorKind::Custom => "custom anchor",
            AnchorKind::Data => "data patch",
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
    /// Whether this rung is climbed with the scan reading the photo as
    /// `binarizer` has it: every rung is, but the one that narrows the
    /// scan's own local threshold where one reading is set in place of
    /// the scan's three, which is then the reading, at the reach it was
    /// set to.
    fn climbed_under(self, binarizer: Binarizer) -> bool {
        !(binarizer.is_set() && matches!(self, Retry::Thresholded { .. }))
    }

    /// The ladder, cheapest step first: a smaller copy costs less than a
    /// filtered one, and that less than a larger one. The narrowed
    /// threshold costs a pass over the photo as it is, between the two,
    /// and so does the photo sharpened at its size. The photo deblurred
    /// by the blur it shows comes after the masks, which cost less and
    /// take back the pixel or two most photographs blur by; where it
    /// shows no blur wider than that, the rung makes no copy and is
    /// passed over.
    pub const LADDER: [Retry; 11] = [
        Retry::Downscaled { factor: 2 },
        Retry::Thresholded { radius: NARROW_RADIUS },
        Retry::Enhanced(Enhance { enlarge: 1, sharpen: Sharpen::Light }),
        Retry::Enhanced(Enhance { enlarge: 1, sharpen: Sharpen::Strong }),
        Retry::DEBLURRED,
        Retry::Closed { radius: 1 },
        Retry::Upscaled { factor: 2 },
        Retry::Enhanced(Enhance { enlarge: 2, sharpen: Sharpen::Light }),
        Retry::Downscaled { factor: 3 },
        Retry::Closed { radius: 2 },
        Retry::Upscaled { factor: 3 },
    ];

    /// The photo deblurred by the blur it shows, at its size: a rung of
    /// the ladder, and the look beside what the search found that
    /// [`search_deblurred`] takes, through the same copy.
    const DEBLURRED: Retry = Retry::Enhanced(Enhance { enlarge: 1, sharpen: Sharpen::Deblur });
}

impl fmt::Display for Retry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Retry::Closed { radius } => write!(f, "the dark closed over gaps of {radius} px"),
            Retry::Downscaled { factor } => write!(f, "the photo shrunk {factor} times"),
            Retry::Upscaled { factor } => write!(f, "the photo enlarged {factor} times"),
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
    /// Quarter turns clockwise of the photo, or of the copy `retry`
    /// names, the symbol was found in, as zxing-cpp retries a failed read
    /// with `tryRotate`: the fits that place the corners and anchors walk
    /// the photo's axes, and a symbol that reads at one turn of the photo
    /// can sit a fraction of a module off at another. Zero when it was
    /// found with the photo as it lies. Kept apart from `retry`, since a
    /// symbol found turned inside a copy was found both ways.
    pub turned: u8,
    /// Whether the symbol decodes as found (see [`decodes_as_found`]).
    /// Asked by whoever lists detections for choosing among, which lists
    /// what decodes first; `None` when it was not asked.
    pub decodes: Option<bool>,
    /// How far, in modules, the symbol's boundaries are taken to lie into
    /// its dark modules from where its modules have them: on a symbol dark
    /// on light, the bleed of an over-exposed photo, negative where the ink
    /// spread; on one light on dark, whose dark modules read light, the
    /// other way about. Its anchors and corners are refined expecting it
    /// ([`refine_anchors`], [`super::refine::refine_corners_expecting`]).
    /// Nought as the symbol is found; where no read of the placement
    /// settles the search, the read measures it ([`bloom_of`]) and reads
    /// the symbol again expecting it.
    pub bloom: f64,
}

impl Detection {
    pub fn count(&self, kind: AnchorKind) -> usize {
        self.anchors.iter().filter(|a| a.kind == kind).count()
    }

    /// What the photo had to be made into for the symbol to be found,
    /// in words: the step of the ladder or the copy, the turn, or both;
    /// `None` when it was found in the photo as it lies.
    pub fn found_with(&self) -> Option<String> {
        let turned = (!self.turned.is_multiple_of(4)).then(|| format!("the photo turned {}°", 90 * u16::from(self.turned % 4)));
        match (&self.retry, turned) {
            (None, None) => None,
            (Some(retry), None) => Some(retry.to_string()),
            (None, Some(turned)) => Some(turned),
            (Some(retry), Some(turned)) => Some(format!("{retry}, and {turned}")),
        }
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

    /// The polarity the refinement should hold, light on dark when
    /// `Some(true)`: the detection's own, where it was placed expecting a
    /// bleed ([`Self::bloom`]), which was measured at that polarity and is
    /// into the symbol's dark, so that at the other it would be the
    /// opposite bleed. `None` for a placement made as printed, which the
    /// refinement places at whichever polarity reads the better.
    pub fn held_polarity(&self) -> Option<bool> {
        (self.bloom != 0.0).then_some(self.inverted)
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

    /// Quarter turns clockwise the symbol sits in `outline`, to the
    /// nearest quarter ([`turns_between`]), reckoned as for a placement
    /// seen from the front: the corners of a mirrored symbol run in the
    /// symbol's own order, which is mirror order in the photo, and are
    /// taken in the photo's order. Reckoned on the corners as they ran,
    /// a mirrored symbol that sat upright in its outline was told to be
    /// a quarter turn out.
    pub(crate) fn turns_in(&self, outline: [[f64; 2]; 4]) -> u8 {
        let c = self.corners;
        let front = if self.mirrored { [c[0], c[3], c[2], c[1]] } else { c };
        turns_between(outline, front)
    }

    /// This placement relabelled for its symbol turned or reflected by
    /// `symmetry`, as [`ModuleGrid::symmetric`] turns a grid: the same
    /// outline with the symbol's corners at other corners of it, so that
    /// a grid sampled through the result is the grid sampled through
    /// this one, turned so. Each anchor names the module its pattern has
    /// gone to, and one that lands where the version has no pattern of
    /// its kind goes, as a finder pattern turned into the corner without
    /// one does: it has nothing there to be refined against. A point
    /// traced along an edge is on the edge wherever the turn takes it,
    /// and one set by hand where no pattern is stands on its module.
    /// The information bits are as they were read here, for the caller
    /// to read again.
    fn turned(&self, symmetry: Symmetry) -> Detection {
        let func = FunctionMap::new(self.version);
        let sites = sites(&func);
        let mut turned = self.clone();
        turned.corners = symmetry.map_corners(self.corners);
        // The corner each corner of the result was.
        let was = symmetry.map_corners([0usize, 1, 2, 3]);
        let moved = |corners: &[usize]| corners.iter().filter_map(|i| was.iter().position(|j| j == i)).collect::<Vec<usize>>();
        turned.edged_corners = moved(&self.edged_corners);
        turned.measured_corners = moved(&self.measured_corners);
        turned.anchors = self
            .anchors
            .iter()
            .filter_map(|anchor| {
                let at = symmetry.map_point(func.width, func.height, anchor.at);
                let same_place = |site: &&Site| (site.at[0] - at[0]).abs() < 1e-6 && (site.at[1] - at[1]).abs() < 1e-6;
                // A custom anchor the turn sets on a module with a
                // pattern of its own becomes that pattern's anchor, so
                // that no module carries two.
                let (sited, kind) = match anchor.kind {
                    AnchorKind::Edge => (true, anchor.kind),
                    AnchorKind::Custom | AnchorKind::Data => (true, sites.iter().find(same_place).map_or(anchor.kind, |site| site.kind)),
                    kind => (sites.iter().filter(same_place).any(|site| site.kind == kind), kind),
                };
                sited.then_some(Anchor { at, kind, ..*anchor })
            })
            .collect();
        turned.mirrored = self.mirrored != symmetry.is_reflection();
        // The turn is reckoned as for a placement seen from the front:
        // a reflection is that placement in mirror order.
        let front = if symmetry.is_reflection() { [turned.corners[0], turned.corners[3], turned.corners[2], turned.corners[1]] } else { turned.corners };
        turned.turns = (self.turns + turns_between(self.corners, front)) % 4;
        turned
    }

    /// Whether `other` places the same version on the same outline with
    /// each corner of the symbol at the same corner of it
    /// ([`placements_alike`]): the same reading of the photo, whatever
    /// anchors each stands on.
    pub(crate) fn placed_as(&self, other: &Detection) -> bool {
        self.version == other.version && placements_alike(self.corners, other.corners)
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
            .filter(|step| step.climbed_under(params.binarizer))
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
    // A reading set for the scan reaches as far across the copy as it
    // was set to reach across the photo.
    let params = &SampleParams { binarizer: params.binarizer.scaled(1.0 / view.mean_scale()), ..*params };
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
    let Some(rect) = Rectified::with_longest(image, region, longest).map(|rect| rect.narrowed(local_radius).ruled(params.binarizer)) else {
        return Vec::new();
    };
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
            fit.turns = fit.turns_in(region);
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

/// A placement whose fixed patterns fit its symbol turned or reflected
/// by `symmetry` and not as placed, put the way they fit: the same
/// outline with the symbol's corners at other corners of it
/// ([`Detection::turned`]), and the symbol's information bits read
/// again through them.
///
/// The fit that ranks the placements of an outline can put a square a
/// quarter turn out on two of a symbol's finders above the same square
/// the right way up on all three, and the few offered to the decode need
/// not include the second. A grid sampled through the first shows a
/// finder pattern in the corner that has none and data where one should
/// be, which the grid's own analysis reads and names the turn for
/// ([`crate::decode::pipeline::GridAnalysis::orientation`]); BoofCV's
/// Micro QR reader, with one position pattern to go by, tries each of
/// the four turns and keeps the one that decodes.
///
/// A square symbol's fixed patterns are their own transpose but for one
/// module, so a turn and the same turn seen from behind fit alike, and
/// which of the two was named is chance. The format bits decide between
/// them, as they do for a placement fitted ([`align_assignment`]): the
/// one seen from the front, unless the bits spell a word more clearly
/// through the mirror order. An rMQR symbol's patterns fit one way only.
pub(crate) fn turned_on_symbol(image: &Luma, found: &Detection, symmetry: Symmetry, params: &SampleParams) -> Detection {
    let func = FunctionMap::new(found.version);
    let threshold = fixed_threshold(params);
    let reread = |mut placed: Detection| {
        placed.mirrored_by_decode = false;
        placed.info = read_info(image, &func, placed.corners, placed.inverted, threshold, params.band);
        placed
    };
    let turned = reread(found.turned(symmetry));
    if found.version.family() == Family::Rmqr {
        return turned;
    }
    let other = reread(turned.transposed());
    let (front, behind) = if turned.mirrored { (other, turned) } else { (turned, other) };
    if bch::mirror_wins(front.info.format, behind.info.format) { behind } else { front }
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
        turned: 0,
        decodes: None,
        bloom: 0.0,
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

/// A symbol of `version` placed by its own four corners, as a hand
/// places them in the Align stage: nothing searched for, no anchors, and
/// the corners taken where they are given, for the read to refine. They
/// may be given round either way and from any corner of the symbol: they
/// are put clockwise here, the read seats a placement that is a turn
/// out on its symbol ([`super::read`]), and the information bits read
/// through them settle a mirror image as they do for a symbol rendered
/// pixel for pixel ([`pure_detection`]). The ink is taken to be
/// whichever colour the fixed patterns read better as. `None` when the
/// corners make no outline.
///
/// A benchmark whose symbols are marked by hand has its oracle here: a
/// symbol such a placement reads and the search of the photo does not
/// was lost in the search, and one that neither reads, in the reading.
pub fn placed_by_hand(image: &Luma, version: SymbolVersion, corners: [[f64; 2]; 4], params: &SampleParams) -> Option<Detection> {
    // Clockwise down the photo, as the symbol's own corners run.
    let corners = clockwise(corners);
    let func = FunctionMap::new(version);
    let threshold = fixed_threshold(params);
    let projection = Projection::new(func.width, func.height, corners, &[])?;
    let scorer = Scorer::new(image, &Template::whole(&func), corners, threshold);
    let (dark_on_light, _) = scorer.agreement_known(&projection, false)?;
    let (light_on_dark, _) = scorer.agreement_known(&projection, true)?;
    let inverted = light_on_dark > dark_on_light;
    let info = read_info(image, &func, corners, inverted, threshold, params.band);
    let mut found = Detection {
        corners,
        version,
        inverted,
        mirrored: false,
        mirrored_by_decode: false,
        turns: 0,
        anchors: Vec::new(),
        agreement: (dark_on_light.max(light_on_dark), scorer.fixed()),
        info,
        edged_corners: Vec::new(),
        measured_corners: Vec::new(),
        retry: None,
        turned: 0,
        decodes: None,
        bloom: 0.0,
    };
    if version.family() != Family::Rmqr {
        let swapped = [corners[0], corners[3], corners[2], corners[1]];
        let swapped_info = read_info(image, &func, swapped, inverted, threshold, params.band);
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
///
/// A name that reads as its word outright stands on less: on the fixed
/// patterns alone reading that well, whatever the quiet zone does. A
/// string of bits is a word outright once in a thousand, not once in
/// two, and the fit read the fixed modules and the quiet zone against
/// one split of them all, which light the paper has and the symbol has
/// not threw: in a dark photograph with a lamp's bright spot on the
/// paper beside a version 7 symbol the split fell between the spot and
/// everything else, and the placement whose version information and
/// format word both read outright, with 97 fixed modules in a hundred
/// right against their own split, was given a fit of 44 in a hundred and
/// ranked below a version 6 symbol fitted to the same finders. The fit
/// is read against a threshold that follows the light across the symbol
/// now ([`Scorer::agreement_shaded`]): a version 2 symbol beside such a
/// lamp has no word to name its version, and fell to the same split. A
/// name read outright stands on the fixed patterns alone as before, for
/// the light no four corners can follow.
type Rank = (i8, f64);

fn rank(found: &Detection, fit: f64, trusted: Option<Prename>) -> Rank {
    let fits = fit >= MIN_NAMED_FIT || (found.version_confirmed() && found.score() >= MIN_NAMED_FIT);
    let verdict = match (trusted, found.info.named_version) {
        (Some(p), _) if p.strong && p.version == found.version => 2,
        (_, Some(named)) if named == found.version && fits => 1,
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
    let readings = Readings::of(rect, threshold, true);
    let mut candidates = scan_for_finders(rect, &readings, max_module);
    let regions = region_candidates(rect, &readings, max_module, &mut candidates);
    let mut all: Vec<Capstone> = Vec::new();
    // Whole candidates first, then the half ones, which passed the
    // cross-check on one side only and come from the data as often as
    // from a torn finder, then the regions the scan did not propose.
    let whole = within_cap(candidates.iter().enumerate().filter(|(_, c)| !c.half), MAX_CANDIDATES);
    let half = within_cap(candidates.iter().enumerate().filter(|(_, c)| c.half), MAX_HALF_CANDIDATES);
    // Each fitted on a core to spare, where there are any, and gathered
    // in its turn: the fits take nothing from one another, and are
    // nearly all the time a search takes.
    let in_turn: Vec<&Candidate> = whole.chain(half).map(|(_, c)| c).chain(regions.iter()).collect();
    for capstone in work::map(&in_turn, |candidate| refine_capstone(image, rect, candidate, threshold)).into_iter().flatten() {
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

/// The candidates of the scan that are refined: the first `most` of them
/// as they come, strongest first, and past those every one a region bore
/// out. The cap is there for the candidates a busy frame throws up, lines
/// of text and the grain of a table that read 1:1:3:1:1 along a row or
/// two; a candidate with a region on it has read so along two lines
/// across a quadrilateral of its own besides, and is as likely a finder
/// as any the cap lets through. A finder turned well off the scan's axes
/// polls few lines and ranks low, and on a sheet of symbols every upright
/// finder ranks above it: three sheets of twenty, one of them turned,
/// spent the whole cap on the upright forty.
fn within_cap<'a>(candidates: impl Iterator<Item = (usize, &'a Candidate)>, most: usize) -> impl Iterator<Item = (usize, &'a Candidate)> {
    candidates.enumerate().filter(move |(nth, (_, c))| *nth < most || c.quad.is_some()).map(|(_, numbered)| numbered)
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

/// Finder candidates from the connected regions of the binarised image,
/// as BoofCV finds position patterns: a quadrilateral fitted to each
/// region and the 1:1:3:1:1 proportion read along two lines through it.
/// The scan wants the proportion intact along two rows or columns through
/// the centre, which a finder scratched through both ways no longer has,
/// and reads the runs square to the frame, which a finder seen steeply
/// through a frame no outline rectified does not keep; a region keeps its
/// shape through either.
///
/// A region within a module of a candidate the scan made is the same
/// pattern seen a second way, and its quadrilateral is handed to that
/// candidate, the strongest when several lie on it, for the template fit
/// to start from besides the candidate's own starts
/// ([`refine_capstone`]). Such a region was once dropped and the scan's
/// candidate left to itself: the scan finds the centre of a finder turned
/// thirty-seven degrees, or one foreshortened to a rhombus, along the few
/// lines that still cross it in proportion, and the template, started
/// square to the frame about that centre, cannot turn so far, so that a
/// region which had the finder's four corners to the pixel was thrown
/// away for a candidate that came to nothing. BoofCV has no such quarrel
/// to settle: the polygon is the only start its position patterns have.
/// The regions the scan made nothing of are returned as candidates of
/// their own, and the template fit verifies those as it verifies the
/// scan's.
///
/// Both readings of the image are labelled, the one against one threshold
/// before the one that follows the lighting, and a region either shows is
/// kept once, as the first shows it; with one reading set for the scan,
/// that reading alone is, one threshold at Otsu's split whatever the
/// sampling's is fixed at. None is turned away for there being
/// too many: a region has already read as a finder along two lines of
/// its own, which is BoofCV's whole test of a position pattern, and a
/// sheet of sixty symbols has a hundred and eighty of them.
fn region_candidates(rect: &Rectified, readings: &Readings, max_module: f64, scanned: &mut [Candidate]) -> Vec<Candidate> {
    let mut out: Vec<Candidate> = Vec::new();
    for found in regions::finder_regions(rect.frame.width, rect.frame.height, &readings.labelled(), MIN_MODULE, max_module) {
        for region in found {
            let module = (region.module[0] + region.module[1]) / 2.0;
            let near = |c: &Candidate| c.inverted == region.inverted && distance([c.x, c.y], [region.x, region.y]) < module.max(c.module_mean());
            let quad = region.corners.map(|[x, y]| rect.frame.to_image.apply(x, y));
            if let Some(candidate) = scanned.iter_mut().find(|c| near(c)) {
                candidate.quad.get_or_insert(quad);
                continue;
            }
            if out.iter().any(near) {
                continue;
            }
            out.push(Candidate {
                x: region.x,
                y: region.y,
                module: region.module,
                inverted: region.inverted,
                votes: MIN_VOTES,
                half: false,
                quad: Some(quad),
                scanned: false,
            });
        }
    }
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
    /// Rectified pixels to a pixel of the image: one, or less for a
    /// region that comes out reduced.
    scale: f64,
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
        Some(Self { width, height, to_image, from_image, origin, size, scale })
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
    /// How the luminance falls ([`super::sample::histogram_of`]): what one
    /// threshold for the outline and the spread are read from.
    histogram: [u32; BINS],
    /// The luminance handed the card, where it holds it whole, for the
    /// scan's readings of it to be made there; let go of once they are,
    /// whose room on the card a large photo's frame would otherwise take
    /// for as long as its candidates are fitted.
    card: Mutex<Option<gpu::Image>>,
    /// How far the luminance reaches from the darkest to the lightest
    /// hundredth of the pixels: the scale the contrast floors are set on.
    spread: f32,
    /// How far around each pixel the scan's local threshold reaches, in
    /// rectified pixels: what the outline's size sets, unless a rung of
    /// the ladder narrowed it, or one reading set for the scan has its
    /// own reach.
    local_radius: f64,
    /// How the scan reads it: its three readings pooled, or the one set
    /// for it, reaching as far as that is set to in rectified pixels.
    binarizer: Binarizer,
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
        let (w, h) = (frame.width, frame.height);
        // On the cores, a band of rows to each, each place in the photo in
        // double precision. Resampled on the graphics card, in single, a
        // place some four thousand pixels into a photo was put a few
        // ten-thousandths of a pixel off, enough to move a pixel near the
        // threshold to the other side of it here and there, and a symbol
        // on a bottle that had read was fitted elsewhere and did not. Each
        // band is counted as it is made, while its values are at hand, and
        // the histogram is the bands' counts added up, where counted on the
        // card it waited on the card for them.
        let mut data = vec![0.0f32; w * h];
        let counted = {
            let bands: Vec<Mutex<(usize, &mut [f32])>> = data.chunks_mut((RECTIFIED_BAND * w).max(1)).enumerate().map(|(i, band)| Mutex::new((i * RECTIFIED_BAND, band))).collect();
            work::map(&bands, |band| {
                let mut band = band.lock().expect("a band is resampled once, by one worker");
                let (y0, values) = &mut *band;
                for (i, value) in values.iter_mut().enumerate() {
                    let [x, y] = frame.to_image.apply((i % w) as f64 + 0.5, (*y0 + i / w) as f64 + 0.5);
                    *value = image.sample(x, y).unwrap_or(f32::NAN);
                }
                histogram_of(values.iter().copied().filter(|v| v.is_finite()))
            })
        };
        let histogram = counted.iter().fold([0u32; BINS], |mut all, counts| {
            all.iter_mut().zip(counts).for_each(|(all, &n)| *all += n);
            all
        });
        // Handed the card once, where it holds the whole, for the scan's
        // readings to be made there ([`Readings`]).
        let card = gpu::holds(w, h).then(|| gpu::Image::handed(&data, w, h));
        let spread = robust_spread_of(&histogram);
        let local_radius = local_radius_for(frame.size[0].min(frame.size[1]));
        Some(Self { frame, data, histogram, card: Mutex::new(card), spread, local_radius, binarizer: Binarizer::Pooled })
    }

    /// The same, with the scan's local threshold reaching `radius`
    /// pixels of the image about each pixel when one is given, instead of
    /// what the outline's size sets: brought to rectified pixels, as
    /// [`Self::ruled`] brings a reading's reach, so that a region scanned
    /// reduced is narrowed to the same stretch of the photo.
    fn narrowed(mut self, radius: Option<f64>) -> Self {
        if let Some(radius) = radius {
            self.local_radius = radius * self.frame.scale;
        }
        self
    }

    /// The same, read as `binarizer` has it: against the one reading
    /// set for the scan alone, when one is, its reach brought from the
    /// image's pixels to rectified ones, which for the local reading is
    /// then the reach of the scan's local threshold.
    fn ruled(mut self, binarizer: Binarizer) -> Self {
        let binarizer = binarizer.scaled(self.frame.scale);
        if let Binarizer::Local { radius } = binarizer {
            self.local_radius = radius;
        }
        self.binarizer = binarizer;
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
    /// Photo pixels per view pixel, across and down: the same both ways,
    /// but for a copy a reduced photo is made again as
    /// ([`Enhanced::factor`]).
    scale: [f64; 2],
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
        // A photo adjusted more than enlarged is made again for the view,
        // by its own adjustments in their order: enlarged or shrunk
        // first, and closed in place of its own filter, last.
        let remade = match step {
            Retry::Upscaled { factor } => Some(Enhanced::window(image, [x0, y0, x1, y1], Enhance { enlarge: factor, sharpen: Sharpen::Off })),
            Retry::Downscaled { factor } if image.as_shot.is_some() => {
                let k = usize::from(factor);
                if w / k < 8 || h / k < 8 {
                    return None;
                }
                Some(Enhanced::shrunk(image, [x0, y0, x0 + w / k * k, y0 + h / k * k], factor))
            }
            Retry::Closed { radius } if image.as_shot.is_some() => Some(Enhanced::closed(image, [x0, y0, x1, y1], radius)),
            _ => None,
        };
        if let Some(view) = remade {
            let view = view?;
            return Some(Self { luma: view.luma, origin: view.origin, scale: view.factor.map(|factor| 1.0 / factor), local_radius: None });
        }
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
            Retry::Upscaled { .. } => unreachable!("an upscaled view is an enlarged copy"),
            // Sharpened and enlarged from the pixels the photo had, and
            // begun where a whole one of those begins.
            Retry::Enhanced(enhance) => {
                let view = Enhanced::searched(image, [x0, y0, x1, y1], enhance)?;
                return Some(Self { luma: view.luma, origin: view.origin, scale: view.factor.map(|factor| 1.0 / factor), local_radius: None });
            }
        };
        Some(Self { luma, origin: [x0 as f64, y0 as f64], scale: [scale; 2], local_radius })
    }

    /// Photo pixels per view pixel as a length, whichever way it runs: the
    /// mean of the two ways.
    fn mean_scale(&self) -> f64 {
        (self.scale[0] + self.scale[1]) / 2.0
    }

    fn to_view(&self, p: [f64; 2]) -> [f64; 2] {
        [(p[0] - self.origin[0]) / self.scale[0], (p[1] - self.origin[1]) / self.scale[1]]
    }

    fn to_photo(&self, p: [f64; 2]) -> [f64; 2] {
        [self.origin[0] + p[0] * self.scale[0], self.origin[1] + p[1] * self.scale[1]]
    }

    /// A detection made in the view, brought back to photo pixels.
    fn unscale(&self, found: &mut Detection) {
        found.corners = found.corners.map(|c| self.to_photo(c));
        for anchor in &mut found.anchors {
            anchor.center = self.to_photo(anchor.center);
        }
    }
}

/// What the scan reads the rectified image as, every way it reads it,
/// made on the graphics card at once ([`gpu::scan::readings`]), packed
/// together, as few bits to each as it reads in: against one threshold for the outline, at the one given
/// or at Otsu's split, and against the lighting about each pixel, the
/// mean of the blocks of [`super::binarize`] within the reach the
/// outline's size sets, or the one a rung of the ladder narrowed it to,
/// each with its closings, which the regions are labelled from; and
/// besides, where the scan pools them, against the mean of a window about
/// each pixel at each width [`window_binarizers`] gives. With one reading
/// set for the scan, that reading alone, closed, and one threshold set so
/// is Otsu's split, as the view shows it, whatever the sampling's
/// threshold is fixed at. Both the scan and the regions read the same
/// readings, which the cores once made twice over for them; and where the
/// scan reads them, each is turned over its diagonal on the card besides,
/// for the scan down the columns.
struct Readings {
    /// One threshold for the outline and the lighting about each pixel,
    /// or the one reading set for the scan, closed; and then the windows',
    /// where the scan pools them. Each turned over its diagonal besides,
    /// its columns as rows, where the scan reads them
    /// ([`gpu::scan::turned_readings`]).
    packed: gpu::scan::Packed,
    /// How many of them, the first, are labelled: the closed ones.
    labelled: usize,
}

impl Readings {
    /// The readings of `rect`, and where they are `scanned`, the windows'
    /// among them and each turned over its diagonal besides.
    fn of(rect: &Rectified, threshold: Option<f32>, scanned: bool) -> Self {
        let otsu = || otsu_of(&rect.histogram);
        let window = |radius: f64| Rule::Window { radius: radius.round().max(1.0) as usize, bias: WINDOW_BIAS * rect.spread };
        let local = local_rule(rect.local_radius, rect.spread);
        let mut rules: Vec<(Rule, bool)> = match rect.binarizer {
            Binarizer::Global => vec![(Rule::Global { threshold: otsu() }, true)],
            Binarizer::Window { radius } => vec![(window(radius), true)],
            Binarizer::Local { .. } => vec![(local, true)],
            Binarizer::Pooled => vec![(Rule::Global { threshold: threshold.unwrap_or_else(otsu) }, true), (local, true)],
        };
        let labelled = rules.len();
        if scanned && !rect.binarizer.is_set() {
            rules.extend(window_binarizers(rect).map(|binarizer| match binarizer {
                Binarizer::Window { radius } => (window(radius), false),
                _ => unreachable!("the windows are windows"),
            }));
        }
        // The luminance on the card is read there once, and let go of: a
        // second look reads it handed the card again.
        let card = rect.card.lock().expect("nothing panics holding the card").take();
        let image = card.as_ref().map_or(Values::Cores(&rect.data), Values::Card);
        let size = (rect.frame.width, rect.frame.height);
        let packed = if scanned { gpu::scan::turned_readings(image, size, &rules) } else { gpu::scan::readings(image, size, &rules, false) };
        Self { packed, labelled }
    }

    /// The readings the regions are labelled from.
    fn labelled(&self) -> Vec<Reading<'_>> {
        (0..self.labelled).map(|r| self.packed.reading(r)).collect()
    }

    /// The readings the scan reads, each beside itself turned over its
    /// diagonal: the labelled ones and then the windows'.
    fn scanned(&self) -> Vec<(Reading<'_>, Reading<'_>)> {
        (0..self.packed.len()).map(|r| (self.packed.reading(r), self.packed.turned(r))).collect()
    }
}

/// Whether the pixel at (`x`, `y`) reads dark in `mask`, a reading of an
/// image `width` by `height` ([`Readings`]); `None` off the image or
/// where it has no pixel.
fn dark_in(mask: Reading<'_>, width: usize, height: usize, x: i64, y: i64) -> Option<bool> {
    if x < 0 || y < 0 || x as usize >= width || y as usize >= height {
        return None;
    }
    match mask.kind(y as usize * width + x as usize) {
        NONE => None,
        kind => Some(kind == DARK),
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
    /// first and clockwise, where a region shows the pattern: it carries
    /// the turn and the perspective the scan's runs do not.
    quad: Option<[[f64; 2]; 4]>,
    /// The scan made the candidate, and the quadrilateral it carries, if
    /// any, is that of a region lying on it; false for a candidate a
    /// region proposed where the scan made none. See [`refine_capstone`]
    /// for what each starts its fit from.
    scanned: bool,
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

/// The side, in rectified pixels, of the cells [`Clusters`] files its
/// clusters under.
const CLUSTER_CELL: f64 = 8.0;

/// The clusters the hits of a scan gather into, in the order they were
/// started, each filed under the cell of [`CLUSTER_CELL`] pixels its
/// centre lies in.
///
/// A hit joins the first cluster it belongs to, and was once held
/// against every cluster in turn to find it. On a sound photo there are
/// a few dozen. A frame of grain starts one for nearly every hit, tens
/// of thousands of them, and the scan of a twelve-megapixel photograph
/// of four symbols on a noisy ground, which reads all four, had not
/// finished gathering after six minutes. A cluster a hit belongs to has
/// a module of at most twice the hit's and lies within that module of it
/// along each axis, so only the clusters filed that near are asked, and
/// the first of them in the order they were started is the one every
/// cluster in turn would have come to.
struct Clusters {
    all: Vec<Candidate>,
    /// Indices into `all` by cell, the cells row by row.
    cells: Vec<Vec<u32>>,
    across: usize,
    down: usize,
}

impl Clusters {
    /// No clusters yet, for the hits of scans of a `width` × `height`
    /// image.
    fn new(width: usize, height: usize) -> Self {
        let across = (width as f64 / CLUSTER_CELL).ceil().max(1.0) as usize;
        let down = (height as f64 / CLUSTER_CELL).ceil().max(1.0) as usize;
        Clusters { all: Vec::new(), cells: vec![Vec::new(); across * down], across, down }
    }

    /// The cell a point is filed under: the one it lies in, or the
    /// nearest for a point off the image.
    fn cell(&self, x: f64, y: f64) -> (usize, usize) {
        let along = |v: f64, cells: usize| (v / CLUSTER_CELL).floor().clamp(0.0, (cells - 1) as f64) as usize;
        (along(x, self.across), along(y, self.down))
    }

    /// Merge a hit into the cluster it belongs to, or start a new one.
    fn merge(&mut self, hit: Candidate) {
        let size = (hit.module[0] + hit.module[1]) / 2.0;
        let belongs = |c: &Candidate| {
            let own = (c.module[0] + c.module[1]) / 2.0;
            c.inverted == hit.inverted
                && (c.x - hit.x).abs() <= own
                && (c.y - hit.y).abs() <= own
                && (own - size).abs() <= 0.5 * own.max(size)
        };
        // The last of those tests holds a cluster's module to twice the
        // hit's, and the two before it hold the cluster within its own
        // module of the hit.
        let reach = 2.0 * size + 1e-6;
        let (x0, y0) = self.cell(hit.x - reach, hit.y - reach);
        let (x1, y1) = self.cell(hit.x + reach, hit.y + reach);
        // A large finder among few clusters has more cells within reach
        // than there are clusters to ask.
        let first = if (x1 - x0 + 1) * (y1 - y0 + 1) >= self.all.len() {
            self.all.iter().position(belongs)
        } else {
            let mut first: Option<u32> = None;
            for row in y0..=y1 {
                for filed in &self.cells[row * self.across + x0..=row * self.across + x1] {
                    for &i in filed {
                        if first.is_none_or(|earlier| i < earlier) && belongs(&self.all[i as usize]) {
                            first = Some(i);
                        }
                    }
                }
            }
            first.map(|i| i as usize)
        };
        let Some(i) = first else {
            let (x, y) = self.cell(hit.x, hit.y);
            self.cells[y * self.across + x].push(self.all.len() as u32);
            self.all.push(hit);
            return;
        };
        let before = self.cell(self.all[i].x, self.all[i].y);
        let c = &mut self.all[i];
        let n = c.votes as f64;
        c.x = (c.x * n + hit.x) / (n + 1.0);
        c.y = (c.y * n + hit.y) / (n + 1.0);
        c.module = [(c.module[0] * n + hit.module[0]) / (n + 1.0), (c.module[1] * n + hit.module[1]) / (n + 1.0)];
        c.votes += 1;
        c.half &= hit.half;
        let after = self.cell(self.all[i].x, self.all[i].y);
        if after != before {
            self.cells[before.1 * self.across + before.0].retain(|&filed| filed as usize != i);
            self.cells[after.1 * self.across + after.0].push(i as u32);
        }
    }

    /// The clusters enough lines agreed on ([`quorate`]), in the order
    /// they were started.
    fn quorate(mut self) -> Vec<Candidate> {
        self.all.retain(quorate);
        self.all
    }
}

/// Scan every line of a `width` × `height` image along its first axis,
/// binarised as `mask` reads it, one of the rectified image's
/// [`Readings`], for finder-shaped runs, and cross-check each along the
/// other axis and back along its own. With `transposed` the image is the
/// rectified image on its side, `mask` its reading turned over its
/// diagonal ([`gpu::scan::Packed::turned`]), and each candidate is turned back
/// before it joins the clusters: the columns are scanned as well as the
/// rows, as ZBar's scanner runs both ways, so that a finder whose rows a
/// vertical scratch breaks is still seen down its columns.
fn scan(width: usize, height: usize, mask: Reading<'_>, min_module: f64, max_module: f64, hits: &mut Vec<Candidate>, transposed: bool) {
    if width == 0 || height == 0 {
        return;
    }
    let push = |hits: &mut Vec<Candidate>, hit: Candidate| {
        let hit = if transposed {
            Candidate { x: hit.y, y: hit.x, module: [hit.module[1], hit.module[0]], ..hit }
        } else {
            hit
        };
        hits.push(hit);
    };
    let dark = |x: i64, y: i64| dark_in(mask, width, height, x, y);
    // (start, length, kind): the reading's own kind of each run, 0 light,
    // 1 dark, 2 no pixel ([`super::binarize::LIGHT`], [`DARK`], [`NONE`]),
    // taken from the row as it lies, from the reading's own bits of it.
    let (shift, own) = (mask.shift, 3u8 << mask.shift);
    let mut runs: Vec<(usize, usize, u8)> = Vec::new();
    for (y, line) in mask.bytes.chunks_exact(width).enumerate() {
        runs.clear();
        let mut start = 0;
        for x in 1..width {
            if (line[x] ^ line[start]) & own != 0 {
                runs.push((start, x - start, (line[start] >> shift) & 3));
                start = x;
            }
        }
        runs.push((start, width - start, (line[start] >> shift) & 3));
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
                    push(hits, Candidate { x: cx, y: cy, module: [row_module, row_module], inverted: !centre_dark, votes: 1, half: true, quad: None, scanned: true });
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
            push(hits, Candidate { x: cx, y: cy, module, inverted: !centre_dark, votes: 1, half: false, quad: None, scanned: true });
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
fn scan_for_finders(rect: &Rectified, readings: &Readings, max_module: f64) -> Vec<Candidate> {
    // The centre run is three modules across, so that many lines could
    // vote, along each axis.
    let rate = |c: &Candidate| c.votes as f64 / (3.0 * (c.module[0] + c.module[1])).max(1.0);
    let strongest_first = |a: &Candidate, b: &Candidate| {
        a.half.cmp(&b.half).then(rate(b).total_cmp(&rate(a))).then((b.module[0] + b.module[1]).total_cmp(&(a.module[0] + a.module[1])))
    };
    // One reading set for the scan is the scan: its candidates alone,
    // whole ones first and each strongest first.
    if rect.binarizer.is_set() {
        let mut alone = clustered(rect, scan_readings(rect, &readings.scanned(), max_module));
        alone.sort_by(strongest_first);
        return alone;
    }
    // The two thresholds and the windows, every reading scanned side by
    // side on the cores to spare, each reading's hits gathered in the
    // order its scan made them.
    let mut hits = scan_readings(rect, &readings.scanned(), max_module);
    let windows = hits.split_off(2);
    let mut clusters = clustered(rect, hits);
    // The windows' hits are gathered apart, and stand only where the
    // two thresholds made no whole candidate: pooled into one gathering
    // they shifted the centre and the module of a candidate those two
    // had already made, by the little the windows read it differently,
    // and a symbol on curled paper that had read outright was then
    // placed a few pixels off at its far corner and read only with its
    // padding pinned.
    let mut windowed = clustered(rect, windows);
    windowed.retain(|c| {
        let near = |o: &Candidate| o.inverted == c.inverted && distance([o.x, o.y], [c.x, c.y]) < c.module_mean().max(o.module_mean());
        !clusters.iter().any(|o| !o.half && near(o))
    });
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
/// rows and columns each, pooled, in no order: what [`scan_for_finders`]
/// makes of the two before it adds the windows.
#[cfg(test)]
fn scan_thresholds(rect: &Rectified, threshold: Option<f32>, max_module: f64) -> Vec<Candidate> {
    let readings = Readings::of(rect, threshold, true);
    clustered(rect, scan_readings(rect, &readings.scanned()[..readings.labelled], max_module))
}

/// The hits of the scans of each of `masks`, readings of the rectified
/// image each beside itself turned over its diagonal
/// ([`Readings::scanned`]), rows and columns, each reading's in the order
/// its scan makes them: the readings side by side on the cores to spare,
/// since none asks anything of another.
fn scan_readings(rect: &Rectified, masks: &[(Reading<'_>, Reading<'_>)], max_module: f64) -> Vec<Vec<Candidate>> {
    work::map(masks, |&(mask, turned)| scan_both_ways(rect, mask, turned, max_module))
}

/// The hits of scans gathered into candidates, each reading's in turn
/// and in its order, as one scan after another gathers them: what a
/// hit merges with depends on the hits before it.
fn clustered(rect: &Rectified, hits: Vec<Vec<Candidate>>) -> Vec<Candidate> {
    let mut clusters = Clusters::new(rect.frame.width, rect.frame.height);
    for hit in hits.into_iter().flatten() {
        clusters.merge(hit);
    }
    clusters.quorate()
}

/// [`scan`] of the rows of the rectified image, binarised as `mask`, one
/// of its [`Readings`], reads it, and then of its columns: the hits, in
/// the order the scan made them. The columns are scanned as the rows of
/// `turned`, the reading turned over its diagonal, a row at a time, where
/// read down the columns as they lie each pixel was a row's length from
/// the last.
fn scan_both_ways(rect: &Rectified, mask: Reading<'_>, turned: Reading<'_>, max_module: f64) -> Vec<Candidate> {
    let (w, h) = (rect.frame.width, rect.frame.height);
    let mut hits = Vec::new();
    scan(w, h, mask, MIN_MODULE, max_module, &mut hits, false);
    scan(h, w, turned, MIN_MODULE, max_module, &mut hits, true);
    hits
}

/// The readings against the mean of a window a few pixels about each
/// pixel, at each width narrower than the neighbourhood the local
/// threshold has, that [`scan_for_finders`] scans besides the two
/// thresholds: a finder blurred until its rings are ripples reads as
/// rings only against what lies right beside them. See
/// [`super::binarize`].
fn window_binarizers(rect: &Rectified) -> impl Iterator<Item = Binarizer> {
    let reach = rect.local_radius;
    WINDOW_RADII.into_iter().filter(move |&radius| radius as f64 <= reach).map(|radius| Binarizer::Window { radius: radius as f64 })
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
    /// A region of the binarised image lies on the candidate this was
    /// fitted to ([`region_candidates`]): a ring of dark closed on
    /// itself about a middle of its own, which a finder pattern is and
    /// a run of 1:1:3:1:1 across a symbol's data is not. See
    /// [`passed_over`].
    regioned: bool,
}

impl Capstone {
    fn module_mean(&self) -> f64 {
        (self.module[0] + self.module[1]) / 2.0
    }

    /// The capstone is a finder pattern in its own right wherever it
    /// lies, on another symbol as much as on bare paper: a region bears
    /// it out, the whole of it read, and its corners were fitted. See
    /// [`passed_over`].
    fn stands_alone(&self) -> bool {
        self.regioned && !self.partial && self.measured
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
    let fit = |start: [[f64; 2]; 4], thorough: bool| refine_capstone_from(image, rect, candidate, threshold, start, thorough);
    // A finder that is turned, and bloomed or torn besides, is not
    // looked for: the turned start takes the plain fit alone.
    let own = || {
        offsets
            .into_iter()
            .enumerate()
            .find_map(|(i, offsets)| fit(offsets.map(|[dx, dy]| rect.frame.to_image.apply(candidate.x + dx, candidate.y + dy)), i == 0))
    };
    let unfitted = || concentric_capstone(image, rect, candidate, threshold).or_else(|| line_capstone(image, rect, candidate, threshold));
    match candidate.quad {
        None => own().or_else(unfitted),
        // A region's own quadrilateral already has the turn and the
        // perspective of a finder seen steeply, as BoofCV fits its position
        // patterns from the polygon it found: it is tried first.
        Some(quad) if !candidate.scanned => fit(quad, true).or_else(own).or_else(unfitted),
        // A candidate of the scan's with a region lying on it is fitted
        // from its own starts and from the region's quadrilateral, and
        // the scan's fit stands unless the region's is plainly the
        // better ([`region_fit_stands`]); what the quadrilateral fits in
        // part stands behind whatever the candidate would have had
        // without it.
        Some(quad) => {
            let scan = own();
            match fit(quad, true) {
                Some(region) if region_fit_stands(scan.as_ref(), &region) => Some(region),
                torn => scan.or_else(unfitted).or(torn),
            }
        }
    }
}

/// How much more crisply the fit started from a region's quadrilateral
/// must read than the one from the scan's own starts to stand in its
/// place. Both descents climb the one score, the contrast across the
/// finder's boundaries, and where both reach the pattern they settle a
/// fraction of a pixel apart with the contrast between them alike to
/// half a part in a hundred, now the one ahead and now the other: taking
/// whichever was ahead moved a capstone a tenth of a pixel on two
/// benchmark photographs of symbols four pixels a module, and with it
/// the format word of one and the timing pattern of the other, which
/// had read outright and no longer did. Where the scan's starts settle
/// off the pattern, on a finder turned between them or foreshortened,
/// the region's fit reads a third as crisply again or more.
const REGION_FIT_GAIN: f64 = 1.1;

/// Whether the fit of a candidate started from a region's quadrilateral
/// stands before the fit `scan` from the candidate's own starts: when it
/// is the whole pattern, and the scan's starts fitted nothing, or a
/// pattern in part, or one that reads less crisply by
/// [`REGION_FIT_GAIN`]. The scan's fit is the one every photo so far was
/// read through, and stands wherever the two are one fit.
fn region_fit_stands(scan: Option<&Capstone>, region: &Capstone) -> bool {
    !region.partial && scan.is_none_or(|scan| scan.partial || region.contrast > REGION_FIT_GAIN * scan.contrast)
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
        regioned: candidate.quad.is_some(),
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
        regioned: candidate.quad.is_some(),
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
    Some(Capstone {
        corners,
        center,
        unit: rect.unit(center),
        module,
        contrast,
        inverted: candidate.inverted,
        partial,
        threshold,
        line_only: false,
        measured: true,
        regioned: candidate.quad.is_some(),
    })
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
    // A count read by its phase may lie further from the spacing: the blur
    // that kept it from being read by its runs measures the finders'
    // modules wrong, and the spacing over them with them.
    let agrees = |c: &TimingCount| {
        let slack = if c.locked { NAMED_SIZE_SLACK.max(LOCKED_SIZE_SLACK * c.size) } else { NAMED_SIZE_SLACK };
        spacing.is_none_or(|s| (c.size - s).abs() <= slack)
    };
    // What a pattern says, read by its runs, or by its phase where the
    // runs say nothing exact that the spacing agrees with: a blur that
    // runs some of its modules together leaves a count of the rest, or of
    // runs that are no modules, as surely as it leaves none. The count by
    // the runs stands where the phase locks on nothing the spacing agrees
    // with either.
    let believed = |by_runs: Option<TimingCount>, by_phase: &dyn Fn() -> Option<TimingCount>| match by_runs {
        Some(count) if count.exact && agrees(&count) => Some(count),
        _ => by_phase().filter(|count| agrees(count)).or(by_runs),
    };

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
                    let by_runs = timing_between(image, frame, pool[tl], pool[o], p[third]);
                    let by_phase = || timing_by_phase(image, frame, pool[tl], pool[o], p[third], family).map(|(count, _)| count);
                    counts.extend(believed(by_runs, &by_phase));
                }
            }
            None if pool.len() == 2 => {
                let mid = [(p[0][0] + p[1][0]) / 2.0, (p[0][1] + p[1][1]) / 2.0];
                let across = [p[0][1] - p[1][1], p[1][0] - p[0][0]];
                let sides = [1.0, -1.0].map(|side| [mid[0] + side * across[0], mid[1] + side * across[1]]);
                let by_runs = sides.iter().find_map(|&inside| timing_between(image, frame, pool[0], pool[1], inside));
                // By their phase, both sides are read and, of those the
                // spacing agrees with, the one that stands out the more is
                // taken: a timing pattern on one side, and on the other
                // data that may follow some size's alternation well enough
                // to lock, though less closely.
                let by_phase = || {
                    sides
                        .iter()
                        .filter_map(|&inside| timing_by_phase(image, frame, pool[0], pool[1], inside, family))
                        .filter(|(count, _)| agrees(count))
                        .max_by(|a, b| a.1.total_cmp(&b.1))
                        .map(|(count, _)| count)
                };
                counts.extend(believed(by_runs, &by_phase));
            }
            None => {}
        }
    }
    let exact: Vec<f64> = counts.iter().filter(|c| c.exact && agrees(c)).map(|c| c.size).collect();
    if let Some(&n) = exact.first() {
        // Two exact counts that differ: the one nearer the spacing.
        let n = exact.iter().copied().min_by(|a, b| spacing.map_or(0.0, |s| (a - s).abs()).total_cmp(&spacing.map_or(0.0, |s| (b - s).abs()))).unwrap_or(n);
        return ([n, n], true);
    }
    let rough: Vec<f64> = counts.iter().filter(|c| agrees(c)).map(|c| c.size).collect();
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

/// How far, as a share of its size, a count read by its phase may lie
/// from what the finders' spacing says ([`size_hint`]): the version 5
/// symbol blurred by four fifths of a module had its spacing over its
/// finders' modules come to the size of version 8, twelve modules
/// over.
const LOCKED_SIZE_SLACK: f64 = 0.35;

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
    /// Read by the phase of the pattern rather than by its runs
    /// ([`timing_by_phase`]): the size is one a version has, and the
    /// pattern alternated at its pitch and nowhere else, which a blur
    /// that ran the runs together leaves as it was.
    locked: bool,
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
    Some(TimingCount { size: modules as f64 + 16.0, exact: ends && (modules + 16) % 4 == 1, locked: false })
}

/// Samples to a module along a timing pattern read by its phase.
const PHASE_SAMPLES: f64 = 4.0;
/// Rows read either side of where a size puts the timing pattern, in
/// modules: the finders' centres place it to a fraction of a module,
/// and the row read best is taken.
const PHASE_ROWS: [f64; 3] = [-0.25, 0.0, 0.25];
/// Fewest timing modules a size must put between the finders to be told
/// by its phase: four alternations and the next size's pitch is not yet
/// out of step with them.
const PHASE_MIN_MODULES: f64 = 7.0;
/// Fewest pixels of the frame to a module a size may have to be read by
/// its phase: the alternation runs a cycle every two modules, and under
/// a pixel a module that is under two pixels a cycle, finer than the
/// pixels can hold.
const PHASE_MIN_PX: f64 = 1.0;
/// How closely the pattern must follow the alternation a size puts there
/// for the size to be taken from it: its correlation with the
/// alternation times the square root of the timing modules the size has,
/// which is how far it stands out from what data that is no timing
/// pattern gives, in the spreads of that. A blur of six tenths of a
/// module runs the modules of the rows either side into the pattern, at
/// two fifths of its weight, and those are data: a version 7 symbol's
/// pattern so blurred, a ripple a fifteenth of the contrast deep,
/// followed its alternation to 0.43, where sharp it followed it to 0.92,
/// all a square wave follows a cosine. The best of the forty sizes,
/// where nothing alternates, stands out by about two of those spreads.
const LOCKED_TIMING: f64 = 3.0;
/// How much further the size taken must stand out than any other, on
/// the same scale: the sizes either side of the right one put the
/// alternation two cycles out of step over the pattern, and are
/// followed scarcely at all.
const LOCKED_MARGIN: f64 = 1.0;
/// How far a symbol's two timing patterns must stand out together, on the
/// scale of [`LOCKED_TIMING`], their sum over the square root of two, to
/// follow the alternation a placement puts there ([`timing_follows`]),
/// on a photo blurred enough about it to run their modules together
/// ([`RUNS_TOGETHER`]): with nothing to choose among, two and a half
/// spreads. A version 7 symbol blurred by six tenths of a module stood
/// out by 2.3 along its row and 1.9 down its column, the shortest
/// patterns there are that are read so. Random modules blurred so
/// smooth the alternation away with the rest: three thousand placements
/// on them stood out by 1.8 at the most.
const FOLLOWED_TIMING: f64 = 2.5;
/// How far they must stand out together on a photo that shows no such
/// blur: four spreads. Read sharp, random modules, whose fixed modules
/// read no better than chance and leave the timing rows alone to vouch
/// for them, stand out by two and a half about once in three hundred
/// placements; nine thousand stood out by 3.2 at the most. A version 7
/// symbol read sharp stands out by seven, and blurred by half a module,
/// short of where its modules run together, by four and a half.
const FOLLOWED_TIMING_SHARP: f64 = 4.0;
/// The share of a module, at least, by which the blur about a placement
/// spreads an edge along its widest way, as [`super::deblur`] measures
/// it, for its timing patterns to follow at [`FOLLOWED_TIMING`] rather
/// than [`FOLLOWED_TIMING_SHARP`]: three tenths, which a Gaussian blur of
/// four tenths of a module measures as, the measure reading a little
/// under where edges lie close. Past that the patterns' own ripple falls
/// short of four spreads, at six tenths to three, and the alternation
/// is smoothed out of what is no timing pattern. Short of it, random
/// modules stood out by two and a half a few times in a thousand.
const RUNS_TOGETHER: f64 = 0.3;
/// And how far each must stand out on its own: one pattern that follows
/// the alternation does not carry the other that does not.
const FOLLOWED_EACH: f64 = 1.0;

/// The timing pattern between two capstones that stand as the finders
/// along one side of a QR symbol, read by its phase rather than its runs,
/// for when [`timing_between`] reads no runs, or none to believe
/// ([`size_hint`]): each size the family has puts the timing modules at
/// a pitch of the finders' spacing over the size less seven, the first
/// five modules on from a finder's centre and dark, three modules in
/// toward the symbol, which `inside` (a rectified point) says the side
/// of; the photo is read along that row, in the frame, and the size
/// whose alternation the row follows most closely is taken, when it
/// stands out from noise and from every other size ([`LOCKED_TIMING`],
/// [`LOCKED_MARGIN`]), with how far it stands out. `None` otherwise.
///
/// A lens that blurs by half a module and more leaves a timing pattern's
/// modules as a ripple of grey a few hundredths of the contrast deep,
/// which no threshold reads as runs, and the runs were all the size had
/// to go on beside the spacing of the finders over their own modules,
/// which the blur measures wrong: a version 5 symbol blurred by four
/// fifths of a module was fitted as version 7, 9 or 12. The ripple
/// keeps its pitch and its phase however shallow it is, and the finders'
/// centres, which a blur does not move, say what both must be for each
/// size. Nothing here is measured off the finders' own modules, which
/// the blur makes wrong, but the side the symbol lies on.
fn timing_by_phase(image: &Luma, frame: &Frame, a: &Capstone, b: &Capstone, inside: [f64; 2], family: Family) -> Option<(TimingCount, f64)> {
    let (pa, pb) = (frame.to_rectified(a.center), frame.to_rectified(b.center));
    let apart = distance(pa, pb);
    if apart < 1e-6 {
        return None;
    }
    let u = [(pb[0] - pa[0]) / apart, (pb[1] - pa[1]) / apart];
    let mid = [(pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0];
    let mut n = [-u[1], u[0]];
    if (inside[0] - mid[0]) * n[0] + (inside[1] - mid[1]) * n[1] < 0.0 {
        n = [-n[0], -n[1]];
    }
    // Dark as the symbol prints it reads high.
    let ink = if a.inverted { 1.0 } else { -1.0 };
    let mut sizes: Vec<usize> = family.versions().into_iter().map(|v| v.width()).collect();
    sizes.dedup();
    let mut followed: Vec<(f64, usize)> = Vec::new();
    for size in sizes {
        let module = apart / (size as f64 - 7.0);
        // The timing modules lie from five modules past one finder's
        // centre to five short of the other's.
        let (first, last) = (5.0, size as f64 - 12.0);
        if last - first + 1.0 < PHASE_MIN_MODULES || module < PHASE_MIN_PX {
            continue;
        }
        let samples = ((last - first + 1.0) * PHASE_SAMPLES).round() as usize;
        let along: Vec<f64> = (0..samples).map(|k| first - 0.5 + (k as f64 + 0.5) / PHASE_SAMPLES).collect();
        let template: Vec<f64> = along.iter().map(|s| (std::f64::consts::PI * (s - first)).cos()).collect();
        let best = PHASE_ROWS
            .iter()
            .filter_map(|&off| {
                let across = (3.0 + off) * module;
                let profile: Option<Vec<f64>> = along
                    .iter()
                    .map(|s| {
                        let p = [pa[0] + u[0] * s * module + n[0] * across, pa[1] + u[1] * s * module + n[1] * across];
                        let [x, y] = frame.to_image.apply(p[0], p[1]);
                        image.sample(x, y).map(|v| ink * f64::from(v))
                    })
                    .collect();
                correlation(&detrended(&profile?), &template)
            })
            .fold(f64::NEG_INFINITY, f64::max);
        if best.is_finite() {
            followed.push((best * (last - first + 1.0).sqrt(), size));
        }
    }
    followed.sort_by(|x, y| y.0.total_cmp(&x.0));
    let (best, size) = *followed.first()?;
    let next = followed.get(1).map_or(0.0, |f| f.0);
    (best >= LOCKED_TIMING && best - next >= LOCKED_MARGIN).then_some((TimingCount { size: size as f64, exact: true, locked: true }, best))
}

/// `values` less the straight line that fits them best: the light falling
/// off along a row, which says nothing of what alternates along it.
fn detrended(values: &[f64]) -> Vec<f64> {
    let n = values.len() as f64;
    let mean_x = (n - 1.0) / 2.0;
    let mean_y = values.iter().sum::<f64>() / n.max(1.0);
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (i, v) in values.iter().enumerate() {
        let dx = i as f64 - mean_x;
        sxy += dx * (v - mean_y);
        sxx += dx * dx;
    }
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    values.iter().enumerate().map(|(i, v)| v - mean_y - slope * (i as f64 - mean_x)).collect()
}

/// How closely `values` follow `template`, from -1 to 1: their
/// correlation about nought. `None` where either is flat.
fn correlation(values: &[f64], template: &[f64]) -> Option<f64> {
    let (mut vt, mut vv, mut tt) = (0.0, 0.0, 0.0);
    for (v, t) in values.iter().zip(template) {
        vt += v * t;
        vv += v * v;
        tt += t * t;
    }
    (vv > 0.0 && tt > 0.0).then(|| vt / (vv * tt).sqrt())
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

/// Rows of a site's window [`search_site`] scores at once, the caller's
/// among them. A window is a few milliseconds of work, and every thread
/// woken for one idles a while after it before it sleeps: with every
/// window spread over every core, reads of the corpus took a tenth more
/// of the processor's time than with each on one, and spread over four,
/// a thirtieth more, for most of the time saved.
const SEARCH_SITE_THREADS: usize = 4;

/// Look for a site's pattern within `window` modules of where `predicted`
/// puts its centre, with the shape of the patch taken from `map`, expecting
/// its boundaries `bloom` modules into the dark ([`Detection::bloom`]).
/// Returns the anchor and its correspondences: the four corners of a
/// finder, whose shape the template pins down, or the centre alone of a
/// smaller pattern.
#[allow(clippy::too_many_arguments)]
fn search_site(
    image: &Luma,
    func: &FunctionMap,
    site: &Site,
    map: &(impl Project + Sync),
    window: f64,
    inverted: bool,
    threshold: Option<f32>,
    bloom: f64,
) -> Option<(Anchor, Vec<Pair>)> {
    let template = site.template(func);
    if template.fixed() < 4 {
        return None;
    }
    let rect = site.rect_corners();
    let placed = |dx: f64, dy: f64| rect.map(|[x, y]| map.apply(x + dx, y + dy));
    let scorer = Scorer::new(image, &template, placed(0.0, 0.0), threshold).with_along(6).with_bloom(bloom);
    let sign = if inverted { -1.0 } else { 1.0 };
    let steps = (window / SEARCH_STEP).ceil() as i32;
    let timing = site.kind == AnchorKind::Timing;
    let penalty = if timing { TIMING_DISTANCE_PENALTY } else { DISTANCE_PENALTY };
    // Every placement in the window, scored by contrast less a little for
    // its distance from the prediction: a timing pattern repeats every two
    // modules, and a match two modules off is more likely a false one, as
    // ZBar keeps its alignment searches close for the same reason.
    // A row of the window at a time, side by side on a few of the cores
    // to spare and taken in their order: this is the search, and nothing
    // else of it can be spread.
    let rows: Vec<i32> = (-steps..=steps).collect();
    let mut placements: Vec<([[f64; 2]; 4], f64)> = work::Crew::gather(SEARCH_SITE_THREADS - 1)
        .map(&rows, |&iy| {
            let mut row = Vec::new();
            for ix in -steps..=steps {
                let (dx, dy) = (ix as f64 * SEARCH_STEP, iy as f64 * SEARCH_STEP);
                let corners = placed(dx, dy);
                if let Some(score) = scorer.contrast_signed(corners, sign) {
                    row.push((corners, score - penalty * (dx * dx + dy * dy)));
                }
            }
            row
        })
        .concat();
    placements.sort_by(|a, b| b.1.total_cmp(&a.1));
    let moves: &[[f64; 4]] = if site.kind == AnchorKind::Finder { &MOVES } else { &SHIFT };
    let mut evaluations = 0;
    // The best few placements are each settled and read; the first that
    // reads as the pattern wins. A finder looked for expecting a bleed
    // keeps the crispest of them that reads as its pattern instead: it is
    // looked for through corners fitted as printed, as far off as the
    // bleed, and its template begins in the shape they give it. Refitted
    // so, the top-left finder of a version 2 symbol whose dark had
    // thinned by three tenths of a module, with edge points dropped by
    // hand on the edge of its ink, was now and then settled from the
    // first placement squeezed a third of a module across, at a contrast
    // of 0.68 where it read 0.71 on its pattern. Its fit places the
    // symbol's corner ([`bled_outline`]): Refine pressed six times over
    // put that corner from 0.06 of a module off to 0.49, and keeping the
    // crispest from 0.05 to 0.13. One that reads whole still comes before
    // one read with a quarter torn ([`judge`]), however crisply that
    // reads: its corners are the ones a finder's fit places the symbol's
    // corner by.
    let crispest = bloom != 0.0 && site.kind == AnchorKind::Finder;
    let mut found: Option<([[f64; 2]; 4], bool, f64)> = None;
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
                found = Some((corners, false, score));
                break;
            }
            continue;
        }
        if let Some(partial) = accept(&scorer, corners, inverted) {
            if found.is_none_or(|(_, torn, best)| (!partial, score) > (!torn, best)) {
                found = Some((corners, partial, score));
            }
            if !crispest {
                break;
            }
        }
    }
    let (corners, partial, _) = found?;
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
    /// A QR Code's alignment pattern whose template read whole where it
    /// was looked for. It stands as a finder's capstone does, whatever
    /// the plane through the other anchors makes of it: on a bottle the
    /// plane is a module and more out at every pattern away from the
    /// finders, and BoofCV, which has no plane to judge its alignment
    /// patterns by, carries the offset each was found at on to the next
    /// instead ([`carried_offset`]). While the patterns beside it bear
    /// it out ([`agrees_with_neighbours`]) the plane is still fitted
    /// through it with the rest: fitted without the patterns that lie
    /// off it, as it was when they were dropped, the plane placed the
    /// patterns after them worse over renders of bottles, though no
    /// plane fits a bottle. With no pattern beside it, it is taken off
    /// the plane where the plane cannot explain it ([`fit_anchors`]).
    firm: bool,
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
/// carry the geometry. Nor is an alignment pattern that stands firm
/// ([`Placed::firm`]) while its neighbours in the grid bear it out
/// ([`agrees_with_neighbours`]): its template of twenty-five modules
/// read whole, within the window it was looked for in, and on curled
/// paper it is the plane that is out and not the pattern. Under an affine fit the far side of a
/// tilted symbol is off for want of perspective, not for being wrong, so
/// only a perspective fit of the others may condemn an anchor.
fn worst_outlier(placed: &[Placed], grid: Option<&AlignmentGrid>) -> Option<usize> {
    if placed.len() < OUTLIER_MIN_ANCHORS {
        return None;
    }
    // The plane through every anchor, which the offsets of the patterns
    // standing firm are taken from.
    let plane = grid.and_then(|_| Homography::fit(choose_model(&anchors_of(placed), pairs_of(placed).len()), &pairs_of(placed)));
    let mut worst: Option<(usize, f64)> = None;
    for k in 0..placed.len() {
        if placed[k].anchor.kind == AnchorKind::Finder {
            continue;
        }
        // A pattern standing firm is not doubted while the patterns
        // beside it bear it out, nor once it has been taken off the
        // plane ([`fit_anchors`]).
        if placed[k].firm
            && let (Some(grid), Some(plane)) = (grid, &plane)
            && (placed[k].pairs.is_empty() || (has_pattern_beside(grid, placed, placed[k].anchor.at) && agrees_with_neighbours(grid, plane, placed, k)))
        {
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
        let others_placed: Vec<Placed> = others.iter().map(|p| Placed { anchor: p.anchor, pairs: Vec::new(), firm: p.firm }).collect();
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
/// without them. An alignment pattern standing firm with no other
/// beside it ([`Placed::firm`]) is not dropped but taken off the plane:
/// its pairs are emptied, the plane is fitted without it, as it was
/// when such a pattern was dropped, and the map still bends through it.
/// Left in the plane it drew the plane to itself, and the corner beyond
/// it, which the plane places, came out a module short on two symbols
/// that read with the corner where the plane without it puts it.
fn fit_anchors(placed: &mut Vec<Placed>, grid: Option<&AlignmentGrid>) -> Option<(Homography, Model)> {
    loop {
        let anchors = anchors_of(placed);
        let pairs = pairs_of(placed);
        let model = choose_model(&anchors, pairs.len());
        let map = Homography::fit(model, &pairs)?;
        match worst_outlier(placed, grid) {
            Some(k) if placed[k].firm && grid.is_some_and(|grid| !has_pattern_beside(grid, placed, placed[k].anchor.at)) => placed[k].pairs.clear(),
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

/// Whether a QR Code's format word reads outright beside the finders an
/// assignment stands the capstones as, at `turns`: each bit through the
/// corners of the capstone it lies beside ([`finder_map`]) and against
/// that capstone's own dark and light, as BoofCV reads the word, through
/// the corner position pattern's own square (`readFormatRegion0`) and,
/// for the second copy, half through the right-hand pattern's and half
/// through the lower one's (`readFormatRegion1`). The copy round the
/// top-left finder lies whole beside that one capstone; the other is
/// split, eight bits below the top-right finder and seven beside the
/// bottom-left, and takes both.
///
/// The read needs no outline and no version, only which capstone is
/// which finder and how the symbol is turned, and it says whether that
/// is so: taken a quarter turn out, or about another finder, the bits
/// read are data. That makes it the judge where the capstones stand as
/// finders more ways than one ([`align_version`]).
///
/// Only a whole copy that is a word to the bit counts, and when both
/// copies are whole, both spelling the one word. There are thirty-two
/// words among the 2^15 strings of fifteen bits: bits that are no format
/// word spell one outright once in a thousand reads, and lie within the
/// three bits a word corrects more often than not, so that over the
/// handful of placements asked a word with errors would nearly always
/// be found somewhere, and say nothing. A word painted without its mask
/// or with the other model's is not looked for, for the same reason: it
/// would treble the words. A symbol seen from the back spells no word
/// either, its bits running the other way round, and is placed as it
/// was before this was asked.
#[allow(clippy::too_many_arguments)]
fn format_reads_beside(
    image: &Luma,
    func: &FunctionMap,
    finders: &[&Site],
    capstones: &[Capstone],
    assignment: &[Option<usize>],
    turns: u8,
    inverted: bool,
    threshold: Option<f32>,
    band: f32,
) -> bool {
    // Each finder's own map and levels, where a capstone with measured
    // corners stands as it.
    let beside: Vec<Option<(Homography, f32, f32)>> = finders
        .iter()
        .zip(assignment)
        .map(|(site, standing)| {
            let capstone = &capstones[(*standing)?];
            if capstone.line_only {
                return None;
            }
            let (threshold, band_abs) = capstone_levels(image, capstone, threshold, band)?;
            Some((finder_map(capstone, site, turns, false, func)?, threshold, band_abs))
        })
        .collect();
    let words: Vec<u32> = bch::format_words(func.version).into_iter().map(|(_, word)| word).collect();
    let mut spelt: Vec<u32> = Vec::new();
    for copy in &func.format {
        let read = bch::info_read_from(copy, |i| {
            let at = [copy.bits[i].x as f64 + 0.5, copy.bits[i].y as f64 + 0.5];
            let nearest = (0..finders.len()).min_by(|&a, &b| distance(finders[a].at, at).total_cmp(&distance(finders[b].at, at)))?;
            let (map, threshold, band_abs) = beside[nearest].as_ref()?;
            let value = sample_positions(image, map, &copy.bits[i..=i], INNER_FRACTION, POINTS)[0]?;
            ((value - threshold).abs() >= *band_abs).then_some((value < *threshold) != inverted)
        });
        if read.known_count() == u32::from(read.len) {
            if !words.contains(&read.bits) {
                return false;
            }
            spelt.push(read.bits);
        }
    }
    !spelt.is_empty() && spelt.iter().all(|&word| word == spelt[0])
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
    /// full size only when the copy shows nothing, shows the finders of
    /// a symbol it could not fit, or shows a symbol too small for it to
    /// show well, and otherwise only by the finder patterns its regions
    /// show beside the symbols found. BoofCV's detector
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
/// less was found by luck. It stood at a pixel and a half, the least
/// the scan looks for at all ([`MIN_MODULE`]), which a finder square to
/// the copy's rows can be read at and one turned cannot: of three sheets
/// of twenty symbols in one photograph the copy showed the two upright
/// ones at 1.7 pixels a module and nothing of the one turned
/// thirty-seven degrees, and settled the search.
pub const COARSE_MIN_MODULE: f64 = 2.0;
/// Least module size, in pixels of the last shrunk copy searched, of a
/// finder pattern that copy is taken to have shown, whatever it made of
/// it: the photo at its own size is asked only for smaller ones
/// ([`search_beside`]). A copy makes out some finders of two pixels a
/// module and not others, which is why the photo is asked at all: one
/// showed two symbols at 2.1 and nothing of a third of the same module.
/// A finder it did show and could fit no symbol to is no better fitted
/// at full size: two symbols lying mostly under others on a sheet of
/// eight, a finder of each showing at four and a half pixels a module
/// in the copy, cost the look twenty-six seconds to make nothing of a
/// second time.
const COARSE_SHOWN_MODULE: f64 = 3.0;

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
/// any family, no finders of a symbol it could not fit, and no claim it
/// may not rest on ([`doubtful_claims`]): the renders
/// read off their pixels, then the shrunk copies when `search` asks for
/// them, then the photo as it is and the ladder of transformed copies,
/// each copy leaving alone the symbols the copies before it found, and
/// not what is set within one that is no part of it ([`passed_over`]). A
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
    search_for_families(image, families, params, search, credible).detections
}

/// What a search of the whole photo came to.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Search {
    /// The symbols found, as [`find_symbols_of_families`] gives them.
    pub detections: Vec<Detection>,
    /// The finder patterns that stand as a symbol's on which no copy of
    /// the photo had a symbol fitted, and none was found by another.
    pub leads: Vec<Lead>,
}

/// [`find_symbols_of_families`], with what the search found the finders
/// of and could not fit beside what it found.
pub fn search_for_families(
    image: &Luma,
    families: &[Family],
    params: &SampleParams,
    search: &SearchParams,
    credible: &(dyn Fn(&Detection) -> bool + Sync),
) -> Search {
    let mut leads = Vec::new();
    let Some(copy) = reduced_for_search(image) else {
        let detections = search_whole(image, families, params, search, credible, &mut leads);
        return Search { leads: leads_left(leads, &detections), detections };
    };
    let (across, down) = (image.width as f64 / copy.width as f64, image.height as f64 / copy.height as f64);
    // A reading set for the scan reaches as far across the copy as it
    // was set to reach across the photo.
    let in_copy = SampleParams { binarizer: params.binarizer.scaled(2.0 / (across + down)), ..*params };
    let found = search_whole(&copy, families, &in_copy, search, credible, &mut leads);
    let detections = work::map(&found, |found| {
        let mut found = found.clone();
        found.corners = found.corners.map(|[x, y]| [x * across, y * down]);
        for anchor in &mut found.anchors {
            anchor.center = [anchor.center[0] * across, anchor.center[1] * down];
        }
        if work::cancelled() { found } else { refine_found(image, found, params) }
    });
    for lead in &mut leads {
        lead.outline = lead.outline.map(|[x, y]| [x * across, y * down]);
        for finder in &mut lead.finders {
            *finder = [finder[0] * across, finder[1] * down];
        }
    }
    Search { leads: leads_left(leads, &detections), detections }
}

/// The leads of every copy of the photo that the search ends with: those
/// no symbol found lies on, each place once. One copy's lead is another's
/// symbol, and the copies that fit nothing to a set of finders each say
/// so.
fn leads_left(leads: Vec<Lead>, found: &[Detection]) -> Vec<Lead> {
    let within = |outline: &[[f64; 2]; 4], at: [f64; 2]| {
        let side = |i: usize| {
            let (a, b) = (outline[i], outline[(i + 1) % 4]);
            (b[0] - a[0]) * (at[1] - a[1]) - (b[1] - a[1]) * (at[0] - a[0])
        };
        (0..4).all(|i| side(i) >= 0.0) || (0..4).all(|i| side(i) <= 0.0)
    };
    let mut left: Vec<Lead> = Vec::new();
    for lead in leads {
        let centre = quad_centre(lead.outline);
        let settled = found.iter().any(|d| covers(d, centre) || lead.finders.iter().any(|&at| covers(d, at)));
        let said = left.iter().any(|other| within(&other.outline, centre) || within(&lead.outline, quad_centre(other.outline)));
        if !settled && !said {
            left.push(lead);
        }
    }
    left
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
    leads: &mut Vec<Lead>,
) -> Vec<Detection> {
    let whole = [[0.0, 0.0], [image.width as f64, 0.0], [image.width as f64, image.height as f64], [0.0, image.height as f64]];
    let mut all: Vec<Detection> = Vec::new();
    let mut taken: Vec<Detection> = Vec::new();
    // A screenshot or a file an encoder wrote holds one symbol, pixel for
    // pixel, and is read off its pixels; the families it does not read
    // as search the photo. So does the family of a QR Code so read: the
    // box its ink fills is the symbol, and whatever else the file holds
    // is set within it, a second symbol where a logo might be, whose
    // finders the search tells from the first symbol's own
    // ([`passed_over`]).
    let mut pending: Vec<Family> = Vec::new();
    // A symbol read off the pixels may have another set within it.
    let mut render = false;
    for &family in families {
        match detect_pure(image, family, whole, params) {
            Some(found) => {
                let before = taken.len();
                note(&mut all, &mut taken, family, vec![found], credible);
                if taken.len() > before {
                    render = true;
                    pending.push(family);
                }
            }
            None => pending.push(family),
        }
    }
    if pending.is_empty() {
        return all;
    }
    let mut known: Vec<Detection> = Vec::new();
    let mut doubts: Vec<[f64; 2]> = Vec::new();
    // The photo deblurred is looked through for every family not read off
    // the pixels, whatever the copies leave to be searched for at full
    // size: finders a blur ran into blobs are not left over for a copy to
    // show.
    let unread = pending.clone();
    // The families the ladder has looked for through the photo deblurred
    // already, which that look leaves out.
    let mut looked: Vec<Family> = Vec::new();
    if search.coarse_to_fine {
        let mut coarse = search_coarse(image, &pending, params, &mut taken, &mut doubts, credible);
        leads.append(&mut coarse.leads);
        // Only when no family shows anywhere, in a render or a shrunk
        // copy, or the last copy shows a symbol too small for it to
        // show well, is the photo searched at full size for every
        // family; and when the last copy shows finders of a symbol it
        // could not fit, for the families whose finders they are.
        if !coarse.detections.is_empty() || !all.is_empty() {
            if coarse.more.is_empty() {
                // What the copies showed, and what the photo at its own
                // size shows beside it that they did not. Within a symbol
                // read off the pixels it is asked whatever the copies
                // showed, and where there were none: a symbol set within
                // another is the smaller of the two.
                let shown = !coarse.detections.is_empty();
                all.extend(coarse.detections);
                if (render || (shown && coarse.shrunk.is_some())) && !work::cancelled() {
                    let beside = search_beside(image, &pending, params, &mut taken, &all, coarse.shrunk, &mut doubts, credible, leads, &mut looked);
                    all.extend(beside);
                }
                let deblurred = search_deblurred(image, &unread, &looked, params, &mut taken, &all, &mut doubts, credible, leads);
                all.extend(deblurred);
                drop_settled_claims(&mut all, &doubts);
                return all;
            }
            pending = coarse.more;
        }
        known = coarse.detections;
    }
    let fine = search_fine(image, &pending, params, &mut taken, &known, &mut doubts, credible, leads, &mut looked);
    all.extend(known);
    all.extend(fine);
    // Where nothing was found, the ladder has looked through the photo
    // deblurred already.
    if !all.is_empty() {
        let deblurred = search_deblurred(image, &unread, &looked, params, &mut taken, &all, &mut doubts, credible, leads);
        all.extend(deblurred);
    }
    drop_settled_claims(&mut all, &doubts);
    all
}

/// What the photo deblurred shows ([`super::deblur`]) beside the symbols
/// `known`, which the search found in it as it is: nothing, without a
/// look, where the photo shows no blur worth undoing, or none that
/// spreads an edge by a fifth of the smallest module of those symbols
/// ([`super::deblur::WORTH_UNDOING`]). A photograph whose symbols read
/// sharp is not searched again for others blurred past finding; a look
/// costs a search of the photo, and measuring its blur a fraction of
/// one. The ladder looks
/// through the photo deblurred only while it shows no symbol at all, or
/// finders of one it could not fit, and a photograph of three symbols
/// out of focus, one of them large enough to read through the blur,
/// showed the other two to nothing but the corners marked for them:
/// their finder patterns had run into blobs, and the search had a symbol
/// to show and climbed no further. The symbols found are withheld, as a
/// rung withholds them. The families the ladder looked for through the
/// same copy, `looked`, are not looked for again: the rung has searched
/// that copy for them with the symbols found before it withheld, and a
/// second look differs only in withholding besides what the rung and
/// those after it found, whose finders it took already.
#[allow(clippy::too_many_arguments)]
fn search_deblurred(
    image: &Luma,
    families: &[Family],
    looked: &[Family],
    params: &SampleParams,
    taken: &mut Vec<Detection>,
    known: &[Detection],
    doubts: &mut Vec<[f64; 2]>,
    credible: &(dyn Fn(&Detection) -> bool + Sync),
    leads: &mut Vec<Lead>,
) -> Vec<Detection> {
    let families: Vec<Family> = families.iter().copied().filter(|family| !looked.contains(family)).collect();
    if families.is_empty() || work::cancelled() {
        return Vec::new();
    }
    let smallest = known.iter().map(module_size_of).fold(f64::INFINITY, f64::min);
    let Some(spread) = super::deblur::spread_of(&image.data, image.width, image.height) else { return Vec::new() };
    let (widest, _) = spread.extremes();
    if widest < super::deblur::LEAST_SPREAD || widest < super::deblur::WORTH_UNDOING * smallest {
        return Vec::new();
    }
    let Some(view) = ScannedView::through(image, Retry::DEBLURRED, params, &families) else { return Vec::new() };
    let found = search_view(&view, &families, params, taken, known, credible);
    leads.extend(found.leads);
    doubts.extend(doubtful_claims(image, &found.detections, params));
    found.detections
}

/// Add what one family's search of a copy found to `all`, and its
/// credible QR Codes to `taken`.
fn note(all: &mut Vec<Detection>, taken: &mut Vec<Detection>, family: Family, found: Vec<Detection>, credible: &(dyn Fn(&Detection) -> bool + Sync)) {
    if family == Family::Qr {
        taken.extend(found.iter().filter(|d| credible(d)).cloned());
    }
    all.extend(found);
}

/// Where the finder pattern of a symbol that has one lies in the photo.
fn lone_finder_centre(found: &Detection) -> Option<[f64; 2]> {
    let (w, h) = (found.version.width() as f64, found.version.height() as f64);
    Homography::rect_to_quad(w, h, found.corners).map(|map| map.apply(3.5, 3.5))
}

/// The claims among `found` that a search may not rest on: symbols of a
/// family with one finder pattern that do not decode as found, each by
/// where its finder lies in the photo.
///
/// Every finder pattern of a QR Code passes for the finder of a Micro QR
/// or rMQR symbol of its own, and the modules beside it spell a format
/// word of theirs within correction range more often than not. Where a
/// copy shows the QR Code whole, its finders are withheld from the
/// families after it; where it does not, a finder turned too far for
/// that copy's fit or lost in a shadow, the one finder the copy does
/// show is claimed as a symbol of its own, and a search that ended at
/// the first copy to show any symbol ended there: a clean version 1
/// symbol turned thirty-seven degrees, which a search for QR Codes
/// alone read some rungs up the ladder, came back as a Micro QR symbol
/// of version M4 with no block solved. Such a claim is what a reader's
/// decode is the last word on, as ZXing, zxing-cpp and BoofCV return
/// nothing that does not decode, so the decode is asked here: a claim
/// that decodes is a symbol, and one that does not may be a part of one.
fn doubtful_claims(image: &Luma, found: &[Detection], params: &SampleParams) -> Vec<[f64; 2]> {
    found
        .iter()
        .filter(|d| !d.version.family().has_three_finders() && !work::cancelled() && !decodes_as_found(image, d, params))
        .filter_map(lone_finder_centre)
        .collect()
}

/// The families a search goes on looking for after a copy: those the
/// copy's own search left [more](Found::more) of, and, while a
/// [doubtful claim](doubtful_claims) stands on a finder that no symbol
/// with three covers, the families with three finder patterns among
/// those searched, whose finder it may be. The families with one are
/// owed nothing by it: they have had the look they always had, and a QR
/// photo does not go hunting Micro QR symbols up the ladder for a claim
/// it has already made.
fn still_wanted(more: &[Family], families: &[Family], doubts: &[[f64; 2]], found: &[Detection]) -> Vec<Family> {
    let whole = |at: [f64; 2]| found.iter().any(|d| d.version.family().has_three_finders() && covers(d, at));
    let owed = doubts.iter().any(|&at| !whole(at));
    families.iter().copied().filter(|family| more.contains(family) || (owed && family.has_three_finders())).collect()
}

/// Drop the [doubtful claims](doubtful_claims) a symbol with three finder
/// patterns, found since, lies over: the claim stood on one of its
/// finders, which would have been withheld had the symbol shown first.
fn drop_settled_claims(all: &mut Vec<Detection>, doubts: &[[f64; 2]]) {
    let wholes: Vec<Detection> = all.iter().filter(|d| d.version.family().has_three_finders()).cloned().collect();
    all.retain(|d| {
        d.version.family().has_three_finders()
            || !lone_finder_centre(d).is_some_and(|at| doubts.contains(&at) && wholes.iter().any(|whole| covers(whole, at)))
    });
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
            all.leads.extend(found.leads);
            note(&mut all.detections, taken, family, found.detections, credible);
        }
        rest = later;
    }
    all
}

/// The shrink factor of the first copy the coarse search looks at, of an
/// image `width` by `height`: what brings its longer side to
/// [`COARSE_SIDE`].
fn coarse_factor(width: usize, height: usize) -> u8 {
    (width.max(height) as f64 / COARSE_SIDE).ceil().clamp(1.0, 255.0) as u8
}

/// The most a search shrinks an image `width` by `height` to look at it:
/// the coarse search's first copy ([`coarse_factors`]), or the ladder's
/// smallest ([`Retry::LADDER`]). A photo adjusted more than enlarged is
/// made again for each such copy from the photo as loaded
/// ([`Enhanced::shrunk`]), and keeps as much of it as the smallest draws
/// on ([`super::adjust::Adjust::apply_to`]).
pub(crate) fn most_shrunk(width: usize, height: usize) -> u8 {
    let ladder = Retry::LADDER.iter().filter_map(|retry| if let Retry::Downscaled { factor } = retry { Some(*factor) } else { None }).max().unwrap_or(1);
    coarse_factor(width, height).max(ladder)
}

/// The shrink factors of the copies the coarse search looks at, in order:
/// to [`COARSE_SIDE`] on the longer side, then half as much each time,
/// down to half size. None for a photo too small to shrink.
fn coarse_factors(image: &Luma) -> Vec<u8> {
    let mut factor = coarse_factor(image.width, image.height);
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
/// only for the families that have such finders left, or are owed a look
/// by a claim the search may not rest on ([`still_wanted`]), whose finder
/// centres gather in `doubts`. No detections when no copy shows a
/// symbol, or the photo is too small to shrink; `more` is the last
/// copy's, the families to search the photo at full size for.
///
/// The last copy searched is held to what the copies before it are. One
/// that shows a symbol under [`COARSE_MIN_MODULE`] sends the search on
/// to the next larger copy for every family, and after the half-size
/// copy the next larger is the photo: `more` is then every family. It
/// was once whatever finders the last copy left, which for a copy that
/// could not make the finders out is none, and the search ended there:
/// of three symbols in a twelve-megapixel photograph of a dim room, the
/// half-size copy showed one at 1.8 pixels a module, and a version 1
/// symbol of the same module beside it, which the photo at its own size
/// shows and an outline reads, was not looked for.
fn search_coarse(
    image: &Luma,
    families: &[Family],
    params: &SampleParams,
    taken: &mut Vec<Detection>,
    doubts: &mut Vec<[f64; 2]>,
    credible: &(dyn Fn(&Detection) -> bool + Sync),
) -> Searched {
    let mut all = Searched::default();
    let mut wanted: Vec<Family> = families.to_vec();
    // The copy last searched showed a symbol too small for it to show
    // well.
    let mut small = false;
    for factor in coarse_factors(image) {
        if work::cancelled() {
            break;
        }
        let Some(view) = ScannedView::through(image, Retry::Downscaled { factor }, params, &wanted) else { continue };
        let found = search_view(&view, &wanted, params, taken, &all.detections, credible);
        all.leads.extend(found.leads);
        all.shrunk = Some(factor);
        // The modules in the copy's own pixels.
        let smallest = found.detections.iter().map(|d| module_size_of(d) / view.mean_scale()).fold(f64::INFINITY, f64::min);
        let fresh = all.detections.len();
        // Each symbol the copy shows is found again in the photo's own
        // pixels side by side with the rest: a sheet of sixty waited a
        // second and a half for each in turn.
        all.detections.extend(work::map(&found.detections, |found| refine_found(image, found.clone(), params)));
        doubts.extend(doubtful_claims(image, &all.detections[fresh..], params));
        all.more = still_wanted(&found.more, families, doubts, &all.detections);
        if all.detections.is_empty() {
            continue;
        }
        small = smallest < COARSE_MIN_MODULE;
        if !small && all.more.is_empty() {
            break;
        }
        // Symbols too small for the copy to show well may have others
        // beside them it did not show at all, of any family; finders left
        // over are their own family's business.
        if !small {
            wanted = all.more.clone();
        }
    }
    if small {
        all.more = families.to_vec();
    }
    all
}

/// What the photo at its own size shows of `families` beside the symbols
/// `known`, which its shrunk copies found and settled the search on: the
/// symbols its regions' finder patterns propose ([`ScannedView::beside`]),
/// and, where those leave finders of a symbol they could not fit, what
/// the fine search of the families that have them makes of the photo,
/// as it would had a copy left them. `shrunk` is how many times the
/// last copy searched was shrunk, and `None` where the photo was too
/// small to have any, and the symbols known were read off its pixels:
/// every region is then asked.
///
/// A copy says what it shows and nothing of what it does not. The search
/// ended at the first copy whose symbols were all of two pixels a module
/// or more with no finders left over: in a twelve-megapixel photograph
/// of three symbols in a dim room the half-size copy showed two of them
/// at 2.1 pixels a module and nothing of the third, a version 1 symbol
/// of the same module, which read through an outline and was found by a
/// search of the photo at its own size from the start. That search is
/// what the copies are there to spare: it fits a template to every
/// cluster its scan of the rows throws up, grain for the most part, and
/// took a hundred seconds where the copies took one.
///
/// BoofCV looks at every photo at its own size and nowhere else, and can:
/// its position patterns are the squares among the contours of one
/// binarisation, and a twelve-megapixel photograph costs it a tenth of a
/// second. The regions here are that pass, and they alone are asked,
/// those that are the finders of a symbol already found left out, and
/// those of a module the last copy searched, shrunk `shrunk` times,
/// will have shown ([`COARSE_SHOWN_MODULE`]), so that the look costs
/// what the finders the copies missed cost and no more.
#[allow(clippy::too_many_arguments)]
fn search_beside(
    image: &Luma,
    families: &[Family],
    params: &SampleParams,
    taken: &mut Vec<Detection>,
    known: &[Detection],
    shrunk: Option<u8>,
    doubts: &mut Vec<[f64; 2]>,
    credible: &(dyn Fn(&Detection) -> bool + Sync),
    leads: &mut Vec<Lead>,
    looked: &mut Vec<Family>,
) -> Vec<Detection> {
    let shown = shrunk.map_or(f64::INFINITY, |shrunk| COARSE_SHOWN_MODULE * f64::from(shrunk));
    let Some(view) = ScannedView::beside(image, params, families, known, shown) else { return Vec::new() };
    if view.regions.is_empty() {
        return Vec::new();
    }
    let found = search_view(&view, families, params, taken, known, credible);
    leads.extend(found.leads);
    let mut all: Vec<Detection> = known.to_vec();
    all.extend(found.detections);
    doubts.extend(doubtful_claims(image, &all[known.len()..], params));
    let wanted = still_wanted(&found.more, families, doubts, &all);
    if !wanted.is_empty() && !work::cancelled() {
        let more = search_fine(image, &wanted, params, taken, &all, doubts, credible, leads, looked);
        all.extend(more);
    }
    all.split_off(known.len())
}

/// The fine search of `families`: the photo as it is, then the ladder of
/// transformed copies, cheapest step first, until a copy shows a symbol
/// and no finders of another it could not fit. The symbols `known`
/// already, from the shrunk copies, are left alone, and so are those each
/// copy finds by the copies after it: what comes back is what is new.
/// Once any symbol is known, a copy is searched only for the families
/// with such finders left: the others have had the look they always had.
/// A claim of a family with one finder pattern that does not decode
/// keeps the families with three wanted for as long as no symbol of
/// theirs is found over its finder ([`still_wanted`]), up the whole
/// ladder if need be; the centres of such finders gather in `doubts`,
/// those of the shrunk copies' claims among them. The families looked
/// for through the photo deblurred ([`Retry::DEBLURRED`]) gather in
/// `looked`, for [`search_deblurred`] to leave out.
///
/// The search ended at the first copy that showed any symbol, so that of
/// two symbols in a photo, one in shadow, the copy that showed the one
/// was the last looked at, and the rung that would have shown the other
/// was never climbed to. OpenCV's ArUco-based detector thresholds at
/// every window size before it groups anything, and pools the lot.
#[allow(clippy::too_many_arguments)]
fn search_fine(
    image: &Luma,
    families: &[Family],
    params: &SampleParams,
    taken: &mut Vec<Detection>,
    known: &[Detection],
    doubts: &mut Vec<[f64; 2]>,
    credible: &(dyn Fn(&Detection) -> bool + Sync),
    leads: &mut Vec<Lead>,
    looked: &mut Vec<Family>,
) -> Vec<Detection> {
    let mut all: Vec<Detection> = known.to_vec();
    let mut wanted: Vec<Family> = families.to_vec();
    let steps: Vec<Option<Retry>> =
        std::iter::once(None).chain(Retry::LADDER.iter().copied().filter(|step| step.climbed_under(params.binarizer)).map(Some)).collect();
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
                let view = (!work::cancelled()).then(|| view_of(step, &wanted)).flatten();
                let deblurred = view.is_some() && step == Some(Retry::DEBLURRED);
                let found = match view {
                    Some(view) => search_view(&view, &wanted, params, &mut withheld, &[], credible),
                    None => Searched::default(),
                };
                (found, withheld, deblurred)
            },
            |(mut found, withheld, deblurred)| {
                climbed += 1;
                if deblurred {
                    looked.extend(wanted.iter().copied());
                }
                leads.append(&mut found.leads);
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
        doubts.extend(doubtful_claims(image, &all, params));
        wanted = still_wanted(&found.more, families, doubts, &all);
        if wanted.is_empty() {
            return all;
        }
    }
    for &step in &steps[climbed..] {
        if work::cancelled() {
            break;
        }
        let Some(view) = view_of(step, &wanted) else { continue };
        if step == Some(Retry::DEBLURRED) {
            looked.extend(wanted.iter().copied());
        }
        let found = search_view(&view, &wanted, params, taken, &all, credible);
        leads.extend(found.leads);
        let fresh = all.len();
        all.extend(found.detections);
        if all.is_empty() {
            continue;
        }
        doubts.extend(doubtful_claims(image, &all[fresh..], params));
        wanted = still_wanted(&found.more, families, doubts, &all);
        if wanted.is_empty() {
            break;
        }
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
    let found = search_coarse(image, &[family], params, &mut taken.to_vec(), &mut Vec::new(), &|_| true).detections;
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
    search_fine(image, &[family], params, &mut taken.to_vec(), &[], &mut Vec::new(), &|_| true, &mut Vec::new(), &mut Vec::new())
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
    let fine = detect_candidates_in(image, coarse.version, clockwise(coarse.corners), params, false, None, MAX_RECTIFIED).into_iter().next();
    fine.and_then(|fine| vouched(image, fine, params)).unwrap_or(coarse)
}

/// A detection as a search of the whole photo offers it, or `None` when
/// it offers none: its fixed patterns fit, with a little less asked of
/// the fit when the symbol's own bits name the version fitted, and
/// something of the symbol besides its finders reads.
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
///
/// The fit is asked two ways, and either will do. The detection's own
/// score reads every fixed module against one split of them all, and a
/// shadow that lies across the symbol defeats it: the paper in the shade
/// is darker than the split, every light module there counts as wrong,
/// and a version 2 symbol with a straight shadow over two fifths of it,
/// found, timed and fitted, scored seven parts in ten and was not
/// offered, at six pixels a module and at forty, though it read at once
/// through an outline. So the fixed modules are counted again as a read
/// will first read them, each against its own neighbourhood unless
/// another rule is asked for, over every fixed module of the version, so
/// that one off the photo counts against the fit as it does in the
/// score; that is how the patterns beyond the finders have been asked
/// all along. BoofCV reads every module against its position patterns'
/// own thresholds, carried across the symbol, to the same end. What the
/// second count lets through is a symbol in uneven light and nothing
/// else: a module too faint to call counts against the fit, as bare
/// paper and flat print read, the share asked is the same, and whatever
/// fits must still be vouched for by a word of the symbol's own or by
/// its patterns beyond the finders, which three marks in any light are
/// not.
///
/// A detection the second count alone lets through comes back with that
/// count as its agreement. Whoever lists the symbols of every family
/// ranks two claims on one place by their scores and takes a symbol
/// below three parts in four for no symbol; scored by the split the
/// shadow defeats, the QR Code was offered and then dropped for a Micro
/// QR symbol that fitted one of its finders better. One its timing
/// patterns alone let through keeps the score it had: the blur that
/// reads its fixed modules wrong reads them wrong in the second count
/// too, which would only score it lower.
fn vouched(image: &Luma, mut found: Detection, params: &SampleParams) -> Option<Detection> {
    let named = found.info.named_version == Some(found.version);
    let floor = if named { MIN_NAMED_SCORE } else { MIN_FOUND_SCORE };
    let worded = found.version_confirmed() || found.format_reads_outright();
    if found.score() >= floor && worded {
        return Some(found);
    }
    // Sampled once, as a read will sample it, for both the questions
    // still open.
    let func = FunctionMap::new(found.version);
    let grid = grid_as_found(image, &func, &found, params)?;
    // Both timing patterns following their alternation through the
    // placement vouch for it whatever its fixed modules read as: a blur
    // that runs a symbol's thin patterns together reads half its fixed
    // modules wrong, where the alternation keeps its pitch and its phase.
    // Asked only where nothing else vouches, since it may measure the
    // blur about the placement ([`timing_follows`]).
    let timed = std::cell::OnceCell::new();
    let timed = || *timed.get_or_init(|| timing_follows(image, &func, &found));
    let mut agreement = found.agreement;
    if found.score() < floor {
        let as_read = func.fixed_agreement(&grid).0;
        let counted = as_read as f64 >= floor * found.agreement.1.max(1) as f64;
        if !counted && !timed() {
            return None;
        }
        if counted {
            agreement.0 = as_read;
        }
    }
    let vouched = worded || reads_beyond_finders(&func, &grid) || timed();
    found.agreement = agreement;
    vouched.then_some(found)
}

/// Whether both timing patterns of a QR Code follow the alternation a
/// placement puts there: read through the placement at
/// [`PHASE_SAMPLES`] to a module, from the first timing module to the
/// last, the two standing out together ([`timing_stands`]) to
/// [`FOLLOWED_TIMING`] where the photo about it is blurred enough to run
/// their modules together ([`blurred_about`]), and elsewhere to
/// [`FOLLOWED_TIMING_SHARP`], and each to [`FOLLOWED_EACH`], as
/// [`timing_by_phase`] has a pattern follow the alternation of a size.
/// The placement fixes the phase, and there is nothing to choose among.
/// False for the other families, and for a version whose patterns are
/// too short to tell ([`PHASE_MIN_MODULES`]).
///
/// The search vouches for a placement by its patterns beyond the
/// finders ([`reads_beyond_finders`]), each module of them read dark or
/// light, and a symbol blurred by a hand that moved, which read through
/// an outline drawn round it, had its timing modules read no better
/// than a coin would and went unlisted. The alternation they make is
/// there as a ripple, at the pitch and the phase the placement says,
/// and two patterns of print that are not a symbol's do not follow it.
/// Followed at two and a half spreads whatever the blur, random modules
/// read sharp vouched for a placement on them about once in three
/// hundred times, where the blur that is the reason to ask has smoothed
/// the alternation out of them.
fn timing_follows(image: &Luma, func: &FunctionMap, found: &Detection) -> bool {
    timing_stands(image, func, found).is_some_and(|z| z >= FOLLOWED_TIMING_SHARP || (z >= FOLLOWED_TIMING && blurred_about(image, found)))
}

/// How far both timing patterns of a QR Code stand out together from
/// data that is no timing pattern, following the alternation a placement
/// puts there ([`timing_follows`]): their sum over the square root of two,
/// on the scale of [`LOCKED_TIMING`]. `None` where either stands out by
/// less than [`FOLLOWED_EACH`], for the other families, and for a version
/// whose patterns are too short to tell ([`PHASE_MIN_MODULES`]).
fn timing_stands(image: &Luma, func: &FunctionMap, found: &Detection) -> Option<f64> {
    if !matches!(func.version.family(), Family::Qr | Family::Model1) {
        return None;
    }
    let map = found.projection()?;
    let (first, last) = (8.0, func.width as f64 - 9.0);
    if last - first + 1.0 < PHASE_MIN_MODULES {
        return None;
    }
    let ink = if found.inverted { 1.0 } else { -1.0 };
    let samples = ((last - first + 1.0) * PHASE_SAMPLES).round() as usize;
    let along: Vec<f64> = (0..samples).map(|k| first - 0.5 + (k as f64 + 0.5) / PHASE_SAMPLES).collect();
    let template: Vec<f64> = along.iter().map(|s| (std::f64::consts::PI * (s - first)).cos()).collect();
    // The row and the column, each through its modules' centres, and how
    // far each stands out.
    let stands: Option<Vec<f64>> = [true, false]
        .into_iter()
        .map(|row| {
            let profile: Option<Vec<f64>> = along
                .iter()
                .map(|&s| {
                    let [x, y] = if row { map.apply(s + 0.5, 6.5) } else { map.apply(6.5, s + 0.5) };
                    image.sample(x, y).map(|v| ink * f64::from(v))
                })
                .collect();
            correlation(&detrended(&profile?), &template).map(|r| r * (last - first + 1.0).sqrt())
        })
        .collect();
    let stands = stands?;
    stands.iter().all(|&z| z >= FOLLOWED_EACH).then(|| stands.iter().sum::<f64>() / 2f64.sqrt())
}

/// Whether the photo about a placement shows a blur that runs its
/// timing patterns' modules together: one that spreads an edge, along
/// its widest way, by [`RUNS_TOGETHER`] of a module or more, measured
/// over the box about its corners. A share of a module alone, and no
/// least spread in pixels: a symbol of two and a half pixels a module
/// blurred by six tenths of one measures a pixel across, and its timing
/// patterns are as far run together as those of a larger one.
fn blurred_about(image: &Luma, found: &Detection) -> bool {
    let (x0, x1) = found.corners.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p[0]), a.1.max(p[0])));
    let (y0, y1) = found.corners.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p[1]), a.1.max(p[1])));
    if !(x0.is_finite() && x1.is_finite() && y0.is_finite() && y1.is_finite()) {
        return false;
    }
    let window = [
        x0.floor().clamp(0.0, image.width as f64) as usize,
        y0.floor().clamp(0.0, image.height as f64) as usize,
        x1.ceil().clamp(0.0, image.width as f64) as usize,
        y1.ceil().clamp(0.0, image.height as f64) as usize,
    ];
    let (w, h) = (window[2].saturating_sub(window[0]), window[3].saturating_sub(window[1]));
    if w == 0 || h == 0 {
        return false;
    }
    let part = super::adjust::cut(&image.data, image.width, 1, window);
    super::deblur::spread_of(&part, w, h).is_some_and(|spread| spread.extremes().0 >= RUNS_TOGETHER * module_size_of(found))
}

/// Whether the patterns of a symbol that are none of its finders read in
/// `grid`, the symbol sampled through a detection as a read would sample
/// it: [`MIN_BEYOND_FINDERS`] of their modules the right colour, a module
/// too faint to call counting against, as blank paper reads. Both timing
/// patterns of a Micro QR symbol are asked, each on its own: hung off a
/// finder of a QR Code and turned to lie along that symbol's edge, one
/// of the two lies on the QR Code's own timing pattern and reads without
/// a fault. An rMQR symbol's timing patterns are taken together, and so
/// are a QR Code's, whose alignment patterns will do in their place: a
/// scratch along a timing pattern takes all of it, and the three finders
/// of a QR Code already say more than the one of the other families.
fn reads_beyond_finders(func: &FunctionMap, grid: &ModuleGrid) -> bool {
    let family = func.version.family();
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
    /// of the copy, across and down ([`View::scale`]).
    origin: [f64; 2],
    scale: [f64; 2],
    /// The rung of the ladder the copy was made by; `None` for the photo
    /// as it is.
    step: Option<Retry>,
    /// The reach of the scan's local threshold, where the rung narrows it.
    local_radius: Option<f64>,
    /// How the scan reads the copy, its reach in the copy's pixels.
    binarizer: Binarizer,
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
        Self::new(Cow::Borrowed(image), [0.0, 0.0], [1.0; 2], None, None, params, families, true)
    }

    /// The photo transformed by `step`, scanned for `families`; `None`
    /// when the copy cannot be made.
    fn through(image: &'a Luma, step: Retry, params: &SampleParams, families: &[Family]) -> Option<Self> {
        let view = View::whole(image, step)?;
        let params = SampleParams { binarizer: params.binarizer.scaled(1.0 / view.mean_scale()), ..*params };
        Self::new(Cow::Owned(view.luma), view.origin, view.scale, Some(step), view.local_radius, &params, families, true)
    }

    /// The photo as it is, with the finder patterns its regions show
    /// that lie on none of the symbols `known`, and none of the scan's:
    /// what [`search_beside`] looks at. The rows and columns are not
    /// scanned. A region has read as a finder along two lines of its
    /// own, and the clusters of the scan have passed no test, are most
    /// of them grain, and are what a search of a large photo at its own
    /// size spends its time refining. The finders of a symbol with one
    /// finder pattern are kept: one may have been claimed off a finder
    /// of a QR Code no copy showed whole. So is a region that lies on a
    /// symbol and on none of its finders, which may be the finder of
    /// another set within it ([`passed_over`]). Those of `shown` photo
    /// pixels a module or more are left out with those on a symbol: the
    /// copies have shown them, and made of them what there is to make.
    fn beside(image: &'a Luma, params: &SampleParams, families: &[Family], known: &[Detection], shown: f64) -> Option<Self> {
        let mut view = Self::new(Cow::Borrowed(image), [0.0, 0.0], [1.0; 2], None, None, params, families, false)?;
        let frame = view.rect.frame;
        // Rectified pixels to a photo pixel.
        let scale = frame.size[0] / image.width.max(1) as f64;
        view.regions.retain(|region| {
            let [x, y] = frame.to_image.apply(region.x, region.y);
            region.module_mean() < shown * scale && !known.iter().any(|found| found.version.family().has_three_finders() && passed_over(found, [x, y], true))
        });
        view.refined[1] = vec![OnceLock::new(); view.regions.len()];
        Some(view)
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        luma: Cow<'a, Luma>,
        origin: [f64; 2],
        scale: [f64; 2],
        step: Option<Retry>,
        local_radius: Option<f64>,
        params: &SampleParams,
        families: &[Family],
        scanned: bool,
    ) -> Option<Self> {
        let (w, h) = (luma.width as f64, luma.height as f64);
        let region = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
        let rect = Rectified::with_longest(&luma, region, MAX_RECTIFIED_WHOLE)?.narrowed(local_radius).ruled(params.binarizer);
        let threshold = fixed_threshold(params);
        let widest = families.iter().map(|&f| max_module_fraction(f)).fold(0.0, f64::max);
        let max_module = rect.frame.size[0].min(rect.frame.size[1]) * widest;
        let readings = Readings::of(&rect, threshold, scanned);
        let mut candidates = if scanned { scan_for_finders(&rect, &readings, max_module) } else { Vec::new() };
        let regions = region_candidates(&rect, &readings, max_module, &mut candidates);
        let refined = [vec![OnceLock::new(); candidates.len()], vec![OnceLock::new(); regions.len()]];
        Some(Self { luma, origin, scale, step, local_radius, binarizer: params.binarizer, rect, threshold, candidates, regions, refined })
    }

    fn to_photo(&self, p: [f64; 2]) -> [f64; 2] {
        [self.origin[0] + p[0] * self.scale[0], self.origin[1] + p[1] * self.scale[1]]
    }

    /// Photo pixels per pixel of the copy as a length ([`View::mean_scale`]).
    fn mean_scale(&self) -> f64 {
        (self.scale[0] + self.scale[1]) / 2.0
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
    /// first, then its half ones, as many of each as [`within_cap`]
    /// lets through, then the regions', each within the family's cap on
    /// the module size, with repeats merged.
    ///
    /// `settled` says of a candidate, and the point of the photo it lies
    /// at, that it lies within a symbol already found and is one the
    /// caller will not offer again. Such a candidate is not refined, where
    /// that changes nothing of what the caller is left with: every copy
    /// after the one that found a symbol fitted a template again to each
    /// of its finders and to every pattern in its data that passed for
    /// one, a third to a half of the candidates of a sheet of sixty
    /// symbols half found, and then dropped the lot. One thing a
    /// capstone that is to be dropped still decides is which partial
    /// ones are too faint beside it to keep ([`drop_faint_partials`]),
    /// so where a partial capstone would stand for want of anything
    /// stronger among those refined, every candidate is refined after
    /// all, as they all once were.
    fn capstones(&self, family: Family, settled: &dyn Fn(&Candidate, [f64; 2]) -> bool) -> Vec<Capstone> {
        let max_module = self.rect.frame.size[0].min(self.rect.frame.size[1]) * max_module_fraction(family);
        let within = |c: &Candidate| c.module.iter().all(|&m| m <= max_module);
        let scan = |half: bool, most: usize| {
            within_cap(self.candidates.iter().enumerate().filter(move |(_, c)| c.half == half && within(c)), most).map(|(i, _)| (0, i))
        };
        let regions = (0..self.regions.len()).filter(|&i| within(&self.regions[i])).map(|i| (1, i));
        let in_turn: Vec<(usize, usize)> = scan(false, MAX_SEARCH_CANDIDATES).chain(scan(true, MAX_HALF_CANDIDATES)).chain(regions).collect();
        let spared = |source: usize, i: usize| {
            let candidate = if source == 0 { &self.candidates[i] } else { &self.regions[i] };
            settled(candidate, self.to_photo(self.rect.frame.to_image.apply(candidate.x, candidate.y)))
        };
        // The candidates are fitted side by side on the cores to spare,
        // each fit being its own, and gathered in their turn.
        let gather = |spare: bool| {
            let wanted: Vec<(usize, usize)> = in_turn.iter().copied().filter(|&(source, i)| !(spare && spared(source, i))).collect();
            let mut all: Vec<Capstone> = Vec::new();
            for capstone in work::map(&wanted, |&(source, i)| self.refined(source, i)).into_iter().flatten() {
                merge_capstone(&mut all, &self.rect, capstone);
            }
            all
        };
        let mut all = gather(true);
        let strongest = all.iter().filter(|c| !c.partial).map(|c| c.contrast).fold(f64::NEG_INFINITY, f64::max);
        if all.iter().any(|c| c.partial && !(c.contrast < PARTIAL_CONTRAST_SHARE * strongest)) {
            all = gather(false);
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
    /// where one does ([`lies_at`]), is passed by: on a sheet of symbols
    /// most triples are three finders of three neighbours, and finding
    /// the neighbours disposes of them. Then the capstones no symbol was
    /// found on propose in pairs, as they do inside an outline, for the
    /// symbol with a finder torn off, cut by the edge of the photo, or
    /// missed: two along a side with the timing pattern between them, or
    /// two across the diagonal that lie turned as a symbol's would.
    /// OpenCV's detectors and BoofCV's want all three finders; the pairs
    /// are the outline search's own, and without them a symbol that read
    /// in a loose outline was not found in the photo around it.
    ///
    /// Two along a side with no timing pattern read between them propose
    /// too, where they face each other as the finders along a side do
    /// ([`finders_face`]), among the proposals nothing speaks for. They
    /// once proposed nothing, and a copy that showed one symbol whole
    /// and two finders of another was the last the search looked at: no
    /// triple, no pair, so nothing fitted in vain, and nothing to say
    /// there was [more](Found::more). A version 5 symbol turned thirty
    /// degrees and blurred by six tenths of a module, which a search
    /// read where it lay alone, some rungs up the ladder, was not found
    /// beside a crisp version 4: the photo as it was showed the crisp
    /// one, and two finders of the other with a timing pattern too
    /// blurred to count between them. BoofCV joins two position patterns
    /// on how they face and nothing else, and reads what lies between
    /// afterwards; the fit of the pair's outline looks for the third
    /// finder at the outline's own scale, as a hand-drawn outline would
    /// have it looked for, and found it. On the photographs of BoofCV's
    /// benchmark two symbols are found so that were not found at all.
    fn find(&self, family: Family, params: &SampleParams, taken: &[Detection], known: &[Detection]) -> Found {
        let image: &Luma = &self.luma;
        let threshold = self.threshold;
        // The proposals are fitted in the copy's own pixels, where a
        // reading set for the scan reaches as the copy was scanned.
        let params = &SampleParams { binarizer: self.binarizer, ..*params };
        // The finders of a symbol already found are not offered again, as
        // those of a symbol found here are not below. A symbol with one
        // finder may have been claimed off a finder of a QR Code no copy
        // has yet shown whole, so a search for QR Codes leaves alone only
        // the finders of the QR Codes known.
        let theirs = |d: &&Detection| !family.has_three_finders() || d.version.family().has_three_finders();
        let settled = || taken.iter().chain(known.iter().filter(theirs));
        // Not offered, so not refined either, where the scan has them
        // within the symbol's own outline: the fit moves a candidate a
        // fraction of a module, and a symbol found covers two modules
        // about itself. The finder of another symbol set within one
        // found is offered, and so refined ([`passed_over`]).
        let mut capstones = self.capstones(family, &|candidate, at| settled().any(|d| holds(d, at) && passed_over(d, at, candidate.quad.is_some())));
        capstones.retain(|c| {
            let at = self.to_photo(c.center);
            !settled().any(|d| passed_over(d, at, c.stands_alone()))
        });
        /// An outline to detect, the capstones it stands on, and what
        /// the photo shows of a symbol there beyond them.
        struct Proposal {
            badness: f64,
            outline: [[f64; 2]; 4],
            members: Vec<usize>,
            /// A timing pattern reads between its capstones.
            timed: bool,
            /// The format word reads outright beside its capstones.
            worded: bool,
            /// Something a symbol alone would put there speaks for it:
            /// a timing pattern between its capstones or running from
            /// its one, a format word outright beside them, or the
            /// symbol decoding through its one finder's own square.
            /// See [`MAX_UNSPOKEN_PROPOSALS`].
            spoken: bool,
        }
        // Whether the format word reads outright beside capstones standing
        // as a QR Code's top-left, top-right and bottom-left finders, each
        // bit through the square of the finder it lies by, whichever way
        // up the capstones' corners have the symbol. With a finder
        // wanting, the copy of the word that lies whole beside those
        // there are is the one read.
        let reference = FunctionMap::new(reference_version(family));
        let sites = sites(&reference);
        let finders: Vec<&Site> = sites.iter().filter(|s| s.kind == AnchorKind::Finder).collect();
        let worded = |standing: [Option<usize>; 3]| {
            let Some(&first) = standing.iter().flatten().next() else { return false };
            let inverted = capstones[first].inverted;
            (0..4u8).any(|turns| format_reads_beside(image, &reference, &finders, &capstones, &standing, turns, inverted, threshold, params.band))
        };
        let mut proposals: Vec<Proposal> = match family {
            Family::Qr | Family::Model1 => qr_triples(image, &self.rect.frame, &capstones)
                .iter()
                .map(|t| {
                    let (timed, worded) = (t.timed > 0, worded(t.roles.map(Some)));
                    Proposal { badness: t.badness, outline: t.outline, members: t.members.to_vec(), timed, worded, spoken: timed || worded }
                })
                .collect(),
            Family::Micro | Family::Rmqr => capstones
                .iter()
                .enumerate()
                .flat_map(|(i, c)| {
                    lone_finder_proposals(image, c, family, threshold, params).into_iter().map(move |p| Proposal {
                        badness: p.badness,
                        outline: p.outline,
                        members: vec![i],
                        timed: false,
                        worded: false,
                        spoken: p.spoken,
                    })
                })
                .collect(),
        };
        // Those something speaks for first, and each kind best first.
        let in_turn = |a: &Proposal, b: &Proposal| b.spoken.cmp(&a.spoken).then(a.badness.total_cmp(&b.badness));
        proposals.sort_by(in_turn);
        let start = match family {
            Family::Qr => SymbolVersion::Qr(2),
            Family::Micro => SymbolVersion::Micro(2),
            Family::Rmqr => SymbolVersion::Rmqr(11),
            Family::Model1 => SymbolVersion::Model1(2),
        };
        // Whether a capstone is one of a symbol's found here, or lies on
        // one and is passed over for it.
        let spent = |found: &[Detection], m: usize| found.iter().any(|d| passed_over(d, capstones[m].center, capstones[m].stands_alone()));
        // A capstone kept on a symbol already found is the finder of a
        // symbol set within it or of none: an outline that stands on one
        // and lies where the symbol found does is that symbol's own,
        // proposed again by a finder the symbol's corners put more than
        // a separator's width from where it lies.
        let lodged: Vec<bool> = capstones.iter().map(|c| settled().any(|d| covers(d, self.to_photo(c.center)))).collect();
        let stale = |p: &Proposal| {
            p.members.iter().any(|&m| lodged[m]) && {
                let outline = p.outline.map(|c| self.to_photo(c));
                p.members.iter().any(|&m| settled().any(|d| covers(d, self.to_photo(capstones[m].center)) && lies_at(d, &outline)))
            }
        };
        // Whether a proposal is still worth detecting: no symbol found
        // lies where it does, or stands on a capstone it stands on.
        let stands = |found: &[Detection], p: &Proposal| !found.iter().any(|d| lies_at(d, &p.outline)) && !p.members.iter().any(|&m| spent(found, m)) && !stale(p);
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
                .and_then(|d| vouched(image, d, params))
        };
        // The proposals in their turn, until `most` have come to nothing,
        // or of those nothing speaks for `most_unspoken`, after which
        // only the others are fitted; each detected unless a symbol found
        // by then lies where it does.
        // The fit is nearly all the time a search takes, and says the
        // same whatever was found before it. The first is made alone: on
        // a sound photo the best proposal is the symbol, and disposes of
        // the rest. Once one has come to nothing, the proposals next in
        // line are fitted side by side on the threads standing idle, and
        // still taken one at a time: a fit made for a proposal that a
        // symbol found meanwhile disposes of is thrown away, as on one
        // core it would never have been made. With no thread idle, as in
        // a search beside others, they are fitted one at a time, as they
        // would be on one core, rather than a batch at once by the search
        // alone.
        let detect_each = |proposals: &[Proposal], most: usize, most_unspoken: usize, found: &mut Vec<Detection>, unfitted: &mut Vec<Unfitted>| {
            let mut fits: Vec<Option<Option<Detection>>> = vec![None; proposals.len()];
            let (mut failed, mut unspoken) = (0, 0);
            let mut reached = proposals.len();
            for (i, p) in proposals.iter().enumerate() {
                if failed >= most || work::cancelled() {
                    reached = i;
                    break;
                }
                if !stands(found.as_slice(), p) || (!p.spoken && unspoken >= most_unspoken) {
                    continue;
                }
                if fits[i].is_none() {
                    let crew = work::Crew::gather_idle(if failed == 0 { 0 } else { FITS_AT_ONCE.min(most - failed) - 1 });
                    // No more of those nothing speaks for are fitted
                    // ahead than may still fail.
                    let mut room = most_unspoken.saturating_sub(unspoken);
                    let ahead: Vec<usize> = (i..proposals.len())
                        .filter(|&j| fits[j].is_none() && stands(found.as_slice(), &proposals[j]))
                        .filter(|&j| {
                            let fitted = proposals[j].spoken || room > 0;
                            if fitted && !proposals[j].spoken {
                                room -= 1;
                            }
                            fitted
                        })
                        .take(crew.width())
                        .collect();
                    let fitted = crew.map(&ahead, |&j| if work::cancelled() { None } else { fit(&proposals[j]) });
                    for (&j, fitted) in ahead.iter().zip(fitted) {
                        fits[j] = Some(fitted);
                    }
                }
                let detection = fits[i].take().flatten().filter(|d| !found.iter().any(|f| lies_at(f, &d.corners)));
                match detection {
                    Some(detection) => found.push(detection),
                    None => {
                        failed += 1;
                        if !p.spoken {
                            unspoken += 1;
                        }
                        if p.spoken && family.has_three_finders() {
                            unfitted.push(Unfitted { members: p.members.clone(), outline: p.outline, timed: p.timed, worded: p.worded });
                        }
                    }
                }
            }
            // A proposal the budget ran out before, with a timing pattern
            // between its capstones, is as much a symbol the copy shows
            // and did not fit as one fitted to no avail: it was once
            // passed over without a word, and the copy that had spent
            // its dozen on other print settled the search.
            for p in proposals[reached..].iter().filter(|p| p.spoken && family.has_three_finders() && stands(found.as_slice(), p)) {
                unfitted.push(Unfitted { members: p.members.clone(), outline: p.outline, timed: p.timed, worded: p.worded });
            }
        };
        let mut found: Vec<Detection> = Vec::new();
        // The proposals something spoke for that came to nothing, of a
        // family whose symbols have three finders: one finder and a few
        // modules beside it are imitated too easily to be worth a word.
        let mut unfitted: Vec<Unfitted> = Vec::new();
        let most_unspoken = if family.has_three_finders() { MAX_UNSPOKEN_PROPOSALS } else { MAX_UNSPOKEN_LONE_PROPOSALS };
        detect_each(&proposals, MAX_PROPOSALS, most_unspoken, &mut found, &mut unfitted);
        if family.has_three_finders() {
            // The capstones no symbol was found on, in pairs. A partial
            // capstone's centre is placed by three quarters of it, and a
            // pair's outline hangs on both centres.
            let left: Vec<usize> = (0..capstones.len())
                .filter(|&i| !capstones[i].partial && capstones[i].measured && !spent(&found, i))
                .collect();
            let spare: Vec<Capstone> = left.iter().map(|&i| capstones[i]).collect();
            let proposed = qr_pairs(image, &self.rect.frame, &spare);
            // The timing pattern between two along a side lies on the
            // symbol's side of them and says which that is. Where none
            // reads on either, each side is proposed for two that face
            // each other, with nothing to speak for it.
            let timed: Vec<[usize; 2]> = proposed.iter().filter(|p| matches!(p.kind, PairKind::Side { timed: true })).map(|p| p.members).collect();
            let mut pairs: Vec<Proposal> = proposed
                .into_iter()
                .filter(|p| match p.kind {
                    PairKind::Side { timed: true } => true,
                    PairKind::Side { timed: false } => !timed.contains(&p.members) && finders_face(&spare[p.members[0]], &spare[p.members[1]], &self.rect.frame),
                    PairKind::Diagonal => finders_lie_diagonal(&spare[p.members[0]], &spare[p.members[1]]),
                })
                .map(|p| {
                    let timed = matches!(p.kind, PairKind::Side { timed: true });
                    let side = matches!(p.kind, PairKind::Side { .. });
                    // Two along a side are the top-left finder and
                    // another, either way about, and the word's first
                    // copy lies whole beside the top-left one; two at the
                    // ends of the diagonal have its second between them.
                    let (a, b) = (left[p.members[0]], left[p.members[1]]);
                    let worded = if side {
                        [[Some(a), Some(b), None], [Some(b), Some(a), None]].into_iter().any(&worded)
                    } else {
                        [[None, Some(a), Some(b)], [None, Some(b), Some(a)]].into_iter().any(&worded)
                    };
                    Proposal { badness: p.badness, outline: p.outline, members: vec![a, b], timed, worded, spoken: timed || worded }
                })
                .collect();
            pairs.sort_by(in_turn);
            detect_each(&pairs, MAX_PAIR_PROPOSALS, MAX_UNSPOKEN_PAIR_PROPOSALS, &mut found, &mut unfitted);
        }
        // Capstones with a timing pattern between them and no symbol
        // found on any of them: a symbol this copy shows and could not
        // fit, which another copy may.
        let loose = |u: &Unfitted| u.members.iter().all(|&m| !spent(&found, m));
        let more = unfitted.iter().any(|u| u.timed && loose(u));
        // And what is left to say of them should no copy fit them: where
        // those lie that a format word spoke for, each set of finders
        // once, as its best proposal had it. A timing pattern is not
        // enough to say it on: two letters of a line of print, or two
        // cells of a screen's weave, have the like between them, and of
        // thirty-one sets of finders so timed and not fitted in the
        // photographs of BoofCV's benchmark three were a symbol's.
        let mut leads: Vec<Lead> = Vec::new();
        let mut led: Vec<usize> = Vec::new();
        for u in unfitted.iter().filter(|u| u.worded && loose(u) && !found.iter().any(|d| lies_at(d, &u.outline))) {
            if u.members.iter().any(|m| led.contains(m)) {
                continue;
            }
            led.extend(&u.members);
            leads.push(Lead {
                family,
                outline: u.outline.map(|c| self.to_photo(c)),
                finders: u.members.iter().map(|&m| self.to_photo(capstones[m].center)).collect(),
            });
        }
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
        Found { detections: found, more, leads }
    }
}

/// A proposal something in the photo spoke for, to which no symbol was
/// fitted: the capstones it stood on, its outline in the copy's pixels,
/// and whether a timing pattern read between its capstones and a format
/// word outright beside them.
struct Unfitted {
    members: Vec<usize>,
    outline: [[f64; 2]; 4],
    timed: bool,
    worded: bool,
}

/// Finder patterns that stand as one symbol's, the symbol's format word
/// reading outright beside them, on which a search of the whole photo
/// fitted no symbol: what it has to say of a symbol it could not find.
/// BoofCV hands back the candidates it could not decode beside its
/// detections, each with the stage it failed at and its bounds, and a
/// damaged symbol among sound ones is still a symbol; here a detection
/// that does not decode is handed back already, and this is the stage
/// before it, the finders found and the fit to them failed: a symbol
/// half under another sheet of paper, or cut by the edge of the photo,
/// two of its finders and the word between them showing. The outline is
/// what the search proposed and fitted in vain, loose about where the
/// symbol would lie: drawn by hand a little otherwise, or read with the
/// retries a read through an outline has and a search has not, it may
/// yet give the symbol up.
#[derive(Clone, Debug, PartialEq)]
pub struct Lead {
    /// The family the finders were taken for.
    pub family: Family,
    /// Top-left, top-right, bottom-right, bottom-left as the finders
    /// have the symbol, in photo pixels: an outline as
    /// [`detect_candidates`] takes one.
    pub outline: [[f64; 2]; 4],
    /// The centres of the finder patterns, in photo pixels.
    pub finders: Vec<[f64; 2]>,
}

/// What one family's search of one copy of the photo came to.
struct Found {
    /// The symbols found, in the photo's pixels.
    detections: Vec<Detection>,
    /// The copy shows finders that stand as a symbol's, with a timing
    /// pattern between them, on which no symbol was fitted, or which
    /// the search's budget of proposals ran out before: there is more
    /// in the photo than was found, which another copy may show better.
    more: bool,
    /// Where finders lie that a format word spoke for and no symbol
    /// was fitted to.
    leads: Vec<Lead>,
}

/// What every family's search of one copy of the photo came to.
#[derive(Default)]
struct Searched {
    /// The symbols found, in the photo's pixels, in the families' order.
    detections: Vec<Detection>,
    /// The families whose search left [more](Found::more) to find.
    more: Vec<Family>,
    /// How many times the last copy searched was shrunk, where the
    /// search was of shrunk copies ([`search_coarse`]).
    shrunk: Option<u8>,
    /// The finders on which the copy fitted no symbol ([`Lead`]).
    leads: Vec<Lead>,
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

/// Whether a photo point lies within the outline of a symbol found:
/// [`covers`] without the two modules about the symbol.
fn holds(found: &Detection, point: [f64; 2]) -> bool {
    let (w, h) = (found.version.width() as f64, found.version.height() as f64);
    let Some(inverse) = Homography::rect_to_quad(w, h, found.corners).and_then(|m| m.inverse()) else { return false };
    let [x, y] = inverse.apply(point[0], point[1]);
    (0.0..=w).contains(&x) && (0.0..=h).contains(&y)
}

/// Whether a photo point lies on a finder pattern of a symbol found, or
/// on the separator about it: within four and a half modules of the
/// pattern's centre along each of the symbol's axes.
fn on_own_finder(found: &Detection, point: [f64; 2]) -> bool {
    let (w, h) = (found.version.width() as f64, found.version.height() as f64);
    let Some(inverse) = Homography::rect_to_quad(w, h, found.corners).and_then(|m| m.inverse()) else { return false };
    let [x, y] = inverse.apply(point[0], point[1]);
    let near = |[fx, fy]: [f64; 2]| (x - fx).abs() <= 4.5 && (y - fy).abs() <= 4.5;
    near([3.5, 3.5]) || (found.version.family().has_three_finders() && (near([w - 3.5, 3.5]) || near([3.5, h - 3.5])))
}

/// Whether a candidate at a photo point is passed over for a symbol
/// found: it lies on the symbol, or just around it ([`covers`]), and is
/// the symbol's own, as far as can be told. `regioned` is whether a
/// region of the binarised image bears the candidate out, as a whole
/// pattern with measured corners where it has been fitted
/// ([`Capstone::stands_alone`]).
///
/// Everything on a symbol found was once passed over, and most of it is
/// the symbol's: its three finders, and the runs across its data that
/// read 1:1:3:1:1 along a row or two, which every large symbol has a
/// dozen of and which cost a fit apiece to make nothing of. But a symbol
/// may be set within another, as a logo is, and its finders went the
/// same way: a version 1 symbol printed in the middle of a version 7
/// read through an outline and was never found, with nothing said.
/// OpenCV's detectors take from the pool only the three patterns of the
/// symbol they have chosen, and WeChat's tells two detections apart by
/// their corners alone, so both find the two. Here the finder of a
/// symbol set within another is told from the outer symbol's data by
/// what BoofCV asks of a position pattern before anything else: a ring
/// of dark closed on itself about a middle of its own. Such a candidate
/// is kept wherever it lies, unless it is one of the found symbol's own
/// three, and the rest are passed over as before.
fn passed_over(found: &Detection, point: [f64; 2], regioned: bool) -> bool {
    covers(found, point) && !(regioned && !on_own_finder(found, point))
}

/// The most of a symbol's area that an outline set within it may have,
/// for the two to be two symbols and not two readings of one. Two
/// readings of one symbol have areas within a factor of two of each
/// other, as the list of what a search found has it.
const SET_WITHIN_AREA: f64 = 0.5;

/// Whether a symbol found lies where an outline does, so that the
/// outline has nothing more to give: the outline's centre is on the
/// symbol, and neither is set within the other, every corner on it and
/// under [`SET_WITHIN_AREA`] of its area. See [`passed_over`].
fn lies_at(found: &Detection, outline: &[[f64; 2]; 4]) -> bool {
    let (drawn, symbol) = (quad_area(outline), quad_area(&found.corners));
    let within = drawn < SET_WITHIN_AREA * symbol && outline.iter().all(|&corner| covers(found, corner));
    let about = symbol < SET_WITHIN_AREA * drawn && found.corners.iter().all(|&corner| quad_holds(outline, corner));
    covers(found, quad_centre(*outline)) && !within && !about
}

fn quad_area(quad: &[[f64; 2]; 4]) -> f64 {
    (0..4).map(|i| quad[i][0] * quad[(i + 1) % 4][1] - quad[(i + 1) % 4][0] * quad[i][1]).sum::<f64>().abs() / 2.0
}

/// Whether a point lies within a convex quadrilateral, whichever way
/// round its corners run.
fn quad_holds(quad: &[[f64; 2]; 4], point: [f64; 2]) -> bool {
    let side = |i: usize| {
        let (a, b) = (quad[i], quad[(i + 1) % 4]);
        (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
    };
    (0..4).all(|i| side(i) >= 0.0) || (0..4).all(|i| side(i) <= 0.0)
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
    /// The three as the top-left, top-right and bottom-left finders.
    roles: [usize; 3],
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
                if hi > STEEP_SIZE_RATIO * lo {
                    continue;
                }
                // Finders this unlike in size are one symbol's only if
                // it is seen steeply, which their squares must say.
                let steep = hi > SIZE_RATIO * lo;
                // How the three stand as one symbol's finders: by the
                // triangle their centres make in the frame's pixels, and
                // where that says no, by the finders' own squares, which
                // a symbol seen steeply still has to show
                // ([`stands_by_squares`]).
                let p = trio.map(|c| frame.to_rectified(c.center));
                let module = sizes.iter().sum::<f64>() / 3.0;
                let by_pixels = if steep { None } else { stands_by_pixels(&trio, &p, frame, module) };
                let Some(TripleShape { tl, distortion, spacing, fourth }) = by_pixels.or_else(|| stands_by_squares(&trio, &p, frame)) else {
                    continue;
                };
                // The other two are told apart by the handedness of the
                // photo's axes.
                let others: Vec<usize> = (0..3).filter(|&v| v != tl).collect();
                let (mut tr, mut bl) = (others[0], others[1]);
                let t = trio[tl].center;
                let cross = |r: [f64; 2], l: [f64; 2]| (r[0] - t[0]) * (l[1] - t[1]) - (r[1] - t[1]) * (l[0] - t[0]);
                if cross(trio[tr].center, trio[bl].center) < 0.0 {
                    std::mem::swap(&mut tr, &mut bl);
                }
                let (r, l) = (trio[tr].center, trio[bl].center);
                // No finder that could be another symbol's may lie within
                // the symbol proposed: three finders of a sheet of symbols
                // stand as readily at the corners of a square as three of
                // one symbol do, and OpenCV's multi-symbol detector
                // likewise rejects a triple with a fourth finder inside it.
                let (u_r, v_r) = ([p[tr][0] - p[tl][0], p[tr][1] - p[tl][1]], [p[bl][0] - p[tl][0], p[bl][1] - p[tl][1]]);
                let across = (spacing[tr] + spacing[bl]) / 2.0;
                if encloses_a_finder(frame, capstones, [i, j, k], p[tl], u_r, v_r, across, (lo, hi)) {
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
                // The reach, as a share of the finder spacing along each
                // side.
                let (m, n) = (PROPOSAL_REACH / spacing[tr], PROPOSAL_REACH / spacing[bl]);
                let parallelogram = [
                    [t[0] - m * u[0] - n * v[0], t[1] - m * u[1] - n * v[1]],
                    [t[0] + u[0] + m * u[0] - n * v[0], t[1] + u[1] + m * u[1] - n * v[1]],
                    [t[0] + u[0] + v[0] + m * u[0] + n * v[0], t[1] + u[1] + v[1] + m * u[1] + n * v[1]],
                    [t[0] + v[0] + n * v[0] - m * u[0], t[1] + v[1] + n * v[1] - m * u[1]],
                ];
                // Where the finders' own squares say how the symbol
                // closes, the outline takes in what both say, each corner
                // as far out as either puts it: a loose outline costs the
                // fit nothing, and one that cuts a corner off costs it
                // the symbol. The squares of a symbol of two pixels a
                // module point where its edges run to within a few
                // modules, short as often as long. A symbol seen steeply
                // takes the squares' outline alone: the parallelogram
                // puts its far corner as far again from the middle as
                // it lies, and the fit, made square to an outline so
                // drawn, read a version 5 symbol as a version 8.
                let middle = [(r[0] + l[0]) / 2.0, (r[1] + l[1]) / 2.0];
                let outline = fourth
                    .and_then(|f| outline_through([t, r, f, l], spacing[tr], spacing[bl]))
                    .map(|closed| if steep { closed } else { std::array::from_fn(|c| if distance(closed[c], middle) > distance(parallelogram[c], middle) { closed[c] } else { parallelogram[c] }) })
                    .filter(convex)
                    .unwrap_or(parallelogram);
                let badness = distortion + (hi / lo - 1.0) + faintness + TIMING_PENALTY * unread;
                out.push(TripleProposal { badness, outline, members: [i, j, k], timed, roles: [[i, j, k][tl], [i, j, k][tr], [i, j, k][bl]] });
            }
        }
    }
    out
}

/// How three capstones stand as the finders of one symbol.
struct TripleShape {
    /// Which of the three is the top-left finder.
    tl: usize,
    /// How far the three depart from the shape a symbol's finders make:
    /// nothing for two legs alike and square to each other.
    distortion: f64,
    /// Modules from the top-left finder's centre to each capstone's, by
    /// its place in the three: finders along a side lie n - 7 modules
    /// apart in a symbol of n modules. Nothing for the top-left one.
    spacing: [f64; 3],
    /// Where the finder of the fourth corner would be centred, had the
    /// symbol one, in photo pixels, when the finders' own squares say:
    /// `None` leaves it to the parallelogram the three centres make.
    fourth: Option<[f64; 2]>,
}

/// How three capstones stand as one symbol's finders by the triangle
/// their centres `p` make in the frame's pixels, their modules `module`
/// pixels on average. The top-left finder is the vertex opposite the
/// longest side, and the legs from it must be alike in length and near
/// square to each other, within what perspective does to them; how far
/// they are from a right isosceles pair ranks the proposal, as ZXing
/// ranks by it. The finders along each leg must face each other as a
/// symbol's do ([`finders_face`]).
fn stands_by_pixels(trio: &[&Capstone; 3], p: &[[f64; 2]; 3], frame: &Frame, module: f64) -> Option<TripleShape> {
    // Squared side lengths in rectified pixels, each with the vertex it
    // lies opposite.
    let d2 = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
    let mut sides = [(d2(p[1], p[2]), 0usize), (d2(p[0], p[2]), 1), (d2(p[0], p[1]), 2)];
    sides.sort_by(|a, b| a.0.total_cmp(&b.0));
    let [(a, _), (b, _), (c, tl)] = sides;
    if a <= 0.0 || c <= 0.0 || (b / a).sqrt() > MAX_LEG_RATIO {
        return None;
    }
    let cos = (a + b - c) / (2.0 * (a * b).sqrt());
    if cos.abs() > MAX_CORNER_TILT.to_radians().sin() {
        return None;
    }
    let distortion = ((c - 2.0 * b).abs() + (c - 2.0 * a).abs()) / c;
    let spacing = (a.sqrt() + b.sqrt()) / 2.0 / module;
    if !(12.0..=176.0).contains(&spacing) {
        return None;
    }
    let faces = (0..3).filter(|&v| v != tl).all(|v| finders_face(trio[tl], trio[v], frame));
    faces.then_some(TripleShape { tl, distortion, spacing: [spacing; 3], fourth: None })
}

/// How three capstones stand as one symbol's finders by their own
/// squares, for three whose triangle says they are none
/// ([`stands_by_pixels`]). A symbol seen from far to one side has one leg
/// a third the length of the other in the photo's pixels, and its
/// finders are bars three times as tall as they are wide; the triangle
/// is nothing like a right isosceles one, and the legs are still each as
/// many of the finders' own modules long as the other. BoofCV asks
/// nothing of the triangle at all: it joins two position patterns when
/// the line between their centres leaves each by the middle of a side,
/// the two sides alike, and takes for the corner of a symbol the pattern
/// so joined to two others by sides that adjoin.
///
/// So here: the top-left finder is the one that faces both the others
/// ([`side_crossed`]) through two sides of its square that meet at a
/// corner, and each leg is measured in the modules the two finders at
/// its ends show along it, seven across each square. The legs so
/// measured must be alike as the pixels' are asked to be, and the
/// corner's lean is not asked: sides that adjoin are the corner. Every
/// capstone must have its corners measured, there being no square to go
/// by otherwise.
fn stands_by_squares(trio: &[&Capstone; 3], p: &[[f64; 2]; 3], frame: &Frame) -> Option<TripleShape> {
    if trio.iter().any(|c| !c.measured) {
        return None;
    }
    (0..3).find_map(|tl| {
        let mut spacing = [0.0; 3];
        let mut left_by = [0usize; 2];
        // The way each of the other two finders runs across its leg: the
        // side the leg crosses and the side opposite, taken together.
        let mut across = [[0.0; 2]; 2];
        let others = [(tl + 1) % 3, (tl + 2) % 3];
        for (n, &other) in others.iter().enumerate() {
            let here = side_crossed(trio[tl], p[tl], p[other], frame)?;
            let there = side_crossed(trio[other], p[other], p[tl], frame)?;
            if !sides_run_alike(here.side, there.side) {
                return None;
            }
            // Three and a half modules of each finder lie between its
            // centre and the side the line leaves it by.
            let module = (here.reach + there.reach) / 7.0;
            spacing[other] = distance(p[tl], p[other]) / module.max(1e-9);
            left_by[n] = here.index;
            let quad = trio[other].corners.map(|c| frame.to_rectified(c));
            let (k, sub) = (there.index, |a: [f64; 2], b: [f64; 2]| [a[0] - b[0], a[1] - b[1]]);
            let (near, far) = (sub(quad[(k + 1) % 4], quad[k]), sub(quad[(k + 2) % 4], quad[(k + 3) % 4]));
            across[n] = [near[0] + far[0], near[1] + far[1]];
        }
        if (left_by[0] + 4 - left_by[1]) % 2 == 0 {
            return None;
        }
        let (a, b) = (spacing[(tl + 1) % 3].min(spacing[(tl + 2) % 3]), spacing[(tl + 1) % 3].max(spacing[(tl + 2) % 3]));
        if b > MAX_LEG_RATIO * a || !(12.0..=176.0).contains(&((a + b) / 2.0)) {
            return None;
        }
        // The triangle's own measure, of legs square to each other.
        let distortion = 2.0 * (b * b - a * a) / (a * a + b * b);
        // The fourth corner lies where the edges the other two finders
        // stand on meet: on the line through each one's centre that runs
        // as its own sides across the leg do. BoofCV closes a symbol's
        // bounds so, by the outer sides of its right and lower position
        // patterns. Seen steeply the near edge of a symbol is longer
        // than the far one, and the parallelogram of the three centres
        // puts the corner inside the symbol; where the lines meet
        // nowhere near it they are not to be believed, and it stands.
        let (q, r) = (p[others[0]], p[others[1]]);
        let parallel = [q[0] + r[0] - p[tl][0], q[1] + r[1] - p[tl][1]];
        let det = across[0][0] * across[1][1] - across[0][1] * across[1][0];
        let fourth = (det.abs() > 1e-12)
            .then(|| {
                let along = ((r[0] - q[0]) * across[1][1] - (r[1] - q[1]) * across[1][0]) / det;
                [q[0] + along * across[0][0], q[1] + along * across[0][1]]
            })
            .filter(|&met| distance(met, parallel) <= 0.5 * distance(p[tl], q).max(distance(p[tl], r)))
            .map(|[x, y]| frame.to_image.apply(x, y));
        Some(TripleShape { tl, distortion, spacing, fourth })
    })
}

/// The outline about a symbol whose finders are centred at the first,
/// second and fourth of `centres`, `across` modules apart along the top
/// and `down` along the left, and whose fourth corner's finder, had it
/// one, would be centred at the third: [`PROPOSAL_REACH`] modules beyond
/// the centres every way, through the perspective the four centres set.
/// `None` when they set none, or one that folds the outline over.
fn outline_through(centres: [[f64; 2]; 4], across: f64, down: f64) -> Option<[[f64; 2]; 4]> {
    let map = Homography::from_points([[0.0, 0.0], [across, 0.0], [across, down], [0.0, down]], centres)?;
    let reach = PROPOSAL_REACH;
    let outline = [[-reach, -reach], [across + reach, -reach], [across + reach, down + reach], [-reach, down + reach]].map(|[x, y]| map.apply(x, y));
    // Wound the way the centres are, and not folded over.
    let wound = (centres[1][0] - centres[0][0]) * (centres[3][1] - centres[0][1]) - (centres[1][1] - centres[0][1]) * (centres[3][0] - centres[0][0]);
    let area: f64 = (0..4).map(|i| outline[i][0] * outline[(i + 1) % 4][1] - outline[(i + 1) % 4][0] * outline[i][1]).sum();
    (convex(&outline) && area * wound > 0.0).then_some(outline)
}

/// Whether a quadrilateral turns the same way at every corner.
fn convex(quad: &[[f64; 2]; 4]) -> bool {
    let turn = |i: usize| {
        let (a, b, c) = (quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]);
        (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
    };
    let first = turn(0);
    first.is_finite() && first != 0.0 && (1..4).all(|i| turn(i) * first > 0.0)
}

/// Where the line from a capstone's centre towards another point leaves
/// the capstone's square.
struct Crossing {
    /// The side crossed, counted from the corner it starts at.
    index: usize,
    /// The side, from that corner to the next, in the frame's pixels.
    side: [f64; 2],
    /// How far from the centre the line crosses it, in the frame's pixels.
    reach: f64,
}

/// The side of a capstone's square that the line from its centre `from`
/// towards `to` leaves it by, when it crosses within [`FACING_MIDPOINT`]
/// of that side's middle and short of `to`, in the frame's pixels.
fn side_crossed(c: &Capstone, from: [f64; 2], to: [f64; 2], frame: &Frame) -> Option<Crossing> {
    let quad = c.corners.map(|p| frame.to_rectified(p));
    let line = [to[0] - from[0], to[1] - from[1]];
    (0..4).find_map(|i| {
        let (p, q) = (quad[i], quad[(i + 1) % 4]);
        let side = [q[0] - p[0], q[1] - p[1]];
        // from + s·line = p + t·side, with both on their segments.
        let det = line[0] * side[1] - line[1] * side[0];
        if det.abs() < 1e-12 {
            return None;
        }
        let (dx, dy) = (p[0] - from[0], p[1] - from[1]);
        let s = (dx * side[1] - dy * side[0]) / det;
        let t = (dx * line[1] - dy * line[0]) / det;
        ((0.0..=1.0).contains(&s) && (0.0..=1.0).contains(&t) && (t - 0.5).abs() <= FACING_MIDPOINT)
            .then(|| Crossing { index: i, side, reach: s * line[0].hypot(line[1]) })
    })
}

/// Whether two sides lie within [`FACING_PARALLEL`] of parallel.
fn sides_run_alike(u: [f64; 2], v: [f64; 2]) -> bool {
    let lengths = u[0].hypot(u[1]) * v[0].hypot(v[1]);
    let cos = (u[0] * v[0] + u[1] * v[1]) / lengths.max(1e-12);
    cos.abs() >= FACING_PARALLEL.to_radians().cos()
}

/// Whether two capstones face each other as the finders along one side
/// of a symbol do: the line between their centres crosses a side of each
/// within [`FACING_MIDPOINT`] of that side's middle, and the two sides
/// crossed lie within [`FACING_PARALLEL`] of parallel to each other, in
/// the frame's pixels. Nothing is asked of the angle the line makes with
/// them, which a symbol seen steeply shears. A capstone whose corners
/// were not measured faces anything: its square was put square to the
/// frame and says nothing of its turn.
fn finders_face(a: &Capstone, b: &Capstone, frame: &Frame) -> bool {
    if !a.measured || !b.measured {
        return true;
    }
    let (pa, pb) = (frame.to_rectified(a.center), frame.to_rectified(b.center));
    let (Some(u), Some(v)) = (side_crossed(a, pa, pb, frame), side_crossed(b, pb, pa, frame)) else { return false };
    sides_run_alike(u.side, v.side)
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
    let corner_of = |turns: u8, mirrored: bool| lone_corner(turns, mirrored, w, h);
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
    boxes.into_iter().map(|(badness, corner, turns, version)| (badness, lone_box(&patch, corner, turns, version))).collect()
}

/// The outline corner a lone finder sits in under a placement, in a
/// symbol `w` by `h` modules.
fn lone_corner(turns: u8, mirrored: bool, w: f64, h: f64) -> usize {
    let [u, v] = to_unit(turns, mirrored, w, h, [3.5, 3.5]);
    match (u < 0.5, v < 0.5) {
        (true, true) => 0,
        (false, true) => 1,
        (false, false) => 2,
        (true, false) => 3,
    }
}

/// The box about a symbol of `version` whose one finder, mapped from its
/// own seven modules square by `patch`, sits in the outline's `corner`,
/// the symbol turned a quarter when `turns` is odd.
fn lone_box(patch: &Homography, corner: usize, turns: u8, version: SymbolVersion) -> [[f64; 2]; 4] {
    // In the finder's own module coordinates the finder is the square
    // [0, 7]²; the symbol is `sw` by `sh` modules, with a quiet zone of
    // two and a module of slack around it. Turned a quarter, its width
    // runs up the finder's own y-axis.
    let (sw, sh) = if turns % 2 == 1 {
        (version.height() as f64, version.width() as f64)
    } else {
        (version.width() as f64, version.height() as f64)
    };
    let reach = 3.0;
    let (x0, x1) = if corner == 0 || corner == 3 { (-reach, sw + reach) } else { (7.0 - sw - reach, 7.0 + reach) };
    let (y0, y1) = if corner == 0 || corner == 1 { (-reach, sh + reach) } else { (7.0 - sh - reach, 7.0 + reach) };
    [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(|[x, y]| patch.apply(x, y))
}

/// An outline a lone finder proposes to a search of the whole photo, and
/// whether anything beside the finder speaks for a symbol there.
struct LoneProposal {
    badness: f64,
    outline: [[f64; 2]; 4],
    spoken: bool,
}

/// [`single_finder_proposals`] for a search of the whole photo: the same
/// boxes about one capstone, each with what the photo shows beside the
/// capstone of a symbol lying so, read through the capstone's own square
/// before anything is fitted ([`MAX_UNSPOKEN_PROPOSALS`]).
///
/// BoofCV's Micro QR reader has nothing but the one position pattern to
/// go by, and goes by it all the way: under each of the eight ways the
/// pattern may stand as a symbol's it reads the format word and then the
/// whole symbol through the pattern's own square, and keeps the way that
/// decodes. A symbol of seventeen modules lies within a finder's width
/// and a half of its finder, and a clean one reads so; that is asked
/// here first, of every placement the word reads under, and a box whose
/// symbol decodes is the word's box whatever else read cleaner. Short of
/// that, a format word that reads outright speaks for its box, as it
/// vouches for a fit, and so does a timing pattern setting out from the
/// finder along each edge the placement gives one: three or four modules
/// of it, as far as the smallest version has them, which the square
/// reaches as surely as it does the word.
///
/// A box nothing speaks for is ranked by how nearly something did, the
/// bits its word was off by and the modules of timing pattern that read
/// wrong, a box with no word standing behind any with one. The boxes
/// with no word stood in the order of the corners, the first corner of
/// every capstone before the second of any.
fn lone_finder_proposals(image: &Luma, capstone: &Capstone, family: Family, threshold: Option<f32>, params: &SampleParams) -> Vec<LoneProposal> {
    let Some(patch) = Homography::rect_to_quad(7.0, 7.0, capstone.corners) else { return Vec::new() };
    let reference = FunctionMap::new(reference_version(family));
    let (w, h) = (reference.width as f64, reference.height as f64);
    let corner_of = |turns: u8, mirrored: bool| lone_corner(turns, mirrored, w, h);
    let stubs = timing_stubs(&reference);
    let sites = sites(&reference);
    let site = sites.iter().find(|s| s.kind == AnchorKind::Finder);
    let levels = capstone_levels(image, capstone, threshold, params.band).filter(|_| !capstone.line_only);
    // Modules of timing pattern beside the finder that read wrong under
    // a placement: all of them where they cannot be read.
    let misread = |turns: u8, mirrored: bool| {
        let read = site.zip(levels).and_then(|(site, (threshold, band_abs))| {
            let map = finder_map(capstone, site, turns, mirrored, &reference)?;
            let luma = sample_positions(image, &map, &stubs, INNER_FRACTION, POINTS);
            Some(
                stubs
                    .iter()
                    .zip(luma)
                    .filter(|(at, value)| {
                        let dark = value.filter(|v| (v - threshold).abs() >= band_abs).map(|v| (v < threshold) != capstone.inverted);
                        dark.is_none() || dark != reference.expected(at.x as usize, at.y as usize)
                    })
                    .count(),
            )
        });
        read.unwrap_or(stubs.len())
    };
    let placements = format_placements(image, capstone, family, threshold, params.band);
    let decoded = placements.iter().find(|placement| decodes_through_finder(image, capstone, placement, params));
    let outright = |placement: &Placement| placement.mismatches == 0 && placement.compared as usize == reference.format[0].bits.len();
    let mut boxes: Vec<(f64, bool, usize, u8, SymbolVersion)> = decoded
        .or(placements.first())
        .map(|placement| {
            let wrong = misread(placement.turns, placement.mirrored);
            let spoken = decoded.is_some() || outright(placement) || wrong == 0;
            let doubt = f64::from(placement.mismatches) + wrong as f64;
            (doubt - capstone.contrast, spoken, corner_of(placement.turns, placement.mirrored), placement.turns, placement.version)
        })
        .into_iter()
        .collect();
    let largest = family.versions().into_iter().max_by_key(|v| v.width() * v.height()).expect("a family has versions");
    // A square symbol stands the same way whichever quarter turn it has;
    // an rMQR on end is as long one way as it is wide the other.
    let turn_options: &[u8] = if family == Family::Rmqr { &[0, 1] } else { &[0] };
    for &lying in turn_options {
        for corner in 0..4 {
            // The placements that put the finder in this corner with the
            // symbol lying this way, plain and mirrored.
            let wrong = [false, true]
                .into_iter()
                .flat_map(|mirrored| (0..4u8).map(move |turns| (turns, mirrored)))
                .filter(|&(turns, mirrored)| corner_of(turns, mirrored) == corner && (family != Family::Rmqr || turns % 2 == lying))
                .map(|(turns, mirrored)| misread(turns, mirrored))
                .min()
                .unwrap_or(stubs.len());
            boxes.push((UNNAMED_DOUBT + wrong as f64 - capstone.contrast, wrong == 0, corner, lying, largest));
        }
    }
    boxes
        .into_iter()
        .map(|(badness, spoken, corner, turns, version)| LoneProposal { badness, outline: lone_box(&patch, corner, turns, version), spoken })
        .collect()
}

/// The timing modules nearest a lone finder, in the module coordinates
/// of the family's reference version `func`: [`MICRO_STUB`] along the
/// top and down the side of a Micro QR symbol, [`RMQR_STUB`] along the
/// top of an rMQR, whose left side below the finder is the corner
/// pattern's in some heights and nothing in the lowest.
fn timing_stubs(func: &FunctionMap) -> Vec<Pos> {
    let family = func.version.family();
    let reach = if family == Family::Rmqr { RMQR_STUB } else { MICRO_STUB };
    let along = (8..8 + reach).map(|x| Pos::new(x, 0));
    let down = (8..8 + reach).map(|y| Pos::new(0, y)).filter(|_| family != Family::Rmqr);
    along
        .chain(down)
        .filter(|at| (at.x as usize) < func.width && (at.y as usize) < func.height && func.role(at.x as usize, at.y as usize) == Role::Timing)
        .collect()
}

/// Whether a lone finder's symbol decodes when read through the finder's
/// own square alone, standing as `placement` has it: sampled once, as
/// [`grid_as_found`] samples a detection, and decoded under every
/// format, every block solving with confirmation and the stream parsing
/// ([`decodes_as_found`]).
fn decodes_through_finder(image: &Luma, capstone: &Capstone, placement: &Placement, params: &SampleParams) -> bool {
    let func = Arc::new(FunctionMap::new(placement.version));
    let sites = sites(&func);
    let Some(site) = sites.iter().find(|s| s.kind == AnchorKind::Finder) else { return false };
    let Some(map) = finder_map(capstone, site, placement.turns, placement.mirrored, &func) else { return false };
    let sampled = sample_modules_as(image, &map, func.width, func.height, &SampleParams { kernel: Kernel::Area, enhance: Enhance::NONE, ..*params });
    let grid = classify(&sampled, params).to_grid(func.width, func.height);
    let grid = if capstone.inverted { grid.inverted() } else { grid };
    let opts = DecodeOptions { reliability: false, ..DecodeOptions::default() };
    brute_force(&grid, &func, None, &opts).first().is_some_and(|attempt| attempt.report.score.all_verified && attempt.report.score.stream_valid)
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
pub(crate) fn clockwise(quad: [[f64; 2]; 4]) -> [[f64; 2]; 4] {
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
        self.side(p).abs()
    }

    /// How far `p` lies from the line, signed by the side it lies on.
    fn side(&self, p: [f64; 2]) -> f64 {
        (p[0] - self.point[0]) * self.dir[1] - (p[1] - self.point[1]) * self.dir[0]
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
/// them; whether it was a bend; whether paper lies beyond the fit near
/// the corner ([`BEYOND_STRIP`]); and the crossings the fit kept, each as
/// the point of the symbol's edge it lies on, in module coordinates, with
/// where the paper turned to ink there in the photo, set on the fit: one
/// crossing is read to a fifth of a module, and the fit through thirty of
/// them far better.
struct Traced {
    line: Line,
    bent: bool,
    clear: bool,
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
///
/// A profile sees the edge only within about half a module of where the
/// map predicts it. Where the prediction runs from a corner placed right
/// to one placed a module inside the symbol, the profiles near the first
/// find the edge and those near the second the boundary between the
/// border row, where its module is light, and the row within, where that
/// one is dark: both turn from light to dark, a line fits the lot within
/// the tolerance, slanting from the edge to a row inside it, and the
/// corner its crossing with the other edge gives is a module short. Such
/// a fit has the dark modules of the symbol's border beyond it, and the
/// trace says whether its fit is clear of them ([`Traced::clear`]). It
/// is not refused for them: on a label round a bottle a true edge bulges
/// past the line fitted to the rest of it, over a few modules, and the
/// line still crosses the other edge at the corner.
///
/// Where the prediction runs from a corner placed a module out, or along
/// an edge that bows further than half a module from it, the profiles
/// there find nothing: paper on either side, a border module read where
/// the paper is, or quiet zone off the photo. Those are taken again about
/// where the crossings found along the rest of the edge put it: further
/// in, and, with `outward`, as the traces that place a corner ask,
/// further out as well. The fourth corner of a version 2 symbol on paper
/// curled across it was put
/// a module out, past the photo's edge, and the bottom-right quarter read
/// sixteen modules wrong; a version 5 symbol whose right edge bowed in
/// toward its foot read nineteen wrong there. Taken again, the profiles
/// found both edges up to the corners, and both symbols read with none
/// wrong.
fn trace_edge(
    image: &Luma,
    map: &impl Project,
    edge: &EdgeFrame,
    sign: f64,
    threshold: f32,
    contrast: f32,
    outward: bool,
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
    let t_of = |j: usize| -PROFILE_REACH + j as f64 * PROFILE_STEP;
    // The border module, half a module in, and where the walk to it
    // starts.
    let border_j = ((0.5 + PROFILE_REACH) / PROFILE_STEP).round() as usize;
    let start_j = ((-QUIET_FROM + PROFILE_REACH) / PROFILE_STEP).round() as usize;
    // One profile across the edge `s` modules along it, about where the
    // edge is expected, `shift` modules in from where the map has it:
    // whether paper lies outside it, `None` where none of that is on the
    // photo, and where the paper turns to ink, in modules in from where
    // the map has the edge. Moved, it reads paper and ink as the symbol's
    // own threshold has them, and not against what lies beside it.
    let probe = |s: f64, shift: f64| -> (Option<bool>, Option<f64>) {
        let profile: Vec<Option<f32>> = (0..=steps).map(|j| sample(s, shift + t_of(j))).collect();
        let outside: Vec<f32> = (0..=steps)
            .filter(|&j| (-QUIET_TO..=-QUIET_FROM).contains(&t_of(j)))
            .filter_map(|j| profile[j])
            .collect();
        if outside.is_empty() {
            return (None, None);
        }
        let paper = outside.iter().sum::<f32>() / outside.len() as f32;
        if !is_quiet(paper) {
            return (Some(false), None);
        }
        let Some(border) = profile.get(border_j).copied().flatten() else { return (Some(true), None) };
        if (sign as f32) * (paper - border) < min_contrast || (shift != 0.0 && is_quiet(border)) {
            // A light border module: no edge to see here.
            return (Some(true), None);
        }
        let level = if shift == 0.0 { (paper + border) / 2.0 } else { threshold };
        // Walk inward from the quiet zone to the first crossing of the level.
        for j in start_j..border_j {
            let (Some(a), Some(b)) = (profile[j], profile[j + 1]) else { continue };
            let (fa, fb) = ((sign as f32) * (level - a), (sign as f32) * (level - b));
            if fa <= 0.0 && fb > 0.0 {
                let frac = if (fb - fa).abs() > 1e-6 { -fa / (fb - fa) } else { 0.5 };
                return (Some(true), Some(shift + t_of(j) + frac as f64 * PROFILE_STEP));
            }
        }
        (Some(true), None)
    };
    let mut probed: Vec<(Option<bool>, Option<f64>)> = positions.iter().map(|&k| probe(k as f64 + 0.5, 0.0)).collect();
    // A profile sees the edge only within half a module or so of where the
    // map has it, and an edge that bows further, or a map run on from a
    // corner a module out, leaves the profiles there seeing paper alone,
    // or a border module that is light where the edge has drawn in, or
    // that is not on the photo. They are taken again about where the
    // crossings the others found run, fitted across the edge
    // ([`edge_offsets`]), from [`RECENTRE_LEAST`] modules off where they
    // were taken to [`RECENTRE_MOST`]. A profile so moved has only the
    // others' word for where it is, and it reads ink as the symbol's own
    // threshold has it, not as what is darker than what lies outside it.
    // Moved in, it looks past where the first saw paper; moved out, it
    // reaches past the symbol into whatever lies beyond it, and only the
    // traces that place a corner move one out, whose crossing must stand
    // clear of the symbol and be seen near the corner to place it
    // ([`traced_corner`]). On a photo with a lamp's reflection drawn as a
    // white line a module out from a symbol on dark paper, the profiles
    // beside its finders that traced the points along its edges, moved
    // out to where the crossings by its corners had the edge, took the
    // line for the paper and the paper inside it for ink, and an edge a
    // module out cost the symbol its placement. Moved in, the profiles
    // along the top of a label round a bottle, which bowed in by most of
    // a module between the symbol's corners, found the bow, and the
    // points along it held the map there.
    let seen: Vec<[f64; 2]> = positions.iter().zip(&probed).filter_map(|(&k, (_, t))| t.map(|t| [k as f64 + 0.5, t])).collect();
    if let Some(run) = edge_offsets(&seen) {
        for (&k, was) in positions.iter().zip(probed.iter_mut()) {
            let s = k as f64 + 0.5;
            let shift = (run[0] + run[1] * s + run[2] * s * s).clamp(if outward { -RECENTRE_MOST } else { 0.0 }, RECENTRE_MOST);
            if was.1.is_some() || shift.abs() < RECENTRE_LEAST {
                continue;
            }
            let again = probe(s, shift);
            if again.0 == Some(true) {
                *was = again;
            }
        }
    }
    let quiet = probed.iter().filter(|(quiet, _)| *quiet == Some(true)).count();
    let mut crossings: Vec<[f64; 2]> = Vec::new();
    // The point of the symbol's edge each crossing lies on, in module
    // coordinates: where the map should put the crossing.
    let mut sites: Vec<[f64; 2]> = Vec::new();
    // The profiles a crossing was found in.
    let mut crossed: Vec<usize> = Vec::new();
    for (&k, &(_, t)) in positions.iter().zip(&probed) {
        let Some(t) = t else { continue };
        let s = k as f64 + 0.5;
        let [x, y] = [
            edge.corner[0] + edge.along[0] * s + edge.inward[0] * t,
            edge.corner[1] + edge.along[1] * s + edge.inward[1] * t,
        ];
        crossings.push(map.apply(x, y));
        sites.push([edge.corner[0] + edge.along[0] * s, edge.corner[1] + edge.along[1] * s]);
        crossed.push(k);
    }
    if (quiet as f64) < QUIET_SHARE * probes as f64 || crossings.len() < MIN_CROSSINGS {
        return None;
    }
    let module = module_size_at(map, [edge.corner[0] + edge.along[0] * 2.0, edge.corner[1] + edge.along[1] * 2.0]);
    let limit = MAX_EDGE_RMS * module;
    // The crossings within reach of what was fitted through them, each
    // set on the fit, as control points; the rest are stray print.
    let kept = |near: &dyn Fn([f64; 2]) -> f64, onto: &dyn Fn([f64; 2]) -> [f64; 2]| -> Vec<ControlPoint> {
        crossings
            .iter()
            .zip(&sites)
            .filter(|(p, _)| near(**p) <= 2.0 * limit)
            .map(|(p, at)| ControlPoint { at: *at, px: onto(*p) })
            .collect()
    };
    // Whether paper lies beyond a fit near the corner. `across` is how
    // far a point of the photo lies from the fit, signed by its side.
    let ink_drop = (BEYOND_DROP * contrast).max(NOISE_RANGE);
    let near = (edge.length / 3.0).max(BEYOND_NEAR);
    let clear = |across: &dyn Fn([f64; 2]) -> f64| -> bool {
        let at = |s: f64, t: f64| {
            map.apply(edge.corner[0] + edge.along[0] * s + edge.inward[0] * t, edge.corner[1] + edge.along[1] * s + edge.inward[1] * t)
        };
        let (mut read, mut inked) = (0usize, 0usize);
        for &k in positions.iter().filter(|k| !crossed.contains(k)) {
            let s = k as f64 + 0.5;
            if s > near {
                continue;
            }
            // Where the fit crosses this profile, from how far two points
            // of the profile lie from it.
            let (outside, inside) = (across(at(s, -1.0)), across(at(s, 1.0)));
            if (inside - outside).abs() < 1e-9 {
                continue;
            }
            let t = -1.0 - 2.0 * outside / (inside - outside);
            let read_at = |out: f64| sample(s, t - out);
            let strip: Option<Vec<f32>> =
                (0..=5).map(|j| read_at(BEYOND_STRIP[0] + (BEYOND_STRIP[1] - BEYOND_STRIP[0]) * j as f64 / 5.0)).collect();
            let Some(strip) = strip else { continue };
            // The paper further out, the lightest of it; off the photo,
            // the threshold says what is ink.
            let further: Option<Vec<f32>> =
                (0..=2).map(|j| read_at(BEYOND_PAPER[0] + (BEYOND_PAPER[1] - BEYOND_PAPER[0]) * j as f64 / 2.0)).collect();
            let paper = further.map(|f| f.into_iter().fold(f32::NEG_INFINITY, |m, v| m.max(sign as f32 * v)));
            let ink = |v: f32| match paper {
                Some(paper) => paper - sign as f32 * v >= ink_drop,
                None => !is_quiet(v),
            };
            read += 1;
            inked += usize::from(strip.into_iter().any(ink));
        }
        inked as f64 <= BEYOND_INKED * read as f64
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
        let crossings = kept(&|p| curve.residual(p).abs(), &|p| curve.project(p));
        return Some(Traced { line: tangent, bent: true, clear: clear(&|p| curve.residual(p)), crossings });
    }
    if rms_all <= limit {
        let crossings = kept(&|p| straight.distance(p), &|p| straight.project(p));
        return Some(Traced { line: straight, bent: false, clear: clear(&|p| straight.side(p)), crossings });
    }
    // Otherwise the worst crossings are stray print, and are left out.
    robust_line(&crossings).filter(|(_, rms, kept)| *kept >= MIN_CROSSINGS && *rms <= limit).map(|(line, _, _)| Traced {
        line,
        bent: false,
        clear: clear(&|p| line.side(p)),
        crossings: kept(&|p| line.distance(p), &|p| line.project(p)),
    })
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

    /// The point of the bend straight across it from `p`.
    fn project(&self, p: [f64; 2]) -> [f64; 2] {
        let r = self.residual(p);
        [p[0] - r * self.normal[0], p[1] - r * self.normal[1]]
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

/// Where the crossings of an edge found so far say it runs, as `[s, t]`
/// pairs, `s` modules along the edge and `t` in from where the map has
/// it: `t` as a quadratic in `s` through six crossings or more, as a line
/// through fewer, with the quarter that fit worst dropped and the rest
/// fitted again, as the coefficients of `1, s, s²`. `None` for fewer than
/// [`MIN_CROSSINGS`] crossings.
fn edge_offsets(seen: &[[f64; 2]]) -> Option<[f64; 3]> {
    if seen.len() < MIN_CROSSINGS {
        return None;
    }
    let terms = if seen.len() >= MIN_CURVE_CROSSINGS { 3 } else { 2 };
    let fit = |points: &[[f64; 2]]| -> Option<[f64; 3]> {
        let mut a = [[0.0f64; 3]; 3];
        let mut b = [0.0f64; 3];
        for &[s, t] in points {
            let basis = [1.0, s, s * s];
            for i in 0..3 {
                b[i] += basis[i] * t;
                for j in 0..3 {
                    a[i][j] += basis[i] * basis[j];
                }
            }
        }
        // A line: the square term held at nought.
        if terms == 2 {
            a[2] = [0.0, 0.0, 1.0];
            a[0][2] = 0.0;
            a[1][2] = 0.0;
            b[2] = 0.0;
        }
        solve3(a, b)
    };
    let residual = |c: [f64; 3], [s, t]: [f64; 2]| t - (c[0] + c[1] * s + c[2] * s * s);
    let first = fit(seen)?;
    let mut ranked = seen.to_vec();
    ranked.sort_by(|p, q| residual(first, *p).abs().total_cmp(&residual(first, *q).abs()));
    ranked.truncate((seen.len() - seen.len() / 4).max(MIN_CROSSINGS));
    fit(&ranked)
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
/// of the fixed modules there, where there are any, expecting their
/// boundaries `bloom` modules into the dark ([`Detection::bloom`]), and
/// how crisply the data modules read dark or light rather than grey,
/// which a grid off by part of a module cannot manage.
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
    fn new(image: &'a Luma, func: &FunctionMap, corners: [[f64; 2]; 4], i: usize, threshold: Option<f32>, bloom: f64) -> Option<Self> {
        let (w, h) = (func.width, func.height);
        let (qx0, qx1) = if i == 0 || i == 3 { (0, w.div_ceil(2)) } else { (w / 2, w) };
        let (qy0, qy1) = if i == 0 || i == 1 { (0, h.div_ceil(2)) } else { (h / 2, h) };
        let template = Template::window(func, qx0 as i64, qy0 as i64, qx1 - qx0, qy1 - qy0);
        let quarter = [[qx0, qy0], [qx1, qy0], [qx1, qy1], [qx0, qy1]].map(|[x, y]| [x as f64, y as f64]);
        let map = Homography::rect_to_quad(w as f64, h as f64, corners)?;
        let base = quarter.map(|[x, y]| map.apply(x, y));
        let fixed = (template.fixed() >= 4).then(|| Scorer::new(image, &template, base, threshold).with_along(6).with_bloom(bloom));
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
/// How far, in modules, the anchors inside the symbol must bend the map
/// from the plane through its corners, somewhere along a straight edge,
/// for that edge to be given points of its own.
const EDGE_BENT_BY: f64 = 0.25;
/// The least depth of a module across a straight edge, in photo pixels,
/// for the edge to be given points: a crossing is read to a fraction of
/// a pixel, which of a module a pixel or two deep, as a symbol seen from
/// far to one side has them, is too large a share to put into the map.
const EDGE_MIN_DEPTH: f64 = 3.0;
/// How far from a corner, in modules, an edge anchor must lie: nearer, it
/// says nothing the corner does not, and the projection drops it anyway.
const EDGE_ANCHOR_CLEARANCE: f64 = 2.5;

/// Anchors along the edges of the symbol, for paper that curls: each of
/// the four edges is traced against the quiet zone through the map the
/// corners and `anchors` make, and a few of the crossings, spread along
/// the edge, become anchors of their own, the crossing in the photo
/// standing for the point of the symbol's edge it was traced from. An
/// edge without a quiet zone cannot be traced, and a straight line that
/// is not clear of the symbol near the corner it was traced from
/// ([`Traced::clear`]) may be a row of modules inside it, and gives no
/// points; a bend is taken as it always was.
///
/// A straight edge is kept as a bowed one is where the map bends at it.
/// Between the corners alone a straight edge says nothing the corners do
/// not, and its points, each read to a fraction of a module, would only
/// add what they are out by: a symbol seen from so far to one side that
/// its modules were three pixels deep read without them and not with
/// them. But the map inside the outline bends through every alignment
/// pattern and timing module found, and on a bottle those bends run on
/// to the edge and past it, where nothing but the two corners at its
/// ends would hold them: the columns beside a straight side of a symbol
/// on a cylinder came out most of a module wide of it. So a straight
/// edge gives points where the anchors bend the map by
/// [`EDGE_BENT_BY`] modules somewhere along it and its modules are
/// [`EDGE_MIN_DEPTH`] pixels deep across it. The points hold the map
/// across the edge only ([`super::projection`]), since the trace cannot
/// say which point of the edge a crossing is. Returns the points of the
/// bowed edges and those of the straight ones, which the caller keeps
/// only where the symbol reads no worse for them.
fn edge_anchors(
    image: &Luma,
    func: &FunctionMap,
    corners: [[f64; 2]; 4],
    anchors: &[Anchor],
    sign: f64,
    threshold: Option<f32>,
) -> (Vec<Anchor>, Vec<Anchor>) {
    let (w, h) = (func.width as f64, func.height as f64);
    let controls: Vec<ControlPoint> = anchors.iter().map(Anchor::control_point).collect();
    let Some(map) = Projection::new(func.width, func.height, corners, &controls) else { return (Vec::new(), Vec::new()) };
    let (level, contrast) = edge_levels(image, func, &map, threshold);
    let outline = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
    let (mut bowed, mut straight) = (Vec::new(), Vec::new());
    // The top and left edges from the top-left corner, the bottom and
    // right from the bottom-right.
    for edge in edges_at(0, w, h).iter().chain(edges_at(2, w, h).iter()) {
        let Some(traced) = trace_edge(image, &map, edge, sign, level, contrast, false) else { continue };
        // A straight edge gives points only where the anchors inside
        // the symbol bend the map at it, and its modules are deep enough
        // across it for a crossing to say where it runs.
        let bends = |c: &ControlPoint| {
            let [dx, dy] = map.displacement(c.at[0], c.at[1]);
            distance([0.0, 0.0], [dx, dy]) >= EDGE_BENT_BY * module_size_at(&map, c.at)
        };
        let deep = |c: &ControlPoint| {
            distance(map.apply(c.at[0], c.at[1]), map.apply(c.at[0] + edge.inward[0], c.at[1] + edge.inward[1])) >= EDGE_MIN_DEPTH
        };
        if !(traced.bent || (traced.clear && traced.crossings.iter().any(bends) && traced.crossings.iter().all(deep))) {
            continue;
        }
        let clear: Vec<&ControlPoint> =
            traced.crossings.iter().filter(|c| outline.iter().all(|&o| distance(o, c.at) >= EDGE_ANCHOR_CLEARANCE)).collect();
        if clear.is_empty() {
            continue;
        }
        let stride = clear.len().div_ceil(EDGE_ANCHORS_PER_EDGE).max(1);
        for c in clear.iter().step_by(stride) {
            let point = Anchor { kind: AnchorKind::Edge, at: c.at, center: c.px, partial: false, inferred: false };
            if traced.bent { bowed.push(point) } else { straight.push(point) }
        }
    }
    (bowed, straight)
}

/// A corner traced from a start other than where the map puts it must
/// come out within this many modules of that start, and each of its edges
/// must have been seen within [`CORNER_SEEN_WITHIN`] modules of the
/// corner. A profile sees an edge only half a module or so from where the
/// outline through the start runs, so edges truly traced up to the corner
/// cross near the start; a crossing further off was run on to from the
/// far ends of the edges, where every start sees them, and on a symbol
/// whose edges bow that lands modules out, with paper beyond it all the
/// same.
const STARTED_WITHIN: f64 = 1.0;
const CORNER_SEEN_WITHIN: f64 = 5.0;
/// How much further out than the corner as traced from where the map
/// puts it a corner traced from another start must lie, in modules, to
/// be taken for it: nearer, the two are one corner read twice.
const OUTER_CORNER: f64 = 0.5;

/// Where the two edges that meet at corner `i` cross, when both trace
/// against the quiet zone ([`trace_edge`]). The edges are traced from
/// the corner as it stands, and the crossing stands when both are clear
/// of the symbol there, with paper beyond them. Where they are not, or
/// do not both trace, they are traced from each of `starts` in turn,
/// other places the corner may be, and from the corner as it stands
/// moved a module out along either edge and along both: a profile sees
/// an edge only within half a module or so of where the outline through
/// the start runs, and a start a module inside the symbol sees none, or
/// sees a row of modules for it. The first of those starts from which
/// both edges trace clear, and are seen near the corner, and cross
/// beside the start ([`STARTED_WITHIN`]), places a corner that did not
/// trace as it stood. With `fourth`, for the corner of a QR Code whose
/// other three finders measured, it also takes the place of a crossing
/// that was not clear, when it lies further out than that by
/// [`OUTER_CORNER`] modules: a line with paper beyond it does not run
/// inside the symbol, and what lies short of it does. Otherwise the
/// crossing from the corner as it stands is kept, clear or not, as it
/// always was: a placement on two finders of three has the third where
/// it takes a corner to be, with nothing clear about it, and a label
/// photographed round a bottle read only through such a placement.
/// `reach` is how far, in modules, the crossing may lie from the corner
/// as it stands.
#[allow(clippy::too_many_arguments)]
fn traced_corner(
    image: &Luma,
    func: &FunctionMap,
    corners: &[[f64; 2]; 4],
    i: usize,
    sign: f64,
    threshold: Option<f32>,
    starts: &[[f64; 2]],
    fourth: bool,
    reach: f64,
) -> Option<[f64; 2]> {
    let (w, h) = (func.width as f64, func.height as f64);
    let [cx, cy] = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]][i];
    // The crossing, whether both edges are clear, and the size of a
    // module there. `own` for the corner as it stands.
    let trace = |start: [f64; 2], own: bool| -> Option<([f64; 2], bool, f64)> {
        let mut outline = *corners;
        outline[i] = start;
        if !own && !convex(&outline) {
            return None;
        }
        let map = Homography::rect_to_quad(w, h, outline)?;
        let (level, contrast) = edge_levels(image, func, &map, threshold);
        let traced: Vec<Traced> = edges_at(i, w, h).iter().filter_map(|e| trace_edge(image, &map, e, sign, level, contrast, true)).collect();
        let [a, b] = traced.as_slice() else { return None };
        let p = a.line.intersect(&b.line)?;
        let seen = |t: &Traced| t.crossings.iter().any(|c| distance(c.at, [cx, cy]) <= CORNER_SEEN_WITHIN);
        let clear = a.clear && b.clear;
        if !own && !(clear && seen(a) && seen(b)) {
            return None;
        }
        let module = module_size_at(&map, [cx, cy]);
        let reach = if own { reach } else { STARTED_WITHIN };
        (distance(p, start) <= reach * module).then_some((p, clear, module))
    };
    let own = trace(corners[i], true);
    if let Some((p, clear, _)) = own
        && (clear || !fourth)
    {
        return Some(p);
    }
    // The other starts, and outwards from the corner in the symbol's own
    // modules.
    let outwards = Homography::rect_to_quad(w, h, *corners).map(|map| {
        let (ox, oy) = (if cx > 0.0 { 1.0 } else { -1.0 }, if cy > 0.0 { 1.0 } else { -1.0 });
        [(ox, 0.0), (0.0, oy), (ox, oy)].map(|(dx, dy)| map.apply(cx + dx, cy + dy))
    });
    let other = starts.iter().chain(outwards.iter().flatten()).find_map(|&start| trace(start, false));
    let out = |p: [f64; 2]| distance(p, corners[(i + 2) % 4]);
    match (own, other) {
        (Some((p, _, module)), Some((q, _, _))) => Some(if out(q) > out(p) + OUTER_CORNER * module { q } else { p }),
        (Some((p, _, _)), None) => Some(p),
        (None, other) => other.map(|(q, _, _)| q),
    }
}

/// Move corner `i` of an outline that an anchor holds, an alignment
/// pattern or a timing module some modules in from it, to where its two
/// edges meet, when both trace against the quiet zone from the corner as
/// it stands or from one of `starts` ([`traced_corner`]). The map through
/// the anchor runs on to the corner as the paper lies between them,
/// which on curled paper is not where the edge goes, and on a blurred
/// photo the anchor itself is read to a few tenths of a module; the
/// edges, traced over many modules, place the corner better either way.
/// Every boundary is expected `bloom` modules into the dark
/// ([`Detection::bloom`]), as [`settle_corner`] expects it. Returns
/// whether the corner was moved.
#[allow(clippy::too_many_arguments)]
fn settle_traced_corner(
    image: &Luma,
    func: &FunctionMap,
    corners: &mut [[f64; 2]; 4],
    i: usize,
    sign: f64,
    threshold: Option<f32>,
    starts: &[[f64; 2]],
    fourth: bool,
    bloom: f64,
) -> bool {
    let (w, h) = (func.width as f64, func.height as f64);
    let [cx, cy] = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]][i];
    // No further than a bend could carry it: beyond that the trace has
    // found other print.
    let Some(traced) = traced_corner(image, func, corners, i, sign, threshold, starts, fourth, MAX_EDGE_SAGITTA) else { return false };
    let [dx, dy] = Homography::rect_to_quad(w, h, *corners).map_or([0.0, 0.0], |map| bled_out(&map, w, h, [cx, cy], bloom));
    let p = [traced[0] + dx, traced[1] + dy];
    corners[i] = p;
    let Some(map) = Homography::rect_to_quad(w, h, *corners) else { return true };
    polish_corner_by_data(image, func, corners, i, sign, threshold, bloom);
    if distance(corners[i], p) > POLISH_LIMIT * module_size_at(&map, [cx, cy]) {
        corners[i] = p;
    }
    true
}

/// Move corner `i` of the outline to where its quarter of the symbol reads
/// best, by the symbol's edge where there is a quiet zone to trace it
/// against, and otherwise by the fixed and data modules of the quarter
/// alone. The edges are traced from the corner as it stands and from each
/// of `starts`, other places it may be, `fourth` saying whether it is a
/// QR Code's fourth corner ([`traced_corner`]). Every boundary is
/// expected `bloom` modules into the dark ([`Detection::bloom`]): the
/// edge of the ink a trace finds lies as far in from the symbol's edge,
/// and the quarter's boundaries, fitted as printed, settle to one side of
/// themselves. Returns whether the edge was traced on both sides.
#[allow(clippy::too_many_arguments)]
fn settle_corner(
    image: &Luma,
    func: &FunctionMap,
    corners: &mut [[f64; 2]; 4],
    i: usize,
    sign: f64,
    threshold: Option<f32>,
    starts: &[[f64; 2]],
    fourth: bool,
    bloom: f64,
) -> bool {
    let (w, h) = (func.width as f64, func.height as f64);
    let Some(map) = Homography::rect_to_quad(w, h, *corners) else { return false };
    let Some(scorer) = CornerScorer::new(image, func, *corners, i, threshold, bloom) else { return false };
    let [cx, cy] = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]][i];

    if let Some(traced) = traced_corner(image, func, corners, i, sign, threshold, starts, fourth, f64::INFINITY) {
        let [dx, dy] = bled_out(&map, w, h, [cx, cy], bloom);
        let p = [traced[0] + dx, traced[1] + dy];
        // Two traced edges fix the corner outright: their crossing is a
        // measurement, where the search below is only a preference. The
        // profiles that found the edges stepped a tenth of a module, and
        // the polish takes the corner the last fraction of a pixel. The
        // polish reads the quarter through a plane, and on curled paper
        // would walk the corner off its edges to where a plane reads the
        // quarter best, so it may take the corner only so far.
        corners[i] = p;
        polish_corner_by_data(image, func, corners, i, sign, threshold, bloom);
        if distance(corners[i], p) > POLISH_LIMIT * module_size_at(&map, [cx, cy]) {
            corners[i] = p;
        }
        return true;
    }
    let (edge_threshold, contrast) = edge_levels(image, func, &map, threshold);
    // Each traced edge moved out across itself by the bleed.
    let lines: Vec<Line> = edges_at(i, w, h)
        .iter()
        .filter_map(|e| {
            let Line { point: [x, y], dir } = trace_edge(image, &map, e, sign, edge_threshold, contrast, true)?.line;
            let (from, to) = (map.apply(e.corner[0], e.corner[1]), map.apply(e.corner[0] - e.inward[0] * bloom, e.corner[1] - e.inward[1] * bloom));
            Some(Line { point: [x + to[0] - from[0], y + to[1] - from[1]], dir })
        })
        .collect();
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
    // Every trial scored a row at a time, side by side on the cores to
    // spare, and the best taken in the order the rows and columns run.
    let rows: Vec<i32> = (-steps..=steps).collect();
    let scored = work::map(&rows, |&iy| {
        (-steps..=steps)
            .map(|ix| {
                let mut trial = *corners;
                trial[i] = map.apply(start_module[0] + ix as f64 * SEARCH_STEP, start_module[1] + iy as f64 * SEARCH_STEP);
                if let [a] = lines.as_slice() {
                    trial[i] = a.project(trial[i]);
                }
                (trial, scorer.score(trial, sign))
            })
            .collect::<Vec<_>>()
    });
    for (trial, score) in scored.into_iter().flatten() {
        if let Some(score) = score
            && score > best
        {
            best = score;
            best_outline = trial;
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
    polish_corner_by_data(image, func, corners, i, sign, threshold, bloom);
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
/// when the boundaries read sharper for it, expecting them `bloom`
/// modules into the dark ([`Detection::bloom`]).
fn polish_corner_by_data(
    image: &Luma,
    func: &FunctionMap,
    corners: &mut [[f64; 2]; 4],
    i: usize,
    sign: f64,
    threshold: Option<f32>,
    bloom: f64,
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
    let scorer = Scorer::new(image, &template, quarter.map(|[x, y]| map.apply(x, y)), threshold).with_along(6).with_bloom(bloom);
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

/// The offset, in modules along the symbol's own axes, at which the point
/// `at` of the symbol was found, at `found` in the photo, from where `map`
/// puts it.
fn offset_in_modules(map: &Homography, at: [f64; 2], found: [f64; 2]) -> Option<[f64; 2]> {
    let [x, y] = at;
    let o = map.apply(x, y);
    let (ex, ey) = (map.apply(x + 1.0, y), map.apply(x, y + 1.0));
    let (ex, ey) = ([ex[0] - o[0], ex[1] - o[1]], [ey[0] - o[0], ey[1] - o[1]]);
    let d = [found[0] - o[0], found[1] - o[1]];
    let det = ex[0] * ey[1] - ex[1] * ey[0];
    if det.abs() < 1e-12 {
        return None;
    }
    Some([(d[0] * ey[1] - d[1] * ey[0]) / det, (ex[0] * d[1] - ex[1] * d[0]) / det])
}

/// A QR Code version's alignment patterns as the grid they make: the
/// module coordinate of each of its columns, which are its rows' too.
/// The first and last of them cross at the three finders, where no
/// alignment pattern is.
struct AlignmentGrid {
    centers: Vec<usize>,
    width: f64,
    height: f64,
}

impl AlignmentGrid {
    /// `None` for a version with no alignment pattern, and for the other
    /// families: an rMQR symbol's alignment patterns lie along its edges
    /// and make no grid.
    fn of(func: &FunctionMap) -> Option<Self> {
        let SymbolVersion::Qr(v) = func.version else { return None };
        let centers = tables::qr_alignment_centers(v);
        (centers.len() >= 2).then_some(Self { centers, width: func.width as f64, height: func.height as f64 })
    }

    /// The column and row of the grid that a pattern centred at `at`
    /// stands in.
    fn place(&self, at: [f64; 2]) -> Option<(usize, usize)> {
        let index = |c: f64| self.centers.iter().position(|&k| (k as f64 + 0.5 - c).abs() < 1e-6);
        Some((index(at[0])?, index(at[1])?))
    }

    /// What stands at column `i`, row `j`: an alignment pattern and its
    /// centre, or, in three of the corners, the finder there and its.
    fn stands_at(&self, i: usize, j: usize) -> (AnchorKind, [f64; 2]) {
        let last = self.centers.len() - 1;
        if (i == 0 && j == 0) || (i == last && j == 0) || (i == 0 && j == last) {
            (AnchorKind::Finder, [if i == 0 { 3.5 } else { self.width - 3.5 }, if j == 0 { 3.5 } else { self.height - 3.5 }])
        } else {
            (AnchorKind::Alignment, [self.centers[i] as f64 + 0.5, self.centers[j] as f64 + 0.5])
        }
    }
}

/// The offset an alignment pattern is looked for at, in modules from
/// where `plane` puts it: the mean of the offsets at which its neighbours
/// in the grid, along its row and down its column, were found from where
/// the same plane puts them. This is BoofCV's
/// `QrCodeAlignmentPatternLocator`, which takes the patterns row by row
/// and starts each from its place under the homography through the
/// position patterns, moved by what the pattern to its left was found
/// off by across and the pattern above it down: paper that curls carries
/// a pattern much as far as it carried the pattern beside it, and a
/// plane cannot follow it at all. BoofCV carries one coordinate from each
/// of two neighbours, which over the geometry of symbols on cylinders
/// left the worst-placed pattern as far from its prediction as no carry
/// did; the mean of both coordinates over every neighbour that stands,
/// the finders among them at next to nothing, left it nearer than
/// either. The search here takes
/// its sites nearest what is known first, not row by row, so a neighbour
/// on any of the four sides may stand already. Only a neighbour standing
/// firm ([`Placed::firm`]) is carried from, and `None` is returned when
/// no alignment pattern beside this one does: the finders' offsets alone
/// say what the plane already says.
fn carried_offset(grid: &AlignmentGrid, site: &Site, plane: &Homography, placed: &[Placed]) -> Option<[f64; 2]> {
    let (i, j) = grid.place(site.at)?;
    let last = grid.centers.len() - 1;
    let mut sum = [0.0, 0.0];
    let (mut standing, mut patterns) = (0usize, 0usize);
    for (ni, nj) in [(i.wrapping_sub(1), j), (i + 1, j), (i, j.wrapping_sub(1)), (i, j + 1)] {
        if ni > last || nj > last {
            continue;
        }
        let (kind, at) = grid.stands_at(ni, nj);
        let Some(found) = placed.iter().find(|p| p.anchor.kind == kind && distance(p.anchor.at, at) < 1e-6 && (p.firm || kind == AnchorKind::Finder))
        else {
            continue;
        };
        let Some(offset) = offset_in_modules(plane, found.anchor.at, found.anchor.center) else { continue };
        sum = [sum[0] + offset[0], sum[1] + offset[1]];
        standing += 1;
        patterns += usize::from(kind == AnchorKind::Alignment);
    }
    (patterns > 0).then(|| [sum[0] / standing as f64, sum[1] / standing as f64])
}

/// Whether the offset at which an alignment pattern standing firm was
/// found from `plane` is what its neighbours in the grid would have it:
/// within the allowance any anchor has of where its neighbours predict
/// it, over the grid's own spacing. Along its row, and down its column,
/// the neighbours say: the mean of the two beside it where both stand;
/// the one beside it run on by its difference from the one beyond,
/// where two stand on one side, since paper round a bottle carries each
/// pattern a little further than the last; or the one beside it alone.
/// A finder stands for a neighbour in its corner of the grid. With no
/// alignment pattern standing beside it there is nothing to ask, and it
/// agrees: the one pattern of versions 2 to 6 has only its template to
/// speak for it. On flat paper the neighbours of a patch of data that
/// reads as a pattern, a module off the true one's place, all stand
/// where the plane has them, and it does not agree, and is doubted
/// against the plane as any anchor is; round a bottle every pattern is
/// off the plane, each much as its neighbours are.
fn agrees_with_neighbours(grid: &AlignmentGrid, plane: &Homography, placed: &[Placed], k: usize) -> bool {
    let Some((i, j)) = grid.place(placed[k].anchor.at) else { return true };
    let last = grid.centers.len() as i64 - 1;
    // The offset of what stands `steps` along the row or down the
    // column, and whether it is an alignment pattern.
    let stands = |di: i64, dj: i64| -> Option<([f64; 2], bool)> {
        let (ni, nj) = (i as i64 + di, j as i64 + dj);
        if ni < 0 || nj < 0 || ni > last || nj > last {
            return None;
        }
        let (kind, at) = grid.stands_at(ni as usize, nj as usize);
        let found = placed.iter().find(|p| p.anchor.kind == kind && distance(p.anchor.at, at) < 1e-6 && (p.firm || kind == AnchorKind::Finder))?;
        Some((offset_in_modules(plane, found.anchor.at, found.anchor.center)?, kind == AnchorKind::Alignment))
    };
    let mut said: Vec<[f64; 2]> = Vec::new();
    let mut patterns = false;
    for (di, dj) in [(1, 0), (0, 1)] {
        let (before, after) = (stands(-di, -dj), stands(di, dj));
        let run_on = |near: ([f64; 2], bool), sign: i64| match stands(2 * sign * di, 2 * sign * dj) {
            Some((far, _)) => [2.0 * near.0[0] - far[0], 2.0 * near.0[1] - far[1]],
            None => near.0,
        };
        patterns |= before.is_some_and(|b| b.1) || after.is_some_and(|a| a.1);
        match (before, after) {
            (Some(b), Some(a)) => said.push([(a.0[0] + b.0[0]) / 2.0, (a.0[1] + b.0[1]) / 2.0]),
            (Some(b), None) => said.push(run_on(b, -1)),
            (None, Some(a)) => said.push(run_on(a, 1)),
            (None, None) => {}
        }
    }
    if !patterns || said.is_empty() {
        return true;
    }
    let Some(own) = offset_in_modules(plane, placed[k].anchor.at, placed[k].anchor.center) else { return true };
    let n = said.len() as f64;
    let expected = [said.iter().map(|o| o[0]).sum::<f64>() / n, said.iter().map(|o| o[1]).sum::<f64>() / n];
    let step = (0..2)
        .map(|axis| {
            let c = [i, j][axis];
            let beside = if c > 0 { c - 1 } else { c + 1 };
            grid.centers[c].abs_diff(grid.centers[beside]) as f64
        })
        .fold(f64::INFINITY, f64::min);
    distance(own, expected) <= OUTLIER_MODULES.max(OUTLIER_BASE + OUTLIER_RATE * step)
}

/// Where a QR Code's alignment pattern lies when the template cannot be
/// fitted to it, by the mean shift BoofCV's `QrCodeAlignmentPatternLocator`
/// places every one of its alignment patterns with. The pattern is looked
/// for about where `map`, the fit through the anchors found so far bent
/// through the nearest of them, predicts it; the template has by then
/// been tried there and where the pattern's neighbours carry it to
/// ([`carried_offset`]). The shift works in module coordinates through the map, so that
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

/// Other places the corner `i` of the symbol may lie than where the fit
/// through the anchors puts it, for its edges to be traced from
/// ([`traced_corner`]): the fit is a plane, which on a creased or curled
/// symbol runs the corner no finder stands at a module and more inside it
/// or outside it, and every reader has its own way to that corner, none
/// of them right every time.
///
/// - Where the outer sides of the finders at the two corners beside it
///   cross, each side run on through the corners its own capstone
///   measured: BoofCV's `QrCodeDecoderImage` and OpenCV's
///   `QRCodeDetectorAruco` place the fourth corner so. Right for a crease
///   or a curl that leaves those two sides straight, and out where the
///   paper turns between a finder and the corner.
/// - Where the three other corners complete a parallelogram, once
///   finders have measured all three: ZXing's guess, which wechat_qrcode
///   keeps. Right for a symbol seen square on however it is creased, and
///   out by the foreshortening of one seen steeply.
/// - The fit moved by the offset at which an alignment pattern standing
///   firm within [`ANCHORED_WITHIN`] modules of the corner was found
///   from it, the nearest such, as [`carried_offset`] moves a pattern by
///   its neighbours'.
fn corner_starts(
    placed: &[Placed],
    plane: &Homography,
    corners: &[[f64; 2]; 4],
    measured: &[usize],
    outline: [[f64; 2]; 4],
    i: usize,
) -> Vec<[f64; 2]> {
    let mut starts = Vec::new();
    let beside = [(i + 1) % 4, (i + 3) % 4];
    // The side of the finder at corner `j` that lies along the symbol's
    // edge from there to corner `i`: the two corners of its patch on it.
    let side = |j: usize| -> Option<Line> {
        let axis = if (outline[j][0] - outline[i][0]).abs() < 1e-9 { 0 } else { 1 };
        let finder = placed.iter().find(|p| p.anchor.kind == AnchorKind::Finder && p.pairs.iter().any(|(at, _)| distance(*at, outline[j]) < 1e-9))?;
        let on: Vec<[f64; 2]> = finder.pairs.iter().filter(|(at, _)| (at[axis] - outline[i][axis]).abs() < 1e-9).map(|(_, px)| *px).collect();
        let [a, b] = on.as_slice() else { return None };
        let length = distance(*a, *b);
        (length > 0.0).then(|| Line { point: *a, dir: [(b[0] - a[0]) / length, (b[1] - a[1]) / length] })
    };
    if let (Some(a), Some(b)) = (side(beside[0]), side(beside[1]))
        && let Some(p) = a.intersect(&b)
    {
        starts.push(p);
    }
    if [beside[0], beside[1], (i + 2) % 4].iter().all(|j| measured.contains(j)) {
        starts.push(std::array::from_fn(|axis| corners[beside[0]][axis] + corners[beside[1]][axis] - corners[(i + 2) % 4][axis]));
    }
    let firm = placed
        .iter()
        .filter(|p| p.firm && distance(p.anchor.at, outline[i]) <= ANCHORED_WITHIN)
        .min_by(|a, b| distance(a.anchor.at, outline[i]).total_cmp(&distance(b.anchor.at, outline[i])));
    if let Some(firm) = firm
        && let Some([dx, dy]) = offset_in_modules(plane, firm.anchor.at, firm.anchor.center)
    {
        starts.push(plane.apply(outline[i][0] + dx, outline[i][1] + dy));
    }
    starts
}

/// How far about a lone alignment pattern, in modules each way, the data
/// modules are read to say whether it is the pattern, and how many rows
/// and columns in from the symbol's edges they are read to say whether
/// the points of a straight edge help.
const LONE_REACH: f64 = 6.0;
const EDGE_REACH: usize = 4;

/// Whether an alignment pattern standing firm lies beside `at` in the
/// grid, along its row or down its column.
fn has_pattern_beside(grid: &AlignmentGrid, placed: &[Placed], at: [f64; 2]) -> bool {
    let Some((i, j)) = grid.place(at) else { return false };
    let last = grid.centers.len() - 1;
    [(i.wrapping_sub(1), j), (i + 1, j), (i, j.wrapping_sub(1)), (i, j + 1)].into_iter().any(|(ni, nj)| {
        ni <= last && nj <= last && {
            let (kind, beside) = grid.stands_at(ni, nj);
            kind == AnchorKind::Alignment && placed.iter().any(|p| p.firm && distance(p.anchor.at, beside) < 1e-6)
        }
    })
}

/// How crisply the data modules of the symbol that `wanted` picks read
/// through `map`: the mean distance of their samples from the split
/// between dark and light, as the corner search judges a quarter of the
/// symbol. A map a fraction of a module out samples the boundaries
/// between modules, which read grey. `None` with fewer than eight of
/// them on the photo.
fn crispness(image: &Luma, func: &FunctionMap, map: &impl Project, wanted: impl Fn(usize, usize) -> bool, threshold: Option<f32>) -> Option<f64> {
    let positions: Vec<Pos> = (0..func.height)
        .flat_map(|y| (0..func.width).map(move |x| (x, y)))
        .filter(|&(x, y)| func.expected(x, y).is_none() && wanted(x, y))
        .map(|(x, y)| Pos::new(x, y))
        .collect();
    let values: Vec<f32> = sample_positions(image, map, &positions, INNER_FRACTION, 3).into_iter().flatten().collect();
    if values.len() < 8 {
        return None;
    }
    let split = threshold.unwrap_or_else(|| otsu(values.iter().copied()));
    Some(values.iter().map(|v| f64::from((v - split).abs())).sum::<f64>() / values.len() as f64)
}

/// Most rounds of [`look_again_for_alignment`]: BoofCV's ten steps.
const ALIGNMENT_ROUNDS: usize = 10;
/// A corner of the symbol within this many modules of an alignment
/// pattern is carried from as a neighbour in the grid is: the pattern in
/// the corner with no finder lies nine modules from that corner.
const CORNER_CARRIES: f64 = 10.0;

/// A second look for each of a QR Code's alignment patterns that does not
/// stand firm, missed or placed by its rings alone, once the first pass
/// over the sites is done and the corners are settled. On curled paper
/// the first look is from the wrong side: the sites are taken outwards
/// from the finders, each predicted from what lies behind it, and the
/// pattern furthest out, which the paper has carried furthest, is looked
/// for from the plane alone when it is the only one the version has. By
/// now there may be patterns standing firm on its other sides, and a
/// corner of the symbol beside it that its two edges or a finder placed;
/// each is a seed, the plane moved by the offsets the pattern's
/// neighbours carry ([`carried_offset`]) or by the corner's own, and the
/// template is looked for about each within the window any site is
/// searched in. A pattern found stands firm and takes the place of what
/// stood for it, and its own neighbours are looked for again, for at
/// most [`ALIGNMENT_ROUNDS`] rounds: BoofCV walks its guess to the
/// pattern in ten steps of a module or less along the gradient of nine
/// samples, which is not taken here for what
/// [`locate_alignment_by_meanshift`] says of it, and the walk here is
/// from pattern to pattern instead, each step checked by a template of
/// twenty-five modules. Returns whether any pattern was found.
#[allow(clippy::too_many_arguments)]
fn look_again_for_alignment(
    image: &Luma,
    func: &FunctionMap,
    grid: &AlignmentGrid,
    sites: &[Site],
    plane: &Homography,
    corners: [[f64; 2]; 4],
    held: &[usize],
    placed: &mut Vec<Placed>,
    inverted: bool,
    threshold: Option<f32>,
) -> bool {
    let outline = [[0.0, 0.0], [grid.width, 0.0], [grid.width, grid.height], [0.0, grid.height]];
    let window = |far: f64| (0.5 + 0.03 * far).clamp(0.75, WINDOW_PROJECTIVE);
    let mut any = false;
    // The patterns found in the round before: only their neighbours have
    // anything new to be looked for by. `None` in the first round, when
    // every site has.
    let mut fresh: Option<Vec<(usize, usize)>> = None;
    for _ in 0..ALIGNMENT_ROUNDS {
        let mut found_now = Vec::new();
        for site in sites.iter().filter(|s| s.kind == AnchorKind::Alignment) {
            let stands = |p: &Placed| p.anchor.kind == AnchorKind::Alignment && distance(p.anchor.at, site.at) < 1e-6;
            if placed.iter().any(|p| stands(p) && p.firm) {
                continue;
            }
            let Some((i, j)) = grid.place(site.at) else { continue };
            let beside = |&(fi, fj): &(usize, usize)| fi.abs_diff(i) + fj.abs_diff(j) == 1;
            let mut seeds: Vec<([f64; 2], f64)> = Vec::new();
            if fresh.as_ref().is_none_or(|fresh| fresh.iter().any(beside))
                && let Some(offset) = carried_offset(grid, site, plane, placed)
            {
                let step = |a: usize, b: usize| grid.centers[a.max(b)] - grid.centers[a.min(b)];
                let far = step(i, if i > 0 { i - 1 } else { i + 1 }).min(step(j, if j > 0 { j - 1 } else { j + 1 }));
                seeds.push((offset, far as f64));
            }
            if fresh.is_none() {
                for &k in held {
                    let far = distance(outline[k], site.at);
                    if far <= CORNER_CARRIES
                        && let Some(offset) = offset_in_modules(plane, outline[k], corners[k])
                    {
                        seeds.push((offset, far));
                    }
                }
            }
            for ([dx, dy], far) in seeds {
                let moved = super::refine::Shifted { map: plane, dx, dy };
                if let Some((anchor, pairs)) = search_site(image, func, site, &moved, window(far), inverted, threshold, 0.0)
                    && !anchor.partial
                {
                    // Only where its neighbours bear it out: a patch of
                    // data that reads as a pattern, dropped in the first
                    // pass for lying where no neighbour has it, is found
                    // here a second time.
                    placed.push(Placed { anchor, pairs, firm: true });
                    let agrees = agrees_with_neighbours(grid, plane, placed, placed.len() - 1);
                    let found = placed.pop().expect("just pushed");
                    if agrees {
                        placed.retain(|p| !stands(p));
                        placed.push(found);
                        found_now.push((i, j));
                        break;
                    }
                }
            }
        }
        if found_now.is_empty() {
            break;
        }
        any = true;
        fresh = Some(found_now);
    }
    any
}

/// A QR Code's alignment patterns placed by walking their grid from the
/// finders, as zxing-cpp's sampler places them: row by row, each pattern
/// looked for where the three found before it beside, above, and
/// diagonally above put it, as the fourth corner of their parallelogram,
/// bent on as the rows and columns before it bend. The patterns along the
/// top row and down the left column have only the finders and each other
/// before them; they are looked for along the cubic each edge traces from
/// one finder to the other, leaving and reaching each finder as that
/// finder's own capstone runs, moved off it as far as the pattern before
/// was found off it, less as the far finder nears. `None` unless all
/// three finders stand with their capstones' corners and the walk reads
/// three in four of the patterns whole by their template, and more of
/// them than the search did; and `None` where it places every pattern
/// the search found within half a module of where the search put it,
/// since it is then the same placement. The walk is given up as soon as
/// it can no longer read three in four. On the photos looked at, walks
/// through versions other than the symbol's own read a few patterns
/// whole and found the rest by the meanshift's rings; offered without
/// this test, the placements such walks made, each read in turn, slowed
/// the BoofCV benchmark by a ninth and read nothing more.
///
/// The search ([`search_assignment`]) looks for each site where the plane
/// through everything placed so far puts it, within a few modules. On a
/// page bent where a finger held its corner up, the top-right finder lay
/// at an angle to the other two, the plane through the three ran the
/// patterns beside the top-left finder four modules off, and the search
/// took patches of data a few modules from the plane for the patterns,
/// one after another; left without the bent finder, it placed the far
/// half of the symbol by the plane through the near half. Walked, every
/// one of the thirty-three patterns of that version 34 symbol was found,
/// the farthest of them a module and a half from where the walk looked
/// for it, and the symbol read as zxing-cpp reads it. As parallelograms
/// alone, the farthest was 2.3 modules off, at the edge of the window
/// looked in.
/// The walk goes no further than the next pattern from what it has found:
/// a pattern it does not find is taken to stand where it was looked for,
/// for the walk on, and is left to its neighbours' lines.
#[allow(clippy::too_many_arguments)]
fn walk_alignment(
    image: &Luma,
    func: &FunctionMap,
    grid: &AlignmentGrid,
    sites: &[Site],
    placed: &[Placed],
    inverted: bool,
    threshold: Option<f32>,
    levels: Option<(f32, f32)>,
) -> Option<(Vec<Placed>, Vec<Site>)> {
    let last = grid.centers.len() - 1;
    if last < 2 {
        return None;
    }
    let finders: Vec<&Placed> = placed.iter().filter(|p| p.anchor.kind == AnchorKind::Finder).collect();
    // Each finder's own map, through its capstone's corners.
    let own_map = |i: usize, j: usize| -> Option<Homography> {
        let (_, at) = grid.stands_at(i, j);
        let finder = finders.iter().find(|p| distance(p.anchor.at, at) < 1e-6)?;
        if finder.pairs.len() < 4 {
            return None;
        }
        Homography::fit(Model::Projective, &finder.pairs)
    };
    // The top-left finder's, and the one each edge from it runs to.
    let (corner, ends) = (own_map(0, 0)?, [own_map(last, 0)?, own_map(0, last)?]);
    let node = |i: usize, j: usize| [grid.centers[i] as f64 + 0.5, grid.centers[j] as f64 + 0.5];
    let step = |map: &Homography, [x, y]: [f64; 2], axis: usize| {
        let (here, there) = (map.apply(x, y), if axis == 0 { map.apply(x + 1.0, y) } else { map.apply(x, y + 1.0) });
        [there[0] - here[0], there[1] - here[1]]
    };
    let span = (grid.centers[last] - grid.centers[0]) as f64;
    let share = |c: usize| (grid.centers[c] - grid.centers[0]) as f64 / span;
    let (start, far) = (node(0, 0), [node(last, 0), node(0, last)]);
    // Where the cubic along `edge`, the top row or the left column, puts
    // the node a share `t` of the way from the top-left finder to the
    // finder at its end, and the module's steps across and down there,
    // from the finders' at either end in proportion.
    let on_edge = |edge: usize, t: f64| -> ([f64; 2], [[f64; 2]; 2]) {
        let end = &ends[edge];
        let (p0, p1) = (corner.apply(start[0], start[1]), end.apply(far[edge][0], far[edge][1]));
        let (t0, t1) = (step(&corner, start, edge), step(end, far[edge], edge));
        let h = [2.0 * t.powi(3) - 3.0 * t * t + 1.0, t.powi(3) - 2.0 * t * t + t, 3.0 * t * t - 2.0 * t.powi(3), t.powi(3) - t * t];
        let on = std::array::from_fn(|k| h[0] * p0[k] + h[1] * span * t0[k] + h[2] * p1[k] + h[3] * span * t1[k]);
        let axes = [0, 1].map(|axis| {
            let (a, b) = (step(&corner, start, axis), step(end, far[edge], axis));
            std::array::from_fn(|k| (1.0 - t) * a[k] + t * b[k])
        });
        (on, axes)
    };
    let n = last + 1;
    // Where each node of the grid stands in the photo, found or guessed.
    let mut at: Vec<Option<[f64; 2]>> = vec![None; n * n];
    at[0] = Some(corner.apply(start[0], start[1]));
    at[last] = Some(ends[0].apply(far[0][0], far[0][1]));
    at[last * n] = Some(ends[1].apply(far[1][0], far[1][1]));
    let mut walked: Vec<Placed> = finders.iter().map(|p| Placed { anchor: p.anchor, pairs: p.pairs.clone(), firm: p.firm }).collect();
    let mut missed = Vec::new();
    let patterns = sites.iter().filter(|s| s.kind == AnchorKind::Alignment).count();
    let mut loose = 0;
    let less = |a: [f64; 2], b: [f64; 2]| [a[0] - b[0], a[1] - b[1]];
    for (i, j) in (0..n).flat_map(|j| (0..n).map(move |i| (i, j))) {
        if at[j * n + i].is_some() {
            continue;
        }
        let (guess, [across, down]) = if i == 0 || j == 0 {
            let (edge, c) = if j == 0 { (0, i) } else { (1, j) };
            let (on, axes) = on_edge(edge, share(c));
            let mut off = [0.0, 0.0];
            if c >= 2 {
                let before = if edge == 0 { at[c - 1] } else { at[(c - 1) * n] }?;
                let fade = (1.0 - share(c)) / (1.0 - share(c - 1));
                off = less(before, on_edge(edge, share(c - 1)).0).map(|v| fade * v);
            }
            ([on[0] + off[0], on[1] + off[1]], axes)
        } else {
            let (a, b, c) = (at[j * n + i - 1]?, at[(j - 1) * n + i]?, at[(j - 1) * n + i - 1]?);
            let (di, dj) = ((grid.centers[i] - grid.centers[i - 1]) as f64, (grid.centers[j] - grid.centers[j - 1]) as f64);
            let (across, down) = (less(b, c), less(a, c));
            // How the step across changed from the row before last to
            // the row before, and the step down from the column before
            // last to the column before: carried on once more, in
            // proportion to the gaps between the rows and columns, the
            // first of which is narrower in most versions.
            let mut bends: Vec<[f64; 2]> = Vec::new();
            if j >= 2 {
                let scale = dj / (grid.centers[j - 1] - grid.centers[j - 2]) as f64;
                bends.push(less(across, less(at[(j - 2) * n + i]?, at[(j - 2) * n + i - 1]?)).map(|v| scale * v));
            }
            if i >= 2 {
                let scale = di / (grid.centers[i - 1] - grid.centers[i - 2]) as f64;
                bends.push(less(down, less(at[j * n + i - 2]?, at[(j - 1) * n + i - 2]?)).map(|v| scale * v));
            }
            let bend = if bends.is_empty() {
                [0.0, 0.0]
            } else {
                let k = bends.len() as f64;
                [bends.iter().map(|b| b[0]).sum::<f64>() / k, bends.iter().map(|b| b[1]).sum::<f64>() / k]
            };
            let guess = [a[0] + across[0] + bend[0], a[1] + across[1] + bend[1]];
            (guess, [across.map(|v| v / di), down.map(|v| v / dj)])
        };
        let here = node(i, j);
        let site = sites.iter().find(|s| s.kind == AnchorKind::Alignment && distance(s.at, here) < 1e-6)?;
        // The module's steps about the guess, as a map for the search.
        let local = Homography::from_matrix([
            across[0],
            down[0],
            guess[0] - here[0] * across[0] - here[1] * down[0],
            across[1],
            down[1],
            guess[1] - here[0] * across[1] - here[1] * down[1],
            0.0,
            0.0,
            1.0,
        ])?;
        // A pattern the search read whole within half a module of where
        // the walk looks is the pattern there, and is not looked for
        // again: on flat paper the walk is then the search's placement,
        // at the cost of predicting it.
        let module = (across[0].hypot(across[1]) + down[0].hypot(down[1])) / 2.0;
        let held = placed
            .iter()
            .find(|p| p.firm && p.anchor.kind == AnchorKind::Alignment && distance(p.anchor.at, here) < 1e-6)
            .filter(|p| distance(p.anchor.center, guess) < 0.5 * module)
            .map(|p| Placed { anchor: p.anchor, pairs: p.pairs.clone(), firm: true });
        let found = held
            .or_else(|| search_site(image, func, site, &local, WINDOW_PROJECTIVE, inverted, threshold, 0.0).map(|(anchor, pairs)| Placed { firm: !anchor.partial, anchor, pairs }))
            .or_else(|| {
                let (split, contrast) = levels?;
                let center = locate_alignment_by_meanshift(image, site, &local, WINDOW_PROJECTIVE, inverted, split, contrast)?;
                Some(Placed { anchor: Anchor { kind: AnchorKind::Alignment, at: here, center, partial: false, inferred: false }, pairs: vec![(here, center)], firm: false })
            });
        if found.as_ref().is_none_or(|found| !found.firm) {
            loose += 1;
            if 4 * loose > patterns {
                return None;
            }
        }
        match found {
            Some(found) => {
                at[j * n + i] = Some(found.anchor.center);
                walked.push(found);
            }
            None => {
                at[j * n + i] = Some(guess);
                missed.push(*site);
            }
        }
    }
    if patterns - loose <= placed.iter().filter(|p| p.firm).count() {
        return None;
    }
    let module = module_size_at(&corner, start);
    let elsewhere = walked.iter().filter(|w| w.anchor.kind == AnchorKind::Alignment && w.firm).any(|w| {
        placed
            .iter()
            .find(|p| p.anchor.kind == AnchorKind::Alignment && distance(p.anchor.at, w.anchor.at) < 1e-6)
            .is_none_or(|p| distance(p.anchor.center, w.anchor.center) > 0.5 * module)
    });
    elsewhere.then_some((walked, missed))
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
    // Where the capstones stand as a QR Code's finders more ways than
    // one, the symbol's own format word says which, when it reads
    // outright beside the finders of some and not of others
    // ([`format_reads_beside`]): those are fitted, and the others not.
    // Two capstones stand as two of three finders at two turns alike,
    // and three stand at every turn within the slack. The fits were all
    // made and the best taken, and the fit reads the fixed modules and
    // the quiet zone against one split of them: in a photo whose light
    // failed toward one side the split of the right turn's modules
    // called more than half the quiet zone dark, the split of a wrong
    // turn's none of it, and a symbol with all three finders showing
    // was placed a quarter turn out on two of them, its third standing
    // where its data should be. The word is read beside the capstones
    // themselves, each against its own dark and light, and does not
    // hang on the fit or on the light; it misleads once in a thousand,
    // and where it reads, three fits in four need not be made.
    if matches!(func.version, SymbolVersion::Qr(_) | SymbolVersion::Model1(_)) && chosen.len() > 1 {
        let site_refs: Vec<&Site> = finders.iter().map(|&i| &sites[i]).collect();
        let worded: Vec<usize> = chosen
            .iter()
            .copied()
            .filter(|&i| format_reads_beside(image, func, &site_refs, capstones, &matches[i].3, matches[i].1, inverted, threshold, band))
            .collect();
        if !worded.is_empty() {
            chosen = worded;
        }
    }
    let mut out = Vec::new();
    for i in chosen {
        let (_, turns, mirrored, assignment) = &matches[i];
        let standing = Standing { sites: &sites, finders: &finders, rmqr, turns: *turns, mirrored: *mirrored, assignment };
        out.extend(align_assignment(image, func, capstones, inverted, threshold, band, prename, standing));
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
/// its finders, as [`align_version`] describes: the placement the search
/// grows from the finders, and with it, where the walk from the finders
/// places some alignment pattern elsewhere ([`walk_alignment`]), the
/// walk's, with the timing patterns' modules looked for again where the
/// walk puts them ([`grow_from`]).
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
) -> Vec<(Detection, f64)> {
    let Some((placed, missed, grid, levels)) = search_assignment(image, func, capstones, inverted, threshold, standing) else { return Vec::new() };
    // Walked only at the version the symbol's own bits name, where they
    // name one: at any other the patterns are not where the walk looks.
    let walked = grid
        .as_ref()
        .filter(|_| prename.is_none_or(|p| p.version == func.version))
        .and_then(|grid| walk_alignment(image, func, grid, standing.sites, &placed, inverted, threshold, levels))
        .and_then(|(mut walked, mut missed)| {
            // The walk places the finders and the alignment patterns. The
            // rest, the timing patterns' modules, are looked for again
            // where the walk's placement puts them, as the search looks
            // for every site, and not taken from the search: where the walk
            // is wanted, the plane the search grew from the finders runs
            // off along the side that bends, and finds few of them there.
            // On a page lifted at one corner it found none along the top
            // row, and the walk's placement nine in twelve of them, each
            // where it lies.
            let sites = standing.sites;
            let done = |site: &Site| walked.iter().any(|p| p.anchor.kind == site.kind && p.anchor.at == site.at) || missed.iter().any(|m| m.kind == site.kind && m.at == site.at);
            let todo: Vec<usize> = (0..sites.len()).filter(|&i| !done(&sites[i])).collect();
            grow_from(image, func, sites, &mut walked, todo, grid.as_ref(), levels, inverted, threshold, &mut missed)?;
            Some((walked, missed))
        });
    let finish = |placed, missed| finish_assignment(image, func, placed, missed, &grid, inverted, threshold, band, prename, standing);
    let mut out: Vec<(Detection, f64)> = finish(placed, missed).into_iter().collect();
    if let Some((walked, missed)) = walked {
        out.extend(finish(walked, missed));
    }
    out
}

/// The anchors the search grows from the finders under one assignment:
/// each site looked for nearest what is already placed, where the fit
/// through everything placed so far puts it. With them, the alignment
/// patterns it did not find, the grid of a QR Code's alignment patterns
/// when all three finders stand, and the finders' own dark and light.
#[allow(clippy::type_complexity)]
fn search_assignment(
    image: &Luma,
    func: &FunctionMap,
    capstones: &[Capstone],
    inverted: bool,
    threshold: Option<f32>,
    standing: Standing,
) -> Option<(Vec<Placed>, Vec<Site>, Option<AlignmentGrid>, Option<(f32, f32)>)> {
    let (w, h) = (func.width as f64, func.height as f64);
    let Standing { sites, finders, turns, mirrored, assignment, .. } = standing;
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
                firm: false,
            });
            todo.retain(|&i| i != site_index);
        }
    }
    if placed.is_empty() {
        return None;
    }

    // The finders' own dark and light, for the locator of an alignment
    // pattern the template cannot be fitted to: BoofCV reads its
    // alignment patterns against the corner position pattern's threshold.
    let levels = match func.version {
        SymbolVersion::Qr(_) => finder_capstone_levels(image, capstones, assignment, threshold),
        _ => None,
    };
    // Only with all three finders standing: the plane through fewer runs
    // modules out on the side where one is missing, there is nothing
    // there for a pattern read to be near, and a patch of data that reads
    // as a pattern, kept, bends the map to itself and carries its offset
    // on to its neighbours' search.
    let grid = AlignmentGrid::of(func).filter(|_| assignment.iter().flatten().count() == 3);
    let mut missed: Vec<Site> = Vec::new();
    grow_from(image, func, sites, &mut placed, todo, grid.as_ref(), levels, inverted, threshold, &mut missed)?;
    Some((placed, missed, grid, levels))
}

/// The sites `todo` of `sites` placed as [`search_assignment`] places
/// them, one after another, each the nearest to what is placed already,
/// looked for where the fit through everything placed so far puts it,
/// and added to `placed`: a QR Code's alignment pattern first where its
/// neighbours in `grid` carry it to, and then by its rings alone against
/// the finders' own `levels`, and where neither finds it, put among the
/// `missed`. `None` where the anchors placed fit no map.
#[allow(clippy::too_many_arguments)]
fn grow_from(
    image: &Luma,
    func: &FunctionMap,
    sites: &[Site],
    placed: &mut Vec<Placed>,
    mut todo: Vec<usize>,
    grid: Option<&AlignmentGrid>,
    levels: Option<(f32, f32)>,
    inverted: bool,
    threshold: Option<f32>,
    missed: &mut Vec<Site>,
) -> Option<()> {
    let nearest = |anchors: &[Anchor], at: [f64; 2]| anchors.iter().map(|a| distance(a.at, at)).fold(f64::INFINITY, f64::min);
    while !todo.is_empty() {
        let (map, model) = fit_anchors(placed, grid)?;
        let anchors = anchors_of(placed);
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
        let local = local_map(map, placed);
        // A QR Code's alignment pattern is looked for first where its
        // neighbours in the grid carry it to, and then, should its
        // template not read whole there, where the fit bent through
        // every anchor puts it, as every other site is.
        let carried = match grid {
            Some(grid) if site.kind == AnchorKind::Alignment => carried_offset(grid, &site, &map, placed),
            _ => None,
        };
        let mut found = carried.and_then(|[dx, dy]| search_site(image, func, &site, &super::refine::Shifted { map: &map, dx, dy }, window, inverted, threshold, 0.0));
        if found.as_ref().is_none_or(|(anchor, _)| anchor.partial) {
            let plain = search_site(image, func, &site, &local, window, inverted, threshold, 0.0);
            if found.is_none() || plain.as_ref().is_some_and(|(anchor, _)| !anchor.partial) {
                found = plain;
            }
        }
        if let Some((anchor, pairs)) = found {
            let firm = grid.is_some() && site.kind == AnchorKind::Alignment && !anchor.partial;
            placed.push(Placed { anchor, pairs, firm });
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
                    firm: false,
                }),
                None => missed.push(site),
            }
        }
    }
    Some(())
}

/// The detection the anchors `placed` make, and its fit, as
/// [`align_assignment`] describes: the alignment patterns `missed` placed
/// where their neighbours' lines cross, the map fitted through every
/// anchor, the corners measured or settled, and the fit read.
#[allow(clippy::too_many_arguments)]
fn finish_assignment(
    image: &Luma,
    func: &FunctionMap,
    mut placed: Vec<Placed>,
    missed: Vec<Site>,
    grid: &Option<AlignmentGrid>,
    inverted: bool,
    threshold: Option<f32>,
    band: f32,
    prename: Option<Prename>,
    standing: Standing,
) -> Option<(Detection, f64)> {
    let (w, h) = (func.width as f64, func.height as f64);
    let Standing { sites, rmqr, turns, mirrored, .. } = standing;
    let nearest = |anchors: &[Anchor], at: [f64; 2]| anchors.iter().map(|a| distance(a.at, at)).fold(f64::INFINITY, f64::min);
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
            placed.push(Placed { anchor, pairs: vec![(site.at, center)], firm: false });
        }
    }

    let (mut map, _) = fit_anchors(&mut placed, grid.as_ref())?;
    // A timing anchor off the bend its line-mates trace has read a row of
    // data or a smudge: it is dropped, and the map refitted without it.
    let strays = timing_line_strays(&anchors_of(&placed), &map);
    if !strays.is_empty() {
        let mut drop: Vec<usize> = strays.iter().map(|(i, _)| *i).collect();
        drop.sort_unstable_by(|a, b| b.cmp(a));
        for i in drop {
            placed.remove(i);
        }
        map = fit_anchors(&mut placed, grid.as_ref())?.0;
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
    // The one corner of a QR Code that no finder stands at, the other
    // three measured.
    let fourth = matches!(func.version, SymbolVersion::Qr(_) | SymbolVersion::Model1(_)) && measured_corners.len() == 3;
    let mut loose: Vec<usize> = (0..4).filter(|&i| nearest(&anchors, outline[i]) > ANCHORED_WITHIN).collect();
    loose.sort_by(|&a, &b| nearest(&anchors, outline[a]).total_cmp(&nearest(&anchors, outline[b])));
    let mut edged_corners = Vec::new();
    // A corner no finder measured but an anchor holds follows the map
    // between that anchor and the edge; where both its edges trace, they
    // place it instead. First, so that a loose corner beside it starts
    // from the better place.
    for i in (0..4).filter(|i| !measured_corners.contains(i) && !loose.contains(i)) {
        let starts = corner_starts(&placed, &map, &corners, &measured_corners, outline, i);
        if settle_traced_corner(image, func, &mut corners, i, sign, threshold, &starts, fourth, 0.0) {
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
            let starts = corner_starts(&placed, &map, &corners, &measured_corners, outline, i);
            let mut edged = settle_corner(image, func, &mut corners, i, sign, threshold, &starts, fourth, 0.0);
            if pass == 0 && !edged && lone_finder && n + 1 == loose.len() {
                // The edges were not found from where the map put the
                // corner: the corner opposite the one finder is tried
                // again from where the three corners now placed complete
                // the parallelogram, which on flat paper is near enough
                // for the edges to be traced from.
                let mut again = before;
                again[i] = std::array::from_fn(|axis| before[(i + 1) % 4][axis] + before[(i + 3) % 4][axis] - before[(i + 2) % 4][axis]);
                if settle_corner(image, func, &mut again, i, sign, threshold, &[], false, 0.0) {
                    corners = again;
                    edged = true;
                }
            }
            if edged && pass == 1 {
                edged_corners.push(i);
            }
        }
    }

    // An alignment pattern that does not yet stand firm is looked for
    // again, now that the corners are placed and every pattern has had
    // its turn.
    if let Some(grid) = &grid {
        let held: Vec<usize> = measured_corners.iter().chain(&edged_corners).copied().collect();
        if look_again_for_alignment(image, func, grid, sites, &map, corners, &held, &mut placed, inverted, threshold) {
            anchors = anchors_of(&placed);
        }
    }

    // Between the anchors and an edge of the symbol the map is the
    // spline's guess: points along each edge that traces anchor it there,
    // so that the rows beside the edge follow the paper and not the
    // plane. A version 1 symbol on a bottle has anchors on its timing
    // patterns alone, and its far quadrant is placed by nothing else.
    let (bowed, straight) = edge_anchors(image, func, corners, &anchors, sign, threshold);
    anchors.extend(bowed);
    let project = |anchors: &[Anchor]| {
        let controls: Vec<ControlPoint> = anchors.iter().map(Anchor::control_point).collect();
        Projection::new(func.width, func.height, corners, &controls)
    };
    // An alignment pattern standing firm with no other beside it to
    // bear it out, as the one pattern of versions 2 to 6 stands, and
    // lying where the map through everything else would not have it, is
    // either the pattern on paper that curls or a patch of data that
    // reads as one. The data modules about it say which: through the
    // right map they read dark or light, and through a map a module out
    // they read grey. It is kept where they read no less crisply with
    // it than without.
    if let Some(grid) = &grid {
        for lone in placed.iter().filter(|p| p.firm && !has_pattern_beside(grid, &placed, p.anchor.at)) {
            let at = lone.anchor.at;
            let without: Vec<Anchor> = anchors.iter().filter(|a| !(a.kind == AnchorKind::Alignment && a.at == at)).copied().collect();
            let (Some(with_it), Some(without_it)) = (project(&anchors), project(&without)) else { continue };
            if distance(without_it.apply(at[0], at[1]), lone.anchor.center) < OUTLIER_MODULES * module_size_at(&without_it, at) {
                continue;
            }
            let about = |x: usize, y: usize| (x as f64 + 0.5 - at[0]).abs() <= LONE_REACH && (y as f64 + 0.5 - at[1]).abs() <= LONE_REACH;
            if let (Some(with_it), Some(without_it)) = (crispness(image, func, &with_it, about, threshold), crispness(image, func, &without_it, about, threshold))
                && without_it > with_it
            {
                anchors = without;
            }
        }
    }
    // The points of a straight edge are kept where the modules beside
    // the symbol's edges read no less crisply for them: they are there
    // to hold a map the anchors have bent, and where the bend was an
    // anchor read badly under a lamp's glare they would hold it wrong.
    if !straight.is_empty() {
        let mut held = anchors.clone();
        held.extend(straight);
        let beside = |x: usize, y: usize| x < EDGE_REACH || y < EDGE_REACH || x + EDGE_REACH >= func.width || y + EDGE_REACH >= func.height;
        let crisp = |anchors: &[Anchor]| project(anchors).and_then(|map| crispness(image, func, &map, beside, threshold));
        if let (Some(with_them), Some(without_them)) = (crisp(&held), crisp(&anchors))
            && with_them >= without_them
        {
            anchors = held;
        }
    }
    let projection = project(&anchors)?;
    let scorer = Scorer::new(image, &Template::whole(func), corners, threshold);
    let agreement = scorer.agreement_known(&projection, inverted)?.0;
    let around = [[-1.0, -1.0], [w + 1.0, -1.0], [w + 1.0, h + 1.0], [-1.0, h + 1.0]].map(|[x, y]| projection.apply(x, y));
    let quiet = Scorer::new(image, &Template::with_quiet_zone(func), around, threshold);
    // Against a threshold that follows the light from corner to corner:
    // the fit ranks the placements, and against one split of the fixed
    // modules and the quiet zone together a symbol lit unevenly ranks
    // below the same square turned to stand on two of its finders, and
    // is not among the few the decode is offered.
    let (matching, _) = quiet.agreement_shaded(&super::refine::Shifted { map: &projection, dx: -1.0, dy: -1.0 }, inverted)?;
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
        turned: 0,
        decodes: None,
        bloom: 0.0,
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
    settle_loose_corners_with(image, func, corners, &anchored, sign, 0.0, params)
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
/// for light on dark), with the anchors given by their module coordinates,
/// expecting every boundary `bloom` modules into the dark
/// ([`Detection::bloom`]).
pub(crate) fn settle_loose_corners_with(
    image: &Luma,
    func: &FunctionMap,
    corners: [[f64; 2]; 4],
    anchored: &[[f64; 2]],
    sign: f64,
    bloom: f64,
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
            // Called off, the corners come back as they stand.
            if work::cancelled() {
                return corners;
            }
            settle_corner(image, func, &mut corners, i, sign, threshold, &[], false, bloom);
        }
    }
    corners
}

/// The bleed a pattern's boundaries show ([`Scorer::with_bloom`]): the
/// `template` fitted from `from`, where fitted as printed it reads
/// `plain` contrast, expecting each bleed of [`BLOOMS`], and then a
/// [`BLEED_STEP`] either side of the crispest. The fit, its contrast and
/// its bleed, when it reads more than [`BLOOM_GAIN`] times as crisply as
/// the plain fit; `None` when no bleed does.
///
/// A pattern that has bled is fitted off itself by the plain template:
/// where the dark has thinned by a third of a module, the template a
/// third of a module to one side lies right on the boundaries of that
/// side, which reads more crisply than lying a third off every boundary,
/// and the plain fit settles there. A template that expects the bleed
/// lies on every boundary at once, and reads more crisply still; one that
/// expects a bleed the photo does not have reads less crisply than the
/// plain one.
fn measured_bleed(
    image: &Luma,
    template: &Template,
    from: [[f64; 2]; 4],
    plain: f64,
    sign: f64,
    threshold: Option<f32>,
    evaluations: &mut usize,
) -> Option<([[f64; 2]; 4], f64, f64)> {
    if plain <= 0.0 {
        return None;
    }
    let fit = |b: f64, evaluations: &mut usize| {
        let bloomed = Scorer::new(image, template, from, threshold).with_along(8).with_bloom(b);
        bloomed.descend(from, sign, &MOVES, 0.5 * bloomed.module_px(), evaluations).map(|(corners, score)| (corners, score, b))
    };
    let crispest = |best: Option<([[f64; 2]; 4], f64, f64)>, next: Option<([[f64; 2]; 4], f64, f64)>| match (best, next) {
        (Some(best), Some(next)) if next.1 > best.1 => Some(next),
        (None, next) => next,
        (best, _) => best,
    };
    let coarse = BLOOMS.into_iter().map(|b| fit(b, evaluations)).fold(None, crispest)?;
    let fine = [coarse.2 - BLEED_STEP, coarse.2 + BLEED_STEP].into_iter().map(|b| fit(b, evaluations)).fold(Some(coarse), crispest)?;
    (fine.1 > BLOOM_GAIN * plain).then_some(fine)
}

/// A symbol's bleed from the bleeds measured of its finders
/// ([`measured_bleed`]): their middle, when there are two or more and
/// every one bled the same way, and nought otherwise. A bleed is the
/// print's and the exposure's, and shows at every finder alike. A finder
/// blurred, as against bled, can read more crisply expecting a bleed of
/// either sign, since a blur thins whichever of its rings is narrower
/// than its neighbours: on BoofCV's photographs of a symbol out of focus
/// one finder or two of a placement measured a bleed of an eighth of a
/// module into the light where the others measured none.
fn symbol_bleed(bleeds: impl IntoIterator<Item = f64>) -> f64 {
    let mut bleeds: Vec<f64> = bleeds.into_iter().collect();
    let alike = bleeds.iter().all(|&b| b > 0.0) || bleeds.iter().all(|&b| b < 0.0);
    if bleeds.len() < 2 || !alike {
        return 0.0;
    }
    bleeds.sort_by(f64::total_cmp);
    (bleeds[(bleeds.len() - 1) / 2] + bleeds[bleeds.len() / 2]) / 2.0
}

/// What the finders of a placed symbol are measured through
/// ([`bloom_of`]): the map its corners and anchors make, and the sign of
/// its polarity, -1 for light on dark.
struct BleedFrame {
    map: Projection,
    sign: f64,
    threshold: Option<f32>,
}

impl BleedFrame {
    /// For a symbol light on dark when `inverted`, as whoever placed it
    /// settled. `None` when the corners and anchors make no map.
    fn new(func: &FunctionMap, corners: [[f64; 2]; 4], anchors: &[Anchor], inverted: bool, params: &SampleParams) -> Option<Self> {
        let controls: Vec<ControlPoint> = anchors.iter().map(Anchor::control_point).collect();
        let map = Projection::new(func.width, func.height, corners, &controls)?;
        Some(Self { map, sign: if inverted { -1.0 } else { 1.0 }, threshold: fixed_threshold(params) })
    }

    /// The bleed the finder at `anchor` shows: fitted as printed from
    /// where the map and the anchor put it, and its bleed measured from
    /// there ([`measured_bleed`]); nought where it shows none.
    fn finder(&self, image: &Luma, anchor: &Anchor) -> f64 {
        let (map, sign, threshold) = (&self.map, self.sign, self.threshold);
        let template = Template::finder_core();
        let [ax, ay] = map.apply(anchor.at[0], anchor.at[1]);
        let start = [[-3.5, -3.5], [3.5, -3.5], [3.5, 3.5], [-3.5, 3.5]].map(|[dx, dy]| {
            let [px, py] = map.apply(anchor.at[0] + dx, anchor.at[1] + dy);
            [px + anchor.center[0] - ax, py + anchor.center[1] - ay]
        });
        let mut evaluations = 0;
        let plain = Scorer::new(image, &template, start, threshold).with_along(8);
        plain
            .descend(start, sign, &MOVES, 0.5 * plain.module_px(), &mut evaluations)
            .and_then(|(fitted, contrast)| measured_bleed(image, &template, fitted, contrast, sign, threshold, &mut evaluations))
            .map_or(0.0, |(.., b)| b)
    }
}

/// The bleed of a placed symbol ([`Detection::bloom`]), read off its
/// finder anchors ([`symbol_bleed`]): each finder fitted as printed from
/// where the anchors and `corners` put it, light on dark when `inverted`
/// as the placement has it, and its bleed measured from there
/// ([`measured_bleed`]).
pub fn bloom_of(image: &Luma, func: &FunctionMap, corners: [[f64; 2]; 4], anchors: &[Anchor], inverted: bool, params: &SampleParams) -> f64 {
    bloom_given(image, func, corners, anchors, inverted, None, params)
}

/// [`bloom_of`], with the bleed of the first finder among `anchors`
/// given as `first` where it has been measured already
/// ([`first_finder_bleed`]).
fn bloom_given(
    image: &Luma,
    func: &FunctionMap,
    corners: [[f64; 2]; 4],
    anchors: &[Anchor],
    inverted: bool,
    first: Option<f64>,
    params: &SampleParams,
) -> f64 {
    let Some(frame) = BleedFrame::new(func, corners, anchors, inverted, params) else { return 0.0 };
    let finders = anchors.iter().filter(|a| a.kind == AnchorKind::Finder).enumerate();
    symbol_bleed(finders.map(|(k, anchor)| match first {
        Some(bleed) if k == 0 => bleed,
        _ => frame.finder(image, anchor),
    }))
}

/// The bleed the first finder of `found` shows, as [`bloom_of`] measures
/// it: nought where it shows none, and then the symbol shows none
/// either, since a symbol bleeds only where every finder does
/// ([`symbol_bleed`]). For a read that counts the reads it may make
/// before it makes the first ([`super::read`]), and would not measure
/// every finder of a symbol that reads at once.
pub(crate) fn first_finder_bleed(image: &Luma, found: &Detection, params: &SampleParams) -> f64 {
    let func = FunctionMap::new(found.version);
    let Some(first) = found.anchors.iter().find(|a| a.kind == AnchorKind::Finder) else { return 0.0 };
    BleedFrame::new(&func, found.corners, &found.anchors, found.inverted, params).map_or(0.0, |frame| frame.finder(image, first))
}

/// Whether [`bled_placement`] may place `found` again at all: a symbol not
/// yet placed expecting a bleed, with two finders or more to measure one
/// by ([`symbol_bleed`]).
pub(crate) fn may_bleed(found: &Detection) -> bool {
    found.bloom == 0.0 && found.count(AnchorKind::Finder) >= 2
}

/// How far, in the photo's pixels through `map`, the edge of a `w` × `h`
/// symbol at `at`, a point of its outline, lies out from the edge of its
/// ink, on a photo whose boundaries lie `bloom` modules into the dark
/// ([`Detection::bloom`]): out of the symbol across whichever of its
/// edges `at` lies on, both at a corner, by the bleed, which drew the
/// edge of the ink in by as much; in, by the spread of ink.
fn bled_out(map: &Homography, w: f64, h: f64, [x, y]: [f64; 2], bloom: f64) -> [f64; 2] {
    let across = |v: f64, end: f64| if v <= 0.0 { -bloom } else if v >= end { bloom } else { 0.0 };
    let (from, to) = (map.apply(x, y), map.apply(x + across(x, w), y + across(y, h)));
    [to[0] - from[0], to[1] - from[1]]
}

/// A placement as a bleed puts it ([`bled_outline`]).
pub struct BledOutline {
    pub corners: [[f64; 2]; 4],
    pub anchors: Vec<Anchor>,
    /// The corners a finder's own fit placed, as
    /// [`Detection::measured_corners`].
    pub measured_corners: Vec<usize>,
    /// The corners the symbol's edges placed, as
    /// [`Detection::edged_corners`].
    pub edged_corners: Vec<usize>,
}

impl BledOutline {
    /// The corners a refinement holds where they were placed, as
    /// [`Detection::held_corners`].
    pub fn held_corners(&self) -> Vec<usize> {
        let mut held: Vec<usize> = self.measured_corners.iter().chain(&self.edged_corners).copied().collect();
        held.sort_unstable();
        held.dedup();
        held
    }
}

/// A placement fitted as printed, as a bleed of `bloom` modules puts it
/// ([`Detection::bloom`]), placed afresh as the detector first places a
/// symbol ([`align_assignment`]): every anchor refined expecting the
/// bleed ([`refine_anchors`]); each edge point, traced against the quiet
/// zone, moved out by the bleed, which drew the edge of the ink in by as
/// much, or in by the spread of ink, less however far `already` says it
/// was moved so before; the plane fitted through the four
/// corners of each finder found and the centre of every other anchor
/// ([`choose_model`]); a corner a finder stands at placed by that
/// finder's fit, and any other an anchor holds by its two edges, traced
/// expecting the bleed ([`settle_traced_corner`]); and the rest where the
/// plane puts them. Every pattern and edge is looked for light on dark
/// when `inverted`, as the placement has it, the same as the bleed was
/// measured ([`bloom_of`]). Whoever reads the symbol refines it expecting
/// the bleed ([`super::refine::refine_corners_expecting`]), holding the
/// corners a finder or the edges placed ([`BledOutline::held_corners`])
/// as the detector's are held. `None` when the corners given form no
/// usable outline, or the anchors no map or no plane.
///
/// A corner is placed afresh, and not moved by how it was placed, since
/// neither the corner nor its placement says how far the bleed put it
/// off. A corner a finder's fit placed was off as far as that finder
/// when it was placed, but before a read places its symbol again the
/// anchors have been refitted as printed: on a photo of a version 2
/// symbol whose dark had thinned by three tenths of a module, the
/// top-left corner lay 0.39 of a module off where its finder lay 0.40
/// off another way, and moved with the finder it came to lie 0.56 off.
/// A corner Refine placed says nothing of how it was placed at all, and
/// a refinement leaves the corners the anchors hold where they were
/// given: of another such symbol, two corners held so stayed four
/// tenths off. Nor will a plane through the centres of the anchors alone
/// do, for a symbol with no alignment pattern: a version 1 symbol's
/// finders and timing modules all lie about one corner of it, and a
/// perspective through their centres, with the top-left finder's a fifth
/// of a module out and a timing module's more, put the top-left corner
/// 0.59 of a module off, which the refinement then held there, and the
/// far one two modules off. Left free once placed so, the corners the
/// finders placed were walked off them: on the same symbol the descent
/// took the bottom-left corner from 0.06 of a module off to 0.18.
///
/// `already[k]` is how far, in modules, `anchors[k]` was moved out for a
/// bleed before, as a placement made expecting one moves its edge points;
/// an anchor with no entry was not. An edge point is never looked for
/// again, only moved, and one already moved would be moved out again by
/// the whole bleed: Refine pressed twice on a version 2 symbol whose
/// dark had thinned by three tenths of a module, with edge points
/// dropped by hand on the edge of its ink, kept the fit as printed the
/// second time, the finders four tenths of a module off, since the
/// placement through edge points moved twice read worse; and the third
/// time placed the symbol through them, a quarter of a module out.
#[allow(clippy::too_many_arguments)]
pub fn bled_outline(
    image: &Luma,
    func: &FunctionMap,
    corners: [[f64; 2]; 4],
    anchors: &[Anchor],
    bloom: f64,
    inverted: bool,
    already: &[f64],
    params: &SampleParams,
) -> Option<BledOutline> {
    // Twice over: each pattern is looked for through the map the others
    // make, and the first time they are where the fit as printed put
    // them, each as far off as the bleed. The top-left finder of a version
    // 1 symbol whose dark had thinned by three tenths of a module stopped
    // a fifth of a module short of its pattern so, and came to within a
    // fiftieth looked for again.
    let first: Vec<Anchor> =
        refine_anchors_paired(image, func, corners, anchors, bloom, Some(inverted), params)?.into_iter().map(|(anchor, _)| anchor).collect();
    let paired = refine_anchors_paired(image, func, corners, &first, bloom, Some(inverted), params)?;
    let (w, h) = (func.width as f64, func.height as f64);
    let printed = Homography::rect_to_quad(w, h, corners)?;
    let mut bled: Vec<Anchor> = paired.iter().map(|(anchor, _)| *anchor).collect();
    for (k, anchor) in bled.iter_mut().enumerate().filter(|(_, a)| a.kind == AnchorKind::Edge) {
        let moved = already.get(k).copied().unwrap_or(0.0);
        let [dx, dy] = bled_out(&printed, w, h, anchor.at, bloom - moved);
        anchor.center = [anchor.center[0] + dx, anchor.center[1] + dy];
    }
    // An edge point holds the map across the edge alone, and says nothing
    // of where along it a module lies.
    let planar: Vec<&(Anchor, Vec<Pair>)> = paired.iter().filter(|(anchor, _)| anchor.kind != AnchorKind::Edge).collect();
    let pairs: Vec<Pair> = planar.iter().flat_map(|(_, pairs)| pairs.iter().copied()).collect();
    let model = choose_model(&planar.iter().map(|(anchor, _)| *anchor).collect::<Vec<_>>(), pairs.len());
    let plane = Homography::fit(model, &pairs)?;
    let outline = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
    let mut placed = outline.map(|[x, y]| plane.apply(x, y));
    let mut measured_corners = Vec::new();
    for (i, corner) in outline.iter().enumerate() {
        let at_corner = planar
            .iter()
            .filter(|(anchor, _)| anchor.kind == AnchorKind::Finder)
            .flat_map(|(_, pairs)| pairs.iter())
            .find(|(at, _)| distance(*at, *corner) < 1e-9);
        if let Some(&(_, px)) = at_corner {
            placed[i] = px;
            measured_corners.push(i);
        }
    }
    let (sign, threshold) = (if inverted { -1.0 } else { 1.0 }, fixed_threshold(params));
    let fourth = matches!(func.version, SymbolVersion::Qr(_) | SymbolVersion::Model1(_)) && measured_corners.len() == 3;
    let anchored: Vec<[f64; 2]> = bled.iter().map(|a| a.at).collect();
    let loose = loose_corners(func, &anchored);
    let mut edged_corners = Vec::new();
    for i in (0..4).filter(|i| !measured_corners.contains(i) && !loose.contains(i)) {
        if settle_traced_corner(image, func, &mut placed, i, sign, threshold, &[], fourth, bloom) {
            edged_corners.push(i);
        }
    }
    Some(BledOutline { corners: placed, anchors: bled, measured_corners, edged_corners })
}

/// A symbol placed as printed placed again expecting the bleed its finders
/// show ([`bloom_of`]): its anchors refined expecting it and its corners
/// placed afresh by them ([`bled_outline`]), and the timing modules looked
/// for afresh through the map the rest now make, within
/// [`BLED_TIMING_WINDOW`], with any that breaks the trend of its line
/// dropped, as the detector drops one ([`timing_line_strays`]). Refined
/// from where the placement as printed had them, a third of a module off
/// on a photo whose dark had thinned by a third, not all of them came
/// back. The corners a finder's fit or the symbol's edges placed expecting
/// the bleed are held as the detector's are ([`Detection::held_corners`]),
/// and the read refines the rest expecting it. The bleed of `found`'s
/// first finder is given as `first`, measured already
/// ([`first_finder_bleed`]) and not measured again. `None` when it is
/// nought, the finders show no bleed alike, or they are too few to show
/// one ([`may_bleed`]).
pub(crate) fn bled_placement(image: &Luma, found: &Detection, first: f64, params: &SampleParams) -> Option<Detection> {
    if !may_bleed(found) || first == 0.0 {
        return None;
    }
    let func = FunctionMap::new(found.version);
    // Light on dark or not as the detection settled it, for the bleed
    // and every pattern looked for again alike.
    let bloom = bloom_given(image, &func, found.corners, &found.anchors, found.inverted, Some(first), params);
    if bloom == 0.0 {
        return None;
    }
    // A placement not yet made expecting a bleed ([`may_bleed`]) has moved
    // no edge point for one.
    let BledOutline { corners, mut anchors, measured_corners, edged_corners } =
        bled_outline(image, &func, found.corners, &found.anchors, bloom, found.inverted, &[], params)?;
    if anchors.iter().any(|a| a.kind == AnchorKind::Timing) {
        anchors.retain(|a| a.kind != AnchorKind::Timing);
        let controls: Vec<ControlPoint> = anchors.iter().map(Anchor::control_point).collect();
        let map = Projection::new(func.width, func.height, corners, &controls)?;
        let threshold = fixed_threshold(params);
        let timing: Vec<Site> = sites(&func).into_iter().filter(|site| site.kind == AnchorKind::Timing).collect();
        anchors.extend(timing.iter().filter_map(|site| search_site(image, &func, site, &map, BLED_TIMING_WINDOW, found.inverted, threshold, bloom).map(|(anchor, _)| anchor)));
        // Looked for all at once, through one map, the timing modules
        // are not checked against each other as the detector's search
        // checks them, one at a time through the map the rest make; one
        // off the bend its line-mates trace has read a row of data or a
        // smudge.
        let (w, h) = (func.width as f64, func.height as f64);
        let pairs: Vec<Pair> = anchors.iter().map(|a| (a.at, a.center)).chain([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]].into_iter().zip(corners)).collect();
        if let Some(base) = Homography::fit(Model::Projective, &pairs) {
            let mut strays: Vec<usize> = timing_line_strays(&anchors, &base).into_iter().map(|(i, _)| i).collect();
            strays.sort_unstable_by(|a, b| b.cmp(a));
            strays.dedup();
            for i in strays {
                anchors.remove(i);
            }
        }
    }
    Some(Detection { corners, anchors, bloom, measured_corners, edged_corners, ..found.clone() })
}

/// Move each anchor to where its own pattern reads best in the photo,
/// through the outline given by `corners` and the other anchors, by at
/// most [`REFINE_LIMIT`] modules, expecting its boundaries `bloom`
/// modules into the dark ([`Detection::bloom`]). Anchors whose pattern
/// cannot be found are returned as they were.
pub fn refine_anchors(
    image: &Luma,
    func: &FunctionMap,
    corners: [[f64; 2]; 4],
    anchors: &[Anchor],
    bloom: f64,
    params: &SampleParams,
) -> Vec<Anchor> {
    refine_anchors_with(image, func, corners, anchors, bloom, None, params)
}

/// [`refine_anchors`], each pattern looked for light on dark when
/// `inverted` says so, and where it is `None` as the fixed patterns read
/// best through the map: a bleed is into the symbol's dark, and one
/// measured at one polarity is the opposite bleed at the other
/// ([`Detection::held_polarity`]).
pub fn refine_anchors_with(
    image: &Luma,
    func: &FunctionMap,
    corners: [[f64; 2]; 4],
    anchors: &[Anchor],
    bloom: f64,
    inverted: Option<bool>,
    params: &SampleParams,
) -> Vec<Anchor> {
    refine_anchors_paired(image, func, corners, anchors, bloom, inverted, params)
        .map_or_else(|| anchors.to_vec(), |paired| paired.into_iter().map(|(anchor, _)| anchor).collect())
}

/// [`refine_anchors`], each anchor with the correspondences it gives a
/// fit, as [`search_site`] gives them: the four corners of a finder
/// found, which its template pins down, and the centre alone of any other
/// pattern, or of an anchor not found or put where the trend of its line
/// has it. Each pattern is looked for light on dark when `inverted` says
/// so, and where it is `None` as the fixed patterns read best through the
/// map. `None` when the corners and anchors make no map.
fn refine_anchors_paired(
    image: &Luma,
    func: &FunctionMap,
    corners: [[f64; 2]; 4],
    anchors: &[Anchor],
    bloom: f64,
    inverted: Option<bool>,
    params: &SampleParams,
) -> Option<Vec<(Anchor, Vec<Pair>)>> {
    let as_given = |anchor: &Anchor| (*anchor, vec![(anchor.at, anchor.center)]);
    let threshold = fixed_threshold(params);
    let controls: Vec<ControlPoint> = anchors.iter().map(Anchor::control_point).collect();
    let map = Projection::new(func.width, func.height, corners, &controls)?;
    let inverted = inverted.unwrap_or_else(|| {
        let scorer = Scorer::new(image, &Template::whole(func), corners, threshold);
        let (plain, flipped) = (scorer.agreement_known(&map, false), scorer.agreement_known(&map, true));
        flipped.map(|a| a.0) > plain.map(|a| a.0)
    });
    // Each anchor through the map the others make as they were given, so
    // that none waits on another: side by side on the cores to spare.
    let numbered: Vec<(usize, &Anchor)> = anchors.iter().enumerate().collect();
    let mut refined: Vec<(Anchor, Vec<Pair>)> = work::map(&numbered, |&(k, anchor)| {
        // Called off, an anchor not yet looked for stays where it is.
        if work::cancelled() {
            return as_given(anchor);
        }
        let Some(site) = site_of(func, anchor) else { return as_given(anchor) };
        // Search a small window around where the anchor already is,
        // through the map the other anchors make: the map through
        // this one too passes through wherever it was put, and an
        // anchor dropped by hand half a module off bends it there,
        // which an anchor two modules away then holds against.
        let others: Vec<ControlPoint> = controls.iter().enumerate().filter(|(i, _)| *i != k).map(|(_, c)| *c).collect();
        let Some(map) = Projection::new(func.width, func.height, corners, &others) else { return as_given(anchor) };
        let shifted = |x: f64, y: f64| {
            let [px, py] = map.apply(x, y);
            let [ax, ay] = map.apply(anchor.at[0], anchor.at[1]);
            [px + anchor.center[0] - ax, py + anchor.center[1] - ay]
        };
        let local = ShiftedMap { f: &shifted };
        // A timing anchor placed as printed on a photo that has bled lies
        // as far as the bleed off its pattern besides.
        let limit = if anchor.kind == AnchorKind::Timing { TIMING_REFINE_LIMIT + bloom.abs() } else { REFINE_LIMIT };
        match search_site(image, func, &site, &local, SEARCH_STEP, inverted, threshold, bloom) {
            Some((found, pairs)) if distance(found.center, anchor.center) <= limit * module_size_at(&map, anchor.at) => (found, pairs),
            _ => as_given(anchor),
        }
    });
    // A timing anchor that still breaks the trend of its line is put
    // where the trend has it: that is what its neighbours say of the
    // paper there, and its own reading has gone astray.
    let anchors: Vec<Anchor> = refined.iter().map(|(anchor, _)| *anchor).collect();
    let pairs: Vec<Pair> = anchors.iter().map(|a| (a.at, a.center)).collect();
    let corner_pairs: Vec<Pair> = [[0.0, 0.0], [func.width as f64, 0.0], [func.width as f64, func.height as f64], [0.0, func.height as f64]]
        .iter()
        .zip(corners)
        .map(|(at, px)| (*at, px))
        .collect();
    if let Some(base) = Homography::fit(Model::Projective, &pairs.iter().chain(&corner_pairs).copied().collect::<Vec<_>>()) {
        for (i, position) in timing_line_strays(&anchors, &base) {
            refined[i] = as_given(&Anchor { center: position, ..anchors[i] });
        }
    }
    Some(refined)
}

/// A map given as a closure.
struct ShiftedMap<'a> {
    f: &'a (dyn Fn(f64, f64) -> [f64; 2] + Sync),
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
    fn the_readings_scanned_down_the_columns_are_the_readings_turned_over_their_diagonal() {
        // Squares of ink, with the margin of the frame off the photo.
        let (width, height) = (131, 67);
        let image = Luma::new(width, height, (0..width * height).map(|i| if (i % width / 5 + i / width / 3) % 2 == 0 { 0.1 } else { 0.9 }).collect());
        let rect = Rectified::new(&image, [[0.0, 0.0], [width as f64, 0.0], [width as f64, height as f64], [0.0, height as f64]]).unwrap();
        let (w, h) = (rect.frame.width, rect.frame.height);
        let readings = Readings::of(&rect, None, true);
        // The two thresholds and the windows.
        assert!(readings.scanned().len() > 2);
        for (mask, turned) in readings.scanned() {
            for (x, y) in (0..h).flat_map(|y| (0..w).map(move |x| (x, y))) {
                assert_eq!(turned.kind(x * h + y), mask.kind(y * w + x), "({x}, {y}) of {w} by {h}");
                assert_eq!(dark_in(turned, h, w, y as i64, x as i64), dark_in(mask, w, h, x as i64, y as i64));
            }
        }
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
            regioned: false,
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

    #[test]
    fn an_alignment_pattern_is_looked_for_where_its_neighbours_carry_it() {
        // A version 7 symbol ten pixels a module, with its finders where
        // the plane has them.
        let func = FunctionMap::new(SymbolVersion::Qr(7));
        let grid = AlignmentGrid::of(&func).unwrap();
        let plane = Homography::rect_to_quad(45.0, 45.0, [[0.0, 0.0], [450.0, 0.0], [450.0, 450.0], [0.0, 450.0]]).unwrap();
        let sites = sites(&func);
        let site = |x: f64, y: f64| *sites.iter().find(|s| s.kind == AnchorKind::Alignment && s.at == [x, y]).unwrap();
        // An anchor found `off` modules from where the plane puts it.
        let stand = |kind: AnchorKind, at: [f64; 2], off: [f64; 2], firm: bool| {
            let center = [10.0 * (at[0] + off[0]), 10.0 * (at[1] + off[1])];
            Placed { anchor: Anchor { kind, at, center, partial: false, inferred: false }, pairs: vec![(at, center)], firm }
        };
        let mut placed: Vec<Placed> = [[3.5, 3.5], [41.5, 3.5], [3.5, 41.5]].map(|at| stand(AnchorKind::Finder, at, [0.0, 0.0], false)).into();
        // With nothing but finders beside a pattern there is nothing to
        // carry: they stand where the plane has them.
        assert_eq!(carried_offset(&grid, &site(22.5, 6.5), &plane, &placed), None);
        // The middle pattern is carried by the two beside it that stand.
        placed.push(stand(AnchorKind::Alignment, [22.5, 6.5], [1.0, 0.2], true));
        placed.push(stand(AnchorKind::Alignment, [6.5, 22.5], [0.0, 0.6], true));
        let carried = carried_offset(&grid, &site(22.5, 22.5), &plane, &placed).unwrap();
        assert!((carried[0] - 0.5).abs() < 1e-9 && (carried[1] - 0.4).abs() < 1e-9, "{carried:?}");
        // A pattern placed by its rings alone carries nothing on.
        placed.push(stand(AnchorKind::Alignment, [22.5, 22.5], [3.0, 3.0], false));
        assert_eq!(carried_offset(&grid, &site(38.5, 22.5), &plane, &placed), None);
        // Standing firm it does, with the finder in the corner above,
        // which stands where the plane has it.
        placed.last_mut().unwrap().firm = true;
        let carried = carried_offset(&grid, &site(38.5, 22.5), &plane, &placed).unwrap();
        assert!((carried[0] - 1.5).abs() < 1e-9 && (carried[1] - 1.5).abs() < 1e-9, "{carried:?}");
        // The offsets are in the symbol's own modules whatever the plane.
        let turned = Homography::rect_to_quad(45.0, 45.0, [[450.0, 0.0], [450.0, 900.0], [0.0, 900.0], [0.0, 0.0]]).unwrap();
        let [x, y] = turned.apply(22.5 + 1.0, 6.5 - 2.0);
        let offset = offset_in_modules(&turned, [22.5, 6.5], [x, y]).unwrap();
        assert!((offset[0] - 1.0).abs() < 1e-9 && (offset[1] + 2.0).abs() < 1e-9, "{offset:?}");
        // Versions with one alignment pattern have a grid of it and the
        // finders; version 1 and the other families have none.
        assert_eq!(AlignmentGrid::of(&FunctionMap::new(SymbolVersion::Qr(2))).map(|g| g.centers), Some(vec![6, 18]));
        assert!(AlignmentGrid::of(&FunctionMap::new(SymbolVersion::Qr(1))).is_none());
        assert!(AlignmentGrid::of(&FunctionMap::new(SymbolVersion::Rmqr(11))).is_none());
    }

    #[test]
    fn an_alignment_pattern_that_stands_firm_is_not_dropped_for_lying_off_the_plane() {
        // A version 7 symbol ten pixels a module, its three finders where
        // the plane has them and its middle alignment pattern two modules
        // along the row from there, as on a bottle.
        let finder = |x0: f64, y0: f64| Placed {
            anchor: Anchor { kind: AnchorKind::Finder, at: [x0 + 3.5, y0 + 3.5], center: [10.0 * (x0 + 3.5), 10.0 * (y0 + 3.5)], partial: false, inferred: false },
            pairs: [[0.0, 0.0], [7.0, 0.0], [7.0, 7.0], [0.0, 7.0]].iter().map(|[dx, dy]| ([x0 + dx, y0 + dy], [10.0 * (x0 + dx), 10.0 * (y0 + dy)])).collect(),
            firm: false,
        };
        let pattern = |firm: bool| Placed {
            anchor: Anchor { kind: AnchorKind::Alignment, at: [22.5, 22.5], center: [245.0, 225.0], partial: false, inferred: false },
            pairs: vec![([22.5, 22.5], [245.0, 225.0])],
            firm,
        };
        // Placed by its rings alone, the plane through the finders
        // condemns it.
        let mut placed = vec![finder(0.0, 0.0), finder(38.0, 0.0), finder(0.0, 38.0), pattern(false)];
        fit_anchors(&mut placed, None).unwrap();
        assert_eq!(placed.iter().filter(|p| p.anchor.kind == AnchorKind::Alignment).count(), 0);
        // Its template read whole, it stands, off the plane, and the
        // pattern beside it is looked for as far off the plane as it was
        // found.
        let func = FunctionMap::new(SymbolVersion::Qr(7));
        let grid = AlignmentGrid::of(&func).unwrap();
        let mut placed = vec![finder(0.0, 0.0), finder(38.0, 0.0), finder(0.0, 38.0), pattern(true)];
        let (plane, _) = fit_anchors(&mut placed, Some(&grid)).unwrap();
        assert_eq!(placed.iter().filter(|p| p.anchor.kind == AnchorKind::Alignment).count(), 1);
        let beside = *sites(&func).iter().find(|s| s.kind == AnchorKind::Alignment && s.at == [22.5, 38.5]).unwrap();
        let carried = carried_offset(&grid, &beside, &plane, &placed).unwrap();
        let found = offset_in_modules(&plane, [22.5, 22.5], [245.0, 225.0]).unwrap();
        assert!((found[0] - 2.0).abs() < 1e-6 && placed.iter().any(|p| p.firm && p.pairs.is_empty()), "the pattern was left in the plane: {found:?}");
        // The mean of what the pattern above was found off by and what
        // the finder in the corner beside this one was, next to nothing.
        let corner = offset_in_modules(&plane, [3.5, 41.5], [35.0, 415.0]).unwrap();
        assert!(corner[0].abs() < 0.2 && corner[1].abs() < 0.2, "{corner:?}");
        for axis in 0..2 {
            assert!((carried[axis] - (found[axis] + corner[axis]) / 2.0).abs() < 1e-9, "{carried:?}");
        }
        // With the patterns beside it standing where the plane has them,
        // as on flat paper, it is a patch of data two modules from where
        // a pattern should be, and is dropped as any anchor would be;
        // with them carried as far as it is, as round a bottle, it stays.
        let beside = |off: f64| -> Vec<Placed> {
            [[6.5, 22.5], [38.5, 22.5], [22.5, 6.5], [22.5, 38.5]]
                .iter()
                .map(|&at| {
                    let center = [10.0 * (at[0] + off), 10.0 * at[1]];
                    Placed { anchor: Anchor { kind: AnchorKind::Alignment, at, center, partial: false, inferred: false }, pairs: vec![(at, center)], firm: true }
                })
                .collect()
        };
        let stands = |placed: &[Placed]| placed.iter().any(|p| p.anchor.at == [22.5, 22.5]);
        let mut flat = vec![finder(0.0, 0.0), finder(38.0, 0.0), finder(0.0, 38.0), pattern(true)];
        flat.extend(beside(0.0));
        fit_anchors(&mut flat, Some(&grid)).unwrap();
        assert!(!stands(&flat) && flat.len() == 7, "the misleading pattern was kept");
        let mut curled = vec![finder(0.0, 0.0), finder(38.0, 0.0), finder(0.0, 38.0), pattern(true)];
        curled.extend(beside(1.6));
        fit_anchors(&mut curled, Some(&grid)).unwrap();
        assert!(stands(&curled) && curled.len() == 8, "a pattern carried as its neighbours are was dropped");
    }

    #[test]
    fn the_alignment_patterns_of_a_page_lifted_at_one_corner_are_walked_to() {
        // A version 20 symbol four pixels a module on a page lifted at its
        // top-right corner, ten modules there and nothing along the left
        // and bottom edges. Each column of the photo shows the rows moved
        // up by the cube of how far across it lies, less toward the bottom.
        use crate::symbol::grid::ModuleGrid;
        let symbol = qrcode_generator::qr::Encoder::new(qrcode_generator::qr::ErrorCorrection::Medium)
            .version(qrcode_generator::qr::Version::new(20).unwrap())
            .boost_error_correction(false)
            .encode_segments(&[qrcode_generator::Segment::bytes("the patterns of a bent page, one after another")])
            .unwrap();
        let grid = ModuleGrid::from_bools(symbol.width(), symbol.height(), symbol.modules());
        let func = FunctionMap::new(SymbolVersion::Qr(20));
        let (module, lifted, w) = (4.0, 10.0, func.width as f64);
        let (left, top) = (20.0, 20.0 + module * lifted);
        let lift = |x: f64| lifted * (x / w).powi(3);
        let to_symbol = |[px, py]: [f64; 2]| {
            let (u, v) = ((px - left) / module, (py - top) / module);
            [u, v + lift(u) * (1.0 - v / w)]
        };
        let to_photo = |[x, y]: [f64; 2]| [left + module * x, top + module * (y - lift(x)) / (1.0 - lift(x) / w)];
        let size = ((w * module + 2.0 * left) as usize, (w * module + left + top) as usize);
        let mut data = Vec::with_capacity(size.0 * size.1);
        for py in 0..size.1 {
            for px in 0..size.0 {
                let mut sum = 0.0;
                for sy in 0..3 {
                    for sx in 0..3 {
                        let [mx, my] = to_symbol([px as f64 + (sx as f64 + 0.5) / 3.0, py as f64 + (sy as f64 + 0.5) / 3.0]);
                        let inside = mx >= 0.0 && my >= 0.0 && mx < w && my < w;
                        sum += if inside && grid.get(mx as usize, my as usize) == crate::symbol::grid::Module::Dark { 0.08f32 } else { 0.92 };
                    }
                }
                data.push(sum / 9.0);
            }
        }
        let image = Luma::new(size.0, size.1, data);
        // The three finders, each placed by its capstone's corners.
        let finder = |x0: f64, y0: f64| Placed {
            anchor: Anchor { kind: AnchorKind::Finder, at: [x0 + 3.5, y0 + 3.5], center: to_photo([x0 + 3.5, y0 + 3.5]), partial: false, inferred: false },
            pairs: [[0.0, 0.0], [7.0, 0.0], [7.0, 7.0], [0.0, 7.0]].iter().map(|[dx, dy]| ([x0 + dx, y0 + dy], to_photo([x0 + dx, y0 + dy]))).collect(),
            firm: false,
        };
        let placed = vec![finder(0.0, 0.0), finder(w - 7.0, 0.0), finder(0.0, w - 7.0)];
        let alignment = AlignmentGrid::of(&func).unwrap();
        let sites = sites(&func);
        let patterns: Vec<[f64; 2]> = sites.iter().filter(|s| s.kind == AnchorKind::Alignment).map(|s| s.at).collect();
        // The plane through the finders runs some pattern further off
        // than the search looks.
        let plane = Homography::fit(Model::Projective, &pairs_of(&placed)).unwrap();
        let plane_off = patterns.iter().map(|&at| distance(to_symbol(plane.apply(at[0], at[1])), at)).fold(0.0, f64::max);
        assert!(plane_off > WINDOW_PROJECTIVE, "the plane puts every pattern within {plane_off:.2} modules");
        // The walk finds every one, each where it lies.
        let (walked, missed) = walk_alignment(&image, &func, &alignment, &sites, &placed, false, None, None).expect("a walk");
        assert!(missed.is_empty(), "{} patterns missed", missed.len());
        let found: Vec<&Placed> = walked.iter().filter(|p| p.anchor.kind == AnchorKind::Alignment).collect();
        assert_eq!(found.len(), patterns.len());
        for p in found {
            let off = distance(to_symbol(p.anchor.center), p.anchor.at);
            assert!(p.firm && off < 0.25, "the pattern at {:?} was found {off:.2} modules off", p.anchor.at);
        }
        // Where the search had already found them all firmly, the walk
        // holds them as they are and has nothing to add.
        let found_already = |firm: bool, shift: f64| {
            let mut placed = vec![finder(0.0, 0.0), finder(w - 7.0, 0.0), finder(0.0, w - 7.0)];
            placed.extend(walked.iter().filter(|p| p.anchor.kind == AnchorKind::Alignment).map(|p| Placed {
                anchor: Anchor { center: [p.anchor.center[0] + shift * module, p.anchor.center[1]], ..p.anchor },
                pairs: p.pairs.clone(),
                firm,
            }));
            placed
        };
        assert!(walk_alignment(&image, &func, &alignment, &sites, &found_already(true, 0.0), false, None, None).is_none());
        // Where it had found them all where they lie, but none firmly,
        // the walk finds each again, firmly, where it was: the same
        // placement, which it does not offer. Found a module off, the
        // walk is another placement, and is offered.
        assert!(walk_alignment(&image, &func, &alignment, &sites, &found_already(false, 0.0), false, None, None).is_none());
        assert!(walk_alignment(&image, &func, &alignment, &sites, &found_already(false, 1.0), false, None, None).is_some());
        // The timing patterns' modules, which the walk does not look for,
        // looked for again where its placement puts them, as a placement
        // walked has them ([`align_assignment`]): three in four of them
        // found, each where it lies, those along the lifted top row among
        // them. Grown from the finders, as the search grows its own, the
        // plane through them finds not one along that row.
        let copy = |placed: &[Placed]| placed.iter().map(|p| Placed { anchor: p.anchor, pairs: p.pairs.clone(), firm: p.firm }).collect::<Vec<_>>();
        let timing: Vec<usize> = (0..sites.len()).filter(|&i| sites[i].kind == AnchorKind::Timing).collect();
        let (mut grown, mut gone) = (copy(&walked), Vec::new());
        grow_from(&image, &func, &sites, &mut grown, timing.clone(), Some(&alignment), None, false, None, &mut gone).expect("the walk's placement fits");
        let found: Vec<&Placed> = grown.iter().filter(|p| p.anchor.kind == AnchorKind::Timing).collect();
        assert!(4 * found.len() >= 3 * timing.len(), "{} of {} timing modules found", found.len(), timing.len());
        for p in &found {
            let off = distance(to_symbol(p.anchor.center), p.anchor.at);
            assert!(off < 0.25, "the timing module at {:?} was found {off:.2} modules off", p.anchor.at);
        }
        assert!(found.iter().any(|p| p.anchor.at[1] == 6.5), "none found along the top row");
        let rest: Vec<usize> = (0..sites.len()).filter(|&i| sites[i].kind != AnchorKind::Finder).collect();
        let (mut searched, mut lost) = (copy(&placed), Vec::new());
        grow_from(&image, &func, &sites, &mut searched, rest, Some(&alignment), None, false, None, &mut lost).expect("the finders fit");
        assert!(!searched.iter().any(|p| p.anchor.kind == AnchorKind::Timing && p.anchor.at[1] == 6.5), "the plane found the top row");
    }

    /// A version 1 symbol's fixed patterns with data made up for it: no
    /// message, but dark and light modules along its edges as a symbol
    /// has them.
    fn made_up_symbol() -> ModuleGrid {
        use crate::symbol::grid::Module;
        let func = FunctionMap::new(SymbolVersion::Qr(1));
        let mut grid = ModuleGrid::new(func.width, func.height);
        let mut state = 0x2545_f491u32;
        for y in 0..func.height {
            for x in 0..func.width {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let dark = func.expected(x, y).unwrap_or(state >> 31 == 1);
                grid.set(x, y, if dark { Module::Dark } else { Module::Light });
            }
        }
        grid
    }

    #[test]
    fn a_row_of_modules_inside_the_symbol_is_not_traced_as_its_edge() {
        // A version 1 symbol eight pixels a module, upright, its corners
        // known but the bottom-right, which is put a module up the right
        // edge: the outline's bottom edge then runs from the symbol's own
        // at the left to a row inside it at the right.
        let grid = made_up_symbol();
        let func = FunctionMap::new(SymbolVersion::Qr(1));
        let image = draw_grid(&grid, [140.0, 130.0], 8.0, 0.0, (280, 260));
        let truth = [[56.0, 46.0], [224.0, 46.0], [224.0, 214.0], [56.0, 214.0]];
        let mut corners = truth;
        corners[2] = [224.0, 206.0];
        let map = Homography::rect_to_quad(21.0, 21.0, corners).unwrap();
        let (level, contrast) = edge_levels(&image, &func, &map, None);
        let [bottom, right] = edges_at(2, 21.0, 21.0);
        // The profiles towards the corner find the boundary between the
        // border row and the row within, and a line fits those with the
        // true crossings near the finder; the dark modules of the border
        // beyond that line give it away. Or the profiles that found
        // nothing, taken again about where the true crossings run, find
        // the edge itself, and the line is the edge's.
        let slanted = trace_edge(&image, &map, &bottom, 1.0, level, contrast, true);
        assert!(
            slanted.is_none_or(|traced| !traced.clear || (traced.line.distance([80.0, 214.0]) < 0.8 && traced.line.distance([210.0, 214.0]) < 0.8)),
            "a line a row inside the symbol was taken to be clear of it"
        );
        // The right edge, which the outline has, traces clear, and so do
        // both from the true corner, on their own lines.
        let traced = trace_edge(&image, &map, &right, 1.0, level, contrast, true).expect("the right edge");
        assert!(traced.clear && !traced.bent);
        assert!(traced.line.distance([224.0, 100.0]) < 0.8 && traced.line.distance([224.0, 200.0]) < 0.8, "{:?}", traced.line);
        let flat = Homography::rect_to_quad(21.0, 21.0, truth).unwrap();
        let traced = trace_edge(&image, &flat, &bottom, 1.0, level, contrast, true).expect("the bottom edge");
        assert!(traced.clear && !traced.bent);
        assert!(traced.line.distance([100.0, 214.0]) < 0.8 && traced.line.distance([200.0, 214.0]) < 0.8, "{:?}", traced.line);
        // The crossings kept lie on the fitted line, each filed under
        // the point of the edge it was traced from.
        assert!(traced.crossings.len() >= MIN_CROSSINGS);
        for crossing in &traced.crossings {
            assert!(crossing.at[1] == 21.0 && traced.line.distance(crossing.px) < 1e-9, "{crossing:?}");
        }
        // From the corner a module inside, the edges do not both trace
        // clear; from a module further out they do, and place the corner.
        let p = traced_corner(&image, &func, &corners, 2, 1.0, None, &[], true, f64::INFINITY).expect("the corner was not traced from a module out");
        assert!(distance(p, truth[2]) < 0.2 * 8.0, "{p:?}");
        // So they do from another start given, taken before those.
        let p = traced_corner(&image, &func, &corners, 2, 1.0, None, &[[224.5, 213.0]], true, f64::INFINITY).expect("the corner was not traced from the start given");
        assert!(distance(p, truth[2]) < 0.2 * 8.0, "{p:?}");
        // A start that the corner does not come out beside is not
        // believed: the edges were run on to it from their far ends.
        let mut far = truth;
        far[2] = [246.0, 236.0];
        assert!(traced_corner(&image, &func, &far, 2, 1.0, None, &[], true, MAX_EDGE_SAGITTA).is_none_or(|p| distance(p, truth[2]) < 0.2 * 8.0));
        // A corner that is not the fourth of a QR Code's stands where it
        // traced from where it was put, clear or not: which, with the
        // profiles towards the corner taken again about the edge the
        // rest of it found, is the corner.
        let own = traced_corner(&image, &func, &corners, 2, 1.0, None, &[], false, f64::INFINITY).expect("the corner was not traced from where it was put");
        assert!(distance(own, truth[2]) < 0.2 * 8.0, "{own:?}");
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
            regioned: false,
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
        // Turned forty degrees against its partner, the line leaves it
        // nearer a corner than the middle of a side: not one symbol's
        // finders.
        assert!(!finders_face(&a, &square_capstone([300.0, 100.0], 8.0, 40.0, &frame), &frame));
        // Across the diagonal the line leaves each square by a corner.
        assert!(!finders_face(&a, &square_capstone([300.0, 300.0], 8.0, 0.0, &frame), &frame));
        // Far enough off the row that the line crosses near a corner.
        assert!(!finders_face(&a, &square_capstone([300.0, 250.0], 8.0, 0.0, &frame), &frame));
        // Corners never measured say nothing against a partner.
        let unmeasured = Capstone { measured: false, ..square_capstone([300.0, 100.0], 8.0, 40.0, &frame) };
        assert!(finders_face(&a, &unmeasured, &frame));
    }

    #[test]
    fn the_finders_of_a_symbol_seen_steeply_face_each_other() {
        // A symbol sheared until its columns run forty-five degrees off
        // square to its rows, as one seen steeply along its diagonal is:
        // each finder is a rhombus, and the line between two of them
        // along a row meets the sides it crosses forty-five degrees off
        // square, through their middles. The sides crossed are parallel,
        // which is all that is asked of the angle.
        let frame = Frame::new([[0.0, 0.0], [800.0, 0.0], [800.0, 800.0], [0.0, 800.0]], MAX_RECTIFIED).unwrap();
        let (module, lean) = (8.0, std::f64::consts::FRAC_1_SQRT_2);
        // A point of the symbol, in modules along its row and its column.
        let at = |[across, down]: [f64; 2]| [100.0 + module * (across + lean * down), 100.0 + module * lean * down];
        let sheared = |[across, down]: [f64; 2]| Capstone {
            corners: order_in_frame([[-3.5, -3.5], [3.5, -3.5], [3.5, 3.5], [-3.5, 3.5]].map(|[dx, dy]| at([across + dx, down + dy])), &frame),
            ..square_capstone(at([across, down]), module, 0.0, &frame)
        };
        let (top_left, top_right, bottom_left) = (sheared([3.5, 3.5]), sheared([21.5, 3.5]), sheared([3.5, 21.5]));
        assert!(finders_face(&top_left, &top_right, &frame));
        assert!(finders_face(&top_left, &bottom_left, &frame));
        // The two at the ends of the diagonal still do not: the line
        // between them leaves each by a corner.
        assert!(!finders_face(&top_right, &bottom_left, &frame));
        // A square finder beside a sheared one is no partner of it once
        // their sides lie past forty-five degrees apart.
        let steeper = |[across, down]: [f64; 2]| {
            let at = |[across, down]: [f64; 2]| [100.0 + module * (across + 1.2 * down), 100.0 + module * 0.6 * down];
            Capstone {
                corners: order_in_frame([[-3.5, -3.5], [3.5, -3.5], [3.5, 3.5], [-3.5, 3.5]].map(|[dx, dy]| at([across + dx, down + dy])), &frame),
                ..square_capstone(at([across, down]), module, 0.0, &frame)
            }
        };
        let square = square_capstone([400.0, steeper([3.5, 3.5]).center[1]], module, 0.0, &frame);
        assert!(!finders_face(&steeper([3.5, 3.5]), &square, &frame));
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
            regioned: false,
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
    fn what_the_symbol_was_found_with_is_said_in_full() {
        let mut found = Detection {
            corners: [[0.0; 2]; 4],
            version: SymbolVersion::Qr(1),
            inverted: false,
            mirrored: false,
            mirrored_by_decode: false,
            turns: 0,
            anchors: Vec::new(),
            agreement: (0, 0),
            info: InfoEvidence::default(),
            edged_corners: Vec::new(),
            measured_corners: Vec::new(),
            retry: None,
            turned: 0,
            decodes: None,
            bloom: 0.0,
        };
        assert_eq!(found.found_with(), None);
        // The turn alone, the copy alone, and a symbol found turned
        // inside a copy, which was found both ways.
        found.turned = 1;
        assert_eq!(found.found_with().as_deref(), Some("the photo turned 90°"));
        let enhanced = Retry::Enhanced(Enhance { enlarge: 2, sharpen: Sharpen::Light });
        found.retry = Some(enhanced);
        assert_eq!(found.found_with(), Some(format!("{enhanced}, and the photo turned 90°")));
        found.turned = 0;
        assert_eq!(found.found_with(), Some(enhanced.to_string()));
        found.retry = Some(Retry::Downscaled { factor: 2 });
        found.turned = 3;
        assert_eq!(found.found_with().as_deref(), Some("the photo shrunk 2 times, and the photo turned 270°"));
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
            turned: 0,
            decodes: None,
            bloom: 0.0,
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
            turned: 0,
            decodes: None,
            bloom: 0.0,
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
        let windowed = scan_for_finders(&rect, &Readings::of(&rect, None, true), max_module);
        assert!(windowed.iter().any(|c| near(c) && !c.half), "{windowed:?}");
        let narrowed = rect.narrowed(Some(f64::from(NARROW_RADIUS)));
        let found = scan_thresholds(&narrowed, None, max_module);
        assert!(found.iter().any(|c| near(c) && !c.half), "{found:?}");
    }

    #[test]
    fn one_reading_set_for_the_scan_is_the_scan() {
        // The finder in a shadow of the test above: the two thresholds
        // the scan pools make nothing of it, and the windows do.
        let finder = draw_turned(SymbolVersion::Micro(1), [128.0, 128.0], 8.0, 0.0, (300, 300));
        let image = lit(&finder, |x, y| if x >= 64 && y >= 64 { 0.4 } else { 1.0 });
        let region = [[0.0, 0.0], [300.0, 0.0], [300.0, 300.0], [0.0, 300.0]];
        let max_module = 300.0 * max_module_fraction(Family::Qr);
        let rect = |binarizer: Binarizer| Rectified::new(&image, region).unwrap().ruled(binarizer);
        let own = rect(Binarizer::Pooled).local_radius;
        let centre = rect(Binarizer::Pooled).to_rectified([128.0 - 16.0, 128.0 - 16.0]);
        let near = |c: &Candidate| distance([c.x, c.y], centre) < 8.0 && !c.inverted;
        let finds = |binarizer: Binarizer| { let rect = rect(binarizer); scan_for_finders(&rect, &Readings::of(&rect, None, true), max_module) }.iter().any(|c| near(c) && !c.half);
        // One threshold alone reads the shadowed paper as ink, and the
        // local threshold at the reach the outline sets alone pulls the
        // finder's light runs dark: neither finds it, and neither has
        // the windows to find it with.
        assert!(!finds(Binarizer::Global));
        assert!(!finds(Binarizer::Local { radius: own }));
        // Narrowed into the shadow, the local threshold alone does, and
        // so does a window alone, as the view shows each.
        assert!(finds(Binarizer::Local { radius: f64::from(NARROW_RADIUS) }));
        assert!(finds(Binarizer::Window { radius: 16.0 }));
        // The regions are labelled under the one reading set, and none
        // other: one threshold alone shows no finder there either way.
        let global = rect(Binarizer::Global);
        let regions = region_candidates(&global, &Readings::of(&global, None, false), max_module, &mut Vec::new());
        assert!(!regions.iter().any(near), "{regions:?}");
        let local = rect(Binarizer::Local { radius: f64::from(NARROW_RADIUS) });
        let regions = region_candidates(&local, &Readings::of(&local, None, false), max_module, &mut Vec::new());
        assert!(regions.iter().any(near), "{regions:?}");
        // One threshold set for the scan is Otsu's split, as the view
        // shows it, whatever the sampling's threshold is fixed at: in
        // even light it reads the finder with that one fixed where
        // nothing reads dark.
        let even = Rectified::new(&finder, region).unwrap().ruled(Binarizer::Global);
        assert!(scan_for_finders(&even, &Readings::of(&even, Some(0.0), true), max_module).iter().any(|c| near(c) && !c.half));
        let regions = region_candidates(&even, &Readings::of(&even, Some(0.0), false), max_module, &mut Vec::new());
        assert!(regions.iter().any(near), "{regions:?}");
        // The reach set is in the image's pixels, and is brought to the
        // rectified image's: a region scanned reduced to a third has it
        // a third as long, and the local reading's reach is the scan's.
        let reduced = Rectified::with_longest(&image, region, 130.0).unwrap();
        assert!((reduced.frame.scale - 1.0 / 3.0).abs() < 1e-9, "{}", reduced.frame.scale);
        let reduced = reduced.ruled(Binarizer::Local { radius: 48.0 });
        assert_eq!(reduced.binarizer, Binarizer::Local { radius: 16.0 });
        assert!((reduced.local_radius - 16.0).abs() < 1e-9, "{}", reduced.local_radius);
        // The rung that narrows the scan's threshold reaches as far
        // across the reduced region as across the photo.
        let narrowed = Rectified::with_longest(&image, region, 130.0).unwrap().narrowed(Some(48.0));
        assert!((narrowed.local_radius - 16.0).abs() < 1e-9, "{}", narrowed.local_radius);
        let unreduced = Rectified::new(&image, region).unwrap().narrowed(Some(48.0));
        assert!((unreduced.local_radius - 48.0).abs() < 1e-9, "{}", unreduced.local_radius);
        let windowed = Rectified::with_longest(&image, region, 130.0).unwrap().ruled(Binarizer::Window { radius: 12.0 });
        assert_eq!(windowed.binarizer, Binarizer::Window { radius: 4.0 });
        assert!((windowed.local_radius - local_radius_for(100.0)).abs() < 1e-9, "a window leaves the local reach alone");
        // A copy of the photo made for a look takes the reach scaled
        // with it, and the proposals it fits are fitted with the same.
        let params = SampleParams { binarizer: Binarizer::Local { radius: 40.0 }, ..SampleParams::default() };
        let view = ScannedView::through(&image, Retry::Downscaled { factor: 2 }, &params, &[Family::Qr]).unwrap();
        assert_eq!(view.binarizer, Binarizer::Local { radius: 20.0 });
        assert_eq!(view.rect.binarizer, Binarizer::Local { radius: 20.0 });
        assert!((view.rect.local_radius - 20.0).abs() < 1e-9);
        let whole = ScannedView::of(&image, &params, &[Family::Qr]).unwrap();
        assert_eq!(whole.rect.binarizer, Binarizer::Local { radius: 40.0 });
        // The rung that narrows the scan's own threshold is not climbed
        // where a reading is set in its place: that reading is the
        // scan, at the reach it was set to.
        assert!(Retry::Thresholded { radius: NARROW_RADIUS }.climbed_under(Binarizer::Pooled));
        assert!(!Retry::Thresholded { radius: NARROW_RADIUS }.climbed_under(Binarizer::Global));
        assert!(!Retry::Thresholded { radius: NARROW_RADIUS }.climbed_under(Binarizer::Local { radius: 40.0 }));
        assert!(Retry::Downscaled { factor: 2 }.climbed_under(Binarizer::Global));
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
        assert_eq!(up.scale, [1.0 / 3.0; 2]);
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
        assert_eq!(down.scale, [2.0; 2]);
        assert_eq!(down.luma, image.resized(20, 15));
        let thirds = View::whole(&image, Retry::Downscaled { factor: 3 }).unwrap();
        assert_eq!((thirds.luma.width, thirds.luma.height, thirds.scale), (13, 10, [3.0; 2]));
        let whole = Luma::new(39, 30, (0..1200).filter(|i| i % 40 < 39).map(|i| image.data[i]).collect());
        assert_eq!(thirds.luma, whole.resized(13, 10));
        assert!(View::whole(&image, Retry::Downscaled { factor: 5 }).is_none(), "too small to hold a symbol");
        let closed = View::whole(&image, Retry::Closed { radius: 1 }).unwrap();
        assert_eq!((closed.luma.width, closed.luma.height), (40, 30));
        assert_eq!(closed.scale, [1.0; 2]);
        assert_eq!(needed_upscale([[0.0, 0.0], [25.0, 0.0], [25.0, 25.0], [0.0, 25.0]], Family::Qr), 3);
        assert_eq!(needed_upscale([[0.0, 0.0], [200.0, 0.0], [200.0, 200.0], [0.0, 200.0]], Family::Qr), 1);
        assert_eq!(needed_upscale([[0.0, 0.0], [100.0, 0.0], [100.0, 10.0], [0.0, 10.0]], Family::Rmqr), 3);
    }

    #[test]
    fn a_large_photo_adjusted_is_shrunk_and_closed_at_any_size() {
        // A photo wider than a copy made to be read is held to, toned, and
        // so made again from the photo as loaded for every view: shrunk
        // to more than that width still, and closed at its own size, as
        // the photo as it is would be. Held to that width, as copies
        // enlarged for a read are, the views were once none, and the
        // search passed over the rungs and the shrunk copies they are.
        let (w, h) = (2 * crate::imaging::enhance::MAX_VIEW_SIDE + 8, 40);
        let rgba: Vec<u8> = (0..w * h).flat_map(|i| [((i % w) * 7 % 256) as u8; 3].into_iter().chain([255])).collect();
        let toned = crate::imaging::adjust::Adjust { contrast: 0.2, ..Default::default() }.apply(&rgba, w, h).1;
        assert!(toned.as_shot.is_some());
        let down = View::whole(&toned, Retry::Downscaled { factor: 2 }).expect("shrunk");
        assert_eq!((down.luma.width, down.luma.height, down.scale), (w / 2, h / 2, [2.0; 2]));
        let closed = View::whole(&toned, Retry::Closed { radius: 1 }).expect("closed");
        assert_eq!((closed.luma.width, closed.luma.height, closed.scale), (w, h, [1.0; 2]));
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

    #[test]
    fn a_timing_pattern_blurred_past_its_runs_follows_its_phase() {
        // A version 7 symbol at five pixels a module, blurred by three and
        // grainy: its timing modules are a ripple a few hundredths deep,
        // and the search, which vouched for a placement by reading them
        // one by one, left it unlisted. The alternation they make keeps
        // its pitch and its phase through the placement, and a placement
        // a module off follows it the wrong way round.
        use crate::symbol::grid::ModuleGrid;
        let symbol = qrcode_generator::qr::Encoder::new(qrcode_generator::qr::ErrorCorrection::Medium)
            .version(qrcode_generator::qr::Version::new(7).unwrap())
            .boost_error_correction(false)
            .encode_segments(&[qrcode_generator::Segment::bytes("a timing pattern read by its phase")])
            .unwrap();
        let grid = ModuleGrid::from_bools(symbol.width(), symbol.height(), symbol.modules());
        let version = SymbolVersion::Qr(7);
        let module = 5.0;
        let size = (grid.width() as f64 * module + 60.0) as usize;
        let centre = [size as f64 / 2.0; 2];
        let sharp = draw_grid(&grid, centre, module, 0.0, (size, size));
        let mut grain = 7u64;
        let blurred: Vec<f32> = crate::imaging::enhance::unsharp(&sharp.data, size, size, 1, 3.0, -1.0)
            .into_iter()
            .map(|v| {
                grain ^= grain << 13;
                grain ^= grain >> 7;
                grain ^= grain << 17;
                v + ((grain % 1000) as f32 / 1000.0 - 0.5) * 0.06
            })
            .collect();
        let blurred = Luma::new(size, size, blurred);
        let half = grid.width() as f64 * module / 2.0;
        let corners = [[centre[0] - half, centre[1] - half], [centre[0] + half, centre[1] - half], [centre[0] + half, centre[1] + half], [centre[0] - half, centre[1] + half]];
        let func = FunctionMap::new(version);
        assert!(timing_follows(&sharp, &func, &placed(version, corners)));
        assert!(timing_follows(&blurred, &func, &placed(version, corners)));
        // Sharp, the patterns stand out far past the bar a photo with no
        // blur sets; blurred, short of it, and they follow because the
        // blur runs their modules together.
        let stands = |image: &Luma| timing_stands(image, &func, &placed(version, corners)).expect("both patterns stand out");
        assert!(stands(&sharp) >= FOLLOWED_TIMING_SHARP, "{}", stands(&sharp));
        assert!(stands(&blurred) < FOLLOWED_TIMING_SHARP && blurred_about(&blurred, &placed(version, corners)), "{}", stands(&blurred));
        assert!(!blurred_about(&sharp, &placed(version, corners)));
        let off = corners.map(|[x, y]| [x + module, y]);
        assert!(!timing_follows(&blurred, &func, &placed(version, off)));
        // The four alignment patterns off the timing patterns printed the
        // other way round, as a stain or a sticker might leave them: the
        // fixed modules read wrong by a hundred, as the count a read makes
        // first has them, and the timing patterns follow their phase as
        // before. Let through by those alone, the placement keeps the
        // score it had, short of the floor; it once came back with that
        // count as its score, lower still.
        use crate::symbol::grid::Module;
        let mut stained = grid.clone();
        for (cx, cy) in [(22, 22), (22, 38), (38, 22), (38, 38)] {
            for y in cy - 2..=cy + 2 {
                for x in cx - 2..=cx + 2 {
                    stained.set(x, y, Module::from_dark(stained.get(x, y) != Module::Dark));
                }
            }
        }
        let stained = draw_grid(&stained, centre, module, 0.0, (size, size));
        let params = SampleParams::default();
        let (as_read, known) = func.fixed_agreement(&grid_as_found(&stained, &func, &placed(version, corners), &params).expect("a grid"));
        let agreement = ((MIN_FOUND_SCORE * known as f64).ceil() as usize - 1, known);
        assert!(as_read < agreement.0, "the second count reads {as_read} of {known}, under the {} scored", agreement.0);
        assert!(timing_follows(&stained, &func, &placed(version, corners)));
        let found = vouched(&stained, Detection { agreement, ..placed(version, corners) }, &params).expect("the timing patterns vouch for it");
        assert_eq!(found.agreement, agreement);
        // Random modules read sharp, four pixels to each, whose rows
        // through where the timing patterns would lie happen to follow
        // the alternation by over three spreads, as about one placement on
        // such modules in three hundred follows it by two and a half:
        // their fixed modules read no better than chance, and with no blur
        // to run a symbol's patterns together the timing rows, followed
        // only so far, vouch for nothing. Followed at two and a half
        // whatever the blur, they once let the placement through.
        let w = func.width;
        let mut state = 305u64.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let bools: Vec<bool> = (0..w * w)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state >> 63 != 0
            })
            .collect();
        let (module, size) = (4.0, w * 4 + 40);
        let centre = [size as f64 / 2.0; 2];
        let half = w as f64 * module / 2.0;
        let on_noise = placed(version, [[centre[0] - half, centre[1] - half], [centre[0] + half, centre[1] - half], [centre[0] + half, centre[1] + half], [centre[0] - half, centre[1] + half]]);
        let noise = draw_grid(&ModuleGrid::from_bools(w, w, &bools), centre, module, 0.0, (size, size));
        let stood = timing_stands(&noise, &func, &on_noise).expect("both rows stand out");
        assert!((FOLLOWED_TIMING..FOLLOWED_TIMING_SHARP).contains(&stood) && !blurred_about(&noise, &on_noise), "{stood}");
        assert!(!timing_follows(&noise, &func, &on_noise));
        let (as_read, known) = func.fixed_agreement(&grid_as_found(&noise, &func, &on_noise, &params).expect("a grid"));
        let agreement = ((MIN_FOUND_SCORE * known as f64).ceil() as usize - 1, known);
        assert!((as_read as f64) < MIN_FOUND_SCORE * known as f64, "{as_read} of {known}");
        assert!(vouched(&noise, Detection { agreement, ..on_noise }, &params).is_none());
    }

    /// A detection of `version` placed at `corners`, with nothing else
    /// said of it.
    fn placed(version: SymbolVersion, corners: [[f64; 2]; 4]) -> Detection {
        Detection {
            corners,
            version,
            inverted: false,
            mirrored: false,
            mirrored_by_decode: false,
            turns: 0,
            anchors: Vec::new(),
            agreement: (0, 0),
            info: InfoEvidence::default(),
            edged_corners: Vec::new(),
            measured_corners: Vec::new(),
            retry: None,
            turned: 0,
            decodes: None,
            bloom: 0.0,
        }
    }

    #[test]
    fn candidates_a_region_bears_out_are_refined_past_the_cap() {
        let candidate = |i: usize, quad: bool| Candidate {
            x: 10.0 * i as f64,
            y: 0.0,
            module: [3.0, 3.0],
            inverted: false,
            votes: MIN_VOTES,
            half: false,
            quad: quad.then_some([[0.0, 0.0], [21.0, 0.0], [21.0, 21.0], [0.0, 21.0]]),
            scanned: true,
        };
        let candidates: Vec<Candidate> = (0..10).map(|i| candidate(i, i == 2 || i == 8)).collect();
        let kept = |most: usize| within_cap(candidates.iter().enumerate(), most).map(|(i, _)| i).collect::<Vec<_>>();
        // The first four as they come, and the one past them with a
        // region on it; the one among them counts toward the four.
        assert_eq!(kept(4), [0, 1, 2, 3, 8]);
        assert_eq!(kept(0), [2, 8]);
        assert_eq!(kept(10), (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn a_region_hands_its_quadrilateral_to_the_scans_candidate() {
        // A version 1 symbol turned thirty-seven degrees, between the
        // template's two starts: the scan finds the centre of each finder
        // and a region its four corners, and the candidate that goes to
        // the fit is the scan's with the region's corners on it, not the
        // scan's alone.
        let (version, centre, module_px, degrees) = (SymbolVersion::Qr(1), [150.0, 150.0], 7.0, 37.0);
        let image = draw_turned(version, centre, module_px, degrees, (300, 300));
        let whole = [[0.0, 0.0], [300.0, 0.0], [300.0, 300.0], [0.0, 300.0]];
        let rect = Rectified::new(&image, whole).unwrap();
        let max_module = rect.frame.size[0] * max_module_fraction(Family::Qr);
        let readings = Readings::of(&rect, None, true);
        let mut candidates = scan_for_finders(&rect, &readings, max_module);
        let regions = region_candidates(&rect, &readings, max_module, &mut candidates);
        let (s, c) = degrees.to_radians().sin_cos();
        for [mx, my] in [[-7.0, -7.0], [7.0, -7.0], [-7.0, 7.0]] {
            let (dx, dy) = (mx * module_px, my * module_px);
            let finder = [centre[0] + dx * c - dy * s, centre[1] + dx * s + dy * c];
            let at = |candidate: &&Candidate| distance(rect.frame.to_image.apply(candidate.x, candidate.y), finder) < module_px;
            let scanned: Vec<&Candidate> = candidates.iter().filter(at).collect();
            assert!(!scanned.is_empty(), "the scan saw nothing at {finder:?}");
            let borne: Vec<&&Candidate> = scanned.iter().filter(|candidate| candidate.quad.is_some()).collect();
            assert_eq!(borne.len(), 1, "one candidate at {finder:?} takes the region's corners");
            assert!(regions.iter().filter(at).count() == 0, "and the region proposes nothing of its own there");
            // The corners are the finder's own, seven modules square.
            let quad = borne[0].quad.unwrap();
            for corner in quad {
                let off = distance(corner, finder);
                assert!((off - 3.5 * module_px * SQRT_2).abs() < 1.5, "a corner {off:.1} px from the centre of {finder:?}");
            }
            let capstone = refine_capstone(&image, &rect, borne[0], None).unwrap_or_else(|| panic!("no capstone at {finder:?}"));
            assert!(capstone.measured && !capstone.partial, "{capstone:?}");
            assert!(distance(capstone.center, finder) < 0.5, "{capstone:?}");
            for corner in capstone.corners {
                assert!(quad.iter().any(|&q| distance(corner, q) < module_px), "{capstone:?} against {quad:?}");
            }
        }
    }

    #[test]
    fn the_fit_from_a_region_stands_only_where_it_is_plainly_the_better() {
        let frame = Frame::new([[0.0, 0.0], [400.0, 0.0], [400.0, 400.0], [0.0, 400.0]], MAX_RECTIFIED).unwrap();
        let fitted = |contrast: f64, partial: bool| Capstone { contrast, partial, ..square_capstone([200.0, 200.0], 10.0, 37.0, &frame) };
        // Nothing fitted from the scan's own starts, or a pattern in part.
        assert!(region_fit_stands(None, &fitted(0.3, false)));
        assert!(region_fit_stands(Some(&fitted(0.6, true)), &fitted(0.3, false)));
        // One fit reached from two starts: the contrasts differ by a
        // part in a thousand, either way, and the scan's stands.
        assert!(!region_fit_stands(Some(&fitted(0.0738, false)), &fitted(0.0739, false)));
        assert!(!region_fit_stands(Some(&fitted(0.3494, false)), &fitted(0.3488, false)));
        // The scan's starts settled off the pattern.
        assert!(region_fit_stands(Some(&fitted(0.525, false)), &fitted(0.772, false)));
        // A pattern the region fits in part never stands before the
        // scan's, whole or not.
        assert!(!region_fit_stands(Some(&fitted(0.2, false)), &fitted(0.9, true)));
        assert!(!region_fit_stands(Some(&fitted(0.2, true)), &fitted(0.9, true)));
        assert!(!region_fit_stands(None, &fitted(0.9, true)));
    }

    #[test]
    fn a_claim_that_does_not_decode_keeps_the_search_for_qr_codes_open() {
        let square = |x: f64, y: f64, side: f64| [[x, y], [x + side, y], [x + side, y + side], [x, y + side]];
        // A Micro QR symbol of version M4 claimed on the top-left finder
        // of where a version 1 QR Code lies, ten pixels a module.
        let claim = placed(SymbolVersion::Micro(4), square(100.0, 100.0, 170.0));
        let finder = lone_finder_centre(&claim).unwrap();
        assert!(distance(finder, [135.0, 135.0]) < 1e-9);
        let families = [Family::Qr, Family::Micro, Family::Rmqr];
        let doubts = [finder];
        // The claim alone: QR Codes are owed a look, and the families
        // with one finder are not.
        let found = vec![claim.clone()];
        assert_eq!(still_wanted(&[], &families, &doubts, &found), [Family::Qr]);
        assert_eq!(still_wanted(&[Family::Rmqr], &families, &doubts, &found), [Family::Qr, Family::Rmqr]);
        // A search that is not for QR Codes owes them nothing.
        assert!(still_wanted(&[], &[Family::Micro, Family::Rmqr], &doubts, &found).is_empty());
        // No claim in doubt, nothing owed.
        assert!(still_wanted(&[], &families, &[], &found).is_empty());
        assert_eq!(still_wanted(&[Family::Qr], &families, &[], &found), [Family::Qr]);
        // A QR Code elsewhere in the photo settles nothing.
        let elsewhere = placed(SymbolVersion::Qr(1), square(600.0, 600.0, 210.0));
        let mut found = vec![claim.clone(), elsewhere.clone()];
        assert_eq!(still_wanted(&[], &families, &doubts, &found), [Family::Qr]);
        drop_settled_claims(&mut found, &doubts);
        assert_eq!(found.len(), 2);
        // The QR Code the finder belongs to does, and the claim goes.
        let whole = placed(SymbolVersion::Qr(1), square(100.0, 100.0, 210.0));
        let mut found = vec![claim.clone(), elsewhere, whole];
        assert!(still_wanted(&[], &families, &doubts, &found).is_empty());
        drop_settled_claims(&mut found, &doubts);
        assert_eq!(found.iter().map(|d| d.version).collect::<Vec<_>>(), [SymbolVersion::Qr(1), SymbolVersion::Qr(1)]);
        // A claim that was never in doubt, one that decoded, stands
        // whatever lies over it.
        let mut found = vec![claim, placed(SymbolVersion::Qr(1), square(100.0, 100.0, 210.0))];
        drop_settled_claims(&mut found, &[]);
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn a_version_word_read_outright_stands_on_the_fixed_patterns_alone() {
        // A version 7 placement whose version information reads outright
        // and whose fixed modules read 97 in a hundred right, given a fit
        // of 44 by a quiet zone under a lamp's bright spot: it is named
        // all the same, and ranks above a version 6 placement on the
        // same finders that nothing names and that fits a little better.
        let square = [[0.0, 0.0], [450.0, 0.0], [450.0, 450.0], [0.0, 450.0]];
        let named = |mismatches: u32, agreement: (usize, usize)| {
            let mut found = placed(SymbolVersion::Qr(7), square);
            found.info.named_version = Some(SymbolVersion::Qr(7));
            found.info.named_mismatches = Some(mismatches);
            found.agreement = agreement;
            found
        };
        let unnamed = placed(SymbolVersion::Qr(6), square);
        assert_eq!(rank(&named(0, (378, 391)), 0.44, None), (1, 0.44));
        assert!(rank(&named(0, (378, 391)), 0.44, None) > rank(&unnamed, 0.475, None));
        // A word read with a bit corrected is one that half of all
        // strings lie as near, and the fit must bear it out, as it had to.
        assert_eq!(rank(&named(1, (378, 391)), 0.44, None), (0, 0.44));
        assert_eq!(rank(&named(1, (378, 391)), 0.75, None), (1, 0.75));
        // Fixed patterns that read no better than the fit bear out
        // nothing, word or no word.
        assert_eq!(rank(&named(0, (250, 391)), 0.44, None), (0, 0.44));
        // And a word for another version counts against the placement
        // however its patterns read.
        let mut misnamed = named(0, (378, 391));
        misnamed.info.named_version = Some(SymbolVersion::Qr(8));
        assert_eq!(rank(&misnamed, 0.44, None), (-1, 0.44));
    }

    #[test]
    fn the_finders_of_a_turned_symbol_are_not_crowded_out_by_a_sheet_of_upright_ones() {
        // Forty-four symbols square to the photo and four turned
        // thirty-seven degrees, of four pixels a module: the upright
        // finders, a hundred and thirty-two of them, poll every line
        // across their stones and fill the cap on the scan's candidates,
        // and the turned ones, which poll a few, came after it.
        let (version, module_px, pitch) = (SymbolVersion::Qr(1), 4.0, 126usize);
        let tile = |degrees: f64| draw_turned(version, [pitch as f64 / 2.0; 2], module_px, degrees, (pitch, pitch));
        let (upright, turned) = (tile(0.0), tile(37.0));
        let (columns, rows) = (8usize, 6usize);
        let (width, height) = (columns * pitch, rows * pitch);
        let mut data = vec![0.92f32; width * height];
        let is_turned = |i: usize| i >= 44;
        for i in 0..columns * rows {
            let (x, y) = ((i % columns) * pitch, (i / columns) * pitch);
            let tile = if is_turned(i) { &turned } else { &upright };
            for row in 0..pitch {
                data[(y + row) * width + x..][..pitch].copy_from_slice(&tile.data[row * pitch..][..pitch]);
            }
        }
        let image = Luma::new(width, height, data);
        let view = ScannedView::of(&image, &SampleParams::default(), &[Family::Qr]).unwrap();
        let capstones = view.capstones(Family::Qr, &|_, _| false);
        let (s, c) = 37f64.to_radians().sin_cos();
        for i in (0..columns * rows).filter(|&i| is_turned(i)) {
            let centre = [((i % columns) * pitch + pitch / 2) as f64, ((i / columns) * pitch + pitch / 2) as f64];
            for [mx, my] in [[-7.0, -7.0], [7.0, -7.0], [-7.0, 7.0]] {
                let (dx, dy) = (mx * module_px, my * module_px);
                let finder = [centre[0] + dx * c - dy * s, centre[1] + dx * s + dy * c];
                let seen = capstones.iter().any(|capstone| !capstone.inverted && capstone.measured && distance(capstone.center, finder) < module_px);
                assert!(seen, "no capstone at {finder:?}, of {} in all", capstones.len());
            }
        }
        // With the first symbol of the sheet found already, what lies on
        // it is not fitted again: the rest come back as they did, and
        // no candidate there has been refined.
        let half = 10.5 * module_px;
        let centre = pitch as f64 / 2.0;
        let first = placed(version, [[centre - half, centre - half], [centre + half, centre - half], [centre + half, centre + half], [centre - half, centre + half]]);
        let again = ScannedView::of(&image, &SampleParams::default(), &[Family::Qr]).unwrap();
        let rest = again.capstones(Family::Qr, &|_, at| holds(&first, at));
        let elsewhere: Vec<[f64; 2]> = capstones.iter().filter(|c| !holds(&first, c.center)).map(|c| c.center).collect();
        assert_eq!(rest.iter().map(|c| c.center).collect::<Vec<_>>(), elsewhere);
        assert!(rest.len() + 3 <= capstones.len(), "{} of {}", rest.len(), capstones.len());
        let on_first = |c: &Candidate| holds(&first, again.to_photo(again.rect.frame.to_image.apply(c.x, c.y)));
        assert!(again.candidates.iter().zip(&again.refined[0]).filter(|(c, _)| on_first(c)).all(|(_, fit)| fit.get().is_none()));
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
    fn a_symbol_too_small_for_the_copies_is_found_beside_one_they_show() {
        // Paper over two thousand pixels across, searched first in a
        // copy a third its size. A symbol of twelve pixels a module has
        // four in the copy and settles the search there; one of three
        // has one, and the copy shows nothing of it, not so much as a
        // finder left over.
        let (width, height) = (2100, 1300);
        let (large, small) = ([600usize, 650usize], [1500usize, 400usize]);
        let (with_large, with_small) = (paper_with_symbols(width, height, 12.0, &[large]), paper_with_symbols(width, height, 3.0, &[small]));
        let image = Luma::new(width, height, with_large.data.iter().zip(&with_small.data).map(|(a, b)| a.min(*b)).collect());
        let (params, families) = (SampleParams::default(), [Family::Micro]);
        assert_eq!(coarse_factors(&image), [3]);
        let coarse = search_coarse(&image, &families, &params, &mut Vec::new(), &mut Vec::new(), &|_| true);
        assert_eq!(coarse.detections.len(), 1, "{:?}", coarse.detections);
        assert!(distance(quad_centre(coarse.detections[0].corners), large.map(|v| v as f64)) < 1.5);
        assert!(coarse.more.is_empty());
        assert_eq!(coarse.shrunk, Some(3));
        // The photo at its own size shows the small one's finder as a
        // region on no symbol found, and the symbol is found by it.
        let shown = COARSE_SHOWN_MODULE * 3.0;
        let beside = ScannedView::beside(&image, &params, &families, &coarse.detections, shown).expect("a view");
        assert!(beside.candidates.is_empty(), "the rows are not scanned");
        assert!(!beside.regions.is_empty());
        let found = find_symbols(&image, Family::Micro, &params);
        assert_eq!(found.iter().map(|d| d.version).collect::<Vec<_>>(), [SymbolVersion::Micro(2); 2]);
        let there = found.iter().find(|d| distance(quad_centre(d.corners), small.map(|v| v as f64)) < 1.5).expect("the small symbol");
        let side = distance(there.corners[0], there.corners[1]);
        assert!((side - 39.0).abs() < 2.0, "thirteen modules of three pixels, not {side}");
        // A region on a finder of a QR Code already found proposes
        // nothing: it is left out before anything is fitted to it. One
        // elsewhere on the QR Code is kept, for the symbol set within it
        // whose finder it may be, and so is one on a symbol with a
        // single finder pattern, for the QR Code that finder may be a
        // third of. And a finder the copy will have shown, as the large
        // symbol's at four pixels a module there, is the copy's to have
        // made a symbol of.
        let over = |version: SymbolVersion, [x, y]: [usize; 2], half: f64| {
            let (x, y) = (x as f64, y as f64);
            placed(version, [[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]])
        };
        let regions = |known: &[Detection], shown: f64| ScannedView::beside(&image, &params, &families, known, shown).expect("a view").regions.len();
        let all = regions(&[], f64::INFINITY);
        assert!(all >= 2, "a finder of each symbol, of {all} regions");
        assert_eq!(regions(&[over(SymbolVersion::Micro(2), small, 20.0)], f64::INFINITY), all);
        // A version 1 QR Code of the small symbol's module, its top-left
        // finder the small symbol's own: thirteen modules of three
        // pixels put the finder nine pixels up and left of the centre,
        // and twenty-one put the QR Code's centre twenty-one down and
        // right of its finder's.
        let on_finder = over(SymbolVersion::Qr(1), [small[0] + 12, small[1] + 12], 31.5);
        assert!(regions(&[on_finder], f64::INFINITY) < all);
        // A version 7 of the same module with the small symbol in the
        // middle of its data.
        assert_eq!(regions(&[over(SymbolVersion::Qr(7), small, 67.5)], f64::INFINITY), all);
        assert!((1..all).contains(&regions(&[], shown)), "the small symbol's finder and not the large one's");
    }

    #[test]
    fn a_last_copy_that_shows_a_symbol_small_sends_the_search_on_to_the_photo() {
        // The photo's one copy is half its size. A symbol of 3.6 pixels
        // a module has 1.8 there, which the copy makes out, this one
        // being nearly square to its rows; what else of that size it
        // did not make out it cannot say, and every family is owed the
        // photo itself.
        let (params, families) = (SampleParams::default(), [Family::Micro, Family::Qr]);
        let small = paper_with_symbols(2048, 1536, 3.6, &[[700, 800]]);
        assert_eq!(coarse_factors(&small), [2]);
        let coarse = search_coarse(&small, &families, &params, &mut Vec::new(), &mut Vec::new(), &|_| true);
        assert_eq!(coarse.detections.iter().map(|d| d.version).collect::<Vec<_>>(), [SymbolVersion::Micro(2)]);
        assert_eq!(coarse.more, families);
        // One of five has two and a half, and the copy settles it.
        let sound = paper_with_symbols(2048, 1536, 5.0, &[[700, 800]]);
        let coarse = search_coarse(&sound, &families, &params, &mut Vec::new(), &mut Vec::new(), &|_| true);
        assert_eq!(coarse.detections.len(), 1);
        assert!(coarse.more.is_empty(), "{:?}", coarse.more);
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

    #[test]
    fn three_finders_seen_from_far_to_one_side_stand_by_their_own_squares() {
        // A version 2 symbol seen from far along its wall: nine pixels a
        // module down and three across. The legs between its finders'
        // centres are fifty-four pixels and a hundred and sixty-two, no
        // triangle a symbol's finders make, and each is eighteen of the
        // finders' own modules, which is the symbol.
        let frame = Frame::new([[0.0, 0.0], [800.0, 0.0], [800.0, 800.0], [0.0, 800.0]], MAX_RECTIFIED).unwrap();
        let image = Luma::new(800, 800, vec![0.92; 800 * 800]);
        let (across, down) = (3.0, 9.0);
        let at = |[x, y]: [f64; 2]| [300.0 + across * x, 200.0 + down * y];
        let finder = |[x, y]: [f64; 2]| Capstone {
            corners: order_in_frame([[-3.5, -3.5], [3.5, -3.5], [3.5, 3.5], [-3.5, 3.5]].map(|[dx, dy]| at([x + dx, y + dy])), &frame),
            module: [across, down],
            ..square_capstone(at([x, y]), (across + down) / 2.0, 0.0, &frame)
        };
        let trio = vec![finder([3.5, 3.5]), finder([21.5, 3.5]), finder([3.5, 21.5])];
        let standing = [&trio[0], &trio[1], &trio[2]];
        let p = standing.map(|c| frame.to_rectified(c.center));
        assert!(stands_by_pixels(&standing, &p, &frame, (across + down) / 2.0).is_none());
        let shape = stands_by_squares(&standing, &p, &frame).expect("the three stand by their squares");
        assert_eq!(shape.tl, 0);
        assert!((shape.spacing[1] - 18.0).abs() < 1e-9 && (shape.spacing[2] - 18.0).abs() < 1e-9, "{:?}", shape.spacing);
        assert!(shape.distortion.abs() < 1e-9);
        // The fourth corner's finder would lie where the other two's
        // own sides run to.
        assert!(distance(shape.fourth.expect("the squares close the symbol"), at([21.5, 21.5])) < 1e-6);
        // One proposal, about the symbol and two modules of quiet zone.
        let proposals = qr_proposals(&image, &frame, &trio);
        assert_eq!(proposals.len(), 1, "{proposals:?}");
        for (corner, want) in proposals[0].1.iter().zip([[-2.0, -2.0], [27.0, -2.0], [27.0, 27.0], [-2.0, 27.0]]) {
            assert!(distance(*corner, at(want)) < 1e-6, "{proposals:?}");
        }
        // The two at the ends of the diagonal face neither each other
        // nor, by sides that adjoin, a third set down between them.
        let between = vec![finder([21.5, 3.5]), finder([3.5, 21.5]), finder([12.5, 12.5])];
        assert!(qr_proposals(&image, &frame, &between).is_empty());
        // Three in a row face each other by opposite sides of the one in
        // the middle: a row of three symbols' finders, and no corner.
        let row = vec![finder([3.5, 3.5]), finder([21.5, 3.5]), finder([39.5, 3.5])];
        assert!(qr_proposals(&image, &frame, &row).is_empty());
        // With corners never measured there are no squares to go by.
        let unmeasured: Vec<Capstone> = trio.iter().map(|c| Capstone { measured: false, ..*c }).collect();
        assert!(qr_proposals(&image, &frame, &unmeasured).is_empty());
    }

    #[test]
    fn a_photo_as_long_as_a_search_looks_at_is_scanned_at_its_own_size() {
        let (width, height) = (SEARCH_SIDE as f64, 0.75 * SEARCH_SIDE as f64);
        let frame = Frame::new([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], MAX_RECTIFIED_WHOLE).unwrap();
        assert!((frame.size[0] - width).abs() < 1e-6 && (frame.size[1] - height).abs() < 1e-6, "{:?}", frame.size);
        assert!((distance(frame.to_rectified([100.0, 100.0]), frame.to_rectified([101.0, 100.0])) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_lone_finder_is_spoken_for_by_what_lies_beside_it() {
        use crate::symbol::grid::Module;
        let params = SampleParams::default();
        let whole = [[0.0, 0.0], [400.0, 0.0], [400.0, 400.0], [0.0, 400.0]];
        let capstone = |image: &Luma| {
            let rect = Rectified::with_longest(image, whole, MAX_RECTIFIED_WHOLE).unwrap();
            let capstones: Vec<Capstone> = collect_capstones(image, &rect, None, Family::Micro).into_iter().filter(|c| !c.inverted).collect();
            assert_eq!(capstones.len(), 1, "{capstones:?}");
            capstones[0]
        };
        let in_turn = |proposals: &mut Vec<LoneProposal>| proposals.sort_by(|a, b| b.spoken.cmp(&a.spoken).then(a.badness.total_cmp(&b.badness)));
        let holds = |outline: &[[f64; 2]; 4], at: [f64; 2]| {
            (0..4).all(|i| {
                let (a, b) = (outline[i], outline[(i + 1) % 4]);
                (b[0] - a[0]) * (at[1] - a[1]) - (b[1] - a[1]) * (at[0] - a[0]) > 0.0
            })
        };
        // An M2 symbol turned a little, eight pixels a module: it decodes
        // through its one finder's own square, and the box its word
        // names is spoken for and stands first.
        let grid = micro_m2();
        let func = FunctionMap::new(SymbolVersion::Micro(2));
        let symbol = draw_grid(&grid, [200.0, 200.0], 8.0, 4.0, (400, 400));
        let finder = capstone(&symbol);
        let placements = format_placements(&symbol, &finder, Family::Micro, None, params.band);
        assert!(placements.iter().any(|placement| decodes_through_finder(&symbol, &finder, placement, &params)), "{placements:?}");
        let mut proposals = lone_finder_proposals(&symbol, &finder, Family::Micro, None, &params);
        in_turn(&mut proposals);
        assert!(proposals[0].spoken && proposals[0].badness < 0.0, "{:?}", proposals[0].badness);
        assert!(distance(quad_centre(proposals[0].outline), [200.0, 200.0]) < 8.0);
        // Its word and its data rubbed out, the timing patterns left:
        // of the boxes about the finder the one the timing patterns run
        // into is spoken for, and no other, whatever word a timing
        // pattern spells through some other placement.
        let mut timed = grid.clone();
        let mut bare = grid.clone();
        for y in 0..func.height {
            for x in 0..func.width {
                match func.role(x, y) {
                    Role::Finder | Role::Separator => {}
                    Role::Timing => bare.set(x, y, Module::Light),
                    _ => {
                        timed.set(x, y, Module::Light);
                        bare.set(x, y, Module::Light);
                    }
                }
            }
        }
        let image = draw_grid(&timed, [200.0, 200.0], 8.0, 4.0, (400, 400));
        let finder = capstone(&image);
        let mut proposals = lone_finder_proposals(&image, &finder, Family::Micro, None, &params);
        in_turn(&mut proposals);
        assert_eq!(proposals.iter().filter(|p| p.spoken).count(), 1);
        assert!(holds(&proposals[0].outline, [200.0, 200.0]), "{:?}", proposals[0].outline);
        // A finder with nothing beside it speaks for no box at all.
        let image = draw_grid(&bare, [200.0, 200.0], 8.0, 4.0, (400, 400));
        let proposals = lone_finder_proposals(&image, &capstone(&image), Family::Micro, None, &params);
        assert_eq!(proposals.len(), 4);
        assert!(proposals.iter().all(|p| !p.spoken && p.badness > UNNAMED_DOUBT), "{:?}", proposals.iter().map(|p| p.badness).collect::<Vec<_>>());
    }

    #[test]
    #[ignore = "two searches that climb the whole ladder and fit a pair at every rung: half a minute"]
    fn two_finders_with_the_format_word_beside_them_are_a_lead() {
        use crate::symbol::grid::Module;
        // What a sheet of paper laid over the lower half of a version 2
        // symbol leaves showing: its top two finders, the timing pattern
        // between them and the format word about the first, turned a
        // little. No symbol can be fitted to that, and the search says
        // where it would have been.
        let version = SymbolVersion::Qr(2);
        let func = FunctionMap::new(version);
        let showing = |worded: bool| {
            let mut grid = ModuleGrid::new(func.width, func.height);
            for y in 0..func.height {
                for x in 0..func.width {
                    let kept = y < 8 && matches!(func.role(x, y), Role::Finder | Role::Separator | Role::Timing);
                    let dark = kept && func.expected(x, y) == Some(true);
                    grid.set(x, y, if dark { Module::Dark } else { Module::Light });
                }
            }
            if worded {
                let copy = &func.format[0];
                let word = bch::format_words(version)[5].1 ^ copy.xor;
                for (bit, at) in copy.bits.iter().enumerate() {
                    grid.set(at.x as usize, at.y as usize, if word >> bit & 1 == 1 { Module::Dark } else { Module::Light });
                }
            }
            draw_grid(&grid, [200.0, 200.0], 8.0, 4.0, (400, 400))
        };
        let search = |image: &Luma| search_for_families(image, &[Family::Qr], &SampleParams::default(), &SearchParams::default(), &|_| true);
        // Without the word there are two finders and a timing pattern,
        // which two letters of a line of print have as well: nothing is
        // said of them.
        let unworded = search(&showing(false));
        assert!(unworded.detections.is_empty() && unworded.leads.is_empty(), "{unworded:?}");
        let sought = search(&showing(true));
        assert!(sought.detections.is_empty(), "{:?}", sought.detections);
        assert_eq!(sought.leads.len(), 1, "{:?}", sought.leads);
        let lead = &sought.leads[0];
        assert_eq!((lead.family, lead.finders.len()), (Family::Qr, 2));
        // The finders' centres, three and a half modules in from the
        // symbol's top corners, turned with it about the middle.
        let (sin, cos) = 4.0f64.to_radians().sin_cos();
        let at = |x: f64, y: f64| [200.0 + 8.0 * ((x - 12.5) * cos - (y - 12.5) * sin), 200.0 + 8.0 * ((x - 12.5) * sin + (y - 12.5) * cos)];
        for centre in [at(3.5, 3.5), at(21.5, 3.5)] {
            assert!(lead.finders.iter().any(|&finder| distance(finder, centre) < 4.0), "{:?}", lead.finders);
        }
        // The outline proposed holds where the whole symbol lay.
        assert!(distance(quad_centre(lead.outline), [200.0, 200.0]) < 24.0, "{:?}", lead.outline);
    }

    #[test]
    fn leads_are_left_where_no_symbol_was_found_and_each_place_once() {
        let outline = |x: f64, y: f64| [[x, y], [x + 100.0, y], [x + 100.0, y + 100.0], [x, y + 100.0]];
        let lead = |x: f64, y: f64| Lead {
            family: Family::Qr,
            outline: outline(x, y),
            finders: vec![[x + 20.0, y + 20.0], [x + 80.0, y + 20.0], [x + 20.0, y + 80.0]],
        };
        // The same finders as two copies of the photo had them, a pixel
        // apart, and others elsewhere.
        let leads = vec![lead(100.0, 100.0), lead(101.0, 99.0), lead(400.0, 100.0)];
        assert_eq!(leads_left(leads.clone(), &[]), vec![lead(100.0, 100.0), lead(400.0, 100.0)]);
        // A symbol another copy found on the first set disposes of them.
        let found = placed(SymbolVersion::Qr(2), [[110.0, 110.0], [190.0, 110.0], [190.0, 190.0], [110.0, 190.0]]);
        assert_eq!(leads_left(leads, &[found]), vec![lead(400.0, 100.0)]);
    }
}
