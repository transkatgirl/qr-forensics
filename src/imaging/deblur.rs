//! Undoing a blur measured off the photo itself: how far, and which way,
//! the photo spreads an edge, taken from the edges it shows, and a
//! deconvolution by that spread.
//!
//! The sharpening a read tries when a symbol does not decode
//! ([`super::enhance`]) is an unsharp mask of a fixed width, a pixel or
//! two, the same every way: it puts back what a lens takes from modules
//! a pixel or two across, and nothing of what a photo out of focus by
//! ten pixels takes from modules of seventeen, nor of what a hand that
//! moved takes along the way it moved. In BoofCV's benchmark the
//! photographs of the `blurred` set are both: some out of focus, and
//! some shaken, their strokes smeared one way and their edges doubled.
//! Their symbols read through the corners marked for them, and the
//! search did not find them: a finder pattern's rings run into its
//! stone, and no threshold reads 1:1:3:1:1 across it.
//!
//! The blur is measured as it is in a photograph of a symbol, from the
//! edges: across an edge the light changes from ink to paper over a
//! width the blur sets, and that width along the edge's normal is the
//! blur's spread that way. At each pixel where the change is steepest
//! along its own direction, the step in light either side of it over
//! the steepness is the edge's width, which for a Gaussian blur of
//! spread `σ` comes to `σ√(2π)`, and does not depend on how dark the
//! ink or how light the paper. Edges that lie close together, as the
//! modules of a symbol blurred by a third of a module and more do, read
//! narrower than that: a band of ink a spread or two wide never reaches
//! its full dark, and the step across its edge falls short where the
//! steepness at the edge barely falls. Read so, symbols blurred in light
//! by from three to six tenths of a module came a tenth to a quarter
//! under their blur. So the light across each edge is read outwards as
//! far as it turns back, where the next edge begins, and the width the
//! edge would have alone is solved for from where it turned ([`alone`]),
//! which brings the same symbols to within about a twentieth; an edge
//! whose neighbours lie so near that the edges beyond them count too is
//! left out. The middle width of each direction's edges is taken: a
//! blur read over would ring where it is undone. A hand that moved
//! spreads the edges across its path more than those along it, and the
//! widths of the edges of each direction are fitted by the spread a
//! Gaussian of any shape gives them, turned and stretched as the photo
//! was: one number each way for a lens out of focus, and a long narrow
//! one for a shake.
//!
//! How a shake is turned is told by edges that run every way, as print
//! and text about a symbol do. A symbol's own modules have their edges
//! along two ways only, and a shake slanting across both widens them as
//! a shorter, rounder blur along them would, which is what is measured
//! where they are all there is: a shake three pixels long laid thirty
//! degrees off the edges of a field of squares came to a spread two and
//! a quarter long and one wide, turned eighteen degrees off them. Fitted
//! at each direction's own mean rather than its middle, it turned
//! twenty-one, and a round blur over squares a lens had bent a little,
//! whose corners read narrower one way, came to one and a half long
//! against one wide; so the edges are fitted at the middle of their
//! direction, which holds a round blur round. Edges that all run within
//! a fan narrower than a quarter turn, as a barcode's or ruled paper's
//! do, say nothing of how far the blur spreads along them, and a shape
//! fitted to them is a guess there; the blur is then taken the same
//! every way ([`LEAST_FAN`]).
//!
//! The deconvolution is Richardson and Lucy's: the estimate of the
//! sharp photo is blurred by the spread measured, the photo is divided
//! by that, and the quotient, blurred again, multiplies the estimate,
//! twenty-five times over. It keeps light positive and a flat tone
//! flat, and sharpens where the photo has edges to sharpen, where an
//! inverse filter raises the noise everywhere. A shake is not a
//! Gaussian, and one fitted to it undoes the most of it that one can:
//! enough to give a finder pattern back its rings.
//!
//! All of it is done in the light the photo is worked in ([`Light`]),
//! as the photo's other adjustments are. A lens spreads the light from a
//! point, and a photo stores that light blurred on sRGB's curve, which is
//! no blur of anything it stores: worked as light, its values are taken
//! as the light they stand for before the blur is measured
//! ([`gpu::edges`]) or undone, the photo reduced and enlarged back in
//! light with it, and the light undone is stored again ([`gpu::in_light`],
//! [`gpu::encoded`]), for all that comes after to read as it reads any
//! photo. Worked as stored, the default, it is measured and undone as
//! stored, which is how a program blurs an image, and how a photo
//! inverted or toned for contrast before it is filtered stores what it
//! shows: its values are no longer on sRGB's curve, and taken as light
//! they spread the blur's edges unevenly, further than as stored. Always
//! worked as light, an inverted photo's edges read a blur's middle at a
//! ninth of the way from ink to paper, where as stored they had read it
//! at a third. Measured as stored,
//! an edge from ink at 0.16 to paper at 0.94 blurred by a spread `σ` came
//! to 2.8`σ` wide rather than 2.5, its steepest change two thirds of a
//! spread into the ink, and how much too wide went with how dark the ink
//! was. That had all but cancelled the shortfall of edges close together,
//! which measured in light was left whole: until it was solved for, the
//! search of BoofCV's blurred photographs lost a symbol it had read and
//! one it had found. And a band of ink a spread wide, which reaches 38%
//! of the way to the ink as light, reaches 23% as stored, so that undone
//! as stored, a thin dark module had less of its ink given back than a
//! thin light one had of its paper. An image a program blurred, which
//! mixed what it stores, is blurred as stored, and is undone so.
//!
//! A blur many pixels wide would take a kernel of hundreds of pixels to
//! undo where it lies. The photo is reduced first, until the blur is no
//! more than [`MOST_SPREAD`] pixels, and the blur measured is carried to
//! the size it comes to: a symbol of seventeen pixels a module blurred
//! by eight is a symbol of five blurred by two and a half. It is not
//! measured again there. Magic Kernel Sharp adds next to nothing to the
//! width of an edge it reduces, and a measure at the smaller size reads
//! the edges, nearer together there, narrower than they are, and reads
//! nothing at all where the modules come to a pixel or two: a photo of
//! twelve megapixels reduced to the most a deconvolution works on put
//! modules of six pixels at two, and its blur of three pixels went
//! unread. The reduction is by a ratio of small whole numbers
//! ([`Reduction`]), laid on the image from its first pixel, so that a
//! part of the image that begins a whole number of the ratio's steps in
//! is reduced, deblurred and enlarged back as the whole is there.
//!
//! A copy the search looks through is reduced besides until it is no
//! more than [`MOST_PIXELS`], as a photo is searched in a reduced copy.
//! The photo's Deblur filter, and a copy a read is made through, are
//! reduced only past [`MOST_READ_PIXELS`], which no copy a read makes
//! comes to: the filter set from a read shows what the read read.
//!
//! A photo larger than [`MOST_PIXELS`] is measured in parts at its own
//! size, those with the most edges in them, and not in a copy reduced
//! to that: reduced, the same modules of six pixels came to two, and no
//! edge of them said how far the blur spread. Nor is it measured whole:
//! the steepest change and the contrast an edge is judged against are
//! the image's, and where a symbol is a small part of a photograph of
//! grainy paper, the grain sets both, and its edges, the narrowest of
//! all, are most of those measured.

use std::borrow::Cow;

use super::gpu::{self, scan::Values};
use super::resize::resize;
use super::sample::Light;

/// The spread of the light from a point in a photo, as a Gaussian of any
/// shape: its covariance, in square pixels, across (`xx`), down (`yy`)
/// and between the two (`xy`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spread {
    pub xx: f64,
    pub xy: f64,
    pub yy: f64,
}

impl Spread {
    /// The spread the same every way, `sigma` pixels.
    pub fn round(sigma: f64) -> Self {
        Self { xx: sigma * sigma, xy: 0.0, yy: sigma * sigma }
    }

    /// The spread along its widest way and its narrowest, in pixels: the
    /// square roots of the covariance's two eigenvalues.
    pub fn extremes(&self) -> (f64, f64) {
        let mean = (self.xx + self.yy) / 2.0;
        let half = (((self.xx - self.yy) / 2.0).powi(2) + self.xy * self.xy).sqrt();
        ((mean + half).max(0.0).sqrt(), (mean - half).max(0.0).sqrt())
    }

    /// The spread in a copy of the photo `times` its size.
    pub fn scaled(&self, times: f64) -> Self {
        let t2 = times * times;
        Self { xx: self.xx * t2, xy: self.xy * t2, yy: self.yy * t2 }
    }

    /// The spread in a copy of the photo `across` times its width and
    /// `down` times its height: [`Self::scaled`], each way by its own.
    pub fn stretched(&self, [across, down]: [f64; 2]) -> Self {
        Self { xx: self.xx * across * across, xy: self.xy * across * down, yy: self.yy * down * down }
    }

    /// The way the spread is widest, in degrees from across towards down,
    /// from -90 to 90; nought for a spread the same every way.
    pub fn angle(&self) -> f64 {
        (0.5 * (2.0 * self.xy).atan2(self.xx - self.yy)).to_degrees()
    }

    /// The spread in the photo turned a quarter clockwise: across is
    /// down, and down across the other way.
    pub fn turned_cw(&self) -> Self {
        Self { xx: self.yy, xy: -self.xy, yy: self.xx }
    }
}

/// Least spread, in pixels, along the blur's widest way, worth undoing:
/// a lens and a sensor spread a sharp edge over a pixel or so, and the
/// unsharp masks of [`super::enhance`] take that back.
pub const LEAST_SPREAD: f64 = 1.2;
/// Most spread, in pixels, the deconvolution works with: a blur wider
/// is undone in a copy reduced until it is no wider. Three times this
/// either way is the reach of the kernel.
pub const MOST_SPREAD: f64 = 2.0;
/// Most pixels a copy the search looks through is deconvolved at: a
/// larger one is reduced to this first, as a photo is searched in a
/// reduced copy. And the most of a larger image measured, in parts
/// ([`spread_of`]).
pub const MOST_PIXELS: usize = 1_500_000;
/// Most pixels the photo's Deblur filter is deconvolved at, and a copy a
/// read is made through: a larger one is reduced to this first. As many
/// as the largest copy a read makes, [`super::enhance::MAX_VIEW_SIDE`]
/// each way, so that a read that decoded only through a copy deblurred,
/// and set the filter to what it undid, has the photo show what it read,
/// and Sample read it so. A photograph of twelve megapixels, deconvolved
/// at [`MOST_PIXELS`] as a search's copy is, put modules of three pixels
/// at one, and read a hundred modules of a symbol wrong that the copy
/// had read right; at its own size its filter costs half a second.
pub const MOST_READ_PIXELS: usize = super::enhance::MAX_VIEW_SIDE * super::enhance::MAX_VIEW_SIDE;
/// The share of a module, at least, the blur must spread an edge by,
/// along its widest way, for undoing it to be worth a look: a fifth.
/// Under that, a threshold or a mask of a pixel or two reads the
/// modules, and the finder patterns keep their rings.
pub const WORTH_UNDOING: f64 = 0.2;
/// Side, in pixels, of the parts a larger image is measured in.
const PART_SIDE: usize = 512;
/// Pixels about a part measured that its edges are read with, past the
/// part: the smoothing's reach and the gradient's pixel, and the
/// furthest a step is read across an edge, so that the edges of a part
/// read as they would in the whole image.
const PART_MARGIN: usize = 56;
/// Rounds of the deconvolution.
pub(crate) const ROUNDS: usize = 25;
/// The furthest, in pixels, a step across an edge is read either side
/// of it ([`measured_over`]).
const MOST_REACH: f64 = 48.0;
/// Directions the edges are sorted into, over half a turn.
const DIRECTIONS: usize = 12;
/// Fewest edges a direction must have to be fitted on.
const LEAST_EDGES: usize = 24;
/// Least reach of the directions a blur's shape is fitted on, in
/// [`DIRECTIONS`]: from the first of them to the last round the half
/// turn, the widest gap between them left out. A quarter turn, as two
/// edges at right angles reach. Over a narrower fan the shape is a guess
/// along the edges, where none was measured: three directions within
/// thirty degrees, each reading a tenth off, may put a round blur at
/// twice its width one way, where over a quarter turn they hold it
/// within about a fifth.
const LEAST_FAN: usize = DIRECTIONS / 2;
/// The spread, in pixels, of the Gaussian the photo is smoothed by before
/// its edges are measured: a photograph's grain changes the light by a
/// few hundredths from pixel to pixel, as steeply as an edge out of
/// focus does over several, and unsmoothed the grain beside an edge is
/// taken for the edge's own steepness. What it adds to every edge's
/// spread is taken off again.
const SMOOTHING: f64 = 1.0;
/// The variance, in square pixels, of the spread the measure itself
/// gives an edge drawn sharp: the smoothing's, and the gradient's over
/// three pixels, by which a step between two pixels reads two wide.
/// Taken from what is measured of symbols rendered sharp.
const OWN_VARIANCE: f64 = SMOOTHING * SMOOTHING + 0.64;
/// The share of the steepest change in the window that an edge must
/// come to, to be measured: fainter changes are grain.
const EDGE_SHARE: f32 = 0.15;
/// The share of the window's contrast, from its darkest to its lightest
/// few pixels, that the step across an edge must come to: grain on the
/// paper changes steeply over a pixel and by a few hundredths, and read
/// as an edge it is the narrowest of all, where an edge between ink and
/// paper steps by most of the contrast.
const STEP_SHARE: f32 = 0.5;
/// The share of each direction's edges, narrowest first, at which the
/// blur is read off them: the middle one.
const NARROWEST: f64 = 0.5;
/// The share of the step across an edge's window by which the light must
/// turn back, from its lightest beyond the edge or its darkest before it,
/// for a neighbouring edge to be taken to begin there ([`alone`]): more
/// than grain moves it.
const TURN_SHARE: f32 = 0.05;
/// The variance, in square pixels, that the gradient's three pixels add to
/// an edge's steepness beyond what the light read across it has
/// ([`alone`]): of what the measure gives an edge drawn sharp
/// ([`OWN_VARIANCE`]), all but the smoothing's and the pixel's own twelfth.
const GRADIENT_OWN: f64 = OWN_VARIANCE - SMOOTHING * SMOOTHING - 1.0 / 12.0;
/// The most of an edge's steepness its neighbours may take for its width to
/// be read ([`alone`]). Nearer, the step and the steepness both turn on
/// the edges beyond the neighbours, which the model of an edge between two
/// does not hold: over symbols blurred by eight tenths of a module, the
/// edges whose neighbours took up to half their steepness read the blur a
/// fifth over, and those that took a tenth, a twentieth.
const MOST_TAKEN: f64 = 0.1;
/// Lowest light the deconvolution divides by: what a hundredth stored
/// stands for ([`super::sample::linear_light`]), the floor it has where
/// it works on stored values ([`STORED_FLOOR`]). A hundredth of light is a tenth stored, which
/// would have held the ink of a dark photo off black.
pub(crate) const FLOOR: f32 = 0.01 / 12.92;
/// Lowest value the deconvolution divides by where it works on values as
/// stored ([`Light::Encoded`]).
const STORED_FLOOR: f32 = 0.01;

/// The lowest value the deconvolution divides by, worked in `light`.
fn floor_in(light: Light) -> f32 {
    match light {
        Light::Linear => FLOOR,
        Light::Encoded => STORED_FLOOR,
    }
}

/// `data`, `width` by `height`, stored as a photo stores it, in `light`:
/// as the light it stands for, or as it is.
fn into_light(data: &[f32], width: usize, height: usize, light: Light) -> Cow<'_, [f32]> {
    match light {
        Light::Linear => Cow::Owned(gpu::in_light(data, width, height)),
        Light::Encoded => Cow::Borrowed(data),
    }
}

/// `data`, `width` by `height`, in `light`, stored again as a photo
/// stores it ([`into_light`]).
fn stored(data: Vec<f32>, width: usize, height: usize, light: Light) -> Vec<f32> {
    match light {
        Light::Linear => gpu::encoded(&data, width, height),
        Light::Encoded => data,
    }
}

/// [`into_light`] of an image on the graphics card.
fn image_into_light(image: &gpu::Image, light: Light) -> gpu::Image {
    match light {
        Light::Linear => image.in_light(),
        Light::Encoded => image.clone(),
    }
}

/// [`stored`] of an image on the graphics card.
fn image_stored(image: gpu::Image, light: Light) -> gpu::Image {
    match light {
        Light::Linear => image.encoded(),
        Light::Encoded => image,
    }
}

/// How far the photo `width` by `height`, its values stored as a photo
/// stores them, spreads an edge, measured from the edges it shows in
/// `light`, as the module describes. `None` when it shows too
/// few edges to say ([`LEAST_EDGES`]), or no spread at all, none so wide
/// as the narrowest kernel undone ([`KERNEL_LEAST`]): so a photo drawn
/// sharp to the pixel, and one of a single grey, whose rounding leaves
/// differences of a few millionths between its pixels that the gradient
/// takes for edges of no width.
pub fn measure(data: &[f32], width: usize, height: usize, light: Light) -> Option<Spread> {
    measure_of(Values::Cores(data), width, height, light)
}

/// [`measure`] of an image on the graphics card, or held by the cores.
fn measure_of(image: Values<'_>, width: usize, height: usize, light: Light) -> Option<Spread> {
    if width < 8 || height < 8 {
        return None;
    }
    measured_over(image, width, height, &[[0, 0, width, height]], light)
}

/// How far the image `width` by `height` spreads an edge, as the module
/// describes, from the edges in `parts` of it, boxes `[x0, y0, x1, y1]`
/// of its pixels: the whole of it, or the parts of a large image
/// [`spread_of`] picks, each read with the pixels about it as the whole
/// image would read them, and the steepest change and the contrast that
/// an edge is judged against taken over them all. The edges are read on
/// the graphics card ([`gpu::edges`]), from the image it holds or from
/// the parts of it handed it, in `light`, and only their widths taken
/// off it.
fn measured_over(image: Values<'_>, width: usize, height: usize, parts: &[[usize; 4]], light: Light) -> Option<Spread> {
    let edges = gpu::edges::Edges::of(image, (width, height), parts, PART_MARGIN, &kernel(&Spread::round(SMOOTHING)), light == Light::Linear);
    let (changing, ranked) = edges.ranked(|n| (n * 99 / 100).min(n.saturating_sub(1)), |m| m / 50, |m| m * 49 / 50);
    if changing < LEAST_EDGES {
        return None;
    }
    let [steepest, darkest, lightest] = ranked?;
    let floor = EDGE_SHARE * steepest;
    let least_step = STEP_SHARE * (lightest - darkest);
    // Each edge's width and the direction across it, measured with the
    // step read `reach` pixels either side, in no order. The step is read
    // over twice the reach at the most, and an edge that ramps the whole
    // of the widest reach is that wide; one measured wider is no edge,
    // but a step taken from whatever lies about a pixel barely steeper
    // than its neighbours, as a grey dithered to pixel-sized checks is
    // once smoothed, where widths of a hundred thousand pixels were read.
    // Held to the reach of the look that measured them, the first look's
    // few pixels lost the wide edges that say to look further, and a
    // photograph blurred by many pixels was measured as barely blurred.
    // Each read as wide as it would be alone ([`alone`]).
    let widths = |reach: f64| -> Vec<(f64, f64)> {
        let measured = edges.widths(reach, floor, least_step, TURN_SHARE);
        let chunks: Vec<&[gpu::edges::Crossing]> = measured.chunks(4096).collect();
        crate::work::map(&chunks, |chunk| {
            chunk
                .iter()
                .filter_map(|crossing| {
                    let w = alone(crossing).filter(|&w| w <= 2.0 * MOST_REACH + 2.0)?;
                    let [ux, uy] = crossing.across.map(f64::from);
                    Some((w, uy.atan2(ux).rem_euclid(std::f64::consts::PI)))
                })
                .collect::<Vec<_>>()
        })
        .concat()
    };
    // A first look with a reach of a few pixels says how wide the edges
    // are; the next reads each step over as far as such an edge reaches,
    // with room for the light to turn back where a neighbouring edge lies,
    // and again while that moves the reach.
    let mut reach = 8.0;
    let mut measured = widths(reach);
    for _ in 0..2 {
        let typical = narrowest_share(measured.iter().map(|w| w.0).collect(), 0.5)?;
        let wider = (1.5 * typical).clamp(4.0, MOST_REACH);
        if (wider - reach).abs() < 1.0 {
            break;
        }
        reach = wider;
        measured = widths(reach);
    }
    if measured.len() < LEAST_EDGES {
        return None;
    }
    // The narrowest of each direction's edges, as a variance of the blur.
    let mut bins: Vec<Vec<f64>> = vec![Vec::new(); DIRECTIONS];
    for &(w, direction) in &measured {
        let bin = ((direction / std::f64::consts::PI * DIRECTIONS as f64) as usize).min(DIRECTIONS - 1);
        bins[bin].push(w);
    }
    let rows: Vec<(usize, [f64; 3], f64, f64)> = bins
        .into_iter()
        .enumerate()
        .filter(|(_, widths)| widths.len() >= LEAST_EDGES)
        .filter_map(|(bin, widths)| {
            let count = widths.len() as f64;
            let w = narrowest_share(widths, NARROWEST)?;
            let theta = (bin as f64 + 0.5) * std::f64::consts::PI / DIRECTIONS as f64;
            let (c, s) = (theta.cos(), theta.sin());
            // The spread along the edge's normal, less the measure's own.
            let variance = (w * w / (2.0 * std::f64::consts::PI) - OWN_VARIANCE).max(0.0);
            Some((bin, [c * c, 2.0 * c * s, s * s], variance, count))
        })
        .collect();
    // How far round the half turn the directions reach ([`LEAST_FAN`]):
    // all of it but the widest gap between two of them, the gap from the
    // last back round to the first among them.
    let gaps = rows.windows(2).map(|pair| pair[1].0 - pair[0].0);
    let round_again = rows.first().zip(rows.last()).map(|(first, last)| DIRECTIONS + first.0 - last.0);
    let fan = gaps.chain(round_again).max().map_or(0, |widest| DIRECTIONS - widest);
    let some = |spread: Spread| (spread.extremes().0 >= KERNEL_LEAST).then_some(spread);
    // The same every way, from every edge.
    let round = || {
        let all: Vec<f64> = measured.iter().map(|w| w.0).collect();
        let w = narrowest_share(all, NARROWEST)?;
        let variance = w * w / (2.0 * std::f64::consts::PI) - OWN_VARIANCE;
        (variance > 0.0).then(|| Spread::round(variance.sqrt())).and_then(some)
    };
    if rows.len() < 3 || fan < LEAST_FAN {
        // Too few directions, or too narrow a fan of them, to tell a
        // shape.
        return round();
    }
    // Least squares, each direction weighed by its edges.
    let mut a = [[0.0f64; 3]; 3];
    let mut b = [0.0f64; 3];
    for (_, row, value, weight) in &rows {
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] += weight * row[i] * row[j];
            }
            b[i] += weight * row[i] * value;
        }
    }
    let [xx, xy, yy] = solve3(a, b)?;
    // A covariance is never less than nought either way.
    let spread = Spread { xx, xy, yy };
    let mean = (xx + yy) / 2.0;
    let half = (((xx - yy) / 2.0).powi(2) + xy * xy).sqrt();
    let (big, small) = ((mean + half).max(0.0), (mean - half).max(0.0));
    if big <= 0.0 {
        return None;
    }
    // The spread along any way is the covariance's between its widest and
    // its narrowest, and over a fan of a quarter turn and more one of the
    // directions measured lies within an eighth of a turn of the widest,
    // where it reads at least half of it: a fit wider than twice the
    // widest direction measured is the least squares running off where
    // the directions say nothing, as over a field of pixel-sized checks,
    // whose edges every way are of no width at all. One such once put a
    // blur of a hundred thousand pixels on a dithered grey, and the
    // reduction it called for asked the card for forty gigabytes. Twice is
    // the bound exactly where the fan is a quarter turn, the widest way
    // midway across the quarter it leaves, and the blur a line, and the
    // widths read are only so near: a narrow shake so lain can read past
    // it, and such a fit is taken as no shape at all, the same every way,
    // as a fan too narrow is. It was once taken as no blur.
    let widest_measured = rows.iter().map(|row| row.2).fold(0.0, f64::max);
    if big > 2.0 * widest_measured {
        return round();
    }
    // The eigenvalues held at nought and above, the axes kept.
    let angle = 0.5 * (2.0 * spread.xy).atan2(spread.xx - spread.yy);
    let (c, s) = (angle.cos(), angle.sin());
    some(Spread { xx: big * c * c + small * s * s, xy: (big - small) * c * s, yy: big * s * s + small * c * c })
}

/// The width an edge would read at alone, `σ√(2π)` of the spread `σ` of
/// its steepness, from what is read across it ([`gpu::edges::Crossing`]).
/// Where the light turned back beyond the edge, a band of the other colour
/// lies there, and its far edge about twice as far out, as the turn lies
/// midway between two edges of a blur; where it turned before it,
/// likewise. Between two such edges the band never reaches its full
/// light or dark, and the step read falls short, where the steepness at
/// the edge barely falls: the step over the steepness of an edge between
/// two others at those places, as a Gaussian of spread `σ` blurs them, is
/// solved for `σ`. Over symbols blurred in light by from three to eight
/// tenths of a module, the middle of the widths so read came within a
/// twentieth of the blur, where the step itself over the steepness read
/// from a tenth to a quarter under it. `None` where no spread gives what
/// was read, or the neighbours take more than [`MOST_TAKEN`] of the
/// steepness.
fn alone(crossing: &gpu::edges::Crossing) -> Option<f64> {
    let read = f64::from(crossing.width);
    let [beyond, before] = crossing.out.map(f64::from);
    let neighbour = |out: f64, turned: bool| turned.then_some(2.0 * out);
    let (far, near) = (neighbour(beyond, crossing.turned[0]), neighbour(before, crossing.turned[1]));
    let taken = |s: f64| (far.map_or(0.0, |d| normal_pdf(d / s)) + near.map_or(0.0, |d| normal_pdf(d / s))) / normal_pdf(0.0);
    // The step over the steepness of an edge whose steepness spreads `s`,
    // the light itself less by what the gradient adds.
    let modelled = |s: f64| -> f64 {
        let light = (s * s - GRADIENT_OWN).max(1e-4).sqrt();
        let at = |x: f64| {
            normal_cdf(x / light) + 1.0 - far.map_or(0.0, |d| normal_cdf((x - d) / light)) - near.map_or(1.0, |d| normal_cdf((x + d) / light))
        };
        s * (at(beyond) - at(-before)) / (normal_pdf(0.0) * (1.0 - taken(s)))
    };
    // A spread past which either neighbour alone takes more than is let is
    // no answer.
    let nearest = (-2.0 * MOST_TAKEN.ln()).sqrt();
    let mut high = 4.0 * beyond.max(before).max(1.0);
    for d in [far, near].into_iter().flatten() {
        high = high.min(d / nearest);
    }
    let low = GRADIENT_OWN.sqrt();
    let (above, below) = (modelled(high) - read, modelled(low) - read);
    if high <= low || above < 0.0 {
        return None;
    }
    let s = if below >= 0.0 { low } else { root_between((low, below), (high, above), |s| modelled(s) - read) };
    (taken(s) <= MOST_TAKEN).then(|| s * (2.0 * std::f64::consts::PI).sqrt())
}

/// Where `f`, rising, crosses nought between `low`, where it is below,
/// and `high`, where it is above, each given with `f` there, to within a
/// millionth of a pixel: by false position, the Illinois way, which
/// halves what is kept of an end that holds twice running. Halving the
/// span forty times, the widths of a photo of a megapixel and a half took
/// twice as long to solve for as to read on the card.
fn root_between((mut low, mut f_low): (f64, f64), (mut high, mut f_high): (f64, f64), f: impl Fn(f64) -> f64) -> f64 {
    let mut kept = 0i8;
    for _ in 0..60 {
        if high - low < 1e-6 {
            break;
        }
        let at = (low * f_high - high * f_low) / (f_high - f_low);
        let f_at = f(at);
        if f_at < 0.0 {
            (low, f_low) = (at, f_at);
            if kept < 0 {
                f_high /= 2.0;
            }
            kept = -1;
        } else {
            (high, f_high) = (at, f_at);
            if kept > 0 {
                f_low /= 2.0;
            }
            kept = 1;
        }
    }
    (low + high) / 2.0
}

/// The standard normal's density at `x`.
fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// The standard normal's distribution below `x`.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// The error function, to within two ten-millionths: Abramowitz and
/// Stegun's 7.1.26.
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let polynomial = ((((1.061_405_429 * t - 1.453_152_027) * t + 1.421_413_741) * t - 0.284_496_736) * t + 0.254_829_592) * t;
    (1.0 - polynomial * (-x * x).exp()).copysign(x)
}

/// The value below which `share` of `values` lie; `None` for none.
fn narrowest_share(mut values: Vec<f64>, share: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let k = ((values.len() as f64 * share) as usize).min(values.len() - 1);
    Some(*values.select_nth_unstable_by(k, f64::total_cmp).1)
}

/// Three equations in three unknowns, by elimination; `None` where they
/// do not settle them.
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
                for (value, pivot) in a[row].iter_mut().zip(pivot_row) {
                    *value -= f * pivot;
                }
                b[row] -= f * b[col];
            }
        }
    }
    Some([b[0] / a[0][0], b[1] / a[1][1], b[2] / a[2][2]])
}

/// Least spread, in pixels, of a kernel along each of its axes: held
/// away from flat, so that it can be inverted.
pub const KERNEL_LEAST: f64 = 0.3;
/// Least spread, in pixels, of a kernel across its narrowest way, where
/// it is no less along its widest: a Gaussian narrower than half a pixel
/// is not one on the pixels' grid. Laid at a slant and sampled at whole
/// pixels, its taps fall as the grid falls across it, and spread along
/// it further or less far than it does: a shake at forty-five degrees,
/// flat across, was once a line of taps two and a half times as long as
/// the shake, where laid along either axis it was right.
pub const KERNEL_LEAST_ACROSS: f64 = 0.5;

/// The kernel of `spread`: its taps as (dx, dy, weight), the weights
/// summing to one, out to three spreads along its widest way and
/// leaving out those too faint to count. Each axis is held to
/// [`KERNEL_LEAST`] and the narrowest to [`KERNEL_LEAST_ACROSS`], or to
/// the widest where that is less, so that the taps spread as `spread`
/// does whichever way it lies, and a spread that is next to nothing
/// either way is next to a single tap.
pub fn kernel(spread: &Spread) -> Vec<(isize, isize, f32)> {
    let (widest, narrowest) = spread.extremes();
    let big = widest.max(KERNEL_LEAST).powi(2);
    let small = narrowest.max(KERNEL_LEAST).powi(2).max(KERNEL_LEAST_ACROSS.powi(2).min(big));
    // The axes held, turned back the way the spread lies.
    let (s, c) = spread.angle().to_radians().sin_cos();
    let (xx, xy, yy) = (big * c * c + small * s * s, (big - small) * c * s, big * s * s + small * c * c);
    let det = xx * yy - xy * xy;
    let (ixx, iyy, ixy) = (yy / det, xx / det, -xy / det);
    let reach = (3.0 * big.sqrt()).ceil() as isize;
    let mut taps = Vec::new();
    for dy in -reach..=reach {
        for dx in -reach..=reach {
            let (fx, fy) = (dx as f64, dy as f64);
            let w = (-0.5 * (ixx * fx * fx + 2.0 * ixy * fx * fy + iyy * fy * fy)).exp();
            if w > 1e-3 {
                taps.push((dx, dy, w as f32));
            }
        }
    }
    let total: f32 = taps.iter().map(|t| t.2).sum();
    for tap in &mut taps {
        tap.2 /= total;
    }
    taps
}

/// Rows worked as one piece on the cores to spare.
#[cfg(test)]
const BAND: usize = 32;

/// `data` blurred by `taps`, the pixel on the image's edge standing for
/// those beyond it, on the cores: what the tests blur a sharp image by.
#[cfg(test)]
fn blurred(data: &[f32], width: usize, height: usize, taps: &[(isize, isize, f32)]) -> Vec<f32> {
    let bands: Vec<(usize, usize)> = (0..height).step_by(BAND).map(|y| (y, (y + BAND).min(height))).collect();
    let (w, h) = (width as isize, height as isize);
    crate::work::map(&bands, |&(y0, y1)| {
        let mut out = vec![0.0f32; (y1 - y0) * width];
        for y in y0..y1 {
            let row = &mut out[(y - y0) * width..(y - y0 + 1) * width];
            for &(dx, dy, weight) in taps {
                let sy = (y as isize + dy).clamp(0, h - 1) as usize;
                let source = &data[sy * width..(sy + 1) * width];
                // The pixels whose tap lies on the row, and those before
                // and after them, whose tap is the pixel on its edge.
                let lo = (-dx).clamp(0, w) as usize;
                let hi = (w - dx).clamp(lo as isize, w) as usize;
                for value in &mut row[..lo] {
                    *value += weight * source[0];
                }
                if hi > lo {
                    let from = (lo as isize + dx) as usize;
                    for (value, &v) in row[lo..hi].iter_mut().zip(&source[from..]) {
                        *value += weight * v;
                    }
                }
                for value in &mut row[hi..] {
                    *value += weight * source[width - 1];
                }
            }
        }
        out
    })
    .concat()
}

/// `data`, `width` by `height`, light, with the blur `spread` undone by
/// Richardson and Lucy's deconvolution, as the module describes, held
/// to black and white: on the graphics card ([`super::gpu`]), held to
/// [`FLOOR`] and above there, fewer rounds where the work is called off.
/// What a photo stores is undone by [`deconvolved_stored`].
pub fn deconvolve(data: &[f32], width: usize, height: usize, spread: &Spread) -> Vec<f32> {
    super::gpu::deconvolved(data, width, height, &kernel(spread), ROUNDS, FLOOR)
}

/// `data`, `width` by `height`, stored as a photo stores it, with the blur
/// `spread` undone in `light` ([`deconvolve`]), and stored again.
fn deconvolved_stored(data: &[f32], width: usize, height: usize, spread: &Spread, light: Light) -> Vec<f32> {
    let lit = into_light(data, width, height, light);
    stored(super::gpu::deconvolved(&lit, width, height, &kernel(spread), ROUNDS, floor_in(light)), width, height, light)
}

/// How far a deconvolution reduces an image before it undoes the blur in
/// it: to `p` of its pixels for every `q` of the image's, each way, on
/// a grid laid from the image's first pixel. A whole number of the
/// image's pixels to a whole number of the copy's, so that a part of the
/// image beginning a whole number of steps of `q` in is reduced among
/// the pixels the whole is ([`super::resize`] centres each pixel of the
/// copy where it centres it in the whole), and comes back as the whole
/// does there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reduction {
    pub p: usize,
    pub q: usize,
}

/// Most of an image's pixels a step of a reduction spans, short of a
/// reduction further than one in so many ([`Reduction::for_blur`]).
const MOST_STEP: usize = 8;
/// Fewest pixels along its shorter side an image reduced to undo its blur
/// may come to ([`Reduction::keeps`]): a blur so wide that undoing it would
/// leave fewer is wider than anything in the image a read could use, and
/// the image is carried to a whole number of the reduction's steps first,
/// so that a reduction of one in a hundred thousand once carried a copy
/// three hundred pixels a side to a hundred thousand a side.
const LEAST_REDUCED_SIDE: usize = 16;

impl Reduction {
    /// No reduction: the blur is undone where it lies.
    pub const NONE: Reduction = Reduction { p: 1, q: 1 };

    /// The reduction a blur spreading `widest` pixels along its widest
    /// way is undone at, in an image of `pixels`, or a part of one so
    /// large: as far as brings the blur to [`MOST_SPREAD`] pixels and the
    /// image to `most_pixels` ([`MOST_PIXELS`] for a copy the search
    /// looks through, [`MOST_READ_PIXELS`] otherwise), and to `most`
    /// times the image's own size, and no further than the largest ratio
    /// short of that whose step is no more than [`MOST_STEP`] pixels, or
    /// one in as many as it takes.
    pub fn for_blur(widest: f64, pixels: usize, most_pixels: usize, most: f64) -> Self {
        let times = (MOST_SPREAD / widest).min((most_pixels as f64 / pixels.max(1) as f64).sqrt()).min(most);
        if times.is_nan() || times >= 1.0 {
            return Self::NONE;
        }
        if times * (MOST_STEP as f64) < 1.0 {
            return Self { p: 1, q: (1.0 / times).ceil() as usize };
        }
        // Of every ratio at or short of `times`, the largest, in its least
        // terms: a larger step is taken only for a larger ratio.
        let mut best = Self { p: 1, q: MOST_STEP };
        for q in 1..=MOST_STEP {
            let p = (times * q as f64 + 1e-9).floor() as usize;
            if p >= 1 && p * best.q > best.p * q {
                best = Self { p, q };
            }
        }
        best
    }

    /// The copy's pixels to one of the image's.
    pub fn factor(self) -> f64 {
        self.p as f64 / self.q as f64
    }

    /// Whether an image `width` by `height` reduced so keeps
    /// [`LEAST_REDUCED_SIDE`] pixels along its shorter side.
    fn keeps(self, width: usize, height: usize) -> bool {
        width.min(height).saturating_mul(self.p) >= LEAST_REDUCED_SIDE.saturating_mul(self.q)
    }

    /// The size an image `width` by `height` is carried to before it is
    /// reduced ([`reduced`]): a whole number of steps of `q` each way.
    fn carried(self, width: usize, height: usize) -> (usize, usize) {
        (width.div_ceil(self.q) * self.q, height.div_ceil(self.q) * self.q)
    }
}

/// `data`, `width` by `height`, reduced by `reduction` on its grid
/// ([`Reduction`]): carried past its far edges by the pixels on them to a
/// whole number of steps, as the resizing reads past an edge, and resized
/// by exactly `p` in `q`. With the size it comes to.
fn reduced(data: &[f32], width: usize, height: usize, reduction: Reduction) -> (Vec<f32>, usize, usize) {
    let Reduction { p, q } = reduction;
    let (ew, eh) = reduction.carried(width, height);
    let carried: Cow<[f32]> = if (ew, eh) == (width, height) {
        Cow::Borrowed(data)
    } else {
        Cow::Owned(
            (0..eh)
                .flat_map(|y| {
                    let row = &data[y.min(height - 1) * width..(y.min(height - 1) + 1) * width];
                    row.iter().copied().chain(std::iter::repeat_n(row[width - 1], ew - width))
                })
                .collect(),
        )
    };
    let (rw, rh) = (ew / q * p, eh / q * p);
    (resize(&carried, ew, eh, 1, rw, rh), rw, rh)
}

/// `sharp`, `rw` by `rh`, the image `width` by `height` reduced by
/// `reduction` ([`reduced`]), enlarged back by exactly `q` in `p` and
/// cut to the image's size.
fn restored(sharp: &[f32], (rw, rh): (usize, usize), reduction: Reduction, (width, height): (usize, usize)) -> Vec<f32> {
    let Reduction { p, q } = reduction;
    let (ew, eh) = (rw / p * q, rh / p * q);
    let back = resize(sharp, rw, rh, 1, ew, eh);
    if ew == width {
        back[..width * height].to_vec()
    } else {
        (0..height).flat_map(|y| back[y * ew..y * ew + width].iter().copied()).collect()
    }
}

/// `data`, `width` by `height`, in the light it is worked in, reduced by
/// `reduction` ([`reduced`]) and the blur `at_size`, the blur at the
/// reduced size, undone in it, held to `floor` ([`floor_in`]): a step at
/// a time, each handed the graphics card and taken off it in turn. With
/// the size it comes to.
fn reduced_deconvolved(data: &[f32], width: usize, height: usize, at_size: &Spread, reduction: Reduction, floor: f32) -> (Vec<f32>, usize, usize) {
    let (small, rw, rh) = reduced(data, width, height, reduction);
    (super::gpu::deconvolved(&small, rw, rh, &kernel(at_size), ROUNDS, floor), rw, rh)
}

/// [`reduced_deconvolved`] of an image on the graphics card, left there:
/// carried and resized as [`reduced`] carries and resizes it, and
/// deconvolved as [`reduced_deconvolved`] deconvolves it, down to the
/// same `floor`, to the same values. Not as [`deconvolve`] deconvolves
/// an image, which always floors it as light does ([`FLOOR`]): an image
/// as stored is floored at [`STORED_FLOOR`].
fn reduced_deconvolved_on(image: &gpu::Image, at_size: &Spread, reduction: Reduction, floor: f32) -> gpu::Image {
    let Reduction { p, q } = reduction;
    let (ew, eh) = reduction.carried(image.width, image.height);
    image.carried(ew, eh).resized(ew / q * p, eh / q * p).deconvolved(&kernel(at_size), ROUNDS, floor)
}

/// How many of the deconvolution's kernels a part of an image must reach
/// past where it is wanted for it to come out there as it does in the
/// whole image, to well within a grey level: what lies further affects
/// it by less than that over the deconvolution's rounds, where the
/// rounds themselves reach a kernel further each time.
const DECONVOLVED_REACH: f64 = 4.0;
/// And the reduced pixels besides that the reduction and the enlargement
/// back draw on past a pixel: half the kernel of [`super::resize`], and
/// one over.
const RESIZED_REACH: f64 = 4.0;

/// How far past a part of an image, in the image's pixels, the
/// deconvolution of the image by `spread` at `strength` under `reduction`
/// reads it for the part to come out as it does in the whole
/// ([`DECONVOLVED_REACH`]).
pub fn reach(spread: &Spread, strength: f64, reduction: Reduction) -> f64 {
    let (widest, _) = spread.scaled(strength.max(0.0) * reduction.factor()).extremes();
    let kernel = (3.0 * widest.max(KERNEL_LEAST)).ceil();
    (DECONVOLVED_REACH * kernel + RESIZED_REACH) / reduction.factor()
}

/// The image `width` by `height`, deblurred: its blur measured, and, when
/// it is worth undoing ([`LEAST_SPREAD`]), undone at a size that holds it
/// to [`MOST_SPREAD`] pixels and the copy to `most_pixels`, but never
/// larger than the image's own size ([`Reduction`]); in `light`, and the
/// copy stored again, as the module describes. The copy, covering
/// what the image covers, with how many times the image's size it is,
/// and the blur measured carried to that size; `None` where the image
/// shows no blur worth undoing, or too few edges to measure one, or is
/// too small to keep [`LEAST_REDUCED_SIDE`] pixels a side reduced as far
/// as its blur asks.
pub fn deblurred(data: &[f32], width: usize, height: usize, most_pixels: usize, light: Light) -> Option<(Vec<f32>, usize, usize, f64, Spread)> {
    let spread = spread_of(data, width, height, light)?;
    let (widest, _) = spread.extremes();
    if widest < LEAST_SPREAD {
        return None;
    }
    let reduction = Reduction::for_blur(widest, width * height, most_pixels, 1.0);
    if reduction == Reduction::NONE {
        return Some((deconvolved_stored(data, width, height, &spread, light), width, height, 1.0, spread));
    }
    if !reduction.keeps(width, height) {
        return None;
    }
    let at_size = spread.scaled(reduction.factor());
    let floor = floor_in(light);
    // As far as the image itself reaches: made light, reduced, undone, cut
    // and stored again on the card, handed it once and taken off it once,
    // where it holds the image whole; and otherwise a step at a time.
    let (w, h) = ((width * reduction.p).div_ceil(reduction.q), (height * reduction.p).div_ceil(reduction.q));
    let sharp = gpu::through(data, (width, height), reduction.carried(width, height), |image| {
        image_stored(reduced_deconvolved_on(&image_into_light(&image, light), &at_size, reduction, floor).carried(w, h), light)
    })
    .unwrap_or_else(|| {
        let (sharp, rw, _) = reduced_deconvolved(&into_light(data, width, height, light), width, height, &at_size, reduction, floor);
        let cut: Vec<f32> = (0..h).flat_map(|y| sharp[y * rw..y * rw + w].iter().copied()).collect();
        stored(cut, w, h, light)
    });
    Some((sharp, w, h, reduction.factor(), at_size))
}

/// How far the image `width` by `height` spreads an edge, in its own
/// pixels, in `light` ([`measure`]): measured
/// whole, or, for an image of more than [`MOST_PIXELS`], over the parts
/// of it that show the most edges ([`busiest_parts`]), at its own size.
/// `None` where it shows no blur, or too few edges to say.
pub fn spread_of(data: &[f32], width: usize, height: usize, light: Light) -> Option<Spread> {
    spread_of_values(Values::Cores(data), width, height, light)
}

/// [`spread_of`] of an image on the graphics card, or held by the cores.
fn spread_of_values(image: Values<'_>, width: usize, height: usize, light: Light) -> Option<Spread> {
    if width * height <= MOST_PIXELS {
        return measure_of(image, width, height, light);
    }
    if width < 8 || height < 8 {
        return None;
    }
    measured_over(image, width, height, &busiest_parts(image, width, height), light)
}

/// The parts of the image `width` by `height`, squares [`PART_SIDE`]
/// pixels a side on a grid from its corner, that change the most across
/// their edges, the busiest first, as many as come to [`MOST_PIXELS`]:
/// the symbols and the print about them, where a photograph is mostly
/// paper, desk and wall. A part is judged by how far its light changes
/// over two pixels each way at every other pixel of every other row,
/// added up over the changes as steep as an edge must be to be measured
/// ([`EDGE_SHARE`] of the steepest), which grain does not reach. However
/// far an edge is blurred, the light changes across it by the same in
/// all, so a blur that spreads it wide does not hide it.
fn busiest_parts(image: Values<'_>, width: usize, height: usize) -> Vec<[usize; 4]> {
    let across = width.div_ceil(PART_SIDE);
    let busy = gpu::edges::busy(image, (width, height), PART_SIDE, EDGE_SHARE);
    let mut order: Vec<usize> = (0..busy.len()).filter(|&i| busy[i] > 0.0).collect();
    order.sort_by(|&a, &b| busy[b].total_cmp(&busy[a]).then(a.cmp(&b)));
    let mut parts = Vec::new();
    let mut pixels = 0;
    for i in order {
        if pixels >= MOST_PIXELS {
            break;
        }
        let (x0, y0) = (i % across * PART_SIDE, i / across * PART_SIDE);
        let part = [x0, y0, (x0 + PART_SIDE).min(width), (y0 + PART_SIDE).min(height)];
        pixels += (part[2] - part[0]) * (part[3] - part[1]);
        parts.push(part);
    }
    parts
}

/// `data`, `width` by `height`, deblurred in `light` and stored again, at
/// its own size, as the photo's Deblur filter shows it
/// ([`super::adjust::Filter::Deblur`]): by `kernel`, where one is given,
/// or by the blur the image shows,
/// measured over the whole of it, either spread `strength` times as
/// wide, however little that is, since the filter was asked for; at a
/// size that holds it to [`MOST_SPREAD`] pixels and the copy to
/// [`MOST_READ_PIXELS`] ([`Reduction`]), and enlarged back. With the deblurred
/// image, the spread before its strength: `None`, and the image as it
/// is, where it shows no blur, or too few edges to measure one, or a blur
/// too wide to undo in so few pixels ([`LEAST_REDUCED_SIDE`]); and the
/// image as it is at a strength of nought.
pub fn undone_by(data: &[f32], width: usize, height: usize, kernel: Option<Spread>, strength: f64, light: Light) -> (Vec<f32>, Option<Spread>) {
    undone_within(data, width, height, kernel, strength, (width, height), light)
}

/// [`undone_by`] of an image that is a part of one `whole` pixels across
/// and down, as the whole is deblurred: reduced as far as the whole is, on the grid
/// the whole is reduced on where the part begins a whole number of the
/// reduction's steps into it ([`Reduction`]), so that, given as much of
/// the whole about it as the deconvolution reads ([`reach`]), it comes
/// out as the whole does there.
pub fn undone_within(data: &[f32], width: usize, height: usize, kernel: Option<Spread>, strength: f64, whole: (usize, usize), light: Light) -> (Vec<f32>, Option<Spread>) {
    match undoing(kernel, strength, whole, || spread_of(data, width, height, light)) {
        Undoing::Nothing(spread) => (data.to_vec(), spread),
        Undoing::Undo { spread, undone, reduction: Reduction::NONE } => (deconvolved_stored(data, width, height, &undone, light), Some(spread)),
        Undoing::Undo { spread, undone, reduction } => {
            // Made light, reduced, undone, brought back and stored again on
            // the card, handed it once and taken off it once, where it holds
            // the image whole; and otherwise a step at a time.
            let (at_size, floor) = (undone.scaled(reduction.factor()), floor_in(light));
            let carried = reduction.carried(width, height);
            let sharp = gpu::through(data, (width, height), carried, |image| {
                image_stored(reduced_deconvolved_on(&image_into_light(&image, light), &at_size, reduction, floor).resized_part(carried, (width, height)), light)
            })
            .unwrap_or_else(|| {
                let (sharp, rw, rh) = reduced_deconvolved(&into_light(data, width, height, light), width, height, &at_size, reduction, floor);
                stored(restored(&sharp, (rw, rh), reduction, (width, height)), width, height, light)
            });
            (sharp, Some(spread))
        }
    }
}

/// [`undone_within`] of an image on the graphics card, left there: by the
/// same steps and the same kernels, to the same values ([`gpu::Image`]).
/// The blur it shows, where no kernel is `given`, is measured there too,
/// and the image is not taken off the card.
pub(crate) fn undone_on(image: &gpu::Image, given: Option<Spread>, strength: f64, whole: (usize, usize), light: Light) -> (gpu::Image, Option<Spread>) {
    let (width, height) = (image.width, image.height);
    let floor = floor_in(light);
    match undoing(given, strength, whole, || spread_of_values(Values::Card(image), width, height, light)) {
        Undoing::Nothing(spread) => (image.clone(), spread),
        Undoing::Undo { spread, undone, reduction: Reduction::NONE } => {
            (image_stored(image_into_light(image, light).deconvolved(&kernel(&undone), ROUNDS, floor), light), Some(spread))
        }
        Undoing::Undo { spread, undone, reduction } => {
            // Brought back as [`restored`] brings it: enlarged to the size
            // it was carried to, of which only the image's own is made.
            let sharp = reduced_deconvolved_on(&image_into_light(image, light), &undone.scaled(reduction.factor()), reduction, floor);
            (image_stored(sharp.resized_part(reduction.carried(width, height), (width, height)), light), Some(spread))
        }
    }
}

/// What undoing a blur in an image comes to before any pixel of it is
/// touched ([`undoing`]).
enum Undoing {
    /// Nothing: no blur shown, with too few edges to measure one, or one
    /// too wide to undo in so few pixels; or one so faint as next to none
    /// at the strength asked for, and the blur, before its strength.
    Nothing(Option<Spread>),
    /// The blur `spread`, `undone` as far as the strength asks, at
    /// `reduction`.
    Undo { spread: Spread, undone: Spread, reduction: Reduction },
}

/// What undoing the blur `given`, or that `measured` gives where none is
/// given, `strength` times as wide, comes to in an image that is a part of
/// one `whole` pixels across and down.
fn undoing(given: Option<Spread>, strength: f64, (whole_width, whole_height): (usize, usize), measured: impl FnOnce() -> Option<Spread>) -> Undoing {
    let Some(spread) = given.or_else(measured) else { return Undoing::Nothing(None) };
    let undone = spread.scaled(strength.max(0.0));
    let (widest, _) = undone.extremes();
    if widest <= 0.0 || !widest.is_finite() {
        return Undoing::Nothing(Some(spread));
    }
    // An image undone at its own size is undone however small, as
    // [`deblurred`] undoes one; only a reduction may leave too little, and
    // then no blur is undone, and none said to be: the filter once showed
    // the blur as undone over the image left as it was. A copy of it
    // enlarged does no better, the reduction taking the copy back down to
    // as few pixels as the blur grew wider. Its shorter side is the
    // whole's own, as [`deblurred`] takes it: it was once taken as the
    // side of a square of as many pixels, and a strip of a photo six
    // thousand pixels by twenty was reduced to a copy four rows tall.
    let reduction = Reduction::for_blur(widest, whole_width * whole_height, MOST_READ_PIXELS, 1.0);
    if reduction != Reduction::NONE && !reduction.keeps(whole_width, whole_height) {
        return Undoing::Nothing(None);
    }
    Undoing::Undo { spread, undone, reduction }
}

/// `data`, `width` by `height`, deblurred by the blur it shows, at full
/// strength, in `light` ([`undone_by`]).
pub fn undone(data: &[f32], width: usize, height: usize, light: Light) -> Vec<f32> {
    undone_by(data, width, height, None, 1.0, light).0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A field of squares `cell` pixels a side, dark and light by a fixed
    /// pattern, `size` pixels a side: edges every way.
    fn squares(size: usize, cell: usize) -> Vec<f32> {
        (0..size * size)
            .map(|i| {
                let (x, y) = (i % size / cell, i / size / cell);
                if (x * 7 + y * 13 + x * y) % 3 == 0 { 0.1 } else { 0.9 }
            })
            .collect()
    }

    #[test]
    fn a_dithered_grey_is_given_no_blur_wider_than_its_edges_say() {
        // A grey dithered to checks a pixel apart, and to one dark pixel in
        // four, as a print made in black alone or a screenshot of one lays
        // it down: the first once measured as a blur a hundred thousand
        // pixels wide, and the copy it was undone in as a hundred thousand
        // a side, which the card could not hold; the second ran the
        // processor out of memory carrying it.
        let checks = |size: usize, dark: &dyn Fn(usize, usize) -> bool| -> Vec<f32> {
            (0..size * size).map(|i| if dark(i % size, i / size) { 0.0 } else { 1.0 }).collect()
        };
        for size in [300, 1200] {
            for (name, image) in [
                ("checks", checks(size, &|x, y| (x + y) % 2 == 0)),
                ("one in four", checks(size, &|x, y| x % 2 == 0 && y % 2 == 0)),
            ] {
                if let Some(spread) = measure(&image, size, size, Light::Linear) {
                    assert!(spread.extremes().0 < 64.0, "{name} at {size}: {spread:?}");
                }
                if let Some((_, w, h, _, _)) = deblurred(&image, size, size, MOST_PIXELS, Light::Linear) {
                    assert!(w <= size && h <= size && w.min(h) >= LEAST_REDUCED_SIDE, "{name} at {size}: {w} by {h}");
                }
            }
        }
        // However wide a blur is given, the copy it is undone in keeps some
        // pixels, or it is not undone.
        assert!(!Reduction::for_blur(1e5, 300 * 300, MOST_PIXELS, 1.0).keeps(300, 300));
        assert!(Reduction::for_blur(16.0, 300 * 300, MOST_PIXELS, 1.0).keeps(300, 300));
    }

    /// `data` blurred by `taps` in the light its values stand for, as a
    /// lens blurs what it photographs, and stored again.
    fn blurred_in_light(data: &[f32], width: usize, height: usize, taps: &[(isize, isize, f32)]) -> Vec<f32> {
        use crate::imaging::sample::{encoded_light, linear_light};
        let light: Vec<f32> = data.iter().map(|&v| linear_light(v)).collect();
        blurred(&light, width, height, taps).into_iter().map(encoded_light).collect()
    }

    #[test]
    fn a_blur_is_measured_as_it_was_laid() {
        let (size, cell) = (240, 8);
        let sharp = squares(size, cell);
        assert_eq!(measure(&sharp, size, size, Light::Linear), None, "a sharp field shows no spread");
        for sigma in [2.0, 3.0] {
            let blurred_image = blurred_in_light(&sharp, size, size, &kernel(&Spread::round(sigma)));
            let (widest, narrowest) = measure(&blurred_image, size, size, Light::Linear).unwrap().extremes();
            assert!((widest - sigma).abs() < 0.35 * sigma && (narrowest - sigma).abs() < 0.35 * sigma, "{sigma}: {widest} by {narrowest}");
        }
        // A shake across: wide that way, and narrow down.
        let shaken = blurred_in_light(&sharp, size, size, &kernel(&Spread { xx: 9.0, xy: 0.0, yy: 0.36 }));
        let spread = measure(&shaken, size, size, Light::Linear).unwrap();
        assert!(spread.xx > 4.0 * spread.yy, "{spread:?}");
    }

    #[test]
    fn a_blur_laid_as_stored_is_measured_and_undone_as_stored() {
        // A blur a program laid on the values it stores, as one blurs an
        // image, or as a photo inverted for its light symbol shows a lens's
        // blur: worked as stored, it is measured as laid, and undone with
        // no turn into light and back; taken as light, its edges read
        // otherwise.
        let (size, cell) = (240, 8);
        let sharp = squares(size, cell);
        let sigma = 2.0;
        let soft = blurred(&sharp, size, size, &kernel(&Spread::round(sigma)));
        let (widest, narrowest) = measure(&soft, size, size, Light::Encoded).unwrap().extremes();
        assert!((widest - sigma).abs() < 0.35 * sigma && (narrowest - sigma).abs() < 0.35 * sigma, "{widest} by {narrowest}");
        assert_ne!(measure(&soft, size, size, Light::Linear), measure(&soft, size, size, Light::Encoded));
        let given = Spread::round(sigma);
        let undone = undone_by(&soft, size, size, Some(given), 1.0, Light::Encoded).0;
        assert_eq!(undone, super::gpu::deconvolved(&soft, size, size, &kernel(&given), ROUNDS, STORED_FLOOR));
        assert_ne!(undone, undone_by(&soft, size, size, Some(given), 1.0, Light::Linear).0);
    }

    #[test]
    fn an_image_too_small_to_reduce_is_undone_at_its_own_size() {
        // Fifteen pixels a side, short of the least side a reduction keeps:
        // undone at its own size, it keeps them all, and the filter undoes
        // it as a copy is undone. It was once left as it was, its blur
        // said to be undone all the same.
        let size = 15;
        assert!(size < LEAST_REDUCED_SIDE);
        let given = Spread::round(1.0);
        let soft = blurred(&squares(size, 3), size, size, &kernel(&given));
        for light in [Light::Linear, Light::Encoded] {
            let (undone, spread) = undone_by(&soft, size, size, Some(given), 1.0, light);
            assert_eq!(spread, Some(given));
            assert_eq!(undone, deconvolved_stored(&soft, size, size, &given, light), "{light:?}");
            assert_ne!(undone, soft, "{light:?}");
        }
        // A blur so wide that the copy reduced for it would keep too few
        // pixels is not undone, and no blur is said to be: it was once
        // said to be undone over the image left as it was.
        let (size, given) = (60, Spread::round(10.0));
        let reduction = Reduction::for_blur(10.0, size * size, MOST_READ_PIXELS, 1.0);
        assert!(reduction != Reduction::NONE && !reduction.keeps(size, size));
        let soft = blurred(&squares(size, 6), size, size, &kernel(&given));
        for light in [Light::Linear, Light::Encoded] {
            assert_eq!(undone_by(&soft, size, size, Some(given), 1.0, light), (soft.clone(), None), "{light:?}");
        }
        // Nor in a part of a whole too narrow for it, however many pixels
        // the whole holds: a strip six thousand pixels by twenty, reduced
        // to a fifth, keeps four rows. Taken for a square of as many
        // pixels, three hundred and forty-six a side, it once kept enough.
        let undoes = |whole| matches!(undoing(Some(given), 1.0, whole, || None), Undoing::Undo { reduction: Reduction { p: 1, q: 5 }, .. });
        assert!(!undoes((6000, 20)) && undoes((346, 346)));
    }

    #[test]
    fn undoing_a_blur_brings_the_edges_back() {
        let (size, cell) = (160, 8);
        let sharp = squares(size, cell);
        let spread = Spread::round(2.5);
        let blurred_image = blurred(&sharp, size, size, &kernel(&spread));
        let restored = deconvolve(&blurred_image, size, size, &spread);
        // How far the middle of each square reads from grey, and how many
        // read the wrong side of it: what a scan for finder patterns and
        // a sampler of modules each ask of a photo.
        let middles = |image: &[f32]| -> (f32, usize) {
            let (mut far, mut wrong) = (0.0, 0);
            for cy in 0..size / cell {
                for cx in 0..size / cell {
                    let i = (cy * cell + cell / 2) * size + cx * cell + cell / 2;
                    far += (image[i] - 0.5).abs();
                    wrong += usize::from((image[i] < 0.5) != (sharp[i] < 0.5));
                }
            }
            (far / ((size / cell) * (size / cell)) as f32, wrong)
        };
        let (blurred_far, blurred_wrong) = middles(&blurred_image);
        let (restored_far, restored_wrong) = middles(&restored);
        assert!(restored_far > 1.3 * blurred_far, "{restored_far} against {blurred_far}");
        assert!(restored_wrong <= blurred_wrong, "{restored_wrong} against {blurred_wrong}");
        // Flat light stays flat.
        let flat = vec![0.6f32; 64 * 64];
        assert!(deconvolve(&flat, 64, 64, &spread).iter().all(|v| (v - 0.6).abs() < 1e-4));
    }

    #[test]
    fn the_filter_undoes_what_it_is_given_as_far_as_it_is_asked() {
        let (size, cell) = (96, 8);
        let sharp = squares(size, cell);
        // Blurred by less than a read thinks worth undoing: the filter,
        // asked for, undoes it all the same, by the blur it measures,
        // which it gives back.
        let little = blurred(&sharp, size, size, &kernel(&Spread::round(0.8)));
        assert!(deblurred(&little, size, size, MOST_READ_PIXELS, Light::Linear).is_none(), "a read leaves it");
        let (undone, measured) = undone_by(&little, size, size, None, 1.0, Light::Linear);
        let measured = measured.expect("edges enough to measure");
        assert!(measured.extremes().0 < LEAST_SPREAD, "{:?}", measured.extremes());
        assert_eq!(undone, deconvolved_stored(&little, size, size, &measured, Light::Linear));
        assert_eq!(super::undone(&little, size, size, Light::Linear), undone);
        // A kernel given is undone in place of what the image shows, and
        // given back as it was; so many times as wide as the strength
        // asks; and at a strength of nought, not at all.
        let spread = Spread::round(2.0);
        let wide = blurred(&sharp, size, size, &kernel(&spread));
        let given = Spread { xx: 1.2, xy: 0.3, yy: 0.8 };
        assert!(given.scaled(1.5).extremes().0 <= MOST_SPREAD, "undone at the image's own size");
        assert_eq!(undone_by(&wide, size, size, Some(given), 1.0, Light::Linear), (deconvolved_stored(&wide, size, size, &given, Light::Linear), Some(given)));
        assert_eq!(undone_by(&wide, size, size, Some(given), 1.5, Light::Linear).0, deconvolved_stored(&wide, size, size, &given.scaled(1.5), Light::Linear));
        assert_eq!(undone_by(&wide, size, size, Some(given), 0.0, Light::Linear), (wide.clone(), Some(given)));
        // The further, the sharper the squares' middles read.
        let far = |image: &[f32]| {
            let middles: Vec<f32> = (0..size / cell).flat_map(|cy| (0..size / cell).map(move |cx| (cy * cell + cell / 2) * size + cx * cell + cell / 2)).map(|i| (image[i] - 0.5).abs()).collect();
            middles.iter().sum::<f32>() / middles.len() as f32
        };
        let fars: Vec<f32> = [0.0, 0.5, 1.0].iter().map(|&strength| far(&undone_by(&wide, size, size, Some(spread), strength, Light::Linear).0)).collect();
        assert!(fars.windows(2).all(|pair| pair[0] < pair[1]), "{fars:?}");
        // With no edges to measure a blur off, the image as it is: of a
        // single grey, however its rounding leaves its pixels, a few
        // millionths apart, which the gradient reads as edges of no width.
        for grey in [0.6f32, 128.0 / 255.0, 0.5] {
            let flat = vec![grey; 64 * 64];
            assert_eq!(measure(&flat, 64, 64, Light::Linear), None, "{grey}");
            assert_eq!(undone_by(&flat, 64, 64, None, 1.0, Light::Linear), (flat.clone(), None), "{grey}");
        }
    }

    #[test]
    fn a_kernel_spreads_as_asked_whichever_way_it_lies() {
        // What the taps themselves spread by, along their widest way and
        // their narrowest.
        let spread_of_taps = |taps: &[(isize, isize, f32)]| {
            let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
            for &(dx, dy, weight) in taps {
                let (x, y, weight) = (dx as f64, dy as f64, f64::from(weight));
                xx += weight * x * x;
                xy += weight * x * y;
                yy += weight * y * y;
            }
            Spread { xx, xy, yy }.extremes()
        };
        // Round, long and thin, and flat across, every way round: as wide
        // as asked along the widest way, and across it as asked or half a
        // pixel, whichever is the more.
        for sigma in [1.0, 2.0, 3.0] {
            for across in [0.0, 0.2, 0.8, sigma] {
                for degrees in (0..180).step_by(15) {
                    let (s, c) = f64::from(degrees).to_radians().sin_cos();
                    let (big, small) = (sigma * sigma, across * across);
                    let spread = Spread { xx: big * c * c + small * s * s, xy: (big - small) * c * s, yy: big * s * s + small * c * c };
                    let (widest, narrowest) = spread_of_taps(&kernel(&spread));
                    let want = across.max(KERNEL_LEAST_ACROSS);
                    assert!((widest / sigma - 1.0).abs() < 0.03, "{sigma} by {across} at {degrees}°: {widest} along");
                    assert!((narrowest / want - 1.0).abs() < 0.1, "{sigma} by {across} at {degrees}°: {narrowest} across");
                }
            }
        }
        // Next to nothing either way is next to a single tap, round or a
        // shake undone at next to no strength, so that the filter's
        // strength runs smoothly up from nought.
        let (s, c) = 30f64.to_radians().sin_cos();
        let shake = Spread { xx: 9.0 * c * c, xy: 9.0 * c * s, yy: 9.0 * s * s };
        for little in [Spread::round(0.05), shake.scaled(0.02)] {
            let middle = kernel(&little).iter().find(|tap| (tap.0, tap.1) == (0, 0)).map_or(0.0, |tap| tap.2);
            assert!(middle > 0.95, "{little:?}: {middle} in the middle");
        }
    }

    #[test]
    fn a_spread_turns_and_says_which_way_it_is_widest() {
        let shake = Spread { xx: 4.0, xy: 0.0, yy: 1.0 };
        assert_eq!(shake.angle(), 0.0);
        assert_eq!(shake.turned_cw(), Spread { xx: 1.0, xy: 0.0, yy: 4.0 });
        assert!((shake.turned_cw().angle().abs() - 90.0).abs() < 1e-9);
        let leaning = Spread { xx: 2.5, xy: 1.5, yy: 2.5 };
        assert!((leaning.angle() - 45.0).abs() < 1e-9 && (leaning.turned_cw().angle() + 45.0).abs() < 1e-9);
        assert_eq!(leaning.turned_cw().turned_cw().turned_cw().turned_cw(), leaning);
        assert_eq!(Spread::round(2.0).angle(), 0.0);
    }

    #[test]
    fn a_large_image_is_measured_at_its_own_size() {
        // A page of grainy paper, larger than is measured whole, with a
        // patch of squares six pixels a side blurred by three. Reduced to
        // the pixels a measure takes, the squares come to three pixels and
        // their blur to one and a half, and the measure reads under two
        // thirds of it; the parts of the page with the most edges,
        // measured at its own size, read the blur as the patch alone does.
        let (w, h) = (2600, 2400);
        assert!(w * h > MOST_PIXELS);
        let (pw, ph, cell) = (900, 700, 6);
        let squares: Vec<f32> = (0..pw * ph)
            .map(|i| {
                let (x, y) = (i % pw / cell, i / pw / cell);
                if (x * 7 + y * 13 + x * y) % 3 == 0 { 0.1 } else { 0.9 }
            })
            .collect();
        let patch = blurred_in_light(&squares, pw, ph, &kernel(&Spread::round(3.0)));
        let mut grain = 0x9E37_79B9u64;
        let mut page: Vec<f32> = (0..w * h)
            .map(|_| {
                grain ^= grain << 13;
                grain ^= grain >> 7;
                grain ^= grain << 17;
                0.85 + ((grain % 1000) as f32 / 1000.0 - 0.5) * 0.03
            })
            .collect();
        let (px, py) = (1500, 1100);
        for y in 0..ph {
            page[(py + y) * w + px..(py + y) * w + px + pw].copy_from_slice(&patch[y * pw..(y + 1) * pw]);
        }
        let alone = measure(&patch, pw, ph, Light::Linear).expect("the patch shows its blur").extremes().0;
        let widest = spread_of(&page, w, h, Light::Linear).expect("the page shows the patch's blur").extremes().0;
        assert!((widest - alone).abs() < 0.1 * alone, "{widest} against {alone} for the patch alone");
        let times = (MOST_PIXELS as f64 / (w * h) as f64).sqrt();
        let (rw, rh) = ((w as f64 * times).round() as usize, (h as f64 * times).round() as usize);
        let reduced = measure(&resize(&page, w, h, 1, rw, rh), rw, rh, Light::Linear);
        assert!(reduced.is_none_or(|spread| spread.extremes().0 / times < 0.7 * alone), "{reduced:?} reduced");
    }

    #[test]
    fn a_small_patch_on_grainy_paper_is_measured_in_parts() {
        // A field of squares two hundred pixels a side, blurred by three,
        // on a page of grainy paper under six megapixels, which was once
        // measured whole: the grain set the steepest change and the
        // contrast an edge is judged against, its edges were most of
        // those measured, and the page showed no blur. Measured over its
        // busiest parts, it shows the field's.
        let (w, h, side) = (2400, 2400, 200);
        assert!(w * h > MOST_PIXELS);
        let field = blurred(&squares(side, 8), side, side, &kernel(&Spread::round(3.0)));
        let mut grain = 12345u64;
        let mut page: Vec<f32> = (0..w * h)
            .map(|_| {
                grain = grain.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                0.9 + ((grain >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.03
            })
            .collect();
        for y in 0..side {
            page[(500 + y) * w + 500..(500 + y) * w + 500 + side].copy_from_slice(&field[y * side..(y + 1) * side]);
        }
        let alone = measure(&field, side, side, Light::Linear).expect("the field shows its blur").extremes().0;
        assert!(measure(&page, w, h, Light::Linear).is_none(), "measured whole, the grain hides it");
        let widest = spread_of(&page, w, h, Light::Linear).expect("the page shows the field's blur").extremes().0;
        assert!((widest - alone).abs() < 0.1 * alone, "{widest} against {alone} for the field alone");
    }

    #[test]
    fn a_few_edges_say_nothing() {
        // A dot of ink four pixels a side, blurred, on clean paper: a
        // score of edges round it, fewer than a direction must have to be
        // fitted on, which once gave a spread all the same.
        let (size, side) = (40, 4);
        let dot: Vec<f32> = (0..size * size).map(|i| if (18..18 + side).contains(&(i % size)) && (18..18 + side).contains(&(i / size)) { 0.1 } else { 0.9 }).collect();
        assert_eq!(measure(&blurred(&dot, size, size, &kernel(&Spread::round(1.5))), size, size, Light::Linear), None);
    }

    #[test]
    fn edges_within_a_narrow_fan_of_directions_say_nothing_of_the_shape() {
        // Three fields of stripes side by side, a barcode or ruled paper,
        // their edges at three directions within thirty degrees, blurred
        // the same every way and a little grainy. Fitted on those
        // directions alone, the shape was a guess along the stripes,
        // where no edge was measured: a blur of two and a half pixels
        // came to two and a half across them and to nothing along them,
        // and with the stripes closer together, to three and a half
        // along them. Over a fan so narrow the blur is taken the same
        // every way; over one reaching a quarter turn its shape is
        // fitted, and a round blur comes out round.
        let (w, h) = (600, 200);
        let stripes = |normals: [f64; 3]| -> Vec<f32> {
            (0..w * h)
                .map(|i| {
                    let (x, y) = ((i % w) as f64, (i / w) as f64);
                    let (s, c) = normals[i % w / 200].to_radians().sin_cos();
                    // Four samples across the pixel, so that a stripe at a
                    // slant is not drawn in steps; twenty pixels of ink and
                    // twenty of paper, so that each edge reads alone.
                    let ink = (0..4).filter(|k| ((c * x + s * y + (*k as f64 - 1.5) * 0.25) / 40.0).rem_euclid(1.0) < 0.5).count();
                    (0.9 - 0.2 * ink as f64) as f32
                })
                .collect()
        };
        let mut grain = 99u64;
        let mut grainy = |image: Vec<f32>| -> Vec<f32> {
            image
                .into_iter()
                .map(|v| {
                    grain = grain.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    v + ((grain >> 40) % 1000) as f32 / 1000.0 * 0.01 - 0.005
                })
                .collect()
        };
        for normals in [[5.0, 20.0, 35.0], [2.0, 10.0, 20.0]] {
            let image = grainy(blurred(&stripes(normals), w, h, &kernel(&Spread::round(2.5))));
            let (widest, narrowest) = measure(&image, w, h, Light::Linear).expect("edges enough").extremes();
            assert!(widest == narrowest && (widest - 2.5).abs() < 0.35 * 2.5, "{normals:?}: {widest} by {narrowest}");
        }
        let image = grainy(blurred(&stripes([0.0, 45.0, 90.0]), w, h, &kernel(&Spread::round(2.5))));
        let (widest, narrowest) = measure(&image, w, h, Light::Linear).expect("edges enough").extremes();
        assert!(widest < 1.2 * narrowest && (widest - 2.5).abs() < 0.35 * 2.5, "a quarter turn: {widest} by {narrowest}");
    }

    #[test]
    fn an_image_on_the_card_is_undone_as_one_handed_it_is() {
        // Its height no whole number of a reduction's steps.
        let (w, h, cell) = (150, 143, 8);
        let data = blurred(&squares(w, cell)[..w * h], w, h, &kernel(&Spread::round(2.6)));
        let card = gpu::Image::handed(&data, w, h);
        // Measured, given, given so wide that it is undone reduced, and at a
        // strength of nought.
        let wide = Spread { xx: 9.0, xy: 1.0, yy: 6.0 };
        assert!(Reduction::for_blur(wide.extremes().0, w * h, MOST_READ_PIXELS, 1.0).q > 1);
        for light in [Light::Linear, Light::Encoded] {
            for (given, strength) in [(None, 1.0), (Some(Spread::round(1.5)), 1.0), (Some(wide), 1.0), (Some(Spread::round(2.0)), 0.0)] {
                let (handed, spread) = undone_within(&data, w, h, given, strength, (w, h), light);
                let (on, on_spread) = undone_on(&card, given, strength, (w, h), light);
                assert_eq!(on_spread, spread, "{given:?} at {strength} {light:?}");
                assert!(on.read() == handed, "{given:?} at {strength} {light:?}");
            }
        }
        // With no edges to measure a blur off, as it is.
        let flat = vec![0.6f32; 64 * 64];
        let (on, spread) = undone_on(&gpu::Image::handed(&flat, 64, 64), None, 1.0, (64, 64), Light::Linear);
        assert_eq!((on.read(), spread), (flat, None));
    }

    #[test]
    fn a_reduction_undone_on_the_card_at_once_is_as_it_is_a_step_at_a_time() {
        // Its height no whole number of a reduction's steps.
        let (w, h, cell) = (150, 143, 8);
        let data = blurred(&squares(w, cell)[..w * h], w, h, &kernel(&Spread::round(2.6)));
        let at_size = Spread::round(1.4);
        for reduction in [Reduction { p: 1, q: 2 }, Reduction { p: 2, q: 3 }, Reduction { p: 1, q: 5 }] {
            let carried = reduction.carried(w, h);
            let (stepped, rw, rh) = reduced_deconvolved(&data, w, h, &at_size, reduction, FLOOR);
            // Reduced and undone, as a search's copy is deblurred.
            let card = gpu::through(&data, (w, h), carried, |image| reduced_deconvolved_on(&image, &at_size, reduction, FLOOR)).expect("on the card");
            assert!(card == stepped, "{reduction:?}");
            // And brought back, as the photo's Deblur filter is.
            let card = gpu::through(&data, (w, h), carried, |image| reduced_deconvolved_on(&image, &at_size, reduction, FLOOR).resized_part(carried, (w, h))).expect("on the card");
            assert!(card == restored(&stepped, (rw, rh), reduction, (w, h)), "{reduction:?} brought back");
        }
    }

    #[test]
    fn a_part_of_an_image_is_undone_as_the_whole_is_there() {
        // The largest ratio with a step of eight pixels or fewer that
        // brings the blur to two pixels and the image to the most a
        // deconvolution works on, or one in so many.
        let few = 100_000;
        let for_blur = |widest: f64, pixels: usize, most: f64| Reduction::for_blur(widest, pixels, MOST_PIXELS, most);
        assert_eq!(for_blur(1.5, few, 1.0), Reduction::NONE);
        assert_eq!(for_blur(2.4, few, 1.0), Reduction { p: 5, q: 6 });
        assert_eq!(for_blur(3.0, few, 1.0), Reduction { p: 2, q: 3 });
        assert_eq!(for_blur(3.5, few, 1.0), Reduction { p: 4, q: 7 });
        assert_eq!(for_blur(16.0, few, 1.0), Reduction { p: 1, q: 8 });
        assert_eq!(for_blur(50.0, few, 1.0), Reduction { p: 1, q: 25 });
        assert_eq!(for_blur(1.0, 4 * MOST_PIXELS, 1.0), Reduction { p: 1, q: 2 });
        assert_eq!(for_blur(3.0, few, 0.5), Reduction { p: 1, q: 2 });
        // A photograph of twelve megapixels, reduced for a search's copy and
        // not for the filter or a read's.
        assert_eq!(for_blur(1.0, 12_000_000, 1.0), Reduction { p: 1, q: 3 });
        assert_eq!(Reduction::for_blur(1.0, 12_000_000, MOST_READ_PIXELS, 1.0), Reduction::NONE);
        // A part of a blurred image that begins a whole number of the
        // reduction's steps into it, with as much of the image about it as
        // the deconvolution reads, comes out as the whole does there; one
        // that begins a pixel off the grid differs by most of a hundredth
        // of the contrast.
        let (size, cell) = (240, 8);
        let image = blurred(&squares(size, cell), size, size, &kernel(&Spread::round(3.0)));
        let spread = Spread { xx: 9.0, xy: 1.0, yy: 6.0 };
        let reduction = Reduction::for_blur(spread.extremes().0, size * size, MOST_READ_PIXELS, 1.0);
        assert!(reduction.q > 1, "{reduction:?}");
        let whole = undone_by(&image, size, size, Some(spread), 1.0, Light::Linear).0;
        let pad = reach(&spread, 1.0, reduction).ceil() as usize;
        let wanted = [100, 90, 180, 170];
        let differs = |x0: usize, y0: usize| {
            let part = [x0, y0, (wanted[2] + pad).min(size), (wanted[3] + pad).min(size)];
            let (w, h) = (part[2] - part[0], part[3] - part[1]);
            let pixels: Vec<f32> = (part[1]..part[3]).flat_map(|y| image[y * size + part[0]..y * size + part[2]].iter().copied()).collect();
            let undone = undone_within(&pixels, w, h, Some(spread), 1.0, (size, size), Light::Linear).0;
            (wanted[1]..wanted[3])
                .flat_map(|y| (wanted[0]..wanted[2]).map(move |x| (x, y)))
                .map(|(x, y)| (undone[(y - part[1]) * w + x - part[0]] - whole[y * size + x]).abs())
                .fold(0.0f32, f32::max)
        };
        let on_grid = |v: usize| (v - pad) / reduction.q * reduction.q;
        let (x0, y0) = (on_grid(wanted[0]), on_grid(wanted[1]));
        assert!(differs(x0, y0) < 1e-4, "{} on the grid", differs(x0, y0));
        assert!(differs(x0 - 1, y0) > 3e-3, "{} off it", differs(x0 - 1, y0));
    }

    /// The blur's measure as the cores made it, before its edges were
    /// read on the card: what the card's is held to.
    mod on_cores {
        use std::borrow::Cow;

        use super::super::*;
        use crate::work;

        /// A part of an image as [`measured_over`] reads its edges: the image
        /// smoothed and its gradient each way, over the part and the margin
        /// about it ([`PART_MARGIN`]) within the image.
        struct PartRead {
            /// The box `[x0, y0, x1, y1]` of the image the pixels are of.
            held: [usize; 4],
            /// The part, within it, that the edges are taken in.
            part: [usize; 4],
            smoothed: Vec<f32>,
            gx: Vec<f32>,
            gy: Vec<f32>,
        }

        impl PartRead {
            fn new(data: &[f32], width: usize, height: usize, part @ [x0, y0, x1, y1]: [usize; 4]) -> Self {
                let held = [x0.saturating_sub(PART_MARGIN), y0.saturating_sub(PART_MARGIN), (x1 + PART_MARGIN).min(width), (y1 + PART_MARGIN).min(height)];
                let (w, h) = (held[2] - held[0], held[3] - held[1]);
                let pixels: Cow<[f32]> = if held == [0, 0, width, height] {
                    Cow::Borrowed(data)
                } else {
                    Cow::Owned((held[1]..held[3]).flat_map(|y| data[y * width + held[0]..y * width + held[2]].iter().copied()).collect())
                };
                let smoothed = blurred(&pixels, w, h, &kernel(&Spread::round(SMOOTHING)));
                let at = |x: usize, y: usize| smoothed[y * w + x];
                // The gradient over three pixels each way, a Sobel's, a pixel apart.
                let mut gx = vec![0.0f32; w * h];
                let mut gy = vec![0.0f32; w * h];
                for y in 1..h.saturating_sub(1) {
                    for x in 1..w.saturating_sub(1) {
                        let i = y * w + x;
                        gx[i] = (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2.0 * at(x - 1, y) - at(x - 1, y + 1)) / 8.0;
                        gy[i] = (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2.0 * at(x, y - 1) - at(x + 1, y - 1)) / 8.0;
                    }
                }
                Self { held, part, smoothed, gx, gy }
            }

            fn width(&self) -> usize {
                self.held[2] - self.held[0]
            }

            /// The index among the pixels held of the image's pixel `(x, y)`.
            fn index(&self, x: usize, y: usize) -> usize {
                (y - self.held[1]) * self.width() + x - self.held[0]
            }

            fn steepness(&self, i: usize) -> f32 {
                (self.gx[i] * self.gx[i] + self.gy[i] * self.gy[i]).sqrt()
            }

            /// The indices of the part's own pixels among those held.
            fn part_pixels(&self) -> impl Iterator<Item = usize> + '_ {
                let [x0, y0, x1, y1] = self.part;
                (y0..y1).flat_map(move |y| (x0..x1).map(move |x| self.index(x, y)))
            }

            /// The light smoothed at `(x, y)` in the image's pixels, bilinear,
            /// held within the pixels held: within the image, where they reach
            /// its edges, and elsewhere further than any step is read.
            fn light(&self, x: f64, y: f64) -> f32 {
                let (w, h) = (self.width(), self.held[3] - self.held[1]);
                let (fx, fy) = ((x - 0.5 - self.held[0] as f64).clamp(0.0, (w - 1) as f64), (y - 0.5 - self.held[1] as f64).clamp(0.0, (h - 1) as f64));
                let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
                let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
                let (tx, ty) = ((fx - x0 as f64) as f32, (fy - y0 as f64) as f32);
                let at = |x: usize, y: usize| self.smoothed[y * w + x];
                let top = at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx;
                let bottom = at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx;
                top * (1.0 - ty) + bottom * ty
            }
        }

        /// How far the image `width` by `height` spreads an edge, as the module
        /// describes, from the edges in `parts` of it, boxes `[x0, y0, x1, y1]`
        /// of its pixels: the whole of it, or the parts of a large image
        /// [`spread_of`] picks, each read with the pixels about it as the whole
        /// image would read them, and the steepest change and the contrast that
        /// an edge is judged against taken over them all.
        fn measured_over(data: &[f32], width: usize, height: usize, parts: &[[usize; 4]]) -> Option<Spread> {
            let reads: Vec<PartRead> = work::map(parts, |&part| PartRead::new(data, width, height, part));
            let mut all: Vec<f32> = reads.iter().flat_map(|read| read.part_pixels().map(|i| read.steepness(i))).filter(|&g| g > 0.0).collect();
            if all.len() < LEAST_EDGES {
                return None;
            }
            let top = (all.len() * 99 / 100).min(all.len() - 1);
            let steepest = *all.select_nth_unstable_by(top, f32::total_cmp).1;
            let floor = EDGE_SHARE * steepest;
            let mut sorted: Vec<f32> = reads.iter().flat_map(|read| read.part_pixels().map(|i| read.smoothed[i])).collect();
            let (low, high) = (sorted.len() / 50, sorted.len() * 49 / 50);
            let darkest = *sorted.select_nth_unstable_by(low, f32::total_cmp).1;
            let lightest = *sorted.select_nth_unstable_by(high, f32::total_cmp).1;
            let least_step = STEP_SHARE * (lightest - darkest);
            // Each edge's width and the direction across it, measured with the
            // step read `reach` pixels either side, the parts in their order.
            let widths = |reach: f64| -> Vec<(f64, f64)> {
                work::map(&reads, |read| {
                    let mut out = Vec::new();
                    let margin = 2;
                    let [x0, y0, x1, y1] = read.part;
                    for y in y0.max(margin)..y1.min(height - margin) {
                        for x in x0.max(margin)..x1.min(width - margin) {
                            let i = read.index(x, y);
                            let g = read.steepness(i);
                            if g < floor {
                                continue;
                            }
                            let (ux, uy) = (f64::from(read.gx[i] / g), f64::from(read.gy[i] / g));
                            // Steepest along its own direction: the edge's middle.
                            let (sx, sy) = ((ux.round() as isize), (uy.round() as isize));
                            let ahead = read.steepness(read.index((x as isize + sx) as usize, (y as isize + sy) as usize));
                            let behind = read.steepness(read.index((x as isize - sx) as usize, (y as isize - sy) as usize));
                            if g < ahead || g < behind {
                                continue;
                            }
                            let (cx, cy) = (x as f64 + 0.5, y as f64 + 0.5);
                            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                            let steps = (4.0 * reach).ceil() as i64;
                            let at = |k: i64| read.light(cx + ux * k as f64 * reach / steps.max(1) as f64, cy + uy * k as f64 * reach / steps.max(1) as f64);
                            for k in -steps..=steps {
                                let v = at(k);
                                lo = lo.min(v);
                                hi = hi.max(v);
                            }
                            if hi - lo < least_step {
                                continue;
                            }
                            // Outwards each way as far as the light turns back.
                            let fall = TURN_SHARE * (hi - lo);
                            let outwards = |sign: i64| {
                                let (mut extreme, mut out, mut turned) = (at(0), 0, false);
                                for k in 1..=steps {
                                    let v = at(sign * k);
                                    if (v - extreme) * sign as f32 > 0.0 {
                                        (extreme, out) = (v, k);
                                    } else if (extreme - v) * sign as f32 > fall {
                                        turned = true;
                                        break;
                                    }
                                }
                                (extreme, (out as f64 * reach / steps.max(1) as f64) as f32, turned)
                            };
                            let ((lightest, beyond, far), (darkest, before, near)) = (outwards(1), outwards(-1));
                            if lightest - darkest < least_step {
                                continue;
                            }
                            let crossing = gpu::edges::Crossing { width: (lightest - darkest) / g, across: [ux as f32, uy as f32], out: [beyond, before], turned: [far, near] };
                            let Some(w) = alone(&crossing).filter(|&w| w <= 2.0 * MOST_REACH + 2.0) else { continue };
                            let direction = uy.atan2(ux).rem_euclid(std::f64::consts::PI);
                            out.push((w, direction));
                        }
                    }
                    out
                })
                .concat()
            };
            // A first look with a reach of a few pixels says how wide the edges
            // are; the next reads each step over as far as such an edge reaches,
            // and again while that moves the reach.
            let mut reach = 8.0;
            let mut measured = widths(reach);
            for _ in 0..2 {
                let typical = narrowest_share(measured.iter().map(|w| w.0).collect(), 0.5)?;
                let wider = (1.5 * typical).clamp(4.0, MOST_REACH);
                if (wider - reach).abs() < 1.0 {
                    break;
                }
                reach = wider;
                measured = widths(reach);
            }
            if measured.len() < LEAST_EDGES {
                return None;
            }
            // The narrowest of each direction's edges, as a variance of the blur.
            let mut bins: Vec<Vec<f64>> = vec![Vec::new(); DIRECTIONS];
            for &(w, direction) in &measured {
                let bin = ((direction / std::f64::consts::PI * DIRECTIONS as f64) as usize).min(DIRECTIONS - 1);
                bins[bin].push(w);
            }
            let rows: Vec<(usize, [f64; 3], f64, f64)> = bins
                .into_iter()
                .enumerate()
                .filter(|(_, widths)| widths.len() >= LEAST_EDGES)
                .filter_map(|(bin, widths)| {
                    let count = widths.len() as f64;
                    let w = narrowest_share(widths, NARROWEST)?;
                    let theta = (bin as f64 + 0.5) * std::f64::consts::PI / DIRECTIONS as f64;
                    let (c, s) = (theta.cos(), theta.sin());
                    // The spread along the edge's normal, less the measure's own.
                    let variance = (w * w / (2.0 * std::f64::consts::PI) - OWN_VARIANCE).max(0.0);
                    Some((bin, [c * c, 2.0 * c * s, s * s], variance, count))
                })
                .collect();
            // How far round the half turn the directions reach ([`LEAST_FAN`]):
            // all of it but the widest gap between two of them, the gap from the
            // last back round to the first among them.
            let gaps = rows.windows(2).map(|pair| pair[1].0 - pair[0].0);
            let round_again = rows.first().zip(rows.last()).map(|(first, last)| DIRECTIONS + first.0 - last.0);
            let fan = gaps.chain(round_again).max().map_or(0, |widest| DIRECTIONS - widest);
            let some = |spread: Spread| (spread.extremes().0 >= KERNEL_LEAST).then_some(spread);
            let round = || {
                let all: Vec<f64> = measured.iter().map(|w| w.0).collect();
                let w = narrowest_share(all, NARROWEST)?;
                let variance = w * w / (2.0 * std::f64::consts::PI) - OWN_VARIANCE;
                (variance > 0.0).then(|| Spread::round(variance.sqrt())).and_then(some)
            };
            if rows.len() < 3 || fan < LEAST_FAN {
                // Too few directions, or too narrow a fan of them, to tell a
                // shape: the same every way.
                return round();
            }
            // Least squares, each direction weighed by its edges.
            let mut a = [[0.0f64; 3]; 3];
            let mut b = [0.0f64; 3];
            for (_, row, value, weight) in &rows {
                for i in 0..3 {
                    for j in 0..3 {
                        a[i][j] += weight * row[i] * row[j];
                    }
                    b[i] += weight * row[i] * value;
                }
            }
            let [xx, xy, yy] = solve3(a, b)?;
            // A covariance is never less than nought either way.
            let spread = Spread { xx, xy, yy };
            let mean = (xx + yy) / 2.0;
            let half = (((xx - yy) / 2.0).powi(2) + xy * xy).sqrt();
            let (big, small) = ((mean + half).max(0.0), (mean - half).max(0.0));
            if big <= 0.0 {
                return None;
            }
            // A fit run off past twice the widest direction measured.
            if big > 2.0 * rows.iter().map(|row| row.2).fold(0.0, f64::max) {
                return round();
            }
            // The eigenvalues held at nought and above, the axes kept.
            let angle = 0.5 * (2.0 * spread.xy).atan2(spread.xx - spread.yy);
            let (c, s) = (angle.cos(), angle.sin());
            some(Spread { xx: big * c * c + small * s * s, xy: (big - small) * c * s, yy: big * s * s + small * c * c })
        }

        /// The parts of the image `width` by `height`, squares [`PART_SIDE`]
        /// pixels a side on a grid from its corner, that change the most across
        /// their edges, the busiest first, as many as come to [`MOST_PIXELS`]:
        /// the symbols and the print about them, where a photograph is mostly
        /// paper, desk and wall. A part is judged by how far its light changes
        /// over two pixels each way at every other pixel of every other row,
        /// added up over the changes as steep as an edge must be to be measured
        /// ([`EDGE_SHARE`] of the steepest), which grain does not reach. However
        /// far an edge is blurred, the light changes across it by the same in
        /// all, so a blur that spreads it wide does not hide it.
        fn busiest_parts(data: &[f32], width: usize, height: usize) -> Vec<[usize; 4]> {
            let across = width.div_ceil(PART_SIDE);
            let rows: Vec<usize> = (0..height.saturating_sub(2)).step_by(2).collect();
            let changes: Vec<Vec<f32>> = work::map(&rows, |&y| {
                (0..width.saturating_sub(2))
                    .step_by(2)
                    .map(|x| {
                        let here = data[y * width + x];
                        (data[y * width + x + 2] - here).abs() + (data[(y + 2) * width + x] - here).abs()
                    })
                    .collect()
            });
            let mut all: Vec<f32> = changes.iter().flatten().copied().filter(|&change| change > 0.0).collect();
            if all.is_empty() {
                return Vec::new();
            }
            let top = (all.len() * 99 / 100).min(all.len() - 1);
            let floor = EDGE_SHARE * *all.select_nth_unstable_by(top, f32::total_cmp).1;
            let mut busy = vec![0.0f64; across * height.div_ceil(PART_SIDE)];
            for (&y, row) in rows.iter().zip(&changes) {
                for (k, &change) in row.iter().enumerate() {
                    if change >= floor {
                        busy[y / PART_SIDE * across + 2 * k / PART_SIDE] += f64::from(change);
                    }
                }
            }
            let mut order: Vec<usize> = (0..busy.len()).filter(|&i| busy[i] > 0.0).collect();
            order.sort_by(|&a, &b| busy[b].total_cmp(&busy[a]).then(a.cmp(&b)));
            let mut parts = Vec::new();
            let mut pixels = 0;
            for i in order {
                if pixels >= MOST_PIXELS {
                    break;
                }
                let (x0, y0) = (i % across * PART_SIDE, i / across * PART_SIDE);
                let part = [x0, y0, (x0 + PART_SIDE).min(width), (y0 + PART_SIDE).min(height)];
                pixels += (part[2] - part[0]) * (part[3] - part[1]);
                parts.push(part);
            }
            parts
        }

        /// `spread_of` on the cores: the busiest parts as stored, and their
        /// edges in `light`.
        pub(super) fn spread_of(data: &[f32], width: usize, height: usize, light: Light) -> Option<Spread> {
            let light: Vec<f32> = data.iter().map(|&v| light.from_encoded(v)).collect();
            if width * height <= MOST_PIXELS {
                return measured_over(&light, width, height, &[[0, 0, width, height]]);
            }
            measured_over(&light, width, height, &busiest_parts(data, width, height))
        }
    }

    #[test]
    fn the_blur_is_measured_on_the_card_as_the_cores_measured_it() {
        let grainy = |data: Vec<f32>| {
            let mut seed = 0x9E37_79B9_u32;
            data.into_iter()
                .map(|v| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (v + ((seed >> 8) as f32 / (1 << 24) as f32 - 0.5) * 0.04).clamp(0.0, 1.0)
                })
                .collect::<Vec<f32>>()
        };
        let size = 256;
        let cases: Vec<(Vec<f32>, usize, usize)> = vec![
            (grainy(blurred(&squares(size, 13), size, size, &kernel(&Spread::round(2.0)))), size, size),
            (grainy(blurred(&squares(size, 9), size, size, &kernel(&Spread { xx: 6.0, xy: 1.5, yy: 1.0 }))), size, size),
            // Large enough to be measured in its busiest parts.
            (grainy(blurred(&squares(1400, 11)[..1400 * 1200], 1400, 1200, &kernel(&Spread::round(2.6)))), 1400, 1200),
        ];
        for (data, w, h) in cases {
            for light in [Light::Linear, Light::Encoded] {
                let (card, cores) = (spread_of(&data, w, h, light).expect("a blur shown"), on_cores::spread_of(&data, w, h, light).expect("a blur shown"));
                let ((cw, cn), (rw, rn)) = (card.extremes(), cores.extremes());
                assert!((cw - rw).abs() < 0.01 * rw && (cn - rn).abs() < 0.01 * rw, "{w} by {h} {light:?}: {card:?} against {cores:?}");
                let on_card = gpu::Image::handed(&data, w, h);
                assert_eq!(spread_of_values(Values::Card(&on_card), w, h, light), Some(card), "measured where it lies on the card");
            }
        }
    }
}
