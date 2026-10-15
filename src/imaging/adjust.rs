//! The adjustments made to the photo before it is shown and sampled: its
//! size, its sharpness, its tone, and a filter over each pixel's
//! neighbourhood.
//!
//! A photo is enlarged or reduced, and sharpened or blurred, as it is
//! toned: the whole of it, in its colours, so that what is shown is
//! what the pipeline reads. Both were once steps of their own, the enlargement a
//! part of loading a small photo and the two together a copy made about
//! one symbol for the sampling to read off ([`super::enhance`]), in
//! grey; a read that does not decode still makes itself such copies,
//! and what it read through one the photo so adjusted shows again.

use std::fmt;
use std::sync::{Arc, Mutex};

use super::Luma;
use super::enhance::{mask_for, unsharp_rgba, unsharp_values};
use super::resize::{Sample, resize, resize_into};
use crate::work;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Channel {
    /// Full colour on screen, perceptual luminance for sampling.
    #[default]
    Color,
    Red,
    Green,
    Blue,
}

impl Channel {
    pub const ALL: [Channel; 4] = [Channel::Color, Channel::Red, Channel::Green, Channel::Blue];

    pub fn label(self) -> &'static str {
        match self {
            Channel::Color => "Colour",
            Channel::Red => "Red",
            Channel::Green => "Green",
            Channel::Blue => "Blue",
        }
    }
}

/// A filter over each pixel's neighbourhood, applied to the luminance after
/// the tone adjustments. The finder scan reads runs of whole pixels and
/// breaks on a symbol printed as dots or a scan full of specks, where the
/// averaging of the module sampler copes; the filter mends the photo for
/// the scan. zxing-cpp's `tryDenoise` closes 2D symbols before binarising
/// them for the same reason, and its `tryDownscale` blurs by shrinking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Filter {
    #[default]
    None,
    /// Each pixel becomes the median of its neighbourhood: specks of
    /// dust, scanner noise and dropouts a pixel or two across vanish, and
    /// edges stay where they were.
    Median,
    /// Morphological closing of the dark: the darkest value of the
    /// neighbourhood, then the lightest of that. The dots of a dot-matrix
    /// or inkjet print join into solid modules.
    Close,
    /// A plain average over the neighbourhood, for grain and halftone
    /// screens.
    Blur,
}

impl Filter {
    pub const ALL: [Filter; 4] = [Filter::None, Filter::Median, Filter::Close, Filter::Blur];
    /// Widest neighbourhood offered, as a half-width in pixels.
    pub const MAX_RADIUS: u8 = 3;

    pub fn label(self) -> &'static str {
        match self {
            Filter::None => "None",
            Filter::Median => "Median",
            Filter::Close => "Close dots",
            Filter::Blur => "Blur",
        }
    }

    pub fn describe(self) -> &'static str {
        match self {
            Filter::None => "The photo as it is",
            Filter::Median => "Removes specks and dropouts a pixel or two across while keeping edges sharp",
            Filter::Close => "Joins the dots of a dot-matrix or inkjet print into solid dark modules",
            Filter::Blur => "Averages away grain and halftone screens",
        }
    }

    /// The filter over a `width` × `height` luminance image, with a
    /// neighbourhood `radius` pixels either side of each pixel. The
    /// median's cost grows with the square of the radius, so it is held
    /// to two.
    pub fn apply(self, data: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
        let radius = radius.max(1);
        match self {
            Filter::None => data.to_vec(),
            Filter::Median => median(data, width, height, radius.min(2)),
            Filter::Close => {
                let darkest = separable(data, width, height, radius, |w| w.iter().copied().fold(f32::INFINITY, f32::min));
                separable(&darkest, width, height, radius, |w| w.iter().copied().fold(f32::NEG_INFINITY, f32::max))
            }
            Filter::Blur => separable(data, width, height, radius, |w| w.iter().sum::<f32>() / w.len() as f32),
        }
    }
}

/// `reduce` over the row window and then the column window around each
/// pixel, the image's edges repeated beyond it.
fn separable(data: &[f32], width: usize, height: usize, radius: usize, reduce: impl Fn(&[f32]) -> f32) -> Vec<f32> {
    let mut rows = vec![0.0f32; width * height];
    let mut window = Vec::with_capacity(2 * radius + 1);
    for y in 0..height {
        let row = &data[y * width..(y + 1) * width];
        for x in 0..width {
            window.clear();
            for dx in -(radius as isize)..=radius as isize {
                let sx = (x as isize + dx).clamp(0, width as isize - 1) as usize;
                window.push(row[sx]);
            }
            rows[y * width + x] = reduce(&window);
        }
    }
    let mut out = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            window.clear();
            for dy in -(radius as isize)..=radius as isize {
                let sy = (y as isize + dy).clamp(0, height as isize - 1) as usize;
                window.push(rows[sy * width + x]);
            }
            out[y * width + x] = reduce(&window);
        }
    }
    out
}

/// The median of the square neighbourhood around each pixel, the image's
/// edges repeated beyond it.
fn median(data: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; width * height];
    let side = 2 * radius + 1;
    let mut window = Vec::with_capacity(side * side);
    for y in 0..height {
        for x in 0..width {
            window.clear();
            for dy in -(radius as isize)..=radius as isize {
                let sy = (y as isize + dy).clamp(0, height as isize - 1) as usize;
                for dx in -(radius as isize)..=radius as isize {
                    let sx = (x as isize + dx).clamp(0, width as isize - 1) as usize;
                    window.push(data[sy * width + sx]);
                }
            }
            let middle = window.len() / 2;
            let (_, value, _) = window.select_nth_unstable_by(middle, f32::total_cmp);
            out[y * width + x] = *value;
        }
    }
    out
}

/// Pixels toned as one piece of a photo spread over the cores.
const TONED_AT_ONCE: usize = 1 << 18;

/// The longest a photo's longer side is enlarged to, past the few times
/// its size any photo may be made ([`Enlarge::ALWAYS`]), and the longest
/// it is enlarged to at all without its being asked for.
pub const MAX_ENLARGED_SIDE: u32 = 4096;
/// Photos smaller than this on their shorter side start enlarged by a
/// whole factor, so that a symbol rendered at a few pixels a module gets
/// modules the finder scan can see. A long thin rMQR render is judged by
/// its height. zxing-cpp has a separate path for such "pure" images; here
/// they are simply made bigger.
pub const MIN_PHOTO_SIDE: u32 = 256;
/// The shortest a photo's shorter side comes to at a reduction offered
/// for it: a photo may be worked on at less, as a small part cropped
/// from a photo reduced is, but less is not put forward.
pub const MIN_REDUCED_SIDE: u32 = 32;

/// The size the photo is worked on at, against its own: a quarter or a
/// half of it, or a whole number of times it. The photo is resized with
/// Magic Kernel Sharp 2013 ([`super::resize`]), in its colours, and the
/// outline and the anchors are in the pixels it comes to.
///
/// Enlarged, a symbol of a pixel or two a module has modules the scan
/// and the sampling can see, and the luminance keeps the photo's own
/// pixels beside it ([`Luma::unenlarged`]) for a symbol rendered pixel
/// for pixel, whose square corners the kernel rounds. Reduced, grain and
/// the dots of a halftone screen run together, as zxing-cpp's
/// `tryDownscale` has them, and a large photo is the quicker to work on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Enlarge {
    /// Quarters of the photo's own size: one, two, or four to each
    /// whole time.
    quarters: u16,
}

impl Default for Enlarge {
    fn default() -> Self {
        Self::NONE
    }
}

impl Enlarge {
    pub const QUARTER: Enlarge = Enlarge { quarters: 1 };
    pub const HALF: Enlarge = Enlarge { quarters: 2 };
    /// The photo at its own size.
    pub const NONE: Enlarge = Enlarge { quarters: 4 };
    /// The most times its size a photo is ever made.
    pub const MAX_TIMES: u16 = 1024;
    /// What a photo of any size may be enlarged to when that is asked
    /// for, however large it is: four times its size. A large photo so
    /// enlarged is sixteen times the pixels to hold and to work on, and
    /// is never made so of itself ([`Self::fits_unasked`]).
    pub const ALWAYS: Enlarge = Self::times(4);
    /// The steps offered of themselves, each twice the last. A photo
    /// may be at a whole number of times its size between them besides:
    /// what a small one starts at ([`Self::for_photo`]), or what a read
    /// decoded through ([`Self::by`]).
    pub const LADDER: [Enlarge; 9] = [
        Self::QUARTER,
        Self::HALF,
        Self::NONE,
        Self::times(2),
        Self::times(4),
        Self::times(8),
        Self::times(16),
        Self::times(32),
        Self::times(64),
    ];

    /// The photo at `times` its own size, once at the least.
    pub const fn times(times: u16) -> Self {
        let times = if times < 1 {
            1
        } else if times > Self::MAX_TIMES {
            Self::MAX_TIMES
        } else {
            times
        };
        Enlarge { quarters: 4 * times }
    }

    /// How many times its own size the photo is made.
    pub fn factor(self) -> f64 {
        f64::from(self.quarters) / 4.0
    }

    /// The whole number of times, when it is one: 1 for the photo at
    /// its own size, `None` for a photo reduced.
    pub fn whole(self) -> Option<usize> {
        self.quarters.is_multiple_of(4).then(|| usize::from(self.quarters / 4))
    }

    /// What a photo `width` by `height` measures so resized: a side of a
    /// photo reduced comes to the nearest whole pixel, and to one at the
    /// least.
    pub fn size(self, width: usize, height: usize) -> (usize, usize) {
        let of = |side: usize| if side == 0 { 0 } else { ((side * usize::from(self.quarters) + 2) / 4).max(1) };
        (of(width), of(height))
    }

    /// Whether a photo `width` by `height` may be worked on so: enlarged
    /// as far as [`Self::ALWAYS`] whatever its size, and further while
    /// its longer side stays within [`MAX_ENLARGED_SIDE`]. Its own size
    /// it may always have, and any reduction: a reduced photo is the
    /// less to hold.
    pub fn fits(self, width: u32, height: u32) -> bool {
        self <= Self::ALWAYS || self.fits_unasked(width, height)
    }

    /// Whether this is put forward as a step for a photo `width` by
    /// `height`: an enlargement that fits it, or a reduction that
    /// leaves its shorter side [`MIN_REDUCED_SIDE`] at the least.
    fn offered(self, width: u32, height: u32) -> bool {
        self.fits(width, height) && (self >= Self::NONE || self.across(width.min(height)) >= u64::from(MIN_REDUCED_SIDE))
    }

    /// Whether a photo `width` by `height` is enlarged so without its
    /// being asked for, when it is loaded or when a read that decoded
    /// only through a copy of it so enlarged sets it: while its longer
    /// side stays within [`MAX_ENLARGED_SIDE`]. A photo not enlarged at
    /// all is.
    pub fn fits_unasked(self, width: u32, height: u32) -> bool {
        self <= Self::NONE || self.across(width.max(height)) <= u64::from(MAX_ENLARGED_SIDE)
    }

    /// What a side of `side` pixels comes to, to the pixel below.
    fn across(self, side: u32) -> u64 {
        u64::from(side) * u64::from(self.quarters) / 4
    }

    /// This, when a photo `width` by `height` may be worked on so, and
    /// otherwise the most times its size that it may be.
    pub fn within(self, width: u32, height: u32) -> Self {
        if self.fits(width, height) {
            return self;
        }
        let within_side = (MAX_ENLARGED_SIDE / width.max(height).max(1)).clamp(1, u32::from(Self::MAX_TIMES)) as u16;
        Self::times(within_side).max(Self::ALWAYS)
    }

    /// What a photo `width` by `height` starts at: its own size, unless
    /// it is under [`MIN_PHOTO_SIDE`] on its shorter side, and then the
    /// whole number of times that brings it there, as far as the longer
    /// side allows. Cubic and linear enlargement were tried before Magic
    /// Kernel Sharp 2013 and lost more of the corpus than they gained: a
    /// cubic rings at every edge, and even a linear ramp moves the
    /// corners the edge tracer finds on crumpled paper. This kernel
    /// reads the corpus as enlarging pixel for pixel did.
    ///
    /// A large photo starts at its size, and is never made smaller
    /// unasked. It was once reduced to 4096 pixels on its longer side,
    /// and a small symbol in a large photo lost to that the pixels its
    /// modules had. Only the search of the whole photo still looks at a
    /// copy that size ([`super::detect::SEARCH_SIDE`]): what it finds,
    /// like an outline drawn by hand, is refined and read in the photo's
    /// own pixels.
    pub fn for_photo(width: u32, height: u32) -> Self {
        let longest = width.max(height).max(1);
        let shortest = width.min(height).max(1);
        if shortest >= MIN_PHOTO_SIDE {
            return Self::NONE;
        }
        Self::times(MIN_PHOTO_SIDE.div_ceil(shortest).min(MAX_ENLARGED_SIDE / longest).clamp(1, u32::from(Self::MAX_TIMES)) as u16)
    }

    /// The steps offered for a photo `width` by `height`, least first:
    /// those of the [`Self::LADDER`] it may be worked on at, which for a
    /// photo of any size go as far as [`Self::ALWAYS`], less the
    /// reductions that would leave next to nothing of it, and what it
    /// starts at among them.
    pub fn steps_for(width: u32, height: u32) -> Vec<Enlarge> {
        let mut steps: Vec<Enlarge> = Self::LADDER.into_iter().filter(|step| step.offered(width, height)).collect();
        steps.push(Self::for_photo(width, height));
        steps.sort();
        steps.dedup();
        steps
    }

    /// The photo `times` as large again as this has it: what a read
    /// made through a copy of the photo so enlarged read it at. More
    /// than the photo's own size is a whole number of times it, the
    /// next one up where the two do not come to one.
    pub fn by(self, times: u8) -> Self {
        let quarters = self.quarters.saturating_mul(u16::from(times.max(1)));
        match quarters {
            1 | 2 => Enlarge { quarters },
            _ => Self::times(quarters.div_ceil(4)),
        }
    }

    /// What `text` names, as it is shown or as a bare number: `0.5×`,
    /// `0.25`, `3x`. A number between the sizes there are is the nearest
    /// of them.
    pub fn parse(text: &str) -> Option<Self> {
        let factor: f64 = text.trim().trim_end_matches(['×', 'x', 'X']).trim_end().parse().ok()?;
        if !(factor > 0.0 && factor.is_finite()) {
            return None;
        }
        Some(if factor < 0.375 {
            Self::QUARTER
        } else if factor < 0.75 {
            Self::HALF
        } else {
            Self::times(factor.round().min(f64::from(Self::MAX_TIMES)) as u16)
        })
    }
}

impl fmt::Display for Enlarge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.whole() {
            Some(times) => write!(f, "{times}×"),
            None => write!(f, "{}×", self.factor()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Adjust {
    /// A single channel often separates ink from a coloured stain better
    /// than luminance does.
    pub channel: Channel,
    /// For light-on-dark symbols.
    pub invert: bool,
    /// -1.0 to 1.0, added after contrast.
    pub brightness: f32,
    /// -1.0 (flat grey) to 1.0 (hard threshold at mid grey).
    pub contrast: f32,
    /// Neighbourhood filter applied last.
    pub filter: Filter,
    /// Half-width of the filter's neighbourhood, in pixels.
    pub radius: u8,
    /// Radial lens distortion to take out first, as Brown's model has it:
    /// negative straightens barrel, positive pincushion; 0.0 leaves the
    /// photo as shot. Corners and anchors are in the straightened image.
    pub k1: f32,
    /// The size the photo is worked on at, against its own. Corners and
    /// anchors are in the pixels it comes to.
    pub enlarge: Enlarge,
    /// How far the photo is sharpened by an unsharp mask, from 0.0 for
    /// not at all to 1.0, or below nought blurred instead, with the
    /// mask's own Gaussian, as far as -1.0: see [`mask_for`]. The mask
    /// is laid on the photo at the smaller of its own size and the size
    /// it is worked on at: before it is enlarged, or after it is reduced.
    pub sharpen: f32,
}

impl Default for Adjust {
    fn default() -> Self {
        Self {
            channel: Channel::Color,
            invert: false,
            brightness: 0.0,
            contrast: 0.0,
            filter: Filter::None,
            radius: 1,
            k1: 0.0,
            enlarge: Enlarge::NONE,
            sharpen: 0.0,
        }
    }
}

/// Where a straightened pixel at normalised radius `r` (half the diagonal
/// is 1) comes from in the photo as shot, by Brown's radial model to its
/// first term, `r (1 + k1 r²)`, out to the half diagonal, and along the
/// model's tangent there beyond it. The model is for the photo, which
/// lies within the half diagonal; a barrel term as strong as the slider
/// allows turns it back on itself a little past the photo's corners, and
/// the corner of an outline or an anchor dragged out there came from
/// nearer the centre than a point within, or from across it. Along the
/// tangent the source keeps climbing with the radius, so every point has
/// one source and the inverse ([`straightened_radius`]) one answer.
pub fn source_radius(r: f32, k1: f32) -> f32 {
    if r <= 1.0 { r * (1.0 + k1 * r * r) } else { (1.0 + k1) + (r - 1.0) * edge_slope(f64::from(k1)) as f32 }
}

/// [`source_radius`] in the precision the remapping of points works in.
fn shot_radius(r: f64, k1: f64) -> f64 {
    if r <= 1.0 { r * (1.0 + k1 * r * r) } else { (1.0 + k1) + (r - 1.0) * edge_slope(k1) }
}

/// The model's slope at the half diagonal, `1 + 3 k1`, which the source
/// climbs at beyond it. The slider keeps `k1` above a third's barrel,
/// where the slope is a tenth; a term from elsewhere is held to that.
fn edge_slope(k1: f64) -> f64 {
    (1.0 + 3.0 * k1).max(0.1)
}

/// The normalised radius of a straightened point whose source lies at
/// normalised radius `source` in the photo as shot: the inverse of
/// [`source_radius`], by Newton's method within the half diagonal, which
/// the gentle distortions a lens has converge in a few steps, and along
/// the tangent beyond it.
fn straightened_radius(source: f64, k1: f64) -> f64 {
    if k1 == 0.0 || source == 0.0 {
        return source;
    }
    // Past a third's barrel the model turns back within the half
    // diagonal and a source has two radii, neither of which Newton's
    // method is sure to find: every term set comes from the slider or
    // the estimate, both held within [`K1_RANGE`].
    debug_assert!(k1 > -1.0 / 3.0, "lens term {k1} is beyond the model's inverse");
    let edge = 1.0 + k1;
    if source > edge {
        return 1.0 + (source - edge) / edge_slope(k1);
    }
    let mut r = source;
    for _ in 0..12 {
        let f = r * (1.0 + k1 * r * r) - source;
        let slope = 1.0 + 3.0 * k1 * r * r;
        if slope.abs() < 1e-9 {
            break;
        }
        let step = f / slope;
        r -= step;
        if step.abs() < 1e-12 {
            break;
        }
    }
    r
}

/// A point of the photo straightened with `from_k1`, in the pixel
/// coordinates the corners and anchors use, as it lies in the photo
/// straightened with `to_k1` instead: through the photo as shot and back.
/// The identity when the two are the same.
pub fn remap_point(point: [f64; 2], width: usize, height: usize, from_k1: f32, to_k1: f32) -> [f64; 2] {
    if from_k1 == to_k1 {
        return point;
    }
    let (w, h) = (width as f64, height as f64);
    let (cx, cy) = (w / 2.0, h / 2.0);
    let norm = (w * w + h * h).sqrt() / 2.0;
    let (dx, dy) = (point[0] - cx, point[1] - cy);
    let r = (dx * dx + dy * dy).sqrt() / norm;
    if r == 0.0 {
        return point;
    }
    // Out to the photo as shot, then in to the other straightening.
    let shot = shot_radius(r, f64::from(from_k1));
    let back = straightened_radius(shot, f64::from(to_k1));
    [cx + dx * back / r, cy + dy * back / r]
}

/// The lens terms the slider offers, the widest barrel to the widest
/// pincushion, and what an estimate is held to. Brown's model to its
/// first term turns back on itself within the half diagonal once the
/// barrel term passes a third, and a source then has two radii
/// ([`straightened_radius`]): the range stops well short of that.
pub const K1_MIN: f32 = -0.3;
pub const K1_MAX: f32 = 0.3;
pub const K1_RANGE: std::ops::RangeInclusive<f32> = K1_MIN..=K1_MAX;
/// Lens distortions tried first, from the widest barrel to the widest
/// pincushion the slider offers, and the step between them.
const LENS_SEARCH: (f32, f32, f32) = (K1_MIN, K1_MAX, 0.01);
/// Pairs needed before the lens term can be fitted with anything to spare:
/// a perspective map has eight degrees of freedom and the lens one more.
pub const LENS_MIN_PAIRS: usize = 7;
/// By what share the fit must improve for the estimate to be worth
/// anything: the residual of a few pairs falls a little under any extra
/// parameter.
pub const LENS_MIN_GAIN: f64 = 0.03;

/// What estimating the lens term found.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LensEstimate {
    /// The term to set, replacing the one the photo is read with now.
    pub k1: f32,
    /// Root-mean-square distance of the pairs from a plane, in pixels,
    /// with the lens term as it is and as estimated.
    pub rms_before: f64,
    pub rms_after: f64,
}

/// Estimate the radial lens term from points whose place in the symbol
/// and in the photo are both known: the corners of the outline and the
/// anchors, as `pairs` of module coordinates and pixels in the photo as
/// it is read now, straightened with `current_k1`. A flat symbol seen
/// through a lens with no distortion lies on a plane, so the term that
/// brings the pairs nearest a plane is the lens's, as BoofCV takes its
/// lens model from calibration and samples through it. The whole range of
/// the slider is tried, and the best term polished. `None` with too few
/// pairs, or when no term fits better than the current one by
/// [`LENS_MIN_GAIN`].
pub fn estimate_k1(width: usize, height: usize, current_k1: f32, pairs: &[([f64; 2], [f64; 2])]) -> Option<LensEstimate> {
    use super::homography::{Homography, Model};
    if pairs.len() < LENS_MIN_PAIRS {
        return None;
    }
    let residual = |k1: f32| -> Option<f64> {
        let moved: Vec<([f64; 2], [f64; 2])> =
            pairs.iter().map(|&(at, px)| (at, remap_point(px, width, height, current_k1, k1))).collect();
        let map = Homography::fit(Model::Projective, &moved)?;
        let sum: f64 = moved
            .iter()
            .map(|&(at, px)| {
                let [x, y] = map.apply(at[0], at[1]);
                (x - px[0]).powi(2) + (y - px[1]).powi(2)
            })
            .sum();
        Some((sum / moved.len() as f64).sqrt())
    };
    let rms_before = residual(current_k1)?;
    let (lo, hi, step) = LENS_SEARCH;
    let mut best: Option<(f32, f64)> = None;
    let mut k = lo;
    while k <= hi + step / 2.0 {
        if let Some(rms) = residual(k)
            && best.is_none_or(|(_, r)| rms < r)
        {
            best = Some((k, rms));
        }
        k += step;
    }
    let (mut k, mut rms) = best?;
    // Polish by halving the step about the best, to a thousandth.
    let mut span = step;
    while span > 0.0005 {
        span /= 2.0;
        for candidate in [k - span, k + span] {
            if let Some(r) = residual(candidate)
                && r < rms
            {
                k = candidate;
                rms = r;
            }
        }
    }
    // To the thousandth, and within the range the slider offers: the
    // polish may step a hair past its end, and the model's inverse is
    // vouched for within it alone.
    let k = ((k * 1000.0).round() / 1000.0).clamp(K1_MIN, K1_MAX);
    let rms_after = residual(k)?;
    (rms_after < rms_before * (1.0 - LENS_MIN_GAIN) && k != current_k1).then_some(LensEstimate { k1: k, rms_before, rms_after })
}

/// The photo with its radial distortion taken out, the same size: each
/// output pixel is read from the source at [`source_radius`] along the
/// same ray from the centre, bilinearly, and white where that falls off
/// the photo. Inverse mapping leaves no holes.
pub fn undistort(rgba: &[u8], width: usize, height: usize, k1: f32) -> Vec<u8> {
    assert_eq!(rgba.len(), width * height * 4);
    let (w, h) = (width as f32, height as f32);
    let (cx, cy) = (w / 2.0, h / 2.0);
    let norm = (w * w + h * h).sqrt() / 2.0;
    let at = |x: usize, y: usize| &rgba[(y * width + x) * 4..(y * width + x) * 4 + 4];
    let mut out = Vec::with_capacity(rgba.len());
    for y in 0..height {
        for x in 0..width {
            let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
            let r = (dx * dx + dy * dy).sqrt() / norm;
            let f = if r > 0.0 { source_radius(r, k1) / r } else { 1.0 };
            // Source pixel centres sit at half-integers, as in `Luma::sample`.
            let (sx, sy) = (cx + dx * f - 0.5, cy + dy * f - 0.5);
            if !(sx >= -0.5 && sy >= -0.5 && sx <= w - 0.5 && sy <= h - 0.5) {
                out.extend_from_slice(&[255, 255, 255, 255]);
                continue;
            }
            let x0 = sx.floor().clamp(0.0, w - 1.0) as usize;
            let y0 = sy.floor().clamp(0.0, h - 1.0) as usize;
            let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(height - 1));
            let tx = (sx - x0 as f32).clamp(0.0, 1.0);
            let ty = (sy - y0 as f32).clamp(0.0, 1.0);
            let (p00, p10, p01, p11) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
            for c in 0..4 {
                let top = p00[c] as f32 * (1.0 - tx) + p10[c] as f32 * tx;
                let bottom = p01[c] as f32 * (1.0 - tx) + p11[c] as f32 * tx;
                out.push((top * (1.0 - ty) + bottom * ty).round() as u8);
            }
        }
    }
    out
}

impl Adjust {
    /// The adjustments a photo `width` by `height` starts with: none,
    /// but for the enlargement a small one is given
    /// ([`Enlarge::for_photo`]).
    pub fn for_photo(width: u32, height: u32) -> Self {
        Self { enlarge: Enlarge::for_photo(width, height), ..Self::default() }
    }

    fn lookup_table(&self) -> [u8; 256] {
        let gain = ((self.contrast.clamp(-1.0, 0.999) + 1.0) * std::f32::consts::FRAC_PI_4).tan();
        std::array::from_fn(|i| {
            let mut value = i as f32 / 255.0;
            if self.invert {
                value = 1.0 - value;
            }
            value = (value - 0.5) * gain + 0.5 + self.brightness;
            (value.clamp(0.0, 1.0) * 255.0).round() as u8
        })
    }

    /// Returns the RGBA image to display and the luminance to sample
    /// from, of a photo `width` by `height`: both at the size the
    /// enlargement makes of it ([`Enlarge::size`]), which is the
    /// luminance's. With a filter on, the display is the filtered
    /// luminance in grey, so that what is shown is what is sampled.
    ///
    /// The lens comes first, which bends the photo about its own centre.
    /// Then the photo is sharpened and resized, in its colours: sharpened
    /// at its own size and then enlarged, where a mask a pixel or two
    /// wide is as wide as what a lens blurred and the enlargement is the
    /// better for the mask's overshoot, which it holds its own result
    /// to; or reduced and then sharpened, at the size the reduction left
    /// it. A blur, the mask the other way up, is laid at the same size.
    /// The tone comes after, a curve for each pixel alone, and the
    /// filter last, whose neighbourhood is measured in the pixels the
    /// photo is worked on at.
    ///
    /// The luminance of a photo enlarged keeps that of the photo at its
    /// own size beside it ([`Luma::unenlarged`]), under the same
    /// adjustments less the filter, whose neighbourhood would be that
    /// many times too wide there. An enlargement the photo's size does
    /// not allow is held to the most that it does ([`Enlarge::within`]).
    pub fn apply(&self, rgba: &[u8], width: usize, height: usize) -> (Vec<u8>, Luma) {
        assert_eq!(rgba.len(), width * height * 4);
        if width == 0 || height == 0 {
            return self.tone(rgba.to_vec(), width, height);
        }
        // Nothing is resampled while there is no distortion to take out.
        let straightened;
        let rgba = if self.k1 != 0.0 {
            straightened = undistort(rgba, width, height, self.k1);
            &straightened[..]
        } else {
            rgba
        };
        let enlarge = self.enlarge.within(width as u32, height as u32);
        let (to_width, to_height) = enlarge.size(width, height);
        let mask = mask_for(self.sharpen);
        if enlarge <= Enlarge::NONE {
            let reduced;
            let rgba = if (to_width, to_height) != (width, height) {
                reduced = resize(rgba, width, height, 4, to_width, to_height);
                &reduced[..]
            } else {
                rgba
            };
            let sharp = match mask {
                Some((sigma, amount)) => unsharp_rgba(rgba, to_width, to_height, sigma, amount),
                None => rgba.to_vec(),
            };
            return self.tone(sharp, to_width, to_height);
        }
        // The photo at its own size as it is enlarged from, when that is
        // not the photo as given, and the photo enlarged.
        let (own, larger) = match mask {
            Some((sigma, amount)) => {
                let sharp = unsharp_values(rgba, width, height, sigma, amount);
                let larger = resize_into(&sharp, width, height, 4, to_width, to_height);
                (Some(sharp.iter().map(|&value| u8::from_f32(value)).collect::<Vec<u8>>()), larger)
            }
            None => (None, resize(rgba, width, height, 4, to_width, to_height)),
        };
        let (display, mut luma) = self.tone(larger, to_width, to_height);
        if enlarge.whole().is_some() {
            let unfiltered = Adjust { filter: Filter::None, ..*self };
            luma.unenlarged = Some(Arc::new(unfiltered.tone(own.unwrap_or_else(|| rgba.to_vec()), width, height).1));
        }
        (display, luma)
    }

    /// The tone and the filter, over pixels already straightened,
    /// resized and sharpened: the RGBA image to display, which is made
    /// of those pixels where they stand, and the luminance to sample
    /// from. A photo enlarged is many times the photo, and is not held
    /// a second time to be toned.
    fn tone(&self, mut rgba: Vec<u8>, width: usize, height: usize) -> (Vec<u8>, Luma) {
        let table = self.lookup_table();
        let mut luma = vec![0.0f32; width * height];
        // A curve for each pixel alone: the pixels are toned a piece at
        // a time on the cores to spare, each piece with its luminance.
        let pieces: Vec<Mutex<(&mut [u8], &mut [f32])>> = rgba.chunks_mut(4 * TONED_AT_ONCE).zip(luma.chunks_mut(TONED_AT_ONCE)).map(Mutex::new).collect();
        work::map(&pieces, |piece| {
            let mut piece = piece.lock().expect("a piece is toned once, by one worker");
            let (pixels, values) = &mut *piece;
            for (pixel, value) in pixels.as_chunks_mut::<4>().0.iter_mut().zip(values.iter_mut()) {
                let [r, g, b] = [table[pixel[0] as usize], table[pixel[1] as usize], table[pixel[2] as usize]];
                let shown = match self.channel {
                    Channel::Color => [r, g, b],
                    Channel::Red => [r; 3],
                    Channel::Green => [g; 3],
                    Channel::Blue => [b; 3],
                };
                *pixel = [shown[0], shown[1], shown[2], 255];
                let [r, g, b] = shown.map(|c| c as f32 / 255.0);
                *value = 0.299 * r + 0.587 * g + 0.114 * b;
            }
        });
        drop(pieces);
        if self.filter != Filter::None && width > 0 && height > 0 {
            luma = self.filter.apply(&luma, width, height, self.radius.clamp(1, Filter::MAX_RADIUS) as usize);
            for (pixel, &value) in rgba.as_chunks_mut::<4>().0.iter_mut().zip(&luma) {
                let grey = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
                pixel[0] = grey;
                pixel[1] = grey;
                pixel[2] = grey;
            }
        }
        (rgba, Luma::new(width, height, luma))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_settings_change_nothing() {
        let rgba = [10, 120, 250, 255, 0, 0, 0, 255];
        let (display, luma) = Adjust::default().apply(&rgba, 2, 1);
        assert_eq!(display, rgba);
        assert!((luma.data[0] - (0.299 * 10.0 + 0.587 * 120.0 + 0.114 * 250.0) / 255.0).abs() < 1e-6);
        assert_eq!(luma.data[1], 0.0);
    }

    /// Opaque pixels `width` by `height`, each of the colour `at` gives.
    fn pixels(width: usize, height: usize, at: impl Fn(usize, usize) -> [u8; 3]) -> Vec<u8> {
        (0..width * height)
            .flat_map(|i| {
                let [r, g, b] = at(i % width, i / width);
                [r, g, b, 255]
            })
            .collect()
    }

    #[test]
    fn a_photo_is_worked_on_at_a_fraction_or_a_whole_number_of_times_its_size() {
        assert_eq!(Enlarge::default(), Enlarge::NONE);
        assert_eq!((Enlarge::QUARTER.factor(), Enlarge::HALF.factor(), Enlarge::NONE.factor(), Enlarge::times(3).factor()), (0.25, 0.5, 1.0, 3.0));
        assert_eq!((Enlarge::HALF.whole(), Enlarge::NONE.whole(), Enlarge::times(24).whole()), (None, Some(1), Some(24)));
        assert_eq!(Enlarge::times(0), Enlarge::NONE);
        assert!(Enlarge::LADDER.windows(2).all(|pair| pair[1].factor() == 2.0 * pair[0].factor()), "each step twice the last");
        // A whole number of times is that many pixels to each; a photo
        // reduced comes to the nearest whole pixel, and never to none.
        assert_eq!(Enlarge::times(3).size(110, 60), (330, 180));
        assert_eq!((Enlarge::HALF.size(5000, 1501), Enlarge::QUARTER.size(5000, 1501)), ((2500, 751), (1250, 375)));
        assert_eq!((Enlarge::QUARTER.size(1, 9), Enlarge::NONE.size(7, 0)), ((1, 2), (7, 0)));
        // As it is shown, and read back.
        let shown: Vec<String> = [Enlarge::QUARTER, Enlarge::HALF, Enlarge::NONE, Enlarge::times(24)].iter().map(Enlarge::to_string).collect();
        assert_eq!(shown, ["0.25×", "0.5×", "1×", "24×"]);
        for step in Enlarge::LADDER.into_iter().chain([Enlarge::times(3), Enlarge::times(37)]) {
            assert_eq!(Enlarge::parse(&step.to_string()), Some(step));
        }
        assert_eq!((Enlarge::parse(" 3x "), Enlarge::parse("0.3"), Enlarge::parse("0.6"), Enlarge::parse("1.4")), (Some(Enlarge::times(3)), Some(Enlarge::QUARTER), Some(Enlarge::HALF), Some(Enlarge::NONE)));
        assert_eq!((Enlarge::parse("large"), Enlarge::parse("0"), Enlarge::parse("-2"), Enlarge::parse("")), (None, None, None, None));
        // A read made through a copy so many times as large again.
        assert_eq!((Enlarge::NONE.by(4), Enlarge::times(3).by(2), Enlarge::times(2).by(1), Enlarge::times(2).by(0)), (Enlarge::times(4), Enlarge::times(6), Enlarge::times(2), Enlarge::times(2)));
        assert_eq!((Enlarge::QUARTER.by(2), Enlarge::HALF.by(2), Enlarge::QUARTER.by(4)), (Enlarge::HALF, Enlarge::NONE, Enlarge::NONE));
        assert_eq!((Enlarge::QUARTER.by(3), Enlarge::HALF.by(3)), (Enlarge::NONE, Enlarge::times(2)), "the next whole number up");
    }

    #[test]
    fn a_small_photo_starts_enlarged_and_a_large_one_at_its_size() {
        // By the whole number of times that brings its shorter side to
        // 256 pixels, as far as its longer side allows.
        assert_eq!(Enlarge::for_photo(11, 11), Enlarge::times(24));
        assert_eq!(Enlarge::for_photo(110, 110), Enlarge::times(3));
        assert_eq!(Enlarge::for_photo(100, 60), Enlarge::times(5));
        assert_eq!(Enlarge::for_photo(255, 4000), Enlarge::NONE, "no whole factor fits");
        // A long thin render is judged by its height, within the limit
        // on its width.
        assert_eq!(Enlarge::for_photo(587, 86), Enlarge::times(3));
        assert_eq!((Enlarge::for_photo(3000, 100), Enlarge::for_photo(5000, 100)), (Enlarge::NONE, Enlarge::NONE));
        // A large photo keeps every pixel it has.
        assert_eq!((Enlarge::for_photo(256, 256), Enlarge::for_photo(640, 480), Enlarge::for_photo(8192, 4096)), (Enlarge::NONE, Enlarge::NONE, Enlarge::NONE));
        assert_eq!(Adjust::for_photo(110, 110), Adjust { enlarge: Enlarge::times(3), ..Adjust::default() });
        assert_eq!(Adjust::for_photo(640, 480), Adjust::default());
        // What a photo may be worked on at: enlarged four times whatever
        // its size, and further within 4096 pixels on its longer side;
        // at its own size whatever that is; and reduced, however little
        // that leaves of it.
        assert_eq!(Enlarge::ALWAYS, Enlarge::times(4));
        assert!(Enlarge::times(2).fits(20_000, 20_000) && Enlarge::times(4).fits(20_000, 20_000) && !Enlarge::times(5).fits(20_000, 20_000));
        assert!(Enlarge::times(8).fits(512, 100) && !Enlarge::times(8).fits(513, 100));
        assert!(Enlarge::NONE.fits(1, 1) && Enlarge::NONE.fits(20_000, 20_000));
        assert!(Enlarge::HALF.fits(20, 20) && Enlarge::QUARTER.fits(20_000, 16));
        assert_eq!(Enlarge::times(64).within(1000, 700), Enlarge::times(4));
        assert_eq!((Enlarge::times(2).within(5000, 1500), Enlarge::times(8).within(5000, 1500)), (Enlarge::times(2), Enlarge::times(4)));
        assert_eq!(Enlarge::times(64).within(330, 330), Enlarge::times(12));
        assert_eq!((Enlarge::QUARTER.within(40, 40), Enlarge::HALF.within(16, 900)), (Enlarge::QUARTER, Enlarge::HALF));
        // Without its being asked for, a photo is enlarged only within
        // that many pixels, as one is when it is loaded.
        assert!(Enlarge::times(2).fits_unasked(2048, 100) && !Enlarge::times(2).fits_unasked(2049, 100));
        assert!(Enlarge::NONE.fits_unasked(20_000, 20_000) && Enlarge::HALF.fits_unasked(20_000, 20_000));
        assert!(Enlarge::for_photo(86, 587).fits_unasked(86, 587) && Enlarge::for_photo(11, 11).fits_unasked(11, 11));
        assert_eq!(Enlarge::times(3).within(110, 110), Enlarge::times(3));
        // The steps offered: each twice the last as far as they fit,
        // down to a reduction that leaves 32 pixels on the shorter side,
        // and what the photo starts at among them.
        let steps = |width, height| Enlarge::steps_for(width, height).iter().map(Enlarge::to_string).collect::<Vec<_>>().join(" ");
        assert_eq!(steps(110, 110), "0.5× 1× 2× 3× 4× 8× 16× 32×");
        assert_eq!(steps(350, 350), "0.25× 0.5× 1× 2× 4× 8×");
        assert_eq!(steps(5000, 1500), "0.25× 0.5× 1× 2× 4×", "a large photo may still be enlarged four times");
        assert_eq!(steps(11, 11), "1× 2× 4× 8× 16× 24× 32× 64×");
        assert_eq!((steps(64, 900), steps(63, 900)), ("0.5× 1× 2× 4×".to_owned(), "1× 2× 4×".to_owned()));
    }

    #[test]
    fn a_photo_is_enlarged_and_reduced_in_its_colours() {
        // Red ink on the left half and white paper on the right.
        let ink = |width: usize, height: usize| pixels(width, height, |x, _| if x < width / 2 { [200, 30, 40] } else { [255; 3] });
        let own = ink(6, 4);
        let twice = Adjust { enlarge: Enlarge::times(2), ..Default::default() };
        let (display, luma) = twice.apply(&own, 6, 4);
        assert_eq!((display.len(), luma.width, luma.height), (12 * 8 * 4, 12, 8));
        // The kernel's own enlargement of every channel, and the
        // luminance of that.
        assert_eq!(display, resize(&own, 6, 4, 4, 12, 8));
        assert_eq!(&display[..4], &[200, 30, 40, 255]);
        assert_eq!(luma.data, Adjust::default().apply(&display, 12, 8).1.data);
        // Beside it, the photo at its own size.
        let small = luma.unenlarged.as_ref().expect("the photo's own pixels are kept");
        assert_eq!(**small, Adjust::default().apply(&own, 6, 4).1);
        assert!(small.unenlarged.is_none());
        // Reduced, it keeps nothing beside it, as at its own size.
        let wide = ink(80, 66);
        let half = Adjust { enlarge: Enlarge::HALF, ..Default::default() };
        let (display, luma) = half.apply(&wide, 80, 66);
        assert_eq!((luma.width, luma.height, luma.unenlarged.is_none()), (40, 33, true));
        assert_eq!(display, resize(&wide, 80, 66, 4, 40, 33));
        assert_eq!((&display[..4], &display[39 * 4..40 * 4]), (&[200, 30, 40, 255][..], &[255; 4][..]));
        assert!(Adjust::default().apply(&wide, 80, 66).1.unenlarged.is_none());
        // More than the photo's size allows is held to what it does.
        let strip = ink(2049, 2);
        let (display, luma) = Adjust { enlarge: Enlarge::times(8), ..Default::default() }.apply(&strip, 2049, 2);
        assert_eq!((luma.width, luma.height, luma.unenlarged.is_some()), (4 * 2049, 8, true));
        assert_eq!(display, resize(&strip, 2049, 2, 4, 4 * 2049, 8));
        // A reduction is made however little it leaves: a part cropped
        // from a photo reduced is worked on as the photo was.
        let (_, luma) = Adjust { enlarge: Enlarge::QUARTER, ..Default::default() }.apply(&own, 6, 4);
        assert_eq!((luma.width, luma.height), (2, 1));
    }

    #[test]
    fn a_photo_enlarged_keeps_its_own_size_adjusted_but_not_filtered() {
        // Ink on the left half and paper on the right, enlarged twice.
        let own = pixels(4, 2, |x, _| if x < 2 { [0; 3] } else { [255; 3] });
        let adjust = Adjust { invert: true, filter: Filter::Blur, radius: 1, enlarge: Enlarge::times(2), ..Default::default() };
        let (display, luma) = adjust.apply(&own, 4, 2);
        assert_eq!((display.len(), luma.width, luma.height), (8 * 4 * 4, 8, 4));
        // The working pixels are inverted and blurred across the edge...
        assert!((luma.data[0] - 1.0).abs() < 1e-6 && (luma.data[7] - 0.0).abs() < 1e-6, "{:?}", &luma.data[..8]);
        assert!(luma.data[3] < 0.9 && luma.data[4] > 0.1, "{:?}", &luma.data[..8]);
        // ...and the photo's own inverted alike, with its edge as it was.
        let small = luma.unenlarged.as_ref().expect("the photo's own pixels are kept");
        assert_eq!((small.width, small.height), (4, 2));
        for (value, want) in small.data.iter().zip([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]) {
            assert!((value - want).abs() < 1e-6, "{:?}", small.data);
        }
        assert!(small.unenlarged.is_none());
        // A photo worked on at its own size keeps nothing beside it.
        assert!(Adjust { enlarge: Enlarge::NONE, ..adjust }.apply(&own, 4, 2).1.unenlarged.is_none());
    }

    #[test]
    fn a_photo_is_sharpened_in_its_colours() {
        // A soft edge from red ink to white paper: two pixels between
        // them, a third and two thirds of the way.
        let soft = |x: usize, width: usize| match (x + 1).cmp(&(width / 2)) {
            std::cmp::Ordering::Less => 0.0,
            std::cmp::Ordering::Equal => 1.0 / 3.0,
            std::cmp::Ordering::Greater if x == width / 2 => 2.0 / 3.0,
            std::cmp::Ordering::Greater => 1.0,
        };
        let edge = |width: usize, height: usize| pixels(width, height, |x, _| [220, (40.0 + 180.0 * soft(x, width)) as u8, (60.0 + 160.0 * soft(x, width)) as u8]);
        let photo = edge(40, 12);
        assert_eq!(Adjust { sharpen: 0.0, ..Default::default() }.apply(&photo, 40, 12).0, photo, "no sharpening leaves the photo alone");
        let pixel = |display: &[u8], x: usize| [display[(5 * 40 + x) * 4], display[(5 * 40 + x) * 4 + 1], display[(5 * 40 + x) * 4 + 2]];
        let mut steps = Vec::new();
        for sharpen in [0.25, 0.5, 1.0] {
            let (display, luma) = Adjust { sharpen, ..Default::default() }.apply(&photo, 40, 12);
            // Flat tone keeps its colour; the channel with no edge in it
            // is as it was all the way across; and the others step more
            // steeply between the two pixels of the edge.
            assert_eq!((pixel(&display, 3), pixel(&display, 36)), ([220, 40, 60], [220, 220, 220]), "{sharpen}");
            assert!((0..40).all(|x| pixel(&display, x)[0] == 220), "{sharpen}");
            let step = i32::from(pixel(&display, 20)[1]) - i32::from(pixel(&display, 19)[1]);
            assert!(step > 60, "{sharpen}: {step}");
            steps.push(step);
            // What is sampled is the luminance of what is shown.
            assert_eq!(luma, Adjust::default().apply(&display, 40, 12).1);
        }
        assert!(steps.windows(2).all(|pair| pair[0] < pair[1]), "the more the sharper: {steps:?}");
        // Sharpened at its own size and then enlarged, with the photo
        // at its own size, sharpened, beside it.
        let sharp = Adjust { sharpen: 0.5, ..Default::default() };
        let both = Adjust { enlarge: Enlarge::times(3), ..sharp };
        let (display, luma) = both.apply(&photo, 40, 12);
        assert_eq!((display.len(), luma.width, luma.height), (120 * 36 * 4, 120, 36));
        assert_eq!(luma.unenlarged.as_deref(), Some(&sharp.apply(&photo, 40, 12).1));
        assert_eq!(&display[(15 * 120 + 9) * 4..(15 * 120 + 9) * 4 + 4], &[220, 40, 60, 255]);
        // Reduced and then sharpened, at the size the reduction left.
        let large = edge(80, 64);
        let reduced = Adjust { enlarge: Enlarge::HALF, ..Default::default() }.apply(&large, 80, 64).0;
        let (display, luma) = Adjust { enlarge: Enlarge::HALF, ..sharp }.apply(&large, 80, 64);
        assert_eq!((luma.width, luma.height), (40, 32));
        assert_eq!(display, sharp.apply(&reduced, 40, 32).0);
    }

    #[test]
    fn a_photo_is_blurred_in_its_colours_below_nought() {
        // The soft edge from red ink to white paper of the sharpening
        // test, with specks of paper in the ink and of ink in the paper.
        let soft = |x: usize, width: usize| match (x + 1).cmp(&(width / 2)) {
            std::cmp::Ordering::Less => 0.0,
            std::cmp::Ordering::Equal => 1.0 / 3.0,
            std::cmp::Ordering::Greater if x == width / 2 => 2.0 / 3.0,
            std::cmp::Ordering::Greater => 1.0,
        };
        let edge = |width: usize, height: usize| {
            pixels(width, height, |x, y| {
                let speck = (x, y) == (8, 5) || (x, y) == (31, 5);
                let t = if speck { 1.0 - soft(x, width) } else { soft(x, width) };
                [220, (40.0 + 180.0 * t) as u8, (60.0 + 160.0 * t) as u8]
            })
        };
        let photo = edge(40, 12);
        let pixel = |display: &[u8], x: usize| [display[(5 * 40 + x) * 4], display[(5 * 40 + x) * 4 + 1], display[(5 * 40 + x) * 4 + 2]];
        assert_eq!((pixel(&photo, 8), pixel(&photo, 31)), ([220, 220, 220], [220, 40, 60]), "the specks");
        let mut spreads = Vec::new();
        let mut specks = Vec::new();
        for sharpen in [-0.25, -0.5, -1.0] {
            let (display, luma) = Adjust { sharpen, ..Default::default() }.apply(&photo, 40, 12);
            // Flat tone far from the edge and the specks keeps its
            // colour; the channel with no edge in it is as it was all
            // the way across; the others step no more steeply between
            // the two pixels of the edge, and run out past them into
            // the flat tone either side, which the edge did not; and
            // the specks fade toward their surroundings, in their
            // colours.
            assert_eq!((pixel(&display, 3), pixel(&display, 36)), ([220, 40, 60], [220, 220, 220]), "{sharpen}");
            assert!((0..40).all(|x| pixel(&display, x)[0] == 220), "{sharpen}");
            let green = |x: usize| i32::from(pixel(&display, x)[1]);
            let step = green(20) - green(19);
            assert!(step > 0 && step <= 60, "{sharpen}: {step}");
            assert!(green(18) > 40 && green(21) < 220, "{sharpen}: {} and {}", green(18), green(21));
            spreads.push(green(21) - green(18));
            let speck = pixel(&display, 8);
            assert!(speck[1] < 220 && speck[1] > 40 && speck[2] < 220 && speck[2] > 60, "{sharpen}: {speck:?}");
            assert!(speck[1] < speck[2], "{sharpen}: in its colours, {speck:?}");
            specks.push(speck[1]);
            // Nothing overshoots: every channel lies between the tones
            // about it.
            assert!(display.as_chunks::<4>().0.iter().all(|&[_, g, b, _]| (40..=220).contains(&g) && (60..=220).contains(&b)), "{sharpen}");
            // What is sampled is the luminance of what is shown.
            assert_eq!(luma, Adjust::default().apply(&display, 40, 12).1);
        }
        assert!(spreads.windows(2).all(|pair| pair[0] > pair[1]), "the further down the softer: {spreads:?}");
        assert!(specks.windows(2).all(|pair| pair[0] > pair[1]), "and the fainter the specks: {specks:?}");
        // Half way down is the whole of a Gaussian blur a pixel wide:
        // a speck a pixel across on flat ground keeps the middle weight
        // of the blur, across and down, over the ground, and no more.
        let (display, _) = Adjust { sharpen: -0.5, ..Default::default() }.apply(&photo, 40, 12);
        let middle = 1.0 / (1.0 + 2.0 * ((-0.5f32).exp() + (-2.0f32).exp() + (-4.5f32).exp()));
        let want = 40.0 + 180.0 * middle * middle;
        assert!((f32::from(pixel(&display, 8)[1]) - want).abs() < 1.0, "{} against {want}", pixel(&display, 8)[1]);
        // Blurred at its own size and then enlarged, with the photo at
        // its own size, blurred, beside it; and reduced and then
        // blurred, at the size the reduction left.
        let blur = Adjust { sharpen: -0.5, ..Default::default() };
        let (display, luma) = Adjust { enlarge: Enlarge::times(3), ..blur }.apply(&photo, 40, 12);
        assert_eq!((display.len(), luma.width, luma.height), (120 * 36 * 4, 120, 36));
        assert_eq!(luma.unenlarged.as_deref(), Some(&blur.apply(&photo, 40, 12).1));
        assert_eq!(&display[(15 * 120 + 9) * 4..(15 * 120 + 9) * 4 + 4], &[220, 40, 60, 255]);
        let large = edge(80, 64);
        let reduced = Adjust { enlarge: Enlarge::HALF, ..Default::default() }.apply(&large, 80, 64).0;
        let (display, luma) = Adjust { enlarge: Enlarge::HALF, ..blur }.apply(&large, 80, 64);
        assert_eq!((luma.width, luma.height), (40, 32));
        assert_eq!(display, blur.apply(&reduced, 40, 32).0);
    }

    #[test]
    fn a_photo_sharpened_and_enlarged_reads_as_a_copy_of_it_made_for_a_read() {
        use crate::imaging::enhance::{Enhance, Enhanced, Sharpen};
        // Busy grey pixels. What a read sees through a copy of the photo
        // about a symbol, sharpened and enlarged, the whole photo under
        // the same two adjustments shows, to the half of a grey level a
        // byte rounds away.
        let (w, h) = (48, 36);
        let photo = pixels(w, h, |x, y| [((((x * 31 + y * 17) % 23) * 255 / 22 + ((x / 5 + y / 3) % 2) * 255) / 2) as u8; 3]);
        let plain = Adjust::default().apply(&photo, w, h).1;
        let near = |adjusted: &Luma, copy: &Luma, what: &str| {
            assert_eq!((adjusted.width, adjusted.height), (copy.width, copy.height), "{what}");
            let worst = adjusted.data.iter().zip(&copy.data).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(worst < 0.0025, "{what}: {worst}");
        };
        for (sharpen, enlarge) in [(Sharpen::Light, 1u8), (Sharpen::Strong, 1), (Sharpen::Light, 3), (Sharpen::Strong, 4), (Sharpen::Off, 2)] {
            let copy = Enhanced::window(&plain, [0, 0, w, h], Enhance { enlarge, sharpen }).unwrap();
            let adjust = Adjust { sharpen: sharpen.amount(), enlarge: Enlarge::NONE.by(enlarge), ..Default::default() };
            near(&adjust.apply(&photo, w, h).1, &copy.luma, &format!("{sharpen:?}, {enlarge} times"));
        }
        // A photo enlarged already is read through a copy made from the
        // pixels it had, in one step: so is the photo enlarged as many
        // times again.
        let twice = Adjust { enlarge: Enlarge::times(2), ..Default::default() };
        let copy = Enhanced::window(&twice.apply(&photo, w, h).1, [0, 0, 2 * w, 2 * h], Enhance { enlarge: 2, sharpen: Sharpen::Light }).unwrap();
        let adjust = Adjust { sharpen: Sharpen::Light.amount(), enlarge: twice.enlarge.by(2), ..Default::default() };
        assert_eq!(adjust.enlarge, Enlarge::times(4));
        near(&adjust.apply(&photo, w, h).1, &copy.luma, "enlarged twice, and twice again");
    }

    #[test]
    fn channel_invert_and_contrast() {
        let rgba = [200, 40, 90, 255];
        let red = Adjust { channel: Channel::Red, ..Default::default() };
        assert_eq!(red.apply(&rgba, 1, 1).0, [200, 200, 200, 255]);
        let inverted = Adjust { invert: true, ..Default::default() };
        assert_eq!(inverted.apply(&rgba, 1, 1).0, [55, 215, 165, 255]);
        let hard = Adjust { contrast: 1.0, ..Default::default() };
        assert_eq!(hard.apply(&rgba, 1, 1).0, [255, 0, 0, 255]);
        let bright = Adjust { brightness: 0.5, ..Default::default() };
        assert_eq!(bright.apply(&[0, 128, 255, 255], 1, 1).0[..3], [128, 255, 255]);
    }

    #[test]
    fn filters_mend_specks_dots_and_grain() {
        // A 7×7 light field with one dark speck in the middle.
        let (w, h) = (7, 7);
        let mut speck = vec![1.0f32; w * h];
        speck[3 * w + 3] = 0.0;
        let out = Filter::Median.apply(&speck, w, h, 1);
        assert!(out.iter().all(|&v| v == 1.0), "the median removes a lone speck");
        let out = Filter::Blur.apply(&speck, w, h, 1);
        assert!((out[3 * w + 3] - 8.0 / 9.0).abs() < 1e-6 && out[0] == 1.0, "the blur spreads it");
        // A dark module printed as a dotted grid: every other pixel dark.
        let dotted: Vec<f32> = (0..w * h).map(|i| if (i % w) % 2 == 0 && (i / w) % 2 == 0 { 0.0 } else { 1.0 }).collect();
        let closed = Filter::Close.apply(&dotted, w, h, 1);
        assert!(closed.iter().all(|&v| v == 0.0), "closing joins the dots into solid dark: {closed:?}");
        // Closing a solid light field leaves it alone. A light gap three
        // pixels wide survives a radius of one, where one of two pixels is
        // closed: the darkest-of-three pass reaches one pixel each way.
        assert!(Filter::Close.apply(&vec![1.0f32; w * h], w, h, 1).iter().all(|&v| v == 1.0));
        let mut gap = vec![0.0f32; w * h];
        let mut narrow = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 2..5 {
                gap[y * w + x] = 1.0;
            }
            narrow[y * w + 3] = 1.0;
            narrow[y * w + 4] = 1.0;
        }
        let closed = Filter::Close.apply(&gap, w, h, 1);
        assert!(closed.contains(&1.0), "{closed:?}");
        assert!(Filter::Close.apply(&narrow, w, h, 1).iter().all(|&v| v == 0.0));
        assert_eq!(Filter::None.apply(&speck, w, h, 2), speck);
        // Through the adjustments the display turns grey and follows the
        // filtered luminance.
        let rgba: Vec<u8> = speck.iter().flat_map(|&v| [(v * 255.0) as u8, 0, 0, 255]).collect();
        let (display, luma) = Adjust { filter: Filter::Median, radius: 1, ..Default::default() }.apply(&rgba, w, h);
        assert!(luma.data.iter().all(|&v| (v - 0.299).abs() < 1e-3), "{:?}", &luma.data[..3]);
        assert_eq!(&display[(3 * w + 3) * 4..(3 * w + 3) * 4 + 4], &[76, 76, 76, 255]);
    }

    #[test]
    fn the_lens_term_is_estimated_from_the_anchors() {
        use crate::imaging::homography::Homography;
        // A 25-module symbol on a plane, photographed through a lens with
        // barrel distortion: every point of it is pulled towards the
        // centre of the photo, as `undistort` reads a straightened photo
        // from its shot.
        let (w, h) = (800, 600);
        let quad = [[150.0, 120.0], [650.0, 100.0], [680.0, 520.0], [130.0, 500.0]];
        let plane = Homography::rect_to_quad(25.0, 25.0, quad).unwrap();
        let truth = -0.12f32;
        // The plane is what the photo straightened with the true term
        // shows; the photo as shot is the one straightened with none.
        let distort = |p: [f64; 2]| remap_point(p, w, h, truth, 0.0);
        let sites = [
            [0.0, 0.0], [25.0, 0.0], [25.0, 25.0], [0.0, 25.0], [3.5, 3.5], [21.5, 3.5], [3.5, 21.5], [18.0, 18.0],
            [8.5, 6.5], [16.5, 6.5], [6.5, 8.5], [6.5, 16.5],
        ];
        // The pairs as seen in the photo as shot: straightened with k1 = 0.
        let pairs: Vec<([f64; 2], [f64; 2])> = sites.iter().map(|&at| (at, distort(plane.apply(at[0], at[1])))).collect();
        let estimate = estimate_k1(w, h, 0.0, &pairs).expect("a term is found");
        assert!((f64::from(estimate.k1) - f64::from(truth)).abs() < 0.01, "{estimate:?}");
        assert!(estimate.rms_after < estimate.rms_before / 5.0, "{estimate:?}");
        // The same pairs, in the photo straightened with the right term,
        // lie on the plane: nothing to gain.
        let straight: Vec<([f64; 2], [f64; 2])> =
            pairs.iter().map(|&(at, px)| (at, remap_point(px, w, h, 0.0, truth))).collect();
        assert!(estimate_k1(w, h, truth, &straight).is_none());
        // Remapping goes there and back.
        for &(_, px) in &pairs {
            let there = remap_point(px, w, h, 0.0, 0.2);
            let back = remap_point(there, w, h, 0.2, 0.0);
            assert!((back[0] - px[0]).abs() < 1e-6 && (back[1] - px[1]).abs() < 1e-6, "{px:?} -> {there:?} -> {back:?}");
        }
        assert_eq!(remap_point([400.0, 300.0], w, h, 0.0, 0.2), [400.0, 300.0], "the centre stays put");
        // A barrel wider than the slider offers: the estimate stops at
        // the slider's end, where the polish would have stepped a hair
        // past it, so the term set is one the model's inverse is vouched
        // for.
        let wide = |p: [f64; 2]| remap_point(p, w, h, -0.32, 0.0);
        let wider: Vec<([f64; 2], [f64; 2])> = sites.iter().map(|&at| (at, wide(plane.apply(at[0], at[1])))).collect();
        let estimate = estimate_k1(w, h, 0.0, &wider).expect("a term is found");
        assert_eq!(estimate.k1, K1_MIN, "{estimate:?}");
        assert!(K1_RANGE.contains(&estimate.k1) && f64::from(K1_MIN) > -1.0 / 3.0);
        // Too few pairs: nothing.
        assert!(estimate_k1(w, h, 0.0, &pairs[..6]).is_none());
    }

    #[test]
    fn the_lens_model_climbs_beyond_the_photo() {
        // Within the half diagonal the model is Brown's; beyond it the
        // source keeps climbing with the radius at every term the slider
        // offers, where the barrel terms once turned back, so that a
        // point far outside the photo has one source, further out than
        // any within, and comes back where it was.
        for k1 in [K1_MIN, -0.2, -0.1, 0.0, 0.1, K1_MAX] {
            let mut last = 0.0f32;
            for i in 1..=400 {
                let r = i as f32 / 100.0;
                let source = source_radius(r, k1);
                if r <= 1.0 {
                    assert!((source - r * (1.0 + k1 * r * r)).abs() < 1e-6, "k1 {k1}, r {r}");
                }
                assert!(source > last, "k1 {k1}: r {r} reads from {source}, a step nearer from {last}");
                last = source;
            }
        }
        let (w, h) = (4000, 3000);
        for point in [[-3000.0, -2000.0], [9000.0, 1500.0], [2000.0, 12000.0], [100.0, 100.0]] {
            for (from, to) in [(K1_MIN, 0.1f32), (K1_MAX, K1_MIN), (0.0, K1_MIN)] {
                let there = remap_point(point, w, h, from, to);
                // On the same ray from the centre, the same side of it.
                for axis in 0..2 {
                    let centre = [2000.0, 1500.0][axis];
                    assert_eq!((there[axis] - centre).signum(), (point[axis] - centre).signum(), "{point:?} {from} -> {to}: {there:?}");
                }
                let back = remap_point(there, w, h, to, from);
                assert!((back[0] - point[0]).abs() < 1e-5 && (back[1] - point[1]).abs() < 1e-5, "{point:?} {from} -> {to}: {there:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn lens_model_and_identity() {
        assert!((source_radius(0.5, 0.1) - 0.5 * 1.025).abs() < 1e-6);
        assert_eq!(source_radius(0.5, 0.0), 0.5);
        // A 160 × 120 photo has a diagonal of 200, so half of it is 100
        // and the normalised radius is plain to reckon; a ramp along x
        // makes every bilinear read exact.
        let (w, h) = (160, 120);
        let ramp: Vec<u8> = (0..w * h).flat_map(|i| [(i % w) as u8, 0, 0, 255]).collect();
        assert_eq!(undistort(&ramp, w, h, 0.0), ramp, "k1 = 0 is the identity");
        let (display, _) = Adjust::default().apply(&ramp, w, h);
        assert_eq!(display, ramp);
        // Straight right of the centre, the pixel whose centre is 49.5 px
        // out on the row just above the middle lies at r = 0.4950, and
        // with k1 = 0.1 reads from 1.0245 times as far out.
        let out = undistort(&ramp, w, h, 0.1);
        let (x, y) = (129usize, 59usize);
        let (dx, dy) = (x as f32 + 0.5 - 80.0, y as f32 + 0.5 - 60.0);
        let r = (dx * dx + dy * dy).sqrt() / 100.0;
        let sx = 80.0 + dx * source_radius(r, 0.1) / r;
        let expected = (sx - 0.5).round() as u8;
        assert_eq!(out[(y * w + x) * 4], expected);
        assert!(expected > x as u8, "a positive k1 reads from further out");
        // The far corners fall off the photo and come back white.
        assert_eq!(&out[..4], &[255, 255, 255, 255]);
        let (display, luma) = Adjust { k1: 0.1, ..Default::default() }.apply(&ramp, w, h);
        assert_eq!(display[(y * w + x) * 4], expected);
        assert!((luma.data[0] - 1.0).abs() < 1e-5);
    }
}
