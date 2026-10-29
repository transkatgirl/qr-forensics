//! The adjustments made to the photo before it is shown and sampled: its
//! size, its lens, its sharpness, its tone, and a filter over each
//! pixel's neighbourhood.
//!
//! A photo is enlarged or reduced, and sharpened or blurred, as it is
//! toned: the whole of it, in its colours, so that what is shown is
//! what the pipeline reads. Both were once steps of their own, the enlargement a
//! part of loading a small photo and the two together a copy made about
//! one symbol for the sampling to read off ([`super::enhance`]), in
//! grey; a read that does not decode still makes itself such copies,
//! and what it read through one the photo so adjusted shows again.
//!
//! The size, the lens and the sharpness mix the photo's pixels: as they
//! are stored, or, where [`Adjust::light`] asks, as the light they stand
//! for, which is stored again before the tone. A lens mixes light, and
//! sRGB stores it on a curve that spends its levels on the darks: fine
//! ink on paper reduced as stored comes out darker than the light that
//! fell there, a pixel half black and half white at 128 where its light
//! stores as 188.
//!
//! Every adjustment is worked on the graphics card ([`super::gpu`]); what
//! is worked out here is what it is asked to do and where: the weights of
//! a resize, the box of the photo a part draws on, the curve of the tone.

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

use super::enhance::{gaussian, mask_for};
use super::deblur::{self, Spread};
use super::gpu::{Lens, Shown, Step};
use super::resize::{Sample, Taps};
use super::sample::{Light, encoded_light, linear_light};
use super::detect::most_shrunk;
use super::{AsShot, Luma, SharedRgba, gpu};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Channel {
    /// Full colour on screen, perceptual luminance for sampling.
    #[default]
    Color,
    Red,
    Green,
    Blue,
}

/// What the transparent parts of a photo are laid on before anything is
/// read of it: mid grey, #808080, unless another is chosen, which ink of
/// either kind stands out against; white, as a viewer shows them; or
/// black, for a symbol drawn in light ink on nothing, to be shown on a
/// dark page, which on white is light on light and no symbol at all. The
/// photo as loaded is flattened onto it
/// ([`super::photo::flatten_alpha_onto`]), and what the lens pulls in
/// from beyond the photo's edges is filled with it too ([`Adjust::k1`]).
/// So a photo with no transparency is the same on any but there, and
/// the app lays such a photo on white.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Backdrop {
    White,
    Black,
    #[default]
    Grey,
}

impl Backdrop {
    pub const ALL: [Backdrop; 3] = [Backdrop::White, Backdrop::Black, Backdrop::Grey];

    pub fn label(self) -> &'static str {
        match self {
            Backdrop::White => "White",
            Backdrop::Black => "Black",
            Backdrop::Grey => "Grey",
        }
    }

    /// What a channel of a pixel wholly transparent comes to on it.
    pub fn level(self) -> u8 {
        match self {
            Backdrop::White => 255,
            Backdrop::Black => 0,
            Backdrop::Grey => 128,
        }
    }
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
/// the tone adjustments: last of all, once the photo is enlarged, among
/// the pixels it is worked on at, as it is over a copy a read makes of
/// it once the copy is enlarged ([`Luma::as_shot`]). The finder scan
/// reads runs of whole pixels and breaks on a symbol printed as dots or
/// a scan full of specks, where the averaging of the module sampler
/// copes; the filter mends the photo for the scan. zxing-cpp's
/// `tryDenoise` closes 2D symbols before binarising them for the same
/// reason, and its `tryDownscale` blurs by shrinking.
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
    /// The blur the photo shows, measured off its own edges, undone
    /// ([`super::deblur`]): a photo out of focus, or shaken. It has no
    /// neighbourhood of its own to set; what it undoes is the blur
    /// measured over the whole photo, as a read that decoded only through
    /// a copy of the photo deblurred so made it, or inside an outline
    /// ([`Adjust::kernel`]), as far as its strength asks, however little.
    Deblur,
}

impl Filter {
    pub const ALL: [Filter; 4] = [Filter::None, Filter::Median, Filter::Close, Filter::Deblur];
    /// Widest neighbourhood offered, as a half-width in pixels.
    pub const MAX_RADIUS: u8 = 3;

    pub fn label(self) -> &'static str {
        match self {
            Filter::None => "None",
            Filter::Median => "Median",
            Filter::Close => "Close dots",
            Filter::Deblur => "Deblur",
        }
    }

    pub fn describe(self) -> &'static str {
        match self {
            Filter::None => "The photo as it is",
            Filter::Median => "Removes specks and dropouts a pixel or two across while keeping edges sharp",
            Filter::Close => "Joins the dots of a dot-matrix or inkjet print into solid dark modules",
            Filter::Deblur => {
                "Estimates the blur from edges in the photo (or inside the outline) and deconvolves it, for a photo \
                 so out of focus or shaken that the finder patterns' rings run into their centres. Works in linear \
                 light when Work in linear light is on"
            }
        }
    }

    /// Whether the filter has a neighbourhood whose radius may be set.
    pub fn has_radius(self) -> bool {
        matches!(self, Filter::Median | Filter::Close)
    }

    /// The filter over a `width` × `height` luminance image worked in
    /// `light`, with a neighbourhood `radius` pixels either side of each
    /// pixel. The median's cost grows with the square of the radius, so
    /// it is held to two. Worked on the graphics card ([`super::gpu`]),
    /// the image's edges repeated beyond it. Only Deblur asks the light:
    /// the median and closing pick one of the values about a pixel, which
    /// is the same value in either.
    pub fn apply(self, data: &[f32], width: usize, height: usize, radius: usize, light: Light) -> Vec<f32> {
        let radius = radius.max(1);
        match self {
            Filter::None => data.to_vec(),
            Filter::Median => super::gpu::median(data, width, height, radius.min(2)),
            Filter::Close => super::gpu::closed(data, width, height, radius),
            Filter::Deblur => super::deblur::undone(data, width, height, light),
        }
    }
}

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
    /// Stops the photo's exposure is moved by, within
    /// [`EXPOSURE_RANGE`]: the light each value stands for
    /// ([`linear_light`]) twice as much for each stop up, and half for
    /// each down, stored again and held at white. Made in light whatever
    /// light the photo is worked in ([`Self::light`]), after invert and
    /// before contrast, whose threshold it moves the photo across.
    pub exposure: f32,
    /// -1.0 (flat grey) to 1.0 (hard threshold at mid grey).
    pub contrast: f32,
    /// Neighbourhood filter applied last.
    pub filter: Filter,
    /// Half-width of the filter's neighbourhood, in pixels.
    pub radius: u8,
    /// Radial lens distortion to take out once the photo is resized, as
    /// Brown's model has it, about the photo's centre: negative
    /// straightens barrel, positive pincushion; 0.0 leaves the photo as
    /// shot. Corners and anchors are in the straightened image.
    pub k1: f32,
    /// The size the photo is worked on at, against its own. Corners and
    /// anchors are in the pixels it comes to.
    pub enlarge: Enlarge,
    /// How far the photo is sharpened by an unsharp mask, from 0.0 for
    /// not at all to 2.0, or below nought blurred instead, with the
    /// mask's own Gaussian, as far as -2.0: see [`mask_for`]. The mask
    /// is laid on the photo after it is resized, as wide in the photo's
    /// own pixels, or in those a reduction leaves: on the photo
    /// enlarged, as many times as wide.
    pub sharpen: f32,
    /// The blur the Deblur filter undoes, in the photo's own pixels, as
    /// measured inside an outline, or as a read that decoded through a
    /// copy of the photo undid it: `None` for the blur the photo shows,
    /// measured over the whole of it whenever it is filtered.
    pub kernel: Option<Spread>,
    /// How far the Deblur filter undoes the blur: its kernel spread so
    /// many times as wide, from nought, which leaves the photo as it is,
    /// past 1.0, as measured, to [`STRENGTH_RANGE`]'s end.
    pub strength: f32,
    /// What the photo's transparent parts are laid on. The photo is
    /// flattened onto it as it is loaded, before every other adjustment,
    /// which are made to the photo so flattened: whoever holds the photo
    /// hands them the pixels on this backdrop. What the lens term draws in
    /// from beyond the photo's edges shows it too: those were once white
    /// whatever it was, wedges at the corners of a photo laid on grey or
    /// black that stood out as hard as any edge of the symbol.
    pub backdrop: Backdrop,
    /// What the photo's pixels are mixed as where it is resized,
    /// straightened and sharpened or blurred: as they are stored, or as
    /// the light they stand for ([`Light`]), as a lens mixed them, and
    /// stored again before the tone. A photo made in light is searched and
    /// read in it besides: every copy of it is made again from the photo as
    /// loaded in light ([`Luma::as_shot`]), what it is reduced to for a
    /// search of the whole of it is reduced in light ([`Luma::resized`]),
    /// and its modules are read as light ([`Luma::light`]). The tone is a
    /// curve of what is stored either way. The median and the closing pick
    /// one of the values about each pixel, whose order a curve keeps, and
    /// are what they are either way; Deblur measures and undoes the blur
    /// in this light too, after the tone, so that a photo inverted or toned
    /// for contrast, whose values are off sRGB's curve, is deblurred as
    /// stored where it is worked so. Deblur once worked in light always,
    /// and undid an inverted photo's blur as if its paper were its ink.
    pub light: Light,
}

/// What is shown of a part of the photo and what is read of it, filtered
/// on the graphics card ([`Adjust::filtered_on`]), with the blur its
/// Deblur filter undid.
type Filtered = (Shown, Vec<f32>, Option<Spread>);

/// Whether what a part of the photo shows is made, and where it is left:
/// not at all, for a copy that is read and never shown; taken off the
/// graphics card as bytes; or left on it where it was made there whole,
/// for a window drawn by the card to lay into its texture
/// ([`gpu::shared_device`]), in its colours, or in the grey of what is
/// read of it, as the view in grey shows the photo.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Show {
    Nothing,
    Bytes,
    Card,
    Grey,
}

impl Show {
    fn of(shown: bool) -> Self {
        if shown { Show::Bytes } else { Show::Nothing }
    }
}

/// How far the exposure goes, in stops ([`Adjust::exposure`]).
pub const EXPOSURE_RANGE: std::ops::RangeInclusive<f32> = -4.0..=4.0;

/// How far the Deblur filter's strength goes ([`Adjust::strength`]).
pub const STRENGTH_RANGE: std::ops::RangeInclusive<f32> = 0.0..=2.0;

impl Default for Adjust {
    fn default() -> Self {
        Self {
            channel: Channel::Color,
            invert: false,
            exposure: 0.0,
            contrast: 0.0,
            filter: Filter::None,
            radius: 1,
            k1: 0.0,
            enlarge: Enlarge::NONE,
            sharpen: 0.0,
            kernel: None,
            strength: 1.0,
            backdrop: Backdrop::Grey,
            light: Light::Encoded,
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
    straightened(rgba, width, height, k1, [0, 0, width, height], 255)
}

/// The part `[x, y, w, h]` of the photo with its radial distortion taken
/// out about the whole photo's centre, as [`undistort`] takes it out of
/// the whole and pixel for pixel the same, read from no more of the
/// photo than the part draws on, and `fill` in every colour where that
/// falls off the photo: what it is laid on ([`Backdrop::level`]).
pub fn straightened(rgba: &[u8], width: usize, height: usize, k1: f32, [x, y, w, h]: [usize; 4], fill: u8) -> Vec<u8> {
    assert_eq!(rgba.len(), width * height * 4);
    let frame = [width as f32, height as f32];
    let [x0, y0, x1, y1] = shot_about(frame, k1, [x as f32, y as f32], (w, h), [1.0; 2], [width, height]);
    let region = if [x0, y0, x1, y1] == [0, 0, width, height] { Cow::Borrowed(rgba) } else { Cow::Owned(cut(rgba, width, 4, [x0, y0, x1, y1])) };
    let shot = Shot { pixels: &region, width: x1 - x0, height: y1 - y0, channels: 4, at: [x0 as f32, y0 as f32], step: [1.0; 2] };
    straighten(&shot, frame, k1, [x as f32, y as f32], (w, h), &[fill, fill, fill, 255])
}

/// Pixels of a photo as loaded, RGBA, that a part of it is adjusted
/// from ([`Adjust::made`]): `width` by `height` of them, beginning at
/// `at` in the photo, which is `photo` pixels across and down. `shared`
/// is the same bytes held to be shared, when they are, for the
/// luminance to keep beside it rather than a copy of them.
#[derive(Clone, Copy)]
pub(crate) struct Loaded<'a> {
    pub rgba: &'a [u8],
    pub width: usize,
    pub height: usize,
    pub at: [usize; 2],
    pub photo: [usize; 2],
    pub shared: Option<&'a SharedRgba>,
}

/// Pixels of a photo as shot, `channels` values to each, that a part of
/// it is straightened from: in the pixels of a frame the whole photo is
/// resized to, where they begin and how many of the frame's pixels apart
/// they lie.
pub(crate) struct Shot<'a, T> {
    pub pixels: &'a [T],
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub at: [f32; 2],
    pub step: [f32; 2],
}

/// The part `width` by `height` of a frame beginning at `out_at` in it,
/// the whole photo `frame` of the frame's pixels across and down, with
/// the photo's radial distortion `k1` taken out about its centre: each
/// pixel read from `shot` at [`source_radius`] along the same ray from
/// the centre, bilinearly, and `fill` where that falls off the photo.
/// The arithmetic is of the frame's own places, so that a part and the
/// whole straightened in a frame of whole pixels come to the same values
/// where they meet, whatever part of the photo as shot each is read
/// from: `shot` holds what the part draws on ([`drawn_on`]). A `shot`
/// that holds nothing is of a part that draws on none of the photo, and
/// the part is `fill` throughout. Straightened on the graphics card
/// ([`gpu::straightened`]).
pub(crate) fn straighten<T: Sample>(shot: &Shot<'_, T>, frame: [f32; 2], k1: f32, out_at: [f32; 2], (width, height): (usize, usize), fill: &[T]) -> Vec<T> {
    let channels = shot.channels;
    assert_eq!(shot.pixels.len(), shot.width * shot.height * channels);
    assert_eq!(fill.len(), channels);
    if width == 0 || height == 0 {
        return Vec::new();
    }
    if shot.width == 0 || shot.height == 0 {
        return fill.repeat(width * height);
    }
    let lens = gpu::Lens { frame, k1, slope: edge_slope(f64::from(k1)) as f32 };
    let fill: Vec<f32> = fill.iter().map(|value| value.to_f32()).collect();
    let made = gpu::straightened(T::handed(shot.pixels), (shot.width, shot.height), channels, [shot.at, shot.step], &lens, out_at, (width, height), &fill, T::BYTES);
    T::taken(made)
}

/// The box of a frame, `[x0, y0, x1, y1]` in its pixels, that the part
/// `width` by `height` of it beginning at `out_at` draws on straightened
/// ([`straighten`]): the sources of the pixels along the part's edges,
/// which bound those of the pixels within, since the straightening moves
/// every place along its ray from the centre and keeps the order of
/// places along it; and the pixels beside them that the bilinear
/// sampling reads, and one more for the edges' bending between their
/// pixels. Within the whole photo: a part whose every pixel is read from
/// past one edge of it, as a strip along that edge is where a lens term
/// straightens pincushion, draws on none of it, and comes to a box with
/// no pixels in it, on that edge.
pub(crate) fn drawn_on(frame: [f32; 2], k1: f32, out_at: [f32; 2], (width, height): (usize, usize)) -> [f32; 4] {
    let (w, h) = (frame[0], frame[1]);
    let (cx, cy) = (w / 2.0, h / 2.0);
    let norm = (w * w + h * h).sqrt() / 2.0;
    let source = |x: usize, y: usize| {
        let (dx, dy) = (out_at[0] + x as f32 + 0.5 - cx, out_at[1] + y as f32 + 0.5 - cy);
        let r = (dx * dx + dy * dy).sqrt() / norm;
        let f = if r > 0.0 { source_radius(r, k1) / r } else { 1.0 };
        [cx + dx * f - 0.5, cy + dy * f - 0.5]
    };
    let (last_x, last_y) = (width.saturating_sub(1), height.saturating_sub(1));
    let edges = (0..width).flat_map(|x| [source(x, 0), source(x, last_y)]).chain((0..height).flat_map(|y| [source(0, y), source(last_x, y)]));
    let [x0, y0, x1, y1] = edges.fold([f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY], |[x0, y0, x1, y1], [x, y]| {
        [x0.min(x), y0.min(y), x1.max(x), y1.max(y)]
    });
    [(x0.floor() - 1.0).clamp(0.0, w), (y0.floor() - 1.0).clamp(0.0, h), (x1.floor() + 3.0).clamp(0.0, w), (y1.floor() + 3.0).clamp(0.0, h)]
}

/// How far past what a straightening reads, in the pixels the photo is
/// resized from, the photo as shot is kept: past what the enlargement
/// draws on beyond a pixel, so that the part as shot enlarged is the
/// whole as shot enlarged where the straightening reads it.
pub(crate) const SHOT_MARGIN: usize = 4;

/// What of a photo as shot, `[x0, y0, x1, y1]` of its own `size` in
/// pixels, the part `width` by `height` of a frame beginning at `out_at`
/// draws on straightened, the frame `scale` of its pixels to one of the
/// photo's: [`drawn_on`] in the photo's pixels, with [`SHOT_MARGIN`] of
/// the frame's pixels about it, or of the photo's where they are the
/// larger.
fn shot_about(frame: [f32; 2], k1: f32, out_at: [f32; 2], out: (usize, usize), scale: [f32; 2], size: [usize; 2]) -> [usize; 4] {
    let [x0, y0, x1, y1] = drawn_on(frame, k1, out_at, out);
    let margin = |scale: f32| SHOT_MARGIN as f32 * (1.0 / scale).max(1.0);
    let low = |at: f32, scale: f32| ((at / scale).floor() - margin(scale)).max(0.0) as usize;
    let high = |at: f32, scale: f32, size: usize| (((at / scale).ceil() + margin(scale)) as usize).min(size);
    [low(x0, scale[0]), low(y0, scale[1]), high(x1, scale[0], size[0]), high(y1, scale[1], size[1])]
}

/// Most of a photo's own pixels [`whole_after`] looks among for a number
/// that a scale makes whole.
const MOST_WHOLE_AFTER: usize = 64;

/// The fewest of a photo's own pixels, along one way, that come to a
/// whole number of the pixels of a copy `scale` times its size: one at a
/// whole scale, two at a half, three at a third or two, four at a
/// quarter, eight at an eighth. A part of the photo cut at a whole
/// number of these from where the copy's pixels begin is resized among
/// the pixels the whole copy is ([`super::resize`] centres each pixel of
/// the part's copy where it centres it in the whole's), and its own come
/// to a whole number of the copy's. Four, a quarter's, where no number
/// up to [`MOST_WHOLE_AFTER`] does.
pub(crate) fn whole_after(scale: f64) -> usize {
    (1..=MOST_WHOLE_AFTER)
        .find(|&n| {
            let pixels = n as f64 * scale;
            pixels.round() >= 1.0 && (pixels - pixels.round()).abs() < 1e-5 * pixels
        })
        .unwrap_or(4)
}

/// The box `[x0, y0, x1, y1]` of a photo's own pixels within `held`, and
/// out to whole `steps` of them each way from the photo's corner: a part
/// so cut is resized among the pixels the whole photo is, where its step
/// is what [`whole_after`] gives for the scale it is resized by. A box
/// that lies outside `held` comes to one with no pixels in it, on its
/// edge.
fn aligned([x0, y0, x1, y1]: [usize; 4], [across, down]: [usize; 2], held: [usize; 4]) -> [usize; 4] {
    let (left, right) = (x0 / across * across, x1.div_ceil(across) * across);
    let (top, bottom) = (y0 / down * down, y1.div_ceil(down) * down);
    let horizontal = |v: usize| v.clamp(held[0], held[2]);
    let vertical = |v: usize| v.clamp(held[1], held[3]);
    [horizontal(left), vertical(top), horizontal(right), vertical(bottom)]
}

/// The rectangle `[x0, y0, x1, y1]`, far sides exclusive, of an image
/// `width` pixels across with `channels` values to a pixel.
pub(crate) fn cut<T: Copy>(data: &[T], width: usize, channels: usize, [x0, y0, x1, y1]: [usize; 4]) -> Vec<T> {
    // Whole rows lie one after another, and are copied at once; otherwise
    // a row at a time, each copied whole rather than a value at a time. No
    // rows at all are none, wherever they would begin.
    if x0 == 0 && x1 == width && y0 < y1 {
        return data[y0 * width * channels..y1 * width * channels].to_vec();
    }
    let mut out = Vec::with_capacity(x1.saturating_sub(x0) * y1.saturating_sub(y0) * channels);
    for y in y0..y1 {
        out.extend_from_slice(&data[(y * width + x0) * channels..(y * width + x1) * channels]);
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
        let exposure = self.exposure.exp2();
        std::array::from_fn(|i| {
            let mut value = i as f32 / 255.0;
            if self.invert {
                value = 1.0 - value;
            }
            if self.exposure != 0.0 {
                value = encoded_light(linear_light(value) * exposure);
            }
            value = (value - 0.5) * gain + 0.5;
            (value.clamp(0.0, 1.0) * 255.0).round() as u8
        })
    }

    /// Returns the RGBA image to display and the luminance to sample
    /// from, of a photo `width` by `height`: both at the size the
    /// enlargement makes of it ([`Enlarge::size`]), which is the
    /// luminance's. With a filter on, the display is the filtered
    /// luminance in grey, so that what is shown is what is sampled.
    pub fn apply(&self, rgba: &[u8], width: usize, height: usize) -> (Vec<u8>, Luma) {
        self.apply_to(rgba, width, height, [0, 0, width, height])
    }

    /// [`Self::apply`] to the part `[x, y, w, h]` of a photo `width` by
    /// `height`, whose lens distortion is the whole photo's, taken out
    /// about its centre: the part is what the whole photo so adjusted
    /// shows there, but for the mask and the filter, which read no
    /// further than the part's edges. With no lens term the enlargement
    /// reads no further either: the part is resized on its own, and its
    /// edges come out as a photo's edges do, not as the whole photo's
    /// pixels there.
    ///
    /// The adjustments are made in the order the Photo section sets
    /// them in. The photo is resized first, in its colours, and its lens
    /// distortion taken out at the size it comes to, which bends it
    /// about its centre: straightened among the pixels it is worked on
    /// at, a photo enlarged keeps the edges the enlargement drew between
    /// its pixels. Then it is sharpened or blurred, in its colours: the
    /// mask is as wide in the photo's own pixels as Sharpen says, or in
    /// those a reduction leaves, and on the photo enlarged as many times
    /// as wide, as wide as what a lens blurred however large the photo
    /// is drawn. These three mix the photo's pixels as stored, or as the
    /// light they stand for ([`Self::light`]), stored again after them.
    /// The tone comes after, a curve for each pixel alone, and the filter
    /// last, whose neighbourhood is measured in the pixels the photo is
    /// worked on at.
    ///
    /// The luminance of a photo enlarged keeps that of the photo at its
    /// own size beside it ([`Luma::unenlarged`]), made as the photo
    /// would be at that size less the filter, whose neighbourhood would
    /// be that many times too wide there; and that of a photo adjusted more than
    /// that keeps the photo as loaded beside it, with the adjustments
    /// ([`Luma::as_shot`]), for a copy of it to be made from in the same
    /// order. An enlargement the part's size does not allow is held to
    /// the most that it does ([`Enlarge::within`]).
    pub fn apply_to(&self, rgba: &[u8], width: usize, height: usize, part: [usize; 4]) -> (Vec<u8>, Luma) {
        let (display, luma) = self.applied(rgba, None, width, height, part, Show::Bytes);
        (display.into_bytes(), luma)
    }

    /// [`Self::apply_to`] of a photo whose bytes are held to be shared,
    /// as a window that shows the photo holds them: what the luminance
    /// keeps beside it of the photo as loaded ([`Luma::as_shot`]) is
    /// those same bytes where it is the whole of them, and not a second
    /// copy of a photo of many megapixels.
    pub fn apply_to_shared(&self, rgba: &SharedRgba, width: usize, height: usize, part: [usize; 4]) -> (Vec<u8>, Luma) {
        let (display, luma) = self.applied(rgba, Some(rgba), width, height, part, Show::Bytes);
        (display.into_bytes(), luma)
    }

    /// The luminance alone of [`Self::apply_to_shared`], for a photo that
    /// is read and not shown: what would be shown of it is never taken off
    /// the graphics card, which is as many bytes again as the luminance.
    pub fn luma_of_shared(&self, rgba: &SharedRgba, width: usize, height: usize, part: [usize; 4]) -> Luma {
        self.applied(rgba, Some(rgba), width, height, part, Show::Nothing).1
    }

    /// [`Self::apply_to_shared`], with the RGBA image to display left on
    /// the graphics card where it was made there whole, for a window drawn
    /// by the card to lay into its own texture ([`gpu::OnCard::copy_to`]),
    /// and taken off it otherwise.
    pub fn apply_to_shared_shown(&self, rgba: &SharedRgba, width: usize, height: usize, part: [usize; 4]) -> (Shown, Luma) {
        self.applied(rgba, Some(rgba), width, height, part, Show::Card)
    }

    /// [`Self::apply_to_shared_shown`], with what is shown the grey of the
    /// luminance, as the view in grey shows the photo ([`Luma::to_rgba`]):
    /// greyed on the card where the luminance was made there whole, and the
    /// photo's colours, which would be thrown away, never taken off it.
    pub fn apply_to_shared_grey(&self, rgba: &SharedRgba, width: usize, height: usize, part: [usize; 4]) -> (Shown, Luma) {
        self.applied(rgba, Some(rgba), width, height, part, Show::Grey)
    }

    /// [`Self::apply_to`], with the bytes held to be shared, if they are,
    /// and the RGBA image to display made only where it is shown, as
    /// `show` says ([`Self::made`]).
    fn applied(&self, rgba: &[u8], shared: Option<&SharedRgba>, width: usize, height: usize, [x, y, w, h]: [usize; 4], show: Show) -> (Shown, Luma) {
        assert_eq!(rgba.len(), width * height * 4);
        assert!(x + w <= width && y + h <= height, "the part [{x}, {y}, {w}, {h}] lies within the photo, {width} by {height}");
        if w == 0 || h == 0 {
            let (display, mut luma) = self.filtered(Vec::new(), Vec::new(), w, h, 1.0, (0, 0));
            luma.light = self.light;
            return (Shown::Bytes(display), luma);
        }
        let enlarge = self.enlarge.within(w as u32, h as u32);
        let photo = Loaded { rgba, width, height, at: [0, 0], photo: [width, height], shared };
        let size = enlarge.size(w, h);
        Adjust { enlarge, ..*self }.made_as(photo, [x, y, w, h], size, size, show, true)
    }

    /// Whether the tone is the photo's own: a copy made of the photo's
    /// luminance and toned no further is the copy toned.
    fn plain_tone(&self) -> bool {
        self.channel == Channel::Color && !self.invert && self.exposure == 0.0 && self.contrast == 0.0
    }

    /// The part `[x, y, w, h]` of a photo, made from what of it `from`
    /// holds as loaded, `to_width` by `to_height`, by every adjustment
    /// in its turn ([`Self::apply_to`]) but the size, which `to_width`
    /// and `to_height` give: what the whole photo so adjusted would show
    /// there, at that size. `whole` is that whole's width and height at
    /// that size, which the Deblur filter is reduced for as the whole is
    /// ([`deblur::undone_within`]). The RGBA image to display is
    /// made where it is `shown`, and is otherwise empty, and never taken off
    /// the graphics card: a copy made for a read is read and not shown.
    /// With `keep`, the luminance keeps the part made at its own size
    /// beside it, when it is made a whole number of times larger, and the
    /// photo as loaded, when it is adjusted more than that.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn made(&self, from: Loaded<'_>, part: [usize; 4], size: (usize, usize), whole: (usize, usize), shown: bool, keep: bool) -> (Vec<u8>, Luma) {
        let (display, luma) = self.made_as(from, part, size, whole, Show::of(shown), keep);
        (display.into_bytes(), luma)
    }

    /// [`Self::made`], with what is shown made only where `show` says, and
    /// left on the graphics card where it says so and it was made there
    /// whole.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn made_as(&self, from: Loaded<'_>, [x, y, w, h]: [usize; 4], (to_width, to_height): (usize, usize), whole: (usize, usize), show: Show, keep: bool) -> (Shown, Luma) {
        let [ax, ay] = from.at;
        let held = [ax, ay, ax + from.width, ay + from.height];
        // Unbent, the part is read where it lies. Bent, it is read from
        // what it draws on, which may lie elsewhere, a lens term that
        // straightens barrel drawing the photo's edges in from nearer its
        // centre, and which is held to what `from` holds ([`aligned`]).
        assert!(
            self.k1 != 0.0 || (x >= ax && y >= ay && x + w <= held[2] && y + h <= held[3]),
            "the part [{x}, {y}, {w}, {h}] is held, in {held:?}"
        );
        let pixels = |[x0, y0, x1, y1]: [usize; 4]| {
            if [x0, y0, x1, y1] == held { Cow::Borrowed(from.rgba) } else { Cow::Owned(cut(from.rgba, from.width, 4, [x0 - ax, y0 - ay, x1 - ax, y1 - ay])) }
        };
        let scale = [to_width as f32 / w as f32, to_height as f32 / h as f32];
        let mask = mask_for(self.sharpen);
        // The chain of steps the part is made by on the card, and what of
        // the photo it is made from, which is what of it the part draws on.
        let mut chain = Vec::new();
        let drawn = if self.k1 == 0.0 {
            if (to_width, to_height) != (w, h) {
                chain.push(Step::Resized { across: Taps::new(w, to_width), down: Taps::new(h, to_height) });
            }
            [x, y, x + w, y + h]
        } else {
            // In a frame of the pixels the photo is worked on at, the
            // photo as loaded about the part, resized into it, and
            // straightened.
            let [pw, ph] = from.photo;
            let frame = [pw as f32 * scale[0], ph as f32 * scale[1]];
            let out_at = [x as f32 * scale[0], y as f32 * scale[1]];
            let steps = scale.map(|scale| whole_after(f64::from(scale)));
            let region = aligned(shot_about(frame, self.k1, out_at, (to_width, to_height), scale, from.photo), steps, held);
            let (gw, gh) = (region[2] - region[0], region[3] - region[1]);
            // A part that draws on none of the photo has nothing to resize,
            // and is the backdrop throughout.
            let (rw, rh) = if gw == 0 || gh == 0 {
                (0, 0)
            } else {
                (((gw as f32 * scale[0]).round() as usize).max(1), ((gh as f32 * scale[1]).round() as usize).max(1))
            };
            if (rw, rh) != (gw, gh) && rw > 0 {
                chain.push(Step::Resized { across: Taps::new(gw, rw), down: Taps::new(gh, rh) });
            }
            let at = [region[0] as f32 * scale[0], region[1] as f32 * scale[1]];
            let step = [gw as f32 * scale[0] / rw.max(1) as f32, gh as f32 * scale[1] / rh.max(1) as f32];
            let lens = Lens { frame, k1: self.k1, slope: edge_slope(f64::from(self.k1)) as f32 };
            chain.push(Step::Straightened { place: [at, step], lens, out_at, size: (to_width, to_height), fill: self.backdrop.level() });
            region
        };
        if let Some((sigma, amount)) = mask {
            chain.push(Step::Sharpened { weights: gaussian(sigma * f64::from(scale[0]).max(1.0)), amount });
        }
        // Made on the card from the photo to what is shown and read, the
        // filter and all, where the card holds it whole; and otherwise a
        // step at a time, and filtered after.
        let source = gpu::Source { rgba: from.rgba, width: from.width, part: [drawn[0] - ax, drawn[1] - ay, drawn[2] - ax, drawn[3] - ay], shared: from.shared };
        let table = self.lookup_table();
        let on_card = gpu::adjusted(&source, &chain, &table, self.pick(), self.tone(show), self.light).map(|toned| self.filtered_on(toned, scale[0], whole, show));
        let (display, mut luma) = match on_card {
            Some((display, toned, deblurred)) => {
                let mut luma = Luma::new(to_width, to_height, toned);
                luma.deblurred = deblurred;
                (display, luma)
            }
            None => {
                let (display, toned) = gpu::stepped(&source, &chain, &table, self.pick(), matches!(show, Show::Bytes | Show::Card), self.light);
                let (display, luma) = self.filtered(display, toned, to_width, to_height, scale[0], whole);
                let display = if show == Show::Grey { luma.to_rgba() } else { display };
                (Shown::Bytes(display), luma)
            }
        };
        luma.light = self.light;
        if !keep {
            return (display, luma);
        }
        let times = to_width / w;
        let whole = to_width == w * times && to_height == h * times && times >= 1;
        if whole && times > 1 {
            // Less the filter, whose neighbourhood would be that many
            // times too wide there: a symbol rendered pixel for pixel is
            // read off these pixels ([`super::pure`]), and a median a
            // pixel either way of each of them would take out every
            // module that stands alone, and a finder pattern's ring.
            let unfiltered = Adjust { filter: Filter::None, ..*self };
            luma.unenlarged = Some(Arc::new(unfiltered.made(from, [x, y, w, h], (w, h), (w, h), false, false).1));
        }
        // Made in light, a copy of the part is made again from the photo as
        // loaded whatever else the part was made by: the part mixed the
        // light of each colour, and a copy made of its luminance would mix
        // the light of its grey, which of a coloured photo is not the grey
        // of that.
        if !whole || self.k1 != 0.0 || mask.is_some() || self.filter != Filter::None || !self.plain_tone() || self.light == Light::Linear {
            // As much of the photo as a copy of the part at any size
            // draws on straightened: what the part draws on at its own
            // size, and as far again as what a copy draws on beyond a
            // pixel of its own, at the smallest a search makes of the
            // part ([`most_shrunk`]), and as its cut out to whole steps
            // of its pixels reaches past that; out to whole fours of the
            // photo's pixels. Kept short of that, a copy shrunk further
            // was cut at the edge of what was kept and not at a whole
            // step, and resized among other pixels than the whole's.
            let region = if self.k1 == 0.0 {
                drawn
            } else {
                let [rx0, ry0, rx1, ry1] = shot_about([from.photo[0] as f32, from.photo[1] as f32], self.k1, [x as f32, y as f32], (w, h), [1.0; 2], from.photo);
                // A copy's pixels are each `1 / least` of the photo's,
                // and what it draws on reaches three of them past what it
                // maps ([`drawn_on`]) and [`SHOT_MARGIN`] past that.
                let least = f64::from(scale[0]) / f64::from(most_shrunk(to_width, to_height));
                let beyond = (3 + SHOT_MARGIN) as f64 * ((1.0 / least).max(1.0) - 1.0);
                let far = 3 * SHOT_MARGIN + beyond.ceil() as usize + whole_after(least);
                aligned([rx0.saturating_sub(far), ry0.saturating_sub(far), rx1 + far, ry1 + far], [whole_after(0.25); 2], held)
            };
            // The bytes as they were handed over, where they are the
            // whole of what is kept and are held to be shared.
            let rgba: SharedRgba = match from.shared {
                Some(shared) if region == held => shared.clone(),
                _ => Arc::new(pixels(region).into_owned()),
            };
            luma.as_shot = Some(Arc::new(AsShot {
                rgba,
                width: region[2] - region[0],
                height: region[3] - region[1],
                at: [region[0], region[1]],
                photo: from.photo,
                part: [x, y, w, h],
                adjust: *self,
                turned: 0,
            }));
        }
        (display, luma)
    }

    /// Which of the photo's colours the tone shows in all three, as the
    /// card is told ([`gpu::adjusted`]): none for the photo in colour.
    fn pick(&self) -> Option<usize> {
        match self.channel {
            Channel::Color => None,
            Channel::Red => Some(0),
            Channel::Green => Some(1),
            Channel::Blue => Some(2),
        }
    }

    /// What the part is toned to show on the card ([`gpu::Tone`]), where
    /// `show` asks it: nothing where a filter is laid over it, whose grey
    /// is shown in its place; its colours, laid out to be copied to a
    /// texture where they are left on the card; or the grey of what is read.
    fn tone(&self, show: Show) -> gpu::Tone {
        match show {
            _ if self.filter != Filter::None => gpu::Tone::Nothing,
            Show::Nothing => gpu::Tone::Nothing,
            Show::Bytes => gpu::Tone::Colours { padded: false },
            Show::Card => gpu::Tone::Colours { padded: true },
            Show::Grey => gpu::Tone::Grey,
        }
    }

    /// [`Self::filtered`] on the card, over a part of the photo adjusted
    /// and toned there ([`gpu::Toned`]), by the same kernels, to the same
    /// values: what is shown, where `show` asks for it, left on the card
    /// where it asks that, and what is read taken off the card after the
    /// filter, with the blur the Deblur filter undid.
    fn filtered_on(&self, toned: gpu::Toned, scale: f32, whole: (usize, usize), show: Show) -> Filtered {
        let radius = usize::from(self.radius.clamp(1, Filter::MAX_RADIUS));
        let (filtered, deblurred) = match self.filter {
            Filter::None => (None, None),
            Filter::Median => (Some(toned.luma.median(radius.min(2))), None),
            Filter::Close => (Some(toned.luma.closed(radius)), None),
            Filter::Deblur => {
                let kernel = self.kernel.map(|kernel| kernel.scaled(f64::from(scale)));
                let (sharp, spread) = deblur::undone_on(&toned.luma, kernel, f64::from(self.strength), whole, self.light);
                (Some(sharp), spread)
            }
        };
        let (display, luma) = match show {
            Show::Card | Show::Grey => {
                let (on_card, luma) = toned.kept(filtered);
                (Shown::Card(on_card), luma)
            }
            _ => {
                let (display, luma) = toned.taken(filtered, show == Show::Bytes);
                (Shown::Bytes(display), luma)
            }
        };
        (display, luma, deblurred)
    }

    /// The filter, over pixels already straightened, resized, sharpened
    /// and toned, `rgba` as they are shown and `luma` as they are read:
    /// the RGBA image to display, made of those pixels where they stand,
    /// and the luminance to sample from. A photo enlarged is many times
    /// the photo, and is not held a second time to be filtered. The
    /// pixels are a part of a whole `whole` pixels across and down at
    /// this size, which the Deblur filter is reduced for ([`Self::made`]).
    /// Pixels not shown come empty, and stay so.
    fn filtered(&self, mut rgba: Vec<u8>, mut luma: Vec<f32>, width: usize, height: usize, scale: f32, whole: (usize, usize)) -> (Vec<u8>, Luma) {
        let mut deblurred = None;
        if self.filter != Filter::None && width > 0 && height > 0 {
            luma = if self.filter == Filter::Deblur {
                // By the kernel set, carried to the pixels the photo is
                // worked on at, or by the blur it shows there.
                let kernel = self.kernel.map(|kernel| kernel.scaled(f64::from(scale)));
                let (sharp, spread) = deblur::undone_within(&luma, width, height, kernel, f64::from(self.strength), whole, self.light);
                deblurred = spread;
                sharp
            } else {
                self.filter.apply(&luma, width, height, self.radius.clamp(1, Filter::MAX_RADIUS) as usize, self.light)
            };
            if !rgba.is_empty() {
                super::greyed(&luma, &mut rgba);
            }
        }
        let mut luma = Luma::new(width, height, luma);
        luma.deblurred = deblurred;
        (rgba, luma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imaging::resize::resize;

    /// `data`, `width` by `height`, blurred by a Gaussian `sigma` pixels
    /// wide in the light its values stand for, as a lens blurs what it
    /// photographs, and stored again as opaque grey pixels: what the
    /// Deblur filter is for.
    fn photographed(data: &[f32], width: usize, height: usize, sigma: f64) -> Vec<u8> {
        use crate::imaging::sample::{encoded_light, linear_light};
        let light: Vec<f32> = data.iter().map(|&v| linear_light(v)).collect();
        crate::imaging::enhance::unsharp(&light, width, height, 1, sigma, -1.0)
            .iter()
            .flat_map(|&v| [(encoded_light(v) * 255.0).round() as u8; 3].into_iter().chain([255]))
            .collect()
    }

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
    fn a_photo_enlarged_keeps_its_own_size_made_as_it_would_be_there_but_unfiltered() {
        // A speck of ink on paper, enlarged twice.
        let own = pixels(6, 6, |x, y| if (x, y) == (2, 2) { [0; 3] } else { [255; 3] });
        let adjust = Adjust { invert: true, filter: Filter::Median, radius: 1, enlarge: Enlarge::times(2), ..Default::default() };
        let (display, luma) = adjust.apply(&own, 6, 6);
        assert_eq!((display.len(), luma.width, luma.height), (12 * 4 * 12, 12, 12));
        // The working pixels are inverted and the speck taken out of
        // them, where without the filter it shows...
        let brightest = |data: &[f32]| data.iter().copied().fold(0.0f32, f32::max);
        let unfiltered = Adjust { filter: Filter::None, ..adjust }.apply(&own, 6, 6).1;
        assert!(luma.data[0].abs() < 1e-6 && brightest(&unfiltered.data) > 0.8, "{:?}", unfiltered.data);
        assert!(brightest(&luma.data) < 0.3, "{:?}", luma.data);
        // ...and the photo's own pixels are the photo at its own size
        // under the same adjustments less the filter, whose median of
        // three pixels a side would take out a module a pixel across:
        // the speck inverted where it was, filtered or not.
        let small = luma.unenlarged.as_ref().expect("the photo's own pixels are kept");
        assert_eq!((small.width, small.height), (6, 6));
        assert_eq!(small.data, Adjust { enlarge: Enlarge::NONE, filter: Filter::None, ..adjust }.apply(&own, 6, 6).1.data);
        for (i, value) in small.data.iter().enumerate() {
            let want = if i == 2 * 6 + 2 { 1.0 } else { 0.0 };
            assert!((value - want).abs() < 1e-6, "{:?}", small.data);
        }
        assert_eq!(unfiltered.unenlarged.as_ref().map(|plain| &plain.data), Some(&small.data));
        assert!(small.unenlarged.is_none());
        // A photo worked on at its own size keeps nothing beside it.
        assert!(Adjust { enlarge: Enlarge::NONE, ..adjust }.apply(&own, 6, 6).1.unenlarged.is_none());
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
            assert_eq!(luma.data, Adjust::default().apply(&display, 40, 12).1.data);
        }
        assert!(steps.windows(2).all(|pair| pair[0] < pair[1]), "the more the sharper: {steps:?}");
        // Enlarged and then sharpened, with the mask as wide in the
        // photo's own pixels, and the photo at its own size, sharpened,
        // beside it.
        let sharp = Adjust { sharpen: 0.5, ..Default::default() };
        let both = Adjust { enlarge: Enlarge::times(3), ..sharp };
        let (display, luma) = both.apply(&photo, 40, 12);
        assert_eq!((display.len(), luma.width, luma.height), (120 * 36 * 4, 120, 36));
        assert_eq!(luma.unenlarged.as_ref().map(|own| &own.data), Some(&sharp.apply(&photo, 40, 12).1.data));
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
            assert_eq!(luma.data, Adjust::default().apply(&display, 40, 12).1.data);
        }
        assert!(spreads.windows(2).all(|pair| pair[0] > pair[1]), "the further down the softer: {spreads:?}");
        assert!(specks.windows(2).all(|pair| pair[0] > pair[1]), "and the fainter the specks: {specks:?}");
        // Past where the mask stops, at the end, it is twice as wide
        // again: softer and fainter still, and still with nothing past
        // the tones about it.
        let (display, _) = Adjust { sharpen: -2.0, ..Default::default() }.apply(&photo, 40, 12);
        let green = |x: usize| i32::from(pixel(&display, x)[1]);
        assert!(green(21) - green(18) < spreads[2] && green(8) < specks[2].into(), "{} and {}", green(21) - green(18), green(8));
        assert!(display.as_chunks::<4>().0.iter().all(|&[_, g, b, _]| (40..=220).contains(&g) && (60..=220).contains(&b)));
        // Half way down is the whole of a Gaussian blur a pixel wide:
        // a speck a pixel across on flat ground keeps the middle weight
        // of the blur, across and down, over the ground, and no more.
        let (display, _) = Adjust { sharpen: -0.5, ..Default::default() }.apply(&photo, 40, 12);
        let middle = 1.0 / (1.0 + 2.0 * ((-0.5f32).exp() + (-2.0f32).exp() + (-4.5f32).exp()));
        let want = 40.0 + 180.0 * middle * middle;
        assert!((f32::from(pixel(&display, 8)[1]) - want).abs() < 1.0, "{} against {want}", pixel(&display, 8)[1]);
        // Enlarged and then blurred, as wide in the photo's own pixels,
        // with the photo at its own size, blurred, beside it; and
        // reduced and then blurred, at the size the reduction left.
        let blur = Adjust { sharpen: -0.5, ..Default::default() };
        let (display, luma) = Adjust { enlarge: Enlarge::times(3), ..blur }.apply(&photo, 40, 12);
        assert_eq!((display.len(), luma.width, luma.height), (120 * 36 * 4, 120, 36));
        assert_eq!(luma.unenlarged.as_ref().map(|own| &own.data), Some(&blur.apply(&photo, 40, 12).1.data));
        assert_eq!(&display[(15 * 120 + 9) * 4..(15 * 120 + 9) * 4 + 4], &[220, 40, 60, 255]);
        let large = edge(80, 64);
        let reduced = Adjust { enlarge: Enlarge::HALF, ..Default::default() }.apply(&large, 80, 64).0;
        let (display, luma) = Adjust { enlarge: Enlarge::HALF, ..blur }.apply(&large, 80, 64);
        assert_eq!((luma.width, luma.height), (40, 32));
        assert_eq!(display, blur.apply(&reduced, 40, 32).0);
    }

    #[test]
    fn a_photo_is_resized_and_blurred_in_light_where_it_is_asked_to_be() {
        let lit = Adjust { light: Light::Linear, ..Default::default() };
        // Stripes a pixel wide of black ink and light grey paper, reduced
        // to half: as stored, each pixel clear of the photo's edges is the
        // stored values' midway, 100, and in light it is the light midway
        // between theirs, stored again, which is lighter.
        let stripes = pixels(16, 8, |x, _| if x % 2 == 0 { [0; 3] } else { [200; 3] });
        let half = |adjust: Adjust| Adjust { enlarge: Enlarge::HALF, ..adjust }.apply(&stripes, 16, 8);
        let within = |display: &[u8]| (0..4).flat_map(|y| (2..6).map(move |x| display[(y * 8 + x) * 4])).collect::<Vec<u8>>();
        let (stored, light) = (half(Adjust::default()), half(lit));
        assert!(within(&stored.0).iter().all(|&v| v == 100), "{:?}", within(&stored.0));
        assert!(within(&light.0).iter().all(|&v| v == 146), "{:?}", within(&light.0));
        // What is read is the luminance of what is shown, and says what
        // light it was made in.
        assert_eq!(light.1.data, Adjust::default().apply(&light.0, 8, 4).1.data);
        assert_eq!((stored.1.light, light.1.light), (Light::Encoded, Light::Linear));
        // So is an image made in light resized, as a search reduces one.
        let grey: Vec<f32> = (0..16 * 8).map(|i| if i % 2 == 0 { 0.0 } else { 200.0 / 255.0 }).collect();
        let reduced = Luma::new_in(16, 8, grey.clone(), Light::Linear).resized(8, 4);
        let midway = crate::imaging::sample::encoded_light(crate::imaging::sample::linear_light(200.0 / 255.0) / 2.0);
        assert!(reduced.light == Light::Linear && (0..4).all(|y| (2..6).all(|x| (reduced.data[y * 8 + x] - midway).abs() < 1e-4)), "{:?}", reduced.data);
        assert!((Luma::new(16, 8, grey).resized(8, 4).data[2] - 100.0 / 255.0).abs() < 1e-4);
        // A line of ink a pixel wide blurred by a pixel: in light the
        // paper's light spreads over the line as a lens spreads it, and
        // the line reads lighter than its stored values blurred.
        let line = pixels(40, 12, |x, _| if x == 20 { [40; 3] } else { [220; 3] });
        let blurred = |adjust: Adjust| Adjust { sharpen: -0.5, ..adjust }.apply(&line, 40, 12).0[(5 * 40 + 20) * 4];
        assert_eq!((blurred(Adjust::default()), blurred(lit)), (148, 177));
        // Enlarged in light, the photo keeps its own pixels beside it made
        // in light too, and the photo as loaded for its copies to be made
        // again from.
        let (_, twice) = Adjust { enlarge: Enlarge::times(2), ..lit }.apply(&line, 40, 12);
        assert_eq!(twice.unenlarged.as_ref().map(|own| own.light), Some(Light::Linear));
        assert!(twice.as_shot.is_some() && Adjust { enlarge: Enlarge::times(2), ..Default::default() }.apply(&line, 40, 12).1.as_shot.is_none());
        // With nothing that mixes its pixels, the photo is the same in
        // either light: the tone is a curve of what is stored, and the
        // median picks one of the values about each pixel.
        let photo = pixels(30, 20, |x, y| [((x * 37 + y * 11) % 256) as u8, ((x * 13 + y * 29) % 256) as u8, (((x / 3 + y / 2) % 2) * 255) as u8]);
        let toned = Adjust { invert: true, contrast: 0.4, filter: Filter::Median, ..Default::default() };
        let (stored, light) = (toned.apply(&photo, 30, 20), Adjust { light: Light::Linear, ..toned }.apply(&photo, 30, 20));
        assert!(stored.0 == light.0 && stored.1.data == light.1.data);
    }

    #[test]
    fn a_photo_sharpened_and_enlarged_reads_as_a_copy_of_it_made_for_a_read() {
        use crate::imaging::enhance::{Enhance, Enhanced, Sharpen};
        // Busy grey pixels. What a read sees through a copy of the photo
        // about a symbol, enlarged and sharpened, the whole photo under
        // the same two adjustments shows, edges and all: to the half of
        // a grey level a byte rounds away where only one of the two is
        // made, and where both are, to what the mask makes of the half
        // a level the bytes the photo is enlarged into round away before
        // it is sharpened, which the copy holds as values.
        let (w, h) = (48, 36);
        let photo = pixels(w, h, |x, y| [((((x * 31 + y * 17) % 23) * 255 / 22 + ((x / 5 + y / 3) % 2) * 255) / 2) as u8; 3]);
        let plain = Adjust::default().apply(&photo, w, h).1;
        let differences = |adjusted: &Luma, copy: &Luma, what: &str| {
            assert_eq!((adjusted.width, adjusted.height), (copy.width, copy.height), "{what}");
            let differences: Vec<f32> = adjusted.data.iter().zip(&copy.data).map(|(a, b)| (a - b).abs()).collect();
            (differences.iter().copied().fold(0.0f32, f32::max), differences.iter().sum::<f32>() / differences.len() as f32)
        };
        let near = |adjusted: &Luma, copy: &Luma, both: bool, what: &str| {
            let (worst, mean) = differences(adjusted, copy, what);
            if both {
                assert!(worst < 0.01 && mean < 0.002, "{what}: {worst} at the most, {mean} on the whole");
            } else {
                assert!(worst < 0.0025, "{what}: {worst}");
            }
        };
        for (sharpen, enlarge) in [(Sharpen::Light, 1u8), (Sharpen::Strong, 1), (Sharpen::Light, 3), (Sharpen::Strong, 4), (Sharpen::Off, 2)] {
            let copy = Enhanced::window(&plain, [0, 0, w, h], Enhance { enlarge, sharpen }).unwrap();
            let adjust = Adjust { sharpen: sharpen.amount(), enlarge: Enlarge::NONE.by(enlarge), ..Default::default() };
            let both = enlarge > 1 && sharpen != Sharpen::Off;
            near(&adjust.apply(&photo, w, h).1, &copy.luma, both, &format!("{sharpen:?}, {enlarge} times"));
        }
        // A photo enlarged already is read through a copy made from the
        // pixels it had, in one step: so is the photo enlarged as many
        // times again.
        let twice = Adjust { enlarge: Enlarge::times(2), ..Default::default() };
        let copy = Enhanced::window(&twice.apply(&photo, w, h).1, [0, 0, 2 * w, 2 * h], Enhance { enlarge: 2, sharpen: Sharpen::Light }).unwrap();
        let adjust = Adjust { sharpen: Sharpen::Light.amount(), enlarge: twice.enlarge.by(2), ..Default::default() };
        assert_eq!(adjust.enlarge, Enlarge::times(4));
        near(&adjust.apply(&photo, w, h).1, &copy.luma, true, "enlarged twice, and twice again");
        // Bent by a lens, the photo is straightened after it is enlarged,
        // and a copy of it is the photo made again from the photo as
        // loaded under the copy's Enlarge and Sharpen, by the same steps:
        // the photo so adjusted, pixel for pixel. So for a window of it,
        // and a part of it cut alone.
        let lensed = |what: &str, adjusted: &Luma, copy: &Luma| {
            let (worst, _) = differences(adjusted, copy, what);
            assert!(worst == 0.0, "{what}: {worst}");
        };
        for k1 in [0.12, -0.2] {
            let plain = Adjust { k1, ..Default::default() }.apply(&photo, w, h).1;
            for (sharpen, enlarge) in [(Sharpen::Light, 2u8), (Sharpen::Strong, 4), (Sharpen::Off, 2)] {
                let copy = Enhanced::window(&plain, [0, 0, w, h], Enhance { enlarge, sharpen }).unwrap();
                let adjust = Adjust { k1, sharpen: sharpen.amount(), enlarge: Enlarge::NONE.by(enlarge), ..Default::default() };
                lensed(&format!("{k1}: {sharpen:?}, {enlarge} times"), &adjust.apply(&photo, w, h).1, &copy.luma);
            }
            let twice = Adjust { k1, ..twice };
            let copy = Enhanced::window(&twice.apply(&photo, w, h).1, [0, 0, 2 * w, 2 * h], Enhance { enlarge: 2, sharpen: Sharpen::Light }).unwrap();
            let adjust = Adjust { sharpen: Sharpen::Light.amount(), enlarge: twice.enlarge.by(2), ..twice };
            lensed(&format!("{k1}: enlarged twice, and twice again"), &adjust.apply(&photo, w, h).1, &copy.luma);
            let copy = Enhanced::window(&plain, [10, 8, 40, 30], Enhance { enlarge: 3, sharpen: Sharpen::Strong }).unwrap();
            let adjusted = Adjust { k1, sharpen: 1.0, enlarge: Enlarge::times(3), ..Default::default() }.apply(&photo, w, h).1;
            lensed(&format!("{k1}: a window"), &Luma::new(90, 66, cut(&adjusted.data, adjusted.width, 1, [30, 24, 120, 90])), &copy.luma);
            let part = [6, 4, 36, 28];
            let copy = Enhanced::window(&Adjust { k1, ..Default::default() }.apply_to(&photo, w, h, part).1, [0, 0, 36, 28], Enhance { enlarge: 2, sharpen: Sharpen::Light }).unwrap();
            let adjusted = Adjust { k1, sharpen: 0.5, enlarge: Enlarge::times(2), ..Default::default() }.apply_to(&photo, w, h, part).1;
            lensed(&format!("{k1}: a part"), &adjusted, &copy.luma);
            // The photo turned gives the copy turned, pixel for pixel: the
            // copy is made of the photo as it lay, and turned after. Made of
            // the photo as loaded turned, it was straightened among other
            // pixels, and a half grey level rounded the other way here and
            // there, which the mask made a few.
            let enhance = Enhance { enlarge: 2, sharpen: Sharpen::Light };
            for quarters in 1..4u8 {
                let (tw, th) = if quarters % 2 == 1 { (h, w) } else { (w, h) };
                let turned = Enhanced::window(&plain.rotated_cw(quarters), [0, 0, tw, th], enhance).unwrap();
                let copy = Enhanced::window(&plain, [0, 0, w, h], enhance).unwrap();
                lensed(&format!("{k1}: turned {quarters} times"), &copy.luma.rotated_cw(quarters), &turned.luma);
                assert_eq!((turned.origin, turned.factor), ([0.0; 2], copy.factor), "{k1}: turned {quarters} times");
            }
        }
        // A photo sharpened or blurred already: a copy that lays a mask
        // of its own is made of the photo without the photo's, and one
        // that lays none of the photo with it, so that the photo with
        // Sharpen set to the copy's, or left as it was, shows the copy
        // again; bent by a lens or not, and enlarged or not.
        for k1 in [0.0, 0.12] {
            for own in [0.3, 1.5, -0.6] {
                let plain = Adjust { k1, sharpen: own, ..Default::default() }.apply(&photo, w, h).1;
                for (sharpen, enlarge) in [(Sharpen::Light, 1u8), (Sharpen::Strong, 3), (Sharpen::Off, 2)] {
                    let copy = Enhanced::window(&plain, [0, 0, w, h], Enhance { enlarge, sharpen }).unwrap();
                    let set = if sharpen == Sharpen::Off { own } else { sharpen.amount() };
                    let adjusted = Adjust { k1, sharpen: set, enlarge: Enlarge::NONE.by(enlarge), ..Default::default() }.apply(&photo, w, h).1;
                    lensed(&format!("{k1}: sharpened {own} already, then {sharpen:?}, {enlarge} times"), &adjusted, &copy.luma);
                }
            }
        }
        // A photo filtered: a copy runs the photo's filter over it last,
        // once it is enlarged, straightened and sharpened, with the same
        // half-width in its own pixels, as the photo so adjusted does.
        for k1 in [0.0, 0.12] {
            for (filter, radius) in [(Filter::Median, 1), (Filter::Close, 2)] {
                let plain = Adjust { k1, filter, radius, ..Default::default() }.apply(&photo, w, h).1;
                for (sharpen, enlarge) in [(Sharpen::Off, 2u8), (Sharpen::Light, 3), (Sharpen::Light, 1)] {
                    let copy = Enhanced::window(&plain, [0, 0, w, h], Enhance { enlarge, sharpen }).unwrap();
                    let adjusted = Adjust { k1, filter, radius, sharpen: sharpen.amount(), enlarge: Enlarge::NONE.by(enlarge), ..Default::default() }.apply(&photo, w, h).1;
                    lensed(&format!("{k1}: {filter:?} {radius}, then {sharpen:?}, {enlarge} times"), &adjusted, &copy.luma);
                }
            }
        }
    }

    #[test]
    fn a_copy_of_a_photo_adjusted_is_the_photo_made_again() {
        use crate::imaging::enhance::{Enhance, Enhanced, Sharpen};
        // Busy colours. Whatever the photo's adjustments, its tone among
        // them, and whatever size it is worked on at, a reduction among
        // them, a copy a read makes of a window of it is that window of
        // the photo made again under the Enlarge and Sharpen a read that
        // decoded through the copy sets, pixel for pixel: the tone after
        // the mask, as the photo has it, and a reduced photo enlarged
        // once from its own pixels, with its mask as wide as it is there.
        let (w, h) = (48, 36);
        let photo = pixels(w, h, |x, y| [((x * 37 + y * 11) % 256) as u8, ((x * 13 + y * 29) % 256) as u8, (((x / 3 + y / 2) % 2) * 255) as u8]);
        let same = |what: &str, copy: &Enhanced, whole: &Luma| {
            let [x, y] = std::array::from_fn(|i| (copy.origin[i] * copy.factor[i]).round() as usize);
            let there = cut(&whole.data, whole.width, 1, [x, y, x + copy.luma.width, y + copy.luma.height]);
            assert!(copy.luma.data == there, "{what}: the copy is not the photo so adjusted at ({x}, {y})");
        };
        // In light as well: a photo made in it is copied from the photo as
        // loaded however little else it is adjusted, its own size and tone
        // among them.
        let tones = [
            Adjust::default(),
            Adjust { contrast: 0.6, exposure: -0.5, ..Default::default() },
            Adjust { invert: true, channel: Channel::Green, contrast: 0.3, ..Default::default() },
            Adjust { light: Light::Linear, ..Default::default() },
            Adjust { light: Light::Linear, contrast: 0.6, exposure: -0.5, ..Default::default() },
        ];
        for tone in tones {
            for (k1, sharpen, filter) in [(0.0, 0.3, Filter::None), (0.12, 0.0, Filter::Median), (-0.2, -0.6, Filter::Close), (0.0, 0.0, Filter::None)] {
                for base in [Enlarge::NONE, Enlarge::times(2), Enlarge::HALF] {
                    let adjust = Adjust { k1, sharpen, filter, radius: 1, enlarge: base, ..tone };
                    let plain = adjust.apply(&photo, w, h).1;
                    assert_eq!(plain.light, adjust.light);
                    // A photo adjusted no more than enlarged is copied from
                    // its luminance, as the test above has it.
                    if plain.as_shot.is_none() {
                        assert_eq!(adjust.light, Light::Encoded, "{adjust:?}: made in light, and kept nothing to be made again from");
                        continue;
                    }
                    for (read, enlarge) in [(Sharpen::Light, 2u8), (Sharpen::Strong, 1), (Sharpen::Off, 3)] {
                        let set = Adjust { enlarge: base.by(enlarge), sharpen: if read == Sharpen::Off { sharpen } else { read.amount() }, ..adjust };
                        let whole = set.apply(&photo, w, h).1;
                        for window in [[0, 0, plain.width, plain.height], [5, 3, plain.width / 2 + 3, plain.height - 4]] {
                            let copy = Enhanced::window(&plain, window, Enhance { enlarge, sharpen: read }).unwrap();
                            same(&format!("{adjust:?}, read {read:?} {enlarge} times through {window:?}"), &copy, &whole);
                            assert_eq!(copy.luma.light, adjust.light, "{adjust:?}: the copy is in the light the photo was made in");
                        }
                    }
                }
            }
        }
        // So for a part of the photo cut alone; and so for the views the
        // search looks at the photo through, shrunk, or closed in place of
        // its own filter.
        let part = [8, 4, 32, 28];
        for (k1, sharpen, contrast, light) in [(0.0, 0.3, 0.4, Light::Encoded), (0.12, -0.6, 0.0, Light::Encoded), (0.0, 0.3, 0.4, Light::Linear), (0.12, -0.6, 0.0, Light::Linear)] {
            for (base, half) in [(Enlarge::NONE, Enlarge::HALF), (Enlarge::times(2), Enlarge::NONE), (Enlarge::HALF, Enlarge::QUARTER)] {
                let adjust = Adjust { k1, sharpen, contrast, filter: Filter::Median, radius: 1, enlarge: base, light, ..Default::default() };
                let what = format!("{adjust:?}");
                let plain = adjust.apply_to(&photo, w, h, part).1;
                let copy = Enhanced::window(&plain, [2, 2, plain.width - 1, plain.height - 3], Enhance { enlarge: 2, sharpen: Sharpen::Light }).unwrap();
                same(&format!("{what}, a part"), &copy, &Adjust { enlarge: base.by(2), sharpen: 0.5, ..adjust }.apply_to(&photo, w, h, part).1);
                let plain = adjust.apply(&photo, w, h).1;
                let shrunk = Enhanced::shrunk(&plain, [0, 0, plain.width, plain.height], 2).unwrap();
                same(&format!("{what}, shrunk"), &shrunk, &Adjust { enlarge: half, ..adjust }.apply(&photo, w, h).1);
                let closed = Enhanced::closed(&plain, [3, 3, plain.width - 2, plain.height - 2], 2).unwrap();
                same(&format!("{what}, closed"), &closed, &Adjust { filter: Filter::Close, radius: 2, ..adjust }.apply(&photo, w, h).1);
            }
        }
        // Shrunk, a window well inside a larger photo closed over gaps of
        // three pixels, with no mask to widen the margin taken about it: a
        // closing draws on twice its neighbourhood, and a margin of it
        // once left the copy's edges a grey level off.
        let (bw, bh) = (300, 240);
        let busy = pixels(bw, bh, |x, y| [((x * 37 + y * 11) % 256) as u8, ((x * 13 + y * 29) % 256) as u8, (((x / 3 + y / 2) % 2) * 255) as u8]);
        let adjust = Adjust { filter: Filter::Close, radius: 3, ..Default::default() };
        let plain = adjust.apply(&busy, bw, bh).1;
        for (factor, smaller) in [(2, Enlarge::HALF), (4, Enlarge::QUARTER)] {
            let shrunk = Enhanced::shrunk(&plain, [76, 60, 224, 180], factor).unwrap();
            same(&format!("closed over 3 px, shrunk {factor} times"), &shrunk, &Adjust { enlarge: smaller, ..adjust }.apply(&busy, bw, bh).1);
        }
        // A photo deblurred by a kernel set for it, so far: a copy of the
        // whole of it is the photo made again, the kernel carried to the
        // copy's pixels and undone as far as it was.
        for strength in [0.7, 1.4] {
            let kernel = Some(deblur::Spread { xx: 0.8, xy: 0.2, yy: 0.5 });
            let adjust = Adjust { filter: Filter::Deblur, kernel, strength, contrast: 0.2, ..Default::default() };
            let plain = adjust.apply(&photo, w, h).1;
            assert_eq!(plain.deblurred, kernel);
            let copy = Enhanced::window(&plain, [0, 0, w, h], Enhance { enlarge: 2, sharpen: Sharpen::Light }).unwrap();
            let whole = Adjust { enlarge: Enlarge::times(2), sharpen: 0.5, ..adjust }.apply(&photo, w, h).1;
            assert_eq!(whole.deblurred, kernel.map(|kernel| kernel.scaled(2.0)));
            same(&format!("deblurred at {strength}"), &copy, &whole);
        }
        // With none set, the blur the photo's filter measured over the
        // whole of it, and not one measured again over the copy, which
        // enlarged or shrunk measures otherwise: a copy is the photo made
        // again with that blur set, carried to the copy's pixels, as are
        // the views the search makes. Squares of ink six pixels a side
        // blurred by a pixel and a half, for a blur to measure: the busy
        // colours, drawn sharp to the pixel and their edges two or three
        // pixels apart, once measured nearly two pixels long and none
        // wide, fitted over a fan of directions too narrow to say, and
        // now measure nothing.
        let squares = pixels(w, h, |x, y| if (x / 6 * 7 + y / 6 * 13 + x / 6 * (y / 6)) % 3 == 0 { [40, 30, 120] } else { [230, 220, 200] });
        let blurry = crate::imaging::enhance::unsharp_rgba(&squares, w, h, 1.5, -1.0);
        let adjust = Adjust { filter: Filter::Deblur, contrast: 0.2, ..Default::default() };
        let plain = adjust.apply(&blurry, w, h).1;
        let measured = plain.deblurred.expect("edges enough to measure");
        let set = Adjust { kernel: Some(measured), ..adjust };
        let window = [0, 0, w, h];
        let copy = Enhanced::window(&plain, window, Enhance { enlarge: 2, sharpen: Sharpen::Light }).unwrap();
        let larger = Adjust { enlarge: Enlarge::times(2), sharpen: 0.5, ..adjust };
        assert_ne!(larger.apply(&blurry, w, h).1.deblurred, Some(measured.scaled(2.0)), "the photo enlarged measures otherwise");
        same("deblurred by the blur measured", &copy, &Adjust { kernel: Some(measured), ..larger }.apply(&blurry, w, h).1);
        let shrunk = Enhanced::shrunk(&plain, [0, 0, w, h], 2).unwrap();
        same("deblurred by the blur measured, shrunk", &shrunk, &Adjust { enlarge: Enlarge::HALF, ..set }.apply(&blurry, w, h).1);
        // And where it measured none and left the photo as it was, a copy
        // is left so too.
        let flat = pixels(w, h, |_, _| [128; 3]);
        let plain = adjust.apply(&flat, w, h).1;
        assert_eq!(plain.deblurred, None);
        let copy = Enhanced::window(&plain, window, Enhance { enlarge: 2, sharpen: Sharpen::Light }).unwrap();
        same("not deblurred", &copy, &Adjust { enlarge: Enlarge::times(2), sharpen: 0.5, filter: Filter::None, ..adjust }.apply(&flat, w, h).1);
        // A photo adjusted no more than enlarged is shrunk and closed as
        // it is, where there is nothing to make in any other order.
        let plain = Adjust { enlarge: Enlarge::times(2), ..Default::default() }.apply(&photo, w, h).1;
        assert!(Enhanced::shrunk(&plain, [0, 0, 96, 72], 2).is_none() && Enhanced::closed(&plain, [0, 0, 96, 72], 1).is_none());
    }

    #[test]
    fn the_deblur_filter_undoes_its_kernel_where_the_photo_is_worked_on() {
        // Squares blurred by eight tenths of a pixel.
        let (w, h) = (60, 48);
        let squares: Vec<f32> = (0..w * h).map(|i| if ((i % w) / 6 + (i / w) / 6) % 2 == 0 { 0.1 } else { 0.9 }).collect();
        let photo: Vec<u8> = photographed(&squares, w, h, 0.8);
        // A kernel set in the photo's own pixels is carried to those it
        // is worked on at, and undone there as far as the strength asks,
        // after everything else.
        let kernel = Spread::round(0.8);
        for (enlarge, strength) in [(1u16, 1.0), (2, 0.6), (3, 1.7)] {
            let adjust = Adjust { filter: Filter::Deblur, kernel: Some(kernel), strength, enlarge: Enlarge::times(enlarge), ..Default::default() };
            let luma = adjust.apply(&photo, w, h).1;
            let carried = kernel.scaled(f64::from(enlarge));
            assert_eq!(luma.deblurred, Some(carried), "{enlarge} times");
            let unfiltered = Adjust { filter: Filter::None, ..adjust }.apply(&photo, w, h).1;
            assert!(luma.data == deblur::undone_by(&unfiltered.data, luma.width, luma.height, Some(carried), f64::from(strength), luma.light).0, "{enlarge} times, {strength}");
        }
        // With none set, the blur measured over the whole photo is undone,
        // however little, and said.
        let luma = Adjust { filter: Filter::Deblur, ..Default::default() }.apply(&photo, w, h).1;
        let measured = luma.deblurred.expect("edges enough to measure");
        assert!(measured.extremes().0 < deblur::LEAST_SPREAD && luma.data != Adjust::default().apply(&photo, w, h).1.data);
        // The blur undone turns with the photo; the photo as loaded is
        // kept as it lay, with its kernel, and the turn.
        let leaning = Spread { xx: 2.0, xy: 0.5, yy: 1.0 };
        let turned = Adjust { filter: Filter::Deblur, kernel: Some(leaning), ..Default::default() }.apply(&photo, w, h).1.rotated_cw(1);
        assert_eq!(turned.deblurred, Some(leaning.turned_cw()));
        assert_eq!(turned.as_shot.as_ref().map(|shot| (shot.adjust.kernel, shot.turned)), Some((Some(leaning), 1)));
    }

    #[test]
    fn a_copy_is_deblurred_after_it_is_enlarged() {
        use crate::imaging::enhance::{Enhance, Enhanced, Sharpen};
        // Squares six pixels a side, blurred by eight tenths of a pixel:
        // too little to be worth undoing at the photo's own size, and as
        // much again once it is enlarged twice.
        let (w, h) = (60, 48);
        let squares: Vec<f32> = (0..w * h).map(|i| if ((i % w) / 6 + (i / w) / 6) % 2 == 0 { 0.1 } else { 0.9 }).collect();
        let photo: Vec<u8> = photographed(&squares, w, h, 0.8);
        let deblurred = Enhance { enlarge: 1, sharpen: Sharpen::Deblur };
        // Worked as light, as the lens blurred it, and as stored, which
        // measures that blur over the copy and over the photo a little
        // less alike.
        for (light, most_mean) in [(Light::Linear, 0.003), (Light::Encoded, 0.004)] {
            let plain = Adjust { light, ..Default::default() };
            let own = plain.apply(&photo, w, h).1;
            assert!(Enhanced::window(&own, [0, 0, w, h], deblurred).is_none(), "{light:?}: nothing worth undoing at its own size");
            // Deblurred among the pixels it was enlarged to, a copy of the
            // photo is what the photo enlarged so shows under its Deblur
            // filter, bent by a lens or not: enlarged by the copy itself,
            // as a read that decoded only so sets Enlarge and Filter, and
            // as the photo was enlarged before.
            for k1 in [0.0, 0.12] {
                let twice = Adjust { k1, enlarge: Enlarge::times(2), ..plain };
                let filtered = Adjust { filter: Filter::Deblur, ..twice }.apply(&photo, w, h).1;
                let copies = [
                    Enhanced::window(&Adjust { k1, ..plain }.apply(&photo, w, h).1, [0, 0, w, h], Enhance { enlarge: 2, ..deblurred }),
                    Enhanced::window(&twice.apply(&photo, w, h).1, [0, 0, 2 * w, 2 * h], deblurred),
                ];
                for (copy, how) in copies.into_iter().zip(["by the copy", "before"]) {
                    let copy = copy.unwrap_or_else(|| panic!("{light:?} {k1}, enlarged {how}: the blur is worth undoing"));
                    assert_eq!((copy.luma.width, copy.luma.height, copy.factor), (2 * w, 2 * h, [if how == "before" { 1.0 } else { 2.0 }; 2]));
                    let differences: Vec<f32> = copy.luma.data.iter().zip(&filtered.data).map(|(a, b)| (a - b).abs()).collect();
                    let worst = differences.iter().copied().fold(0.0f32, f32::max);
                    let mean = differences.iter().sum::<f32>() / differences.len() as f32;
                    assert!(worst < 0.02 && mean < most_mean, "{light:?} {k1}, enlarged {how}: {worst} at the most, {mean} on the whole");
                }
            }
        }
    }

    #[test]
    fn a_photo_larger_than_the_search_deblurs_is_deblurred_as_a_read_copy_of_it_is() {
        use crate::imaging::enhance::{Enhance, Enhanced, Sharpen};
        // A photograph larger than the copies the search deblurs, with a
        // patch of squares four pixels a side blurred by two and a fifth:
        // under the Deblur filter set to the blur a read's copy of the
        // patch undid, as a read that decoded only so sets it, the photo
        // shows the patch as the copy did. Deconvolved at the most the
        // search's copies are, the photo was reduced first, and read
        // modules wrong that the copy read right.
        let (w, h) = (1600, 1000);
        assert!(w * h > deblur::MOST_PIXELS);
        let sharp: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let inside = (300..420).contains(&x) && (300..420).contains(&y);
                if inside && ((x / 4) * 7 + (y / 4) * 13 + (x / 4) * (y / 4)) % 3 == 0 { 0.1 } else { 0.9 }
            })
            .collect();
        let photo: Vec<u8> = photographed(&sharp, w, h, 2.2);
        let plain = Adjust::default().apply(&photo, w, h).1;
        let window = [280, 280, 440, 440];
        let copy = Enhanced::window(&plain, window, Enhance { enlarge: 1, sharpen: Sharpen::Deblur }).expect("a blur worth undoing");
        assert_eq!((copy.origin, copy.factor), ([280.0, 280.0], [1.0; 2]));
        let filtered = Adjust { filter: Filter::Deblur, kernel: copy.luma.deblurred, ..Default::default() }.apply(&photo, w, h).1;
        let there = cut(&filtered.data, w, 1, window);
        let worst = copy.luma.data.iter().zip(&there).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "{worst}");
    }

    #[test]
    fn a_window_of_a_photo_deblurred_is_the_photo_made_again_there() {
        use crate::imaging::enhance::{Enhance, Enhanced, Sharpen};
        // Squares blurred, under a Deblur filter whose kernel, carried to
        // the copy's pixels, is wider than a deconvolution works at, and
        // narrower: a copy of a window of the photo is reduced as the
        // whole photo made again at the copy's size is, on its grid, and
        // reads as far about the window as the deconvolution does, and it
        // comes out as the whole does there. Cut off that grid and short
        // of that reach, as it once was, it differed by a quarter of the
        // contrast.
        let (w, h) = (150, 120);
        let squares: Vec<f32> = (0..w * h).map(|i| if ((i % w) / 7 + (i / w) / 7 + (i % w) / 21) % 3 == 0 { 0.1 } else { 0.9 }).collect();
        let photo: Vec<u8> = photographed(&squares, w, h, 1.3);
        let mut reduced = 0;
        for (kernel, strength) in [(Spread::round(1.2), 1.0), (Spread { xx: 1.6, xy: 0.4, yy: 0.7 }, 1.4), (Spread::round(0.6), 0.8)] {
            let adjust = Adjust { filter: Filter::Deblur, kernel: Some(kernel), strength, contrast: 0.2, ..Default::default() };
            let plain = adjust.apply(&photo, w, h).1;
            for (enlarge, read) in [(2u8, Sharpen::Light), (2, Sharpen::Off), (1, Sharpen::Strong), (3, Sharpen::Light)] {
                let set = Adjust { enlarge: Enlarge::times(enlarge.into()), sharpen: if read == Sharpen::Off { 0.0 } else { read.amount() }, ..adjust };
                let whole = set.apply(&photo, w, h).1;
                let widest = kernel.scaled(f64::from(enlarge) * f64::from(strength)).extremes().0;
                reduced += usize::from(deblur::Reduction::for_blur(widest, whole.width * whole.height, deblur::MOST_READ_PIXELS, 1.0) != deblur::Reduction::NONE);
                for window in [[30, 24, 110, 90], [0, 0, 60, 50], [95, 70, 150, 120]] {
                    let copy = Enhanced::window(&plain, window, Enhance { enlarge, sharpen: read }).unwrap();
                    let [cx, cy] = std::array::from_fn(|i| (copy.origin[i] * copy.factor[i]).round() as usize);
                    let there = cut(&whole.data, whole.width, 1, [cx, cy, cx + copy.luma.width, cy + copy.luma.height]);
                    let worst = copy.luma.data.iter().zip(&there).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                    assert!(worst < 1e-4, "{kernel:?} at {strength}, read {read:?} {enlarge} times through {window:?}: {worst}");
                }
            }
        }
        assert!(reduced >= 3, "the deconvolution was reduced for {reduced} of the copies");
    }

    #[test]
    fn a_photo_deblurred_and_turned_is_made_again_as_it_lay_and_turned() {
        use crate::imaging::enhance::{Enhance, Enhanced, Sharpen};
        // Squares blurred, a photo whose height the deconvolution's
        // reduction does not divide, under a Deblur filter with a kernel
        // set wide enough to be reduced for, and bent by a lens with the
        // blur it measures. Turned each way, the photo made again from the
        // photo as loaded at its size is the photo turned, and a window of
        // it enlarged and sharpened is what the photo so adjusted, turned,
        // shows there. Made again from the photo as loaded turned, its
        // reduction's grid was laid from the corner the turn brought
        // round, and the copy differed from the photo by more than a
        // quarter of the way from black to white along its edges.
        let (w, h) = (300, 241);
        let squares: Vec<f32> = (0..w * h).map(|i| if ((i % w) / 9 + (i / w) / 9 + (i % w) / 27) % 3 == 0 { 0.1 } else { 0.9 }).collect();
        let photo: Vec<u8> = photographed(&squares, w, h, 2.5);
        let kernel = Spread { xx: 12.0, xy: 2.0, yy: 7.0 };
        assert!(deblur::Reduction::for_blur(kernel.extremes().0, w * h, deblur::MOST_READ_PIXELS, 1.0) != deblur::Reduction::NONE);
        for adjust in [Adjust { filter: Filter::Deblur, kernel: Some(kernel), ..Default::default() }, Adjust { filter: Filter::Deblur, k1: 0.1, contrast: 0.2, ..Default::default() }] {
            let plain = adjust.apply(&photo, w, h).1;
            let enhance = Enhance { enlarge: 2, sharpen: Sharpen::Light };
            let larger = Adjust { enlarge: Enlarge::times(2), sharpen: Sharpen::Light.amount(), kernel: adjust.kernel.or(plain.deblurred), ..adjust }.apply(&photo, w, h).1;
            for quarters in 1..4u8 {
                let turned = plain.rotated_cw(quarters);
                let what = format!("{:?}, {quarters} turns", adjust.kernel);
                let worst = |copy: &Enhanced, whole: &Luma| {
                    let [cx, cy] = std::array::from_fn(|i| (copy.origin[i] * copy.factor[i]).round() as usize);
                    let there = cut(&whole.data, whole.width, 1, [cx, cy, cx + copy.luma.width, cy + copy.luma.height]);
                    copy.luma.data.iter().zip(&there).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
                };
                let again = Enhanced::shrunk(&turned, [0, 0, turned.width, turned.height], 1).expect("the photo made again");
                assert_eq!((again.origin, again.factor, again.luma.width, again.luma.height), ([0.0; 2], [1.0; 2], turned.width, turned.height), "{what}");
                assert!(worst(&again, &turned) < 1e-4, "{what}: made again, {}", worst(&again, &turned));
                for window in [[20, 30, 140, 110], [0, 0, 90, 70], [150, 170, 241, 241]] {
                    let part = Enhanced::shrunk(&turned, window, 1).expect("a window made again");
                    assert!(worst(&part, &turned) < 1e-4, "{what}: {window:?} made again, {}", worst(&part, &turned));
                    let copy = Enhanced::window(&turned, window, enhance).expect("a copy");
                    let whole = larger.rotated_cw(quarters);
                    assert!(worst(&copy, &whole) < 1e-4, "{what}: {window:?} enlarged, {}", worst(&copy, &whole));
                }
            }
        }
    }

    #[test]
    fn a_copy_of_a_photo_adjusted_shrunk_any_number_of_times_lies_where_its_pixels_were() {
        use crate::imaging::enhance::Enhanced;
        // A step from ink to paper at the photo's 600th pixel across, the
        // photo toned, and so made again for every copy, at its own size,
        // twice and half and a quarter of it. Shrunk two or three times,
        // a window of it is cut at whole steps of the copy's own pixels,
        // and the step lies in the copy where the copy says it does. Cut
        // at whole fours of the photo's pixels, as it once was, a copy a
        // third of its size or an eighth lay up to a pixel and a half
        // off.
        let (w, h) = (1200, 160);
        let photo = pixels(w, h, |x, _| if x < 600 { [30; 3] } else { [220; 3] });
        for (base, factor) in [(Enlarge::NONE, 3u8), (Enlarge::times(2), 3), (Enlarge::HALF, 3), (Enlarge::QUARTER, 2), (Enlarge::HALF, 2)] {
            let luma = Adjust { contrast: 0.2, enlarge: base, ..Default::default() }.apply(&photo, w, h).1;
            let (dark, light) = (luma.data[0], luma.data[luma.width - 1]);
            let k = usize::from(factor);
            for x0 in [0, 5, 37, 61, 98] {
                let (ww, hh) = (luma.width - x0, luma.height - 4);
                let copy = Enhanced::shrunk(&luma, [x0, 4, x0 + ww / k * k, 4 + hh / k * k], factor).expect("an adjusted photo is made again");
                assert!(copy.factor.iter().all(|f| (f - 1.0 / f64::from(factor)).abs() < 1e-12), "{:?}", copy.factor);
                let row = &copy.luma.data[copy.luma.height / 2 * copy.luma.width..(copy.luma.height / 2 + 1) * copy.luma.width];
                let middle = (dark + light) / 2.0;
                let i = row.windows(2).position(|pair| pair[0] < middle && pair[1] >= middle).expect("the step");
                let at = i as f64 + 0.5 + f64::from((middle - row[i]) / (row[i + 1] - row[i]));
                let step = copy.to_photo([at, 0.0])[0];
                assert!((step - 600.0 * base.factor()).abs() < 0.1, "{base} shrunk {factor} times from {x0}: the step at {step}");
            }
        }
    }

    #[test]
    fn a_copy_of_a_photo_reduced_lies_where_the_photo_reduced_has_its_pixels() {
        use crate::imaging::enhance::{Enhance, Enhanced, Sharpen};
        // A block of ink in the top-left of a photo 1,203 pixels across
        // and 802 down, its edges near the far sides on whole eighths of
        // the photo's pixels, where every copy's pixels meet and none
        // reads the edge off by the share of a pixel it falls in. Reduced,
        // each side comes to the nearest whole pixel, so the photo at half
        // its size is reduced a little less across than it says and exactly
        // as it says down, and at a quarter a little less both ways. A copy
        // of it made again at the photo's own size, or shrunk twice more,
        // is made on the photo's own grid, and says where its edges lie in
        // the photo reduced as the photo reduced has them. Placed by the
        // enlargement alone, such a copy lay a quarter of a pixel off and
        // more, and half a pixel across at half the photo's size.
        let (w, h, ex, ey) = (1203, 802, 1160, 760);
        let photo = pixels(w, h, |x, y| if x < ex && y < ey { [30; 3] } else { [220; 3] });
        // Where the light crosses the middle along `values`, a pixel apart.
        let crossing = |values: &[f32], middle: f32| {
            let i = values.windows(2).position(|pair| pair[0] < middle && pair[1] >= middle).expect("the edge");
            i as f64 + 0.5 + f64::from((middle - values[i]) / (values[i + 1] - values[i]))
        };
        let row = |luma: &Luma, y: usize| luma.data[y * luma.width..(y + 1) * luma.width].to_vec();
        let column = |luma: &Luma, x: usize| (0..luma.height).map(|y| luma.data[y * luma.width + x]).collect::<Vec<f32>>();
        for base in [Enlarge::HALF, Enlarge::QUARTER] {
            let luma = Adjust { enlarge: base, ..Default::default() }.apply(&photo, w, h).1;
            let middle = (luma.data[0] + luma.data[luma.data.len() - 1]) / 2.0;
            // Each edge where the photo reduced has it, read across the
            // middle of the block.
            let (mx, my) = (luma.width / 2, luma.height / 2);
            let (across, down) = (crossing(&row(&luma, my), middle), crossing(&column(&luma, mx), middle));
            let times = (1.0 / base.factor()).round() as u8;
            let whole = [0, 0, luma.width, luma.height];
            let own = Enhanced::window(&luma, whole, Enhance { enlarge: times, sharpen: Sharpen::Off }).expect("a copy at the photo's own size");
            let shrunk = Enhanced::shrunk(&luma, whole, 2).expect("a copy shrunk");
            for (what, copy) in [("at its own size", own), ("shrunk", shrunk)] {
                let [cx, cy] = copy.to_view([mx as f64 + 0.5, my as f64 + 0.5]).map(|v| v as usize);
                let found = [copy.to_photo([crossing(&row(&copy.luma, cy), middle), 0.0])[0], copy.to_photo([0.0, crossing(&column(&copy.luma, cx), middle)])[1]];
                assert!((found[0] - across).abs() < 0.1, "{base}, {what}: the edge across at {} where the photo has it at {across}", found[0]);
                assert!((found[1] - down).abs() < 0.1, "{base}, {what}: the edge down at {} where the photo has it at {down}", found[1]);
            }
        }
    }

    #[test]
    fn a_photo_is_straightened_after_it_is_enlarged() {
        // Busy colours, bent by a lens: enlarged twice, the photo is
        // straightened among the pixels the enlargement drew, and not
        // straightened at its own size and enlarged after.
        let (w, h) = (48, 36);
        let photo = pixels(w, h, |x, y| [((x * 37 + y * 11) % 256) as u8, ((x * 13 + y * 29) % 256) as u8, (((x / 3 + y / 2) % 2) * 255) as u8]);
        // On white, as [`undistort`] lays what falls off the photo.
        let lens = Adjust { k1: 0.12, enlarge: Enlarge::times(2), backdrop: Backdrop::White, ..Default::default() };
        let (display, luma) = lens.apply(&photo, w, h);
        let enlarged = resize(&photo, w, h, 4, 2 * w, 2 * h);
        assert_eq!(display, undistort(&enlarged, 2 * w, 2 * h, 0.12));
        assert_ne!(display, resize(&undistort(&photo, w, h, 0.12), w, h, 4, 2 * w, 2 * h));
        // Beside it, the photo at its own size straightened there.
        let own = luma.unenlarged.as_ref().expect("an enlarged photo keeps its own pixels");
        assert_eq!(own.data, Adjust { enlarge: Enlarge::NONE, ..lens }.apply(&photo, w, h).1.data);
        // And the photo as loaded, with the adjustments, for a copy of it
        // to be made from: all of it, which the corners of a photo bent
        // so draw on.
        let shot = luma.as_shot.as_ref().expect("a straightened photo keeps the photo as loaded");
        assert!(shot.rgba[..] == photo[..]);
        assert_eq!((shot.at, shot.photo, shot.part, shot.adjust), ([0, 0], [w, h], [0, 0, w, h], lens));
        assert!(Adjust::default().apply(&photo, w, h).1.as_shot.is_none(), "nothing to keep with no adjustment");
        assert!(Adjust { enlarge: Enlarge::times(3), ..Default::default() }.apply(&photo, w, h).1.as_shot.is_none(), "nor with an enlargement alone");
        for more in [Adjust { contrast: 0.2, ..Default::default() }, Adjust { enlarge: Enlarge::HALF, ..Default::default() }, Adjust { filter: Filter::Median, ..Default::default() }] {
            assert!(more.apply(&photo, w, h).1.as_shot.is_some(), "{more:?}");
        }
        // Handed over shared, the photo as loaded is kept as the same
        // bytes, and not copied, and the photo is adjusted the same; a
        // part drawing on less of it keeps a copy of what it draws on.
        let shared: SharedRgba = Arc::new(photo.clone());
        let (display, kept) = lens.apply_to_shared(&shared, w, h, [0, 0, w, h]);
        assert!(display == lens.apply(&photo, w, h).0 && kept.data == luma.data);
        let shot = kept.as_shot.as_ref().expect("a straightened photo keeps the photo as loaded");
        assert!(Arc::ptr_eq(&shot.rgba, &shared), "the photo as loaded is held once");
        assert_eq!(**shot, **luma.as_shot.as_ref().unwrap());
        let (w, h) = (160, 120);
        let larger: SharedRgba = Arc::new(pixels(w, h, |x, y| [((x * 37 + y * 11) % 256) as u8, ((x * 13 + y * 29) % 256) as u8, 128]));
        let part = lens.apply_to_shared(&larger, w, h, [70, 50, 10, 10]).1;
        let shot = part.as_shot.as_ref().expect("a straightened part keeps the photo as loaded about it");
        assert!(!Arc::ptr_eq(&shot.rgba, &larger) && shot.rgba.len() < larger.len(), "{} of {} bytes", shot.rgba.len(), larger.len());
    }

    #[test]
    fn a_photo_read_and_not_shown_reads_as_it_does_shown() {
        // Its luminance alone, with all it keeps beside it: the photo at
        // its own size where it is enlarged, and as loaded where it is
        // adjusted more than that; filtered, toned and straightened.
        let (w, h) = (90, 70);
        let photo: SharedRgba = Arc::new(pixels(w, h, |x, y| [((x * 37 + y * 11) % 256) as u8, ((x * 13 + y * 29) % 256) as u8, (((x / 3 + y / 2) % 2) * 255) as u8]));
        for adjust in [
            Adjust::default(),
            Adjust { enlarge: Enlarge::times(2), ..Default::default() },
            Adjust { enlarge: Enlarge::times(2), filter: Filter::Median, contrast: 0.3, ..Default::default() },
            Adjust { k1: 0.15, sharpen: 1.0, filter: Filter::Close, radius: 2, channel: Channel::Green, ..Default::default() },
            Adjust { filter: Filter::Deblur, kernel: Some(Spread::round(1.2)), ..Default::default() },
        ] {
            for part in [[0, 0, w, h], [10, 5, 50, 40]] {
                assert!(adjust.luma_of_shared(&photo, w, h, part) == adjust.apply_to_shared(&photo, w, h, part).1, "{adjust:?} of {part:?}");
            }
        }
    }

    #[test]
    fn a_part_by_the_edge_is_straightened_as_the_whole_is_there() {
        use crate::imaging::enhance::{Enhance, Enhanced, Sharpen};
        let (w, h) = (400, 300);
        let photo = pixels(w, h, |x, y| [((x * 37 + y * 11) % 256) as u8, ((x * 13 + y * 29) % 256) as u8, (((x / 3 + y / 2) % 2) * 255) as u8]);
        // A strip along the right edge of a photo whose lens term
        // straightens pincushion reads every one of its pixels from past
        // that edge: it draws on none of the photo, and is the backdrop,
        // as the whole photo so straightened is there, at its own size
        // and enlarged.
        for part @ [x, y, pw, ph] in [[390, 140, 10, 20], [395, 0, 5, 300], [380, 0, 20, 300]] {
            let whole = undistort(&photo, w, h, 0.3);
            assert!(straightened(&photo, w, h, 0.3, part, 255) == cut(&whole, w, 4, [x, y, x + pw, y + ph]), "{part:?}");
            for enlarge in [1, 2] {
                let adjust = Adjust { k1: 0.3, enlarge: Enlarge::times(enlarge), ..Default::default() };
                let e = usize::from(enlarge);
                let whole = adjust.apply(&photo, w, h).0;
                let there = cut(&whole, w * e, 4, [x * e, y * e, (x + pw) * e, (y + ph) * e]);
                assert!(adjust.apply_to(&photo, w, h, part).0 == there, "{part:?}, {enlarge} times");
            }
        }
        // A copy of a window by the photo's edges, bent either way, is
        // the photo made again there: a lens term that straightens barrel
        // draws the photo's edges in from nearer its centre, from pixels
        // the window does not cover.
        for k1 in [-0.3, 0.3] {
            let adjust = Adjust { k1, contrast: 0.2, ..Default::default() };
            let plain = adjust.apply(&photo, w, h).1;
            let whole = Adjust { enlarge: Enlarge::times(2), sharpen: 0.5, ..adjust }.apply(&photo, w, h).1;
            for window in [[0, 0, 60, 60], [340, 240, 400, 300], [380, 100, 400, 200]] {
                let copy = Enhanced::window(&plain, window, Enhance { enlarge: 2, sharpen: Sharpen::Light }).unwrap();
                let [cx, cy] = std::array::from_fn(|i| (copy.origin[i] * copy.factor[i]).round() as usize);
                let there = cut(&whole.data, whole.width, 1, [cx, cy, cx + copy.luma.width, cy + copy.luma.height]);
                assert!(copy.luma.data == there, "{k1}: {window:?}");
            }
        }
    }

    #[test]
    fn a_part_is_straightened_as_the_whole_is_there() {
        // The part of a photo bent by a lens, straightened about the
        // whole photo's centre from what of the photo it draws on alone:
        // what the whole photo so adjusted shows there, pixel for pixel,
        // at its own size and enlarged, and by pincushion and barrel.
        let (w, h) = (128, 96);
        let photo = pixels(w, h, |x, y| [((x * 37 + y * 11) % 256) as u8, ((x * 13 + y * 29) % 256) as u8, (((x / 3 + y / 2) % 2) * 255) as u8]);
        let part = [5, 30, 22, 14];
        for (k1, enlarge) in [(0.12, 1u16), (0.12, 3), (-0.2, 2)] {
            let adjust = Adjust { k1, enlarge: Enlarge::times(enlarge), ..Default::default() };
            let e = usize::from(enlarge);
            let whole = adjust.apply(&photo, w, h).0;
            let (display, luma) = adjust.apply_to(&photo, w, h, part);
            let there = cut(&whole, w * e, 4, [part[0] * e, part[1] * e, (part[0] + part[2]) * e, (part[1] + part[3]) * e]);
            assert_eq!((luma.width, luma.height), (part[2] * e, part[3] * e));
            assert!(display == there, "{k1}, {enlarge} times");
            if e == 1 {
                assert!(display == straightened(&photo, w, h, k1, part, adjust.backdrop.level()), "{k1}");
                assert!(straightened(&photo, w, h, k1, part, 255) == cut(&undistort(&photo, w, h, k1), w, 4, [5, 30, 27, 44]));
            }
            let shot = luma.as_shot.as_ref().expect("a straightened part keeps the photo as loaded about it");
            assert!(shot.width < w && shot.height < h, "{k1}: no more of the photo than the part draws on");
            assert_eq!((shot.part, shot.adjust), (part, adjust));
        }
    }

    #[test]
    fn a_part_bent_by_a_lens_is_made_again_shrunk_as_the_whole_photo_makes_it() {
        use crate::imaging::enhance::Enhanced;
        // A part of a photo bent by a lens, kept with as much of the photo
        // as loaded as it draws on, and shrunk for a search as far as a
        // search shrinks it ([`most_shrunk`]), at its own size and below
        // it: each copy is the one the whole photo as loaded makes, pixel
        // for pixel. Kept for copies no smaller than a quarter of the
        // photo, a part at half its size shrunk three times was cut at the
        // edge of what was kept, and not at a whole step of its pixels,
        // and came out as much as eight grey levels off.
        let (w, h) = (600, 450);
        let photo = pixels(w, h, |x, y| [((x * 37 + y * 11) % 256) as u8, ((x * 13 + y * 29) % 256) as u8, (((x / 3 + y / 2) % 2) * 255) as u8]);
        let part = [200, 150, 160, 120];
        for k1 in [0.2, -0.25] {
            for enlarge in [Enlarge::NONE, Enlarge::HALF, Enlarge::QUARTER] {
                let luma = Adjust { k1, contrast: 0.2, enlarge, ..Default::default() }.apply_to(&photo, w, h, part).1;
                let kept = luma.as_shot.as_deref().expect("a straightened part keeps the photo as loaded about it");
                assert!(kept.width < w && kept.height < h, "{k1}, {enlarge}: no more of the photo than the part draws on");
                let mut whole = luma.clone();
                whole.as_shot = Some(Arc::new(AsShot { rgba: Arc::new(photo.clone()), width: w, height: h, at: [0, 0], ..kept.clone() }));
                for factor in 2..=most_shrunk(luma.width, luma.height) {
                    let k = usize::from(factor);
                    let window = [0, 0, luma.width / k * k, luma.height / k * k];
                    let copy = Enhanced::shrunk(&luma, window, factor).expect("a copy of the part");
                    let all = Enhanced::shrunk(&whole, window, factor).expect("a copy of the whole");
                    let worst = copy.luma.data.iter().zip(&all.luma.data).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                    assert!(
                        (copy.origin, copy.factor, copy.luma.width, copy.luma.height) == (all.origin, all.factor, all.luma.width, all.luma.height) && worst == 0.0,
                        "{k1}, {enlarge} shrunk {factor} times: {worst} at the most"
                    );
                }
            }
        }
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
        // A stop up doubles the light, and a stop down halves it: white
        // comes to 188, which stores half its light, and 128 to 176, which
        // stores twice its.
        let up = Adjust { exposure: 1.0, ..Default::default() };
        assert_eq!(up.apply(&[0, 128, 255, 255], 1, 1).0[..3], [0, 176, 255]);
        let down = Adjust { exposure: -1.0, ..Default::default() };
        assert_eq!(down.apply(&[0, 176, 255, 255], 1, 1).0[..3], [0, 128, 188]);
        // In light, whatever light the photo is worked in; and after the
        // photo is inverted, so that up is lighter either way.
        assert_eq!(Adjust { light: Light::Linear, ..up }.apply(&[0, 128, 255, 255], 1, 1).0[..3], [0, 176, 255]);
        assert_eq!(Adjust { invert: true, ..up }.apply(&[255, 127, 0, 255], 1, 1).0[..3], [0, 176, 255]);
        // Before the contrast: a hard threshold at mid grey cuts the photo
        // where the exposure moved it to, a stop up taking 100 to light.
        assert_eq!(Adjust { contrast: 1.0, ..up }.apply(&[100, 100, 100, 255], 1, 1).0[..3], [255; 3]);
        assert_eq!(hard.apply(&[100, 100, 100, 255], 1, 1).0[..3], [0; 3]);
        // None is the photo as it is, value for value.
        assert!(Adjust::default().lookup_table().iter().enumerate().all(|(i, &v)| usize::from(v) == i));
    }

    #[test]
    fn filters_mend_specks_and_dots() {
        // A 7×7 light field with one dark speck in the middle.
        let (w, h) = (7, 7);
        let mut speck = vec![1.0f32; w * h];
        speck[3 * w + 3] = 0.0;
        let out = Filter::Median.apply(&speck, w, h, 1, Light::Encoded);
        assert!(out.iter().all(|&v| v == 1.0), "the median removes a lone speck");
        // A dark module printed as a dotted grid: every other pixel dark.
        let dotted: Vec<f32> = (0..w * h).map(|i| if (i % w) % 2 == 0 && (i / w) % 2 == 0 { 0.0 } else { 1.0 }).collect();
        let closed = Filter::Close.apply(&dotted, w, h, 1, Light::Encoded);
        assert!(closed.iter().all(|&v| v == 0.0), "closing joins the dots into solid dark: {closed:?}");
        // Closing a solid light field leaves it alone. A light gap three
        // pixels wide survives a radius of one, where one of two pixels is
        // closed: the darkest-of-three pass reaches one pixel each way.
        assert!(Filter::Close.apply(&vec![1.0f32; w * h], w, h, 1, Light::Encoded).iter().all(|&v| v == 1.0));
        let mut gap = vec![0.0f32; w * h];
        let mut narrow = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 2..5 {
                gap[y * w + x] = 1.0;
            }
            narrow[y * w + 3] = 1.0;
            narrow[y * w + 4] = 1.0;
        }
        let closed = Filter::Close.apply(&gap, w, h, 1, Light::Encoded);
        assert!(closed.contains(&1.0), "{closed:?}");
        assert!(Filter::Close.apply(&narrow, w, h, 1, Light::Encoded).iter().all(|&v| v == 0.0));
        assert_eq!(Filter::None.apply(&speck, w, h, 2, Light::Encoded), speck);
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
        // The far corners fall off the photo and come back white; under
        // the adjustments, the backdrop the photo is laid on, as its
        // transparent parts are.
        assert_eq!(&out[..4], &[255, 255, 255, 255]);
        for backdrop in Backdrop::ALL {
            let (display, luma) = Adjust { k1: 0.1, backdrop, ..Default::default() }.apply(&ramp, w, h);
            assert_eq!(display[(y * w + x) * 4], expected);
            assert_eq!(&display[..4], &[backdrop.level(), backdrop.level(), backdrop.level(), 255], "{backdrop:?}");
            assert!((luma.data[0] - f32::from(backdrop.level()) / 255.0).abs() < 1e-5, "{backdrop:?}: {}", luma.data[0]);
        }
    }

    #[test]
    fn a_rectangle_is_cut_as_its_rows_are_joined() {
        let (width, height, channels) = (7, 5, 3);
        let data: Vec<u16> = (0..(width * height * channels) as u16).collect();
        let joined = |[x0, y0, x1, y1]: [usize; 4]| -> Vec<u16> { (y0..y1).flat_map(|y| data[(y * width + x0) * channels..(y * width + x1) * channels].iter().copied()).collect() };
        // Whole rows, some of them, and none, wherever none would begin:
        // past the last row, or with the rows the wrong way about.
        for rect in [[0, 0, width, height], [0, 1, width, 4], [2, 1, 6, 4], [3, 0, 3, height], [0, 2, width, 2], [0, height + 3, width, height + 3], [0, 4, width, 1], [2, 4, 6, 1]] {
            assert_eq!(cut(&data, width, channels, rect), joined(rect), "{rect:?}");
        }
    }
}
