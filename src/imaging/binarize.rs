//! Thresholding the photo as the finder scan sees it.
//!
//! The scan in [`super::detect`] reads the outline two ways and pools
//! what it finds: against one threshold for the whole outline, Otsu's
//! split of its luminance, and against a threshold that follows the
//! lighting, the mean of the blocks of pixels around each pixel, as
//! ZXing's HybridBinarizer and OpenCV's `adaptiveThreshold` take it. The
//! local rule lives here, so that the scan and the app's view of what it
//! sees are one code.
//!
//! A block whose neighbourhood has no spread of luminance is flat, and a
//! flat block reads as background: noise there makes no runs. ZXing's
//! HybridBinarizer does otherwise, and lets a flat block inside a dark
//! region take its threshold from the blocks around it, so that the
//! region stays dark. A finder pattern eight pixels a module or more is
//! flat in the middle of its stone and reads light there under this
//! rule; the other threshold reads it whole, and a trial over a lighting
//! gradient at every module size found no finder lost to the difference.
//! What counts as no spread is a share of the spread of the image
//! thresholded, not a fixed luminance: ZXing's fixed 24 of 255 reads an
//! underexposed photo, whose symbol spans less than that from ink to
//! paper, as flat all over, where BoofCV's block Otsu, which has no such
//! floor, reads it.
//!
//! The scan reads the outline a third way besides, against the mean of a
//! window a few pixels about each pixel ([`window_dark`]), at each of a
//! few widths, and pools those hits with the rest. A finder pattern
//! blurred by half a module or more keeps its rings only as ripples: the
//! light ring between the dark one and the stone rises barely past the
//! middle of the scale and the dark ring outside it sinks barely below,
//! so that one threshold for the outline, or the mean of a neighbourhood
//! as wide as the finder and wider, which is one threshold as far as the
//! finder is concerned, runs the rings together or loses them. Against a
//! window of a module or two either side, each ring is darker or lighter
//! than what lies about it, however faint. OpenCV's ArUco-based QR
//! detector thresholds at windows of three, thirteen and twenty-three
//! pixels and keeps what any of them finds, and wechat_qrcode reads each
//! crop through four binarizers whose windows run from a tenth to a
//! quarter of it.

use std::fmt;
use std::sync::Arc;

use super::Luma;
use super::gpu::scan::{self, Rule, Values};
use super::sample::{BINS, NOISE_RANGE, histogram_of, otsu_of, robust_spread_of};
use crate::work;

/// What a reading makes of a pixel, a byte to each: light, dark, or
/// nothing, where the image has no value, as a rectified image has none
/// off the photo.
pub const LIGHT: u8 = 0;
pub const DARK: u8 = 1;
pub const NONE: u8 = 2;

/// Pixels on a side of the blocks the local threshold averages.
pub const BLOCK: usize = 8;
/// Least spread of luminance within a block and its eight neighbours for
/// the block to carry any pattern at all, as a share of the spread of the
/// whole image thresholded between its darkest and lightest hundredth:
/// a tenth of the scale on a photo that spans it, as ZXing has it, and
/// never less than the noise of the sensor.
pub const RANGE_SHARE: f32 = 0.12;

/// The local threshold's radius, in pixels, for a symbol whose outline is
/// `side` pixels on its shorter side: wider than a finder pattern, so the
/// finder's own dark does not pull the threshold down over it.
pub fn local_radius_for(side: f64) -> f64 {
    (side / 6.0).clamp(8.0, 256.0)
}

/// The spread of luminance below which a block and its neighbours are
/// flat, for an image whose own spread is `spread`: [`RANGE_SHARE`] of
/// it, or the sensor's noise if that is more.
pub fn min_range(spread: f32) -> f32 {
    (RANGE_SHARE * spread).max(NOISE_RANGE)
}

/// The local threshold reaching `radius_px` pixels about each pixel, as
/// the card is given it: the neighbourhood rounded up to whole blocks.
pub(crate) fn local_rule(radius_px: f64, spread: f32) -> Rule {
    Rule::Local { radius: (radius_px.max(0.0) / BLOCK as f64).ceil() as usize, min_range: min_range(spread) }
}

/// Each pixel of a `width` × `height` image against the lighting around
/// it, for a threshold that follows shadows and glare: [`DARK`] where it
/// lies below the mean of the blocks within `radius_px` pixels of its
/// own block, rounded up to whole blocks, each block's mean weighing the
/// same, whole or clipped by the image's edge; [`LIGHT`] otherwise, and
/// where its block and that block's eight neighbours spread less than
/// the flat floor ([`min_range`]), which tells a flat patch from one with
/// pattern in it; and [`NONE`] where it has no value, `NaN` where a
/// rectified image has no photo, which counts for nothing. The floor is
/// set by the spread of the pixels inside `region`, `[x0, y0, x1, y1]`
/// with the far sides exclusive, when one is given, and of the whole
/// image otherwise: the scan sets its floor by the spread of the outline
/// it looks at, and a view of that outline must do the same, or a dim
/// symbol in a bright photo reads as flat in the view where the scan
/// reads it, and the reverse.
pub fn local_dark(data: &[f32], width: usize, height: usize, radius_px: f64, region: Option<[usize; 4]>) -> Vec<u8> {
    assert_eq!(data.len(), width * height);
    let spread = robust_spread_of(&histogram(data, width, height, region));
    scan::readings(Values::Cores(data), (width, height), &[(local_rule(radius_px, spread), false)], false).into_only()
}

/// Rows of a region counted as one piece on the cores to spare
/// ([`histogram`]).
const HISTOGRAM_BAND: usize = 64;

/// How the values of `region` of `data`, an image `width` by `height`,
/// fall ([`histogram_of`]), or of the whole image where none is given, the
/// pixels with no value left out: counted on the cores that hold them, a
/// band of rows to each. Counted on the graphics card, where the
/// binarised view reads the image, the view waited on the card for the
/// counts before it could read it, every frame a corner was dragged; and
/// the card counts as the cores do.
fn histogram(data: &[f32], width: usize, height: usize, region: Option<[usize; 4]>) -> [u32; BINS] {
    let [x0, y0, x1, y1] = region.map_or([0, 0, width, height], |r| [r[0].min(width), r[1].min(height), r[2].min(width), r[3].min(height)]);
    if x0 >= x1 || y0 >= y1 {
        return [0; BINS];
    }
    let bands: Vec<usize> = (y0..y1).step_by(HISTOGRAM_BAND).collect();
    let counted = work::map(&bands, |&from| {
        let rows = from..(from + HISTOGRAM_BAND).min(y1);
        histogram_of(rows.flat_map(|y| data[y * width + x0..y * width + x1].iter().copied()).filter(|v| v.is_finite()))
    });
    counted.iter().fold([0u32; BINS], |mut all, counts| {
        all.iter_mut().zip(counts).for_each(|(all, &n)| *all += n);
        all
    })
}

/// Half-widths, in pixels, of the windows the scan thresholds each pixel
/// against besides ([`window_dark`]). A window must reach past the stone
/// of a finder, three modules across, into the ring about it, or the
/// middle of the stone reads as light as itself; and it must stop short
/// of the finder's far side, or it is the neighbourhood rule over again.
/// So each width serves modules from a third of it to about the whole of
/// it, and the four between them modules from a pixel and a half, the
/// least the scan looks for, to thirty pixels, past which blur no longer
/// runs a finder's rings together.
pub const WINDOW_RADII: [usize; 4] = [4, 8, 16, 32];
/// How far below the mean of its window a pixel must lie to read dark,
/// as a share of the spread of the image thresholded: without it the
/// grain of flat paper reads as a scatter of dark pixels, half of them.
/// OpenCV's ArUco-based detector takes seven of 255 off its window means
/// for the same reason.
pub const WINDOW_BIAS: f32 = 0.03;

/// How the finder scan thresholds the photo: the three readings it pools,
/// or one of them alone, reaching as far as it is set to, as the
/// binarised view shows one ([`BinarizeParams::binarizer`]). The scan
/// reads the outline, or the whole photo, against one reading set for
/// it and nothing else, so that what the view shows is what the scan
/// sees. The reaches are in the pixels of the image set for, and a copy
/// of it made larger or smaller for a look takes them scaled with it
/// ([`Binarizer::scaled`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Binarizer {
    /// The three readings pooled: the scan as it is, against the
    /// lighting around each pixel within a sixth of the outline, against
    /// one threshold for the outline, and against the windows at each of
    /// [`WINDOW_RADII`].
    Pooled,
    /// Each pixel against the lighting around it alone: the mean of the
    /// blocks within `radius` pixels ([`local_dark`]).
    Local { radius: f64 },
    /// One threshold alone, Otsu's split of the luminance of what is
    /// scanned.
    Global,
    /// Each pixel against the mean of the window `radius` pixels either
    /// side of it alone ([`window_dark`]).
    Window { radius: f64 },
}

impl Binarizer {
    /// Whether one reading is set in place of the scan's three.
    pub fn is_set(self) -> bool {
        self != Self::Pooled
    }

    /// The same reading for a copy of the image `factor` times as large:
    /// each reach `factor` times as long, so that it reaches as far
    /// across the copy as it was set to reach across the image.
    pub fn scaled(self, factor: f64) -> Self {
        match self {
            Self::Local { radius } => Self::Local { radius: radius * factor },
            Self::Window { radius } => Self::Window { radius: radius * factor },
            other => other,
        }
    }
}

impl fmt::Display for Binarizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pooled => write!(f, "its three readings pooled"),
            Self::Local { radius } => write!(f, "the local threshold, reaching {radius:.0} px about each pixel"),
            Self::Global => write!(f, "one threshold for the whole of what it scans"),
            Self::Window { radius } => write!(f, "a window reaching {radius:.0} px either side of each pixel"),
        }
    }
}

/// Each pixel of a `width` × `height` image against the mean of the
/// window `radius` pixels either side of it, clipped to the image:
/// [`DARK`] where it lies more than `bias` below that mean, [`LIGHT`]
/// otherwise, and [`NONE`] where it has no value, `NaN` where a rectified
/// image has no photo. Pixels without a value count for nothing in the
/// mean.
pub fn window_dark(data: &[f32], width: usize, height: usize, radius: usize, bias: f32) -> Vec<u8> {
    assert_eq!(data.len(), width * height);
    scan::readings(Values::Cores(data), (width, height), &[(Rule::Window { radius, bias }, false)], false).into_only()
}

/// Settings of the view: the scan's rule as it runs for one outline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinarizeParams {
    /// How far around each pixel the blocks it is judged against reach,
    /// in pixels.
    pub radius: usize,
    /// One threshold for the region instead, Otsu's split of its
    /// luminance, the other way the scan reads the outline.
    pub global: bool,
    /// The half-width, in pixels, of the window each pixel is judged
    /// against the mean of instead ([`window_dark`]), the third way the
    /// scan reads the outline, at each of [`WINDOW_RADII`] narrower than
    /// its neighbourhood. Neither `radius` nor `global` is looked at
    /// while one is given.
    pub window: Option<usize>,
    /// The photo pixels the scan looks at, `[x0, y0, x1, y1]` with the far
    /// sides exclusive: the outline with the search's margin around it.
    /// The whole photo when `None`.
    pub region: Option<[usize; 4]>,
}

impl BinarizeParams {
    /// Smallest and largest radius offered.
    pub const MIN_RADIUS: usize = 8;
    pub const MAX_RADIUS: usize = 256;

    /// The settings the scan uses for a symbol outlined by `corners`,
    /// top-left first and clockwise, in a `width` × `height` photo: the
    /// radius a sixth of the outline's shorter side, and the region the
    /// outline with the margin the search reaches beyond it. The radius
    /// is whole pixels, taken upward: the blocks it reaches, whole
    /// blocks rounded up ([`local_dark`]), are then the blocks
    /// the scan's radius reaches, which is not rounded. Taken to the
    /// nearest pixel it fell a block short of the scan's for a side a
    /// hair over a multiple of forty-eight.
    pub fn for_outline(corners: [[f64; 2]; 4], width: usize, height: usize) -> Self {
        let distance = |a: [f64; 2], b: [f64; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
        let across = (distance(corners[0], corners[1]) + distance(corners[3], corners[2])) / 2.0;
        let down = (distance(corners[0], corners[3]) + distance(corners[1], corners[2])) / 2.0;
        let side = across.min(down);
        let radius = if side.is_finite() && side > 0.0 { local_radius_for(side).ceil() as usize } else { Self::MIN_RADIUS };
        let (x0, x1) = corners.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p[0]), a.1.max(p[0])));
        let (y0, y1) = corners.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p[1]), a.1.max(p[1])));
        let region = if [x0, x1, y0, y1].iter().all(|v| v.is_finite()) {
            let (mx, my) = (super::detect::MARGIN * (x1 - x0), super::detect::MARGIN * (y1 - y0));
            let clip = |v: f64, limit: usize| v.clamp(0.0, limit as f64) as usize;
            let region = [clip((x0 - mx).floor(), width), clip((y0 - my).floor(), height), clip((x1 + mx).ceil(), width), clip((y1 + my).ceil(), height)];
            (region[2] > region[0] && region[3] > region[1]).then_some(region)
        } else {
            None
        };
        Self { radius: radius.clamp(Self::MIN_RADIUS, Self::MAX_RADIUS), global: false, window: None, region }
    }

    /// The one reading these settings show, as the scan takes it in
    /// place of its three when it is set for the scan: the window when
    /// one is given, else the one threshold when that is asked for, else
    /// the local threshold at this radius. The region is the view's own:
    /// the scan takes the level and the floor of a reading from what it
    /// scans, which for an outline is what the view's region is.
    pub fn binarizer(&self) -> Binarizer {
        match self.window {
            Some(radius) => Binarizer::Window { radius: radius as f64 },
            None if self.global => Binarizer::Global,
            None => Binarizer::Local { radius: self.radius as f64 },
        }
    }
}

impl Default for BinarizeParams {
    /// For a symbol about four hundred pixels across, the whole photo
    /// being its region.
    fn default() -> Self {
        Self { radius: 64, global: false, window: None, region: None }
    }
}

/// The photo thresholded as the scan thresholds the outline: 0.0 where a
/// pixel reads dark, 1.0 elsewhere. Against the blocks around each pixel,
/// with the flat floor set by the region's spread as the scan sets it, or,
/// with `global`, against Otsu's split of the region's luminance, or,
/// with a `window`, against the mean of that window about each pixel,
/// less the share of the region's spread the scan takes off it.
pub fn binarize(luma: &Luma, params: &BinarizeParams) -> Luma {
    let (width, height) = (luma.width, luma.height);
    if width == 0 || height == 0 {
        return Luma::new(width, height, Vec::new());
    }
    let rule = rule_of(&histogram(&luma.data, width, height, params.region), params);
    let mask = scan::readings(Values::Cores(&luma.data), (width, height), &[(rule, false)], false).into_only();
    Luma::new(width, height, mask.into_iter().map(|kind| if kind == DARK { 0.0 } else { 1.0 }).collect())
}

/// A photo's luminance held on the graphics card for the binarised view
/// to be made from: handed the card once, and read there again as often
/// as the outline or the view's settings change, which while a corner is
/// dragged is every frame. The levels and floors of the readings are
/// counted from the cores' own luminance ([`histogram`]), so that nothing
/// made on the card is waited for until the view is shown. A luminance
/// the card does not hold whole, as an enlarged photo may be for a card
/// whose buffers are small, is read from the cores' own, in tiles.
pub struct ViewSource {
    luma: Arc<Luma>,
    /// None for a luminance the card does not hold whole
    /// ([`super::gpu::holds`]), or of no pixels, which has nothing to hand.
    image: Option<super::gpu::Image>,
}

impl ViewSource {
    /// `luma`, handed the card where it holds it whole.
    pub fn new(luma: Arc<Luma>) -> Self {
        let image = super::gpu::holds(luma.width, luma.height).then(|| super::gpu::Image::handed(&luma.data, luma.width, luma.height));
        Self { luma, image }
    }

    /// [`binarize`] of the luminance, as the binarised view shows it: a
    /// byte to a pixel, black where it reads dark and white elsewhere, made
    /// on the card and taken off it in grey.
    pub fn binarized(&self, params: &BinarizeParams) -> Vec<u8> {
        scan::readings(self.values(), (self.luma.width, self.luma.height), &[(self.rule(params), false)], true).into_only()
    }

    /// The same, as opaque pixels left on the card, for a window drawn by
    /// the card to lay into its texture ([`super::gpu::OnCard::copy_to`]),
    /// where the card holds the luminance whole; and otherwise taken off
    /// it, a byte of grey to a pixel.
    pub fn binarized_shown(&self, params: &BinarizeParams) -> Result<super::gpu::OnCard, Vec<u8>> {
        // The rule's level and floor taken from the region once, for either.
        let rule = self.rule(params);
        let shown = self.image.as_ref().and_then(|image| scan::reading_shown(image, rule));
        shown.ok_or_else(|| scan::readings(self.values(), (self.luma.width, self.luma.height), &[(rule, false)], true).into_only())
    }

    /// The luminance as the readings are made from it: where the card
    /// holds it, and otherwise the cores' own.
    fn values(&self) -> Values<'_> {
        self.image.as_ref().map_or(Values::Cores(&self.luma.data), Values::Card)
    }

    /// The one reading `params` set over the luminance ([`rule_of`]).
    fn rule(&self, params: &BinarizeParams) -> Rule {
        rule_of(&histogram(&self.luma.data, self.luma.width, self.luma.height, params.region), params)
    }
}

/// The one reading `params` set over an image whose region's values fall
/// as `counts` counts them ([`histogram`]): its level and floor taken
/// from them.
fn rule_of(counts: &[u32; BINS], params: &BinarizeParams) -> Rule {
    if let Some(radius) = params.window {
        Rule::Window { radius, bias: WINDOW_BIAS * robust_spread_of(counts) }
    } else if params.global {
        Rule::Global { threshold: otsu_of(counts) }
    } else {
        local_rule(params.radius as f64, robust_spread_of(counts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imaging::sample::robust_spread;

    fn image(width: usize, height: usize, value: impl Fn(usize, usize) -> f32) -> Luma {
        Luma::new(width, height, (0..width * height).map(|i| value(i % width, i / width)).collect())
    }

    #[test]
    fn defaults_follow_the_outline() {
        let square = |x0: f64, y0: f64, side: f64| [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]];
        let params = BinarizeParams::for_outline(square(100.0, 100.0, 300.0), 1000, 800);
        assert_eq!(params.radius, 50);
        assert!(!params.global);
        // The outline and the search's margin of fifteen per cent, clipped to the photo.
        assert_eq!(params.region, Some([55, 55, 445, 445]));
        assert_eq!(BinarizeParams::for_outline(square(-50.0, -50.0, 4000.0), 1000, 800).radius, BinarizeParams::MAX_RADIUS);
        assert_eq!(BinarizeParams::for_outline(square(-50.0, -50.0, 4000.0), 1000, 800).region, Some([0, 0, 1000, 800]));
        assert_eq!(BinarizeParams::for_outline(square(10.0, 10.0, 20.0), 1000, 800).radius, BinarizeParams::MIN_RADIUS);
        assert_eq!(BinarizeParams::for_outline([[0.0; 2]; 4], 1000, 800).region, None);
        assert_eq!(BinarizeParams::default().region, None);
        // The view reaches the blocks the scan does, whatever the size of
        // the outline: a radius taken to the nearest pixel fell a block
        // short of the scan's for a side of 49, of 97 and so on.
        for side in 48..=1536 {
            let view = BinarizeParams::for_outline(square(0.0, 0.0, side as f64), 2000, 2000).radius as f64;
            let scan = local_radius_for(side as f64);
            assert_eq!((view / BLOCK as f64).ceil(), (scan / BLOCK as f64).ceil(), "a side of {side}");
        }
    }

    #[test]
    fn the_view_names_the_one_reading_it_shows() {
        let local = BinarizeParams { radius: 40, global: false, window: None, region: None };
        assert_eq!(local.binarizer(), Binarizer::Local { radius: 40.0 });
        assert_eq!(BinarizeParams { global: true, ..local }.binarizer(), Binarizer::Global);
        // A window given is the reading, whatever else is set beside it.
        assert_eq!(BinarizeParams { global: true, window: Some(8), ..local }.binarizer(), Binarizer::Window { radius: 8.0 });
        assert!(local.binarizer().is_set() && !Binarizer::Pooled.is_set());
        // A copy twice as large has each reach twice as long, and one
        // threshold reaches as far as ever.
        assert_eq!(Binarizer::Local { radius: 40.0 }.scaled(2.0), Binarizer::Local { radius: 80.0 });
        assert_eq!(Binarizer::Window { radius: 8.0 }.scaled(0.25), Binarizer::Window { radius: 2.0 });
        assert_eq!(Binarizer::Global.scaled(3.0), Binarizer::Global);
        assert_eq!(Binarizer::Pooled.scaled(3.0), Binarizer::Pooled);
        assert_eq!(Binarizer::Local { radius: 33.3 }.to_string(), "the local threshold, reaching 33 px about each pixel");
        assert_eq!(Binarizer::Window { radius: 8.0 }.to_string(), "a window reaching 8 px either side of each pixel");
        assert_eq!(Binarizer::Global.to_string(), "one threshold for the whole of what it scans");
        assert_eq!(Binarizer::Pooled.to_string(), "its three readings pooled");
    }

    #[test]
    fn flat_fields_read_light() {
        for level in [0.0, 0.3, 1.0] {
            let out = binarize(&image(40, 30, |_, _| level), &BinarizeParams { radius: 16, global: false, window: None, region: None });
            assert!(out.data.iter().all(|&v| v == 1.0), "level {level}");
        }
        assert!(binarize(&image(0, 0, |_, _| 0.0), &BinarizeParams::default()).data.is_empty());
    }

    #[test]
    fn the_flat_floor_follows_the_spread_of_the_image() {
        // A square of ink on paper, both a twentieth as bright as they
        // would be in a well exposed photo: the whole image spans 0.06, so
        // the floor sinks below the square's spread and it reads dark. A
        // fixed floor of a tenth would read the image as flat all over.
        let inside = |x: usize, y: usize| (27..37).contains(&x) && (27..37).contains(&y);
        let dim = image(64, 64, |x, y| if inside(x, y) { 0.004 } else { 0.06 });
        let floor = |data: &[f32]| min_range(robust_spread(data.iter().copied()));
        assert!((floor(&dim.data) - NOISE_RANGE).abs() < 1e-6, "{}", floor(&dim.data));
        let out = binarize(&dim, &BinarizeParams { radius: 16, global: false, window: None, region: None });
        assert_eq!(out.data[32 * 64 + 32], 0.0);
        assert_eq!(out.data[5 * 64 + 5], 1.0);
        // On a photo that spans the scale the floor is ZXing's tenth or
        // so, and a patch that varies by less is flat.
        let bright = image(64, 64, |x, y| if inside(x, y) { 0.0 } else { 1.0 });
        assert!((floor(&bright.data) - RANGE_SHARE).abs() < 1e-6);
        // The dim square again, with a strip of bright paper down the
        // right of the photo: the whole photo spans the scale, and its
        // floor reads the square as flat, where the floor of the region
        // the scan looks at, the square's own part of the photo, reads
        // it dark, as the scan does.
        let mixed = image(64, 64, |x, y| if x >= 56 { 1.0 } else if inside(x, y) { 0.004 } else { 0.06 });
        assert!(floor(&mixed.data) > 0.1, "{}", floor(&mixed.data));
        assert_eq!(local_dark(&mixed.data, 64, 64, 16.0, None)[32 * 64 + 32], LIGHT);
        let part: Vec<f32> = (0..64).flat_map(|y| mixed.data[y * 64..y * 64 + 48].to_vec()).collect();
        assert!((floor(&part) - NOISE_RANGE).abs() < 1e-6, "{}", floor(&part));
        assert_eq!(local_dark(&mixed.data, 64, 64, 16.0, Some([0, 0, 48, 64]))[32 * 64 + 32], DARK);
        let region = BinarizeParams { radius: 16, global: false, window: None, region: Some([0, 0, 48, 64]) };
        assert_eq!(binarize(&mixed, &region).data[32 * 64 + 32], 0.0);
        assert_eq!(binarize(&mixed, &BinarizeParams { region: None, ..region }).data[32 * 64 + 32], 1.0);
    }

    #[test]
    fn a_small_dark_square_reads_dark_and_a_large_one_hollow() {
        let inside = |x: usize, y: usize| (27..37).contains(&x) && (27..37).contains(&y);
        let photo = image(64, 64, |x, y| if inside(x, y) { 0.0 } else { 1.0 });
        let out = binarize(&photo, &BinarizeParams { radius: 16, global: false, window: None, region: None });
        for y in 0..64 {
            for x in 0..64 {
                let expected = if inside(x, y) { 0.0 } else { 1.0 };
                assert_eq!(out.data[y * 64 + x], expected, "at ({x}, {y})");
            }
        }
        // A square of 120 pixels: its border is dark against the paper, but
        // its middle is flat over a block and its neighbours, and reads
        // light under the local rule, as the module doc says; the global
        // split reads it whole.
        let big = |x: usize, y: usize| (40..160).contains(&x) && (40..160).contains(&y);
        let photo = image(200, 200, |x, y| if big(x, y) { 0.0 } else { 1.0 });
        let local = binarize(&photo, &BinarizeParams { radius: 32, global: false, window: None, region: None });
        assert_eq!(local.data[44 * 200 + 100], 0.0, "just inside the edge");
        assert_eq!(local.data[100 * 200 + 100], 1.0, "the flat middle");
        let global = binarize(&photo, &BinarizeParams { radius: 32, global: true, window: None, region: None });
        assert_eq!(global.data[100 * 200 + 100], 0.0);
        assert_eq!(global.data[10 * 200 + 10], 1.0);
        // The region bounds the global split: inside a box of paper alone,
        // the split falls where it may, and the paper still reads light
        // where the square is darker than it.
        let boxed = binarize(&photo, &BinarizeParams { radius: 32, global: true, window: None, region: Some([0, 0, 30, 200]) });
        assert_eq!(boxed.data[100 * 200 + 100], 0.0);
    }

    #[test]
    fn window_means_match_a_direct_mean() {
        // A busy image with a hole in it, against the mean over the
        // window computed the slow way.
        let (w, h) = (23, 17);
        let mut photo = image(w, h, |x, y| ((x * 7 + y * 13) % 11) as f32 / 10.0);
        for i in [5 * w + 9, 5 * w + 10, 16 * w] {
            photo.data[i] = f32::NAN;
        }
        for radius in [0, 1, 3, 8, 40] {
            let mask = window_dark(&photo.data, w, h, radius, 0.05);
            for y in 0..h {
                for x in 0..w {
                    let value = photo.data[y * w + x];
                    if !value.is_finite() {
                        assert_eq!(mask[y * w + x], 2, "{radius}: ({x}, {y})");
                        continue;
                    }
                    let (mut sum, mut n) = (0.0f64, 0);
                    for ny in y.saturating_sub(radius)..(y + radius + 1).min(h) {
                        for nx in x.saturating_sub(radius)..(x + radius + 1).min(w) {
                            if photo.data[ny * w + nx].is_finite() {
                                sum += f64::from(photo.data[ny * w + nx]);
                                n += 1;
                            }
                        }
                    }
                    let margin = f64::from(value) - (sum / f64::from(n) - 0.05);
                    if margin.abs() > 1e-6 {
                        assert_eq!(mask[y * w + x], u8::from(margin < 0.0), "{radius}: ({x}, {y})");
                    }
                }
            }
        }
        assert!(window_dark(&[], 0, 0, 4, 0.0).is_empty());
    }

    #[test]
    fn a_window_keeps_the_rings_a_blur_has_run_together() {
        // A finder pattern three pixels a module on paper, blurred by
        // two thirds of a module: across its middle the dark ring sinks
        // to 0.44 and the light ring inside it rises to 0.57, so that a
        // threshold a tenth of the scale either side of the middle, or
        // the mean of the paper about the finder, reads one dark run
        // where a window two modules either side still reads dark,
        // light, dark, light, dark.
        let (module, side) = (3usize, 60usize);
        let finder = |x: usize, y: usize| {
            let (mx, my) = ((x as i64 - 18).div_euclid(module as i64), (y as i64 - 18).div_euclid(module as i64));
            let ring = (mx - 3).abs().max((my - 3).abs());
            (0..7).contains(&mx) && (0..7).contains(&my) && ring != 2
        };
        let sharp = image(side, side, |x, y| if finder(x, y) { 0.0 } else { 1.0 });
        let sigma = 2.0f64;
        let kernel: Vec<f64> = (-6..=6).map(|k| (-0.5 * (f64::from(k) / sigma).powi(2)).exp()).collect();
        let total: f64 = kernel.iter().sum();
        let pass = |from: &Luma, across: bool| {
            image(side, side, |x, y| {
                let at = |k: i64| {
                    let (nx, ny) = if across { (x as i64 + k, y as i64) } else { (x as i64, y as i64 + k) };
                    f64::from(from.data[ny.clamp(0, side as i64 - 1) as usize * side + nx.clamp(0, side as i64 - 1) as usize])
                };
                ((-6..=6).zip(&kernel).map(|(k, weight)| weight * at(k)).sum::<f64>() / total) as f32
            })
        };
        let blurred = pass(&pass(&sharp, true), false);
        let row = 18 + 3 * module + 1;
        let runs = |dark: &dyn Fn(usize) -> bool| {
            let mut runs: Vec<(bool, usize)> = Vec::new();
            for x in 0..side {
                match runs.last_mut() {
                    Some(run) if run.0 == dark(x) => run.1 += 1,
                    _ => runs.push((dark(x), 1)),
                }
            }
            runs.into_iter().filter(|run| run.0).count()
        };
        let mean = blurred.data.iter().sum::<f32>() / blurred.data.len() as f32;
        for threshold in [0.4, 0.6, mean] {
            assert_eq!(runs(&|x| blurred.data[row * side + x] < threshold), 1, "one threshold, {threshold}, loses the rings");
        }
        let mask = window_dark(&blurred.data, side, side, 2 * module, WINDOW_BIAS);
        assert_eq!(runs(&|x| mask[row * side + x] == 1), 3, "a window reads ring, stone and ring apart");
    }

    #[test]
    fn block_means_match_a_direct_mean() {
        // A busy image, so that no block is flat, against the mean over the
        // blocks within the radius computed the slow way: each block's
        // mean weighs the same, whole or clipped by the edge.
        let (w, h) = (37, 29);
        let photo = image(w, h, |x, y| ((x * 7 + y * 13) % 11) as f32 / 10.0);
        let blocks = local_dark(&photo.data, w, h, 8.0, None);
        let (bw, bh) = (w.div_ceil(BLOCK), h.div_ceil(BLOCK));
        let block_mean = |bx: usize, by: usize| {
            let (mut sum, mut n) = (0.0f64, 0);
            for y in by * BLOCK..((by + 1) * BLOCK).min(h) {
                for x in bx * BLOCK..((bx + 1) * BLOCK).min(w) {
                    sum += photo.data[y * w + x] as f64;
                    n += 1;
                }
            }
            sum / n as f64
        };
        for y in 0..h {
            for x in 0..w {
                let (bx, by) = (x / BLOCK, y / BLOCK);
                let (mut sum, mut n) = (0.0f64, 0);
                for ny in by.saturating_sub(1)..(by + 2).min(bh) {
                    for nx in bx.saturating_sub(1)..(bx + 2).min(bw) {
                        sum += block_mean(nx, ny);
                        n += 1;
                    }
                }
                let mean = (sum / n as f64) as f32;
                let value = photo.data[y * w + x];
                if (value - mean).abs() > 1e-4 {
                    assert_eq!(blocks[y * w + x] == DARK, value < mean, "at ({x}, {y})");
                }
            }
        }
    }

    #[test]
    fn the_view_shows_the_window_rule_as_the_scan_reads_it() {
        // Paper that darkens from left to right, with a square of ink in
        // the middle no darker than the paper is at its far side: one
        // threshold cannot have the square without the paper, and the
        // mean of a window a few pixels about each pixel can.
        let (side, radius) = (64, 4);
        let inked = |x: usize, y: usize| (28..36).contains(&x) && (28..36).contains(&y);
        let photo = image(side, side, |x, y| if inked(x, y) { 0.45 } else { 0.9 - 0.4 * x as f32 / side as f32 });
        let params = BinarizeParams { window: Some(radius), ..BinarizeParams::default() };
        let out = binarize(&photo, &params);
        assert_eq!(out.data[32 * side + 32], 0.0, "the ink reads dark");
        assert_eq!(out.data[32 * side + 8], 1.0, "the paper reads light where it is light");
        assert_eq!(out.data[8 * side + 60], 1.0, "and where it is as dark as the ink");
        let global = binarize(&photo, &BinarizeParams { global: true, ..BinarizeParams::default() });
        assert_eq!(global.data[8 * side + 60], 0.0, "one threshold takes the dark paper with the ink");
        // It is the scan's own mask, with the bias the scan takes off.
        let bias = WINDOW_BIAS * robust_spread(photo.data.iter().copied());
        let mask = window_dark(&photo.data, side, side, radius, bias);
        assert!(out.data.iter().zip(&mask).all(|(&value, &kind)| (value == 0.0) == (kind == 1)));
        // A window given is the rule, whatever else is set beside it.
        assert_eq!(binarize(&photo, &BinarizeParams { global: true, ..params }), out);
    }

    #[test]
    fn the_view_held_on_the_card_shows_what_binarize_makes() {
        let (side, radius) = (90, 6);
        let photo = image(side, side + 7, |x, y| ((x * 7 + y * 13) % 11) as f32 / 10.0 * (0.4 + 0.6 * x as f32 / side as f32));
        let source = ViewSource::new(Arc::new(photo.clone()));
        assert!(source.image.is_some(), "held on the card");
        // A luminance the card does not hold whole is never handed it, and
        // is read from the cores' own.
        let unheld = ViewSource { luma: Arc::new(photo.clone()), image: None };
        let region = Some([5, 9, 70, 80]);
        for params in [
            BinarizeParams { radius: 16, global: false, window: None, region },
            BinarizeParams { radius: 40, global: true, window: None, region: None },
            BinarizeParams { radius: 16, global: false, window: Some(radius), region },
        ] {
            let grey: Vec<u8> = binarize(&photo, &params).data.iter().map(|&v| if v < 0.5 { 0 } else { 255 }).collect();
            assert_eq!(source.binarized(&params), grey, "{params:?}");
            let shown = source.binarized_shown(&params).expect("held whole").read();
            assert!(shown.as_chunks::<4>().0.iter().zip(&grey).all(|(&[r, g, b, a], &grey)| [r, g, b, a] == [grey, grey, grey, 255]), "{params:?} shown");
            assert_eq!(unheld.binarized(&params), grey, "{params:?}, not held");
            assert_eq!(unheld.binarized_shown(&params).err(), Some(grey), "{params:?}, not held, shown");
        }
        let empty = ViewSource::new(Arc::new(image(0, 0, |_, _| 0.0)));
        assert!(empty.image.is_none() && empty.binarized(&BinarizeParams::default()).is_empty());
        assert_eq!(empty.binarized_shown(&BinarizeParams::default()).err(), Some(Vec::new()));
    }
}
