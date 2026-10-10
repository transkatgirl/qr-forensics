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

use super::Luma;
use super::sample::{NOISE_RANGE, otsu, robust_spread};

/// Pixels on a side of the blocks the local threshold averages.
pub const BLOCK: usize = 8;
/// Least spread of luminance within a block and its eight neighbours for
/// the block to carry any pattern at all, as a share of the spread of the
/// whole image thresholded between its darkest and lightest hundredth:
/// a tenth of the scale on a photo that spans it, as ZXing has it, and
/// never less than the noise of the sensor.
pub const RANGE_SHARE: f32 = 0.12;

/// Mean luminance around each pixel, over blocks, for a threshold that
/// follows shadows and glare, and the spread of luminance within each
/// block and its neighbours, which tells a flat patch from one with
/// pattern in it. A pixel with no value, `NaN` where a rectified image
/// has no photo, counts for nothing.
pub struct BlockMean {
    blocks_wide: usize,
    blocks_high: usize,
    /// Summed-area tables over the block means and the blocks with a value.
    sums: Vec<f64>,
    counts: Vec<f64>,
    /// The neighbourhood's half-width, in blocks.
    radius: usize,
    /// Per block: the spread between the darkest and lightest pixel of the
    /// block and its eight neighbours.
    range: Vec<f32>,
    /// The spread below which a block is flat: [`RANGE_SHARE`] of the
    /// image's own spread, or the sensor's noise if that is more.
    min_range: f32,
}

impl BlockMean {
    /// The neighbourhood's radius, in pixels, for a symbol whose outline
    /// is `side` pixels on its shorter side: wider than a finder pattern,
    /// so the finder's own dark does not pull the threshold down over it.
    pub fn radius_for(side: f64) -> f64 {
        (side / 6.0).clamp(8.0, 256.0)
    }

    /// The block means of a `width` × `height` image, which must not be
    /// empty, with the neighbourhood reaching `radius_px` pixels, rounded
    /// up to whole blocks. The flat floor is set by the spread of the
    /// whole image.
    pub fn new(data: &[f32], width: usize, height: usize, radius_px: f64) -> Self {
        Self::within(data, width, height, radius_px, None)
    }

    /// [`BlockMean::new`] with the flat floor set by the spread of the
    /// pixels inside `region`, `[x0, y0, x1, y1]` with the far sides
    /// exclusive, when one is given: the scan sets its floor by the
    /// spread of the outline it looks at, and a view of that outline
    /// must do the same, or a dim symbol in a bright photo reads as flat
    /// in the view where the scan reads it, and the reverse.
    pub fn within(data: &[f32], width: usize, height: usize, radius_px: f64, region: Option<[usize; 4]>) -> Self {
        assert!(width > 0 && height > 0 && data.len() == width * height, "a {width} × {height} image of {} values", data.len());
        let blocks_wide = width.div_ceil(BLOCK);
        let blocks_high = height.div_ceil(BLOCK);
        let mut sums = vec![0.0f64; (blocks_wide + 1) * (blocks_high + 1)];
        let mut counts = vec![0.0f64; (blocks_wide + 1) * (blocks_high + 1)];
        let mut mins = vec![f32::INFINITY; blocks_wide * blocks_high];
        let mut maxs = vec![f32::NEG_INFINITY; blocks_wide * blocks_high];
        let stride = blocks_wide + 1;
        for by in 0..blocks_high {
            for bx in 0..blocks_wide {
                let (mut sum, mut count) = (0.0f64, 0u32);
                for y in by * BLOCK..((by + 1) * BLOCK).min(height) {
                    for x in bx * BLOCK..((bx + 1) * BLOCK).min(width) {
                        let value = data[y * width + x];
                        if value.is_finite() {
                            sum += value as f64;
                            count += 1;
                            mins[by * blocks_wide + bx] = mins[by * blocks_wide + bx].min(value);
                            maxs[by * blocks_wide + bx] = maxs[by * blocks_wide + bx].max(value);
                        }
                    }
                }
                let i = (by + 1) * stride + bx + 1;
                let mean = if count > 0 { sum / count as f64 } else { 0.0 };
                sums[i] = mean + sums[i - 1] + sums[i - stride] - sums[i - stride - 1];
                counts[i] = f64::from(u8::from(count > 0)) + counts[i - 1] + counts[i - stride] - counts[i - stride - 1];
            }
        }
        let mut range = vec![0.0f32; blocks_wide * blocks_high];
        for by in 0..blocks_high {
            for bx in 0..blocks_wide {
                let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                for ny in by.saturating_sub(1)..(by + 2).min(blocks_high) {
                    for nx in bx.saturating_sub(1)..(bx + 2).min(blocks_wide) {
                        lo = lo.min(mins[ny * blocks_wide + nx]);
                        hi = hi.max(maxs[ny * blocks_wide + nx]);
                    }
                }
                range[by * blocks_wide + bx] = if hi.is_finite() && lo.is_finite() { hi - lo } else { 0.0 };
            }
        }
        let radius = (radius_px.max(0.0) / BLOCK as f64).ceil() as usize;
        let [x0, y0, x1, y1] = region.map_or([0, 0, width, height], |r| [r[0].min(width), r[1].min(height), r[2].min(width), r[3].min(height)]);
        let spread = robust_spread((y0..y1).flat_map(|y| (x0..x1).map(move |x| data[y * width + x])).filter(|v| v.is_finite()));
        let min_range = (RANGE_SHARE * spread).max(NOISE_RANGE);
        Self { blocks_wide, blocks_high, sums, counts, radius, range, min_range }
    }

    /// The spread below which a block and its neighbours are flat.
    pub fn min_range(&self) -> f32 {
        self.min_range
    }

    /// Whether a pixel of luminance `value` at (`x`, `y`) is dark against
    /// its surroundings. A pixel in a flat patch never is.
    pub fn is_dark(&self, value: f32, x: usize, y: usize) -> bool {
        let (bx, by) = ((x / BLOCK).min(self.blocks_wide - 1), (y / BLOCK).min(self.blocks_high - 1));
        if self.range[by * self.blocks_wide + bx] < self.min_range {
            return false;
        }
        let stride = self.blocks_wide + 1;
        let (x0, y0) = (bx.saturating_sub(self.radius), by.saturating_sub(self.radius));
        let (x1, y1) = ((bx + self.radius + 1).min(self.blocks_wide), (by + self.radius + 1).min(self.blocks_high));
        let sum = |table: &[f64]| {
            table[y1 * stride + x1] - table[y0 * stride + x1] - table[y1 * stride + x0] + table[y0 * stride + x0]
        };
        let n = sum(&self.counts);
        n > 0.5 && value < (sum(&self.sums) / n) as f32
    }
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
    /// outline with the margin the search reaches beyond it.
    pub fn for_outline(corners: [[f64; 2]; 4], width: usize, height: usize) -> Self {
        let distance = |a: [f64; 2], b: [f64; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
        let across = (distance(corners[0], corners[1]) + distance(corners[3], corners[2])) / 2.0;
        let down = (distance(corners[0], corners[3]) + distance(corners[1], corners[2])) / 2.0;
        let side = across.min(down);
        let radius = if side.is_finite() && side > 0.0 { BlockMean::radius_for(side).round() as usize } else { Self::MIN_RADIUS };
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
        Self { radius: radius.clamp(Self::MIN_RADIUS, Self::MAX_RADIUS), global: false, region }
    }
}

impl Default for BinarizeParams {
    /// For a symbol about four hundred pixels across, the whole photo
    /// being its region.
    fn default() -> Self {
        Self { radius: 64, global: false, region: None }
    }
}

/// The photo thresholded as the scan thresholds the outline: 0.0 where a
/// pixel reads dark, 1.0 elsewhere. Against the blocks around each pixel,
/// with the flat floor set by the region's spread as the scan sets it, or,
/// with `global`, against Otsu's split of the region's luminance.
pub fn binarize(luma: &Luma, params: &BinarizeParams) -> Luma {
    let (width, height) = (luma.width, luma.height);
    if width == 0 || height == 0 {
        return Luma::new(width, height, Vec::new());
    }
    let [x0, y0, x1, y1] = params.region.map_or([0, 0, width, height], |r| [r[0].min(width), r[1].min(height), r[2].min(width), r[3].min(height)]);
    let dark: Vec<bool> = if params.global {
        let threshold = otsu((y0..y1).flat_map(|y| (x0..x1).map(move |x| luma.data[y * width + x])));
        luma.data.iter().map(|&v| v < threshold).collect()
    } else {
        let blocks = BlockMean::within(&luma.data, width, height, params.radius as f64, params.region);
        (0..height).flat_map(|y| (0..width).map(move |x| (x, y))).map(|(x, y)| blocks.is_dark(luma.data[y * width + x], x, y)).collect()
    };
    Luma::new(width, height, dark.into_iter().map(|d| if d { 0.0 } else { 1.0 }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }

    #[test]
    fn flat_fields_read_light() {
        for level in [0.0, 0.3, 1.0] {
            let out = binarize(&image(40, 30, |_, _| level), &BinarizeParams { radius: 16, global: false, region: None });
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
        let blocks = BlockMean::new(&dim.data, 64, 64, 16.0);
        assert!((blocks.min_range() - NOISE_RANGE).abs() < 1e-6, "{}", blocks.min_range());
        let out = binarize(&dim, &BinarizeParams { radius: 16, global: false, region: None });
        assert_eq!(out.data[32 * 64 + 32], 0.0);
        assert_eq!(out.data[5 * 64 + 5], 1.0);
        // On a photo that spans the scale the floor is ZXing's tenth or
        // so, and a patch that varies by less is flat.
        let bright = image(64, 64, |x, y| if inside(x, y) { 0.0 } else { 1.0 });
        assert!((BlockMean::new(&bright.data, 64, 64, 16.0).min_range() - RANGE_SHARE).abs() < 1e-6);
        // The dim square again, with a strip of bright paper down the
        // right of the photo: the whole photo spans the scale, and its
        // floor reads the square as flat, where the floor of the region
        // the scan looks at, the square's own part of the photo, reads
        // it dark, as the scan does.
        let mixed = image(64, 64, |x, y| if x >= 56 { 1.0 } else if inside(x, y) { 0.004 } else { 0.06 });
        let whole = BlockMean::new(&mixed.data, 64, 64, 16.0);
        assert!(whole.min_range() > 0.1, "{}", whole.min_range());
        assert!(!whole.is_dark(0.004, 32, 32));
        let part = BlockMean::within(&mixed.data, 64, 64, 16.0, Some([0, 0, 48, 64]));
        assert!((part.min_range() - NOISE_RANGE).abs() < 1e-6, "{}", part.min_range());
        assert!(part.is_dark(0.004, 32, 32));
        let region = BinarizeParams { radius: 16, global: false, region: Some([0, 0, 48, 64]) };
        assert_eq!(binarize(&mixed, &region).data[32 * 64 + 32], 0.0);
        assert_eq!(binarize(&mixed, &BinarizeParams { region: None, ..region }).data[32 * 64 + 32], 1.0);
    }

    #[test]
    fn a_small_dark_square_reads_dark_and_a_large_one_hollow() {
        let inside = |x: usize, y: usize| (27..37).contains(&x) && (27..37).contains(&y);
        let photo = image(64, 64, |x, y| if inside(x, y) { 0.0 } else { 1.0 });
        let out = binarize(&photo, &BinarizeParams { radius: 16, global: false, region: None });
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
        let local = binarize(&photo, &BinarizeParams { radius: 32, global: false, region: None });
        assert_eq!(local.data[44 * 200 + 100], 0.0, "just inside the edge");
        assert_eq!(local.data[100 * 200 + 100], 1.0, "the flat middle");
        let global = binarize(&photo, &BinarizeParams { radius: 32, global: true, region: None });
        assert_eq!(global.data[100 * 200 + 100], 0.0);
        assert_eq!(global.data[10 * 200 + 10], 1.0);
        // The region bounds the global split: inside a box of paper alone,
        // the split falls where it may, and the paper still reads light
        // where the square is darker than it.
        let boxed = binarize(&photo, &BinarizeParams { radius: 32, global: true, region: Some([0, 0, 30, 200]) });
        assert_eq!(boxed.data[100 * 200 + 100], 0.0);
    }

    #[test]
    fn block_means_match_a_direct_mean() {
        // A busy image, so that no block is flat, against the mean over the
        // blocks within the radius computed the slow way: each block's
        // mean weighs the same, whole or clipped by the edge.
        let (w, h) = (37, 29);
        let photo = image(w, h, |x, y| ((x * 7 + y * 13) % 11) as f32 / 10.0);
        let blocks = BlockMean::new(&photo.data, w, h, 8.0);
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
                    assert_eq!(blocks.is_dark(value, x, y), value < mean, "at ({x}, {y})");
                }
            }
        }
    }
}
