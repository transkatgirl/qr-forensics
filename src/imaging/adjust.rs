//! Tone adjustments and neighbourhood filters applied to the photo before
//! it is shown and sampled.

use super::Luma;

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
        }
    }
}

/// Where a straightened pixel at normalised radius `r` (half the diagonal
/// is 1) comes from in the photo as shot, by Brown's radial model to its
/// first term.
pub fn source_radius(r: f32, k1: f32) -> f32 {
    r * (1.0 + k1 * r * r)
}

/// The normalised radius of a straightened point whose source lies at
/// normalised radius `source` in the photo as shot: the inverse of
/// [`source_radius`], by Newton's method, which the gentle distortions a
/// lens has converge in a few steps.
fn straightened_radius(source: f64, k1: f64) -> f64 {
    if k1 == 0.0 || source == 0.0 {
        return source;
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
    let shot = r * (1.0 + f64::from(from_k1) * r * r);
    let back = straightened_radius(shot, f64::from(to_k1));
    [cx + dx * back / r, cy + dy * back / r]
}

/// Lens distortions tried first, from the widest barrel to the widest
/// pincushion the slider offers, and the step between them.
const LENS_SEARCH: (f32, f32, f32) = (-0.3, 0.3, 0.01);
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
    let k = (k * 1000.0).round() / 1000.0;
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

    /// Returns the RGBA image to display and the luminance to sample from.
    /// With a filter on, the display is the filtered luminance in grey, so
    /// that what is shown is what is sampled.
    pub fn apply(&self, rgba: &[u8], width: usize, height: usize) -> (Vec<u8>, Luma) {
        assert_eq!(rgba.len(), width * height * 4);
        // The lens comes first: the tone curve is per pixel and does not
        // care, but the filter's neighbourhoods should be of the
        // straightened photo. Nothing is resampled while there is no
        // distortion to take out.
        let straightened;
        let rgba = if self.k1 != 0.0 && width > 0 && height > 0 {
            straightened = undistort(rgba, width, height, self.k1);
            &straightened[..]
        } else {
            rgba
        };
        let table = self.lookup_table();
        let mut display = Vec::with_capacity(rgba.len());
        let mut luma = Vec::with_capacity(width * height);
        for pixel in rgba.as_chunks::<4>().0 {
            let [r, g, b] = [table[pixel[0] as usize], table[pixel[1] as usize], table[pixel[2] as usize]];
            let shown = match self.channel {
                Channel::Color => [r, g, b],
                Channel::Red => [r; 3],
                Channel::Green => [g; 3],
                Channel::Blue => [b; 3],
            };
            display.extend_from_slice(&[shown[0], shown[1], shown[2], 255]);
            let [r, g, b] = shown.map(|c| c as f32 / 255.0);
            luma.push(0.299 * r + 0.587 * g + 0.114 * b);
        }
        if self.filter != Filter::None && width > 0 && height > 0 {
            luma = self.filter.apply(&luma, width, height, self.radius.clamp(1, Filter::MAX_RADIUS) as usize);
            for (pixel, &value) in display.as_chunks_mut::<4>().0.iter_mut().zip(&luma) {
                let grey = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
                pixel[0] = grey;
                pixel[1] = grey;
                pixel[2] = grey;
            }
        }
        (display, Luma::new(width, height, luma))
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
        // Too few pairs: nothing.
        assert!(estimate_k1(w, h, 0.0, &pairs[..6]).is_none());
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
