//! The photo about a symbol enlarged and sharpened, for modules too
//! small or too blurred to read as they lie.
//!
//! A symbol a pixel or two a module, or one a lens has blurred by half a
//! module and more, keeps its modules only as ripples of grey: a dark
//! module between two light ones never reaches black, and the rings of a
//! finder pattern run into its stone. wechat_qrcode reads a small crop
//! at its own size first and then at twice it, through a network that
//! is, by its size and its layers, a sharpening interpolator: a few
//! thousand weights that see seven or eight pixels of the crop at once
//! and add what they make of them to the crop enlarged pixel for pixel.
//! The same is done here with no network. The photo is sharpened by an
//! unsharp mask, which puts back a share of what a blur of a pixel or
//! two would take from it, and then enlarged with Magic Kernel Sharp
//! 2013 ([`super::resize`]), which lays the grey of each edge where the
//! edge runs between the pixels rather than on the pixels' own grid.
//!
//! Over photographs made of symbols at one to one and a half pixels a
//! module, half of them blurred by seven tenths of a pixel, a search
//! read 51 of 72 without these copies and 58 with them, where
//! wechat_qrcode read 48, and 30 without its network. Over symbols of
//! two and three pixels a module blurred by half to seven tenths of a
//! module it read 46 of 72 without and all 72 with, where wechat_qrcode
//! read 22. Both the scan for finder patterns and the reading of the
//! modules gain. Through the true corners of the symbols a pixel and a
//! quarter a module, blurred, three modules in a hundred read wrong off
//! the photo as it lay and six more could not be called; off the copy
//! sharpened lightly, three in a thousand and two in a hundred. Through
//! those of the symbols three pixels a module, blurred by seven tenths
//! of a module, nine in a hundred read wrong and ten could not be
//! called; off the copy sharpened strongly, one and a half and four and
//! a half. The light mask reads the smallest modules best and the
//! strong one the most blurred, so a read tries both.
//!
//! The mask is laid on the photo before it is enlarged, at the photo's
//! own size, where a pixel or two is three taps or seven: laid on the
//! enlargement it would be that many times as wide and as dear, and
//! the two come to the same picture, since a blur of the pixels and the
//! same blur of what is drawn between them differ only in what the
//! enlargement cannot hold. The mask reads the photo past the edges of
//! the part that is enlarged, so that a symbol reads the same through
//! whatever part was cut about it.
//!
//! A photo enlarged already ([`super::adjust::Enlarge`]) is sharpened
//! and enlarged from the pixels it had ([`Luma::unenlarged`]), in one
//! step, and comes out in the pixels it has now.
//!
//! The copies are a read's own, made about one symbol when the photo as
//! it lies does not decode. The same mask and the same kernel are an
//! adjustment of the whole photo besides, in colour
//! ([`super::adjust::Adjust::sharpen`] and its `enlarge`), and what a
//! read saw through a copy the photo so adjusted shows again.

use std::fmt;

use super::Luma;
use super::projection::Project;
use super::resize::Sample;
use crate::work;

/// Longest side a copy of the photo made for one look at it may have.
pub(crate) const MAX_VIEW_SIDE: usize = 4096;

/// How much the photo is sharpened before a symbol is read off it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Sharpen {
    #[default]
    Off,
    /// For modules a pixel or two across, which the lens and the sensor
    /// have run together.
    Light,
    /// For a symbol out of focus, blurred by half a module and more.
    Strong,
}

impl Sharpen {
    /// Where this lies between no sharpening, 0.0, and the most, 1.0:
    /// what the photo's Sharpen adjustment is set to for the same mask
    /// ([`mask_for`]).
    pub fn amount(self) -> f32 {
        match self {
            Sharpen::Off => 0.0,
            Sharpen::Light => 0.5,
            Sharpen::Strong => 1.0,
        }
    }

    /// The mask: the width of its blur, as the standard deviation of a
    /// Gaussian in the photo's own pixels, and how many times over what
    /// the blur takes is added back.
    fn mask(self) -> Option<(f64, f32)> {
        mask_for(self.amount())
    }
}

/// The unsharp mask a sharpening of `amount` lays, from 0.0 for none to
/// 1.0 for the most: the width of its blur, as the standard deviation
/// of a Gaussian in pixels, and how many times over what the blur takes
/// is added back. Half way is the light mask, a pixel wide and one and
/// a half times over, and 1.0 the strong one, two pixels wide and two
/// and a half times over; between them, and below the light one down
/// to a mask half a pixel wide that adds nothing back, the two run
/// straight. `None` for no sharpening at all.
pub fn mask_for(amount: f32) -> Option<(f64, f32)> {
    let amount = amount.clamp(0.0, 1.0);
    if amount.is_nan() || amount <= 0.0 {
        return None;
    }
    Some(if amount <= 0.5 {
        let t = amount / 0.5;
        (0.5 + 0.5 * f64::from(t), 1.5 * t)
    } else {
        let t = (amount - 0.5) / 0.5;
        (1.0 + f64::from(t), 1.5 + t)
    })
}

/// What is done to the photo about a symbol before the symbol is looked
/// for or read there.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Enhance {
    /// The whole factor the photo is enlarged by, with Magic Kernel
    /// Sharp 2013; 1 leaves it at its size.
    pub enlarge: u8,
    pub sharpen: Sharpen,
}

impl Default for Enhance {
    fn default() -> Self {
        Self::NONE
    }
}

/// The side, in pixels, an outline is enlarged towards before a read
/// that failed is made again: wechat_qrcode doubles a crop under 320
/// pixels on a side.
const READ_SIDE: f64 = 320.0;

impl Enhance {
    /// The photo as it is.
    pub const NONE: Enhance = Enhance { enlarge: 1, sharpen: Sharpen::Off };
    /// The most the photo is enlarged.
    pub const MAX_ENLARGE: u8 = 4;

    pub fn is_none(self) -> bool {
        self.enlarge <= 1 && self.sharpen == Sharpen::Off
    }

    /// What a read of `region` that did not decode is made again
    /// through, in order: the photo enlarged so that the outline's
    /// shorter side comes to [`READ_SIDE`] pixels or so, sharpened
    /// lightly and then strongly. A large outline is sharpened at its
    /// size.
    pub fn rungs(region: [[f64; 2]; 4]) -> [Enhance; 2] {
        let side = |a: usize, b: usize| (region[a][0] - region[b][0]).hypot(region[a][1] - region[b][1]);
        let shorter = side(0, 1).min(side(3, 2)).min(side(0, 3)).min(side(1, 2));
        let enlarge = if shorter > 0.0 && shorter.is_finite() { (READ_SIDE / shorter).ceil().clamp(1.0, f64::from(Self::MAX_ENLARGE)) as u8 } else { 1 };
        [Sharpen::Light, Sharpen::Strong].map(|sharpen| Enhance { enlarge, sharpen })
    }
}

impl fmt::Display for Enhance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sharpened = match self.sharpen {
            Sharpen::Off => None,
            Sharpen::Light => Some("sharpened lightly"),
            Sharpen::Strong => Some("sharpened strongly"),
        };
        match (self.enlarge > 1, sharpened) {
            (true, Some(sharpened)) => write!(f, "enlarged {} times and {sharpened}", self.enlarge),
            (true, None) => write!(f, "enlarged {} times", self.enlarge),
            (false, Some(sharpened)) => write!(f, "{sharpened}"),
            (false, None) => write!(f, "as it is"),
        }
    }
}

/// A part of the photo enlarged and sharpened: what a symbol is looked
/// for and read in, and where it lies in the photo.
#[derive(Clone, Debug, PartialEq)]
pub struct Enhanced {
    pub luma: Luma,
    /// The photo pixel at the copy's origin.
    pub origin: [f64; 2],
    /// Pixels of the copy to a pixel of the photo.
    pub factor: f64,
}

/// How far beyond an outline the copy made for it reaches, as a share of
/// the outline's size: what the search reaches beyond an outline.
const MARGIN: f64 = 0.15;
/// And how far besides, in the photo's own pixels: past what the
/// enlargement draws on beyond a pixel, so that what is read inside the
/// outline does not depend on where the copy was cut.
const MARGIN_PX: f64 = 5.0;

impl Enhanced {
    /// The window `[x0, y0, x1, y1]` of the photo, far sides exclusive,
    /// as `enhance` makes it; `None` when the window is empty, or the
    /// copy would be too large to hold. The copy may begin a little
    /// before the window: at a whole pixel of a small photo's own.
    pub fn window(image: &Luma, [x0, y0, x1, y1]: [usize; 4], enhance: Enhance) -> Option<Self> {
        let (x1, y1) = (x1.min(image.width), y1.min(image.height));
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        // The pixels the photo had, each `times` of those it has now.
        let (source, times) = match image.unenlarged.as_deref() {
            Some(own) if own.width > 0 && own.height > 0 && image.width == own.width * (image.width / own.width) && image.height * own.width == own.height * image.width => {
                (own, image.width / own.width)
            }
            _ => (image, 1),
        };
        let (sx0, sy0) = (x0 / times, y0 / times);
        let (sx1, sy1) = (x1.div_ceil(times).min(source.width), y1.div_ceil(times).min(source.height));
        let (w, h) = (sx1 - sx0, sy1 - sy0);
        let enlarge = usize::from(enhance.enlarge.clamp(1, Enhance::MAX_ENLARGE));
        let (vw, vh) = (w * times * enlarge, h * times * enlarge);
        if vw.max(vh) > MAX_VIEW_SIDE {
            return None;
        }
        let part = match enhance.sharpen.mask() {
            Some((sigma, amount)) => unsharp_window(source, [sx0, sy0, sx1, sy1], sigma, amount),
            None => (sy0..sy1).flat_map(|y| source.data[y * source.width + sx0..y * source.width + sx1].iter().copied()).collect(),
        };
        let part = Luma::new(w, h, part);
        let luma = if (vw, vh) == (w, h) { Luma::new(w, h, part.data.iter().map(|v| v.clamp(0.0, 1.0)).collect()) } else { part.resized(vw, vh) };
        Some(Self { luma, origin: [(sx0 * times) as f64, (sy0 * times) as f64], factor: enlarge as f64 })
    }

    /// The photo about `outline`, with a margin, as `enhance` makes it.
    pub fn around(image: &Luma, outline: [[f64; 2]; 4], enhance: Enhance) -> Option<Self> {
        Self::window(image, Self::window_about(image, outline)?, enhance)
    }

    /// The window of the photo a copy about `outline` is made of: the
    /// upright box about the outline with a margin, within the photo.
    /// `None` for an outline that is no outline.
    pub fn window_about(image: &Luma, outline: [[f64; 2]; 4]) -> Option<[usize; 4]> {
        let (x0, x1) = outline.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p[0]), a.1.max(p[0])));
        let (y0, y1) = outline.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p[1]), a.1.max(p[1])));
        if !(x0.is_finite() && x1.is_finite() && y0.is_finite() && y1.is_finite()) {
            return None;
        }
        let times = image.unenlarged.as_ref().map_or(1, |own| image.width / own.width.max(1)).max(1) as f64;
        let (mx, my) = (MARGIN * (x1 - x0) + MARGIN_PX * times, MARGIN * (y1 - y0) + MARGIN_PX * times);
        Some([
            (x0 - mx).floor().max(0.0) as usize,
            (y0 - my).floor().max(0.0) as usize,
            (x1 + mx).ceil().clamp(0.0, image.width as f64) as usize,
            (y1 + my).ceil().clamp(0.0, image.height as f64) as usize,
        ])
    }

    /// The box of the photo the copy covers: its near corner and its
    /// far one.
    pub fn covers(&self) -> [[f64; 2]; 2] {
        [self.origin, self.to_photo([self.luma.width as f64, self.luma.height as f64])]
    }

    /// Where a point of the photo lies in the copy.
    pub fn to_view(&self, p: [f64; 2]) -> [f64; 2] {
        [(p[0] - self.origin[0]) * self.factor, (p[1] - self.origin[1]) * self.factor]
    }

    /// Where a point of the copy lies in the photo.
    pub fn to_photo(&self, p: [f64; 2]) -> [f64; 2] {
        [self.origin[0] + p[0] / self.factor, self.origin[1] + p[1] / self.factor]
    }

    /// A map from modules to the photo, as a map from modules to the
    /// copy.
    pub fn through<'a, P: Project>(&self, map: &'a P) -> Through<'a, P> {
        Through { map, origin: self.origin, factor: self.factor }
    }
}

/// A map into the photo carried into an [`Enhanced`] copy of it.
pub struct Through<'a, P: Project> {
    map: &'a P,
    origin: [f64; 2],
    factor: f64,
}

impl<P: Project> Project for Through<'_, P> {
    fn base(&self, x: f64, y: f64) -> [f64; 2] {
        let [px, py] = self.map.base(x, y);
        [(px - self.origin[0]) * self.factor, (py - self.origin[1]) * self.factor]
    }

    fn displacement(&self, x: f64, y: f64) -> [f64; 2] {
        let [dx, dy] = self.map.displacement(x, y);
        [dx * self.factor, dy * self.factor]
    }
}

/// The window `[x0, y0, x1, y1]` of `source` under an unsharp mask: each
/// pixel moved away from a Gaussian blur of the image, `sigma` pixels
/// wide, by `amount` times its difference from it. The blur reads the
/// image past the window, and past the image the pixel on its edge. The
/// values are not held to black and white: the enlargement that follows
/// is the better for the overshoot, and holds its own result to them.
fn unsharp_window(source: &Luma, [x0, y0, x1, y1]: [usize; 4], sigma: f64, amount: f32) -> Vec<f32> {
    let weights = gaussian(sigma);
    let reach = (weights.len() / 2) as i64;
    let (w, h) = (x1 - x0, y1 - y0);
    let (last_x, last_y) = (source.width as i64 - 1, source.height as i64 - 1);
    // Every row the window's blur draws on, blurred across.
    let (top, bottom) = ((y0 as i64 - reach).max(0), (y1 as i64 - 1 + reach).min(last_y));
    let mut across = Vec::with_capacity((bottom - top + 1) as usize * w);
    for y in top..=bottom {
        let line = &source.data[y as usize * source.width..(y as usize + 1) * source.width];
        for x in x0 as i64..x1 as i64 {
            across.push((-reach..=reach).zip(&weights).map(|(k, weight)| weight * line[(x + k).clamp(0, last_x) as usize]).sum::<f32>());
        }
    }
    // And those blurred down, against the pixels themselves.
    let mut out = Vec::with_capacity(w * h);
    for y in y0 as i64..y1 as i64 {
        for x in 0..w {
            let blurred: f32 = (-reach..=reach).zip(&weights).map(|(k, weight)| weight * across[((y + k).clamp(0, last_y) - top) as usize * w + x]).sum();
            let value = source.data[y as usize * source.width + x0 + x];
            out.push(value + amount * (value - blurred));
        }
    }
    out
}

/// The weights of a Gaussian blur `sigma` pixels wide, out to three
/// times that either side of the pixel, summing to one.
fn gaussian(sigma: f64) -> Vec<f32> {
    let reach = (3.0 * sigma).ceil().max(1.0) as i64;
    let raw: Vec<f64> = (-reach..=reach).map(|k| (-0.5 * (k as f64 / sigma).powi(2)).exp()).collect();
    let total: f64 = raw.iter().sum();
    raw.iter().map(|w| (w / total) as f32).collect()
}

/// Rows worked as one piece of an image spread over the cores.
const BAND: usize = 64;

/// An image `width` by `height`, of `channels` values to a pixel, under
/// an unsharp mask, as [`Enhanced::window`] lays one on a part of the
/// photo: every channel sharpened alike and apart from the others, so
/// that a photo is sharpened in its colours, and past the image the
/// pixel on its edge standing for those beyond it. `row` fills in the
/// values of one row of the image, and `emit` adds to the result what a
/// row of sharpened values comes to, which is not held to black and
/// white before it is handed over.
///
/// The image is worked a band of rows at a time, on the cores to spare,
/// each band blurring for itself the rows above and below that it draws
/// on: what is held besides the result is a few rows a core, where the
/// whole of a large photo as values to sharpen, and again blurred, would
/// be several times the photo.
fn unsharp_rows<T: Send>(
    (width, height, channels): (usize, usize, usize),
    (sigma, amount): (f64, f32),
    row: impl Fn(usize, &mut [f32]) + Sync,
    emit: impl Fn(&[f32], &mut Vec<T>) + Sync,
) -> Vec<T> {
    if width == 0 || height == 0 || channels == 0 {
        return Vec::new();
    }
    let weights = gaussian(sigma);
    let reach = weights.len() / 2;
    let stride = width * channels;
    let bands: Vec<(usize, usize)> = (0..height).step_by(BAND).map(|y| (y, (y + BAND).min(height))).collect();
    work::map(&bands, |&(y0, y1)| {
        // The rows the band's blur draws on: its own, and as many
        // either side as the mask reaches, within the image.
        let (top, bottom) = (y0.saturating_sub(reach), (y1 - 1 + reach).min(height - 1));
        let mut values = vec![0.0f32; (bottom - top + 1) * stride];
        let mut across = vec![0.0f32; values.len()];
        for ((y, line), sums) in (top..=bottom).zip(values.chunks_exact_mut(stride)).zip(across.chunks_exact_mut(stride)) {
            row(y, line);
            // Blurred across, tap by tap, so that the middle of the row,
            // where no tap falls off it, is one run of sums.
            for (k, &weight) in weights.iter().enumerate() {
                let shift = k as isize - reach as isize;
                // The pixels whose tap lies on the row, and those before
                // and after them, whose tap is the pixel on its edge.
                let lo = (-shift).clamp(0, width as isize) as usize;
                let hi = (width as isize - shift).clamp(lo as isize, width as isize) as usize;
                for x in 0..lo {
                    for c in 0..channels {
                        sums[x * channels + c] += weight * line[c];
                    }
                }
                if hi > lo {
                    let from = (lo as isize + shift) as usize * channels;
                    for (sum, &value) in sums[lo * channels..hi * channels].iter_mut().zip(&line[from..]) {
                        *sum += weight * value;
                    }
                }
                for x in hi..width {
                    for c in 0..channels {
                        sums[x * channels + c] += weight * line[(width - 1) * channels + c];
                    }
                }
            }
        }
        // And those blurred down, against the pixels themselves.
        let mut out = Vec::new();
        let mut sharp = vec![0.0f32; stride];
        for y in y0..y1 {
            sharp.fill(0.0);
            for (k, &weight) in weights.iter().enumerate() {
                let from = (y + k).saturating_sub(reach).clamp(top, bottom) - top;
                for (sum, &value) in sharp.iter_mut().zip(&across[from * stride..(from + 1) * stride]) {
                    *sum += weight * value;
                }
            }
            for (blurred, &value) in sharp.iter_mut().zip(&values[(y - top) * stride..(y - top + 1) * stride]) {
                *blurred = value + amount * (value - *blurred);
            }
            emit(&sharp, &mut out);
        }
        out
    })
    .into_iter()
    .flatten()
    .collect()
}

/// A whole image under an unsharp mask `sigma` pixels wide and `amount`
/// strong: `width` by `height` pixels of `channels` interleaved values
/// each, every channel sharpened apart from the others. The values are
/// not held to black and white: the enlargement that follows is the
/// better for the overshoot, and holds its own result to them.
pub fn unsharp(data: &[f32], width: usize, height: usize, channels: usize, sigma: f64, amount: f32) -> Vec<f32> {
    assert_eq!(data.len(), width * height * channels);
    let stride = width * channels;
    unsharp_rows(
        (width, height, channels),
        (sigma, amount),
        |y, line| line.copy_from_slice(&data[y * stride..(y + 1) * stride]),
        |sharp, out| out.extend_from_slice(sharp),
    )
}

/// Opaque pixels, four bytes to each, under an unsharp mask, in their
/// colours, as the values the bytes come to: four to a pixel, from 0.0
/// to 255.0 and past either where the mask overshoots, the fourth the
/// opacity as it was. What a photo is enlarged from, sharpened.
pub fn unsharp_values(rgba: &[u8], width: usize, height: usize, sigma: f64, amount: f32) -> Vec<f32> {
    assert_eq!(rgba.len(), width * height * 4);
    unsharp_rows(
        (width, height, 3),
        (sigma, amount),
        |y, line| colours_of_row(rgba, width, y, line),
        |sharp, out| out.extend(sharp.as_chunks::<3>().0.iter().flat_map(|&[r, g, b]| [r, g, b, 255.0])),
    )
}

/// Opaque pixels, four bytes to each, under an unsharp mask, in their
/// colours: [`unsharp_values`] held to what a byte holds, and made
/// without the whole photo ever being held as anything wider than its
/// bytes.
pub fn unsharp_rgba(rgba: &[u8], width: usize, height: usize, sigma: f64, amount: f32) -> Vec<u8> {
    assert_eq!(rgba.len(), width * height * 4);
    unsharp_rows(
        (width, height, 3),
        (sigma, amount),
        |y, line| colours_of_row(rgba, width, y, line),
        |sharp, out| out.extend(sharp.as_chunks::<3>().0.iter().flat_map(|&[r, g, b]| [r, g, b, 255.0]).map(u8::from_f32)),
    )
}

/// The colours of row `y` of opaque pixels `width` to a row, three
/// values to a pixel, as the bytes have them.
fn colours_of_row(rgba: &[u8], width: usize, y: usize, line: &mut [f32]) {
    let pixels = rgba[y * width * 4..(y + 1) * width * 4].as_chunks::<4>().0;
    for (colours, pixel) in line.as_chunks_mut::<3>().0.iter_mut().zip(pixels) {
        *colours = [pixel[0], pixel[1], pixel[2]].map(f32::from);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::imaging::homography::Homography;

    fn image(width: usize, height: usize, value: impl Fn(usize, usize) -> f32) -> Luma {
        Luma::new(width, height, (0..width * height).map(|i| value(i % width, i / width)).collect())
    }

    /// A soft edge down the middle of a photo: ink left, paper right,
    /// two grey pixels between.
    fn soft_edge(width: usize, height: usize) -> Luma {
        image(width, height, |x, _| match (x + 1).cmp(&(width / 2)) {
            std::cmp::Ordering::Less => 0.1,
            std::cmp::Ordering::Equal => 0.35,
            std::cmp::Ordering::Greater if x == width / 2 => 0.65,
            std::cmp::Ordering::Greater => 0.9,
        })
    }

    #[test]
    fn nothing_asked_is_the_photo_as_it_is() {
        assert!(Enhance::NONE.is_none() && Enhance::default().is_none());
        assert!(!Enhance { enlarge: 2, sharpen: Sharpen::Off }.is_none());
        assert!(!Enhance { enlarge: 1, sharpen: Sharpen::Light }.is_none());
        let photo = image(30, 20, |x, y| ((x * 7 + y * 13) % 11) as f32 / 10.0);
        let view = Enhanced::window(&photo, [4, 3, 19, 15], Enhance::NONE).unwrap();
        assert_eq!((view.luma.width, view.luma.height, view.origin, view.factor), (15, 12, [4.0, 3.0], 1.0));
        assert_eq!(view.luma.data[0], photo.data[3 * 30 + 4]);
        assert_eq!(view.luma.data[11 * 15 + 14], photo.data[14 * 30 + 18]);
        assert!(Enhanced::window(&photo, [4, 3, 4, 15], Enhance::NONE).is_none());
        assert!(Enhanced::window(&image(1100, 8, |_, _| 0.5), [0, 0, 1100, 8], Enhance { enlarge: 4, sharpen: Sharpen::Off }).is_none(), "too large to hold");
    }

    #[test]
    fn the_mask_steepens_an_edge_and_leaves_flat_tone_alone() {
        let photo = soft_edge(40, 12);
        for (sharpen, steeper) in [(Sharpen::Light, 0.05), (Sharpen::Strong, 0.15)] {
            let view = Enhanced::window(&photo, [0, 0, 40, 12], Enhance { enlarge: 1, sharpen }).unwrap();
            let row = &view.luma.data[5 * 40..6 * 40];
            // Far from the edge, the tone it had; across the edge, a
            // greater step between the two grey pixels; and nothing past
            // black or white.
            assert!((row[3] - 0.1).abs() < 1e-3 && (row[36] - 0.9).abs() < 1e-3, "{sharpen:?}: {row:?}");
            assert!(row[20] - row[19] > 0.3 + steeper, "{sharpen:?}: {} to {}", row[19], row[20]);
            assert!(view.luma.data.iter().all(|v| (0.0..=1.0).contains(v)));
        }
    }

    #[test]
    fn the_masks_run_from_none_through_the_light_one_to_the_strong() {
        assert_eq!(mask_for(0.0), None);
        assert_eq!(mask_for(-1.0), None);
        assert_eq!(mask_for(f32::NAN), None);
        // The two a read tries lie half way and at the end.
        assert_eq!((Sharpen::Off.amount(), Sharpen::Light.amount(), Sharpen::Strong.amount()), (0.0, 0.5, 1.0));
        assert_eq!(mask_for(0.5), Some((1.0, 1.5)));
        assert_eq!(mask_for(1.0), Some((2.0, 2.5)));
        assert_eq!(mask_for(7.0), mask_for(1.0));
        assert_eq!((Sharpen::Off.mask(), Sharpen::Light.mask(), Sharpen::Strong.mask()), (None, mask_for(0.5), mask_for(1.0)));
        // Between them each is wider and stronger than the last, and
        // the least of them adds next to nothing back.
        let masks: Vec<(f64, f32)> = (1..=20).map(|step| mask_for(step as f32 / 20.0).unwrap()).collect();
        assert!(masks.windows(2).all(|pair| pair[0].0 < pair[1].0 && pair[0].1 < pair[1].1), "{masks:?}");
        assert!(masks[0].0 > 0.5 && masks[0].1 < 0.2, "{:?}", masks[0]);
    }

    #[test]
    fn a_whole_image_is_sharpened_as_a_window_of_it_is_in_every_channel() {
        // Busy pixels, three values to each, every channel its own
        // picture: each comes out as the same picture alone does through
        // a copy of the whole of it, at any width of mask and on images
        // smaller than the mask reaches.
        for (width, height) in [(64, 48), (5, 3), (1, 7), (200, 130)] {
            let channel = |c: usize| image(width, height, |x, y| (((x * (31 + c) + y * 17) % 23) as f32 / 22.0 + ((x / 5 + y / 3 + c) % 2) as f32) / 2.0);
            let planes = [channel(0), channel(1), channel(2)];
            let interleaved: Vec<f32> = (0..width * height).flat_map(|i| [planes[0].data[i], planes[1].data[i], planes[2].data[i]]).collect();
            for (sigma, amount) in [(1.0, 1.5), (2.0, 2.5), (0.6, 0.3)] {
                let sharp = unsharp(&interleaved, width, height, 3, sigma, amount);
                assert_eq!(sharp.len(), interleaved.len());
                for (c, plane) in planes.iter().enumerate() {
                    let alone = unsharp_window(plane, [0, 0, width, height], sigma, amount);
                    for (i, want) in alone.iter().enumerate() {
                        assert!((sharp[i * 3 + c] - want).abs() < 1e-5, "{width} × {height}, {sigma}: channel {c} at {i}");
                    }
                    assert_eq!(unsharp(&plane.data, width, height, 1, sigma, amount), sharp.iter().skip(c).step_by(3).copied().collect::<Vec<f32>>());
                }
            }
        }
        // Flat tone stays flat, and the overshoot at an edge is kept.
        let edge = soft_edge(40, 12);
        let sharp = unsharp(&edge.data, 40, 12, 1, 2.0, 2.5);
        assert!((sharp[5 * 40 + 3] - 0.1).abs() < 1e-5 && (sharp[5 * 40 + 36] - 0.9).abs() < 1e-5);
        assert!(sharp.iter().any(|&v| v < 0.0) && sharp.iter().any(|&v| v > 1.0), "not held to black and white");
        assert!(unsharp(&[], 0, 0, 3, 1.0, 1.5).is_empty());
        // Opaque pixels are sharpened in their colours, the same whether
        // they come back as values or as bytes, on an image of more
        // rows than are worked at once as on one of fewer.
        for (width, height) in [(37, 150), (9, 4)] {
            let rgba: Vec<u8> = (0..width * height).flat_map(|i| [(i * 37 % 256) as u8, (i * 11 % 256) as u8, ((i / width) * 29 % 256) as u8, 255]).collect();
            let colours: Vec<f32> = rgba.as_chunks::<4>().0.iter().flat_map(|pixel| [pixel[0], pixel[1], pixel[2]]).map(f32::from).collect();
            let alone = unsharp(&colours, width, height, 3, 2.0, 2.5);
            let sharp = unsharp_values(&rgba, width, height, 2.0, 2.5);
            assert_eq!(sharp, alone.as_chunks::<3>().0.iter().flat_map(|&[r, g, b]| [r, g, b, 255.0]).collect::<Vec<f32>>());
            assert!(sharp.iter().any(|&v| v < 0.0) && sharp.iter().any(|&v| v > 255.0), "not held to what a byte holds");
            let bytes = unsharp_rgba(&rgba, width, height, 2.0, 2.5);
            assert_eq!(bytes, sharp.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect::<Vec<u8>>());
            assert!(bytes.as_chunks::<4>().0.iter().all(|pixel| pixel[3] == 255) && bytes != rgba);
        }
    }

    #[test]
    fn a_part_reads_as_the_whole_does_wherever_it_is_cut() {
        // Busy pixels, enlarged and sharpened whole and in two parts cut
        // at different places: away from the cuts, the same values.
        let photo = image(64, 48, |x, y| (((x * 31 + y * 17) % 23) as f32 / 22.0 + ((x / 5 + y / 3) % 2) as f32) / 2.0);
        let enhance = Enhance { enlarge: 3, sharpen: Sharpen::Strong };
        let whole = Enhanced::window(&photo, [0, 0, 64, 48], enhance).unwrap();
        for window in [[10, 8, 50, 40], [13, 11, 47, 37]] {
            let part = Enhanced::window(&photo, window, enhance).unwrap();
            assert_eq!(part.origin, [window[0] as f64, window[1] as f64]);
            for (px, py) in [(20.5, 16.5), (30.2, 24.9), (40.0, 30.0)] {
                let (a, b) = (whole.to_view([px, py]), part.to_view([px, py]));
                assert_eq!(part.to_photo(b), [px, py]);
                let (va, vb) = (whole.luma.sample(a[0], a[1]).unwrap(), part.luma.sample(b[0], b[1]).unwrap());
                assert!((va - vb).abs() < 1e-5, "{window:?} at ({px}, {py}): {va} against {vb}");
            }
        }
    }

    #[test]
    fn a_small_photo_is_enlarged_from_the_pixels_it_had() {
        // A photo of 20 by 12 enlarged four times when it was loaded:
        // enhanced, it comes out in the enlarged pixels, made from its
        // own in one step.
        let own = soft_edge(20, 12);
        let loaded = Luma { unenlarged: Some(Arc::new(own.clone())), ..own.resized(80, 48) };
        let enhance = Enhance { enlarge: 2, sharpen: Sharpen::Light };
        let view = Enhanced::window(&loaded, [9, 6, 70, 42], enhance).unwrap();
        // Begun at a whole pixel of its own, and twice the loaded size.
        assert_eq!((view.origin, view.factor), ([8.0, 4.0], 2.0));
        assert_eq!((view.luma.width, view.luma.height), ((18 - 2) * 8, (11 - 1) * 8));
        // A point of the loaded photo lies where it should in the copy.
        assert_eq!(view.to_view([40.0, 24.0]), [64.0, 40.0]);
        // Sharpened and no more, it is its loaded size, and what its own
        // pixels come to enlarged that many times: not the loaded
        // pixels, which were enlarged unsharpened.
        let same = Enhanced::window(&loaded, [8, 4, 72, 44], Enhance { enlarge: 1, sharpen: Sharpen::Light }).unwrap();
        let direct = Enhanced::window(&own, [2, 1, 18, 11], Enhance { enlarge: 4, sharpen: Sharpen::Light }).unwrap();
        assert_eq!((same.luma.width, same.luma.height, same.origin, same.factor), (64, 40, [8.0, 4.0], 1.0));
        assert_eq!(same.luma, direct.luma);
        assert!(same.luma.data.iter().zip(&loaded.data[4 * 80 + 8..]).any(|(a, b)| (a - b).abs() > 0.02));
    }

    #[test]
    fn a_copy_about_an_outline_carries_a_map_into_it() {
        let photo = image(200, 150, |x, y| ((x + y) % 2) as f32);
        let outline = [[60.0, 40.0], [120.0, 40.0], [120.0, 100.0], [60.0, 100.0]];
        let view = Enhanced::around(&photo, outline, Enhance { enlarge: 2, sharpen: Sharpen::Off }).unwrap();
        // Fifteen per cent of the outline and five pixels beyond it.
        assert_eq!(view.origin, [46.0, 26.0]);
        assert_eq!((view.luma.width, view.luma.height), (2 * 88, 2 * 88));
        let map = Homography::rect_to_quad(21.0, 21.0, outline).unwrap();
        let carried = view.through(&map);
        for (mx, my) in [(0.0, 0.0), (10.5, 3.25), (21.0, 21.0)] {
            assert_eq!(carried.apply(mx, my), view.to_view(Project::apply(&map, mx, my)));
        }
        assert_eq!(view.covers(), [[46.0, 26.0], [134.0, 114.0]]);
        assert_eq!(Enhanced::window_about(&photo, outline), Some([46, 26, 134, 114]));
        assert!(Enhanced::around(&photo, [[f64::NAN, 0.0]; 4], Enhance::NONE).is_none());
    }

    #[test]
    fn a_failed_read_is_made_again_larger_the_smaller_its_outline() {
        let square = |side: f64| [[10.0, 10.0], [10.0 + side, 10.0], [10.0 + side, 10.0 + side], [10.0, 10.0 + side]];
        let enlarge = |side: f64| Enhance::rungs(square(side)).map(|rung| rung.enlarge);
        assert_eq!(enlarge(40.0), [4, 4]);
        assert_eq!(enlarge(100.0), [4, 4]);
        assert_eq!(enlarge(120.0), [3, 3]);
        assert_eq!(enlarge(200.0), [2, 2]);
        assert_eq!(enlarge(320.0), [1, 1]);
        assert_eq!(enlarge(2000.0), [1, 1]);
        assert_eq!(Enhance::rungs(square(200.0)).map(|rung| rung.sharpen), [Sharpen::Light, Sharpen::Strong]);
        assert_eq!(Enhance::rungs([[0.0; 2]; 4])[0].enlarge, 1);
        assert_eq!(Enhance { enlarge: 3, sharpen: Sharpen::Light }.to_string(), "enlarged 3 times and sharpened lightly");
        assert_eq!(Enhance { enlarge: 1, sharpen: Sharpen::Strong }.to_string(), "sharpened strongly");
        assert_eq!(Enhance { enlarge: 2, sharpen: Sharpen::Off }.to_string(), "enlarged 2 times");
    }
}
