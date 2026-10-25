//! Getting modules out of a photograph: finding the symbol, perspective
//! mapping, sampling and thresholding.

pub mod adjust;
pub mod binarize;
pub mod deblur;
pub mod detect;
pub mod enhance;
pub mod gpu;
pub mod homography;
pub mod measure;
pub mod patches;
pub mod photo;
pub mod projection;
pub mod pure;
pub mod read;
pub mod refine;
pub(crate) mod regions;
pub mod resize;
pub mod sample;

use std::fmt;
use std::ops::Deref;
use std::panic::RefUnwindSafe;
use std::sync::{Arc, Mutex};

/// RGBA bytes held once and shared: a photo as loaded, by the window
/// that shows it and by the luminance made of it, which keeps it beside
/// itself ([`AsShot`]) and would otherwise hold it a second time. Safe
/// to hand across a panic, as a [`Luma`] holding plain bytes was.
pub type SharedRgba = Arc<dyn Deref<Target = [u8]> + Send + Sync + RefUnwindSafe>;

/// Single-channel image, row-major, 0.0 black to 1.0 white.
#[derive(Clone, Debug, PartialEq)]
pub struct Luma {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
    /// The image at its own size, when this is a photo enlarged for the
    /// pipeline ([`adjust::Enlarge`]): a whole number of times smaller
    /// each way, and made as the photo would be at its own size, under
    /// the same adjustments in the same order but the filter, whose
    /// neighbourhood would be that many times too wide there. The enlargement
    /// smooths every edge and rounds every corner, and a symbol rendered
    /// pixel for pixel is read off the pixels it had ([`pure`]), which
    /// are these.
    pub unenlarged: Option<Arc<Luma>>,
    /// The photo as loaded, with the adjustments the image was made
    /// under, when they are more than a whole number of times its size
    /// ([`adjust::Adjust`]): what a copy of the image made otherwise is
    /// made from ([`enhance::Enhanced::window`]), by the same steps in
    /// the same order, so that the photo so adjusted shows the copy
    /// again.
    pub as_shot: Option<Arc<AsShot>>,
    /// The blur the photo's Deblur filter undid, in the image's pixels
    /// and before its strength ([`adjust::Adjust::strength`]): the one
    /// set for it, or the one measured over the image; `None` where it
    /// measured none, and left the image as it was.
    pub deblurred: Option<deblur::Spread>,
}

/// The photo as loaded about the part of it an image is of, in its
/// colours, and the adjustments the image was made under.
#[derive(Clone)]
pub struct AsShot {
    /// RGBA, `width` by `height`, beginning at `at` in the photo as
    /// loaded, which is `photo` pixels across and down: the part, and as
    /// much about it as straightening the part draws on. The photo's own
    /// bytes where that is the whole of what they were made from and they
    /// were handed over shared ([`adjust::Adjust::apply_to_shared`]).
    pub rgba: SharedRgba,
    pub width: usize,
    pub height: usize,
    pub at: [usize; 2],
    pub photo: [usize; 2],
    /// The part of the photo the image is of: left, top, width and
    /// height.
    pub part: [usize; 4],
    /// The adjustments, with the enlargement the part's size held them
    /// to.
    pub adjust: adjust::Adjust,
    /// Quarter turns clockwise the image has been turned since it was
    /// made ([`Luma::rotated_cw`]); the rest is of the image as it was
    /// made, and lies as it did. A copy of the image is made again as
    /// the image lay and turned after ([`enhance::Enhanced::window`]), so
    /// that it is straightened, reduced and deblurred among the pixels
    /// the image was, and not among those of the photo turned, which lie
    /// otherwise against its edges: a deblurred image turned, and a copy
    /// of it made again from the photo turned, once differed by more
    /// than a quarter of the way from black to white along their edges.
    pub turned: u8,
}

/// The same pixels of the same photo under the same adjustments, whether
/// held apart or shared.
impl PartialEq for AsShot {
    fn eq(&self, other: &Self) -> bool {
        self.rgba[..] == other.rgba[..]
            && (self.width, self.height, self.at, self.photo, self.part) == (other.width, other.height, other.at, other.photo, other.part)
            && (self.adjust, self.turned) == (other.adjust, other.turned)
    }
}

/// All but the bytes, of which their count.
impl fmt::Debug for AsShot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AsShot")
            .field("rgba", &format_args!("{} bytes", self.rgba.len()))
            .field("width", &self.width)
            .field("height", &self.height)
            .field("at", &self.at)
            .field("photo", &self.photo)
            .field("part", &self.part)
            .field("adjust", &self.adjust)
            .field("turned", &self.turned)
            .finish()
    }
}

/// The pixels that copies of a photo worked side by side may come to
/// between them. A photo is searched and read through copies of itself,
/// closed, thresholded or turned, and with a core to spare for each a
/// large photo would have them all in memory at once: four bytes a pixel
/// for each copy, and as much again for what a scan makes of it.
const PIXELS_AT_ONCE: usize = 48_000_000;

impl Luma {
    pub fn new(width: usize, height: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), width * height);
        Self { width, height, data, unenlarged: None, as_shot: None, deblurred: None }
    }

    /// How many copies of this image may be worked side by side: as many
    /// as come to [`PIXELS_AT_ONCE`], and one at least. A copy turned
    /// holds the photo at its own size turned too. A copy made again from
    /// the photo as loaded ([`Luma::as_shot`]) is worked in its colours at
    /// its own size before it is toned, four bytes to the pixel as the
    /// image's own are, and counts twice; the photo as loaded it is made
    /// from is held once, shared by every copy and every copy turned, and
    /// counts against none of them. Counted by its bytes, a photo of
    /// forty-eight megapixels shown at a quarter of its size was worked a
    /// copy at a time.
    pub(crate) fn copies_at_once(&self) -> usize {
        let pixels = self.width * self.height;
        let own = self.unenlarged.as_ref().map_or(0, |own| own.width * own.height);
        let worked = if self.as_shot.is_some() { pixels } else { 0 };
        (PIXELS_AT_ONCE / (pixels + own + worked).max(1)).max(1)
    }

    /// Bilinear sample at pixel coordinates, where (0.5, 0.5) is the centre
    /// of the top-left pixel. `None` outside the image.
    ///
    /// Nearly every sample a search takes is one of a template's, well
    /// inside the photo, and the fitting of templates is nearly all the
    /// time a search takes: such a sample is read here, in line, with
    /// the arithmetic of [`Self::sample_by_border`] and none of its
    /// care for the border, where nothing needs clamping. The two rows
    /// the sample lies between are one slice, the second at the image's
    /// width into it, so that the four pixels are bounds checked once.
    ///
    /// The border's own sampling is in line as well: a call in the
    /// middle of the loops that sample, rare as it is, has them keep
    /// what they are working on in memory rather than in registers
    /// around every sample.
    #[inline]
    pub fn sample(&self, x: f64, y: f64) -> Option<f32> {
        let (fx, fy) = (x - 0.5, y - 0.5);
        if fx >= 0.0 && fy >= 0.0 && fx < self.width as f64 - 1.0 && fy < self.height as f64 - 1.0 {
            let (x0, y0) = (fx as usize, fy as usize);
            let (tx, ty) = ((fx - x0 as f64) as f32, (fy - y0 as f64) as f32);
            let first = y0 * self.width + x0;
            let pixels = &self.data[first..first + self.width + 2];
            let (above, below) = (&pixels[..2], &pixels[self.width..]);
            let top = above[0] * (1.0 - tx) + above[1] * tx;
            let bottom = below[0] * (1.0 - tx) + below[1] * tx;
            return Some(top * (1.0 - ty) + bottom * ty);
        }
        self.sample_by_border(x, y)
    }

    /// [`Self::sample`] for a point within half a pixel of the image's
    /// border, or off the image.
    #[inline]
    fn sample_by_border(&self, x: f64, y: f64) -> Option<f32> {
        // An image with no pixels has no border to be within: the clamps
        // below would be asked for a range that ends before it begins.
        if self.width == 0 || self.height == 0 {
            return None;
        }
        let (fx, fy) = (x - 0.5, y - 0.5);
        if !(fx >= -0.5 && fy >= -0.5 && fx <= self.width as f64 - 0.5 && fy <= self.height as f64 - 0.5) {
            return None;
        }
        let x0 = fx.floor().clamp(0.0, self.width as f64 - 1.0) as usize;
        let y0 = fy.floor().clamp(0.0, self.height as f64 - 1.0) as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = (fx - x0 as f64).clamp(0.0, 1.0) as f32;
        let ty = (fy - y0 as f64).clamp(0.0, 1.0) as f32;
        let at = |x: usize, y: usize| self.data[y * self.width + x];
        let top = at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx;
        let bottom = at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx;
        Some(top * (1.0 - ty) + bottom * ty)
    }

    /// The image made `width` by `height`, larger or smaller, with Magic
    /// Kernel Sharp 2013: see [`resize`].
    pub fn resized(&self, width: usize, height: usize) -> Luma {
        Luma::new(width, height, resize::resize(&self.data, self.width, self.height, 1, width, height))
    }

    /// The image as opaque grey pixels, four bytes to each: what a
    /// photo shown in grey is drawn from, so that what is shown is what
    /// is read ([`greyed`]).
    pub fn to_rgba(&self) -> Vec<u8> {
        let mut rgba = vec![0u8; self.data.len() * 4];
        greyed(&self.data, &mut rgba);
        rgba
    }

    /// The image turned clockwise by `quarters` quarter turns, pixel for
    /// pixel, and the image at its own size and the blur undone turned
    /// with it; the photo as loaded is kept as it lay, with the turn
    /// ([`AsShot::turned`]).
    pub fn rotated_cw(&self, quarters: u8) -> Luma {
        let mut out = Luma::new(self.width, self.height, self.data.clone());
        for _ in 0..quarters % 4 {
            let (w, h) = (out.width, out.height);
            // Each column of the source becomes a row of the result, read
            // from the bottom of the source upward.
            let mut data = Vec::with_capacity(w * h);
            for row in 0..w {
                for col in 0..h {
                    data.push(out.data[(h - 1 - col) * w + row]);
                }
            }
            out = Luma::new(h, w, data);
        }
        out.unenlarged = self.unenlarged.as_ref().map(|small| Arc::new(small.rotated_cw(quarters)));
        out.as_shot = self.as_shot.as_ref().map(|shot| Arc::new(AsShot { turned: (shot.turned + quarters % 4) % 4, ..(**shot).clone() }));
        out.deblurred = self.deblurred.map(|spread| (0..quarters % 4).fold(spread, |spread, _| spread.turned_cw()));
        out
    }
}

/// Pixels greyed as one piece of an image spread over the cores.
const GREYED_AT_ONCE: usize = 1 << 18;

/// `luma` as opaque grey pixels, four bytes to each, written over `rgba`:
/// each value held from black to white and rounded to a byte. On the
/// cores, a piece at a time on those to spare: so, a photo of twelve
/// megapixels is greyed in well under a millisecond, where handed the
/// graphics card and taken back it took ten.
pub(crate) fn greyed(luma: &[f32], rgba: &mut [u8]) {
    assert_eq!(rgba.len(), luma.len() * 4);
    let pieces: Vec<Mutex<(&[f32], &mut [u8])>> = luma.chunks(GREYED_AT_ONCE).zip(rgba.chunks_mut(4 * GREYED_AT_ONCE)).map(Mutex::new).collect();
    crate::work::map(&pieces, |piece| {
        let mut piece = piece.lock().expect("a piece is greyed once, by one worker");
        let (values, pixels) = &mut *piece;
        for (pixel, &value) in pixels.as_chunks_mut::<4>().0.iter_mut().zip(values.iter()) {
            let grey = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
            *pixel = [grey, grey, grey, 255];
        }
    });
}

/// A point of an image `width` by `height` pixels, in the pixel
/// coordinates outlines use (the image spans `0..width` by `0..height`),
/// as it lies in the image turned clockwise by `quarters` quarter turns.
pub fn rotate_point_cw(point: [f64; 2], quarters: u8, width: usize, height: usize) -> [f64; 2] {
    let (mut p, mut dims) = (point, [width as f64, height as f64]);
    for _ in 0..quarters % 4 {
        // The top-left corner goes to the top right, so x becomes y and y
        // counts from the far edge.
        p = [dims[1] - p[1], p[0]];
        dims.swap(0, 1);
    }
    p
}

/// The inverse of [`rotate_point_cw`]: a point of the turned image back in
/// the image it was turned from, which is `width` by `height` pixels.
pub fn unrotate_point_cw(point: [f64; 2], quarters: u8, width: usize, height: usize) -> [f64; 2] {
    let (turned_w, turned_h) = if quarters % 2 == 1 { (height, width) } else { (width, height) };
    rotate_point_cw(point, (4 - quarters % 4) % 4, turned_w, turned_h)
}

/// An outline turned with its image: the corners mapped into the turned
/// image and put back in the order a hand would draw them there, top-left
/// first. A clockwise turn carries the old bottom-left corner to the top
/// left, so the order shifts by one place a turn.
pub fn rotate_outline_cw(outline: [[f64; 2]; 4], quarters: u8, width: usize, height: usize) -> [[f64; 2]; 4] {
    let mut out = outline.map(|p| rotate_point_cw(p, quarters, width, height));
    out.rotate_right(usize::from(quarters % 4));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turns_come_back_round() {
        let image = Luma::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let once = image.rotated_cw(1);
        assert_eq!((once.width, once.height), (3, 2));
        assert_eq!(once.data, vec![5.0, 3.0, 1.0, 6.0, 4.0, 2.0]);
        assert_eq!(image.rotated_cw(4), image);
        assert_eq!(once.rotated_cw(3), image);
        assert_eq!(Luma::new(3, 1, vec![0.0, 0.5, 1.2]).to_rgba(), [0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255]);
        // An enlarged image turns with the image it was enlarged from.
        let small = Luma::new(1, 2, vec![0.25, 0.75]);
        let large = Luma { unenlarged: Some(Arc::new(small.clone())), ..Luma::new(2, 4, vec![0.5; 8]) };
        assert_eq!(large.rotated_cw(1).unenlarged.as_deref(), Some(&small.rotated_cw(1)));
        assert_eq!(large.rotated_cw(4), large);
        for quarters in 0..4u8 {
            for p in [[0.0, 0.0], [2.0, 3.0], [0.7, 2.1]] {
                let turned = rotate_point_cw(p, quarters, 2, 3);
                let back = unrotate_point_cw(turned, quarters, 2, 3);
                assert!((back[0] - p[0]).abs() < 1e-12 && (back[1] - p[1]).abs() < 1e-12, "{quarters}: {p:?} -> {turned:?} -> {back:?}");
            }
        }
        let frame = [[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]];
        assert_eq!(rotate_outline_cw(frame, 1, 2, 3), [[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]]);
        assert_eq!(rotate_outline_cw(frame, 2, 2, 3), frame);
    }

    #[test]
    fn sampling_stays_inside_the_image() {
        let image = Luma::new(2, 3, vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        // Pixel centres read the pixel; the border reads the edge pixel.
        assert_eq!(image.sample(0.5, 0.5), Some(0.0));
        assert_eq!(image.sample(1.5, 0.5), Some(1.0));
        assert_eq!(image.sample(0.0, 0.0), Some(0.0));
        assert_eq!(image.sample(2.0, 3.0), Some(1.0));
        assert_eq!(image.sample(-0.1, 1.0), None);
        assert_eq!(image.sample(2.1, 1.0), None);
        assert_eq!(image.sample(f64::NAN, 1.0), None);
        // An image with no pixels has nothing to read anywhere, its own
        // corner included.
        for (width, height) in [(0, 0), (0, 3), (3, 0)] {
            let empty = Luma::new(width, height, Vec::new());
            for point in [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.5, 0.5]] {
                assert_eq!(empty.sample(point[0], point[1]), None, "{width}x{height} at {point:?}");
            }
        }
    }

    #[test]
    fn the_larger_the_photo_the_fewer_copies_of_it_at_once() {
        let of = |width: usize, height: usize| Luma { width, height, data: Vec::new(), unenlarged: None, as_shot: None, deblurred: None }.copies_at_once();
        assert!(of(1280, 960) >= 8, "a small photo has every rung of the ladder at once");
        assert_eq!(of(4032, 3024), 3);
        assert_eq!(of(8064, 6048), 1);
        assert_eq!(of(20_000, 20_000), 1, "never none");
        assert_eq!(of(2, 3), 8_000_000);
        // A copy of a photo adjusted is made again from the photo as
        // loaded, worked in its colours at its own size, and counts twice;
        // the photo as loaded, shared by every copy, counts against none,
        // however much larger than the image it is. Enlarged, the image
        // keeps its own pixels as well, which a copy turned turns.
        let adjusted = |(width, height): (usize, usize), (w, h): (usize, usize), enlarged: bool| {
            let own = enlarged.then(|| Arc::new(Luma::new(w, h, vec![0.0; w * h])));
            let shot: SharedRgba = Arc::new(vec![0u8; w * h * 4]);
            let as_shot = AsShot { rgba: shot, width: w, height: h, at: [0, 0], photo: [w, h], part: [0, 0, w, h], adjust: adjust::Adjust::default(), turned: 0 };
            Luma { width, height, data: Vec::new(), unenlarged: own, as_shot: Some(Arc::new(as_shot)), deblurred: None }.copies_at_once()
        };
        assert_eq!(adjusted((4032, 3024), (4032, 3024), false), 1);
        assert_eq!(adjusted((2000, 1500), (2000, 1500), false), 8);
        assert_eq!(adjusted((2000, 1500), (1000, 750), true), 7);
        // A photo of forty-eight megapixels shown at a quarter of its
        // size: as many at once as one of three megapixels adjusted at its
        // own. Counted by the bytes kept of it, it was one at a time.
        assert_eq!(adjusted((2000, 1500), (8000, 6000), false), 8);
    }
}
