//! Getting modules out of a photograph: finding the symbol, perspective
//! mapping, sampling and thresholding.

pub mod adjust;
pub mod binarize;
pub mod detect;
pub mod enhance;
pub mod homography;
pub mod photo;
pub mod projection;
pub mod pure;
pub mod read;
pub mod refine;
pub(crate) mod regions;
pub mod resize;
pub mod sample;

use std::sync::Arc;

/// Single-channel image, row-major, 0.0 black to 1.0 white.
#[derive(Clone, Debug, PartialEq)]
pub struct Luma {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
    /// The image at its own size, when this is a photo enlarged for the
    /// pipeline ([`adjust::Enlarge`]): a whole number of times smaller
    /// each way. The enlargement smooths every edge and rounds every
    /// corner, and a symbol rendered pixel for pixel is read off the
    /// pixels it had ([`pure`]), which are these.
    pub unenlarged: Option<Arc<Luma>>,
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
        Self { width, height, data, unenlarged: None }
    }

    /// How many copies the size of this image may be worked side by
    /// side: as many as come to [`PIXELS_AT_ONCE`], and one at least.
    pub(crate) fn copies_at_once(&self) -> usize {
        (PIXELS_AT_ONCE / (self.width * self.height).max(1)).max(1)
    }

    /// Bilinear sample at pixel coordinates, where (0.5, 0.5) is the centre
    /// of the top-left pixel. `None` outside the image.
    ///
    /// Nearly every sample a search takes is one of a template's, well
    /// inside the photo, and the fitting of templates is nearly all the
    /// time a search takes: such a sample is read here, in line, with
    /// the arithmetic of [`Self::sample_by_border`] and none of its
    /// care for the border, where nothing needs clamping.
    #[inline]
    pub fn sample(&self, x: f64, y: f64) -> Option<f32> {
        let (fx, fy) = (x - 0.5, y - 0.5);
        if fx >= 0.0 && fy >= 0.0 && fx < self.width as f64 - 1.0 && fy < self.height as f64 - 1.0 {
            let (x0, y0) = (fx as usize, fy as usize);
            let (tx, ty) = ((fx - x0 as f64) as f32, (fy - y0 as f64) as f32);
            let first = y0 * self.width + x0;
            let (above, below) = (&self.data[first..first + 2], &self.data[first + self.width..first + self.width + 2]);
            let top = above[0] * (1.0 - tx) + above[1] * tx;
            let bottom = below[0] * (1.0 - tx) + below[1] * tx;
            return Some(top * (1.0 - ty) + bottom * ty);
        }
        self.sample_by_border(x, y)
    }

    /// [`Self::sample`] for a point within half a pixel of the image's
    /// border, or off the image.
    fn sample_by_border(&self, x: f64, y: f64) -> Option<f32> {
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
    /// is read.
    pub fn to_rgba(&self) -> Vec<u8> {
        let mut rgba = Vec::with_capacity(self.data.len() * 4);
        for &value in &self.data {
            let grey = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
            rgba.extend_from_slice(&[grey, grey, grey, 255]);
        }
        rgba
    }

    /// The image turned clockwise by `quarters` quarter turns, pixel for
    /// pixel, and the image at its own size turned with it.
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
        out
    }
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
    fn the_larger_the_photo_the_fewer_copies_of_it_at_once() {
        let of = |width: usize, height: usize| Luma { width, height, data: Vec::new(), unenlarged: None }.copies_at_once();
        assert!(of(1280, 960) >= 8, "a small photo has every rung of the ladder at once");
        assert_eq!(of(4032, 3024), 3);
        assert_eq!(of(8064, 6048), 1);
        assert_eq!(of(20_000, 20_000), 1, "never none");
        assert_eq!(of(2, 3), 8_000_000);
    }
}
