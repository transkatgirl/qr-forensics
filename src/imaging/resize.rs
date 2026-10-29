//! Making an image larger or smaller: every copy of a photo that has
//! more or fewer pixels than the photo is made with Magic Kernel Sharp
//! 2013, John Costella's resizing kernel
//! (<https://johncostella.com/magic/>).
//!
//! The kernel is the Magic Kernel, a quadratic B-spline, convolved with
//! the three-tap sharpening step `[-1/4, 3/2, -1/4]` that undoes the
//! slight blur the spline alone leaves: five parabolas over five pixels,
//! with a zero a pixel either side of the centre. It keeps a flat tone
//! flat and a ramp a ramp, and a sharp edge comes out a pixel or so wide
//! with a faint halo either side, which a result held to black and white
//! clips.
//!
//! Enlarging, the kernel is laid among the source's pixels as it is.
//! Reducing, it is stretched by the reduction, so that it spans five of
//! the result's pixels and takes in every source pixel under them. The
//! weights a result pixel draws with are made to sum to one, and past an
//! edge of the image the pixel on the edge stands for those beyond it.
//! Pixels are tiles with their centres at the half-integers, so that the
//! result covers what the source covers: result pixel `i` is centred on
//! the source at `(i + 1/2) / ratio - 1/2`.
//!
//! The values are resized as they are given: a photo is resized as it
//! is stored, as the rest of the pipeline weighs it, or as the light it
//! stands for, where it is worked in linear light
//! ([`super::adjust::Adjust::light`]), its values converted before they
//! are handed here and stored again after ([`super::gpu::adjusted`],
//! [`super::Luma::resized`]).
//!
//! The weights are worked out here, and the resizing on the graphics
//! card ([`super::gpu::resized`]): a photo enlarged four times is sixteen
//! times its pixels, made on every change of its adjustments. Each pixel
//! of the result is made of the source alone, so a result too large for
//! the card to make at once is made a tile at a time, and comes out as
//! it would have whole.

use super::gpu::{self, Made, Samples};

/// Magic Kernel Sharp 2013 at `x` pixels from its centre.
pub fn mks2013(x: f64) -> f64 {
    let x = x.abs();
    if x <= 0.5 {
        17.0 / 16.0 - 7.0 / 4.0 * x * x
    } else if x <= 1.5 {
        (1.0 - x) * (7.0 / 4.0 - x)
    } else if x <= 2.5 {
        -(x - 2.5) * (x - 2.5) / 8.0
    } else {
        0.0
    }
}

/// How far the kernel reaches either side of its centre.
const REACH: f64 = 2.5;

/// What a pixel holds in each of its channels.
pub trait Sample: Copy + Send + Sync {
    fn to_f32(self) -> f32;
    /// The value kept within what the channel holds: beside a sharp edge
    /// the kernel's negative lobes carry it past black or white. The card
    /// keeps a value made there so too.
    fn from_f32(value: f32) -> Self;
    /// Whether the card makes samples of this kind as bytes, rounded,
    /// rather than as values.
    const BYTES: bool;
    /// The samples, as the card is handed them.
    fn handed(samples: &[Self]) -> Samples<'_>;
    /// What the card made, as samples of this kind: what it was asked to
    /// make by [`Self::BYTES`].
    fn taken(made: Made) -> Vec<Self>;
}

impl Sample for u8 {
    fn to_f32(self) -> f32 {
        f32::from(self)
    }

    fn from_f32(value: f32) -> Self {
        value.round().clamp(0.0, 255.0) as u8
    }

    const BYTES: bool = true;

    fn handed(samples: &[Self]) -> Samples<'_> {
        Samples::Bytes(samples)
    }

    fn taken(made: Made) -> Vec<Self> {
        match made {
            Made::Bytes(bytes) => bytes,
            Made::Values(_) => unreachable!("the card makes bytes where it is asked to"),
        }
    }
}

/// Luminance, 0.0 black to 1.0 white.
impl Sample for f32 {
    fn to_f32(self) -> f32 {
        self
    }

    fn from_f32(value: f32) -> Self {
        value.clamp(0.0, 1.0)
    }

    const BYTES: bool = false;

    fn handed(samples: &[Self]) -> Samples<'_> {
        Samples::Values(samples)
    }

    fn taken(made: Made) -> Vec<Self> {
        match made {
            Made::Values(values) => values,
            Made::Bytes(_) => unreachable!("the card makes values where it is asked to"),
        }
    }
}

/// What each pixel along one side of the result draws on: a run of the
/// source's pixels along that side, and a weight for each.
pub(crate) struct Taps {
    /// The first source pixel each result pixel draws on.
    pub(crate) first: Vec<usize>,
    /// How many it draws on.
    pub(crate) count: Vec<usize>,
    /// The weights, `per` to a result pixel, of which it uses its count.
    pub(crate) weights: Vec<f32>,
    pub(crate) per: usize,
}

impl Taps {
    /// A side of `from` pixels resized to `to`.
    pub(crate) fn new(from: usize, to: usize) -> Self {
        let ratio = to as f64 / from as f64;
        // The kernel measures in the pixels of the smaller of the two.
        let scale = ratio.min(1.0);
        let reach = REACH / scale;
        let per = (2.0 * reach).floor() as usize + 1;
        let last = from as i64 - 1;
        let (mut first, mut count, mut weights) = (Vec::with_capacity(to), Vec::with_capacity(to), vec![0.0f32; to * per]);
        let mut raw = vec![0.0f64; per];
        for i in 0..to {
            let centre = (i as f64 + 0.5) / ratio - 0.5;
            let start = (centre - reach).ceil() as i64;
            let (low, high) = (start.clamp(0, last), (start + per as i64 - 1).clamp(0, last));
            raw.fill(0.0);
            for tap in 0..per as i64 {
                let source = start + tap;
                // Beyond the image, the pixel on its edge.
                raw[(source.clamp(0, last) - low) as usize] += mks2013((source as f64 - centre) * scale);
            }
            let sum: f64 = raw.iter().sum();
            for (weight, raw) in weights[i * per..(i + 1) * per].iter_mut().zip(&raw) {
                *weight = (raw / sum) as f32;
            }
            first.push(low as usize);
            count.push((high - low + 1) as usize);
        }
        Self { first, count, weights, per }
    }

    /// The source pixels the result pixels `from..to` draw on between
    /// them: from the first of the first to the last of the last, since
    /// both move on with the result pixel.
    pub(crate) fn span(&self, from: usize, to: usize) -> (usize, usize) {
        (self.first[from], self.first[to - 1] + self.count[to - 1])
    }
}

/// An image `width` by `height` of `channels` interleaved samples a
/// pixel, row-major, resized to `to_width` by `to_height` with Magic
/// Kernel Sharp 2013. Every channel is resized alike, so a photo with
/// transparency is flattened first ([`super::photo::flatten_alpha_onto`]).
pub fn resize<T: Sample>(data: &[T], width: usize, height: usize, channels: usize, to_width: usize, to_height: usize) -> Vec<T> {
    resize_into(data, width, height, channels, to_width, to_height)
}

/// [`resize`] of samples of one kind into samples of another, each
/// result held to what its own kind holds: values that run past black
/// and white, as a sharpened image's do, resized into the bytes of a
/// photo, with no image of the result's size held in between.
pub fn resize_into<I: Sample, O: Sample>(data: &[I], width: usize, height: usize, channels: usize, to_width: usize, to_height: usize) -> Vec<O> {
    assert_eq!(data.len(), width * height * channels);
    assert!((1..=4).contains(&channels), "one to four channels a pixel");
    assert!(width > 0 && height > 0 && to_width > 0 && to_height > 0, "neither image may be empty");
    let (across, down) = (Taps::new(width, to_width), Taps::new(height, to_height));
    O::taken(gpu::resized(I::handed(data), (width, height), channels, &across, &down, O::BYTES))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kernel_is_the_five_parabolas() {
        // At the pixels themselves it is the same-size kernel the page
        // gives: -1/32, 0, 17/16, 0, -1/32.
        for (x, value) in [(0.0, 17.0 / 16.0), (1.0, 0.0), (2.0, -1.0 / 32.0), (3.0, 0.0)] {
            assert!((mks2013(x) - value).abs() < 1e-12 && (mks2013(-x) - value).abs() < 1e-12, "{x}");
        }
        // The parabolas meet where one hands over to the next.
        for joint in [0.5, 1.5, 2.5] {
            assert!((mks2013(joint - 1e-9) - mks2013(joint + 1e-9)).abs() < 1e-8, "{joint}");
        }
        let magic = |x: f64| {
            let x = x.abs();
            if x <= 0.5 {
                0.75 - x * x
            } else if x <= 1.5 {
                0.5 * (x - 1.5) * (x - 1.5)
            } else {
                0.0
            }
        };
        for step in -300..=300 {
            let x = f64::from(step) / 100.0;
            // It is the Magic Kernel sharpened: [-m(x - 1) + 6 m(x) - m(x + 1)] / 4.
            let sharpened = (-magic(x - 1.0) + 6.0 * magic(x) - magic(x + 1.0)) / 4.0;
            assert!((mks2013(x) - sharpened).abs() < 1e-12, "{x}");
            // Wherever it is laid among the pixels, its weights sum to
            // one and their centre is where it lies.
            let taps = (-6..=6).map(|j| (f64::from(j), mks2013(f64::from(j) - x)));
            let (sum, centre) = taps.fold((0.0, 0.0), |(sum, centre), (j, weight)| (sum + weight, centre + j * weight));
            assert!((sum - 1.0).abs() < 1e-12 && (centre - x).abs() < 1e-12, "{x}: {sum}, {centre}");
        }
    }

    #[test]
    fn a_flat_tone_stays_flat_and_a_ramp_a_ramp() {
        for (to_width, to_height) in [(120, 90), (13, 10), (40, 30), (97, 7), (5, 61)] {
            let flat = resize(&vec![0.37f32; 40 * 30], 40, 30, 1, to_width, to_height);
            assert_eq!(flat.len(), to_width * to_height);
            assert!(flat.iter().all(|v| (v - 0.37).abs() < 1e-5), "{to_width} × {to_height}");
        }
        // A ramp across, away from the edges, where the pixel on the
        // edge stands for the ramp going on: each result pixel holds
        // what the ramp has at its centre.
        let ramp: Vec<f32> = (0..30).flat_map(|_| (0..40).map(|x| (x as f32 + 0.5) / 40.0)).collect();
        for to_width in [120usize, 100, 20, 10, 27] {
            let out = resize(&ramp, 40, 30, 1, to_width, 30);
            let margin = (to_width * 3).div_ceil(40).max(3);
            for x in margin..to_width - margin {
                let expected = (x as f32 + 0.5) / to_width as f32;
                assert!((out[15 * to_width + x] - expected).abs() < 2e-3, "{to_width}: {x}: {} against {expected}", out[15 * to_width + x]);
            }
        }
    }

    #[test]
    fn an_edge_stays_where_it_is_and_within_black_and_white() {
        // Ink left of the middle, paper right of it, a byte a pixel.
        let edge: Vec<u8> = (0..8).flat_map(|_| (0..20).map(|x| if x < 10 { 0 } else { 255 })).collect();
        for to_width in [60usize, 80, 10, 4] {
            let out = resize(&edge, 20, 8, 1, to_width, 8);
            let row = &out[3 * to_width..4 * to_width];
            assert!(row.windows(2).all(|pair| pair[0] <= pair[1]), "{to_width}: {row:?}");
            assert_eq!((row[0], row[to_width - 1]), (0, 255), "{to_width}");
            let dark = row.iter().filter(|&&v| v < 128).count();
            assert_eq!(dark, to_width / 2, "{to_width}: {row:?}");
        }
        // Enlarged a whole odd number of times, a pixel of a checkerboard
        // keeps its value at its centre: the kernel is zero a pixel away,
        // and what lies two away is the pixel's own colour.
        let board: Vec<u8> = (0..9 * 9).map(|i| if i % 2 == 0 { 0 } else { 255 }).collect();
        let out = resize(&board, 9, 9, 1, 27, 27);
        for y in 0..9 {
            for x in 0..9 {
                assert_eq!(out[(3 * y + 1) * 27 + 3 * x + 1], board[y * 9 + x], "{x}, {y}");
            }
        }
    }

    #[test]
    fn a_large_result_made_in_bands_is_the_result_made_whole() {
        // A large result: every row of it is the row a result of that row
        // alone would be, since a row is made of the source and nothing
        // else. Rows of the source repeated make every result row the
        // same, wherever it lies. (A result too large for the card to make
        // whole is made in tiles: see the tests of [`super::gpu`].)
        let (width, to_width, to_height) = (600usize, 1800usize, 2400usize);
        let line: Vec<u8> = (0..width).map(|x| ((x * 37) % 256) as u8).collect();
        let source: Vec<u8> = (0..800).flat_map(|_| line.iter().copied()).collect();
        let large = resize(&source, width, 800, 1, to_width, to_height);
        let one = resize(&line, width, 1, 1, to_width, 1);
        assert_eq!(large.len(), to_width * to_height);
        for (y, row) in large.chunks_exact(to_width).enumerate() {
            assert!(row == &one[..], "row {y}");
        }
        // And busy pixels in several channels, against the same resize
        // made of the top of the image alone, where the rows it shares
        // draw on the same source rows.
        let busy: Vec<u8> = (0..400 * 300 * 4).map(|i| ((i * 31 + (i / 1600) * 7) % 256) as u8).collect();
        let whole = resize(&busy, 400, 300, 4, 1600, 1200);
        let top = resize(&busy[..400 * 100 * 4], 400, 100, 4, 1600, 400);
        assert!(whole[..1600 * 380 * 4] == top[..1600 * 380 * 4]);
    }

    #[test]
    fn values_past_black_and_white_are_resized_into_bytes() {
        // Bytes as values, with an overshoot either side of an edge as
        // a sharpened image has: resized into bytes, they are what the
        // values resized come to, each held to what a byte holds.
        let values: Vec<f32> = (0..6).flat_map(|_| [10.0, 10.0, -40.0, 300.0, 250.0, 250.0]).collect();
        let bytes: Vec<u8> = resize_into(&values, 6, 6, 1, 18, 12);
        assert_eq!(bytes.len(), 18 * 12);
        assert_eq!((bytes[0], bytes[17]), (10, 250));
        assert!(bytes.contains(&0) && bytes.contains(&255), "{:?}", &bytes[..18]);
        // The same kind in and out is the resize itself.
        let same: Vec<u8> = resize_into(&bytes, 18, 12, 1, 9, 5);
        assert_eq!(same, resize(&bytes, 18, 12, 1, 9, 5));
    }

    #[test]
    fn every_channel_is_resized_alike() {
        let gray: Vec<u8> = (0..24 * 18).map(|i| ((i * 37) % 256) as u8).collect();
        let rgba: Vec<u8> = gray.iter().flat_map(|&v| [v, 255 - v, v / 2, 255]).collect();
        for (to_width, to_height) in [(48, 54), (8, 6), (31, 5)] {
            let out = resize(&rgba, 24, 18, 4, to_width, to_height);
            let channel = |c: usize| out.iter().skip(c).step_by(4).copied().collect::<Vec<u8>>();
            assert_eq!(channel(0), resize(&gray, 24, 18, 1, to_width, to_height));
            let inverse: Vec<u8> = gray.iter().map(|&v| 255 - v).collect();
            assert_eq!(channel(1), resize(&inverse, 24, 18, 1, to_width, to_height));
            assert!(channel(3).iter().all(|&alpha| alpha == 255), "opaque stays opaque");
        }
    }
}
