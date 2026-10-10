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
//! The values are resized as they are stored, as the rest of the
//! pipeline weighs them, and not in linear light.

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
pub trait Sample: Copy {
    fn to_f32(self) -> f32;
    /// The value kept within what the channel holds: beside a sharp edge
    /// the kernel's negative lobes carry it past black or white.
    fn from_f32(value: f32) -> Self;
}

impl Sample for u8 {
    fn to_f32(self) -> f32 {
        f32::from(self)
    }

    fn from_f32(value: f32) -> Self {
        value.round().clamp(0.0, 255.0) as u8
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
}

/// What each pixel along one side of the result draws on: a run of the
/// source's pixels along that side, and a weight for each.
struct Taps {
    /// The first source pixel each result pixel draws on.
    first: Vec<usize>,
    /// How many it draws on.
    count: Vec<usize>,
    /// The weights, `per` to a result pixel, of which it uses its count.
    weights: Vec<f32>,
    per: usize,
}

impl Taps {
    /// A side of `from` pixels resized to `to`.
    fn new(from: usize, to: usize) -> Self {
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

    /// The first source pixel result pixel `i` draws on, and the weights
    /// of it and those after it.
    fn of(&self, i: usize) -> (usize, &[f32]) {
        (self.first[i], &self.weights[i * self.per..i * self.per + self.count[i]])
    }
}

/// An image `width` by `height` of `channels` interleaved samples a
/// pixel, row-major, resized to `to_width` by `to_height` with Magic
/// Kernel Sharp 2013. Every channel is resized alike, so a photo with
/// transparency is flattened first ([`super::photo::flatten_alpha`]).
pub fn resize<T: Sample>(data: &[T], width: usize, height: usize, channels: usize, to_width: usize, to_height: usize) -> Vec<T> {
    assert_eq!(data.len(), width * height * channels);
    assert!((1..=4).contains(&channels), "one to four channels a pixel");
    assert!(width > 0 && height > 0 && to_width > 0 && to_height > 0, "neither image may be empty");
    let (across, down) = (Taps::new(width, to_width), Taps::new(height, to_height));
    let stride = width * channels;
    let mut out = Vec::with_capacity(to_width * to_height * channels);
    let mut row = vec![0.0f32; stride];
    for y in 0..to_height {
        // The source rows this row draws on, weighed into one row as wide
        // as the source...
        row.fill(0.0);
        let (first, weights) = down.of(y);
        for (k, &weight) in weights.iter().enumerate() {
            let source = &data[(first + k) * stride..(first + k + 1) * stride];
            for (sum, &value) in row.iter_mut().zip(source) {
                *sum += weight * value.to_f32();
            }
        }
        // ...and that row resized across.
        for x in 0..to_width {
            let (first, weights) = across.of(x);
            let mut sums = [0.0f32; 4];
            for (k, &weight) in weights.iter().enumerate() {
                let pixel = &row[(first + k) * channels..(first + k + 1) * channels];
                for (sum, &value) in sums.iter_mut().zip(pixel) {
                    *sum += weight * value;
                }
            }
            out.extend(sums[..channels].iter().map(|&sum| T::from_f32(sum)));
        }
    }
    out
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
