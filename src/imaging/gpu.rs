//! The photo's adjustments, worked on the graphics card: its resizing
//! ([`super::resize`]), the lens taken out of it
//! ([`super::adjust::straighten`]), its sharpening and blurring
//! ([`super::enhance::unsharp`]), its tone, and the filters over each
//! pixel's neighbourhood, the median and the closing of dots
//! ([`super::adjust::Filter`]) and the deconvolution that undoes a blur
//! ([`super::deblur::deconvolve`]); for the photo as it is shown, and
//! for every copy of it a search or a read makes. They are written in
//! CubeCL and run through wgpu, on Metal, or on Vulkan, DirectX 12 or
//! OpenGL, whichever the machine has first.
//!
//! Each is the same small piece of arithmetic at every pixel, drawing on
//! those about it: five taps of Magic Kernel Sharp each way, the four
//! pixels about where a lens bent one from, a Gaussian a few pixels wide,
//! a curve for each pixel alone, a median of nine or twenty-five, the
//! darkest and then the lightest of a square up to seven across, a blur
//! of up to some hundred and fifty taps twice over in each of twenty-five
//! rounds. A card works thousands of pixels side by side where the cores
//! work a dozen. What does little at each pixel is no quicker so: handing
//! the card a photo of twelve megapixels and taking it back takes longer
//! than the cores take to tone it, or to blur it a pixel or two, and
//! such an adjustment alone takes some ten or twenty milliseconds where
//! the cores took a few.
//!
//! A photo stays on the card while it is adjusted: handed it once, made
//! by every step and toned there, its filter laid over its luminance
//! there, and what is shown and read taken off it once ([`adjusted`],
//! [`Image`]). A photo whose bytes are shared, as a window that shows it
//! holds them, stays on the card from one adjustment of it to the next,
//! and the part a copy is made of is cut from it there ([`photo_on`]).
//! Handed the card afresh each time, a photo of twelve megapixels toned
//! took twice as long, and most of the time a curve or a blur takes is
//! still what it takes to take what is shown and read off the card.
//!
//! The card is required: the adjustments have no way to run on the
//! cores. The program asks for it as it starts ([`ready`]), and where
//! there is none says so and stops, rather than failing at the first
//! photo it opens.
//!
//! What the card makes of a pixel is what it makes of it wherever it
//! lies: in the photo, in a copy of a part of it, in a tile of either.
//! A pixel is worked out from the same values, by the same weights, added
//! up in the same order, as the cores did it; so a copy a read makes of a
//! part of the photo is still the photo so adjusted there, to the bit.
//! Against the cores, the median and the closing take one of the pixels'
//! own values and come out the same to the bit; the rest come within a
//! few millionths, a card rounding a product and the sum it is added to
//! once where the cores round them twice, and a byte made of such a value
//! may round the other way at a half. Bytes are rounded half away from
//! nought, as the cores round them, and not to the even.
//!
//! The closing is the darkest of each pixel's square and then the lightest
//! of that, a pass over the image each ([`extreme_kernel`]). Taken as a
//! network pools the lightest, of the photo negated and then of that
//! negated back, it took two passes more to negate it.
//!
//! A large image is worked out in tiles ([`tiled_within`]), the fewest
//! that the card takes ([`fewest_tiles`]): each no more than a launch
//! reaches either way ([`PIECE_SIDE`]), with what it draws on where that
//! is worked out with it, and each read from no more than the card holds
//! in a buffer ([`most_values`]): from what the tile
//! draws on, which for a resize is the run of source
//! pixels its taps reach, for the lens the box its edges are bent from
//! ([`super::adjust::drawn_on`]), and for the rest the tile and as much
//! about it as a pixel draws on: the reach of a mask, a median's radius,
//! twice a closing's, whose lightest is of darkest that themselves reach
//! that far, and for a deconvolution twice the reach of its taps for every
//! round, which blurs twice, each estimate drawing so much further on the
//! photo than the last. A curve for each pixel alone is worked out a run
//! of pixels at a time.
//!
//! The tiles are as large as a buffer of the card's takes, and not as
//! all its memory does. A card that fails, out of memory or otherwise,
//! stops the work with what it said ([`off_card`]).

use std::borrow::Cow;
use std::ops::{Deref, Range};
use std::panic::{self, RefUnwindSafe};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::server::{Handle, ServerError};
use cubecl::device::{WgpuDevice, WgpuDeviceKind};
use cubecl::wgpu::{AutoGraphicsApi, RuntimeOptions, WgpuInitError, try_init_setup};
use cubecl::Device;

use super::SharedRgba;
use super::adjust::{cut, drawn_on};
use super::resize::Taps;
use crate::work;

/// Side of the square of units a cube is: as many units as WebGPU
/// promises a cube may have. A unit of the median's, the negation's or the
/// lens's works out one pixel, and of the resize's and the mask's one
/// value of one; of the deconvolution's, [`ACROSS`] by [`DOWN`] pixels.
const SIDE: u32 = 16;
/// Units in a cube of a kernel over a line of values, as many as a cube
/// of [`SIDE`] by [`SIDE`].
const LINE: u32 = 256;
/// Most cubes a launch may have each way, as WebGPU promises them.
const MOST_CUBES: u32 = 65_535;
/// Pixels a unit of the deconvolution works out across and down, each
/// [`SIDE`] apart. A tap is read once for all eight of them, where with a
/// pixel to each unit the reading of the taps took most of the time:
/// eight to a unit undid a blur three times as fast as one to a unit, and
/// five times as fast as a unit reading every pixel it drew on from the
/// image itself.
const ACROSS: u32 = 4;
const DOWN: u32 = 2;
/// Most pixels a tap of the deconvolution reaches from its pixel, either
/// way, for the tile a cube shares, the pixels it works out and this many
/// about them: 80 by 48 pixels, within the 16 KiB WebGPU promises a cube.
/// A blur is undone at a size that holds it to two pixels
/// ([`super::deblur::MOST_SPREAD`]), whose taps reach six. Taps that reach
/// further are read from the image itself ([`ratio_wide_kernel`]).
pub const MOST_HALO: usize = 8;

/// Most pixels a tile is across or down: as many as cubes of [`SIDE`]
/// cover, [`MOST_CUBES`] of them, the most a launch may have each way.
pub const PIECE_SIDE: usize = MOST_CUBES as usize * SIDE as usize;
/// Fewest pixels across or down a tile is, however small the buffers the
/// card takes or however far what it works out draws on.
const LEAST_SIDE: usize = 64;

/// The place of this unit among those of a launch over a line of items
/// ([`line_cubes`]).
#[cube]
fn line_place() -> u32 {
    (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * LINE + UNIT_POS_X
}

/// The place across of this unit among those of a launch over the values
/// of the rows of an image ([`columns`]).
#[cube]
fn column() -> u32 {
    (CUBE_POS_Z * CUBE_COUNT_X + CUBE_POS_X) * SIDE + UNIT_POS_X
}

/// A value as a byte holds it: rounded half away from nought, as
/// [`f32::round`] rounds, and held from black to white.
#[cube]
fn rounded_byte(value: f32) -> u32 {
    let below = value.floor();
    let mut rounded = below;
    if value - below >= 0.5f32 {
        rounded = below + 1.0f32;
    }
    clamp(rounded, 0.0f32, 255.0f32) as u32
}

/// `value` as a kernel stores it: as a byte holds it ([`rounded_byte`]),
/// as a value, where `round`. Between two adjustments on the card, what a
/// byte made of it and read back in would be, made as it is stored rather
/// than by a pass of its own over the image.
#[cube]
fn stored(value: f32, #[comptime] round: bool) -> f32 {
    let mut kept = value;
    if comptime!(round) {
        kept = rounded_byte(value) as f32;
    }
    kept
}

/// Bytes, `channels` to a pixel and four to a word, as `count` values:
/// the first `keep` channels of each pixel.
#[cube(launch)]
fn unpacked_kernel(words: &[u32], values: &mut [f32], count: u32, channels: u32, keep: u32) {
    let i = line_place();
    if i < count {
        let byte = (i / keep) * channels + i % keep;
        values[i as usize] = ((words[(byte / 4) as usize] >> ((byte % 4) * 8)) & 255) as f32;
    }
}

/// Values, `channels` to a pixel, as `count` bytes, `per` to a pixel and
/// four to a word: each value as a byte holds it ([`rounded_byte`]), and
/// a byte past the values' channels white.
#[cube(launch)]
fn packed_kernel(values: &[f32], words: &mut [u32], count: u32, channels: u32, per: u32) {
    let w = line_place();
    if w * 4 < count {
        let mut word = 0u32;
        #[unroll]
        for b in 0..4u32 {
            let byte = w * 4 + b;
            if byte < count {
                let (pixel, channel) = (byte / per, byte % per);
                let mut value = 255u32;
                if channel < channels {
                    value = rounded_byte(values[(pixel * channels + channel) as usize]);
                }
                word |= value << (b * 8);
            }
        }
        words[w as usize] = word;
    }
}

/// `count` values held between black and white.
#[cube(launch)]
fn clamped_kernel(values: &mut [f32], count: u32) {
    let i = line_place();
    if i < count {
        values[i as usize] = clamp(values[i as usize], 0.0f32, 1.0f32);
    }
}

/// The first half of a resize: each row of the result of the source's
/// rows, `samples` values across, weighed together by the taps down, from
/// row `first` and `count` of them, `per` weights to a row of the result;
/// `height` rows. Added up in their order, from nought.
#[cube(launch)]
fn resized_down_kernel(source: &[f32], rows: &mut [f32], first: &[u32], count: &[u32], weights: &[f32], per: u32, samples: u32, height: u32) {
    let (j, y) = (column(), ABSOLUTE_POS_Y);
    if j < samples && y < height {
        let start = first[y as usize];
        let mut sum = 0.0f32;
        for k in 0..count[y as usize] {
            sum += weights[(y * per + k) as usize] * source[((start + k) * samples + j) as usize];
        }
        rows[(y * samples + j) as usize] = sum;
    }
}

/// The second: each of those rows, `from` values across, resized across
/// by the taps across into `width` pixels of `channels`, every channel
/// alike; each value as a byte holds it where `round` ([`stored`]).
#[cube(launch)]
fn resized_across_kernel(rows: &[f32], out: &mut [f32], first: &[u32], count: &[u32], weights: &[f32], per: u32, from: u32, width: u32, height: u32, channels: u32, #[comptime] round: bool) {
    let (s, y) = (column(), ABSOLUTE_POS_Y);
    if s < width * channels && y < height {
        let (x, c) = (s / channels, s % channels);
        let start = first[x as usize];
        let mut sum = 0.0f32;
        for k in 0..count[x as usize] {
            sum += weights[(x * per + k) as usize] * rows[(y * from + (start + k) * channels + c) as usize];
        }
        out[(y * width * channels + s) as usize] = stored(sum, round);
    }
}

/// How a lens bends a frame of pixels about its centre: its centre and
/// half its diagonal, its size, the lens's term and the model's slope past
/// the half diagonal ([`super::adjust::source_radius`]), and where in the
/// frame the part straightened begins.
#[derive(CubeLaunch, CubeType)]
struct Bend {
    cx: f32,
    cy: f32,
    norm: f32,
    width: f32,
    height: f32,
    k1: f32,
    slope: f32,
    out_x: f32,
    out_y: f32,
}

/// The photo as shot, as a straightening reads it: where its pixels begin
/// in the frame and how many of the frame's apart they lie, how many there
/// are, and of those the box held, where it begins and how large it is.
#[derive(CubeLaunch, CubeType)]
struct Shot {
    at_x: f32,
    at_y: f32,
    step_x: f32,
    step_y: f32,
    width: u32,
    height: u32,
    held_x: u32,
    held_y: u32,
    held_width: u32,
    held_height: u32,
}

/// A tile `width` by `height` of a part of the frame, `tile_x` and
/// `tile_y` into the part, with the lens taken out: each pixel read from
/// the photo as shot at [`super::adjust::source_radius`] along the same
/// ray from the centre, bilinearly, and `fill` where that falls off the
/// frame. The arithmetic is the cores' own, of the frame's places. Each
/// value as a byte holds it where `round` ([`stored`]).
#[cube(launch)]
fn straightened_kernel(shot: &[f32], out: &mut [f32], fill: &[f32], bend: Bend, from: Shot, tile_x: u32, tile_y: u32, width: u32, height: u32, channels: u32, #[comptime] round: bool) {
    let (x, y) = (ABSOLUTE_POS_X, ABSOLUTE_POS_Y);
    if x < width && y < height {
        let o = (y * width + x) * channels;
        let dx = bend.out_x + (tile_x + x) as f32 + 0.5f32 - bend.cx;
        let dy = bend.out_y + (tile_y + y) as f32 + 0.5f32 - bend.cy;
        let r = (dx * dx + dy * dy).sqrt() / bend.norm;
        let mut f = 1.0f32;
        if r > 0.0f32 {
            let mut source = (1.0f32 + bend.k1) + (r - 1.0f32) * bend.slope;
            if r <= 1.0f32 {
                source = r * (1.0f32 + bend.k1 * r * r);
            }
            f = source / r;
        }
        // Source pixel centres sit at half-integers, as in `Luma::sample`.
        let (sx, sy) = (bend.cx + dx * f - 0.5f32, bend.cy + dy * f - 0.5f32);
        if sx >= -0.5f32 && sy >= -0.5f32 && sx <= bend.width - 0.5f32 && sy <= bend.height - 0.5f32 {
            let (lx, ly) = ((sx - from.at_x) / from.step_x, (sy - from.at_y) / from.step_y);
            let left = clamp(lx.floor(), 0.0f32, (from.width - 1) as f32);
            let top = clamp(ly.floor(), 0.0f32, (from.height - 1) as f32);
            let (tx, ty) = (clamp(lx - left, 0.0f32, 1.0f32), clamp(ly - top, 0.0f32, 1.0f32));
            let (x0, y0) = (left as u32, top as u32);
            let (x1, y1) = (min(x0 + 1, from.width - 1), min(y0 + 1, from.height - 1));
            // Among the pixels held, which hold every one the tile reads,
            // and are kept to.
            let row0 = min(max(y0, from.held_y) - from.held_y, from.held_height - 1) * from.held_width;
            let row1 = min(max(y1, from.held_y) - from.held_y, from.held_height - 1) * from.held_width;
            let column0 = min(max(x0, from.held_x) - from.held_x, from.held_width - 1);
            let column1 = min(max(x1, from.held_x) - from.held_x, from.held_width - 1);
            for c in 0..channels {
                let p00 = shot[((row0 + column0) * channels + c) as usize];
                let p10 = shot[((row0 + column1) * channels + c) as usize];
                let p01 = shot[((row1 + column0) * channels + c) as usize];
                let p11 = shot[((row1 + column1) * channels + c) as usize];
                let above = p00 * (1.0f32 - tx) + p10 * tx;
                let below = p01 * (1.0f32 - tx) + p11 * tx;
                out[(o + c) as usize] = stored(above * (1.0f32 - ty) + below * ty, round);
            }
        } else {
            for c in 0..channels {
                out[(o + c) as usize] = stored(fill[c as usize], round);
            }
        }
    }
}

/// The first half of an unsharp mask: each value of `values`, `width` by
/// `height` of `channels`, blurred across by `weights`, `reach` either
/// side of it, the pixel on the image's edge standing for those beyond
/// it, every channel alike and apart from the others. Added up in their
/// order, from nought.
// A kernel indexes its slices; it has no iterators to take them by.
#[allow(clippy::needless_range_loop)]
#[cube(launch)]
fn blurred_across_kernel(values: &[f32], across: &mut [f32], weights: &[f32], reach: u32, width: u32, height: u32, channels: u32) {
    let (s, y) = (column(), ABSOLUTE_POS_Y);
    if s < width * channels && y < height {
        let (x, c) = ((s / channels) as i32, s % channels);
        let row = y * width * channels;
        let mut sum = 0.0f32;
        for k in 0..weights.len() {
            let sx = clamp(x + k as i32 - reach as i32, 0, width as i32 - 1) as u32;
            sum += weights[k] * values[(row + sx * channels + c) as usize];
        }
        across[(row + s) as usize] = sum;
    }
}

/// The second: those blurred down, and `amount` of what each value
/// stands from its blur added to it; as a byte holds it where `round`
/// ([`stored`]).
// A kernel indexes its slices; it has no iterators to take them by.
#[allow(clippy::needless_range_loop)]
#[cube(launch)]
fn unsharp_down_kernel(values: &[f32], across: &[f32], out: &mut [f32], weights: &[f32], reach: u32, amount: f32, width: u32, height: u32, channels: u32, #[comptime] round: bool) {
    let (s, y) = (column(), ABSOLUTE_POS_Y);
    let stride = width * channels;
    if s < stride && y < height {
        let mut sum = 0.0f32;
        for k in 0..weights.len() {
            let sy = clamp(y as i32 + k as i32 - reach as i32, 0, height as i32 - 1) as u32;
            sum += weights[k] * across[(sy * stride + s) as usize];
        }
        let value = values[(y * stride + s) as usize];
        out[(y * stride + s) as usize] = stored(value + amount * (value - sum), round);
    }
}

/// Pixel `i`, of the colours `red`, `green` and `blue`, bytes, toned: each
/// colour through the curve `table`, or the one `pick` names through it
/// for all three where it is less than three; the pixel shown so, opaque,
/// four bytes to a word, and its luminance.
#[cube]
fn toned_pixel(red: u32, green: u32, blue: u32, table: &[u32], pick: u32, shown: &mut [u32], luma: &mut [f32], i: u32) {
    let mut colours = Array::<u32>::new(3usize);
    colours[0] = table[red as usize];
    colours[1] = table[green as usize];
    colours[2] = table[blue as usize];
    if pick < 3 {
        let one = colours[pick as usize];
        colours[0] = one;
        colours[1] = one;
        colours[2] = one;
    }
    shown[i as usize] = colours[0] | (colours[1] << 8) | (colours[2] << 16) | (255u32 << 24);
    let (r, g, b) = (colours[0] as f32 / 255.0f32, colours[1] as f32 / 255.0f32, colours[2] as f32 / 255.0f32);
    luma[i as usize] = 0.299f32 * r + 0.587f32 * g + 0.114f32 * b;
}

/// `count` pixels of `values`, `channels` to each, bytes as values, the
/// first three their colours, toned ([`toned_pixel`]).
#[cube(launch)]
fn toned_kernel(values: &[f32], shown: &mut [u32], luma: &mut [f32], table: &[u32], pick: u32, count: u32, channels: u32) {
    let i = line_place();
    if i < count {
        let at = i * channels;
        toned_pixel(values[at as usize] as u32, values[(at + 1) as usize] as u32, values[(at + 2) as usize] as u32, table, pick, shown, luma, i);
    }
}

/// The `count` pixels of the box `width` across beginning at `(x0, y0)`
/// of an image `across` pixels wide, four bytes to each and a word, the
/// first three their colours, toned ([`toned_pixel`]) as they lie there,
/// with no values made of them first.
#[cube(launch)]
fn toned_words_kernel(words: &[u32], shown: &mut [u32], luma: &mut [f32], table: &[u32], pick: u32, count: u32, across: u32, x0: u32, y0: u32, width: u32) {
    let i = line_place();
    if i < count {
        let word = words[((y0 + i / width) * across + x0 + i % width) as usize];
        toned_pixel(word & 255, (word >> 8) & 255, (word >> 16) & 255, table, pick, shown, luma, i);
    }
}

/// The colours of the pixels of the box `width` across beginning at
/// `(x0, y0)` of an image `across` pixels wide, four bytes to each and a
/// word, as `count` values, three to a pixel.
#[cube(launch)]
fn unpacked_part_kernel(words: &[u32], values: &mut [f32], count: u32, across: u32, x0: u32, y0: u32, width: u32) {
    let i = line_place();
    if i < count {
        let (pixel, channel) = (i / 3, i % 3);
        let word = words[((y0 + pixel / width) * across + x0 + pixel % width) as usize];
        values[i as usize] = ((word >> (channel * 8)) & 255) as f32;
    }
}

/// `count` values held between `floor` and white, into both `seen` and
/// `estimate`: what a deconvolution begins from.
#[cube(launch)]
fn floored_kernel(values: &[f32], seen: &mut [f32], estimate: &mut [f32], floor: f32, count: u32) {
    let i = line_place();
    if i < count {
        let value = clamp(values[i as usize], floor, 1.0f32);
        seen[i as usize] = value;
        estimate[i as usize] = value;
    }
}

/// `input`, `width` by `height`, as an image `to_width` by `to_height`
/// from `(x0, y0)` in it: carried past its far edges by the pixels on them
/// where that is larger, and cut short where it is smaller; held between
/// black and white where `held`.
#[cube(launch)]
fn carried_kernel(input: &[f32], output: &mut [f32], width: u32, height: u32, x0: u32, y0: u32, to_width: u32, to_height: u32, #[comptime] held: bool) {
    let (x, y) = (ABSOLUTE_POS_X, ABSOLUTE_POS_Y);
    if x < to_width && y < to_height {
        let mut value = input[(min(y0 + y, height - 1) * width + min(x0 + x, width - 1)) as usize];
        if comptime!(held) {
            value = clamp(value, 0.0f32, 1.0f32);
        }
        output[(y * to_width + x) as usize] = value;
    }
}

/// The compare-swaps, each leaving the lesser of two places in the first
/// and the greater in the second, that leave the median of nine values in
/// the middle place: a network of 19, where sorting the square as far as
/// its middle took 30. As Devillard collects it.
const MEDIAN_OF_9: [(usize, usize); 19] =
    [(1, 2), (4, 5), (7, 8), (0, 1), (3, 4), (6, 7), (1, 2), (4, 5), (7, 8), (0, 3), (5, 8), (4, 7), (3, 6), (1, 4), (2, 5), (4, 7), (4, 2), (6, 4), (4, 2)];
/// And of twenty-five: a network of 99, where sorting took 234.
#[rustfmt::skip]
const MEDIAN_OF_25: [(usize, usize); 99] = [
    (0, 1), (3, 4), (2, 4), (2, 3), (6, 7), (5, 7), (5, 6), (9, 10), (8, 10), (8, 9), (12, 13), (11, 13), (11, 12), (15, 16), (14, 16), (14, 15), (18, 19),
    (17, 19), (17, 18), (21, 22), (20, 22), (20, 21), (23, 24), (2, 5), (3, 6), (0, 6), (0, 3), (4, 7), (1, 7), (1, 4), (11, 14), (8, 14), (8, 11), (12, 15),
    (9, 15), (9, 12), (13, 16), (10, 16), (10, 13), (20, 23), (17, 23), (17, 20), (21, 24), (18, 24), (18, 21), (19, 22), (8, 17), (9, 18), (0, 18), (0, 9),
    (10, 19), (1, 19), (1, 10), (11, 20), (2, 20), (2, 11), (12, 21), (3, 21), (3, 12), (13, 22), (4, 22), (4, 13), (14, 23), (5, 23), (5, 14), (15, 24),
    (6, 24), (6, 15), (7, 16), (7, 19), (13, 21), (15, 23), (7, 13), (7, 15), (1, 9), (3, 11), (5, 17), (11, 17), (9, 17), (4, 10), (6, 12), (7, 14), (4, 6),
    (4, 7), (12, 14), (10, 14), (6, 7), (10, 12), (6, 10), (6, 17), (12, 17), (7, 17), (7, 10), (12, 18), (7, 12), (10, 18), (12, 20), (10, 20), (10, 12),
];

/// The median of the square `2 * radius + 1` across about each pixel of
/// `input`, `width` by `height`, the pixels on its edge standing for those
/// beyond it: the square, a row after another, through the network for its
/// size ([`MEDIAN_OF_9`], [`MEDIAN_OF_25`]), and its middle place taken.
#[cube(launch)]
fn median_kernel(input: &[f32], output: &mut [f32], width: u32, height: u32, #[comptime] radius: u32) {
    let (x, y) = (ABSOLUTE_POS_X, ABSOLUTE_POS_Y);
    if x >= width || y >= height {
        terminate!();
    }
    let side = comptime!(2 * radius + 1);
    let count = comptime!((side * side) as usize);
    let mut square = Array::<f32>::new(count);
    #[unroll]
    for dy in 0..side {
        // `y + dy - radius`, held within the image.
        let sy = min(max(y + dy, radius) - radius, height - 1);
        #[unroll]
        for dx in 0..side {
            let sx = min(max(x + dx, radius) - radius, width - 1);
            square[(dy * side + dx) as usize] = input[(sy * width + sx) as usize];
        }
    }
    // Unrolled as the kernel is made: each pair of places known then.
    let network = comptime!(if radius == 1 { &MEDIAN_OF_9[..] } else { &MEDIAN_OF_25[..] });
    for k in comptime!(0..network.len()) {
        let i = comptime!(network[k].0);
        let j = comptime!(network[k].1);
        let (a, b) = (square[i], square[j]);
        square[i] = min(a, b);
        square[j] = max(a, b);
    }
    output[(y * width + x) as usize] = square[comptime!(count / 2)];
}

/// The darkest of the square `2 * radius + 1` across about each pixel of
/// `input`, `width` by `height`, or where `lightest` the lightest: the
/// pixels on its edge standing for those beyond it, which for the darkest
/// and the lightest is as good as leaving out what lies beyond, since
/// each is in the square already.
#[cube(launch)]
fn extreme_kernel(input: &[f32], output: &mut [f32], width: u32, height: u32, #[comptime] radius: u32, #[comptime] lightest: bool) {
    let (x, y) = (ABSOLUTE_POS_X, ABSOLUTE_POS_Y);
    if x < width && y < height {
        let side = comptime!(2 * radius + 1);
        let mut extreme = input[(y * width + x) as usize];
        #[unroll]
        for dy in 0..side {
            // `y + dy - radius`, held within the image.
            let sy = min(max(y + dy, radius) - radius, height - 1);
            #[unroll]
            for dx in 0..side {
                let value = input[(sy * width + min(max(x + dx, radius) - radius, width - 1)) as usize];
                if comptime!(lightest) {
                    extreme = max(extreme, value);
                } else {
                    extreme = min(extreme, value);
                }
            }
        }
        output[(y * width + x) as usize] = extreme;
    }
}

/// `source`, `width` by `height`, blurred by the taps `offsets` and
/// `weights` at the [`ACROSS`] by [`DOWN`] pixels of this unit, the pixels
/// on the image's edge standing for those beyond it: the cube's pixels and
/// `halo` about them read into a tile it shares, held within the image as
/// they are read, and each tap then a place in the tile, `offsets` from
/// the pixel's own. The taps are added up in their order, from nought.
#[cube]
fn blurred_tile(source: &[f32], offsets: &[i32], weights: &[f32], width: u32, height: u32, #[comptime] halo: u32) -> Array<f32> {
    let wide = comptime!(SIDE * ACROSS + 2 * halo);
    let cells = comptime!(wide * (SIDE * DOWN + 2 * halo));
    let mut tile = Shared::<[f32]>::new_slice(comptime!(cells as usize));
    let (x0, y0) = ((CUBE_POS_X * SIDE * ACROSS) as i32 - halo as i32, (CUBE_POS_Y * SIDE * DOWN) as i32 - halo as i32);
    let (w, h) = (width as i32, height as i32);
    #[unroll]
    for share in 0..comptime!(cells.div_ceil(SIDE * SIDE)) {
        let k = share * SIDE * SIDE + UNIT_POS;
        if k < cells {
            let sx = clamp(x0 + (k % wide) as i32, 0, w - 1);
            let sy = clamp(y0 + (k / wide) as i32, 0, h - 1);
            tile[k as usize] = source[(sy * w + sx) as usize];
        }
    }
    sync_cube();
    let own = ((UNIT_POS_Y + halo) * wide + UNIT_POS_X + halo) as i32;
    let mut sums = Array::<f32>::new(comptime!((ACROSS * DOWN) as usize));
    #[unroll]
    for p in 0..comptime!(ACROSS * DOWN) {
        sums[p as usize] = 0.0f32;
    }
    for t in 0..weights.len() {
        let (at, weight) = (own + offsets[t], weights[t]);
        #[unroll]
        for row in 0..DOWN {
            #[unroll]
            for column in 0..ACROSS {
                sums[(row * ACROSS + column) as usize] += weight * tile[(at + (row * SIDE * wide + column * SIDE) as i32) as usize];
            }
        }
    }
    sums
}

/// The first half of a round of the deconvolution: what the photo shows,
/// `observed`, over the `estimate` of it sharp blurred again, which is
/// held to `floor` and above.
#[cube(launch_unchecked)]
fn ratio_kernel(observed: &[f32], estimate: &[f32], ratio: &mut [f32], offsets: &[i32], weights: &[f32], width: u32, height: u32, floor: f32, #[comptime] halo: u32) {
    let blurred = blurred_tile(estimate, offsets, weights, width, height, halo);
    #[unroll]
    for row in 0..DOWN {
        #[unroll]
        for column in 0..ACROSS {
            let x = (CUBE_POS_X * ACROSS + column) * SIDE + UNIT_POS_X;
            let y = (CUBE_POS_Y * DOWN + row) * SIDE + UNIT_POS_Y;
            if x < width && y < height {
                let i = (y * width + x) as usize;
                ratio[i] = observed[i] / max(blurred[(row * ACROSS + column) as usize], floor);
            }
        }
    }
}

/// The second half: the `estimate` times the `ratio` blurred, held
/// between `floor` and white.
#[cube(launch_unchecked)]
fn update_kernel(ratio: &[f32], estimate: &mut [f32], offsets: &[i32], weights: &[f32], width: u32, height: u32, floor: f32, #[comptime] halo: u32) {
    let blurred = blurred_tile(ratio, offsets, weights, width, height, halo);
    #[unroll]
    for row in 0..DOWN {
        #[unroll]
        for column in 0..ACROSS {
            let x = (CUBE_POS_X * ACROSS + column) * SIDE + UNIT_POS_X;
            let y = (CUBE_POS_Y * DOWN + row) * SIDE + UNIT_POS_Y;
            if x < width && y < height {
                let i = (y * width + x) as usize;
                estimate[i] = clamp(estimate[i] * blurred[(row * ACROSS + column) as usize], floor, 1.0f32);
            }
        }
    }
}

/// `source`, `width` by `height`, blurred at `(x, y)` by taps reaching
/// `across` and `down` from it with `weights`, read from the image itself
/// and held within it, and added up in their order, from nought: as
/// [`blurred_tile`] adds them up, for taps that reach further than its
/// tile.
#[cube]
fn blurred_at(source: &[f32], across: &[i32], down: &[i32], weights: &[f32], x: u32, y: u32, width: u32, height: u32) -> f32 {
    let (w, h) = (width as i32, height as i32);
    let mut sum = 0.0f32;
    for t in 0..weights.len() {
        let sx = clamp(x as i32 + across[t], 0, w - 1);
        let sy = clamp(y as i32 + down[t], 0, h - 1);
        sum += weights[t] * source[(sy * w + sx) as usize];
    }
    sum
}

/// [`ratio_kernel`] for taps that reach further than [`MOST_HALO`].
#[cube(launch)]
fn ratio_wide_kernel(observed: &[f32], estimate: &[f32], ratio: &mut [f32], across: &[i32], down: &[i32], weights: &[f32], width: u32, height: u32, floor: f32) {
    let (x, y) = (ABSOLUTE_POS_X, ABSOLUTE_POS_Y);
    if x < width && y < height {
        let i = (y * width + x) as usize;
        ratio[i] = observed[i] / max(blurred_at(estimate, across, down, weights, x, y, width, height), floor);
    }
}

/// [`update_kernel`] for taps that reach further than [`MOST_HALO`].
#[cube(launch)]
fn update_wide_kernel(ratio: &[f32], estimate: &mut [f32], across: &[i32], down: &[i32], weights: &[f32], width: u32, height: u32, floor: f32) {
    let (x, y) = (ABSOLUTE_POS_X, ABSOLUTE_POS_Y);
    if x < width && y < height {
        let i = (y * width + x) as usize;
        estimate[i] = clamp(estimate[i] * blurred_at(ratio, across, down, weights, x, y, width, height), floor, 1.0f32);
    }
}

/// The card's client, made the first time it is asked for, or why there
/// is none: no graphics API wgpu could open, or no adapter on any of them.
fn card() -> &'static Result<Client, String> {
    static CARD: OnceLock<Result<Client, String>> = OnceLock::new();
    CARD.get_or_init(|| {
        // Opened by the call that says why it cannot be, rather than by
        // making the client, which panics, and the panic's message would
        // stand before the program's own.
        let opened = panic::catch_unwind(|| {
            let device = WgpuDevice::new(WgpuDeviceKind::DefaultDevice);
            RuntimeOptions::try_default().and_then(|options| try_init_setup::<AutoGraphicsApi>(&device, options))?;
            Ok::<_, WgpuInitError>(Device::Wgpu(device).client())
        });
        match opened {
            Ok(Ok(client)) => Ok(client),
            Ok(Err(why)) => Err(format!("QR Forensics adjusts photos on a graphics card, and could not open one: {why}")),
            Err(_) => Err("QR Forensics adjusts photos on a graphics card, and the one found would not open.".to_owned()),
        }
    })
}

/// The card made ready for the adjustments, or why it could not be: what
/// the program asks as it starts, to say so and stop where there is none.
pub fn ready() -> Result<(), String> {
    card().as_ref().map(|_| ()).map_err(Clone::clone)
}

/// The card's client. An adjustment asked for without one is a program
/// that did not ask whether it was [`ready`].
///
/// Every piece of work on the card asks for it, and the photos the card
/// holds whose bytes nothing else holds any more are let go first
/// ([`PHOTOS`]): a photo closed is let go by the next work on the card,
/// which is when its room is wanted, rather than by the next photo handed
/// it. The card takes the room back on its way to making room for more.
fn client() -> &'static Client {
    let client = card().as_ref().unwrap_or_else(|why| panic!("{why}"));
    // Not while the photos are held: whoever holds them lets go of these
    // as it does.
    if let Ok(mut photos) = PHOTOS.try_lock() {
        let_go_of_dropped(&mut photos);
    }
    client
}

/// Values held on the card, or words of bytes, and how many.
#[derive(Clone)]
struct Held {
    handle: Handle,
    len: usize,
}

impl Held {
    /// The values, copied once to be handed the card: handed it as a slice
    /// of bytes, by the client's `create_from_slice`, they were copied twice
    /// on the way, and a photo of twelve megapixels was 96 MB copied.
    fn of<T: CubeElement>(client: &Client, values: &[T]) -> Self {
        Self::owned(client, values.to_vec())
    }

    /// The values, handed the card as they are, with no copy made of them.
    fn owned<T: CubeElement>(client: &Client, values: Vec<T>) -> Self {
        let len = values.len();
        Self { handle: client.create(Bytes::from_elems(values)), len }
    }

    /// Bytes, four to a word, the last word filled out with nought: copied
    /// once where they are borrowed, and handed the card as they are where
    /// they are owned.
    fn bytes<'a>(client: &Client, bytes: impl Into<Cow<'a, [u8]>>) -> Self {
        let mut whole = bytes.into().into_owned();
        let len = whole.len().div_ceil(4);
        whole.resize(len * 4, 0);
        Self { handle: client.create(Bytes::from_elems(whole)), len }
    }

    /// Room for `len` values, or words.
    fn empty(client: &Client, len: usize) -> Self {
        Self { handle: client.empty(len * size_of::<f32>()), len }
    }

    /// The values, as a kernel's argument.
    fn arg(&self) -> BufferArg {
        // SAFETY: the length is that of the values the handle was made
        // to hold.
        unsafe { BufferArg::from_raw_parts(self.handle.clone(), self.len) }
    }

    fn read(&self, client: &Client) -> Vec<f32> {
        f32::from_bytes(&off_card(client.read_one(self.handle.clone()))).to_vec()
    }

    /// The first `count` bytes of the words.
    fn read_bytes(&self, client: &Client, count: usize) -> Vec<u8> {
        let mut bytes = off_card(client.read_one(self.handle.clone())).to_vec();
        bytes.truncate(count);
        bytes
    }

    /// The first `count` bytes of the words, and the values of `values`,
    /// read together: one wait for the card to hand both over, where a read
    /// of each waits for its own.
    fn read_bytes_with(&self, values: &Held, client: &Client, count: usize) -> (Vec<u8>, Vec<f32>) {
        let read = off_card(client.read(vec![self.handle.clone(), values.handle.clone()]));
        let [words, values] = &read[..] else { unreachable!("a read for each handle") };
        let mut bytes = words.to_vec();
        bytes.truncate(count);
        (bytes, f32::from_bytes(values).to_vec())
    }
}

/// What a read off the card came to. The card reports at a read how the
/// work that made what is read went; and where it failed, out of memory
/// or with its device lost, the work stops with what the card said, since
/// there is no other way to do it.
fn off_card<T>(read: Result<T, ServerError>) -> T {
    read.unwrap_or_else(|error| panic!("the graphics card failed: {error}"))
}

/// The cubes that cover an image `width` by `height`, `per_cube` pixels
/// to a cube across and down.
fn cubes(width: usize, height: usize, [across, down]: [u32; 2]) -> CubeCount {
    CubeCount::Static((width as u32).div_ceil(across), (height as u32).div_ceil(down), 1)
}

/// The cubes of [`SIDE`] by [`SIDE`] units that cover `samples` values
/// across and `rows` down: as many across as a launch may have, and as
/// many times that over as it takes ([`column`]).
fn columns(samples: usize, rows: usize) -> CubeCount {
    let across = (samples as u32).div_ceil(SIDE);
    CubeCount::Static(across.min(MOST_CUBES), (rows as u32).div_ceil(SIDE), across.div_ceil(MOST_CUBES))
}

/// The cubes of [`LINE`] units that cover `count` items: as many across
/// as a launch may have, and as many rows of them as that takes
/// ([`line_place`]).
fn line_cubes(count: usize) -> CubeCount {
    let cubes = (count as u32).div_ceil(LINE);
    CubeCount::Static(cubes.min(MOST_CUBES), cubes.div_ceil(MOST_CUBES), 1)
}

/// A box `[x0, y0, x1, y1]` of an image, the far sides exclusive.
type Part = [usize; 4];

fn area([x0, y0, x1, y1]: Part) -> usize {
    (x1 - x0) * (y1 - y0)
}

/// `part` as it lies in `within`.
fn inside([x0, y0, x1, y1]: Part, within: Part) -> Part {
    [x0 - within[0], y0 - within[1], x1 - within[0], y1 - within[1]]
}

/// Samples handed to the card: bytes, or values.
#[derive(Clone, Copy)]
pub enum Samples<'a> {
    Bytes(&'a [u8]),
    Values(&'a [f32]),
}

/// What the card made: bytes, or values.
pub enum Made {
    Bytes(Vec<u8>),
    Values(Vec<f32>),
}

impl Samples<'_> {
    /// The box `part` of the samples, of an image `width` across with
    /// `channels` to a pixel.
    fn cut(self, width: usize, channels: usize, part: Part) -> Made {
        match self {
            Samples::Bytes(bytes) => Made::Bytes(cut(bytes, width, channels, part)),
            Samples::Values(values) => Made::Values(cut(values, width, channels, part)),
        }
    }
}

impl Made {
    fn handed(&self) -> Samples<'_> {
        match self {
            Made::Bytes(bytes) => Samples::Bytes(bytes),
            Made::Values(values) => Samples::Values(values),
        }
    }

    fn cut(self, width: usize, channels: usize, part: Part) -> Made {
        self.handed().cut(width, channels, part)
    }

    fn into_values(self) -> Vec<f32> {
        match self {
            Made::Values(values) => values,
            Made::Bytes(_) => unreachable!("values are made as values"),
        }
    }

    fn into_bytes(self) -> Vec<u8> {
        match self {
            Made::Bytes(bytes) => bytes,
            Made::Values(_) => unreachable!("bytes are made as bytes"),
        }
    }
}

/// The samples a tile is worked out from ([`values_on`]): the image's own,
/// where the tile draws on the whole of it, and otherwise the box of it the
/// tile draws on, cut from it to be handed the card, which is handed it as
/// it was cut, with no copy made of it.
enum Piece<'a> {
    Whole(Samples<'a>),
    Cut(Made),
}

impl<'a> Piece<'a> {
    /// The box `part` of `samples`, `width` by `height` of `channels` to a
    /// pixel.
    fn of(samples: Samples<'a>, (width, height): (usize, usize), channels: usize, part: Part) -> Self {
        if part == [0, 0, width, height] { Piece::Whole(samples) } else { Piece::Cut(samples.cut(width, channels, part)) }
    }
}

/// The tiles an output `width` by `height` is worked out in, `across` by
/// `down` of them, row after row: as alike as they may be, none more than
/// a pixel wider or taller than another.
fn tiles(width: usize, height: usize, [across, down]: [usize; 2]) -> impl Iterator<Item = Part> {
    let edge = |length: usize, count: usize, i: usize| i * length / count;
    (0..down).flat_map(move |j| (0..across).map(move |i| [edge(width, across, i), edge(height, down, j), edge(width, across, i + 1), edge(height, down, j + 1)]))
}

/// How many tiles across and down an output `width` by `height` is worked
/// out in ([`tiles`]): the fewest, each no more than `side` either way,
/// for which `holds`, the most values the card holds in a buffer to work
/// a tile out, is no more than `most` for every one; and of as few, those
/// that hold the fewest values all told, which read the least about their
/// edges twice. Where none fit, as many as there may be, each at least
/// [`LEAST_SIDE`] either way, or the whole of the image that way where it
/// is less.
fn fewest_tiles(width: usize, height: usize, side: usize, most: usize, holds: impl Fn(Part) -> usize) -> [usize; 2] {
    let fewest = [width.div_ceil(side), height.div_ceil(side)];
    let finest = [(width / LEAST_SIDE).max(fewest[0]), (height / LEAST_SIDE).max(fewest[1])];
    // What the tiles hold all told, where every one fits.
    let held = |grid| tiles(width, height, grid).try_fold(0, |all, tile| Some(holds(tile)).filter(|&held| held <= most).map(|held| all + held));
    let mut best: Option<([usize; 2], usize)> = None;
    let more = |count: usize, best: Option<([usize; 2], usize)>| best.is_some_and(|([a, d], _)| count > a * d);
    for across in fewest[0]..=finest[0] {
        if more(across * fewest[1], best) {
            break;
        }
        // The fewest down that fit so many across: more down are more
        // tiles.
        for down in fewest[1]..=finest[1] {
            if more(across * down, best) {
                break;
            }
            if let Some(all) = held([across, down]) {
                if best.is_none_or(|([a, d], least)| across * down < a * d || all < least) {
                    best = Some(([across, down], all));
                }
                break;
            }
        }
    }
    best.map_or(finest, |(grid, _)| grid)
}

/// An output `width` by `height` worked out on the card tile by tile,
/// each tile with what `work` made of it: in the fewest tiles, each no
/// more than `side` either way, for which `holds`, the most values the
/// card holds in a buffer to work a tile out, is no more than `most` for
/// every one ([`fewest_tiles`]), the whole of it in one where that fits.
fn tiled_within<R>(most: usize, side: usize, width: usize, height: usize, holds: impl Fn(Part) -> usize, work: impl Fn(&Client, Part) -> R) -> Vec<(Part, R)> {
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let client = client();
    tiles(width, height, fewest_tiles(width, height, side, most, holds)).map(|tile| (tile, work(client, tile))).collect()
}

/// The most values the card is given to hold in a buffer at once: as
/// many as its largest buffer holds, and no more than leaves the bytes of
/// them, four at the most to a value, counted in the `u32` a kernel
/// counts in. WebGPU promises no more than 128 MiB a buffer, and a card
/// that holds less is given tiles to fit.
fn most_values(client: &Client) -> usize {
    let largest = client.properties().memory.max_page_size / size_of::<f32>() as u64;
    usize::try_from(largest).unwrap_or(usize::MAX).min(u32::MAX as usize / 4)
}

/// What the tiles made, `channels` to a pixel, laid where they lie in an
/// output `width` by `height`.
fn assembled(mut tiles: Vec<(Part, Made)>, width: usize, height: usize, channels: usize) -> Made {
    if let [(part, _)] = tiles[..]
        && part == [0, 0, width, height]
    {
        return tiles.pop().expect("one tile").1;
    }
    fn laid<T: Copy + Default>(tiles: impl Iterator<Item = (Part, Vec<T>)>, width: usize, height: usize, channels: usize) -> Vec<T> {
        let mut out = vec![T::default(); width * height * channels];
        for ([x0, y0, x1, y1], values) in tiles {
            let across = (x1 - x0) * channels;
            for (y, row) in (y0..y1).zip(values.chunks_exact(across)) {
                out[(y * width + x0) * channels..(y * width + x1) * channels].copy_from_slice(row);
            }
        }
        out
    }
    match tiles.first() {
        Some((_, Made::Bytes(_))) => Made::Bytes(laid(tiles.into_iter().map(|(part, made)| (part, if let Made::Bytes(bytes) = made { bytes } else { unreachable!() })), width, height, channels)),
        _ => Made::Values(laid(tiles.into_iter().map(|(part, made)| (part, made.into_values())), width, height, channels)),
    }
}

/// The box of an image `width` by `height` that `tile` draws on, where a
/// pixel draws on those up to `reach` away: the tile and that about it,
/// within the image.
fn about(tile: Part, reach: usize, width: usize, height: usize) -> Part {
    [tile[0].saturating_sub(reach), tile[1].saturating_sub(reach), (tile[2] + reach).min(width), (tile[3] + reach).min(height)]
}

/// `work` of `samples`, `width` by `height` of `channels` to a pixel,
/// where a pixel of what it makes draws on those up to `reach` away: tile
/// by tile ([`tiled_within`]), each worked out whole from as much of the
/// image as it draws on ([`about`]), which the card holds `per_pixel`
/// values to a pixel of, and its own part of that kept, `made` channels
/// to a pixel. What the whole would come to there.
///
/// The launches that work a tile out reach across all it draws on, so a
/// tile is less than [`PIECE_SIDE`] either way by twice the reach: a tile
/// as wide as a launch reaches, and the reach about it, took more cubes
/// than a launch may have.
#[allow(clippy::too_many_arguments)]
fn around_within(most: usize, samples: Samples<'_>, (width, height): (usize, usize), channels: usize, reach: usize, per_pixel: usize, made: usize, work: impl Fn(&Client, Piece<'_>, usize, usize) -> Made) -> Made {
    let side = PIECE_SIDE.saturating_sub(2 * reach).max(1);
    let tiles = tiled_within(most, side, width, height, |tile| area(about(tile, reach, width, height)) * per_pixel, |client, tile| {
        let held = about(tile, reach, width, height);
        let (w, h) = (held[2] - held[0], held[3] - held[1]);
        let worked = work(client, Piece::of(samples, (width, height), channels, held), w, h);
        if held == tile { worked } else { worked.cut(w, made, inside(tile, held)) }
    });
    assembled(tiles, width, height, made)
}

/// `count` pixels worked out on the card a run of them at a time, `most`
/// of them at the most, by `work`, which writes what it makes of a run
/// where it belongs.
fn in_runs_within(most: usize, count: usize, mut work: impl FnMut(&Client, Range<usize>)) {
    let client = client();
    let run = most.max(1);
    for at in (0..count).step_by(run) {
        work(client, at..(at + run).min(count));
    }
}

/// The samples of `piece`, `pixels` of `channels`, on the card as values,
/// the first `keep` channels of each pixel: bytes unpacked there, values as
/// they are.
fn values_on(client: &Client, piece: Piece<'_>, pixels: usize, channels: usize, keep: usize) -> Held {
    let unpacked = |words: Held| {
        let (count, values) = (pixels * keep, Held::empty(client, pixels * keep));
        unpacked_kernel::launch(client, line_cubes(count), CubeDim::new_1d(LINE), words.arg(), values.arg(), count as u32, channels as u32, keep as u32);
        values
    };
    let as_they_are = |values: Held| {
        assert_eq!(keep, channels, "values are kept as they are");
        values
    };
    match piece {
        Piece::Whole(Samples::Bytes(bytes)) => unpacked(Held::bytes(client, bytes)),
        Piece::Cut(Made::Bytes(bytes)) => unpacked(Held::bytes(client, bytes)),
        Piece::Whole(Samples::Values(values)) => as_they_are(Held::of(client, values)),
        Piece::Cut(Made::Values(values)) => as_they_are(Held::owned(client, values)),
    }
}

/// What values on the card, `pixels` of `channels`, are taken off as.
#[derive(Clone, Copy)]
enum Out {
    /// Values, as they are.
    Values,
    /// Values held between black and white.
    Held,
    /// Bytes, so many to a pixel, the values as a byte holds them, and
    /// white past their channels.
    Bytes(usize),
}

fn taken(client: &Client, values: Held, pixels: usize, channels: usize, out: Out) -> Made {
    let count = pixels * channels;
    match out {
        Out::Values => Made::Values(values.read(client)),
        Out::Held => {
            clamped_kernel::launch(client, line_cubes(count), CubeDim::new_1d(LINE), values.arg(), count as u32);
            Made::Values(values.read(client))
        }
        Out::Bytes(per) => {
            let bytes = pixels * per;
            let words = Held::empty(client, bytes.div_ceil(4));
            packed_kernel::launch(client, line_cubes(bytes.div_ceil(4)), CubeDim::new_1d(LINE), values.arg(), words.arg(), bytes as u32, channels as u32, per as u32);
            Made::Bytes(words.read_bytes(client, bytes))
        }
    }
}

/// The taps of the result pixels `from..to` of `taps`, on the card: the
/// first source pixel of each counted from `base`, how many, and the
/// weights.
fn taps_on(client: &Client, taps: &Taps, from: usize, to: usize, base: usize) -> [Held; 3] {
    let first: Vec<u32> = taps.first[from..to].iter().map(|&first| (first - base) as u32).collect();
    let count: Vec<u32> = taps.count[from..to].iter().map(|&count| count as u32).collect();
    [Held::owned(client, first), Held::owned(client, count), Held::of(client, &taps.weights[from * taps.per..to * taps.per])]
}

/// The box `tile` of the result of resizing by the taps `across` and
/// `down` ([`Taps`]), from `source` on the card, the box `read` of the
/// image resized, `channels` to a pixel: each row of the tile the source's
/// rows weighed by `down` into one, and that weighed by `across`. The
/// values as the taps add them up, not yet held to what a sample holds;
/// or, where `round`, as a byte holds each ([`stored`]).
#[allow(clippy::too_many_arguments)]
fn resized_on(client: &Client, source: &Held, read: Part, channels: usize, across: &Taps, down: &Taps, tile: Part, round: bool) -> Held {
    let (w, (to_w, to_h)) = (read[2] - read[0], (tile[2] - tile[0], tile[3] - tile[1]));
    let [down_first, down_count, down_weights] = taps_on(client, down, tile[1], tile[3], read[1]);
    let [across_first, across_count, across_weights] = taps_on(client, across, tile[0], tile[2], read[0]);
    let rows = Held::empty(client, to_h * w * channels);
    let unit = CubeDim::new_2d(SIDE, SIDE);
    resized_down_kernel::launch(client, columns(w * channels, to_h), unit, source.arg(), rows.arg(), down_first.arg(), down_count.arg(), down_weights.arg(), down.per as u32, (w * channels) as u32, to_h as u32);
    let out = Held::empty(client, to_w * to_h * channels);
    resized_across_kernel::launch(
        client,
        columns(to_w * channels, to_h),
        unit,
        rows.arg(),
        out.arg(),
        across_first.arg(),
        across_count.arg(),
        across_weights.arg(),
        across.per as u32,
        (w * channels) as u32,
        to_w as u32,
        to_h as u32,
        channels as u32,
        round,
    );
    out
}

/// `samples`, `width` by `height` of `channels` to a pixel, resized by the
/// taps `across` and `down` ([`Taps`]) to as many pixels as they have:
/// each row of the result the source's rows weighed by `down` into one,
/// and that weighed by `across`, as bytes or as values held between black
/// and white. A tile of the result is made from the run of source pixels
/// its taps reach each way.
pub(crate) fn resized(samples: Samples<'_>, (width, height): (usize, usize), channels: usize, across: &Taps, down: &Taps, bytes: bool) -> Made {
    resized_within(most_values(client()), samples, (width, height), channels, across, down, bytes)
}

#[allow(clippy::too_many_arguments)]
fn resized_within(most: usize, samples: Samples<'_>, (width, height): (usize, usize), channels: usize, across: &Taps, down: &Taps, bytes: bool) -> Made {
    let (to_width, to_height) = (across.first.len(), down.first.len());
    let reads = |[x0, y0, x1, y1]: Part| {
        let ((left, right), (top, bottom)) = (across.span(x0, x1), down.span(y0, y1));
        [left, top, right, bottom]
    };
    let holds = |tile: Part| {
        let read = reads(tile);
        let rows = tile[3] - tile[1];
        area(read).max(rows * (read[2] - read[0])).max(area(tile)) * channels
    };
    let tiles = tiled_within(most, PIECE_SIDE, to_width, to_height, holds, |client, tile| {
        let read = reads(tile);
        let source = values_on(client, Piece::of(samples, (width, height), channels, read), area(read), channels, channels);
        let out = resized_on(client, &source, read, channels, across, down, tile, false);
        taken(client, out, area(tile), channels, if bytes { Out::Bytes(channels) } else { Out::Held })
    });
    assembled(tiles, to_width, to_height, channels)
}

/// How a lens is taken out of a frame ([`super::adjust::straighten`]):
/// the frame's size in its pixels, the lens's term, and the model's slope
/// past the half diagonal.
pub(crate) struct Lens {
    pub frame: [f32; 2],
    pub k1: f32,
    pub slope: f32,
}

/// The box `tile` of the part of a frame beginning at `out_at` in it, with
/// the lens taken out, from `shot` on the card, the box `read` of the
/// photo as shot, which is `shot_size` pixels of `channels` that begin at
/// `place[0]` in the frame and lie `place[1]` of its pixels apart; `fill`
/// where a pixel falls off the frame. The values as the bilinear sampling
/// makes them, not yet held to what a sample holds; or, where `round`, as
/// a byte holds each ([`stored`]).
#[allow(clippy::too_many_arguments)]
fn straightened_on(client: &Client, shot: &Held, read: Part, shot_size: (usize, usize), channels: usize, [at, step]: [[f32; 2]; 2], lens: &Lens, out_at: [f32; 2], tile: Part, fill: &[f32], round: bool) -> Held {
    let [w, h] = lens.frame;
    let (cx, cy) = (w / 2.0, h / 2.0);
    let norm = (w * w + h * h).sqrt() / 2.0;
    let (tw, th) = (tile[2] - tile[0], tile[3] - tile[1]);
    let (out, fill) = (Held::empty(client, tw * th * channels), Held::of(client, fill));
    let bend = BendLaunch::new(cx, cy, norm, w, h, lens.k1, lens.slope, out_at[0], out_at[1]);
    let from = ShotLaunch::new(at[0], at[1], step[0], step[1], shot_size.0 as u32, shot_size.1 as u32, read[0] as u32, read[1] as u32, (read[2] - read[0]) as u32, (read[3] - read[1]) as u32);
    straightened_kernel::launch(client, cubes(tw, th, [SIDE; 2]), CubeDim::new_2d(SIDE, SIDE), shot.arg(), out.arg(), fill.arg(), bend, from, tile[0] as u32, tile[1] as u32, tw as u32, th as u32, channels as u32, round);
    out
}

/// The part `width` by `height` of a frame beginning at `out_at` in it,
/// with the lens taken out ([`super::adjust::straighten`]): each pixel
/// read from `shot`, `shot_size` pixels of `channels`, which begin at
/// `place[0]` in the frame and lie `place[1]` of its pixels apart, and
/// `fill` where that falls off the frame; as bytes or as values held
/// between black and white. A tile of the part is straightened from the
/// box of `shot` its edges are bent from ([`drawn_on`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn straightened(shot: Samples<'_>, shot_size: (usize, usize), channels: usize, place: [[f32; 2]; 2], lens: &Lens, out_at: [f32; 2], size: (usize, usize), fill: &[f32], bytes: bool) -> Made {
    straightened_within(most_values(client()), shot, shot_size, channels, place, lens, out_at, size, fill, bytes)
}

#[allow(clippy::too_many_arguments)]
fn straightened_within(most: usize, shot: Samples<'_>, (shot_width, shot_height): (usize, usize), channels: usize, place @ [at, step]: [[f32; 2]; 2], lens: &Lens, out_at: [f32; 2], (width, height): (usize, usize), fill: &[f32], bytes: bool) -> Made {
    assert!(shot_width > 0 && shot_height > 0, "a part straightened from none of the photo is filled where it is asked for");
    // The box of the photo as shot a tile reads: the pixels about the
    // places its edges are bent from, and one more either way for what a
    // card makes of those places against what the cores made of them.
    let reads = |tile: Part| {
        let [fx0, fy0, fx1, fy1] = drawn_on(lens.frame, lens.k1, [out_at[0] + tile[0] as f32, out_at[1] + tile[1] as f32], (tile[2] - tile[0], tile[3] - tile[1]));
        let index = |frame: f32, at: f32, step: f32, size: usize| ((frame - at) / step).floor().clamp(0.0, (size - 1) as f32) as usize;
        let (x0, y0) = (index(fx0, at[0], step[0], shot_width), index(fy0, at[1], step[1], shot_height));
        let (x1, y1) = (index(fx1, at[0], step[0], shot_width), index(fy1, at[1], step[1], shot_height));
        [x0.saturating_sub(1), y0.saturating_sub(1), (x1 + 3).min(shot_width), (y1 + 3).min(shot_height)]
    };
    let tiles = tiled_within(most, PIECE_SIDE, width, height, |tile| area(reads(tile)).max(area(tile)) * channels, |client, tile| {
        let read = reads(tile);
        let source = values_on(client, Piece::of(shot, (shot_width, shot_height), channels, read), area(read), channels, channels);
        let out = straightened_on(client, &source, read, (shot_width, shot_height), channels, place, lens, out_at, tile, fill, false);
        taken(client, out, area(tile), channels, if bytes { Out::Bytes(channels) } else { Out::Held })
    });
    assembled(tiles, width, height, channels)
}

/// `values` on the card, `width` by `height` of `channels`, under an
/// unsharp mask of `weights` and `amount`, every channel alike and apart
/// from the others, past the image the pixel on its edge standing for
/// those beyond it. The values not held to black and white; or, where
/// `round`, as a byte holds each ([`stored`]).
fn unsharp_on(client: &Client, values: &Held, (w, h): (usize, usize), channels: usize, weights: &[f32], amount: f32, round: bool) -> Held {
    let reach = weights.len() / 2;
    let (across, out, taps) = (Held::empty(client, w * h * channels), Held::empty(client, w * h * channels), Held::of(client, weights));
    let (unit, every) = (CubeDim::new_2d(SIDE, SIDE), columns(w * channels, h));
    let dims = (w as u32, h as u32, channels as u32);
    blurred_across_kernel::launch(client, every.clone(), unit, values.arg(), across.arg(), taps.arg(), reach as u32, dims.0, dims.1, dims.2);
    unsharp_down_kernel::launch(client, every, unit, values.arg(), across.arg(), out.arg(), taps.arg(), reach as u32, amount, dims.0, dims.1, dims.2, round);
    out
}

/// `samples`, `width` by `height` of `channels[0]` to a pixel, the first
/// `channels[1]` of them under an unsharp mask of `weights`, a Gaussian
/// whose middle tap is the pixel's own, and `amount`: every channel
/// sharpened alike and apart from the others, past the image the pixel on
/// its edge standing for those beyond it; as bytes, four to a pixel and
/// the fourth white, or as values not held to black and white.
pub(crate) fn unsharp(samples: Samples<'_>, size: (usize, usize), channels: [usize; 2], weights: &[f32], amount: f32, bytes: bool) -> Made {
    unsharp_within(most_values(client()), samples, size, channels, weights, amount, bytes)
}

fn unsharp_within(most: usize, samples: Samples<'_>, (width, height): (usize, usize), [channels, keep]: [usize; 2], weights: &[f32], amount: f32, bytes: bool) -> Made {
    let made = if bytes { 4 } else { keep };
    if width == 0 || height == 0 || keep == 0 {
        return if bytes { Made::Bytes(Vec::new()) } else { Made::Values(Vec::new()) };
    }
    let reach = weights.len() / 2;
    around_within(most, samples, (width, height), channels, reach, keep, made, |client, piece, w, h| {
        let values = values_on(client, piece, w * h, channels, keep);
        let out = unsharp_on(client, &values, (w, h), keep, weights, amount, false);
        taken(client, out, w * h, keep, if bytes { Out::Bytes(4) } else { Out::Values })
    })
}

/// `values` on the card, `pixels` of `channels` that are bytes as values,
/// toned: each of their colours through the curve `table`, or the colour
/// `pick` names through it for all three, and the pixel made opaque; the
/// pixels shown so, four bytes to each and a word, and their luminance.
fn toned_on(client: &Client, values: &Held, pixels: usize, channels: usize, table: &[u8; 256], pick: Option<usize>) -> [Held; 2] {
    let (shown, luma, curve) = (Held::empty(client, pixels), Held::empty(client, pixels), Held::of(client, &table.map(u32::from)));
    let pick = pick.map_or(3, |pick| pick as u32);
    toned_kernel::launch(client, line_cubes(pixels), CubeDim::new_1d(LINE), values.arg(), shown.arg(), luma.arg(), curve.arg(), pick, pixels as u32, channels as u32);
    [shown, luma]
}

/// The box `width` by `height` beginning at `at` of an image `across`
/// pixels wide on the card, four bytes to each pixel and a word, toned as
/// it lies there ([`toned_on`]), with no values made of its bytes first.
fn toned_words_on(client: &Client, words: &Held, across: usize, [x0, y0]: [usize; 2], (width, height): (usize, usize), table: &[u8; 256], pick: Option<usize>) -> [Held; 2] {
    let pixels = width * height;
    let (shown, luma, curve) = (Held::empty(client, pixels), Held::empty(client, pixels), Held::of(client, &table.map(u32::from)));
    let pick = pick.map_or(3, |pick| pick as u32);
    toned_words_kernel::launch(client, line_cubes(pixels), CubeDim::new_1d(LINE), words.arg(), shown.arg(), luma.arg(), curve.arg(), pick, pixels as u32, across as u32, x0 as u32, y0 as u32, width as u32);
    [shown, luma]
}

/// The pixels of `rgba`, four bytes to each, toned where they lie: each
/// of their colours through the curve `table`, or the colour `pick` names
/// through it for all three, and the pixel made opaque; with the
/// luminance of each pixel so shown, which is what is read. Where not
/// `shown`, the pixels are left as they were, and only their luminance is
/// taken off the card. A run of no more than `most` values at a time.
fn toned_within(most: usize, rgba: &mut [u8], table: &[u8; 256], pick: Option<usize>, shown: bool) -> Vec<f32> {
    assert_eq!(rgba.len() % 4, 0);
    let count = rgba.len() / 4;
    let mut luma = vec![0.0f32; count];
    // Three words a pixel are held on the card for each pixel of a run:
    // its bytes, the pixel shown, and its luminance.
    in_runs_within(most / 3, count, |client, run| {
        let n = run.len();
        let pixels = &mut rgba[run.start * 4..run.end * 4];
        let words = Held::bytes(client, &*pixels);
        let [toned, values] = toned_words_on(client, &words, n, [0, 0], (n, 1), table, pick);
        if shown {
            let (toned, values) = toned.read_bytes_with(&values, client, n * 4);
            pixels.copy_from_slice(&toned);
            luma[run].copy_from_slice(&values);
        } else {
            luma[run].copy_from_slice(&values.read(client));
        }
    });
    luma
}

/// Photos as loaded that the card holds from one adjustment of them to
/// the next, each with its bytes there, four to a word, the latest last:
/// what a photo is held by whose bytes are shared ([`SharedRgba`]), as a
/// window that shows it holds them. A photo is let go by the next work on
/// the card once nothing else holds its bytes ([`client`]), or once two
/// photos have been held since it was.
static PHOTOS: Mutex<Vec<(Weak<PhotoBytes>, Held)>> = Mutex::new(Vec::new());
/// The bytes of a photo as loaded, as they are shared ([`SharedRgba`]).
type PhotoBytes = dyn Deref<Target = [u8]> + Send + Sync + RefUnwindSafe;
/// Most photos the card holds at once.
const PHOTOS_HELD: usize = 2;

/// The photos the card holds, less those whose bytes nothing else holds.
fn let_go_of_dropped(photos: &mut Vec<(Weak<PhotoBytes>, Held)>) {
    photos.retain(|(photo, _)| photo.strong_count() > 0);
}

/// The bytes of the photo `shared` on the card, four to a word: those it
/// holds from the last time, or those handed it now, which it holds from
/// now on ([`PHOTOS`]).
fn photo_on(client: &Client, shared: &SharedRgba) -> Held {
    let mut photos = PHOTOS.lock().expect("nothing panics holding the photos");
    let_go_of_dropped(&mut photos);
    if let Some(i) = photos.iter().position(|(photo, _)| photo.upgrade().is_some_and(|photo| Arc::ptr_eq(&photo, shared))) {
        let held = photos.remove(i);
        let words = held.1.clone();
        photos.push(held);
        return words;
    }
    let words = Held::bytes(client, &shared[..]);
    photos.push((Arc::downgrade(shared), words.clone()));
    if photos.len() > PHOTOS_HELD {
        photos.remove(0);
    }
    words
}

/// What of a photo as loaded a part of it is made from ([`adjusted`]):
/// `rgba`, `width` pixels across, four bytes to each, and the box `part`
/// of it; with the same bytes held to be shared, where they are, which the
/// card holds from one adjustment of the photo to the next and cuts the
/// part from itself ([`photo_on`]).
pub(crate) struct Source<'a> {
    pub rgba: &'a [u8],
    pub width: usize,
    pub part: [usize; 4],
    pub shared: Option<&'a SharedRgba>,
}

impl Source<'_> {
    fn size(&self) -> (usize, usize) {
        (self.part[2] - self.part[0], self.part[3] - self.part[1])
    }

    /// The part's bytes, cut from the photo's where it is not the whole.
    fn bytes(&self) -> Cow<'_, [u8]> {
        if self.part == [0, 0, self.width, self.rgba.len() / 4 / self.width.max(1)] { Cow::Borrowed(self.rgba) } else { Cow::Owned(cut(self.rgba, self.width, 4, self.part)) }
    }

    /// The bytes on the card that hold the part, four to a word, with how
    /// many pixels across they hold and where the part begins in them: the
    /// photo the card holds, where its bytes are shared and the card has
    /// room for them ([`photo_on`]), and otherwise the part's own, handed it.
    fn words(&self, client: &Client) -> (Held, usize, [usize; 2]) {
        match self.shared {
            Some(shared) if shared.len() / 4 <= most_values(client) => {
                assert!(shared.len() == self.rgba.len(), "the bytes shared are the photo's own");
                (photo_on(client, shared), self.width, [self.part[0], self.part[1]])
            }
            _ => (Held::bytes(client, self.bytes()), self.size().0, [0, 0]),
        }
    }

    /// The part's colours on the card as values, three to a pixel, cut there
    /// from the bytes that hold it ([`Self::words`]). Its alpha is left
    /// behind: no step reads it, and the part is toned opaque.
    fn on(&self, client: &Client) -> Held {
        let (w, h) = self.size();
        let (words, across, [x0, y0]) = self.words(client);
        let (count, values) = (w * h * 3, Held::empty(client, w * h * 3));
        unpacked_part_kernel::launch(client, line_cubes(count), CubeDim::new_1d(LINE), words.arg(), values.arg(), count as u32, across as u32, x0 as u32, y0 as u32, w as u32);
        values
    }
}

/// A step a part of a photo is made by, in its colours, before it is
/// toned ([`adjusted`]).
pub(crate) enum Step {
    /// Resized by the taps across and down ([`resized`]).
    Resized { across: Taps, down: Taps },
    /// The lens taken out ([`straightened`]): the part `size` of the frame
    /// beginning at `out_at` in it, read from the image so far as the
    /// photo as shot, which begins at `place[0]` in the frame and lies
    /// `place[1]` of its pixels apart, and white where that falls off the
    /// frame.
    Straightened { place: [[f32; 2]; 2], lens: Lens, out_at: [f32; 2], size: (usize, usize) },
    /// Under an unsharp mask of `weights` and `amount`, in its colours
    /// ([`unsharp`]).
    Sharpened { weights: Vec<f32>, amount: f32 },
}

/// A part of a photo adjusted and toned on the card, not yet taken off
/// it: its pixels as they are shown, four bytes to each and a word, and
/// their luminance, which a filter may yet be laid over there
/// ([`Self::taken`]).
pub(crate) struct Toned {
    shown: Held,
    pub luma: Image,
}

impl Toned {
    /// What is shown of the part and what is read of it, off the card:
    /// the pixels as toned and their luminance; or, where the luminance
    /// was `filtered`, that, and its grey shown, as a filter shows the
    /// photo. What is shown is taken only where it is `shown`, and is
    /// otherwise empty: a copy made for a read is read and never shown.
    ///
    /// The grey of a filtered luminance is made on the cores from the
    /// luminance taken off the card ([`super::greyed`]), in well under a
    /// millisecond, where making it on the card would have taken as many
    /// bytes off it again as the luminance itself.
    pub(crate) fn taken(self, filtered: Option<Image>, shown: bool) -> (Vec<u8>, Vec<f32>) {
        let client = client();
        let pixels = self.luma.width * self.luma.height;
        match filtered {
            None if shown => self.shown.read_bytes_with(&self.luma.values, client, pixels * 4),
            None => (Vec::new(), self.luma.read()),
            Some(filtered) => {
                let luma = filtered.read();
                let mut grey = Vec::new();
                if shown {
                    grey = vec![0u8; pixels * 4];
                    super::greyed(&luma, &mut grey);
                }
                (grey, luma)
            }
        }
    }
}

/// The part of a photo `source` holds, made by `steps` and toned by
/// `table` and `pick` ([`toned_within`]), on the card from first to last:
/// the photo handed the card once, or not at all where it holds it from
/// the last time ([`Source::words`]), and the part left there, its
/// luminance for a filter to be laid over ([`Toned`]). Each step made as
/// its own call makes it, each value stored as a byte holds it, so that
/// the part comes out the same to the bit as it does a step at a time
/// ([`stepped`]). Only the part's colours are worked: its alpha is never
/// read, and the part is toned opaque. With no step, the part is toned
/// from the photo's bytes as they lie. `None` where an image along the way
/// has no pixels, or does not fit what the card is given to hold at once
/// ([`most_values`]) or [`PIECE_SIDE`] either way.
///
/// Taken off the card between steps, a photo of twelve megapixels
/// enlarged twice, straightened and sharpened spent most of its time going
/// to the card and back, and took longer than the cores had; made on the
/// card from first to last, it took three fifths as long as they did.
pub(crate) fn adjusted(source: &Source<'_>, steps: &[Step], table: &[u8; 256], pick: Option<usize>) -> Option<Toned> {
    adjusted_within(most_values(client()), source, steps, table, pick)
}

fn adjusted_within(most: usize, source: &Source<'_>, steps: &[Step], table: &[u8; 256], pick: Option<usize>) -> Option<Toned> {
    // The images along the way, three colours to a pixel, and the most
    // values any holds at once.
    let (width, height) = source.size();
    let mut sizes = vec![(width, height)];
    let mut largest = width * height * 3;
    for step in steps {
        let &(w, h) = sizes.last().expect("the part");
        let made = match step {
            Step::Resized { across, down } => {
                let (to_w, to_h) = (across.first.len(), down.first.len());
                largest = largest.max(to_h * w * 3);
                (to_w, to_h)
            }
            Step::Straightened { size, .. } => *size,
            Step::Sharpened { .. } => (w, h),
        };
        largest = largest.max(made.0 * made.1 * 3);
        sizes.push(made);
    }
    if largest > most || sizes.iter().any(|&(w, h)| w == 0 || h == 0 || w.max(h) > PIECE_SIDE) {
        return None;
    }
    let client = client();
    let &(w, h) = sizes.last().expect("the part");
    let [shown, luma] = if steps.is_empty() {
        let (words, across, at) = source.words(client);
        toned_words_on(client, &words, across, at, (w, h), table, pick)
    } else {
        let mut image = source.on(client);
        for (step, pair) in steps.iter().zip(sizes.windows(2)) {
            let [(w, h), (to_w, to_h)] = [pair[0], pair[1]];
            // Bytes, as a step made alone hands on.
            image = match step {
                Step::Resized { across, down } => resized_on(client, &image, [0, 0, w, h], 3, across, down, [0, 0, to_w, to_h], true),
                Step::Straightened { place, lens, out_at, .. } => straightened_on(client, &image, [0, 0, w, h], (w, h), 3, *place, lens, *out_at, [0, 0, to_w, to_h], &[255.0; 3], true),
                Step::Sharpened { weights, amount } => unsharp_on(client, &image, (w, h), 3, weights, *amount, true),
            };
        }
        toned_on(client, &image, w * h, 3, table, pick)
    };
    Some(Toned { shown, luma: Image { values: luma, width: w, height: h } })
}

/// [`adjusted`] a step at a time, each step in tiles, through the cores'
/// memory, and taken off the card: the pixels shown, four bytes to each,
/// where they are `shown`, and otherwise none ([`Toned::taken`]), and
/// their luminance. For a part too large for the card to hold at once.
pub(crate) fn stepped(source: &Source<'_>, steps: &[Step], table: &[u8; 256], pick: Option<usize>, shown: bool) -> (Vec<u8>, Vec<f32>) {
    stepped_within(most_values(client()), source, steps, table, pick, shown)
}

fn stepped_within(most: usize, source: &Source<'_>, steps: &[Step], table: &[u8; 256], pick: Option<usize>, shown: bool) -> (Vec<u8>, Vec<f32>) {
    let mut image = source.bytes().into_owned();
    let mut size = source.size();
    for step in steps {
        image = match step {
            Step::Resized { across, down } => {
                let made = resized_within(most, Samples::Bytes(&image), size, 4, across, down, true);
                size = (across.first.len(), down.first.len());
                made
            }
            Step::Straightened { place, lens, out_at, size: part } => {
                // A part that draws on none of the photo is white throughout.
                let made = if size.0 == 0 || size.1 == 0 { Made::Bytes([255; 4].repeat(part.0 * part.1)) } else { straightened_within(most, Samples::Bytes(&image), size, 4, *place, lens, *out_at, *part, &[255.0; 4], true) };
                size = *part;
                made
            }
            Step::Sharpened { weights, amount } => unsharp_within(most, Samples::Bytes(&image), size, [4, 3], weights, *amount, true),
        }
        .into_bytes();
    }
    let luma = toned_within(most, &mut image, table, pick, shown);
    (if shown { image } else { Vec::new() }, luma)
}

/// An image of a channel on the card, `width` by `height`: the luminance
/// of a part of a photo adjusted there ([`Toned`]), which its filter is
/// laid over without its leaving the card, by the kernels the filters use
/// on an image handed the card ([`median`], [`closed`], [`deconvolved`]),
/// so that it comes out as it would have there; or an image handed the
/// card to be made by several such steps and taken off it once, where
/// each step alone would take it off and hand it back.
#[derive(Clone)]
pub(crate) struct Image {
    values: Held,
    pub width: usize,
    pub height: usize,
}

/// Whether an image of a channel `width` by `height` may be worked on the
/// card whole, as an [`Image`]: it has pixels, and is no more than the card
/// is given to hold at once ([`most_values`]) nor [`PIECE_SIDE`] either way.
pub(crate) fn holds(width: usize, height: usize) -> bool {
    width > 0 && height > 0 && width.max(height) <= PIECE_SIDE && width * height <= most_values(client())
}

/// `data`, `width` by `height`, handed the card as an [`Image`], made by
/// `steps` there, and taken off it once: where the card holds it and the
/// largest image along the way, `largest`, whole ([`holds`]). `None`
/// otherwise, for it to be made a step at a time, each step handed the
/// card and taken off it in turn, in tiles where it must be.
pub(crate) fn through(data: &[f32], (width, height): (usize, usize), largest: (usize, usize), steps: impl FnOnce(Image) -> Image) -> Option<Vec<f32>> {
    if !holds(width, height) || !holds(largest.0, largest.1) {
        return None;
    }
    Some(steps(Image::handed(data, width, height)).read())
}

impl Image {
    fn of(values: Held, (width, height): (usize, usize)) -> Self {
        Self { values, width, height }
    }

    fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// `values`, `width` by `height`, handed the card: an image it
    /// [`holds`].
    pub(crate) fn handed(values: &[f32], width: usize, height: usize) -> Image {
        assert_eq!(values.len(), width * height);
        Image::of(Held::of(client(), values), (width, height))
    }

    /// The values, off the card.
    pub(crate) fn read(&self) -> Vec<f32> {
        self.values.read(client())
    }

    /// The median of the square `2 * radius + 1` across about each pixel
    /// ([`median`]).
    pub(crate) fn median(&self, radius: usize) -> Image {
        Image::of(median_on(client(), &self.values, self.size(), radius), self.size())
    }

    /// Closed over gaps of `radius` pixels ([`closed`]).
    pub(crate) fn closed(&self, radius: usize) -> Image {
        Image::of(closed_on(client(), &self.values, self.size(), radius), self.size())
    }

    /// Held to `floor` and above, and deconvolved by `taps` for `rounds`
    /// ([`deconvolved`]).
    pub(crate) fn deconvolved(&self, taps: &[(isize, isize, f32)], rounds: usize, floor: f32) -> Image {
        Image::of(deconvolved_on(client(), &self.values, self.size(), taps, rounds, floor), self.size())
    }

    /// The image from its top-left corner `width` by `height`: carried
    /// past its far edges by the pixels on them, and cut short.
    pub(crate) fn carried(&self, width: usize, height: usize) -> Image {
        self.carried_from([0, 0], (width, height), false)
    }

    /// The box `[x0, y0, x1, y1]` of the image, held between black and
    /// white.
    pub(crate) fn held_part(&self, [x0, y0, x1, y1]: Part) -> Image {
        assert!(x0 <= x1 && y0 <= y1 && x1 <= self.width && y1 <= self.height, "the part lies within the image");
        self.carried_from([x0, y0], (x1 - x0, y1 - y0), true)
    }

    fn carried_from(&self, [x0, y0]: [usize; 2], (width, height): (usize, usize), held: bool) -> Image {
        let client = client();
        let out = Held::empty(client, width * height);
        let (from, to) = ((self.width as u32, self.height as u32), (width as u32, height as u32));
        carried_kernel::launch(client, cubes(width, height, [SIDE; 2]), CubeDim::new_2d(SIDE, SIDE), self.values.arg(), out.arg(), from.0, from.1, x0 as u32, y0 as u32, to.0, to.1, held);
        Image::of(out, (width, height))
    }

    /// The image resized to `width` by `height` ([`super::resize`]), held
    /// between black and white.
    pub(crate) fn resized(&self, width: usize, height: usize) -> Image {
        self.resized_part((width, height), (width, height))
    }

    /// The box `width` by `height` from the top-left corner of the image
    /// resized to `size` ([`Self::resized`]), and only that made: what the
    /// whole resized and cut so would come to.
    pub(crate) fn resized_part(&self, size: (usize, usize), (width, height): (usize, usize)) -> Image {
        assert!(width <= size.0 && height <= size.1, "the part lies within the image resized");
        let client = client();
        let (across, down) = (Taps::new(self.width, size.0), Taps::new(self.height, size.1));
        let out = resized_on(client, &self.values, [0, 0, self.width, self.height], 1, &across, &down, [0, 0, width, height], false);
        clamped_kernel::launch(client, line_cubes(width * height), CubeDim::new_1d(LINE), out.arg(), (width * height) as u32);
        Image::of(out, (width, height))
    }

    /// The image under an unsharp mask of `weights` and `amount`
    /// ([`unsharp`]), not held to black and white.
    pub(crate) fn sharpened(&self, weights: &[f32], amount: f32) -> Image {
        Image::of(unsharp_on(client(), &self.values, self.size(), 1, weights, amount, false), self.size())
    }
}

/// `filter` of `data`, `width` by `height`, where a pixel of what it makes
/// draws on those up to `reach` away, tile by tile ([`around_within`]):
/// each piece handed the card, filtered there, and taken off it.
fn filtered_within(most: usize, data: &[f32], width: usize, height: usize, reach: usize, filter: impl Fn(&Client, &Held, (usize, usize)) -> Held) -> Vec<f32> {
    assert_eq!(data.len(), width * height);
    if data.is_empty() {
        return Vec::new();
    }
    let worked = around_within(most, Samples::Values(data), (width, height), 1, reach, 1, 1, |client, piece, w, h| {
        Made::Values(filter(client, &values_on(client, piece, w * h, 1, 1), (w, h)).read(client))
    });
    worked.into_values()
}

/// [`filtered_within`] in as large tiles as the card takes.
fn filtered(data: &[f32], width: usize, height: usize, reach: usize, filter: impl Fn(&Client, &Held, (usize, usize)) -> Held) -> Vec<f32> {
    filtered_within(most_values(client()), data, width, height, reach, filter)
}

/// The median of the square `2 * radius + 1` pixels across about each
/// pixel of `data`, `width` by `height`, the pixels on its edge standing
/// for those beyond it. A radius of one or two.
pub(crate) fn median(data: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    filtered(data, width, height, radius, |client, piece, size| median_on(client, piece, size, radius))
}

fn median_on(client: &Client, input: &Held, (width, height): (usize, usize), radius: usize) -> Held {
    assert!(matches!(radius, 1 | 2), "a median of radius {radius}");
    let output = Held::empty(client, width * height);
    median_kernel::launch(client, cubes(width, height, [SIDE; 2]), CubeDim::new_2d(SIDE, SIDE), input.arg(), output.arg(), width as u32, height as u32, radius as u32);
    output
}

/// `data`, `width` by `height`, closed over gaps of `radius` pixels: the
/// lightest of the darkest of the square `2 * radius + 1` pixels across
/// about each pixel, the pixels on its edge standing for those beyond it.
pub(crate) fn closed(data: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    filtered(data, width, height, 2 * radius, |client, piece, size| closed_on(client, piece, size, radius))
}

fn closed_on(client: &Client, input: &Held, (width, height): (usize, usize), radius: usize) -> Held {
    let (darkest, closed) = (Held::empty(client, width * height), Held::empty(client, width * height));
    let (w, h, every) = (width as u32, height as u32, cubes(width, height, [SIDE; 2]));
    let unit = CubeDim::new_2d(SIDE, SIDE);
    extreme_kernel::launch(client, every.clone(), unit, input.arg(), darkest.arg(), w, h, radius as u32, false);
    extreme_kernel::launch(client, every, unit, darkest.arg(), closed.arg(), w, h, radius as u32, true);
    closed
}

/// How far a tap of `taps` reaches from its pixel, either way.
fn halo_of(taps: &[(isize, isize, f32)]) -> usize {
    taps.iter().map(|&(dx, dy, _)| dx.unsigned_abs().max(dy.unsigned_abs())).max().unwrap_or(0)
}

/// `rounds` of Richardson and Lucy's deconvolution of `data`, `width` by
/// `height`, by the blur `taps`, as [`super::deblur::kernel`] gives them:
/// what is observed is `data` held to `floor` and above, on the card, and
/// the estimate begins as that; each round the estimate blurred again,
/// what is observed over that, and the estimate times the quotient
/// blurred, held between `floor` and white. Fewer rounds where the work is
/// called off. A pixel of the result draws on the photo twice as far as
/// the taps reach for every round.
pub(crate) fn deconvolved(data: &[f32], width: usize, height: usize, taps: &[(isize, isize, f32)], rounds: usize, floor: f32) -> Vec<f32> {
    let reach = 2 * rounds * halo_of(taps);
    filtered(data, width, height, reach, |client, piece, size| deconvolved_on(client, piece, size, taps, rounds, floor))
}

/// [`deconvolved`] of `values` on the card: held to `floor` and above, in
/// one pass, into both what is observed and the estimate, which is the
/// result.
fn deconvolved_on(client: &Client, values: &Held, (width, height): (usize, usize), taps: &[(isize, isize, f32)], rounds: usize, floor: f32) -> Held {
    assert!(!taps.is_empty(), "a blur has a tap");
    let count = width * height;
    let (seen, estimate) = (Held::empty(client, count), Held::empty(client, count));
    floored_kernel::launch(client, line_cubes(count), CubeDim::new_1d(LINE), values.arg(), seen.arg(), estimate.arg(), floor, count as u32);
    let halo = halo_of(taps);
    let ratio = Held::empty(client, width * height);
    let weights = Held::owned(client, taps.iter().map(|&(_, _, weight)| weight).collect::<Vec<f32>>());
    let (w, h) = (width as u32, height as u32);
    let unit = CubeDim::new_2d(SIDE, SIDE);
    // The tile a cube shares, where its taps reach no further than it.
    let wide = SIDE * ACROSS + 2 * halo as u32;
    let tile = (wide * (SIDE * DOWN + 2 * halo as u32)) as usize * size_of::<f32>();
    // Each round blurs twice.
    let per_round = 2 * width * height * taps.len();
    if halo <= MOST_HALO && tile <= client.properties().hardware.max_shared_memory_size {
        let offsets = Held::owned(client, taps.iter().map(|&(dx, dy, _)| (dy * wide as isize + dx) as i32).collect::<Vec<i32>>());
        let every = cubes(width, height, [SIDE * ACROSS, SIDE * DOWN]);
        paced(client, rounds, per_round, || {
            // SAFETY: every place read and written is within its values.
            // A pixel is written only within the image, and read within
            // it, held there as the tile is read; a tap reaches no further
            // than the halo, so that every place in the tile it reads is
            // within the tile; and there are as many offsets as weights.
            unsafe {
                ratio_kernel::launch_unchecked(client, every.clone(), unit, seen.arg(), estimate.arg(), ratio.arg(), offsets.arg(), weights.arg(), w, h, floor, halo as u32);
                update_kernel::launch_unchecked(client, every.clone(), unit, ratio.arg(), estimate.arg(), offsets.arg(), weights.arg(), w, h, floor, halo as u32);
            }
        });
    } else {
        let across = Held::owned(client, taps.iter().map(|&(dx, _, _)| dx as i32).collect::<Vec<i32>>());
        let down = Held::owned(client, taps.iter().map(|&(_, dy, _)| dy as i32).collect::<Vec<i32>>());
        let every = cubes(width, height, [SIDE; 2]);
        paced(client, rounds, per_round, || {
            ratio_wide_kernel::launch(client, every.clone(), unit, seen.arg(), estimate.arg(), ratio.arg(), across.arg(), down.arg(), weights.arg(), w, h, floor);
            update_wide_kernel::launch(client, every.clone(), unit, ratio.arg(), estimate.arg(), across.arg(), down.arg(), weights.arg(), w, h, floor);
        });
    }
    estimate
}

/// How long the card works at a deconvolution's rounds between two waits
/// for it ([`paced`]): a frame or so. Each wait leaves the card idle for a
/// quarter of a millisecond or so: waited for every three rounds, a
/// deconvolution of a megapixel and a half took a tenth as long again.
const PACED: Duration = Duration::from_millis(16);
/// Taps of a deconvolution's rounds, a pixel's tap each, queued before it
/// is first waited for: some thirteen milliseconds of them on a card that
/// works out two thirds of a million million taps a second, as an M4 Pro
/// undoes a blur two pixels wide, of 157 taps, over sixteen megapixels.
const PACED_TAPS: usize = 1 << 33;

/// `round` queued on the card `rounds` times, each round working out
/// `taps` taps, until the work is called off. A launch returns before the
/// card begins it, so that every round would be queued before a call to
/// stop could be seen, and the card would work them all. So the rounds are
/// waited for every so often before the call is asked after: first after
/// [`PACED_TAPS`] taps of them, and from then on after as many as the card
/// took [`PACED`] over, by the time it took over those since the last wait;
/// and a deconvolution called off stops within that. A fault of the card's
/// ends the rounds, and is reported by the read of their result, as every
/// call on a device lost is.
fn paced(client: &Client, rounds: usize, taps: usize, mut round: impl FnMut()) {
    let mut every = (PACED_TAPS / taps.max(1)).max(1);
    let (mut since, mut queued) = (Instant::now(), 0);
    for _ in 0..rounds {
        if queued == every {
            if cubecl::future::block_on(client.sync()).is_err() {
                break;
            }
            let each = since.elapsed().as_nanos() / queued as u128;
            every = (PACED.as_nanos() / each.max(1)).clamp(1, rounds as u128) as usize;
            (since, queued) = (Instant::now(), 0);
        }
        if work::cancelled() {
            break;
        }
        round();
        queued += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imaging::SharedRgba;
    use crate::imaging::adjust::source_radius;
    use crate::imaging::deblur::{self, Spread};
    use crate::imaging::enhance::gaussian;
    use crate::imaging::resize::Sample;

    /// Squares of ink on paper, with grain, `width` by `height`.
    fn photo(width: usize, height: usize) -> Vec<f32> {
        let mut seed = 0x2545_f491_u32;
        (0..width * height)
            .map(|i| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let (x, y) = (i % width, i / width);
                let ink = if (x / 9 + y / 7) % 2 == 0 { 0.15 } else { 0.85 };
                (ink + ((seed >> 8) as f32 / (1 << 24) as f32 - 0.5) * 0.2).clamp(0.0, 1.0)
            })
            .collect()
    }

    /// The same, as opaque pixels in colours, four bytes to each.
    fn colours(width: usize, height: usize) -> Vec<u8> {
        photo(width, height).iter().enumerate().flat_map(|(i, &v)| [(v * 255.0) as u8, ((1.0 - v) * 200.0) as u8 + (i % 7) as u8, (v * v * 255.0) as u8, 255]).collect()
    }

    /// The values of the square `2 * radius + 1` across about the pixel
    /// `(x, y)` of `data`, `width` by `height`, the edges repeated beyond
    /// the image.
    fn square(data: &[f32], width: usize, height: usize, radius: usize, x: usize, y: usize) -> Vec<f32> {
        let r = radius as isize;
        (-r..=r)
            .flat_map(|dy| (-r..=r).map(move |dx| (dx, dy)))
            .map(|(dx, dy)| {
                let sx = (x as isize + dx).clamp(0, width as isize - 1) as usize;
                let sy = (y as isize + dy).clamp(0, height as isize - 1) as usize;
                data[sy * width + sx]
            })
            .collect()
    }

    /// Each pixel of `data` as `of` its square makes it, on the cores.
    fn each(data: &[f32], width: usize, height: usize, radius: usize, of: impl Fn(Vec<f32>) -> f32) -> Vec<f32> {
        (0..width * height).map(|i| of(square(data, width, height, radius, i % width, i / width))).collect()
    }

    /// The median, by sorting each square on the cores.
    fn median_sorted(data: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
        each(data, width, height, radius, |mut values| {
            let middle = values.len() / 2;
            *values.select_nth_unstable_by(middle, f32::total_cmp).1
        })
    }

    /// The closing, by comparing each square on the cores.
    fn closed_compared(data: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
        let darkest = each(data, width, height, radius, |values| values.into_iter().fold(f32::INFINITY, f32::min));
        each(&darkest, width, height, radius, |values| values.into_iter().fold(f32::NEG_INFINITY, f32::max))
    }

    /// The deconvolution, by adding up each pixel's taps in their order
    /// on the cores.
    fn deconvolved_summed(observed: &[f32], width: usize, height: usize, taps: &[(isize, isize, f32)], rounds: usize, floor: f32) -> Vec<f32> {
        let blurred = |source: &[f32]| -> Vec<f32> {
            (0..width * height)
                .map(|i| {
                    let (x, y) = ((i % width) as isize, (i / width) as isize);
                    taps.iter().fold(0.0f32, |sum, &(dx, dy, weight)| {
                        let sx = (x + dx).clamp(0, width as isize - 1) as usize;
                        let sy = (y + dy).clamp(0, height as isize - 1) as usize;
                        sum + weight * source[sy * width + sx]
                    })
                })
                .collect()
        };
        let mut estimate = observed.to_vec();
        for _ in 0..rounds {
            let ratio: Vec<f32> = observed.iter().zip(blurred(&estimate)).map(|(o, a)| o / a.max(floor)).collect();
            for (e, c) in estimate.iter_mut().zip(blurred(&ratio)) {
                *e = (*e * c).clamp(floor, 1.0);
            }
        }
        estimate
    }

    /// The resize, as the cores made it: the source's rows weighed down
    /// into each row of the result, and that weighed across; each value
    /// as its sum, before it is held to what a sample holds.
    fn resized_on_cores<T: Sample>(data: &[T], width: usize, channels: usize, across: &Taps, down: &Taps) -> Vec<f32> {
        let stride = width * channels;
        let mut out = Vec::new();
        for y in 0..down.first.len() {
            let mut row = vec![0.0f32; stride];
            for k in 0..down.count[y] {
                let weight = down.weights[y * down.per + k];
                for (sum, value) in row.iter_mut().zip(&data[(down.first[y] + k) * stride..]) {
                    *sum += weight * value.to_f32();
                }
            }
            for x in 0..across.first.len() {
                for c in 0..channels {
                    let mut sum = 0.0f32;
                    for k in 0..across.count[x] {
                        sum += across.weights[x * across.per + k] * row[(across.first[x] + k) * channels + c];
                    }
                    out.push(sum);
                }
            }
        }
        out
    }

    /// The lens taken out of the whole of `data`, `width` by `height` of
    /// `channels`, as the cores took it out; each value before it is held
    /// to what a sample holds.
    fn straightened_on_cores<T: Sample>(data: &[T], (width, height): (usize, usize), channels: usize, k1: f32, out_at: [f32; 2], (out_w, out_h): (usize, usize), fill: f32) -> Vec<f32> {
        let (w, h) = (width as f32, height as f32);
        let (cx, cy) = (w / 2.0, h / 2.0);
        let norm = (w * w + h * h).sqrt() / 2.0;
        let mut out = Vec::new();
        for y in 0..out_h {
            for x in 0..out_w {
                let (dx, dy) = (out_at[0] + x as f32 + 0.5 - cx, out_at[1] + y as f32 + 0.5 - cy);
                let r = (dx * dx + dy * dy).sqrt() / norm;
                let f = if r > 0.0 { source_radius(r, k1) / r } else { 1.0 };
                let (sx, sy) = (cx + dx * f - 0.5, cy + dy * f - 0.5);
                if !(sx >= -0.5 && sy >= -0.5 && sx <= w - 0.5 && sy <= h - 0.5) {
                    out.extend(std::iter::repeat_n(fill, channels));
                    continue;
                }
                let x0 = sx.floor().clamp(0.0, w - 1.0) as usize;
                let y0 = sy.floor().clamp(0.0, h - 1.0) as usize;
                let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(height - 1));
                let (tx, ty) = ((sx - x0 as f32).clamp(0.0, 1.0), (sy - y0 as f32).clamp(0.0, 1.0));
                let at = |x: usize, y: usize, c: usize| data[(y * width + x) * channels + c].to_f32();
                for c in 0..channels {
                    let top = at(x0, y0, c) * (1.0 - tx) + at(x1, y0, c) * tx;
                    let bottom = at(x0, y1, c) * (1.0 - tx) + at(x1, y1, c) * tx;
                    out.push(top * (1.0 - ty) + bottom * ty);
                }
            }
        }
        out
    }

    /// The unsharp mask, as the cores laid it: blurred across, then down,
    /// against the values themselves.
    fn unsharp_on_cores(values: &[f32], width: usize, height: usize, channels: usize, weights: &[f32], amount: f32) -> Vec<f32> {
        let reach = (weights.len() / 2) as isize;
        let at = |x: isize, y: isize| (y.clamp(0, height as isize - 1) as usize * width + x.clamp(0, width as isize - 1) as usize) * channels;
        let across: Vec<f32> = (0..width * height * channels)
            .map(|i| {
                let (x, y, c) = (((i / channels) % width) as isize, (i / channels / width) as isize, i % channels);
                weights.iter().enumerate().fold(0.0f32, |sum, (k, &weight)| sum + weight * values[at(x + k as isize - reach, y) + c])
            })
            .collect();
        (0..width * height * channels)
            .map(|i| {
                let (x, y, c) = (((i / channels) % width) as isize, (i / channels / width) as isize, i % channels);
                let blurred = weights.iter().enumerate().fold(0.0f32, |sum, (k, &weight)| sum + weight * across[at(x, y + k as isize - reach) + c]);
                values[i] + amount * (values[i] - blurred)
            })
            .collect()
    }

    fn worst(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max)
    }

    /// Whether every byte the card made is within one of what the cores'
    /// values come to as bytes: a value at a half the card made a hair
    /// off may round the other way.
    fn bytes_within_one(card: &[u8], cores: &[f32]) -> bool {
        card.len() == cores.len() && card.iter().zip(cores).all(|(&card, &cores)| card.abs_diff(u8::from_f32(cores)) <= 1)
    }

    fn values_of(made: Made) -> Vec<f32> {
        made.into_values()
    }

    fn bytes_of(made: Made) -> Vec<u8> {
        match made {
            Made::Bytes(bytes) => bytes,
            Made::Values(_) => panic!("bytes asked for"),
        }
    }

    /// Sizes about the edges of a cube's pixels and a tile's: a single
    /// pixel, a single row and a single column, and sizes that leave a
    /// part of a cube over each way.
    const SIZES: [(usize, usize); 6] = [(1, 1), (1, 37), (53, 1), (37, 23), (65, 33), (130, 97)];

    #[test]
    fn the_card_filters_as_sorting_and_adding_up_do() {
        // A round blur, one shaken at a slant, the widest the tile holds,
        // whose taps reach the whole of its halo, and one wider, whose
        // taps are read from the image itself.
        let spreads = [Spread::round(1.2), Spread { xx: 1.8, xy: 0.7, yy: 0.5 }, Spread::round(2.6), Spread::round(3.0)];
        assert_eq!(halo_of(&deblur::kernel(&spreads[2])), MOST_HALO);
        assert!(halo_of(&deblur::kernel(&spreads[3])) > MOST_HALO);
        for (width, height) in SIZES {
            let image = photo(width, height);
            for radius in 1..=2 {
                assert!(median(&image, width, height, radius) == median_sorted(&image, width, height, radius), "median of {radius} over {width} by {height}");
            }
            for radius in 1..=3 {
                assert!(closed(&image, width, height, radius) == closed_compared(&image, width, height, radius), "closed over {radius} in {width} by {height}");
            }
            let observed: Vec<f32> = image.iter().map(|v| v.clamp(deblur::FLOOR, 1.0)).collect();
            for spread in spreads {
                let taps = deblur::kernel(&spread);
                let card = deconvolved(&observed, width, height, &taps, deblur::ROUNDS, deblur::FLOOR);
                let summed = deconvolved_summed(&observed, width, height, &taps, deblur::ROUNDS, deblur::FLOOR);
                assert!(worst(&card, &summed) < 1e-5, "{spread:?} over {width} by {height}: {}", worst(&card, &summed));
                assert_eq!(deblur::deconvolve(&image, width, height, &spread), card, "the card deconvolves alike every time");
            }
        }
    }

    #[test]
    fn the_card_adjusts_as_the_cores_did() {
        for (width, height) in SIZES {
            let (grey, rgba) = (photo(width, height), colours(width, height));
            // Larger, smaller, and by other numbers each way.
            for (to_w, to_h) in [(3 * width, 3 * height), (width.div_ceil(2), height.div_ceil(3)), (width * 5 / 2 + 1, height + 4)] {
                let (across, down) = (Taps::new(width, to_w), Taps::new(height, to_h));
                let card = bytes_of(resized(Samples::Bytes(&rgba), (width, height), 4, &across, &down, true));
                assert!(bytes_within_one(&card, &resized_on_cores(&rgba, width, 4, &across, &down)), "{width} by {height} to {to_w} by {to_h}");
                let card = values_of(resized(Samples::Values(&grey), (width, height), 1, &across, &down, false));
                let cores: Vec<f32> = resized_on_cores(&grey, width, 1, &across, &down).into_iter().map(f32::from_f32).collect();
                assert!(worst(&card, &cores) < 1e-5, "{width} by {height} to {to_w} by {to_h}: {}", worst(&card, &cores));
            }
            // Barrel, none and pincushion, of the whole and of a part.
            for k1 in [-0.25f32, 0.0, 0.18] {
                let lens = Lens { frame: [width as f32, height as f32], k1, slope: (1.0 + 3.0 * k1).max(0.1) };
                for (out_at, size) in [([0.0, 0.0], (width, height)), ([1.0, 2.0], (width.div_ceil(2), height.div_ceil(2)))] {
                    let card = bytes_of(straightened(Samples::Bytes(&rgba), (width, height), 4, [[0.0; 2], [1.0; 2]], &lens, out_at, size, &[255.0; 4], true));
                    let cores = straightened_on_cores(&rgba, (width, height), 4, k1, out_at, size, 255.0);
                    assert!(bytes_within_one(&card, &cores), "{k1} over {width} by {height} at {out_at:?}");
                }
            }
            // Sharpened, and blurred below nought.
            for (sigma, amount) in [(1.0, 1.5), (2.2, -1.0)] {
                let weights = gaussian(sigma);
                let card = values_of(unsharp(Samples::Values(&grey), (width, height), [1, 1], &weights, amount, false));
                let cores = unsharp_on_cores(&grey, width, height, 1, &weights, amount);
                assert!(worst(&card, &cores) < 1e-5, "{sigma} {amount} over {width} by {height}: {}", worst(&card, &cores));
                let colour: Vec<f32> = rgba.as_chunks::<4>().0.iter().flat_map(|&[r, g, b, _]| [r, g, b].map(f32::from)).collect();
                let card = bytes_of(unsharp(Samples::Bytes(&rgba), (width, height), [4, 3], &weights, amount, true));
                let cores: Vec<f32> = unsharp_on_cores(&colour, width, height, 3, &weights, amount).as_chunks::<3>().0.iter().flat_map(|&[r, g, b]| [r, g, b, 255.0]).collect();
                assert!(bytes_within_one(&card, &cores), "{sigma} {amount} over {width} by {height}");
            }
            // Toned through a curve, in colour and by one channel of it;
            // and the grey of its luminance.
            let table: [u8; 256] = std::array::from_fn(|i| (255 - i as u8).wrapping_mul(3));
            for pick in [None, Some(1)] {
                let mut toned_rgba = rgba.clone();
                let luma = toned_within(usize::MAX, &mut toned_rgba, &table, pick, true);
                for ((pixel, shown), &value) in rgba.as_chunks::<4>().0.iter().zip(toned_rgba.as_chunks::<4>().0).zip(&luma) {
                    let curved = [0, 1, 2].map(|c| table[pixel[pick.unwrap_or(c)] as usize]);
                    assert_eq!(shown, &[curved[0], curved[1], curved[2], 255]);
                    let [r, g, b] = curved.map(|c| c as f32 / 255.0);
                    assert!((value - (0.299 * r + 0.587 * g + 0.114 * b)).abs() < 1e-6);
                }
                // Not shown: the pixels left as they were, and the same
                // luminance read.
                let mut unshown = rgba.clone();
                assert!(toned_within(usize::MAX, &mut unshown, &table, pick, false) == luma && unshown == rgba, "{pick:?} over {width} by {height}");
            }
            let mut halves = grey.clone();
            halves[0] = 0.5 / 255.0;
            halves[width * height - 1] = 1.7;
            let cores: Vec<u8> = halves.iter().flat_map(|&v| {
                let g = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                [g, g, g, 255]
            }).collect();
            assert_eq!(greyed_as_taken(&halves, width, height), cores, "the grey of {width} by {height}");
        }
    }

    /// `luma`, `width` by `height`, handed the card and taken off it in
    /// grey, as a filter laid over a part adjusted there shows it
    /// ([`Toned::taken`]).
    fn greyed_as_taken(luma: &[f32], width: usize, height: usize) -> Vec<u8> {
        let image = Image::handed(luma, width, height);
        Toned { shown: Held::empty(client(), width * height), luma: image.clone() }.taken(Some(image), true).0
    }

    #[test]
    fn a_large_image_is_worked_out_in_tiles_as_it_is_whole() {
        let (width, height) = (300, 220);
        let (grey, rgba) = (photo(width, height), colours(width, height));
        // So few values at once that the image takes many tiles, and the
        // deconvolution's are at the fewest pixels across a tile.
        let (few, all) = (1 << 13, usize::MAX);
        assert!(width * height > 8 * few, "the image is more than the card is given at once, many times over");
        type Filter<'a> = dyn Fn(&Client, &Held, (usize, usize)) -> Held + 'a;
        let both = |reach: usize, filter: &Filter<'_>| {
            (filtered_within(few, &grey, width, height, reach, filter), filtered_within(all, &grey, width, height, reach, filter))
        };
        let (tiled, whole) = both(2, &|client, piece, size| median_on(client, piece, size, 2));
        assert!(tiled == whole, "median");
        let (tiled, whole) = both(6, &|client, piece, size| closed_on(client, piece, size, 3));
        assert!(tiled == whole, "closed");
        // In the tile and out of it, and every round of a deconvolution.
        for (spread, rounds) in [(Spread::round(0.5), deblur::ROUNDS), (Spread::round(1.0), 4), (Spread::round(3.0), 2)] {
            let taps = deblur::kernel(&spread);
            let (tiled, whole) = both(2 * rounds * halo_of(&taps), &|client, piece, size| deconvolved_on(client, piece, size, &taps, rounds, deblur::FLOOR));
            assert!(tiled == whole, "{spread:?} over {rounds} rounds: {}", worst(&tiled, &whole));
        }
        // Resized larger in colour, smaller in grey, and by a fraction.
        for (to_w, to_h) in [(3 * width, 3 * height), (width / 3, height / 2), (width * 5 / 4, height * 3 / 4)] {
            let (across, down) = (Taps::new(width, to_w), Taps::new(height, to_h));
            let tiled = bytes_of(resized_within(few, Samples::Bytes(&rgba), (width, height), 4, &across, &down, true));
            assert!(tiled == bytes_of(resized_within(all, Samples::Bytes(&rgba), (width, height), 4, &across, &down, true)), "{to_w} by {to_h}");
            let tiled = values_of(resized_within(few, Samples::Values(&grey), (width, height), 1, &across, &down, false));
            assert!(tiled == values_of(resized_within(all, Samples::Values(&grey), (width, height), 1, &across, &down, false)), "{to_w} by {to_h}");
        }
        // Straightened, pincushion and barrel, of a frame enlarged twice,
        // from a part of the photo as shot beginning a little way into it.
        for k1 in [0.2f32, -0.25] {
            let lens = Lens { frame: [2.0 * width as f32, 2.0 * height as f32], k1, slope: (1.0 + 3.0 * k1).max(0.1) };
            let shot = [[3.0, 5.0], [2.0, 2.0]];
            let straight = |most: usize| bytes_of(straightened_within(most, Samples::Bytes(&rgba), (width, height), 4, shot, &lens, [40.0, 30.0], (500, 380), &[255.0; 4], true));
            assert!(straight(few) == straight(all), "straightened by {k1}");
        }
        // Sharpened in colour and blurred in grey.
        let weights = gaussian(1.7);
        let sharp = |most: usize| bytes_of(unsharp_within(most, Samples::Bytes(&rgba), (width, height), [4, 3], &weights, 1.2, true));
        assert!(sharp(few) == sharp(all), "sharpened");
        let blurred = |most: usize| values_of(unsharp_within(most, Samples::Values(&grey), (width, height), [1, 1], &weights, -1.0, false));
        assert!(blurred(few) == blurred(all), "blurred");
        // Toned a run of pixels at a time.
        let table: [u8; 256] = std::array::from_fn(|i| (i as u8) / 2 + 60);
        let (mut runs, mut once) = (rgba.clone(), rgba.clone());
        assert!(toned_within(1000, &mut runs, &table, None, true) == toned_within(all, &mut once, &table, None, true));
        assert!(runs == once);
    }

    #[test]
    fn a_part_is_adjusted_on_the_card_whole_as_it_is_a_step_at_a_time() {
        let (width, height) = (150, 110);
        let rgba = colours(width, height);
        let table: [u8; 256] = std::array::from_fn(|i| 255 - i as u8 / 3);
        // Every chain of steps, for a part `width` by `height`.
        let chains = |(width, height): (usize, usize)| -> Vec<Vec<Step>> {
            let lens = |frame: [f32; 2]| Lens { frame, k1: -0.15, slope: 0.55 };
            let resized = || Step::Resized { across: Taps::new(width, 2 * width), down: Taps::new(height, 2 * height) };
            let reduced = || Step::Resized { across: Taps::new(width, width / 2), down: Taps::new(height, height / 2) };
            let bent = |size: (usize, usize), scale: f32| Step::Straightened { place: [[0.0; 2], [scale; 2]], lens: lens([size.0 as f32 * scale, size.1 as f32 * scale]), out_at: [4.0, 3.0], size };
            let sharpened = || Step::Sharpened { weights: gaussian(1.4), amount: 1.1 };
            let blurred = || Step::Sharpened { weights: gaussian(0.9), amount: -1.0 };
            vec![
                vec![],
                vec![resized()],
                vec![reduced(), blurred()],
                vec![bent((width - 6, height - 5), 1.0)],
                vec![sharpened()],
                vec![resized(), bent((2 * width - 9, 2 * height - 7), 2.0), sharpened()],
            ]
        };
        // The whole photo handed the card; a part of it handed it; and the
        // same part cut on the card from the photo it holds.
        let shared: SharedRgba = Arc::new(rgba.clone());
        let part = [7, 5, width - 3, height - 2];
        let sources = [
            Source { rgba: &rgba, width, part: [0, 0, width, height], shared: None },
            Source { rgba: &rgba, width, part, shared: None },
            Source { rgba: &shared, width, part, shared: Some(&shared) },
        ];
        for pick in [None, Some(2)] {
            for (s, source) in sources.iter().enumerate() {
                for (i, steps) in chains(source.size()).iter().enumerate() {
                    let on_card = |shown: bool| adjusted_within(usize::MAX, source, steps, &table, pick).expect("the part fits the card whole").taken(None, shown);
                    let whole = on_card(true);
                    // A step at a time, through the cores' memory, and in
                    // tiles.
                    let stepped = stepped_within(1 << 12, source, steps, &table, pick, true);
                    assert!(whole == stepped, "chain {i}, {pick:?}, source {s}");
                    // And not shown, the same luminance alone, both ways.
                    let unshown = (Vec::new(), whole.1.clone());
                    assert!(on_card(false) == unshown && stepped_within(1 << 12, source, steps, &table, pick, false) == unshown, "chain {i}, {pick:?}, source {s}, not shown");
                }
            }
            let made = |source: &Source<'_>, steps: &[Step]| adjusted_within(usize::MAX, source, steps, &table, pick).expect("fits").taken(None, true);
            for (i, steps) in chains(sources[1].size()).iter().enumerate() {
                assert!(made(&sources[1], steps) == made(&sources[2], steps), "chain {i}, {pick:?}: a part cut on the card is the part handed it");
            }
        }
    }

    #[test]
    fn a_photo_nothing_holds_is_let_go_by_the_next_work_on_the_card() {
        let shared: SharedRgba = Arc::new(colours(40, 30));
        let held = Arc::downgrade(&shared);
        photo_on(client(), &shared);
        drop(shared);
        // Any work on the card, and not another photo handed it: here a
        // filter of an image of a channel.
        median(&photo(8, 8), 8, 8, 1);
        let photos = PHOTOS.lock().expect("nothing panics holding the photos");
        assert!(!photos.iter().any(|(photo, _)| Weak::ptr_eq(photo, &held)), "the photo is still held on the card");
    }

    #[test]
    fn a_filter_laid_over_an_image_on_the_card_is_the_filter_of_it_handed_the_card() {
        let (width, height) = (150, 110);
        let grey = photo(width, height);
        let image = Image::of(Held::of(client(), &grey), (width, height));
        for radius in 1..=2 {
            assert!(image.median(radius).read() == median(&grey, width, height, radius), "median of {radius}");
        }
        for radius in 1..=3 {
            assert!(image.closed(radius).read() == closed(&grey, width, height, radius), "closed over {radius}");
        }
        let observed: Vec<f32> = grey.iter().map(|v| v.clamp(deblur::FLOOR, 1.0)).collect();
        for spread in [Spread::round(1.4), Spread::round(3.0)] {
            let taps = deblur::kernel(&spread);
            let card = image.deconvolved(&taps, deblur::ROUNDS, deblur::FLOOR).read();
            assert!(card == deconvolved(&observed, width, height, &taps, deblur::ROUNDS, deblur::FLOOR), "{spread:?}");
        }
        // Carried past its edges, and cut short; resized larger and smaller.
        let carried = image.carried(width + 9, height + 4).read();
        assert!(carried == (0..height + 4).flat_map(|y| (0..width + 9).map(move |x| (x.min(width - 1), y.min(height - 1)))).map(|(x, y)| grey[y * width + x]).collect::<Vec<f32>>());
        assert!(image.carried(width - 9, height - 4).read() == cut(&grey, width, 1, [0, 0, width - 9, height - 4]));
        for (to_w, to_h) in [(2 * width, 2 * height), (width / 3, height / 2)] {
            let (across, down) = (Taps::new(width, to_w), Taps::new(height, to_h));
            let whole = values_of(resized(Samples::Values(&grey), (width, height), 1, &across, &down, false));
            assert!(image.resized(to_w, to_h).read() == whole, "{to_w} by {to_h}");
            // And only the part of that from its corner made.
            let part = [0, 0, to_w - 7, to_h - 3];
            assert!(image.resized_part((to_w, to_h), (part[2], part[3])).read() == cut(&whole, to_w, 1, part), "{to_w} by {to_h}, a part");
        }
        // Sharpened and blurred, and a part of that held to black and white.
        for (sigma, amount) in [(1.3, 2.5), (2.0, -1.0)] {
            let weights = gaussian(sigma);
            let sharp = image.sharpened(&weights, amount);
            let cores = values_of(unsharp(Samples::Values(&grey), (width, height), [1, 1], &weights, amount, false));
            assert!(sharp.read() == cores, "{sigma} {amount}");
            let part = [5, 3, width - 8, height - 1];
            let held: Vec<f32> = cut(&cores, width, 1, part).into_iter().map(|v| v.clamp(0.0, 1.0)).collect();
            assert!(sharp.held_part(part).read() == held, "{sigma} {amount}, a part held");
        }
        // Several steps on the card at once, taken off it once.
        let steps = |image: Image| image.resized(2 * width, 2 * height).sharpened(&gaussian(1.0), 1.5).held_part([4, 4, 2 * width - 4, 2 * height - 4]);
        assert!(through(&grey, (width, height), (2 * width, 2 * height), steps) == Some(steps(Image::handed(&grey, width, height)).read()));
        assert!(through(&grey, (width, height), (PIECE_SIDE + 1, 1), steps).is_none(), "an image along the way too wide for the card whole");
        // The grey of a luminance filtered there, shown as a filter shows the photo.
        let rgba = colours(width, height);
        let source = Source { rgba: &rgba, width, part: [0, 0, width, height], shared: None };
        let toned = || adjusted_within(usize::MAX, &source, &[], &std::array::from_fn(|i| i as u8), None).expect("fits");
        let toned = (toned(), toned());
        let filtered = toned.0.luma.median(1);
        let luma = filtered.read();
        let (shown, read) = toned.0.taken(Some(filtered.clone()), true);
        let mut grey = vec![0u8; width * height * 4];
        crate::imaging::greyed(&luma, &mut grey);
        assert!(shown == grey && read == luma);
        // And not shown, the luminance alone.
        assert!(toned.1.taken(Some(filtered), false) == (Vec::new(), luma));
    }

    #[test]
    fn the_median_networks_leave_the_median_in_the_middle() {
        // By the zero-one principle, a network of compare-swaps leaves the
        // median of every input in the middle place where it does so for
        // every input of noughts and ones: here every one of those, a bit
        // of each word for each of sixty-four inputs side by side.
        for (network, count) in [(&MEDIAN_OF_9[..], 9usize), (&MEDIAN_OF_25[..], 25)] {
            let middle = count / 2;
            // The inputs of each of a word's bits, by its first six places.
            let low = [0xAAAA_AAAA_AAAA_AAAA_u64, 0xCCCC_CCCC_CCCC_CCCC, 0xF0F0_F0F0_F0F0_F0F0, 0xFF00_FF00_FF00_FF00, 0xFFFF_0000_FFFF_0000, 0xFFFF_FFFF_0000_0000];
            // Which of a word's inputs have more than `t` ones among those.
            let above = |t: i64| (0..64u64).filter(|b| i64::from(b.count_ones()) > t).fold(0u64, |word, b| word | 1 << b);
            for word in 0..1u64 << (count - 6) {
                let mut places: Vec<u64> = (0..count).map(|i| if i < 6 { low[i] } else if word >> (i - 6) & 1 == 1 { u64::MAX } else { 0 }).collect();
                for &(i, j) in network {
                    (places[i], places[j]) = (places[i] & places[j], places[i] | places[j]);
                }
                assert_eq!(places[middle], above(middle as i64 - i64::from(word.count_ones())), "of {count}, the inputs of word {word}");
            }
        }
    }

    #[test]
    fn a_deconvolution_called_off_stops_part_way() {
        let (width, height) = (2048, 2048);
        let observed: Vec<f32> = photo(width, height).iter().map(|v| v.clamp(deblur::FLOOR, 1.0)).collect();
        let taps = deblur::kernel(&Spread::round(2.0));
        // The kernels made first, so that the deconvolutions timed are all
        // rounds; and as many rounds as take the card ten times as long as
        // it is left to work between waits, however fast it is.
        deconvolved(&observed[..64 * 64], 64, 64, &taps, 1, deblur::FLOOR);
        let begun = Instant::now();
        deconvolved(&observed, width, height, &taps, deblur::ROUNDS, deblur::FLOOR);
        let rounds = ((10 * PACED).as_secs_f64() / begun.elapsed().as_secs_f64() * deblur::ROUNDS as f64).ceil().max(deblur::ROUNDS as f64) as usize;
        let begun = Instant::now();
        let all = deconvolved(&observed, width, height, &taps, rounds, deblur::FLOOR);
        let taken = begun.elapsed();
        // Called off a fifth of the way in, it stops within a wait or so,
        // where every round had been queued by then.
        let cancel = work::Cancel::new();
        let (asked, observed) = (cancel.clone(), &observed);
        let part = std::thread::scope(|scope| {
            let worker = scope.spawn(move || asked.scope(|| deconvolved(observed, width, height, &taps, rounds, deblur::FLOOR)));
            std::thread::sleep(taken / 5);
            cancel.cancel();
            worker.join().expect("the deconvolution")
        });
        assert!(part != all, "called off at {:?} of {taken:?}", taken / 5);
    }

    #[test]
    fn an_image_wider_than_a_tile_is_worked_out_in_tiles() {
        let (width, height) = (PIECE_SIDE + 617, 3);
        assert_eq!(fewest_tiles(width, height, PIECE_SIDE, usize::MAX, area), [2, 1]);
        let image = photo(width, height);
        assert!(median(&image, width, height, 1) == median_sorted(&image, width, height, 1));
        assert!(closed(&image, width, height, 2) == closed_compared(&image, width, height, 2));
        let rgba = colours(width, height);
        let (across, down) = (Taps::new(width, width / 2), Taps::new(height, 2 * height));
        let card = bytes_of(resized(Samples::Bytes(&rgba), (width, height), 4, &across, &down, true));
        assert!(bytes_within_one(&card, &resized_on_cores(&rgba, width, 4, &across, &down)));
    }

    #[test]
    fn a_filter_is_worked_out_in_tiles_a_launch_reaches_across_with_what_they_draw_on() {
        // Twice a tile across or down: two tiles as wide as a launch
        // reaches would each be worked out with the pixels about them,
        // wider than that, and took more cubes than a launch may have.
        for (width, height) in [(2 * PIECE_SIDE, 2), (2, 2 * PIECE_SIDE)] {
            let image = photo(width, height);
            assert!(median(&image, width, height, 1) == median_sorted(&image, width, height, 1), "median over {width} by {height}");
            assert!(closed(&image, width, height, 2) == closed_compared(&image, width, height, 2), "closed over {width} by {height}");
            let weights = gaussian(1.0);
            let card = values_of(unsharp(Samples::Values(&image), (width, height), [1, 1], &weights, 1.5, false));
            let cores = unsharp_on_cores(&image, width, height, 1, &weights, 1.5);
            assert!(worst(&card, &cores) < 1e-5, "sharpened over {width} by {height}: {}", worst(&card, &cores));
        }
    }

    #[test]
    fn tiles_work_out_every_pixel_once() {
        for (width, height) in [(1, 1), (5000, 3), (3, 70_000), (900, 700), (16_384, 16_384), (PIECE_SIDE + 1, 2)] {
            for grid in [[1, 1], [2, 1], [7, 3], [width / LEAST_SIDE, height / LEAST_SIDE]] {
                let [across, down] = [grid[0].clamp(1, width), grid[1].clamp(1, height)];
                let tiles: Vec<Part> = tiles(width, height, [across, down]).collect();
                assert_eq!(tiles.len(), across * down);
                assert_eq!(tiles.iter().map(|&tile| area(tile)).sum::<usize>(), width * height, "{width} by {height}, {across} by {down}");
                for &[x0, y0, x1, y1] in &tiles {
                    assert!(x0 < x1 && y0 < y1 && x1 <= width && y1 <= height);
                    assert!((width / across..=width.div_ceil(across)).contains(&(x1 - x0)) && (height / down..=height.div_ceil(down)).contains(&(y1 - y0)));
                }
                if width * height <= 1 << 20 {
                    let mut times = vec![0u8; width * height];
                    for tile in &tiles {
                        for y in tile[1]..tile[3] {
                            times[y * width + tile[0]..y * width + tile[2]].iter_mut().for_each(|t| *t += 1);
                        }
                    }
                    assert!(times.iter().all(|&t| t == 1));
                }
            }
        }
    }

    #[test]
    fn an_image_is_worked_out_in_the_fewest_tiles_that_fit() {
        for (width, height) in [(900, 600), (1000, 1000), (5000, 70), (130, 4100)] {
            for reach in [0, 10, 75] {
                let holds = |tile| area(about(tile, reach, width, height)) * 3;
                for most in [usize::MAX, 3 * width * height, width * height, width * height / 10, 20_000] {
                    let fits = |grid| tiles(width, height, grid).all(|tile| holds(tile) <= most);
                    // Every grid there may be, each tile at least the
                    // fewest pixels either way, and the fewest tiles of
                    // those that fit.
                    let finest = [(width / LEAST_SIDE).max(1), (height / LEAST_SIDE).max(1)];
                    let fewest = (1..=finest[0]).flat_map(|across| (1..=finest[1]).map(move |down| [across, down])).filter(|&grid| fits(grid)).map(|[across, down]| across * down).min();
                    let grid = fewest_tiles(width, height, PIECE_SIDE, most, holds);
                    let case = format!("{width} by {height}, reaching {reach}, {most} at once: {grid:?}");
                    match fewest {
                        Some(count) => assert!(fits(grid) && grid[0] * grid[1] == count, "{case}, where {count} fit"),
                        None => assert_eq!(grid, finest, "{case}, where none fit"),
                    }
                }
            }
        }
        // The whole where it fits; a third of it at once in three tiles,
        // where halving the side of a square tile came to twelve; and of
        // as few tiles that fit, those that read the least about them
        // twice, here squares rather than strips.
        assert_eq!(fewest_tiles(900, 600, PIECE_SIDE, 900 * 600, area), [1, 1]);
        let [across, down] = fewest_tiles(900, 600, PIECE_SIDE, 900 * 600 / 3, area);
        assert_eq!(across * down, 3);
        let reaching = |tile| area(about(tile, 10, 1000, 1000));
        assert!([[1, 4], [4, 1]].iter().all(|&strips| tiles(1000, 1000, strips).all(|tile| reaching(tile) <= 270_000)));
        assert_eq!(fewest_tiles(1000, 1000, PIECE_SIDE, 270_000, reaching), [2, 2]);
        // And none wider or taller than a tile may be, however much fits.
        assert_eq!(fewest_tiles(1000, 1000, 400, usize::MAX, area), [3, 3]);
    }

    #[test]
    fn an_image_with_no_pixels_comes_back_empty() {
        for (width, height) in [(0, 0), (0, 5), (5, 0)] {
            assert!(median(&[], width, height, 1).is_empty());
            assert!(closed(&[], width, height, 1).is_empty());
            assert!(deconvolved(&[], width, height, &deblur::kernel(&Spread::round(1.0)), 3, 0.01).is_empty());
            assert!(values_of(unsharp(Samples::Values(&[]), (width, height), [1, 1], &gaussian(1.0), 1.5, false)).is_empty());
        }
        assert!(toned_within(usize::MAX, &mut [], &[0; 256], None, true).is_empty());
    }
}
