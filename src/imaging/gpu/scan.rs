//! The finder scan's view of an image, on the graphics card: each way the
//! scan reads the image as dark and light ([`super::super::binarize`]),
//! with the closings its regions are labelled from
//! ([`super::super::regions`]).
//!
//! A frame of a whole photo of twelve megapixels is some twenty million
//! pixels, and on the cores each of them was read against the blocks
//! about it twice over, and against four windows, each a pass of one core
//! over the lot: a third of a second and more before the scan of the rows
//! began, with the other cores idle. Here each pixel of each reading is
//! the same small piece of arithmetic, side by side, and what is taken
//! off the card is the readings packed together, as few bits to each as
//! it reads in ([`slots`]). The frame itself is resampled on the cores, in
//! double precision ([`super::super::detect`]): on the card, in single, a
//! place some four thousand pixels into a photo is put a few
//! ten-thousandths of a pixel off, which moved the finders of a symbol on
//! a bottle.
//!
//! The card's arithmetic is single precision, where the cores added up a
//! block's values and its neighbourhood's in double: a mean comes out
//! within a few parts in ten million of what they made of it, and a pixel
//! that close to its threshold may read the other way. Each value is
//! worked out from the same values in the same order wherever it lies,
//! so that a tile reads as the whole does there.
//!
//! The card's shaders are compiled for fast arithmetic, which may take a
//! value for a number whatever it is. So a pixel the frame has no photo
//! for is told by its bits, which are a NaN's, and never by comparing
//! the value: the image is held as the bits of its values, as the cores
//! hold them, NaN and all.

use super::*;
use crate::imaging::binarize::{BLOCK, DARK, LIGHT, NONE};

/// Above the reading of a pixel ([`LIGHT`], [`DARK`], [`NONE`]), where a
/// reading is closed: the pixel dark once the dark
/// pixels are closed over ([`super::super::regions`]), and light once the
/// light ones are.
pub(crate) const CLOSED_DARK: u8 = 4;
pub(crate) const CLOSED_LIGHT: u8 = 8;
/// Darker and lighter than any value: what a block with no values reads
/// as at its darkest and its lightest, and a threshold no value is below.
const FAR: f32 = 1.0e30;

/// Whether `bits` are those of a value, and not of a NaN or an infinity:
/// what [`f32::is_finite`] says of them.
#[cube]
fn has(bits: u32) -> bool {
    (bits & 0x7f80_0000u32) != 0x7f80_0000u32
}

/// A pixel's reading as it is written ([`Rule`]): as it is, or where
/// `shown`, as an opaque pixel the binarised view shows, black where it
/// reads dark and white elsewhere.
#[cube]
fn written(kind: u32, #[comptime] shown: bool) -> u32 {
    let mut word = kind;
    if comptime!(shown) {
        word = 0xffff_ffffu32;
        if kind == DARK as u32 {
            word = 0xff00_0000u32;
        }
    }
    word
}

/// Each pixel of an image `width` by `height` against one `threshold`:
/// dark below it, light otherwise, and nothing where there is no value;
/// written `across` words to a row ([`written`]).
#[cube(launch)]
fn global_kernel(values: &[u32], kinds: &mut [u32], threshold: f32, width: u32, height: u32, across: u32, #[comptime] shown: bool) {
    let (x, y) = (ABSOLUTE_POS_X, ABSOLUTE_POS_Y);
    if x < width && y < height {
        let bits = values[(y * width + x) as usize];
        let mut kind = (NONE as u32).runtime();
        if has(bits) {
            kind = LIGHT as u32;
            if f32::reinterpret(bits) < threshold {
                kind = DARK as u32;
            }
        }
        kinds[(y * across + x) as usize] = written(kind, shown);
    }
}

/// The first half of the windows' means: each pixel's sum of the values of
/// its row `radius` either side of it, within the image, and how many
/// there are, the pixels with no value counting for nothing. Launched
/// unchecked ([`read_into`]).
#[cube(launch_unchecked)]
fn window_across_kernel(values: &[u32], sums: &mut [f32], counts: &mut [f32], width: u32, height: u32, radius: u32) {
    let (x, y) = (ABSOLUTE_POS_X, ABSOLUTE_POS_Y);
    if x < width && y < height {
        let row = y * width;
        let mut sum = 0.0f32;
        let mut count = 0.0f32;
        for k in max(x, radius) - radius..min(x + radius + 1, width) {
            let bits = values[(row + k) as usize];
            if has(bits) {
                sum += f32::reinterpret(bits);
                count += 1.0f32;
            }
        }
        sums[(row + x) as usize] = sum;
        counts[(row + x) as usize] = count;
    }
}

/// The second: those sums added up down the column `radius` either side of
/// each pixel, and the pixel dark where it lies more than `bias` below their
/// mean, light otherwise, and nothing where it has no value, as
/// [`super::super::binarize::window_dark`] reads it; written `across`
/// words to a row ([`written`]). Launched unchecked ([`read_into`]).
#[cube(launch_unchecked)]
fn window_kernel(values: &[u32], sums: &[f32], counts: &[f32], kinds: &mut [u32], width: u32, height: u32, radius: u32, bias: f32, across: u32, #[comptime] shown: bool) {
    let (x, y) = (ABSOLUTE_POS_X, ABSOLUTE_POS_Y);
    if x < width && y < height {
        let i = y * width + x;
        let bits = values[i as usize];
        let mut kind = (NONE as u32).runtime();
        if has(bits) {
            let mut sum = 0.0f32;
            let mut count = 0.0f32;
            for k in max(y, radius) - radius..min(y + radius + 1, height) {
                sum += sums[(k * width + x) as usize];
                count += counts[(k * width + x) as usize];
            }
            kind = LIGHT as u32;
            if count > 0.5f32 && f32::reinterpret(bits) < sum / count - bias {
                kind = DARK as u32;
            }
        }
        kinds[(y * across + x) as usize] = written(kind, shown);
    }
}

/// Of each block [`BLOCK`] pixels square of an image `width` by `height`,
/// `blocks_wide` across and `blocks_high` down, cut short by the image's
/// far edges: the mean of its values, or nought, whether it has any, one or
/// nought, and the darkest and lightest of them, or [`FAR`] each way where
/// there are none; four values to a block.
#[cube(launch)]
fn blocks_kernel(values: &[u32], blocks: &mut [f32], width: u32, height: u32, blocks_wide: u32, blocks_high: u32) {
    let (bx, by) = (ABSOLUTE_POS_X, ABSOLUTE_POS_Y);
    if bx < blocks_wide && by < blocks_high {
        let mut sum = 0.0f32;
        let mut count = 0u32;
        let mut darkest = FAR.runtime();
        let mut lightest = (-FAR).runtime();
        for y in by * BLOCK as u32..min((by + 1) * BLOCK as u32, height) {
            for x in bx * BLOCK as u32..min((bx + 1) * BLOCK as u32, width) {
                let bits = values[(y * width + x) as usize];
                if has(bits) {
                    let value = f32::reinterpret(bits);
                    sum += value;
                    count += 1;
                    darkest = min(darkest, value);
                    lightest = max(lightest, value);
                }
            }
        }
        let at = ((by * blocks_wide + bx) * 4) as usize;
        let mut mean = 0.0f32;
        let mut some = 0.0f32;
        if count > 0 {
            mean = sum / count as f32;
            some = 1.0f32;
        }
        blocks[at] = mean;
        blocks[at + 1] = some;
        blocks[at + 2] = darkest;
        blocks[at + 3] = lightest;
    }
}

/// The first half of the neighbourhood of each block: the sum of the means
/// of the blocks of its row `radius` either side of it, within the image,
/// and how many of them have values; and the darkest and lightest of it
/// and the block either side. Four values to a block.
#[cube(launch)]
fn blocks_across_kernel(blocks: &[f32], across: &mut [f32], blocks_wide: u32, blocks_high: u32, radius: u32) {
    let (bx, by) = (ABSOLUTE_POS_X, ABSOLUTE_POS_Y);
    if bx < blocks_wide && by < blocks_high {
        let row = by * blocks_wide;
        let mut sum = 0.0f32;
        let mut some = 0.0f32;
        for k in max(bx, radius) - radius..min(bx + radius + 1, blocks_wide) {
            sum += blocks[((row + k) * 4) as usize];
            some += blocks[((row + k) * 4 + 1) as usize];
        }
        let mut darkest = FAR.runtime();
        let mut lightest = (-FAR).runtime();
        for k in max(bx, 1) - 1..min(bx + 2, blocks_wide) {
            darkest = min(darkest, blocks[((row + k) * 4 + 2) as usize]);
            lightest = max(lightest, blocks[((row + k) * 4 + 3) as usize]);
        }
        let at = ((row + bx) * 4) as usize;
        across[at] = sum;
        across[at + 1] = some;
        across[at + 2] = darkest;
        across[at + 3] = lightest;
    }
}

/// The second: the threshold of each block, the mean of the blocks
/// `radius` either side of it down its column of those sums, where they
/// have values, and the block and its eight neighbours spread at least
/// `min_range` from their darkest to their lightest; and otherwise
/// [`FAR`] below any value, so that a flat block reads light, as
/// [`super::super::binarize`] reads one.
#[cube(launch)]
fn blocks_down_kernel(across: &[f32], thresholds: &mut [f32], blocks_wide: u32, blocks_high: u32, radius: u32, min_range: f32) {
    let (bx, by) = (ABSOLUTE_POS_X, ABSOLUTE_POS_Y);
    if bx < blocks_wide && by < blocks_high {
        let mut sum = 0.0f32;
        let mut some = 0.0f32;
        for k in max(by, radius) - radius..min(by + radius + 1, blocks_high) {
            sum += across[((k * blocks_wide + bx) * 4) as usize];
            some += across[((k * blocks_wide + bx) * 4 + 1) as usize];
        }
        let mut darkest = FAR.runtime();
        let mut lightest = (-FAR).runtime();
        for k in max(by, 1) - 1..min(by + 2, blocks_high) {
            darkest = min(darkest, across[((k * blocks_wide + bx) * 4 + 2) as usize]);
            lightest = max(lightest, across[((k * blocks_wide + bx) * 4 + 3) as usize]);
        }
        let mut range = 0.0f32;
        if lightest > -0.5f32 * FAR && darkest < 0.5f32 * FAR {
            range = lightest - darkest;
        }
        let mut threshold = (-FAR).runtime();
        if range >= min_range && some > 0.5f32 {
            threshold = sum / some;
        }
        thresholds[(by * blocks_wide + bx) as usize] = threshold;
    }
}

/// Each pixel against the threshold of its block: dark below it, light
/// otherwise, and nothing where it has no value; written `across` words
/// to a row ([`written`]).
#[cube(launch)]
fn blocks_read_kernel(values: &[u32], thresholds: &[f32], kinds: &mut [u32], width: u32, height: u32, blocks_wide: u32, across: u32, #[comptime] shown: bool) {
    let (x, y) = (ABSOLUTE_POS_X, ABSOLUTE_POS_Y);
    if x < width && y < height {
        let bits = values[(y * width + x) as usize];
        let mut kind = (NONE as u32).runtime();
        if has(bits) {
            kind = LIGHT as u32;
            if f32::reinterpret(bits) < thresholds[((y / BLOCK as u32) * blocks_wide + x / BLOCK as u32) as usize] {
                kind = DARK as u32;
            }
        }
        kinds[(y * across + x) as usize] = written(kind, shown);
    }
}

/// The first half of a closing of the dark pixels and of the light ones:
/// whether each pixel has a dark one in the square three across about it,
/// within the image, and whether a light one; two bits.
#[cube(launch_unchecked)]
fn grown_kernel(kinds: &[u32], grown: &mut [u32], width: u32, height: u32) {
    let (x, y) = (ABSOLUTE_POS_X, ABSOLUTE_POS_Y);
    if x < width && y < height {
        let mut bits = 0u32;
        for sy in max(y, 1) - 1..min(y + 2, height) {
            for sx in max(x, 1) - 1..min(x + 2, width) {
                let kind = kinds[(sy * width + sx) as usize] & 3u32;
                if kind == DARK as u32 {
                    bits |= 1u32;
                }
                if kind == LIGHT as u32 {
                    bits |= 2u32;
                }
            }
        }
        grown[(y * width + x) as usize] = bits;
    }
}

/// The second: each pixel dark once the dark pixels are closed over where
/// the whole square three across about it, within the image, has a dark
/// pixel within its own square, and light once the light ones are so;
/// set beside the pixel's own reading ([`CLOSED_DARK`], [`CLOSED_LIGHT`]).
#[cube(launch_unchecked)]
fn closed_kernel(kinds: &mut [u32], grown: &[u32], width: u32, height: u32) {
    let (x, y) = (ABSOLUTE_POS_X, ABSOLUTE_POS_Y);
    if x < width && y < height {
        let mut all = 3u32;
        for sy in max(y, 1) - 1..min(y + 2, height) {
            for sx in max(x, 1) - 1..min(x + 2, width) {
                all &= grown[(sy * width + sx) as usize];
            }
        }
        let i = (y * width + x) as usize;
        kinds[i] |= all << 2u32;
    }
}

/// The readings of the box `box_w` across, `count` pixels in all,
/// beginning at `(x0, y0)` of an image `width` across, packed into its
/// slot of a plane ([`Slot`]): a byte to a pixel and four to a word, row
/// after row of the box, the reading `shift` bits up each byte; or, where
/// `grey`, the whole byte black where the pixel reads dark and white
/// otherwise, as the binarised view shows it. The reading at the foot of
/// each byte writes the plane's words, and each above it adds to them.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn packed_kinds_kernel(kinds: &[u32], words: &mut [u32], width: u32, x0: u32, y0: u32, box_w: u32, count: u32, shift: u32, #[comptime] grey: bool) {
    let w = line_place();
    if w * 4 < count {
        let mut word = 0u32;
        if shift > 0 {
            word = words[w as usize];
        }
        #[unroll]
        for b in 0..4u32 {
            let p = w * 4 + b;
            if p < count {
                let kind = kinds[((y0 + p / box_w) * width + x0 + p % box_w) as usize];
                let mut byte = kind & 15u32;
                if comptime!(grey) {
                    byte = 255u32;
                    if (kind & 3u32) == DARK as u32 {
                        byte = 0u32;
                    }
                }
                word |= byte << (b * 8 + shift);
            }
        }
        words[w as usize] = word;
    }
}

/// The readings of the box `box_w` by `box_h` beginning at `(x0, y0)` of
/// an image `width` across, turned over its diagonal: each column of the
/// box a row, a byte to a pixel and four to a word, `across` words to a
/// row, packed into its slot of a plane `shift` bits up each byte, as
/// [`packed_kinds_kernel`] packs them. Units side by side read a row of
/// the box together, each its own column, and each writes four pixels
/// down that column as a word.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn turned_kinds_kernel(kinds: &[u32], words: &mut [u32], width: u32, x0: u32, y0: u32, box_w: u32, box_h: u32, across: u32, shift: u32) {
    let (column, w) = (ABSOLUTE_POS_X, ABSOLUTE_POS_Y);
    if column < box_w && w < across {
        let at = (column * across + w) as usize;
        let mut word = 0u32;
        if shift > 0 {
            word = words[at];
        }
        #[unroll]
        for b in 0..4u32 {
            let row = w * 4 + b;
            if row < box_h {
                word |= (kinds[((y0 + row) * width + x0 + column) as usize] & 15u32) << (b * 8 + shift);
            }
        }
        words[at] = word;
    }
}

/// What `rule` reads `image`, an image of a channel on the card, as, to be
/// shown, left on the card: black where it reads dark, white elsewhere,
/// written so by the reading itself, its rows laid out as a copy to a
/// texture wants them ([`OnCard::padded`]). `None` where the card is not
/// given to hold the image whole, which is then read a tile at a time and
/// taken off it ([`readings`]).
pub(crate) fn reading_shown(image: &Image, rule: Rule) -> Option<OnCard> {
    let (width, height) = (image.width, image.height);
    let across = super::shown::stride(width);
    if across * height > most_values(client()) || width.max(height) > PIECE_SIDE {
        return None;
    }
    let client = client();
    let words = Held::empty(client, across * height);
    read_into(client, &image.values, (width, height), rule, &words, across, true);
    Some(OnCard::padded(words, (width, height)))
}

/// A way the scan reads an image of a channel as dark and light
/// ([`super::super::binarize::Binarizer`]), as the card is given it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Rule {
    /// Dark below one `threshold`.
    Global { threshold: f32 },
    /// Dark below the mean of the blocks `radius` blocks about the
    /// pixel's own, where that block and its eight neighbours spread at
    /// least `min_range`.
    Local { radius: usize, min_range: f32 },
    /// Dark more than `bias` below the mean of the window `radius` pixels
    /// either side of the pixel.
    Window { radius: usize, bias: f32 },
}

impl Rule {
    /// How far from a pixel its reading draws on the image, and for a
    /// reading by blocks, how far from the block the pixel is in.
    fn reach(self) -> usize {
        match self {
            Rule::Global { .. } => 0,
            Rule::Window { radius, .. } => radius,
            // The blocks of the neighbourhood, and the block either side
            // of the pixel's own whose spread the flat floor reads, whole.
            Rule::Local { radius, .. } => (radius.max(1) + 1) * BLOCK,
        }
    }
}

/// An image of a channel on the card, as the bits of its values, or its
/// values held by the cores to be handed it.
#[derive(Clone, Copy)]
pub(crate) enum Values<'a> {
    Card(&'a Image),
    Cores(&'a [f32]),
}

/// The values of the box `part` of `image`, `width` across, on the card:
/// the card's own where it holds the whole of what is asked, and
/// otherwise cut from the cores' and handed it.
fn values_of(client: &Client, image: Values<'_>, width: usize, part: Part) -> Held {
    match image {
        Values::Card(image) if part == [0, 0, image.width, image.height] => image.values.clone(),
        Values::Card(_) => unreachable!("an image on the card is read whole"),
        Values::Cores(values) => Held::owned(client, cut(values, width, 1, part)),
    }
}

/// Where a reading lies among the planes of bytes the readings of an
/// image are packed into ([`slots`]): which plane, how many bits up each
/// byte of it, and whether its closings lie above its kind there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Slot {
    plane: usize,
    shift: u32,
    closed: bool,
}

/// Where each of `rules` lies packed: in the order of the rules, each in
/// as many bits as it reads in, two for its kind ([`LIGHT`], [`DARK`],
/// [`NONE`]) and two more for its closings where it is closed
/// ([`CLOSED_DARK`], [`CLOSED_LIGHT`]), as many to a byte as fit; or,
/// where `grey`, a byte to each. The six readings the scan pools, two
/// closed and four windows, take two bytes a pixel, and two more turned
/// over their diagonal, where a byte to each reading took twelve: some
/// two hundred and forty megabytes taken off the card for the frame of a
/// whole photo, and held while it was scanned.
fn slots(rules: &[(Rule, bool)], grey: bool) -> Vec<Slot> {
    let (mut plane, mut used) = (0, 0);
    let mut out = Vec::with_capacity(rules.len());
    for &(_, closed) in rules {
        let bits = if grey { 8 } else if closed { 4 } else { 2 };
        if used + bits > 8 {
            plane += 1;
            used = 0;
        }
        out.push(Slot { plane, shift: used, closed: closed && !grey });
        used += bits;
    }
    out
}

/// What the scan's readings of an image come to, packed into planes of
/// bytes, a byte to a pixel, row after row ([`slots`]); and where they
/// were turned over their diagonal, those planes turned, each column of
/// the image a row of them ([`turned_readings`]).
#[derive(Debug, PartialEq)]
pub(crate) struct Packed {
    planes: Vec<Vec<u8>>,
    turned: Vec<Vec<u8>>,
    slots: Vec<Slot>,
}

impl Packed {
    /// How many readings there are.
    pub(crate) fn len(&self) -> usize {
        self.slots.len()
    }

    /// The reading of rule `r`, where it lies among the planes.
    pub(crate) fn reading(&self, r: usize) -> Reading<'_> {
        let Slot { plane, shift, closed } = self.slots[r];
        Reading { bytes: &self.planes[plane], shift, closings: closed }
    }

    /// The reading of rule `r` turned over its diagonal.
    pub(crate) fn turned(&self, r: usize) -> Reading<'_> {
        assert!(!self.turned.is_empty(), "the readings were turned");
        let Slot { plane, shift, closed } = self.slots[r];
        Reading { bytes: &self.turned[plane], shift, closings: closed }
    }

    /// The one reading made, a byte to a pixel, as it reads alone: its
    /// kind and its closings, or the grey it is shown in.
    pub(crate) fn into_only(mut self) -> Vec<u8> {
        assert_eq!(self.slots.len(), 1, "one reading was made");
        self.planes.swap_remove(0)
    }
}

/// A reading as it lies packed among others ([`Packed`]): its plane, a
/// byte to a pixel, how many bits up each byte it lies, and whether its
/// closings lie above its kind there ([`CLOSED_DARK`], [`CLOSED_LIGHT`]),
/// as they do of a reading closed, where otherwise the next reading's
/// bits do.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Reading<'a> {
    pub bytes: &'a [u8],
    pub shift: u32,
    pub closings: bool,
}

impl Reading<'_> {
    /// Pixel `i`'s kind: [`LIGHT`], [`DARK`] or [`NONE`].
    pub(crate) fn kind(&self, i: usize) -> u8 {
        (self.bytes[i] >> self.shift) & 3
    }

    /// Pixel `i`'s kind, and of a reading closed, its closings above it
    /// ([`CLOSED_DARK`], [`CLOSED_LIGHT`]): what the tests read a reading
    /// by.
    #[cfg(test)]
    pub(crate) fn closed(&self, i: usize) -> u8 {
        (self.bytes[i] >> self.shift) & 15
    }
}

/// What `rules` read `values` as, an image of a channel on the card
/// `width` by `height` ([`read_into`]), closed where it is asked
/// ([`CLOSED_DARK`], [`CLOSED_LIGHT`]): of the box `own` of it, each
/// reading packed into its slot of planes of bytes ([`slots`],
/// [`packed_kinds_kernel`]) as soon as it is made, and where `turned`, into
/// its slot of those planes turned over their diagonal besides
/// ([`turned_kinds_kernel`]). The card holds one reading a pixel to a word
/// at a time, and lets go of it once it is packed: held until all six the
/// scan pools were made, the readings of the frame of a whole photo were
/// most of what the card held at once.
#[allow(clippy::too_many_arguments)]
fn packed_on(client: &Client, values: &Held, (width, height): (usize, usize), rules: &[(Rule, bool)], slots: &[Slot], own: Part, grey: bool, turned: bool) -> (Vec<Held>, Vec<Held>) {
    assert!(!(grey && turned), "a reading in grey is shown, and never turned");
    let (w, h, pixels) = (width as u32, height as u32, width * height);
    let unit = CubeDim::new_2d(SIDE, SIDE);
    let every = cubes(width, height, [SIDE; 2]);
    let planes = slots.last().map_or(0, |slot| slot.plane + 1);
    let (count, box_w, box_h) = (area(own), own[2] - own[0], own[3] - own[1]);
    let across = box_h.div_ceil(4);
    let packed: Vec<Held> = (0..planes).map(|_| Held::empty(client, count.div_ceil(4))).collect();
    let turned: Vec<Held> = if turned { (0..planes).map(|_| Held::empty(client, across * box_w)).collect() } else { Vec::new() };
    for (&(rule, closed), slot) in rules.iter().zip(slots) {
        let kinds = Held::empty(client, pixels);
        read_into(client, values, (width, height), rule, &kinds, width, false);
        if closed {
            let grown = Held::empty(client, pixels);
            // SAFETY: every place read and written is within its values.
            // A pixel is worked out only within the image, of which `kinds`
            // and `grown` hold a word to a pixel, from the square about it
            // within the image. Unchecked, a reading was closed in two
            // thirds of the time.
            unsafe {
                grown_kernel::launch_unchecked(client, every.clone(), unit, kinds.arg(), grown.arg(), w, h);
                closed_kernel::launch_unchecked(client, every.clone(), unit, kinds.arg(), grown.arg(), w, h);
            }
        }
        let (x0, y0) = (own[0] as u32, own[1] as u32);
        packed_kinds_kernel::launch(client, line_cubes(count.div_ceil(4)), CubeDim::new_1d(LINE), kinds.arg(), packed[slot.plane].arg(), w, x0, y0, box_w as u32, count as u32, slot.shift, grey);
        if let Some(turned) = turned.get(slot.plane) {
            turned_kinds_kernel::launch(client, cubes(box_w, across, [SIDE; 2]), unit, kinds.arg(), turned.arg(), w, x0, y0, box_w as u32, box_h as u32, across as u32, slot.shift);
        }
    }
    (packed, turned)
}

/// What `rule` reads `values`, an image of a channel on the card `width`
/// by `height`, as, written into `out` `across` words to a row: a pixel to
/// a word, as it reads ([`LIGHT`], [`DARK`], [`NONE`]), or where `shown`,
/// as the binarised view shows it ([`written`]).
fn read_into(client: &Client, values: &Held, (width, height): (usize, usize), rule: Rule, out: &Held, across: usize, shown: bool) {
    // What the readings launched unchecked write within.
    assert!(across >= width && out.len >= across * height, "{height} rows of {across} words, each the image's {width} and more, are made room for");
    let (w, h, pixels, across) = (width as u32, height as u32, width * height, across as u32);
    let unit = CubeDim::new_2d(SIDE, SIDE);
    let every = cubes(width, height, [SIDE; 2]);
    match rule {
        Rule::Global { threshold } => {
            global_kernel::launch(client, every, unit, values.arg(), out.arg(), threshold, w, h, across, shown);
        }
        Rule::Window { radius, bias } => {
            let (sums, counts) = (Held::empty(client, pixels), Held::empty(client, pixels));
            // Unchecked, where each value a window reads is checked to lie
            // in its buffer: the windows are most of the scan's readings,
            // and each of their places read is a step of a loop whose
            // arithmetic is little more than the check. The widest window
            // the scan pools was read in seven tenths of the time.
            //
            // SAFETY: every place read and written is within its values.
            // A pixel is worked out only within the image, of `values`,
            // `sums` and `counts` a value to each pixel, and written at a
            // row of `out` within its `across` words to a row, which the
            // caller made room for `height` of; and a window reads only the
            // pixels of its row or its column within the image.
            unsafe {
                window_across_kernel::launch_unchecked(client, every.clone(), unit, values.arg(), sums.arg(), counts.arg(), w, h, radius as u32);
                window_kernel::launch_unchecked(client, every, unit, values.arg(), sums.arg(), counts.arg(), out.arg(), w, h, radius as u32, bias, across, shown);
            }
        }
        Rule::Local { radius, min_range } => {
            let (bw, bh) = (width.div_ceil(BLOCK), height.div_ceil(BLOCK));
            let (blocks, sums, thresholds) = (Held::empty(client, 4 * bw * bh), Held::empty(client, 4 * bw * bh), Held::empty(client, bw * bh));
            let over_blocks = cubes(bw, bh, [SIDE; 2]);
            let (bw, bh) = (bw as u32, bh as u32);
            blocks_kernel::launch(client, over_blocks.clone(), unit, values.arg(), blocks.arg(), w, h, bw, bh);
            blocks_across_kernel::launch(client, over_blocks.clone(), unit, blocks.arg(), sums.arg(), bw, bh, radius as u32);
            blocks_down_kernel::launch(client, over_blocks, unit, sums.arg(), thresholds.arg(), bw, bh, radius as u32, min_range);
            blocks_read_kernel::launch(client, every, unit, values.arg(), thresholds.arg(), out.arg(), w, h, bw, across, shown);
        }
    }
}

/// The `rows` rows of `bytes`, each `row` bytes filled out to a whole
/// word, without what fills them out ([`turned_kinds_kernel`]).
fn unfilled(bytes: &[u8], row: usize, rows: usize) -> Vec<u8> {
    let filled = row.next_multiple_of(4);
    if filled == row {
        return bytes[..row * rows].to_vec();
    }
    let mut out = Vec::with_capacity(row * rows);
    for laid in bytes.chunks_exact(filled).take(rows) {
        out.extend_from_slice(&laid[..row]);
    }
    out
}

/// What each of `rules` reads `image`, an image of a channel `width` by
/// `height`, as, packed ([`Packed`]): a reading's kind, [`LIGHT`], [`DARK`]
/// or [`NONE`], in its two low bits, and where its rule is closed,
/// [`CLOSED_DARK`] and [`CLOSED_LIGHT`] above them; or, where `grey`, a
/// byte to each reading, black for dark and white otherwise. On the card
/// whole where it holds it, and otherwise in tiles, each worked out with
/// as much about it as its readings draw on; a reading by blocks has its
/// tiles' pixels about them begin and end where the image's blocks do, so
/// that every block it reads is whole.
pub(crate) fn readings(image: Values<'_>, (width, height): (usize, usize), rules: &[(Rule, bool)], grey: bool) -> Packed {
    readings_within(most_values(client()), image, (width, height), rules, grey)
}

pub(crate) fn readings_within(most: usize, image: Values<'_>, size: (usize, usize), rules: &[(Rule, bool)], grey: bool) -> Packed {
    made_within(most, image, size, rules, grey, false)
}

/// [`readings`], and each reading turned over its diagonal besides: its
/// columns as rows, `height` across, which is how the scan reads down the
/// columns ([`super::super::detect`]). Turned on the card as it is packed,
/// where turned on the cores each byte went a column's length from the
/// last, which was a quarter of the cores' time the readings and the scan
/// of them took for the frame of a whole photo.
pub(crate) fn turned_readings(image: Values<'_>, size: (usize, usize), rules: &[(Rule, bool)]) -> Packed {
    made_within(most_values(client()), image, size, rules, false, true)
}

/// What each of `rules` reads `image` as ([`readings`]), and where
/// `turned`, each reading turned over its diagonal besides
/// ([`turned_readings`]), and otherwise none.
fn made_within(most: usize, image: Values<'_>, (width, height): (usize, usize), rules: &[(Rule, bool)], grey: bool, turned: bool) -> Packed {
    let slots = slots(rules, grey);
    let planes = slots.last().map_or(0, |slot| slot.plane + 1);
    if width == 0 || height == 0 || rules.is_empty() {
        let none = |made: bool| if made { vec![Vec::new(); planes] } else { Vec::new() };
        return Packed { planes: none(true), turned: none(turned), slots };
    }
    // An image the card holds whole is read whole, from where it lies.
    if let Values::Card(on_card) = image {
        assert_eq!((on_card.width, on_card.height), (width, height));
        if width * height <= most && width.max(height) <= PIECE_SIDE {
            let client = client();
            let (packed, turned) = packed_on(client, &on_card.values, (width, height), rules, &slots, [0, 0, width, height], grey, turned);
            let mut read = off_card(client.read(packed.iter().chain(&turned).map(|held| held.handle.clone()).collect())).into_iter();
            let made = read.by_ref().take(planes).map(|bytes| bytes[..width * height].to_vec()).collect();
            return Packed { planes: made, turned: read.map(|bytes| unfilled(&bytes, height, width)).collect(), slots };
        }
        return made_within(most, Values::Cores(&on_card.read()), (width, height), rules, grey, turned);
    }
    let blocked = rules.iter().any(|(rule, _)| matches!(rule, Rule::Local { .. }));
    let reach = rules.iter().map(|&(rule, closed)| rule.reach() + if closed { 2 } else { 0 }).max().unwrap_or(0);
    let held = |tile: Part| {
        let [x0, y0, x1, y1] = about(tile, reach, width, height);
        if blocked {
            [x0 / BLOCK * BLOCK, y0 / BLOCK * BLOCK, x1.next_multiple_of(BLOCK).min(width), y1.next_multiple_of(BLOCK).min(height)]
        } else {
            [x0, y0, x1, y1]
        }
    };
    let side = PIECE_SIDE.saturating_sub(2 * (reach + BLOCK)).max(1);
    let tiles = tiled_within(most, side, width, height, |tile| area(held(tile)), |client, tile| {
        let piece = held(tile);
        let size = (piece[2] - piece[0], piece[3] - piece[1]);
        let values = values_of(client, image, width, piece);
        let (packed, turned) = packed_on(client, &values, size, rules, &slots, inside(tile, piece), grey, turned);
        let (tile_w, tile_h) = (tile[2] - tile[0], tile[3] - tile[1]);
        let mut read = off_card(client.read(packed.iter().chain(&turned).map(|held| held.handle.clone()).collect())).into_iter();
        let made: Vec<Vec<u8>> = read.by_ref().take(planes).map(|bytes| bytes[..area(tile)].to_vec()).collect();
        (made, read.map(|bytes| unfilled(&bytes, tile_h, tile_w)).collect::<Vec<Vec<u8>>>())
    });
    let made = (0..planes)
        .map(|p| {
            let mut out = vec![0u8; width * height];
            for ([x0, y0, x1, y1], (made, _)) in &tiles {
                for (y, row) in (*y0..*y1).zip(made[p].chunks_exact(x1 - x0)) {
                    out[y * width + x0..y * width + x1].copy_from_slice(row);
                }
            }
            out
        })
        .collect();
    // Turned, each column of a tile is a part of a row of the whole turned.
    let columns = |p: usize| {
        let mut out = vec![0u8; width * height];
        for ([x0, y0, x1, y1], (_, turned)) in &tiles {
            for (x, column) in (*x0..*x1).zip(turned[p].chunks_exact(y1 - y0)) {
                out[x * height + y0..x * height + y1].copy_from_slice(column);
            }
        }
        out
    };
    Packed { planes: made, turned: if turned { (0..planes).map(columns).collect() } else { Vec::new() }, slots }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Squares of ink on paper, with grain and a hole, `width` by `height`.
    fn photo(width: usize, height: usize) -> Vec<f32> {
        let mut seed = 0x2545_f491_u32;
        let mut values: Vec<f32> = (0..width * height)
            .map(|i| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let (x, y) = (i % width, i / width);
                let ink = if (x / 9 + y / 7) % 2 == 0 { 0.15 } else { 0.85 };
                (ink + ((seed >> 8) as f32 / (1 << 24) as f32 - 0.5) * 0.2 + 0.3 * x as f32 / width as f32).clamp(0.0, 1.0)
            })
            .collect();
        for i in (0..values.len()).step_by(97) {
            values[i] = f32::NAN;
        }
        values
    }

    /// Each pixel's reading as the cores make it of `rule`, and how far the
    /// pixel lies from where it would read the other way.
    fn read_on_cores(data: &[f32], width: usize, height: usize, rule: Rule) -> Vec<(u8, f64)> {
        let at = |x: usize, y: usize| data[y * width + x];
        let mean_over = |x0: usize, y0: usize, x1: usize, y1: usize| {
            let (mut sum, mut n) = (0.0f64, 0.0f64);
            for y in y0..y1 {
                for x in x0..x1 {
                    if at(x, y).is_finite() {
                        sum += f64::from(at(x, y));
                        n += 1.0;
                    }
                }
            }
            (sum, n)
        };
        (0..width * height)
            .map(|i| {
                let (x, y) = (i % width, i / width);
                let value = data[i];
                if !value.is_finite() {
                    return (NONE, f64::INFINITY);
                }
                let below = |threshold: f64| (if f64::from(value) < threshold { DARK } else { LIGHT }, (f64::from(value) - threshold).abs());
                match rule {
                    Rule::Global { threshold } => below(f64::from(threshold)),
                    Rule::Window { radius, bias } => {
                        let (sum, n) = mean_over(x.saturating_sub(radius), y.saturating_sub(radius), (x + radius + 1).min(width), (y + radius + 1).min(height));
                        below(sum / n - f64::from(bias))
                    }
                    Rule::Local { radius, min_range } => {
                        let (bw, bh) = (width.div_ceil(BLOCK), height.div_ceil(BLOCK));
                        let (bx, by) = (x / BLOCK, y / BLOCK);
                        let block = |bx: usize, by: usize| mean_over(bx * BLOCK, by * BLOCK, ((bx + 1) * BLOCK).min(width), ((by + 1) * BLOCK).min(height));
                        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                        for y in (by.saturating_sub(1) * BLOCK)..((by + 2) * BLOCK).min(height) {
                            for x in (bx.saturating_sub(1) * BLOCK)..((bx + 2) * BLOCK).min(width) {
                                if at(x, y).is_finite() {
                                    lo = lo.min(at(x, y));
                                    hi = hi.max(at(x, y));
                                }
                            }
                        }
                        if hi - lo < min_range {
                            return (LIGHT, f64::INFINITY);
                        }
                        let (mut sum, mut n) = (0.0f64, 0.0f64);
                        for ny in by.saturating_sub(radius)..(by + radius + 1).min(bh) {
                            for nx in bx.saturating_sub(radius)..(bx + radius + 1).min(bw) {
                                let (s, count) = block(nx, ny);
                                if count > 0.0 {
                                    sum += s / count;
                                    n += 1.0;
                                }
                            }
                        }
                        if n > 0.0 { below(sum / n) } else { (LIGHT, f64::INFINITY) }
                    }
                }
            })
            .collect()
    }

    /// Closed over as [`super::super::super::regions`] closed a mask on the
    /// cores: the darkest of the lightest of each square three across.
    fn closed_on_cores(kinds: &[u8], width: usize, height: usize, ink: u8) -> Vec<bool> {
        let square = |of: &dyn Fn(usize, usize) -> bool, x: usize, y: usize, all: bool| {
            let mut out = all;
            for sy in y.saturating_sub(1)..(y + 2).min(height) {
                for sx in x.saturating_sub(1)..(x + 2).min(width) {
                    out = if all { out && of(sx, sy) } else { out || of(sx, sy) };
                }
            }
            out
        };
        let grown: Vec<bool> = (0..width * height).map(|i| square(&|x, y| kinds[y * width + x] & 3 == ink, i % width, i / width, false)).collect();
        (0..width * height).map(|i| square(&|x, y| grown[y * width + x], i % width, i / width, true)).collect()
    }

    /// Each pixel of `reading` as it reads, with its closings.
    fn unpacked(reading: Reading<'_>) -> Vec<u8> {
        (0..reading.bytes.len()).map(|i| reading.closed(i)).collect()
    }

    const RULES: [Rule; 7] = [
        Rule::Global { threshold: 0.5 },
        Rule::Window { radius: 1, bias: 0.0 },
        Rule::Window { radius: 8, bias: 0.02 },
        Rule::Window { radius: 40, bias: 0.01 },
        Rule::Local { radius: 0, min_range: 0.05 },
        Rule::Local { radius: 2, min_range: 0.1 },
        Rule::Local { radius: 5, min_range: 0.5 },
    ];

    #[test]
    fn the_card_reads_an_image_as_the_cores_did() {
        for (width, height) in [(1, 1), (1, 37), (53, 1), (37, 23), (65, 33), (130, 97)] {
            let data = photo(width, height);
            let rules: Vec<(Rule, bool)> = RULES.iter().map(|&rule| (rule, true)).collect();
            let card = readings(Values::Cores(&data), (width, height), &rules, false);
            assert_eq!(card.len(), RULES.len());
            for (r, rule) in RULES.iter().enumerate() {
                let made = unpacked(card.reading(r));
                let cores = read_on_cores(&data, width, height, *rule);
                // A pixel a hair from its threshold may read the other way.
                for (i, (&kind, &(expected, margin))) in made.iter().zip(&cores).enumerate() {
                    assert!(kind & 3 == expected || margin < 1e-5, "{rule:?} over {width} by {height} at {i}: {kind} against {expected}");
                }
                for (ink, bit) in [(DARK, CLOSED_DARK), (LIGHT, CLOSED_LIGHT)] {
                    let closed = closed_on_cores(&made, width, height, ink);
                    assert!(made.iter().zip(&closed).all(|(&kind, &closed)| (kind & bit != 0) == closed), "{rule:?} closed over {width} by {height}");
                }
            }
            // Unclosed, the readings alone; and in grey, black for dark.
            let bare = readings(Values::Cores(&data), (width, height), &[(RULES[5], false)], false).into_only();
            assert!(bare.iter().zip(unpacked(card.reading(5))).all(|(&bare, closed)| bare == closed & 3));
            let grey = readings(Values::Cores(&data), (width, height), &[(RULES[5], false)], true).into_only();
            assert!(grey.iter().zip(&bare).all(|(&grey, &kind)| grey == if kind == DARK { 0 } else { 255 }));
        }
    }

    #[test]
    fn readings_are_packed_as_few_bits_to_each_as_they_read_in() {
        let (width, height) = (37, 23);
        let data = photo(width, height);
        // Closed and not, side by side: each reads packed as it does alone,
        // and the six the scan pools take two bytes a pixel.
        let rules: Vec<(Rule, bool)> = [(RULES[0], true), (RULES[5], true), (RULES[1], false), (RULES[2], false), (RULES[3], false), (RULES[4], false), (RULES[6], true)].to_vec();
        let together = readings(Values::Cores(&data), (width, height), &rules, false);
        assert_eq!(together.planes.len(), 3);
        assert_eq!(slots(&rules[..6], false).last(), Some(&Slot { plane: 1, shift: 6, closed: false }));
        for (r, &(rule, closed)) in rules.iter().enumerate() {
            let alone = readings(Values::Cores(&data), (width, height), &[(rule, closed)], false).into_only();
            let reading = together.reading(r);
            assert!((0..alone.len()).all(|i| reading.kind(i) == alone[i] & 3), "{rule:?}");
            // A reading closed has its closings above its kind, and says so;
            // one that is not has the next reading's there.
            assert_eq!(reading.closings, closed, "{rule:?}");
            if closed {
                assert_eq!(unpacked(reading), alone, "{rule:?} closed");
            }
        }
        // In grey, a byte to each, with no closings in it.
        assert_eq!(slots(&rules, true).iter().map(|slot| (slot.plane, slot.shift, slot.closed)).collect::<Vec<_>>(), (0..rules.len()).map(|r| (r, 0, false)).collect::<Vec<_>>());
    }

    #[test]
    fn an_image_is_read_in_tiles_as_it_is_whole() {
        let (width, height) = (300, 220);
        let data = photo(width, height);
        let rules: Vec<(Rule, bool)> = RULES.iter().map(|&rule| (rule, true)).collect();
        let whole = readings_within(usize::MAX, Values::Cores(&data), (width, height), &rules, false);
        let tiled = readings_within(1 << 13, Values::Cores(&data), (width, height), &rules, false);
        assert!(whole == tiled);
        let on_card = Image::of(Held::of(client(), &data), (width, height));
        assert!(readings(Values::Card(&on_card), (width, height), &rules, false) == whole, "read where it lies on the card");
        assert!(readings_within(1 << 13, Values::Card(&on_card), (width, height), &rules, false) == whole, "and in tiles from it");
    }

    #[test]
    fn a_reading_turned_is_its_columns_as_rows() {
        let rules: Vec<(Rule, bool)> = RULES.iter().enumerate().map(|(r, &rule)| (rule, r % 2 == 0)).collect();
        let turn = |reading: &[u8], width: usize, height: usize| -> Vec<u8> { (0..width * height).map(|i| reading[(i % height) * width + i / height]).collect() };
        // Neither side a whole number of words, a pixel across or down;
        // and in tiles, from the cores and from the card.
        for (width, height) in [(1, 1), (1, 37), (53, 1), (37, 23), (65, 34), (300, 221)] {
            let data = photo(width, height);
            let whole = readings(Values::Cores(&data), (width, height), &rules, false);
            let on_card = Image::of(Held::of(client(), &data), (width, height));
            for (most, image) in [(usize::MAX, Values::Cores(&data)), (1 << 13, Values::Cores(&data)), (usize::MAX, Values::Card(&on_card)), (1 << 13, Values::Card(&on_card))] {
                let made = made_within(most, image, (width, height), &rules, false, true);
                assert!(made.planes == whole.planes, "{width} by {height}, {most} at most");
                assert_eq!(made.turned.len(), made.planes.len());
                for (plane, turned) in made.planes.iter().zip(&made.turned) {
                    assert!(*turned == turn(plane, width, height), "{width} by {height}, {most} at most, turned");
                }
            }
        }
    }
}
