//! The edges of an image as the blur's measure reads them
//! ([`super::super::deblur`]), on the graphics card: the image smoothed
//! and its gradient, each part measured with the pixels about it as the
//! whole image would read them; the steepest of its changes and its
//! darkest and lightest few pixels, ranked among all of them; each edge's
//! width and the way across it; and, for an image too large to measure
//! whole, how much each part of it changes, by which the busiest parts are
//! chosen.
//!
//! The photo's Deblur filter undoes the blur of a part of the photo
//! adjusted on the card, and measured that blur by taking the whole part
//! off it first, sixty-odd megabytes for a photo of sixteen megapixels,
//! and measuring it on the cores. Measured here, nothing of it is taken
//! off but the widths of its edges. A read's copy deblurred, which is
//! made from the cores' pixels, is measured here too, so that what the
//! filter shows is what the read read.
//!
//! What is ranked is ranked exactly: the value of the given rank among the
//! card's own values, found a byte of it at a time from the highest
//! ([`ordered`]), each byte by the count of the values that share the
//! bytes above it. Sums and quotients are the card's single precision, as
//! the cores' were but for the busyness of a part, which they added up in
//! double, and the places across an edge, which they put in double and
//! the card puts as a pixel and a fraction of one apart.

use super::scan::Values;
use super::*;

/// What lies beyond either end of the values ranked: darker and lighter
/// than any.
const FAR: f32 = 1.0e30;
/// Most ranks a pass of [`ranked_kernel`] seeks among the same values at
/// once: a histogram of each, which its cube holds, eight kilobytes for
/// as many as this.
const MOST_SEEKS: usize = 8;
// A cube of [`ranked_kernel`] clears each rank's 256 counts, and adds them
// to the whole, a unit to a count: as many units as a byte has values.
const _: () = assert!(LINE == 256, "a unit of a ranking's cube to each value of a byte");

/// Of the bits of a value, a word that orders as the value does: the sign
/// flipped for a value of nought or more, and every bit for one below.
#[cube]
fn ordered(bits: u32) -> u32 {
    let mut key = bits | 0x8000_0000u32;
    if (bits & 0x8000_0000u32) != 0u32 {
        key = !bits;
    }
    key
}

/// The value whose bits [`ordered`] makes `key` of.
fn unordered(key: u32) -> f32 {
    f32::from_bits(if key & 0x8000_0000 != 0 { key & 0x7fff_ffff } else { !key })
}

/// `values`, `width` by `height`, smoothed by the taps `across`, `down`
/// and `weights` ([`blurred_at`]), the pixels on the edge standing for
/// those beyond it.
#[cube(launch_unchecked)]
fn smoothed_kernel(values: &[f32], smoothed: &mut [f32], across: &[i32], down: &[i32], weights: &[f32], width: u32, height: u32) {
    let (x, y) = (ABSOLUTE_POS_X, ABSOLUTE_POS_Y);
    if x < width && y < height {
        smoothed[(y * width + x) as usize] = blurred_at(values, across, down, weights, x, y, width, height);
    }
}

/// The gradient of `smoothed`, an image `width` across, across the pixel
/// `i`, which is not on its edge: a Sobel's, over the three pixels about
/// it each way, a pixel apart.
#[cube]
fn gradient_across(smoothed: &[f32], i: u32, width: u32) -> f32 {
    let (above, here, below) = ((i - width) as usize, i as usize, (i + width) as usize);
    (smoothed[above + 1] + 2.0f32 * smoothed[here + 1] + smoothed[below + 1] - smoothed[above - 1] - 2.0f32 * smoothed[here - 1] - smoothed[below - 1]) / 8.0f32
}

/// And down it.
#[cube]
fn gradient_down(smoothed: &[f32], i: u32, width: u32) -> f32 {
    let (above, below) = ((i - width) as usize, (i + width) as usize);
    (smoothed[below - 1] + 2.0f32 * smoothed[below] + smoothed[below + 1] - smoothed[above - 1] - 2.0f32 * smoothed[above] - smoothed[above + 1]) / 8.0f32
}

/// The steepness of `smoothed`, `width` by `height`: the length of its
/// gradient each way ([`gradient_across`], [`gradient_down`]), nought on
/// the image's edge. The gradient itself is worked out again where an
/// edge is measured ([`widths_kernel`]), at the few pixels steep enough to
/// be one, where held for every pixel it was two images more on the card.
#[cube(launch)]
fn steep_kernel(smoothed: &[f32], steep: &mut [f32], width: u32, height: u32) {
    let (x, y) = (ABSOLUTE_POS_X, ABSOLUTE_POS_Y);
    if x < width && y < height {
        let i = y * width + x;
        let mut across = 0.0f32;
        let mut down = 0.0f32;
        if x >= 1 && y >= 1 && x + 1 < width && y + 1 < height {
            across = gradient_across(smoothed, i, width);
            down = gradient_down(smoothed, i, width);
        }
        steep[i as usize] = (across * across + down * down).sqrt();
    }
}

/// Of the box `box_w` across, `count` pixels in all, beginning at
/// `(x0, y0)` of `values`, `width` across, the values more than nought
/// alone where `positive`: for each of the `seeks` ranks sought among them,
/// how many of those whose [`ordered`] bits are its prefix in `prefixes`
/// under its mask in `masks` have each byte `shift` bits up, added to its
/// 256 of `counts`. Each value is read once for every rank sought, where a
/// pass of each rank read them all again. Each cube counts its own and
/// adds them to the whole.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn ranked_kernel(values: &[f32], counts: &mut [Atomic<u32>], prefixes: &[u32], masks: &[u32], width: u32, x0: u32, y0: u32, box_w: u32, count: u32, shift: u32, #[comptime] seeks: u32, #[comptime] positive: bool) {
    let bins = Shared::<[Atomic<u32>]>::new_slice(comptime!(256 * seeks as usize));
    let mut prefix = Array::<u32>::new(comptime!(seeks as usize));
    let mut mask = Array::<u32>::new(comptime!(seeks as usize));
    #[unroll]
    for s in 0..seeks {
        bins[(s * 256 + UNIT_POS) as usize].store(0u32);
        prefix[s as usize] = prefixes[s as usize];
        mask[s as usize] = masks[s as usize];
    }
    sync_cube();
    let mut i = CUBE_POS_X * LINE + UNIT_POS;
    let stride = CUBE_COUNT_X * LINE;
    while i < count {
        let value = values[((y0 + i / box_w) * width + x0 + i % box_w) as usize];
        let mut kept = true.runtime();
        if comptime!(positive) {
            kept = value > 0.0f32;
        }
        let key = ordered(u32::reinterpret(value));
        let byte = (key >> shift) & 255u32;
        #[unroll]
        for s in 0..seeks {
            if kept && (key & mask[s as usize]) == prefix[s as usize] {
                bins[(s * 256 + byte) as usize].fetch_add(1u32);
            }
        }
        i += stride;
    }
    sync_cube();
    #[unroll]
    for s in 0..seeks {
        let own = bins[(s * 256 + UNIT_POS) as usize].load();
        if own > 0 {
            counts[(s * 256 + UNIT_POS) as usize].fetch_add(own);
        }
    }
}

/// What `values`, `width` by `height`, hold at `(x, y)` moved by the
/// fraction `(dx, dy)` of a pixel and more, bilinearly, the place held
/// within the image: a pixel and a fraction apart, so that the fraction is
/// as fine as a place a few pixels from a pixel of the image may be.
#[cube]
fn light_at(values: &[f32], width: u32, height: u32, x: u32, y: u32, dx: f32, dy: f32) -> f32 {
    let (fx, fy) = (dx.floor(), dy.floor());
    let mut column = x as i32 + fx as i32;
    let mut row = y as i32 + fy as i32;
    let mut tx = dx - fx;
    let mut ty = dy - fy;
    if column < 0 {
        column = 0;
        tx = 0.0f32;
    }
    if column >= width as i32 - 1 {
        column = width as i32 - 1;
        tx = 0.0f32;
    }
    if row < 0 {
        row = 0;
        ty = 0.0f32;
    }
    if row >= height as i32 - 1 {
        row = height as i32 - 1;
        ty = 0.0f32;
    }
    let (x0, y0) = (column as u32, row as u32);
    let (x1, y1) = (min(x0 + 1, width - 1), min(y0 + 1, height - 1));
    let top = values[(y0 * width + x0) as usize] * (1.0f32 - tx) + values[(y0 * width + x1) as usize] * tx;
    let bottom = values[(y1 * width + x0) as usize] * (1.0f32 - tx) + values[(y1 * width + x1) as usize] * tx;
    top * (1.0f32 - ty) + bottom * ty
}

/// A value rounded half away from nought, as [`f64::round`] rounds: up
/// from its whole part where what is left over comes to a half, which is
/// worked out exactly. A half added and the sum's whole part taken, the
/// last value short of a half, 0.49999997, was rounded to one, the sum
/// rounding up to one.
#[cube]
fn rounded(value: f32) -> i32 {
    let magnitude = value.abs();
    let below = magnitude.floor();
    let mut out = below;
    if magnitude - below >= 0.5f32 {
        out = below + 1.0f32;
    }
    if value < 0.0f32 {
        out = -out;
    }
    out as i32
}

/// Each edge among the pixels of the box `box_w` by `box_h` beginning at
/// `(x0, y0)` of an image `width` by `height`: a pixel whose steepness
/// comes to `floor`, and is the steepest of it and the pixels either side
/// of it along its own direction, its gradient each way worked out from
/// `smoothed` there; and across which the light read
/// `reach` pixels either way, at `2 * steps + 1` places, steps by
/// `least_step` or more. Its width, the step over its steepness, and the
/// way across it, are written three values to an edge into the next slot
/// of `out`, of `capacity`. A cube's edges count off their slots among
/// themselves and claim them from `slots` at once, where each claiming its
/// own had every edge of the image wait on the one count.
#[cube(launch_unchecked)]
fn widths_kernel(smoothed: &[f32], steep: &[f32], out: &mut [f32], slots: &mut [Atomic<u32>], frame: Frame, reach: f32, steps: u32, capacity: u32) {
    let (bx, by) = (ABSOLUTE_POS_X, ABSOLUTE_POS_Y);
    let mut found = false.runtime();
    let mut wide = 0.0f32;
    let mut across = 0.0f32;
    let mut down = 0.0f32;
    if bx < frame.box_w && by < frame.box_h {
        let (x, y) = (frame.x0 + bx, frame.y0 + by);
        let at = y * frame.width + x;
        let g = steep[at as usize];
        if g >= frame.floor {
            let (ux, uy) = (gradient_across(smoothed, at, frame.width) / g, gradient_down(smoothed, at, frame.width) / g);
            // A pixel along a unit of the way across: held there, so that
            // no place read lies further.
            let (sx, sy) = (clamp(rounded(ux), -1, 1), clamp(rounded(uy), -1, 1));
            let ahead = steep[((y as i32 + sy) as u32 * frame.width + (x as i32 + sx) as u32) as usize];
            let behind = steep[((y as i32 - sy) as u32 * frame.width + (x as i32 - sx) as u32) as usize];
            if !(g < ahead || g < behind) {
                let mut lo = FAR.runtime();
                let mut hi = (-FAR).runtime();
                let each = max(steps, 1u32) as f32;
                for k in 0..2 * steps + 1 {
                    let t = (k as i32 - steps as i32) as f32 * reach / each;
                    let value = light_at(smoothed, frame.width, frame.height, x, y, ux * t, uy * t);
                    lo = min(lo, value);
                    hi = max(hi, value);
                }
                if hi - lo >= frame.least_step {
                    found = true;
                    wide = (hi - lo) / g;
                    across = ux;
                    down = uy;
                }
            }
        }
    }
    // How many of the cube's units found an edge, and where the slots
    // claimed for them begin.
    let tally = Shared::<[Atomic<u32>]>::new_slice(2usize);
    if UNIT_POS == 0 {
        tally[0].store(0u32);
    }
    sync_cube();
    let mut own = 0u32;
    if found {
        own = tally[0].fetch_add(1u32);
    }
    sync_cube();
    if UNIT_POS == 0 {
        let count = tally[0].load();
        if count > 0 {
            tally[1].store(slots[0].fetch_add(count));
        }
    }
    sync_cube();
    if found {
        let slot = tally[1].load() + own;
        if slot < capacity {
            out[(3 * slot) as usize] = wide;
            out[(3 * slot + 1) as usize] = across;
            out[(3 * slot + 2) as usize] = down;
        }
    }
}

/// Where in an image `width` by `height` the edges are looked for: the
/// box `box_w` by `box_h` beginning at `(x0, y0)`; and how steep and how
/// far across an edge must step ([`widths_kernel`]).
#[derive(CubeLaunch, CubeType)]
struct Frame {
    width: u32,
    height: u32,
    x0: u32,
    y0: u32,
    box_w: u32,
    box_h: u32,
    floor: f32,
    least_step: f32,
}

/// How far the light changes at every other pixel of every other row of a
/// band of an image, over two pixels across and two down from it: `cols`
/// of them across and `rows` down. `values` holds the band `row` values to
/// a row: each of its pixels where `step` is two, and where it is one,
/// only every other pixel of every other row, the pixels read.
#[cube(launch)]
fn changes_kernel(values: &[f32], changes: &mut [f32], row: u32, step: u32, cols: u32, rows: u32) {
    let (k, j) = (ABSOLUTE_POS_X, ABSOLUTE_POS_Y);
    if k < cols && j < rows {
        let at = (step * (j * row + k)) as usize;
        let here = values[at];
        changes[(j * cols + k) as usize] = (values[at + step as usize] - here).abs() + (values[at + (step * row) as usize] - here).abs();
    }
}

/// How much each part `side` pixels square of a band of an image changes:
/// of `changes`, `cols` across and `rows` down, a change at every other
/// pixel of every other row ([`changes_kernel`]), those of `floor` or more
/// added up, a cube to each part, `across` of them along a row. Each unit
/// adds up its share in its order, and the first adds up theirs in theirs.
#[cube(launch)]
fn busy_kernel(changes: &[f32], busy: &mut [f32], cols: u32, rows: u32, side: u32, across: u32, floor: f32) {
    let (px, py) = (CUBE_POS_X, CUBE_POS_Y);
    // The changes of the part: each a pixel two of the image apart.
    let (k0, j0) = (px * side / 2, py * side / 2);
    let (k1, j1) = (min((px + 1) * side / 2, cols), min((py + 1) * side / 2, rows));
    let box_w = k1 - k0;
    let count = box_w * (j1 - j0);
    let mut sum = 0.0f32;
    let mut i = UNIT_POS;
    while i < count {
        let change = changes[((j0 + i / box_w) * cols + k0 + i % box_w) as usize];
        if change >= floor {
            sum += change;
        }
        i += LINE;
    }
    let mut shared = Shared::<[f32]>::new_slice(comptime!(LINE as usize));
    shared[UNIT_POS as usize] = sum;
    sync_cube();
    if UNIT_POS == 0 {
        let mut total = 0.0f32;
        for u in 0..LINE {
            total += shared[u as usize];
        }
        busy[(py * across + px) as usize] = total;
    }
}

/// A box of values on the card to rank: of `values`, `width` across, the
/// box `part` of it, and those more than nought alone where `positive`.
struct Ranked<'a> {
    values: &'a Held,
    width: usize,
    part: Part,
    positive: bool,
}

/// The values of each rank of `ranks` among the values of every box of
/// `boxes` together, from the least, counted from nought; with how many
/// values there are, and `None` for a rank that there are not so many
/// values to reach. Found a byte at a time from the highest of their
/// [`ordered`] bits, the counts of every box taken off the card together
/// ([`ranked_together`]).
fn ranked(client: &Client, boxes: &[Ranked<'_>], ranks: impl Fn(usize) -> Vec<usize>) -> (usize, Vec<Option<f32>>) {
    ranked_together(client, &[(boxes, &ranks)]).remove(0)
}

/// A ranking [`ranked_together`] finds: boxes of values on the card, and
/// the ranks asked among them, by how many values they hold.
type Ranking<'a> = (&'a [Ranked<'a>], &'a dyn Fn(usize) -> Vec<usize>);

/// The ranks a pass of [`ranked_together`] seeks among the boxes of a
/// ranking: the prefix of the [`ordered`] bits of each, and their mask.
type Asked<'a> = (&'a [Ranked<'a>], Vec<(u32, u32)>);

/// A rank sought a byte at a time ([`ranked_together`]): the bits of its
/// value found so far and the mask of them, its rank among the values
/// that share them, and how those fall by the next byte.
#[derive(Clone, Copy)]
struct Sought {
    prefix: u32,
    mask: u32,
    rank: usize,
    counts: [u32; 256],
}

/// [`ranked`] of each of `rankings`, side by side: the counts of every
/// rank of every ranking taken off the card together at each byte, four
/// waits on the card in all, where ranked one after another each ranking
/// waits four times of its own; and each ranking's values read once a
/// byte for all its ranks ([`ranked_kernel`]).
fn ranked_together(client: &Client, rankings: &[Ranking<'_>]) -> Vec<(usize, Vec<Option<f32>>)> {
    // How the values of the boxes of each of `asked` fall by their byte
    // `shift` bits up, for each rank sought among them: those whose
    // ordered bits are its prefix under its mask.
    let pass = |asked: &[Asked<'_>], shift: u32| -> Vec<Vec<[u32; 256]>> {
        // Which ranking's counts each of `counts` holds, of how many ranks.
        let mut counts: Vec<(usize, usize, Held)> = Vec::new();
        for (a, (boxes, seeks)) in asked.iter().enumerate() {
            for seeks in seeks.chunks(MOST_SEEKS) {
                let held = Held::owned(client, vec![0u32; 256 * seeks.len()]);
                let prefixes = Held::owned(client, seeks.iter().map(|&(prefix, _)| prefix).collect::<Vec<u32>>());
                let masks = Held::owned(client, seeks.iter().map(|&(_, mask)| mask).collect::<Vec<u32>>());
                for ranked in boxes.iter().filter(|ranked| area(ranked.part) > 0) {
                    let count = area(ranked.part);
                    let cubes = count.div_ceil(LINE as usize * 64).clamp(1, MOST_CUBES as usize) as u32;
                    let [x0, y0, x1, _] = ranked.part;
                    ranked_kernel::launch(
                        client,
                        CubeCount::Static(cubes, 1, 1),
                        CubeDim::new_1d(LINE),
                        ranked.values.arg(),
                        held.arg(),
                        prefixes.arg(),
                        masks.arg(),
                        ranked.width as u32,
                        x0 as u32,
                        y0 as u32,
                        (x1 - x0) as u32,
                        count as u32,
                        shift,
                        seeks.len() as u32,
                        ranked.positive,
                    );
                }
                counts.push((a, seeks.len(), held));
            }
        }
        let mut out = vec![Vec::new(); asked.len()];
        if counts.is_empty() {
            return out;
        }
        let read = off_card(client.read(counts.iter().map(|(_, _, held)| held.handle.clone()).collect()));
        for ((a, seeks, _), bytes) in counts.iter().zip(&read) {
            for each in u32::from_bytes(bytes)[..256 * seeks].as_chunks::<256>().0 {
                out[*a].push(*each);
            }
        }
        out
    };
    // The first pass counts every ranking's values, which sets its ranks.
    let first = pass(&rankings.iter().map(|&(boxes, _)| (boxes, vec![(0, 0)])).collect::<Vec<_>>(), 24);
    let totals: Vec<usize> = first.iter().map(|counts| counts[0].iter().map(|&n| n as usize).sum()).collect();
    // Each rank asked of each ranking, where there are values enough to
    // reach it.
    let mut sought: Vec<Vec<Option<Sought>>> = rankings
        .iter()
        .zip(totals.iter().zip(&first))
        .map(|(&(_, ranks), (&total, first))| ranks(total).into_iter().map(|rank| (rank < total).then_some(Sought { prefix: 0, mask: 0, rank, counts: first[0] })).collect())
        .collect();
    for shift in [24u32, 16, 8, 0] {
        if shift != 24 {
            let asked: Vec<Asked<'_>> = rankings.iter().zip(&sought).map(|(&(boxes, _), seeks)| (boxes, seeks.iter().flatten().map(|seek| (seek.prefix, seek.mask)).collect())).collect();
            for (seeks, counted) in sought.iter_mut().zip(pass(&asked, shift)) {
                let mut counted = counted.into_iter();
                for seek in seeks.iter_mut().flatten() {
                    seek.counts = counted.next().expect("counts for every rank sought");
                }
            }
        }
        for seek in sought.iter_mut().flatten().flatten() {
            let mut seen = 0usize;
            let byte = (0..256).find(|&b| {
                seen += seek.counts[b] as usize;
                seen > seek.rank
            });
            let byte = byte.expect("the rank lies among the values counted");
            seek.rank -= seen - seek.counts[byte] as usize;
            seek.prefix |= (byte as u32) << shift;
            seek.mask |= 0xff << shift;
        }
    }
    totals.iter().zip(sought).map(|(&total, seeks)| (total, seeks.into_iter().map(|seek| seek.map(|seek| unordered(seek.prefix))).collect())).collect()
}

/// A part of an image as the blur's measure reads its edges, on the card:
/// the box `held` of the image, which is the part and as much about it as
/// is read with it, smoothed, and its steepness.
struct PartRead {
    held: Part,
    part: Part,
    smoothed: Held,
    steep: Held,
}

/// The parts of an image `width` by `height` read on the card for the
/// blur's measure, each with `margin` pixels about it within the image,
/// smoothed by `smoothing` ([`super::super::deblur::kernel`]).
pub(crate) struct Edges {
    width: usize,
    height: usize,
    parts: Vec<PartRead>,
}

impl Edges {
    /// The boxes `parts` of `image`, `width` by `height`, each read with
    /// `margin` pixels about it, within the image.
    pub(crate) fn of(image: Values<'_>, (width, height): (usize, usize), parts: &[Part], margin: usize, smoothing: &[(isize, isize, f32)]) -> Self {
        let client = client();
        let across = Held::owned(client, smoothing.iter().map(|&(dx, _, _)| dx as i32).collect::<Vec<i32>>());
        let down = Held::owned(client, smoothing.iter().map(|&(_, dy, _)| dy as i32).collect::<Vec<i32>>());
        let weights = Held::owned(client, smoothing.iter().map(|&(_, _, w)| w).collect::<Vec<f32>>());
        let parts = parts
            .iter()
            .filter(|&&part| area(part) > 0)
            .map(|&part| {
                let held = about(part, margin, width, height);
                let (w, h) = (held[2] - held[0], held[3] - held[1]);
                let values = match image {
                    Values::Card(on_card) if held == [0, 0, width, height] => on_card.values.clone(),
                    Values::Card(on_card) => on_card.carried_from([held[0], held[1]], (w, h), false).values,
                    Values::Cores(values) => Held::owned(client, cut(values, width, 1, held)),
                };
                let (smoothed, steep) = (Held::empty(client, w * h), Held::empty(client, w * h));
                let (every, unit) = (cubes(w, h, [SIDE; 2]), CubeDim::new_2d(SIDE, SIDE));
                // SAFETY: every place read and written is within its
                // values. A pixel is worked out only within the box held,
                // of which `values` and `smoothed` hold a value to a pixel;
                // each tap is read at a place held within it
                // ([`blurred_at`]); and there are as many places across and
                // down as weights. Unchecked, the image was smoothed in four
                // fifths of the time.
                unsafe { smoothed_kernel::launch_unchecked(client, every.clone(), unit, values.arg(), smoothed.arg(), across.arg(), down.arg(), weights.arg(), w as u32, h as u32) };
                steep_kernel::launch(client, every, unit, smoothed.arg(), steep.arg(), w as u32, h as u32);
                PartRead { held, part, smoothed, steep }
            })
            .collect();
        Self { width, height, parts }
    }

    /// The steepness of rank `top(n)` from the least among the parts' `n`
    /// pixels that change at all, and the light of ranks `low(m)` and
    /// `high(m)` among all `m` of their pixels; with `n`. `None` where
    /// there is no pixel.
    pub(crate) fn ranked(&self, top: impl Fn(usize) -> usize, low: impl Fn(usize) -> usize, high: impl Fn(usize) -> usize) -> (usize, Option<[f32; 3]>) {
        let client = client();
        let boxes = |positive: bool| -> Vec<Ranked<'_>> {
            self.parts
                .iter()
                .map(|read| {
                    let source = if positive { &read.steep } else { &read.smoothed };
                    Ranked { values: source, width: read.held[2] - read.held[0], part: inside(read.part, read.held), positive }
                })
                .collect()
        };
        // The steepness and the light ranked side by side.
        let (steep, light) = (boxes(true), boxes(false));
        let (steepest_rank, light_ranks) = (|n| vec![top(n)], |m| vec![low(m), high(m)]);
        let [(changing, steepest), (_, lights)]: [(usize, Vec<Option<f32>>); 2] =
            ranked_together(client, &[(&steep, &steepest_rank), (&light, &light_ranks)]).try_into().expect("a ranking of each");
        match (steepest[0], lights[0], lights[1]) {
            (Some(steepest), Some(darkest), Some(lightest)) => (changing, Some([steepest, darkest, lightest])),
            _ => (changing, None),
        }
    }

    /// Each edge of the parts' pixels at least two from the image's own
    /// edges whose steepness comes to `floor` and whose step, read `reach`
    /// pixels either side of it, comes to `least_step`
    /// ([`widths_kernel`]): its width, and the way across it, as the x and
    /// y of a unit vector.
    pub(crate) fn widths(&self, reach: f64, floor: f32, least_step: f32) -> Vec<(f32, f32, f32)> {
        let client = client();
        let margin = 2;
        let boxes: Vec<(usize, Part)> = self
            .parts
            .iter()
            .enumerate()
            .map(|(i, read)| {
                let [x0, y0, x1, y1] = read.part;
                let held = [x0.max(margin), y0.max(margin), x1.min(self.width.saturating_sub(margin)), y1.min(self.height.saturating_sub(margin))];
                (i, held)
            })
            .filter(|(_, part)| part[0] < part[2] && part[1] < part[3])
            .collect();
        let capacity: usize = boxes.iter().map(|&(_, part)| area(part)).sum();
        if capacity == 0 {
            return Vec::new();
        }
        let (out, slots) = (Held::empty(client, 3 * capacity), Held::owned(client, vec![0u32; 1]));
        let steps = reach.ceil() as u32;
        for &(i, part) in &boxes {
            let read = &self.parts[i];
            let (w, h) = (read.held[2] - read.held[0], read.held[3] - read.held[1]);
            let [x0, y0, x1, y1] = inside(part, read.held);
            assert!(x0 >= 1 && y0 >= 1 && x1 < w && y1 < h, "an edge is measured a pixel or more within the box held about it");
            let frame = FrameLaunch::new(w as u32, h as u32, x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32, floor, least_step);
            // Unchecked, the edges were measured in seven eighths of the
            // time.
            //
            // SAFETY: every place read and written is within its values.
            // An edge is measured only within the box, which lies a pixel
            // or more within the box held, as asserted, of which `smoothed`
            // and `steep` hold a value to a pixel: its gradient is read
            // from the pixels about it, and its steepness beside that of a
            // pixel along a unit of the way across, held there; the light
            // across it is read at places held within the box held
            // ([`light_at`]); and an edge is written only into a slot below
            // `capacity`, three values to a slot, of which `out` holds as
            // many.
            unsafe {
                widths_kernel::launch_unchecked(
                    client,
                    cubes(x1 - x0, y1 - y0, [SIDE; 2]),
                    CubeDim::new_2d(SIDE, SIDE),
                    read.smoothed.arg(),
                    read.steep.arg(),
                    out.arg(),
                    slots.arg(),
                    frame,
                    reach as f32,
                    steps,
                    capacity as u32,
                );
            }
        }
        // How many edges there are, and then theirs alone taken off the
        // card: the room made for them is an edge to every pixel, and
        // taken off whole it was many times what they fill of it.
        let found = (u32::from_bytes(&off_card(client.read_one(slots.handle.clone())))[0] as usize).min(capacity);
        if found == 0 {
            return Vec::new();
        }
        let filled = off_card(client.read_one(out.handle.clone().offset_end((3 * (capacity - found) * size_of::<f32>()) as u64)));
        f32::from_bytes(&filled)[..3 * found].as_chunks::<3>().0.iter().map(|&[w, ux, uy]| (w, ux, uy)).collect()
    }
}

/// How much each part `side` pixels square of `image`, `width` by
/// `height`, laid on a grid from its corner, changes, row after row of
/// the parts: added up over the changes at every other pixel of every
/// other row, over two pixels each way, as steep as `share` of the
/// steepest hundredth of them, which grain does not reach. Empty where
/// nothing changes. An image the card does not hold at once is read in
/// bands of whole rows of parts.
pub(crate) fn busy(image: Values<'_>, (width, height): (usize, usize), side: usize, share: f32) -> Vec<f64> {
    busy_within(most_values(client()), image, (width, height), side, share)
}

pub(crate) fn busy_within(most: usize, image: Values<'_>, (width, height): (usize, usize), side: usize, share: f32) -> Vec<f64> {
    let (across, down) = (width.div_ceil(side), height.div_ceil(side));
    let (cols, rows) = (width.saturating_sub(2).div_ceil(2), height.saturating_sub(2).div_ceil(2));
    if cols == 0 || rows == 0 {
        return Vec::new();
    }
    assert!(side.is_multiple_of(2), "a part begins on a pixel a change is read at");
    let client = client();
    // Bands of whole rows of parts, each with the two rows below it its
    // changes read down to, as many as the card takes at once.
    let rows_of_parts = (most / (width * side).max(1)).clamp(1, down);
    let bands: Vec<(usize, usize)> = (0..down).step_by(rows_of_parts).map(|p| (p, (p + rows_of_parts).min(down))).collect();
    let changes: Vec<(usize, usize, Held)> = bands
        .iter()
        .filter(|&&(p0, _)| p0 * side < height - 2)
        .map(|&(p0, p1)| {
            let (y0, y1) = (p0 * side, (p1 * side).min(height));
            let band_rows = (y1.min(height - 2) - y0).div_ceil(2);
            let (values, row, step) = match image {
                Values::Card(on_card) if bands.len() == 1 => (on_card.values.clone(), width, 2),
                Values::Card(on_card) => (on_card.carried_from([0, y0], (width, (y1 + 2).min(height) - y0), false).values, width, 2),
                // Of the cores' pixels, those read alone handed the card: a
                // quarter of them, the band's every other pixel of every
                // other row, from the first to the last a change reaches.
                Values::Cores(values) => {
                    let mut read = Vec::with_capacity((cols + 1) * (band_rows + 1));
                    for m in 0..=band_rows {
                        // The first of each pair, and the last pixel read.
                        let (pairs, last) = values[(y0 + 2 * m) * width..][..2 * cols + 1].as_chunks::<2>();
                        read.extend(pairs.iter().map(|pair| pair[0]));
                        read.extend_from_slice(last);
                    }
                    (Held::owned(client, read), cols + 1, 1)
                }
            };
            let changes = Held::empty(client, cols * band_rows);
            changes_kernel::launch(client, cubes(cols, band_rows, [SIDE; 2]), CubeDim::new_2d(SIDE, SIDE), values.arg(), changes.arg(), row as u32, step, cols as u32, band_rows as u32);
            (p0, band_rows, changes)
        })
        .collect();
    let boxes: Vec<Ranked<'_>> = changes.iter().map(|(_, band_rows, changes)| Ranked { values: changes, width: cols, part: [0, 0, cols, *band_rows], positive: true }).collect();
    let (_, top) = ranked(client, &boxes, |n| vec![(n * 99 / 100).min(n.saturating_sub(1))]);
    let Some(top) = top[0] else { return Vec::new() };
    let floor = share * top;
    let mut busy = vec![0.0f64; across * down];
    let sums: Vec<(usize, usize, Held)> = changes
        .iter()
        .map(|(p0, band_rows, changes)| {
            let parts_down = band_rows.div_ceil(side / 2);
            let sums = Held::empty(client, across * parts_down);
            busy_kernel::launch(client, CubeCount::Static(across as u32, parts_down as u32, 1), CubeDim::new_1d(LINE), changes.arg(), sums.arg(), cols as u32, *band_rows as u32, side as u32, across as u32, floor);
            (*p0, parts_down, sums)
        })
        .collect();
    let read = off_card(client.read(sums.iter().map(|(_, _, sums)| sums.handle.clone()).collect()));
    for ((p0, parts_down, _), bytes) in sums.iter().zip(&read) {
        for (i, &sum) in f32::from_bytes(bytes)[..across * parts_down].iter().enumerate() {
            busy[(p0 + i / across) * across + i % across] = f64::from(sum);
        }
    }
    busy
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Blurred squares of ink on paper, with grain.
    fn photo(width: usize, height: usize) -> Vec<f32> {
        let mut seed = 0x2545_f491_u32;
        (0..width * height)
            .map(|i| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let (x, y) = ((i % width) as f32, (i / width) as f32);
                let ink = 0.5 + 0.35 * ((x / 6.0).sin() * (y / 5.0).cos()).tanh();
                (ink + ((seed >> 8) as f32 / (1 << 24) as f32 - 0.5) * 0.05).clamp(0.0, 1.0)
            })
            .collect()
    }

    /// [`rounded`] of each of `values`, as a value.
    #[cube(launch)]
    fn rounded_kernel(values: &[f32], out: &mut [f32], count: u32) {
        let i = line_place();
        if i < count {
            out[i as usize] = rounded(values[i as usize]) as f32;
        }
    }

    #[test]
    fn a_value_is_rounded_on_the_card_as_on_the_cores() {
        let client = client();
        // The last value short of a half, which a half added to rounds up
        // to one, among halves and values either side of them.
        let short = 0.5f32 - 2f32.powi(-25);
        let values = [0.0, 0.25, short, 0.5, 0.75, 1.5 - 2f32.powi(-23), 1.5, 2.5, -short, -0.5, -1.5, -2.5, -0.75];
        let (on_card, out) = (Held::of(client, &values), Held::empty(client, values.len()));
        rounded_kernel::launch(client, line_cubes(values.len()), CubeDim::new_1d(LINE), on_card.arg(), out.arg(), values.len() as u32);
        assert_eq!(out.read(client), values.map(|value| f64::from(value).round() as f32));
    }

    #[test]
    fn values_are_ranked_on_the_card_as_by_sorting() {
        let client = client();
        let (width, height) = (130, 97);
        let mut data = photo(width, height);
        data[3] = 0.0;
        data[5] = -0.25;
        data[8] = 2.0;
        let on_card = Held::of(client, &data);
        for (part, positive) in [([0, 0, width, height], false), ([7, 5, 90, 60], true), ([3, 3, 4, 4], false)] {
            let mut sorted: Vec<f32> = (part[1]..part[3]).flat_map(|y| (part[0]..part[2]).map(move |x| (x, y))).map(|(x, y)| data[y * width + x]).filter(|&v| !positive || v > 0.0).collect();
            sorted.sort_by(f32::total_cmp);
            let n = sorted.len();
            let (total, found) = ranked(client, &[Ranked { values: &on_card, width, part, positive }], |n| vec![0, n / 50, n * 99 / 100, n.saturating_sub(1), n]);
            assert_eq!(total, n);
            assert_eq!(found, vec![Some(sorted[0]), Some(sorted[n / 50]), Some(sorted[n * 99 / 100]), Some(sorted[n - 1]), None], "{part:?}");
        }
        // Several boxes ranked as one.
        let boxes = [Ranked { values: &on_card, width, part: [0, 0, 40, 30], positive: false }, Ranked { values: &on_card, width, part: [60, 50, 130, 97], positive: false }];
        let mut sorted: Vec<f32> = [[0, 0, 40, 30], [60, 50, 130, 97]].iter().flat_map(|p: &Part| (p[1]..p[3]).flat_map(move |y| (p[0]..p[2]).map(move |x| (x, y)))).map(|(x, y)| data[y * width + x]).collect();
        sorted.sort_by(f32::total_cmp);
        assert_eq!(ranked(client, &boxes, |n| vec![n / 3]).1, vec![Some(sorted[sorted.len() / 3])]);
        // Rankings side by side, each as it is alone, a rank past the
        // values among them.
        let (steep, light) = ([Ranked { values: &on_card, width, part: [0, 0, 40, 30], positive: true }], [Ranked { values: &on_card, width, part: [7, 5, 90, 60], positive: false }]);
        let (halfway, ends) = (|n: usize| vec![n / 2], |n: usize| vec![0, n * 49 / 50, n + 3]);
        assert_eq!(ranked_together(client, &[(&steep, &halfway), (&light, &ends)]), vec![ranked(client, &steep, halfway), ranked(client, &light, ends)]);
        // More ranks than a pass seeks at once.
        let many = |n: usize| (0..2 * MOST_SEEKS + 3).map(|k| k * n / (2 * MOST_SEEKS + 3)).collect::<Vec<usize>>();
        let mut sorted = data.clone();
        sorted.sort_by(f32::total_cmp);
        let whole = Ranked { values: &on_card, width, part: [0, 0, width, height], positive: false };
        assert_eq!(ranked(client, &[whole], many).1, many(sorted.len()).into_iter().map(|rank| Some(sorted[rank])).collect::<Vec<_>>());
    }

    #[test]
    fn a_large_image_is_found_as_busy_in_bands_as_whole() {
        let (width, height) = (300, 260);
        let data = photo(width, height);
        let whole = busy_within(usize::MAX, Values::Cores(&data), (width, height), 64, 0.15);
        assert_eq!(whole.len(), 5 * 5);
        assert!(whole.iter().all(|&b| b > 0.0));
        assert_eq!(busy_within(width * 64, Values::Cores(&data), (width, height), 64, 0.15), whole, "a row of parts at a time");
        let on_card = Image::of(Held::of(client(), &data), (width, height));
        assert_eq!(busy(Values::Card(&on_card), (width, height), 64, 0.15), whole);
        assert_eq!(busy_within(width * 64, Values::Card(&on_card), (width, height), 64, 0.15), whole);
        // Odd either way, so that the last pixel a change reaches is the
        // last of its row or its column: the cores hand the card the
        // pixels read alone, and it reads its own whole.
        for (width, height) in [(301, 263), (131, 67), (5, 3)] {
            let data = photo(width, height);
            let on_card = Image::of(Held::of(client(), &data), (width, height));
            let whole = busy(Values::Card(&on_card), (width, height), 64, 0.15);
            assert_eq!(busy_within(usize::MAX, Values::Cores(&data), (width, height), 64, 0.15), whole, "{width} by {height}");
            assert_eq!(busy_within(width * 64, Values::Cores(&data), (width, height), 64, 0.15), whole, "{width} by {height}, a row of parts at a time");
        }
        // What the cores added up, within the card's rounding.
        let changes: Vec<(usize, usize, f32)> = (0..height - 2)
            .step_by(2)
            .flat_map(|y| (0..width - 2).step_by(2).map(move |x| (x, y)))
            .map(|(x, y)| (x, y, (data[y * width + x + 2] - data[y * width + x]).abs() + (data[(y + 2) * width + x] - data[y * width + x]).abs()))
            .collect();
        let mut positive: Vec<f32> = changes.iter().map(|c| c.2).filter(|&c| c > 0.0).collect();
        let top = (positive.len() * 99 / 100).min(positive.len() - 1);
        let floor = 0.15 * *positive.select_nth_unstable_by(top, f32::total_cmp).1;
        let mut cores = vec![0.0f64; 25];
        for &(x, y, c) in &changes {
            if c >= floor {
                cores[y / 64 * 5 + x / 64] += f64::from(c);
            }
        }
        for (card, cores) in whole.iter().zip(&cores) {
            assert!((card - cores).abs() < 1e-4 * cores.max(1.0), "{card} against {cores}");
        }
    }
}
