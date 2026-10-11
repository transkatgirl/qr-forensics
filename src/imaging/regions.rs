//! Finder patterns from connected regions: a second way of proposing
//! where the finders of a symbol are, after BoofCV's
//! QrCodePositionPatternDetector.
//!
//! The row scan in `detect` needs the 1:1:3:1:1 cross-section of a finder
//! to be intact along at least two lines through its centre. A scratch
//! across a finder breaks every row it crosses and a second scratch down
//! it breaks the columns as well; a finder turned far from square, or seen
//! at a steep angle, keeps the proportion only along its own axes, which
//! the scan does not follow. BoofCV goes the other way about: it labels the
//! connected regions of the binarised image, fits a polygon to the contour
//! of each, keeps the quadrilaterals, and then asks the pixels along two
//! lines through each quadrilateral, each joining the midpoints of a pair
//! of opposite sides, whether they read as 1:1:3:1:1, and is content when
//! either does. The region gives the square's own axes whatever its turn
//! or perspective, and a ring survives as one region through any number of
//! dark scratches, and through a light one that cuts it in one place.
//!
//! Here the regions are labelled by union-find over the runs of each row,
//! with eight-way connectivity, and the quadrilateral is the one of
//! greatest area whose corners are vertices of the region's convex hull:
//! for a square ring seen in perspective that is the ring's own outline,
//! and a region that is nothing like a quadrilateral gives one that covers
//! too little of its hull, and is dropped. The hull is taken over the
//! corners of the pixels rather than their centres, so the corners come
//! out on the edge of the ink. Both lines must read, where BoofCV asks one
//! of the two: each is tried at the midpoints and a little to either side
//! of them, taking the best, so that a scratch along the very middle of a
//! finder does not fail it, and a region that reads along both is let past
//! the detector's cap on the candidates it refines, which a test that one
//! line could pass would not earn it. The labelling is run twice over each
//! polarity, once on the image as it is and once with gaps of up to two
//! pixels closed over (a dilation followed by an erosion: the
//! morphological closing), so that a finder cut into pieces by thin
//! scratches is seen whole. The line checks read whichever image the
//! labelling was made from: a scratch along the centre row of a finder
//! cuts every line across it, however far from the middle, and only the
//! closed image reads clean, as zxing-cpp retries a failed read on a
//! denoised copy with `tryDenoise`. Light-on-dark symbols are found by
//! running the whole procedure again on the complemented image.
//!
//! Every candidate goes on to the detector's own template fit, which has
//! the last word, so the checks here are generous: they are there to keep
//! the count of candidates small, not to make the final decision.

use std::f64::consts::SQRT_2;

/// A finder pattern suggested by a connected region of the binarised image.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RegionCandidate {
    /// Centre of the pattern, in the pixel coordinates of `dark`: where the
    /// diagonals of the quadrilateral cross, which is where the centre of a
    /// square lands under perspective.
    pub x: f64,
    pub y: f64,
    /// Module size along each axis (quad width / 7, quad height / 7).
    pub module: [f64; 2],
    /// Light on dark: found as a light region in dark surroundings.
    pub inverted: bool,
    /// The outer quadrilateral of the pattern, in pixel coordinates, going
    /// clockwise as seen on screen (the y-axis points down) from the corner
    /// nearest the top left: the detector starts its template fit from it.
    pub corners: [[f64; 2]; 4],
    /// How far the two line profiles stray from 1:1:3:1:1: the sum, over
    /// both lines, of the relative deviation of each of the five runs, and
    /// of how far the span of the five falls short of the quadrilateral,
    /// weighted. Zero is a perfect finder, and candidates are returned
    /// best first.
    pub score: f64,
}

/// What a pixel of the image reads as: light, dark, or nothing, for a
/// pixel the photo does not cover.
const LIGHT: u8 = 0;
const DARK: u8 = 1;
const NONE: u8 = 2;

/// A finder's outer ring is seven modules on a side, so a region whose
/// bounding box is shorter than this many modules on its longer side
/// cannot be one; the allowance is for a ring eroded by light bleeding
/// into it.
const MIN_LONG_SIDE: f64 = 6.0;
/// The shorter side may be much less: a finder seen at a steep angle is
/// foreshortened along one axis.
const MIN_SHORT_SIDE: f64 = 3.5;
/// A square turned forty-five degrees spans 7·√2 modules across its
/// bounding box; a little more is allowed for a ring grown by blur.
const MAX_LONG_SIDE: f64 = 7.0 * SQRT_2 * 1.3;
/// Least share of its bounding box a region's pixels must fill. A finder's
/// ring fills about half of its box when square to the image and a
/// quarter when turned forty-five degrees, less again when light has bled
/// into it; a thin diagonal stroke fills far less, and is not worth a hull.
const MIN_FILL: f64 = 0.1;
/// Least share of its convex hull's area the fitted quadrilateral must
/// cover. A square ring at any turn or perspective gives nearly all of it;
/// a round blob gives its inscribed square, about two thirds.
const MIN_QUAD_SHARE: f64 = 0.7;
/// How far the module read off the quadrilateral may fall outside the
/// range asked for: the quadrilateral is fitted to the ink, which blur
/// grows or shrinks.
const MODULE_SLACK: (f64, f64) = (0.8, 1.25);
/// The two mean side lengths of the quadrilateral may differ by this
/// factor at most: a square in perspective is foreshortened, but not into
/// a bar. Three to one is a finder seen seventy degrees off the square,
/// past which its modules are too thin along one axis to read. The
/// factor was one and a half, forty-eight degrees, and the finders of a
/// symbol photographed from low across a table, two to one and more,
/// were turned away here with their four corners found: the scan sees
/// such a finder along a line or two if at all, and no template started
/// square to the frame settles on it. At two, and at two and a half, one
/// photograph of five so taken was still found only up the ladder.
const MAX_SIDE_RATIO: f64 = 3.0;
/// How far beyond the quadrilateral each line is sampled, as a share of
/// its length: about half a module, enough that the outer ring is measured
/// whole when the quadrilateral falls a little short of it, and little
/// enough that print the finder stands hard against does not swell the
/// outer runs past their band.
const LINE_EXTENSION: f64 = 0.075;
/// Where the lines cross the sides, as offsets from the midpoint in shares
/// of the side: the midpoint itself, as BoofCV samples it, and 0.7 of a
/// module to either side, still well within the three-module stone. The
/// best of the three is taken, so a scratch along the middle of the finder
/// does not fail it.
const LINE_OFFSETS: [f64; 3] = [0.0, -0.1, 0.1];
/// Samples along a line: BoofCV's forty-six at the least, and four a pixel
/// up to a cap, so that the runs are measured to a quarter pixel.
const MIN_SAMPLES: usize = 46;
const MAX_SAMPLES: usize = 800;
/// The most runs a line through a finder may have: five for the finder
/// itself, two more for the paper either side, and two for some print or
/// noise, but no more, or the region is not a finder at all.
const MAX_RUNS: usize = 9;
/// The band, in modules, each of the outer four runs must lie in, and the
/// one for the centre run.
const OUTER_RUN: (f64, f64) = (0.4, 1.8);
const CENTRE_RUN: (f64, f64) = (2.0, 4.4);
/// Weight of the span term of the score: how far the five runs of a line,
/// taken together, fall short of or overrun the quadrilateral's length,
/// as a share of it. A quadrilateral that has taken in a speck of noise
/// beside the ring reads the same five runs as the right one, and this is
/// what tells them apart, so a tenth of overrun must count for more than
/// a run a tenth of a module out.
const SPAN_WEIGHT: f64 = 5.0;
/// How far from the centre, as a share of the quadrilateral's sides, its
/// four diagonal companions lie: 0.7 of a module along each axis, within
/// the stone, and off any scratch along the centre row or column.
const CENTRE_OFFSET: f64 = 0.1;
/// Most candidates returned, so a busy page cannot flood the detector.
/// A candidate has read as a finder along two lines, which print seldom
/// does, and the cap is far above what a sheet of symbols holds: at
/// sixty-four it cut a sheet of sixty symbols, a hundred and eighty
/// finders, to a third.
const MAX_CANDIDATES: usize = 1024;
/// Hull vertices the search for the hull's diameter looks at, at most:
/// the search is quadratic in them, and the corners are polished against
/// the whole hull afterwards.
const DIAMETER_SAMPLES: usize = 256;
/// Rounds of polishing of the quadrilateral's corners against the hull.
const POLISH_ROUNDS: usize = 3;

/// Finder patterns suggested by the connected regions of a binarised image
/// of `width` × `height` pixels. `dark(x, y)` is `None` off the image; a
/// module is between `min_module` and `max_module` pixels. Pixel centres
/// are at (x + 0.5, y + 0.5), as elsewhere in the crate. Candidates come
/// best first, at most [`MAX_CANDIDATES`] of them, with any two of the same
/// polarity whose centres lie within a module of each other merged into
/// the better.
pub(crate) fn finder_regions(
    width: usize,
    height: usize,
    dark: &dyn Fn(i64, i64) -> Option<bool>,
    min_module: f64,
    max_module: f64,
) -> Vec<RegionCandidate> {
    if width == 0
        || height == 0
        || !(min_module.is_finite() && max_module.is_finite())
        || min_module <= 0.0
        || max_module < min_module
    {
        return Vec::new();
    }
    let mut image = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            image.push(match dark(x as i64, y as i64) {
                Some(true) => DARK,
                Some(false) => LIGHT,
                None => NONE,
            });
        }
    }
    let limits = Limits::new(min_module, max_module);
    let mut found = Vec::new();
    for inverted in [false, true] {
        let ink = if inverted { LIGHT } else { DARK };
        let mask: Vec<u8> = image.iter().map(|&v| u8::from(v == ink)).collect();
        let closed = close(&mask, width, height);
        for labelling in [&mask, &closed] {
            let field = Field { mask: labelling, image: &image, width, height };
            for points in region_points(labelling, width, height, &limits) {
                let Some(quad) = quad_of(points) else { continue };
                if let Some(candidate) = candidate(&quad, &field, &limits, inverted) {
                    found.push(candidate);
                }
            }
        }
    }
    rank(found)
}

/// Sizes a region must have to be worth a hull, and a module may have.
struct Limits {
    min_long: f64,
    min_short: f64,
    max_long: f64,
    module: (f64, f64),
}

impl Limits {
    fn new(min_module: f64, max_module: f64) -> Self {
        Self {
            min_long: MIN_LONG_SIDE * min_module,
            min_short: MIN_SHORT_SIDE * min_module,
            max_long: MAX_LONG_SIDE * max_module,
            module: (MODULE_SLACK.0 * min_module, MODULE_SLACK.1 * max_module),
        }
    }

    /// Whether a region of this extent could be a finder's ring.
    fn plausible(&self, extent: &Extent) -> bool {
        let (w, h) = ((extent.x1 - extent.x0) as f64, (extent.y1 - extent.y0) as f64);
        let (long, short) = (w.max(h), w.min(h));
        long >= self.min_long && short >= self.min_short && long <= self.max_long && extent.count as f64 >= MIN_FILL * w * h
    }
}

/// The image the lines are read from: a mask of the ink being looked for,
/// as it is or closed, and the image as read, for where the photo is not.
struct Field<'a> {
    mask: &'a [u8],
    image: &'a [u8],
    width: usize,
    height: usize,
}

impl Field<'_> {
    /// What the pixel under the continuous point `p` reads: ink, paper, or
    /// nothing, off the image or off the photo.
    fn kind(&self, p: [f64; 2]) -> Option<bool> {
        let (x, y) = (p[0].floor(), p[1].floor());
        if x < 0.0 || y < 0.0 || x >= self.width as f64 || y >= self.height as f64 {
            return None;
        }
        let i = y as usize * self.width + x as usize;
        (self.image[i] != NONE).then(|| self.mask[i] == 1)
    }
}

/// Morphological closing of a mask of 0 and 1 by a 3 × 3 square: a
/// dilation, then an erosion. Gaps of one or two pixels between pixels of
/// the mask are filled, and nothing else changes, so a ring cut by thin
/// scratches comes back whole. Each step is separable into a pass along
/// the rows and one down the columns.
fn close(mask: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut scratch = vec![0u8; mask.len()];
    let mut dilated = vec![0u8; mask.len()];
    spread(mask, &mut scratch, &mut dilated, width, height, u8::max);
    let mut closed = vec![0u8; mask.len()];
    spread(&dilated, &mut scratch, &mut closed, width, height, u8::min);
    closed
}

/// `dst` takes `op` over each pixel of `src` and its eight neighbours, the
/// edge pixels standing in for whatever lies beyond the image, so that a
/// region touching the edge is neither grown nor shrunk there.
fn spread(src: &[u8], scratch: &mut [u8], dst: &mut [u8], width: usize, height: usize, op: fn(u8, u8) -> u8) {
    for (row, out) in src.chunks_exact(width).zip(scratch.chunks_exact_mut(width)) {
        if width == 1 {
            out[0] = row[0];
            continue;
        }
        out[0] = op(row[0], row[1]);
        out[width - 1] = op(row[width - 2], row[width - 1]);
        for (window, o) in row.windows(3).zip(out[1..width - 1].iter_mut()) {
            *o = op(op(window[0], window[1]), window[2]);
        }
    }
    for y in 0..height {
        let above = &scratch[y.saturating_sub(1) * width..][..width];
        let here = &scratch[y * width..][..width];
        let below = &scratch[(y + 1).min(height - 1) * width..][..width];
        for (((o, &a), &h), &b) in dst[y * width..][..width].iter_mut().zip(above).zip(here).zip(below) {
            *o = op(op(a, h), b);
        }
    }
}

/// A run of mask pixels [x0, x1) in one row, and its union-find link to
/// the region it belongs to: a run is its own parent while it is a root.
#[derive(Clone, Copy)]
struct Run {
    x0: u32,
    x1: u32,
    parent: u32,
}

/// What is known of a region: its pixel count and bounding box, with the
/// far sides exclusive. Kept per root.
#[derive(Clone, Copy)]
struct Extent {
    count: u32,
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
}

impl Extent {
    fn absorb(&mut self, other: Extent) {
        self.count += other.count;
        self.x0 = self.x0.min(other.x0);
        self.x1 = self.x1.max(other.x1);
        self.y0 = self.y0.min(other.y0);
        self.y1 = self.y1.max(other.y1);
    }
}

/// Root of the region run `i` belongs to, halving the path on the way up.
fn find(runs: &mut [Run], mut i: u32) -> u32 {
    loop {
        let parent = runs[i as usize].parent;
        if parent == i {
            return i;
        }
        let grandparent = runs[parent as usize].parent;
        runs[i as usize].parent = grandparent;
        i = grandparent;
    }
}

/// Join the regions of runs `a` and `b`. The root of lower index stays the
/// root, so a region's root is always its first run.
fn unite(runs: &mut [Run], extents: &mut [Extent], a: u32, b: u32) {
    let (ra, rb) = (find(runs, a), find(runs, b));
    if ra == rb {
        return;
    }
    let (keep, drop) = if ra < rb { (ra, rb) } else { (rb, ra) };
    runs[drop as usize].parent = keep;
    let absorbed = extents[drop as usize];
    extents[keep as usize].absorb(absorbed);
}

/// The corners of the pixels of every region of `mask` (1 where the ink
/// is) whose extent is plausible for a finder, one list of points per
/// region. The regions are the eight-connected components, labelled in one
/// pass over the rows: each row's runs are matched against the previous
/// row's with two cursors, and runs that touch are united. Every pixel in
/// a run lies between the run's ends, so the hull of the run corners is
/// the hull of the region.
fn region_points(mask: &[u8], width: usize, height: usize, limits: &Limits) -> Vec<Vec<[f64; 2]>> {
    let mut runs: Vec<Run> = Vec::new();
    let mut extents: Vec<Extent> = Vec::new();
    let mut rows: Vec<usize> = Vec::with_capacity(height + 1);
    rows.push(0);
    for y in 0..height {
        let row = &mask[y * width..][..width];
        let first = runs.len();
        let mut x = 0;
        while let Some(skip) = row[x..].iter().position(|&v| v != 0) {
            let x0 = x + skip;
            x = x0 + row[x0..].iter().position(|&v| v == 0).unwrap_or(width - x0);
            let i = runs.len() as u32;
            runs.push(Run { x0: x0 as u32, x1: x as u32, parent: i });
            extents.push(Extent { count: (x - x0) as u32, x0: x0 as u32, x1: x as u32, y0: y as u32, y1: y as u32 + 1 });
            if x >= width {
                break;
            }
        }
        // Two runs in adjacent rows touch, eight ways, when their spans
        // overlap or abut diagonally. Whichever ends first can touch
        // nothing further along the other row.
        let (mut i, mut j) = (if y > 0 { rows[y - 1] } else { 0 }, first);
        let end = runs.len();
        while i < first && j < end {
            let (a, b) = (runs[i], runs[j]);
            if a.x0 <= b.x1 && b.x0 <= a.x1 {
                unite(&mut runs, &mut extents, i as u32, j as u32);
            }
            if a.x1 < b.x1 {
                i += 1;
            } else {
                j += 1;
            }
        }
        rows.push(end);
    }

    let mut slot = vec![u32::MAX; runs.len()];
    let mut regions = 0u32;
    for i in 0..runs.len() {
        if runs[i].parent == i as u32 && limits.plausible(&extents[i]) {
            slot[i] = regions;
            regions += 1;
        }
    }
    let mut points = vec![Vec::new(); regions as usize];
    if regions == 0 {
        return points;
    }
    for y in 0..height {
        for i in rows[y]..rows[y + 1] {
            let s = slot[find(&mut runs, i as u32) as usize];
            if s != u32::MAX {
                let (x0, x1, y0, y1) = (runs[i].x0 as f64, runs[i].x1 as f64, y as f64, y as f64 + 1.0);
                points[s as usize].extend([[x0, y0], [x1, y0], [x0, y1], [x1, y1]]);
            }
        }
    }
    points
}

/// Twice the signed area of the triangle `o`, `a`, `b`.
fn cross(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// Area of a simple polygon, by the shoelace formula.
fn polygon_area(polygon: &[[f64; 2]]) -> f64 {
    let n = polygon.len();
    (0..n).map(|i| polygon[i][0] * polygon[(i + 1) % n][1] - polygon[(i + 1) % n][0] * polygon[i][1]).sum::<f64>().abs() / 2.0
}

/// Convex hull of `points` by Andrew's monotone chain, without collinear
/// vertices, in the order that runs clockwise as seen on screen.
fn convex_hull(mut points: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    points.sort_unstable_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    points.dedup();
    if points.len() < 3 {
        return points;
    }
    let mut hull: Vec<[f64; 2]> = Vec::with_capacity(points.len() + 1);
    for &p in &points {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    let lower = hull.len() + 1;
    for &p in points.iter().rev().skip(1) {
        while hull.len() >= lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    hull
}

/// Indices of a ring of `n` strictly between `from` and `to`, going round.
fn between(n: usize, from: usize, to: usize) -> impl Iterator<Item = usize> {
    (1..n).map(move |d| (from + d) % n).take_while(move |&k| k != to)
}

/// The two hull vertices farthest apart. Of a large hull every few
/// vertices are looked at; the polishing afterwards finds the exact
/// corners.
fn diameter(hull: &[[f64; 2]]) -> (usize, usize) {
    let stride = hull.len().div_ceil(DIAMETER_SAMPLES).max(1);
    let mut best = (0, 0, -1.0);
    for i in (0..hull.len()).step_by(stride) {
        for j in (i + stride..hull.len()).step_by(stride) {
            let d = (hull[i][0] - hull[j][0]).powi(2) + (hull[i][1] - hull[j][1]).powi(2);
            if d > best.2 {
                best = (i, j, d);
            }
        }
    }
    (best.0, best.1)
}

/// The hull vertex between `from` and `to` farthest from the line through
/// them, if there is one.
fn farthest(hull: &[[f64; 2]], from: usize, to: usize) -> Option<usize> {
    between(hull.len(), from, to)
        .max_by(|&a, &b| cross(hull[from], hull[to], hull[a]).abs().total_cmp(&cross(hull[from], hull[to], hull[b]).abs()))
}

/// The quadrilateral of greatest area with its corners on the hull of
/// `points`, when it covers enough of the hull to be the hull's shape. The
/// hull's diameter is a diagonal of the square the region is hoped to be,
/// and the vertices farthest from it on either side are the other two
/// corners; a few rounds of moving each corner, in turn, to the hull
/// vertex that makes the quadrilateral largest then settle it exactly.
fn quad_of(points: Vec<[f64; 2]>) -> Option<[[f64; 2]; 4]> {
    let hull = convex_hull(points);
    if hull.len() < 4 {
        return None;
    }
    let (i, j) = diameter(&hull);
    let mut quad = [i, farthest(&hull, i, j)?, j, farthest(&hull, j, i)?];
    let area_of = |quad: [usize; 4]| polygon_area(&quad.map(|k| hull[k]));
    for _ in 0..POLISH_ROUNDS {
        let mut moved = false;
        for k in 0..4 {
            let (previous, next) = (quad[(k + 3) % 4], quad[(k + 1) % 4]);
            let mut best = (quad[k], area_of(quad));
            for v in between(hull.len(), previous, next) {
                let mut trial = quad;
                trial[k] = v;
                let area = area_of(trial);
                if area > best.1 + 1e-9 {
                    best = (v, area);
                }
            }
            if best.0 != quad[k] {
                quad[k] = best.0;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    let corners = quad.map(|k| hull[k]);
    (polygon_area(&corners) >= MIN_QUAD_SHARE * polygon_area(&hull)).then_some(corners)
}

/// The point a share `t` of the way from `a` to `b`.
fn lerp(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// Where the diagonals of a quadrilateral cross.
fn diagonal_crossing(quad: &[[f64; 2]; 4]) -> Option<[f64; 2]> {
    let [c0, c1, c2, c3] = *quad;
    let d1 = [c2[0] - c0[0], c2[1] - c0[1]];
    let d2 = [c3[0] - c1[0], c3[1] - c1[1]];
    let denominator = d1[0] * d2[1] - d1[1] * d2[0];
    if denominator.abs() < 1e-12 {
        return None;
    }
    let t = ((c1[0] - c0[0]) * d2[1] - (c1[1] - c0[1]) * d2[0]) / denominator;
    Some([c0[0] + t * d1[0], c0[1] + t * d1[1]])
}

/// How far the pixels along the line from `a` to `b`, and a little beyond
/// each end, stray from reading as a finder's 1:1:3:1:1, or `None` when
/// they do not read as one at all. The samples are run-length encoded; of
/// the windows of five runs that go ink, paper, ink, paper, ink and whose
/// runs each lie within their band of the module the window implies, the
/// one closest to the proportion counts, with the five runs together
/// expected to span the line from `a` to `b` and no more. Too many runs,
/// and the line crosses something other than a finder.
fn line_profile(field: &Field, a: [f64; 2], b: [f64; 2]) -> Option<f64> {
    let samples = ((4.0 * distance(a, b)).ceil() as usize).clamp(MIN_SAMPLES, MAX_SAMPLES);
    let mut runs: Vec<(Option<bool>, u32)> = Vec::with_capacity(16);
    for k in 0..samples {
        let t = -LINE_EXTENSION + (1.0 + 2.0 * LINE_EXTENSION) * (k as f64 + 0.5) / samples as f64;
        let kind = field.kind(lerp(a, b, t));
        match runs.last_mut() {
            Some(run) if run.0 == kind => run.1 += 1,
            _ => runs.push((kind, 1)),
        }
    }
    let counted = runs.iter().filter(|run| run.0.is_some()).count();
    if !(5..=MAX_RUNS).contains(&counted) {
        return None;
    }
    let expected = [1.0, 1.0, 3.0, 1.0, 1.0];
    let mut best: Option<f64> = None;
    for window in runs.windows(5) {
        if window.iter().enumerate().any(|(i, run)| run.0 != Some(i % 2 == 0)) {
            continue;
        }
        let lengths: [f64; 5] = std::array::from_fn(|i| window[i].1 as f64);
        let module = lengths.iter().sum::<f64>() / 7.0;
        let within = lengths.iter().enumerate().all(|(i, &length)| {
            let (low, high) = if i == 2 { CENTRE_RUN } else { OUTER_RUN };
            length >= low * module && length <= high * module
        });
        if !within {
            continue;
        }
        let span = 7.0 * module / samples as f64 * (1.0 + 2.0 * LINE_EXTENSION);
        let deviation = lengths.iter().zip(expected).map(|(&length, e)| (length / module - e).abs() / e).sum::<f64>()
            + SPAN_WEIGHT * (span - 1.0).abs();
        if best.is_none_or(|b| deviation < b) {
            best = Some(deviation);
        }
    }
    best
}

/// The best line profile across the quadrilateral from side `from` to the
/// opposite side `to`, each given from the corner the sides share a
/// direction with, at the midpoints and offset to either side of them.
fn best_line(field: &Field, from: ([f64; 2], [f64; 2]), to: ([f64; 2], [f64; 2])) -> Option<f64> {
    LINE_OFFSETS
        .iter()
        .filter_map(|&offset| line_profile(field, lerp(from.0, from.1, 0.5 + offset), lerp(to.0, to.1, 0.5 + offset)))
        .min_by(f64::total_cmp)
}

/// Judge a quadrilateral as a finder pattern: its sides must be those of a
/// square in perspective, its centre and the four points about it must
/// read as ink, and both lines across it must read as 1:1:3:1:1.
fn candidate(quad: &[[f64; 2]; 4], field: &Field, limits: &Limits, inverted: bool) -> Option<RegionCandidate> {
    let [c0, c1, c2, c3] = *quad;
    let mean_a = (distance(c0, c1) + distance(c2, c3)) / 2.0;
    let mean_b = (distance(c1, c2) + distance(c3, c0)) / 2.0;
    if mean_a.min(mean_b) <= 0.0 || mean_a.max(mean_b) > MAX_SIDE_RATIO * mean_a.min(mean_b) {
        return None;
    }
    // The pair of sides that runs closer to the x-axis gives the module
    // along x.
    let along_a = [c1[0] - c0[0] + c2[0] - c3[0], c1[1] - c0[1] + c2[1] - c3[1]];
    let along_b = [c2[0] - c1[0] + c3[0] - c0[0], c2[1] - c1[1] + c3[1] - c0[1]];
    let module = if along_a[0].abs() >= along_a[1].abs() { [mean_a / 7.0, mean_b / 7.0] } else { [mean_b / 7.0, mean_a / 7.0] };
    if !module.iter().all(|m| (limits.module.0..=limits.module.1).contains(m)) {
        return None;
    }
    let centre = diagonal_crossing(quad)?;
    let companions = [[0.0, 0.0], [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]];
    let ink = companions
        .iter()
        .filter(|&&[p, q]| {
            let (p, q) = (p * CENTRE_OFFSET / 2.0, q * CENTRE_OFFSET / 2.0);
            field.kind([centre[0] + p * along_a[0] + q * along_b[0], centre[1] + p * along_a[1] + q * along_b[1]]) == Some(true)
        })
        .count();
    if ink < 3 {
        return None;
    }
    let across_a = best_line(field, (c0, c1), (c3, c2))?;
    let across_b = best_line(field, (c1, c2), (c0, c3))?;
    let start = (0..4).min_by(|&i, &j| {
        (quad[i][0] + quad[i][1]).total_cmp(&(quad[j][0] + quad[j][1])).then(quad[i][1].total_cmp(&quad[j][1]))
    })?;
    let corners = std::array::from_fn(|k| quad[(start + k) % 4]);
    Some(RegionCandidate { x: centre[0], y: centre[1], module, inverted, corners, score: across_a + across_b })
}

/// Order candidates best first, larger first among equals, and keep the
/// best of any that lie within a module of each other at the same
/// polarity, up to the cap.
fn rank(mut found: Vec<RegionCandidate>) -> Vec<RegionCandidate> {
    let size = |c: &RegionCandidate| (c.module[0] + c.module[1]) / 2.0;
    found.sort_by(|a, b| a.score.total_cmp(&b.score).then_with(|| size(b).total_cmp(&size(a))));
    let mut kept: Vec<RegionCandidate> = Vec::new();
    for c in found {
        if kept.len() >= MAX_CANDIDATES {
            break;
        }
        let duplicate =
            kept.iter().any(|k| k.inverted == c.inverted && distance([k.x, k.y], [c.x, c.y]) <= size(k).max(size(&c)));
        if !duplicate {
            kept.push(c);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// A binary image drawn from a predicate on pixel centres.
    struct Bitmap {
        width: usize,
        height: usize,
        dark: Vec<bool>,
    }

    impl Bitmap {
        fn draw(width: usize, height: usize, is_dark: impl Fn(f64, f64) -> bool) -> Self {
            let dark = (0..height)
                .flat_map(|y| (0..width).map(move |x| (x, y)))
                .map(|(x, y)| is_dark(x as f64 + 0.5, y as f64 + 0.5))
                .collect();
            Self { width, height, dark }
        }

        fn at(&self, x: i64, y: i64) -> Option<bool> {
            (x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height)
                .then(|| self.dark[y as usize * self.width + x as usize])
        }

        /// Turn every pixel `light` names light.
        fn scratch(&mut self, light: impl Fn(usize, usize) -> bool) {
            for y in 0..self.height {
                for x in 0..self.width {
                    if light(x, y) {
                        self.dark[y * self.width + x] = false;
                    }
                }
            }
        }

        fn find(&self, min_module: f64, max_module: f64) -> Vec<RegionCandidate> {
            finder_regions(self.width, self.height, &|x, y| self.at(x, y), min_module, max_module)
        }
    }

    /// Whether module coordinates (u, v) of a finder pattern, which occupies
    /// [0, 7)², are dark; `None` outside it.
    fn finder(u: f64, v: f64) -> Option<bool> {
        if !(0.0..7.0).contains(&u) || !(0.0..7.0).contains(&v) {
            return None;
        }
        let ring = |t: f64| !(1.0..6.0).contains(&t);
        let stone = |t: f64| (2.0..5.0).contains(&t);
        Some(ring(u) || ring(v) || (stone(u) && stone(v)))
    }

    /// Module coordinates of a pixel, for a finder of `module` pixels a
    /// module turned `degrees` about `centre`.
    fn turned(centre: [f64; 2], module: f64, degrees: f64) -> impl Fn(f64, f64) -> (f64, f64) {
        let (s, c) = degrees.to_radians().sin_cos();
        move |x, y| {
            let (dx, dy) = (x - centre[0], y - centre[1]);
            ((c * dx + s * dy) / module + 3.5, (-s * dx + c * dy) / module + 3.5)
        }
    }

    /// A projective map applied to (u, v).
    fn apply(h: &[[f64; 3]; 3], u: f64, v: f64) -> [f64; 2] {
        let w = h[2][0] * u + h[2][1] * v + h[2][2];
        [(h[0][0] * u + h[0][1] * v + h[0][2]) / w, (h[1][0] * u + h[1][1] * v + h[1][2]) / w]
    }

    /// The inverse of a projective map, up to scale: its adjugate.
    fn inverse(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
        [
            [m[1][1] * m[2][2] - m[1][2] * m[2][1], m[0][2] * m[2][1] - m[0][1] * m[2][2], m[0][1] * m[1][2] - m[0][2] * m[1][1]],
            [m[1][2] * m[2][0] - m[1][0] * m[2][2], m[0][0] * m[2][2] - m[0][2] * m[2][0], m[0][2] * m[1][0] - m[0][0] * m[1][2]],
            [m[1][0] * m[2][1] - m[1][1] * m[2][0], m[0][1] * m[2][0] - m[0][0] * m[2][1], m[0][0] * m[1][1] - m[0][1] * m[1][0]],
        ]
    }

    /// A finder of five pixels a module with its top left corner at (10, 10).
    fn upright() -> Bitmap {
        Bitmap::draw(64, 64, |x, y| finder((x - 10.0) / 5.0, (y - 10.0) / 5.0).unwrap_or(false))
    }

    #[test]
    fn an_upright_finder_is_found_with_its_corners() {
        let found = upright().find(2.0, 10.0);
        assert_eq!(found.len(), 1, "{found:?}");
        let c = found[0];
        assert!((c.x - 27.5).abs() < 1e-9 && (c.y - 27.5).abs() < 1e-9, "{c:?}");
        assert!(c.module.iter().all(|m| (m - 5.0).abs() < 1e-9), "{c:?}");
        assert!(!c.inverted);
        assert_eq!(c.corners, [[10.0, 10.0], [45.0, 10.0], [45.0, 45.0], [10.0, 45.0]]);
        assert!(c.score < 0.5, "{c:?}");
    }

    #[test]
    fn a_turned_finder_is_found() {
        for degrees in [30.0, 45.0, 75.0] {
            let to_module = turned([32.0, 32.0], 5.0, degrees);
            let bitmap = Bitmap::draw(64, 64, |x, y| {
                let (u, v) = to_module(x, y);
                finder(u, v).unwrap_or(false)
            });
            let found = bitmap.find(2.0, 10.0);
            assert_eq!(found.len(), 1, "{degrees}°: {found:?}");
            let c = found[0];
            assert!((c.x - 32.0).abs() < 1.0 && (c.y - 32.0).abs() < 1.0, "{degrees}°: {c:?}");
            assert!(c.module.iter().all(|m| (m - 5.0).abs() < 0.75), "{degrees}°: {c:?}");
            assert!(!c.inverted);
            // Every corner lies close to one of the square's own.
            let true_corners: Vec<[f64; 2]> = [[0.0, 0.0], [7.0, 0.0], [7.0, 7.0], [0.0, 7.0]]
                .iter()
                .map(|&[u, v]| {
                    let (s, co) = degrees.to_radians().sin_cos();
                    let (du, dv) = ((u - 3.5) * 5.0, (v - 3.5) * 5.0);
                    [32.0 + co * du - s * dv, 32.0 + s * du + co * dv]
                })
                .collect();
            for corner in c.corners {
                let nearest = true_corners.iter().map(|&t| distance(t, corner)).fold(f64::INFINITY, f64::min);
                assert!(nearest < 1.5, "{degrees}°: corner {corner:?} is {nearest} from the square's, {c:?}");
            }
        }
    }

    #[test]
    fn a_finder_in_perspective_is_found() {
        // Module coordinates to pixels, with the far side smaller.
        let h = [[5.0, 1.0, 12.0], [0.5, 4.5, 14.0], [0.004, 0.006, 1.0]];
        let back = inverse(&h);
        let bitmap = Bitmap::draw(64, 64, |x, y| {
            let [u, v] = apply(&back, x, y);
            finder(u, v).unwrap_or(false)
        });
        let found = bitmap.find(2.0, 10.0);
        assert_eq!(found.len(), 1, "{found:?}");
        let c = found[0];
        let centre = apply(&h, 3.5, 3.5);
        assert!(distance([c.x, c.y], centre) < 1.0, "{c:?} against {centre:?}");
        let corners = [[0.0, 0.0], [7.0, 0.0], [7.0, 7.0], [0.0, 7.0]].map(|[u, v]| apply(&h, u, v));
        let module = (0..4).map(|i| distance(corners[i], corners[(i + 1) % 4])).sum::<f64>() / 28.0;
        let mean = (c.module[0] + c.module[1]) / 2.0;
        assert!((mean - module).abs() < 0.15 * module, "{c:?} against {module}");
        for (corner, truth) in c.corners.iter().zip(corners) {
            assert!(distance(*corner, truth) < 1.5, "{c:?} against {corners:?}");
        }
    }

    #[test]
    fn a_thin_scratch_through_the_centre_both_ways_is_closed_over() {
        // A one-pixel light line along the centre row and another down the
        // centre column cut the ring into four pieces and the stone into
        // quarters, and no row or column through the centre reads as a
        // finder any more. The closing joins the pieces, and the lines
        // either side of the middle read clean.
        let mut bitmap = upright();
        bitmap.scratch(|x, y| x == 27 || y == 27);
        let found = bitmap.find(2.0, 10.0);
        assert_eq!(found.len(), 1, "{found:?}");
        let c = found[0];
        assert!((c.x - 27.5).abs() < 0.5 && (c.y - 27.5).abs() < 0.5, "{c:?}");
        assert!(c.module.iter().all(|m| (m - 5.0).abs() < 0.5), "{c:?}");
        // Two pixels wide is still closed over.
        let mut bitmap = upright();
        bitmap.scratch(|x, y| (27..29).contains(&x) || (27..29).contains(&y));
        assert_eq!(bitmap.find(2.0, 10.0).len(), 1);
    }

    #[test]
    fn a_wide_scratch_through_the_centre_both_ways_is_beyond_the_closing() {
        // Known limitation: the closing bridges two pixels, not three. A
        // wider cross leaves four separate pieces, none of them a finder.
        let mut bitmap = upright();
        bitmap.scratch(|x, y| (26..29).contains(&x) || (26..29).contains(&y));
        assert!(bitmap.find(2.0, 10.0).is_empty());
    }

    #[test]
    fn a_scratch_through_the_ring_alone_is_survived() {
        // Three pixels wide, down through the top of the ring only: the
        // ring is cut in one place and stays one region, and the line down
        // the middle fails while those either side of it pass.
        let mut bitmap = upright();
        bitmap.scratch(|x, y| (26..29).contains(&x) && y < 16);
        let found = bitmap.find(2.0, 10.0);
        assert_eq!(found.len(), 1, "{found:?}");
        let c = found[0];
        assert!((c.x - 27.5).abs() < 0.5 && (c.y - 27.5).abs() < 0.5, "{c:?}");
        // A dark scratch joining the ring to the stone changes nothing of
        // the outline.
        let bitmap = Bitmap::draw(64, 64, |x, y| {
            finder((x - 10.0) / 5.0, (y - 10.0) / 5.0).unwrap_or(false) || (x > 10.0 && x < 45.0 && (y - 20.0).abs() < 0.5)
        });
        let found = bitmap.find(2.0, 10.0);
        assert_eq!(found.len(), 1, "{found:?}");
        assert!((found[0].x - 27.5).abs() < 0.5 && (found[0].y - 27.5).abs() < 0.5, "{found:?}");
    }

    #[test]
    fn squares_that_are_not_finders_are_rejected() {
        let square = |u: f64, v: f64| (0.0..7.0).contains(&u) && (0.0..7.0).contains(&v);
        let solid = Bitmap::draw(64, 64, |x, y| square((x - 10.0) / 5.0, (y - 10.0) / 5.0));
        assert!(solid.find(2.0, 10.0).is_empty());
        // A ring round a light hole, with no stone.
        let hollow = Bitmap::draw(64, 64, |x, y| {
            let (u, v) = ((x - 10.0) / 5.0, (y - 10.0) / 5.0);
            square(u, v) && !((1.0..6.0).contains(&u) && (1.0..6.0).contains(&v))
        });
        assert!(hollow.find(2.0, 10.0).is_empty());
        // A square with a one-module hole at its centre.
        let pierced = Bitmap::draw(64, 64, |x, y| {
            let (u, v) = ((x - 10.0) / 5.0, (y - 10.0) / 5.0);
            square(u, v) && !((3.0..4.0).contains(&u) && (3.0..4.0).contains(&v))
        });
        assert!(pierced.find(2.0, 10.0).is_empty());
        // A ring filled with stripes a module wide, so that one axis reads
        // 2:1:1:1:2 and the other solid.
        let striped = Bitmap::draw(64, 64, |x, y| {
            let (u, v) = ((x - 10.0) / 5.0, (y - 10.0) / 5.0);
            square(u, v) && (!(1.0..6.0).contains(&v) || u.floor() as i64 % 2 == 1)
        });
        assert!(striped.find(2.0, 10.0).is_empty());
        // A round blob with the right rings: a bullseye.
        let bullseye = Bitmap::draw(64, 64, |x, y| {
            let r = (x - 32.0).hypot(y - 32.0) / 5.0;
            r < 3.5 && !(1.5..2.5).contains(&r)
        });
        assert!(bullseye.find(2.0, 10.0).is_empty());
    }

    #[test]
    fn a_light_finder_on_dark_is_found_inverted() {
        let bitmap = Bitmap::draw(64, 64, |x, y| !finder((x - 10.0) / 5.0, (y - 10.0) / 5.0).unwrap_or(false));
        let found = bitmap.find(2.0, 10.0);
        assert_eq!(found.len(), 1, "{found:?}");
        let c = found[0];
        assert!(c.inverted);
        assert!((c.x - 27.5).abs() < 1e-9 && (c.y - 27.5).abs() < 1e-9, "{c:?}");
        assert!(c.module.iter().all(|m| (m - 5.0).abs() < 1e-9), "{c:?}");
    }

    #[test]
    fn candidates_are_deduplicated_and_sorted() {
        // Two finders: one clean, one whose ink has bled a fifth of a
        // module outward. Each is found by the plain labelling and by the
        // closed one, and once only is reported; the clean one first.
        let bitmap = Bitmap::draw(128, 64, |x, y| {
            let clean = finder((x - 10.0) / 5.0, (y - 10.0) / 5.0).unwrap_or(false);
            let (u, v) = ((x - 74.0) / 5.0, (y - 10.0) / 5.0);
            let ring = |t: f64| !(1.2..5.8).contains(&t);
            let stone = |t: f64| (1.8..5.2).contains(&t);
            let bled = (0.0..7.0).contains(&u) && (0.0..7.0).contains(&v) && (ring(u) || ring(v) || (stone(u) && stone(v)));
            clean || bled
        });
        let found = bitmap.find(2.0, 10.0);
        assert_eq!(found.len(), 2, "{found:?}");
        assert!((found[0].x - 27.5).abs() < 1e-9 && (found[1].x - 91.5).abs() < 1e-9, "{found:?}");
        assert!(found[0].score < found[1].score, "{found:?}");
        assert!(found[0].score < 0.5 && found[1].score > 1.0, "{found:?}");
    }

    #[test]
    fn every_finder_of_a_sheet_is_a_candidate() {
        // A ten by ten grid of finders of four pixels a module: a sheet
        // of thirty-three symbols has as many, and a cap of sixty-four
        // kept two in three of them.
        let bitmap = Bitmap::draw(400, 400, |x, y| {
            let (u, v) = ((x % 40.0) / 4.0, (y % 40.0) / 4.0);
            finder(u, v).unwrap_or(false)
        });
        let found = bitmap.find(2.0, 10.0);
        assert_eq!(found.len(), 100);
        assert!(found.len() <= MAX_CANDIDATES);
        assert!(found.windows(2).all(|w| w[0].score <= w[1].score));
    }

    #[test]
    fn a_finder_seen_steeply_is_found_and_a_bar_is_not() {
        // A finder foreshortened two and a half to one, as a symbol
        // photographed from low across a table shows it: eight pixels a
        // module along the rows and three and a fifth down the columns.
        let steep = |down: f64| Bitmap::draw(96, 48, move |x, y| finder((x - 10.0) / 8.0, (y - 10.0) / down).unwrap_or(false));
        let found = steep(3.2).find(2.0, 10.0);
        assert_eq!(found.len(), 1, "{found:?}");
        let c = found[0];
        assert!((c.x - 38.0).abs() < 0.5 && (c.y - 21.2).abs() < 0.5, "{c:?}");
        assert!((c.module[0] - 8.0).abs() < 0.2 && (c.module[1] - 3.2).abs() < 0.2, "{c:?}");
        for (corner, truth) in c.corners.iter().zip([[10.0, 10.0], [66.0, 10.0], [66.0, 32.4], [10.0, 32.4]]) {
            assert!(distance(*corner, truth) < 1.0, "{c:?}");
        }
        // Three and a half to one is past what a finder is read at.
        assert!(steep(2.3).find(2.0, 10.0).is_empty());
    }

    #[test]
    fn a_large_image_is_handled_quickly() {
        // A 2048 × 2048 image of speckle noise with eight finders of
        // various sizes and turns scattered over it, each with a module
        // of clear paper round it, as a symbol's separator gives. Printed
        // with --nocapture; the time is the thing to look at, in release.
        let finders: [([f64; 2], f64, f64); 8] = [
            ([200.0, 200.0], 4.0, 0.0),
            ([600.0, 300.0], 6.0, 20.0),
            ([1200.0, 250.0], 9.0, 45.0),
            ([1800.0, 400.0], 12.0, 70.0),
            ([300.0, 1000.0], 5.0, 10.0),
            ([1000.0, 1100.0], 16.0, 33.0),
            ([1700.0, 1300.0], 7.0, 85.0),
            ([900.0, 1800.0], 24.0, 5.0),
        ];
        let maps: Vec<_> = finders.iter().map(|&(centre, module, degrees)| turned(centre, module, degrees)).collect();
        let (width, height) = (2048usize, 2048usize);
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut dark = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                let speck = (seed >> 33) % 100 < 5;
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                let mut value = speck;
                for to_module in &maps {
                    let (u, v) = to_module(px, py);
                    if (-1.0..8.0).contains(&u) && (-1.0..8.0).contains(&v) {
                        value = finder(u, v).unwrap_or(false);
                    }
                }
                dark.push(value);
            }
        }
        let bitmap = Bitmap { width, height, dark };
        let start = Instant::now();
        let found = bitmap.find(3.0, 30.0);
        let elapsed = start.elapsed();
        println!("regions: {} candidates from a {width} x {height} image in {elapsed:?}", found.len());
        for (centre, module, degrees) in finders {
            let hit = found.iter().any(|c| distance([c.x, c.y], centre) < 1.5 && (c.module[0] - module).abs() < 0.15 * module);
            assert!(hit, "finder at {centre:?} of {module} px a module turned {degrees}° missed: {found:?}");
        }
        assert!(found.len() <= MAX_CANDIDATES);
    }
}
