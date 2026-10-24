//! Corner refinement: nudge the four corners of the outline to where the
//! version's fixed patterns read best.
//!
//! The score is the contrast across every boundary between two fixed
//! modules of opposite colour: luminance a little way into the light side
//! minus luminance the same distance into the dark side, measured at a few
//! points along the boundary and at a few distances from it. Uneven
//! lighting changes both sides alike and cancels, and a boundary placed a
//! fraction of a pixel off the real edge loses contrast at once, so the
//! score has a sharp peak at the true outline. Coordinate descent then
//! finds it down to a twentieth of a pixel. Besides moving one corner at a
//! time it tries joint moves of all four, since on a long thin symbol a
//! slightly turned outline is a local maximum that no single corner can
//! leave.
//!
//! The boundaries are combined with a trimmed mean: the fifth of them that
//! read worst are left out. A finder pattern with a corner torn off, or an
//! alignment pattern under a stain, keeps most of its boundaries, and
//! those place it as well as ever, where a plain mean would be dragged
//! towards the damage. ZBar fits its finder edges with RANSAC for the same
//! reason.
//!
//! A light-on-dark symbol reverses the contrast across every boundary, so
//! the search is run for both polarities and the outline with the stronger
//! contrast wins; the result says which polarity that was.
//!
//! When anchors are known, the outline is a [`Projection`] through them
//! rather than a plain homography, and the corners are refined with the
//! anchors held fixed.
//!
//! The same scorer serves the coarse alignment in [`super::detect`], there
//! applied to one finder or alignment pattern at a time through a
//! [`Template`] of that patch alone.

use std::collections::HashSet;
use std::sync::Arc;

use super::Luma;
use super::homography::Homography;
use super::projection::{ControlPoint, Project, Projection, Spline};
use super::sample::{INNER_FRACTION, POINTS, SampleParams, otsu, sample_positions};
use crate::symbol::Pos;
use crate::symbol::function_map::FunctionMap;
use crate::work;

/// How far the refinement may move a corner a finder pattern's own fit
/// placed, in modules.
const HELD_TRAVEL: f64 = 0.05;
/// Smallest move tried, in photo pixels.
const MIN_STEP: f64 = 0.05;
/// Passes over the corners at one step size before it is halved.
const MAX_PASSES: usize = 20;
/// Distances from a boundary, in photo pixels, at which contrast is taken.
/// Several, so that a blurrier edge still gives a peak at the right place.
const SCALES: [f64; 3] = [0.5, 1.0, 2.0];
/// A further distance as a share of the module: an over-exposed photo
/// moves every boundary a fraction of a module into the dark, and a pair
/// this far apart still straddles the moved boundary.
const MODULE_SCALE_FRACTION: f64 = 0.3;
/// No further across a boundary than this fraction of a module, so the
/// sample stays inside the neighbouring module.
const MAX_SCALE_FRACTION: f64 = 0.4;
/// Part of a boundary's length that is sampled, keeping clear of the module
/// corners where diagonal neighbours intrude.
const ALONG_FRACTION: f64 = 0.7;
/// Share of the boundaries, those reading worst, left out of the score.
pub const TRIM_FRACTION: f64 = 0.2;
/// The corner of a template that [`Scorer::agreement_shaded`] takes one
/// threshold from, as a share of the template along each side: a finder
/// pattern with its separator and the paper beyond it is just under two
/// fifths of the smallest QR Code with its quiet zone, and two fifths of
/// a larger symbol holds more of its patterns.
const SHADE_CORNER: f64 = 0.4;
/// Fewest modules of each colour a corner must hold for the split of
/// what it reads to stand as its threshold: a corner of paper alone, or
/// of paper and two or three timing modules, has no split to speak of.
const SHADE_MIN_EACH: usize = 4;
/// How far each corner (top-left, top-right, bottom-right, bottom-left)
/// moves in one trial, as multiples of the step along one axis: each corner
/// alone, then the four patterns that together span every joint move of
/// the outline: a shift, a turn, and a squeeze across each diagonal pair.
pub(crate) const MOVES: [[f64; 4]; 8] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, -1.0, -1.0, 1.0],
    [1.0, 1.0, -1.0, -1.0],
    [1.0, -1.0, 1.0, -1.0],
];
/// The joint shift alone: for a patch too small to pin down its own shape.
pub(crate) const SHIFT: [[f64; 4]; 1] = [[1.0, 1.0, 1.0, 1.0]];

/// The fixed modules of a rectangular patch of a symbol, in the patch's own
/// module coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Template {
    pub width: usize,
    pub height: usize,
    /// Colour of each module, row-major: `Some(true)` is dark and `None`
    /// means the module's colour is not fixed.
    pub expected: Vec<Option<bool>>,
}

impl Template {
    /// Every fixed module of a version.
    pub fn whole(func: &FunctionMap) -> Self {
        Self::window(func, 0, 0, func.width, func.height)
    }

    /// The fixed modules inside a rectangle of the symbol, whose top-left
    /// module is (`x0`, `y0`). Modules outside the symbol are not fixed.
    pub fn window(func: &FunctionMap, x0: i64, y0: i64, width: usize, height: usize) -> Self {
        let expected = (0..height)
            .flat_map(|dy| (0..width).map(move |dx| (dx, dy)))
            .map(|(dx, dy)| {
                let (x, y) = (x0 + dx as i64, y0 + dy as i64);
                let inside = x >= 0 && y >= 0 && (x as usize) < func.width && (y as usize) < func.height;
                inside.then(|| func.expected(x as usize, y as usize)).flatten()
            })
            .collect();
        Self { width, height, expected }
    }

    /// Every fixed module of a version and the first module of quiet zone
    /// around it, which must be light: two modules wider and higher than
    /// the symbol, whose top-left module is at (1, 1).
    pub fn with_quiet_zone(func: &FunctionMap) -> Self {
        let mut template = Self::window(func, -1, -1, func.width + 2, func.height + 2);
        for y in 0..template.height {
            for x in 0..template.width {
                if x == 0 || y == 0 || x + 1 == template.width || y + 1 == template.height {
                    template.expected[y * template.width + x] = Some(false);
                }
            }
        }
        template
    }

    /// A finder pattern with the ring of light modules around it: the
    /// separator on its inner sides and the quiet zone on the outer ones.
    /// Nine modules square, the finder's centre at (4.5, 4.5).
    pub fn finder_with_ring() -> Self {
        let expected = (0..9usize)
            .flat_map(|y| (0..9usize).map(move |x| (x, y)))
            .map(|(x, y)| {
                let ring = x.abs_diff(4).max(y.abs_diff(4));
                Some(ring == 3 || ring <= 1)
            })
            .collect();
        Self { width: 9, height: 9, expected }
    }

    /// A finder pattern alone, seven modules square.
    pub fn finder_core() -> Self {
        let expected = (0..7usize)
            .flat_map(|y| (0..7usize).map(move |x| (x, y)))
            .map(|(x, y)| Some(x.abs_diff(3).max(y.abs_diff(3)) != 2))
            .collect();
        Self { width: 7, height: 7, expected }
    }

    /// The template with one quarter (as [`Scorer::agreement_quarters`]
    /// numbers them) left unfixed: for a pattern torn there.
    pub fn without_quarter(&self, quarter: usize) -> Self {
        let mut out = self.clone();
        for y in 0..self.height {
            for x in 0..self.width {
                let q = usize::from(2 * x + 1 >= self.width) + 2 * usize::from(2 * y + 1 >= self.height);
                if q == quarter {
                    out.expected[y * self.width + x] = None;
                }
            }
        }
        out
    }

    pub fn expected(&self, x: usize, y: usize) -> Option<bool> {
        self.expected[y * self.width + x]
    }

    /// How many modules are fixed.
    pub fn fixed(&self) -> usize {
        self.expected.iter().flatten().count()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Refinement {
    pub corners: [[f64; 2]; 4],
    /// (matching, fixed) modules at the corners given, read the way the
    /// sampler reads them. When `inverted`, matching means the inverse of
    /// the specified colour.
    pub before: (usize, usize),
    /// The same after refinement.
    pub after: (usize, usize),
    pub evaluations: usize,
    /// The outline was found with dark and light swapped: the photo shows
    /// a light symbol on a dark background.
    pub inverted: bool,
}

/// A boundary between two fixed modules of opposite colours.
struct Edge {
    /// Centre of the boundary, in module units.
    at: [f64; 2],
    /// Unit step from the dark side to the light side.
    across: [f64; 2],
    /// Unit step along the boundary.
    along: [f64; 2],
}

/// A map shifted in module space: for scoring a template whose origin is
/// not the symbol's.
pub struct Shifted<'a, P: Project> {
    pub map: &'a P,
    /// Added to template coordinates to get the map's coordinates.
    pub dx: f64,
    pub dy: f64,
}

impl<P: Project> Project for Shifted<'_, P> {
    fn base(&self, x: f64, y: f64) -> [f64; 2] {
        self.map.base(x + self.dx, y + self.dy)
    }

    fn displacement(&self, x: f64, y: f64) -> [f64; 2] {
        self.map.displacement(x + self.dx, y + self.dy)
    }
}

/// A spline's weights at the centre of every boundary of a template, laid
/// out a site at a time: the weight of site `j` at boundary `e` is
/// `by_site[j * edges + e]`. The displacement at each boundary is its
/// weights times the residuals, summed site by site; laid out so, the
/// sums of all the boundaries are taken side by side, each in the order
/// it would be taken alone, where a boundary at a time waits on every
/// addition before the next.
struct EdgeWeights {
    /// Sites each boundary has a weight for: none when there is no
    /// boundary.
    sites: usize,
    edges: usize,
    by_site: Vec<f64>,
}

impl EdgeWeights {
    fn new(per_edge: &[Vec<f64>]) -> Self {
        let (sites, edges) = (per_edge.first().map_or(0, Vec::len), per_edge.len());
        let by_site = (0..sites).flat_map(|j| per_edge.iter().map(move |weights| weights[j])).collect();
        Self { sites, edges, by_site }
    }

    /// Every `stride`th boundary's weights, as [`Scorer::with_max_edges`]
    /// keeps the boundaries.
    fn step_by(&self, stride: usize) -> Self {
        let edges = self.edges.div_ceil(stride);
        let by_site = self.by_site.chunks_exact(self.edges.max(1)).flat_map(|column| column.iter().step_by(stride).copied()).collect();
        Self { sites: self.sites, edges, by_site }
    }

    /// The displacement at every boundary, by the residuals of a
    /// projection with this spline, as
    /// [`Projection::displacement_from_weights`] gives each.
    fn displacements(&self, residuals: &[[f64; 2]]) -> Vec<[f64; 2]> {
        let mut out = vec![[0.0, 0.0]; self.edges];
        if self.edges == 0 {
            return out;
        }
        for (column, r) in self.by_site.chunks_exact(self.edges).zip(residuals) {
            for (d, &w) in out.iter_mut().zip(column) {
                d[0] += w * r[0];
                d[1] += w * r[1];
            }
        }
        out
    }
}

/// Scores placements of one template against the photo.
pub(crate) struct Scorer<'a> {
    image: &'a Luma,
    width: f64,
    height: f64,
    edges: Vec<Edge>,
    scales: Vec<f64>,
    /// Sample pairs along each boundary.
    along: usize,
    positions: Vec<Pos>,
    expected_dark: Vec<bool>,
    threshold: Option<f32>,
    module_px: f64,
    /// Spline weights at each boundary's centre, for projections that
    /// share one spline: the displacement there is then a weighted sum of
    /// the projection's residuals.
    edge_weights: Option<EdgeWeights>,
    /// How far, in modules, every boundary is expected to have moved into
    /// the dark: the bleed of an over-exposed photo. Negative for light
    /// bled into by dark.
    bloom: f64,
}

impl<'a> Scorer<'a> {
    /// A scorer for `template` placed at `corners`, which set the scale of
    /// the sampling. `threshold` fixes the dark/light split used by
    /// [`Scorer::agreement`]; without it the split is found per call.
    pub(crate) fn new(image: &'a Luma, template: &Template, corners: [[f64; 2]; 4], threshold: Option<f32>) -> Self {
        let mut positions = Vec::new();
        let mut expected_dark = Vec::new();
        let mut edges = Vec::new();
        for y in 0..template.height {
            for x in 0..template.width {
                let Some(dark) = template.expected(x, y) else { continue };
                positions.push(Pos::new(x, y));
                expected_dark.push(dark);
                let sign = if dark { 1.0 } else { -1.0 };
                if x + 1 < template.width && template.expected(x + 1, y) == Some(!dark) {
                    edges.push(Edge {
                        at: [x as f64 + 1.0, y as f64 + 0.5],
                        across: [sign, 0.0],
                        along: [0.0, 1.0],
                    });
                }
                if y + 1 < template.height && template.expected(x, y + 1) == Some(!dark) {
                    edges.push(Edge {
                        at: [x as f64 + 0.5, y as f64 + 1.0],
                        across: [0.0, sign],
                        along: [1.0, 0.0],
                    });
                }
            }
        }
        let width = template.width as f64;
        let height = template.height as f64;
        let module_px = module_pixels(corners, width, height);
        let mut scales: Vec<f64> = SCALES.iter().copied().filter(|&s| s <= MAX_SCALE_FRACTION * module_px).collect();
        let wide = MODULE_SCALE_FRACTION * module_px;
        if scales.last().is_none_or(|&s| s < wide) {
            scales.push(wide.max(MIN_STEP));
        }
        Scorer {
            image,
            width,
            height,
            edges,
            scales,
            // About two sample pairs per pixel of boundary.
            along: ((2.0 * module_px).round() as usize).clamp(2, 16),
            positions,
            expected_dark,
            threshold,
            module_px,
            edge_weights: None,
            bloom: 0.0,
        }
    }

    /// Expect every boundary `bloom` modules into the dark side of where
    /// the template has it.
    pub(crate) fn with_bloom(mut self, bloom: f64) -> Self {
        self.bloom = bloom;
        self
    }

    /// Take at most `max` sample pairs along each boundary: faster, for
    /// searches that only need to tell a good placement from a bad one.
    pub(crate) fn with_along(mut self, max: usize) -> Self {
        self.along = self.along.min(max.max(1));
        self
    }

    /// Keep at most `max` boundaries, every n-th in raster order, for a
    /// template with far more than a descent needs to place the corners.
    pub(crate) fn with_max_edges(mut self, max: usize) -> Self {
        let max = max.max(1);
        if self.edges.len() > max {
            let stride = self.edges.len().div_ceil(max);
            self.edges = std::mem::take(&mut self.edges).into_iter().step_by(stride).collect();
            if let Some(weights) = &mut self.edge_weights {
                *weights = weights.step_by(stride);
            }
        }
        self
    }

    /// Precompute the spline's weights at every boundary, for scoring
    /// projections that share `spline`. Only right for a template in the
    /// symbol's own coordinates.
    pub(crate) fn with_spline(mut self, spline: Option<&Spline>) -> Self {
        self.edge_weights = spline.map(|s| EdgeWeights::new(&self.edges.iter().map(|e| s.weights(e.at)).collect::<Vec<_>>()));
        self
    }

    /// Size of a module in photo pixels at the corners given to [`Scorer::new`].
    pub(crate) fn module_px(&self) -> f64 {
        self.module_px
    }

    /// Fixed modules in the template.
    pub(crate) fn fixed(&self) -> usize {
        self.positions.len()
    }

    /// Trimmed mean contrast across the boundaries of the template placed
    /// through `map`, times `sign`, where -1 looks for a light-on-dark
    /// symbol; `None` for a degenerate placement.
    pub(crate) fn contrast_with(&self, map: &impl Project, sign: f64) -> Option<f64> {
        self.contrast_edges(sign, |e, edge| {
            let _ = e;
            map.displacement(edge.at[0], edge.at[1])
        }, |x, y| map.base(x, y))
    }

    /// The same for a projection sharing the spline given to
    /// [`Scorer::with_spline`], which avoids re-evaluating the spline.
    pub(crate) fn contrast_projection(&self, map: &Projection, sign: f64) -> Option<f64> {
        match &self.edge_weights {
            Some(weights) if map.spline().is_some_and(|s| s.len() == weights.sites) => {
                let displacements = weights.displacements(map.residuals());
                self.contrast_edges(sign, |e, _| displacements[e], |x, y| map.base(x, y))
            }
            _ => self.contrast_with(map, sign),
        }
    }

    fn contrast_edges(
        &self,
        sign: f64,
        displacement: impl Fn(usize, &Edge) -> [f64; 2],
        base: impl Fn(f64, f64) -> [f64; 2],
    ) -> Option<f64> {
        let mut per_edge: Vec<f64> = Vec::with_capacity(self.edges.len());
        // Where each sample pair lies along a boundary, the same for
        // every boundary.
        let along: Vec<f64> = (0..self.along).map(|k| ALONG_FRACTION * ((k as f64 + 0.5) / self.along as f64 - 0.5)).collect();
        for (e, edge) in self.edges.iter().enumerate() {
            let [dx, dy] = displacement(e, edge);
            let (mut total, mut count) = (0.0f64, 0usize);
            for &t in &along {
                let (mx, my) = (
                    edge.at[0] + edge.along[0] * t - edge.across[0] * self.bloom,
                    edge.at[1] + edge.along[1] * t - edge.across[1] * self.bloom,
                );
                let p = base(mx, my);
                // The across direction in photo pixels, from a short step.
                let q = base(mx + 0.01 * edge.across[0], my + 0.01 * edge.across[1]);
                let (ax, ay) = (q[0] - p[0], q[1] - p[1]);
                let length = (ax * ax + ay * ay).sqrt();
                if length == 0.0 || !length.is_finite() {
                    return None;
                }
                let (ux, uy) = (ax / length, ay / length);
                let (px, py) = (p[0] + dx, p[1] + dy);
                for &scale in &self.scales {
                    let light = self.image.sample(px + ux * scale, py + uy * scale);
                    let dark = light.and_then(|_| self.image.sample(px - ux * scale, py - uy * scale));
                    total += match (light, dark) {
                        (Some(light), Some(dark)) => sign * (light - dark) as f64,
                        // Off the image counts as wrong by the widest margin,
                        // whichever polarity is sought: the penalty must not
                        // turn into a reward when the sign is reversed, or a
                        // light-on-dark search is drawn off the edge of the
                        // photo, where everything reads as it pleases.
                        _ => -1.0,
                    };
                    count += 1;
                }
            }
            if count > 0 {
                per_edge.push(total / count as f64);
            }
        }
        trimmed_mean(&mut per_edge)
    }

    /// Mean contrast across the boundaries for the template placed at
    /// `corners`; `None` for degenerate corners.
    pub(crate) fn contrast(&self, corners: [[f64; 2]; 4]) -> Option<f64> {
        self.contrast_signed(corners, 1.0)
    }

    /// The same times `sign`, where -1 looks for a light-on-dark symbol.
    /// The sign goes in with the samples: negating the contrast of a
    /// dark-on-light reading afterwards would turn its penalty for
    /// sampling off the image into a reward, and trim away the edges that
    /// match best instead of those that match worst.
    pub(crate) fn contrast_signed(&self, corners: [[f64; 2]; 4], sign: f64) -> Option<f64> {
        let h = Homography::rect_to_quad(self.width, self.height, corners)?;
        self.contrast_with(&h, sign)
    }

    /// Fixed modules that come out the right colour when sampled the way
    /// the reader samples them, or the inverse colour when `inverted`.
    pub(crate) fn agreement(&self, corners: [[f64; 2]; 4], inverted: bool) -> Option<usize> {
        let h = Homography::rect_to_quad(self.width, self.height, corners)?;
        self.agreement_known(&h, inverted).map(|(matching, _)| matching)
    }

    /// The same, with the number of fixed modules the photo covers at all,
    /// for a template placed through `map`.
    pub(crate) fn agreement_known(&self, map: &impl Project, inverted: bool) -> Option<(usize, usize)> {
        let luma = sample_positions(self.image, map, &self.positions, INNER_FRACTION, POINTS);
        let threshold = self.threshold.unwrap_or_else(|| otsu(luma.iter().flatten().copied()));
        let known = luma.iter().flatten().count();
        let matching = luma
            .iter()
            .zip(&self.expected_dark)
            .filter(|(value, dark)| value.is_some_and(|v| (v < threshold) == (**dark != inverted)))
            .count();
        Some((matching, known))
    }

    /// [`Scorer::agreement_known`] against a threshold that follows the
    /// light across the template, where that reads every module against
    /// one split of the lot: each corner of the template has the split
    /// of its own samples, and a module is read against the four
    /// interpolated to where it lies. BoofCV reads a module so, against
    /// the thresholds of its position patterns carried across the symbol
    /// (`bitIntensityToBitValue`), and a read here may
    /// ([`super::sample::ThresholdSurface`]).
    ///
    /// One split serves a pattern a few modules across, and a whole
    /// symbol in even light. Light that falls off across a symbol leaves
    /// the paper at its dim end darker than the split of the whole, and
    /// every light module there counts as wrong. A version 2 symbol
    /// beside a lamp, the paper nearly three times as bright at one
    /// finder as at another, was placed on all three finders with 226 of
    /// its 236 fixed modules right and read 65 in a hundred with its
    /// quiet zone against one split; the same square a quarter turn out,
    /// on two of the finders, read 87.
    ///
    /// A corner with too few modules of either colour to have a split
    /// ([`SHADE_MIN_EACH`]) takes its threshold from the others: the
    /// plane through the other three when it alone is wanting, which is
    /// where a steady fall of light puts it, and their mean otherwise. A
    /// symbol with one finder pattern has little that is fixed beyond
    /// that corner, and is read against that corner's split all over. A
    /// threshold the scorer was given is used as it is.
    pub(crate) fn agreement_shaded(&self, map: &impl Project, inverted: bool) -> Option<(usize, usize)> {
        if self.threshold.is_some() {
            return self.agreement_known(map, inverted);
        }
        let luma = sample_positions(self.image, map, &self.positions, INNER_FRACTION, POINTS);
        let unit = |pos: &Pos| [(pos.x as f64 + 0.5) / self.width, (pos.y as f64 + 0.5) / self.height];
        let within = |f: f64, far: bool| if far { f > 1.0 - SHADE_CORNER } else { f < SHADE_CORNER };
        // Top-left, top-right, bottom-left, bottom-right.
        let read: [Option<f32>; 4] = std::array::from_fn(|i| {
            let (right, bottom) = (i % 2 == 1, i >= 2);
            let mut values = Vec::new();
            let mut colours = [0usize; 2];
            for ((value, &dark), pos) in luma.iter().zip(&self.expected_dark).zip(&self.positions) {
                let [fx, fy] = unit(pos);
                if let Some(value) = value
                    && within(fx, right)
                    && within(fy, bottom)
                {
                    values.push(*value);
                    colours[usize::from(dark)] += 1;
                }
            }
            colours.iter().all(|&n| n >= SHADE_MIN_EACH).then(|| otsu(values.into_iter()))
        });
        let (count, sum) = read.iter().flatten().fold((0usize, 0.0f32), |(n, sum), &t| (n + 1, sum + t));
        let rest = if count > 0 { sum / count as f32 } else { otsu(luma.iter().flatten().copied()) };
        let (low, high) = luma.iter().flatten().fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), &v| (low.min(v), high.max(v)));
        let [tl, tr, bl, br]: [f32; 4] = std::array::from_fn(|i| match (read[i], read[i ^ 1], read[i ^ 2], read[3 - i]) {
            (Some(own), ..) => own,
            // Beside it along the row, beside it down the column, and
            // across the diagonal: no further than anything read.
            (None, Some(across), Some(down), Some(opposite)) => (across + down - opposite).clamp(low, high),
            _ => rest,
        });
        let share = |f: f64| ((f - SHADE_CORNER / 2.0) / (1.0 - SHADE_CORNER)).clamp(0.0, 1.0) as f32;
        let known = luma.iter().flatten().count();
        let matching = luma
            .iter()
            .zip(&self.expected_dark)
            .zip(&self.positions)
            .filter(|((value, dark), pos)| {
                let [fx, fy] = unit(pos);
                let (u, v) = (share(fx), share(fy));
                let threshold = (1.0 - v) * ((1.0 - u) * tl + u * tr) + v * ((1.0 - u) * bl + u * br);
                value.is_some_and(|value| (value < threshold) == (**dark != inverted))
            })
            .count();
        Some((matching, known))
    }

    /// (matching, known) in each quarter of the template, top-left,
    /// top-right, bottom-left, bottom-right. A pattern with one quarter
    /// torn away still reads right in the other three; one that is not
    /// there at all reads wrong everywhere.
    pub(crate) fn agreement_quarters(&self, corners: [[f64; 2]; 4], inverted: bool) -> Option<[(usize, usize); 4]> {
        self.agreement_quarters_at(corners, inverted, None)
    }

    /// The same against a threshold of the caller's choosing, when there
    /// is a better one than the split of the template's own samples.
    pub(crate) fn agreement_quarters_at(
        &self,
        corners: [[f64; 2]; 4],
        inverted: bool,
        threshold: Option<f32>,
    ) -> Option<[(usize, usize); 4]> {
        let h = Homography::rect_to_quad(self.width, self.height, corners)?;
        let luma = sample_positions(self.image, &h, &self.positions, INNER_FRACTION, POINTS);
        let threshold = threshold.or(self.threshold).unwrap_or_else(|| otsu(luma.iter().flatten().copied()));
        let mut out = [(0usize, 0usize); 4];
        for ((value, dark), pos) in luma.iter().zip(&self.expected_dark).zip(&self.positions) {
            let Some(v) = value else { continue };
            let quarter = usize::from(2.0 * (pos.x as f64 + 0.5) >= self.width)
                + 2 * usize::from(2.0 * (pos.y as f64 + 0.5) >= self.height);
            out[quarter].1 += 1;
            out[quarter].0 += usize::from((*v < threshold) == (*dark != inverted));
        }
        Some(out)
    }

    /// Coordinate descent from `corners` on `sign` times the contrast,
    /// where -1 looks for the outline of a light-on-dark symbol, trying the
    /// given `moves` at `step` photo pixels first and halving from there.
    /// Returns the corners reached and the score there.
    pub(crate) fn descend(
        &self,
        corners: [[f64; 2]; 4],
        sign: f64,
        moves: &[[f64; 4]],
        step: f64,
        evaluations: &mut usize,
    ) -> Option<([[f64; 2]; 4], f64)> {
        self.descend_by(corners, moves, step, evaluations, |trial| {
            let h = Homography::rect_to_quad(self.width, self.height, trial)?;
            self.contrast_with(&h, sign)
        })
    }

    /// Coordinate descent on any score of the corners.
    pub(crate) fn descend_by(
        &self,
        corners: [[f64; 2]; 4],
        moves: &[[f64; 4]],
        step: f64,
        evaluations: &mut usize,
        score: impl Fn([[f64; 2]; 4]) -> Option<f64>,
    ) -> Option<([[f64; 2]; 4], f64)> {
        descend_by(corners, moves, step, evaluations, score)
    }

    /// The template's size in modules.
    pub(crate) fn template_width(&self) -> f64 {
        self.width
    }

    pub(crate) fn template_height(&self) -> f64 {
        self.height
    }
}

/// Coordinate descent from `corners` on `score`, trying the given `moves`
/// at `step` photo pixels first and halving from there, down to a
/// twentieth of a pixel. Returns the corners reached and the score there;
/// `None` when the start cannot be scored.
///
/// Each pass takes the first move that improves the score, so where a
/// descent ends up depends a little on the order the axes and directions
/// are swept in: a photo turned a quarter swaps the axes, and the same
/// start can then lead to another peak a fraction of a module off.
/// Sweeping in both orders, alternately or keeping the better of two
/// descents, was tried and found to settle on wrong peaks of the contrast
/// as often as it avoided them, so the sweep is left as it is, and a read
/// that fails is tried again with the photo turned instead, which
/// [`super::read`] does.
///
/// The best only rises, so corners scored once in a descent are refused
/// every time they come round again: they scored no more than the best
/// then, or were the best then and have been bettered since, or are the
/// corners as they stand. They are not scored again. That spares a pass
/// after one that improved the moves the last improvement left refused,
/// which it would otherwise score all over to find nothing; a move
/// `moves` holds twice; and the move straight back to where an
/// improvement came from, which lands on those very corners as often as
/// the step undone rounds away to nothing. The descent ends where it
/// ended when every trial was scored.
pub(crate) fn descend_by(
    corners: [[f64; 2]; 4],
    moves: &[[f64; 4]],
    step: f64,
    evaluations: &mut usize,
    score: impl Fn([[f64; 2]; 4]) -> Option<f64>,
) -> Option<([[f64; 2]; 4], f64)> {
    let mut best = score(corners)?;
    *evaluations += 1;
    let mut current = corners;
    let mut step = step.max(MIN_STEP);
    let mut scored: HashSet<[u64; 8]> = HashSet::new();
    scored.insert(bits_of(corners));
    while step >= MIN_STEP {
        for _ in 0..MAX_PASSES {
            let mut improved = false;
            for &pattern in moves {
                // Called off, the descent comes back with what it has.
                if work::cancelled() {
                    return Some((current, best));
                }
                for axis in 0..2 {
                    for direction in [1.0, -1.0] {
                        let mut trial = current;
                        for (corner, weight) in trial.iter_mut().zip(pattern) {
                            corner[axis] += direction * weight * step;
                        }
                        if !scored.insert(bits_of(trial)) {
                            continue;
                        }
                        *evaluations += 1;
                        if let Some(value) = score(trial)
                            && value > best
                        {
                            best = value;
                            current = trial;
                            improved = true;
                        }
                    }
                }
            }
            if !improved {
                break;
            }
        }
        step /= 2.0;
    }
    Some((current, best))
}

/// The corners as their bits: the same bits are the same corners, and
/// score the same.
fn bits_of(corners: [[f64; 2]; 4]) -> [u64; 8] {
    let [[a, b], [c, d], [e, f], [g, h]] = corners.map(|p| p.map(f64::to_bits));
    [a, b, c, d, e, f, g, h]
}

/// Mean of the values with the lowest [`TRIM_FRACTION`] of them left out.
/// Reorders `values`. `None` when there are none.
pub(crate) fn trimmed_mean(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let drop = ((values.len() as f64 * TRIM_FRACTION).floor() as usize).min(values.len() - 1);
    // A partition about the last value dropped, not a sort: the mean of
    // the kept values does not care for their order, and this runs in the
    // innermost loop of every placement scored.
    if drop > 0 {
        values.select_nth_unstable_by(drop, f64::total_cmp);
    }
    let kept = &values[drop..];
    Some(kept.iter().sum::<f64>() / kept.len() as f64)
}

/// Size of a module in photo pixels along the outline's shortest side.
fn module_pixels(corners: [[f64; 2]; 4], width: f64, height: f64) -> f64 {
    let distance = |a: [f64; 2], b: [f64; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    (distance(corners[0], corners[1]).min(distance(corners[3], corners[2])) / width)
        .min(distance(corners[0], corners[3]).min(distance(corners[1], corners[2])) / height)
}

/// (matching fixed modules, mean boundary contrast) for an outline, as the
/// refinement sees it. `None` when the corners do not form a usable outline.
pub fn alignment_score(
    image: &Luma,
    func: &FunctionMap,
    corners: [[f64; 2]; 4],
    params: &SampleParams,
) -> Option<(usize, f64)> {
    let scorer = Scorer::new(image, &Template::whole(func), corners, fixed_threshold(params));
    Some((scorer.agreement(corners, false)?, scorer.contrast(corners)?))
}

/// (matching, fixed) modules of a version's fixed patterns read through
/// `corners` and the map `anchors` bend, as a [`Refinement`] counts them,
/// with dark and light swapped when `inverted`. `None` when they form no
/// usable outline.
pub fn placement_agreement(
    image: &Luma,
    func: &FunctionMap,
    corners: [[f64; 2]; 4],
    anchors: &[ControlPoint],
    inverted: bool,
    params: &SampleParams,
) -> Option<(usize, usize)> {
    let scorer = Scorer::new(image, &Template::whole(func), corners, fixed_threshold(params));
    let map = Projection::new(func.width, func.height, corners, anchors)?;
    Some((scorer.agreement_known(&map, inverted)?.0, scorer.fixed()))
}

/// The threshold the sampler would apply to every module alike. A fixed
/// global threshold is honoured; local thresholds are not meaningful over
/// fixed modules alone.
pub(crate) fn fixed_threshold(params: &SampleParams) -> Option<f32> {
    params.threshold.filter(|_| !params.local)
}

/// The projections of one symbol's outline that share a set of anchors:
/// the spline is solved once and each placement of the corners reuses it.
pub struct Warp {
    width: usize,
    height: usize,
    anchors: Vec<ControlPoint>,
    spline: Option<Arc<Spline>>,
}

impl Warp {
    pub fn new(width: usize, height: usize, anchors: &[ControlPoint]) -> Self {
        // Build once to learn which sites the spline keeps, in which order.
        let probe = Projection::new(width, height, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], anchors);
        let spline = probe.and_then(|p| p.spline().cloned()).map(Arc::new);
        Self { width, height, anchors: anchors.to_vec(), spline }
    }

    pub fn spline(&self) -> Option<&Spline> {
        self.spline.as_deref()
    }

    /// The projection for one placement of the corners.
    pub fn place(&self, corners: [[f64; 2]; 4]) -> Option<Projection> {
        Projection::new(self.width, self.height, corners, &self.anchors)
    }
}

/// Move the corners to maximise how well the version's fixed patterns read,
/// at whichever polarity the photo turns out to have, with `anchors` held
/// fixed, and then settle any corner the fixed patterns cannot place by
/// the symbol's edge or its data modules. `None` when the corners given
/// do not form a usable outline.
pub fn refine_corners(
    image: &Luma,
    func: &FunctionMap,
    corners: [[f64; 2]; 4],
    anchors: &[ControlPoint],
    params: &SampleParams,
) -> Option<Refinement> {
    refine_corners_holding(image, func, corners, anchors, &[], params)
}

/// [`refine_corners`] with the corners in `held` kept where they are:
/// those a finder pattern's own fit has placed, which no descent of the
/// whole outline's contrast places better, and which on curled paper it
/// places worse.
pub fn refine_corners_holding(
    image: &Luma,
    func: &FunctionMap,
    corners: [[f64; 2]; 4],
    anchors: &[ControlPoint],
    held: &[usize],
    params: &SampleParams,
) -> Option<Refinement> {
    refine_corners_expecting(image, func, corners, anchors, held, 0.0, None, params)
}

/// [`refine_corners_holding`] expecting every boundary `bloom` modules
/// into the dark ([`super::detect::Detection::bloom`]), as the anchors of
/// a photo that has bled are refined ([`super::detect::refine_anchors`]):
/// fitted as printed, the outline of a symbol whose dark has thinned
/// settles to one side of it, as a finder pattern does. The corners
/// settled by the symbol's edge are settled expecting it too: the edge of
/// the ink a trace finds lies as far in as the bleed, and a QR Code of
/// version 1, with no alignment pattern, has its fourth corner so. The
/// symbol is taken to be light on dark when `inverted` says so, and where
/// it is `None` at whichever polarity reads the more crisply: a bleed is
/// into the symbol's dark, and one measured at one polarity is the
/// opposite bleed at the other.
#[allow(clippy::too_many_arguments)]
pub fn refine_corners_expecting(
    image: &Luma,
    func: &FunctionMap,
    corners: [[f64; 2]; 4],
    anchors: &[ControlPoint],
    held: &[usize],
    bloom: f64,
    inverted: Option<bool>,
    params: &SampleParams,
) -> Option<Refinement> {
    let warp = Warp::new(func.width, func.height, anchors);
    let scorer = Scorer::new(image, &Template::whole(func), corners, fixed_threshold(params)).with_spline(warp.spline()).with_bloom(bloom);
    let fixed = scorer.fixed();
    let mut evaluations = 0;
    // Start at half a module: a corner further off than that is misplaced,
    // not merely imprecise, and the search could settle a whole module away.
    let step = scorer.module_px() * 0.5;
    let (warp_ref, scorer_ref) = (&warp, &scorer);
    let score = |sign: f64| {
        move |trial: [[f64; 2]; 4]| {
            let map = warp_ref.place(trial)?;
            scorer_ref.contrast_projection(&map, sign)
        }
    };
    // A corner that no fixed pattern is near is moved by the contrast only
    // through the perspective of the whole outline, which is to say by
    // noise: on a blurred photo it wandered two modules off and dragged
    // the corners beside it along. Such a corner is held where it was
    // given through the descent, and placed once the others are settled
    // as the detector first placed it: by the symbol's edge where there
    // is a quiet zone to trace it against, and otherwise by how crisply
    // the data modules of its quarter read dark or light rather than grey.
    let anchored: Vec<[f64; 2]> = anchors.iter().map(|a| a.at).collect();
    let loose = super::detect::loose_corners(func, &anchored);
    // A corner whose only fixed patterns the anchors hold is likewise
    // beyond the contrast's reach, and keeps its place.
    let pinned = super::detect::corners_held_by_controls(func, anchors);
    let still: Vec<usize> = loose.iter().chain(&pinned).copied().collect();
    let moves = holding(&still);
    // The two polarities, where the polarity is not given, descend from
    // the same corners and take nothing from one another: side by side,
    // on a core to spare.
    let polarities: Vec<bool> = inverted.map_or_else(|| vec![false, true], |inverted| vec![inverted]);
    let descents = work::map(&polarities, |&inverted| {
        let mut evaluations = 0;
        (scorer.descend_by(corners, &moves, step, &mut evaluations, score(if inverted { -1.0 } else { 1.0 })), evaluations)
    });
    evaluations += descents.iter().map(|(_, n)| n).sum::<usize>();
    // Light on dark only where that reads more crisply.
    let mut crispest: Option<(bool, _, f64)> = None;
    for (&inverted, (descent, _)) in polarities.iter().zip(descents) {
        let (placed, contrast) = descent?;
        if crispest.is_none_or(|(_, _, best)| contrast > best) {
            crispest = Some((inverted, placed, contrast));
        }
    }
    let (inverted, mut current, _) = crispest?;
    // A corner a finder pattern's own fit placed is let move only the
    // last fraction of a pixel, which the whole outline's contrast does
    // place better on a flat photo: on curled paper the contrast would
    // walk it off the finder to where a plane reads every pattern best.
    for &i in held {
        if distance(current[i], corners[i]) > HELD_TRAVEL * scorer.module_px() {
            current[i] = corners[i];
        }
    }
    // The descent is believed only while it reads at least as many fixed
    // modules right as where it started: a peak of boundary contrast that
    // reads fewer is a wrong peak, as blur makes them, and the corners
    // given are kept instead, as quirc keeps a move of its perspective
    // only when the count of cells it reads right improves.
    let before = scorer.agreement_known(&warp.place(corners)?, inverted)?.0;
    if scorer.agreement_known(&warp.place(current)?, inverted)?.0 < before {
        current = corners;
    }
    let sign = if inverted { -1.0 } else { 1.0 };
    let current = super::detect::settle_loose_corners_with(image, func, current, &anchored, sign, bloom, params);
    let after = scorer.agreement_known(&warp.place(current)?, inverted)?.0;
    Some(Refinement { corners: current, before: (before, fixed), after: (after, fixed), evaluations, inverted })
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// The corner moves with the corners in `held` kept still: their part in
/// every pattern zeroed, and patterns that then move nothing dropped.
fn holding(held: &[usize]) -> Vec<[f64; 4]> {
    let mut moves: Vec<[f64; 4]> = MOVES
        .iter()
        .map(|pattern| {
            let mut pattern = *pattern;
            for &i in held {
                pattern[i] = 0.0;
            }
            pattern
        })
        .filter(|pattern| pattern.iter().any(|&v| v != 0.0))
        .collect();
    moves.dedup();
    moves
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::SymbolVersion;

    #[test]
    fn trimmed_mean_drops_the_worst_fifth() {
        let mut values = vec![0.5, 0.6, 0.55, 0.58, -0.9];
        assert!((trimmed_mean(&mut values).unwrap() - 0.5575).abs() < 1e-9);
        let mut one = vec![0.3];
        assert_eq!(trimmed_mean(&mut one), Some(0.3));
        assert_eq!(trimmed_mean(&mut []), None);
        // Fewer than five values: nothing is dropped.
        let mut four = vec![1.0, 2.0, 3.0, -4.0];
        assert_eq!(trimmed_mean(&mut four), Some(0.5));
    }

    /// The descent as it was before it passed over corners it had
    /// scored: every trial scored, which [`descend_by`] is held to.
    fn descend_scoring_every_trial(
        corners: [[f64; 2]; 4],
        moves: &[[f64; 4]],
        step: f64,
        evaluations: &mut usize,
        score: impl Fn([[f64; 2]; 4]) -> Option<f64>,
    ) -> Option<([[f64; 2]; 4], f64)> {
        let mut best = score(corners)?;
        *evaluations += 1;
        let mut current = corners;
        let mut step = step.max(MIN_STEP);
        while step >= MIN_STEP {
            for _ in 0..MAX_PASSES {
                let mut improved = false;
                for &pattern in moves {
                    for axis in 0..2 {
                        for direction in [1.0, -1.0] {
                            let mut trial = current;
                            for (corner, weight) in trial.iter_mut().zip(pattern) {
                                corner[axis] += direction * weight * step;
                            }
                            *evaluations += 1;
                            if let Some(value) = score(trial)
                                && value > best
                            {
                                best = value;
                                current = trial;
                                improved = true;
                            }
                        }
                    }
                }
                if !improved {
                    break;
                }
            }
            step /= 2.0;
        }
        Some((current, best))
    }

    #[test]
    fn a_refinement_called_off_comes_back_at_once_with_the_corners_given() {
        use crate::work::Cancel;
        let func = FunctionMap::new(SymbolVersion::Qr(2));
        let (image, corners) = drawn(&func, |_| 1.0);
        let loose = corners.map(|[x, y]| [x + 1.5, y - 1.0]);
        let params = SampleParams::default();
        // Not called off, the refinement moves the corners given.
        let refined = refine_corners(&image, &func, loose, &[], &params).expect("the corners given can be scored");
        assert_ne!(refined.corners, loose);
        assert!(refined.evaluations > 2);
        // Called off before it starts, it scores the corners given and
        // no trial, and comes back with them.
        let request = Cancel::new();
        request.cancel();
        let started = std::time::Instant::now();
        let refined = request.scope(|| refine_corners(&image, &func, loose, &[], &params)).expect("the corners given can be scored");
        assert_eq!(refined.corners, loose);
        assert_eq!(refined.evaluations, 2, "one score of the start at each polarity");
        assert!(started.elapsed().as_secs() < 5, "{:?}", started.elapsed());
    }

    #[test]
    fn a_descent_ends_where_scoring_every_trial_ends() {
        let func = FunctionMap::new(SymbolVersion::Qr(2));
        let (image, corners) = drawn(&func, |across| 1.0 - 0.3 * across as f32);
        let scorer = Scorer::new(&image, &Template::whole(&func), corners, None);
        let target = corners.map(|[x, y]| [x + 1.7, y - 0.9]);
        let smooth = |c: [[f64; 2]; 4]| Some(-c.iter().flatten().zip(target.iter().flatten()).map(|(a, b)| (a - b).powi(2)).sum::<f64>());
        // Plateaus, where a trial ties the best and is refused, and a
        // region that cannot be scored.
        let rugged = |c: [[f64; 2]; 4]| {
            let v: f64 = c.iter().flatten().enumerate().map(|(i, &x)| ((x * (1.3 + i as f64 * 0.17)).sin() * 4.0).floor()).sum();
            (v > -12.0).then_some(v)
        };
        let contrast = |c: [[f64; 2]; 4]| scorer.contrast_signed(c, 1.0);
        let scores: [&dyn Fn([[f64; 2]; 4]) -> Option<f64>; 3] = [&smooth, &rugged, &contrast];
        // `holding` leaves a move twice in the list when corners are held.
        let held = holding(&[2, 3]);
        let movesets: [&[[f64; 4]]; 4] = [&MOVES, &SHIFT, &held, &[[1.0, 0.0, 0.0, 0.0]]];
        let (mut fewer, mut runs) = (0, 0);
        for (s, score) in scores.iter().enumerate() {
            for moves in movesets {
                for (k, step) in [0.05, 0.4, 3.0].into_iter().enumerate() {
                    let start = corners.map(|[x, y]| [x + 0.37 * k as f64 - 0.5, y + 0.81 - 0.29 * k as f64]);
                    let (mut skipping, mut every) = (0, 0);
                    let ours = descend_by(start, moves, step, &mut skipping, score);
                    let reference = descend_scoring_every_trial(start, moves, step, &mut every, score);
                    let bits = |(c, v): ([[f64; 2]; 4], f64)| (c.map(|p| p.map(f64::to_bits)), v.to_bits());
                    assert_eq!(ours.map(bits), reference.map(bits), "score {s}, moves {moves:?}, step {step}");
                    assert!(skipping <= every);
                    fewer += usize::from(skipping < every);
                    runs += 1;
                }
            }
        }
        assert!(fewer * 2 > runs, "{fewer} of {runs} descents scored fewer trials");
    }

    /// The spline's weights laid out a site at a time displace every
    /// boundary as each boundary's own weights do, to the bit, and so
    /// score a projection as the spline evaluated afresh at each does.
    #[test]
    fn boundaries_displaced_together_are_displaced_as_each_alone() {
        let func = FunctionMap::new(SymbolVersion::Qr(7));
        let (image, corners) = drawn(&func, |_| 1.0);
        let (w, h) = (func.width as f64, func.height as f64);
        let plane = Homography::rect_to_quad(w, h, corners).unwrap();
        let anchors: Vec<ControlPoint> = [[3.5, 3.5], [41.5, 3.5], [3.5, 41.5], [22.5, 22.5], [38.5, 38.5], [6.5, 22.5]]
            .iter()
            .enumerate()
            .map(|(k, &at)| {
                let [x, y] = plane.apply(at[0], at[1]);
                ControlPoint { at, px: [x + 0.3 * (k as f64).sin(), y - 0.4 * (k as f64).cos()] }
            })
            .collect();
        let projection = Projection::new(func.width, func.height, corners, &anchors).unwrap();
        let spline = projection.spline().expect("the anchors bend the map");
        for max_edges in [usize::MAX, 333] {
            let scorer = Scorer::new(&image, &Template::whole(&func), corners, None).with_spline(Some(spline)).with_max_edges(max_edges);
            let together = scorer.edge_weights.as_ref().unwrap().displacements(projection.residuals());
            assert_eq!(together.len(), scorer.edges.len());
            for (edge, displaced) in scorer.edges.iter().zip(&together) {
                let alone = projection.displacement_from_weights(&spline.weights(edge.at));
                assert_eq!(displaced.map(f64::to_bits), alone.map(f64::to_bits));
            }
            for nudge in [0.0, 0.6, -1.3] {
                let moved = corners.map(|[x, y]| [x + nudge, y - 0.5 * nudge]);
                let map = Projection::new(func.width, func.height, moved, &anchors).unwrap();
                let shared = scorer.contrast_projection(&map, 1.0).unwrap();
                assert_eq!(shared.to_bits(), scorer.contrast_with(&map, 1.0).unwrap().to_bits());
            }
        }
    }

    #[test]
    fn shifted_maps_offset_their_input() {
        let h = Homography::from_matrix([2.0, 0.0, 10.0, 0.0, 2.0, 20.0, 0.0, 0.0, 1.0]).unwrap();
        let shifted = Shifted { map: &h, dx: -1.0, dy: -1.0 };
        assert_eq!(shifted.apply(1.0, 1.0), h.apply(0.0, 0.0));
    }

    /// A symbol of the version `func` maps, eight pixels a module with
    /// three modules of paper round it, its fixed modules as the version
    /// has them and its data modules dark and light by turns, every
    /// pixel's luminance times `light` of how far across the image it
    /// lies; and the symbol's corners in it.
    fn drawn(func: &FunctionMap, light: impl Fn(f64) -> f32) -> (Luma, [[f64; 2]; 4]) {
        let (px, margin) = (8usize, 3usize);
        let (width, height) = ((func.width + 2 * margin) * px, (func.height + 2 * margin) * px);
        let mut data = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let (mx, my) = ((x / px) as i64 - margin as i64, (y / px) as i64 - margin as i64);
                let inside = mx >= 0 && my >= 0 && (mx as usize) < func.width && (my as usize) < func.height;
                let dark = inside && func.expected(mx as usize, my as usize).unwrap_or((mx + my) % 2 == 0);
                data.push(if dark { 0.25 } else { 0.9 } * light(x as f64 / width as f64));
            }
        }
        let (x0, y0) = ((margin * px) as f64, (margin * px) as f64);
        let (x1, y1) = (x0 + (func.width * px) as f64, y0 + (func.height * px) as f64);
        (Luma::new(width, height, data), [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    }

    /// (matching, known) of a symbol's fixed modules and quiet zone at
    /// `corners`, against one split and against the light.
    fn fit_of(image: &Luma, func: &FunctionMap, corners: [[f64; 2]; 4], threshold: Option<f32>) -> [(usize, usize); 2] {
        let (w, h) = (func.width as f64, func.height as f64);
        let map = Homography::rect_to_quad(w, h, corners).unwrap();
        let around = [[-1.0, -1.0], [w + 1.0, -1.0], [w + 1.0, h + 1.0], [-1.0, h + 1.0]].map(|[x, y]| map.apply(x, y));
        let scorer = Scorer::new(image, &Template::with_quiet_zone(func), around, threshold);
        let shifted = Shifted { map: &map, dx: -1.0, dy: -1.0 };
        [scorer.agreement_known(&shifted, false).unwrap(), scorer.agreement_shaded(&shifted, false).unwrap()]
    }

    #[test]
    fn a_template_in_failing_light_is_read_against_the_light_where_it_lies() {
        let func = FunctionMap::new(SymbolVersion::Qr(2));
        // In even light one split reads every module, and so does the
        // threshold that follows the light.
        let (even, corners) = drawn(&func, |_| 1.0);
        let [split, shaded] = fit_of(&even, &func, corners, None);
        assert_eq!(split.0, split.1, "{split:?}");
        assert_eq!(shaded, split);
        // Light that fails to a quarter from the left of the image to the
        // right leaves the ink on the left as light as the paper on the
        // right: no one split tells the two apart all the way across,
        // and the split of the lot calls the paper at the dim end dark.
        let (failing, corners) = drawn(&func, |across| 1.0 - 0.75 * across as f32);
        let [split, shaded] = fit_of(&failing, &func, corners, None);
        assert!(split.0 * 10 < split.1 * 9, "one split read {split:?}");
        assert_eq!(shaded.0, shaded.1, "{shaded:?}");
        // The same square a quarter turn out stands a finder where the
        // symbol has none and none where it has one, in any light.
        let turned = [corners[1], corners[2], corners[3], corners[0]];
        let [_, shaded] = fit_of(&failing, &func, turned, None);
        assert!(shaded.0 * 10 < shaded.1 * 9, "{shaded:?}");
        // A threshold given is the threshold used.
        let [split, shaded] = fit_of(&failing, &func, corners, Some(0.3));
        assert_eq!(shaded, split);
        assert!(split.0 < split.1);
        // A symbol with one finder pattern has a split in that corner
        // alone, and is read against it all over, as it was against the
        // split of the lot.
        let micro = FunctionMap::new(SymbolVersion::Micro(2));
        let (even, corners) = drawn(&micro, |_| 1.0);
        let [split, shaded] = fit_of(&even, &micro, corners, None);
        assert_eq!((shaded, split.0), (split, split.1));
    }
}
