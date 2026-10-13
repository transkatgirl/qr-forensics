//! The map from module coordinates to photo pixels.
//!
//! A flat symbol photographed with a lens is a plane seen in perspective, and
//! a homography through its four corners places every module. Paper that has
//! curled, a label on a bottle or a crumpled receipt is not a plane, and one
//! homography can be a module or more off in the middle of a large symbol.
//! Every fixed pattern the detector finds, or the user places, says where
//! one known module really lies, so the map here is the corner homography
//! plus a smooth correction that passes exactly through those anchors: a
//! thin-plate spline over the anchors' residuals, the surface of least
//! bending that fits them, as ZBar and zxing-cpp fit a transform per cell
//! between alignment patterns, but smooth, and indifferent to how the
//! anchors are arranged. With no anchors the correction is zero and the map
//! is the homography alone.
//!
//! The spline need not pass exactly through every anchor. An anchor is
//! read from a blurred photo to a few tenths of a module, and a surface
//! forced through every reading bends to each one's error; BoofCV fits
//! its grid-to-image transform by least squares over every pattern it
//! located, and smooths those errors away. A smoothing term on the
//! anchors does the same here: the surface may miss an anchor by a little
//! where that saves bending, and a lone anchor off its pattern pulls the
//! map less than its neighbours do together. The corners, which are
//! measured or traced, are held exactly.
//!
//! An anchor on the outline itself, a point of the symbol's edge traced
//! against the quiet zone or set by hand, holds the map across the edge
//! only. The trace says where the edge runs; it cannot say which point
//! of the edge it met, and the place along the edge it was filed under
//! is only where the map of the moment had that point. Held there both
//! ways, on a bottle, whose modules crowd together towards its sides,
//! every such point pins the columns beside the edge to the spacing a
//! plane gave them. So the part of its residual along the edge is left
//! free, and takes the value that bends the surface least
//! ([`Spline::relax`]): the map through it runs where the edge was
//! traced, and along the edge goes where the anchors inside the symbol
//! take it.

use super::homography::Homography;

/// Anything that maps module coordinates to photo pixels.
pub trait Project {
    /// The smooth global part of the map: a homography.
    fn base(&self, x: f64, y: f64) -> [f64; 2];

    /// The local correction on top of the base at a module point. It
    /// changes slowly, so callers may take it once per module or per
    /// boundary and apply it to every sample nearby.
    fn displacement(&self, _x: f64, _y: f64) -> [f64; 2] {
        [0.0, 0.0]
    }

    fn apply(&self, x: f64, y: f64) -> [f64; 2] {
        let [bx, by] = self.base(x, y);
        let [dx, dy] = self.displacement(x, y);
        [bx + dx, by + dy]
    }
}

impl Project for Homography {
    fn base(&self, x: f64, y: f64) -> [f64; 2] {
        Homography::apply(self, x, y)
    }
}

/// A point whose module coordinates and photo position are both known.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControlPoint {
    /// In module coordinates.
    pub at: [f64; 2],
    /// In photo pixels.
    pub px: [f64; 2],
}

/// Radial basis of the thin-plate spline: r² ln r, with U(0) = 0.
fn kernel(dx: f64, dy: f64) -> f64 {
    let r2 = dx * dx + dy * dy;
    if r2 <= 0.0 { 0.0 } else { 0.5 * r2 * r2.ln() }
}

/// How near a corner of the outline, in modules, an anchor yields to it.
pub const CORNER_CLEARANCE: f64 = 1.5;
/// Smallest spread of control points, in modules, that lets a spline be
/// solved: closer to a line than this and the affine part is not determined.
const MIN_SPREAD: f64 = 1.0;

/// How much the map may miss an anchor by, to save bending: the smoothing
/// term added to the kernel at each anchor, in the kernel's units of
/// square modules, where the kernel between two anchors two modules
/// apart is about 1.4 and three apart about 5. Zero makes the surface
/// pass exactly through every anchor. One is gentle: an anchor's own
/// reading still counts for about as much as a neighbour two modules
/// off, and a lone stray anchor is followed a little less far. Over the
/// corpus of photographs the reads are the same at zero, one, five and
/// twenty-five, so the term is set by reasoning rather than measurement.
/// The corners carry none.
pub const ANCHOR_SMOOTHING: f64 = 1.0;

/// A thin-plate spline over fixed sites in module space. The sites decide
/// the linear system; the values interpolated can change without solving
/// it again, so one spline serves every trial placement of the corners.
#[derive(Clone, Debug, PartialEq)]
pub struct Spline {
    sites: Vec<[f64; 2]>,
    /// Inverse of the bordered kernel matrix, (n + 3) square, row-major.
    inverse: Vec<f64>,
}

impl Spline {
    /// The spline that passes exactly through every site. `None` when the
    /// sites are too few, repeated or nearly collinear.
    pub fn new(sites: &[[f64; 2]]) -> Option<Self> {
        Self::smoothed(sites, &vec![0.0; sites.len()])
    }

    /// The smoothing spline: at each site the surface may miss the value
    /// there in proportion to that site's `smoothing`, zero holding it
    /// exactly. `None` as for [`Spline::new`].
    pub fn smoothed(sites: &[[f64; 2]], smoothing: &[f64]) -> Option<Self> {
        let n = sites.len();
        assert_eq!(smoothing.len(), n);
        if n < 3 || spread(sites) < MIN_SPREAD {
            return None;
        }
        let m = n + 3;
        let mut l = vec![0.0f64; m * m];
        for i in 0..n {
            for j in 0..n {
                l[i * m + j] = kernel(sites[i][0] - sites[j][0], sites[i][1] - sites[j][1]);
            }
            // The kernel is zero at a site itself; the smoothing term
            // takes its place on the diagonal.
            l[i * m + i] = smoothing[i].max(0.0);
            l[i * m + n] = 1.0;
            l[i * m + n + 1] = sites[i][0];
            l[i * m + n + 2] = sites[i][1];
            l[n * m + i] = 1.0;
            l[(n + 1) * m + i] = sites[i][0];
            l[(n + 2) * m + i] = sites[i][1];
        }
        let inverse = invert(&l, m)?;
        Some(Self { sites: sites.to_vec(), inverse })
    }

    pub fn len(&self) -> usize {
        self.sites.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }

    pub fn sites(&self) -> &[[f64; 2]] {
        &self.sites
    }

    /// The weight each site's value carries at `p`: the interpolant there
    /// is the weighted sum of the values. The weights sum to one and depend
    /// only on the sites.
    pub fn weights(&self, p: [f64; 2]) -> Vec<f64> {
        let n = self.sites.len();
        let m = n + 3;
        let mut k = Vec::with_capacity(m);
        for site in &self.sites {
            k.push(kernel(p[0] - site[0], p[1] - site[1]));
        }
        k.extend([1.0, p[0], p[1]]);
        // The matrix is symmetric, so its inverse is too, and the weight on
        // site i is row i of the inverse times k.
        (0..n).map(|i| (0..m).map(|j| self.inverse[i * m + j] * k[j]).sum()).collect()
    }

    /// Let the values at some sites slide: each of `free` names a site
    /// and a direction, a unit vector, along which that site's value is
    /// not known, and the part of each such value along its direction is
    /// set to what bends the surface least, the parts across them and
    /// every other value staying as they are. The bending of the surface
    /// through values `v` is `v`ᵀ·A·`v` in each coordinate, A the sites'
    /// own block of the inverse; it is least over the free parts where,
    /// at each freed site, the weight the solution gives that site has
    /// nothing along its direction, which is a small linear system in
    /// the free parts. Left as they were when that system cannot be
    /// solved: no site but the freed ones held.
    pub fn relax(&self, values: &mut [[f64; 2]], free: &[(usize, [f64; 2])]) {
        let n = self.sites.len();
        let m = n + 3;
        let k = free.len();
        if k == 0 {
            return;
        }
        debug_assert_eq!(values.len(), n);
        // Each value with its free part taken out.
        let mut held = values.to_vec();
        for &(i, t) in free {
            let along = held[i][0] * t[0] + held[i][1] * t[1];
            held[i] = [held[i][0] - along * t[0], held[i][1] - along * t[1]];
        }
        let mut a = vec![0.0f64; k * k];
        let mut b = vec![0.0f64; k];
        for (r, &(i, ti)) in free.iter().enumerate() {
            for (c, &(j, tj)) in free.iter().enumerate() {
                a[r * k + c] = self.inverse[i * m + j] * (ti[0] * tj[0] + ti[1] * tj[1]);
            }
            b[r] = -(0..n).map(|j| self.inverse[i * m + j] * (held[j][0] * ti[0] + held[j][1] * ti[1])).sum::<f64>();
        }
        let Some(inverse) = invert(&a, k) else { return };
        for (r, &(i, t)) in free.iter().enumerate() {
            let along: f64 = (0..k).map(|c| inverse[r * k + c] * b[c]).sum();
            values[i] = [held[i][0] + along * t[0], held[i][1] + along * t[1]];
        }
    }

    /// Interpolate two-dimensional `values`, one per site, at `p`.
    pub fn interpolate(&self, values: &[[f64; 2]], p: [f64; 2]) -> [f64; 2] {
        debug_assert_eq!(values.len(), self.sites.len());
        let weights = self.weights(p);
        let mut out = [0.0, 0.0];
        for (w, v) in weights.iter().zip(values) {
            out[0] += w * v[0];
            out[1] += w * v[1];
        }
        out
    }
}

/// Greatest distance of a point from the line through the two farthest
/// apart, in the points' own units.
pub fn spread(points: &[[f64; 2]]) -> f64 {
    let mut farthest = (0, 0, 0.0f64);
    for i in 0..points.len() {
        for j in i + 1..points.len() {
            let d = ((points[i][0] - points[j][0]).powi(2) + (points[i][1] - points[j][1]).powi(2)).sqrt();
            if d > farthest.2 {
                farthest = (i, j, d);
            }
        }
    }
    let (a, b, length) = (points[farthest.0], points[farthest.1], farthest.2);
    if length == 0.0 {
        return 0.0;
    }
    points
        .iter()
        .map(|c| ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs() / length)
        .fold(0.0, f64::max)
}

/// Gauss-Jordan inverse of an `m` × `m` row-major matrix with partial
/// pivoting; `None` when singular.
fn invert(a: &[f64], m: usize) -> Option<Vec<f64>> {
    let mut work = a.to_vec();
    let mut inv = vec![0.0f64; m * m];
    for i in 0..m {
        inv[i * m + i] = 1.0;
    }
    let scale = work.iter().fold(0.0f64, |s, v| s.max(v.abs()));
    if scale == 0.0 {
        return None;
    }
    for col in 0..m {
        let pivot = (col..m).max_by(|&a, &b| work[a * m + col].abs().total_cmp(&work[b * m + col].abs()))?;
        if work[pivot * m + col].abs() < 1e-12 * scale {
            return None;
        }
        if pivot != col {
            for j in 0..m {
                work.swap(col * m + j, pivot * m + j);
                inv.swap(col * m + j, pivot * m + j);
            }
        }
        let lead = work[col * m + col];
        for j in 0..m {
            work[col * m + j] /= lead;
            inv[col * m + j] /= lead;
        }
        for r in 0..m {
            if r == col {
                continue;
            }
            let factor = work[r * m + col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..m {
                work[r * m + j] -= factor * work[col * m + j];
                inv[r * m + j] -= factor * inv[col * m + j];
            }
        }
    }
    inv.iter().all(|v| v.is_finite()).then_some(inv)
}

/// Module coordinates to photo pixels for one placement of a symbol: the
/// homography through its four corners, corrected to pass through every
/// anchor.
#[derive(Clone, Debug, PartialEq)]
pub struct Projection {
    base: Homography,
    spline: Option<Spline>,
    /// Each anchor's photo position less where the base puts it.
    residuals: Vec<[f64; 2]>,
}

impl Projection {
    /// The homography alone.
    pub fn from_homography(base: Homography) -> Self {
        Self { base, spline: None, residuals: Vec::new() }
    }

    /// For a `width` × `height` symbol whose corners (top-left, top-right,
    /// bottom-right, bottom-left) lie at `corners` in the photo, corrected
    /// through `anchors`, with [`ANCHOR_SMOOTHING`] on the anchors. Anchors
    /// that fall on a corner are ignored, since the corners already fix
    /// those points. `None` when the corners are degenerate.
    pub fn new(width: usize, height: usize, corners: [[f64; 2]; 4], anchors: &[ControlPoint]) -> Option<Self> {
        Self::with_smoothing(width, height, corners, anchors, ANCHOR_SMOOTHING)
    }

    /// [`Projection::new`] with the smoothing on the anchors given: zero
    /// passes the map exactly through every anchor.
    pub fn with_smoothing(
        width: usize,
        height: usize,
        corners: [[f64; 2]; 4],
        anchors: &[ControlPoint],
        smoothing: f64,
    ) -> Option<Self> {
        let base = Homography::rect_to_quad(width as f64, height as f64, corners)?;
        if anchors.is_empty() {
            return Some(Self::from_homography(base));
        }
        // The corners are control points too, with no residual, so the
        // correction dies away towards them and the outline stays where
        // the corners say.
        let (w, h) = (width as f64, height as f64);
        let outline = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
        let mut all: Vec<ControlPoint> = outline.into_iter().zip(corners).map(|(at, px)| ControlPoint { at, px }).collect();
        // An anchor within a module or so of a corner yields to the
        // corner: two control points that close whose positions disagree
        // by a fraction of a module, as a blurred anchor and a traced
        // corner may, would ripple the spline across the whole symbol.
        let near_corner = |c: &ControlPoint| outline.iter().any(|o| ((o[0] - c.at[0]).powi(2) + (o[1] - c.at[1]).powi(2)).sqrt() < CORNER_CLEARANCE);
        all.extend(anchors.iter().filter(|c| !near_corner(c)).copied());
        // The corners are held exactly; the anchors may be missed a little.
        let smoothing: Vec<f64> = (0..all.len()).map(|i| if i < 4 { 0.0 } else { smoothing }).collect();
        // An anchor on the outline slides along the edge it lies on.
        let on = |v: f64, edge: f64| (v - edge).abs() < 1e-9;
        let slides: Vec<Option<[f64; 2]>> = all
            .iter()
            .enumerate()
            .map(|(i, c)| {
                if i < 4 {
                    None
                } else if on(c.at[1], 0.0) || on(c.at[1], h) {
                    Some([1.0, 0.0])
                } else if on(c.at[0], 0.0) || on(c.at[0], w) {
                    Some([0.0, 1.0])
                } else {
                    None
                }
            })
            .collect();
        Some(Self::over_sliding(base, &all, &smoothing, &slides))
    }

    /// A base map corrected through `anchors`, exactly. Anchors too few or
    /// too nearly in a line for a spline leave the base alone.
    pub fn over(base: Homography, anchors: &[ControlPoint]) -> Self {
        Self::over_smoothed(base, anchors, &vec![0.0; anchors.len()])
    }

    /// [`Projection::over`] with a smoothing term for each anchor.
    pub fn over_smoothed(base: Homography, anchors: &[ControlPoint], smoothing: &[f64]) -> Self {
        Self::over_sliding(base, anchors, smoothing, &vec![None; anchors.len()])
    }

    /// [`Projection::over_smoothed`] with some anchors free to slide: an
    /// anchor with a direction in `slides`, in module coordinates, holds
    /// the map across that direction only, as a point of the symbol's
    /// edge does across the edge ([`Spline::relax`]).
    pub fn over_sliding(base: Homography, anchors: &[ControlPoint], smoothing: &[f64], slides: &[Option<[f64; 2]>]) -> Self {
        let mut sites: Vec<[f64; 2]> = Vec::with_capacity(anchors.len());
        let mut kept: Vec<&ControlPoint> = Vec::with_capacity(anchors.len());
        let mut terms: Vec<f64> = Vec::with_capacity(anchors.len());
        let mut free: Vec<(usize, [f64; 2])> = Vec::new();
        for ((anchor, &term), slide) in anchors.iter().zip(smoothing).zip(slides) {
            // A repeated site makes the system singular; the first wins.
            if sites.iter().any(|s| (s[0] - anchor.at[0]).abs() < 1e-9 && (s[1] - anchor.at[1]).abs() < 1e-9) {
                continue;
            }
            // The direction in the photo, as the base has it there.
            if let Some([dx, dy]) = slide {
                let (here, there) = (base.apply(anchor.at[0], anchor.at[1]), base.apply(anchor.at[0] + dx, anchor.at[1] + dy));
                let length = ((there[0] - here[0]).powi(2) + (there[1] - here[1]).powi(2)).sqrt();
                if length > 0.0 && length.is_finite() {
                    free.push((sites.len(), [(there[0] - here[0]) / length, (there[1] - here[1]) / length]));
                }
            }
            sites.push(anchor.at);
            kept.push(anchor);
            terms.push(term);
        }
        let spline = Spline::smoothed(&sites, &terms);
        let residuals = match &spline {
            Some(spline) => {
                let mut residuals: Vec<[f64; 2]> = kept
                    .iter()
                    .map(|a| {
                        let [bx, by] = base.apply(a.at[0], a.at[1]);
                        [a.px[0] - bx, a.px[1] - by]
                    })
                    .collect();
                spline.relax(&mut residuals, &free);
                residuals
            }
            None => Vec::new(),
        };
        Self { base, spline, residuals }
    }

    pub fn homography(&self) -> &Homography {
        &self.base
    }

    /// True when anchors bend the map away from the homography.
    pub fn is_warped(&self) -> bool {
        self.spline.is_some()
    }

    pub fn spline(&self) -> Option<&Spline> {
        self.spline.as_ref()
    }

    /// The anchors' residuals, in the spline's site order.
    pub fn residuals(&self) -> &[[f64; 2]] {
        &self.residuals
    }

    /// The displacement at a point from precomputed spline weights, as
    /// [`Spline::weights`] gives them for that point.
    pub fn displacement_from_weights(&self, weights: &[f64]) -> [f64; 2] {
        let mut out = [0.0, 0.0];
        for (w, r) in weights.iter().zip(&self.residuals) {
            out[0] += w * r[0];
            out[1] += w * r[1];
        }
        out
    }

    /// The photo positions of every module corner of a `width` × `height`
    /// symbol, row-major, `(width + 1) * (height + 1)` of them.
    pub fn lattice(&self, width: usize, height: usize) -> Vec<[f64; 2]> {
        let mut out = Vec::with_capacity((width + 1) * (height + 1));
        for y in 0..=height {
            for x in 0..=width {
                out.push(self.apply(x as f64, y as f64));
            }
        }
        out
    }
}

impl Project for Projection {
    fn base(&self, x: f64, y: f64) -> [f64; 2] {
        self.base.apply(x, y)
    }

    fn displacement(&self, x: f64, y: f64) -> [f64; 2] {
        match &self.spline {
            Some(spline) => spline.interpolate(&self.residuals, [x, y]),
            None => [0.0, 0.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f64; 2], b: [f64; 2], tol: f64) -> bool {
        (a[0] - b[0]).abs() < tol && (a[1] - b[1]).abs() < tol
    }

    #[test]
    fn spline_passes_through_its_sites_and_weights_sum_to_one() {
        let sites = [[3.5, 3.5], [21.5, 3.5], [3.5, 21.5], [18.0, 18.0], [10.0, 12.0]];
        let spline = Spline::new(&sites).unwrap();
        let values = [[1.0, -2.0], [0.5, 0.25], [-1.0, 1.0], [2.0, 2.0], [0.0, 3.0]];
        for (site, value) in sites.iter().zip(&values) {
            assert!(close(spline.interpolate(&values, *site), *value, 1e-9), "{site:?}");
        }
        for p in [[0.0, 0.0], [12.3, 7.7], [25.0, 25.0]] {
            let sum: f64 = spline.weights(p).iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "{p:?}: {sum}");
        }
        // Affine values are reproduced exactly everywhere: no bending needed.
        let affine: Vec<[f64; 2]> = sites.iter().map(|s| [2.0 * s[0] - s[1] + 1.0, 0.5 * s[1] + 3.0]).collect();
        for p in [[0.0, 0.0], [12.3, 7.7], [30.0, -4.0]] {
            assert!(close(spline.interpolate(&affine, p), [2.0 * p[0] - p[1] + 1.0, 0.5 * p[1] + 3.0], 1e-7));
        }
    }

    #[test]
    fn degenerate_sites_are_refused() {
        assert!(Spline::new(&[[0.0, 0.0], [1.0, 1.0]]).is_none(), "too few");
        assert!(Spline::new(&[[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]]).is_none(), "collinear");
        assert!(Spline::new(&[[0.0, 0.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.5]]).is_none(), "nearly so");
        assert!(Spline::new(&[[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]]).is_some());
    }

    #[test]
    fn projection_without_anchors_is_the_homography() {
        let quad = [[120.5, 80.25], [530.75, 110.0], [560.0, 520.5], [90.0, 480.75]];
        let h = Homography::rect_to_quad(25.0, 25.0, quad).unwrap();
        let p = Projection::new(25, 25, quad, &[]).unwrap();
        assert!(!p.is_warped());
        for (x, y) in [(0.0, 0.0), (12.5, 3.0), (25.0, 25.0)] {
            assert_eq!(p.apply(x, y), h.apply(x, y));
        }
        // Anchors that agree with the plane change nothing measurable.
        let anchors: Vec<ControlPoint> = [[3.5, 3.5], [21.5, 3.5], [3.5, 21.5], [18.0, 18.0]]
            .iter()
            .map(|&at| ControlPoint { at, px: h.apply(at[0], at[1]) })
            .collect();
        let p = Projection::new(25, 25, quad, &anchors).unwrap();
        assert!(p.is_warped());
        for (x, y) in [(0.0, 0.0), (12.5, 3.0), (25.0, 25.0), (7.0, 19.0)] {
            assert!(close(p.apply(x, y), h.apply(x, y), 1e-6), "{x},{y}");
        }
    }

    #[test]
    fn anchors_pull_the_map_through_themselves() {
        let quad = [[100.0, 100.0], [600.0, 100.0], [600.0, 600.0], [100.0, 600.0]];
        let h = Homography::rect_to_quad(25.0, 25.0, quad).unwrap();
        // The middle of the symbol bulges 12 pixels down and right.
        let bulge = |at: [f64; 2]| {
            let [x, y] = h.apply(at[0], at[1]);
            let r = ((at[0] - 12.5).powi(2) + (at[1] - 12.5).powi(2)).sqrt() / 12.5;
            let s = 12.0 * (1.0 - r * r).max(0.0);
            [x + s, y + s]
        };
        let anchors: Vec<ControlPoint> = [[3.5, 3.5], [21.5, 3.5], [3.5, 21.5], [18.0, 18.0], [12.5, 12.5], [6.0, 12.0], [12.0, 20.0]]
            .iter()
            .map(|&at| ControlPoint { at, px: bulge(at) })
            .collect();
        let p = Projection::with_smoothing(25, 25, quad, &anchors, 0.0).unwrap();
        for anchor in &anchors {
            assert!(close(p.apply(anchor.at[0], anchor.at[1]), anchor.px, 1e-7));
        }
        // Corners stay put; between anchors the map follows the bulge
        // closely, and the displacement is what the weights give.
        for (i, [x, y]) in [[0.0, 0.0], [25.0, 0.0], [25.0, 25.0], [0.0, 25.0]].into_iter().enumerate() {
            assert!(close(p.apply(x, y), quad[i], 1e-7));
        }
        let at = [15.0, 10.0];
        assert!(close(p.apply(at[0], at[1]), bulge(at), 2.0), "{:?} vs {:?}", p.apply(at[0], at[1]), bulge(at));
        let weights = p.spline().unwrap().weights(at);
        assert!(close(p.displacement_from_weights(&weights), p.displacement(at[0], at[1]), 1e-9));
        let lattice = p.lattice(25, 25);
        assert_eq!(lattice.len(), 26 * 26);
        assert!(close(lattice[0], quad[0], 1e-7));
        assert!(close(lattice[26 * 25 + 25], quad[2], 1e-7));
    }

    #[test]
    fn smoothing_absorbs_a_stray_anchor_and_holds_the_corners() {
        let quad = [[100.0, 100.0], [600.0, 100.0], [600.0, 600.0], [100.0, 600.0]];
        let h = Homography::rect_to_quad(25.0, 25.0, quad).unwrap();
        // Anchors on the plane, and one read six pixels off its pattern.
        let sites = [[3.5, 3.5], [21.5, 3.5], [3.5, 21.5], [18.0, 18.0], [12.5, 12.5], [6.0, 12.0], [12.0, 20.0]];
        let mut anchors: Vec<ControlPoint> = sites.iter().map(|&at| ControlPoint { at, px: h.apply(at[0], at[1]) }).collect();
        anchors[4].px[0] += 6.0;
        let exact = Projection::with_smoothing(25, 25, quad, &anchors, 0.0).unwrap();
        let smooth = Projection::with_smoothing(25, 25, quad, &anchors, 20.0).unwrap();
        // Exact: the map goes through the stray reading. Smoothed: it
        // goes part of the way, and the plane beside it is bent less.
        let stray = anchors[4];
        assert!(close(exact.apply(stray.at[0], stray.at[1]), stray.px, 1e-7));
        let at_stray = smooth.apply(stray.at[0], stray.at[1]);
        let miss = at_stray[0] - h.apply(stray.at[0], stray.at[1])[0];
        assert!(miss > 0.5 && miss < 5.5, "{miss}");
        let beside = [10.0, 11.0];
        let bend = |p: &Projection| (p.apply(beside[0], beside[1])[0] - h.apply(beside[0], beside[1])[0]).abs();
        assert!(bend(&smooth) < bend(&exact), "{} vs {}", bend(&smooth), bend(&exact));
        // More smoothing, less of the stray.
        let smoother = Projection::with_smoothing(25, 25, quad, &anchors, 200.0).unwrap();
        let miss_more = smoother.apply(stray.at[0], stray.at[1])[0] - h.apply(stray.at[0], stray.at[1])[0];
        assert!(miss_more < miss, "{miss_more} vs {miss}");
        // The corners stay where they were put, and anchors that agree
        // with the plane are still met closely.
        for (i, [x, y]) in [[0.0, 0.0], [25.0, 0.0], [25.0, 25.0], [0.0, 25.0]].into_iter().enumerate() {
            assert!(close(smooth.apply(x, y), quad[i], 1e-7));
        }
        assert!(close(smooth.apply(21.5, 3.5), h.apply(21.5, 3.5), 1.0));
        // The default smoothing is what `new` applies.
        let by_default = Projection::new(25, 25, quad, &anchors).unwrap();
        let same = Projection::with_smoothing(25, 25, quad, &anchors, ANCHOR_SMOOTHING).unwrap();
        assert_eq!(by_default, same);
        // Smoothing weights still sum to one and reproduce affine values.
        let spline = Spline::smoothed(&sites, &[1.0; 7]).unwrap();
        for p in [[0.0, 0.0], [12.3, 7.7]] {
            let sum: f64 = spline.weights(p).iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "{p:?}: {sum}");
        }
        let affine: Vec<[f64; 2]> = sites.iter().map(|s| [2.0 * s[0] - s[1] + 1.0, 0.5 * s[1] + 3.0]).collect();
        let p = [8.0, 9.0];
        assert!(close(spline.interpolate(&affine, p), [2.0 * p[0] - p[1] + 1.0, 0.5 * p[1] + 3.0], 1e-7));
    }

    #[test]
    fn a_point_of_the_edge_holds_the_map_across_the_edge_only() {
        // A symbol whose modules crowd together towards its right side,
        // as on a bottle: the anchors inside it say so, and its top edge
        // is straight, twelve pixels higher than the corners have it.
        let quad = [[100.0, 100.0], [600.0, 100.0], [600.0, 600.0], [100.0, 600.0]];
        let h = Homography::rect_to_quad(25.0, 25.0, quad).unwrap();
        let crowded = |[x, y]: [f64; 2]| {
            let [px, py] = h.apply(x, y);
            [px + 40.0 * (std::f64::consts::PI * x / 25.0).sin(), py]
        };
        let inside: Vec<ControlPoint> =
            [[6.5, 6.5], [12.5, 6.5], [18.5, 6.5], [6.5, 18.5], [12.5, 18.5], [18.5, 18.5]].iter().map(|&at| ControlPoint { at, px: crowded(at) }).collect();
        // Points of the top edge as a trace through the plane files them:
        // each under the place along the edge the plane has it at.
        let edge = |x: f64, along: f64| ControlPoint { at: [x, 0.0], px: [h.apply(x, 0.0)[0] + along, 88.0] };
        let with = |along: f64| {
            let mut anchors = inside.clone();
            anchors.extend([6.0, 12.0, 19.0].map(|x| edge(x, along)));
            Projection::with_smoothing(25, 25, quad, &anchors, 0.0).unwrap()
        };
        let held = with(0.0);
        // The map runs along the traced line at each of them.
        for x in [6.0, 12.0, 19.0] {
            assert!((held.apply(x, 0.0)[1] - 88.0).abs() < 1e-6, "{x}: {:?}", held.apply(x, 0.0));
        }
        // Where along the line a point was filed changes nothing.
        let filed_elsewhere = with(25.0);
        for at in [[6.0, 0.0], [12.0, 0.0], [12.5, 3.0], [20.0, 1.0], [12.5, 12.5]] {
            assert!(close(held.apply(at[0], at[1]), filed_elsewhere.apply(at[0], at[1]), 1e-6), "{at:?}");
        }
        // Along the edge the map goes where the anchors inside take it:
        // the column over the middle of the left three is carried right
        // with them, further than the plane has it and further than the
        // same points pinned both ways leave it.
        let plane = h.apply(6.0, 0.0)[0];
        let pinned = {
            let mut anchors = inside.clone();
            anchors.extend([6.0, 12.0, 19.0].map(|x| ControlPoint { at: [x, 0.5], px: [h.apply(x, 0.0)[0], 98.0] }));
            Projection::with_smoothing(25, 25, quad, &anchors, 0.0).unwrap()
        };
        let slid = held.apply(6.0, 0.0)[0] - plane;
        assert!(slid > 10.0, "the edge point slid {slid} px");
        assert!(slid > pinned.apply(6.0, 0.5)[0] - plane + 5.0, "{slid} against {}", pinned.apply(6.0, 0.5)[0] - plane);
        // The freed parts are those that bend the surface least: any
        // other value along the edge at a freed site bends it more.
        let spline = held.spline().unwrap();
        let bending = |values: &[[f64; 2]]| -> f64 {
            // The bending of the surface through `values`: the values
            // times the sites' block of the inverse times the values.
            let n = spline.sites.len();
            let m = n + 3;
            let mut sum = 0.0;
            for i in 0..n {
                for j in 0..n {
                    sum += spline.inverse[i * m + j] * (values[i][0] * values[j][0] + values[i][1] * values[j][1]);
                }
            }
            sum.abs()
        };
        let relaxed = held.residuals().to_vec();
        let site = spline.sites().iter().position(|s| *s == [12.0, 0.0]).unwrap();
        for nudge in [-6.0, 6.0] {
            let mut other = relaxed.clone();
            other[site][0] += nudge;
            assert!(bending(&relaxed) < bending(&other), "{nudge}: {} against {}", bending(&relaxed), bending(&other));
        }
        // An anchor inside the symbol is held both ways as ever.
        for anchor in &inside {
            assert!(close(held.apply(anchor.at[0], anchor.at[1]), anchor.px, 1e-6));
        }
    }
}
