//! Plane-to-plane perspective transform fitted to four point pairs.

/// Reweightings of a fit after the plain least squares: enough for a
/// gross outlier's weight to fall to nothing.
const REWEIGHTINGS: usize = 3;
/// Huber's threshold, in residual scales: residuals beyond it count only
/// linearly. 1.345 keeps 95% efficiency on clean Gaussian data.
const HUBER_K: f64 = 1.345;
/// The residual scale from the median absolute residual, for Gaussian
/// residuals.
const MAD_TO_SIGMA: f64 = 1.4826;
/// Smallest residual scale, as a share of the points' mean distance from
/// their centroid: a fiftieth of a pixel on a symbol two hundred pixels
/// across. Exact data's residuals are rounding noise, well within it, so
/// every pair keeps its full weight and the plain fit stands.
const SCALE_FLOOR: f64 = 1e-4;

/// Maps (x, y) to ((ax + by + c) / (gx + hy + i), (dx + ey + f) / (gx + hy + i)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Homography {
    m: [f64; 9],
}

impl Homography {
    pub const IDENTITY: Self = Self { m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] };

    /// The transform taking each `from[i]` to `to[i]`. `None` when the points
    /// are degenerate (three in a line, or repeated).
    pub fn from_points(from: [[f64; 2]; 4], to: [[f64; 2]; 4]) -> Option<Self> {
        // Eight equations in the eight unknowns a..h, with i fixed at 1.
        let mut rows = [[0.0f64; 9]; 8];
        for k in 0..4 {
            let [x, y] = from[k];
            let [u, v] = to[k];
            rows[2 * k] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
            rows[2 * k + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
        }
        for col in 0..8 {
            let pivot = (col..8).max_by(|&a, &b| rows[a][col].abs().total_cmp(&rows[b][col].abs()))?;
            if rows[pivot][col].abs() < 1e-10 {
                return None;
            }
            rows.swap(col, pivot);
            let lead = rows[col];
            for (r, row) in rows.iter_mut().enumerate() {
                if r != col {
                    let factor = row[col] / lead[col];
                    for (value, &l) in row.iter_mut().zip(&lead).skip(col) {
                        *value -= factor * l;
                    }
                }
            }
        }
        let mut m = [1.0; 9];
        for (i, row) in rows.iter().enumerate() {
            m[i] = row[8] / row[i];
        }
        let h = Self { m };
        // Coincident or collinear targets can still solve, to a transform
        // that squashes the plane onto a line.
        let scale = m.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        (m.iter().all(|v| v.is_finite()) && h.determinant().abs() > 1e-12 * scale.powi(3)).then_some(h)
    }

    fn determinant(&self) -> f64 {
        let [a, b, c, d, e, f, g, h, i] = self.m;
        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    }

    /// The transform taking the rectangle [0, width] × [0, height] to a
    /// quadrilateral given as top-left, top-right, bottom-right, bottom-left.
    pub fn rect_to_quad(width: f64, height: f64, quad: [[f64; 2]; 4]) -> Option<Self> {
        Self::from_points([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], quad)
    }

    pub fn apply(&self, x: f64, y: f64) -> [f64; 2] {
        let m = &self.m;
        let w = m[6] * x + m[7] * y + m[8];
        [(m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w]
    }

    pub fn inverse(&self) -> Option<Self> {
        let [a, b, c, d, e, f, g, h, i] = self.m;
        let det = self.determinant();
        if det == 0.0 || !det.is_finite() {
            return None;
        }
        let adjugate = [
            e * i - f * h,
            c * h - b * i,
            b * f - c * e,
            f * g - d * i,
            a * i - c * g,
            c * d - a * f,
            d * h - e * g,
            b * g - a * h,
            a * e - b * d,
        ];
        Some(Self { m: adjugate.map(|v| v / det) })
    }

    /// A transform from its matrix, row-major. `None` when it is singular.
    pub fn from_matrix(m: [f64; 9]) -> Option<Self> {
        let h = Self { m };
        let scale = m.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        (scale > 0.0 && m.iter().all(|v| v.is_finite()) && h.determinant().abs() > 1e-12 * scale.powi(3)).then_some(h)
    }

    pub fn matrix(&self) -> [f64; 9] {
        self.m
    }

    /// The transform that applies `inner` first and then `self`.
    pub fn compose(&self, inner: &Self) -> Self {
        let (a, b) = (&self.m, &inner.m);
        let mut m = [0.0; 9];
        for row in 0..3 {
            for col in 0..3 {
                m[3 * row + col] = (0..3).map(|k| a[3 * row + k] * b[3 * k + col]).sum();
            }
        }
        Self { m }
    }

    /// Least-squares fit of a transform of the given kind taking each
    /// `from` to its `to`. Needs at least two, three or four pairs for a
    /// similarity, an affine map or a projective one; `None` when there are
    /// too few or they are degenerate.
    ///
    /// A pair far off the fit of the rest, as a pattern read in the wrong
    /// place gives, would drag a plain least-squares fit towards itself
    /// and spoil the pairs around it. After the plain fit, pairs whose
    /// residual exceeds Huber's threshold of the residual scale are
    /// reweighted down to count only linearly, and the fit repeated, a few
    /// times over; a gross outlier is then all but ignored, while clean
    /// data keeps every pair at full weight and the plain fit unchanged.
    pub fn fit(model: Model, pairs: &[([f64; 2], [f64; 2])]) -> Option<Self> {
        Self::fit_reweighted(model, pairs, REWEIGHTINGS)
    }

    /// [`Homography::fit`] with the number of reweightings given: none is
    /// plain least squares.
    fn fit_reweighted(model: Model, pairs: &[([f64; 2], [f64; 2])], reweightings: usize) -> Option<Self> {
        let needed = match model {
            Model::Similarity => 2,
            Model::Affine => 3,
            Model::Projective => 4,
        };
        if pairs.len() < needed {
            return None;
        }
        // Both sides are centred and scaled first, so that photo pixels in
        // the thousands and module coordinates in the tens condition the
        // normal equations alike.
        let from_norm = normalizer(pairs.iter().map(|p| p.0))?;
        let to_norm = normalizer(pairs.iter().map(|p| p.1))?;
        let normalized: Vec<([f64; 2], [f64; 2])> = pairs
            .iter()
            .map(|&(f, t)| (from_norm.apply(f[0], f[1]), to_norm.apply(t[0], t[1])))
            .collect();
        let mut weights = vec![1.0; pairs.len()];
        let mut m = solve_weighted(model, &normalized, &weights)?;
        for _ in 0..reweightings {
            // Residuals in the normalised target's units, where the mean
            // distance from the centroid is √2.
            let h = Self { m };
            let residuals: Vec<f64> = normalized
                .iter()
                .map(|&([x, y], [u, v])| {
                    let [px, py] = h.apply(x, y);
                    ((px - u).powi(2) + (py - v).powi(2)).sqrt()
                })
                .collect();
            let mut sorted = residuals.clone();
            sorted.sort_by(f64::total_cmp);
            let median = (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0;
            let threshold = HUBER_K * (MAD_TO_SIGMA * median).max(SCALE_FLOOR * std::f64::consts::SQRT_2);
            let reweighted: Vec<f64> = residuals.iter().map(|&r| if r <= threshold { 1.0 } else { threshold / r }).collect();
            if reweighted == weights {
                break;
            }
            weights = reweighted;
            m = solve_weighted(model, &normalized, &weights)?;
        }
        Self::from_matrix(to_norm.inverse()?.compose(&Self { m }).compose(&from_norm).m)
    }
}

/// The matrix of the transform of the given kind fitted to the pairs by
/// least squares, each pair's equations weighted by `weights`: the rows
/// are scaled by the weight's square root, so that the squared residuals
/// are weighted by it.
fn solve_weighted(model: Model, pairs: &[([f64; 2], [f64; 2])], weights: &[f64]) -> Option<[f64; 9]> {
    let weighted = pairs.iter().zip(weights).map(|(&(f, t), &w)| (f, t, w.sqrt()));
    let scaled = |row: [f64; 8], value: f64, s: f64| (row.map(|c| c * s), value * s);
    Some(match model {
        Model::Similarity => {
            let [a, b, c, d] = least_squares(weighted.flat_map(|([x, y], [u, v], s)| {
                [([x * s, -y * s, s, 0.0], u * s), ([y * s, x * s, 0.0, s], v * s)]
            }))?;
            [a, -b, c, b, a, d, 0.0, 0.0, 1.0]
        }
        Model::Affine => {
            let [a, b, c, d, e, f] = least_squares(weighted.flat_map(|([x, y], [u, v], s)| {
                [([x * s, y * s, s, 0.0, 0.0, 0.0], u * s), ([0.0, 0.0, 0.0, x * s, y * s, s], v * s)]
            }))?;
            [a, b, c, d, e, f, 0.0, 0.0, 1.0]
        }
        Model::Projective => {
            let [a, b, c, d, e, f, g, h] = least_squares(weighted.flat_map(|([x, y], [u, v], s)| {
                [
                    scaled([x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u], u, s),
                    scaled([0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v], v, s),
                ]
            }))?;
            [a, b, c, d, e, f, g, h, 1.0]
        }
    })
}

/// The kind of plane transform [`Homography::fit`] looks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    /// Rotation, one scale and a translation: four degrees of freedom.
    Similarity,
    /// Any linear map and a translation: six.
    Affine,
    /// Full perspective: eight.
    Projective,
}

/// The similarity taking the points' centroid to the origin and their mean
/// distance from it to √2.
fn normalizer(points: impl Iterator<Item = [f64; 2]> + Clone) -> Option<Homography> {
    let count = points.clone().count() as f64;
    if count == 0.0 {
        return None;
    }
    let (cx, cy) = points.clone().fold((0.0, 0.0), |(sx, sy), [x, y]| (sx + x / count, sy + y / count));
    let mean = points.map(|[x, y]| ((x - cx).powi(2) + (y - cy).powi(2)).sqrt()).sum::<f64>() / count;
    if !(mean > 0.0 && mean.is_finite()) {
        return None;
    }
    let s = std::f64::consts::SQRT_2 / mean;
    Some(Homography { m: [s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0] })
}

/// Least-squares solution of the rows `(coefficients, value)`, through the
/// normal equations. `None` when the system is rank deficient.
fn least_squares<const N: usize>(rows: impl Iterator<Item = ([f64; N], f64)>) -> Option<[f64; N]> {
    let mut ata = [[0.0f64; N]; N];
    let mut atb = [0.0f64; N];
    for (a, b) in rows {
        for i in 0..N {
            atb[i] += a[i] * b;
            for j in 0..N {
                ata[i][j] += a[i] * a[j];
            }
        }
    }
    let scale = (0..N).fold(0.0f64, |m, i| m.max(ata[i][i].abs()));
    if scale == 0.0 {
        return None;
    }
    for col in 0..N {
        let pivot = (col..N).max_by(|&a, &b| ata[a][col].abs().total_cmp(&ata[b][col].abs()))?;
        if ata[pivot][col].abs() <= 1e-12 * scale {
            return None;
        }
        ata.swap(col, pivot);
        atb.swap(col, pivot);
        let (lead, lead_b) = (ata[col], atb[col]);
        for r in 0..N {
            if r != col {
                let factor = ata[r][col] / lead[col];
                if factor != 0.0 {
                    for c in col..N {
                        ata[r][c] -= factor * lead[c];
                    }
                    atb[r] -= factor * lead_b;
                }
            }
        }
    }
    let x: [f64; N] = std::array::from_fn(|i| atb[i] / ata[i][i]);
    x.iter().all(|v| v.is_finite()).then_some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f64; 2], b: [f64; 2]) -> bool {
        (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6
    }

    #[test]
    fn corners_map_exactly() {
        let quad = [[120.5, 80.25], [530.75, 110.0], [560.0, 520.5], [90.0, 480.75]];
        let h = Homography::rect_to_quad(21.0, 21.0, quad).unwrap();
        assert!(close(h.apply(0.0, 0.0), quad[0]));
        assert!(close(h.apply(21.0, 0.0), quad[1]));
        assert!(close(h.apply(21.0, 21.0), quad[2]));
        assert!(close(h.apply(0.0, 21.0), quad[3]));
        // A perspective map keeps the intersection of the diagonals.
        let centre = h.apply(10.5, 10.5);
        let (p, q, r, s) = (quad[0], quad[2], quad[1], quad[3]);
        let cross = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        assert!(cross(p, q, centre).abs() < 1e-6);
        assert!(cross(r, s, centre).abs() < 1e-6);
    }

    #[test]
    fn inverse_round_trips() {
        let quad = [[10.0, 400.0], [15.0, 20.0], [700.0, 60.0], [650.0, 380.0]];
        let h = Homography::rect_to_quad(139.0, 17.0, quad).unwrap();
        let inv = h.inverse().unwrap();
        for (x, y) in [(0.0, 0.0), (3.5, 16.0), (70.0, 8.5), (139.0, 17.0), (-4.0, 20.0)] {
            let [u, v] = h.apply(x, y);
            assert!(close(inv.apply(u, v), [x, y]), "{x},{y}");
        }
    }

    #[test]
    fn mirrored_quads_work() {
        // Corners given as seen from the back of the paper.
        let quad = [[500.0, 100.0], [100.0, 100.0], [100.0, 500.0], [500.0, 500.0]];
        let h = Homography::rect_to_quad(10.0, 10.0, quad).unwrap();
        assert!(close(h.apply(2.5, 5.0), [400.0, 300.0]));
        assert_eq!(Homography::IDENTITY.apply(3.0, 4.0), [3.0, 4.0]);
    }

    #[test]
    fn degenerate_points_are_rejected() {
        let line = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        assert!(Homography::rect_to_quad(1.0, 1.0, line).is_none());
        let repeated = [[5.0, 5.0], [5.0, 5.0], [9.0, 9.0], [0.0, 9.0]];
        assert!(Homography::rect_to_quad(1.0, 1.0, repeated).is_none());
    }

    #[test]
    fn compose_applies_the_inner_transform_first() {
        let quad = [[120.5, 80.25], [530.75, 110.0], [560.0, 520.5], [90.0, 480.75]];
        let outer = Homography::rect_to_quad(21.0, 21.0, quad).unwrap();
        let inner = Homography::from_matrix([2.0, 0.0, 1.0, 0.0, 2.0, -1.0, 0.0, 0.0, 1.0]).unwrap();
        let both = outer.compose(&inner);
        let [x, y] = inner.apply(3.0, 4.0);
        assert!(close(both.apply(3.0, 4.0), outer.apply(x, y)));
        assert!(Homography::from_matrix([1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0, 0.0, 1.0]).is_none());
    }

    #[test]
    fn fits_recover_each_kind_of_transform() {
        // Points of a 25-module symbol, as three finder patches and two
        // alignment centres would give them.
        let mut from = Vec::new();
        for (x0, y0) in [(-1.0, -1.0), (17.0, -1.0), (-1.0, 17.0)] {
            from.extend([[x0, y0], [x0 + 9.0, y0], [x0 + 9.0, y0 + 9.0], [x0, y0 + 9.0]]);
        }
        from.extend([[18.0, 18.0], [12.0, 20.0]]);
        let check = |truth: &Homography, model: Model| {
            let pairs: Vec<_> = from.iter().map(|&p| (p, truth.apply(p[0], p[1]))).collect();
            let fitted = Homography::fit(model, &pairs).unwrap();
            for (x, y) in [(0.0, 0.0), (25.0, 25.0), (25.0, 0.0), (0.0, 25.0), (13.0, 7.0)] {
                let want = truth.apply(x, y);
                let got = fitted.apply(x, y);
                assert!(
                    (want[0] - got[0]).abs() < 1e-6 && (want[1] - got[1]).abs() < 1e-6,
                    "{model:?} at ({x}, {y}): {want:?} vs {got:?}"
                );
            }
        };
        let (c, s) = (0.3f64.cos() * 7.5, 0.3f64.sin() * 7.5);
        let similarity = Homography::from_matrix([c, -s, 300.0, s, c, 120.0, 0.0, 0.0, 1.0]).unwrap();
        check(&similarity, Model::Similarity);
        check(&similarity, Model::Affine);
        check(&similarity, Model::Projective);
        let affine = Homography::from_matrix([8.0, 1.5, 200.0, -0.5, 6.0, 90.0, 0.0, 0.0, 1.0]).unwrap();
        check(&affine, Model::Affine);
        check(&affine, Model::Projective);
        let quad = [[150.5, 70.25], [655.0, 120.5], [610.75, 560.0], [95.25, 500.5]];
        let projective = Homography::rect_to_quad(25.0, 25.0, quad).unwrap();
        check(&projective, Model::Projective);

        // Too few or collinear points give nothing.
        let pairs: Vec<_> = from.iter().take(3).map(|&p| (p, projective.apply(p[0], p[1]))).collect();
        assert!(Homography::fit(Model::Projective, &pairs).is_none());
        let line: Vec<_> = (0..6).map(|i| ([i as f64, 2.0 * i as f64], [3.0 * i as f64, i as f64])).collect();
        assert!(Homography::fit(Model::Affine, &line).is_none());
        assert!(Homography::fit(Model::Similarity, &line[..1]).is_none());
    }

    /// Twelve points of a 25-module symbol: three finder corners and
    /// three more fixed points, as the anchors give them.
    fn twelve() -> Vec<[f64; 2]> {
        let mut from = Vec::new();
        for (x0, y0) in [(-1.0, -1.0), (17.0, -1.0), (-1.0, 17.0)] {
            from.extend([[x0, y0], [x0 + 9.0, y0], [x0, y0 + 9.0]]);
        }
        from.extend([[18.0, 18.0], [12.0, 20.0], [6.5, 6.5]]);
        from
    }

    #[test]
    fn clean_data_keeps_the_plain_fit() {
        let quad = [[150.5, 70.25], [655.0, 120.5], [610.75, 560.0], [95.25, 500.5]];
        let truth = Homography::rect_to_quad(25.0, 25.0, quad).unwrap();
        let pairs: Vec<_> = twelve().iter().map(|&p| (p, truth.apply(p[0], p[1]))).collect();
        for model in [Model::Projective, Model::Affine, Model::Similarity] {
            let plain = Homography::fit_reweighted(model, &pairs, 0).unwrap();
            let robust = Homography::fit(model, &pairs).unwrap();
            for (a, b) in plain.matrix().iter().zip(robust.matrix()) {
                assert!((a - b).abs() <= 1e-9 * a.abs().max(1.0), "{model:?}: {plain:?} vs {robust:?}");
            }
        }
    }

    #[test]
    fn reweighting_shrugs_off_an_outlier() {
        let quad = [[150.5, 70.25], [655.0, 120.5], [610.75, 560.0], [95.25, 500.5]];
        let projective = Homography::rect_to_quad(25.0, 25.0, quad).unwrap();
        let affine = Homography::from_matrix([8.0, 1.5, 200.0, -0.5, 6.0, 90.0, 0.0, 0.0, 1.0]).unwrap();
        for (model, truth) in [(Model::Projective, projective), (Model::Affine, affine)] {
            let inliers: Vec<_> = twelve().iter().map(|&p| (p, truth.apply(p[0], p[1]))).collect();
            // One more pair, twenty pixels from where the truth puts it.
            let mut pairs = inliers.clone();
            let [u, v] = truth.apply(12.0, 12.0);
            pairs.push(([12.0, 12.0], [u + 12.0, v - 16.0]));
            let worst = |fitted: &Homography| {
                inliers
                    .iter()
                    .map(|&([x, y], [u, v])| {
                        let [px, py] = fitted.apply(x, y);
                        ((px - u).powi(2) + (py - v).powi(2)).sqrt()
                    })
                    .fold(0.0f64, f64::max)
            };
            let plain = worst(&Homography::fit_reweighted(model, &pairs, 0).unwrap());
            let robust = worst(&Homography::fit(model, &pairs).unwrap());
            assert!(robust < 0.05, "{model:?}: {robust} px");
            assert!(plain > 0.5, "{model:?}: plain {plain} px");
        }
    }
}
