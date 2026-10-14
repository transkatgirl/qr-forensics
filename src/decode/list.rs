//! Algebraic soft-decision list decoding of one Reed-Solomon block:
//! Koetter and Vardy's multiplicities from how far each module's reading
//! is to be believed, Guruswami and Sudan's interpolation through the
//! codewords so weighed, and Roth and Ruckenstein's factorisation, which
//! reads every codeword of the code off the curve interpolated.
//!
//! The decoders that stop at half the code's distance, every reader's
//! and [`super::rs::rs_decode`] here, correct `⌊r/2⌋` wrong codewords of
//! a block with `r` of error correction and no more: past that the block
//! as read may lie as near another codeword as the right one, and a
//! decoder that returns one codeword cannot say which. A list decoder
//! returns every codeword within reach, and reaches further: Guruswami
//! and Sudan reach `n - √(n(k-1))` wrong codewords of a block of `n`
//! with `k` of data, which is eleven of the twenty-six codewords of a
//! version 1-H block where half the distance is eight, and nineteen of
//! the forty-five of a version 40-H block where it is fifteen. The
//! trials of erasures ([`super::rs::rs_decode_trying`]) reach one or two
//! past the half, which is as far as small sets go; this is as far as
//! the algebra goes, and on a block of low rate that is some way.
//!
//! The curve is a polynomial `Q(x, y)` that passes through each point
//! `(x_i, y_i)` the block gives, codeword `i` read as `y_i` at the place
//! `x_i` the code evaluates it, with a multiplicity `m_i`: it is tangent
//! there to order `m_i`. The codewords of the code are the graphs of
//! polynomials `y = H(x)` of degree below `k`, and any that agrees with
//! enough of the points, weighed by their multiplicities, divides `Q` as
//! `y - H(x)`; the factors are read off. Koetter and Vardy choose the
//! multiplicities from the likelihood of each reading, and give a point
//! to every value a codeword may well hold, not only the one it read as:
//! a faint module's other colour is a second point at the same place,
//! and the curve is bent toward whichever the rest of the block bears
//! out. That is soft decision decoding, which the chase
//! ([`super::rs::rs_decode_chasing`]) does by trying every pattern of the
//! ten faintest bits, and this does for every bit at once.
//!
//! What such a solution confirms is counted as any other's
//! ([`super::rs::list_solution_bits`]): how rarely a block of noise has
//! any codeword as near. A solution past half the distance has no slack,
//! and stands on how few blocks of noise show a codeword within so many
//! errors, which for a block of much redundancy is a great deal and for
//! one of little is nothing: a version 2-L block has a codeword within
//! six of noise once in six hundred blocks, and no solution so far out
//! is offered there. Where the list holds two codewords the block lies
//! as near to, both are returned ([`super::rs::nearest`]) for the
//! message to tell apart, as the other solvers that try many ways do.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::gf256::{self, div, exp, mul, poly_eval};
use super::rs::{CwRead, DOUBT_CONFIRMING_BITS, RsSolution, doubted_agreement, list_solution_bits, nearest, preferring_the_standard, settle, soft_distance};
use crate::symbol::grid::CERTAIN;
use crate::work;

/// Constraints the interpolation is given by default, counting a point
/// of multiplicity `m` as `m(m+1)/2`: what the curve costs, since every
/// constraint is a pass over every coefficient of every polynomial in
/// hand, and how far it reaches, since the multiplicities grow with the
/// square root of it. Two thousand puts a multiplicity of nine on every
/// codeword of a 40-H block read whole, which reaches nineteen wrong of
/// its forty-five, and of twelve on a 1-H block's twenty-six, which
/// reaches eleven; a block of a hundred and fifty codewords gets five,
/// which on a block of high rate reaches no further than half the
/// distance does, and nothing would. A tenth of a second on the largest
/// blocks, and a few milliseconds on the smallest.
pub const LIST_COST: usize = 2048;
/// Fewest constraints a decode that is one of many is given
/// ([`super::pipeline::DecodeOptions::among`]).
pub const LIST_COST_MIN: usize = 512;
/// Most constraints the interpolation may be set to by hand
/// ([`super::pipeline::DecodeOptions::list_cost`]): seconds on the
/// largest blocks.
pub const LIST_COST_MAX: usize = 1 << 15;

/// Values a codeword is given points for, at the most: the one it read
/// as and the likeliest others, made by flipping its faintest bits and
/// filling its unknown ones.
const VALUES_PER_CODEWORD: usize = 16;
/// Faintest known bits of a codeword whose flips make its other values.
const FLIP_BITS: usize = 4;
/// Most unknown bits a codeword may have and still be given points:
/// with more, no filling of them is likely enough to be worth one, and
/// the codeword is an erasure, which constrains the curve nowhere.
const UNKNOWN_BITS_MAX: usize = 3;
/// Largest degree in `y` the curve is given, which is how many codewords
/// the list may hold and how many polynomials the interpolation carries.
const LIST_SIZE_MAX: usize = 32;

/// How likely a module read at `reliability` is to have been misread:
/// the scale of [`crate::imaging::sample::reliability_bits`] read back, a
/// coin's toss at nought and a bit of likelihood more for every eight,
/// and never for a module painted by hand ([`CERTAIN`]).
fn misread(reliability: u8) -> f64 {
    if reliability == CERTAIN { 0.0 } else { (-(f64::from(reliability) / 8.0 + 1.0)).exp2() }
}

/// The place the code evaluates each codeword of a block of `n` with `r`
/// EC codewords at, and what it multiplies the value there by: codeword
/// `i`, which carries `x^(n-1-i)` of the block polynomial, is
/// `v_i · H(x_i)` for the data polynomial `H` of degree below `n - r`.
///
/// The block polynomial has the roots `α^0..α^(r-1)`, so by the finite
/// field's Fourier transform the codeword at position `p` is
/// `u^r G(u)` at `u = α^(-p)`, for a `G` of degree below `255 - r`; and
/// the block being shortened to `n` positions, the codewords at the
/// positions `n..255` are nought, which makes `G` a multiple of the
/// polynomial with a root at each of those positions' `u`. What is left
/// is `H`, of degree below `n - r`, and `v_i` is the rest: `u^r` times
/// that polynomial at `u`.
pub fn grs_points(n: usize, r: usize) -> Vec<(u8, u8)> {
    (0..n)
        .map(|i| {
            let position = n - 1 - i;
            let x = exp(255 - position);
            let mut v = gf256::pow(x, r);
            for shortened in n..255 {
                v = mul(v, x ^ exp(255 - shortened));
            }
            (x, v)
        })
        .collect()
}

/// A value a codeword may hold, and how likely it is to.
struct Point {
    codeword: usize,
    value: u8,
    likelihood: f64,
}

/// The values each codeword may hold, with the likelihood of each: the
/// value read, and those made by flipping its [`FLIP_BITS`] faintest
/// bits every way and filling its unknown bits every way, the
/// [`VALUES_PER_CODEWORD`] likeliest. Pinned bits and bits painted by
/// hand are never flipped, and a codeword with more than
/// [`UNKNOWN_BITS_MAX`] unknown bits is given nothing.
fn points(block: &[CwRead], bit_reliability: &[[u8; 8]]) -> Vec<Point> {
    let mut out = Vec::new();
    for (i, cw) in block.iter().enumerate() {
        let unknown = !cw.known;
        if unknown.count_ones() as usize > UNKNOWN_BITS_MAX {
            continue;
        }
        let misread_of = |b: usize| {
            if cw.known >> b & 1 != 0 && cw.pinned >> b & 1 == 0 { misread(bit_reliability[i][b]) } else { 0.0 }
        };
        let mut faint: Vec<(u8, usize)> = (0..8usize)
            .filter(|&b| cw.known >> b & 1 != 0 && cw.pinned >> b & 1 == 0 && bit_reliability[i][b] < CERTAIN)
            .map(|b| (bit_reliability[i][b], b))
            .collect();
        faint.sort();
        faint.truncate(FLIP_BITS);
        let mut as_read = 0.5f64.powi(unknown.count_ones() as i32);
        for b in 0..8 {
            if cw.known >> b & 1 != 0 {
                as_read *= 1.0 - misread_of(b);
            }
        }
        let base = cw.value & cw.known;
        let mut values: Vec<(f64, u8)> = Vec::new();
        for flips in 0..1u32 << faint.len() {
            let mut flipped = 0u8;
            let mut odds = 1.0;
            for (k, &(_, b)) in faint.iter().enumerate() {
                if flips >> k & 1 != 0 {
                    flipped |= 1 << b;
                    let q = misread_of(b);
                    odds *= q / (1.0 - q);
                }
            }
            let likelihood = as_read * odds;
            if likelihood <= 0.0 {
                continue;
            }
            // Every filling of the unknown bits, as subsets of them.
            let mut fill = 0u8;
            loop {
                values.push((likelihood, base ^ flipped | fill));
                fill = fill.wrapping_sub(unknown) & unknown;
                if fill == 0 {
                    break;
                }
            }
        }
        values.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        values.truncate(VALUES_PER_CODEWORD);
        out.extend(values.into_iter().map(|(likelihood, value)| Point { codeword: i, value, likelihood }));
    }
    out
}

/// A point's claim on the next unit of multiplicity.
#[derive(PartialEq)]
struct Claim {
    key: f64,
    point: usize,
}

impl Eq for Claim {}

impl PartialOrd for Claim {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Claim {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.total_cmp(&other.key).then(other.point.cmp(&self.point))
    }
}

/// Koetter and Vardy's multiplicities for the points, within `cost`
/// constraints: a unit at a time to the point whose likelihood divided
/// by one more than its multiplicity is the greatest, which is their
/// greedy algorithm, until the next unit would not fit. A point of
/// likelihood one, a codeword painted by hand or pinned, draws the most,
/// and a faint bit's other colour draws its share.
fn multiplicities(points: &[Point], cost: usize) -> Vec<usize> {
    let mut m = vec![0usize; points.len()];
    let mut claims: BinaryHeap<Claim> = points
        .iter()
        .enumerate()
        .filter(|(_, p)| p.likelihood > 0.0)
        .map(|(i, p)| Claim { key: p.likelihood, point: i })
        .collect();
    let mut spent = 0;
    while let Some(Claim { point, .. }) = claims.pop() {
        let unit = m[point] + 1;
        if spent + unit > cost {
            break;
        }
        m[point] += 1;
        spent += unit;
        claims.push(Claim { key: points[point].likelihood / (m[point] + 1) as f64, point });
    }
    m
}

/// A polynomial in `x` and `y`: `coef[t]` is the coefficient of `y^t`, a
/// polynomial in `x` with `coef[t][d]` the coefficient of `x^d`, trimmed
/// of trailing zeros.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Curve {
    coef: Vec<Vec<u8>>,
}

fn trim(poly: &mut Vec<u8>) {
    while poly.last() == Some(&0) {
        poly.pop();
    }
}

/// `base^0 .. base^(count-1)`.
fn powers(base: u8, count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(count);
    let mut p = 1u8;
    for _ in 0..count {
        out.push(p);
        p = mul(p, base);
    }
    out
}

impl Curve {
    /// `y^t`.
    fn monomial(t: usize) -> Self {
        let mut coef = vec![Vec::new(); t];
        coef.push(vec![1]);
        Self { coef }
    }

    /// The leading monomial under the `(1, w)`-weighted degree, as
    /// (weighted degree, power of `y`): the greatest weighted degree,
    /// and of those the greatest power of `y`. `None` for nought.
    fn leading(&self, w: usize) -> Option<(usize, usize)> {
        self.coef.iter().enumerate().filter(|(_, p)| !p.is_empty()).map(|(t, p)| (p.len() - 1 + t * w, t)).max()
    }

    /// The Hasse derivative of order `(a, b)` at `(x, y)`: the coefficient
    /// of `(X - x)^a (Y - y)^b` in the curve written about the point.
    /// `xpow` and `ypow` are the powers of `x` and `y`. In characteristic
    /// two a binomial coefficient is odd exactly when the lower index's
    /// bits lie within the upper's.
    fn hasse(&self, a: usize, b: usize, xpow: &[u8], ypow: &[u8]) -> u8 {
        let mut total = 0u8;
        for (t, poly) in self.coef.iter().enumerate().skip(b) {
            if t & b != b {
                continue;
            }
            let mut inner = 0u8;
            for (d, &c) in poly.iter().enumerate().skip(a) {
                if c != 0 && d & a == a {
                    inner ^= mul(c, xpow[d - a]);
                }
            }
            if inner != 0 {
                total ^= mul(inner, ypow[t - b]);
            }
        }
        total
    }

    /// Largest power of `x` among the coefficients.
    fn x_degree(&self) -> usize {
        self.coef.iter().map(|p| p.len()).max().unwrap_or(1)
    }

    /// `self += c · other`.
    fn add_scaled(&mut self, other: &Curve, c: u8) {
        if self.coef.len() < other.coef.len() {
            self.coef.resize(other.coef.len(), Vec::new());
        }
        for (mine, theirs) in self.coef.iter_mut().zip(&other.coef) {
            if mine.len() < theirs.len() {
                mine.resize(theirs.len(), 0);
            }
            for (m, &t) in mine.iter_mut().zip(theirs) {
                *m ^= mul(c, t);
            }
            trim(mine);
        }
    }

    /// `self *= (X - x)`, in place: from the top, each coefficient
    /// moves up a degree and leaves `x` times itself behind.
    fn times_x_minus(&mut self, x: u8) {
        for poly in &mut self.coef {
            if poly.is_empty() {
                continue;
            }
            poly.push(0);
            for d in (0..poly.len() - 1).rev() {
                let c = poly[d];
                poly[d + 1] ^= c;
                poly[d] = mul(c, x);
            }
            trim(poly);
        }
    }

    /// The curve at `(x, gamma + x·y)`: what is left once `gamma` is taken
    /// as the next coefficient of a root.
    fn shift_y(&self, gamma: u8) -> Curve {
        let gpow = powers(gamma, self.coef.len());
        let mut coef = Vec::with_capacity(self.coef.len());
        for b in 0..self.coef.len() {
            let mut sum: Vec<u8> = Vec::new();
            for t in b..self.coef.len() {
                if t & b != b || gpow[t - b] == 0 {
                    continue;
                }
                let g = gpow[t - b];
                for (d, &c) in self.coef[t].iter().enumerate() {
                    if c != 0 {
                        if sum.len() <= d {
                            sum.resize(d + 1, 0);
                        }
                        sum[d] ^= mul(c, g);
                    }
                }
            }
            trim(&mut sum);
            let mut shifted = if sum.is_empty() { Vec::new() } else { vec![0u8; b] };
            shifted.extend(sum);
            coef.push(shifted);
        }
        Curve { coef }
    }

    /// Divide out the largest power of `x` that divides every
    /// coefficient, so that the curve at `x = 0` is not nought.
    fn divide_x(&mut self) {
        let Some(s) = self.coef.iter().filter_map(|p| p.iter().position(|&c| c != 0)).min() else { return };
        if s > 0 {
            for poly in &mut self.coef {
                if !poly.is_empty() {
                    poly.drain(..s);
                }
            }
        }
    }

    /// The curve at `x = 0`, as a polynomial in `y`.
    fn at_zero(&self) -> Vec<u8> {
        self.coef.iter().map(|p| p.first().copied().unwrap_or(0)).collect()
    }
}

/// The degree in `y` the curve is given for `cost` constraints and data
/// of degree below `w + 1`: the smallest that lets the curve have the
/// smallest `(1, w)`-weighted degree the constraints allow. A curve of
/// weighted degree `Δ` and degree `L` in `y` has a coefficient for each
/// monomial `x^d y^t` with `d + t·w ≤ Δ`, and the constraints are met,
/// one way at least, once there are more coefficients than constraints.
fn list_size(cost: usize, w: usize) -> (usize, usize) {
    let mut best: Option<(usize, usize)> = None;
    for l in 1..=LIST_SIZE_MAX {
        let monomials = |delta: usize| (0..=l).map(|t| (delta + 1).saturating_sub(t * w)).sum::<usize>();
        let (mut lo, mut hi) = (0usize, cost + l * w + 1);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if monomials(mid) > cost {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        match best {
            Some((delta, _)) if lo >= delta => break,
            _ => best = Some((lo, l)),
        }
    }
    let (delta, l) = best.unwrap_or((cost, 1));
    (l, delta)
}

/// Koetter's interpolation: the curve of least `(1, w)`-weighted degree
/// and degree at most `list_size` in `y` that passes through each
/// constraint's `(x, y)` with its multiplicity. The curves `y^0..y^L`
/// are carried together and each constraint, a Hasse derivative that
/// must vanish, is met by taking the curve of least order among those
/// that fail it, adding a multiple of it to the rest so that they pass,
/// and raising it by `(X - x)`. `None` when the work was called off.
fn interpolate(constraints: &[(u8, u8, usize)], w: usize, list_size: usize) -> Option<Curve> {
    let mut curves: Vec<Curve> = (0..=list_size).map(Curve::monomial).collect();
    let mut met = 0usize;
    for &(x, y, m) in constraints {
        let ypow = powers(y, list_size + 1);
        let mut xpow = powers(x, 1);
        for a in 0..m {
            for b in 0..m - a {
                let degree = curves.iter().map(Curve::x_degree).max().unwrap_or(1);
                while xpow.len() <= degree {
                    xpow.push(mul(*xpow.last().expect("a power"), x));
                }
                let d: Vec<u8> = curves.iter().map(|c| c.hasse(a, b, &xpow, &ypow)).collect();
                let Some(pivot) = (0..=list_size).filter(|&j| d[j] != 0).min_by_key(|&j| curves[j].leading(w)) else { continue };
                let pivot_curve = std::mem::take(&mut curves[pivot]);
                for j in 0..=list_size {
                    if j != pivot && d[j] != 0 {
                        let c = div(d[j], d[pivot]);
                        curves[j].add_scaled(&pivot_curve, c);
                    }
                }
                curves[pivot] = pivot_curve;
                curves[pivot].times_x_minus(x);
                met += 1;
                if met.is_multiple_of(64) && work::cancelled() {
                    return None;
                }
            }
        }
    }
    curves.into_iter().min_by_key(|c| c.leading(w))
}

/// Roth and Ruckenstein's factorisation: every polynomial `H` of degree
/// below `k` with `Q(x, H(x)) = 0`, as its `k` coefficients, lowest
/// first. The constant term of `H` is a root of `Q(0, y)`; taking it,
/// the curve at `(x, h_0 + x·y)` divided by what power of `x` it has is
/// the curve whose roots are the rest of `H`, and so on `k` times, where
/// `H` is a root exactly when nothing of `y^0` is left.
fn y_roots(curve: &Curve, k: usize) -> Vec<Vec<u8>> {
    fn descend(q: &Curve, k: usize, prefix: &mut Vec<u8>, out: &mut Vec<Vec<u8>>, nodes: &mut usize, budget: usize) {
        *nodes += 1;
        if *nodes > budget || work::cancelled() {
            return;
        }
        if prefix.len() == k {
            if q.coef.first().is_none_or(Vec::is_empty) {
                out.push(prefix.clone());
            }
            return;
        }
        let at_zero = q.at_zero();
        if at_zero.iter().all(|&c| c == 0) {
            return;
        }
        for gamma in 0..=255u8 {
            if poly_eval(&at_zero, gamma) != 0 {
                continue;
            }
            let mut next = q.shift_y(gamma);
            next.divide_x();
            prefix.push(gamma);
            descend(&next, k, prefix, out, nodes, budget);
            prefix.pop();
        }
    }
    let mut start = curve.clone();
    start.divide_x();
    let mut out = Vec::new();
    let budget = 4 * k.max(1) * curve.coef.len().max(1);
    descend(&start, k, &mut Vec::with_capacity(k), &mut out, &mut 0, budget);
    out
}

/// Decode a block by algebraic soft-decision list decoding, with `cost`
/// constraints to spend on the curve ([`LIST_COST`]): every codeword of
/// the code the curve through the block's points, weighed by their
/// reliability, passes through, each restated against the block as read
/// as any solution is, kept when [`DOUBT_CONFIRMING_BITS`] confirm it,
/// those within the standard over any that spend its reserve
/// ([`preferring_the_standard`]), and of those the nearest the
/// block as read by a clear margin, or every one within the margin,
/// nearest first ([`nearest`]). Each comes with the greatest
/// multiplicity any point was given. A pinned bit is never overturned.
/// Nothing for a block whose every EC codeword the standard reserves,
/// or that has no redundancy left after its erasures.
///
/// What confirms a solution is counted as the doubting solvers count
/// it. The bits whose other colour was given a point are the doubted
/// bits, chosen by their reliability before any solution was seen, and
/// a solution that overturns nothing but doubted bits of a codeword is
/// held to the equations the rest of the block's bits make: eight a
/// codeword of redundancy, less one for each doubted bit, and the
/// doubted bits' agreement beyond chance besides
/// ([`super::rs::doubted_agreement`]). A codeword the solution changes
/// in a bit that was not doubted is an error located by the curve, and
/// charged as the plain count charges one
/// ([`list_solution_bits`]): a block of noise shows a codeword within so
/// many whole errors of it once in so many blocks. The slack is what
/// those errors leave of the redundancy, as a chased solution's is what
/// its flipped block's decode left, and the standard's reserve is held
/// against it.
pub fn rs_decode_listing(block: &[CwRead], bit_reliability: &[[u8; 8]], ec_len: usize, reserved: usize, cost: usize) -> Vec<(usize, RsSolution)> {
    assert_eq!(block.len(), bit_reliability.len());
    let n = block.len();
    if cost == 0 || ec_len == 0 || ec_len >= n || reserved >= ec_len {
        return Vec::new();
    }
    let k = n - ec_len;
    let erased: Vec<usize> = (0..n).filter(|&i| block[i].is_erasure()).collect();
    let f = erased.len();
    if f >= ec_len {
        return Vec::new();
    }
    let pts = points(block, bit_reliability);
    let m = multiplicities(&pts, cost.min(LIST_COST_MAX));
    let grs = grs_points(n, ec_len);
    let constraints: Vec<(u8, u8, usize)> = pts
        .iter()
        .zip(&m)
        .filter(|&(_, &m)| m > 0)
        .map(|(p, &m)| (grs[p.codeword].0, div(p.value, grs[p.codeword].1), m))
        .collect();
    if constraints.is_empty() {
        return Vec::new();
    }
    // The bits of each codeword whose other colour was given a point.
    let mut doubted = vec![0u8; n];
    for (p, &m) in pts.iter().zip(&m) {
        if m > 0 {
            doubted[p.codeword] |= (p.value ^ block[p.codeword].value) & block[p.codeword].known;
        }
    }
    // An erased codeword with doubted bits is weighed as a doubted one,
    // its agreement against its disagreement, rather than refused for
    // the known bit it overturns.
    let doubted_erased: Vec<usize> = erased.iter().copied().filter(|&i| doubted[i] != 0).collect();
    let total: usize = constraints.iter().map(|&(_, _, m)| m * (m + 1) / 2).sum();
    let multiplicity = constraints.iter().map(|c| c.2).max().unwrap_or(0);
    let (list_size, _) = list_size(total, k - 1);
    let Some(curve) = interpolate(&constraints, k - 1, list_size) else { return Vec::new() };
    let (read, redundancy) = (n - f, ec_len - f);
    let mut found: Vec<(u32, (usize, RsSolution))> = Vec::new();
    for h in y_roots(&curve, k) {
        let corrected: Vec<u8> = (0..n).map(|i| mul(grs[i].1, poly_eval(&h, grs[i].0))).collect();
        let (mut hard, mut errors, mut agreeing, mut flipped) = (0usize, 0usize, 0u32, 0u32);
        for i in 0..n {
            if block[i].is_erasure() {
                continue;
            }
            let diff = corrected[i] ^ block[i].value;
            if diff != 0 {
                errors += 1;
            }
            if diff & !doubted[i] != 0 {
                hard += 1;
            } else {
                agreeing += (doubted[i] & !diff).count_ones();
                flipped += (doubted[i] & diff).count_ones();
            }
        }
        let bits = list_solution_bits(read, redundancy, hard).saturating_sub(agreeing + flipped) + doubted_agreement(agreeing, flipped);
        // The slack never claims more than the bits confirm: the doubted
        // bits spent equations of it.
        let slack = redundancy.saturating_sub(2 * hard).min(bits as usize / 8);
        let located = bits - 8 * slack as u32;
        let Ok(solution) = settle(block, &doubted_erased, &erased, corrected, errors, slack, located, reserved) else { continue };
        if solution.confidence_bits() < DOUBT_CONFIRMING_BITS {
            continue;
        }
        found.push((soft_distance(block, bit_reliability, &solution.corrected), (multiplicity, solution)));
    }
    let found = preferring_the_standard(found, |(_, solution)| solution);
    nearest(found, |a, b| a.1.corrected == b.1.corrected, |a, b| a.1.confidence_bits() > b.1.confidence_bits())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::rs::{AIDED_VERIFIED_BITS, rs_decode, rs_encode};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        fn byte(&mut self) -> u8 {
            (self.next() >> 24) as u8
        }
        fn nonzero(&mut self) -> u8 {
            loop {
                let b = self.byte();
                if b != 0 {
                    return b;
                }
            }
        }
        fn distinct(&mut self, count: usize, n: usize) -> Vec<usize> {
            let mut picked = Vec::new();
            while picked.len() < count {
                let i = self.below(n);
                if !picked.contains(&i) {
                    picked.push(i);
                }
            }
            picked
        }
    }

    fn codeword(rng: &mut Rng, data_len: usize, ec_len: usize) -> Vec<u8> {
        let mut block: Vec<u8> = (0..data_len).map(|_| rng.byte()).collect();
        let ec = rs_encode(&block, ec_len);
        block.extend(ec);
        block
    }

    /// Lagrange interpolation through the points, little-endian.
    fn lagrange(points: &[(u8, u8)]) -> Vec<u8> {
        let mut out = vec![0u8; points.len()];
        for (i, &(xi, yi)) in points.iter().enumerate() {
            let mut basis = vec![1u8];
            let mut denominator = 1u8;
            for (j, &(xj, _)) in points.iter().enumerate() {
                if i == j {
                    continue;
                }
                basis = gf256::poly_mul(&basis, &[xj, 1]);
                denominator = mul(denominator, xi ^ xj);
            }
            let scale = div(yi, denominator);
            for (o, b) in out.iter_mut().zip(&basis) {
                *o ^= mul(scale, *b);
            }
        }
        out
    }

    #[test]
    fn codewords_are_a_polynomial_of_degree_below_k_at_the_points() {
        let mut rng = Rng(0x6A5);
        for &(data_len, ec_len) in &[(9usize, 17usize), (15, 18), (19, 7), (118, 30), (15, 30), (3, 2)] {
            let n = data_len + ec_len;
            let grs = grs_points(n, ec_len);
            for round in 0..5 {
                let block = codeword(&mut rng, data_len, ec_len);
                let scaled: Vec<(u8, u8)> = (0..n).map(|i| (grs[i].0, div(block[i], grs[i].1))).collect();
                let h = lagrange(&scaled[..data_len]);
                for &(x, y) in &scaled {
                    assert_eq!(poly_eval(&h, x), y, "({data_len}, {ec_len}) round {round}");
                }
                // And every polynomial of degree below k is a codeword.
                let h: Vec<u8> = (0..data_len).map(|_| rng.byte()).collect();
                let word: Vec<u8> = (0..n).map(|i| mul(grs[i].1, poly_eval(&h, grs[i].0))).collect();
                let read: Vec<CwRead> = word.iter().map(|&v| CwRead::known(v)).collect();
                assert_eq!(rs_decode(&read, ec_len).map(|s| s.errors), Ok(0), "({data_len}, {ec_len}) round {round}");
            }
        }
    }

    #[test]
    fn a_curve_is_tangent_where_it_is_told_and_its_roots_are_read_off() {
        // A curve through three points with multiplicity two vanishes
        // with its first derivatives there, and the line through two of
        // them is read off when the third lies on it.
        let constraints = [(exp(1), exp(5), 2usize), (exp(2), exp(6), 2), (exp(3), exp(7), 2)];
        let curve = interpolate(&constraints, 1, 3).unwrap();
        for &(x, y, m) in &constraints {
            let xpow = powers(x, curve.x_degree() + 1);
            let ypow = powers(y, curve.coef.len() + 1);
            for a in 0..m {
                for b in 0..m - a {
                    assert_eq!(curve.hasse(a, b, &xpow, &ypow), 0, "({a}, {b}) at α^{}", gf256::log(x));
                }
            }
        }
        // The line y = α^4 · x passes through all three.
        let roots = y_roots(&curve, 2);
        assert!(roots.contains(&vec![0, exp(4)]), "{roots:?}");
        // A list size: more constraints want a higher degree in y, and
        // a higher weight a lower.
        assert_eq!(list_size(10, 1).0, 2);
        assert!(list_size(2048, 14).0 >= list_size(2048, 117).0);
    }

    #[test]
    fn list_decoding_reaches_past_half_the_distance() {
        let mut rng = Rng(0x115D);
        // A 1-H block: half the distance is eight, the trials of erasures
        // reach ten, and the algebra eleven, verified.
        for round in 0..20 {
            let block = codeword(&mut rng, 9, 17);
            let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
            let wrong = rng.distinct(12, 26);
            for &i in &wrong[..11] {
                read[i].value ^= rng.nonzero();
            }
            let certain = vec![[CERTAIN; 8]; 26];
            assert!(rs_decode(&read, 17).is_err(), "round {round}");
            let found = rs_decode_listing(&read, &certain, 17, 0, LIST_COST);
            assert_eq!(found.len(), 1, "round {round}: {found:?}");
            let (multiplicity, solution) = &found[0];
            assert_eq!(solution.corrected, block, "round {round}");
            assert_eq!((solution.errors, solution.erasures, solution.slack), (11, 0, 0), "round {round}");
            assert!(solution.confidence_bits() >= AIDED_VERIFIED_BITS, "round {round}: {}", solution.confidence_bits());
            assert!(*multiplicity >= 6, "round {round}: multiplicity {multiplicity}");
            // One more is past the algebra: whatever is found is not the
            // block, and not verified.
            read[wrong[11]].value ^= rng.nonzero();
            for (_, solution) in rs_decode_listing(&read, &certain, 17, 0, LIST_COST) {
                assert_ne!(solution.corrected, block, "round {round} and one more");
                assert!(solution.confidence_bits() < AIDED_VERIFIED_BITS, "round {round} and one more");
            }
        }
        // A 40-H block: half the distance is fifteen, pairs reach sixteen,
        // and the algebra nineteen.
        for round in 0..5 {
            let block = codeword(&mut rng, 15, 30);
            let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
            for &i in &rng.distinct(19, 45) {
                read[i].value ^= rng.nonzero();
            }
            let certain = vec![[CERTAIN; 8]; 45];
            let found = rs_decode_listing(&read, &certain, 30, 0, LIST_COST);
            assert_eq!(found.len(), 1, "40-H round {round}: {found:?}");
            assert_eq!((found[0].1.corrected == block, found[0].1.errors), (true, 19), "40-H round {round}");
            assert!(found[0].1.confidence_bits() >= AIDED_VERIFIED_BITS);
        }
        // Erasures take their codewords out of the count: a 1-H block
        // with four unknown and eight wrong of the twenty-two read, where
        // errors and erasures together reach six, and the algebra on
        // twenty-two points eight.
        let block = codeword(&mut rng, 9, 17);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        let picked = rng.distinct(12, 26);
        for &i in &picked[..4] {
            read[i] = CwRead::unknown();
        }
        for &i in &picked[4..] {
            read[i].value ^= rng.nonzero();
        }
        assert!(rs_decode(&read, 17).is_err());
        let found = rs_decode_listing(&read, &vec![[CERTAIN; 8]; 26], 17, 0, LIST_COST);
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!((found[0].1.corrected == block, found[0].1.errors, found[0].1.erasures), (true, 8, 4));
        // A pinned codeword is never overturned, and a block with its
        // every EC codeword reserved is not decoded so.
        let mut pinned = read.clone();
        pinned[picked[4]].pinned = 0xFF;
        assert!(rs_decode_listing(&pinned, &vec![[CERTAIN; 8]; 26], 17, 0, LIST_COST).is_empty());
        assert!(rs_decode_listing(&read, &vec![[CERTAIN; 8]; 26], 17, 17, LIST_COST).is_empty());
    }

    #[test]
    fn faint_bits_bend_the_curve_toward_their_other_colour() {
        // A 5-Q block wrong in thirteen codewords, each in one faint bit:
        // past half the distance, past the algebra on the bits as read,
        // and past the ten the chase flips. Weighed, the faint bits'
        // other colours are points of their own, and the curve passes
        // through the block.
        let mut rng = Rng(0xFA17);
        for round in 0..20 {
            let block = codeword(&mut rng, 15, 18);
            let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
            let mut reliability = vec![[200u8; 8]; 33];
            for &i in &rng.distinct(13, 33) {
                let b = rng.below(8);
                read[i].value ^= 1 << b;
                reliability[i][b] = rng.below(12) as u8;
            }
            assert!(rs_decode_listing(&read, &vec![[200u8; 8]; 33], 18, 0, LIST_COST).iter().all(|(_, s)| s.corrected != block), "round {round}: reached as read");
            let found = rs_decode_listing(&read, &reliability, 18, 0, LIST_COST);
            assert_eq!(found.len(), 1, "round {round}: {found:?}");
            assert_eq!((found[0].1.corrected == block, found[0].1.errors), (true, 13), "round {round}");
            // Nothing but doubted bits was overturned: the solution is
            // held to the equations the rest make, thirteen short of the
            // hundred and forty-four.
            assert_eq!(found[0].1.confidence_bits(), list_solution_bits(33, 18, 0) - 13, "round {round}");
            assert_eq!(found[0].1.slack, 16, "round {round}");
            assert!(found[0].1.confidence_bits() >= AIDED_VERIFIED_BITS, "round {round}: {}", found[0].1.confidence_bits());
            // Bits unknown, the codewords are read at half a byte each;
            // with two unknown bits in each of a few, the fillings are
            // points too.
            let mut read = read.clone();
            for &i in &rng.distinct(3, 33) {
                read[i].known &= 0xFC;
                read[i].value &= 0xFC;
            }
            let found = rs_decode_listing(&read, &reliability, 18, 0, LIST_COST);
            assert!(found.iter().any(|(_, s)| s.corrected == block), "round {round} with unknown bits: {found:?}");
        }
    }

    #[test]
    fn noise_is_never_read_as_verified() {
        let mut rng = Rng(0x0153);
        for (n, ec_len) in [(26usize, 17usize), (33, 18), (44, 10), (26, 7)] {
            for round in 0..150 {
                let read: Vec<CwRead> = (0..n).map(|_| CwRead::known(rng.byte())).collect();
                let reliability: Vec<[u8; 8]> = (0..n).map(|_| [rng.byte().min(CERTAIN - 1); 8]).collect();
                for (_, solution) in rs_decode_listing(&read, &reliability, ec_len, 0, LIST_COST) {
                    assert!(solution.confidence_bits() < AIDED_VERIFIED_BITS, "({n}, {ec_len}) round {round}: {solution:?}");
                }
            }
        }
        // A 2-L block has a codeword within six of noise once in six
        // hundred blocks: nothing so far out is offered at all.
        assert!(list_solution_bits(44, 10, 6) < DOUBT_CONFIRMING_BITS);
        assert!(list_solution_bits(44, 10, 5) >= DOUBT_CONFIRMING_BITS);
        // Within half the distance the count is the plain decoder's.
        assert_eq!(list_solution_bits(33, 18, 9), super::super::rs::located_error_bits(33, 9));
        assert_eq!(list_solution_bits(26, 17, 11), 25);
        assert_eq!(list_solution_bits(45, 30, 19), 46);
    }

    #[test]
    fn the_cost_bounds_the_time() {
        let mut rng = Rng(0x7E);
        let block = codeword(&mut rng, 118, 30);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        for &i in &rng.distinct(16, 148) {
            read[i].value ^= rng.nonzero();
        }
        let started = std::time::Instant::now();
        let found = rs_decode_listing(&read, &vec![[CERTAIN; 8]; 148], 30, 0, LIST_COST);
        let took = started.elapsed();
        println!("40-L block, sixteen wrong, {LIST_COST} constraints: {took:?}, {} found", found.len());
        assert!(took.as_secs() < 5, "{took:?}");
        // Sixteen is one past half the distance, and the bound for this
        // block: the curve reaches it when the multiplicities allow.
        if let Some((_, solution)) = found.first() {
            assert_eq!(solution.corrected, block);
        }
    }
}

