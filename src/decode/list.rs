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
//! ten faintest bits, and this does for every bit at once. Where every
//! codeword is as sure as every other, as in a block painted by hand,
//! each is given the same multiplicity where that is sure to reach
//! further than shares that differ by a unit as they fall
//! ([`multiplicities`]).
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

use super::gf256::{self, Scale, div, exp, mul, poly_eval};
use super::rs::{
    CwRead, Offering, RsSolution, Sightings, counted_against, doubted_agreement, list_solution_balance, list_solution_bits, nearest,
    preferring_the_standard, settle, soft_distance,
};
use crate::symbol::grid::{CERTAIN, misread_chance};
use crate::work;

/// Constraints the interpolation is given by default, counting a point
/// of multiplicity `m` as `m(m+1)/2`: what the curve costs, since every
/// constraint is a pass over every coefficient of every polynomial in
/// hand, and how far it reaches, since the multiplicities grow with the
/// square root of it. Two thousand puts a multiplicity of nine or ten on
/// every codeword of a 40-H block read whole, which reaches nineteen
/// wrong of its forty-five, and of twelve or thirteen on a 1-H block's
/// twenty-six, which reaches eleven; a block of a hundred and fifty
/// codewords gets four or five, which on a block of high rate is sure
/// of no further than half the distance, and nothing would be. A block
/// read whole that one multiplicity alike takes further uses no more of
/// them than that takes ([`multiplicities`]): five on each of an 8-Q
/// block's forty, six hundred, reach thirteen wrong where the units
/// given one at a time were sure of twelve. A tenth of a second on the
/// largest blocks, and a few milliseconds on the smallest.
pub const LIST_COST: usize = 2048;
/// Fewest constraints a decode that is one of many is given
/// ([`super::pipeline::DecodeOptions::among`]).
pub const LIST_COST_MIN: usize = 512;
/// Most constraints the interpolation may be set to by hand
/// ([`super::pipeline::DecodeOptions::list_cost`]): seconds on the
/// largest blocks. A block of low rate uses fewer ([`usable_cost`]).
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
/// A block of low rate given many constraints would want more, and is
/// given no more constraints than this lets it use ([`usable_cost`]): a
/// list of over a hundred, which the most constraints would want on the
/// smallest blocks, reaches a codeword further on a few of them, at ten
/// times the time and more.
const LIST_SIZE_MAX: usize = 32;

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
            if cw.known >> b & 1 != 0 && cw.pinned >> b & 1 == 0 { misread_chance(bit_reliability[i][b]) } else { 0.0 }
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
/// constraints, for a curve whose data is of degree `w` at most: a unit
/// at a time to the point whose likelihood divided by one more than its
/// multiplicity is the greatest, which is their greedy algorithm, until
/// the next unit would not fit. A point of likelihood one, a codeword
/// painted by hand or pinned, draws the most, and a faint bit's other
/// colour draws its share.
///
/// Where every point is as likely as every other, as every codeword of a
/// block painted by hand is, the greedy has no likelihood to choose by,
/// and hands the units left past the last multiplicity it could give
/// them all to the points first in the block, one each: the curve they
/// raise reaches a codeword more where the wrong codewords miss those
/// points, and less far where they fall on them. There one multiplicity
/// alike ([`even_multiplicity`]) is given instead where it is sure to
/// reach further wherever the wrong codewords fall ([`sure_reach`]), or
/// as far where the greedy's are alike too, at fewer constraints. An
/// M4-M block is sure of five wrong at two thousand constraints a unit
/// at a time, and of six at six hundred and seventy-two given alike,
/// and a 3-M block of thirteen and fourteen. Given alike always, a 7-M
/// block at a thousand and twenty-four constraints, just short of a
/// multiplicity of six on every codeword, was sure of nine, no further
/// than half the distance, where the greedy's, six on all but one, are
/// sure of ten; and where the two were sure of as many, a block wrong
/// in one codeword more than that never decoded, where the greedy's
/// reached it as often as not. Short of half the distance, which the
/// plain decoder reaches by itself before the curve is asked, how far
/// either is sure of counts for nothing: a block of high rate, sure of
/// half the distance alike only at a multiplicity of one, which adds
/// nothing to the plain decode, keeps the greedy's, and what they reach
/// past it where the wrong codewords miss the points first in the block.
fn multiplicities(points: &[Point], cost: usize, w: usize) -> Vec<usize> {
    let greedy = greedy_multiplicities(points, cost);
    let Some(even) = even_multiplicity(points, cost, w) else { return greedy };
    let even = vec![even; points.len()];
    let mut codewords: Vec<usize> = points.iter().map(|p| p.codeword).collect();
    codewords.dedup();
    let half = codewords.len().saturating_sub(w + 1) / 2;
    let (sure_even, sure_greedy) = (sure_reach(points, &even, w).max(half), sure_reach(points, &greedy, w).max(half));
    let greedy_alike = greedy.windows(2).all(|pair| pair[0] == pair[1]);
    if sure_even > sure_greedy || (sure_even == sure_greedy && greedy_alike) { even } else { greedy }
}

/// Koetter and Vardy's greedy multiplicities for the points, within
/// `cost` constraints ([`multiplicities`]).
fn greedy_multiplicities(points: &[Point], cost: usize) -> Vec<usize> {
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

/// The one multiplicity to give every point where all are alike in
/// likelihood ([`multiplicities`]), within `cost` constraints for a curve
/// whose data is of degree `w` at most: of those whose curve is sure to
/// reach the most wrong codewords, wherever they fall, the least, which
/// costs the least time. Given alike, a codeword of the code is a factor
/// of the curve once the codewords it agrees with outnumber its weighted
/// degree over the multiplicity, and is not otherwise, so that this is
/// how far the curve reaches, and no further for any block; the greedy's
/// multiplicities, which differ by a unit, may reach further at the
/// same `cost`, and [`multiplicities`] weighs the two. `None` where the
/// points differ, or no multiplicity alike fits `cost`.
fn even_multiplicity(points: &[Point], cost: usize, w: usize) -> Option<usize> {
    let likelihood = points.first()?.likelihood;
    if likelihood <= 0.0 || points.iter().any(|p| p.likelihood != likelihood) {
        return None;
    }
    // A codeword of the code agrees with one point of a codeword at most.
    let mut codewords: Vec<usize> = points.iter().map(|p| p.codeword).collect();
    codewords.dedup();
    let (points, codewords) = (points.len(), codewords.len());
    let constraints = |m: usize| points * m * (m + 1) / 2;
    (1..)
        .take_while(|&m| constraints(m) <= cost)
        .filter_map(|m| {
            let (_, delta) = list_size(constraints(m), w);
            codewords.checked_sub(delta / m + 1).map(|reach| (reach, m))
        })
        .max_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)))
        .map(|(_, m)| m)
}

/// How many wrong codewords a curve whose points have multiplicities `m`,
/// for data of degree `w` at most, is sure to reach wherever they fall:
/// a codeword of the code is a factor of the curve once the
/// multiplicities of the points it passes through sum past the curve's
/// weighted degree, and a codeword read right adds the least of its
/// points' at the least, a wrong one nothing. As [`hard_reach`] counts
/// it, where each codeword has one point.
fn sure_reach(points: &[Point], m: &[usize], w: usize) -> usize {
    let total: usize = m.iter().map(|&m| m * (m + 1) / 2).sum();
    let (_, delta) = list_size(total, w);
    let mut least: Vec<(usize, usize)> = Vec::new();
    for (point, &m) in points.iter().zip(m) {
        match least.last_mut() {
            Some((codeword, least)) if *codeword == point.codeword => *least = (*least).min(m),
            _ => least.push((point.codeword, m)),
        }
    }
    let mut least: Vec<usize> = least.into_iter().map(|(_, m)| m).collect();
    least.sort_unstable();
    (0..=least.len()).take_while(|&t| least[..least.len() - t].iter().sum::<usize>() > delta).last().unwrap_or(0)
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
    #[cfg(test)]
    fn hasse(&self, a: usize, b: usize, xpow: &[u8], ypow: &[u8]) -> u8 {
        let mut total = 0u8;
        // The powers whose bits hold the order's, in turn: the next after
        // `t` is `(t + 1) | b`.
        let mut t = b;
        while t < self.coef.len() {
            let poly = &self.coef[t];
            let mut inner = 0u8;
            let mut d = a;
            while d < poly.len() {
                inner ^= mul(poly[d], xpow[d - a]);
                d = (d + 1) | a;
            }
            if inner != 0 {
                total ^= mul(inner, ypow[t - b]);
            }
            t = (t + 1) | b;
        }
        total
    }

    /// The Hasse derivatives in `x` alone of each coefficient of the curve
    /// at the point whose powers are `xpow`, of the orders below `m`, for
    /// `lists` powers of `y`: entry `t * m + a` is order `a` of the
    /// coefficient of `y^t`, as [`Self::hasse`] takes them one at a time.
    fn x_derivatives(&self, m: usize, lists: usize, xpow: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; lists * m];
        for (t, poly) in self.coef.iter().enumerate() {
            for a in 0..m {
                let mut inner = 0u8;
                let mut d = a;
                while d < poly.len() {
                    inner ^= mul(poly[d], xpow[d - a]);
                    d = (d + 1) | a;
                }
                out[t * m + a] = inner;
            }
        }
        out
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
        let times_c = Scale::new(c);
        for (mine, theirs) in self.coef.iter_mut().zip(&other.coef) {
            if mine.len() < theirs.len() {
                mine.resize(theirs.len(), 0);
            }
            times_c.add_into(mine, theirs);
            trim(mine);
        }
    }

    /// `self *= (X - x)`, in place: from the top, each coefficient
    /// moves up a degree and leaves `x` times itself behind.
    fn times_x_minus(&mut self, x: u8) {
        let times_x = Scale::new(x);
        for poly in &mut self.coef {
            if poly.is_empty() {
                continue;
            }
            // Coefficient `d` becomes `x` times itself and the one below
            // it: from the top, eight at a time, each reading only what
            // has not been written yet.
            poly.push(0);
            let mut top = poly.len();
            while top >= 9 {
                let word = |at: usize| u64::from_le_bytes(poly[at..at + 8].try_into().expect("eight bytes"));
                let raised = times_x.word(word(top - 8)) ^ word(top - 9);
                poly[top - 8..top].copy_from_slice(&raised.to_le_bytes());
                top -= 8;
            }
            for d in (1..top).rev() {
                poly[d] = times_x.byte(poly[d]) ^ poly[d - 1];
            }
            poly[0] = times_x.byte(poly[0]);
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

/// The Hasse derivative of order `(a, b)` of a curve whose derivatives in
/// `x` alone are `derivatives` ([`Curve::x_derivatives`]), at the point
/// whose powers of `y` are `ypow`: what [`Curve::hasse`] makes of the curve
/// itself.
fn y_derivative(derivatives: &[u8], m: usize, a: usize, b: usize, ypow: &[u8]) -> u8 {
    let lists = derivatives.len() / m;
    let mut total = 0u8;
    let mut t = b;
    while t < lists {
        let inner = derivatives[t * m + a];
        if inner != 0 {
            total ^= mul(inner, ypow[t - b]);
        }
        t = (t + 1) | b;
    }
    total
}

/// The degree in `y` the curve is given for `cost` constraints and data
/// of degree below `w + 1`: the smallest that lets the curve have the
/// smallest `(1, w)`-weighted degree the constraints allow. A curve of
/// weighted degree `Δ` and degree `L` in `y` has a coefficient for each
/// monomial `x^d y^t` with `d + t·w ≤ Δ`, and the constraints are met,
/// one way at least, once there are more coefficients than constraints.
fn list_size(cost: usize, w: usize) -> (usize, usize) {
    list_size_within(cost, w, LIST_SIZE_MAX)
}

/// [`list_size`], with the degree in `y` held to `most`.
fn list_size_within(cost: usize, w: usize, most: usize) -> (usize, usize) {
    let mut best: Option<(usize, usize)> = None;
    for l in 1..=most {
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

/// Constraints the interpolation spends on a block of `n` codewords with
/// `ec_len` of error correction, given `cost`: no more than
/// [`LIST_COST_MAX`], and no more than a list of [`LIST_SIZE_MAX`] lets
/// the curve use. Past that the curve would need a greater degree in `y`
/// to keep its weighted degree to the square root of the constraints;
/// held to the list, its weighted degree grows with the constraints
/// themselves, which the multiplicities, growing with their square root,
/// do not keep up with, and more constraints reach less far, at more
/// cost. The constraints stop short of the first count the list holds
/// the curve back at, however many more `cost` gives: a 40-H block uses
/// about eight thousand, a 1-H block about five thousand, and a block
/// of high rate every one it is given.
pub fn usable_cost(n: usize, ec_len: usize, cost: usize) -> usize {
    let cost = cost.min(LIST_COST_MAX);
    if ec_len >= n {
        return cost;
    }
    let w = n - ec_len - 1;
    // The list is too short where one codeword more would lower the
    // curve's weighted degree. That comes and goes, as the weighted
    // degree steps, for some hundreds of constraints before it holds
    // for good, and was once halved over as if it came once: a block
    // used more constraints or fewer as more were given. It cannot come
    // until the curve's weighted degree passes `w` times one more than
    // the list, below which a monomial of one more degree in `y` has no
    // room; a curve of the list's degree in `y` has every monomial up to
    // that degree, and needs a greater, at `first` constraints.
    let held = |cost: usize| list_size_within(cost, w, LIST_SIZE_MAX + 1).1 < list_size(cost, w).1;
    let first = (LIST_SIZE_MAX + 1) * (LIST_SIZE_MAX + 2) / 2 * w + LIST_SIZE_MAX + 1;
    (first..=cost).find(|&cost| held(cost)).map_or(cost, |held_at| held_at - 1)
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
    // Each curve's leading monomial and degree in `x`, kept as the curves
    // change: a constraint changes only the curves that fail it.
    let mut leading: Vec<Option<(usize, usize)>> = curves.iter().map(|c| c.leading(w)).collect();
    let mut degrees: Vec<usize> = curves.iter().map(Curve::x_degree).collect();
    let mut d = vec![0u8; list_size + 1];
    let mut met = 0usize;
    for &(x, y, m) in constraints {
        let ypow = powers(y, list_size + 1);
        let degree = degrees.iter().copied().max().unwrap_or(1);
        let xpow = powers(x, degree + 1);
        // Each curve's Hasse derivatives in `x` alone at the point, of every
        // order the point's constraints ask, for each power of `y`: entry
        // `t * m + a` is the coefficient of `(X - x)^a` in the coefficient
        // of `y^t`. A constraint's derivative is read off them
        // ([`Curve::hasse`]), and they are kept as the curves change, which
        // they do only by adding a multiple of another curve, whose
        // derivatives add as the curves do, and by `(X - x)`, which raises
        // each derivative's order by one. Every constraint once took its
        // derivatives afresh from every coefficient of every curve, and at
        // a point of multiplicity `m` that was `m(m+1)/2` passes over them
        // where this is `m`.
        let mut derivatives: Vec<Vec<u8>> = curves.iter().map(|curve| curve.x_derivatives(m, list_size + 1, &xpow)).collect();
        for a in 0..m {
            for b in 0..m - a {
                for (dj, derivative) in d.iter_mut().zip(&derivatives) {
                    *dj = y_derivative(derivative, m, a, b, &ypow);
                }
                let Some(pivot) = (0..=list_size).filter(|&j| d[j] != 0).min_by_key(|&j| leading[j]) else { continue };
                let pivot_curve = std::mem::take(&mut curves[pivot]);
                let pivot_derivative = std::mem::take(&mut derivatives[pivot]);
                for j in 0..=list_size {
                    if j != pivot && d[j] != 0 {
                        let c = div(d[j], d[pivot]);
                        curves[j].add_scaled(&pivot_curve, c);
                        Scale::new(c).add_into(&mut derivatives[j], &pivot_derivative);
                        leading[j] = curves[j].leading(w);
                        degrees[j] = curves[j].x_degree();
                    }
                }
                curves[pivot] = pivot_curve;
                curves[pivot].times_x_minus(x);
                derivatives[pivot] = pivot_derivative;
                for orders in derivatives[pivot].chunks_exact_mut(m) {
                    orders.copy_within(..m - 1, 1);
                    orders[0] = 0;
                }
                leading[pivot] = curves[pivot].leading(w);
                degrees[pivot] = curves[pivot].x_degree();
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
/// as any solution is, kept when it is offered ([`Offering`]),
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
/// its flipped block's decode left; whether it keeps the standard's
/// reserve is what it changes of the block as read, the codewords it
/// changes only in bits it doubted among them ([`super::rs::keeps_reserve`]).
///
/// Every codeword the curve passes through that overturns no pin is
/// noted in `seen`, those returned among them, whether or not anything
/// confirms it ([`Sightings`]).
pub fn rs_decode_listing(
    block: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    reserved: usize,
    cost: usize,
    seen: &mut Sightings,
) -> Vec<(usize, RsSolution)> {
    rs_decode_listing_against(block, block, bit_reliability, ec_len, reserved, cost, seen)
}

/// [`rs_decode_listing`], what each solution spends of the reserve
/// counted against `as_read` ([`counted_against`]).
pub(super) fn rs_decode_listing_against(
    block: &[CwRead],
    as_read: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    reserved: usize,
    cost: usize,
    seen: &mut Sightings,
) -> Vec<(usize, RsSolution)> {
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
    let m = multiplicities(&pts, usable_cost(n, ec_len, cost), k - 1);
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
    let mut offering = Offering::new(block, bit_reliability, ec_len, seen);
    let mut found: Vec<(u32, (usize, RsSolution))> = Vec::new();
    for h in y_roots(&curve, k) {
        let corrected: Vec<u8> = (0..n).map(|i| mul(grs[i].1, poly_eval(&h, grs[i].0))).collect();
        offering.seen().see_unless_pinned(block, &corrected);
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
        // Each doubted bit spends an equation, and its agreement beyond
        // chance is credited back; the equations are taken from the two
        // together. Taken from the codewords' count alone, where there
        // are more doubted bits than equations, the shortfall was lost:
        // the codewords that agree with every bit not doubted are then
        // two to the shortfall, any block of noise has them, and the one
        // the curve finds is the one that agrees with the most doubted
        // bits, an agreement the search made and not chance. Blocks of
        // noise read as faint were so confirmed by a hundred and fifty
        // bits and more.
        let bits = list_solution_balance(read, redundancy, hard) + i64::from(doubted_agreement(agreeing, flipped)) - i64::from(agreeing + flipped);
        // The slack never claims more than the bits confirm: the doubted
        // bits spent equations of it. What the count falls short of
        // nothing by is taken from the known bits of the erased codewords
        // ([`settle`]).
        let slack = redundancy.saturating_sub(2 * hard).min(bits.max(0) as usize / 8);
        let located = bits - 8 * slack as i64;
        let Ok(solution) = settle(block, ec_len, &doubted_erased, &erased, corrected, errors, slack, located, reserved, offering.seen())
            .map(|solution| counted_against(as_read, ec_len, solution))
        else {
            continue;
        };
        if !offering.offers_solution(&solution) {
            continue;
        }
        found.push((soft_distance(block, bit_reliability, &solution.corrected), (multiplicity, solution)));
    }
    let found = preferring_the_standard(found, |(_, solution)| solution);
    nearest(found, |a, b| a.1.corrected == b.1.corrected, |a, b| a.1.confidence_bits() > b.1.confidence_bits())
}

/// Wrong codewords [`rs_decode_listing`] is sure to reach in a block of
/// `n` codewords with `ec_len` of error correction, read whole and every
/// module certain, for `cost` constraints, and with `needed` bits to
/// confirm it, which is what verifies a block charged for what the
/// solution was found among ([`super::pipeline::verifying_bits`]), or,
/// where it is less, what the block itself must show of the codeword
/// whatever found it ([`super::pipeline::judged_verifying_bits`]): the
/// count here is both its own and the block's. For the app to say how
/// far it goes in a symbol. Each codeword is a
/// point of likelihood one, given its multiplicity as [`multiplicities`]
/// gives it, and a codeword of the code agreeing with the block on
/// points whose multiplicities sum past the curve's weighted degree is a
/// factor of the curve; the wrong codewords are put where the
/// multiplicities are greatest. Unknown codewords and faint bits, which
/// take it less far or further, are left out. The codewords the standard
/// reserves for detecting misreads are left out too: past half the
/// distance a block read certain has none of its redundancy to spare,
/// so where any are reserved, a solution reached past what the standard
/// corrects spends them, and is never confirmed however much else
/// confirms it ([`super::rs::RsSolution::within_standard`]).
pub fn hard_reach(n: usize, ec_len: usize, cost: usize, needed: u32) -> usize {
    if cost == 0 || ec_len == 0 || ec_len >= n {
        return 0;
    }
    let block = vec![CwRead::known(0); n];
    let mut m = multiplicities(&points(&block, &vec![[CERTAIN; 8]; n]), usable_cost(n, ec_len, cost), n - ec_len - 1);
    let total: usize = m.iter().map(|&m| m * (m + 1) / 2).sum();
    let (_, delta) = list_size(total, n - ec_len - 1);
    m.sort_unstable();
    (0..n)
        .take_while(|&t| m[..n - t].iter().sum::<usize>() > delta && list_solution_bits(n, ec_len, t) >= needed)
        .last()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::pipeline::{DecodeOptions, verifying_bits};
    use crate::decode::rs::{VERIFIED_BITS, rs_decode, rs_encode};
    use crate::symbol::SymbolVersion;

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


    /// Koetter's interpolation as it was, each derivative taken afresh.
    fn interpolate_reference(constraints: &[(u8, u8, usize)], w: usize, list_size: usize) -> Option<Curve> {
        let mut curves: Vec<Curve> = (0..=list_size).map(Curve::monomial).collect();
        let mut leading: Vec<Option<(usize, usize)>> = curves.iter().map(|c| c.leading(w)).collect();
        let mut degrees: Vec<usize> = curves.iter().map(Curve::x_degree).collect();
        let mut d = vec![0u8; list_size + 1];
        for &(x, y, m) in constraints {
            let ypow = powers(y, list_size + 1);
            let mut xpow = powers(x, 1);
            for a in 0..m {
                for b in 0..m - a {
                    let degree = degrees.iter().copied().max().unwrap_or(1);
                    while xpow.len() <= degree {
                        xpow.push(mul(*xpow.last().expect("a power"), x));
                    }
                    for (dj, curve) in d.iter_mut().zip(&curves) {
                        *dj = curve.hasse(a, b, &xpow, &ypow);
                    }
                    let Some(pivot) = (0..=list_size).filter(|&j| d[j] != 0).min_by_key(|&j| leading[j]) else { continue };
                    let pivot_curve = std::mem::take(&mut curves[pivot]);
                    for j in 0..=list_size {
                        if j != pivot && d[j] != 0 {
                            let c = div(d[j], d[pivot]);
                            curves[j].add_scaled(&pivot_curve, c);
                            leading[j] = curves[j].leading(w);
                            degrees[j] = curves[j].x_degree();
                        }
                    }
                    curves[pivot] = pivot_curve;
                    curves[pivot].times_x_minus(x);
                    leading[pivot] = curves[pivot].leading(w);
                    degrees[pivot] = curves[pivot].x_degree();
                }
            }
        }
        curves.into_iter().min_by_key(|c| c.leading(w))
    }

    #[test]
    fn derivatives_kept_through_the_interpolation_are_those_taken_afresh() {
        let mut rng = Rng(0x4A55E);
        for round in 0..120 {
            let n = [8usize, 26, 45, 70][round % 4];
            let k = 1 + rng.below(n / 2);
            let points = grs_points(n, n - k);
            let mut constraints = Vec::new();
            for &(x, _) in &points {
                for _ in 0..1 + rng.below(2) {
                    constraints.push((x, rng.byte(), 1 + rng.below(if round % 3 == 0 { 12 } else { 5 })));
                }
            }
            let total: usize = constraints.iter().map(|&(_, _, m)| m * (m + 1) / 2).sum();
            let (list, _) = list_size(total, k - 1);
            assert_eq!(interpolate(&constraints, k - 1, list), interpolate_reference(&constraints, k - 1, list), "round {round}");
        }
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
            let found = rs_decode_listing(&read, &certain, 17, 0, LIST_COST, &mut Sightings::default());
            assert_eq!(found.len(), 1, "round {round}: {found:?}");
            let (multiplicity, solution) = &found[0];
            assert_eq!(solution.corrected, block, "round {round}");
            assert_eq!((solution.errors, solution.erasures, solution.slack), (11, 0, 0), "round {round}");
            assert!(solution.confidence_bits() >= 3 * VERIFIED_BITS, "round {round}: {}", solution.confidence_bits());
            assert!(*multiplicity >= 6, "round {round}: multiplicity {multiplicity}");
            // One more is past the algebra: whatever is found is not the
            // block, and not verified.
            read[wrong[11]].value ^= rng.nonzero();
            for (_, solution) in rs_decode_listing(&read, &certain, 17, 0, LIST_COST, &mut Sightings::default()) {
                assert_ne!(solution.corrected, block, "round {round} and one more");
                assert!(solution.confidence_bits() < 3 * VERIFIED_BITS, "round {round} and one more");
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
            let found = rs_decode_listing(&read, &certain, 30, 0, LIST_COST, &mut Sightings::default());
            assert_eq!(found.len(), 1, "40-H round {round}: {found:?}");
            assert_eq!((found[0].1.corrected == block, found[0].1.errors), (true, 19), "40-H round {round}");
            assert!(found[0].1.confidence_bits() >= 3 * VERIFIED_BITS);
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
        let found = rs_decode_listing(&read, &vec![[CERTAIN; 8]; 26], 17, 0, LIST_COST, &mut Sightings::default());
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!((found[0].1.corrected == block, found[0].1.errors, found[0].1.erasures), (true, 8, 4));
        // A pinned codeword is never overturned, and a block with its
        // every EC codeword reserved is not decoded so.
        let mut pinned = read.clone();
        pinned[picked[4]].pinned = 0xFF;
        assert!(rs_decode_listing(&pinned, &vec![[CERTAIN; 8]; 26], 17, 0, LIST_COST, &mut Sightings::default()).is_empty());
        assert!(rs_decode_listing(&read, &vec![[CERTAIN; 8]; 26], 17, 17, LIST_COST, &mut Sightings::default()).is_empty());
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
            assert!(rs_decode_listing(&read, &vec![[200u8; 8]; 33], 18, 0, LIST_COST, &mut Sightings::default()).iter().all(|(_, s)| s.corrected != block), "round {round}: reached as read");
            let found = rs_decode_listing(&read, &reliability, 18, 0, LIST_COST, &mut Sightings::default());
            assert_eq!(found.len(), 1, "round {round}: {found:?}");
            assert_eq!((found[0].1.corrected == block, found[0].1.errors), (true, 13), "round {round}");
            // Nothing but doubted bits was overturned: the solution is
            // held to the equations the rest make, thirteen short of the
            // hundred and forty-four.
            assert_eq!(found[0].1.confidence_bits(), list_solution_bits(33, 18, 0) - 13, "round {round}");
            assert_eq!(found[0].1.slack, 16, "round {round}");
            assert!(found[0].1.confidence_bits() >= 3 * VERIFIED_BITS, "round {round}: {}", found[0].1.confidence_bits());
            // Bits unknown, the codewords are read at half a byte each;
            // with two unknown bits in each of a few, the fillings are
            // points too.
            let mut read = read.clone();
            for &i in &rng.distinct(3, 33) {
                read[i].known &= 0xFC;
                read[i].value &= 0xFC;
            }
            let found = rs_decode_listing(&read, &reliability, 18, 0, LIST_COST, &mut Sightings::default());
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
                for (_, solution) in rs_decode_listing(&read, &reliability, ec_len, 0, LIST_COST, &mut Sightings::default()) {
                    assert!(solution.confidence_bits() < 3 * VERIFIED_BITS, "({n}, {ec_len}) round {round}: {solution:?}");
                }
            }
        }
        // Noise read faint throughout: four faint bits a codeword have a
        // point at their other colour, more doubted bits than a block of
        // fifteen of error correction has equations, and the curve finds
        // codewords that agree with all but a few of them, as the
        // codewords that agree with every bit not doubted, which are two
        // to the shortfall, would. Such a codeword was once confirmed by
        // a hundred and fifty bits, the shortfall left uncounted.
        for (n, ec_len) in [(70usize, 15usize), (26, 7)] {
            for round in 0..40 {
                let read: Vec<CwRead> = (0..n).map(|_| CwRead::known(rng.byte())).collect();
                let reliability: Vec<[u8; 8]> = (0..n).map(|_| std::array::from_fn(|_| rng.byte() % 48)).collect();
                for (_, solution) in rs_decode_listing(&read, &reliability, ec_len, 0, LIST_COST, &mut Sightings::default()) {
                    assert!(solution.confidence_bits() < 3 * VERIFIED_BITS, "faint ({n}, {ec_len}) round {round}: {solution:?}");
                }
            }
        }
        // A 2-L block has a codeword within six of noise once in six
        // hundred blocks: nothing so far out is verified, charged as one
        // of the solvers of the searches of a block a decode makes, those
        // that run on a block painted by hand, where one within five is.
        let needed = verifying_bits(SymbolVersion::Qr(2), &DecodeOptions::default(), false);
        assert_eq!((list_solution_bits(44, 10, 6), needed), (9, 12));
        assert!(list_solution_bits(44, 10, 5) >= needed);
        // Within half the distance the count is the plain decoder's.
        assert_eq!(list_solution_bits(33, 18, 9), super::super::rs::located_error_bits(33, 9));
        assert_eq!(list_solution_bits(26, 17, 11), 25);
        assert_eq!(list_solution_bits(45, 30, 19), 46);
    }

    /// What a solution the reach is said for must show in these tests:
    /// three codewords' worth, more than a decode of a block painted by
    /// hand asks of any of its fallbacks ([`verifying_bits`]).
    const NEEDED: u32 = 3 * VERIFIED_BITS;

    #[test]
    fn the_reach_said_is_the_reach_found() {
        // What list_decoding_reaches_past_half_the_distance finds.
        assert_eq!(hard_reach(26, 17, LIST_COST, NEEDED), 11);
        assert_eq!(hard_reach(45, 30, LIST_COST, NEEDED), 19);
        // At level L no further than half the distance, versions 2 and 5,
        // however many constraints.
        for cost in [LIST_COST, LIST_COST_MAX] {
            assert!(hard_reach(44, 10, cost, NEEDED) <= 5, "{cost}");
            assert!(hard_reach(134, 26, cost, NEEDED) <= 13, "{cost}");
        }
        assert_eq!(hard_reach(26, 17, 0, NEEDED), 0);
        // The most constraints reach as far as the default in the blocks
        // of low rate, which use no more than their list lets them, and a
        // block of high rate uses every one.
        assert_eq!(hard_reach(26, 17, LIST_COST_MAX, NEEDED), 11);
        assert_eq!(hard_reach(45, 30, LIST_COST_MAX, NEEDED), 19);
        assert!(usable_cost(45, 30, LIST_COST_MAX) < LIST_COST_MAX / 2);
        assert_eq!(usable_cost(148, 30, LIST_COST_MAX), LIST_COST_MAX);
        assert_eq!(usable_cost(45, 30, LIST_COST), LIST_COST);
        // And as far as is said, the decoder goes, in a 5-Q and a 10-H
        // block too, the wrong codewords wherever they fall.
        let mut rng = Rng(0x2EAC);
        for (n, ec_len) in [(33, 18), (43, 28)] {
            let reach = hard_reach(n, ec_len, LIST_COST, NEEDED);
            assert!(reach > ec_len / 2, "{n}, {ec_len}: {reach}");
            for round in 0..5 {
                let block = codeword(&mut rng, n - ec_len, ec_len);
                let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
                for &i in &rng.distinct(reach, n) {
                    read[i].value ^= rng.nonzero();
                }
                let found = rs_decode_listing(&read, &vec![[CERTAIN; 8]; n], ec_len, 0, LIST_COST, &mut Sightings::default());
                assert!(found.iter().any(|(_, s)| s.corrected == block), "{n}, {ec_len} round {round}: {found:?}");
            }
        }
        // A 40-H block wrong in nineteen, at the most constraints: held to
        // its list, the curve once reached fifteen there.
        let block = codeword(&mut rng, 15, 30);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        for &i in &rng.distinct(19, 45) {
            read[i].value ^= rng.nonzero();
        }
        let found = rs_decode_listing(&read, &vec![[CERTAIN; 8]; 45], 30, 0, LIST_COST_MAX, &mut Sightings::default());
        assert!(found.iter().any(|(_, s)| s.corrected == block), "{found:?}");
    }

    #[test]
    fn codewords_read_alike_are_given_one_multiplicity_and_reach_as_far_wherever_the_errors_fall() {
        // An M4-M block, five codewords its half of the distance, and a
        // 3-M block, thirteen: given the default's constraints a unit at
        // a time, their codewords had twelve or thirteen and five or six,
        // the units past the last multiplicity all could have going to
        // the codewords first in the block, and a curve sure of five and
        // thirteen. Wrong in those, a codeword more was past it, where
        // wrong elsewhere it was mostly not. Given one multiplicity alike,
        // seven and five, they are sure of six and fourteen, wherever the
        // wrong codewords fall, the first ones among them, at a third and
        // a half of the constraints.
        let mut rng = Rng(0xE7E1);
        for (n, ec_len, reach, multiplicity) in [(24, 10, 6, 7), (70, 26, 14, 5)] {
            let w = n - ec_len - 1;
            let certain = vec![[CERTAIN; 8]; n];
            let m = multiplicities(&points(&vec![CwRead::known(0); n], &certain), usable_cost(n, ec_len, LIST_COST), w);
            assert!(m.iter().all(|&m| m == multiplicity), "({n}, {ec_len}): {m:?}");
            assert_eq!(hard_reach(n, ec_len, LIST_COST, 0), reach, "({n}, {ec_len})");
            for first in [true, false] {
                let block = codeword(&mut rng, n - ec_len, ec_len);
                let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
                let wrong = if first { (0..reach).collect() } else { rng.distinct(reach, n) };
                for &i in &wrong {
                    read[i].value ^= rng.nonzero();
                }
                let found = rs_decode_listing(&read, &certain, ec_len, 0, LIST_COST, &mut Sightings::default());
                assert!(found.iter().any(|(_, s)| s.corrected == block), "({n}, {ec_len}) wrong in {wrong:?}: {found:?}");
            }
        }
        // Read faintly, the codewords' likelihoods differ, and the units
        // go by them as Koetter and Vardy's greedy gives them.
        let mut read = vec![CwRead::known(0); 24];
        read[3].value = 0x10;
        let mut bits = vec![[CERTAIN; 8]; 24];
        bits[3][4] = 20;
        let m = multiplicities(&points(&read, &bits), usable_cost(24, 10, LIST_COST), 13);
        assert!(m.iter().min() != m.iter().max(), "{m:?}");
    }

    #[test]
    fn codewords_read_alike_keep_the_greedys_units_where_one_multiplicity_reaches_no_further() {
        let mut rng = Rng(0x7A11);
        let decodes = |rng: &mut Rng, n: usize, ec_len: usize, cost: usize, wrong: &[usize]| {
            let block = codeword(rng, n - ec_len, ec_len);
            let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
            for &i in wrong {
                read[i].value ^= rng.nonzero();
            }
            let found = rs_decode_listing(&read, &vec![[CERTAIN; 8]; n], ec_len, 0, cost, &mut Sightings::default());
            found.iter().any(|(_, s)| s.corrected == block)
        };
        let alike = |n: usize, ec_len: usize, cost: usize| {
            let m = multiplicities(&points(&vec![CwRead::known(0); n], &vec![[CERTAIN; 8]; n]), usable_cost(n, ec_len, cost), n - ec_len - 1);
            m.iter().min() == m.iter().max()
        };
        // A 7-M block at a thousand and twenty-four constraints: a
        // multiplicity of six on all forty-nine codewords is just past
        // them, and any less alike is sure of nine, half the distance.
        // The greedy's, six on all but one, are sure of ten, wherever the
        // wrong codewords fall.
        let (n, ec_len, cost) = (49, 18, 1024);
        assert!(!alike(n, ec_len, cost));
        assert_eq!(hard_reach(n, ec_len, cost, 0), 10);
        for _ in 0..4 {
            let wrong = rng.distinct(10, n);
            assert!(decodes(&mut rng, n, ec_len, cost, &wrong), "7-M wrong in {wrong:?}");
        }
        // A 1-H block at five hundred and twelve: alike or a unit at a
        // time, sure of ten. The greedy's units past the last
        // multiplicity all could have went to the first codewords, and
        // eleven wrong past them are reached as well.
        let (n, ec_len, cost) = (26, 17, 512);
        assert!(!alike(n, ec_len, cost));
        assert_eq!(hard_reach(n, ec_len, cost, 0), 10);
        assert!(decodes(&mut rng, n, ec_len, cost, &(n - 11..n).collect::<Vec<_>>()));
        // A block of high rate is sure of half its distance alike only at
        // a multiplicity of one, which adds nothing to the plain decode,
        // and keeps the greedy's.
        assert!(!alike(148, 30, LIST_COST));
    }

    #[test]
    fn the_constraints_used_stop_where_the_list_first_holds_the_curve_back() {
        // Where the list holds the curve back comes and goes before it
        // holds for good, and the constraints used were once halved over
        // it: a block given more past it used more constraints or fewer.
        // They are now the same however many more are given.
        for (n, ec_len) in [(26, 17), (34, 18)] {
            let w = n - ec_len - 1;
            let held = |cost: usize| list_size_within(cost, w, LIST_SIZE_MAX + 1).1 < list_size(cost, w).1;
            let used = usable_cost(n, ec_len, LIST_COST_MAX);
            assert!(held(used + 1) && !(0..=used).any(held), "{n}, {ec_len}: {used}");
            assert!((used + 2..used + 1000).any(|cost| !held(cost)), "{n}, {ec_len}: it comes and goes past {used}");
            for cost in [used + 1, used + 2, used + 500, 20_000, LIST_COST_MAX] {
                assert_eq!(usable_cost(n, ec_len, cost), used, "{n}, {ec_len} given {cost}");
            }
            assert_eq!(usable_cost(n, ec_len, used - 1), used - 1);
        }
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
        let found = rs_decode_listing(&read, &vec![[CERTAIN; 8]; 148], 30, 0, LIST_COST, &mut Sightings::default());
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

