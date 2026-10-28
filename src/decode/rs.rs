//! Reed-Solomon over GF(256) as used by QR, Micro QR and rMQR: generator
//! roots α^0..α^(r-1), first codeword is the highest-degree coefficient.
//!
//! Seven solvers are provided. [`rs_decode`] is the classical
//! errors-and-erasures decoder, where any codeword with an unknown bit is an
//! erasure. [`solve_bit_erasures`] treats each unknown *bit* as an unknown
//! over GF(2), which is far cheaper when damage is scattered, but assumes
//! every known bit is correct. [`rs_decode_doubting`] goes the other way:
//! it assumes the least reliable codewords may be wrong and erases them
//! until the block solves, and [`solve_bit_erasures_doubting`] does the
//! same one bit at a time, which spends one equation per doubted bit
//! instead of eight, and, when no count of doubted bits will do, flips a
//! bit or two or three of those read solidly wrong, found by what the
//! equations are left wanting rather than by trying every pattern.
//! [`rs_decode_chasing`] spends nothing: it flips the least reliable
//! bits outright, every way, and decodes each result. The last two ask
//! nothing of reliability, and so reach a grid painted by hand, whose
//! every module is certain: [`rs_decode_trying`] erases codewords by
//! trial, every set of the smallest size that goes past the code and
//! then of the next, and reaches a block wrong in one codeword more than
//! the code corrects, or two in a small block, and
//! [`solve_bit_erasures_trying`] does the same over GF(2), for a block
//! whose scattered unknown bits would solve but for a wrong codeword or
//! two or three among those read.
//!
//! The solvers that try many ways are list decoders: where their tries
//! reach codewords the block as read lies as near to as each other, they
//! return every one ([`nearest`]), for the caller to tell apart by what
//! the block's data makes of the message, where one alone would have to
//! give the block up. An eighth solver, [`super::list::rs_decode_listing`],
//! is a list decoder proper: it interpolates a curve through the block's
//! codewords weighed by their reliability and reads every codeword of
//! the code off it, which reaches further than any trial of small sets.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;
use std::sync::LazyLock;

use super::gf256::{div, exp, exp_of_sum, log, mul, poly_degree, poly_eval, poly_mul};
use crate::symbol::grid::CERTAIN;
use crate::work;

/// Confirming bits a solution needs before it counts as verified, once
/// every try that found it is charged for: wrong no oftener than once
/// in 256. That is what the decode the standard describes of a block
/// read whole shows with one codeword of slack, or with errors located
/// that as few blocks of noise would show ([`located_error_bits`]), and
/// such a decode never shows less within the standard's limit: the
/// reserve speaks where there is one, and the errors located at the
/// limit where there is none. Every other solution is held to the same
/// once what found it is counted: the tries of its solver
/// ([`RsSolution::among`]), the decodes of the grid it is one of
/// ([`super::pipeline::DecodeOptions::tries`]), and the searches of its
/// block and the solvers of each search ([`super::pipeline`]).
pub const VERIFIED_BITS: u32 = 8;

/// Spare equations the bit-level solvers keep a solution to at the
/// least, before the charge for their tries: [`VERIFIED_BITS`], an
/// equation to a bit.
const SPARE_EQUATIONS: usize = VERIFIED_BITS as usize;

/// How much nearer the block as read, in bits weighed by reliability
/// ([`soft_distance`]), the solution a solver that tries many ways keeps
/// must lie than any other it found that names another codeword, for it
/// to stand alone. Chasing and the trials of erasures are list decoders:
/// several of their tries may each reach a codeword that verifies, and
/// where they reach different ones the nearer to what was read is the
/// likelier, since the bits it overturns are the fainter. A bit
/// overturned costs the likelihood its reading carries, in eighths of a
/// bit ([`bit_cost`]): two hundred and sixty-three for a certain bit,
/// nothing for one read on its threshold. So the margin is a likelihood
/// ratio of two hundred and fifty-six to one between the two codewords,
/// as the format words are told apart ([`super::bch::WEIGHT_MARGIN`]):
/// one certain bit, or one bit read well inside its colour against one
/// read near the threshold, settles it; two faint bits of middling
/// margin do not, and both codewords are returned for the message to
/// tell apart.
pub const DISTANCE_MARGIN: u32 = 64;

/// Sets of codewords a solver erases by trial, at the most, over every
/// size of set it tries: every codeword alone, every pair, which fit for
/// any block, the largest having two hundred and thirty-eight codewords,
/// in Model 1, and twenty-eight thousand pairs, every triple up to
/// seventy-three codewords and every four up to thirty. Each set is a
/// decode of a few thousand field operations, and each is a fresh
/// chance for a block of noise to solve, which what the solution
/// confirms is charged for ([`RsSolution::among`]): sixteen bits for
/// all of them, and a bit or two more for the share of them each size of
/// set is given ([`rs_decode_trying`]), which is why the count stops
/// here.
pub const TRIAL_SETS: usize = 1 << 16;

/// Fewest sets a solver may be given to erase by trial
/// ([`Trials::sets`]): the pairs of the largest block of Model 2, Micro
/// QR and rMQR, a hundred and fifty-three codewords, so that one
/// codeword past the code is reached in any block however many tries
/// the decode is one of. Model 1's larger blocks, of up to two hundred
/// and thirty-eight codewords, are given their pairs whatever the
/// budget ([`rs_decode_trying`]).
pub const TRIAL_SETS_MIN: usize = 12 * 1024;

/// Most sets a solver may be given to erase by trial, where the budget
/// is raised by hand ([`super::pipeline::DecodeOptions::trial_sets`]):
/// a million decodes, which is seconds on a block, and the triples of a
/// block of a hundred and eighty codewords.
pub const TRIAL_SETS_MAX: usize = 1 << 20;

/// What the bit-level solvers may try beyond doubting bits by their
/// reliability, and how far.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Trials {
    /// Flip a bit or two or three read solidly, found by what the
    /// equations are left wanting ([`BitSystem::flipping`]).
    pub flips: bool,
    /// Free a codeword or two or three by trial, whatever its
    /// reliability ([`BitSystem::trying`]).
    pub free: bool,
    /// Sets of codewords the frees may try, over every size, at the
    /// most: [`TRIAL_SETS`] or less.
    pub sets: usize,
}

impl Trials {
    /// Neither flips nor frees.
    pub const NONE: Self = Self { flips: false, free: false, sets: TRIAL_SETS };
    /// Both, as far as [`TRIAL_SETS`] goes.
    pub const ALL: Self = Self { flips: true, free: true, sets: TRIAL_SETS };
}

/// Known bits a flip is looked for among in threes
/// ([`BitSystem::flipping`]), at the most: the least reliable ones,
/// which is every bit of a block of sixty-four codewords. Every single
/// and every pair of a block's known bits is looked at, by what the
/// equations are left wanting, which is one lookup a bit for the pairs;
/// the triples are a lookup a pair, and the pairs of a block of a
/// hundred and fifty codewords are seven hundred thousand. Fewer bits
/// where the decode has fewer sets to spend ([`Trials::sets`]): as many
/// as have no more pairs than four lookups a set, which is every bit of
/// a block of sixty-four codewords at the full budget and of twenty-seven
/// at the least ([`TRIAL_SETS_MIN`]).
pub const FLIP_TRIPLE_BITS: usize = 512;

/// How many sets of `size` among `n` there are, when no more than
/// `limit`.
pub fn sets_of(n: usize, size: usize, limit: usize) -> Option<usize> {
    if size > n {
        return Some(0);
    }
    let mut count: u128 = 1;
    for k in 0..size {
        count = count * (n - k) as u128 / (k + 1) as u128;
    }
    usize::try_from(count).ok().filter(|&count| count <= limit)
}

/// Every set of `size` indices below `n`, in order, to `f`, until `f`
/// says to stop, every so often asking whether the work was called off.
fn each_set(n: usize, size: usize, mut f: impl FnMut(&[usize])) {
    if size > n {
        return;
    }
    let mut set: Vec<usize> = (0..size).collect();
    let mut tried = 0usize;
    loop {
        f(&set);
        tried += 1;
        if tried % 256 == 0 && work::cancelled() {
            return;
        }
        // The rightmost index that can still move up, and those after it
        // in a row behind it.
        let Some(k) = (0..size).rev().find(|&k| set[k] < n - size + k) else { return };
        set[k] += 1;
        for j in k + 1..size {
            set[j] = set[j - 1] + 1;
        }
    }
}

/// What a solution picked among `tries` is charged, in bits: one for
/// each doubling of them, since each try is a fresh chance for a block
/// of noise to pass.
pub fn tries_charge(tries: usize) -> u32 {
    tries.max(1).next_power_of_two().trailing_zeros()
}

/// What overturning a bit read at `reliability` costs, in eighths of a
/// bit of likelihood: the log of the odds that it was read right, on
/// the sampler's scale ([`crate::imaging::sample::reliability_bits`]),
/// a bit of likelihood at nought and a bit more for every eight. That
/// is nothing for a bit read on its threshold, a coin's toss that says
/// nothing either way; thirteen for one read a bit of likelihood inside
/// its colour; eight more than the reliability for one read well; and
/// two hundred and sixty-three for a certain bit. Summed over the bits
/// two codewords overturn, the difference is the likelihood ratio
/// between them, in the same eighths. A bit's reliability and one, as
/// the cost once was, gave a coin's toss an eighth of a bit of weight,
/// and sixty-four of them, as a retry that keeps the unsure modules at
/// the colour they fell on has, told two codewords apart by a byte that
/// nothing had read.
pub fn bit_cost(reliability: u8) -> u32 {
    static COSTS: LazyLock<[u32; 256]> = LazyLock::new(|| {
        std::array::from_fn(|reliability| {
            let bits = reliability as f64 / 8.0 + 1.0;
            (8.0 * (bits.exp2() - 1.0).log2()).round() as u32
        })
    });
    COSTS[usize::from(reliability)]
}

/// How far a solution lies from the block as read: the bits read as
/// known that it overturns, each at what overturning it costs
/// ([`bit_cost`]), so that a certain bit costs two hundred and
/// sixty-three and a bit read on its threshold nothing. Two hundred and
/// sixty-three times the Hamming distance, bit for bit, where nothing
/// was sampled.
pub fn soft_distance(block: &[CwRead], bit_reliability: &[[u8; 8]], corrected: &[u8]) -> u32 {
    block
        .iter()
        .zip(bit_reliability)
        .zip(corrected)
        .map(|((cw, reliability), &c)| {
            let overturned = (cw.value ^ c) & cw.known;
            (0..8).filter(|&b| overturned >> b & 1 != 0).map(|b| bit_cost(reliability[b])).sum::<u32>()
        })
        .sum()
}

/// Of the solutions a solver that tries many ways found, each with its
/// [`soft_distance`] from the block as read, the nearest first, and
/// after it every one that names another codeword and lies within
/// [`DISTANCE_MARGIN`] of it: the rivals the tries cannot tell apart,
/// which the caller may tell apart by what each makes of the message
/// ([`super::pipeline::decode`]), or leave the block unsolved over. Of
/// those naming the same codeword the one `better` prefers stands for
/// them; those lying further off than the margin are dismissed. Empty
/// when nothing was found, and one solution where it stands clear.
pub(super) fn nearest<T>(mut found: Vec<(u32, T)>, same: impl Fn(&T, &T) -> bool, better: impl Fn(&T, &T) -> bool) -> Vec<T> {
    found.sort_by_key(|&(distance, _)| distance);
    let mut kept: Vec<(u32, T)> = Vec::new();
    for (distance, candidate) in found {
        if let Some((_, held)) = kept.iter_mut().find(|(_, held)| same(held, &candidate)) {
            if better(&candidate, held) {
                *held = candidate;
            }
        } else if kept.first().is_none_or(|&(nearest, _)| distance < nearest + DISTANCE_MARGIN) {
            kept.push((distance, candidate));
        }
    }
    kept.into_iter().map(|(_, candidate)| candidate).collect()
}

/// What the agreement of doubted bits with a solution is worth, in
/// bits: how rarely so many of them agree with a wrong solution, under
/// which each doubted bit agrees by chance, a coin's toss. The tail of
/// the binomial, to the bit: for `agreeing` or more of the `agreeing +
/// flipped` to fall right by chance. A faint bit that turns out right
/// confirms the solution, but a run of doubted bits agrees with any
/// solution half the time, and half of them agreeing is worth nothing.
/// Three standard deviations short of the agreement stood for this
/// once, which falls short of the tail by a byte over a few dozen bits
/// and overshoots it by as much over a few hundred.
pub fn doubted_agreement(agreeing: u32, flipped: u32) -> u32 {
    let n = agreeing + flipped;
    if 2 * agreeing <= n {
        return 0;
    }
    // The sum of C(n, j) for j from `agreeing` to n, in logarithms: each
    // term from the one above it, the last and largest the scale.
    let mut log_term = 0.0f64;
    let mut terms = Vec::with_capacity((n - agreeing + 1) as usize);
    terms.push(log_term);
    for j in (agreeing..n).rev() {
        log_term += (f64::from(j + 1) / f64::from(n - j)).log2();
        terms.push(log_term);
    }
    let total = log_term + terms.iter().map(|t| (t - log_term).exp2()).sum::<f64>().log2();
    // Where the sum is a power of two the tail is a whole number of
    // bits: every bit right, a bare majority of an odd count, or six of
    // seven. The sum in logarithms then lands a hair either side of the
    // whole number, and a hair short of it the floor lost a bit; so a
    // hair is allowed, far wider than the rounding and far narrower than
    // any gap that is no rounding.
    (f64::from(n) - total + 1e-9).max(0.0).floor() as u32
}

/// What the errors a solution located confirm of it, in bits, for a
/// solution that corrects `errors` wrong codewords among the `read` it
/// did not take as erased.
///
/// Filling an erasure confirms nothing: an erased codeword is given
/// whatever value fits, so a solution that spends all its redundancy on
/// erasures is one any block would have had. Locating an error is another
/// matter. Each error spends two codewords of redundancy, and a solution
/// with `l` of them exists only where the codewords read lie within `l`
/// of a codeword of the code. About each codeword there are
/// `C(read, l) · 255^l` such readings, and none of them lies as near
/// another, since a decoder corrects no more than half of what tells two
/// codewords apart; against the `256^(2l)` readings the redundancy spent
/// tells apart, a block of noise, or one damaged past correcting, shows
/// such a solution once in `256^(2l) / (C(read, l) · 255^l)`, which is
/// near enough `256^l / C(read, l)`. McEliece and Swanson
/// bound the chance that a Reed-Solomon decoder corrects to a wrong
/// codeword so, which for a code of full length is the one in `l!` they
/// are quoted for. The solutions of fewer errors, which noise shows as
/// well, are counted in: at most a 254th part more.
///
/// It is little for one error and a great deal for many. One error among
/// the 26 codewords of a version 1 symbol is found in a tenth of all
/// blocks of noise, and confirms three bits; the nine a block of version
/// 5-Q corrects, among its 33, are found in one in 10^14. ZXing and
/// BoofCV take any solution their decoders reach, and the standard asks
/// more only of the smallest symbols, where it reserves codewords against
/// exactly this ([`RsSolution::within_standard`]).
pub fn located_error_bits(read: usize, errors: usize) -> u32 {
    // Past half the codewords read the fewer errors are no longer the
    // lesser part, and no code corrects so many.
    if errors == 0 || 2 * errors > read + 1 {
        return 0;
    }
    let ways: f64 = (1..=errors).map(|i| ((read - errors + i) as f64 / i as f64).log2()).sum();
    let bits = errors as f64 * (16.0 - 255f64.log2()) - ways - (255f64 / 254.0).log2();
    bits.max(0.0).floor() as u32
}

/// What confirms a codeword of the code found within `errors` of the
/// `read` codewords of a block, whatever found it, in bits: how rarely
/// a block of noise has any codeword so near. The `redundancy` left
/// after the erasures tells `256^redundancy` readings apart, and about
/// each codeword `C(read, errors) · 255^errors` readings lie within
/// `errors` of it, so noise shows one once in the quotient. Within half
/// the code's distance this is a byte for each codeword of slack and
/// [`located_error_bits`] besides, to the bit; past it, where there is
/// no slack and the decoders that stop at half the distance never go,
/// it falls by eight bits a codeword less what the codewords' ways
/// give back, and reaches nought where any block would have a codeword
/// within reach ([`super::list`]).
pub fn list_solution_bits(read: usize, redundancy: usize, errors: usize) -> u32 {
    list_solution_balance(read, redundancy, errors).max(0) as u32
}

/// [`list_solution_bits`] before it is held at nought: below nought
/// where a block of noise has codewords so near by the score, by the
/// bits of how many. What else confirms such a codeword, the known bits
/// of the codewords it filled in or the doubted bits it agrees with, is
/// weighed against those, since for each of the many codewords noise
/// has within reach the rest of the block is a fresh chance to agree:
/// counted from nought, it was counted whole. A codeword of belief
/// propagation fifteen codewords from a block of noise with two of
/// redundancy, where noise has some two to the hundred such, was
/// confirmed so by the thirty known bits of the five codewords it
/// filled in, and verified.
pub fn list_solution_balance(read: usize, redundancy: usize, errors: usize) -> i64 {
    let errors = errors.min(read);
    let ways: f64 = (1..=errors).map(|i| ((read - errors + i) as f64 / i as f64).log2()).sum();
    let bits = 8.0 * redundancy as f64 - errors as f64 * 255f64.log2() - ways - (255f64 / 254.0).log2();
    bits.floor() as i64
}

/// Furthest the soft count ([`soft_count_bits`]) looks, in
/// reliability-weighted bits: a solution lying further from the block
/// as read than this is confirmed by the hard count alone. Fifteen
/// certain bits, or three hundred read a bit of likelihood inside their
/// colour, which is more than any block has; bits read on their
/// threshold cost nothing and never reach it.
pub const SOFT_COUNT_REACH: u32 = 4096;

/// What confirms a codeword found within `distance` of the block as
/// read, in bits weighed by their reliability ([`soft_distance`]), in
/// bits: how rarely a block of noise has any codeword so near. The
/// code is linear over GF(2) with `redundancy_bits` equations once the
/// unknown bits are spent, so one binary word in `2^redundancy_bits` is
/// a codeword, and about each word there are as many words within
/// `distance` as there are sets of its known bits whose costs
/// ([`bit_cost`]) sum to `distance` or less: counted by dynamic
/// programming over the costs, which are small integers. A bit that
/// costs nothing, read on its threshold, is in a set or out of it
/// alike, and doubles the count: it takes a bit from what confirms the
/// solution whichever way it was read. So noise shows a
/// codeword within `distance` once in the quotient, and a solution
/// that overturns only faint bits is rare in noise however many
/// codewords it changes, where the hard count ([`list_solution_bits`])
/// charges it a byte's worth for each. Pinned bits are never
/// overturned and are not counted. Nought beyond
/// [`SOFT_COUNT_REACH`], where the hard count serves.
///
/// The count asks nothing of the reliabilities but that noise be noise:
/// the sets are what they are whatever the reliabilities mean, and a
/// block of noise lies within `distance` of some codeword as often as
/// the count says. Reliabilities that overstate how well the modules
/// were read make the right codeword lie far off, and go unconfirmed;
/// they never make a wrong one lie near.
pub fn soft_count_bits(block: &[CwRead], bit_reliability: &[[u8; 8]], redundancy_bits: usize, distance: u32) -> u32 {
    soft_count_reaching(block, bit_reliability, redundancy_bits, distance, 0).unwrap_or(0)
}

/// [`soft_count_bits`] where it comes to `needed` or more, and nought
/// where it falls short, which is found as soon as it shows
/// ([`soft_count_reaching`]): for a caller that asks only whether a
/// codeword far from the block as read is confirmed.
pub(super) fn soft_count_short_of(block: &[CwRead], bit_reliability: &[[u8; 8]], redundancy_bits: usize, distance: u32, needed: u32) -> u32 {
    soft_count_reaching(block, bit_reliability, redundancy_bits, distance, needed.max(1)).unwrap_or(0)
}

/// Cost of a bit, in eighths of a bit, that the rough count of
/// [`soft_count_reaching`] rounds every cost up to a multiple of.
const ROUGH_COST: usize = 8;

/// Bits taken into a count of sets between each look at whether it has
/// grown past what the count is asked to come to ([`soft_count_reaching`]).
const LOOK_EVERY: usize = 16;

/// [`soft_count_bits`] where it comes to `needed` or more, and `None`
/// where it falls short, as soon as that shows: the sets within reach
/// only grow as the bits are taken in, so a count already too large to
/// leave `needed` stays so. The sets are counted first with every cost
/// rounded up to a multiple of [`ROUGH_COST`], and the reach down: a set
/// within the rough reach is within the reach, so the rough count is
/// never the more, a few hundred counts where the reach has thousands,
/// and a codeword far from the block as read, which noise lies within
/// reach of many codewords of, is refused on it. Nothing is refused
/// that the whole count would have offered: what it says is lowered by
/// a hair against the rounding, a millionth of a bit where the counts
/// are wrong by a millionth of that. `needed` of nought asks for the
/// whole count, every time.
fn soft_count_reaching(block: &[CwRead], bit_reliability: &[[u8; 8]], redundancy_bits: usize, distance: u32, needed: u32) -> Option<u32> {
    if distance > SOFT_COUNT_REACH {
        return (needed == 0).then_some(0);
    }
    let reach = distance as usize;
    let choice = (255f64 / 254.0).log2();
    let short = |volume: f64| needed > 0 && redundancy_bits as f64 - volume - choice < f64::from(needed) - 1e-6;
    let costs = || {
        block
            .iter()
            .zip(bit_reliability)
            .flat_map(|(cw, reliability)| {
                (0..8).filter(move |&b| cw.known >> b & 1 != 0 && cw.pinned >> b & 1 == 0).map(move |b| bit_cost(reliability[b]) as usize)
            })
            .filter(move |&cost| cost <= reach)
    };
    if needed > 0 {
        let rough = sets_within(costs().map(|cost| cost.div_ceil(ROUGH_COST)), reach / ROUGH_COST, &short)?;
        if short(rough) {
            return None;
        }
    }
    let volume = sets_within(costs(), reach, &short)?;
    Some((redundancy_bits as f64 - volume - (255f64 / 254.0).log2()).max(0.0).floor() as u32)
}

/// The log of how many sets of bits costing `costs` cost `reach` or
/// less, by dynamic programming over the costs ([`soft_count_bits`]);
/// `None` once a look at the count, every [`LOOK_EVERY`] bits, finds it
/// `short`.
fn sets_within(costs: impl Iterator<Item = usize>, reach: usize, short: &impl Fn(f64) -> bool) -> Option<f64> {
    // ways[d]: how many sets of the bits seen so far cost exactly d,
    // scaled down by 2^scaled whenever they grow large.
    let mut ways = vec![0f64; reach + 1];
    ways[0] = 1.0;
    let mut scaled = 0i64;
    // No count is larger than this: a bit at most doubles the largest. The
    // largest itself is looked for only once this passes the scale, and
    // the counts are scaled exactly where they were when it was looked for
    // after every bit, without a comparison an addition to hold the
    // additions back.
    let mut bound = 1f64;
    let volume = |ways: &[f64], scaled: i64| ways.iter().sum::<f64>().log2() + scaled as f64;
    for (taken, cost) in costs.enumerate() {
        if cost == 0 {
            // In every set or out of it for nothing: every count
            // doubles.
            for w in &mut ways {
                *w *= 2.0;
            }
        } else {
            // `ways[d] += ways[d - cost]` from the top down, which
            // reads only counts not yet added to: so done a stretch
            // of `cost` at a time, each stretch from the one below
            // it, which it does not overlap.
            let mut end = reach + 1;
            while end > cost {
                let start = (end - cost).max(cost);
                let (below, from) = ways.split_at_mut(start);
                for (w, &add) in from[..end - start].iter_mut().zip(&below[start - cost..end - cost]) {
                    *w += add;
                }
                end = start;
            }
        }
        bound *= 2.0;
        if bound > 2f64.powi(500) {
            bound = ways.iter().copied().fold(0f64, f64::max);
            if bound > 2f64.powi(500) {
                for w in &mut ways {
                    *w *= 2f64.powi(-500);
                }
                scaled += 500;
                bound *= 2f64.powi(-500);
            }
        }
        if taken % LOOK_EVERY == LOOK_EVERY - 1 && short(volume(&ways, scaled)) {
            return None;
        }
    }
    Some(volume(&ways, scaled))
}

/// Whether a solver that tries many ways offers a solution it found:
/// whether, charged for the tries it was found among, it would be
/// verified standing alone ([`VERIFIED_BITS`]), by what its solver
/// counts for it or by the soft count of the bits it overturns
/// ([`soft_count_bits`]), charged the same and a bit for the choice
/// between the two counts. What it does not offer it keeps back, so
/// that a block of noise, which such a solver reaches some codeword of
/// every so often, does not end its search at the first or crowd the
/// block's solutions with codewords nothing bears out. The solvers were
/// once each held to a figure of their own, two codewords to spare for
/// the chase and sixteen bits for the rest, counted before their tries
/// by some and after them by others and by the codewords' count alone,
/// and a solution the soft count would have verified was kept back.
///
/// The soft count of each codeword is worked out once, however many of
/// the solver's tries reach it. Every codeword it is asked about, offered
/// or kept back, is noted in the solver's [`Sightings`].
pub(super) struct Offering<'a> {
    block: &'a [CwRead],
    bit_reliability: &'a [[u8; 8]],
    redundancy_bits: usize,
    counted: HashMap<Vec<u8>, Counted>,
    seen: &'a mut Sightings,
}

/// The soft count of a codeword an [`Offering`] was asked about: the
/// whole of it, or that it fell short of a figure, where the count
/// stopped as soon as it did ([`soft_count_reaching`]).
#[derive(Clone, Copy)]
enum Counted {
    Exactly(u32),
    ShortOf(u32),
}

impl<'a> Offering<'a> {
    pub(super) fn new(block: &'a [CwRead], bit_reliability: &'a [[u8; 8]], ec_len: usize, seen: &'a mut Sightings) -> Self {
        let unknown: usize = block.iter().map(|c| c.known.count_zeros() as usize).sum();
        Self { block, bit_reliability, redundancy_bits: (8 * ec_len).saturating_sub(unknown), counted: HashMap::new(), seen }
    }

    /// The solver's sightings, for a codeword it reaches that it never
    /// asks to have offered.
    pub(super) fn seen(&mut self) -> &mut Sightings {
        self.seen
    }

    /// Whether `solution` is offered, confirmed as it counts itself after
    /// the charge for its tries ([`RsSolution::charge`]).
    pub(super) fn offers_solution(&mut self, solution: &RsSolution) -> bool {
        self.offers(&solution.corrected, solution.confidence_bits(), solution.charge)
    }

    /// Whether `doubt` is offered, confirmed as it counts itself after
    /// the charge for its tries ([`BitDoubt::tries`]).
    fn offers_doubt(&mut self, doubt: &BitDoubt) -> bool {
        self.offers(&doubt.solution.corrected, doubt.confidence_bits(), tries_charge(doubt.tries))
    }

    /// Whether `corrected`, confirmed by `confidence` bits as its solver
    /// counts them after `charge` for its tries, is offered.
    fn offers(&mut self, corrected: &[u8], confidence: u32, charge: u32) -> bool {
        self.seen.see(corrected);
        if confidence >= VERIFIED_BITS {
            return true;
        }
        let needed = charge + 1 + VERIFIED_BITS;
        let soft = match self.counted.get(corrected) {
            Some(&Counted::Exactly(soft)) => Some(soft),
            Some(&Counted::ShortOf(short_of)) if needed >= short_of => None,
            _ => {
                let distance = soft_distance(self.block, self.bit_reliability, corrected);
                let soft = soft_count_reaching(self.block, self.bit_reliability, self.redundancy_bits, distance, needed);
                self.counted.insert(corrected.to_vec(), soft.map_or(Counted::ShortOf(needed), Counted::Exactly));
                soft
            }
        };
        soft.is_some_and(|soft| soft.saturating_sub(charge + 1) >= VERIFIED_BITS)
    }
}

/// Every codeword of the code a solver that tries many ways reached,
/// whether it offered it ([`Offering`]) or kept it back, whether it
/// returned it or dismissed it for lying far from the nearest
/// ([`nearest`]), and whether the decode that reached it refused it for
/// the known bits it overturns with nothing to confirm it
/// ([`RsFailure::KnownBitConflict`]): for the caller to hold the
/// solutions it is offered against ([`super::pipeline`]). Of two
/// codewords the block as read lies near, the nearer is the likelier by
/// the margin between them, whatever else confirms either, and a
/// codeword kept back for having little to confirm it, dismissed for
/// lying further off, or refused for having nothing, is a codeword for
/// all that. The codewords a solver kept back or dismissed were once
/// forgotten, and after them those a decode refused, and a solution
/// stood on what confirmed it alone over a codeword nearer the block as
/// read that its own solver had reached and said nothing of. A codeword
/// that overturns a pin is never one the block can be, and is not noted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Sightings(BTreeSet<Vec<u8>>);

impl Sightings {
    /// Note that `codeword` was reached.
    pub(super) fn see(&mut self, codeword: &[u8]) {
        if !self.0.contains(codeword) {
            self.0.insert(codeword.to_vec());
        }
    }

    /// Note that `codeword` was reached, unless it overturns a pin of
    /// `block`, which no codeword the block can be does.
    pub(super) fn see_unless_pinned(&mut self, block: &[CwRead], codeword: &[u8]) {
        if block.iter().zip(codeword).all(|(cw, &c)| (cw.value ^ c) & cw.pinned == 0) {
            self.see(codeword);
        }
    }

    /// Note every codeword `other` noted.
    pub(super) fn see_all(&mut self, other: &Sightings) {
        for codeword in other.codewords() {
            self.see(codeword);
        }
    }

    /// Whether `codeword` was reached.
    pub(super) fn saw(&self, codeword: &[u8]) -> bool {
        self.0.contains(codeword)
    }

    /// The codewords reached, each once.
    pub fn codewords(&self) -> impl Iterator<Item = &[u8]> {
        self.0.iter().map(Vec::as_slice)
    }
}

/// One codeword as read from the symbol.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CwRead {
    /// Bits as read; unknown bits are zero.
    pub value: u8,
    /// Mask of bits that came from known modules or pins.
    pub known: u8,
    /// Mask of bits asserted by a pin. Always a subset of `known`. A solution
    /// that would change one of these is rejected.
    pub pinned: u8,
}

impl CwRead {
    pub const fn known(value: u8) -> Self {
        Self { value, known: 0xFF, pinned: 0 }
    }

    pub const fn unknown() -> Self {
        Self { value: 0, known: 0, pinned: 0 }
    }

    pub const fn is_erasure(&self) -> bool {
        self.known != 0xFF
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Erratum {
    /// Index within the block.
    pub index: usize,
    /// XOR applied to the value as read.
    pub magnitude: u8,
    pub was_erasure: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RsSolution {
    pub corrected: Vec<u8>,
    pub errata: Vec<Erratum>,
    /// Codewords that were fully read but wrong.
    pub errors: usize,
    pub erasures: usize,
    /// Unused redundancy in codewords: `ec_len - erasures - 2 * errors`.
    pub slack: usize,
    /// What the known bits inside erased codewords confirm of the solution:
    /// how rarely so many of them would agree with it by chance
    /// ([`doubted_agreement`]), less what the codewords' count falls
    /// short of nothing by ([`list_solution_balance`]), and nothing when
    /// that comes to less than nothing ([`settle`]).
    pub check_bits: u32,
    /// Known bits inside erased codewords that the solution overturned,
    /// those of the doubted codewords among them.
    pub flipped_known_bits: u32,
    /// What the errors the decoder located confirm of the solution
    /// ([`located_error_bits`]); or, for a solution judged against the
    /// block ([`judged`]), what the count that confirms it has beyond
    /// the slack's bytes, less a bit for the choice between the two
    /// counts ([`less_the_choice`]).
    pub located_bits: u32,
    /// What the tries of a solver that made many to find it cost
    /// ([`Self::among`]): a bit for each doubling of them, taken from
    /// everything that confirms the solution ([`Self::confidence_bits`]).
    pub charge: u32,
    /// EC codewords the standard reserves against misdecodes for this
    /// block: a solution that leaves fewer than this many unused has spent
    /// capacity the standard says is for detection only.
    pub reserved: usize,
    /// The solution leaves the reserved codewords unused as the standard
    /// counts them, against the block as read ([`keeps_reserve`]).
    pub reserve_kept: bool,
}

impl RsSolution {
    /// Rough count of independent bits confirming the solution: a byte
    /// for each codeword of redundancy left unused, the known bits of
    /// erased codewords it reproduced, and what the errors it located
    /// confirm, less the charge for the tries of a solver that made many
    /// to find it ([`Self::among`]). A solution that only filled
    /// erasures, with nothing to spare, has none, and any input would
    /// have "solved" so; one that located errors with nothing to spare
    /// has what they confirm, which at the limit of a block's error
    /// correction is a great deal.
    pub fn confidence_bits(&self) -> u32 {
        self.uncharged_bits().saturating_sub(self.charge)
    }

    /// What confirms the solution before the charge for its tries.
    pub fn uncharged_bits(&self) -> u32 {
        8 * self.slack as u32 + self.check_bits + self.located_bits
    }

    /// The solution as the pick of `tries` a solver made: charged a bit
    /// for each doubling of them ([`tries_charge`]), since each try is a
    /// fresh chance for a block of noise to pass whatever the solution
    /// shows, its slack and the known bits it reproduced as much as the
    /// errors it located. The charge was once taken from the located
    /// errors' bits alone, and a solution that located none, as one
    /// whose trial erasures were just the wrong codewords does, went
    /// uncharged: the pick of sixty thousand sets showed the sixteen
    /// bits of two codewords spare as the first try would have.
    pub fn among(mut self, tries: usize) -> Self {
        self.charge += tries_charge(tries);
        self
    }

    /// The solution stays inside the correction capacity the standard
    /// allows, leaving the reserved codewords unused as the standard
    /// counts them against the block as read, whatever found it
    /// ([`keeps_reserve`]). It was once the slack of the decode that found
    /// the solution, which a solver could leave unspent by what it did to
    /// the block before decoding: the trials erasing wrong codewords,
    /// which then cost a unit of redundancy each where the standard's
    /// errors cost two, the chase flipping bits, which cost nothing, the
    /// curve leaving uncounted the codewords it changed only in bits it
    /// had doubted, and a solution judged against the block taking its
    /// slack from the soft count. A 2-L symbol five codewords wrong, past
    /// the four the standard corrects, was so read and confirmed, through
    /// the propagation's first round.
    pub fn within_standard(&self) -> bool {
        self.reserve_kept
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RsFailure {
    TooManyErasures { erasures: usize, capacity: usize },
    /// The error locator needs more redundancy than is left after erasures.
    BudgetExceeded,
    LocatorDegree,
    RootCount { want: usize, got: usize },
    ZeroDerivative,
    ResidualSyndrome,
    /// The only available solution contradicts a pinned bit.
    PinConflict { index: usize },
    /// The solution contradicts bits that were read as known, and nothing
    /// confirms it: no slack, no errors located that noise seldom shows,
    /// and too few of the known bits of the codewords it erased agreeing
    /// with it to be more than chance. A block of noise would have done
    /// as well, and the bits read say it is not the block.
    KnownBitConflict { index: usize },
}

impl std::fmt::Display for RsFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManyErasures { erasures, capacity } => {
                write!(f, "{erasures} unknown codewords, capacity is {capacity}")
            }
            Self::BudgetExceeded | Self::LocatorDegree | Self::RootCount { .. } => {
                write!(f, "too many wrong codewords")
            }
            Self::ZeroDerivative | Self::ResidualSyndrome => write!(f, "no consistent solution"),
            Self::PinConflict { index } => write!(f, "solution contradicts pin at codeword {index}"),
            Self::KnownBitConflict { index } => {
                write!(f, "solution contradicts known bits at codeword {index}")
            }
        }
    }
}

/// Generator polynomial with roots α^0..α^(ec_len-1), little-endian.
pub fn rs_generator(ec_len: usize) -> Vec<u8> {
    (0..ec_len).fold(vec![1u8], |g, i| poly_mul(&g, &[exp(i), 1]))
}

/// Error-correction codewords for `data`.
pub fn rs_encode(data: &[u8], ec_len: usize) -> Vec<u8> {
    let generator = rs_generator(ec_len);
    let mut rem = vec![0u8; ec_len];
    for &d in data {
        let factor = d ^ rem[0];
        rem.rotate_left(1);
        rem[ec_len - 1] = 0;
        for (i, r) in rem.iter_mut().enumerate() {
            // generator[ec_len] is the leading 1, so skip it.
            *r ^= mul(generator[ec_len - 1 - i], factor);
        }
    }
    rem
}

/// S_j = block(α^j) for j in 0..ec_len.
pub fn rs_syndromes(block: &[u8], ec_len: usize) -> Vec<u8> {
    (0..ec_len)
        .map(|j| {
            let x = exp(j);
            block.iter().fold(0, |acc, &c| mul(acc, x) ^ c)
        })
        .collect()
}

/// Errors-and-erasures decode of one block (data followed by EC codewords).
pub fn rs_decode(block: &[CwRead], ec_len: usize) -> Result<RsSolution, RsFailure> {
    rs_decode_reserving(block, ec_len, 0)
}

/// The same, noting `reserved` EC codewords that the standard keeps for
/// detection (ISO/IEC 18004 Table 9). The decode goes as far as the code
/// allows either way; the solution reports whether it stayed within the
/// standard's limit, and callers decide what to make of one that did not.
pub fn rs_decode_reserving(block: &[CwRead], ec_len: usize, reserved: usize) -> Result<RsSolution, RsFailure> {
    rs_decode_reserving_noting(block, ec_len, reserved, &mut Sightings::default())
}

/// [`rs_decode_reserving`], noting in `seen` the codeword of a solution
/// refused for the known bits it overturns ([`Sightings`]); one it
/// returns is the caller's to note.
pub fn rs_decode_reserving_noting(block: &[CwRead], ec_len: usize, reserved: usize, seen: &mut Sightings) -> Result<RsSolution, RsFailure> {
    rs_decode_doubted(block, &[], ec_len, reserved, seen)
}

/// The same, with the codewords at `doubted` indices treated as erasures
/// although they were read in full: their bits are allowed to be wrong,
/// but the bits the solution agrees with still count as checks, as the
/// known bits of any other erasure do. Erasing them outright would throw
/// that confirmation away: a doubted codeword that was right in seven
/// bits of eight confirms the solution by those seven. Only the agreement
/// beyond the disagreement counts, though, since a codeword that was
/// wrong in half its bits agrees with any solution as often as not. The
/// codeword of a solution refused for the known bits it overturns is
/// noted in `seen` ([`settle`]).
fn rs_decode_doubted(
    block: &[CwRead],
    doubted: &[usize],
    ec_len: usize,
    reserved: usize,
    seen: &mut Sightings,
) -> Result<RsSolution, RsFailure> {
    let received: Vec<u8> = block.iter().map(|c| c.value & c.known).collect();
    let syndromes = rs_syndromes(&received, ec_len);
    rs_decode_core(block, doubted, ec_len, reserved, &syndromes, seen)
}

/// What bit `1 << bit` of codeword `i` of a block of `n` adds to each of
/// its `ec_len` syndromes: the syndromes of a block with that bit
/// flipped are those of the block as read with this added, since the
/// syndromes are linear in the block.
fn syndrome_of_bit(n: usize, i: usize, bit: u8, ec_len: usize) -> Vec<u8> {
    (0..ec_len).map(|j| mul(1 << bit, exp((n - 1 - i) * j))).collect()
}

/// [`rs_decode_doubted`], given the syndromes of the block as read, its
/// unknown bits as zeros. Computing them is most of a decode, and a
/// solver that decodes the block many ways over, a codeword or two
/// erased or a bit flipped each time, has them once for all its tries:
/// the erasures by trial change nothing of them, and a flipped bit adds
/// [`syndrome_of_bit`]. The codeword of a solution refused for the known
/// bits it overturns is noted in `seen` ([`settle`]).
pub(super) fn rs_decode_core(
    block: &[CwRead],
    doubted: &[usize],
    ec_len: usize,
    reserved: usize,
    syndromes: &[u8],
    seen: &mut Sightings,
) -> Result<RsSolution, RsFailure> {
    let n = block.len();
    assert!(n <= 255 && ec_len <= n, "block of {n} with {ec_len} EC codewords");
    debug_assert_eq!(syndromes.len(), ec_len);
    let is_doubted = |i: usize| doubted.contains(&i);
    let is_erased = |i: usize| block[i].is_erasure() || is_doubted(i);
    let received: Vec<u8> = block.iter().map(|c| c.value & c.known).collect();
    let erased: Vec<usize> = (0..n).filter(|&i| is_erased(i)).collect();
    let f = erased.len();
    if f > ec_len {
        return Err(RsFailure::TooManyErasures { erasures: f, capacity: ec_len });
    }
    // Position i carries x^(n-1-i).
    let locator = |i: usize| exp(n - 1 - i);
    let locator_inv = |i: usize| exp(255 - (n - 1 - i) % 255);

    let mut gamma = vec![1u8];
    for &i in &erased {
        gamma = poly_mul(&gamma, &[1, locator(i)]);
    }

    // Forney syndromes: the part of S·Γ that erasures cannot explain.
    let forney: Vec<u8> = (f..ec_len)
        .map(|k| (0..=f).fold(0, |acc, m| acc ^ mul(gamma[m], syndromes[k - m])))
        .collect();

    let (lambda, l) = berlekamp_massey(&forney);
    if 2 * l > ec_len - f {
        return Err(RsFailure::BudgetExceeded);
    }
    if poly_degree(&lambda) != l {
        return Err(RsFailure::LocatorDegree);
    }

    let mut error_positions = Vec::with_capacity(l);
    if l > 0 {
        for i in 0..n {
            if poly_eval(&lambda, locator_inv(i)) == 0 {
                if is_erased(i) {
                    return Err(RsFailure::RootCount { want: l, got: error_positions.len() });
                }
                error_positions.push(i);
            }
        }
        if error_positions.len() != l {
            return Err(RsFailure::RootCount { want: l, got: error_positions.len() });
        }
    }

    let psi = poly_mul(&lambda, &gamma);
    let mut omega = poly_mul(syndromes, &psi);
    omega.truncate(ec_len);
    // Formal derivative in characteristic 2: only odd-degree terms survive.
    let psi_prime: Vec<u8> =
        (0..psi.len().saturating_sub(1)).map(|k| if k % 2 == 0 { psi[k + 1] } else { 0 }).collect();

    let mut corrected = received.clone();
    for &i in erased.iter().chain(&error_positions) {
        let x_inv = locator_inv(i);
        let denom = poly_eval(&psi_prime, x_inv);
        if denom == 0 {
            return Err(RsFailure::ZeroDerivative);
        }
        corrected[i] ^= mul(locator(i), div(poly_eval(&omega, x_inv), denom));
    }

    // The syndromes of the corrected block are, by linearity, those of
    // the block as read with the errata's added: zero, or the values
    // found make no codeword of the code.
    for (j, &s) in syndromes.iter().enumerate() {
        let residual = erased
            .iter()
            .chain(&error_positions)
            .fold(s, |acc, &i| acc ^ mul(corrected[i] ^ received[i], exp((n - 1 - i) * j)));
        if residual != 0 {
            return Err(RsFailure::ResidualSyndrome);
        }
    }

    let slack = ec_len - f - 2 * l;
    settle(block, ec_len, doubted, &erased, corrected, l, slack, i64::from(located_error_bits(n - f, l)), reserved, seen)
}

/// Whether `corrected` leaves the `reserved` error correction codewords
/// the standard keeps for detection unused in `block` (ISO/IEC 18004
/// Table 9), counting what it spends of the `ec_len` the way the
/// standard's decode does, eight equations to a codeword: two codewords'
/// worth for each codeword it changes in a bit that was read, and for
/// each it changes in no bit read, an equation for each of its bits that
/// could not be read. A codeword doubted for its faint modules, by a
/// solver or by the propagation's beliefs, is a codeword read, and wrong
/// where the solution changes it: the reserve is there to detect a
/// misread, and a misread module is read as faintly as any. A module that
/// could not be read is an erasure, which takes an equation to solve for
/// and no more, as solving the block bit by bit does; where every module
/// of a codeword is unread that is the unit of the standard's own count.
/// Within the standard's limit so, a solution changes no more codewords
/// read in full than the plain decode of the block corrects with the
/// reserve unused; what the other solvers bring a block that reserves
/// codewords is what confirms such a codeword, the reach of solving bit
/// by bit over modules unread, and a codeword past the limit, which never
/// is confirmed. Nothing is reserved in most symbols, and there every
/// solution keeps the reserve.
pub fn keeps_reserve(block: &[CwRead], ec_len: usize, reserved: usize, corrected: &[u8]) -> bool {
    if reserved == 0 {
        return true;
    }
    let spent: usize = block
        .iter()
        .zip(corrected)
        .map(|(cw, &c)| if (cw.value ^ c) & cw.known != 0 { 16 } else { cw.known.count_zeros() as usize })
        .sum();
    spent + 8 * reserved <= 8 * ec_len
}

/// Whether every codeword that keeps the reserve against `read`, the
/// block as the reserve counts it ([`keeps_reserve`]), lies within half
/// the code's distance of `block`, the block a solver searches, as the
/// plain decode reads it: so where `block` reads as `read` does, and
/// every codeword of it is read whole or not at all. Such a codeword
/// spends two codewords for each it changes and one for each unread, as
/// the plain decode counts them, and the reserve unused leaves it within
/// that decode's reach, where no other codeword lies. Where a codeword is
/// read in part, its unread modules cost an eighth of a codeword each,
/// where the plain decode erases the whole of it, and a codeword that
/// keeps the reserve may lie past that decode's reach: a 1-Q block, one
/// of its thirteen codewords of error correction reserved, six codewords
/// a module short and four read wrong spends seventy of the ninety-six
/// eighths it may, and the plain decode would need fourteen codewords of
/// its thirteen. So too where `block` reads what `read` leaves unread,
/// the modules read at a coin's toss or the message's evidence laid over
/// it.
pub(super) fn plain_reach_holds_the_reserve(block: &[CwRead], read: &[CwRead]) -> bool {
    block.len() == read.len()
        && block.iter().zip(read).all(|(b, r)| b.known == r.known && (b.value ^ r.value) & r.known == 0 && (r.known == 0xFF || r.known == 0))
}

/// A solution's accounts, from the block it corrected: whether it may
/// stand, and what confirms it. `erased` are the codewords taken as
/// erasures, those read with unknown bits and the `doubted` among them,
/// read in full and taken as erased all the same; `errors` how many
/// codewords read in full the solution changed besides; `slack` the
/// redundancy it left unused, in codewords; and `located` what the
/// errors it located confirm ([`located_error_bits`], or
/// [`list_solution_balance`] less the slack's byte each for a solution
/// reached past half the code's distance), which is less than nothing
/// where noise has many codewords as near.
///
/// The bits the solution was free to rewrite, the known bits of the
/// codewords it erased and every bit of those it doubted, are checks on
/// it: a right solution reproduces nearly all of them, and a wrong one,
/// fitted to whatever redundancy the erasures left, half. What they
/// confirm is how rarely so many would agree by chance
/// ([`doubted_agreement`]), one count over all of them, so that the half
/// a wrong solution agrees with by chance count for nothing and the
/// bits it overturns weigh against the rest; and from that is taken
/// what `located` falls short of nothing by, since each of the codewords
/// noise has within reach is a fresh chance for them to agree. The bits
/// once each counted one for and two against, whatever the odds, which
/// let a solution agreeing with nine in ten of a hundred bits claim
/// seventy where chance gives such agreement once in two to the
/// fifty-five, and refused one overturning a single bit of fifty with
/// nothing to spare, which chance gives once in two to the forty-four,
/// however much its errors located confirmed it besides.
///
/// A solution that overturns a pinned bit is refused, and so is one that
/// overturns a bit read as known with nothing at all to confirm it,
/// slack, errors located or the checks: noise would have done as well,
/// and the bits read say it is not the block. The second kind is a
/// codeword the block might be all the same, and is noted in `seen`
/// ([`Sightings`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn settle(
    block: &[CwRead],
    ec_len: usize,
    doubted: &[usize],
    erased: &[usize],
    corrected: Vec<u8>,
    errors: usize,
    slack: usize,
    located: i64,
    reserved: usize,
    seen: &mut Sightings,
) -> Result<RsSolution, RsFailure> {
    let n = block.len();
    let is_erased = |i: usize| erased.contains(&i) || doubted.contains(&i);
    let received: Vec<u8> = block.iter().map(|c| c.value & c.known).collect();
    let (mut agreeing, mut flipped_known_bits) = (0u32, 0u32);
    let mut first_overturned = None;
    for (i, cw) in block.iter().enumerate() {
        let diff = corrected[i] ^ cw.value;
        if diff & cw.pinned != 0 {
            return Err(RsFailure::PinConflict { index: i });
        }
        if is_erased(i) {
            let flipped = (diff & cw.known).count_ones();
            flipped_known_bits += flipped;
            agreeing += cw.known.count_ones() - flipped;
            if flipped > 0 && first_overturned.is_none() {
                first_overturned = Some(i);
            }
        }
    }
    let check_bits = (i64::from(doubted_agreement(agreeing, flipped_known_bits)) + located.min(0)).max(0) as u32;
    let located_bits = located.max(0) as u32;
    if let Some(index) = first_overturned
        && 8 * slack as u32 + check_bits + located_bits == 0
    {
        seen.see(&corrected);
        return Err(RsFailure::KnownBitConflict { index });
    }

    let errata = (0..n)
        .filter(|&i| is_erased(i) || corrected[i] != received[i])
        .map(|i| Erratum { index: i, magnitude: corrected[i] ^ received[i], was_erasure: is_erased(i) })
        .collect();

    let reserve_kept = keeps_reserve(block, ec_len, reserved, &corrected);
    Ok(RsSolution { corrected, errata, errors, erasures: erased.len(), slack, check_bits, flipped_known_bits, located_bits, charge: 0, reserved, reserve_kept })
}

/// A codeword of the code found by searching something other than
/// `block`, the word the beliefs made ([`super::abp`]) or the block with
/// what the message makes likely laid over it
/// ([`super::evidence::Laid`]), judged against `block` alone: how rarely
/// a block of noise has a codeword as near, counted codeword by codeword
/// or bit by bit, whichever counts the more, less one for the choice
/// between them and `tries_cost`, what the decodes made to find it cost
/// ([`tries_charge`]). The tries' cost is the charge the solution
/// carries ([`RsSolution::charge`]), and the choice's bit comes off what
/// confirms it ([`less_the_choice`]): a caller that weighs the solution
/// against its soft count once more, charged for the tries it carries
/// and a bit for that choice ([`super::pipeline`]), then charges the
/// choice once. Carried in the charge, the bit was charged there again,
/// and the propagation's solutions stood a bit below the chase's under
/// a guided search. What it searched from says nothing here, since the
/// search carries its word toward some codeword whatever the block
/// holds.
///
/// Codeword by codeword is as a decode of errors and erasures has it
/// ([`list_solution_bits`], [`settle`]): the redundancy left after the
/// codewords that could not be read, the errors located among those
/// that were, and the known bits of the erased codewords it reproduces,
/// which are checks only while the codewords read in full fix the
/// erased ones, no more of them erased than the block has error
/// correction codewords. Past that the erased codewords' known bits are
/// what the codeword was solved from, and bit by bit is the one count.
/// Bit by bit is the soft count ([`soft_count_bits`]): every known bit
/// at its reliability, against the equations the unknown bits leave,
/// among which the known bits of the erased codewords already stand. So
/// the two are never added: a search that solved bit by bit used every
/// known bit to find its codeword, and its agreeing with them is what
/// the soft count measures, once. The propagation's solutions were once
/// counted so, the larger of the two counts and then the erased
/// codewords' known bits on top of it, and a block with many codewords
/// partly read was shown verified at a risk far below what the search
/// had established. Nor is the codewords' count held at nought before
/// the known bits are weighed: a codeword many codewords from the block
/// as read is one of the many that noise has as near, and the known
/// bits of the erased codewords agreeing with it are what one of those
/// many does by chance ([`list_solution_balance`]). And a codeword of
/// noise more codewords from the block, of which noise has more, is
/// confirmed as much where more of those known bits agree with it: each
/// count of errors they could carry as far is a fresh chance for noise,
/// and the codewords' count is charged for them
/// ([`error_counts_reaching`]). It was once held to its own count of
/// errors alone, and a block with a few codewords partly read was
/// confirmed some bits beyond what noise gives it. A bit by bit
/// count's slack is the codewords' worth of it. Whether the solution
/// keeps the codewords the standard reserves is what it changes of the
/// block as read ([`keeps_reserve`]), whichever count confirms it: the
/// slack once stood for that too, and a 2-L block five codewords wrong,
/// confirmed by the soft count, stood within the standard's four. `None`
/// for a codeword
/// that overturns a pin, or that the counts, charged, would not verify
/// standing alone, which a solver that tries many ways keeps back
/// ([`Offering`]).
pub(super) fn judged(
    block: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    reserved: usize,
    corrected: Vec<u8>,
    tries_cost: u32,
) -> Option<RsSolution> {
    let unknown: usize = block.iter().map(|c| c.known.count_zeros() as usize).sum();
    let distance = soft_distance(block, bit_reliability, &corrected);
    let soft = soft_count_bits(block, bit_reliability, (8 * ec_len).saturating_sub(unknown), distance);
    judged_by(block, ec_len, reserved, corrected, tries_cost, soft)
}

/// [`judged`], given the soft count of the codeword against the block
/// ([`soft_count_bits`]), for a caller that has worked it out already.
pub(super) fn judged_by(block: &[CwRead], ec_len: usize, reserved: usize, corrected: Vec<u8>, tries_cost: u32, soft: u32) -> Option<RsSolution> {
    judged_count(block, ec_len, reserved, corrected, tries_cost, soft).filter(|solution| solution.confidence_bits() >= VERIFIED_BITS)
}

/// [`judged_by`] however little the counts confirm the codeword: `None`
/// only for one that overturns a pin. For a caller that weighs a
/// solution short of verified by the more of what confirms it.
pub(super) fn judged_count(block: &[CwRead], ec_len: usize, reserved: usize, corrected: Vec<u8>, tries_cost: u32, soft: u32) -> Option<RsSolution> {
    if block.iter().zip(&corrected).any(|(cw, &c)| (cw.value ^ c) & cw.pinned != 0) {
        return None;
    }
    let n = block.len();
    let erased: Vec<usize> = (0..n).filter(|&i| block[i].is_erasure()).collect();
    let (read, redundancy) = (n - erased.len(), ec_len.saturating_sub(erased.len()));
    let errors = (0..n).filter(|&i| !block[i].is_erasure() && corrected[i] != block[i].value).count();
    let known: u32 = erased.iter().map(|&i| block[i].known.count_ones()).sum();
    let flipped_known_bits: u32 = erased.iter().map(|&i| ((corrected[i] ^ block[i].value) & block[i].known).count_ones()).sum();
    // What each count is held to: a bit for the choice between them and
    // the tries' cost, and the codewords' count the other counts of
    // errors the known bits of the erased codewords could carry as far
    // as this one's ([`error_counts_reaching`]). The slack never claims
    // more than the bits confirm once charged.
    let charge = 1 + tries_cost;
    let hard = list_solution_balance(read, redundancy, errors);
    let score = hard + i64::from(doubted_agreement(known - flipped_known_bits, flipped_known_bits));
    let rivals = tries_charge(error_counts_reaching(read, redundancy, known, score));
    let slack = redundancy.saturating_sub(2 * errors).min((hard - i64::from(charge + rivals)).max(0) as usize / 8);
    let by_codeword = (erased.len() <= ec_len)
        .then(|| {
            settle(block, ec_len, &[], &erased, corrected.clone(), errors, slack, hard - 8 * slack as i64, reserved, &mut Sightings::default())
                .ok()
                .map(|solution| less_the_choice(less_by(RsSolution { charge: tries_cost, ..solution }, rivals)))
        })
        .flatten();
    let solution = match by_codeword {
        Some(solution) if solution.confidence_bits() >= soft.saturating_sub(charge) => solution,
        _ => {
            let slack = soft.saturating_sub(charge) as usize / 8;
            let reserve_kept = keeps_reserve(block, ec_len, reserved, &corrected);
            let solution = RsSolution {
                corrected,
                errata: Vec::new(),
                errors,
                erasures: erased.len(),
                slack,
                check_bits: 0,
                flipped_known_bits,
                located_bits: soft - 8 * slack as u32,
                charge: tries_cost,
                reserved,
                reserve_kept,
            };
            restated(block, ec_len, less_the_choice(solution))
        }
    };
    Some(solution)
}

/// `solution` with a bit taken from what confirms it, for the choice
/// between the two counts [`judged`] judges by: from what the count has
/// beyond the slack's bytes, or from the checks where it has nothing
/// there, which only a count of nothing leaves, the slack being held to
/// the count less the charge and that bit. The confidence is as if the
/// bit were charged, and the charge stays the tries' cost alone.
fn less_the_choice(solution: RsSolution) -> RsSolution {
    less_by(solution, 1)
}

/// `solution` with `bits` taken from what confirms it as
/// [`less_the_choice`] takes its bit: from what the count has beyond the
/// slack's bytes, and from the checks for what that falls short by.
fn less_by(mut solution: RsSolution, bits: u32) -> RsSolution {
    let beyond = bits.min(solution.located_bits);
    solution.located_bits -= beyond;
    solution.check_bits = solution.check_bits.saturating_sub(bits - beyond);
    solution
}

/// How many counts of errors among the `read` codewords a codeword of
/// noise could lie at and be confirmed by the codewords' count by
/// `score` bits or more, the `known` bits of the erased codewords
/// agreeing with it as they may ([`judged`]): a count of errors whose
/// codewords noise has once in `2^b`, `b` its [`list_solution_balance`],
/// is topped up to `score` by the known bits once in `2^(score - b)`, so
/// noise reaches `score` there once in `2^score`, and the chances of
/// every such count add. Those whose codewords reach `score` alone are
/// one between them, each a byte and more short of the one before; each
/// that the known bits must top up is one more. A block with no erased
/// codeword has one count at most, and is charged nothing.
fn error_counts_reaching(read: usize, redundancy: usize, known: u32, score: i64) -> usize {
    let balance = |errors: usize| list_solution_balance(read, redundancy, errors);
    // The balance falls with every error, so the counts that reach alone
    // come first, and none past the first the known bits fall short of.
    let alone = usize::from(balance(0) >= score);
    let topped = (0..=read).map(balance).take_while(|&b| b + i64::from(known) >= score).filter(|&b| b < score).count();
    alone + topped
}

/// Decode by doubting: erase the least reliable fully read codewords, one
/// more at a time, and keep the solution with the most confirmation,
/// among those offered ([`Offering`]), every count tried charged for: one
/// within the standard's limit when `reserved` codewords are kept for
/// detection, or, when none is, one that spends them, which the caller
/// marks as beyond the standard ([`RsSolution::within_standard`]). A
/// codeword doubted is a codeword read, and one the solution changes is
/// wrong as the standard counts errors ([`keeps_reserve`]), so that past
/// what the standard corrects the doubting reaches the block only so.
/// Returns how many codewords were doubted. Codewords at [`CERTAIN`]
/// reliability and pinned codewords are never doubted. The soft count
/// that may offer a solution takes each bit at its codeword's
/// reliability, which is all this is given.
///
/// The codeword of every count doubted that reaches one is noted in
/// `seen`, the one returned among them and those refused for the known
/// bits they overturn ([`Sightings`]).
pub fn rs_decode_doubting(
    block: &[CwRead],
    reliability: &[u8],
    ec_len: usize,
    reserved: usize,
    seen: &mut Sightings,
) -> Option<(usize, RsSolution)> {
    rs_decode_doubting_against(block, block, reliability, ec_len, reserved, seen)
}

/// [`rs_decode_doubting`], what each solution spends of the reserve
/// counted against `read` ([`counted_against`]).
pub(super) fn rs_decode_doubting_against(
    block: &[CwRead],
    read: &[CwRead],
    reliability: &[u8],
    ec_len: usize,
    reserved: usize,
    seen: &mut Sightings,
) -> Option<(usize, RsSolution)> {
    assert_eq!(block.len(), reliability.len());
    let erasures = block.iter().filter(|c| c.is_erasure()).count();
    let mut doubtful: Vec<usize> = (0..block.len())
        .filter(|&i| !block[i].is_erasure() && block[i].pinned == 0 && reliability[i] < CERTAIN)
        .collect();
    doubtful.sort_by_key(|&i| (reliability[i], i));
    doubtful.truncate(ec_len.saturating_sub(erasures));

    let bits: Vec<[u8; 8]> = reliability.iter().map(|&r| [r; 8]).collect();
    let mut offering = Offering::new(block, &bits, ec_len, seen);
    let mut best: Option<(usize, RsSolution)> = None;
    let mut beyond: Option<(usize, RsSolution)> = None;
    for k in 1..=doubtful.len() {
        if let Ok(solution) =
            rs_decode_doubted(block, &doubtful[..k], ec_len, reserved, offering.seen()).map(|s| counted_against(read, ec_len, s.among(doubtful.len())))
            && offering.offers_solution(&solution)
        {
            let held = if solution.within_standard() { &mut best } else { &mut beyond };
            if held.as_ref().is_none_or(|(_, b)| solution.confidence_bits() > b.confidence_bits()) {
                *held = Some((k, solution));
            }
        }
    }
    best.or(beyond)
}

/// Of solutions found many ways, those within the standard's limit
/// when any is ([`RsSolution::within_standard`]), and all of them
/// otherwise: a solution that spends the codewords the standard reserves
/// for detection is offered only where none that keeps them was found.
pub(super) fn preferring_the_standard<T>(found: Vec<(u32, T)>, solution: impl Fn(&T) -> &RsSolution) -> Vec<(u32, T)> {
    if found.iter().any(|(_, t)| solution(t).within_standard()) {
        found.into_iter().filter(|(_, t)| solution(t).within_standard()).collect()
    } else {
        found
    }
}

/// Bits [`rs_decode_chasing`] flips at the most: every pattern of the
/// least reliable ones is tried, so a block costs `2^CHASE_BITS` decodes
/// at most. Each is a decode from syndromes the pattern amends rather
/// than computes, a few thousand field operations, so a thousand
/// patterns are a few milliseconds for a block of a hundred and fifty
/// codewords; and each is a chance of a misdecode, which the located
/// errors are charged for ([`RsSolution::among`]): ten bits here. A
/// decode that is one of many flips fewer ([`chase_bits`]).
pub const CHASE_BITS: usize = 10;
/// Fewest bits the chase flips, however many decodes it is one of.
pub const CHASE_MIN_BITS: usize = 6;
/// Most bits the chase may be set to flip by hand
/// ([`super::pipeline::DecodeOptions::chase_bits`]): sixty-five thousand
/// patterns, which is a second or so on a large block, and sixteen bits
/// of misdecode risk for the located errors to pay for.
pub const CHASE_MAX_BITS: usize = 16;

/// How many bits the chase flips in a decode that is one of `tries`: the
/// most, less a bit for each doubling of the tries, since the patterns
/// of every try are chances of a misdecode together and a search of
/// every format has thirty-two decodes to spend them over.
pub fn chase_bits(tries: usize) -> usize {
    chase_bits_among(CHASE_BITS, tries)
}

/// [`chase_bits`] from a budget of `bits` set by hand: less a bit for
/// each doubling of the tries, and never below [`CHASE_MIN_BITS`] or
/// the budget itself, whichever is the less.
pub fn chase_bits_among(bits: usize, tries: usize) -> usize {
    let bits = bits.clamp(1, CHASE_MAX_BITS);
    bits.saturating_sub(tries_charge(tries) as usize).max(CHASE_MIN_BITS.min(bits))
}

/// Decode by flipping: take the least reliable bits of the fully read
/// codewords, at most `bits` of them, flip every non-empty subset
/// in turn and decode what results as errors and erasures. Doubting a
/// faint codeword erases it, which spends a unit of redundancy to let
/// its bits be anything; flipping spends none, so a block with a few more
/// wrong codewords than the code corrects is reached when the extra ones
/// read faintly. This is Chase's second algorithm, over the faintest bits
/// of the block rather than of each codeword.
///
/// Trying many patterns is list decoding, and a wrong pattern can land
/// on a wrong codeword that verifies, so every pattern's solution that
/// is offered ([`Offering`]) is kept, those within the standard over any
/// that spend its
/// reserve ([`preferring_the_standard`]), and where they name different
/// codewords the one nearest the block as read, in bits weighed by
/// their reliability, is kept over the rest only when it is nearer by a
/// clear margin ([`nearest`]): the patterns flip the faintest bits, and
/// of two codewords each within reach the one that overturns the fainter
/// bits is the likelier, while two as near as each other mean the
/// patterns cannot tell, and both are returned, the nearer first, for
/// the message to tell apart. Each comes with how many bits its pattern
/// flipped, and is restated against the block as read: a flipped bit it
/// kept is an erratum it found, and `errors` counts every codeword it
/// changed. The slack stays that of the flipped block's decode, since
/// the flipped bits were guessed, not located by the code, and so does
/// what the errors that decode located confirm, less the cost of the
/// patterns tried. Bits that are pinned, and bits at [`CERTAIN`], are
/// never flipped.
///
/// A block whose every EC codeword the standard reserves for detection,
/// as M1's two are, is not chased at all: flipping bits until the check
/// passes is correction by another name, and with sixteen check bits and
/// a thousand patterns to try it passes on noise once in sixty blocks.
pub fn rs_decode_chasing(
    block: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    reserved: usize,
    bits: usize,
) -> Vec<(usize, RsSolution)> {
    rs_decode_chasing_bits(block, bit_reliability, ec_len, reserved, bits, &mut Sightings::default())
        .into_iter()
        .map(|(flipped, solution)| (flipped.iter().map(|bits| bits.count_ones() as usize).sum(), solution))
        .collect()
}

/// [`rs_decode_chasing`], with the bits of each codeword each
/// solution's pattern flipped in place of their count: for showing
/// which they were. `bits` is how many bits are flipped at the most,
/// [`CHASE_BITS`] or fewer ([`chase_bits`]).
///
/// The codeword of every pattern that reaches one is noted in `seen`,
/// those returned among them and those refused for the known bits they
/// overturn ([`Sightings`]).
pub fn rs_decode_chasing_bits(
    block: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    reserved: usize,
    bits: usize,
    seen: &mut Sightings,
) -> Vec<(Vec<u8>, RsSolution)> {
    rs_decode_chasing_bits_against(block, block, bit_reliability, ec_len, reserved, bits, seen)
}

/// [`rs_decode_chasing_bits`], what each solution spends of the reserve
/// counted against `read` ([`counted_against`]).
pub(super) fn rs_decode_chasing_bits_against(
    block: &[CwRead],
    read: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    reserved: usize,
    bits: usize,
    seen: &mut Sightings,
) -> Vec<(Vec<u8>, RsSolution)> {
    assert_eq!(block.len(), bit_reliability.len());
    if reserved >= ec_len {
        return Vec::new();
    }
    let mut doubtful: Vec<(u8, usize, u8)> = block
        .iter()
        .enumerate()
        .filter(|(_, cw)| !cw.is_erasure())
        .flat_map(|(i, cw)| {
            (0..8u8).filter(move |&b| cw.pinned >> b & 1 == 0).map(move |b| (bit_reliability[i][b as usize], i, b))
        })
        .filter(|&(r, _, _)| r < CERTAIN)
        .collect();
    if doubtful.is_empty() {
        return Vec::new();
    }
    doubtful.sort();
    doubtful.truncate(bits.clamp(1, CHASE_MAX_BITS));

    let n = block.len();
    let received: Vec<u8> = block.iter().map(|c| c.value & c.known).collect();
    let base = rs_syndromes(&received, ec_len);
    let deltas: Vec<Vec<u8>> = doubtful.iter().map(|&(_, i, b)| syndrome_of_bit(n, i, b, ec_len)).collect();
    let patterns = (1usize << doubtful.len()) - 1;
    // The erasures are the same under every pattern, so their locator,
    // and what it makes of the syndromes and of each flip's addition to
    // them, is computed once; each pattern then has its Forney
    // syndromes by a few additions, and most patterns are refused on
    // those alone ([`Locator::locates`]), without the decode.
    let erased: Vec<usize> = (0..n).filter(|&i| block[i].is_erasure()).collect();
    let f = erased.len();
    if f > ec_len {
        return Vec::new();
    }
    let gamma = erased.iter().fold(vec![1u8], |gamma, &i| poly_mul(&gamma, &[1, exp(n - 1 - i)]));
    let forney_of = |syndromes: &[u8]| -> Vec<u8> { (f..ec_len).map(|k| (0..=f).fold(0, |acc, m| acc ^ mul(gamma[m], syndromes[k - m]))).collect() };
    let delta_forney: Vec<Vec<u8>> = deltas.iter().map(|d| forney_of(d)).collect();
    let budget = (ec_len - f) / 2;
    let mut offering = Offering::new(block, bit_reliability, ec_len, seen);
    let mut found: Vec<(u32, (u32, RsSolution))> = Vec::new();
    let mut trial = block.to_vec();
    let mut syndromes = base.clone();
    let mut forney = forney_of(&base);
    let mut locator = Locator::new(ec_len, n);
    // The patterns in Gray code order: each differs from the last in
    // one bit, flipped in the trial and added to the syndromes.
    let mut pattern = 0u32;
    for step in 1..=patterns as u32 {
        if step % 256 == 0 && work::cancelled() {
            break;
        }
        let k = step.trailing_zeros() as usize;
        pattern ^= 1 << k;
        let (_, i, b) = doubtful[k];
        trial[i].value ^= 1 << b;
        for (s, d) in syndromes.iter_mut().zip(&deltas[k]) {
            *s ^= d;
        }
        for (s, d) in forney.iter_mut().zip(&delta_forney[k]) {
            *s ^= d;
        }
        if !locator.locates(&forney, budget, &erased) {
            continue;
        }
        let Ok(solution) = rs_decode_core(&trial, &[], ec_len, reserved, &syndromes, offering.seen()).map(|s| s.among(patterns)) else { continue };
        // A thousand trials are a thousand chances for a wrong codeword
        // to lie within reach of the flipped block, which the solution
        // is charged for; and the known bits of the erased codewords it
        // overturns weigh against it ([`settle`]). It was once held to
        // two codewords to spare and to overturning none of those bits,
        // whatever else confirmed it.
        if !offering.offers_solution(&solution) {
            continue;
        }
        let solution = counted_against(read, ec_len, restated(block, ec_len, solution));
        found.push((soft_distance(block, bit_reliability, &solution.corrected), (pattern, solution)));
    }
    let found = preferring_the_standard(found, |(_, solution)| solution);
    nearest(found, |a, b| a.1.corrected == b.1.corrected, |a, b| a.1.confidence_bits() > b.1.confidence_bits())
        .into_iter()
        .map(|(pattern, solution)| {
            let mut flipped = vec![0u8; block.len()];
            for (k, &(_, i, b)) in doubtful.iter().enumerate() {
                if pattern >> k & 1 != 0 {
                    flipped[i] |= 1 << b;
                }
            }
            (flipped, solution)
        })
        .collect()
}

/// Berlekamp-Massey with its buffers kept between calls, for the solvers
/// that decode a block many ways over: the locator of a try is found
/// without an allocation, and the try given up as soon as the locator
/// would be longer than the code allows.
struct Locator {
    c: Vec<u8>,
    b: Vec<u8>,
    previous: Vec<u8>,
    /// The locator's nonzero coefficients past the first, with their
    /// logarithms.
    logs: Vec<(usize, usize)>,
    /// Positions of the block the locator is searched for roots over.
    n: usize,
    /// `inverse[i * (ec_len + 1) + j]` is `255 - j · (n - 1 - i) mod 255`:
    /// the logarithm of the `j`th power of the inverse of position `i`'s
    /// locator, so that the locator is evaluated at a position by a
    /// lookup a coefficient, a position's powers side by side.
    inverse: Vec<u16>,
    stride: usize,
}

impl Locator {
    fn new(ec_len: usize, n: usize) -> Self {
        let stride = ec_len + 1;
        let inverse = (0..n).flat_map(|i| (0..stride).map(move |j| (255 - (j * (n - 1 - i)) % 255) as u16)).collect();
        Self {
            c: Vec::with_capacity(ec_len + 2),
            b: Vec::with_capacity(ec_len + 2),
            previous: Vec::with_capacity(ec_len + 2),
            logs: Vec::with_capacity(ec_len + 1),
            n,
            inverse,
            stride,
        }
    }

    /// Whether the Forney syndromes `s` of a try have an error locator
    /// [`rs_decode_core`] would accept: of length at most `max_len`,
    /// of that degree, with as many roots among the `n` positions of
    /// the block and none at an `erased` position. The same test the
    /// decode makes, made cheaply: the shortest LFSR is grown as
    /// [`berlekamp_massey`] grows it, and the try given up the moment
    /// it grows past `max_len`, since it never shrinks.
    fn locates(&mut self, s: &[u8], max_len: usize, erased: &[usize]) -> bool {
        let Self { c, b, previous, logs, n, inverse, stride } = self;
        let (n, stride) = (*n, *stride);
        c.clear();
        c.push(1);
        b.clear();
        b.push(1);
        let (mut l, mut m, mut last) = (0usize, 1usize, 1u8);
        for i in 0..s.len() {
            let mut d = s[i];
            for j in 1..=l.min(c.len() - 1) {
                d ^= mul(c[j], s[i - j]);
            }
            if d == 0 {
                m += 1;
                continue;
            }
            let scale = div(d, last);
            let grows = 2 * l <= i;
            if grows {
                if i + 1 - l > max_len {
                    return false;
                }
                previous.clear();
                previous.extend_from_slice(c);
            }
            if c.len() < b.len() + m {
                c.resize(b.len() + m, 0);
            }
            for (k, &coef) in b.iter().enumerate() {
                c[k + m] ^= mul(scale, coef);
            }
            if grows {
                l = i + 1 - l;
                std::mem::swap(b, previous);
                last = d;
                m = 1;
            } else {
                m += 1;
            }
        }
        if poly_degree(c) != l {
            return false;
        }
        if l == 0 {
            return true;
        }
        // The locator at the inverse of each position's locator, which
        // is where a root lies for an error there: `c[0]`, which is one,
        // plus each coefficient times the position's locator to the
        // minus `j`, by the logarithms. Past `n - l` positions that are
        // no root, the locator cannot have its `l`.
        logs.clear();
        logs.extend((1..=l).filter(|&j| c[j] != 0).map(|j| (j, log(c[j]))));
        let (mut roots, mut misses) = (0, 0);
        for i in 0..n {
            let at = &inverse[i * stride..(i + 1) * stride];
            let mut value = c[0];
            for &(j, lg) in logs.iter() {
                value ^= exp_of_sum(lg + usize::from(at[j]));
            }
            if value == 0 {
                if erased.binary_search(&i).is_ok() {
                    return false;
                }
                roots += 1;
            } else {
                misses += 1;
                if misses > n - l {
                    return false;
                }
            }
        }
        roots == l
    }
}

/// A solution of a block some of whose codewords were flipped or erased
/// for the try, restated against the block as read: an erratum for every
/// codeword the solution changed from what was read and for every one
/// that could not be read, `errors` counting the former, `erasures` the
/// latter, and whether it keeps the standard's reserve so counted
/// ([`keeps_reserve`]). The slack stays that of the try's decode.
fn restated(block: &[CwRead], ec_len: usize, mut solution: RsSolution) -> RsSolution {
    let received: Vec<u8> = block.iter().map(|c| c.value & c.known).collect();
    solution.errata = (0..block.len())
        .filter(|&i| block[i].is_erasure() || solution.corrected[i] != received[i])
        .map(|i| Erratum { index: i, magnitude: solution.corrected[i] ^ received[i], was_erasure: block[i].is_erasure() })
        .collect();
    solution.errors = solution.errata.iter().filter(|e| !e.was_erasure).count();
    solution.erasures = solution.errata.len() - solution.errors;
    solution.reserve_kept = keeps_reserve(block, ec_len, solution.reserved, &solution.corrected);
    solution
}

/// `solution` with whether it keeps the standard's reserve counted
/// against `read`, the block as read as the reserve counts it
/// ([`keeps_reserve`]), where the block a solver searched is another:
/// one with the message's evidence laid over it, or with its modules
/// sampled right on the threshold read at their lean, which the reserve
/// counts as unread ([`super::pipeline`]). Each solver prefers a
/// solution that keeps the reserve ([`preferring_the_standard`]), and
/// counted against the block searched it could keep one the block as
/// read says spends the reserve, which is never confirmed, over one the
/// block as read lets stand.
pub(super) fn counted_against(read: &[CwRead], ec_len: usize, mut solution: RsSolution) -> RsSolution {
    solution.reserve_kept = keeps_reserve(read, ec_len, solution.reserved, &solution.corrected);
    solution
}

/// Decode by erasing codewords by trial: codewords read in full are
/// taken as unknown besides those that are, a few at a time, and the
/// block decoded as errors and erasures with them so. A codeword erased
/// spends one unit of redundancy where a wrong one the code must locate
/// spends two, so a block wrong in more codewords than the code corrects
/// is reached when the wrong ones are among those erased: with `r` units
/// of redundancy left after the erasures the block has, erasing `e`
/// reaches `⌊(r - e) / 2⌋ + e` wrong codewords, one more than the code
/// corrects at `e = 2` when `r` is even and at `e = 1` when it is odd,
/// and one more again with two erased more. The sets of the smallest
/// size that goes past the code are tried first, every one of them, and
/// only when none solves the block the sets two larger, as far as
/// [`TRIAL_SETS`] lets the sizes go: pairs for any block, which reaches
/// one codeword past the code however large the block, and triples and
/// fours for the smaller blocks, which reach two. This is as far as
/// small sets go of the list decoding Guruswami and Sudan do by
/// interpolation, and it asks nothing of reliability: it is for the grid
/// painted by hand, whose every module is certain and which nothing that
/// doubts or flips by reliability can touch. ZXing, zxing-cpp, quirc and
/// ZBar correct to half the code's distance and no further.
///
/// Many tries are many chances for a wrong codeword to lie within reach,
/// so what a solution confirms is charged for the sets tried
/// ([`RsSolution::among`]): each size of set the budget lets the trials
/// try is given an equal share of the chances, and each set of it a
/// share of that, so that the chances of every size together come to
/// one decode's. A solution was once charged for the sets of the sizes
/// tried so far, which over two sizes gave the trials the chances of two
/// decodes, and over three of three. A solution is
/// kept only where it is offered ([`Offering`]), and where the
/// sets reach different codewords the one nearest the block as read is
/// kept by a clear margin, or every one within the margin is returned,
/// nearest first ([`nearest`]). A size whose every solution spends the
/// codewords the standard reserves for detection does not end the
/// search: the sizes after it may reach the block within the standard,
/// and its solutions are returned only when none of them does
/// ([`preferring_the_standard`]). Where the block searched reads as the
/// block as read does, every codeword of it read whole or not at all, a
/// codeword that keeps the reserve is within what the plain decode
/// corrects, and is its own, which the trials come after
/// ([`plain_reach_holds_the_reserve`]): the sizes after the first to
/// reach anything have nothing to bring, and are not tried. They once
/// were, every size the budget let them, to return the first size's
/// solutions all the same; and then they were not wherever the block as
/// read was so, though the block searched read what it left unread,
/// and a codeword that kept the reserve lay past the plain decode of the
/// block searched, at a size after the first. Each
/// comes with the codewords erased, restated against the block as read
/// as a chased one is. A codeword pinned whole is never erased. `sets`
/// is how many sets may be tried in all, [`TRIAL_SETS`] at the most: a
/// decode that is one of many has less to spend on each
/// ([`super::pipeline::DecodeOptions::trial_sets`]). Every set of the
/// smallest size is tried however few that is, so that one codeword
/// past the code is reached in any block: the pairs of Model 1's largest
/// blocks are more than [`TRIAL_SETS_MIN`].
///
/// The codeword of every set that reaches one is noted in `seen`, those
/// returned among them and those refused for the known bits they
/// overturn ([`Sightings`]).
pub fn rs_decode_trying(
    block: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    reserved: usize,
    sets: usize,
    seen: &mut Sightings,
) -> Vec<(Vec<usize>, RsSolution)> {
    rs_decode_trying_against(block, block, bit_reliability, ec_len, reserved, sets, seen)
}

/// [`rs_decode_trying`], what each solution spends of the reserve
/// counted against `read` ([`counted_against`]): a size of set whose
/// solutions all spend it so is no end to the trials.
pub(super) fn rs_decode_trying_against(
    block: &[CwRead],
    read: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    reserved: usize,
    sets: usize,
    seen: &mut Sightings,
) -> Vec<(Vec<usize>, RsSolution)> {
    assert_eq!(block.len(), bit_reliability.len());
    let erasures = block.iter().filter(|c| c.is_erasure()).count();
    let left = ec_len.saturating_sub(erasures);
    let candidates: Vec<usize> = (0..block.len()).filter(|&i| !block[i].is_erasure() && block[i].pinned != 0xFF).collect();
    let received: Vec<u8> = block.iter().map(|c| c.value & c.known).collect();
    let syndromes = rs_syndromes(&received, ec_len);
    let n = block.len();
    let read_erased: Vec<usize> = (0..n).filter(|&i| block[i].is_erasure()).collect();
    let read_gamma = read_erased.iter().fold(vec![1u8], |gamma, &i| poly_mul(&gamma, &[1, exp(n - 1 - i)]));
    let mut locator = Locator::new(ec_len, n);
    let mut offering = Offering::new(block, bit_reliability, ec_len, seen);
    let mut beyond: Vec<(Vec<usize>, RsSolution)> = Vec::new();
    // No size of set past the first to reach anything reaches a codeword
    // within the standard that the plain decode does not.
    let within_plain_reach = plain_reach_holds_the_reserve(block, read);
    let first = if left % 2 == 1 { 1 } else { 2 };
    let budget = sets.max(sets_of(candidates.len(), first, TRIAL_SETS_MAX).unwrap_or(0)).min(TRIAL_SETS_MAX);
    // The sizes of set the budget lets the trials try, each with how
    // many sets it has: the smallest that goes past the code, and each
    // two larger while the sets of every size so far fit.
    let mut sizes: Vec<(usize, usize)> = Vec::new();
    let (mut size, mut tried) = (first, 0);
    while size < left {
        let Some(count) = sets_of(candidates.len(), size, budget).filter(|&count| count > 0 && tried + count <= budget) else { break };
        sizes.push((size, count));
        tried += count;
        size += 2;
    }
    let shares = sizes.len();
    for &(size, count) in &sizes {
        let tries = shares * count;
        let mut found: Vec<(u32, (Vec<usize>, RsSolution))> = Vec::new();
        let (mut gamma, mut forney, mut all_erased) = (Vec::with_capacity(ec_len + 1), Vec::with_capacity(ec_len), Vec::with_capacity(ec_len));
        each_set(candidates.len(), size, |set| {
            // The set's Forney syndromes, and the cheap test of them
            // ([`Locator::locates`]) before the decode, which most sets
            // fail: the erasure locator of the block as read, times
            // `(1 + X x)` for each codeword of the set, in place.
            gamma.clear();
            gamma.extend_from_slice(&read_gamma);
            for &k in set {
                let x = exp(n - 1 - candidates[k]);
                gamma.push(0);
                for m in (1..gamma.len()).rev() {
                    gamma[m] ^= mul(x, gamma[m - 1]);
                }
            }
            let f = gamma.len() - 1;
            forney.clear();
            forney.extend((f..ec_len).map(|k| (0..=f).fold(0, |acc, m| acc ^ mul(gamma[m], syndromes[k - m]))));
            all_erased.clear();
            all_erased.extend(read_erased.iter().copied());
            all_erased.extend(set.iter().map(|&k| candidates[k]));
            all_erased.sort_unstable();
            if !locator.locates(&forney, (ec_len - f) / 2, &all_erased) {
                return;
            }
            let erased: Vec<usize> = set.iter().map(|&k| candidates[k]).collect();
            let Ok(solution) = rs_decode_core(block, &erased, ec_len, reserved, &syndromes, offering.seen()).map(|s| s.among(tries)) else { return };
            if !offering.offers_solution(&solution) {
                return;
            }
            let solution = counted_against(read, ec_len, restated(block, ec_len, solution));
            found.push((soft_distance(block, bit_reliability, &solution.corrected), (erased, solution)));
        });
        let within = found.iter().any(|(_, (_, solution))| solution.within_standard());
        let found = preferring_the_standard(found, |(_, solution)| solution);
        let kept = nearest(found, |a, b| a.1.corrected == b.1.corrected, |a, b| a.1.confidence_bits() > b.1.confidence_bits());
        if (within || within_plain_reach) && !kept.is_empty() {
            return kept;
        }
        if beyond.is_empty() {
            beyond = kept;
        }
        if work::cancelled() {
            break;
        }
    }
    beyond
}

/// How far the trials of erasures ([`rs_decode_trying`]) go in a block
/// of `n` codewords read whole with `ec_len` of error correction, given
/// `sets` to try: a size of set at a time, as they do, while every set
/// of the sizes so far fits, and the smallest size whatever `sets` is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrialReach {
    /// Sizes of set tried in full, the smallest first: each reaches a
    /// codeword more past half the code's distance than the last.
    pub past: usize,
    /// The smallest size tried: one where the error correction is odd
    /// and two where it is even, each size after it two more.
    pub first_size: usize,
    /// Sets tried in all.
    pub sets: usize,
    /// Sets of the largest size tried.
    pub largest: usize,
    /// Sets the next size would take in all, those tried among them;
    /// `None` where no size of set goes further.
    pub further: Option<usize>,
}

impl TrialReach {
    /// What a solution of the largest size tried is found among, as the
    /// trials charge it: its size's sets, each size given an equal share
    /// of the chances ([`rs_decode_trying`]). The most any solution of
    /// the trials is charged for.
    pub fn most_tries(&self) -> usize {
        self.past.max(1) * self.largest.max(1)
    }

    /// Of the wrong codewords past what the standard corrects, those the
    /// sets tried reach in a block of `n` read whole and certain, with
    /// `ec_len` of error correction and `reserved` of it kept for
    /// detection, and confirm: a set of `s` erased, every one of them
    /// wrong, leaves the code half of `ec_len - s` more to locate, and
    /// each it locates confirms less ([`located_error_bits`]) than the
    /// redundancy it spends. A solution is confirmed as the trials count
    /// it, with `needed` bits to spare besides what the sets cost, or as
    /// the block itself judges it, with `judged` to spare and nothing for
    /// the sets: how rarely noise has a codeword within all its wrong
    /// codewords ([`list_solution_balance`], less a bit for the choice
    /// between the counts, which `judged` holds), which asks nothing of
    /// how many sets were tried to reach it
    /// ([`super::pipeline::judged_verifying_bits`]). Where the sets cost
    /// more than their solutions have to spare, as in the small blocks of
    /// Micro QR, the block's own count is the one that confirms them; the
    /// reach was once said without it, and M3-M, M4-M and M4-Q, whose
    /// blocks it confirms a codeword further, were said to reach a
    /// codeword less. Nothing where any codeword is reserved: a codeword the
    /// trials erased is one read, and wrong, as the standard counts a
    /// solution's errors ([`keeps_reserve`]), and a solution past what
    /// the standard corrects spends the reserve however it was reached.
    /// The trials were once said to reach a codeword past it within the
    /// reserve at 1-H and 3-L, and at 1-M two, the codewords they erased
    /// counted as erasures; and the reach was once counted without what
    /// confirms it, and blocks of a few dozen codewords, those of Micro QR
    /// and the smallest rMQR among them, were said to be corrected by sets
    /// whose count ate what their solutions had to spare.
    pub fn within_standard(&self, n: usize, ec_len: usize, reserved: usize, needed: u32, judged: u32) -> usize {
        if reserved > 0 {
            return 0;
        }
        let reached = (0..self.past)
            .map(|i| self.first_size + 2 * i)
            .filter(|&size| size <= ec_len && size <= n)
            .filter_map(|size| {
                // What a solution of this size of set is charged for its
                // tries, as the trials charge it ([`rs_decode_trying`]).
                let charge = tries_charge(self.past.max(1).saturating_mul(sets_of(n, size, usize::MAX).unwrap_or(usize::MAX)));
                (0..=(ec_len - size) / 2)
                    .rev()
                    .find(|&errors| {
                        let slack = ec_len - size - 2 * errors;
                        8 * slack as u32 + located_error_bits(n - size, errors) >= needed + charge
                            || list_solution_balance(n, ec_len, size + errors) >= i64::from(judged)
                    })
                    .map(|errors| size + errors)
            })
            .max();
        match reached {
            Some(reached) if reached > ec_len / 2 => reached - ec_len / 2,
            _ => 0,
        }
    }
}

pub fn trial_reach(n: usize, ec_len: usize, sets: usize) -> TrialReach {
    let first_size = if ec_len % 2 == 1 { 1 } else { 2 };
    let budget = sets.max(sets_of(n, first_size, TRIAL_SETS_MAX).unwrap_or(0)).min(TRIAL_SETS_MAX);
    let mut reach = TrialReach { past: 0, first_size, sets: 0, largest: 0, further: None };
    let mut size = first_size;
    while size < ec_len && size <= n {
        let count = sets_of(n, size, usize::MAX).unwrap_or(usize::MAX);
        let all = reach.sets.saturating_add(count);
        if all > budget {
            reach.further = Some(all);
            break;
        }
        reach.past += 1;
        reach.sets = all;
        reach.largest = count;
        size += 2;
    }
    reach
}

/// A block solved by doubting single bits, or by flipping a few bits
/// read solidly, or by erasing codewords by trial, or more than one of
/// these.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitDoubt {
    pub solution: BitSolution,
    /// How many bits, read as known, were doubted for their reliability.
    pub doubted_bits: usize,
    /// Which bits of each codeword were doubted, flipped, or erased by
    /// trial.
    pub doubted: Vec<u8>,
    /// Which doubted bits of each codeword the solution overturned: every
    /// flipped bit among them.
    pub changed: Vec<u8>,
    /// Which bits of each codeword were flipped ([`BitSystem::flipping`]):
    /// read solidly, and overturned to make the equations agree. None
    /// where none was.
    pub flipped: Vec<u8>,
    /// Codewords erased by trial ([`solve_bit_erasures_trying`]), whose
    /// bits are among `doubted`; none where none was.
    pub erased: Vec<usize>,
    /// What the solution was found among: the sets of codewords, or of
    /// bits, its trial tried, one where there was none, times the counts
    /// of doubted bits its search may try and the kinds of search it may
    /// make ([`solve_bit_erasures_doubting`]). What it is charged for
    /// ([`tries_charge`]).
    pub tries: usize,
}

impl BitDoubt {
    /// Doubted bits the solution settled to the value read. A faint bit
    /// that turns out right is a check the solution passed, as a known
    /// bit inside an erased codeword is for [`rs_decode`].
    pub fn agreeing_bits(&self) -> u32 {
        self.doubted.iter().zip(&self.changed).map(|(&d, &c)| (d & !c).count_ones()).sum()
    }

    /// Doubted bits the solution overturned.
    pub fn changed_bits(&self) -> u32 {
        self.changed.iter().map(|&c| c.count_ones()).sum()
    }

    /// Bits read solidly that were flipped.
    pub fn flipped_bits(&self) -> u32 {
        self.flipped.iter().map(|&f| f.count_ones()).sum()
    }

    /// Independent bits confirming the solution: the spare equations
    /// and the doubted bits' agreement beyond chance, as
    /// [`doubted_agreement`] weighs it, less what the sets tried cost,
    /// which is taken from the two together, since each set is a fresh
    /// chance for both.
    pub fn confidence_bits(&self) -> u32 {
        self.uncharged_bits().saturating_sub(tries_charge(self.tries))
    }

    /// What confirms the solution before the charge for the sets tried.
    pub fn uncharged_bits(&self) -> u32 {
        self.solution.spare_equations as u32 + doubted_agreement(self.agreeing_bits(), self.changed_bits())
    }

    /// Codewords the solution changed.
    pub fn errors(&self) -> usize {
        self.changed.iter().filter(|&&c| c != 0).count()
    }
}

/// Decode by doubting bits: mark the least reliable bits unknown, fewest
/// first, and solve the rest over GF(2) as [`solve_bit_erasures`] does,
/// stopping at the first count that gives a complete, consistent solution
/// that is offered ([`Offering`]). Doubting one
/// bit spends one of the `8 · ec_len` equations where
/// [`rs_decode_doubting`] spends eight on the whole codeword, so damage
/// scattered one faint module to a codeword over more codewords than
/// there are EC codewords, which no erasure decoder can take, is solved
/// here.
///
/// `bit_reliability[i][b]` is the reliability of bit `1 << b` of codeword
/// `i`. Bits that are pinned or already unknown, and bits at [`CERTAIN`],
/// are never doubted. Neither ZXing nor ZBar has soft decoding; this is
/// the classical ordered-erasure decoding of Forney's GMD, taken down to
/// bits because the code is linear over GF(2).
///
/// With [`Trials::flips`], when no count of doubted bits leaves the
/// known bits consistent, the counts are searched again with a bit or
/// two or three of those read solidly flipped at each, found by what
/// the equations are left wanting rather than by trying every pattern
/// ([`BitSystem::flipping`]): a speck over a module reads solidly
/// wrong, and no doubting by reliability reaches it. This is the
/// reprocessing of ordered-statistics decoding, over the basis the
/// doubted bits leave. With [`Trials::free`], the counts are searched again
/// with a codeword erased by trial at each, then a pair, then a triple
/// ([`solve_bit_erasures_trying`]): a stain over a codeword reads
/// solidly wrong in several bits. The bits as read over every count
/// first, then the flips, then the singles freed before the pairs over
/// every count, since what spends fewer equations is the better
/// confirmed: a codeword freed at a lower count spends eight equations
/// where the single a few more doubted bits would have let do spends
/// one each.
///
/// Every count tried is a fresh chance for a block of noise to come out
/// consistent, and so is every kind of search, so a solution is charged
/// for the counts its search may try, the kinds that may be searched,
/// and the sets its own kind tried at its count ([`BitDoubt::tries`]):
/// each kind is given an equal share of the chances, and each count of
/// it a share of that. The counts tried once went uncharged, held to
/// two codewords of spare equations as by a score of them, which the
/// counts a large block's search may try are more than.
///
/// Returns the solutions of the first kind of search to find any, at
/// the fewest doubted bits it finds any at: one where it stands clear,
/// and the rivals the tries cannot tell apart besides, nearest first
/// ([`nearest`]). Empty when nothing solves.
///
/// The codeword of every count, flip and free that solves is noted in
/// `seen`, those returned among them ([`Sightings`]).
pub fn solve_bit_erasures_doubting(
    block: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    trials: Trials,
    seen: &mut Sightings,
) -> Vec<BitDoubt> {
    assert_eq!(block.len(), bit_reliability.len());
    let unknown_bits: usize = block.iter().map(|c| c.known.count_zeros() as usize).sum();
    let equations = 8 * ec_len;
    let budget = equations.saturating_sub(unknown_bits + SPARE_EQUATIONS);
    if budget == 0 {
        return Vec::new();
    }
    let mut doubtful: Vec<(u8, usize, u8)> = block
        .iter()
        .enumerate()
        .flat_map(|(i, cw)| {
            (0..8u8)
                .filter(move |&b| cw.known >> b & 1 != 0 && cw.pinned >> b & 1 == 0)
                .map(move |b| (bit_reliability[i][b as usize], i, b))
        })
        .filter(|&(r, _, _)| r < CERTAIN)
        .collect();
    if doubtful.is_empty() {
        return Vec::new();
    }
    doubtful.sort();
    doubtful.truncate(budget);

    let n = block.len();
    let mut kinds = vec![Trial::Plain];
    if trials.flips {
        kinds.push(Trial::Flips);
    }
    if trials.free {
        kinds.extend((1..=FREE_SIZES).map(Trial::Free));
    }
    // What every set a trial tries is one of besides: the counts the
    // search may try, and the kinds of search.
    let among = counts_tried(doubtful.len()) * kinds.len();
    let offering = RefCell::new(Offering::new(block, bit_reliability, ec_len, seen));

    /// How a count of doubted bits is tried: as it is, with a few solid
    /// bits flipped, or with codewords of this many freed by trial.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Trial {
        Plain,
        Flips,
        Free(usize),
    }

    // The system of equations with the known bits carried, at each count
    // of doubted bits it was built for: the count as it is, the flips and
    // every size of free ask for the same one, and building it is most of
    // a try. The unknown bits reduce in it as they do in a system that
    // carries no known bit, so the count as it is reads its solution off
    // it ([`BitSystem::solve`]). The rows no unknown has a pivot in are
    // there only where a trial can help: not where the system is
    // consistent, the known bits solving as they are, with too few
    // equations to spare, since the count as it is found nothing; nor
    // where it is incomplete, a trial taking equations the unknown bits
    // already lack. The known bits are carried only where a flip or a
    // free may be tried: the count as it is solves as well without them,
    // and a system that carries them is several times the size.
    //
    // Every count asks for the same equations, the bits not read and the
    // doubted ones taken as unknown and the rest carried, and differs
    // only in how many of the doubtful bits are among the unknowns. So
    // one system is laid for them all, the bits not read first and the
    // doubtful after them in the order they are doubted
    // ([`BitSystem::laid`]), and each count's is the one before it
    // reduced over as many columns more ([`BitSystem::reduce_to`]): a
    // doubted bit is solved for as what overturns its reading, which
    // leaves the right-hand side what the block as read makes it at every
    // count. What a system says of the block is the same whatever order
    // its unknowns were reduced in, and so is what its rows no unknown
    // has a pivot in make of the known bits, a basis of the same
    // equations. Each count's system was once built afresh, a dozen
    // times a search.
    let carried = trials.flips || trials.free;
    let laid = std::cell::OnceCell::new();
    let lay = || {
        let mut doubted_bits = vec![0u8; n];
        for &(_, i, b) in &doubtful {
            doubted_bits[i] |= 1 << b;
        }
        let mut columns: Vec<(usize, u8)> =
            block.iter().enumerate().flat_map(|(i, cw)| (0..8u8).filter(move |b| cw.known >> b & 1 == 0).map(move |b| (i, b))).collect();
        let unread = columns.len();
        columns.extend(doubtful.iter().map(|&(_, i, b)| (i, b)));
        if carried {
            columns.extend(block.iter().enumerate().flat_map(|(i, cw)| {
                let carry = cw.known & !cw.pinned & !doubted_bits[i];
                (0..8u8).filter(move |b| carry >> b & 1 != 0).map(move |b| (i, b))
            }));
        }
        let read_as = doubtful.iter().map(|&(_, i, b)| block[i].value >> b & 1 != 0);
        let read_as = std::iter::repeat_n(false, unread).chain(read_as).collect();
        let mut system = BitSystem::laid(block, ec_len, columns, read_as);
        system.reduce_to(unread);
        system
    };
    let systems: RefCell<BTreeMap<usize, Rc<(BitSystem, Option<ZeroRows>)>>> = RefCell::new(BTreeMap::new());
    let system_at = |k: usize| -> Rc<(BitSystem, Option<ZeroRows>)> {
        if let Some(system) = systems.borrow().get(&k) {
            return system.clone();
        }
        let below = systems.borrow().range(..k).next_back().map(|(_, below)| below.0.clone());
        let mut system = below.unwrap_or_else(|| laid.get_or_init(lay).clone());
        system.reduce_to(unknown_bits + k);
        let zero = (carried && system.rank == system.unknown && !system.consistent()).then(|| system.zero_rows());
        let system = Rc::new((system, zero));
        systems.borrow_mut().insert(k, system.clone());
        system
    };

    let attempt = |k: usize, trial: Trial| -> Option<Vec<BitDoubt>> {
        let mut read = block.to_vec();
        let mut doubted = vec![0u8; n];
        for &(_, i, b) in &doubtful[..k] {
            read[i].known &= !(1 << b);
            read[i].value &= !(1 << b);
            doubted[i] |= 1 << b;
        }
        // Each: the solution, the codewords freed, the bits flipped, and
        // the sets tried.
        let found: Vec<(BitSolution, Vec<usize>, Vec<u8>, usize)> = match trial {
            Trial::Plain => {
                // As [`solve_bit_erasures`] solves the block.
                unknowns_within_reach(&read, ec_len).ok()?;
                let solution = system_at(k).0.solve(&read).ok().filter(BitSolution::is_complete)?;
                vec![(solution, Vec::new(), vec![0; n], 1)]
            }
            Trial::Flips => {
                let shared = system_at(k);
                let (system, zero) = &*shared;
                let (tries, found) = system.flipping(zero.as_ref()?, &read, bit_reliability, trials.sets, offering.borrow_mut().seen());
                found.into_iter().map(|(flipped, solution)| (solution, Vec::new(), flipped, tries)).collect()
            }
            Trial::Free(size) => {
                let shared = system_at(k);
                let (system, zero) = &*shared;
                let (tries, found) = system.trying(zero.as_ref()?, &read, bit_reliability, size, trials.sets, offering.borrow_mut().seen());
                found.into_iter().map(|(erased, _, solution)| (solution, erased, vec![0; n], tries)).collect()
            }
        };
        let doubts: Vec<BitDoubt> = found
            .into_iter()
            .map(|(solution, erased, flipped, tries)| {
                let mut doubted = doubted.clone();
                for &i in &erased {
                    doubted[i] |= block[i].known & !block[i].pinned;
                }
                for (d, &f) in doubted.iter_mut().zip(&flipped) {
                    *d |= f;
                }
                let changed = solution.corrected.iter().zip(block).zip(&doubted).map(|((&got, cw), &m)| (got ^ cw.value) & m).collect();
                BitDoubt { solution, doubted_bits: k, doubted, changed, flipped, erased, tries: tries * among }
            })
            .filter(|doubt| offering.borrow_mut().offers_doubt(doubt))
            .collect();
        (!doubts.is_empty()).then_some(doubts)
    };

    // Fewer doubted bits leave more spare equations, so the first count
    // that solves is the best: find it by growing the count geometrically
    // and then narrowing down between the last failure and that success.
    let search = |attempt: &dyn Fn(usize) -> Option<Vec<BitDoubt>>| -> Option<Vec<BitDoubt>> {
        let mut last_failed = 0;
        let mut k = 1;
        let mut found = None;
        while k <= doubtful.len() {
            match attempt(k) {
                Some(doubts) => {
                    found = Some((k, doubts));
                    break;
                }
                None => last_failed = k,
            }
            k = (k * 3 / 2 + 1).min(doubtful.len()).max(k + 1);
            if last_failed == doubtful.len() {
                break;
            }
        }
        let (mut hi, mut best) = found?;
        let mut lo = last_failed;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            match attempt(mid) {
                Some(doubts) => {
                    hi = mid;
                    best = doubts;
                }
                None => lo = mid,
            }
        }
        Some(best)
    };
    kinds.iter().find_map(|&trial| search(&|k| attempt(k, trial))).unwrap_or_default()
}

/// Counts of doubted bits the search of [`solve_bit_erasures_doubting`]
/// may try among `n` doubtful bits, at the most: every count of its
/// climb, each half again as many as the last, and then as many as
/// halving the widest stretch of the climb takes.
fn counts_tried(n: usize) -> usize {
    let (mut k, mut climbed, mut widest, mut last) = (1usize, 0usize, 1usize, 0usize);
    while k <= n {
        climbed += 1;
        widest = widest.max(k - last);
        last = k;
        if k == n {
            break;
        }
        k = (k * 3 / 2 + 1).min(n).max(k + 1);
    }
    climbed + (usize::BITS - (widest - 1).leading_zeros()) as usize
}

/// Largest set of codewords the bit-level solvers free by trial: a
/// triple, which is twenty-four bits of a [`Span`]'s combination.
pub const FREE_SIZES: usize = 3;

/// Shortest LFSR generating `s`; returns the connection polynomial and its
/// claimed length.
fn berlekamp_massey(s: &[u8]) -> (Vec<u8>, usize) {
    let mut c = vec![1u8];
    let mut b = vec![1u8];
    let mut l = 0usize;
    let mut m = 1usize;
    let mut last = 1u8;
    for n in 0..s.len() {
        let mut d = s[n];
        for i in 1..=l.min(c.len() - 1) {
            d ^= mul(c[i], s[n - i]);
        }
        if d == 0 {
            m += 1;
            continue;
        }
        let scale = div(d, last);
        let previous = c.clone();
        if c.len() < b.len() + m {
            c.resize(b.len() + m, 0);
        }
        for (i, &coef) in b.iter().enumerate() {
            c[i + m] ^= mul(scale, coef);
        }
        if 2 * l <= n {
            l = n + 1 - l;
            b = previous;
            last = d;
            m = 1;
        } else {
            m += 1;
        }
    }
    (c, l)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitSolution {
    /// Known bits as read, plus every unknown bit the syndromes determine.
    /// Unknown bits that stay open are zero.
    pub corrected: Vec<u8>,
    /// The unknown bits that were determined, per codeword.
    pub solved: Vec<u8>,
    pub unknown_bits: usize,
    pub solved_bits: usize,
    /// Equations beyond the rank of the system. When the solution is
    /// complete, every one of them is a check on the known bits that passed.
    pub spare_equations: usize,
}

impl BitSolution {
    /// Every unknown bit was determined.
    pub fn is_complete(&self) -> bool {
        self.solved_bits == self.unknown_bits
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitSolveFailure {
    NoUnknowns,
    TooManyUnknowns { unknown_bits: usize, equations: usize },
    /// No unknown bit is determined by the syndromes.
    Underdetermined { rank: usize, unknown_bits: usize },
    /// The known bits cannot all be right.
    Inconsistent,
}

/// Unknown bits beyond the number of equations that are still worth a try
/// for a partial solution. Further past the equation count the space of
/// solutions is so large that it reaches every unknown bit.
const PARTIAL_MARGIN: usize = 32;

/// Solve for unknown bits directly, assuming every known bit is correct.
///
/// The code is linear over GF(2), so each syndrome byte gives eight equations
/// in the unknown bits. When the equations do not pin down every unknown,
/// the bits they do determine are still returned, as a partial solution.
pub fn solve_bit_erasures(block: &[CwRead], ec_len: usize) -> Result<BitSolution, BitSolveFailure> {
    let n = block.len();
    assert!(n <= 255 && ec_len <= n);
    unknowns_within_reach(block, ec_len)?;
    BitSystem::build(block, ec_len, false).solve(block)
}

/// Whether a block has unknown bits for [`solve_bit_erasures`] to solve,
/// and not so many that no partial solution is worth a try.
fn unknowns_within_reach(block: &[CwRead], ec_len: usize) -> Result<(), BitSolveFailure> {
    let u: usize = block.iter().map(|cw| cw.known.count_zeros() as usize).sum();
    let equations = 8 * ec_len;
    if u == 0 {
        return Err(BitSolveFailure::NoUnknowns);
    }
    if u > equations + PARTIAL_MARGIN {
        return Err(BitSolveFailure::TooManyUnknowns { unknown_bits: u, equations });
    }
    Ok(())
}

/// Solve for unknown bits with a codeword or two or three of those read
/// taken as unknown besides, by trial, when the known bits as read are
/// not consistent: every codeword in turn, then every pair, then every
/// triple, whatever its reliability, as far as [`TRIAL_SETS`] lets the
/// sizes go. A block whose unknown bits are scattered over more
/// codewords than there is redundancy to erase solves bit by bit as long
/// as every bit read is right, and a stain that reads solidly wrong over
/// one codeword puts every one of its equations out; here that codeword
/// is found, among the few whose bits can be freed to make the equations
/// agree, and the rest of the block solved as before. With
/// [`Trials::flips`], a bit or two or three read solidly wrong is looked for first
/// ([`BitSystem::flipping`]), which spends no equations where a
/// codeword freed spends eight.
///
/// Each trial is a fresh chance for a block of noise to come out
/// consistent, so the spare equations are charged for the sets tried,
/// each kind of trial, the flips and each size of free, given an equal
/// share of the chances ([`BitDoubt::confidence_bits`]); a solution is
/// kept only where it is offered ([`Offering`]); and where two sets reach different
/// blocks the one nearest the block as read is kept by a clear margin,
/// or every one within the margin is returned, nearest first
/// ([`nearest`]). Pinned bits are never freed. Empty when the known
/// bits are consistent as they are, which is [`solve_bit_erasures`]'s
/// case, when the unknown bits are not all determined even before the
/// trial, or when no set makes them agree.
///
/// The codeword of every flip and free that solves is noted in `seen`,
/// those returned among them ([`Sightings`]).
pub fn solve_bit_erasures_trying(block: &[CwRead], bit_reliability: &[[u8; 8]], ec_len: usize, trials: Trials, seen: &mut Sightings) -> Vec<BitDoubt> {
    assert_eq!(block.len(), bit_reliability.len());
    let unknown_bits: usize = block.iter().map(|c| c.known.count_zeros() as usize).sum();
    if unknown_bits + SPARE_EQUATIONS >= 8 * ec_len {
        return Vec::new();
    }
    let system = BitSystem::build(block, ec_len, true);
    if system.rank < system.unknown || system.consistent() {
        return Vec::new();
    }
    let n = block.len();
    let kinds = usize::from(trials.flips) + if trials.free { FREE_SIZES } else { 0 };
    let mut offering = Offering::new(block, bit_reliability, ec_len, seen);
    let keep = |offering: &mut Offering, (solution, erased, flipped, tries): (BitSolution, Vec<usize>, Vec<u8>, usize)| -> Option<BitDoubt> {
        let doubted: Vec<u8> =
            (0..n).map(|i| if erased.contains(&i) { block[i].known & !block[i].pinned } else { 0 } | flipped[i]).collect();
        let changed = solution.corrected.iter().zip(block).zip(&doubted).map(|((&got, cw), &m)| (got ^ cw.value) & m).collect();
        let doubt = BitDoubt { solution, doubted_bits: 0, doubted, changed, flipped, erased, tries: tries * kinds.max(1) };
        offering.offers_doubt(&doubt).then_some(doubt)
    };
    let zero = system.zero_rows();
    if trials.flips {
        let (tries, found) = system.flipping(&zero, block, bit_reliability, trials.sets, offering.seen());
        let doubts: Vec<BitDoubt> =
            found.into_iter().filter_map(|(flipped, solution)| keep(&mut offering, (solution, Vec::new(), flipped, tries))).collect();
        if !doubts.is_empty() {
            return doubts;
        }
    }
    for size in 1..=FREE_SIZES {
        if !trials.free {
            break;
        }
        let (tries, found) = system.trying(&zero, block, bit_reliability, size, trials.sets, offering.seen());
        let doubts: Vec<BitDoubt> =
            found.into_iter().filter_map(|(erased, _, solution)| keep(&mut offering, (solution, erased, vec![0; n], tries))).collect();
        if !doubts.is_empty() {
            return doubts;
        }
    }
    Vec::new()
}

/// The equations of one block over GF(2), built and reduced: eight to a
/// syndrome, a column to each bit taken as unknown, and on the right-hand
/// side what the bits read as known make of the syndromes. With
/// `with_known`, a column for every known bit that is not pinned is
/// carried through the reduction beside the unknowns', so that a known
/// bit can afterwards be taken as unknown without building the system
/// again ([`BitSystem::trying`]): the rows no unknown has a pivot in say
/// at once whether the block is consistent with that bit free.
#[derive(Clone)]
struct BitSystem {
    /// Each column's bit, as (codeword, bit): the unknown bits first,
    /// then the known ones where they are carried.
    columns: Vec<(usize, u8)>,
    /// How many of `columns` are unknown bits.
    unknown: usize,
    equations: usize,
    /// Row `8 * j + k` is bit `k` of syndrome `j`; the right-hand side
    /// sits at column `columns.len()`. Reduced on the unknown columns:
    /// a pivot's column is set in its row alone, and the pivot rows come
    /// first. The rows lie end to end, `words` words each.
    rows: Vec<u64>,
    words: usize,
    /// Row holding each unknown column's pivot, `usize::MAX` for none.
    pivot_row: Vec<usize>,
    rank: usize,
    /// For each column taken as unknown though it was read, the bit read
    /// ([`BitSystem::laid`]): what the column is solved for is what
    /// overturns it. False, or past the end, for a bit not read.
    read_as: Vec<bool>,
}

impl BitSystem {
    fn build(block: &[CwRead], ec_len: usize, with_known: bool) -> Self {
        let mut columns: Vec<(usize, u8)> = block
            .iter()
            .enumerate()
            .flat_map(|(i, cw)| (0..8u8).filter(move |b| cw.known >> b & 1 == 0).map(move |b| (i, b)))
            .collect();
        let unknown = columns.len();
        if with_known {
            columns.extend(block.iter().enumerate().flat_map(|(i, cw)| {
                (0..8u8).filter(move |b| cw.known >> b & 1 != 0 && cw.pinned >> b & 1 == 0).map(move |b| (i, b))
            }));
        }
        let mut system = Self::laid(block, ec_len, columns, Vec::new());
        system.reduce_to(unknown);
        system
    }

    /// The equations of `block` over `columns`, none of them reduced yet
    /// ([`Self::reduce_to`]), the right-hand side what every bit read
    /// makes of the syndromes. A column of a bit that was read is solved
    /// for, once reduced as an unknown, as what overturns its reading,
    /// `read_as` holding the bit read: the right-hand side, made with it,
    /// is so whichever of the bits read are taken as unknown.
    fn laid(block: &[CwRead], ec_len: usize, columns: Vec<(usize, u8)>, read_as: Vec<bool>) -> Self {
        let n = block.len();
        let equations = 8 * ec_len;
        let rhs = columns.len();
        let words = rhs / 64 + 1;
        let mut rows = vec![0u64; equations * words];
        // What each column adds to syndrome `j`: its bit times its
        // codeword's locator to the `j`th power, a byte whose bit `k`
        // goes to row `8 * j + k`. Eight columns' bytes at a time are
        // turned about, so that each byte of the result is one row's bits
        // across the eight.
        let locators: Vec<u8> = columns.iter().map(|&(i, _)| exp(n - 1 - i)).collect();
        let mut powers = vec![1u8; rhs];
        let mut contributions = vec![0u8; rhs.next_multiple_of(8)];
        for j in 0..ec_len {
            for (col, &(_, bit)) in columns.iter().enumerate() {
                contributions[col] = mul(1 << bit, powers[col]);
                powers[col] = mul(powers[col], locators[col]);
            }
            for (m, &eight) in contributions.as_chunks::<8>().0.iter().enumerate() {
                let turned = transpose_bits(u64::from_le_bytes(eight));
                let (word, shift) = (m / 8, 8 * (m % 8));
                for k in 0..8 {
                    rows[(8 * j + k) * words + word] |= (turned >> (8 * k) & 0xFF) << shift;
                }
            }
        }
        let received: Vec<u8> = block.iter().map(|c| c.value & c.known).collect();
        for (j, &s) in rs_syndromes(&received, ec_len).iter().enumerate() {
            for k in 0..8 {
                if s >> k & 1 != 0 {
                    rows[(8 * j + k) * words + rhs / 64] |= 1 << (rhs % 64);
                }
            }
        }
        Self { columns, unknown: 0, equations, rows, words, pivot_row: Vec::new(), rank: 0, read_as }
    }

    /// The system reduced over its first `unknown` columns, those it was
    /// reduced over already and as many more: the reduction goes a column
    /// at a time, and leaves the columns before alone.
    fn reduce_to(&mut self, unknown: usize) {
        debug_assert!(self.unknown <= unknown && unknown <= self.columns.len());
        let Self { rows, words, equations, rank, pivot_row, .. } = self;
        let (words, equations) = (*words, *equations);
        let mut pivot = vec![0u64; words];
        for col in self.unknown..unknown {
            let (word, bit) = (col / 64, 1u64 << (col % 64));
            let Some(p) = (*rank..equations).find(|&r| rows[r * words + word] & bit != 0) else {
                pivot_row.push(usize::MAX);
                continue;
            };
            if p != *rank {
                let (head, tail) = rows.split_at_mut(p * words);
                head[*rank * words..(*rank + 1) * words].swap_with_slice(&mut tail[..words]);
            }
            // The pivot row is nought in every column before its own: a
            // column with a pivot was cleared from every other row, and
            // one without is nought in every row from the rank on. So the
            // words before its own add nothing.
            debug_assert!(rows[*rank * words..*rank * words + word].iter().all(|&w| w == 0));
            pivot[word..].copy_from_slice(&rows[*rank * words + word..(*rank + 1) * words]);
            for (r, row) in rows.chunks_exact_mut(words).enumerate() {
                if r != *rank && row[word] & bit != 0 {
                    for (a, b) in row[word..].iter_mut().zip(&pivot[word..]) {
                        *a ^= b;
                    }
                }
            }
            pivot_row.push(*rank);
            *rank += 1;
        }
        self.unknown = unknown;
    }

    fn row(&self, r: usize) -> &[u64] {
        &self.rows[r * self.words..(r + 1) * self.words]
    }

    /// The block solved with every known bit taken as read, as
    /// [`solve_bit_erasures`] solves it: alike whether the system carries
    /// the known bits or not, since the unknown columns and the
    /// right-hand side reduce the same either way.
    fn solve(&self, block: &[CwRead]) -> Result<BitSolution, BitSolveFailure> {
        if !self.consistent() {
            return Err(BitSolveFailure::Inconsistent);
        }
        let solution = self.solution(block, &vec![0; block.len()], self.equations - self.rank);
        if solution.solved_bits == 0 {
            return Err(BitSolveFailure::Underdetermined { rank: self.rank, unknown_bits: self.unknown });
        }
        debug_assert!(solution.solved_bits < self.unknown || rs_syndromes(&solution.corrected, self.equations / 8).iter().all(|&s| s == 0));
        Ok(solution)
    }

    fn bit(row: &[u64], col: usize) -> bool {
        row[col / 64] >> (col % 64) & 1 != 0
    }

    fn rhs(&self, row: &[u64]) -> bool {
        Self::bit(row, self.columns.len())
    }

    /// Whether the known bits can all be right: no row without a pivot
    /// has anything on its right-hand side.
    fn consistent(&self) -> bool {
        self.rows[self.rank * self.words..].chunks_exact(self.words).all(|row| !self.rhs(row))
    }

    /// Unknown columns set in a row.
    fn unknowns_in(&self, row: &[u64]) -> u32 {
        let full = self.unknown / 64;
        let rest = self.unknown % 64;
        row[..full].iter().map(|w| w.count_ones()).sum::<u32>() + (row[full] & ((1u64 << rest) - 1)).count_ones()
    }

    /// The block as the system solves it: the known bits as read, with
    /// the bits of `changed` overturned, and every unknown bit the rows
    /// determine, which a pivot's row does exactly when it involves no
    /// other unknown. A known bit overturned moves the right-hand side
    /// of every row its column is set in. `spare` is what the caller
    /// knows the equations to have beyond the rank.
    fn solution(&self, block: &[CwRead], changed: &[u8], spare: usize) -> BitSolution {
        let moved: Vec<usize> = (self.unknown..self.columns.len())
            .filter(|&col| {
                let (i, b) = self.columns[col];
                changed[i] >> b & 1 != 0
            })
            .collect();
        let mut corrected: Vec<u8> = block.iter().zip(changed).map(|(cw, &c)| (cw.value & cw.known) ^ c).collect();
        let mut solved = vec![0u8; block.len()];
        let mut solved_bits = 0;
        for (col, &(i, bit)) in self.columns[..self.unknown].iter().enumerate() {
            if self.pivot_row[col] == usize::MAX {
                continue;
            }
            let row = self.row(self.pivot_row[col]);
            if self.unknowns_in(row) != 1 {
                continue;
            }
            solved[i] |= 1 << bit;
            solved_bits += 1;
            let read_as = self.read_as.get(col).copied().unwrap_or(false);
            if self.rhs(row) ^ (moved.iter().filter(|&&m| Self::bit(row, m)).count() % 2 == 1) ^ read_as {
                corrected[i] |= 1 << bit;
            }
        }
        BitSolution { corrected, solved, unknown_bits: self.unknown, solved_bits, spare_equations: spare }
    }

    /// The rows no unknown has a pivot in, as every known column and
    /// the right-hand side stand over them, what the known bits must
    /// answer for by themselves: computed once for the flips and every
    /// size of free a block is tried with, a row at a time, each setting
    /// its bit in the columns it holds.
    fn zero_rows(&self) -> ZeroRows {
        let zero = self.equations - self.rank;
        let words = zero / 64 + 1;
        let (first, end) = (self.unknown, self.columns.len());
        let mut columns = vec![0u64; (end - first) * words];
        let mut residual = vec![0u64; words];
        let zero_rows: Vec<&[u64]> = self.rows[self.rank * self.words..].chunks_exact(self.words).collect();
        for (k, row) in zero_rows.iter().enumerate() {
            if self.rhs(row) {
                residual[k / 64] |= 1 << (k % 64);
            }
        }
        // Sixty-four rows by sixty-four columns at a time, turned about the
        // diagonal: what was a word of a row is then a word of a column.
        for (word, rows) in zero_rows.chunks(64).enumerate() {
            for w in first / 64..end.div_ceil(64) {
                let mut square = [0u64; 64];
                for (held, row) in square.iter_mut().zip(rows) {
                    *held = row[w];
                }
                transpose_square(&mut square);
                let (from, to) = ((64 * w).max(first), (64 * w + 64).min(end));
                for col in from..to {
                    columns[(col - first) * words + word] = square[col - 64 * w];
                }
            }
        }
        ZeroRows { words, residual, columns }
    }

    /// The block solved with `size` codewords of those read freed, by
    /// trial, for a system built with its known bits carried, whose
    /// unknown bits are all determined and whose known bits are not
    /// consistent. In the rows no unknown has a pivot in, what is left
    /// of the right-hand side must be made up by the columns of the bits
    /// freed: a codeword's columns there, in echelon form, say whether
    /// they can, and how, and that they settle the codeword, which they
    /// do when they are independent. Every set of `size` codewords is
    /// tried, when the sets of that size and every smaller one fit
    /// `budget` between them, [`TRIAL_SETS`] at the most. Returns the sets tried, of this size
    /// and the smaller ones, and the solutions found, each with the
    /// codewords freed and the bits of each codeword it overturned, the
    /// nearest first and the rivals within the margin after it
    /// ([`nearest`]), every one found noted in `seen` ([`Sightings`]).
    fn trying(
        &self,
        over: &ZeroRows,
        block: &[CwRead],
        bit_reliability: &[[u8; 8]],
        size: usize,
        budget: usize,
        seen: &mut Sightings,
    ) -> (usize, Vec<(Vec<usize>, Vec<u8>, BitSolution)>) {
        debug_assert!(self.rank == self.unknown && !self.consistent());
        let n = block.len();
        let zero = self.equations - self.rank;
        // Each codeword's known columns, in their order, end to end: the
        // `i`th codeword's from `column_starts[i]` to `column_starts[i + 1]`.
        let mut column_starts = vec![0usize; n + 1];
        for col in self.unknown..self.columns.len() {
            column_starts[self.columns[col].0 + 1] += 1;
        }
        for i in 0..n {
            column_starts[i + 1] += column_starts[i];
        }
        let mut by_codeword = vec![0usize; self.columns.len() - self.unknown];
        let mut filled = column_starts.clone();
        for col in self.unknown..self.columns.len() {
            let i = self.columns[col].0;
            by_codeword[filled[i]] = col;
            filled[i] += 1;
        }
        let columns_of = |i: usize| &by_codeword[column_starts[i]..column_starts[i + 1]];
        let candidates: Vec<usize> = (0..n).filter(|&i| !columns_of(i).is_empty()).collect();
        let c = candidates.len();
        let budget = budget.min(TRIAL_SETS_MAX);
        let tries: Option<usize> = (1..=size).map(|s| sets_of(c, s, budget)).sum();
        let Some(tries) = tries.filter(|&tries| tries <= budget) else { return (0, Vec::new()) };
        if size == 0 || size > c || zero < 8 * size + SPARE_EQUATIONS {
            return (tries, Vec::new());
        }
        // Each candidate's known columns over the zero rows, laid end to
        // end in the order of the candidates.
        let mut starts = Vec::with_capacity(c + 1);
        starts.push(0);
        let mut vectors = Vec::with_capacity(by_codeword.len() * over.words);
        for &i in &candidates {
            for &col in columns_of(i) {
                vectors.extend_from_slice(over.column(col - self.unknown));
            }
            starts.push(starts[starts.len() - 1] + columns_of(i).len());
        }
        let mut found: Vec<(u32, (Vec<usize>, Vec<u8>, BitSolution))> = Vec::new();
        each_freeing(&starts, &vectors, &over.residual, size, |set, combination| {
            let freed: usize = set.iter().map(|&k| starts[k + 1] - starts[k]).sum();
            let mut changed = vec![0u8; n];
            for (which, &k) in set.iter().enumerate() {
                let i = candidates[k];
                for (b, &col) in columns_of(i).iter().enumerate() {
                    if combination >> (8 * which + b) & 1 != 0 {
                        changed[i] |= 1 << self.columns[col].1;
                    }
                }
            }
            let solution = self.solution(block, &changed, zero - freed);
            debug_assert!(solution.is_complete());
            seen.see(&solution.corrected);
            let erased = set.iter().map(|&k| candidates[k]).collect();
            found.push((soft_distance(block, bit_reliability, &solution.corrected), (erased, changed, solution)));
        });
        (tries, nearest(found, |a, b| a.2.corrected == b.2.corrected, |a, b| a.2.spare_equations > b.2.spare_equations))
    }

    /// The block solved with a bit, or two, or three of those read
    /// solidly flipped, for a system built with its known bits carried,
    /// whose unknown bits are all determined and whose known bits are
    /// not consistent. A flipped bit moves the right-hand side by its
    /// column, so the bits to flip are those whose columns over the rows
    /// no unknown has a pivot in add up to what is left there: every
    /// bit is looked at for the singles, every bit against a table of
    /// the rest for the pairs, and every pair of the
    /// [`FLIP_TRIPLE_BITS`] least reliable bits against the table for
    /// the triples, or as many of them as have no more pairs than four
    /// lookups a set of `budget`. Only bits sampled from a photo are
    /// flipped, as only
    /// they are doubted: a bit painted by hand is certain. Returns the
    /// sets of bits looked at and the solutions found, each with the
    /// bits of each codeword it flipped, the nearest first and the
    /// rivals within the margin after it ([`nearest`]), every one found
    /// noted in `seen` ([`Sightings`]).
    fn flipping(
        &self,
        over: &ZeroRows,
        block: &[CwRead],
        bit_reliability: &[[u8; 8]],
        budget: usize,
        seen: &mut Sightings,
    ) -> (usize, Vec<(Vec<u8>, BitSolution)>) {
        debug_assert!(self.rank == self.unknown && !self.consistent());
        let n = block.len();
        let zero = self.equations - self.rank;
        let mut cols: Vec<(u8, usize)> = (self.unknown..self.columns.len())
            .filter_map(|col| {
                let (i, b) = self.columns[col];
                let reliability = bit_reliability[i][b as usize];
                (reliability < CERTAIN).then_some((reliability, col))
            })
            .collect();
        // By reliability and then by bit, as the columns of a system that
        // carries its known bits in the block's order are.
        cols.sort_by_key(|&(reliability, col)| (reliability, self.columns[col]));
        let c = cols.len();
        if c == 0 || zero < SPARE_EQUATIONS + tries_charge(c) as usize {
            return (0, Vec::new());
        }
        let residual = &over.residual;
        let vectors: Vec<&[u64]> = cols.iter().map(|&(_, col)| over.column(col - self.unknown)).collect();
        let mut sets: Vec<Vec<usize>> = Vec::new();
        let mut tries = c;
        for (k, &v) in vectors.iter().enumerate() {
            if v == residual.as_slice() {
                sets.push(vec![k]);
            }
        }
        let pairs = c * (c - 1) / 2;
        if c >= 2 && zero >= SPARE_EQUATIONS + tries_charge(tries + pairs) as usize {
            tries += pairs;
            // Every bit by a fingerprint of its column, those sharing one
            // in their order: a pair is a bit and one before it whose
            // column makes up the rest, a triple two bits and one after
            // them that does. The fingerprint is linear, so that of what
            // is left wanting is the sum of the fingerprints, and a column
            // whose fingerprint matches is held to the vector itself.
            let fingerprint = |v: &[u64]| v.iter().enumerate().fold(0u64, |f, (w, &x)| f ^ x.rotate_left((23 * w % 64) as u32));
            let prints: Vec<u64> = vectors.iter().map(|v| fingerprint(v)).collect();
            // The bits in order of their fingerprints, those sharing one in
            // their own order, and where each fingerprint's run of them lies.
            let mut in_print_order: Vec<usize> = (0..c).collect();
            in_print_order.sort_unstable_by_key(|&k| (prints[k], k));
            let mut by_print: HashMap<u64, std::ops::Range<usize>, WordHash> = HashMap::with_capacity_and_hasher(c, WordHash::default());
            // And which fingerprints there are at all, by sixteen bits of
            // them, which the triples ask after hundreds of thousands of
            // times a block and nearly always in vain: most are answered
            // here, without the map.
            let slot = |print: u64| (print.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 48) as usize;
            let mut present = vec![0u64; 1 << 10];
            for (at, &k) in in_print_order.iter().enumerate() {
                let print = prints[k];
                by_print.entry(print).or_insert(at..at).end = at + 1;
                present[slot(print) / 64] |= 1 << (slot(print) % 64);
            }
            let sharing = |print: u64| by_print.get(&print).map(|run| &in_print_order[run.clone()]);
            let maybe = |print: u64| present[slot(print) / 64] >> (slot(print) % 64) & 1 != 0;
            let wanting = fingerprint(residual);
            let mut target = vec![0u64; residual.len()];
            // The bits among `found` whose column is `residual` plus those
            // of `with`.
            let mut matching = |found: &[usize], with: &[&[u64]], keep: &mut dyn FnMut(usize)| {
                target.copy_from_slice(residual);
                for v in with {
                    for (t, x) in target.iter_mut().zip(*v) {
                        *t ^= x;
                    }
                }
                for &k in found {
                    if vectors[k] == target.as_slice() {
                        keep(k);
                    }
                }
            };
            for (k, &v) in vectors.iter().enumerate() {
                if let Some(others) = sharing(wanting ^ prints[k]) {
                    let before = &others[..others.partition_point(|&a| a < k)];
                    if !before.is_empty() {
                        matching(before, &[v], &mut |a| sets.push(vec![a, k]));
                    }
                }
            }
            // As many of the least reliable bits as have no more pairs
            // than four lookups a set of the budget: a lookup is a hash of
            // a few words where a set erased is a decode.
            let s = c.min(FLIP_TRIPLE_BITS).min((8 * budget.min(TRIAL_SETS_MAX)).isqrt());
            let triples: usize = (0..s).map(|a| (a + 1..s).map(|b| c - b - 1).sum::<usize>()).sum();
            if c >= 3 && zero >= SPARE_EQUATIONS + tries_charge(tries + triples) as usize {
                tries += triples;
                for a in 0..s {
                    let partial = wanting ^ prints[a];
                    for b in a + 1..s {
                        let print = partial ^ prints[b];
                        if !maybe(print) {
                            continue;
                        }
                        if let Some(others) = sharing(print) {
                            let after = &others[others.partition_point(|&k| k <= b)..];
                            if !after.is_empty() {
                                matching(after, &[vectors[a], vectors[b]], &mut |k| sets.push(vec![a, b, k]));
                            }
                        }
                    }
                }
            }
        }
        let found: Vec<(u32, (Vec<u8>, BitSolution))> = sets
            .into_iter()
            .map(|set| {
                let mut flipped = vec![0u8; n];
                for &k in &set {
                    let (i, b) = self.columns[cols[k].1];
                    flipped[i] |= 1 << b;
                }
                let solution = self.solution(block, &flipped, zero);
                debug_assert!(solution.is_complete());
                seen.see(&solution.corrected);
                (soft_distance(block, bit_reliability, &solution.corrected), (flipped, solution))
            })
            .collect();
        (tries, nearest(found, |a, b| a.1.corrected == b.1.corrected, |a, b| a.1.spare_equations > b.1.spare_equations))
    }
}

/// An eight by eight matrix of bits turned about its diagonal: bit `c`
/// of byte `r` becomes bit `r` of byte `c`. Three exchanges, of the
/// corners of each two by two block, then of each four by four, then of
/// the whole.
fn transpose_bits(mut x: u64) -> u64 {
    let t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AA;
    x ^= t ^ (t << 7);
    let t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCC;
    x ^= t ^ (t << 14);
    let t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0;
    x ^= t ^ (t << 28);
    x
}

/// A sixty-four by sixty-four matrix of bits turned about its diagonal,
/// in place: bit `c` of word `r` becomes bit `r` of word `c`. The two
/// off-diagonal halves exchanged, then within each half its quarters, and
/// so on down to single bits, six exchanges of every pair of words.
fn transpose_square(square: &mut [u64; 64]) {
    let (mut j, mut low) = (32, 0x0000_0000_FFFF_FFFFu64);
    while j != 0 {
        let mut k = 0;
        while k < 64 {
            let t = ((square[k] >> j) ^ square[k + j]) & low;
            square[k] ^= t << j;
            square[k + j] ^= t;
            k = (k + j + 1) & !j;
        }
        j >>= 1;
        low ^= low << j;
    }
}

/// A hasher for the fingerprints of the columns of a [`BitSystem`] over
/// its zero rows ([`BitSystem::flipping`]): a rotate and a multiply a
/// word. The columns are a code's, as good as random, and a map of them
/// is looked into hundreds of thousands of times a block, where the
/// standard library's hasher, made to withstand keys chosen to collide,
/// costs several times as much.
#[derive(Clone, Copy, Default)]
struct WordHasher(u64);

impl std::hash::Hasher for WordHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for &word in words {
            self.write_u64(u64::from_le_bytes(word));
        }
        for &byte in rest {
            self.write_u64(u64::from(byte));
        }
    }

    fn write_u64(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95);
    }

    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}

type WordHash = std::hash::BuildHasherDefault<WordHasher>;

/// The rows of a [`BitSystem`] no unknown has a pivot in, as the known
/// columns and the right-hand side stand over them ([`BitSystem::zero_rows`]).
struct ZeroRows {
    /// Words each vector over the rows takes.
    words: usize,
    /// The right-hand side: what the known bits as read leave wanting.
    residual: Vec<u64>,
    /// Each known column, by its place among the known columns, laid end
    /// to end.
    columns: Vec<u64>,
}

impl ZeroRows {
    /// The `k`th known column.
    fn column(&self, k: usize) -> &[u64] {
        &self.columns[k * self.words..(k + 1) * self.words]
    }
}

/// Vectors over GF(2) of `W` words in fully reduced echelon form, each
/// with the combination of the vectors added that it is made of, so that
/// a vector the span holds can be expressed in them: no more than `R` of
/// them. Every row is nought at every other row's pivot, so whether a
/// row is added to a vector being reduced is read off the vector as it
/// came, for every row at once, rather than off what the rows before it
/// left, one after another; and is added under a mask rather than a
/// branch, since it is a coin's toss which no predictor guesses. The
/// combinations are kept only where `TRACK`, since the search of
/// [`each_freeing`] asks for one only of a set that solves, which few do.
struct Span<const W: usize, const R: usize, const TRACK: bool> {
    rows: [[u64; W]; R],
    /// Each row's pivot, its lowest bit, as a vector of that bit alone,
    /// and as its word and the bit in it.
    pivots: [[u64; W]; R],
    pivot_at: [(usize, u64); R],
    combinations: [u32; R],
    len: usize,
}

impl<const W: usize, const R: usize, const TRACK: bool> Span<W, R, TRACK> {
    fn new() -> Self {
        Self { rows: [[0; W]; R], pivots: [[0; W]; R], pivot_at: [(0, 0); R], combinations: [0; R], len: 0 }
    }

    fn clear(&mut self) {
        self.len = 0;
    }

    /// All ones where `v` holds the bit of `pivot`, else nought.
    fn holds(v: &[u64; W], pivot: &[u64; W]) -> u64 {
        let any = v.iter().zip(pivot).fold(0, |any, (a, b)| any | (a & b));
        0u64.wrapping_sub(u64::from(any != 0))
    }

    /// Add a vector, `combination` naming it; false when it lay in the
    /// span already.
    fn add(&mut self, mut v: [u64; W], mut combination: u32) -> bool {
        debug_assert!(self.len < R);
        self.reduce(&mut v, &mut combination);
        let Some(word) = v.iter().position(|&x| x != 0) else { return false };
        let mut pivot = [0u64; W];
        pivot[word] = v[word].isolate_lowest_one();
        for r in 0..self.len {
            let mask = Self::holds(&self.rows[r], &pivot);
            for (a, b) in self.rows[r].iter_mut().zip(&v) {
                *a ^= b & mask;
            }
            if TRACK {
                self.combinations[r] ^= combination & mask as u32;
            }
        }
        self.rows[self.len] = v;
        self.pivots[self.len] = pivot;
        self.pivot_at[self.len] = (word, pivot[word]);
        self.combinations[self.len] = combination;
        self.len += 1;
        true
    }

    /// `v` modulo the span, in place, with the combination of the rows
    /// added to it added to `combination`: nought exactly when the span
    /// holds `v`, and `combination` then what makes it.
    fn reduce(&self, v: &mut [u64; W], combination: &mut u32) {
        let came = *v;
        for r in 0..self.len {
            let mask = Self::holds(&came, &self.pivots[r]);
            for (a, b) in v.iter_mut().zip(&self.rows[r]) {
                *a ^= b & mask;
            }
            if TRACK {
                *combination ^= self.combinations[r] & mask as u32;
            }
        }
    }
}

/// Every set of `size` codewords whose freed columns are independent and
/// make up `residual`, as [`BitSystem::trying`] looks for them, to `f`
/// with the combination that does: bit `8 * which + b` for the `b`th
/// column of the `which`th codeword of the set. The `k`th codeword's
/// columns are `vectors` from `starts[k]` to `starts[k + 1]`, each of the
/// residual's length, eight to a codeword at the most. The sets come in
/// the order [`each_set`] makes them, and the work asks every so often
/// whether it was called off.
///
/// A set is reduced a codeword at a time. The columns of the codewords
/// chosen so far span a space, and every codeword after them is taken
/// modulo it once, for all the sets that begin so; whether a codeword's
/// columns add as many dimensions as it has columns, and whether what is
/// left of the residual lies among them, is then an elimination among its
/// own few. A prefix whose columns are dependent already is so in every
/// set it begins, and those are passed over. Columns that are independent
/// make a vector one way only, so the combination of a set that solves is
/// found afterwards from its own columns, and is the one a reduction of
/// the whole set at once finds. The vectors are held in as few whole
/// words as hold them, the words past their end nought, which leaves
/// every pivot where it was.
fn each_freeing(starts: &[usize], vectors: &[u64], residual: &[u64], size: usize, f: impl FnMut(&[usize], u32)) {
    match residual.len() {
        0 | 1 => Freeing::<1>::new(starts, vectors, residual, size).each(f),
        2 => Freeing::<2>::new(starts, vectors, residual, size).each(f),
        3 | 4 => Freeing::<4>::new(starts, vectors, residual, size).each(f),
        5..=9 => Freeing::<9>::new(starts, vectors, residual, size).each(f),
        _ => Freeing::<32>::new(starts, vectors, residual, size).each(f),
    }
}

/// A vector of `W` words folded into one, by a map that is linear: the
/// folds of vectors that are dependent are dependent, so folds that are
/// independent say the vectors are.
fn fold<const W: usize>(v: &[u64; W]) -> u64 {
    v.iter().enumerate().fold(0, |f, (w, &x)| f ^ x.rotate_left((17 * w % 64) as u32))
}

/// Whether the words `vs`, and `last` after them, are independent: a
/// word at a time reduced by those before it, nought when it is not.
fn independent_words(vs: &[u64], last: u64) -> bool {
    let (mut rows, mut pivots, mut len) = ([0u64; 9], [0u64; 9], 0);
    for &v in vs.iter().chain(std::iter::once(&last)) {
        let mut v = v;
        for r in 0..len {
            v ^= rows[r] & 0u64.wrapping_sub(u64::from(v & pivots[r] != 0));
        }
        if v == 0 {
            return false;
        }
        rows[len] = v;
        pivots[len] = v.isolate_lowest_one();
        len += 1;
    }
    true
}

/// The state of [`each_freeing`] over vectors of `W` words.
struct Freeing<'a, const W: usize> {
    starts: &'a [usize],
    size: usize,
    /// For each depth `d` short of the last: every vector modulo the
    /// columns of the first `d` codewords of the set. Only the vectors of
    /// the codewords after the `d`th are kept up to date. At the last
    /// depth only their folds are kept, below, and a codeword's vectors
    /// are reduced in full only where its folds do not settle the set.
    vectors: Vec<Vec<[u64; W]>>,
    /// For each depth, every vector's fold ([`fold`]).
    folds: Vec<Vec<u64>>,
    /// For each depth, the residual so.
    residuals: Vec<[u64; W]>,
    /// The columns of the codeword before the last, modulo those before
    /// it: what the last depth's vectors are the depth before's modulo.
    prefix: Span<W, 8, false>,
    /// The columns of the codeword in hand, modulo those before it.
    span: Span<W, 8, false>,
    set: Vec<usize>,
    /// Sets of full size looked at, for asking every so often whether
    /// the work was called off.
    looked: usize,
}

impl<'a, const W: usize> Freeing<'a, W> {
    fn new(starts: &'a [usize], vectors: &[u64], residual: &[u64], size: usize) -> Self {
        let words = residual.len();
        assert!(words <= W, "vectors of {words} words held in {W}");
        let widen = |v: &[u64]| {
            let mut out = [0u64; W];
            out[..words].copy_from_slice(v);
            out
        };
        let count = starts[starts.len() - 1];
        let first: Vec<[u64; W]> = vectors.chunks_exact(words.max(1)).map(widen).collect();
        debug_assert_eq!(first.len(), count);
        let mut folds = vec![first.iter().map(fold).collect::<Vec<u64>>()];
        folds.resize(size.max(1), vec![0; count]);
        let mut levels = vec![first];
        levels.resize(size.saturating_sub(1).max(1), vec![[0; W]; count]);
        let mut residuals = vec![[0; W]; size.max(1)];
        residuals[0] = widen(residual);
        Self { starts, size, vectors: levels, folds, residuals, prefix: Span::new(), span: Span::new(), set: vec![0; size], looked: 0 }
    }

    fn each(mut self, mut f: impl FnMut(&[usize], u32)) {
        let c = self.starts.len() - 1;
        if self.size == 0 || self.size > c {
            return;
        }
        if self.size == 1 {
            self.leaves(0, 0, &mut f);
        } else {
            self.descend(0, 0, &mut f);
        }
    }

    /// The sets whose first `depth` codewords are those chosen, the next
    /// one from `from` on, short of the last. False once the work was
    /// called off.
    fn descend(&mut self, depth: usize, from: usize, f: &mut impl FnMut(&[usize], u32)) -> bool {
        let c = self.starts.len() - 1;
        for k in from..=c - (self.size - depth) {
            if work::cancelled() {
                return false;
            }
            // The codeword's columns modulo those chosen before it.
            self.span.clear();
            if !self.vectors[depth][self.starts[k]..self.starts[k + 1]].iter().all(|&v| self.span.add(v, 0)) {
                continue;
            }
            self.set[depth] = k;
            let mut residual = self.residuals[depth];
            self.span.reduce(&mut residual, &mut 0);
            self.residuals[depth + 1] = residual;
            // Everything after it, modulo it as well: in full, or, for
            // the last depth, the folds alone, which are linear, so that
            // a vector's is its own and those of the rows it takes.
            let after = self.starts[k + 1];
            let (done, next) = self.folds.split_at_mut(depth + 1);
            let (folds, next_folds) = (&done[depth][after..], &mut next[0][after..]);
            let more = if depth + 2 == self.size {
                let span = &self.span;
                let row_folds: [u64; 8] = std::array::from_fn(|r| if r < span.len { fold(&span.rows[r]) } else { 0 });
                let pivots = &span.pivot_at[..span.len];
                for ((out, &folded), v) in next_folds.iter_mut().zip(folds).zip(&self.vectors[depth][after..]) {
                    *out = pivots.iter().zip(&row_folds).fold(folded, |out, (&(word, bit), &row)| out ^ row & 0u64.wrapping_sub(u64::from(v[word] & bit != 0)));
                }
                std::mem::swap(&mut self.prefix, &mut self.span);
                self.leaves(depth + 1, k + 1, f)
            } else {
                let (done, next) = self.vectors.split_at_mut(depth + 1);
                for ((out, folded), &v) in next[0][after..].iter_mut().zip(next_folds).zip(&done[depth][after..]) {
                    let mut v = v;
                    self.span.reduce(&mut v, &mut 0);
                    *out = v;
                    *folded = fold(&v);
                }
                self.descend(depth + 1, k + 1, f)
            };
            if !more {
                return false;
            }
        }
        true
    }

    /// The sets whose first `depth` codewords are those chosen, the last
    /// one from `from` on. False once the work was called off.
    ///
    /// Most sets fail, the residual lying outside their span, which their
    /// folds ([`fold`]) mostly show at once: the folds of the last
    /// codeword's columns and of the residual independent, the vectors
    /// are, and the residual is not among the columns. Only where the
    /// folds are dependent, which every set that solves is, is the set
    /// reduced in full: the codeword's columns at the depth before,
    /// modulo the codeword chosen there, in `prefix`.
    fn leaves(&mut self, depth: usize, from: usize, f: &mut impl FnMut(&[usize], u32)) -> bool {
        let c = self.starts.len() - 1;
        let residual_fold = fold(&self.residuals[depth]);
        for k in from..c {
            self.looked += 1;
            if self.looked.is_multiple_of(256) && work::cancelled() {
                return false;
            }
            if independent_words(&self.folds[depth][self.starts[k]..self.starts[k + 1]], residual_fold) {
                continue;
            }
            self.span.clear();
            let before = depth.saturating_sub(1);
            let independent = self.vectors[before][self.starts[k]..self.starts[k + 1]].iter().all(|&v| {
                let mut v = v;
                if depth > 0 {
                    self.prefix.reduce(&mut v, &mut 0);
                }
                self.span.add(v, 0)
            });
            if !independent {
                continue;
            }
            let mut residual = self.residuals[depth];
            self.span.reduce(&mut residual, &mut 0);
            if residual.iter().all(|&x| x == 0) {
                self.set[depth] = k;
                f(&self.set, self.combination());
            }
        }
        true
    }

    /// The combination of the columns of the set in hand that makes the
    /// residual, which they do, and in one way only.
    fn combination(&self) -> u32 {
        let mut whole: Span<W, 24, true> = Span::new();
        for (which, &k) in self.set.iter().enumerate() {
            for (b, &v) in self.vectors[0][self.starts[k]..self.starts[k + 1]].iter().enumerate() {
                let independent = whole.add(v, 1 << (8 * which + b));
                debug_assert!(independent);
            }
        }
        let (mut residual, mut combination) = (self.residuals[0], 0);
        whole.reduce(&mut residual, &mut combination);
        debug_assert!(residual.iter().all(|&x| x == 0));
        combination
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_trials_reach_said_is_the_one_they_go() {
        // A 1-H block, of odd error correction: every codeword and every
        // triple, and every set of five is past the default.
        assert_eq!(trial_reach(26, 17, TRIAL_SETS), TrialReach { past: 2, first_size: 1, sets: 26 + 2_600, largest: 2_600, further: Some(26 + 2_600 + 65_780) });
        // A 40-H block, of even: every pair, and every set of four is past it.
        assert_eq!(trial_reach(45, 30, TRIAL_SETS), TrialReach { past: 1, first_size: 2, sets: 990, largest: 990, further: Some(990 + 148_995) });
        // Nothing goes past where the error correction is two.
        assert_eq!(trial_reach(5, 2, TRIAL_SETS), TrialReach { past: 0, first_size: 2, sets: 0, largest: 0, further: None });
        // Past the standard's limit, and confirmed by what a block read
        // certain must show, in the trials' stage of a decode, standing
        // alone: where nothing is reserved, as at 40-H and 5-Q, as far
        // as the sets go, the pairs one past the code and the fours two.
        use crate::decode::pipeline::{DecodeOptions, judged_verifying_bits, verifying_bits};
        use crate::symbol::SymbolVersion;
        let stage = DecodeOptions { erase_by_trial: true, ..DecodeOptions::default().first_stage() };
        let (needed, judged) = (verifying_bits(SymbolVersion::Qr(1), &stage, false), judged_verifying_bits(&stage));
        assert_eq!(trial_reach(45, 30, TRIAL_SETS).within_standard(45, 30, 0, needed, judged), 1);
        assert_eq!(trial_reach(33, 18, TRIAL_SETS).within_standard(33, 18, 0, needed, judged), 2);
        // In a small block the sets cost what their solutions have to
        // spare: at M3-M the pairs, fours and sixes reach three past the
        // code's four, and as the trials count them confirm none, which
        // were once said to be corrected. The block itself confirms the
        // first of them, five wrong, whatever the sets cost, and a decode
        // so verifies it; six it does not. They were then said to confirm
        // none, as were M4-M's, and M4-Q's one where they confirm two.
        let m3_m = trial_reach(17, 8, TRIAL_SETS);
        assert_eq!((m3_m.past, m3_m.within_standard(17, 8, 0, needed, u32::MAX), m3_m.within_standard(17, 8, 0, needed, judged)), (3, 0, 1));
        assert_eq!(trial_reach(24, 10, TRIAL_SETS).within_standard(24, 10, 0, needed, judged), 1);
        assert_eq!(trial_reach(24, 14, TRIAL_SETS).within_standard(24, 14, 0, needed, judged), 2);
        // What the block itself makes of a codeword so many codewords
        // off, read whole and certain, is that count, less the choice's
        // bit ([`judged`]).
        let mut codeword: Vec<u8> = (1..=9).collect();
        codeword.extend(rs_encode(&codeword, 8));
        let mut block: Vec<CwRead> = codeword.iter().map(|&c| CwRead::known(c)).collect();
        for cw in &mut block[..5] {
            cw.value ^= 0xA5;
        }
        let certain = vec![[CERTAIN; 8]; 17];
        let solution = super::judged(&block, &certain, 8, 0, codeword, 0).expect("the block judges it");
        assert_eq!(i64::from(solution.confidence_bits()), list_solution_balance(17, 8, 5) - 1);
        // Where any codeword is reserved, nothing: the codewords a set
        // erases are read, and wrong, as the standard counts errors
        // ([`keeps_reserve`]), so a solution past what it corrects spends
        // the reserve however it was reached. At 1-H the triples were once
        // said to reach nine wrong within it, at 1-M the fours six, and at
        // 3-L, where the least budget tries the single codewords alone,
        // the triples a codeword past it, the codewords they erased taken
        // for erasures.
        assert_eq!(trial_reach(26, 17, TRIAL_SETS).within_standard(26, 17, 1, VERIFIED_BITS, VERIFIED_BITS), 0);
        assert_eq!(trial_reach(26, 10, TRIAL_SETS).within_standard(26, 10, 2, VERIFIED_BITS, VERIFIED_BITS), 0);
        assert_eq!(trial_reach(44, 10, TRIAL_SETS).within_standard(44, 10, 2, VERIFIED_BITS, VERIFIED_BITS), 0);
        let singles = trial_reach(70, 15, TRIAL_SETS_MIN);
        assert_eq!((singles.past, singles.within_standard(70, 15, 1, VERIFIED_BITS, VERIFIED_BITS)), (1, 0));
        assert_eq!(trial_reach(70, 15, TRIAL_SETS).within_standard(70, 15, 1, VERIFIED_BITS, VERIFIED_BITS), 0);
        assert_eq!(trial_reach(10, 5, TRIAL_SETS).within_standard(10, 5, 3, VERIFIED_BITS, VERIFIED_BITS), 0);
        assert_eq!(trial_reach(26, 7, TRIAL_SETS_MAX).within_standard(26, 7, 3, VERIFIED_BITS, VERIFIED_BITS), 0);
        // The least budget reaches one codeword past the code in any
        // block that has room to erase in, Model 1's largest, of 238
        // codewords, among them.
        use crate::symbol::Family;
        use crate::symbol::tables::block_shape;
        assert_eq!(trial_reach(238, 46, TRIAL_SETS_MIN), TrialReach { past: 1, first_size: 2, sets: 28_203, largest: 28_203, further: Some(28_203 + 130_344_865) });
        for family in Family::ALL {
            for version in family.versions() {
                for &ec in version.ec_levels() {
                    let shape = block_shape(version, ec).unwrap();
                    let longest = shape.data_len(shape.blocks - 1) + shape.ec_per_block;
                    if shape.ec_per_block > 2 {
                        assert!(trial_reach(longest, shape.ec_per_block, TRIAL_SETS_MIN).past >= 1, "{version} {ec:?}");
                    }
                }
            }
        }
    }

    /// Small deterministic generator so the tests need no extra dependency.
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

    /// The one solution a solver that tries many ways found, when it
    /// found exactly one: rivals it could not tell apart count as none.
    fn only<T>(mut found: Vec<T>) -> Option<T> {
        if found.len() == 1 { found.pop() } else { None }
    }

    #[test]
    fn hello_world_1m() {
        let data = [
            0x20, 0x5B, 0x0B, 0x78, 0xD1, 0x72, 0xDC, 0x4D, 0x43, 0x40, 0xEC, 0x11, 0xEC, 0x11, 0xEC, 0x11,
        ];
        assert_eq!(rs_encode(&data, 10), [196, 35, 39, 119, 235, 215, 231, 226, 93, 23]);
        let mut block = data.to_vec();
        block.extend(rs_encode(&data, 10));
        assert!(rs_syndromes(&block, 10).iter().all(|&s| s == 0));
    }

    #[test]
    fn clean_block_reports_full_slack() {
        let mut rng = Rng(1);
        let block = codeword(&mut rng, 20, 10);
        let read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        let sol = rs_decode(&read, 10).unwrap();
        assert_eq!(sol.corrected, block);
        assert!(sol.errata.is_empty());
        assert_eq!((sol.errors, sol.erasures, sol.slack), (0, 0, 10));
    }

    #[test]
    fn recovers_everything_within_capacity() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for round in 0..3000 {
            let ec_len = 2 + rng.below(29);
            let data_len = 1 + rng.below(120);
            let n = data_len + ec_len;
            let block = codeword(&mut rng, data_len, ec_len);

            let erasures = rng.below(ec_len + 1);
            let errors = rng.below((ec_len - erasures) / 2 + 1);
            let positions = rng.distinct((erasures + errors).min(n), n);
            let (erased, wrong) = positions.split_at(erasures.min(positions.len()));

            let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
            for &i in erased {
                read[i] = CwRead::unknown();
            }
            for &i in wrong {
                read[i].value ^= rng.nonzero();
            }

            let sol = rs_decode(&read, ec_len).unwrap_or_else(|e| panic!("round {round}: {e:?}"));
            assert_eq!(sol.corrected, block, "round {round}");
            assert_eq!(sol.erasures, erased.len());
            assert_eq!(sol.errors, wrong.len());
            assert_eq!(sol.slack, ec_len - erased.len() - 2 * wrong.len());
            for &i in wrong {
                let e = sol.errata.iter().find(|e| e.index == i).expect("error reported");
                assert_eq!(e.magnitude, read[i].value ^ block[i]);
                assert!(!e.was_erasure);
            }
            for &i in erased {
                assert!(sol.errata.iter().any(|e| e.index == i && e.was_erasure));
            }
            assert_eq!(sol.errata.len(), positions.len());
        }
    }

    #[test]
    fn too_many_erasures_fails_deterministically() {
        let mut rng = Rng(7);
        let block = codeword(&mut rng, 30, 8);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        for i in rng.distinct(9, block.len()) {
            read[i] = CwRead::unknown();
        }
        assert_eq!(
            rs_decode(&read, 8),
            Err(RsFailure::TooManyErasures { erasures: 9, capacity: 8 })
        );
    }

    #[test]
    fn beyond_capacity_is_never_silently_wrong_with_slack_claimed() {
        // With more errors than capacity the decoder either fails or lands on
        // a different codeword; it must never return a non-codeword.
        let mut rng = Rng(99);
        let (mut failed, mut miscorrected) = (0, 0);
        for _ in 0..2000 {
            let ec_len = 6 + 2 * rng.below(8);
            let block = codeword(&mut rng, 40, ec_len);
            let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
            let wrong = ec_len / 2 + 1 + rng.below(3);
            for i in rng.distinct(wrong, block.len()) {
                read[i].value ^= rng.nonzero();
            }
            match rs_decode(&read, ec_len) {
                Err(_) => failed += 1,
                Ok(sol) => {
                    assert!(rs_syndromes(&sol.corrected, ec_len).iter().all(|&s| s == 0));
                    assert_ne!(sol.corrected, block);
                    miscorrected += 1;
                }
            }
        }
        assert!(failed > miscorrected * 20, "failed {failed}, miscorrected {miscorrected}");
    }

    #[test]
    fn known_bits_in_erased_codewords_act_as_checks() {
        let mut rng = Rng(3);
        let ec_len = 6;
        let block = codeword(&mut rng, 20, ec_len);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        let erased = rng.distinct(ec_len, block.len());
        for &i in &erased {
            // Top nibble unreadable, bottom nibble read correctly.
            read[i] = CwRead { value: block[i] & 0x0F, known: 0x0F, pinned: 0 };
        }
        let sol = rs_decode(&read, ec_len).unwrap();
        assert_eq!(sol.corrected, block);
        assert_eq!(sol.slack, 0);
        assert_eq!(sol.check_bits, 4 * ec_len as u32);
        assert!(sol.confidence_bits() >= 8);

        // Corrupt an unerased codeword: the zero-slack solution now lands on
        // the wrong values and the known nibbles expose it: they agree
        // with it no better than chance, and nothing else confirms it.
        let victim = (0..block.len()).find(|i| !erased.contains(i)).unwrap();
        read[victim].value ^= 0x5A;
        match rs_decode(&read, ec_len) {
            Err(failure) => assert!(matches!(failure, RsFailure::KnownBitConflict { .. }), "{failure:?}"),
            Ok(sol) => assert!(sol.corrected != block && sol.confidence_bits() < VERIFIED_BITS && sol.flipped_known_bits > 0, "{sol:?}"),
        }
    }

    #[test]
    fn known_bits_a_solution_overturns_count_against_it() {
        // Nineteen of twenty error correction codewords spent on erasures
        // whose seven known bits each are noise, and the codewords read in
        // full all right: the decoder lands on a codeword with one
        // codeword of slack, and half the known bits it was given
        // contradict it. They agree with it as with any codeword, and
        // confirm nothing: it stands on its one codeword of slack alone.
        // Counted one for and two against, they once refused it.
        let mut rng = Rng(21);
        let ec_len = 20;
        let block = codeword(&mut rng, 80, ec_len);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        let erased = rng.distinct(ec_len - 1, block.len());
        for &i in &erased {
            read[i] = CwRead { value: rng.byte() & 0x7F, known: 0x7F, pinned: 0 };
        }
        let sol = rs_decode(&read, ec_len).unwrap();
        assert_eq!((sol.slack, sol.located_bits), (1, 0));
        assert!(sol.check_bits <= 2 && sol.flipped_known_bits > 7 * 19 / 3, "{} checks, {} overturned", sol.check_bits, sol.flipped_known_bits);

        // The same erasures with their known bits read right but for one
        // in each of three codewords: the solution stands, and is
        // credited how rarely chance agrees with so many of the bits.
        for &i in &erased {
            read[i] = CwRead { value: block[i] & 0x7F, known: 0x7F, pinned: 0 };
        }
        for &i in &erased[..3] {
            read[i].value ^= 0x10;
        }
        let sol = rs_decode(&read, ec_len).unwrap();
        assert_eq!(sol.corrected, block);
        assert_eq!((sol.slack, sol.flipped_known_bits), (1, 3));
        assert_eq!(sol.check_bits, doubted_agreement(7 * 19 - 3, 3));
        assert_eq!(sol.check_bits, 114);

        // Noise in the erased codewords of a block with redundancy to
        // spare says nothing against it: sixteen codewords of slack
        // outweigh the bits of four.
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        for &i in &erased[..4] {
            read[i] = CwRead { value: !block[i] & 0x7F, known: 0x7F, pinned: 0 };
        }
        let sol = rs_decode(&read, ec_len).unwrap();
        assert_eq!(sol.corrected, block);
        assert_eq!((sol.slack, sol.check_bits, sol.flipped_known_bits), (16, 0, 28));
        assert!(sol.confidence_bits() >= 8 * 16);
    }

    #[test]
    fn a_codeword_refused_for_the_bits_it_overturns_is_noted_all_the_same() {
        // Seven codewords of a block with seven of error correction read
        // in seven bits each, every one of them against the block, and the
        // rest read right: the codewords read in full fix the block, which
        // overturns every known bit of the seven and has nothing to spare,
        // and it is refused. It is a codeword of the code for all that,
        // and where those bits are faint the nearest there is; the solvers
        // that note what they reach once forgot it.
        let mut rng = Rng(0x5EE);
        let ec_len = 7;
        let block = codeword(&mut rng, 19, ec_len);
        let wrong = rng.distinct(ec_len, block.len());
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        for &i in &wrong {
            read[i] = CwRead { value: !block[i] & 0x7F, known: 0x7F, pinned: 0 };
        }
        let mut seen = Sightings::default();
        let refused = rs_decode_reserving_noting(&read, ec_len, 0, &mut seen);
        assert!(matches!(refused, Err(RsFailure::KnownBitConflict { .. })), "{refused:?}");
        assert_eq!(seen.codewords().collect::<Vec<_>>(), [&block[..]]);

        // Read in full but against the block in every bit, and faint: the
        // doubting reaches the block only with all seven doubted, where it
        // is refused as above, and notes it.
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        let mut reliability = vec![200u8; block.len()];
        for &i in &wrong {
            read[i] = CwRead::known(!block[i]);
            reliability[i] = 10;
        }
        let mut seen = Sightings::default();
        assert!(rs_decode_doubting(&read, &reliability, ec_len, 0, &mut seen).is_none());
        assert!(seen.codewords().any(|codeword| codeword == block), "{:?}", seen.codewords().collect::<Vec<_>>());
    }

    #[test]
    fn pins_veto_solutions_that_change_them() {
        let mut rng = Rng(11);
        let ec_len = 4;
        let block = codeword(&mut rng, 12, ec_len);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        // A single error is correctable...
        read[3].value ^= 0x40;
        assert_eq!(rs_decode(&read, ec_len).unwrap().corrected, block);
        // ...but not if the wrong value is pinned.
        read[3].pinned = 0xFF;
        assert_eq!(rs_decode(&read, ec_len), Err(RsFailure::PinConflict { index: 3 }));
    }

    #[test]
    fn shortened_blocks_reject_out_of_range_roots() {
        // Short blocks leave most of the 255 locators unused; uncorrectable
        // patterns frequently point there and must be reported as failures.
        let mut rng = Rng(5);
        let mut failures = 0;
        for _ in 0..500 {
            let block = codeword(&mut rng, 6, 4);
            let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
            for i in rng.distinct(3, block.len()) {
                read[i].value ^= rng.nonzero();
            }
            if rs_decode(&read, 4).is_err() {
                failures += 1;
            }
        }
        assert!(failures > 400, "{failures}");
    }

    #[test]
    fn bit_solver_beats_codeword_erasures_on_scattered_damage() {
        let mut rng = Rng(21);
        for _ in 0..200 {
            let ec_len = 4 + rng.below(20);
            let data_len = 10 + rng.below(60);
            let block = codeword(&mut rng, data_len, ec_len);
            let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
            // One unknown bit in each of 2r codewords: twice what erasure
            // decoding can take, a quarter of the bit budget.
            let hit = rng.distinct((2 * ec_len).min(block.len()), block.len());
            for &i in &hit {
                let bit = 1u8 << rng.below(8);
                read[i].known &= !bit;
                read[i].value &= !bit;
            }
            assert!(matches!(rs_decode(&read, ec_len), Err(RsFailure::TooManyErasures { .. })));
            let sol = solve_bit_erasures(&read, ec_len).unwrap();
            assert!(sol.is_complete());
            assert_eq!(sol.corrected, block);
            assert_eq!(sol.unknown_bits, hit.len());
            assert_eq!(sol.spare_equations, 8 * ec_len - hit.len());
            assert_eq!(sol.solved.iter().map(|m| m.count_ones() as usize).sum::<usize>(), hit.len());
        }
    }

    #[test]
    fn bit_solver_agrees_with_erasure_decoder() {
        let mut rng = Rng(22);
        for _ in 0..200 {
            let ec_len = 2 + rng.below(16);
            let data_len = 5 + rng.below(40);
            let block = codeword(&mut rng, data_len, ec_len);
            let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
            let erased = rng.below(ec_len + 1).max(1);
            for i in rng.distinct(erased, block.len()) {
                read[i] = CwRead::unknown();
            }
            let a = rs_decode(&read, ec_len).unwrap();
            let b = solve_bit_erasures(&read, ec_len).unwrap();
            assert_eq!(a.corrected, block);
            assert_eq!(b.corrected, block);
        }
    }

    #[test]
    fn bit_solver_detects_errors_and_underdetermination() {
        let mut rng = Rng(23);
        let ec_len = 6;
        let block = codeword(&mut rng, 20, ec_len);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        assert_eq!(solve_bit_erasures(&read, ec_len), Err(BitSolveFailure::NoUnknowns));

        read[0].known = 0xFE;
        read[0].value &= 0xFE;
        read[9].value ^= 0x10;
        assert_eq!(solve_bit_erasures(&read, ec_len), Err(BitSolveFailure::Inconsistent));
        read[9].value ^= 0x10;

        // Seven whole codewords against six of redundancy: the solutions
        // reach every unknown bit, so nothing is settled.
        for cw in read.iter_mut().take(ec_len + 1) {
            *cw = CwRead::unknown();
        }
        assert_eq!(
            solve_bit_erasures(&read, ec_len),
            Err(BitSolveFailure::Underdetermined { rank: 8 * ec_len, unknown_bits: 8 * (ec_len + 1) })
        );
        for cw in read.iter_mut().take(2 * ec_len) {
            *cw = CwRead::unknown();
        }
        assert!(matches!(solve_bit_erasures(&read, ec_len), Err(BitSolveFailure::TooManyUnknowns { .. })));
    }

    /// The block as read is solved off a system that carries its known
    /// bits, for the doubting to share with its trials, just as it is
    /// off one that does not: complete, partial, inconsistent or
    /// underdetermined alike.
    #[test]
    fn a_system_carrying_its_known_bits_solves_as_one_without_them() {
        let mut rng = Rng(31);
        let mut seen = [0usize; 4];
        for _ in 0..400 {
            let ec_len = 2 + rng.below(10);
            let data_len = 2 + rng.below(30);
            let block = codeword(&mut rng, data_len, ec_len);
            let n = block.len();
            let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
            for _ in 0..rng.below(12 * ec_len) {
                let (i, b) = (rng.below(n), 1u8 << rng.below(8));
                read[i].known &= !b;
                read[i].value &= !b;
            }
            for _ in 0..rng.below(3) {
                let i = rng.below(n);
                read[i].value ^= rng.byte() & read[i].known;
            }
            for _ in 0..rng.below(3) {
                let i = rng.below(n);
                read[i].pinned = read[i].known & rng.byte();
            }
            let alone = solve_bit_erasures(&read, ec_len);
            let carried = unknowns_within_reach(&read, ec_len).and_then(|()| BitSystem::build(&read, ec_len, true).solve(&read));
            assert_eq!(carried, alone);
            seen[match alone {
                Ok(solution) if solution.is_complete() => 0,
                Ok(_) => 1,
                Err(BitSolveFailure::Inconsistent) => 2,
                Err(_) => 3,
            }] += 1;
        }
        assert!(seen.iter().all(|&n| n > 0), "{seen:?}");
    }

    #[test]
    fn partial_solutions_settle_what_they_can() {
        let mut rng = Rng(24);
        let ec_len = 6;
        let mut partial_seen = 0;
        for _ in 0..50 {
            let block = codeword(&mut rng, 20, ec_len);
            let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
            // Seven codewords lose all but one bit each: one more unknown
            // bit than there are equations, so the system cannot close, but
            // the surviving bits cut the solutions down to a line, and the
            // bits that line does not touch are settled.
            for i in rng.distinct(ec_len + 1, block.len()) {
                let keep = 1u8 << rng.below(8);
                read[i] = CwRead { value: block[i] & keep, known: keep, pinned: 0 };
            }
            let Ok(sol) = solve_bit_erasures(&read, ec_len) else { continue };
            assert!(!sol.is_complete());
            assert_eq!(sol.unknown_bits, 7 * (ec_len + 1));
            assert!(sol.solved_bits > 0 && sol.solved_bits < sol.unknown_bits);
            for (i, (&got, &mask)) in sol.corrected.iter().zip(&sol.solved).enumerate() {
                assert_eq!(mask & read[i].known, 0, "only unknown bits are solved");
                assert_eq!(got & mask, block[i] & mask, "codeword {i}");
                assert_eq!(got & read[i].known, read[i].value, "known bits untouched");
            }
            partial_seen += 1;
        }
        assert!(partial_seen > 40, "{partial_seen}");
    }

    #[test]
    fn doubting_erases_the_least_reliable_first() {
        let mut rng = Rng(31);
        let ec_len = 10;
        let block = codeword(&mut rng, 30, ec_len);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        let mut reliability = vec![CERTAIN; block.len()];
        // Seven wrong codewords, two past what errors alone can fix, all
        // reading as doubtful; three clean codewords read as a little
        // doubtful too.
        let picked = rng.distinct(10, block.len());
        for (n, &i) in picked.iter().enumerate() {
            if n < 7 {
                read[i].value ^= rng.nonzero();
                reliability[i] = 10 + n as u8;
            } else {
                reliability[i] = 40 + n as u8;
            }
        }
        assert!(rs_decode(&read, ec_len).is_err());
        let (doubted, sol) = rs_decode_doubting(&read, &reliability, ec_len, 0, &mut Sightings::default()).unwrap();
        assert_eq!(sol.corrected, block);
        // Erasing all seven wrong ones leaves three codewords of slack,
        // more confirmation than any other count of doubted codewords.
        assert_eq!(doubted, 7);
        assert_eq!((sol.errors, sol.erasures, sol.slack), (0, 7, 3));

        // Certain codewords are never doubted, however wrong.
        reliability.iter_mut().for_each(|r| *r = CERTAIN);
        assert_eq!(rs_decode_doubting(&read, &reliability, ec_len, 0, &mut Sightings::default()), None);

        // A solution with nothing to spare is not accepted.
        let ec_len = 4;
        let block = codeword(&mut rng, 12, ec_len);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        let reliability: Vec<u8> = (0..block.len()).map(|i| i as u8).collect();
        for cw in read.iter_mut().take(4) {
            cw.value ^= rng.nonzero();
        }
        assert_eq!(rs_decode_doubting(&read, &reliability, ec_len, 0, &mut Sightings::default()), None);
        // Nor is one with a single codeword spare whose doubted codewords
        // agree with it little better than chance: one syndrome byte and
        // a bit or two of agreement are all that stand behind it, which
        // noise passes once in a few hundred tries, and doubting tries a
        // count of codewords at a time.
        read[3] = CwRead::known(block[3]);
        let plain = rs_decode_doubted(&read, &[0, 1, 2], ec_len, 0, &mut Sightings::default()).unwrap();
        assert_eq!((plain.slack, plain.corrected == block), (1, true));
        assert!((8..=10).contains(&plain.confidence_bits()), "{}", plain.confidence_bits());
        assert_eq!(rs_decode_doubting(&read, &reliability, ec_len, 0, &mut Sightings::default()), None);

        // Codewords misread in one faint bit each bear the solution out
        // by the bits they got right, and one spare codeword is enough.
        let ec_len = 8;
        let block = codeword(&mut rng, 24, ec_len);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        let reliability: Vec<u8> = (0..block.len()).map(|i| i as u8).collect();
        for (i, cw) in read.iter_mut().take(7).enumerate() {
            cw.value ^= 1 << i;
        }
        let (doubted, sol) = rs_decode_doubting(&read, &reliability, ec_len, 0, &mut Sightings::default()).unwrap();
        assert_eq!((doubted, sol.slack), (7, 1));
        // Charged for the eight counts of codewords doubting tried.
        assert_eq!(sol.confidence_bits(), 8 + doubted_agreement(49, 7) - tries_charge(8));
        assert_eq!(sol.uncharged_bits(), 8 + doubted_agreement(49, 7));
        assert_eq!(sol.corrected, block);
        // With two codewords reserved for detection that solution is
        // beyond the standard's limit, and doubting offers it so marked,
        // having none within it.
        let (_, beyond) = rs_decode_doubting(&read, &reliability, ec_len, 2, &mut Sightings::default()).unwrap();
        assert!(!beyond.within_standard() && beyond.corrected == block);
    }

    #[test]
    fn noise_does_not_pass_for_a_doubted_or_chased_solution() {
        // Blocks of random codewords, every one of them doubtful: nothing
        // is there to find, and what the doubting returns, if anything,
        // claims no more than it shows: a block of noise is offered a
        // solution confirmed by `b` bits, charged for the counts tried,
        // no oftener than once in `2^b`, and never by three codewords'
        // worth. Before a doubted solution's agreement was weighed
        // against chance and charged for the counts, one block in two
        // hundred of these verified.
        let mut rng = Rng(0x0153);
        let mut claimed = [0u32; 64];
        let mut blocks = 0;
        for (n, ec_len) in [(26usize, 7usize), (26, 10), (26, 17), (44, 28), (13, 7)] {
            for _ in 0..600 {
                let read: Vec<CwRead> = (0..n).map(|_| CwRead::known(rng.byte())).collect();
                let reliability: Vec<u8> = (0..n).map(|_| rng.byte().min(CERTAIN - 1)).collect();
                if let Some((_, solution)) = rs_decode_doubting(&read, &reliability, ec_len, 0, &mut Sightings::default()) {
                    assert!(solution.confidence_bits() >= VERIFIED_BITS && solution.confidence_bits() < 3 * VERIFIED_BITS, "({n}, {ec_len}): {solution:?}");
                    claimed[solution.confidence_bits() as usize] += 1;
                }
                blocks += 1;
            }
        }
        for bits in VERIFIED_BITS as usize..64 {
            let at_least: u32 = claimed[bits..].iter().sum();
            assert!(f64::from(at_least) <= 2.0 * f64::from(blocks) / 2f64.powi(bits as i32) + 3.0, "{at_least} of {blocks} claim {bits} bits or more");
        }
        // M1's block has two codewords and the standard reserves both: a
        // read of it is right or wrong, and no pattern of flips is tried.
        // Without the reserve, flipping the one faint bit is one pattern,
        // charged nothing, and the sixteen bits of the check confirm it;
        // as one of a thousand patterns it would stand on six.
        let block = codeword(&mut rng, 5, 2);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        read[1].value ^= 0x04;
        let mut reliability = vec![[200u8; 8]; block.len()];
        reliability[1][2] = 3;
        assert!(rs_decode_chasing(&read, &reliability, 2, 2, CHASE_BITS).is_empty());
        assert!(only(rs_decode_chasing(&read, &reliability, 2, 0, 1)).is_some_and(|(_, s)| s.corrected == block && s.confidence_bits() == 16));
        assert!(rs_decode_chasing(&read, &reliability, 2, 0, CHASE_BITS).is_empty());
    }

    #[test]
    fn doubted_codewords_keep_their_agreeing_bits_as_checks() {
        // Beyond chance: a doubted codeword agrees with any solution in
        // half its bits, so four right of eight is worth nothing and
        // seven of eight a few bits, while seven codewords right in
        // seven bits of eight are worth far more than their one slack
        // codeword. The tail of the binomial, to the bit: nine of the
        // two hundred and fifty-six ways eight coins fall show seven or
        // more heads.
        assert_eq!(doubted_agreement(7, 1), 4);
        assert_eq!(doubted_agreement(4, 4), 0);
        assert_eq!(doubted_agreement(0, 0), 0);
        // Tails that are a whole number of bits: a bare majority of an
        // odd count is a coin's toss, and six of seven, eight of the
        // hundred and twenty-eight ways, four bits. The sum in
        // logarithms once landed a hair above the whole number, and the
        // floor a bit short.
        assert_eq!((doubted_agreement(1, 0), doubted_agreement(2, 1), doubted_agreement(11, 10), doubted_agreement(31, 30)), (1, 1, 1, 1));
        assert_eq!((doubted_agreement(6, 1), doubted_agreement(3, 1), doubted_agreement(255, 0)), (4, 1, 255));
        assert_eq!(doubted_agreement(49, 7), 27);
        assert_eq!(doubted_agreement(56, 8), 31);
        // Over a few hundred bits three standard deviations overshot
        // the tail: 164 right of 256 is seventeen bits, where they said
        // twenty-four, and 205 of 256 seventy-five, where they said a
        // hundred and six.
        assert_eq!(doubted_agreement(164, 92), 17);
        assert_eq!(doubted_agreement(205, 51), 75);

        let mut rng = Rng(51);
        let ec_len = 10;
        let block = codeword(&mut rng, 30, ec_len);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        let mut reliability = vec![CERTAIN; block.len()];
        // Seven codewords each wrong in one faint bit.
        let picked = rng.distinct(7, block.len());
        for (n, &i) in picked.iter().enumerate() {
            read[i].value ^= 1 << (n % 8);
            reliability[i] = 10 + n as u8;
        }
        let (doubted, sol) = rs_decode_doubting(&read, &reliability, ec_len, 0, &mut Sightings::default()).unwrap();
        assert_eq!(sol.corrected, block);
        assert_eq!((doubted, sol.erasures, sol.slack), (7, 7, 3));
        // Every doubted codeword counts as an erasure, with the bits the
        // solution agreed with, beyond the one it overturned, as checks.
        assert_eq!(sol.flipped_known_bits, 7);
        assert_eq!(sol.check_bits, doubted_agreement(49, 7));
        assert_eq!(sol.confidence_bits(), 24 + doubted_agreement(49, 7) - tries_charge(7));
        assert!(sol.errata.iter().all(|e| e.was_erasure && e.magnitude.count_ones() == 1));
        assert_eq!(sol.errata.len(), 7);
        // Erasing them outright would have confirmed nothing beyond the slack.
        let mut blanked = read.clone();
        for &i in &picked {
            blanked[i] = CwRead::unknown();
        }
        let plain = rs_decode(&blanked, ec_len).unwrap();
        assert_eq!((plain.check_bits, plain.confidence_bits()), (0, 24));

        // At zero slack a doubted solution is held to the agreement alone:
        // ten codewords right in seven bits of eight pass, ten corrupted
        // at random, which agree in half their bits, do not.
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        reliability.iter_mut().for_each(|r| *r = CERTAIN);
        let picked = rng.distinct(10, block.len());
        for (n, &i) in picked.iter().enumerate() {
            read[i].value ^= 1 << (n % 8);
            reliability[i] = 10 + n as u8;
        }
        let (doubted, sol) = rs_decode_doubting(&read, &reliability, ec_len, 0, &mut Sightings::default()).unwrap();
        assert_eq!((doubted, sol.slack, sol.corrected == block), (10, 0, true));
        assert_eq!(sol.confidence_bits(), doubted_agreement(70, 10) - tries_charge(10));
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        for &i in &picked {
            read[i].value = rng.byte();
        }
        let garbage = rs_decode_doubting(&read, &reliability, ec_len, 0, &mut Sightings::default());
        assert!(garbage.as_ref().is_none_or(|(_, s)| s.slack >= 1), "{garbage:?}");
    }

    #[test]
    fn doubting_bits_solves_scattered_faint_misreads() {
        let mut rng = Rng(61);
        for round in 0..100 {
            let ec_len = 4 + rng.below(20);
            let data_len = 10 + rng.below(60);
            let block = codeword(&mut rng, data_len, ec_len);
            let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
            let mut reliability = vec![[200u8; 8]; block.len()];
            // One faint wrong bit in each of more codewords than there are
            // EC codewords, among a few faint bits that are right.
            let wrong = (ec_len + 2).min(block.len());
            let hit = rng.distinct(wrong + 3, block.len());
            for (n, &i) in hit.iter().enumerate() {
                let bit = rng.below(8);
                if n < wrong {
                    read[i].value ^= 1 << bit;
                }
                reliability[i][bit] = 5 + rng.below(30) as u8;
            }
            // Doubting codewords cannot reach the right answer: the wrong
            // ones outnumber the EC codewords. (Once in a few hundred
            // blocks it lands on a wrong codeword with one of slack, which
            // one syndrome byte cannot tell from the truth.)
            let by_codeword: Vec<u8> = reliability.iter().map(|r| *r.iter().min().unwrap()).collect();
            if let Some((_, sol)) = rs_decode_doubting(&read, &by_codeword, ec_len, 0, &mut Sightings::default()) {
                assert_ne!(sol.corrected, block, "round {round}");
            }
            let doubt = only(solve_bit_erasures_doubting(&read, &reliability, ec_len, Trials::NONE, &mut Sightings::default())).unwrap_or_else(|| panic!("round {round}"));
            assert_eq!(doubt.solution.corrected, block, "round {round}");
            assert!(doubt.doubted_bits >= wrong && doubt.doubted_bits <= wrong + 3, "round {round}: {}", doubt.doubted_bits);
            assert_eq!(doubt.errors(), wrong);
            assert_eq!(doubt.changed_bits(), wrong as u32);
            assert_eq!(doubt.solution.spare_equations, 8 * ec_len - doubt.doubted_bits);
            assert!(doubt.confidence_bits() >= VERIFIED_BITS);
        }
        // Pinned and certain bits are never doubted, and a block with no
        // faint bit has nothing to try.
        let ec_len = 6;
        let block = codeword(&mut rng, 20, ec_len);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        read[4].value ^= 0x08;
        assert!(solve_bit_erasures_doubting(&read, &vec![[CERTAIN; 8]; block.len()], ec_len, Trials::NONE, &mut Sightings::default()).is_empty());
        let mut reliability = vec![[200u8; 8]; block.len()];
        reliability[4][3] = 1;
        assert!(!solve_bit_erasures_doubting(&read, &reliability, ec_len, Trials::NONE, &mut Sightings::default()).is_empty());
        read[4].pinned = 0x08;
        assert!(solve_bit_erasures_doubting(&read, &reliability, ec_len, Trials::NONE, &mut Sightings::default()).iter().all(|d| d.doubted[4] & 0x08 == 0));
    }

    #[test]
    fn chasing_flips_the_faintest_bits() {
        let mut rng = Rng(71);
        let ec_len = 10;
        let block = codeword(&mut rng, 30, ec_len);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        let mut reliability = vec![[200u8; 8]; block.len()];
        // Four solid wrong codewords, one short of capacity, and three
        // more wrong in one faint bit each: seven errors, beyond the code,
        // and erasing the faint three leaves four for seven units of
        // redundancy. Two faint bits that are right come along.
        let picked = rng.distinct(9, block.len());
        for (n, &i) in picked.iter().enumerate() {
            let bit = n % 8;
            if n < 4 {
                read[i].value ^= rng.nonzero();
            } else {
                if n < 7 {
                    read[i].value ^= 1 << bit;
                }
                reliability[i][bit] = 5 + n as u8;
            }
        }
        assert!(rs_decode(&read, ec_len).is_err());
        let by_codeword: Vec<u8> = reliability.iter().map(|r| *r.iter().min().unwrap()).collect();
        assert_eq!(rs_decode_doubting(&read, &by_codeword, ec_len, 0, &mut Sightings::default()), None);
        assert!(solve_bit_erasures_doubting(&read, &reliability, ec_len, Trials::NONE, &mut Sightings::default()).is_empty());
        let (flipped, sol) = only(rs_decode_chasing(&read, &reliability, ec_len, 0, CHASE_BITS)).unwrap();
        assert_eq!(sol.corrected, block);
        assert_eq!((flipped, sol.errors, sol.erasures, sol.slack), (3, 7, 0, 2));
        // Two codewords spare, and the four solid errors the code located
        // among the forty read, less ten bits for the thousand and
        // twenty-three patterns of the ten faintest bits that were tried.
        assert_eq!((sol.located_bits, sol.charge), (located_error_bits(40, 4), 10));
        assert_eq!(sol.confidence_bits(), 16 + 5);
        assert_eq!(sol.errata.len(), 7);
        for e in &sol.errata {
            assert!(!e.was_erasure && e.magnitude == read[e.index].value ^ block[e.index]);
        }
        // Certain bits are never flipped; with three codewords reserved
        // the solution is beyond the standard, and offered so marked.
        assert!(rs_decode_chasing(&read, &vec![[CERTAIN; 8]; block.len()], ec_len, 0, CHASE_BITS).is_empty());
        let (_, beyond) = only(rs_decode_chasing(&read, &reliability, ec_len, 3, CHASE_BITS)).unwrap();
        assert!(!beyond.within_standard() && beyond.corrected == block);
        // One of many decodes flips fewer bits, and never fewer than six.
        assert_eq!((chase_bits(1), chase_bits(2), chase_bits(32), chase_bits(1 << 20)), (10, 9, 6, 6));
        // A pinned bit is never flipped either, however faint.
        read[picked[4]].pinned = 1 << 4;
        assert!(rs_decode_chasing(&read, &reliability, ec_len, 0, CHASE_BITS).iter().all(|(_, s)| s.corrected != block));
    }

    #[test]
    fn a_judged_solution_carries_the_tries_cost_and_takes_the_choice_from_what_confirms_it() {
        // A 5-Q block wrong in three codewords, every bit of each, judged
        // as the block as read makes of its own codeword, as one of
        // thirty-two decodes: too far for the soft count to look, so the
        // hard count confirms it, twelve codewords spare and three errors
        // located among thirty-three read, less one for the choice
        // between the counts and the five bits the decodes cost. The
        // charge is the tries' cost alone, and the choice's bit comes off
        // what the count has beyond the slack's bytes: a caller weighing
        // the solution against its soft count again, charged for the
        // tries it carries and a bit for the choice, charges the choice
        // once. Carried in the charge, the bit was charged there again.
        let mut rng = Rng(0x7D6E);
        let ec_len = 18;
        let block = codeword(&mut rng, 15, ec_len);
        let certain = vec![[CERTAIN; 8]; block.len()];
        let tries_cost = tries_charge(32);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        for &i in &rng.distinct(3, block.len()) {
            read[i].value ^= 0xFF;
        }
        assert!(soft_distance(&read, &certain, &block) > SOFT_COUNT_REACH);
        let hard = list_solution_bits(33, 18, 3);
        let sol = judged(&read, &certain, ec_len, 0, block.clone(), tries_cost).unwrap();
        assert_eq!((sol.errors, sol.slack, sol.check_bits, sol.charge), (3, 12, 0, tries_cost));
        assert_eq!(sol.located_bits, hard - 8 * 12 - 1);
        assert_eq!((sol.confidence_bits(), sol.uncharged_bits()), (hard - 1 - tries_cost, hard - 1));

        // Wrong in one bit each, read on its threshold: the soft count,
        // the hundred and forty-four bits of redundancy less one and the
        // three coin tosses, confirms it where the hard count falls
        // short, and is split the same way.
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        let mut faint = certain.clone();
        for &i in &rng.distinct(3, block.len()) {
            read[i].value ^= 1 << (i % 8);
            faint[i][i % 8] = 0;
        }
        assert_eq!(soft_count_bits(&read, &faint, 8 * ec_len, 0), 140);
        let sol = judged(&read, &faint, ec_len, 0, block.clone(), tries_cost).unwrap();
        assert_eq!((sol.errors, sol.slack, sol.check_bits, sol.charge), (3, 16, 0, tries_cost));
        assert_eq!(sol.located_bits, 140 - 8 * 16 - 1);
        assert_eq!((sol.confidence_bits(), sol.uncharged_bits()), (140 - 1 - tries_cost, 140 - 1));

        // Eighteen codewords half read, their low nibbles unknown, and
        // the rest read right: the codewords read in full fix the half
        // read ones with no redundancy left, so the hard count is a hair
        // less than nothing, noise having the one codeword they fix and
        // the 254th part more the count allows, and the seventy-two
        // known bits of the half read ones, reproduced, are the checks,
        // less that hair, which the count rounds down to a bit. The
        // choice's bit then comes off the checks, the count having
        // nothing beyond the slack's bytes.
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        for &i in &rng.distinct(18, block.len()) {
            read[i] = CwRead { value: block[i] & 0xF0, known: 0xF0, pinned: 0 };
        }
        assert_eq!((list_solution_bits(15, 0, 0), list_solution_balance(15, 0, 0)), (0, -1));
        let sol = judged(&read, &certain, ec_len, 0, block.clone(), tries_cost).unwrap();
        assert_eq!((sol.erasures, sol.slack, sol.located_bits, sol.check_bits, sol.charge), (18, 0, 0, 70, tries_cost));
        assert_eq!((sol.confidence_bits(), sol.uncharged_bits()), (70 - tries_cost, 70));
    }

    #[test]
    fn the_codewords_count_is_charged_for_the_counts_of_errors_the_known_bits_could_carry_as_far() {
        // A 5-Q block wrong in three codewords, every bit of each, and
        // four codewords half read, their low nibbles unknown: too far
        // for the soft count to look, so the codewords' count judges it,
        // the twenty-nine read with fourteen of redundancy and three
        // errors among them, seventy-six bits, and the sixteen known bits
        // of the half read ones, reproduced, ninety-two. Noise has a
        // codeword two errors from the block once in two to the
        // eighty-seven, and sixteen known bits agree with it often enough
        // to make up the five more; and one three errors away is this
        // codeword's own chance. So noise reaches ninety-two through
        // either, and through the counts of fewer errors, which reach it
        // alone between them: three chances, two bits, taken from what
        // the count has beyond the slack's bytes with the choice's bit.
        // Its own count of errors alone was once charged, two bits less.
        let mut rng = Rng(0x7D6E);
        let ec_len = 18;
        let block = codeword(&mut rng, 15, ec_len);
        let certain = vec![[CERTAIN; 8]; block.len()];
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        let picked = rng.distinct(7, block.len());
        for &i in &picked[..3] {
            read[i].value ^= 0xFF;
        }
        for &i in &picked[3..] {
            read[i] = CwRead { value: block[i] & 0xF0, known: 0xF0, pinned: 0 };
        }
        assert!(soft_distance(&read, &certain, &block) > SOFT_COUNT_REACH);
        let hard = list_solution_balance(29, 14, 3);
        assert_eq!((hard, list_solution_balance(29, 14, 2), doubted_agreement(16, 0)), (76, 87, 16));
        assert_eq!(error_counts_reaching(29, 14, 16, hard + 16), 3);
        let sol = judged(&read, &certain, ec_len, 0, block.clone(), 0).unwrap();
        assert_eq!((sol.errors, sol.erasures, sol.slack, sol.check_bits), (3, 4, 8, 16));
        assert_eq!(sol.located_bits, 76 - 8 * 8 - 2 - 1);
        assert_eq!(sol.confidence_bits(), 92 - 2 - 1);
        // Read in full, the four codewords bear nothing that tops up a
        // farther codeword: the count of three errors and those within
        // it are the one chance, and nothing is charged but the choice.
        for &i in &picked[3..] {
            read[i] = CwRead::known(block[i]);
        }
        assert_eq!(error_counts_reaching(33, 18, 0, list_solution_balance(33, 18, 3)), 1);
        let sol = judged(&read, &certain, ec_len, 0, block.clone(), 0).unwrap();
        assert_eq!(sol.confidence_bits(), list_solution_bits(33, 18, 3) - 1);
    }

    #[test]
    fn the_nearest_of_solutions_that_disagree_wins_by_a_clear_margin_or_not_at_all() {
        // A bit overturned costs the likelihood its reading carries, in
        // eighths of a bit: nothing for a coin's toss, about thirteen for
        // one read a bit of likelihood inside its colour, eight over
        // the reliability for one read well, and 263 for a certain one.
        assert_eq!((bit_cost(0), bit_cost(1), bit_cost(8), bit_cost(9), bit_cost(16), bit_cost(24)), (0, 2, 13, 14, 22, 31));
        assert_eq!((bit_cost(200), bit_cost(CERTAIN - 1), bit_cost(CERTAIN)), (208, 262, 263));
        assert!((1..=255u8).all(|r| bit_cost(r) > bit_cost(r - 1)));
        // A certain bit overturned costs 263, a faint one less, one read
        // on its threshold nothing, and a bit that was not read nothing.
        let block = [CwRead::known(0b1010_1010), CwRead { value: 0x0F, known: 0x0F, pinned: 0 }];
        let certain = [[CERTAIN; 8]; 2];
        assert_eq!(soft_distance(&block, &certain, &[0b1010_1010, 0x0F]), 0);
        assert_eq!(soft_distance(&block, &certain, &[0b1010_1011, 0xFF]), 263);
        let mut faint = certain;
        faint[0][0] = 9;
        faint[1][1] = 0;
        assert_eq!(soft_distance(&block, &faint, &[0b1010_1011, 0x0D]), 14);

        let same = |a: &u8, b: &u8| a == b;
        let never = |_: &u8, _: &u8| false;
        assert_eq!(nearest(vec![(300, 1u8), (10, 2), (10, 2)], same, never), vec![2]);
        // Two within the margin of each other are both returned, the
        // nearer first, and a third beyond it is not.
        assert_eq!(nearest(vec![(10 + DISTANCE_MARGIN - 1, 2u8), (10, 1), (10 + DISTANCE_MARGIN, 3)], same, never), vec![1, 2]);
        assert_eq!(nearest(vec![(10 + DISTANCE_MARGIN, 2u8), (10, 1)], same, never), vec![1]);
        assert_eq!(nearest(Vec::<(u32, u8)>::new(), same, never), Vec::<u8>::new());
        // Of those naming the same codeword, the one preferred.
        let better = |a: &(u8, u8), b: &(u8, u8)| a.1 > b.1;
        assert_eq!(nearest(vec![(5, (1, 1)), (5, (1, 7)), (5, (1, 3))], |a, b| a.0 == b.0, better), vec![(1, 7)]);
    }

    #[test]
    fn erasing_by_trial_reaches_one_wrong_codeword_past_the_code() {
        let mut rng = Rng(0x7B1A1);
        for &(data_len, ec_len) in &[(16usize, 18usize), (32, 18), (16, 28), (24, 22), (9, 17)] {
            for round in 0..30 {
                let block = codeword(&mut rng, data_len, ec_len);
                let n = block.len();
                let limit = ec_len / 2;
                let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
                let wrong = rng.distinct(limit + 2, n);
                for &i in &wrong[..limit + 1] {
                    read[i].value ^= rng.nonzero();
                }
                let what = format!("({data_len},{ec_len}) round {round}");
                assert!(rs_decode(&read, ec_len).is_err(), "{what}");
                let certain = vec![[CERTAIN; 8]; n];
                let (erased, solution) = only(rs_decode_trying(&read, &certain, ec_len, 0, TRIAL_SETS, &mut Sightings::default())).unwrap_or_else(|| panic!("{what}"));
                assert_eq!(solution.corrected, block, "{what}");
                assert_eq!(erased.len(), 2 - ec_len % 2, "{what}");
                assert!(erased.iter().all(|i| wrong[..limit + 1].contains(i)), "{what}");
                assert_eq!((solution.errors, solution.erasures), (limit + 1, 0), "{what}");
                assert!(solution.confidence_bits() >= 3 * VERIFIED_BITS, "{what}: {}", solution.confidence_bits());
                // One more is reached by the sets two larger, where the
                // block is small enough for those to fit the budget with
                // the smaller sets before them, and past the trials where
                // it is not.
                read[wrong[limit + 1]].value ^= rng.nonzero();
                let size = 2 - ec_len % 2;
                let fits = sets_of(n, size, TRIAL_SETS).zip(sets_of(n, size + 2, TRIAL_SETS)).is_some_and(|(small, large)| small + large <= TRIAL_SETS);
                let found = rs_decode_trying(&read, &certain, ec_len, 0, TRIAL_SETS, &mut Sightings::default());
                if fits {
                    let (erased, solution) = only(found).unwrap_or_else(|| panic!("{what} and one more"));
                    assert_eq!(solution.corrected, block, "{what} and one more");
                    assert_eq!((erased.len(), solution.errors), (size + 2, limit + 2), "{what} and one more");
                    assert!(erased.iter().all(|i| wrong.contains(i)), "{what} and one more");
                } else {
                    assert!(found.is_empty(), "{what} and one more");
                }
                // Two more are past every size that fits.
                let extra = (0..n).find(|i| !wrong.contains(i)).unwrap();
                read[extra].value ^= rng.nonzero();
                assert!(rs_decode_trying(&read, &certain, ec_len, 0, TRIAL_SETS, &mut Sightings::default()).is_empty(), "{what} and two more");
            }
        }
        // A pinned codeword is never erased.
        let block = codeword(&mut rng, 16, 18);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        for cw in &mut read[..10] {
            cw.value ^= rng.nonzero();
            cw.pinned = 0xFF;
        }
        assert!(rs_decode_trying(&read, &vec![[CERTAIN; 8]; 34], 18, 0, TRIAL_SETS, &mut Sightings::default()).is_empty());
        // The pairs of the largest block fit the budget, so a block of a
        // hundred and thirty codewords is reached one past the code; its
        // triples do not, so with an odd redundancy it is reached no
        // further than one codeword erased takes it.
        let block = codeword(&mut rng, 100, 30);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        for cw in &mut read[..16] {
            cw.value ^= rng.nonzero();
        }
        let certain = vec![[CERTAIN; 8]; 130];
        let (erased, solution) = only(rs_decode_trying(&read, &certain, 30, 0, TRIAL_SETS, &mut Sightings::default())).expect("pairs of a large block");
        assert_eq!((erased.len(), solution.errors, solution.corrected == block), (2, 16, true));
        assert_eq!(sets_of(130, 3, TRIAL_SETS), None);
        assert_eq!(sets_of(130, 3, TRIAL_SETS_MAX), Some(130 * 129 * 128 / 6));
        let block = codeword(&mut rng, 101, 29);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        for cw in &mut read[..15] {
            cw.value ^= rng.nonzero();
        }
        assert_eq!(only(rs_decode_trying(&read, &certain, 29, 0, TRIAL_SETS, &mut Sightings::default())).map(|(erased, s)| (erased.len(), s.errors)), Some((1, 15)));
        read[15].value ^= rng.nonzero();
        assert!(rs_decode_trying(&read, &certain, 29, 0, TRIAL_SETS, &mut Sightings::default()).is_empty());
        // A block of Model 1's largest, 238 codewords, has more pairs
        // than the least budget, and is given them all the same.
        let block = codeword(&mut rng, 192, 46);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        for &i in &rng.distinct(24, 238) {
            read[i].value ^= rng.nonzero();
        }
        assert!(sets_of(238, 2, TRIAL_SETS_MIN).is_none());
        let found = only(rs_decode_trying(&read, &vec![[CERTAIN; 8]; 238], 46, 0, TRIAL_SETS_MIN, &mut Sightings::default()));
        assert_eq!(found.map(|(erased, s)| (erased.len(), s.errors, s.corrected == block)), Some((2, 24, true)));
    }

    #[test]
    fn freeing_a_codeword_by_trial_mends_a_bit_solve_a_stain_put_out() {
        let mut rng = Rng(0x5A1B);
        for &(data_len, ec_len) in &[(16usize, 18usize), (32, 18), (16, 28)] {
            for round in 0..30 {
                let block = codeword(&mut rng, data_len, ec_len);
                let n = block.len();
                let stains = 1 + round % 2;
                // A bit unknown in more codewords than there is redundancy
                // to erase, and a codeword or two wrong throughout.
                let unknown = ec_len + 3;
                let picked = rng.distinct(unknown + stains, n);
                let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
                for &i in &picked[..unknown] {
                    let b = rng.below(8);
                    read[i].known &= !(1 << b);
                    read[i].value &= !(1 << b);
                }
                for &i in &picked[unknown..] {
                    read[i].value ^= rng.nonzero();
                }
                let what = format!("({data_len},{ec_len}) round {round}, {stains} stained");
                assert!(rs_decode(&read, ec_len).is_err(), "{what}");
                assert_eq!(solve_bit_erasures(&read, ec_len), Err(BitSolveFailure::Inconsistent), "{what}");
                let certain = vec![[CERTAIN; 8]; n];
                let doubt = only(solve_bit_erasures_trying(&read, &certain, ec_len, Trials { free: true, ..Trials::NONE }, &mut Sightings::default())).unwrap_or_else(|| panic!("{what}"));
                assert_eq!(doubt.solution.corrected, block, "{what}");
                assert_eq!(doubt.erased.len(), stains, "{what}");
                assert!(doubt.erased.iter().all(|i| picked[unknown..].contains(i)), "{what}");
                assert_eq!((doubt.doubted_bits, doubt.errors()), (0, stains), "{what}");
                assert!(doubt.confidence_bits() >= VERIFIED_BITS, "{what}: {}", doubt.confidence_bits());
                for &i in &doubt.erased {
                    assert_eq!(doubt.doubted[i], 0xFF, "{what}");
                    assert_eq!(doubt.changed[i], read[i].value ^ block[i], "{what}");
                }
                // Three stains are reached by the triples, and four are
                // past them.
                if stains == 2 {
                    let third = (0..n).find(|i| !picked.contains(i)).unwrap();
                    read[third].value ^= rng.nonzero();
                    let doubt = only(solve_bit_erasures_trying(&read, &certain, ec_len, Trials { free: true, ..Trials::NONE }, &mut Sightings::default())).unwrap_or_else(|| panic!("{what} and a third"));
                    assert_eq!(doubt.solution.corrected, block, "{what} and a third");
                    assert_eq!(doubt.erased.len(), 3, "{what} and a third");
                    assert!(doubt.erased.contains(&third), "{what} and a third");
                    let fourth = (0..n).find(|i| !picked.contains(i) && *i != third).unwrap();
                    read[fourth].value ^= rng.nonzero();
                    assert!(solve_bit_erasures_trying(&read, &certain, ec_len, Trials { free: true, ..Trials::NONE }, &mut Sightings::default()).is_empty(), "{what} and a fourth");
                }
            }
        }
        // Consistent as read, there is nothing to free.
        let block = codeword(&mut rng, 16, 18);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        read[3].known = 0xF0;
        assert!(solve_bit_erasures_trying(&read, &vec![[CERTAIN; 8]; 34], 18, Trials::ALL, &mut Sightings::default()).is_empty());
    }

    #[test]
    fn flipping_finds_the_solid_misreads_the_equations_name() {
        // A bit unknown in more codewords than there is redundancy to
        // erase, and a bit or two or three read wrong as solidly as the
        // rest: no doubting by reliability reaches them, and freeing
        // codewords by trial spends eight equations on each where a flip
        // spends none. The equations say which bits, by what they are
        // left wanting.
        let mut rng = Rng(0xF11B);
        // Blocks of sixty-four codewords at the most, every bit of
        // which the triples take in.
        for &(data_len, ec_len) in &[(16usize, 18usize), (32, 18), (16, 28), (40, 24)] {
            for round in 0..30 {
                let block = codeword(&mut rng, data_len, ec_len);
                let n = block.len();
                let flips = 1 + round % 3;
                let unknown = ec_len + 3;
                let picked = rng.distinct(unknown + flips, n);
                let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
                let reliability = vec![[200u8; 8]; n];
                for &i in &picked[..unknown] {
                    let b = rng.below(8);
                    read[i].known &= !(1 << b);
                    read[i].value &= !(1 << b);
                }
                let mut wrong = vec![0u8; n];
                for &i in &picked[unknown..] {
                    let b = rng.below(8);
                    // A bit that is known: the unknown ones are elsewhere.
                    read[i].value ^= 1 << b;
                    wrong[i] |= 1 << b;
                }
                let what = format!("({data_len},{ec_len}) round {round}, {flips} flipped");
                assert!(rs_decode(&read, ec_len).is_err(), "{what}");
                assert_eq!(solve_bit_erasures(&read, ec_len), Err(BitSolveFailure::Inconsistent), "{what}");
                // Nothing faint to doubt, and without flips only the frees
                // reach it, at eight equations a codeword.
                assert!(solve_bit_erasures_trying(&read, &reliability, ec_len, Trials::NONE, &mut Sightings::default()).is_empty(), "{what}");
                let flipping = Trials { flips: true, ..Trials::NONE };
                let doubt = only(solve_bit_erasures_trying(&read, &reliability, ec_len, flipping, &mut Sightings::default())).unwrap_or_else(|| panic!("{what}"));
                assert_eq!(doubt.solution.corrected, block, "{what}");
                assert_eq!(doubt.flipped, wrong, "{what}");
                assert_eq!((doubt.flipped_bits() as usize, doubt.errors(), doubt.erased.len(), doubt.doubted_bits), (flips, flips, 0, 0), "{what}");
                assert_eq!(doubt.solution.spare_equations, 8 * ec_len - unknown, "{what}");
                assert!(doubt.confidence_bits() >= VERIFIED_BITS, "{what}: {}", doubt.confidence_bits());
                // The frees reach the same block, spending more; the
                // flips are preferred, and so is a flip to a free when
                // both are allowed.
                let both = only(solve_bit_erasures_trying(&read, &reliability, ec_len, Trials::ALL, &mut Sightings::default())).unwrap_or_else(|| panic!("{what}"));
                assert_eq!((both.flipped, both.erased.len()), (wrong, 0), "{what}");
                // Painted by hand, every bit is certain and none is
                // flipped; the frees are all there is.
                let certain = vec![[CERTAIN; 8]; n];
                assert!(solve_bit_erasures_trying(&read, &certain, ec_len, flipping, &mut Sightings::default()).is_empty(), "{what}");
                let freed = solve_bit_erasures_trying(&read, &certain, ec_len, Trials::ALL, &mut Sightings::default());
                assert!(freed.iter().all(|d| d.erased.len() == flips && d.flipped_bits() == 0), "{what}");
            }
        }
        // With faint bits besides, doubted by their reliability, the
        // flips are found at the count of doubted bits that lets them.
        let (data_len, ec_len) = (16, 18);
        for round in 0..20 {
            let block = codeword(&mut rng, data_len, ec_len);
            let n = block.len();
            let faint = ec_len + 2;
            // The solid misreads in the last two codewords, which doubting
            // by index does not reach within its budget of equations.
            let picked = rng.distinct(faint, n - 2);
            let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
            let mut reliability = vec![[200u8; 8]; n];
            for &i in &picked {
                let b = rng.below(8);
                read[i].value ^= 1 << b;
                reliability[i][b] = 5 + rng.below(20) as u8;
            }
            for i in [n - 2, n - 1] {
                read[i].value ^= 1 << rng.below(8);
            }
            let what = format!("round {round}");
            assert!(solve_bit_erasures_doubting(&read, &reliability, ec_len, Trials::NONE, &mut Sightings::default()).is_empty(), "{what}");
            let flipping = Trials { flips: true, ..Trials::NONE };
            let doubt = only(solve_bit_erasures_doubting(&read, &reliability, ec_len, flipping, &mut Sightings::default())).unwrap_or_else(|| panic!("{what}"));
            assert_eq!(doubt.solution.corrected, block, "{what}");
            // The two solid bits are flipped, and now and then a faint one
            // not yet doubted with them, at a count of doubted bits one
            // short of it.
            let flipped = doubt.flipped_bits() as usize;
            assert!((2..=3).contains(&flipped), "{what}: flipped {flipped}");
            assert!(doubt.doubted_bits + flipped >= faint + 2, "{what}: doubted {} bits", doubt.doubted_bits);
            assert_eq!(doubt.errors(), faint + 2, "{what}");
            assert!(doubt.confidence_bits() >= VERIFIED_BITS, "{what}: {}", doubt.confidence_bits());
        }
    }

    #[test]
    fn doubting_bits_frees_a_stain_by_trial_among_faint_misreads() {
        // One faint wrong bit in more codewords than there is redundancy
        // to erase, which doubting the bits reaches, and one codeword
        // wrong throughout at full reliability, which it never doubts:
        // nothing solves until the stain is freed by trial.
        let mut rng = Rng(0x0DB7);
        let (data_len, ec_len) = (16, 18);
        for round in 0..30 {
            let block = codeword(&mut rng, data_len, ec_len);
            let n = block.len();
            let faint = ec_len + 2;
            let picked = rng.distinct(faint + 1, n);
            let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
            let mut reliability = vec![[CERTAIN; 8]; n];
            for &i in &picked[..faint] {
                let b = rng.below(8);
                read[i].value ^= 1 << b;
                reliability[i][b] = 5 + rng.below(20) as u8;
            }
            let stain = picked[faint];
            read[stain].value ^= rng.nonzero();
            let what = format!("round {round}");
            assert!(solve_bit_erasures_doubting(&read, &reliability, ec_len, Trials::NONE, &mut Sightings::default()).is_empty(), "{what}");
            let free = Trials { free: true, ..Trials::NONE };
            let doubt = only(solve_bit_erasures_doubting(&read, &reliability, ec_len, free, &mut Sightings::default())).unwrap_or_else(|| panic!("{what}"));
            assert_eq!(doubt.solution.corrected, block, "{what}");
            assert_eq!(doubt.erased, vec![stain], "{what}");
            assert!(doubt.doubted_bits >= faint, "{what}: doubted {} bits", doubt.doubted_bits);
            assert_eq!(doubt.doubted[stain], 0xFF, "{what}");
            assert_eq!(doubt.errors(), faint + 1, "{what}");
            assert!(doubt.confidence_bits() >= VERIFIED_BITS, "{what}: {}", doubt.confidence_bits());
        }
    }

    #[test]
    fn reserved_codewords_mark_solutions_beyond_the_standard() {
        // A 1-L block: seven EC codewords, three reserved, so two errors
        // are allowed and a third is a misdecode risk the standard forbids.
        let mut rng = Rng(41);
        let block = codeword(&mut rng, 19, 7);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        let clean = rs_decode_reserving(&read, 7, 3).unwrap();
        assert!(clean.within_standard() && clean.reserved == 3 && clean.slack == 7);
        for i in [2, 9] {
            read[i].value ^= rng.nonzero();
        }
        let two = rs_decode_reserving(&read, 7, 3).unwrap();
        assert_eq!((two.errors, two.slack), (2, 3));
        assert!(two.within_standard());
        read[14].value ^= rng.nonzero();
        let three = rs_decode_reserving(&read, 7, 3).unwrap();
        assert_eq!((three.errors, three.slack), (3, 1));
        assert_eq!(three.corrected, block, "the decode itself still goes as far as the code allows");
        assert!(!three.within_standard());
        assert!(rs_decode(&read, 7).unwrap().within_standard(), "nothing reserved, nothing exceeded");
        // Erasures count against the same limit.
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        for i in [0, 1, 2, 3, 4] {
            read[i] = CwRead::unknown();
        }
        assert!(!rs_decode_reserving(&read, 7, 3).unwrap().within_standard());
        read[4] = CwRead::known(block[4]);
        assert!(rs_decode_reserving(&read, 7, 3).unwrap().within_standard());
    }

    #[test]
    fn a_solver_holds_its_solutions_to_the_reserve_as_the_block_as_read_counts_it() {
        // A 1-L block, seven codewords of error correction and three of
        // them kept: four codewords wrong are past the two the standard
        // corrects, and erasing one by trial reaches the codeword. Read
        // right on the threshold, those four were read as nothing, and the
        // reserve counts them so: four codewords unread, as much as it
        // allows. Counted against the block the trials searched, they are
        // four read wrong, past it, and a solver that prefers what keeps
        // the reserve would keep another solution over this one.
        let mut rng = Rng(43);
        let block = codeword(&mut rng, 19, 7);
        let mut searched: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        let mut read = searched.clone();
        for i in [3, 8, 13, 18] {
            searched[i].value ^= rng.nonzero();
            read[i] = CwRead::unknown();
        }
        let certain = vec![[CERTAIN; 8]; block.len()];
        let found = rs_decode_trying_against(&searched, &read, &certain, 7, 3, TRIAL_SETS, &mut Sightings::default());
        assert!(found.iter().any(|(_, s)| s.corrected == block && s.within_standard()), "{found:?}");
        let found = rs_decode_trying(&searched, &certain, 7, 3, TRIAL_SETS, &mut Sightings::default());
        assert!(found.iter().all(|(_, s)| !s.within_standard()), "{found:?}");
    }

    #[test]
    fn a_codeword_that_keeps_the_reserve_lies_within_the_plain_decode_only_of_a_block_read_whole_or_not_at_all() {
        // A 1-Q block, thirteen codewords of error correction and one of
        // them kept: six codewords a module short and four read wrong
        // keep it, at seventy eighths of the ninety-six it leaves, where
        // the plain decode, erasing the six whole, would need fourteen.
        let mut rng = Rng(0x1A);
        let block = codeword(&mut rng, 13, 13);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        let picked = rng.distinct(10, block.len());
        for &i in &picked[..4] {
            read[i].value ^= rng.nonzero();
        }
        for &i in &picked[4..] {
            read[i].known = 0xFE;
        }
        assert!(keeps_reserve(&read, 13, 1, &block));
        assert!(!rs_decode_reserving(&read, 13, 1).is_ok_and(|s| s.corrected == block));
        assert!(!plain_reach_holds_the_reserve(&read, &read));
        // Read whole or not at all, it holds; and not of a block searched
        // that reads a codeword the block as read leaves unread.
        for &i in &picked[4..] {
            read[i] = CwRead::unknown();
        }
        assert!(plain_reach_holds_the_reserve(&read, &read));
        let mut searched = read.clone();
        searched[picked[4]] = CwRead::known(block[picked[4]] ^ 1);
        assert!(!plain_reach_holds_the_reserve(&searched, &read));
    }

    #[test]
    fn located_errors_confirm_a_solution_with_nothing_to_spare() {
        // One error among 26 codewords is found in 26 of 257 readings,
        // three bits; none confirms nothing, and one among 255 nothing
        // either. The nine a 5-Q block corrects among its 33 codewords,
        // and the eight of a 2-M block among its 44:
        // 256^18 / (C(33, 9) · 255^9) is 2^46.8 and
        // 256^16 / (C(44, 8) · 255^8) 2^36.6.
        assert_eq!(located_error_bits(26, 0), 0);
        assert_eq!(located_error_bits(26, 1), 3);
        assert_eq!(located_error_bits(255, 1), 0);
        assert_eq!(located_error_bits(26, 3), 12);
        assert_eq!(located_error_bits(33, 9), 46);
        assert_eq!(located_error_bits(44, 8), 36);
        // More errors than half the codewords read is no decoder's claim.
        assert_eq!(located_error_bits(8, 5), 0);

        // A 5-Q block wrong in nine codewords, which is all its eighteen
        // of error correction correct: solved with nothing to spare, and
        // confirmed by the nine it found.
        let mut rng = Rng(0x10CA7E);
        let ec_len = 18;
        let block = codeword(&mut rng, 15, ec_len);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        let wrong = rng.distinct(10, block.len());
        for &i in &wrong[..9] {
            read[i].value ^= rng.nonzero();
        }
        let sol = rs_decode(&read, ec_len).unwrap();
        assert_eq!((sol.errors, sol.erasures, sol.slack, sol.corrected == block), (9, 0, 0, true));
        assert_eq!((sol.check_bits, sol.located_bits, sol.confidence_bits()), (0, 46, 46));
        assert!(sol.within_standard());
        // One more is past the code, and no solution.
        read[wrong[9]].value ^= rng.nonzero();
        assert!(rs_decode(&read, ec_len).is_err());

        // Eighteen unknown codewords are filled with nothing to spare
        // too, and that any block would have allowed: nothing confirms it.
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        for &i in &rng.distinct(18, block.len()) {
            read[i] = CwRead::unknown();
        }
        let sol = rs_decode(&read, ec_len).unwrap();
        assert_eq!((sol.errors, sol.erasures, sol.slack, sol.confidence_bits()), (0, 18, 0, 0));
        // Sixteen unknown and one wrong: the one error is found among the
        // seventeen codewords read, which noise shows once in fifteen
        // blocks, and confirms three bits.
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        let picked = rng.distinct(17, block.len());
        for &i in &picked[..16] {
            read[i] = CwRead::unknown();
        }
        read[picked[16]].value ^= rng.nonzero();
        let sol = rs_decode(&read, ec_len).unwrap();
        assert_eq!((sol.errors, sol.erasures, sol.slack, sol.corrected == block), (1, 16, 0, true));
        assert_eq!(sol.confidence_bits(), located_error_bits(17, 1));
        assert_eq!(sol.confidence_bits(), 3);

        // Reserved codewords are reserved whatever was located: a 2-L
        // block wrong in five codewords solves by spending the two the
        // standard keeps for detection, and is beyond it.
        let block = codeword(&mut rng, 34, 10);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        for &i in &rng.distinct(5, block.len()) {
            read[i].value ^= rng.nonzero();
        }
        let sol = rs_decode_reserving(&read, 10, 2).unwrap();
        assert_eq!((sol.errors, sol.slack, sol.corrected == block), (5, 0, true));
        assert!(sol.confidence_bits() >= VERIFIED_BITS && !sol.within_standard());
    }

    #[test]
    fn one_spare_codeword_confirms_little_of_a_block_mostly_unread() {
        // A 5-Q block with seventeen of its codewords unknown has one
        // syndrome left of its eighteen to check the sixteen that were
        // read, and two of those are wrong. Whatever the first is off
        // by, one value in 255 of the second leaves that syndrome zero,
        // and the block "solves", to codewords it never held.
        let data: Vec<u8> = (0..15u8).map(|i| 0x40 + i).collect();
        let block: Vec<u8> = data.iter().copied().chain(rs_encode(&data, 18)).collect();
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        for cw in read.iter_mut().skip(16) {
            *cw = CwRead::unknown();
        }
        read[0].value ^= 0x01;
        let mut passed = Vec::new();
        for magnitude in 1..=255u8 {
            let mut trial = read.clone();
            trial[1].value ^= magnitude;
            if let Ok(sol) = rs_decode(&trial, 18) {
                passed.push((magnitude, sol));
            }
        }
        assert_eq!(passed.len(), 1);
        let (magnitude, sol) = &passed[0];
        assert_eq!((*magnitude, sol.errors, sol.erasures, sol.slack), (0x04, 0, 17, 1));
        assert!(sol.corrected != block);
        // A byte of confirmation, and no more: one block of noise in 256
        // passes so, as one of the 255 values here does. It is a byte
        // that is said, and the decodes it is one of are charged for
        // where they are counted ([`super::super::pipeline`]).
        assert_eq!(sol.confidence_bits(), VERIFIED_BITS);

        // With fifteen unknown there are three syndromes left, and no
        // value of the second wrong codeword passes them: one block in
        // sixteen million would, or one in two thousand as a single
        // wrong codeword the code located with a codeword to spare.
        read[16] = CwRead::known(block[16]);
        read[17] = CwRead::known(block[17]);
        for magnitude in 1..=255u8 {
            let mut trial = read.clone();
            trial[1].value ^= magnitude;
            assert!(rs_decode(&trial, 18).is_err(), "{magnitude:#x}");
        }
        assert_eq!(8 + located_error_bits(18, 1), 11);
    }

    #[test]
    fn the_locator_test_agrees_with_the_decode() {
        // Over blocks with every kind of damage, erasures among them,
        // the cheap test of a try is right exactly when the decode
        // gets past its locator: the budget, the degree and the roots.
        let mut rng = Rng(0x10CA);
        let (mut passed, mut refused) = (0, 0);
        for round in 0..3000 {
            let ec_len = 2 + rng.below(29);
            let data_len = 1 + rng.below(60);
            let block = codeword(&mut rng, data_len, ec_len);
            let n = block.len();
            let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
            let mut erasures = rng.below(ec_len.min(n) / 2 + 1);
            let wrong = rng.below(ec_len / 2 + 2);
            for &i in &rng.distinct((erasures + wrong).min(n), n) {
                if erasures > 0 && read[i].known == 0xFF && rng.below(2) == 0 {
                    erasures -= 1;
                    read[i] = CwRead::unknown();
                } else {
                    read[i].value ^= rng.nonzero();
                }
            }
            let erased: Vec<usize> = (0..n).filter(|&i| read[i].is_erasure()).collect();
            let f = erased.len();
            let received: Vec<u8> = read.iter().map(|c| c.value & c.known).collect();
            let syndromes = rs_syndromes(&received, ec_len);
            let gamma = erased.iter().fold(vec![1u8], |gamma, &i| poly_mul(&gamma, &[1, exp(n - 1 - i)]));
            let forney: Vec<u8> = (f..ec_len).map(|k| (0..=f).fold(0, |acc, m| acc ^ mul(gamma[m], syndromes[k - m]))).collect();
            let mut locator = Locator::new(ec_len, n);
            let cheap = locator.locates(&forney, (ec_len - f) / 2, &erased);
            let full = rs_decode_core(&read, &[], ec_len, 0, &syndromes, &mut Sightings::default());
            let past_locator = !matches!(full, Err(RsFailure::BudgetExceeded | RsFailure::LocatorDegree | RsFailure::RootCount { .. }));
            assert_eq!(cheap, past_locator, "round {round}: {full:?}");
            if cheap { passed += 1 } else { refused += 1 }
        }
        assert!(passed > 500 && refused > 500, "{passed} passed, {refused} refused");
    }

    /// How long the chase takes at its most, for seeing what a change
    /// to it buys: run by hand, with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn chase_timing() {
        let mut rng = Rng(0x7E);
        for &(data_len, ec_len, wrong) in &[(118usize, 30usize, 10usize), (9, 17, 5), (15, 18, 6)] {
            let block = codeword(&mut rng, data_len, ec_len);
            let n = block.len();
            let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
            let mut reliability = vec![[200u8; 8]; n];
            // Solid errors to the limit of the code, and faint bits that
            // are right beyond them, so that no pattern solves and every
            // one is decoded.
            for &i in &rng.distinct(wrong + 16, n) {
                let b = rng.below(8);
                if wrong > 0 && read[i].value == block[i] && rng.below(2) == 0 {
                    read[i].value ^= rng.nonzero();
                } else {
                    reliability[i][b] = 5 + rng.below(20) as u8;
                }
            }
            for bits in [10usize, 13, 16] {
                let started = std::time::Instant::now();
                let found = rs_decode_chasing(&read, &reliability, ec_len, 0, bits);
                let took = started.elapsed();
                println!("({data_len}, {ec_len}) chase of {bits} bits: {took:?}, {} found", found.len());
            }
            // And the trials of erasures, on the block wrong one past
            // the code: every single and then every pair.
            let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
            for &i in &rng.distinct(ec_len / 2 + 1, n) {
                read[i].value ^= rng.nonzero();
            }
            let started = std::time::Instant::now();
            let found = rs_decode_trying(&read, &vec![[CERTAIN; 8]; n], ec_len, 0, TRIAL_SETS, &mut Sightings::default());
            let took = started.elapsed();
            // Against every set decoded in full, as the trials once did.
            let received: Vec<u8> = read.iter().map(|c| c.value & c.known).collect();
            let syndromes = rs_syndromes(&received, ec_len);
            let (mut sets, mut solved) = (0, 0);
            let started_slow = std::time::Instant::now();
            for size in [1, 2] {
                each_set(n, size, |set| {
                    sets += 1;
                    solved += usize::from(rs_decode_core(&read, set, ec_len, 0, &syndromes, &mut Sightings::default()).is_ok());
                });
            }
            let slow = started_slow.elapsed();
            println!("({data_len}, {ec_len}) trials of erasures: {took:?}, {} found; every set decoded in full: {slow:?} over {sets} sets, {solved} solved", found.len());
        }
    }

    #[test]
    fn the_soft_count_is_the_hard_count_on_a_block_painted_by_hand_and_more_on_faint_bits() {
        // Painted by hand every bit costs 256, and the sets within the
        // reach are the sets of so many bits: within one codeword's
        // worth of distance, the sets of up to eight bits of the block,
        // which are more than the one byte the hard count allows, so
        // the soft count is the lower of the two.
        let block: Vec<CwRead> = (0..33).map(|_| CwRead::known(0)).collect();
        let certain = vec![[CERTAIN; 8]; 33];
        // Nothing overturned: every equation, less the hard count's
        // allowance for the codewords nearer than the one found.
        assert_eq!(soft_count_bits(&block, &certain, 8 * 18, 0), 8 * 18 - 1);
        assert_eq!(soft_count_bits(&block, &certain, 8 * 18, 0), list_solution_bits(33, 18, 0));
        let sets_of_eight: f64 = (0..=8).map(|j| (0..j).map(|i| (264 - i) as f64 / (i + 1) as f64).product::<f64>()).sum();
        let byte = 8 * bit_cost(CERTAIN);
        assert_eq!(soft_count_bits(&block, &certain, 8 * 18, byte), (144.0 - sets_of_eight.log2() - (255f64 / 254.0).log2()).floor() as u32);
        assert!(soft_count_bits(&block, &certain, 8 * 18, byte) < list_solution_bits(33, 18, 1));
        assert_eq!(soft_count_bits(&block, &certain, 8 * 18, SOFT_COUNT_REACH + 1), 0);
        // Twenty faint bits among solid ones: the sets within the cost
        // of overturning ten of them are sets of the faint bits mostly,
        // and the count is far more than the hard count's for the ten
        // codewords those bits lie in.
        let mut faint = vec![[200u8; 8]; 33];
        for i in 0..20 {
            faint[i][i % 8] = 4;
        }
        let ten = 10 * bit_cost(4);
        let soft = soft_count_bits(&block, &faint, 8 * 18, ten);
        assert!(soft > 100 && soft > list_solution_bits(33, 18, 10), "{soft}");
        // Pinned bits are not counted; unknown bits spend equations.
        let mut pinned = block.clone();
        for cw in &mut pinned[..20] {
            cw.pinned = 0xFF;
        }
        assert!(soft_count_bits(&pinned, &faint, 8 * 18, ten) > soft);
        assert_eq!(soft_count_bits(&block, &faint, 8 * 18 - 40, ten), soft - 40);
        // A bit read on its threshold costs nothing to overturn and is
        // in a set or out of it alike: each takes a bit from the count,
        // whatever the distance, where a byte each of reliability and
        // one once made sixty-four of them a byte of margin. Within a
        // byte of distance a little comes back, there being fewer
        // certain bits to make sets of.
        for tossed in [1usize, 5, 64] {
            let mut tosses = certain.clone();
            for i in 0..tossed {
                tosses[i / 8][i % 8] = 0;
            }
            assert_eq!(soft_count_bits(&block, &tosses, 8 * 18, 0), 8 * 18 - 1 - tossed as u32, "{tossed}");
            let want = soft_count_bits(&block, &certain, 8 * 18, byte) - tossed as u32;
            let got = soft_count_bits(&block, &tosses, 8 * 18, byte);
            assert!((want..=want + 4).contains(&got), "{tossed}: {got} for {want}");
        }
    }

    #[test]
    fn blocks_of_noise_pass_no_oftener_than_their_confidence_says() {
        // The count behind [`located_error_bits`], made: blocks of
        // noise, decoded, and those that "solve" counted by the
        // confidence the solution claims. A claim of `c` bits is that at
        // most one block of noise in `2^c` gets so far.
        let mut rng = Rng(0x0153);
        for (len, ec_len, blocks) in [(12usize, 4usize, 200_000u32), (20, 2, 20_000), (16, 6, 400_000)] {
            // How many solved, by the bits claimed.
            let mut claimed = [0u32; 64];
            let mut solved_at_the_limit = 0;
            for _ in 0..blocks {
                let read: Vec<CwRead> = (0..len).map(|_| CwRead::known(rng.byte())).collect();
                if let Ok(sol) = rs_decode(&read, ec_len) {
                    claimed[sol.confidence_bits().min(63) as usize] += 1;
                    solved_at_the_limit += u32::from(sol.slack == 0 && 2 * sol.errors == ec_len);
                }
            }
            for bits in 1..64 {
                let at_least: u32 = claimed[bits..].iter().sum();
                // Twice the expected count allows for chance, where the
                // expected count is more than a few.
                let allowed = (2.0 * f64::from(blocks) / 2f64.powi(bits as i32)).ceil() as u32 + 3;
                assert!(at_least <= allowed, "{len} codewords, {ec_len} EC: {at_least} of {blocks} claim {bits} bits or more");
            }
            // And the claim is no gross understatement: solutions at the
            // limit of the code do turn up in noise, about as often as
            // the count says.
            let located = located_error_bits(len, ec_len / 2);
            let expected = f64::from(blocks) / 2f64.powi(located as i32 + 1);
            assert!(f64::from(solved_at_the_limit) >= expected / 2.0, "{len} codewords, {ec_len} EC: {solved_at_the_limit} at the limit, {expected} expected");
        }
    }


    /// The count as it was, before the largest count was bounded rather
    /// than looked for after every bit, and before it could stop short.
    fn soft_count_bits_reference(block: &[CwRead], bit_reliability: &[[u8; 8]], redundancy_bits: usize, distance: u32) -> u32 {
        if distance > SOFT_COUNT_REACH {
            return 0;
        }
        let reach = distance as usize;
        let mut ways = vec![0f64; reach + 1];
        ways[0] = 1.0;
        let mut scaled = 0i64;
        for (cw, reliability) in block.iter().zip(bit_reliability) {
            for (b, &reliability) in reliability.iter().enumerate() {
                if cw.known >> b & 1 == 0 || cw.pinned >> b & 1 != 0 {
                    continue;
                }
                let cost = bit_cost(reliability) as usize;
                if cost > reach {
                    continue;
                }
                let mut peak = 0f64;
                if cost == 0 {
                    for w in &mut ways {
                        *w *= 2.0;
                        peak = peak.max(*w);
                    }
                } else {
                    let mut end = reach + 1;
                    while end > cost {
                        let start = (end - cost).max(cost);
                        let (below, from) = ways.split_at_mut(start);
                        for (w, &add) in from[..end - start].iter_mut().zip(&below[start - cost..end - cost]) {
                            *w += add;
                            peak = peak.max(*w);
                        }
                        end = start;
                    }
                }
                if peak > 2f64.powi(500) {
                    for w in &mut ways {
                        *w *= 2f64.powi(-500);
                    }
                    scaled += 500;
                }
            }
        }
        let total: f64 = ways.iter().sum();
        let volume = total.log2() + scaled as f64;
        (redundancy_bits as f64 - volume - (255f64 / 254.0).log2()).max(0.0).floor() as u32
    }

    #[test]
    fn the_soft_count_is_what_it_was_and_stops_short_only_where_it_falls_short() {
        let mut rng = Rng(0x50F7);
        let (mut cases, mut stopped, mut rescaled) = (0, 0, 0);
        for round in 0..3000 {
            let n = [4, 10, 26, 60, 150, 255][round % 6];
            let block: Vec<CwRead> = (0..n)
                .map(|_| {
                    let known = if rng.below(10) == 0 { rng.byte() } else { 0xFF };
                    let pinned = if rng.below(20) == 0 { rng.byte() & known } else { 0 };
                    CwRead { value: rng.byte() & known, known, pinned }
                })
                .collect();
            // Every kind of reading: on the threshold, faint, solid, certain.
            let style = rng.below(5);
            let bits: Vec<[u8; 8]> = (0..n)
                .map(|_| {
                    std::array::from_fn(|_| match style {
                        0 => [0, 0, 0, 3, 200][rng.below(5)],
                        1 => rng.below(40) as u8,
                        2 => rng.byte(),
                        3 => CERTAIN,
                        _ => if rng.below(3) == 0 { 0 } else { 255 - rng.below(60) as u8 },
                    })
                })
                .collect();
            let redundancy = rng.below(8 * 68 + 1);
            let distance = match rng.below(4) {
                0 => rng.below(64) as u32,
                1 => rng.below(600) as u32,
                2 => rng.below(SOFT_COUNT_REACH as usize + 1) as u32,
                _ => SOFT_COUNT_REACH - rng.below(8) as u32 + rng.below(16) as u32,
            };
            let want = soft_count_bits_reference(&block, &bits, redundancy, distance);
            assert_eq!(soft_count_bits(&block, &bits, redundancy, distance), want, "round {round}");
            rescaled += usize::from(block.iter().map(|c| (c.known & !c.pinned).count_ones()).sum::<u32>() > 500);
            for needed in [1, 9, 17, 30, 64, 200] {
                cases += 1;
                match soft_count_reaching(&block, &bits, redundancy, distance, needed) {
                    Some(soft) => assert_eq!(soft, want, "round {round}, needed {needed}"),
                    None => {
                        stopped += 1;
                        assert!(want < needed, "round {round}: stopped short of {needed} where the count is {want}");
                    }
                }
            }
        }
        // The test is of something: counts stopped short, and counts large enough to be scaled.
        assert!(stopped > cases / 10, "{stopped} of {cases}");
        assert!(rescaled > 100, "{rescaled}");
    }

    #[test]
    fn squares_of_bits_are_turned_about_the_diagonal() {
        let mut rng = Rng(0x7A5E);
        for _ in 0..200 {
            let square: [u64; 64] = std::array::from_fn(|_| rng.next());
            let mut turned = square;
            transpose_square(&mut turned);
            for r in 0..64 {
                for c in 0..64 {
                    assert_eq!(turned[c] >> r & 1, square[r] >> c & 1);
                }
            }
        }
    }

    /// The rows no unknown has a pivot in, as they were taken a bit at a
    /// time before they were taken a square at a time.
    fn zero_rows_reference(system: &BitSystem) -> ZeroRows {
        let zero = system.equations - system.rank;
        let words = zero / 64 + 1;
        let (first, end) = (system.unknown, system.columns.len());
        let mut columns = vec![0u64; (end - first) * words];
        let mut residual = vec![0u64; words];
        for (k, row) in system.rows[system.rank * system.words..].chunks_exact(system.words).enumerate() {
            let (word, bit) = (k / 64, 1u64 << (k % 64));
            if system.rhs(row) {
                residual[word] |= bit;
            }
            for (w, &held) in row.iter().enumerate().take(end.div_ceil(64)).skip(first / 64) {
                let mut held = held;
                if w == first / 64 {
                    held &= !0u64 << (first % 64);
                }
                if w == end / 64 {
                    held &= (1u64 << (end % 64)) - 1;
                }
                while held != 0 {
                    let col = 64 * w + held.trailing_zeros() as usize;
                    columns[(col - first) * words + word] |= bit;
                    held &= held - 1;
                }
            }
        }
        ZeroRows { words, residual, columns }
    }

    #[test]
    fn zero_rows_taken_a_square_at_a_time_are_those_taken_a_bit_at_a_time() {
        let mut rng = Rng(0x2E80);
        for round in 0..300 {
            let (data_len, ec_len) = [(9, 17), (19, 7), (44, 30), (118, 30), (20, 68), (3, 2)][round % 6];
            let block = codeword(&mut rng, data_len, ec_len);
            let n = block.len();
            let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
            for _ in 0..rng.below(4 * ec_len) {
                let i = rng.below(n);
                read[i].known &= !(1 << rng.below(8));
                read[i].value &= read[i].known;
            }
            for _ in 0..rng.below(4) {
                read[rng.below(n)].value ^= rng.nonzero();
            }
            for _ in 0..rng.below(3) {
                let i = rng.below(n);
                read[i].pinned = read[i].known & rng.byte();
            }
            let system = BitSystem::build(&read, ec_len, true);
            let (got, want) = (system.zero_rows(), zero_rows_reference(&system));
            assert_eq!((got.words, &got.residual, &got.columns), (want.words, &want.residual, &want.columns), "round {round}");
        }
    }

    #[test]
    fn bits_are_turned_about_the_diagonal() {
        let mut rng = Rng(0x7AA5);
        for _ in 0..1000 {
            let x = rng.next();
            let turned = transpose_bits(x);
            for r in 0..8 {
                for c in 0..8 {
                    assert_eq!(turned >> (8 * c + r) & 1, x >> (8 * r + c) & 1, "{x:#x}: ({r}, {c})");
                }
            }
            assert_eq!(transpose_bits(turned), x);
        }
    }

    #[test]
    fn freeing_finds_what_a_reduction_of_each_whole_set_finds() {
        // Every set of codewords, its columns reduced from nothing in a
        // plain elimination, against the sets [`each_freeing`] reports:
        // over vectors of every width it holds, codewords of fewer than
        // eight columns among them, and columns that repeat or are
        // nought, so that dependent sets and prefixes turn up.
        fn bit_at(v: &[u64], at: usize) -> bool {
            v[at / 64] >> (at % 64) & 1 != 0
        }
        fn xor_into(v: &mut [u64], w: &[u64]) {
            v.iter_mut().zip(w).for_each(|(a, b)| *a ^= b);
        }
        fn reference(starts: &[usize], vectors: &[u64], residual: &[u64], size: usize) -> Vec<(Vec<usize>, u32)> {
            let words = residual.len();
            let mut out = Vec::new();
            each_set(starts.len() - 1, size, |set| {
                let mut rows: Vec<(Vec<u64>, u32)> = Vec::new();
                for (which, &k) in set.iter().enumerate() {
                    for j in starts[k]..starts[k + 1] {
                        let mut v = vectors[j * words..(j + 1) * words].to_vec();
                        let mut combination = 1u32 << (8 * which + j - starts[k]);
                        for (row, c) in &rows {
                            let pivot = row.iter().position(|&x| x != 0).map(|w| 64 * w + row[w].trailing_zeros() as usize).unwrap();
                            if bit_at(&v, pivot) {
                                xor_into(&mut v, row);
                                combination ^= c;
                            }
                        }
                        if v.iter().all(|&x| x == 0) {
                            return;
                        }
                        rows.push((v, combination));
                    }
                }
                let mut target = residual.to_vec();
                let mut combination = 0;
                for (row, c) in &rows {
                    let pivot = row.iter().position(|&x| x != 0).map(|w| 64 * w + row[w].trailing_zeros() as usize).unwrap();
                    if bit_at(&target, pivot) {
                        xor_into(&mut target, row);
                        combination ^= c;
                    }
                }
                if target.iter().all(|&x| x == 0) {
                    out.push((set.to_vec(), combination));
                }
            });
            out
        }
        let mut rng = Rng(0xF4EE);
        let mut found = 0;
        for &(bits, codewords) in &[(20usize, 7usize), (40, 9), (64, 8), (100, 10), (200, 8), (300, 7), (600, 6)] {
            let words = bits / 64 + 1;
            let top = if bits % 64 == 0 { 0 } else { (1u64 << (bits % 64)) - 1 };
            for round in 0..12 {
                let mut starts = vec![0];
                let mut vectors: Vec<u64> = Vec::new();
                for _ in 0..codewords {
                    let count = 1 + rng.below(8);
                    for _ in 0..count {
                        let mut v: Vec<u64> = (0..words).map(|_| rng.next()).collect();
                        // Few bits where the space is small, so that sets
                        // do make up the residual.
                        v[words - 1] &= top;
                        if rng.below(12) == 0 {
                            v.iter_mut().for_each(|x| *x = 0);
                        } else if rng.below(10) == 0 && vectors.len() >= words {
                            v.copy_from_slice(&vectors[vectors.len() - words..]);
                        }
                        vectors.extend(v);
                    }
                    starts.push(vectors.len() / words);
                }
                for size in 1..=FREE_SIZES {
                    // A residual some set makes up, or one at random.
                    let mut residual = vec![0u64; words];
                    if round % 3 != 0 {
                        for k in rng.distinct(size.min(codewords), codewords) {
                            for j in starts[k]..starts[k + 1] {
                                if rng.below(2) == 0 {
                                    xor_into(&mut residual, &vectors[j * words..(j + 1) * words]);
                                }
                            }
                        }
                    } else {
                        residual.iter_mut().for_each(|x| *x = rng.next());
                        residual[words - 1] &= top;
                    }
                    let mut got = Vec::new();
                    each_freeing(&starts, &vectors, &residual, size, |set, combination| got.push((set.to_vec(), combination)));
                    let want = reference(&starts, &vectors, &residual, size);
                    assert_eq!(got, want, "{bits} bits, round {round}, size {size}");
                    found += got.len();
                }
            }
        }
        assert!(found > 100, "{found} sets found in all");
    }
}
