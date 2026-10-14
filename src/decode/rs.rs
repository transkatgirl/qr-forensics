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
use std::collections::HashMap;
use std::rc::Rc;

use super::gf256::{self, div, exp, exp_of_sum, log, mul, poly_degree, poly_eval, poly_mul};
use crate::symbol::grid::CERTAIN;
use crate::work;

/// Confirming bits a solution needs before it counts as verified, when
/// it is the decode the standard describes of a block read whole: every
/// codeword known, the wrong ones located by the code, and nothing else
/// tried. One codeword of slack, or errors located that as few blocks of
/// noise would show ([`located_error_bits`]). Such a decode never shows
/// less within the standard's limit: the reserve speaks where there is
/// one, and the errors located at the limit where there is none.
pub const VERIFIED_BITS: u32 = 8;

/// Confirming bits any other solution needs: one that filled in codewords
/// that could not be read, or was reached by doubting codewords or bits
/// that could, by flipping them, bit by bit, or from pins alone. Three
/// codewords of slack, or as much between the slack, the known bits of
/// erased codewords reproduced and the errors located.
///
/// An erased codeword is given whatever value fits, so a block with all
/// but one of its error correction codewords spent on erasures is
/// confirmed by the one syndrome left, and the rest of the block may hold
/// anything: a block of noise passes once in 256, and as often with all
/// but four so spent and two wrong codewords "located" by the four. That
/// is no rare block in a damaged symbol. A tear takes most of one block
/// of four and leaves the others clean, the read is made again under
/// other kernels and thresholds, and the garbage that passes sits inside
/// a byte segment whose stream parses as well as ever: a version 5-Q
/// symbol with seventeen of a block's eighteen error correction
/// codewords' worth unknown and the rest of the block misread was read,
/// one try in four hundred, as its message with a block's worth of bytes
/// it never held. ZXing, BoofCV and OpenCV decode errors alone, within
/// half the code's distance, where the same block of noise passes once
/// in 10^14; three codewords make it once in sixteen million here, and a
/// solution with less is kept, and shown, as one that solved without
/// confirmation.
pub const AIDED_VERIFIED_BITS: u32 = 3 * VERIFIED_BITS;

/// Spare equations a solution found by doubting bits must keep. Doubting
/// tries many counts of doubted bits, and a wrong solution passes `s`
/// spare equations by chance once in `2^s` tries, so one codeword's worth
/// is not enough when a score of counts are tried. Two are enough for
/// the solution to be offered, and no more: it is verified with three
/// ([`AIDED_VERIFIED_BITS`]), between the spare equations and the
/// agreement of the bits doubted.
pub const DOUBT_SPARE_EQUATIONS: usize = 2 * VERIFIED_BITS as usize;

/// Confirming bits a solution found by doubting codewords must show.
/// Doubting tries one count of codewords after another, up to all of the
/// block's redundancy, and each count is a fresh chance for noise to
/// pass: with one codeword spare and nothing else behind it, a block of
/// noise passes once in 256 tries. So a doubted solution needs two
/// codewords spare, or one and a byte's worth of agreement from the
/// doubted codewords beyond chance, which codewords misread in a faint
/// bit or two give and codewords of noise do not, or as much from the
/// errors it located among the codewords it did not doubt, once those
/// have paid for the counts tried: a bit for each doubling of them.
/// That much has it offered; it is verified with a codeword's worth
/// more ([`AIDED_VERIFIED_BITS`]), since with this much one block of
/// noise in thirty thousand still passed.
pub const DOUBT_CONFIRMING_BITS: u32 = 2 * VERIFIED_BITS;

/// What a known bit of an erased codeword costs a solution that overturns
/// it, in bits it reproduced. A codeword with an unknown bit is erased, and
/// the solution is free to rewrite the bits of it that were read: a right
/// solution reproduces nearly all of them, and a wrong one, fitted to
/// whatever redundancy the erasures left, half. Counted at face value the
/// half that agree by chance pass for confirmation: a block with nineteen
/// of its twenty error correction codewords erased, a hundred and sixteen
/// known bits among them, "solved" to a codeword that overturned sixty-two
/// of those bits and was credited with the other fifty-four, and a noisy
/// photo was read as text it did not hold. At two for one, bits read with
/// one error in five still net their solution credit, and bits a solution
/// agrees with by chance net it less than nothing.
pub const OVERTURN_COST: u32 = 2;

/// How much nearer the block as read, in bits weighed by reliability
/// ([`soft_distance`]), the solution a solver that tries many ways keeps
/// must lie than any other it found that names another codeword, for it
/// to stand alone. Chasing and the trials of erasures are list decoders:
/// several of their tries may each reach a codeword that verifies, and
/// where they reach different ones the nearer to what was read is the
/// likelier, since the bits it overturns are the fainter. A bit
/// overturned costs its reliability and one: two hundred and fifty-six
/// for a certain bit, a few for a faint one. Reliability is eight times
/// the bits of likelihood a module's reading carries, once the sampler
/// has calibrated it against the modules whose colour it knew
/// ([`crate::imaging::sample::Classified::calibrate`]), so the margin
/// is a likelihood ratio of two hundred and fifty-six to one between
/// the two codewords, a quarter of a certain bit, as the format words
/// are told apart ([`super::bch::WEIGHT_MARGIN`]): one certain bit, or
/// one bit read well inside its colour against one read near the
/// threshold, settles it; two faint bits of middling margin do not, and
/// both codewords are returned for the message to tell apart.
pub const DISTANCE_MARGIN: u32 = 64;

/// Sets of codewords a solver erases by trial, at the most, over every
/// size of set it tries: every codeword alone, every pair, which fit for
/// any block, the largest having a hundred and fifty-three codewords
/// and eleven thousand six hundred pairs, every triple up to
/// seventy-three codewords and every four up to thirty. Each set is a
/// decode of a few thousand field operations, and each is a fresh
/// chance for a block of noise to solve, which what the solution
/// confirms is charged for ([`RsSolution::among`]): sixteen bits for
/// all of them, which is why the count stops here.
pub const TRIAL_SETS: usize = 1 << 16;

/// Fewest sets a solver may be given to erase by trial
/// ([`Trials::sets`]): the pairs of the largest block, so that one
/// codeword past the code is reached in any block however many tries
/// the decode is one of.
pub const TRIAL_SETS_MIN: usize = 12 * 1024;

/// Most sets a solver may be given to erase by trial, where the budget
/// is raised by hand ([`super::pipeline::DecodeOptions::trial_sets`]):
/// a million decodes, which is seconds on a block, and the triples of a
/// block of a hundred and eighty codewords.
pub const TRIAL_SETS_MAX: usize = 1 << 20;

/// What the bit-level solvers may try beyond doubting bits by their
/// reliability, and how far.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// How far a solution lies from the block as read: the bits read as
/// known that it overturns, each weighed by its reliability and one, so
/// that a certain bit costs two hundred and fifty-six and a bit read on
/// its threshold one. The Hamming distance, bit for bit, where nothing
/// was sampled.
pub fn soft_distance(block: &[CwRead], bit_reliability: &[[u8; 8]], corrected: &[u8]) -> u32 {
    block
        .iter()
        .zip(bit_reliability)
        .zip(corrected)
        .map(|((cw, reliability), &c)| {
            let overturned = (cw.value ^ c) & cw.known;
            (0..8).filter(|&b| overturned >> b & 1 != 0).map(|b| u32::from(reliability[b]) + 1).sum::<u32>()
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

/// What the agreement of doubted bits with a solution is worth: the
/// bits that agree less those that do not, less three standard
/// deviations of that difference under the hypothesis that the solution
/// is wrong and each doubted bit agrees with it by chance. A faint bit
/// that turns out right confirms the solution, but a run of doubted bits
/// agrees with any solution half the time.
pub fn doubted_agreement(agreeing: u32, flipped: u32) -> u32 {
    let chance = (3.0 * f64::from(agreeing + flipped).sqrt()).ceil() as u32;
    agreeing.saturating_sub(flipped).saturating_sub(chance)
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
    if errors > read {
        return 0;
    }
    let ways: f64 = (1..=errors).map(|i| ((read - errors + i) as f64 / i as f64).log2()).sum();
    let bits = 8.0 * redundancy as f64 - errors as f64 * 255f64.log2() - ways - (255f64 / 254.0).log2();
    bits.max(0.0).floor() as u32
}

/// Furthest the soft count ([`soft_count_bits`]) looks, in
/// reliability-weighted bits: a solution lying further from the block
/// as read than this is confirmed by the hard count alone. Sixteen
/// certain bits, or four hundred faint ones, which is more than any
/// block has.
pub const SOFT_COUNT_REACH: u32 = 4096;

/// What confirms a codeword found within `distance` of the block as
/// read, in bits weighed by their reliability ([`soft_distance`]), in
/// bits: how rarely a block of noise has any codeword so near. The
/// code is linear over GF(2) with `redundancy_bits` equations once the
/// unknown bits are spent, so one binary word in `2^redundancy_bits` is
/// a codeword, and about each word there are as many words within
/// `distance` as there are sets of its known bits whose reliabilities
/// and one sum to `distance` or less: counted by dynamic programming
/// over the reliabilities, which are small integers. So noise shows a
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
    if distance > SOFT_COUNT_REACH {
        return 0;
    }
    let reach = distance as usize;
    // ways[d]: how many sets of the bits seen so far cost exactly d,
    // scaled down by 2^scaled whenever they grow large.
    let mut ways = vec![0f64; reach + 1];
    ways[0] = 1.0;
    let mut scaled = 0i64;
    for (cw, reliability) in block.iter().zip(bit_reliability) {
        for (b, &reliability) in reliability.iter().enumerate() {
            if cw.known >> b & 1 == 0 || cw.pinned >> b & 1 != 0 {
                continue;
            }
            let cost = usize::from(reliability) + 1;
            if cost > reach {
                continue;
            }
            for d in (cost..=reach).rev() {
                ways[d] += ways[d - cost];
            }
            if ways.iter().any(|&w| w > 2f64.powi(500)) {
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

/// One codeword as read from the symbol.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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
    /// those it reproduced, less [`OVERTURN_COST`] for each it overturned,
    /// and nothing when that comes to less than nothing.
    pub check_bits: u32,
    /// Known bits inside erased codewords that the solution overturned. Only
    /// ever non-zero when `slack > 0`.
    pub flipped_known_bits: u32,
    /// What the errors the decoder located confirm of the solution
    /// ([`located_error_bits`]), less what the tries of a solver that
    /// made many to find it cost: a bit for each doubling of them, since
    /// each try is a fresh chance for a block of noise to show errors
    /// located.
    pub located_bits: u32,
    /// EC codewords the standard reserves against misdecodes for this
    /// block: a solution that leaves fewer than this many unused has spent
    /// capacity the standard says is for detection only.
    pub reserved: usize,
}

impl RsSolution {
    /// Rough count of independent bits confirming the solution: a byte
    /// for each codeword of redundancy left unused, the known bits of
    /// erased codewords it reproduced, and what the errors it located
    /// confirm. A solution that only filled erasures, with nothing to
    /// spare, has none, and any input would have "solved" so; one that
    /// located errors with nothing to spare has what they confirm, which
    /// at the limit of a block's error correction is a great deal.
    pub fn confidence_bits(&self) -> u32 {
        8 * self.slack as u32 + self.check_bits + self.located_bits
    }

    /// The solution as the pick of `tries` a solver made: what its
    /// located errors confirm is less by the bits it takes to count the
    /// tries. The slack and the known bits reproduced are left as they
    /// are, and the solvers that try many times ask more of those
    /// instead ([`DOUBT_CONFIRMING_BITS`], [`CHASE_MIN_SLACK`]).
    pub fn among(mut self, tries: usize) -> Self {
        self.located_bits = self.located_bits.saturating_sub(tries_charge(tries));
        self
    }

    /// The solution stays inside the correction capacity the standard
    /// allows, leaving the reserved codewords unused.
    pub fn within_standard(&self) -> bool {
        self.slack >= self.reserved
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
    /// The solution contradicts bits that were read as known, with too
    /// little redundancy left to say that it is the bits that are wrong: it
    /// has no slack at all, or it overturns so many known bits of the
    /// codewords it erased that what slack it has does not outweigh them.
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
    rs_decode_doubted(block, &[], ec_len, reserved)
}

/// The same, with the codewords at `doubted` indices treated as erasures
/// although they were read in full: their bits are allowed to be wrong,
/// but the bits the solution agrees with still count as checks, as the
/// known bits of any other erasure do. Erasing them outright would throw
/// that confirmation away: a doubted codeword that was right in seven
/// bits of eight confirms the solution by those seven. Only the agreement
/// beyond the disagreement counts, though, since a codeword that was
/// wrong in half its bits agrees with any solution as often as not.
fn rs_decode_doubted(
    block: &[CwRead],
    doubted: &[usize],
    ec_len: usize,
    reserved: usize,
) -> Result<RsSolution, RsFailure> {
    let received: Vec<u8> = block.iter().map(|c| c.value & c.known).collect();
    let syndromes = rs_syndromes(&received, ec_len);
    rs_decode_core(block, doubted, ec_len, reserved, &syndromes)
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
/// [`syndrome_of_bit`].
pub(super) fn rs_decode_core(block: &[CwRead], doubted: &[usize], ec_len: usize, reserved: usize, syndromes: &[u8]) -> Result<RsSolution, RsFailure> {
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
    settle(block, doubted, &erased, corrected, l, slack, located_error_bits(n - f, l), reserved)
}

/// A solution's accounts, from the block it corrected: whether it may
/// stand, and what confirms it. `erased` are the codewords taken as
/// erasures, those read with unknown bits and the `doubted` among them,
/// read in full and taken as erased all the same; `errors` how many
/// codewords read in full the solution changed besides; `slack` the
/// redundancy it left unused, in codewords; and `located_bits` what the
/// errors it located confirm ([`located_error_bits`], or
/// [`list_solution_bits`] less the slack's byte each for a solution
/// reached past half the code's distance). A solution that overturns a
/// pinned bit is refused, and so is one that overturns the known bits of
/// the codewords it erased with too little slack to say that it is the
/// bits that are wrong ([`OVERTURN_COST`]): a doubted codeword's bits are
/// expected to be wrong, and its agreement is weighed against its
/// disagreement instead ([`doubted_agreement`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn settle(
    block: &[CwRead],
    doubted: &[usize],
    erased: &[usize],
    corrected: Vec<u8>,
    errors: usize,
    slack: usize,
    located_bits: u32,
    reserved: usize,
) -> Result<RsSolution, RsFailure> {
    let n = block.len();
    let is_doubted = |i: usize| doubted.contains(&i);
    let is_erased = |i: usize| erased.contains(&i);
    let received: Vec<u8> = block.iter().map(|c| c.value & c.known).collect();
    let mut check_bits = 0;
    let mut flipped_known_bits = 0;
    let (mut doubted_agreeing, mut doubted_flipped) = (0u32, 0u32);
    let (mut erased_agreeing, mut erased_flipped) = (0u32, 0u32);
    let mut first_overturned = None;
    for (i, cw) in block.iter().enumerate() {
        let diff = corrected[i] ^ cw.value;
        if diff & cw.pinned != 0 {
            return Err(RsFailure::PinConflict { index: i });
        }
        let flipped = (diff & cw.known).count_ones();
        if is_doubted(i) {
            // A doubted codeword's bits are expected to be wrong: that is
            // why it was doubted. Its agreement is weighed against its
            // disagreement below.
            flipped_known_bits += flipped;
            doubted_flipped += flipped;
            doubted_agreeing += cw.known.count_ones() - flipped;
        } else if is_erased(i) {
            if flipped > 0 && slack == 0 {
                return Err(RsFailure::KnownBitConflict { index: i });
            }
            flipped_known_bits += flipped;
            erased_flipped += flipped;
            erased_agreeing += cw.known.count_ones() - flipped;
            if flipped > 0 && first_overturned.is_none() {
                first_overturned = Some(i);
            }
        }
    }
    // The known bits of the erased codewords, weighed: see
    // [`OVERTURN_COST`]. A solution they speak against stands only on its
    // slack, and one left with less than a byte of confirmation between
    // the two is refused, as one with no slack that overturns a single
    // bit is above.
    let net = i64::from(erased_agreeing) - i64::from(OVERTURN_COST) * i64::from(erased_flipped);
    if net >= 0 {
        check_bits += net as u32;
    } else if 8 * slack as i64 + net < i64::from(VERIFIED_BITS) {
        return Err(RsFailure::KnownBitConflict { index: first_overturned.unwrap_or(0) });
    }
    check_bits += doubted_agreement(doubted_agreeing, doubted_flipped);

    let errata = (0..n)
        .filter(|&i| is_erased(i) || corrected[i] != received[i])
        .map(|i| Erratum { index: i, magnitude: corrected[i] ^ received[i], was_erasure: is_erased(i) })
        .collect();

    Ok(RsSolution { corrected, errata, errors, erasures: erased.len(), slack, check_bits, flipped_known_bits, located_bits, reserved })
}

/// A codeword of the code, however it was found, confirmed against the
/// block as read: the hard count of [`list_solution_bits`] for the
/// codewords read in full that it changes, or the soft count of
/// [`soft_count_bits`] for the bits it overturns weighed by
/// `bit_reliability`, whichever is the more, less one for the choice
/// between them and the charge for `tries` ([`tries_charge`]), with the
/// known bits of the codewords that could not be read weighed as
/// [`settle`] weighs them. For a solver that searched from something
/// other than the block as read: the beliefs propagation leaves
/// ([`super::abp`]), or the block with what the message makes likely
/// laid over it ([`super::evidence`]). What it searched from says
/// nothing here, since the search carries its word toward some codeword
/// whatever the block holds; only how rarely a block of noise lies so
/// near a codeword does. `None` for a codeword refused, or one with less
/// than [`DOUBT_CONFIRMING_BITS`] behind it.
pub(super) fn accounted(
    block: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    reserved: usize,
    corrected: Vec<u8>,
    tries: usize,
) -> Option<RsSolution> {
    let n = block.len();
    let erased: Vec<usize> = (0..n).filter(|&i| block[i].is_erasure()).collect();
    let (read, redundancy) = (n - erased.len(), ec_len.saturating_sub(erased.len()));
    let errors = (0..n).filter(|&i| !block[i].is_erasure() && corrected[i] != block[i].value).count();
    let unknown: usize = block.iter().map(|c| c.known.count_zeros() as usize).sum();
    let hard = list_solution_bits(read, redundancy, errors);
    let soft = soft_count_bits(block, bit_reliability, (8 * ec_len).saturating_sub(unknown), soft_distance(block, bit_reliability, &corrected));
    let bits = hard.max(soft).saturating_sub(1).saturating_sub(tries_charge(tries));
    let slack = redundancy.saturating_sub(2 * errors).min(bits as usize / 8);
    let located = bits - 8 * slack as u32;
    let solution = settle(block, &[], &erased, corrected, errors, slack, located, reserved).ok()?;
    (solution.confidence_bits() >= DOUBT_CONFIRMING_BITS).then_some(solution)
}

/// A codeword of the code found by searching something other than
/// `block`, judged against `block` alone ([`super::evidence::Laid`]):
/// how rarely a block of noise has a codeword as near, counted codeword
/// by codeword or bit by bit, whichever counts the more, less one for
/// the choice between them and the charge for `tries`.
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
/// the two are never added, as [`accounted`] adds them: a search that
/// solved bit by bit used every known bit to find its codeword, and its
/// agreeing with them is what the soft count measures, once. A bit by
/// bit count's slack is the codewords' worth of it, which is what the
/// codewords the standard reserves are held to. `None` for a codeword
/// that overturns a pin, or with less than [`DOUBT_CONFIRMING_BITS`]
/// behind it.
pub(super) fn judged(
    block: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    reserved: usize,
    corrected: Vec<u8>,
    tries: usize,
) -> Option<RsSolution> {
    if block.iter().zip(&corrected).any(|(cw, &c)| (cw.value ^ c) & cw.pinned != 0) {
        return None;
    }
    let n = block.len();
    let erased: Vec<usize> = (0..n).filter(|&i| block[i].is_erasure()).collect();
    let (read, redundancy) = (n - erased.len(), ec_len.saturating_sub(erased.len()));
    let errors = (0..n).filter(|&i| !block[i].is_erasure() && corrected[i] != block[i].value).count();
    let unknown: usize = block.iter().map(|c| c.known.count_zeros() as usize).sum();
    let charge = 1 + tries_charge(tries);
    let hard = list_solution_bits(read, redundancy, errors).saturating_sub(charge);
    let slack = redundancy.saturating_sub(2 * errors).min(hard as usize / 8);
    let by_codeword = (erased.len() <= ec_len)
        .then(|| settle(block, &[], &erased, corrected.clone(), errors, slack, hard - 8 * slack as u32, reserved).ok())
        .flatten();
    let distance = soft_distance(block, bit_reliability, &corrected);
    let soft = soft_count_bits(block, bit_reliability, (8 * ec_len).saturating_sub(unknown), distance).saturating_sub(charge);
    let solution = match by_codeword {
        Some(solution) if solution.confidence_bits() >= soft => solution,
        _ => {
            let slack = soft as usize / 8;
            let flipped_known_bits =
                erased.iter().map(|&i| ((corrected[i] ^ block[i].value) & block[i].known).count_ones()).sum();
            let solution = RsSolution {
                corrected,
                errata: Vec::new(),
                errors,
                erasures: erased.len(),
                slack,
                check_bits: 0,
                flipped_known_bits,
                located_bits: soft - 8 * slack as u32,
                reserved,
            };
            restated(block, solution)
        }
    };
    (solution.confidence_bits() >= DOUBT_CONFIRMING_BITS).then_some(solution)
}

/// Decode by doubting: erase the least reliable fully read codewords, one
/// more at a time, and keep the solution with the most confirmation,
/// among those that show [`DOUBT_CONFIRMING_BITS`] of it: one within the
/// standard's limit when `reserved` codewords are kept for detection, or,
/// when none is, one that spends them, which the caller marks as beyond
/// the standard ([`RsSolution::within_standard`]). Returns how many
/// codewords were doubted. Codewords at [`CERTAIN`] reliability and
/// pinned codewords are never doubted.
pub fn rs_decode_doubting(
    block: &[CwRead],
    reliability: &[u8],
    ec_len: usize,
    reserved: usize,
) -> Option<(usize, RsSolution)> {
    assert_eq!(block.len(), reliability.len());
    let erasures = block.iter().filter(|c| c.is_erasure()).count();
    let mut doubtful: Vec<usize> = (0..block.len())
        .filter(|&i| !block[i].is_erasure() && block[i].pinned == 0 && reliability[i] < CERTAIN)
        .collect();
    doubtful.sort_by_key(|&i| (reliability[i], i));
    doubtful.truncate(ec_len.saturating_sub(erasures));

    let mut best: Option<(usize, RsSolution)> = None;
    let mut beyond: Option<(usize, RsSolution)> = None;
    for k in 1..=doubtful.len() {
        if let Ok(solution) = rs_decode_doubted(block, &doubtful[..k], ec_len, reserved).map(|s| s.among(doubtful.len()))
            && solution.confidence_bits() >= DOUBT_CONFIRMING_BITS
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
/// Fewest EC codewords a chased solution must leave unused: the patterns
/// tried are a power of two, and each is a fresh chance of a misdecode,
/// which a second spare codeword makes two hundred and fifty-six times
/// rarer. With that the solution is offered. It is verified with
/// [`AIDED_VERIFIED_BITS`] behind it, which two spare codewords make
/// when the flipped block's decode located errors to the worth of a
/// third, as it does where flipping reaches a block a little past the
/// code's correction, and do not make when the rest of the redundancy
/// went on codewords that could not be read: with sixteen of a block's
/// eighteen so spent, one block of noise in a thousand was chased to a
/// codeword that stood as verified.
pub const CHASE_MIN_SLACK: usize = 2;

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
/// on a wrong codeword that verifies, so every pattern's verified
/// solution is kept, those within the standard over any that spend its
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
    rs_decode_chasing_bits(block, bit_reliability, ec_len, reserved, bits)
        .into_iter()
        .map(|(flipped, solution)| (flipped.iter().map(|bits| bits.count_ones() as usize).sum(), solution))
        .collect()
}

/// [`rs_decode_chasing`], with the bits of each codeword each
/// solution's pattern flipped in place of their count: for showing
/// which they were. `bits` is how many bits are flipped at the most,
/// [`CHASE_BITS`] or fewer ([`chase_bits`]).
pub fn rs_decode_chasing_bits(
    block: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    reserved: usize,
    bits: usize,
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
        let Ok(solution) = rs_decode_core(&trial, &[], ec_len, reserved, &syndromes).map(|s| s.among(patterns)) else { continue };
        // A thousand trials are a thousand chances for a wrong codeword
        // to lie within reach of the flipped block, so a chased solution
        // must show more redundancy than one found at the first try, and
        // must reproduce every known bit of the erased codewords: one
        // that overturns them has been read into the noise.
        if solution.slack < CHASE_MIN_SLACK || solution.flipped_known_bits > 0 {
            continue;
        }
        let solution = restated(block, solution);
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
    /// Positions of the block the locator is searched for roots over.
    n: usize,
    /// `powers[j * n + i]` is `j · (n - 1 - i) mod 255`: the logarithm
    /// of the `j`th power of position `i`'s locator, so that the locator
    /// is evaluated at a position by a lookup a coefficient.
    powers: Vec<u16>,
}

impl Locator {
    fn new(ec_len: usize, n: usize) -> Self {
        let powers = (0..=ec_len).flat_map(|j| (0..n).map(move |i| ((j * (n - 1 - i)) % 255) as u16)).collect();
        Self { c: Vec::with_capacity(ec_len + 2), b: Vec::with_capacity(ec_len + 2), previous: Vec::with_capacity(ec_len + 2), n, powers }
    }

    /// Whether the Forney syndromes `s` of a try have an error locator
    /// [`rs_decode_core`] would accept: of length at most `max_len`,
    /// of that degree, with as many roots among the `n` positions of
    /// the block and none at an `erased` position. The same test the
    /// decode makes, made cheaply: the shortest LFSR is grown as
    /// [`berlekamp_massey`] grows it, and the try given up the moment
    /// it grows past `max_len`, since it never shrinks.
    fn locates(&mut self, s: &[u8], max_len: usize, erased: &[usize]) -> bool {
        let Self { c, b, previous, n, powers } = self;
        let n = *n;
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
        // minus `j`, by the logarithms.
        let logs: Vec<(usize, usize)> = (1..=l).filter(|&j| c[j] != 0).map(|j| (j, log(c[j]))).collect();
        let mut roots = 0;
        for i in 0..n {
            let mut value = c[0];
            for &(j, lg) in &logs {
                // α^(-j·p) is α^(255 - j·p mod 255).
                value ^= exp_of_sum(lg + 255 - usize::from(powers[j * n + i]));
            }
            if value == 0 {
                if erased.binary_search(&i).is_ok() {
                    return false;
                }
                roots += 1;
            }
        }
        roots == l
    }
}

/// A solution of a block some of whose codewords were flipped or erased
/// for the try, restated against the block as read: an erratum for every
/// codeword the solution changed from what was read and for every one
/// that could not be read, `errors` counting the former, `erasures` the
/// latter. The slack stays that of the try's decode.
fn restated(block: &[CwRead], mut solution: RsSolution) -> RsSolution {
    let received: Vec<u8> = block.iter().map(|c| c.value & c.known).collect();
    solution.errata = (0..block.len())
        .filter(|&i| block[i].is_erasure() || solution.corrected[i] != received[i])
        .map(|i| Erratum { index: i, magnitude: solution.corrected[i] ^ received[i], was_erasure: block[i].is_erasure() })
        .collect();
    solution.errors = solution.errata.iter().filter(|e| !e.was_erasure).count();
    solution.erasures = solution.errata.len() - solution.errors;
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
/// so what a solution's located errors confirm is charged for every set
/// tried, of every size so far ([`RsSolution::among`]), a solution is
/// kept only with [`DOUBT_CONFIRMING_BITS`] behind it, and where the
/// sets reach different codewords the one nearest the block as read is
/// kept by a clear margin, or every one within the margin is returned,
/// nearest first ([`nearest`]). A size whose every solution spends the
/// codewords the standard reserves for detection does not end the
/// search: the sizes after it may reach the block within the standard,
/// and its solutions are returned only when none of them does
/// ([`preferring_the_standard`]). Each
/// comes with the codewords erased, restated against the block as read
/// as a chased one is. A codeword pinned whole is never erased. `sets`
/// is how many sets may be tried in all, [`TRIAL_SETS`] at the most: a
/// decode that is one of many has less to spend on each
/// ([`super::pipeline::DecodeOptions::trial_sets`]).
pub fn rs_decode_trying(
    block: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    reserved: usize,
    sets: usize,
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
    let mut tried = 0;
    let mut beyond: Vec<(Vec<usize>, RsSolution)> = Vec::new();
    let mut size = if left % 2 == 1 { 1 } else { 2 };
    while size < left {
        let budget = sets.min(TRIAL_SETS_MAX);
        let Some(count) = sets_of(candidates.len(), size, budget).filter(|&count| count > 0 && tried + count <= budget) else { break };
        tried += count;
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
            let Ok(solution) = rs_decode_core(block, &erased, ec_len, reserved, &syndromes).map(|s| s.among(tried)) else { return };
            if solution.confidence_bits() < DOUBT_CONFIRMING_BITS {
                return;
            }
            let solution = restated(block, solution);
            found.push((soft_distance(block, bit_reliability, &solution.corrected), (erased, solution)));
        });
        let within = found.iter().any(|(_, (_, solution))| solution.within_standard());
        let found = preferring_the_standard(found, |(_, solution)| solution);
        let kept = nearest(found, |a, b| a.1.corrected == b.1.corrected, |a, b| a.1.confidence_bits() > b.1.confidence_bits());
        if within && !kept.is_empty() {
            return kept;
        }
        if beyond.is_empty() {
            beyond = kept;
        }
        if work::cancelled() {
            break;
        }
        size += 2;
    }
    beyond
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
    /// Sets of codewords, or of bits, the trial tried, one where there
    /// was none: what the spare equations are charged for
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

    /// Independent bits confirming the solution: the spare equations,
    /// less what the sets tried cost, and the doubted bits' agreement
    /// beyond chance, as [`doubted_agreement`] weighs it.
    pub fn confidence_bits(&self) -> u32 {
        (self.solution.spare_equations as u32).saturating_sub(tries_charge(self.tries))
            + doubted_agreement(self.agreeing_bits(), self.changed_bits())
    }

    /// Codewords the solution changed.
    pub fn errors(&self) -> usize {
        self.changed.iter().filter(|&&c| c != 0).count()
    }
}

/// Decode by doubting bits: mark the least reliable bits unknown, fewest
/// first, and solve the rest over GF(2) as [`solve_bit_erasures`] does,
/// stopping at the first count that gives a complete, consistent solution
/// with at least [`DOUBT_SPARE_EQUATIONS`] spare equations. Doubting one
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
/// Returns the solutions of the first kind of search to find any, at
/// the fewest doubted bits it finds any at: one where it stands clear,
/// and the rivals the tries cannot tell apart besides, nearest first
/// ([`nearest`]). Empty when nothing solves.
pub fn solve_bit_erasures_doubting(
    block: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    trials: Trials,
) -> Vec<BitDoubt> {
    assert_eq!(block.len(), bit_reliability.len());
    let unknown_bits: usize = block.iter().map(|c| c.known.count_zeros() as usize).sum();
    let equations = 8 * ec_len;
    let budget = equations.saturating_sub(unknown_bits + DOUBT_SPARE_EQUATIONS);
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

    /// How a count of doubted bits is tried: as it is, with a few solid
    /// bits flipped, or with codewords of this many freed by trial.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Trial {
        Plain,
        Flips,
        Free(usize),
    }

    // The system of equations with the known bits carried, at each count
    // of doubted bits it was built for: the flips and every size of free
    // ask for the same one, and building it is most of a try. None where
    // no trial can help: consistent, the known bits solve as they are,
    // with too few equations to spare, since the count as it is found
    // nothing; incomplete, a trial takes equations the unknown bits
    // already lack.
    let systems: RefCell<HashMap<usize, Option<Rc<(BitSystem, ZeroRows)>>>> = RefCell::new(HashMap::new());
    let system_at = |k: usize, read: &[CwRead]| -> Option<Rc<(BitSystem, ZeroRows)>> {
        systems
            .borrow_mut()
            .entry(k)
            .or_insert_with(|| {
                let system = BitSystem::build(read, ec_len, true);
                if system.rank < system.unknown || system.consistent() {
                    return None;
                }
                let zero = system.zero_rows();
                Some(Rc::new((system, zero)))
            })
            .clone()
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
                let solution = solve_bit_erasures(&read, ec_len).ok().filter(BitSolution::is_complete)?;
                vec![(solution, Vec::new(), vec![0; n], 1)]
            }
            Trial::Flips => {
                let shared = system_at(k, &read)?;
                let (system, zero) = &*shared;
                let (tries, found) = system.flipping(zero, &read, bit_reliability, trials.sets);
                found.into_iter().map(|(flipped, solution)| (solution, Vec::new(), flipped, tries)).collect()
            }
            Trial::Free(size) => {
                let shared = system_at(k, &read)?;
                let (system, zero) = &*shared;
                let (tries, found) = system.trying(zero, &read, bit_reliability, size, trials.sets);
                found.into_iter().map(|(erased, _, solution)| (solution, erased, vec![0; n], tries)).collect()
            }
        };
        let doubts: Vec<BitDoubt> = found
            .into_iter()
            .filter(|(solution, _, _, tries)| solution.spare_equations.saturating_sub(tries_charge(*tries) as usize) >= DOUBT_SPARE_EQUATIONS)
            .map(|(solution, erased, flipped, tries)| {
                let mut doubted = doubted.clone();
                for &i in &erased {
                    doubted[i] |= block[i].known & !block[i].pinned;
                }
                for (d, &f) in doubted.iter_mut().zip(&flipped) {
                    *d |= f;
                }
                let changed = solution.corrected.iter().zip(block).zip(&doubted).map(|((&got, cw), &m)| (got ^ cw.value) & m).collect();
                BitDoubt { solution, doubted_bits: k, doubted, changed, flipped, erased, tries }
            })
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
    let mut kinds = vec![Trial::Plain];
    if trials.flips {
        kinds.push(Trial::Flips);
    }
    if trials.free {
        kinds.extend((1..=FREE_SIZES).map(Trial::Free));
    }
    kinds.iter().find_map(|&trial| search(&|k| attempt(k, trial))).unwrap_or_default()
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
    let u: usize = block.iter().map(|cw| cw.known.count_zeros() as usize).sum();
    let equations = 8 * ec_len;
    if u == 0 {
        return Err(BitSolveFailure::NoUnknowns);
    }
    if u > equations + PARTIAL_MARGIN {
        return Err(BitSolveFailure::TooManyUnknowns { unknown_bits: u, equations });
    }
    let system = BitSystem::build(block, ec_len, false);
    if !system.consistent() {
        return Err(BitSolveFailure::Inconsistent);
    }
    let solution = system.solution(block, &vec![0; n], equations - system.rank);
    if solution.solved_bits == 0 {
        return Err(BitSolveFailure::Underdetermined { rank: system.rank, unknown_bits: u });
    }
    debug_assert!(solution.solved_bits < u || rs_syndromes(&solution.corrected, ec_len).iter().all(|&s| s == 0));
    Ok(solution)
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
/// consistent, so the spare equations are charged for the sets tried
/// ([`BitDoubt::confidence_bits`]), and where two sets reach different
/// blocks the one nearest the block as read is kept by a clear margin,
/// or every one within the margin is returned, nearest first
/// ([`nearest`]). Pinned bits are never freed. Empty when the known
/// bits are consistent as they are, which is [`solve_bit_erasures`]'s
/// case, when the unknown bits are not all determined even before the
/// trial, or when no set makes them agree.
pub fn solve_bit_erasures_trying(block: &[CwRead], bit_reliability: &[[u8; 8]], ec_len: usize, trials: Trials) -> Vec<BitDoubt> {
    assert_eq!(block.len(), bit_reliability.len());
    let unknown_bits: usize = block.iter().map(|c| c.known.count_zeros() as usize).sum();
    if unknown_bits + DOUBT_SPARE_EQUATIONS >= 8 * ec_len {
        return Vec::new();
    }
    let system = BitSystem::build(block, ec_len, true);
    if system.rank < system.unknown || system.consistent() {
        return Vec::new();
    }
    let n = block.len();
    let keep = |(solution, erased, flipped, tries): (BitSolution, Vec<usize>, Vec<u8>, usize)| -> Option<BitDoubt> {
        if solution.spare_equations.saturating_sub(tries_charge(tries) as usize) < DOUBT_SPARE_EQUATIONS {
            return None;
        }
        let doubted: Vec<u8> =
            (0..n).map(|i| if erased.contains(&i) { block[i].known & !block[i].pinned } else { 0 } | flipped[i]).collect();
        let changed = solution.corrected.iter().zip(block).zip(&doubted).map(|((&got, cw), &m)| (got ^ cw.value) & m).collect();
        Some(BitDoubt { solution, doubted_bits: 0, doubted, changed, flipped, erased, tries })
    };
    let zero = system.zero_rows();
    if trials.flips {
        let (tries, found) = system.flipping(&zero, block, bit_reliability, trials.sets);
        let doubts: Vec<BitDoubt> =
            found.into_iter().filter_map(|(flipped, solution)| keep((solution, Vec::new(), flipped, tries))).collect();
        if !doubts.is_empty() {
            return doubts;
        }
    }
    for size in 1..=FREE_SIZES {
        if !trials.free {
            break;
        }
        let (tries, found) = system.trying(&zero, block, bit_reliability, size, trials.sets);
        let doubts: Vec<BitDoubt> =
            found.into_iter().filter_map(|(erased, _, solution)| keep((solution, erased, vec![0; n], tries))).collect();
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
    /// first.
    rows: Vec<Vec<u64>>,
    /// Row holding each unknown column's pivot, `usize::MAX` for none.
    pivot_row: Vec<usize>,
    rank: usize,
}

impl BitSystem {
    fn build(block: &[CwRead], ec_len: usize, with_known: bool) -> Self {
        let n = block.len();
        let equations = 8 * ec_len;
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
        let rhs = columns.len();
        let words = rhs / 64 + 1;
        let mut rows = vec![vec![0u64; words]; equations];
        for (col, &(i, bit)) in columns.iter().enumerate() {
            let x = exp(n - 1 - i);
            for j in 0..ec_len {
                let contribution = mul(1 << bit, gf256::pow(x, j));
                for k in 0..8 {
                    if contribution >> k & 1 != 0 {
                        rows[8 * j + k][col / 64] |= 1 << (col % 64);
                    }
                }
            }
        }
        let received: Vec<u8> = block.iter().map(|c| c.value & c.known).collect();
        for (j, &s) in rs_syndromes(&received, ec_len).iter().enumerate() {
            for k in 0..8 {
                if s >> k & 1 != 0 {
                    rows[8 * j + k][rhs / 64] |= 1 << (rhs % 64);
                }
            }
        }
        let mut rank = 0;
        let mut pivot_row = vec![usize::MAX; unknown];
        for (col, pivot_of) in pivot_row.iter_mut().enumerate() {
            let Some(p) = (rank..equations).find(|&r| Self::bit(&rows[r], col)) else { continue };
            rows.swap(rank, p);
            let pivot = rows[rank].clone();
            for (r, row) in rows.iter_mut().enumerate() {
                if r != rank && Self::bit(row, col) {
                    for (a, b) in row.iter_mut().zip(&pivot) {
                        *a ^= b;
                    }
                }
            }
            *pivot_of = rank;
            rank += 1;
        }
        Self { columns, unknown, equations, rows, pivot_row, rank }
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
        self.rows[self.rank..].iter().all(|row| !self.rhs(row))
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
            let row = &self.rows[self.pivot_row[col]];
            if self.unknowns_in(row) != 1 {
                continue;
            }
            solved[i] |= 1 << bit;
            solved_bits += 1;
            if self.rhs(row) ^ (moved.iter().filter(|&&m| Self::bit(row, m)).count() % 2 == 1) {
                corrected[i] |= 1 << bit;
            }
        }
        BitSolution { corrected, solved, unknown_bits: self.unknown, solved_bits, spare_equations: spare }
    }

    /// A column, or the right-hand side, over the rows no unknown has a
    /// pivot in: what the known bits must answer for by themselves.
    fn over_zero(&self, col: usize) -> Vec<u64> {
        let zero = self.equations - self.rank;
        let mut v = vec![0u64; zero / 64 + 1];
        for (k, row) in self.rows[self.rank..].iter().enumerate() {
            if Self::bit(row, col) {
                v[k / 64] |= 1 << (k % 64);
            }
        }
        v
    }

    /// The rows no unknown has a pivot in, as every known column and
    /// the right-hand side stand over them: computed once for the flips
    /// and every size of free a block is tried with.
    fn zero_rows(&self) -> ZeroRows {
        ZeroRows {
            residual: self.over_zero(self.columns.len()),
            columns: (self.unknown..self.columns.len()).map(|col| self.over_zero(col)).collect(),
        }
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
    /// ([`nearest`]).
    fn trying(
        &self,
        over: &ZeroRows,
        block: &[CwRead],
        bit_reliability: &[[u8; 8]],
        size: usize,
        budget: usize,
    ) -> (usize, Vec<(Vec<usize>, Vec<u8>, BitSolution)>) {
        debug_assert!(self.rank == self.unknown && !self.consistent());
        let n = block.len();
        let zero = self.equations - self.rank;
        let mut columns_of: Vec<Vec<usize>> = vec![Vec::new(); n];
        for col in self.unknown..self.columns.len() {
            columns_of[self.columns[col].0].push(col);
        }
        let candidates: Vec<usize> = (0..n).filter(|&i| !columns_of[i].is_empty()).collect();
        let c = candidates.len();
        let budget = budget.min(TRIAL_SETS_MAX);
        let tries: Option<usize> = (1..=size).map(|s| sets_of(c, s, budget)).sum();
        let Some(tries) = tries.filter(|&tries| tries <= budget) else { return (0, Vec::new()) };
        if size == 0 || size > c || zero < 8 * size + DOUBT_SPARE_EQUATIONS {
            return (tries, Vec::new());
        }
        let residual = &over.residual;
        let vectors: Vec<Vec<&[u64]>> =
            candidates.iter().map(|&i| columns_of[i].iter().map(|&col| over.columns[col - self.unknown].as_slice()).collect()).collect();
        let mut found: Vec<(u32, (Vec<usize>, Vec<u8>, BitSolution))> = Vec::new();
        each_set(c, size, |set| {
            let mut span = Span::default();
            let mut freed = 0;
            for (which, &k) in set.iter().enumerate() {
                for (b, &v) in vectors[k].iter().enumerate() {
                    span.add(v.to_vec(), 1 << (8 * which + b));
                    freed += 1;
                }
            }
            if span.rows.len() != freed {
                return;
            }
            let Some(combination) = span.express(residual.clone()) else { return };
            let mut changed = vec![0u8; n];
            for (which, &k) in set.iter().enumerate() {
                let i = candidates[k];
                for (b, &col) in columns_of[i].iter().enumerate() {
                    if combination >> (8 * which + b) & 1 != 0 {
                        changed[i] |= 1 << self.columns[col].1;
                    }
                }
            }
            let solution = self.solution(block, &changed, zero - freed);
            debug_assert!(solution.is_complete());
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
    /// rivals within the margin after it ([`nearest`]).
    fn flipping(&self, over: &ZeroRows, block: &[CwRead], bit_reliability: &[[u8; 8]], budget: usize) -> (usize, Vec<(Vec<u8>, BitSolution)>) {
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
        cols.sort();
        let c = cols.len();
        if c == 0 || zero < DOUBT_SPARE_EQUATIONS + tries_charge(c) as usize {
            return (0, Vec::new());
        }
        let residual = &over.residual;
        let vectors: Vec<&[u64]> = cols.iter().map(|&(_, col)| over.columns[col - self.unknown].as_slice()).collect();
        let mut sets: Vec<Vec<usize>> = Vec::new();
        let mut tries = c;
        for (k, &v) in vectors.iter().enumerate() {
            if v == residual.as_slice() {
                sets.push(vec![k]);
            }
        }
        let pairs = c * (c - 1) / 2;
        if c >= 2 && zero >= DOUBT_SPARE_EQUATIONS + tries_charge(tries + pairs) as usize {
            tries += pairs;
            let mut by_vector: HashMap<&[u64], Vec<usize>> = HashMap::new();
            let xor = |a: &[u64], b: &[u64]| -> Vec<u64> { a.iter().zip(b).map(|(x, y)| x ^ y).collect() };
            for (k, &v) in vectors.iter().enumerate() {
                if let Some(others) = by_vector.get(xor(residual, v).as_slice()) {
                    sets.extend(others.iter().map(|&a| vec![a, k]));
                }
                by_vector.entry(v).or_default().push(k);
            }
            // As many of the least reliable bits as have no more pairs
            // than four lookups a set of the budget: a lookup is a hash of
            // a few words where a set erased is a decode.
            let s = c.min(FLIP_TRIPLE_BITS).min((8 * budget.min(TRIAL_SETS_MAX)).isqrt());
            let triples: usize = (0..s).map(|a| (a + 1..s).map(|b| c - b - 1).sum::<usize>()).sum();
            if c >= 3 && zero >= DOUBT_SPARE_EQUATIONS + tries_charge(tries + triples) as usize {
                tries += triples;
                for a in 0..s {
                    for b in a + 1..s {
                        let target = xor(&xor(residual, vectors[a]), vectors[b]);
                        if let Some(others) = by_vector.get(target.as_slice()) {
                            sets.extend(others.iter().filter(|&&k| k > b).map(|&k| vec![a, b, k]));
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
                (soft_distance(block, bit_reliability, &solution.corrected), (flipped, solution))
            })
            .collect();
        (tries, nearest(found, |a, b| a.1.corrected == b.1.corrected, |a, b| a.1.spare_equations > b.1.spare_equations))
    }
}

/// The rows of a [`BitSystem`] no unknown has a pivot in, as the known
/// columns and the right-hand side stand over them ([`BitSystem::zero_rows`]).
struct ZeroRows {
    /// The right-hand side: what the known bits as read leave wanting.
    residual: Vec<u64>,
    /// Each known column, by its place among the known columns.
    columns: Vec<Vec<u64>>,
}

/// Vectors over GF(2) in fully reduced echelon form, each with the
/// combination of the vectors added that it is made of, so that a vector
/// the span holds can be expressed in them.
#[derive(Clone, Default)]
struct Span {
    /// Each row with the index of its pivot bit and its combination.
    rows: Vec<(Vec<u64>, usize, u32)>,
}

impl Span {
    fn bit(v: &[u64], at: usize) -> bool {
        v[at / 64] >> (at % 64) & 1 != 0
    }

    fn xor(v: &mut [u64], w: &[u64]) {
        for (a, b) in v.iter_mut().zip(w) {
            *a ^= b;
        }
    }

    /// Add a vector, `combination` naming it; false when it lay in the
    /// span already.
    fn add(&mut self, mut v: Vec<u64>, mut combination: u32) -> bool {
        for (row, pivot, c) in &self.rows {
            if Self::bit(&v, *pivot) {
                Self::xor(&mut v, row);
                combination ^= c;
            }
        }
        let Some(word) = v.iter().position(|&w| w != 0) else { return false };
        let pivot = 64 * word + v[word].trailing_zeros() as usize;
        for (row, _, c) in &mut self.rows {
            if Self::bit(row, pivot) {
                Self::xor(row, &v);
                *c ^= combination;
            }
        }
        self.rows.push((v, pivot, combination));
        true
    }

    /// The combination of the vectors added that makes `target`, when
    /// the span holds it.
    fn express(&self, mut target: Vec<u64>) -> Option<u32> {
        let mut combination = 0;
        for (row, pivot, c) in &self.rows {
            if Self::bit(&target, *pivot) {
                Self::xor(&mut target, row);
                combination ^= c;
            }
        }
        target.iter().all(|&w| w == 0).then_some(combination)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // the wrong values and the known nibbles expose it.
        let victim = (0..block.len()).find(|i| !erased.contains(i)).unwrap();
        read[victim].value ^= 0x5A;
        assert!(matches!(rs_decode(&read, ec_len), Err(RsFailure::KnownBitConflict { .. })));
    }

    #[test]
    fn known_bits_a_solution_overturns_count_against_it() {
        // Nineteen of twenty error correction codewords spent on erasures
        // whose seven known bits each are noise, and the codewords read in
        // full all right: the decoder lands on a codeword with one
        // codeword of slack, and half the known bits it was given
        // contradict it. Told from a wrong solution it cannot be, since a
        // wrong one reads just so, and it is refused.
        let mut rng = Rng(21);
        let ec_len = 20;
        let block = codeword(&mut rng, 80, ec_len);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        let erased = rng.distinct(ec_len - 1, block.len());
        for &i in &erased {
            read[i] = CwRead { value: rng.byte() & 0x7F, known: 0x7F, pinned: 0 };
        }
        assert!(matches!(rs_decode(&read, ec_len), Err(RsFailure::KnownBitConflict { .. })));

        // The same erasures with their known bits read right but for one
        // in each of three codewords: the solution stands, and is
        // credited the bits it reproduced less two for each it overturned.
        for &i in &erased {
            read[i] = CwRead { value: block[i] & 0x7F, known: 0x7F, pinned: 0 };
        }
        for &i in &erased[..3] {
            read[i].value ^= 0x10;
        }
        let sol = rs_decode(&read, ec_len).unwrap();
        assert_eq!(sol.corrected, block);
        assert_eq!((sol.slack, sol.flipped_known_bits), (1, 3));
        assert_eq!(sol.check_bits, 7 * 19 - 3 - OVERTURN_COST * 3);

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
        let (doubted, sol) = rs_decode_doubting(&read, &reliability, ec_len, 0).unwrap();
        assert_eq!(sol.corrected, block);
        // Erasing all seven wrong ones leaves three codewords of slack,
        // more confirmation than any other count of doubted codewords.
        assert_eq!(doubted, 7);
        assert_eq!((sol.errors, sol.erasures, sol.slack), (0, 7, 3));

        // Certain codewords are never doubted, however wrong.
        reliability.iter_mut().for_each(|r| *r = CERTAIN);
        assert_eq!(rs_decode_doubting(&read, &reliability, ec_len, 0), None);

        // A solution with nothing to spare is not accepted.
        let ec_len = 4;
        let block = codeword(&mut rng, 12, ec_len);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        let reliability: Vec<u8> = (0..block.len()).map(|i| i as u8).collect();
        for cw in read.iter_mut().take(4) {
            cw.value ^= rng.nonzero();
        }
        assert_eq!(rs_decode_doubting(&read, &reliability, ec_len, 0), None);
        // Nor is one with a single codeword spare whose doubted codewords
        // agree with it no better than chance: one syndrome byte is all
        // that stands behind it, which noise passes once in 256 tries,
        // and doubting tries a count of codewords at a time.
        read[3] = CwRead::known(block[3]);
        let plain = rs_decode_doubted(&read, &[0, 1, 2], ec_len, 0).unwrap();
        assert_eq!((plain.slack, plain.confidence_bits(), plain.corrected == block), (1, 8, true));
        assert_eq!(rs_decode_doubting(&read, &reliability, ec_len, 0), None);

        // Codewords misread in one faint bit each bear the solution out
        // by the bits they got right, and one spare codeword is enough.
        let ec_len = 8;
        let block = codeword(&mut rng, 24, ec_len);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        let reliability: Vec<u8> = (0..block.len()).map(|i| i as u8).collect();
        for (i, cw) in read.iter_mut().take(7).enumerate() {
            cw.value ^= 1 << i;
        }
        let (doubted, sol) = rs_decode_doubting(&read, &reliability, ec_len, 0).unwrap();
        assert_eq!((doubted, sol.slack), (7, 1));
        assert_eq!(sol.confidence_bits(), 8 + doubted_agreement(49, 7));
        assert_eq!(sol.corrected, block);
        // With two codewords reserved for detection that solution is
        // beyond the standard's limit, and doubting offers it so marked,
        // having none within it.
        let (_, beyond) = rs_decode_doubting(&read, &reliability, ec_len, 2).unwrap();
        assert!(!beyond.within_standard() && beyond.corrected == block);
    }

    #[test]
    fn noise_does_not_pass_for_a_doubted_or_chased_solution() {
        // Blocks of random codewords, every one of them doubtful: nothing
        // is there to find, and what the fallbacks return, if anything,
        // must not count as confirmed. Before a doubted solution was
        // asked for more than one spare codeword, one block in two
        // hundred of these verified.
        let mut rng = Rng(0x0153);
        for (n, ec_len) in [(26usize, 7usize), (26, 10), (26, 17), (44, 28), (13, 7)] {
            for _ in 0..600 {
                let read: Vec<CwRead> = (0..n).map(|_| CwRead::known(rng.byte())).collect();
                let reliability: Vec<u8> = (0..n).map(|_| rng.byte().min(CERTAIN - 1)).collect();
                let found = rs_decode_doubting(&read, &reliability, ec_len, 0);
                assert!(found.is_none(), "({n}, {ec_len}): {found:?}");
            }
        }
        // M1's block has two codewords and the standard reserves both: a
        // read of it is right or wrong, and no pattern of flips is tried.
        let block = codeword(&mut rng, 5, 2);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        read[1].value ^= 0x04;
        let mut reliability = vec![[200u8; 8]; block.len()];
        reliability[1][2] = 3;
        assert!(rs_decode_chasing(&read, &reliability, 2, 2, CHASE_BITS).is_empty());
        assert!(only(rs_decode_chasing(&read, &reliability, 2, 0, CHASE_BITS)).is_some_and(|(_, s)| s.corrected == block));
    }

    #[test]
    fn doubted_codewords_keep_their_agreeing_bits_as_checks() {
        // Beyond chance: a doubted codeword agrees with any solution in
        // half its bits, so one bit of disagreement in eight is worth
        // nothing alone, while seven codewords right in seven bits of
        // eight are worth far more than their one slack codeword.
        assert_eq!(doubted_agreement(7, 1), 0);
        assert_eq!(doubted_agreement(4, 4), 0);
        assert_eq!(doubted_agreement(49, 7), 42 - 23);
        assert_eq!(doubted_agreement(56, 8), 48 - 24);

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
        let (doubted, sol) = rs_decode_doubting(&read, &reliability, ec_len, 0).unwrap();
        assert_eq!(sol.corrected, block);
        assert_eq!((doubted, sol.erasures, sol.slack), (7, 7, 3));
        // Every doubted codeword counts as an erasure, with the bits the
        // solution agreed with, beyond the one it overturned, as checks.
        assert_eq!(sol.flipped_known_bits, 7);
        assert_eq!(sol.check_bits, doubted_agreement(49, 7));
        assert_eq!(sol.confidence_bits(), 24 + doubted_agreement(49, 7));
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
        let (doubted, sol) = rs_decode_doubting(&read, &reliability, ec_len, 0).unwrap();
        assert_eq!((doubted, sol.slack, sol.corrected == block), (10, 0, true));
        assert_eq!(sol.confidence_bits(), doubted_agreement(70, 10));
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        for &i in &picked {
            read[i].value = rng.byte();
        }
        let garbage = rs_decode_doubting(&read, &reliability, ec_len, 0);
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
            if let Some((_, sol)) = rs_decode_doubting(&read, &by_codeword, ec_len, 0) {
                assert_ne!(sol.corrected, block, "round {round}");
            }
            let doubt = only(solve_bit_erasures_doubting(&read, &reliability, ec_len, Trials::NONE)).unwrap_or_else(|| panic!("round {round}"));
            assert_eq!(doubt.solution.corrected, block, "round {round}");
            assert!(doubt.doubted_bits >= wrong && doubt.doubted_bits <= wrong + 3, "round {round}: {}", doubt.doubted_bits);
            assert_eq!(doubt.errors(), wrong);
            assert_eq!(doubt.changed_bits(), wrong as u32);
            assert_eq!(doubt.solution.spare_equations, 8 * ec_len - doubt.doubted_bits);
            assert!(doubt.confidence_bits() >= DOUBT_SPARE_EQUATIONS as u32);
        }
        // Pinned and certain bits are never doubted, and a block with no
        // faint bit has nothing to try.
        let ec_len = 6;
        let block = codeword(&mut rng, 20, ec_len);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        read[4].value ^= 0x08;
        assert!(solve_bit_erasures_doubting(&read, &vec![[CERTAIN; 8]; block.len()], ec_len, Trials::NONE).is_empty());
        let mut reliability = vec![[200u8; 8]; block.len()];
        reliability[4][3] = 1;
        assert!(!solve_bit_erasures_doubting(&read, &reliability, ec_len, Trials::NONE).is_empty());
        read[4].pinned = 0x08;
        assert!(solve_bit_erasures_doubting(&read, &reliability, ec_len, Trials::NONE).iter().all(|d| d.doubted[4] & 0x08 == 0));
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
        assert_eq!(rs_decode_doubting(&read, &by_codeword, ec_len, 0), None);
        assert!(solve_bit_erasures_doubting(&read, &reliability, ec_len, Trials::NONE).is_empty());
        let (flipped, sol) = only(rs_decode_chasing(&read, &reliability, ec_len, 0, CHASE_BITS)).unwrap();
        assert_eq!(sol.corrected, block);
        assert_eq!((flipped, sol.errors, sol.erasures, sol.slack), (3, 7, 0, 2));
        // Two codewords spare, and the four solid errors the code located
        // among the forty read, less ten bits for the thousand and
        // twenty-three patterns of the ten faintest bits that were tried.
        assert_eq!(sol.located_bits, located_error_bits(40, 4) - 10);
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
    fn the_nearest_of_solutions_that_disagree_wins_by_a_clear_margin_or_not_at_all() {
        // A certain bit overturned costs 256, a faint one its reliability
        // and one, and a bit that was not read nothing.
        let block = [CwRead::known(0b1010_1010), CwRead { value: 0x0F, known: 0x0F, pinned: 0 }];
        let certain = [[CERTAIN; 8]; 2];
        assert_eq!(soft_distance(&block, &certain, &[0b1010_1010, 0x0F]), 0);
        assert_eq!(soft_distance(&block, &certain, &[0b1010_1011, 0xFF]), 256);
        let mut faint = certain;
        faint[0][0] = 9;
        faint[1][1] = 0;
        assert_eq!(soft_distance(&block, &faint, &[0b1010_1011, 0x0D]), 11);

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
                let (erased, solution) = only(rs_decode_trying(&read, &certain, ec_len, 0, TRIAL_SETS)).unwrap_or_else(|| panic!("{what}"));
                assert_eq!(solution.corrected, block, "{what}");
                assert_eq!(erased.len(), 2 - ec_len % 2, "{what}");
                assert!(erased.iter().all(|i| wrong[..limit + 1].contains(i)), "{what}");
                assert_eq!((solution.errors, solution.erasures), (limit + 1, 0), "{what}");
                assert!(solution.confidence_bits() >= AIDED_VERIFIED_BITS, "{what}: {}", solution.confidence_bits());
                // One more is reached by the sets two larger, where the
                // block is small enough for those to fit the budget with
                // the smaller sets before them, and past the trials where
                // it is not.
                read[wrong[limit + 1]].value ^= rng.nonzero();
                let size = 2 - ec_len % 2;
                let fits = sets_of(n, size, TRIAL_SETS).zip(sets_of(n, size + 2, TRIAL_SETS)).is_some_and(|(small, large)| small + large <= TRIAL_SETS);
                let found = rs_decode_trying(&read, &certain, ec_len, 0, TRIAL_SETS);
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
                assert!(rs_decode_trying(&read, &certain, ec_len, 0, TRIAL_SETS).is_empty(), "{what} and two more");
            }
        }
        // A pinned codeword is never erased.
        let block = codeword(&mut rng, 16, 18);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        for cw in &mut read[..10] {
            cw.value ^= rng.nonzero();
            cw.pinned = 0xFF;
        }
        assert!(rs_decode_trying(&read, &vec![[CERTAIN; 8]; 34], 18, 0, TRIAL_SETS).is_empty());
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
        let (erased, solution) = only(rs_decode_trying(&read, &certain, 30, 0, TRIAL_SETS)).expect("pairs of a large block");
        assert_eq!((erased.len(), solution.errors, solution.corrected == block), (2, 16, true));
        assert_eq!(sets_of(130, 3, TRIAL_SETS), None);
        assert_eq!(sets_of(130, 3, TRIAL_SETS_MAX), Some(130 * 129 * 128 / 6));
        let block = codeword(&mut rng, 101, 29);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        for cw in &mut read[..15] {
            cw.value ^= rng.nonzero();
        }
        assert_eq!(only(rs_decode_trying(&read, &certain, 29, 0, TRIAL_SETS)).map(|(erased, s)| (erased.len(), s.errors)), Some((1, 15)));
        read[15].value ^= rng.nonzero();
        assert!(rs_decode_trying(&read, &certain, 29, 0, TRIAL_SETS).is_empty());
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
                let doubt = only(solve_bit_erasures_trying(&read, &certain, ec_len, Trials { free: true, ..Trials::NONE })).unwrap_or_else(|| panic!("{what}"));
                assert_eq!(doubt.solution.corrected, block, "{what}");
                assert_eq!(doubt.erased.len(), stains, "{what}");
                assert!(doubt.erased.iter().all(|i| picked[unknown..].contains(i)), "{what}");
                assert_eq!((doubt.doubted_bits, doubt.errors()), (0, stains), "{what}");
                assert!(doubt.confidence_bits() >= DOUBT_SPARE_EQUATIONS as u32, "{what}: {}", doubt.confidence_bits());
                for &i in &doubt.erased {
                    assert_eq!(doubt.doubted[i], 0xFF, "{what}");
                    assert_eq!(doubt.changed[i], read[i].value ^ block[i], "{what}");
                }
                // Three stains are reached by the triples, and four are
                // past them.
                if stains == 2 {
                    let third = (0..n).find(|i| !picked.contains(i)).unwrap();
                    read[third].value ^= rng.nonzero();
                    let doubt = only(solve_bit_erasures_trying(&read, &certain, ec_len, Trials { free: true, ..Trials::NONE })).unwrap_or_else(|| panic!("{what} and a third"));
                    assert_eq!(doubt.solution.corrected, block, "{what} and a third");
                    assert_eq!(doubt.erased.len(), 3, "{what} and a third");
                    assert!(doubt.erased.contains(&third), "{what} and a third");
                    let fourth = (0..n).find(|i| !picked.contains(i) && *i != third).unwrap();
                    read[fourth].value ^= rng.nonzero();
                    assert!(solve_bit_erasures_trying(&read, &certain, ec_len, Trials { free: true, ..Trials::NONE }).is_empty(), "{what} and a fourth");
                }
            }
        }
        // Consistent as read, there is nothing to free.
        let block = codeword(&mut rng, 16, 18);
        let mut read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        read[3].known = 0xF0;
        assert!(solve_bit_erasures_trying(&read, &vec![[CERTAIN; 8]; 34], 18, Trials::ALL).is_empty());
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
                assert!(solve_bit_erasures_trying(&read, &reliability, ec_len, Trials::NONE).is_empty(), "{what}");
                let flipping = Trials { flips: true, ..Trials::NONE };
                let doubt = only(solve_bit_erasures_trying(&read, &reliability, ec_len, flipping)).unwrap_or_else(|| panic!("{what}"));
                assert_eq!(doubt.solution.corrected, block, "{what}");
                assert_eq!(doubt.flipped, wrong, "{what}");
                assert_eq!((doubt.flipped_bits() as usize, doubt.errors(), doubt.erased.len(), doubt.doubted_bits), (flips, flips, 0, 0), "{what}");
                assert_eq!(doubt.solution.spare_equations, 8 * ec_len - unknown, "{what}");
                assert!(doubt.confidence_bits() >= DOUBT_SPARE_EQUATIONS as u32, "{what}: {}", doubt.confidence_bits());
                // The frees reach the same block, spending more; the
                // flips are preferred, and so is a flip to a free when
                // both are allowed.
                let both = only(solve_bit_erasures_trying(&read, &reliability, ec_len, Trials::ALL)).unwrap_or_else(|| panic!("{what}"));
                assert_eq!((both.flipped, both.erased.len()), (wrong, 0), "{what}");
                // Painted by hand, every bit is certain and none is
                // flipped; the frees are all there is.
                let certain = vec![[CERTAIN; 8]; n];
                assert!(solve_bit_erasures_trying(&read, &certain, ec_len, flipping).is_empty(), "{what}");
                let freed = solve_bit_erasures_trying(&read, &certain, ec_len, Trials::ALL);
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
            assert!(solve_bit_erasures_doubting(&read, &reliability, ec_len, Trials::NONE).is_empty(), "{what}");
            let flipping = Trials { flips: true, ..Trials::NONE };
            let doubt = only(solve_bit_erasures_doubting(&read, &reliability, ec_len, flipping)).unwrap_or_else(|| panic!("{what}"));
            assert_eq!(doubt.solution.corrected, block, "{what}");
            // The two solid bits are flipped, and now and then a faint one
            // not yet doubted with them, at a count of doubted bits one
            // short of it.
            let flipped = doubt.flipped_bits() as usize;
            assert!((2..=3).contains(&flipped), "{what}: flipped {flipped}");
            assert!(doubt.doubted_bits + flipped >= faint + 2, "{what}: doubted {} bits", doubt.doubted_bits);
            assert_eq!(doubt.errors(), faint + 2, "{what}");
            assert!(doubt.confidence_bits() >= DOUBT_SPARE_EQUATIONS as u32, "{what}: {}", doubt.confidence_bits());
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
            assert!(solve_bit_erasures_doubting(&read, &reliability, ec_len, Trials::NONE).is_empty(), "{what}");
            let free = Trials { free: true, ..Trials::NONE };
            let doubt = only(solve_bit_erasures_doubting(&read, &reliability, ec_len, free)).unwrap_or_else(|| panic!("{what}"));
            assert_eq!(doubt.solution.corrected, block, "{what}");
            assert_eq!(doubt.erased, vec![stain], "{what}");
            assert!(doubt.doubted_bits >= faint, "{what}: doubted {} bits", doubt.doubted_bits);
            assert_eq!(doubt.doubted[stain], 0xFF, "{what}");
            assert_eq!(doubt.errors(), faint + 1, "{what}");
            assert!(doubt.confidence_bits() >= DOUBT_SPARE_EQUATIONS as u32, "{what}: {}", doubt.confidence_bits());
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
        // A byte of confirmation: what the plain decode of a block read
        // whole stands on, and a third of what this must show.
        assert_eq!(sol.confidence_bits(), VERIFIED_BITS);
        assert!(sol.confidence_bits() < AIDED_VERIFIED_BITS);

        // With fifteen unknown there are three syndromes left, and no
        // value of the second wrong codeword passes them: one block in
        // sixteen million would, or one in two thousand as a single
        // wrong codeword the code located with a codeword to spare,
        // which is short of three as well.
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
            let erasures = rng.below(ec_len.min(n) / 2 + 1);
            let wrong = rng.below(ec_len / 2 + 2);
            for &i in &rng.distinct((erasures + wrong).min(n), n) {
                if erasures > 0 && read[i].known == 0xFF && rng.below(2) == 0 {
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
            let full = rs_decode_core(&read, &[], ec_len, 0, &syndromes);
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
            let found = rs_decode_trying(&read, &vec![[CERTAIN; 8]; n], ec_len, 0, TRIAL_SETS);
            let took = started.elapsed();
            // Against every set decoded in full, as the trials once did.
            let received: Vec<u8> = read.iter().map(|c| c.value & c.known).collect();
            let syndromes = rs_syndromes(&received, ec_len);
            let (mut sets, mut solved) = (0, 0);
            let started_slow = std::time::Instant::now();
            for size in [1, 2] {
                each_set(n, size, |set| {
                    sets += 1;
                    solved += usize::from(rs_decode_core(&read, set, ec_len, 0, &syndromes).is_ok());
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
        assert_eq!(soft_count_bits(&block, &certain, 8 * 18, 8 * 256), (144.0 - sets_of_eight.log2() - (255f64 / 254.0).log2()).floor() as u32);
        assert!(soft_count_bits(&block, &certain, 8 * 18, 8 * 256) < list_solution_bits(33, 18, 1));
        assert_eq!(soft_count_bits(&block, &certain, 8 * 18, SOFT_COUNT_REACH + 1), 0);
        // Twenty faint bits among solid ones: the sets within the cost
        // of overturning ten of them are sets of the faint bits mostly,
        // and the count is far more than the hard count's for the ten
        // codewords those bits lie in.
        let mut faint = vec![[200u8; 8]; 33];
        for i in 0..20 {
            faint[i][i % 8] = 4;
        }
        let soft = soft_count_bits(&block, &faint, 8 * 18, 10 * 5);
        assert!(soft > 100 && soft > list_solution_bits(33, 18, 10), "{soft}");
        // Pinned bits are not counted; unknown bits spend equations.
        let mut pinned = block.clone();
        for cw in &mut pinned[..20] {
            cw.pinned = 0xFF;
        }
        assert!(soft_count_bits(&pinned, &faint, 8 * 18, 10 * 5) > soft);
        assert_eq!(soft_count_bits(&block, &faint, 8 * 18 - 40, 10 * 5), soft - 40);
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
}
