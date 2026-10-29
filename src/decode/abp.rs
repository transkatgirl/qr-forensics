//! Adaptive belief propagation over the binary image of one Reed-Solomon
//! block: Jiang and Narayanan's iterative soft decoding, which adapts
//! the parity-check matrix to the reliability of the bits each round and
//! passes beliefs between them along it.
//!
//! Every solver of [`super::rs`] searches from the block as read with
//! each bit's reliability fixed: the doubting erases the faintest, the
//! chase flips them, and the curve of [`super::list`] weighs them, but
//! none lets what the equations say of one bit change what is believed
//! of another. Belief propagation does that. The block's code is linear
//! over GF(2), eight equations to a syndrome, and each equation is a
//! parity check over the bits it involves: a check that fails says one
//! of its bits is wrong, the fainter the likelier, and a check that
//! passes bears its bits out. Passed round the checks a few times, the
//! beliefs settle toward a word every check passes. On a matrix as
//! dense as a Reed-Solomon code's the passing alone goes nowhere, since
//! every check involves half the bits; Jiang and Narayanan's step is to
//! reduce the matrix, each round, so that the least reliable bits each
//! sit in a check of their own, where the solid bits of the rest of the
//! check speak to them and nothing speaks back. The hard decision after
//! each round is decoded as any read is, and the solution restated
//! against the block as read; after the third round and every doubling
//! of it, the word and the reliabilities the beliefs leave are given to
//! the doubting of the faintest bits one by one
//! ([`super::rs::solve_bit_erasures_doubting`]), which is where the
//! propagation earns its keep, since the doubting reaches what the
//! beliefs have sorted right that the reliabilities as read had not,
//! and to the chase ([`super::rs::rs_decode_chasing_bits`]). The rounds
//! stop at the first that finds a solution, so that a block given more
//! rounds is given back what fewer found, whenever they found anything.
//! On the blocks of a photo the doubting alone, over the reliabilities
//! as read, is the stronger solver; over the beliefs it reads blocks
//! besides that nothing else does, where the misread bits are faint but
//! outnumbered by faint bits read right.
//!
//! What such a solution confirms is counted against the block as read,
//! never against the word the beliefs made, which the propagation carries
//! toward some codeword whatever the block holds: the hard count of
//! [`super::rs::list_solution_bits`], for the codewords it changes, or
//! the soft count of [`super::rs::soft_count_bits`], for the bits it
//! overturns weighed by their reliability, whichever is the more. A block
//! of noise is carried to a codeword far from it, overturning solid bits
//! by the dozen, and both counts refuse that; a block misread in faint
//! bits is carried to the codeword that overturns those alone, which the
//! soft count confirms however many codewords they lie in. Nothing is
//! credited to the bits the propagation overturned as such, which were
//! chosen by what the equations wanted and not before any solution was
//! seen, and the rounds are charged as tries: the rounds, the doubting
//! among them and the chase are each given an equal share of the
//! propagation's chances, and each round, set of doubted bits or pattern
//! a share of its own, the same however many rounds follow it.

use std::f32::consts::LN_2;

use super::gf256::{exp, mul};
use super::rs::{
    CHASE_MAX_BITS, CwRead, RsSolution, SPARE_EQUATIONS, Sightings, Trials, counted_against, judged, nearest,
    rs_decode_chasing_bits_against, rs_decode_core, rs_syndromes, soft_distance, solve_bit_erasures_doubting, tries_charge,
};
use crate::symbol::grid::CERTAIN;
use crate::work;

/// A solution found: the rounds it took, the bits of each codeword the
/// propagation overturned, and the solution itself.
pub type Propagated = (usize, Vec<u8>, RsSolution);

/// Rounds of adapting and propagating a block is given at the most.
/// Each is a reduction of the parity-check matrix and a pass over it,
/// a millisecond or two on the largest blocks, and a decode of the
/// hard decision; a block that settles on a codeword, or that a round
/// finds a solution of, stops early.
pub const ABP_ROUNDS: usize = 20;
/// Fewest rounds a decode that is one of many is given.
pub const ABP_ROUNDS_MIN: usize = 6;
/// Most rounds a block may be given ([`super::pipeline::DecodeOptions::abp_rounds`]):
/// a tenth or a fifth of a second on the largest blocks, and a solution
/// found in the last of them needs eleven bits more to confirm it, nine
/// for the round and two for the share of the propagation's chances the
/// rounds leave the doubting and the chase.
pub const ABP_ROUNDS_MAX: usize = 100;
/// The first round the doubting and the chase run after: half the
/// fewest a decode that is one of many is given, so that the last of
/// those is one of them too ([`is_checkpoint`]).
const FIRST_CHECKPOINT: usize = ABP_ROUNDS_MIN / 2;
/// Fewest rounds a block may be given
/// ([`super::pipeline::DecodeOptions::abp_rounds`]): as far as the first
/// round the doubting and the chase run after, the third. Given fewer,
/// the propagation never handed its beliefs to them, which it once did
/// after the last round whatever that was; and run after the last round
/// given, they would take from a block given more rounds what it found
/// with fewer.
pub const ABP_ROUNDS_LEAST: usize = FIRST_CHECKPOINT;
/// Rounds the doubting and the chase run after, at the most: those of
/// the rounds up to [`ABP_ROUNDS_MAX`] ([`is_checkpoint`]).
const CHECKPOINTS_MAX: usize = (ABP_ROUNDS_MAX / FIRST_CHECKPOINT).ilog2() as usize + 1;

/// What a solution the decode of round `round` reaches is charged for,
/// as tries, `chased` where the chase runs among the rounds: the round's
/// share of the rounds' share of the propagation's chances, which the
/// doubting and the chase share equally with them ([`nth_of`]). The same
/// however many rounds follow it, so that the last of the rounds a block
/// is given is the dearest.
pub fn round_tries(round: usize, chased: bool) -> usize {
    as_tries(kinds_of_search(chased) as f64 * nth_of(round, ABP_ROUNDS_MAX))
}

/// The kinds of search a propagation makes, each given an equal share of
/// its chances: the rounds, the doubting among them, and the chase where
/// it is `chased`.
fn kinds_of_search(chased: bool) -> usize {
    2 + usize::from(chased)
}

/// What the `nth` of as many as `most` searches of a kind is found
/// among, as a share of the kind's chances: one in `nth` times the
/// harmonic number of `most`, which over all `most` comes to the whole.
/// A share that falls as the searches go on, and that is the same
/// however many of them follow, where an equal share of as many as are
/// made shrinks for every search made after it. Charged for the
/// searches up to its own, as the rounds once were, the `nth` was given
/// one in `nth`, and twenty rounds together the chances of three
/// decodes and more.
pub(super) fn nth_of(nth: usize, most: usize) -> f64 {
    nth as f64 * (1..=most).map(|k| 1.0 / k as f64).sum::<f64>()
}

/// A count of tries reckoned in shares, to the try above.
pub(super) fn as_tries(tries: f64) -> usize {
    tries.ceil() as usize
}

/// Whether the doubting and the chase run after `round`: after the
/// [`FIRST_CHECKPOINT`] and every doubling of it, the third, the sixth,
/// the twelfth and so on, which are the same however many rounds a
/// block is given. The beliefs move most in the first rounds, and the
/// doubting reaches the most from them. Run after the last round alone,
/// as they once were, the doubting and the chase reached a quarter
/// fewer blocks with three bits in a hundred read as a speck or a stain
/// reads after twenty rounds than after six, the beliefs having settled
/// on a wrong codeword in the meantime. Run after the first round and
/// every doubling of it, they reached a block in two hundred more, at
/// nearly three times the cost on a block of noise.
fn is_checkpoint(round: usize) -> bool {
    round.is_multiple_of(FIRST_CHECKPOINT) && (round / FIRST_CHECKPOINT).is_power_of_two()
}

/// Which of the rounds the doubting and the chase run after `round` is,
/// from one ([`is_checkpoint`]).
fn checkpoint(round: usize) -> usize {
    (round / FIRST_CHECKPOINT).trailing_zeros() as usize + 1
}

/// How much of what the checks say a bit takes on each round. Jiang and
/// Narayanan damp the propagation heavily, since the checks of a dense
/// matrix repeat each other: a bit told the same thing by many checks
/// that share their other bits would otherwise be carried off by it.
pub const ABP_DAMPING: f32 = 0.15;
/// The log of the odds a bit that is never moved stands at: pinned, or
/// painted by hand.
const LLR_FIXED: f32 = 64.0;
/// Most a check may say of a bit, in log-odds.
const LLR_CLAMP: f32 = 32.0;

/// The log of the odds that a module read at `reliability` was read
/// right: the scale of [`crate::imaging::sample::reliability_bits`]
/// read back, a bit of likelihood more for every eight. What belief
/// propagation starts each bit from, which the sampler holds short of
/// eight bits however far inside its colour the reading lies
/// ([`crate::imaging::sample::MISREAD_FLOOR`]), so that the checks can
/// move a bit read well and wrongly, as one under a logo or a stain is.
/// [`reliability_at`] turns it back.
pub(super) fn log_odds(reliability: u8) -> f32 {
    if reliability == CERTAIN {
        return LLR_FIXED;
    }
    let bits = f32::from(reliability) / 8.0 + 1.0;
    // ln((1 - p) / p) for p = 2^-bits.
    bits * LN_2 + (1.0 - 2f32.powf(-bits)).ln()
}

/// The reliability, on the sampler's scale, of a bit believed at `llr`
/// log-odds either way.
pub(super) fn reliability_at(llr: f32) -> u8 {
    let llr = llr.abs();
    let bits = if llr > 40.0 { llr / LN_2 } else { (1.0 + llr.exp()).log2() };
    ((bits - 1.0) * 8.0).round().clamp(0.0, f32::from(CERTAIN - 1)) as u8
}

/// The parity checks of a block of `n` codewords with `ec_len` of error
/// correction, over GF(2): row `8j + k` is bit `k` of syndrome `j`, and
/// column `8i + b` is bit `1 << b` of codeword `i`, as
/// [`super::rs::rs_syndromes`] has them.
fn checks(n: usize, ec_len: usize) -> Vec<Vec<u64>> {
    let cols = 8 * n;
    let words = cols / 64 + 1;
    let mut rows = vec![vec![0u64; words]; 8 * ec_len];
    for i in 0..n {
        let x = exp(n - 1 - i);
        let mut xj = 1u8;
        for j in 0..ec_len {
            for b in 0..8u8 {
                let contribution = mul(1 << b, xj);
                let col = 8 * i + b as usize;
                for k in 0..8 {
                    if contribution >> k & 1 != 0 {
                        rows[8 * j + k][col / 64] |= 1 << (col % 64);
                    }
                }
            }
            xj = mul(xj, x);
        }
    }
    rows
}

/// Reduce the checks so that the columns first in `order` are each set
/// in one row alone, as far as the rows are independent: the least
/// reliable bits are then each in a check of their own.
fn adapt(rows: &mut [Vec<u64>], order: &[usize]) {
    let m = rows.len();
    let mut rank = 0;
    for &col in order {
        if rank == m {
            break;
        }
        let (w, bit) = (col / 64, 1u64 << (col % 64));
        let Some(p) = (rank..m).find(|&r| rows[r][w] & bit != 0) else { continue };
        rows.swap(rank, p);
        let (before, from) = rows.split_at_mut(rank);
        let (pivot, after) = from.split_first_mut().expect("the pivot row");
        for row in before.iter_mut().chain(after.iter_mut()) {
            if row[w] & bit != 0 {
                for (a, b) in row.iter_mut().zip(pivot.iter()) {
                    *a ^= b;
                }
            }
        }
        rank += 1;
    }
}

/// One pass of the sum-product rule over the checks: what every check
/// says of each of its bits from the beliefs of the others, summed into
/// `ext`.
fn propagate(rows: &[Vec<u64>], llr: &[f32], ext: &mut [f32]) {
    // What a bit's belief says through a check is the same in every check
    // it is in: worked out once a round, where it was once worked out in
    // every check, half the bits of each.
    let halves: Vec<f32> = llr.iter().map(|&l| (l * 0.5).tanh()).collect();
    let mut members: Vec<usize> = Vec::new();
    let mut tanh: Vec<f32> = Vec::new();
    let mut prefix: Vec<f32> = Vec::new();
    for row in rows {
        members.clear();
        for (w, &word) in row.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                members.push(64 * w + bits.trailing_zeros() as usize);
                bits &= bits - 1;
            }
        }
        if members.len() < 2 {
            continue;
        }
        tanh.clear();
        tanh.extend(members.iter().map(|&c| halves[c]));
        prefix.clear();
        prefix.push(1.0);
        for &t in &tanh {
            let last = *prefix.last().expect("one");
            prefix.push(last * t);
        }
        let mut suffix = 1.0f32;
        for k in (0..members.len()).rev() {
            let others = prefix[k] * suffix;
            let said = if others.abs() >= 1.0 { LLR_CLAMP.copysign(others) } else { (2.0 * others.atanh()).clamp(-LLR_CLAMP, LLR_CLAMP) };
            ext[members[k]] += said;
            suffix *= tanh[k];
        }
    }
}

/// The block as the beliefs decide it: each bit at the colour its
/// log-odds favour, the colour read where they favour neither, and
/// unknown only where nothing was read and nothing has been said.
fn hard_decision(block: &[CwRead], llr: &[f32]) -> Vec<CwRead> {
    block
        .iter()
        .enumerate()
        .map(|(i, cw)| {
            let mut out = CwRead { value: 0, known: 0, pinned: cw.pinned };
            for b in 0..8 {
                let l = llr[8 * i + b];
                let known = cw.known >> b & 1 != 0;
                let value = if l < 0.0 {
                    true
                } else if l > 0.0 {
                    false
                } else {
                    cw.value >> b & 1 != 0
                };
                if known || l != 0.0 {
                    out.known |= 1 << b;
                    if value {
                        out.value |= 1 << b;
                    }
                }
            }
            out
        })
        .collect()
}

/// What one codeword found confirms, against the block as read
/// ([`judged`]): the hard count for the codewords it changes of those
/// read in full, with the known bits of the erased codewords as checks,
/// or the soft count for the bits it overturns weighed by their
/// reliability, among which those bits already stand, whichever is the
/// more, and never the two added. The propagation
/// reaches a block wrong in more codewords than the hard count can
/// confirm, each in a bit or two that read faintly; the soft count is
/// what confirms those. The bits the propagation overturned that the
/// solution keeps are its `flipped`. `None` for a solution refused, or
/// one it would not offer ([`super::rs::Offering`]). Either way the
/// codeword is noted in `seen`, unless it overturns a pin
/// ([`Sightings`]). What it spends of the reserve is counted against
/// `read` ([`counted_against`]).
#[allow(clippy::too_many_arguments)]
fn account(
    block: &[CwRead],
    read: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    corrected: Vec<u8>,
    reserved: usize,
    tries: usize,
    refined: &[CwRead],
    rounds: usize,
    seen: &mut Sightings,
) -> Option<(u32, Propagated)> {
    let n = block.len();
    seen.see_unless_pinned(block, &corrected);
    let solution = counted_against(read, ec_len, judged(block, bit_reliability, ec_len, reserved, corrected, tries_charge(tries))?);
    let flipped: Vec<u8> =
        (0..n).map(|i| (solution.corrected[i] ^ block[i].value) & (refined[i].value ^ block[i].value) & block[i].known).collect();
    Some((soft_distance(block, bit_reliability, &solution.corrected), (rounds, flipped, solution)))
}

/// Decode a block by adaptive belief propagation: `rounds` of adapting
/// the checks to the bits' reliability and passing beliefs along them,
/// the hard decision after each decoded as errors and erasures; and
/// after some ([`is_checkpoint`]), over the word and the reliabilities
/// the beliefs leave, the doubting of the faintest bits one by one
/// ([`solve_bit_erasures_doubting`], with `trials`), which is where the
/// propagation earns its keep, and the chase over the `chase_bits`
/// faintest, none when `chase_bits` is nought. Every codeword so found
/// is restated against the block as read and confirmed against it
/// ([`account`]), never by what it confirms of the word the beliefs
/// made, which the propagation carries toward some codeword whatever
/// the block holds; of those naming different codewords the nearest the
/// block as read by a clear margin is kept, or every one within the
/// margin, nearest first ([`nearest`]). Each comes with the rounds it
/// took and the bits of each codeword the propagation overturned.
/// Pinned bits and bits painted by hand are never moved; nothing for a
/// block with no other, or whose unread bits leave it too few equations
/// to spare. A block whose every EC codeword the standard reserves, as
/// M1's, was once not propagated either.
///
/// The rounds, the doubting and the chase are each a fresh chance for
/// a block of noise to pass, and each is given an equal share of the
/// propagation's, a solution of one charged for every try of its own
/// and for its round's share ([`nth_of`]): what a round, or a doubting
/// or chase after it, reaches is charged alike however many rounds
/// follow it. The rounds stop at the first that finds a solution, the
/// doubting and the chase after it among them. Given more rounds, a
/// block is so given back what fewer found, whenever they found
/// anything: never a solution more to stand against it, nor one charged
/// more. Every round's solution was once charged for all the rounds, so
/// that more of them confirmed it the less, and the doubting and the
/// chase ran after the last round alone, on beliefs that more rounds
/// had carried elsewhere: a block that six rounds solved was lost to
/// twenty.
///
/// Every codeword the rounds, the doubting and the chase reach that
/// overturns no pin is noted in `seen`, those returned among them
/// ([`Sightings`]).
#[allow(clippy::too_many_arguments)]
pub fn rs_decode_propagating(
    block: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    reserved: usize,
    rounds: usize,
    trials: Trials,
    chase_bits: usize,
    seen: &mut Sightings,
) -> Vec<Propagated> {
    rs_decode_propagating_against(block, block, bit_reliability, ec_len, reserved, rounds, trials, chase_bits, seen)
}

/// [`rs_decode_propagating`], what each solution spends of the reserve
/// counted against `read` ([`counted_against`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn rs_decode_propagating_against(
    block: &[CwRead],
    read: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    reserved: usize,
    rounds: usize,
    trials: Trials,
    chase_bits: usize,
    seen: &mut Sightings,
) -> Vec<Propagated> {
    propagating_damped(block, read, bit_reliability, ec_len, reserved, rounds, trials, chase_bits, ABP_DAMPING, seen)
}

/// [`rs_decode_propagating_against`] with the damping given, for finding
/// out what the damping should be.
#[allow(clippy::too_many_arguments)]
fn propagating_damped(
    block: &[CwRead],
    read: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    reserved: usize,
    rounds: usize,
    trials: Trials,
    chase_bits: usize,
    damping: f32,
    seen: &mut Sightings,
) -> Vec<Propagated> {
    assert_eq!(block.len(), bit_reliability.len());
    let n = block.len();
    if rounds == 0 || ec_len == 0 || ec_len >= n {
        return Vec::new();
    }
    // What the propagation has to work with is the equations its unread
    // bits leave: the beliefs, the doubting and the chase all solve for
    // bits, and a codeword a module short is one bit unread, not a
    // codeword erased. It once gave up wherever as many codewords as the
    // block has of error correction were each a module short, seven in
    // eight of their bits read and a hundred equations to spare.
    let unread: usize = block.iter().map(|c| c.known.count_zeros() as usize).sum();
    if unread + SPARE_EQUATIONS >= 8 * ec_len {
        return Vec::new();
    }
    let mut found: Vec<(u32, Propagated)> = Vec::new();
    let chased = chase_bits > 0;
    believe(block, bit_reliability, ec_len, rounds, damping, |beliefs| {
        let round = beliefs.rounds;
        let refined = beliefs.decided(block);
        let received: Vec<u8> = refined.iter().map(|c| c.value & c.known).collect();
        let syndromes = rs_syndromes(&received, ec_len);
        if let Ok(solution) = rs_decode_core(&refined, &[], ec_len, reserved, &syndromes, seen)
            && let Some(kept) = account(block, read, bit_reliability, ec_len, solution.corrected, reserved, round_tries(round, chased), &refined, round, seen)
        {
            found.push(kept);
        }
        if is_checkpoint(round) && !work::cancelled() {
            let share = kinds_of_search(chased) as f64 * nth_of(checkpoint(round), CHECKPOINTS_MAX);
            let refined_bits = beliefs.reliabilities();
            for doubt in solve_bit_erasures_doubting(&refined, &refined_bits, ec_len, trials, seen) {
                let tries = as_tries(share * doubt.tries as f64);
                if let Some(kept) = account(block, read, bit_reliability, ec_len, doubt.solution.corrected, reserved, tries, &refined, round, seen) {
                    found.push(kept);
                }
            }
            if chased && !work::cancelled() {
                let bits = chase_bits.clamp(1, CHASE_MAX_BITS);
                let tries = as_tries(share * ((1usize << bits) - 1) as f64);
                for (_, solution) in rs_decode_chasing_bits_against(&refined, read, &refined_bits, ec_len, reserved, bits, seen) {
                    if let Some(kept) = account(block, read, bit_reliability, ec_len, solution.corrected, reserved, tries, &refined, round, seen) {
                        found.push(kept);
                    }
                }
            }
        }
        // A round with a solution is the last, so that more rounds
        // never add to what it found.
        !found.is_empty() || syndromes.iter().all(|&s| s == 0)
    });
    nearest(found, |a, b| a.2.corrected == b.2.corrected, |a, b| a.2.confidence_bits() > b.2.confidence_bits())
}

/// The beliefs about a block's bits after the rounds made, as log-odds.
struct Beliefs {
    n: usize,
    llr: Vec<f32>,
    /// Bits never moved: pinned, or painted by hand.
    fixed: Vec<bool>,
    /// Rounds made.
    rounds: usize,
}

impl Beliefs {
    /// The block as the beliefs decide it ([`hard_decision`]).
    fn decided(&self, block: &[CwRead]) -> Vec<CwRead> {
        hard_decision(block, &self.llr)
    }

    /// The reliability of each bit of each codeword under the beliefs,
    /// on the sampler's scale, certain where it was never moved.
    fn reliabilities(&self) -> Vec<[u8; 8]> {
        (0..self.n)
            .map(|i| std::array::from_fn(|b| if self.fixed[8 * i + b] { CERTAIN } else { reliability_at(self.llr[8 * i + b]) }))
            .collect()
    }
}

/// The rounds of adapting and propagating, `each_round` given the
/// beliefs after each, and stopping early when it says to or when the
/// work is called off. `None` for a block with no bit to move.
fn believe(
    block: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    rounds: usize,
    damping: f32,
    mut each_round: impl FnMut(&Beliefs) -> bool,
) -> Option<Beliefs> {
    let n = block.len();
    let cols = 8 * n;
    let mut llr = vec![0f32; cols];
    let mut fixed = vec![false; cols];
    let mut movable = 0;
    for (i, cw) in block.iter().enumerate() {
        for (b, &reliability) in bit_reliability[i].iter().enumerate() {
            if cw.known >> b & 1 == 0 {
                continue;
            }
            let col = 8 * i + b;
            let set = cw.value >> b & 1 != 0;
            let fix = cw.pinned >> b & 1 != 0 || reliability == CERTAIN;
            let odds = if fix { LLR_FIXED } else { log_odds(reliability) };
            llr[col] = if set { -odds } else { odds };
            fixed[col] = fix;
            movable += usize::from(!fix);
        }
    }
    if movable == 0 {
        return None;
    }
    let mut rows = checks(n, ec_len);
    let mut ext = vec![0f32; cols];
    let mut order: Vec<usize> = (0..cols).collect();
    let mut beliefs = Beliefs { n, llr, fixed, rounds: 0 };
    for round in 1..=rounds {
        if work::cancelled() {
            break;
        }
        let llr = &mut beliefs.llr;
        order.sort_by(|&a, &b| llr[a].abs().total_cmp(&llr[b].abs()));
        adapt(&mut rows, &order);
        ext.fill(0.0);
        propagate(&rows, llr, &mut ext);
        for (col, said) in ext.iter().enumerate() {
            if !beliefs.fixed[col] {
                llr[col] += damping * said;
            }
        }
        beliefs.rounds = round;
        if each_round(&beliefs) {
            break;
        }
    }
    Some(beliefs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::rs::{Trials, VERIFIED_BITS, rs_decode, rs_decode_doubting, rs_encode, solve_bit_erasures_doubting};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn byte(&mut self) -> u8 {
            (self.next() >> 24) as u8
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        fn unit(&mut self) -> f32 {
            ((self.next() >> 11) as f64 / (1u64 << 53) as f64) as f32
        }
        /// A normal deviate, by Box and Muller.
        fn normal(&mut self) -> f32 {
            let (u, v) = (self.unit().max(1e-9), self.unit());
            (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
        }
    }

    fn codeword(rng: &mut Rng, data_len: usize, ec_len: usize) -> Vec<u8> {
        let mut block: Vec<u8> = (0..data_len).map(|_| rng.byte()).collect();
        let ec = rs_encode(&block, ec_len);
        block.extend(ec);
        block
    }

    /// The block sent through noise: each bit read as a level one off
    /// its colour with normal noise of `sigma` on it, taken as the colour
    /// it fell nearer, and a share `gross` of the bits read as either
    /// colour whatever was sent, at a level that colour's noise gives,
    /// as a speck or a stain reads; each at the reliability the
    /// sampler's calibration gives such a reading, which the gross share
    /// caps ([`crate::imaging::sample::Calibration::bits`]).
    fn through_noise(rng: &mut Rng, block: &[u8], sigma: f32, gross: f32) -> (Vec<CwRead>, Vec<[u8; 8]>) {
        let mut read = Vec::with_capacity(block.len());
        let mut reliability = Vec::with_capacity(block.len());
        for &value in block {
            let mut cw = CwRead::known(0);
            let mut bits = [0u8; 8];
            for (b, bit) in bits.iter_mut().enumerate() {
                let mut sent = if value >> b & 1 != 0 { -1.0 } else { 1.0 };
                if rng.unit() < gross && rng.unit() < 0.5 {
                    sent = -sent;
                }
                let level = sent + sigma * rng.normal();
                if level < 0.0 {
                    cw.value |= 1 << b;
                }
                let odds = (2.0 * level.abs() / (sigma * sigma)).min(87.0);
                let wrong = gross + (1.0 - gross) / (1.0 + odds.exp());
                let likelihood = -wrong.max(f32::MIN_POSITIVE).log2();
                *bit = ((likelihood - 1.0) * 8.0).round().clamp(0.0, f32::from(CERTAIN - 1)) as u8;
            }
            read.push(cw);
            reliability.push(bits);
        }
        (read, reliability)
    }

    /// How the solvers fare on a 5-Q block through noise of every kind,
    /// for choosing the damping and the rounds: run by hand, with
    /// `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn survey_of_the_channel() {
        use crate::decode::list::{LIST_COST, rs_decode_listing};
        use crate::decode::rs::{CHASE_BITS, rs_decode_chasing};
        let blocks: usize = std::env::var("ABP_BLOCKS").ok().and_then(|v| v.parse().ok()).unwrap_or(60);
        let damping: f32 = std::env::var("ABP_DAMPING").ok().and_then(|v| v.parse().ok()).unwrap_or(ABP_DAMPING);
        let rounds: usize = std::env::var("ABP_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(ABP_ROUNDS);
        println!("sigma gross | plain cw bits chase list all | abp whole abp>bits abp>list | adds: abp abp>bits abp>list any");
        for &sigma in &[0.5f32, 0.6, 0.7] {
            for &gross in &[0.0f32, 0.01, 0.03] {
                let mut rng = Rng(0xAB9 ^ (sigma * 1000.0) as u64 ^ ((gross * 1000.0) as u64) << 20);
                let mut counts = [0usize; 14];
                for _ in 0..blocks {
                    let block = codeword(&mut rng, 15, 18);
                    let (read, reliability) = through_noise(&mut rng, &block, sigma, gross);
                    let by_codeword: Vec<u8> = reliability.iter().map(|r| *r.iter().min().unwrap()).collect();
                    let plain = rs_decode(&read, 18).is_ok_and(|s| s.corrected == block);
                    let cw = rs_decode_doubting(&read, &by_codeword, 18, 0, &mut Sightings::default()).is_some_and(|(_, s)| s.corrected == block);
                    let bits = solve_bit_erasures_doubting(&read, &reliability, 18, Trials::ALL, &mut Sightings::default()).first().is_some_and(|d| d.solution.corrected == block);
                    let chase = rs_decode_chasing(&read, &reliability, 18, 0, CHASE_BITS).first().is_some_and(|(_, s)| s.corrected == block);
                    let list = rs_decode_listing(&read, &reliability, 18, 0, LIST_COST, &mut Sightings::default()).first().is_some_and(|(_, s)| s.corrected == block);
                    let others = plain || cw || bits || chase || list;
                    let found = propagating_damped(&read, &read, &reliability, 18, 0, rounds, Trials::NONE, 0, damping, &mut Sightings::default());
                    assert!(found.iter().all(|(_, _, s)| s.corrected == block), "a wrong codeword offered");
                    let abp = !found.is_empty();
                    let found = propagating_damped(&read, &read, &reliability, 18, 0, rounds, Trials::ALL, CHASE_BITS, damping, &mut Sightings::default());
                    assert!(found.iter().all(|(_, _, s)| s.corrected == block), "a wrong codeword offered: {found:?}");
                    let abp_chase = !found.is_empty();
                    // The beliefs as a refiner: the bits and the list
                    // decoder run on the word and the reliabilities the
                    // beliefs leave, and judged by whether they reach
                    // the block.
                    let (mut abp_bits, mut abp_list) = (false, false);
                    if let Some(beliefs) = believe(&read, &reliability, 18, rounds, damping, |_| false) {
                        let refined = beliefs.decided(&read);
                        let refined_bits = beliefs.reliabilities();
                        abp_bits = solve_bit_erasures_doubting(&refined, &refined_bits, 18, Trials::ALL, &mut Sightings::default()).first().is_some_and(|d| d.solution.corrected == block);
                        abp_list = rs_decode_listing(&refined, &refined_bits, 18, 0, LIST_COST, &mut Sightings::default()).first().is_some_and(|(_, s)| s.corrected == block);
                    }
                    for (k, hit) in [plain, cw, bits, chase, list, others, abp, abp_chase, abp_bits, abp_list].into_iter().enumerate() {
                        counts[k] += usize::from(hit);
                    }
                    counts[10] += usize::from(abp && !others);
                    counts[11] += usize::from(abp_bits && !others);
                    counts[12] += usize::from(abp_list && !others);
                    counts[13] += usize::from(abp_chase && !others);
                }
                println!(
                    "{sigma:.2} {gross:.2} | {:>3} {:>3} {:>3} {:>3} {:>3} {:>3} | {:>3} {:>3} {:>3} {:>3} | {:>3} {:>3} {:>3} {:>3}",
                    counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6], counts[7], counts[8], counts[9], counts[10], counts[11], counts[12], counts[13]
                );
            }
        }
    }

    #[test]
    fn the_checks_are_the_syndromes_bit_by_bit() {
        let mut rng = Rng(0xABF);
        for &(n, ec_len) in &[(26usize, 17usize), (44, 10), (153, 30), (5, 2)] {
            let rows = checks(n, ec_len);
            for _ in 0..5 {
                let word: Vec<u8> = (0..n).map(|_| rng.byte()).collect();
                let syndromes = rs_syndromes(&word, ec_len);
                for (r, row) in rows.iter().enumerate() {
                    let mut parity = false;
                    for (w, &bits) in row.iter().enumerate() {
                        let mut bits = bits;
                        while bits != 0 {
                            let col = 64 * w + bits.trailing_zeros() as usize;
                            parity ^= word[col / 8] >> (col % 8) & 1 != 0;
                            bits &= bits - 1;
                        }
                    }
                    assert_eq!(parity, syndromes[r / 8] >> (r % 8) & 1 != 0, "({n}, {ec_len}) row {r}");
                }
            }
        }
        // The odds and the reliability are each other's inverse.
        for r in [0u8, 8, 40, 120, 200, 254] {
            assert!((i32::from(reliability_at(log_odds(r))) - i32::from(r)).abs() <= 1, "{r}: {}", reliability_at(log_odds(r)));
        }
        assert_eq!(log_odds(0), 0.0);
        assert_eq!(reliability_at(LLR_FIXED), CERTAIN - 1);
    }

    #[test]
    fn the_shares_come_to_no_more_than_the_whole() {
        // Every round up to the most a block may be given, and every
        // round the doubting and the chase run after, each at its share:
        // together no more than the chances of each kind of search.
        let rounds: f64 = (1..=ABP_ROUNDS_MAX).map(|r| 1.0 / nth_of(r, ABP_ROUNDS_MAX)).sum();
        assert!(rounds <= 1.0 + 1e-9, "{rounds}");
        let checkpoints: Vec<usize> = (1..=ABP_ROUNDS_MAX).filter(|&r| is_checkpoint(r)).collect();
        assert_eq!(checkpoints, [3, 6, 12, 24, 48, 96]);
        // The fewest rounds a block may be given reach the first of them.
        assert_eq!(checkpoints[0], ABP_ROUNDS_LEAST);
        assert_eq!(checkpoints.iter().map(|&r| checkpoint(r)).collect::<Vec<_>>(), (1..=CHECKPOINTS_MAX).collect::<Vec<_>>());
        let doubtings: f64 = (1..=CHECKPOINTS_MAX).map(|k| 1.0 / nth_of(k, CHECKPOINTS_MAX)).sum();
        assert!(doubtings <= 1.0 + 1e-9, "{doubtings}");
        // The first round is charged four bits, and the twentieth nine,
        // however many follow; every one of twenty was once charged six.
        assert_eq!(tries_charge(round_tries(1, true)), 4);
        assert_eq!(tries_charge(round_tries(ABP_ROUNDS, true)), 9);
        assert_eq!(tries_charge(round_tries(ABP_ROUNDS_MAX, true)), 11);
    }

    #[test]
    fn propagation_reaches_misreads_the_other_solvers_do_not() {
        // A 5-Q block through noise that misreads one bit in twenty or
        // so: a dozen codewords wrong, past the nine the code corrects,
        // past the ten faintest bits the chase flips, and, the faint
        // right bits outnumbering the wrong ones several to one, past
        // what doubting the faintest can take as unknown. Belief
        // propagation reads a good share of them, every one right.
        let mut rng = Rng(0xAB9);
        let (blocks, sigma) = (60, 0.62f32);
        let (mut plain, mut doubted, mut doubted_bits, mut propagated) = (0, 0, 0, 0);
        for round in 0..blocks {
            let block = codeword(&mut rng, 15, 18);
            let (read, reliability) = through_noise(&mut rng, &block, sigma, 0.0);
            plain += usize::from(rs_decode(&read, 18).is_ok_and(|s| s.corrected == block));
            let by_codeword: Vec<u8> = reliability.iter().map(|r| *r.iter().min().unwrap()).collect();
            doubted += usize::from(rs_decode_doubting(&read, &by_codeword, 18, 0, &mut Sightings::default()).is_some_and(|(_, s)| s.corrected == block));
            doubted_bits += usize::from(solve_bit_erasures_doubting(&read, &reliability, 18, Trials::NONE, &mut Sightings::default()).first().is_some_and(|d| d.solution.corrected == block));
            let found = rs_decode_propagating(&read, &reliability, 18, 0, ABP_ROUNDS, Trials::ALL, 0, &mut Sightings::default());
            for (rounds, flipped, solution) in &found {
                assert_eq!(solution.corrected, block, "round {round}: a wrong codeword offered");
                assert!(*rounds >= 1 && *rounds <= ABP_ROUNDS);
                assert!(flipped.iter().zip(&read).zip(&block).all(|((&f, r), &b)| f & (r.value ^ b) == f), "round {round}: flipped bits the block does not have");
            }
            propagated += usize::from(found.first().is_some_and(|(_, _, s)| s.corrected == block));
        }
        println!("plain {plain}, doubted codewords {doubted}, doubted bits {doubted_bits}, propagated {propagated} of {blocks}");
        assert!(propagated > doubted && propagated > doubted_bits && propagated > plain, "plain {plain}, doubted {doubted}, doubted bits {doubted_bits}, propagated {propagated}");
        assert!(propagated * 2 >= blocks, "propagated {propagated} of {blocks}");
    }

    #[test]
    fn more_rounds_give_back_what_fewer_found() {
        // Blocks through faint noise, some stained besides, given ever
        // more rounds: whatever a count of rounds finds, every count past
        // it finds the same, charged the same. With every round's
        // solution charged for all the rounds, and the doubting and the
        // chase after the last alone, twenty rounds once solved a quarter
        // fewer of the stained blocks than six.
        let mut rng = Rng(0xAB7);
        let counts = [1usize, 2, 3, 4, 6, 9, 12, 20, 40];
        let mut solved = 0;
        for (sigma, gross) in [(0.62f32, 0.0f32), (0.62, 0.03), (0.7, 0.03)] {
            for round in 0..12 {
                let block = codeword(&mut rng, 15, 18);
                let (read, reliability) = through_noise(&mut rng, &block, sigma, gross);
                let mut first: Option<(usize, Vec<Propagated>)> = None;
                for &rounds in &counts {
                    let found = rs_decode_propagating(&read, &reliability, 18, 0, rounds, Trials::ALL, 10, &mut Sightings::default());
                    match &first {
                        Some((fewer, before)) => assert_eq!(&found, before, "{sigma} {gross}, block {round}: {rounds} rounds against {fewer}"),
                        None if !found.is_empty() => first = Some((rounds, found)),
                        None => {}
                    }
                }
                solved += usize::from(first.is_some());
            }
        }
        assert!(solved >= 24, "solved {solved} of 36");
    }

    /// What noise comes to through the beliefs and then the bit-level
    /// doubting: run by hand, with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn survey_of_noise_through_the_refiner() {
        let mut rng = Rng(0x0153);
        let blocks = 400;
        for (n, ec_len) in [(26usize, 17usize), (33, 18), (44, 10), (26, 7)] {
            let (mut offered, mut verified, mut most) = (0, 0, 0);
            let mut doubted_at = Vec::new();
            for _ in 0..blocks {
                let read: Vec<CwRead> = (0..n).map(|_| CwRead::known(rng.byte())).collect();
                let reliability: Vec<[u8; 8]> = (0..n).map(|_| std::array::from_fn(|_| rng.byte().min(CERTAIN - 1))).collect();
                for (_, _, solution) in rs_decode_propagating(&read, &reliability, ec_len, 0, ABP_ROUNDS, Trials::ALL, 10, &mut Sightings::default()) {
                    offered += 1;
                    let bits = solution.confidence_bits();
                    most = most.max(bits);
                    verified += usize::from(bits >= 3 * VERIFIED_BITS);
                    doubted_at.push(solution.errors);
                }
            }
            println!("({n}, {ec_len}): {offered} offered of {blocks}, {verified} at {} bits or more, the most {most}; errors {doubted_at:?}", 3 * VERIFIED_BITS);
        }
    }

    #[test]
    fn noise_is_never_read_as_verified() {
        let mut rng = Rng(0x0153);
        for (n, ec_len) in [(26usize, 17usize), (33, 18), (44, 10), (26, 7)] {
            for round in 0..100 {
                let read: Vec<CwRead> = (0..n).map(|_| CwRead::known(rng.byte())).collect();
                let reliability: Vec<[u8; 8]> = (0..n).map(|_| std::array::from_fn(|_| rng.byte().min(CERTAIN - 1))).collect();
                for (_, _, solution) in rs_decode_propagating(&read, &reliability, ec_len, 0, ABP_ROUNDS, Trials::ALL, 10, &mut Sightings::default()) {
                    assert!(solution.confidence_bits() < 3 * VERIFIED_BITS, "({n}, {ec_len}) round {round}: {solution:?}");
                }
            }
        }
        // Noise with five of a 1-L block's codewords read but for a bit or
        // three: the propagation carries it to codewords a dozen and more
        // from the block as read, where noise has some two to the hundred
        // as near, and the thirty known bits of the five agree with one
        // of those as often as chance has them. Counted from nought, the
        // codewords' count once let them confirm such a codeword by thirty
        // bits, one block of noise in twenty.
        for round in 0..200 {
            let mut read: Vec<CwRead> = (0..26).map(|_| CwRead::known(rng.byte())).collect();
            let mut unread = 0;
            while unread < 5 {
                let i = rng.below(26);
                if read[i].is_erasure() {
                    continue;
                }
                unread += 1;
                for _ in 0..1 + rng.below(3) {
                    read[i].known &= !(1 << rng.below(8));
                }
                read[i].value &= read[i].known;
            }
            let reliability: Vec<[u8; 8]> = (0..26).map(|_| std::array::from_fn(|_| rng.byte().min(CERTAIN - 1))).collect();
            for (_, _, solution) in rs_decode_propagating(&read, &reliability, 7, 0, ABP_ROUNDS, Trials::ALL, 10, &mut Sightings::default()) {
                assert!(solution.confidence_bits() < 3 * VERIFIED_BITS, "partly unread, round {round}: {solution:?}");
            }
        }
        // A block painted by hand has nothing to move.
        let block = codeword(&mut rng, 15, 18);
        let read: Vec<CwRead> = block.iter().map(|&v| CwRead::known(v)).collect();
        assert!(rs_decode_propagating(&read, &vec![[CERTAIN; 8]; 33], 18, 0, ABP_ROUNDS, Trials::ALL, 10, &mut Sightings::default()).is_empty());
    }
}
