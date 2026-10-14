//! Grid to text: read codewords, solve each RS block as far as possible,
//! assemble the data stream with per-bit certainty, and parse it.

use std::sync::Arc;

use super::bch::{self, FormatAnalysis, VersionAnalysis};
use super::bitstream::{self, Certainty, EndOfData, ParseReport, StreamBits};
use super::codewords::{Pin, Pins, apply_pins, codeword_bit_mask, read_codewords};
use super::list::{LIST_COST, LIST_COST_MIN, rs_decode_listing};
use super::padding::{self, TailCheck};
use super::rs::{
    AIDED_VERIFIED_BITS, BitDoubt, BitSolveFailure, CHASE_BITS, CwRead, RsFailure, RsSolution, TRIAL_SETS, TRIAL_SETS_MIN, Trials,
    VERIFIED_BITS, chase_bits_among, rs_decode_chasing_bits, rs_decode_doubting, rs_decode_reserving, rs_decode_trying,
    solve_bit_erasures, solve_bit_erasures_doubting, solve_bit_erasures_trying,
};
use super::text::{self, Charset, TextReport};
use crate::symbol::EcLevel;
use crate::symbol::function_map::{FunctionMap, POLARITY_MIN_KNOWN, looks_inverted};
use crate::symbol::grid::{CERTAIN, Mirror, ModuleGrid, Symmetry};
use crate::symbol::layout::{BlockSpec, SymbolLayout};
use crate::symbol::tables;
use crate::work;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeOptions {
    pub charset: Charset,
    /// Fall back to solving unknown bits over GF(2) when a block has more
    /// erased codewords than Reed-Solomon can take.
    pub bit_level: bool,
    /// When a block fails, doubt its least reliable codewords, and then
    /// its least reliable bits, a few at a time, and retry; then flip its
    /// least reliable bits every way. Only modules sampled from a photo
    /// are ever doubted.
    pub reliability: bool,
    /// While `reliability` holds, doubt the least reliable codewords
    /// ([`SolveMethod::Reliability`]): the first of the three ways it
    /// names, each of which may be left out by itself.
    pub doubt_codewords: bool,
    /// While `reliability` and `bit_level` hold, doubt the least reliable
    /// bits ([`SolveMethod::BitReliability`]).
    pub doubt_bits: bool,
    /// While `reliability` holds, flip the least reliable bits every way
    /// ([`SolveMethod::Chase`]), and, while `bit_level` holds too, flip
    /// a bit or two or three read solidly wrong where the equations say
    /// which ([`SolveMethod::BitFlip`]).
    pub flip_bits: bool,
    /// When a block fails, erase its codewords by trial, every one or
    /// every pair, whatever their reliability, and keep a solution the
    /// rest of the block confirms ([`SolveMethod::Trial`],
    /// [`SolveMethod::BitTrial`]): reaches a block wrong in one codeword
    /// more than the code corrects, and one whose unknown bits solve bit
    /// by bit but for a wrong codeword or two, on a grid painted by hand
    /// as on one sampled.
    pub erase_by_trial: bool,
    /// Decodes of the grid the caller makes, this one among them, under
    /// other masks and levels, samplings and placements, keeping the
    /// best: each is a fresh chance for a block of noise to show errors
    /// located, so what a solution's located errors confirm is charged
    /// a bit for each doubling of them ([`super::rs::RsSolution::among`]).
    /// One where the decode stands alone.
    pub tries: usize,
    /// Sets of codewords a block's trials of erasures may try, over
    /// every size of set, at the most ([`super::rs::TRIAL_SETS`]): what
    /// a block nothing else solves costs, since every set is a decode. A
    /// decode that is one of many ([`Self::among`]) has less to spend on
    /// each, and never less than the pairs of the largest block
    /// ([`super::rs::TRIAL_SETS_MIN`]), so that one codeword past the
    /// code is reached in any block. Set by hand up to
    /// [`super::rs::TRIAL_SETS_MAX`].
    pub trial_sets: usize,
    /// Least reliable bits the chase flips every way
    /// ([`super::rs::CHASE_BITS`]), from one to
    /// [`super::rs::CHASE_MAX_BITS`]: every pattern is a decode, and a
    /// chance of a misdecode, so there are two to the power of this many
    /// of each. Less by a bit for each doubling of the tries a decode is
    /// one of.
    pub chase_bits: usize,
    /// When a block fails, decode it by algebraic soft-decision list
    /// decoding ([`super::list`], [`SolveMethod::List`]): a curve
    /// interpolated through the block's codewords weighed by their
    /// reliability, and every codeword of the code read off it, which
    /// reaches past half the code's distance, further on a block of low
    /// rate, and bends toward the other colour of every faint module at
    /// once.
    pub list_decode: bool,
    /// Constraints the list decoder's curve is given
    /// ([`super::list::LIST_COST`]): how far it reaches, and what it
    /// costs. A decode that is one of many has less, and never less than
    /// [`super::list::LIST_COST_MIN`].
    pub list_cost: usize,
    /// When the decode is no read, find the padding, terminator and
    /// headers the stream shows, pin them, and solve again the blocks
    /// the pins touch, to a fixed point, as padding recovery does by
    /// hand: a pinned codeword costs no error correction, and the pins
    /// are kept when they solve or verify a block more, or the segment
    /// headers confirm them ([`DecodeReport::derived_pins`]). The
    /// decodes padding recovery itself makes never derive, nor does the
    /// first pass of a brute-force search.
    pub derive_pins: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            charset: Charset::Auto,
            bit_level: true,
            reliability: true,
            doubt_codewords: true,
            doubt_bits: true,
            flip_bits: true,
            erase_by_trial: true,
            tries: 1,
            trial_sets: TRIAL_SETS,
            chase_bits: CHASE_BITS,
            list_decode: true,
            list_cost: LIST_COST,
            derive_pins: false,
        }
    }
}

impl DecodeOptions {
    /// These options for a decode that is one of `tries` the caller
    /// makes of the grid, keeping the best, besides those already
    /// counted: the budgets of the dear solvers are shared out over the
    /// tries, down to their floors.
    pub fn among(self, tries: usize) -> Self {
        let tries = self.tries.max(1) * tries.max(1);
        Self {
            tries,
            trial_sets: (self.trial_sets / tries).clamp(TRIAL_SETS_MIN.min(self.trial_sets), self.trial_sets),
            list_cost: (self.list_cost / tries).clamp(LIST_COST_MIN.min(self.list_cost), self.list_cost),
            ..self
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveMethod {
    /// Errors-and-erasures Reed-Solomon.
    ReedSolomon,
    /// Unknown bits solved individually, trusting every known bit.
    BitLevel,
    /// The least reliable codewords erased until the block solved.
    Reliability { doubted: usize },
    /// The least reliable bits, read as known, treated as unknown until
    /// the block solved bit by bit.
    BitReliability { doubted_bits: usize },
    /// The least reliable bits flipped outright, every pattern of them
    /// tried, and of the solutions the patterns reached the one nearest
    /// the block as read kept, with this many bits flipped.
    Chase { flipped_bits: usize },
    /// Solved bit by bit with `flipped_bits` bits read solidly flipped,
    /// found by what the equations were left wanting, and the
    /// `doubted_bits` least reliable bits besides, when any were.
    BitFlip { doubted_bits: usize, flipped_bits: usize },
    /// This many codewords erased by trial, every set of that size of
    /// the block's tried whatever its reliability, and the block decoded
    /// as errors and erasures with them so.
    Trial { erased: usize },
    /// Solved bit by bit with `erased` codewords taken as unknown by
    /// trial, and the `doubted_bits` least reliable bits besides, when
    /// any were.
    BitTrial { doubted_bits: usize, erased: usize },
    /// Read off a curve interpolated through the block's codewords
    /// weighed by their reliability, with a point of this multiplicity
    /// at the most ([`super::list`]).
    List { multiplicity: usize },
    /// Everything except pinned codewords treated as erased.
    PinnedOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockOutcome {
    Solved {
        method: SolveMethod,
        /// Fully read codewords that turned out wrong.
        errors: usize,
        /// Codewords with unknown bits that were filled in.
        erasures: usize,
        /// Redundancy left over, in codewords.
        slack: usize,
        /// Independent bits confirming the solution: the redundancy left
        /// over, the known bits of erased codewords it reproduced, and
        /// what the errors it located confirm
        /// ([`super::rs::located_error_bits`]). Zero means any input would
        /// have produced a "solution", as one does that only fills
        /// unknown codewords and has nothing left over.
        confidence_bits: u32,
        /// The solution spent error correction codewords the standard
        /// reserves for detection in the smallest symbols (ISO/IEC 18004
        /// Table 9), so it is one the standard says not to trust.
        beyond_standard: bool,
    },
    Failed(RsFailure),
}

impl BlockOutcome {
    /// Confirming bits, zero for a failure.
    pub fn confidence_bits(&self) -> u32 {
        match *self {
            BlockOutcome::Solved { confidence_bits, .. } => confidence_bits,
            BlockOutcome::Failed(_) => 0,
        }
    }

    /// Confirming bits the solution must show to count as verified: a
    /// byte's worth for the plain decode of a block read whole, which is
    /// the decode every reader makes and never shows so little within
    /// the standard's limit, and three codewords' worth
    /// ([`AIDED_VERIFIED_BITS`]) for one that filled in codewords that
    /// could not be read or was reached by any of the fallbacks, the
    /// trials of erasures among them. `None` for a failure.
    pub fn bits_to_verify(&self) -> Option<u32> {
        match *self {
            BlockOutcome::Solved { method: SolveMethod::ReedSolomon, erasures: 0, .. } => Some(VERIFIED_BITS),
            BlockOutcome::Solved { .. } => Some(AIDED_VERIFIED_BITS),
            BlockOutcome::Failed(_) => None,
        }
    }

    /// Solved with the confirmation its kind of solution needs
    /// ([`Self::bits_to_verify`]), within the standard's limit.
    pub fn is_verified(&self) -> bool {
        match *self {
            BlockOutcome::Solved { confidence_bits, beyond_standard: false, .. } => {
                self.bits_to_verify().is_some_and(|needed| confidence_bits >= needed)
            }
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockReport {
    pub data_len: usize,
    pub ec_len: usize,
    /// EC codewords reserved against misdecodes.
    pub reserved: usize,
    pub outcome: BlockOutcome,
    /// Codewords with at least one unknown bit, after pins.
    pub erased_codewords: usize,
    pub unknown_bits: usize,
    /// Unknown bits settled by partial bit-level solving although the block
    /// did not solve. They rely on every known bit being right.
    pub inferred_bits: usize,
    /// The block solved only with the missing nibble of its final four-bit
    /// codeword, in an M1 or M3 symbol, taken as ones rather than the
    /// standard's zeros: the symbol's encoder computed its error
    /// correction that way.
    pub half_padded_with_ones: bool,
    /// The solution was one of this many a solver that tries many ways
    /// found and could not tell apart by the block as read, and the
    /// message told it from the rest: under it alone the stream parsed
    /// best ([`arbitrate`]). Nought where the solution stood alone.
    pub rivals: usize,
    /// Modules of the block whose reliability was lowered because an
    /// error was located beside them in a block that had verified, when
    /// the solution was reached so ([`NEAR_ERROR_RELIABILITY`]). Nought
    /// where it was not.
    pub near_errors: usize,
}

impl BlockReport {
    pub fn is_solved(&self) -> bool {
        matches!(self.outcome, BlockOutcome::Solved { .. })
    }

    /// Solved with the confirmation its solution needs, within the
    /// standard's limit ([`BlockOutcome::is_verified`]).
    pub fn is_verified(&self) -> bool {
        self.outcome.is_verified()
    }

    /// Solved only by spending reserved codewords.
    pub fn is_beyond_standard(&self) -> bool {
        matches!(self.outcome, BlockOutcome::Solved { beyond_standard: true, .. })
    }

    /// How many more codewords must become fully known before plain erasure
    /// decoding has a chance.
    pub fn codewords_short(&self) -> usize {
        self.erased_codewords.saturating_sub(self.ec_len)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CwStatus {
    /// Read in full and confirmed.
    Clean,
    /// Read in full, changed by error correction.
    Corrected,
    /// Had unknown bits, filled in by error correction.
    Filled,
    /// Read (or pinned) in full, but its block did not solve.
    Unverified,
    /// Had unknown bits, some of them inferred by partial solving, but its
    /// block did not solve.
    Inferred,
    /// Has unknown bits and its block did not solve.
    Unresolved,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodewordReport {
    /// Value as the modules show it, unmasked; unknown bits are zero.
    pub read: u8,
    /// Which bits of `read` came from known modules.
    pub known: u8,
    pub pin: Option<Pin>,
    /// Value after pins, with unknown bits zero.
    pub pinned_value: u8,
    /// Known bits after pins.
    pub pinned_known: u8,
    /// Bits settled by partial bit-level solving when the block did not
    /// solve, and which bits they are.
    pub inferred_value: u8,
    pub inferred_known: u8,
    /// Value after error correction, when the block solved.
    pub final_value: Option<u8>,
    /// Bits read as known that the solution of the codeword's block was
    /// reached by doubting: all eight of a codeword erased as one of the
    /// block's least reliable or by trial, the single bits treated as
    /// unknown where the block solved bit by bit, and the bits flipped
    /// where it solved by flipping. None under any other solution, or
    /// under none.
    pub doubted: u8,
    pub status: CwStatus,
}

impl CodewordReport {
    /// Value and known bits after pins and inference, before error
    /// correction.
    pub fn partial(&self) -> (u8, u8) {
        (self.pinned_value | self.inferred_value, self.pinned_known | self.inferred_known)
    }

    /// Best available value and the bits of it that are known.
    pub fn best(&self) -> (u8, u8) {
        match self.final_value {
            Some(value) => (value, 0xFF),
            None => self.partial(),
        }
    }
}

/// Ranking of a decode attempt; larger is better. Field order is the
/// comparison order.
/// How attempts at a grid, and reads of a photo, rank against each other:
/// field by field, the first that differs deciding. A read that is
/// trustworthy, with every block confirmed and a stream that parses,
/// outranks all else. Below that the stream comes before the blocks: a
/// placement whose stream parses, through bits no block confirmed, is
/// nearer the symbol than one whose every block "verified" while its
/// stream reads as noise. A wrong placement manages the latter now and
/// then, where a block spends most of its redundancy on what it erased
/// and is confirmed by the little left, which noise passes once in so
/// many blocks however much of it is asked
/// ([`super::rs::AIDED_VERIFIED_BITS`]); BoofCV takes a symbol as read
/// only when its error correction succeeds and its message parses, and
/// the order here does the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AttemptScore {
    /// Segments parse to a clean end with no impossible values; a stream
    /// with no segment at all counts only when padding follows its end.
    pub stream_valid: bool,
    /// Every block solved with confirmation, and the stream not plainly
    /// contradicting it: blocks that all "verified" while not one segment
    /// header read as a mode the symbol has are a false solve of a wrong
    /// placement, and count as none.
    pub all_verified: bool,
    /// What follows the end of data looks like padding.
    pub tail_plausible: bool,
    pub blocks_verified: usize,
    pub blocks_solved: usize,
    /// Total confirming bits over solved blocks. A codeword that satisfies a
    /// higher EC level also satisfies a lower one with the same block count,
    /// so of two clean attempts the one that checked more is the real one.
    pub confidence_bits: u32,
    /// Negated count of codewords error correction had to change or fill.
    pub neg_errata: i64,
}

#[derive(Clone, Debug)]
pub struct DecodeReport {
    pub ec: EcLevel,
    pub mask: u8,
    /// Indexed by placement order, as in [`SymbolLayout::codewords`].
    pub codewords: Vec<CodewordReport>,
    pub blocks: Vec<BlockReport>,
    pub stream: StreamBits,
    pub parse: ParseReport,
    /// How the bits after the end of data compare with standard padding.
    pub tail: Option<TailCheck>,
    pub text: TextReport,
    pub score: AttemptScore,
    /// Pins the decode placed itself, beyond those it was given, on the
    /// padding, terminator and headers the stream showed
    /// ([`DecodeOptions::derive_pins`]), and decoded under: they show in
    /// `codewords` as any pin does. Empty where it placed none.
    pub derived_pins: Pins,
}

impl DecodeReport {
    pub fn blocks_solved(&self) -> usize {
        self.blocks.iter().filter(|b| b.is_solved()).count()
    }

    pub fn blocks_verified(&self) -> usize {
        self.blocks.iter().filter(|b| b.is_verified()).count()
    }

    pub fn is_complete(&self) -> bool {
        self.blocks.iter().all(BlockReport::is_solved)
    }

    /// Complete, with every block confirmed and a stream that parses.
    pub fn is_trustworthy(&self) -> bool {
        self.score.all_verified && self.score.stream_valid
    }
}

/// A solution of a block as [`solve_block`] keeps one: its outcome, the
/// block as it corrected it, and the bits of each codeword it was
/// reached by doubting ([`CodewordReport::doubted`]), none for most.
#[derive(Clone, Debug)]
struct Candidate {
    outcome: BlockOutcome,
    corrected: Vec<u8>,
    doubted: Vec<u8>,
}

impl Candidate {
    fn rank(&self) -> (bool, u32) {
        solve_rank(&self.outcome)
    }
}

#[derive(Clone, Debug)]
struct BlockSolve {
    outcome: BlockOutcome,
    corrected: Option<Vec<u8>>,
    /// When the block did not solve: bits settled anyway, as (value, mask)
    /// per codeword.
    inferred: Vec<(u8, u8)>,
    /// The bits of each codeword the solution was reached by doubting
    /// ([`CodewordReport::doubted`]); empty when it doubted none.
    doubted: Vec<u8>,
    /// Solutions a solver that tries many ways found and could not tell
    /// apart by the block as read, when nothing verified the block: for
    /// the message to tell apart ([`arbitrate`]), best ranked first.
    rivals: Vec<Candidate>,
    /// The solution was told from this many rivals by the message.
    arbitrated_among: usize,
    /// Modules of the block whose reliability was lowered for errors
    /// located beside them, when the solution was reached so.
    near_errors: usize,
    /// See [`BlockReport::half_padded_with_ones`].
    half_padded_with_ones: bool,
}

impl BlockSolve {
    fn solved(candidate: Candidate) -> Self {
        let inferred = vec![(0, 0); candidate.corrected.len()];
        Self {
            outcome: candidate.outcome,
            corrected: Some(candidate.corrected),
            inferred,
            doubted: candidate.doubted,
            rivals: Vec::new(),
            arbitrated_among: 0,
            near_errors: 0,
            half_padded_with_ones: false,
        }
    }

    fn rank(&self) -> (bool, u32) {
        solve_rank(&self.outcome)
    }
}

/// How block solutions rank: verified solutions within the standard first,
/// then by confirmation. A failure ranks below every solution.
fn solve_rank(outcome: &BlockOutcome) -> (bool, u32) {
    (outcome.is_verified(), outcome.confidence_bits())
}

/// Rivals a block keeps for the message to tell apart, at the most: the
/// best ranked. Every one costs a parse of the stream.
const MAX_RIVALS: usize = 8;

/// Solve one block, trying the cheap exact decoder first and then every
/// fallback, keeping the solution with the most confirmation. `reserved`
/// EC codewords are the standard's misdecode protection: a solution that
/// spends them is reported, but marked as beyond the standard, and ranks
/// below any verified solution within it.
///
/// An exact solution ends the search only when it is verified within the
/// standard, as one that corrected to the limit of the block's error
/// correction is by the errors it located. One with too little to confirm
/// it, which is nothing where it only filled unknown codewords and had
/// none to spare, and less than three codewords' worth wherever it
/// filled any ([`BlockOutcome::bits_to_verify`]), or one that spent the
/// reserved codewords, is kept as a last resort while the fallbacks look
/// for better: flipping the faint bit of the one wrong codeword turns a
/// solution that fills fifteen unknown codewords and locates that one
/// with a codeword to spare into a confirmed one, with three.
///
/// A fallback that tries many ways may reach two codewords it cannot
/// tell apart by the block as read ([`super::rs::DISTANCE_MARGIN`]).
/// Those are kept beside the solution as rivals, when nothing verified
/// the block, for the message to tell apart ([`arbitrate`]): a block
/// that would otherwise have been given up over a tie.
fn solve_block(
    block: &[CwRead],
    reliability: &[u8],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    reserved: usize,
    opts: &DecodeOptions,
) -> BlockSolve {
    let mut best: Option<Candidate> = None;
    let mut rivals: Vec<Candidate> = Vec::new();
    fn consider(best: &mut Option<Candidate>, candidate: Candidate) {
        if best.as_ref().is_none_or(|b| candidate.rank() > b.rank()) {
            *best = Some(candidate);
        }
    }
    /// What a solver that tries many ways found: one solution is
    /// considered as any other's; several it could not tell apart are
    /// kept as rivals.
    fn offer(best: &mut Option<Candidate>, rivals: &mut Vec<Candidate>, mut found: Vec<Candidate>) {
        if found.len() == 1 {
            consider(best, found.pop().expect("one"));
        } else {
            rivals.extend(found);
        }
    }

    let verified_so_far = |best: &Option<Candidate>| best.as_ref().is_some_and(|b| b.rank().0);
    let erased = block.iter().filter(|c| c.is_erasure()).count();
    let rs_outcome = |method: SolveMethod, solution: &RsSolution| BlockOutcome::Solved {
        method,
        errors: solution.errors,
        erasures: solution.erasures,
        slack: solution.slack,
        confidence_bits: solution.confidence_bits(),
        beyond_standard: !solution.within_standard(),
    };
    let bit_outcome = |doubt: &BitDoubt| {
        let method = match (doubt.erased.len(), doubt.flipped_bits() as usize) {
            (0, 0) => SolveMethod::BitReliability { doubted_bits: doubt.doubted_bits },
            (0, flipped_bits) => SolveMethod::BitFlip { doubted_bits: doubt.doubted_bits, flipped_bits },
            (erased, _) => SolveMethod::BitTrial { doubted_bits: doubt.doubted_bits, erased },
        };
        BlockOutcome::Solved {
            method,
            errors: doubt.errors(),
            erasures: erased,
            slack: 0,
            confidence_bits: doubt.confidence_bits(),
            beyond_standard: doubt.solution.spare_equations < 8 * reserved,
        }
    };
    let bit_candidate = |doubt: BitDoubt| {
        let outcome = bit_outcome(&doubt);
        Candidate { outcome, corrected: doubt.solution.corrected, doubted: doubt.doubted }
    };
    let flips = opts.reliability && opts.flip_bits;

    let failure = match rs_decode_reserving(block, ec_len, reserved).map(|s| s.among(opts.tries)) {
        Ok(solution) => {
            let outcome = rs_outcome(SolveMethod::ReedSolomon, &solution);
            let candidate = Candidate { outcome, corrected: solution.corrected, doubted: Vec::new() };
            if candidate.rank().0 {
                return BlockSolve::solved(candidate);
            }
            consider(&mut best, candidate);
            RsFailure::BudgetExceeded
        }
        Err(failure) => failure,
    };
    // The trials of erasures are for a block the exact decoder cannot
    // solve at all. On one it solved, erasing by trial the codewords it
    // located as wrong names the same codeword with a unit of redundancy
    // more to spare, for nothing: a wrong codeword costs two and an
    // erasure one.
    let exact_failed = best.is_none();

    let mut partial = None;
    if opts.bit_level {
        let bits = solve_bit_erasures(block, ec_len);
        let inconsistent = matches!(bits, Err(BitSolveFailure::Inconsistent));
        match bits {
            Ok(solution) if solution.is_complete() => {
                let outcome = BlockOutcome::Solved {
                    method: SolveMethod::BitLevel,
                    errors: 0,
                    erasures: erased,
                    slack: 0,
                    confidence_bits: solution.spare_equations as u32,
                    // Spare equations stand in for unused codewords.
                    beyond_standard: solution.spare_equations < 8 * reserved,
                };
                consider(&mut best, Candidate { outcome, corrected: solution.corrected, doubted: Vec::new() });
            }
            Ok(solution) => partial = Some(solution),
            Err(_) => {}
        }
        // The known bits cannot all be right: a bit or two of them
        // flipped, or a codeword or two freed by trial, may make them
        // agree, whatever their reliability.
        let trials = Trials { flips, free: opts.erase_by_trial && exact_failed, sets: opts.trial_sets };
        if (trials.flips || trials.free) && inconsistent {
            let found = solve_bit_erasures_trying(block, bit_reliability, ec_len, trials);
            offer(&mut best, &mut rivals, found.into_iter().map(bit_candidate).collect());
        }
    }

    if opts.reliability
        && opts.doubt_codewords
        && let Some((doubted, solution)) = rs_decode_doubting(block, reliability, ec_len, reserved)
    {
        let solution = solution.among(opts.tries);
        let outcome = rs_outcome(SolveMethod::Reliability { doubted }, &solution);
        // The codewords doubted are those the solution took as erased
        // that were read in full.
        let mut which = vec![0u8; block.len()];
        for erratum in solution.errata.iter().filter(|e| e.was_erasure && !block[e.index].is_erasure()) {
            which[erratum.index] = 0xFF;
        }
        consider(&mut best, Candidate { outcome, corrected: solution.corrected, doubted: which });
    }

    // Doubting bits rather than codewords reaches damage spread over more
    // codewords than there are EC codewords, where no erasure decoder can
    // follow. It competes with doubting codewords on confirmation: a
    // codeword solution with one of slack has one syndrome byte behind
    // it, which a wrong codeword passes once in a few hundred tries, and
    // the bit solution, when there is one, has every spare equation.
    if opts.reliability && opts.doubt_bits && opts.bit_level {
        let trials = Trials { flips, free: opts.erase_by_trial && exact_failed, sets: opts.trial_sets };
        let found = solve_bit_erasures_doubting(block, bit_reliability, ec_len, trials);
        offer(&mut best, &mut rivals, found.into_iter().map(bit_candidate).collect());
    }

    // Flipping the faintest bits outright costs no redundancy where
    // doubting a codeword costs a unit and doubting a bit an equation: a
    // block wrong in a few more codewords than the code corrects is
    // reached when the extra ones read faintly, after doubting has run
    // out of erasures. Many tries are many chances of a misdecode, so of
    // the patterns that solve to different codewords the nearest the
    // block as read is kept only by a clear margin, and the rest kept as
    // rivals.
    if flips {
        let found = rs_decode_chasing_bits(block, bit_reliability, ec_len, reserved, chase_bits_among(opts.chase_bits, opts.tries))
            .into_iter()
            .map(|(flipped, solution)| {
                let solution = solution.among(opts.tries);
                let flipped_bits = flipped.iter().map(|bits| bits.count_ones() as usize).sum();
                let outcome = rs_outcome(SolveMethod::Chase { flipped_bits }, &solution);
                Candidate { outcome, corrected: solution.corrected, doubted: flipped }
            })
            .collect();
        offer(&mut best, &mut rivals, found);
    }

    // A curve through the block's codewords, weighed by their
    // reliability, and every codeword of the code read off it: reaches
    // past half the code's distance where the trials of small sets stop,
    // and bends toward the other colour of every faint module at once
    // where the chase flips ten. Before the trials of erasures, which
    // are dearer, and only while nothing has verified the block; and, as
    // the trials, only for a block the exact decoder cannot solve at all,
    // since within half the distance the curve reaches the codeword the
    // exact decoder found and no other. With doubting by reliability
    // turned off every bit is as certain as painted, and the curve is
    // the hard decision one.
    if opts.list_decode && opts.list_cost > 0 && exact_failed && !verified_so_far(&best) {
        let certain = vec![[CERTAIN; 8]; block.len()];
        let weighed = if opts.reliability { bit_reliability } else { &certain[..] };
        let found = rs_decode_listing(block, weighed, ec_len, reserved, opts.list_cost)
            .into_iter()
            .map(|(multiplicity, solution)| {
                let solution = solution.among(opts.tries);
                let outcome = rs_outcome(SolveMethod::List { multiplicity }, &solution);
                let mut which = vec![0u8; block.len()];
                for erratum in solution.errata.iter().filter(|e| !e.was_erasure) {
                    which[erratum.index] = erratum.magnitude & block[erratum.index].known;
                }
                Candidate { outcome, corrected: solution.corrected, doubted: which }
            })
            .collect();
        offer(&mut best, &mut rivals, found);
    }

    // Erasing codewords by trial reaches a wrong codeword or two more
    // than the code corrects, whatever the modules' reliability: the
    // last resort of a grid painted by hand, which nothing above can
    // doubt, and the dearest, every set of the block's codewords being
    // a decode. Only when nothing above has verified the block.
    if opts.erase_by_trial && exact_failed && !verified_so_far(&best) {
        let found = rs_decode_trying(block, bit_reliability, ec_len, reserved, opts.trial_sets)
            .into_iter()
            .map(|(erased, solution)| {
                let solution = solution.among(opts.tries);
                let outcome = rs_outcome(SolveMethod::Trial { erased: erased.len() }, &solution);
                let mut which = vec![0u8; block.len()];
                for &i in &erased {
                    which[i] = block[i].known & !block[i].pinned;
                }
                Candidate { outcome, corrected: solution.corrected, doubted: which }
            })
            .collect();
        offer(&mut best, &mut rivals, found);
    }

    // With enough pinned codewords the block is determined by pins alone,
    // however badly the rest was read.
    if block.iter().any(|c| c.pinned == 0xFF) {
        let pinned_only: Vec<CwRead> = block
            .iter()
            .map(|c| CwRead { value: c.value & c.pinned, known: c.pinned, pinned: c.pinned })
            .collect();
        if let Ok(solution) = rs_decode_reserving(&pinned_only, ec_len, reserved) {
            let outcome = BlockOutcome::Solved {
                method: SolveMethod::PinnedOnly,
                errors: 0,
                erasures: solution.erasures,
                slack: solution.slack,
                confidence_bits: solution.confidence_bits(),
                beyond_standard: !solution.within_standard(),
            };
            consider(&mut best, Candidate { outcome, corrected: solution.corrected, doubted: Vec::new() });
        }
    }

    // Rivals are for a block nothing verified. Each codeword once, the
    // best ranked solution of it standing for the rest, and the best
    // ranked few of those.
    let rivals = if verified_so_far(&best) {
        Vec::new()
    } else {
        rivals.sort_by(|a, b| b.rank().cmp(&a.rank()));
        let mut distinct: Vec<Candidate> = Vec::new();
        for candidate in rivals {
            if !distinct.iter().any(|d| d.corrected == candidate.corrected) {
                distinct.push(candidate);
            }
        }
        distinct.truncate(MAX_RIVALS);
        distinct
    };

    if let Some(candidate) = best {
        return BlockSolve { rivals, ..BlockSolve::solved(candidate) };
    }
    let inferred = match partial {
        Some(solution) => solution.corrected.iter().zip(&solution.solved).map(|(&v, &m)| (v & m, m)).collect(),
        None => vec![(0, 0); block.len()],
    };
    BlockSolve {
        outcome: BlockOutcome::Failed(failure),
        corrected: None,
        inferred,
        doubted: Vec::new(),
        rivals,
        arbitrated_among: 0,
        near_errors: 0,
        half_padded_with_ones: false,
    }
}

/// The reliability of each bit of each codeword, `bits[cw][b]` for bit
/// `1 << b`, from that of each module: bit `b` of a codeword is its
/// module `7 - b` (less the bits before its first module), and a bit the
/// symbol does not hold, in a four-bit codeword, is certain.
fn bit_reliabilities(layout: &SymbolLayout, module_reliability: &[u8]) -> Vec<[u8; 8]> {
    let width = layout.func.width;
    (0..layout.codewords.len())
        .map(|cw| {
            let mut bits = [CERTAIN; 8];
            let offset = layout.codewords[cw].bit_offset as usize;
            for (i, p) in layout.codeword_modules(cw).iter().enumerate() {
                bits[7 - offset - i] = module_reliability[p.y as usize * width + p.x as usize];
            }
            bits
        })
        .collect()
}

/// Solve the block `spec` describes, from the codewords as pinned and
/// the reliability of every module. The final four-bit codeword of an M1
/// or M3 symbol enters error correction widened to a byte, its missing
/// nibble the standard's zeros. An encoder in the field computes its
/// error correction over the nibble filled with ones instead, which
/// leaves a block of M3 one codeword wrong, and its redundancy corrects
/// that, but a block of M1, which detects errors and corrects none,
/// cannot solve. So a block that did not solve with confirmation is
/// tried again with the nibble filled with ones, and the better solution
/// kept.
fn solve_spec(layout: &SymbolLayout, spec: &BlockSpec, pinned: &[CwRead], bit_reliability: &[[u8; 8]], opts: &DecodeOptions) -> BlockSolve {
    let block: Vec<CwRead> = spec.codewords.iter().map(|&cw| pinned[cw]).collect();
    let block_bits: Vec<[u8; 8]> = spec.codewords.iter().map(|&cw| bit_reliability[cw]).collect();
    // A codeword as a whole is as reliable as its least reliable module.
    let block_reliability: Vec<u8> = block_bits.iter().map(|bits| *bits.iter().min().unwrap_or(&CERTAIN)).collect();
    let mut solve = solve_block(&block, &block_reliability, &block_bits, spec.ec_len, spec.reserved, opts);
    if tables::has_half_codeword(layout.version()) && !solve.rank().0 {
        let i = spec.data_len - 1;
        let missing = !codeword_bit_mask(layout, spec.codewords[i]);
        let mut filled = block.clone();
        filled[i].value |= missing;
        let mut again = solve_block(&filled, &block_reliability, &block_bits, spec.ec_len, spec.reserved, opts);
        // A solution that erased the widened codeword by trial owes
        // nothing to the ones: it filled the codeword with whatever fit,
        // and only counted the nibble as one wrong codeword more.
        let owes_to_ones = again.doubted.get(i).is_none_or(|&doubted| doubted & missing == 0);
        if owes_to_ones && again.rank() > solve.rank() {
            again.half_padded_with_ones = true;
            solve = again;
        }
    }
    solve
}

/// Fill in the codeword reports of one block from its solution, and
/// report the block.
fn report_block(
    layout: &SymbolLayout,
    spec: &BlockSpec,
    solve: &BlockSolve,
    raw: &[CwRead],
    pinned: &[CwRead],
    codewords: &mut [CodewordReport],
) -> BlockReport {
    let block: Vec<CwRead> = spec.codewords.iter().map(|&cw| pinned[cw]).collect();
    for (i, &cw) in spec.codewords.iter().enumerate() {
        let exists = codeword_bit_mask(layout, cw);
        let report = &mut codewords[cw];
        let read_in_full = raw[cw].known & exists == exists;
        match &solve.corrected {
            Some(values) => {
                report.final_value = Some(values[i]);
                report.inferred_value = 0;
                report.inferred_known = 0;
                report.doubted = solve.doubted.get(i).copied().unwrap_or(0) & exists;
                report.status = if !read_in_full {
                    CwStatus::Filled
                } else if (values[i] ^ raw[cw].value) & exists != 0 {
                    CwStatus::Corrected
                } else {
                    CwStatus::Clean
                };
            }
            None => {
                let (value, mask) = solve.inferred[i];
                report.final_value = None;
                report.doubted = 0;
                report.inferred_value = value;
                report.inferred_known = mask;
                report.status = if !block[i].is_erasure() {
                    CwStatus::Unverified
                } else if mask != 0 {
                    CwStatus::Inferred
                } else {
                    CwStatus::Unresolved
                };
            }
        }
    }
    BlockReport {
        data_len: spec.data_len,
        ec_len: spec.ec_len,
        reserved: spec.reserved,
        outcome: solve.outcome,
        erased_codewords: block.iter().filter(|c| c.is_erasure()).count(),
        unknown_bits: block.iter().map(|c| c.known.count_zeros() as usize).sum(),
        inferred_bits: solve.inferred.iter().map(|(_, m)| m.count_ones() as usize).sum(),
        half_padded_with_ones: solve.half_padded_with_ones,
        rivals: solve.arbitrated_among,
        near_errors: solve.near_errors,
    }
}

/// The data stream and what is made of it: assembled from the codeword
/// reports, parsed, its tail checked against the padding, and read as
/// text.
struct Assembled {
    stream: StreamBits,
    parse: ParseReport,
    tail: Option<TailCheck>,
    text: TextReport,
    stream_valid: bool,
    tail_plausible: bool,
}

fn assemble(layout: &SymbolLayout, codewords: &[CodewordReport], blocks: &[BlockReport], pinned: &[CwRead], charset: Charset) -> Assembled {
    let mut stream = StreamBits::default();
    for &cw in &layout.data_stream {
        let span = layout.codewords[cw];
        let block = &blocks[span.block as usize];
        let solved_as = if block.is_verified() { Certainty::Verified } else { Certainty::Unverified };
        let report = &codewords[cw];
        let (value, known) = report.best();
        for bit in (0..span.stream_bits()).map(|i| 0x80u8 >> i) {
            let certainty = if pinned[cw].pinned & bit != 0 {
                Certainty::Verified
            } else if report.final_value.is_some() {
                solved_as
            } else if known & bit != 0 {
                Certainty::Unverified
            } else {
                Certainty::Unknown
            };
            stream.push(value & bit != 0, certainty);
        }
    }
    let parse = bitstream::parse(&stream, &layout.stream);
    let tail = match parse.end {
        EndOfData::At { bit, .. } => Some(padding::check_tail(layout, &stream, bit)),
        EndOfData::Lost { .. } => None,
    };
    let text = text::decode_text_of(layout.version().family(), &parse.segments, charset);
    let tail_plausible = tail.as_ref().is_some_and(TailCheck::is_plausible);
    // A stream that ends before its first segment is a message only when
    // what follows is the padding an empty message has. Under a wrong
    // mask or on a wrong placement the stream is noise, and noise begins
    // with a terminator's zeros once in sixteen tries.
    let empty_unexplained = parse.segments.is_empty() && !tail_plausible;
    let stream_valid = matches!(parse.end, EndOfData::At { .. })
        && !parse.preamble_damaged
        && !empty_unexplained
        && parse.segments.iter().all(|s| !s.damaged && !s.truncated);
    Assembled { stream, parse, tail, text, stream_valid, tail_plausible }
}

/// How well the stream reads under one of a block's rivals, for telling
/// them apart: field by field, the first that differs deciding. Nothing
/// of the rival's own confirmation, which is what the tries could not
/// tell apart by; only what its data makes of the message.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct StreamScore {
    valid: bool,
    tail_plausible: bool,
    /// The stream parses to an end, where a terminator is or capacity
    /// runs out, rather than losing its way at a header.
    ended: bool,
    /// Segments that parse with no impossible value and no overrun.
    sound_segments: usize,
    /// The fewer damaged or truncated segments the better.
    neg_damaged: i64,
}

impl StreamScore {
    fn of(assembled: &Assembled) -> Self {
        let segments = &assembled.parse.segments;
        Self {
            valid: assembled.stream_valid,
            tail_plausible: assembled.tail_plausible,
            ended: matches!(assembled.parse.end, EndOfData::At { .. }),
            sound_segments: segments.iter().filter(|s| !s.damaged && !s.truncated).count(),
            neg_damaged: -(segments.iter().filter(|s| s.damaged || s.truncated).count() as i64),
        }
    }
}

/// Tell apart the rivals of every block that has them by the message:
/// the stream assembled with each rival's data in turn, and the rival
/// under which it parses best taken, when it parses better than under
/// every other. A tie between rivals that agree on every data codeword
/// is no tie for the message, and the nearest the block as read is
/// taken; between rivals that disagree, the block stays as it was. The
/// blocks are taken in order, each with those before it as settled.
///
/// The solvers that try many ways are list decoders, and a list of two
/// is what they return where the block as read lies as near to two
/// codewords ([`super::rs::DISTANCE_MARGIN`]); the message is the one
/// judge of such a tie that is not the block itself, since a wrong
/// codeword's data parses as a segment of noise, with a header it does
/// not have or a value no character takes, where the right one's reads
/// on as the message does.
fn arbitrate(
    layout: &SymbolLayout,
    raw: &[CwRead],
    pinned: &[CwRead],
    solves: &mut [BlockSolve],
    codewords: &mut [CodewordReport],
    blocks: &mut [BlockReport],
    charset: Charset,
) {
    for b in 0..solves.len() {
        if blocks[b].is_verified() || solves[b].rivals.is_empty() {
            continue;
        }
        let spec = &layout.blocks[b];
        let rivals = std::mem::take(&mut solves[b].rivals);
        let mut scored: Vec<(StreamScore, usize)> = rivals
            .iter()
            .enumerate()
            .map(|(r, candidate)| {
                let trial = BlockSolve::solved(candidate.clone());
                blocks[b] = report_block(layout, spec, &trial, raw, pinned, codewords);
                let assembled = assemble(layout, codewords, blocks, pinned, charset);
                (StreamScore::of(&assembled), r)
            })
            .collect();
        scored.sort_by(|a, b| b.cmp(a));
        let data = spec.data_len;
        let agree_on_data = rivals.iter().all(|c| c.corrected[..data] == rivals[0].corrected[..data]);
        let winner = match scored.as_slice() {
            [(best, r), (next, _), ..] if best > next => Some(*r),
            _ if agree_on_data => Some(0),
            _ => None,
        };
        let taken = winner.map(|r| rivals[r].clone()).filter(|candidate| candidate.rank() > solves[b].rank());
        if let Some(candidate) = taken {
            let (near_errors, half) = (solves[b].near_errors, solves[b].half_padded_with_ones);
            solves[b] = BlockSolve { arbitrated_among: rivals.len(), near_errors, half_padded_with_ones: half, ..BlockSolve::solved(candidate) };
        }
        blocks[b] = report_block(layout, spec, &solves[b], raw, pinned, codewords);
    }
}

/// Reliability a module is held at, at the most, when an error was
/// located in the module beside it in a block that verified: damage is
/// contiguous, a stain or a tear or a fold, and a module next to one the
/// photo was shown to have misread is a module to doubt first. On the
/// calibrated scale ([`crate::imaging::sample::Classified::calibrate`])
/// this is three bits of likelihood: one such module in eight misread.
/// Modules painted by hand stay certain; only sampled ones are lowered.
pub const NEAR_ERROR_RELIABILITY: u8 = 16;
/// Rounds of lowering and solving again: each block that then verifies
/// locates errors of its own for the next.
const NEAR_ERROR_ROUNDS: usize = 2;

/// The modules the photo misread, as the blocks that verified show them:
/// every module of a verified block read as known whose bit the solution
/// overturned.
fn lying_modules(layout: &SymbolLayout, raw: &[CwRead], codewords: &[CodewordReport], blocks: &[BlockReport]) -> Vec<(usize, usize)> {
    let mut lying = Vec::new();
    for (b, spec) in layout.blocks.iter().enumerate() {
        if !blocks[b].is_verified() {
            continue;
        }
        for &cw in &spec.codewords {
            let Some(value) = codewords[cw].final_value else { continue };
            let overturned = (value ^ raw[cw].value) & raw[cw].known & codeword_bit_mask(layout, cw);
            if overturned == 0 {
                continue;
            }
            let offset = layout.codewords[cw].bit_offset as usize;
            for (i, pos) in layout.codeword_modules(cw).iter().enumerate() {
                if overturned & (0x80 >> (offset + i)) != 0 {
                    lying.push((pos.x as usize, pos.y as usize));
                }
            }
        }
    }
    lying
}

/// The module reliabilities with every sampled module beside a lying one
/// held at [`NEAR_ERROR_RELIABILITY`] or below.
fn lowered_beside(module_reliability: &[u8], lying: &[(usize, usize)], width: usize, height: usize) -> Vec<u8> {
    let mut lowered = module_reliability.to_vec();
    for &(x, y) in lying {
        for ny in y.saturating_sub(1)..(y + 2).min(height) {
            for nx in x.saturating_sub(1)..(x + 2).min(width) {
                let r = &mut lowered[ny * width + nx];
                if *r < CERTAIN {
                    *r = (*r).min(NEAR_ERROR_RELIABILITY);
                }
            }
        }
    }
    lowered
}

/// The exact decode of a block, when it verifies within the standard:
/// the first thing [`solve_block`] tries, and all a sound block needs,
/// made alone before the hard blocks are shared over the cores, since
/// it is microseconds and a thread is more.
fn exact_solve(block: &[CwRead], ec_len: usize, reserved: usize, opts: &DecodeOptions) -> Option<BlockSolve> {
    let solution = rs_decode_reserving(block, ec_len, reserved).ok()?.among(opts.tries);
    let outcome = BlockOutcome::Solved {
        method: SolveMethod::ReedSolomon,
        errors: solution.errors,
        erasures: solution.erasures,
        slack: solution.slack,
        confidence_bits: solution.confidence_bits(),
        beyond_standard: !solution.within_standard(),
    };
    let candidate = Candidate { outcome, corrected: solution.corrected, doubted: Vec::new() };
    candidate.rank().0.then(|| BlockSolve::solved(candidate))
}

/// What one block was last solved from, and what came of it.
#[derive(Clone)]
struct Solved {
    reads: Vec<CwRead>,
    bits: Vec<[u8; 8]>,
    solve: BlockSolve,
}

/// Every block of the layout solved from the codewords as pinned and the
/// reliability of every bit. A block solved before from the same reads
/// and reliabilities, as `solved` remembers each, is not solved again:
/// a decode made again under more pins solves only the blocks the pins
/// touch, and the rounds of lowering reliabilities beside located errors
/// solve only the blocks whose reliabilities moved. Of the rest, the
/// exact decoder settles the sound ones alone, and the hard ones, each
/// a search of its fallbacks, are solved side by side on the spare
/// cores ([`work::map`]).
fn solve_blocks(
    layout: &SymbolLayout,
    opts: &DecodeOptions,
    solved: &mut [Option<Solved>],
    pinned: &[CwRead],
    bit_reliability: &[[u8; 8]],
) -> Vec<BlockSolve> {
    let inputs = |spec: &BlockSpec| -> (Vec<CwRead>, Vec<[u8; 8]>) {
        (spec.codewords.iter().map(|&cw| pinned[cw]).collect(), spec.codewords.iter().map(|&cw| bit_reliability[cw]).collect())
    };
    let mut hard = Vec::new();
    let mut out: Vec<Option<BlockSolve>> = Vec::with_capacity(layout.blocks.len());
    for (b, spec) in layout.blocks.iter().enumerate() {
        let (reads, bits) = inputs(spec);
        if let Some(before) = &solved[b]
            && before.reads == reads
            && before.bits == bits
        {
            out.push(Some(before.solve.clone()));
            continue;
        }
        match exact_solve(&reads, spec.ec_len, spec.reserved, opts) {
            Some(solve) => {
                solved[b] = Some(Solved { reads, bits, solve: solve.clone() });
                out.push(Some(solve));
            }
            None => {
                hard.push(b);
                out.push(None);
            }
        }
    }
    let solves: Vec<BlockSolve> = if hard.len() >= 2 {
        work::map(&hard, |&b| solve_spec(layout, &layout.blocks[b], pinned, bit_reliability, opts))
    } else {
        hard.iter().map(|&b| solve_spec(layout, &layout.blocks[b], pinned, bit_reliability, opts)).collect()
    };
    for (&b, solve) in hard.iter().zip(solves) {
        let (reads, bits) = inputs(&layout.blocks[b]);
        solved[b] = Some(Solved { reads, bits, solve: solve.clone() });
        out[b] = Some(solve);
    }
    out.into_iter().map(|solve| solve.expect("every block solved")).collect()
}

/// A decode of one grid under one layout and mask, kept so that it can
/// be made again under more pins with only the blocks the pins touch
/// solved anew: what [`decode`] does to pin the padding it finds
/// ([`DecodeOptions::derive_pins`]).
pub struct Decoding<'a> {
    layout: &'a SymbolLayout,
    mask: u8,
    opts: &'a DecodeOptions,
    has_reliability: bool,
    raw: Vec<CwRead>,
    module_reliability: Vec<u8>,
    bit_reliability: Vec<[u8; 8]>,
    /// Each block as last solved, from whatever reads and reliabilities.
    solved: Vec<Option<Solved>>,
}

impl<'a> Decoding<'a> {
    pub fn new(grid: &ModuleGrid, layout: &'a SymbolLayout, mask: u8, opts: &'a DecodeOptions) -> Self {
        let raw = read_codewords(grid, layout, mask);
        let (width, height) = (layout.func.width, layout.func.height);
        let module_reliability: Vec<u8> = (0..width * height).map(|i| grid.reliability(i % width, i / width)).collect();
        let bit_reliability = bit_reliabilities(layout, &module_reliability);
        Self {
            layout,
            mask,
            opts,
            has_reliability: grid.has_reliability(),
            raw,
            module_reliability,
            bit_reliability,
            solved: vec![None; layout.blocks.len()],
        }
    }

    /// The grid decoded under `pins`.
    pub fn solve(&mut self, pins: &Pins) -> DecodeReport {
        let (layout, opts) = (self.layout, self.opts);
        let raw = &self.raw;
        let pinned = apply_pins(layout, raw, pins);
        let (width, height) = (layout.func.width, layout.func.height);

        let mut codewords: Vec<CodewordReport> = (0..raw.len())
            .map(|cw| CodewordReport {
                read: raw[cw].value,
                known: raw[cw].known,
                pin: pins.get(&cw).copied(),
                pinned_value: pinned[cw].value,
                pinned_known: pinned[cw].known,
                inferred_value: 0,
                inferred_known: 0,
                final_value: None,
                doubted: 0,
                status: CwStatus::Unresolved,
            })
            .collect();

        let mut solves = solve_blocks(layout, opts, &mut self.solved, &pinned, &self.bit_reliability);
        let mut blocks: Vec<BlockReport> =
            layout.blocks.iter().zip(&solves).map(|(spec, solve)| report_block(layout, spec, solve, raw, &pinned, &mut codewords)).collect();
        arbitrate(layout, raw, &pinned, &mut solves, &mut codewords, &mut blocks, opts.charset);

        // The blocks that verified say where the photo misread: an error
        // located is a module read as known and overturned, and the modules
        // beside it, in the blocks that did not verify, are doubted first
        // when those are solved again. Only a grid sampled from a photo has
        // modules to doubt.
        if self.has_reliability && opts.reliability {
            for _ in 0..NEAR_ERROR_ROUNDS {
                if blocks.iter().all(BlockReport::is_verified) || work::cancelled() {
                    break;
                }
                let lying = lying_modules(layout, raw, &codewords, &blocks);
                if lying.is_empty() {
                    break;
                }
                let lowered = lowered_beside(&self.module_reliability, &lying, width, height);
                let lowered_bits = bit_reliabilities(layout, &lowered);
                // Every block is asked of the memory; those with modules
                // lowered and not yet solved so are solved anew.
                let again = solve_blocks(layout, opts, &mut self.solved, &pinned, &lowered_bits);
                let mut improved = false;
                for (b, spec) in layout.blocks.iter().enumerate() {
                    if blocks[b].is_verified() {
                        continue;
                    }
                    let near_errors = spec
                        .codewords
                        .iter()
                        .flat_map(|&cw| layout.codeword_modules(cw))
                        .filter(|p| lowered[p.y as usize * width + p.x as usize] < self.module_reliability[p.y as usize * width + p.x as usize])
                        .count();
                    if near_errors == 0 {
                        continue;
                    }
                    let mut again = again[b].clone();
                    again.near_errors = near_errors;
                    if again.rank() > solves[b].rank() || (again.rank() == solves[b].rank() && again.rivals.len() > solves[b].rivals.len()) {
                        solves[b] = again;
                        improved = true;
                    }
                }
                if !improved {
                    break;
                }
                blocks = layout.blocks.iter().zip(&solves).map(|(spec, solve)| report_block(layout, spec, solve, raw, &pinned, &mut codewords)).collect();
                arbitrate(layout, raw, &pinned, &mut solves, &mut codewords, &mut blocks, opts.charset);
            }
        }

        let Assembled { stream, parse, tail, text, stream_valid, tail_plausible } = assemble(layout, &codewords, &blocks, &pinned, opts.charset);
        let blocks_solved = blocks.iter().filter(|b| b.is_solved()).count();
        let blocks_verified = blocks.iter().filter(|b| b.is_verified()).count();
        let errata = codewords.iter().filter(|c| matches!(c.status, CwStatus::Corrected | CwStatus::Filled)).count();
        let confidence_bits = blocks.iter().map(|b| b.outcome.confidence_bits()).sum();
        // Every block "verified" yet nothing parsed, not even the first
        // segment's header: the verification is the false solve of a wrong
        // placement, as [`AttemptScore`] explains, and ranks as none.
        let all_verified = blocks_verified == blocks.len();
        let contradicted = all_verified && !stream_valid && parse.segments.is_empty();
        let score = AttemptScore {
            stream_valid,
            all_verified: all_verified && !contradicted,
            tail_plausible,
            blocks_verified: if contradicted { 0 } else { blocks_verified },
            blocks_solved,
            confidence_bits,
            neg_errata: -(errata as i64),
        };

        DecodeReport { ec: layout.ec, mask: self.mask, codewords, blocks, stream, parse, tail, text, score, derived_pins: Pins::new() }
    }
}

pub fn decode(grid: &ModuleGrid, layout: &SymbolLayout, mask: u8, pins: &Pins, opts: &DecodeOptions) -> DecodeReport {
    let mut decoding = Decoding::new(grid, layout, mask, opts);
    let report = decoding.solve(pins);
    if !opts.derive_pins || report.is_trustworthy() || work::cancelled() {
        return report;
    }
    derive_pins(&mut decoding, pins, report)
}

/// The decode made again with the padding, terminator and headers the
/// stream shows pinned, by padding recovery over the decode in hand,
/// each round solving anew only the blocks its pins touch: taken, with
/// the pins it placed noted, when the pins solve or verify a block more
/// or the segment headers confirm where the padding was placed, and the
/// decode ranks no worse; else the decode as it was. Pins asserted on
/// what the pad pattern alone showed, that solve nothing more, are not
/// taken: they would stand for nothing but themselves.
fn derive_pins(decoding: &mut Decoding, pins: &Pins, report: DecodeReport) -> DecodeReport {
    let layout = decoding.layout;
    let Some(recovery) = padding::recover_padding_with(layout, pins, &report, |more| decoding.solve(more)) else { return report };
    let helped = recovery.solved_after > recovery.solved_before
        || recovery.report.blocks_verified() > report.blocks_verified()
        || recovery.confirmed;
    if !helped || recovery.report.score < report.score {
        return report;
    }
    let mut taken = recovery.report;
    taken.derived_pins = recovery.pins.into_iter().filter(|(cw, pin)| pins.get(cw) != Some(pin)).collect();
    taken
}

/// What the grid says about itself before any decoding.
#[derive(Clone, Debug)]
pub struct GridAnalysis {
    pub format: FormatAnalysis,
    pub version: Option<VersionAnalysis>,
    /// (matching, known) over fixed-pattern modules.
    pub fixed: (usize, usize),
    /// The fixed patterns read as their inverse: the grid is a
    /// light-on-dark symbol taken as is, and nothing in it will decode
    /// until dark and light are swapped.
    pub inverted: bool,
    /// The format bits (and version bits, where there are any) spell
    /// nothing as read but a valid word once the grid is reflected this
    /// way: the symbol was seen from the back, and the grid is its mirror
    /// image. The fixed patterns of a square symbol look the same either
    /// way, so only the information bits can tell.
    pub mirrored: Option<Mirror>,
    /// The fixed patterns do not fit the grid as painted, but fit well
    /// once it is turned or reflected this way: the grid was painted the
    /// wrong way up, or from the back, or both. A reflection moves the
    /// finder patterns to other corners, which the fixed patterns show;
    /// the one reflection that keeps them in place is `mirrored`'s job.
    pub orientation: Option<Orientation>,
}

/// A symmetry under which a grid's fixed patterns read far better than as
/// painted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Orientation {
    pub symmetry: Symmetry,
    /// Dark and light must be swapped as well.
    pub inverted: bool,
    /// (matching, known) over fixed modules once transformed.
    pub agreement: (usize, usize),
}

/// Share of known fixed modules that must read right for the grid as
/// painted to need no turning, and that a turned grid must reach, with a
/// margin between them, to be offered instead. A quarter turn of a square
/// symbol leaves two of its three finder patterns in place, so a turned
/// grid still reads about four fifths right; the bar for "as painted" is
/// set above that.
const ORIENTATION_POOR: f64 = 0.9;
const ORIENTATION_GOOD: f64 = 0.97;
const ORIENTATION_MARGIN: f64 = 0.1;

/// The symmetry, with or without inversion, under which the fixed
/// patterns read best, when they read poorly as painted and well that
/// way. ZXing and ZBar retry a failed decode mirrored; here every turn and
/// reflection is tried on the fixed patterns alone, which costs nothing
/// and names the transform to apply.
fn suggest_orientation(grid: &ModuleGrid, func: &FunctionMap, fixed: (usize, usize)) -> Option<Orientation> {
    let (matching, known) = fixed;
    if known < POLARITY_MIN_KNOWN {
        return None;
    }
    let plain = matching as f64 / known as f64;
    if plain >= ORIENTATION_POOR {
        return None;
    }
    let square = func.width == func.height;
    let mut best: Option<Orientation> = None;
    for &symmetry in Symmetry::of_shape(square) {
        let turned = grid.symmetric(symmetry);
        let (m, k) = func.fixed_agreement(&turned);
        if k < POLARITY_MIN_KNOWN {
            continue;
        }
        for inverted in [false, true] {
            let agreement = if inverted { (k - m, k) } else { (m, k) };
            let share = agreement.0 as f64 / k as f64;
            if share >= ORIENTATION_GOOD
                && share >= plain + ORIENTATION_MARGIN
                && best.is_none_or(|b| share > b.agreement.0 as f64 / b.agreement.1 as f64)
            {
                best = Some(Orientation { symmetry, inverted, agreement });
            }
        }
    }
    best
}

/// The [`GridAnalysis::orientation`] of `grid` alone, as [`analyze`]
/// makes it, for a caller with no use for the rest of an analysis: none
/// for a grid that reads inverted, which is judged once its colours are
/// right.
pub fn orientation_of(grid: &ModuleGrid, func: &FunctionMap) -> Option<Orientation> {
    let fixed = func.fixed_agreement(grid);
    if looks_inverted(fixed) { None } else { suggest_orientation(grid, func, fixed) }
}

pub fn analyze(grid: &ModuleGrid, func: &FunctionMap) -> GridAnalysis {
    let fixed = func.fixed_agreement(grid);
    let format = bch::analyze_format(grid, func);
    let version = bch::analyze_version(grid, func);
    let inverted = looks_inverted(fixed);
    let orientation = if inverted { None } else { suggest_orientation(grid, func, fixed) };
    // An inverted grid reads as noise either way round; it is judged once
    // the colours are right.
    let mirrored = (!inverted)
        .then(|| {
            let plain = format.summary();
            // Of the reflections that read as a word, the one that reads
            // best: a flip the wrong way round still leaves noise.
            func.version
                .mirrors()
                .iter()
                .copied()
                .map(|mirror| (mirror, bch::analyze_format(&grid.mirrored(mirror), func).summary()))
                .filter(|&(_, reflected)| bch::mirror_wins(plain, reflected))
                .min_by_key(|&(_, reflected)| reflected.mismatches)
                .map(|(mirror, _)| mirror)
        })
        .flatten();
    GridAnalysis { format, version, fixed, inverted, mirrored, orientation }
}

#[derive(Clone, Debug)]
pub struct Attempt {
    pub ec: EcLevel,
    pub mask: u8,
    /// Mismatches between this choice's format codeword and the bits read.
    pub format_mismatches: u32,
    /// The grid was reflected this way before decoding: the attempt is of
    /// the symbol's mirror image.
    pub mirror: Option<Mirror>,
    pub report: DecodeReport,
}

/// Block solves the second pass of [`brute_force`] may spend on doubting:
/// enough to retry every attempt of a symbol with a few blocks, and a
/// handful of attempts of one with eighty.
const DOUBTING_BUDGET: usize = 128;
/// Attempts the second pass retries however many blocks they have.
const DOUBTING_MIN_ATTEMPTS: usize = 4;

/// Decode under every EC level and mask the version allows, best first.
///
/// `pins` apply only to the EC level they were made for, since codeword
/// numbering changes with the block structure.
///
/// Doubting is dear, and erasing codewords by trial dearer, and under
/// the wrong mask every block is noise that neither can help, so a grid
/// is decoded in two passes: every attempt without either, then the few
/// that rank best with them. The exact decoder and the bit-level solver,
/// which the first pass keeps, already tell the right mask from the rest
/// when any block of it is clean. Every attempt is one of as many
/// decodes as there are levels and masks ([`DecodeOptions::tries`]).
pub fn brute_force(
    grid: &ModuleGrid,
    func: &Arc<FunctionMap>,
    pins: Option<(EcLevel, &Pins)>,
    opts: &DecodeOptions,
) -> Vec<Attempt> {
    let format = bch::analyze_format(grid, func);
    let no_pins = Pins::new();
    let opts = &opts.among(func.version.ec_levels().len() * func.version.mask_count() as usize);
    let two_passes = (opts.reliability && grid.has_reliability()) || opts.erase_by_trial || opts.list_decode || opts.derive_pins;
    let first_pass = DecodeOptions { reliability: opts.reliability && !two_passes, erase_by_trial: false, list_decode: false, derive_pins: false, ..*opts };
    let mut layouts: Vec<(EcLevel, Arc<SymbolLayout>)> = Vec::new();
    let mut attempts = Vec::new();
    for &ec in func.version.ec_levels() {
        let Some(layout) = SymbolLayout::new(func.clone(), ec) else { continue };
        let layout = Arc::new(layout);
        layouts.push((ec, layout.clone()));
        let pins = match pins {
            Some((level, pins)) if level == ec => pins,
            _ => &no_pins,
        };
        for mask in 0..func.version.mask_count() {
            let format_mismatches = format
                .candidates
                .iter()
                .find(|c| c.fits_grid && c.is_standard() && c.fields.ec == ec && c.fields.mask == mask)
                .map_or(0, |c| c.best_mismatches);
            let report = decode(grid, &layout, mask, pins, &first_pass);
            attempts.push(Attempt { ec, mask, format_mismatches, mirror: None, report });
        }
    }
    let by_score = |a: &Attempt, b: &Attempt| {
        b.report.score.cmp(&a.report.score).then(a.format_mismatches.cmp(&b.format_mismatches))
    };
    attempts.sort_by(by_score);
    if two_passes {
        let blocks = attempts.iter().map(|a| a.report.blocks.len()).max().unwrap_or(1).max(1);
        // An rMQR grid has two attempts, fewer than the minimum: then all of them.
        let retry = (DOUBTING_BUDGET / blocks).max(DOUBTING_MIN_ATTEMPTS).min(attempts.len());
        for attempt in attempts.iter_mut().take(retry) {
            if attempt.report.is_trustworthy() {
                continue;
            }
            let layout = &layouts.iter().find(|(ec, _)| *ec == attempt.ec).expect("layout was built").1;
            let pins = match pins {
                Some((level, pins)) if level == attempt.ec => pins,
                _ => &no_pins,
            };
            attempt.report = decode(grid, layout, attempt.mask, pins, opts);
        }
        attempts.sort_by(by_score);
    }
    attempts
}

/// Decode the grid's mirror images under every EC level and mask, best
/// first, as ZXing and ZBar retry a failed decode with the symbol
/// transposed. The format bits of a mirrored grid usually say so, but
/// damage takes the corners they live in first; the blocks that verify
/// say so whatever the state of the format bits. Pins do not carry over:
/// they name codewords of the grid as it stands.
pub fn brute_force_mirrored(grid: &ModuleGrid, func: &Arc<FunctionMap>, opts: &DecodeOptions) -> Vec<Attempt> {
    let mut attempts = Vec::new();
    for &mirror in func.version.mirrors() {
        let reflected = grid.mirrored(mirror);
        for mut attempt in brute_force(&reflected, func, None, opts) {
            attempt.mirror = Some(mirror);
            attempts.push(attempt);
        }
    }
    attempts.sort_by(|a, b| {
        b.report.score.cmp(&a.report.score).then(a.format_mismatches.cmp(&b.format_mismatches))
    });
    attempts
}
