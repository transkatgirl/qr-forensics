//! Grid to text: read codewords, solve each RS block as far as possible,
//! assemble the data stream with per-bit certainty, and parse it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::abp::{ABP_ROUNDS, ABP_ROUNDS_MAX, ABP_ROUNDS_MIN, Propagated, nth_of, rs_decode_propagating_against};
use super::bch::{self, FormatAnalysis, VersionAnalysis};
use super::bitstream::{self, Certainty, EndOfData, ParseReport, StreamBits};
use super::clustering::Clustering;
use super::codewords::{Pin, Pins, apply_pins, codeword_bit_mask, read_codewords};
use super::evidence::{Laid, MessageEvidence};
use super::likeness::{likeness_bits, stream_as_read};
use super::list::{LIST_COST, LIST_COST_MAX, LIST_COST_MIN, rs_decode_listing_against};
use super::padding::{self, TailCheck};
use super::rs::{
    BitDoubt, BitSolveFailure, CHASE_BITS, CHASE_MAX_BITS, CwRead, DISTANCE_MARGIN, RsFailure, RsSolution, Sightings, TRIAL_SETS,
    TRIAL_SETS_MAX, TRIAL_SETS_MIN, Trials, VERIFIED_BITS, chase_bits_among, judged_by, judged_count, keeps_reserve,
    plain_reach_holds_the_reserve, rs_decode_chasing_bits_against, rs_decode_doubting_against, rs_decode_reserving, rs_decode_reserving_noting,
    rs_decode_trying_against, soft_count_bits, soft_count_short_of, soft_distance, solve_bit_erasures, solve_bit_erasures_doubting,
    solve_bit_erasures_trying, tries_charge,
};
use super::text::{self, Charset, TextReport};
use crate::symbol::{EcLevel, SymbolVersion};
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
    /// which ([`SolveMethod::BitFlip`]). A decode is made without them
    /// first ([`Self::stages`]).
    pub flip_bits: bool,
    /// When a block fails, erase its codewords by trial, every one or
    /// every pair, whatever their reliability, and keep a solution the
    /// rest of the block confirms ([`SolveMethod::Trial`],
    /// [`SolveMethod::BitTrial`]): reaches a block wrong in one codeword
    /// more than the code corrects, and one whose unknown bits solve bit
    /// by bit but for a wrong codeword or two, on a grid painted by hand
    /// as on one sampled. A decode is made without them first
    /// ([`Self::stages`]).
    pub erase_by_trial: bool,
    /// Decodes of the grid the caller makes, this one among them, under
    /// other masks and levels, samplings and placements, keeping the
    /// best: each is a fresh chance for a block of noise to show errors
    /// located, so what a solution's located errors confirm is charged
    /// a bit for each doubling of them ([`super::rs::RsSolution::among`]).
    /// One where the decode stands alone.
    pub tries: usize,
    /// Of the [`Self::tries`], those the dear solvers' budgets are shared
    /// out over ([`Self::among`]): the chase flips a bit fewer and the
    /// propagation runs a share of its rounds for each of them, as the
    /// trials and the curve are given a share of theirs. Those
    /// [`Self::one_of`] counts, as a brute force's, each have the budgets
    /// whole. One where the decode stands alone.
    pub shared: usize,
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
    /// While `reliability` holds, pass beliefs between the bits of a
    /// block that fails ([`super::abp`], [`SolveMethod::BeliefPropagation`]):
    /// the parity checks the bits share, adapted each round so that the
    /// least reliable bits each sit in a check of their own, say of each
    /// bit what the others make likely, and the block is decoded from
    /// the beliefs that settle. Reaches damage the doubting and the
    /// chase cannot: faint misreads among more faint bits read right
    /// than the doubting can take as unknown.
    pub belief_propagation: bool,
    /// Rounds the propagation is given at the most ([`ABP_ROUNDS`]), from
    /// [`super::abp::ABP_ROUNDS_LEAST`] to [`ABP_ROUNDS_MAX`]: each is a
    /// decode, and a chance of a misdecode. A share of them where the
    /// decode is one of many, never fewer than [`ABP_ROUNDS_MIN`].
    pub abp_rounds: usize,
    /// When the decode is no read, find the padding, terminator and
    /// headers the stream shows, pin them, and solve again the blocks
    /// the pins touch, to a fixed point, as padding recovery does by
    /// hand: a pinned codeword costs no error correction, and the pins
    /// are kept when they solve or verify a block more, or the segment
    /// headers confirm them ([`DecodeReport::derived_pins`]). The
    /// decodes padding recovery itself makes never derive, nor does the
    /// first pass of a brute-force search.
    pub derive_pins: bool,
    /// When a block fails, search it again with what the message makes
    /// likely of its bits laid over them ([`super::evidence`]): each
    /// group a segment packs its characters in leans its bits toward
    /// what the message's other groups of its kind hold there, a byte
    /// toward seven-bit text until they say otherwise, and a bit read
    /// against the lean is doubted the sooner, one that could not be
    /// read solved for at the lean. A solution so reached is confirmed
    /// against the block as read ([`super::rs::judged`]), every bit
    /// that was read judged by its reading alone and every one that was
    /// not taken at its lean, as an assumption no stronger than the lean
    /// ([`super::evidence`]), and is charged as one of the searches of
    /// the block the decode makes, a search with the evidence laid over
    /// it in each round ([`Search`]). Never in the first pass of a
    /// brute-force search, and a decode is made without it first
    /// ([`Self::stages`]).
    pub message_evidence: bool,
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
            shared: 1,
            trial_sets: TRIAL_SETS,
            chase_bits: CHASE_BITS,
            list_decode: true,
            list_cost: LIST_COST,
            belief_propagation: true,
            abp_rounds: ABP_ROUNDS,
            derive_pins: false,
            message_evidence: true,
        }
    }
}

impl DecodeOptions {
    /// These options for a decode that is one of `tries` the caller
    /// makes of the grid, keeping the best, besides those already
    /// counted: the budgets of the dear solvers are shared out over the
    /// tries, down to their floors. The trials' and the curve's are held
    /// already shared over those counted before ([`Self::shared`]), and
    /// are shared out over these alone, so that a decode one of `a` and
    /// then of `b` has what one of `a·b` has.
    pub fn among(self, tries: usize) -> Self {
        Self { tries: self.tries.max(1) * tries.max(1), ..self.sharing(tries) }
    }

    /// These options with the dear solvers' budgets shared out over
    /// `searches` of each block, as [`Self::among`] shares them, but not
    /// counted among the decodes of the grid: a decode counts the
    /// searches it makes of each block itself ([`Search`]), and a read
    /// of a photo charges the read asked for nothing for those it makes
    /// after it ([`crate::imaging::read`]).
    pub fn sharing(self, searches: usize) -> Self {
        let searches = searches.max(1);
        Self {
            shared: self.shared.max(1) * searches,
            trial_sets: share(self.trial_sets, TRIAL_SETS_MIN, searches),
            list_cost: share(self.list_cost, LIST_COST_MIN, searches),
            ..self
        }
    }

    /// These options for a decode that is one of `tries` the caller
    /// makes of the grid, as [`Self::among`], but given the budgets
    /// whole: counted, for what confirms a solution, and not shared.
    pub fn one_of(self, tries: usize) -> Self {
        Self { tries: self.tries.max(1) * tries.max(1), ..self }
    }

    /// These options with the budget of every dear solver at its most:
    /// what a block tried harder is searched with ([`decode_trying_harder`]).
    /// The trials' and the curve's are shared out over the decodes the
    /// options are one of ([`Self::shared`]), as the chase's and the
    /// propagation's are where they are spent.
    pub fn at_most(self) -> Self {
        let shared = self.shared.max(1);
        Self {
            chase_bits: CHASE_MAX_BITS,
            abp_rounds: ABP_ROUNDS_MAX,
            list_cost: share(LIST_COST_MAX, LIST_COST_MIN, shared),
            trial_sets: share(TRIAL_SETS_MAX, TRIAL_SETS_MIN, shared),
            ..self
        }
    }

    /// The options a decode under these is made under in turn, each
    /// stage a decode of its own, until one is trustworthy
    /// ([`decode_trying_harder`]): without the flips, the trials of
    /// erasures and the message's evidence, then with the flips alone,
    /// the trials alone, and both, each first without the evidence and
    /// then with it, as far as these options switch them on, these
    /// options last. Each stage is the very decode its options make
    /// alone, charged as that is and not for the stages before it, as
    /// the second pass of a brute force is not for the first
    /// ([`brute_force`]); and the stages of options with any of the
    /// three switched off are stages of these, in the same order, so a
    /// grid that reads with one switched off reads with it on. The three
    /// once ran beside the rest in one decode: every other solution was
    /// charged for the decodes the flips and the trials made and the
    /// searches the evidence might make, and held against the codewords
    /// they reached, a solution of theirs that verified kept the
    /// solvers after them from running, and what they made of one block
    /// moved the message's evidence for the others, so a grid that read
    /// with one off could be lost with it on: a solution found before
    /// any evidence was laid, charged for the evidence's searches, once
    /// fell a bit short of verified. The flips go by reliability, and
    /// without it are no stage of their own.
    pub fn stages(self) -> Vec<Self> {
        let flips = self.reliability && self.flip_bits;
        let mut stages = Vec::new();
        for (flip_bits, erase_by_trial) in [(false, false), (true, false), (false, true), (true, true)] {
            if (flip_bits && !flips) || (erase_by_trial && !self.erase_by_trial) {
                continue;
            }
            for message_evidence in [false, true] {
                if message_evidence && !self.message_evidence {
                    continue;
                }
                stages.push(Self { flip_bits, erase_by_trial, message_evidence, ..self });
            }
        }
        stages
    }

    /// The first of [`Self::stages`], whatever these options switch on:
    /// without the flips, the trials of erasures and the message's
    /// evidence.
    pub fn first_stage(self) -> Self {
        Self { flip_bits: false, erase_by_trial: false, message_evidence: false, ..self }
    }

    /// Rounds the propagation runs here: the options' share of them
    /// ([`Self::shared`]), and never fewer than [`ABP_ROUNDS_MIN`] unless
    /// fewer were set.
    fn rounds(&self) -> usize {
        (self.abp_rounds / self.shared.max(1)).max(ABP_ROUNDS_MIN.min(self.abp_rounds))
    }
}

/// Each of `over` decodes' share of `budget`: never less than `floor`,
/// unless the budget is, and never more than the whole.
fn share(budget: usize, floor: usize, over: usize) -> usize {
    (budget / over.max(1)).clamp(floor.min(budget), budget)
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
    /// Decoded from the beliefs that settled after this many rounds of
    /// passing them between the bits along their parity checks
    /// ([`super::abp`]), which overturned `flipped_bits` of the bits as
    /// read.
    BeliefPropagation { rounds: usize, flipped_bits: usize },
    /// A codeword a solver reached and offered nothing for, verified by
    /// what the block itself makes of it, as any codeword is whatever
    /// found it ([`super::rs::judged`]).
    Reached,
    /// Everything except pinned codewords treated as erased.
    PinnedOnly,
}

/// The risk a solution must keep below for its block to count as
/// verified ([`BlockOutcome::risk_bits`]): wrong no oftener than once in
/// two to this many, everything that found it counted. A byte's worth,
/// which is the least the plain decode of a block read whole shows
/// within the standard's limit, and what every reader takes.
pub const VERIFIED_RISK_BITS: u32 = VERIFIED_BITS;

/// The solvers that may offer a block a solution in one search under
/// `opts`, the plain decode among them, and the decode of its pinned
/// codewords alone where `pinned`, a codeword of the block being pinned
/// whole: the plain decode, bit by bit, the flips and frees of the bits
/// read, the doubting of codewords and of bits, the chase, the
/// propagation, the curve and the trials of erasures, each as the
/// options let it run. Each is a fresh chance for a block of noise to
/// pass, and the search keeps whichever ranks best, so a solution any
/// but the plain decode found is one of them all ([`Search::share`]),
/// besides the tries of its own solver. Those that go by reliability,
/// the doubting, the flips, the chase and the propagation, run only
/// where the block is `faint` with a bit each of them may doubt, flip or
/// move ([`Faint`]): on a block painted by hand they find nothing, and
/// were once counted all the same.
pub fn routes(opts: &DecodeOptions, pinned: bool, faint: Faint) -> usize {
    let by = |faint: bool| opts.reliability && faint;
    1 + usize::from(opts.bit_level)
        + usize::from(opts.bit_level && ((by(faint.bits) && opts.flip_bits) || opts.erase_by_trial))
        + usize::from(by(faint.unpinned_whole) && opts.doubt_codewords)
        + usize::from(by(faint.bits) && opts.doubt_bits && opts.bit_level)
        + usize::from(by(faint.read_whole) && opts.flip_bits)
        + usize::from(by(faint.bits) && opts.belief_propagation)
        + usize::from(opts.list_decode && opts.list_cost > 0)
        + usize::from(opts.erase_by_trial)
        + usize::from(pinned)
}

/// What a block holds for the solvers that go by reliability to doubt,
/// flip or move ([`routes`]): a bit read as known, and not pinned, below
/// [`CERTAIN`], which is all any of them takes, and where it lies, since
/// some take only the codewords read in full. A solver with nothing
/// here finds nothing. Every such bit was once counted for every one of
/// them, those of codewords read in part among them, which the chase and
/// the doubting of codewords never look at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Faint {
    /// Such a bit anywhere in the block: what the doubting of bits, the
    /// flips of solid bits and the propagation take.
    pub bits: bool,
    /// Such a bit in a codeword read in full: what the chase takes.
    pub read_whole: bool,
    /// Such a bit in a codeword read in full that holds no pin: what the
    /// doubting of codewords takes.
    pub unpinned_whole: bool,
}

impl Faint {
    /// What `block` holds, its bits as reliable as `bit_reliability` says.
    fn of(block: &[CwRead], bit_reliability: &[[u8; 8]]) -> Self {
        let mut faint = Self::sampled(false);
        for (cw, bits) in block.iter().zip(bit_reliability) {
            let any = (0..8).any(|b| (cw.known & !cw.pinned) >> b & 1 != 0 && bits[b] < CERTAIN);
            faint.bits |= any;
            faint.read_whole |= any && !cw.is_erasure();
            faint.unpinned_whole |= any && !cw.is_erasure() && cw.pinned == 0;
        }
        faint
    }

    /// Something for every one of the solvers where `sampled`, as a block
    /// read from a photo may hold, and nothing for any of them on a block
    /// painted by hand.
    pub fn sampled(sampled: bool) -> Self {
        Self { bits: sampled, read_whole: sampled, unpinned_whole: sampled }
    }
}

/// Searches a decode may make of a block one after another, at the
/// most: the block as read, and in each round of evidence again with
/// the reliabilities the blocks that verified lowered and again with the
/// message's evidence laid over it ([`EVIDENCE_ROUNDS`]), whichever the
/// options and the grid allow. Each is given its share of the chances by
/// its place among these ([`Search::share`]).
const SEARCHES_MAX: usize = 1 + 2 * EVIDENCE_ROUNDS;

/// The ways each search of a block of a symbol of `version` is made at
/// once: as it is, and again with the final codeword's missing nibble
/// filled with ones, where the version has a codeword of four bits
/// ([`solve_spec`]).
fn ways(version: SymbolVersion) -> usize {
    if tables::has_half_codeword(version) { 2 } else { 1 }
}

/// Confirming bits a solution any but the plain decode found must show,
/// as its solver counts them, for its block to be verified, in a decode
/// of a symbol of `version` under `opts` that searches no block harder,
/// found in the search of the block as read: [`VERIFIED_RISK_BITS`], and
/// what it is charged for the decodes of the grid it is one of and for
/// the search's share of the chances and the solvers of it
/// ([`Search::share`]), those that go by reliability only where it was
/// `sampled`. For saying how far a solver reaches.
pub fn verifying_bits(version: SymbolVersion, opts: &DecodeOptions, sampled: bool) -> u32 {
    let search = Search { as_read: true, nth: 1, ways: ways(version), picked: 1 };
    VERIFIED_RISK_BITS + whole_charge(tries_bits(opts.tries) + search.share(opts, false, Faint::sampled(sampled)))
}

/// Confirming bits the block itself must show of a codeword, by how
/// rarely noise has any codeword so near ([`super::rs::judged`]), for the
/// block to be verified in a decode under `opts`, found in the search of
/// the block as read: [`VERIFIED_RISK_BITS`], a bit for the choice
/// between that count and the soft count, and what it is charged for the
/// decodes of the grid, and nothing for the solver that reached the
/// codeword or its tries ([`Candidate::weigh`]). A codeword a solver
/// reaches is verified so where its solver's own count, charged for its
/// tries and the search's solvers ([`verifying_bits`]), falls short.
/// For saying how far a solver reaches.
pub fn judged_verifying_bits(opts: &DecodeOptions) -> u32 {
    VERIFIED_RISK_BITS + 1 + whole_charge(tries_bits(opts.tries))
}

/// What a pick of `tries` is charged, in bits worked out to fractions of
/// one: the log of them, which [`tries_charge`] takes to the bit above.
fn tries_bits(tries: usize) -> f64 {
    (tries.max(1) as f64).log2()
}

/// A charge worked out in fractions of a bit, to the bit above, a hair
/// allowed for the rounding of the logarithms it was summed from. The
/// charges a solution owes, for the tries of its solver, the decodes of
/// the grid it is one of and its search's share of the chances, were
/// once each taken to the bit above and then added, which lost as much
/// as a bit for each of them after the first.
fn whole_charge(bits: f64) -> u32 {
    (bits - 1e-9).ceil().max(0.0) as u32
}

/// Where a search of a block stands among those a decode makes of it.
///
/// A block is searched by many solvers, and searched again in the rounds
/// of evidence, with the missing nibble of a four-bit codeword filled,
/// and harder, and the best ranked solution found is kept: each solver
/// of each search is a fresh chance for a block of noise to pass, and a
/// solution is charged for them all, a bit for each doubling of them,
/// as a solver charges for its own tries ([`tries_charge`]). All but
/// one: the plain decode of the block as read, codewords filled in or
/// none, the decode the standard describes and every reader makes,
/// stands on its own count and the decodes of the grid it is one of, so
/// that a block read with a codeword to spare is verified as every
/// reader has it. Its share of the chances is the whole, and every
/// other's a share of as much again.
///
/// Two counts judge a codeword by the block itself, whatever found it:
/// how rarely a block of noise has any codeword so few codewords from it,
/// the known bits of those it could not read agreeing with it as far, and
/// how rarely any so near in bits weighed by their reliability
/// ([`super::rs::judged`]). Each bounds the chance that some codeword of
/// the code, any of them, lies so near noise, and so asks nothing of how
/// many tries a solver made or how many solvers looked: a thousand
/// patterns flipped reach no codeword that was not there to be reached.
/// They are charged for the decodes of the grid, the ways the pins were
/// picked among, and a bit for the choice between them, and for the
/// search they judge by its share of as much again once more, by its
/// place among the searches and the ways it is made ([`Charges`]): the
/// lowered reliabilities and the evidence laid weigh the block otherwise,
/// and the nibble filled is another block. In the search of the block as
/// read they stand on the plain decode's whole share, its own count
/// being the first of them at the one codeword it reaches. So what the
/// risks claim holds to within a factor of three, a bit and a half.
/// Those counts were once charged for the tries of the solver that found
/// the codeword and for the solvers of its search as a solver's own
/// count is, ten bits and more for one the chase found, and a codeword
/// the block itself verified stood unverified for the patterns flipped
/// to reach it.
///
/// The searches are made one after another, each where those before it
/// left the block unverified ([`Stage::solve`]), and a solution is
/// charged for its own search's share of the chances, which falls as
/// the searches go on and is the same however many follow it
/// ([`nth_of`]): the block as read has the most, and each round's
/// lowered reliabilities and then its evidence less, by their place
/// among the most a decode may make ([`SEARCHES_MAX`]), whether or not
/// the options make them. A search not as read was once charged for
/// every search the decode might make, and a solution found in the
/// block as read, before any evidence was laid, was charged for the
/// rounds of evidence that might follow it: with the message's
/// evidence switched on it stood a bit short of verified where with it
/// off it stood verified. Charged for the searches up to its own alone,
/// each would be given one in its place, and the five together the
/// chances of two searches and more.
///
/// Every solution but that one was once charged sixteen bits instead,
/// whatever found it, for tries no solver could count: two codewords'
/// worth, set by trial, after one read in four hundred took a version
/// 5-Q symbol with seventeen of a block's eighteen error correction
/// codewords unknown for its message with a block's worth of bytes it
/// never held. That read had one codeword to spare, which noise passes
/// once in 256 decodes, and it was one of four hundred decodes of the
/// photo, which are counted now ([`DecodeOptions::tries`]); and the
/// sixteen bits stood over counts that were wrong by more, the curve's
/// and the propagation's among them, which once claimed blocks of
/// noise by a hundred bits ([`super::rs::list_solution_balance`]).
#[derive(Clone, Copy, Debug)]
struct Search {
    /// The block as read, with the reliabilities as read and nothing
    /// laid over it or filled in: its plain decode is the standard's.
    as_read: bool,
    /// Which of the searches a decode may make of the block one after
    /// another this is, from one: the block as read, then in each round
    /// its reliabilities lowered and then its evidence laid
    /// ([`SEARCHES_MAX`]).
    nth: usize,
    /// The ways the search is made at once, each a fresh chance charged
    /// alike: twice where the final codeword's missing nibble is filled
    /// with ones as well ([`ways`]), and twice again where the block is
    /// searched harder, now or before ([`Harder`]).
    ways: usize,
    /// The ways the pins the search reaches were picked among
    /// ([`PinsPicked`]), which the decodes of the grid count in besides
    /// ([`DecodeOptions::tries`]): what the decode of the block's pinned
    /// codewords alone is charged.
    picked: usize,
}

impl Search {
    /// What a solution any but the plain decode of the block as read
    /// found is charged, in bits worked out to fractions of one, for the
    /// search's share of the chances ([`nth_of`]), the ways it is made,
    /// and the solvers of it ([`routes`]), the plain decode's own among
    /// them: what its solver's own count owes.
    fn share(&self, opts: &DecodeOptions, pinned: bool, faint: Faint) -> f64 {
        (nth_of(self.nth, SEARCHES_MAX) * (self.ways * routes(opts, pinned, faint)) as f64).log2()
    }

    /// What the counts that judge a codeword by the block itself owe the
    /// search, in bits worked out to fractions of one: nothing for the
    /// block as read, whose plain decode they stand beside on its whole
    /// share, and for any other search its share of the chances by its
    /// place and the ways it is made, whatever its solvers.
    fn judged_share(&self) -> f64 {
        if self.as_read { 0.0 } else { (nth_of(self.nth, SEARCHES_MAX) * self.ways as f64).log2() }
    }

    /// What the solutions of this search are charged, under `opts`.
    fn charges(&self, opts: &DecodeOptions, pinned: bool, faint: Faint) -> Charges {
        Charges {
            grid: tries_bits(opts.tries),
            pins: (self.picked.max(1) as f64).log2(),
            share: self.share(opts, pinned, faint),
            judged: self.judged_share(),
            as_read: self.as_read,
        }
    }

    /// The same search of the block with something laid over it or
    /// filled in: not as read.
    fn again(self) -> Self {
        Self { as_read: false, ..self }
    }
}

/// What the solutions of one search of a block are charged, in bits
/// worked out to fractions of one and taken to the bit above once they
/// are summed ([`whole_charge`]).
#[derive(Clone, Copy, Debug)]
struct Charges {
    /// The decodes of the grid the search is one of, the ways the pins
    /// were picked among counted in ([`DecodeOptions::tries`]).
    grid: f64,
    /// The ways the pins were picked among alone ([`Search::picked`]),
    /// for the decode of the block's pinned codewords alone.
    pins: f64,
    /// The search's share of the chances among its solvers, for a
    /// solver's own count ([`Search::share`]).
    share: f64,
    /// The search's share of the chances for the counts that judge a
    /// codeword by the block itself ([`Search::judged_share`]).
    judged: f64,
    /// The search is of the block as read ([`Search::as_read`]).
    as_read: bool,
}

/// The ways the pins a decode is made under were picked among
/// ([`Pin::among`]), as the searches they reach are charged for them.
/// A block's search as read reaches the pins it holds, and is charged
/// for the most ways any of them was picked among: a fresh chance under
/// each, and the same search whichever way pins elsewhere were picked.
/// Every other search, with what the rest of the message makes of the
/// block laid over it or with reliabilities the other blocks lowered,
/// moves with every pin, and is charged for the most of all. The ways
/// were once one count for all the pins in force, and a pick of pins in
/// one block was charged to every block holding a pin picked before it.
struct PinsPicked {
    block: Vec<usize>,
    all: usize,
}

impl PinsPicked {
    fn of(layout: &SymbolLayout, pins: &Pins) -> Self {
        let mut picked = Self { block: vec![1; layout.blocks.len()], all: 1 };
        for (&cw, pin) in pins {
            if cw >= layout.codewords.len() || pin.mask & codeword_bit_mask(layout, cw) == 0 {
                continue;
            }
            let b = layout.codewords[cw].block as usize;
            picked.block[b] = picked.block[b].max(pin.among);
            picked.all = picked.all.max(pin.among);
        }
        picked
    }
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
        /// Independent bits confirming the solution as its solver counts
        /// them: the redundancy left over, the known bits of erased
        /// codewords it reproduced, and what the errors it located
        /// confirm ([`super::rs::located_error_bits`]). Zero means any
        /// input would have produced a "solution", as one does that only
        /// fills unknown codewords and has nothing left over.
        confidence_bits: u32,
        /// How rarely the solution is wrong, as bits: no oftener than
        /// once in two to this many, by everything that speaks for it
        /// weighed on one footing ([`solve_block`]). What
        /// `confidence_bits` counts, or the soft count of
        /// [`super::rs::soft_count_bits`] where that counts more, less
        /// the charge for the solvers and searches of the block it was
        /// one of, for any but the plain decode of the block as read
        /// ([`Search`]); and with that chance, the chance of every other
        /// codeword a solver reached, in any search of the block, offered
        /// or not, being the block's: the margin, in bits of likelihood,
        /// by which the solution lies nearer the block as read than it,
        /// since of two codewords within reach the nearer is the likelier
        /// by just that much, and both may be wrong ([`held_against`]).
        /// The one number verification goes by ([`VERIFIED_RISK_BITS`]).
        risk_bits: u32,
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

    /// Bits of risk, zero for a failure.
    pub fn risk_bits(&self) -> u32 {
        match *self {
            BlockOutcome::Solved { risk_bits, .. } => risk_bits,
            BlockOutcome::Failed(_) => 0,
        }
    }

    /// Solved with its risk below [`VERIFIED_RISK_BITS`], within the
    /// standard's limit.
    pub fn is_verified(&self) -> bool {
        matches!(*self, BlockOutcome::Solved { risk_bits, beyond_standard: false, .. } if risk_bits >= VERIFIED_RISK_BITS)
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
    /// the solution was reached so ([`Clustering::lowered`]). Nought
    /// where it was not.
    pub near_errors: usize,
    /// Bits of the block the message's evidence moved for the search,
    /// when the solution was reached so ([`DecodeOptions::message_evidence`]).
    /// Nought where it was not.
    pub message_evidence: usize,
    /// The block, as it reads now, was searched with every dear solver's
    /// budget at its most ([`decode_trying_harder`]), whether or not that
    /// found anything better than its own options did.
    pub tried_harder: bool,
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
/// How attempts at a grid rank against each other: field by field, the
/// first that differs deciding. An attempt that is trustworthy, with
/// every block confirmed and a stream that parses, outranks all else.
/// Below that the stream comes before the blocks: of two attempts at one
/// grid, under two formats or as the two QR Code models, or of two
/// refits of one read, the one whose stream parses, through bits no
/// block confirmed, is nearer the symbol than one whose every block
/// "verified" while its stream reads as noise. A wrong placement manages
/// the latter now and then, where a block spends most of its redundancy
/// on what it erased and is confirmed by the little left, which noise
/// passes once in so many blocks, as often as its risk says; BoofCV
/// takes a symbol as read only when its error correction succeeds and
/// its message parses, and the order here does the same.
///
/// Reads of a photo, which may be of other placements and other
/// versions, are ranked otherwise ([`crate::imaging::read::ReadResult`]):
/// first by whether a block of them is confirmed, those with none by the
/// fit of their fixed patterns, and only then by this. Between them a
/// stream that parses through bits no block confirmed shows nothing that
/// noise does not show as often, since a version laid over a symbol of
/// another reads one so, and a block confirmed comes first, false
/// confirmations and all.
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

/// The risk, in bits, that one of several solutions each wrong no
/// oftener than once in two to its `risks` is wrong: their chances
/// added, since nothing says how far they are wrong together. Summed
/// from the likeliest down, so that solutions a thousand bits sure add
/// what they add rather than nothing, and taken to a hundredth of a bit
/// before it is rounded down, so that a chance sixteen orders below the
/// rest does not cost the whole a bit: each risk is itself rounded down.
/// None when a solution is missing, nought when the chances come to one
/// or more, and [`u32::MAX`] when there are no solutions to be wrong.
pub fn combined_risk_bits(risks: impl IntoIterator<Item = Option<u32>>) -> Option<u32> {
    let risks: Vec<u32> = risks.into_iter().collect::<Option<_>>()?;
    let Some(&least) = risks.iter().min() else { return Some(u32::MAX) };
    let more: f64 = risks.iter().map(|&bits| (-f64::from(bits - least)).exp2()).sum();
    Some(whole_risk_bits(f64::from(least) - more.log2()))
}

/// A risk worked out in fractions of a bit, in whole bits: taken to a
/// hundredth of a bit and rounded down ([`combined_risk_bits`]), and
/// nought where the chance of being wrong comes to one or more.
fn whole_risk_bits(bits: f64) -> u32 {
    if bits <= 0.0 { 0 } else { (bits + 0.01).floor() as u32 }
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
    /// How the photo's misreads cluster, as the blocks that verified
    /// showed it, when that was learnt to solve again the blocks that
    /// had not ([`Clustering::lowered`]).
    pub clustering: Option<Clustering>,
    /// The decodes of the grid this was one of, which its risks are
    /// charged for ([`DecodeOptions::tries`]): for a caller that takes
    /// the grid this decode was made of as a pick of so many, to charge
    /// its own decodes of it the same.
    pub tries: usize,
    /// The most ways any pin this decode was made under was picked among
    /// ([`Pin::among`]): what the searches drawing on the whole message
    /// were charged for, a block's search as read being charged for the
    /// pins it holds alone ([`PinsPicked`]). One where no pin was picked.
    pub pin_tries: usize,
}

impl DecodeReport {
    /// How rarely the message is wrong, in bits: no oftener than once in
    /// two to this many, when every block solved. A block's risk is how
    /// rarely its own solution is wrong ([`BlockOutcome::risk_bits`]),
    /// and the message is wrong when any one of them is, so their
    /// chances add: two blocks each at one in 2^8 make the message one
    /// in 2^7, and the eighty-one blocks of a version 40-H symbol each
    /// at one in 2^8 make it worse than one in four. None while a block
    /// is unsolved, and nought when the chances come to one or more.
    pub fn message_risk_bits(&self) -> Option<u32> {
        combined_risk_bits(self.blocks.iter().map(|b| b.is_solved().then(|| b.outcome.risk_bits())))
    }

    /// Every block verified, each by itself, and together they leave the
    /// message wrong oftener than a block may be ([`VERIFIED_RISK_BITS`]):
    /// a read to be warned of, though no block of it fails its own test.
    pub fn message_risk_falls_short(&self) -> bool {
        self.blocks.iter().all(BlockReport::is_verified) && self.message_risk_bits().is_some_and(|bits| bits < VERIFIED_RISK_BITS)
    }

    /// The message is empty and every bit after its end is nought: the
    /// data is the zero codeword, which in the grid is the mask's own
    /// pattern. An encoder may make such a symbol, an empty message
    /// padded with zeros, but a texture that runs as a mask does reads
    /// the same, every block of it confirmed as surely as a symbol's:
    /// on a photo of a monitor, whose rows of pixels read through mask 1
    /// as its horizontal stripes do, fifteen placements on the screen
    /// read so, at 1 in 2^9 to 2^89. A read to be warned of, as
    /// nothing in the decode can tell the two apart.
    pub fn data_is_the_mask(&self) -> bool {
        matches!(self.parse.end, EndOfData::At { .. })
            && self.parse.segments.is_empty()
            && self.tail.as_ref().is_some_and(|tail| tail.zero_filled)
    }

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
    /// What its solver counts for it before any charge
    /// ([`RsSolution::uncharged_bits`], [`BitDoubt::uncharged_bits`]).
    own: u32,
    /// What its solver's tries cost, in bits worked out to fractions of
    /// one where the solver counts them so ([`RsSolution::charge`],
    /// [`BitDoubt::tries`]): what its own count and the soft count, taken
    /// as its solver's, are charged besides the decodes of the grid and
    /// the search's share of the chances ([`Candidate::weigh`]). The soft
    /// count is charged the same, and a bit for the choice between the
    /// two counts, so that the two stand on one footing and neither is
    /// the lenient one. The soft count was once charged for the solver's
    /// decodes alone, and under a search of every format a solution the
    /// fallbacks found stood on it as verified where the hard count,
    /// charged for the search, fell short of it.
    tried: f64,
    /// Bits of risk on its own ([`Candidate::weigh`]), before any other
    /// codeword found is held against it.
    alone: u32,
    /// Bits of risk it may stand on once the message has told it from
    /// its rivals ([`arbitrate`]): `alone`, or less where other
    /// codewords found, not among the rivals, hold it down still: one as
    /// near the block as read as they are but left out for their number,
    /// one further from the nearest than the tie allows and yet near
    /// this one, or one a solver reached and did not offer, their
    /// chances added ([`held_against`]). Set where it is offered as a
    /// rival ([`rivals_among`]).
    held: u32,
    /// How far the block as read lies from it ([`soft_distance`]), once
    /// its solve has gathered the codewords found ([`distinct_codewords`]):
    /// what it is held against the others by.
    distance: u32,
}

impl Candidate {
    fn new(outcome: BlockOutcome, corrected: Vec<u8>, doubted: Vec<u8>, own: u32, tried: f64) -> Self {
        Self { outcome, corrected, doubted, own, tried, alone: 0, held: 0, distance: 0 }
    }

    /// The candidate, found by searching the block with the message's
    /// evidence laid over it, confirmed against `block` instead, the
    /// block as read with its unread bits at their lean ([`Laid::judged`],
    /// [`judged`](super::rs::judged)): its errors, erasures and slack,
    /// and what confirms it, are what that block makes of its codeword,
    /// charged for its tries as found and for the decodes of the grid
    /// ([`Charges::grid`]); and it carries its tries, so that the soft
    /// count is charged the same when it is weighed ([`Candidate::weigh`]).
    /// The search with the
    /// evidence is one of the searches of the block the decode counts
    /// ([`Search`]); it was once charged a bit for that here, and a bit
    /// again as one of two decodes of the grid. None when that block
    /// refuses it, or confirms too little of it to offer: as found, or as
    /// it judges any codeword whatever found it, charged for the decodes
    /// of the grid and the search's share of such counts and not for the
    /// solver's tries ([`Candidate::weigh`]). Held to the count as found
    /// alone, a codeword the block verified was kept back, and came back
    /// only as one a solver reached and offered nothing for, the bits its
    /// solver doubted lost ([`promote_reached`]).
    fn restated(self, block: &[CwRead], bit_reliability: &[[u8; 8]], ec_len: usize, reserved: usize, charges: Charges) -> Option<Self> {
        let BlockOutcome::Solved { method, .. } = self.outcome else { return Some(self) };
        let unknown: usize = block.iter().map(|c| c.known.count_zeros() as usize).sum();
        let distance = soft_distance(block, bit_reliability, &self.corrected);
        let soft = soft_count_bits(block, bit_reliability, (8 * ec_len).saturating_sub(unknown), distance);
        let solution = judged_count(block, ec_len, reserved, self.corrected, whole_charge(self.tried + charges.grid), soft)?;
        if solution.confidence_bits() < VERIFIED_BITS
            && judged_by(block, ec_len, reserved, solution.corrected.clone(), whole_charge(charges.grid + charges.judged), soft).is_none()
        {
            return None;
        }
        let outcome = BlockOutcome::Solved {
            method,
            errors: solution.errors,
            erasures: solution.erasures,
            slack: solution.slack,
            confidence_bits: solution.confidence_bits(),
            risk_bits: 0,
            beyond_standard: !solution.within_standard(),
        };
        let own = solution.uncharged_bits();
        Some(Self { outcome, corrected: solution.corrected, ..Self::new(outcome, Vec::new(), self.doubted, own, self.tried) })
    }

    /// The candidate weighed on its own, against `block` as the search
    /// it was found in is charged (`charges`), its risk the more of two.
    ///
    /// As its solver found it: what its solver counted for it
    /// ([`Candidate::own`]), or the soft count ([`soft_count_bits`]) for
    /// the bits it overturns weighed by their reliability with a bit for
    /// the choice between the two, charged for its solver's tries
    /// ([`Candidate::tried`]), the decodes of the grid, and the solvers
    /// and searches of the block it is one of ([`Search::share`]), unless
    /// it is the plain decode of the block as read, which a candidate
    /// found with the message's evidence laid over the block never is
    /// ([`Search::again`]). The charges are summed in fractions of a bit
    /// and taken to the bit above once.
    ///
    /// And as the block itself judges it, whatever found it
    /// ([`judged`](super::rs::judged)): the codewords' count or the soft
    /// count, charged for the decodes of
    /// the grid, the search's share of the chances for such counts
    /// ([`Search::judged_share`]) and a bit for the choice. Where that
    /// comes to the more, the solution stands on it. Whether it is within
    /// the standard's limit is the same either way, what the codewords it
    /// changes of those read and the modules unread leave of the reserve
    /// ([`keeps_reserve`]), whatever found it. That is counted against
    /// `read`, the block as read as the reserve counts it ([`heard`]),
    /// which is `block` but where the message's evidence was laid over
    /// it: a bit `block` takes at its lean ([`Laid::judged`]) is an
    /// assumption, not a reading, and is unread there, as it was. Counted
    /// as read, it cost nothing where the solution bore the lean out, and
    /// a 3-L block six bytes unread and five codewords wrong, past what
    /// the standard corrects, kept the reserve when its bytes' high bits
    /// leaned toward text.
    fn weigh(mut self, block: &[CwRead], read: &[CwRead], bit_reliability: &[[u8; 8]], ec_len: usize, reserved: usize, charges: Charges) -> Self {
        let BlockOutcome::Solved { method, .. } = self.outcome else { return self };
        let unknown: usize = block.iter().map(|c| c.known.count_zeros() as usize).sum();
        let distance = soft_distance(block, bit_reliability, &self.corrected);
        let soft = soft_count_bits(block, bit_reliability, (8 * ec_len).saturating_sub(unknown), distance);
        let plain = matches!(method, SolveMethod::ReedSolomon) && charges.as_read;
        // The decode of the pinned codewords alone is no decode of the
        // grid as read: its own count is charged for the ways its pins
        // were picked among and not for the grid's decodes, which change
        // nothing of it. It was once charged for neither, the two being
        // counted together ([`Charges::grid`]). Its soft count is of the
        // block as read, which each decode of the grid reads afresh, and
        // is charged for them as every other's is: charged for the pins
        // alone, it stood two bits and more above the same count as the
        // block judges it, and the lighter charge was the one kept.
        let solver_grid = if matches!(method, SolveMethod::PinnedOnly) { charges.pins } else { charges.grid };
        let share = if plain { 0.0 } else { charges.share };
        // A codeword a solver reached and did not offer has no count of
        // its solver's to stand on.
        let own = if matches!(method, SolveMethod::Reached) {
            0
        } else {
            let own = self.own.saturating_sub(whole_charge(self.tried + solver_grid + share));
            own.max(soft.saturating_sub(whole_charge(self.tried + charges.grid + 1.0 + share)))
        };
        let charge = whole_charge(charges.grid + charges.judged);
        // However little the block's own count comes to: short of
        // verified, the risk is still the more of the two counts.
        let judged = judged_count(block, ec_len, reserved, self.corrected.clone(), charge, soft).map_or(0, |solution| solution.confidence_bits());
        self.alone = own.max(judged);
        self.set_risk(self.alone);
        // The decode of the pinned codewords alone keeps the reserve as
        // its own block of pins has it, every codeword not pinned unread.
        if !matches!(method, SolveMethod::PinnedOnly)
            && let BlockOutcome::Solved { beyond_standard, .. } = &mut self.outcome
        {
            *beyond_standard = !keeps_reserve(read, ec_len, reserved, &self.corrected);
        }
        self
    }

    fn set_risk(&mut self, risk_bits: u32) {
        if let BlockOutcome::Solved { risk_bits: risk, .. } = &mut self.outcome {
            *risk = risk_bits;
        }
    }

    /// The candidate with its risk set.
    fn at_risk(mut self, risk_bits: u32) -> Self {
        self.set_risk(risk_bits);
        self
    }

    /// Verified on its own: within the standard, with its risk alone
    /// below the threshold.
    fn verified_alone(&self) -> bool {
        matches!(self.outcome, BlockOutcome::Solved { beyond_standard: false, .. }) && self.alone >= VERIFIED_RISK_BITS
    }

    fn rank(&self) -> (bool, u32, u32) {
        solve_rank(&self.outcome)
    }

    /// Solved only by spending the codewords the standard reserves.
    fn is_beyond_standard(&self) -> bool {
        matches!(self.outcome, BlockOutcome::Solved { beyond_standard: true, .. })
    }

    /// The decode of the block from its pins alone, keeping the reserve as
    /// its pins count it ([`Candidate::weigh`]).
    fn kept_by_its_pins(&self) -> bool {
        matches!(self.outcome, BlockOutcome::Solved { method: SolveMethod::PinnedOnly, beyond_standard: false, .. })
    }

    /// The candidate, a solution of the same codeword as `other`, at the
    /// less risk of the two on their own ([`Candidate::alone`]): what
    /// `other` confirms of the codeword added to it, as it stands
    /// ([`distinct_codewords`]).
    fn confirmed_by(mut self, other: &Candidate) -> Self {
        debug_assert_eq!(self.corrected, other.corrected);
        if other.alone > self.alone {
            self.alone = other.alone;
            self.set_risk(self.alone);
        }
        self
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
    /// Codewords the solvers found that the block as read lies as near
    /// to as each other, when nothing verified the block: for the
    /// message to tell apart ([`arbitrate`]), best ranked first.
    rivals: Vec<Candidate>,
    /// The solution was told from this many rivals by the message.
    arbitrated_among: usize,
    /// Modules of the block whose reliability was lowered for errors
    /// located beside them, when the solution was reached so.
    near_errors: usize,
    /// Bits of the block the message's evidence moved for the search,
    /// when the solution was reached so.
    message_evidence: usize,
    /// See [`BlockReport::half_padded_with_ones`].
    half_padded_with_ones: bool,
    /// Every codeword of the code the searches behind this solve
    /// reached, offered or not ([`Sightings`]), its own and its rivals'
    /// among them: what another search of the block is held against
    /// ([`hold_together`]).
    reached: Sightings,
    /// The block the solve was judged against and its bits' reliability
    /// ([`solve_block`]): what the codewords another search reached are
    /// held against it by.
    frame: (Vec<CwRead>, Vec<[u8; 8]>),
}

impl BlockSolve {
    fn solved(candidate: Candidate) -> Self {
        let inferred = vec![(0, 0); candidate.corrected.len()];
        let mut reached = Sightings::default();
        reached.see(&candidate.corrected);
        Self {
            outcome: candidate.outcome,
            reached,
            corrected: Some(candidate.corrected),
            inferred,
            doubted: candidate.doubted,
            rivals: Vec::new(),
            arbitrated_among: 0,
            near_errors: 0,
            message_evidence: 0,
            half_padded_with_ones: false,
            frame: (Vec::new(), Vec::new()),
        }
    }

    /// The solve, judged against `block` with `bit_reliability`, its
    /// searches having reached `reached` besides its own codewords.
    fn framed(mut self, block: &[CwRead], bit_reliability: &[[u8; 8]], reached: &Sightings) -> Self {
        self.frame = (block.to_vec(), bit_reliability.to_vec());
        for rival in &self.rivals {
            self.reached.see(&rival.corrected);
        }
        self.reached.see_all(reached);
        self
    }

    fn rank(&self) -> (bool, u32, u32) {
        solve_rank(&self.outcome)
    }

    /// The solve held against the codewords of `reached` its own searches
    /// did not reach, as another search of its block reached them: its
    /// solution's risk, and each rival's once the message has told it
    /// from the rest, held against them as against any codeword found
    /// ([`held_against`]), by how far its block lies from each. Whether
    /// its outcome changed.
    fn hold_against(&mut self, reached: &Sightings) -> bool {
        let new: Vec<&[u8]> = reached.codewords().filter(|codeword| !self.reached.saw(codeword)).collect();
        if new.is_empty() {
            return false;
        }
        let (block, bits) = &self.frame;
        // With no block to judge them by, every codeword would lie nought
        // from it, and hold the solve to no risk at all ([`Self::framed`]).
        debug_assert!(!block.is_empty(), "a solve held against codewords with no block to judge them by");
        let distances: Vec<u32> = new.iter().map(|codeword| soft_distance(block, bits, codeword)).collect();
        let before = self.outcome;
        if let (Some(corrected), BlockOutcome::Solved { risk_bits, .. }) = (&self.corrected, &mut self.outcome) {
            *risk_bits = held_against(*risk_bits, soft_distance(block, bits, corrected), distances.iter().copied());
        }
        for rival in &mut self.rivals {
            rival.held = held_against(rival.held, rival.distance, distances.iter().copied());
        }
        for codeword in new {
            self.reached.see(codeword);
        }
        self.outcome != before
    }
}

/// Two solves of one block, by searches of it made otherwise, each held
/// against the codewords the other's searches reached that its own did
/// not ([`BlockSolve::hold_against`]), and each then holding all of
/// them: a codeword is no likelier for having been offered in one
/// search than another, and one search's solution was once kept over
/// another's, or taken in its place, with nothing said of the codewords
/// the other had reached. Whether each outcome changed. Once `a` holds
/// all of them, what it reached that `b` did not is what it reached
/// before.
fn hold_together(a: &mut BlockSolve, b: &mut BlockSolve) -> (bool, bool) {
    let a_changed = a.hold_against(&b.reached);
    (a_changed, b.hold_against(&a.reached))
}

/// How block solutions rank: verified solutions within the standard
/// first, then by risk, then by what their solver counted. A failure
/// ranks below every solution.
fn solve_rank(outcome: &BlockOutcome) -> (bool, u32, u32) {
    (outcome.is_verified(), outcome.risk_bits(), outcome.confidence_bits())
}

/// Rivals a block keeps for the message to tell apart, at the most: the
/// best ranked. Every one costs a parse of the stream.
const MAX_RIVALS: usize = 8;

/// Solve one block, trying the cheap exact decoder first and then every
/// fallback, and keeping the solution of least risk. `reserved` EC
/// codewords are the standard's misdecode protection: a solution that
/// spends them is reported, but marked as beyond the standard, and ranks
/// below any verified solution within it.
///
/// Every solution found is weighed on one footing
/// ([`BlockOutcome::risk_bits`]): what its solver counted for it, or
/// the soft count for the bits it overturns, charged for the tries, and
/// for anything but the plain decode of the block as read, for the
/// solvers and searches of the block it is one of (`search_of`,
/// [`Search`]). An exact solution ends the search only when it is
/// verified so, within the standard, as one that corrected to the limit
/// of the block's error correction is by the errors it located. One
/// with too little to confirm it, which is nothing where it only filled
/// unknown codewords and had none to spare, or one that spent the
/// reserved codewords, is kept while the fallbacks look for better:
/// doubting the faint bit of the one wrong codeword, or flipping it,
/// turns a solution that fills fifteen unknown codewords and locates
/// that one with a codeword to spare into one with three, less what the
/// patterns of flips tried are charged, which is nearer verified if not
/// there.
///
/// The solutions found are then held against each other by the block
/// as read, and against every codeword a solver reached and did not
/// offer ([`Sightings`]): of two codewords the solvers reach, the nearer
/// in bits weighed by their reliability is the likelier by just the
/// margin between them, and each other codeword's chance of being the
/// block's, by its margin, is added to a solution's risk
/// ([`held_against`]). Two within [`DISTANCE_MARGIN`] of each other the block
/// as read cannot tell apart; when nothing verifies the block they are
/// kept beside the solution as rivals, for the message to tell apart
/// ([`arbitrate`]): a block that would otherwise have been given up
/// over a tie.
///
/// The solvers search `block`, with `bit_reliability`, or, when
/// `searched` is given, the block with the message's evidence laid over
/// it ([`MessageEvidence::laid`]). Either way every solution is judged by
/// `block`, which is then the block as read with its unread bits at their
/// lean ([`Laid::judged`]): one found in the evidence is restated against
/// it ([`Candidate::restated`]), and what the evidence did to the bits
/// that were read counts for nothing in what confirms it. Whether a
/// solution keeps the codewords the standard reserves is counted against
/// `read`, the block as read as the reserve counts it ([`heard`]), its
/// unread bits unread, which is `block` itself where nothing is laid over
/// it and no bit was read at a coin's toss ([`Candidate::weigh`]).
#[allow(clippy::too_many_arguments)]
fn solve_block(
    block: &[CwRead],
    read: &[CwRead],
    bit_reliability: &[[u8; 8]],
    searched: Option<(&[CwRead], &[[u8; 8]])>,
    ec_len: usize,
    reserved: usize,
    opts: &DecodeOptions,
    search_of: Search,
    memo: &Memo,
) -> BlockSolve {
    let guided = searched.is_some();
    let search_of = if guided { search_of.again() } else { search_of };
    let (search, search_bits) = searched.unwrap_or((block, bit_reliability));
    // What the solvers that go by reliability have to doubt, flip or
    // move: each runs, and is charged for, only where it has a bit.
    let faint = Faint::of(search, search_bits);
    // The decode of the pinned codewords alone, made of a block with a
    // codeword pinned whole where nothing is laid over it, and charged
    // only there: a search with the message's evidence laid over it was
    // once charged for it too.
    let pins_alone = !guided && block.iter().any(|c| c.pinned == 0xFF);
    let charges = search_of.charges(opts, pins_alone, faint);
    // A codeword as a whole is as reliable as its least reliable module.
    let reliability: Vec<u8> = search_bits.iter().map(|bits| *bits.iter().min().unwrap_or(&CERTAIN)).collect();
    let mut pool: Vec<Candidate> = Vec::new();
    // Every codeword any solver reached, offered or not: what the
    // solutions offered are held against besides each other.
    let mut seen = Sightings::default();
    let weigh = |candidate: Candidate| -> Option<Candidate> {
        let candidate = if guided { candidate.restated(block, bit_reliability, ec_len, reserved, charges)? } else { candidate };
        Some(candidate.weigh(block, read, bit_reliability, ec_len, reserved, charges))
    };
    let verified_so_far = |pool: &[Candidate]| pool.iter().any(Candidate::verified_alone);
    let erased = search.iter().filter(|c| c.is_erasure()).count();
    // The standard reserves error correction codewords for detection in
    // the smallest symbols (ISO/IEC 18004 Table 9), and a solution that
    // spends them is one it says not to trust. The plain decoder keeps
    // such a solution all the same, as a last resort, marked as beyond
    // the standard and never verified, and the fallbacks offer theirs
    // the same way, each preferring a solution within the standard where
    // it finds one ([`super::rs::preferring_the_standard`]). Whether a
    // solution is so is what it changes of the block as read, whatever
    // found it ([`keeps_reserve`]): on a version 1 to 3 symbol, or a
    // Micro QR, the wrong codewords the fallbacks reach past what the
    // plain decoder corrects spend the reserve, and what they bring is
    // confirmation of the codeword it finds, and the modules that could
    // not be read solved for bit by bit. A block whose every EC codeword
    // is reserved, as M1's, the chase, the curve and the propagation
    // leave alone: there the reserve is all there is, and flipping bits
    // until the check passes is correction by another name.
    let rs_outcome = |method: SolveMethod, solution: &RsSolution| BlockOutcome::Solved {
        method,
        errors: solution.errors,
        erasures: solution.erasures,
        slack: solution.slack,
        confidence_bits: solution.confidence_bits(),
        risk_bits: 0,
        beyond_standard: !solution.within_standard(),
    };
    // A solution of a solver of codewords: shown as its solver counts it,
    // charged for the decodes of the grid, and weighed from what it counts
    // before any charge and what its solver's tries cost.
    let rs_candidate = |method: SolveMethod, solution: RsSolution, doubted: Vec<u8>| {
        let (own, tried) = (solution.uncharged_bits(), f64::from(solution.charge));
        let outcome = rs_outcome(method, &solution.clone().among(opts.tries));
        weigh(Candidate::new(outcome, solution.corrected, doubted, own, tried))
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
            // Charged for the decodes of which this one is, as every
            // solution is: each is a fresh chance for a block of noise
            // to pass, whatever solved the block.
            confidence_bits: doubt.confidence_bits().saturating_sub(tries_charge(opts.tries)),
            risk_bits: 0,
            // As the standard counts it, whatever bits were solved for:
            // two codewords for each codeword read and changed, and an
            // eighth of one for each module unread ([`keeps_reserve`]).
            // The spare equations stood in for the unused codewords once,
            // eight to a codeword, and a bit doubted took one of them as a
            // bit unread does: a block wrong past the standard's limit,
            // the faint bits of its wrong codewords doubted, kept the
            // reserve so.
            beyond_standard: !keeps_reserve(search, ec_len, reserved, &doubt.solution.corrected),
        }
    };
    let bit_candidate = |doubt: BitDoubt| {
        let outcome = bit_outcome(&doubt);
        let (own, tried) = (doubt.uncharged_bits(), tries_bits(doubt.tries));
        weigh(Candidate::new(outcome, doubt.solution.corrected, doubt.doubted, own, tried))
    };
    let flips = opts.reliability && opts.flip_bits;
    let chase_bits = chase_bits_among(opts.chase_bits, opts.shared);

    let mut exact_solved = false;
    let failure = match rs_decode_reserving_noting(search, ec_len, reserved, &mut seen) {
        Ok(solution) => {
            seen.see(&solution.corrected);
            // One that overturns bits read in the codewords it filled in
            // leaves the search for another open, as it did when such a
            // solution was refused outright; and so does one that spends
            // the codewords the standard reserves where a codeword that
            // keeps them may lie past the exact decoder's reach
            // ([`plain_reach_holds_the_reserve`]).
            exact_solved = solution.flipped_known_bits == 0
                && (keeps_reserve(read, ec_len, reserved, &solution.corrected) || plain_reach_holds_the_reserve(search, read));
            match rs_candidate(SolveMethod::ReedSolomon, solution, Vec::new()) {
                Some(candidate) if candidate.verified_alone() => return BlockSolve::solved(candidate).framed(block, bit_reliability, &seen),
                Some(candidate) => pool.push(candidate),
                None => {}
            }
            RsFailure::BudgetExceeded
        }
        Err(failure) => failure,
    };
    // The trials of erasures are for a block the exact decoder cannot
    // solve at all, or solves only by overturning bits that were read.
    // On one it solved, erasing by trial the codewords it located as
    // wrong names the same codeword with a unit of redundancy more to
    // spare, for nothing: a wrong codeword costs two and an erasure one.
    // Nor, for the same reason, for one it solved as searched, whose
    // solution the block as read refused. Nor where its solution spends
    // the codewords the standard reserves and the block searched reads
    // as the block as read does, every codeword whole or unread: one that
    // keeps them, as the standard counts a solution's errors against the
    // block as read ([`keeps_reserve`]), would lie within half the code's
    // distance as well, where there is no other, and the curve and the
    // trials would find only codewords past the standard's limit too.
    // Where a codeword is read in part, or the search reads what the block
    // as read leaves unread, one that keeps them may lie past the exact
    // decoder's reach ([`plain_reach_holds_the_reserve`]), and the search
    // goes on: it once stopped there, and the exact decoder's codeword,
    // past the standard's limit, stood in the place of one within it.
    let exact_failed = !exact_solved;

    let mut partial = None;
    if opts.bit_level {
        let bits = solve_bit_erasures(search, ec_len);
        let inconsistent = matches!(bits, Err(BitSolveFailure::Inconsistent));
        match bits {
            Ok(solution) if solution.is_complete() => {
                seen.see(&solution.corrected);
                let outcome = BlockOutcome::Solved {
                    method: SolveMethod::BitLevel,
                    errors: 0,
                    erasures: erased,
                    slack: 0,
                    confidence_bits: (solution.spare_equations as u32).saturating_sub(tries_charge(opts.tries)),
                    risk_bits: 0,
                    beyond_standard: !keeps_reserve(search, ec_len, reserved, &solution.corrected),
                };
                pool.extend(weigh(Candidate::new(outcome, solution.corrected, Vec::new(), solution.spare_equations as u32, 0.0)));
            }
            Ok(solution) => partial = Some(solution),
            Err(_) => {}
        }
        // The known bits cannot all be right: a bit or two of them
        // flipped, or a codeword or two freed by trial, may make them
        // agree, whatever their reliability.
        let trials = Trials { flips, free: opts.erase_by_trial && exact_failed, sets: opts.trial_sets };
        if (trials.flips || trials.free) && inconsistent {
            let found = memo.try_bits(search, search_bits, ec_len, trials, &mut seen);
            pool.extend(found.into_iter().filter_map(bit_candidate));
        }
    }

    if opts.reliability
        && opts.doubt_codewords
        && let Some((doubted, solution)) = memo.doubt(search, read, &reliability, ec_len, reserved, &mut seen)
    {
        // The codewords doubted are those the solution took as erased
        // that were read in full.
        let mut which = vec![0u8; block.len()];
        for erratum in solution.errata.iter().filter(|e| e.was_erasure && !search[e.index].is_erasure()) {
            which[erratum.index] = 0xFF;
        }
        pool.extend(rs_candidate(SolveMethod::Reliability { doubted }, solution, which));
    }

    // Doubting bits rather than codewords reaches damage spread over more
    // codewords than there are EC codewords, where no erasure decoder can
    // follow. It competes with doubting codewords on confirmation: a
    // codeword solution with one of slack has one syndrome byte behind
    // it, which a wrong codeword passes once in a few hundred tries, and
    // the bit solution, when there is one, has every spare equation.
    if opts.reliability && opts.doubt_bits && opts.bit_level {
        let trials = Trials { flips, free: opts.erase_by_trial && exact_failed, sets: opts.trial_sets };
        let found = memo.doubt_bits(search, search_bits, ec_len, trials, &mut seen);
        pool.extend(found.into_iter().filter_map(bit_candidate));
    }

    // Flipping the faintest bits outright costs no redundancy where
    // doubting a codeword costs a unit and doubting a bit an equation: a
    // block wrong in a few more codewords than the code corrects is
    // reached when the extra ones read faintly, after doubting has run
    // out of erasures. Only while nothing has verified the block, as the
    // solvers below: a thousand decodes buy nothing a verified solution
    // has not.
    if flips && !verified_so_far(&pool) {
        let found = memo.chase(search, read, search_bits, ec_len, reserved, chase_bits, &mut seen).into_iter().filter_map(|(flipped, solution)| {
            let flipped_bits = flipped.iter().map(|bits| bits.count_ones() as usize).sum();
            rs_candidate(SolveMethod::Chase { flipped_bits }, solution, flipped)
        });
        pool.extend(found);
    }

    // Beliefs passed between the bits along the checks they share, the
    // checks adapted each round to the bits' reliability: what every
    // other solver takes as fixed, the reliability of each bit, is what
    // this moves, by what the rest of the block makes likely. Only
    // while nothing has verified the block, and only where a bit read
    // was sampled ([`Faint::bits`]), as the search is charged for it
    // ([`routes`]): a bit painted by hand is never moved. It once ran on
    // a block whose every bit read was painted, for a sampled bit that
    // was not read, and the search went uncharged for it.
    if opts.reliability && opts.belief_propagation && faint.bits && !verified_so_far(&pool) {
        let rounds = opts.rounds();
        let trials = Trials { flips, free: opts.erase_by_trial && exact_failed, sets: opts.trial_sets };
        let chase = if flips { chase_bits } else { 0 };
        let found = memo.propagate(search, read, search_bits, ec_len, reserved, rounds, trials, chase, &mut seen).into_iter().filter_map(
            |(rounds, flipped, solution)| {
                let flipped_bits = flipped.iter().map(|bits| bits.count_ones() as usize).sum();
                rs_candidate(SolveMethod::BeliefPropagation { rounds, flipped_bits }, solution, flipped)
            },
        );
        pool.extend(found);
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
    if opts.list_decode && opts.list_cost > 0 && exact_failed && !verified_so_far(&pool) {
        let certain = vec![[CERTAIN; 8]; block.len()];
        let weighed = if opts.reliability { search_bits } else { &certain[..] };
        let found = memo.list(search, read, weighed, ec_len, reserved, opts.list_cost, &mut seen).into_iter().filter_map(|(multiplicity, solution)| {
            let mut which = vec![0u8; block.len()];
            for erratum in solution.errata.iter().filter(|e| !e.was_erasure) {
                which[erratum.index] = erratum.magnitude & search[erratum.index].known;
            }
            rs_candidate(SolveMethod::List { multiplicity }, solution, which)
        });
        pool.extend(found);
    }

    // Erasing codewords by trial reaches a wrong codeword or two more
    // than the code corrects, whatever the modules' reliability: the
    // last resort of a grid painted by hand, which nothing above can
    // doubt, and the dearest, every set of the block's codewords being
    // a decode. Only when nothing above has verified the block.
    if opts.erase_by_trial && exact_failed && !verified_so_far(&pool) {
        let found = memo.trial(search, read, search_bits, ec_len, reserved, opts.trial_sets, &mut seen).into_iter().filter_map(|(erased, solution)| {
            let mut which = vec![0u8; block.len()];
            for &i in &erased {
                which[i] = search[i].known & !search[i].pinned;
            }
            rs_candidate(SolveMethod::Trial { erased: erased.len() }, solution, which)
        });
        pool.extend(found);
    }

    // With enough pinned codewords the block is determined by pins alone,
    // however badly the rest was read: the same whatever the search, so
    // only of the block as read.
    if pins_alone {
        let pinned_only: Vec<CwRead> = block
            .iter()
            .map(|c| CwRead { value: c.value & c.pinned, known: c.pinned, pinned: c.pinned })
            .collect();
        if let Ok(solution) = rs_decode_reserving(&pinned_only, ec_len, reserved) {
            seen.see(&solution.corrected);
            let outcome = BlockOutcome::Solved {
                method: SolveMethod::PinnedOnly,
                errors: 0,
                erasures: solution.erasures,
                slack: solution.slack,
                confidence_bits: solution.confidence_bits(),
                risk_bits: 0,
                beyond_standard: !solution.within_standard(),
            };
            let (own, tried) = (solution.uncharged_bits(), f64::from(solution.charge));
            pool.extend(weigh(Candidate::new(outcome, solution.corrected, Vec::new(), own, tried)));
        }
    }

    // Every codeword offered once, the best ranked solution of it
    // standing for the rest, with how far the block as read lies from
    // it; then each held against every other codeword reached by that
    // distance, those no solver offered among them, and the best ranked
    // first.
    let mut distinct = distinct_codewords(pool, block, bit_reliability);
    promote_reached(&mut distinct, &seen, block, read, bit_reliability, ec_len, reserved, charges);
    let unoffered: Vec<u32> = seen
        .codewords()
        .filter(|&codeword| distinct.iter().all(|held| held.corrected != codeword))
        .map(|codeword| soft_distance(block, bit_reliability, codeword))
        .collect();
    hold_against_each_other(&mut distinct, &unoffered);
    let Some(best) = distinct.first() else {
        let inferred = match partial {
            Some(solution) => solution.corrected.iter().zip(&solution.solved).map(|(&v, &m)| (v & m, m)).collect(),
            None => vec![(0, 0); block.len()],
        };
        return BlockSolve {
            outcome: BlockOutcome::Failed(failure),
            corrected: None,
            inferred,
            doubted: Vec::new(),
            rivals: Vec::new(),
            arbitrated_among: 0,
            near_errors: 0,
            message_evidence: 0,
            half_padded_with_ones: false,
            reached: Sightings::default(),
            frame: (Vec::new(), Vec::new()),
        }
        .framed(block, bit_reliability, &seen);
    };
    // Rivals are for a block nothing verified.
    let rivals = if best.outcome.is_verified() { Vec::new() } else { rivals_among(&distinct, &unoffered) };
    let best = distinct.swap_remove(0);
    BlockSolve { rivals, ..BlockSolve::solved(best) }.framed(block, bit_reliability, &seen)
}

/// Every codeword of `pool` once, the best ranked solution of it
/// standing for the rest, each carrying how far the block as read lies
/// from it ([`Candidate::distance`]). The distance was once set only on
/// the first solution found of a codeword, and one ranked above it that
/// took its place carried nought: the message's arbitration then
/// measured its margins from the block as read itself.
///
/// The decode of a block from its pins alone that keeps the reserve as
/// its pins count it stands for its codeword wherever another solution
/// of it spends the reserve as the block as read counts it, with what
/// that one confirms of the codeword where it is the more
/// ([`Candidate::confirmed_by`]): the pins name the codeword, and the
/// standard's reserve, there to catch a misread, has nothing of theirs to
/// catch, while the other solvers can still add confirmation to it. The
/// better ranked was once kept alone, and a block its pins solved within
/// the standard, and a chase confirmed past it, was left unverified.
fn distinct_codewords(pool: Vec<Candidate>, block: &[CwRead], bit_reliability: &[[u8; 8]]) -> Vec<Candidate> {
    let mut distinct: Vec<Candidate> = Vec::new();
    for candidate in pool {
        match distinct.iter_mut().find(|held| held.corrected == candidate.corrected) {
            Some(held) => {
                let distance = held.distance;
                let kept = if held.kept_by_its_pins() && candidate.is_beyond_standard() {
                    held.clone().confirmed_by(&candidate)
                } else if candidate.kept_by_its_pins() && held.is_beyond_standard() {
                    candidate.confirmed_by(held)
                } else if candidate.rank() > held.rank() {
                    candidate
                } else {
                    continue;
                };
                *held = Candidate { distance, ..kept };
            }
            None => {
                let distance = soft_distance(block, bit_reliability, &candidate.corrected);
                distinct.push(Candidate { distance, ..candidate });
            }
        }
    }
    distinct
}

/// `block` as the reserve the standard keeps counts what was read of it
/// ([`keeps_reserve`]): every bit read at a reliability of nought, a
/// coin's toss that says nothing either way ([`super::rs::bit_cost`]),
/// taken as unread, as a module painted as a guess is meant to be. A
/// module read at any reliability above that is a reading, and a misread
/// where a solution overturns it. `bit_reliability` is what the modules
/// were read at, never what a round of the decode lowered it to
/// ([`Clustering::lowered`]): lowering a module beside errors located is
/// doubting it after the fact, and a doubted module is read. Taken from
/// the lowered reliabilities, a solid module lowered to nought was
/// counted unread, at an eighth of a codeword where the standard counts
/// two, and a solution past its limit could keep the reserve.
fn heard(block: &[CwRead], bit_reliability: &[[u8; 8]]) -> Vec<CwRead> {
    block
        .iter()
        .zip(bit_reliability)
        .map(|(cw, bits)| {
            let read = cw.known & !cw.pinned;
            let guessed = (0..8u8).filter(|&b| read >> b & 1 != 0 && bits[usize::from(b)] == 0).fold(0u8, |m, b| m | 1 << b);
            CwRead { value: cw.value & !guessed, known: cw.known & !guessed, pinned: cw.pinned }
        })
        .collect()
}

/// The codewords a solver reached and offered nothing for that the block
/// itself verifies, added to `distinct`, the codewords offered, as
/// solutions of their own ([`SolveMethod::Reached`]). A solver keeps back
/// a codeword its own count, charged for its tries, would not verify
/// ([`super::rs::Offering`]), and the propagation's doubting and chase,
/// and every solver searching the block with the message's evidence laid
/// over it, tell the codewords they reach apart by another block than
/// this one, the beliefs' word or the evidence's, and keep the nearest of
/// that; what the block itself makes of a codeword, by how rarely noise
/// lies so near one ([`judged`](super::rs::judged)), asks nothing of
/// what reached it. Only a codeword the block as read lies as near to as
/// to any reached, within
/// the tie ([`DISTANCE_MARGIN`]), is weighed: of any further off, the
/// nearer codeword is the likelier by more than a byte, and holds it
/// down to nothing ([`held_against`]). Such a codeword once only held
/// down the solutions offered, the block's own among them, and a block
/// failed with the codeword it was in hand, unoffered. Whether such a
/// codeword keeps the reserve is counted against `read`, the block as
/// read ([`Candidate::weigh`]).
#[allow(clippy::too_many_arguments)]
fn promote_reached(
    distinct: &mut Vec<Candidate>,
    seen: &Sightings,
    block: &[CwRead],
    read: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    reserved: usize,
    charges: Charges,
) {
    let unoffered: Vec<(&[u8], u32)> = seen
        .codewords()
        .filter(|&codeword| distinct.iter().all(|held| held.corrected != codeword))
        .map(|codeword| (codeword, soft_distance(block, bit_reliability, codeword)))
        .collect();
    let Some(nearest) = distinct.iter().map(|held| held.distance).chain(unoffered.iter().map(|&(_, distance)| distance)).min() else { return };
    let charge = whole_charge(charges.grid + charges.judged);
    let unknown: usize = block.iter().map(|c| c.known.count_zeros() as usize).sum();
    for (codeword, distance) in unoffered {
        if distance >= nearest + DISTANCE_MARGIN {
            continue;
        }
        // The soft count only as far as it verifies the codeword: one far
        // from the block as read, which noise lies as near to as often as
        // not, is refused on its first few bits.
        let soft = soft_count_short_of(block, bit_reliability, (8 * ec_len).saturating_sub(unknown), distance, VERIFIED_BITS + 1 + charge);
        let Some(solution) = judged_by(block, ec_len, reserved, codeword.to_vec(), charge, soft) else { continue };
        let outcome = BlockOutcome::Solved {
            method: SolveMethod::Reached,
            errors: solution.errors,
            erasures: solution.erasures,
            slack: solution.slack,
            confidence_bits: solution.confidence_bits(),
            risk_bits: 0,
            beyond_standard: !solution.within_standard(),
        };
        let candidate = Candidate::new(outcome, solution.corrected, Vec::new(), 0, 0.0).weigh(block, read, bit_reliability, ec_len, reserved, charges);
        distinct.push(Candidate { distance, ..candidate });
    }
}

/// The risk, in bits, of a solution wrong no oftener than once in two to
/// `alone` on its own, the block as read lying `own` from it, held
/// against other codewords of the code it lies `others` from
/// ([`soft_distance`]): each is the block's codeword, against this one,
/// as often as the likelihood ratio between them says, a bit for every
/// eight by which it lies further off, and any of them being so makes
/// this one wrong, so their chances and its own are added
/// ([`combined_risk_bits`]). A solution was once held to the nearest
/// other alone, and stood a bit or more above what several codewords
/// nearly as near left it.
fn held_against(alone: u32, own: u32, others: impl IntoIterator<Item = u32>) -> u32 {
    let margins = others.into_iter().map(|distance| Some(distance.saturating_sub(own) / 8));
    combined_risk_bits(std::iter::once(Some(alone)).chain(margins)).unwrap_or(0)
}

/// Each candidate held against every other by the block as read, by how
/// far the block lies from each ([`Candidate::distance`]), and against
/// the codewords no solver offered that the block lies `unoffered` from
/// ([`held_against`]). Then the best ranked first.
fn hold_against_each_other(distinct: &mut [Candidate], unoffered: &[u32]) {
    let distances: Vec<u32> = distinct.iter().map(|candidate| candidate.distance).collect();
    for (i, candidate) in distinct.iter_mut().enumerate() {
        let others = distances.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, &d)| d).chain(unoffered.iter().copied());
        let risk = held_against(candidate.alone, candidate.distance, others);
        candidate.set_risk(risk);
    }
    distinct.sort_by_key(|candidate| std::cmp::Reverse(candidate.rank()));
}

/// The rivals of a block nothing verified, from its distinct candidates
/// as [`hold_against_each_other`] leaves them: the codewords the block as
/// read lies as near to as the nearest, which it cannot tell apart, the
/// best ranked [`MAX_RIVALS`] of them, and none where fewer than two are
/// so near. Each carries the risk it may stand on once the message has
/// told it from the others ([`Candidate::held`]): its own, held down by
/// its margin over every codeword found that is not among them
/// ([`held_against`]), since the message is never asked about those.
/// That is one as near as the rivals but left out for their number; one
/// the block as read tells from the nearest, lying past the tie's
/// window, which is drawn about the nearest, and yet within a bit of a
/// rival further off; and one a solver reached and offered nothing for,
/// which the block as read lies `unoffered` from. The rivals once stood
/// clear of the second kind, and a block went from unverified to
/// verified on the message's word for a rival, over a codeword one bit
/// from it that the message was never shown.
fn rivals_among(distinct: &[Candidate], unoffered: &[u32]) -> Vec<Candidate> {
    let nearest = distinct.iter().map(|candidate| candidate.distance).min().unwrap_or(0);
    let tied: Vec<usize> = (0..distinct.len()).filter(|&i| distinct[i].distance < nearest + DISTANCE_MARGIN).collect();
    if tied.len() < 2 {
        return Vec::new();
    }
    let kept = &tied[..tied.len().min(MAX_RIVALS)];
    kept.iter()
        .map(|&i| {
            let others = (0..distinct.len()).filter(|j| !kept.contains(j)).map(|j| distinct[j].distance).chain(unoffered.iter().copied());
            let mut candidate = distinct[i].clone();
            candidate.held = held_against(candidate.alone, candidate.distance, others);
            candidate
        })
        .collect()
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
/// tried again with the nibble filled with ones, each solve held against
/// what the other's searches reached ([`hold_together`]), and the better
/// kept. With `laid`, the block is searched with the message's evidence
/// laid over it, and judged as read with its unread bits at their lean
/// ([`solve_block`]), the nibble filled the same way in both. The search
/// with the nibble filled is one of the ways the search is made
/// ([`ways`]), and is not of the block as read. `sampled` is what each
/// bit was read at, which `bit_reliability` is too but in a round that
/// lowered it: what the reserve counts as read is told by it ([`heard`]).
#[allow(clippy::too_many_arguments)]
fn solve_spec(
    layout: &SymbolLayout,
    spec: &BlockSpec,
    pinned: &[CwRead],
    bit_reliability: &[[u8; 8]],
    sampled: &[[u8; 8]],
    laid: Option<&Laid>,
    opts: &DecodeOptions,
    search_of: Search,
    memo: &Memo,
) -> BlockSolve {
    let read: Vec<CwRead> = spec.codewords.iter().map(|&cw| pinned[cw]).collect();
    let read_bits: Vec<[u8; 8]> = spec.codewords.iter().map(|&cw| bit_reliability[cw]).collect();
    let sampled_bits: Vec<[u8; 8]> = spec.codewords.iter().map(|&cw| sampled[cw]).collect();
    let heard_read = heard(&read, &sampled_bits);
    let (block, block_bits) = match laid {
        Some(laid) => (&laid.judged[..], &laid.judged_bits[..]),
        None => (&read[..], &read_bits[..]),
    };
    let search = laid.map(|laid| (&laid.searched[..], &laid.searched_bits[..]));
    let mut solve = solve_block(block, &heard_read, block_bits, search, spec.ec_len, spec.reserved, opts, search_of, memo);
    if tables::has_half_codeword(layout.version()) && !solve.rank().0 {
        let i = spec.data_len - 1;
        let missing = !codeword_bit_mask(layout, spec.codewords[i]);
        let fill = |block: &[CwRead]| {
            let mut filled = block.to_vec();
            filled[i].value |= missing;
            filled
        };
        let (filled, filled_read) = (fill(block), fill(&heard_read));
        let filled_search = laid.map(|laid| (fill(&laid.searched), &laid.searched_bits[..]));
        let search = filled_search.as_ref().map(|(block, bits)| (&block[..], *bits));
        let mut again = solve_block(&filled, &filled_read, block_bits, search, spec.ec_len, spec.reserved, opts, search_of.again(), memo);
        // Two searches of one block, as any two are: the filled one was
        // once taken with nothing said of the codewords the other had
        // reached, which were then forgotten.
        hold_together(&mut solve, &mut again);
        // A solution that erased the widened codeword by trial owes
        // nothing to the ones: it filled the codeword with whatever fit,
        // and only counted the nibble as one wrong codeword more. Nor
        // does one whose codeword there holds anything but ones in the
        // nibble, whatever reached it: a codeword a solver reached and
        // offered nothing for doubts no bit ([`promote_reached`]), and
        // one so reached with the nibble at nought was once said to be
        // padded with ones.
        let padded = again.corrected.as_ref().is_none_or(|corrected| corrected[i] & missing == missing);
        let owes_to_ones = padded && again.doubted.get(i).is_none_or(|&doubted| doubted & missing == 0);
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
        message_evidence: solve.message_evidence,
        tried_harder: false,
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
    let Parsed { stream, parse, tail, stream_valid, tail_plausible } = parsed(layout, codewords, blocks, pinned);
    let text = text::decode_text_of(layout.version().family(), &parse.segments, charset);
    Assembled { stream, parse, tail, text, stream_valid, tail_plausible }
}

/// What [`Assembled`] holds but the text: all the message is asked
/// for when its rivals are told apart ([`StreamScore`]).
struct Parsed {
    stream: StreamBits,
    parse: ParseReport,
    tail: Option<TailCheck>,
    stream_valid: bool,
    tail_plausible: bool,
}

fn parsed(layout: &SymbolLayout, codewords: &[CodewordReport], blocks: &[BlockReport], pinned: &[CwRead]) -> Parsed {
    parsed_of(layout, stream_of(layout, codewords, blocks, pinned))
}

/// The data stream the codeword reports make: each data codeword's bits
/// in the stream's order ([`codeword_bits`]).
fn stream_of(layout: &SymbolLayout, codewords: &[CodewordReport], blocks: &[BlockReport], pinned: &[CwRead]) -> StreamBits {
    let mut stream = StreamBits::default();
    for &cw in &layout.data_stream {
        let report = &codewords[cw];
        let (value, known) = report.best();
        for (bit, certainty) in codeword_bits(layout, blocks, pinned, cw, value, known, report.final_value.is_some()) {
            stream.push(bit, certainty);
        }
    }
    stream
}

/// The bits data codeword `cw` lays in the stream, read as `value` where
/// `known`, each with how sure it is: a pinned bit verified, a bit of a
/// codeword its block `solved` as sure as the block, another known bit
/// unverified, and the rest unknown.
fn codeword_bits(
    layout: &SymbolLayout,
    blocks: &[BlockReport],
    pinned: &[CwRead],
    cw: usize,
    value: u8,
    known: u8,
    solved: bool,
) -> impl Iterator<Item = (bool, Certainty)> {
    let span = layout.codewords[cw];
    let solved_as = if blocks[span.block as usize].is_verified() { Certainty::Verified } else { Certainty::Unverified };
    let pin = pinned[cw].pinned;
    (0..span.stream_bits()).map(|i| 0x80u8 >> i).map(move |bit| {
        let certainty = if pin & bit != 0 {
            Certainty::Verified
        } else if solved {
            solved_as
        } else if known & bit != 0 {
            Certainty::Unverified
        } else {
            Certainty::Unknown
        };
        (value & bit != 0, certainty)
    })
}

/// `stream` parsed, its end found and the bits after it checked against
/// the padding.
fn parsed_of(layout: &SymbolLayout, stream: StreamBits) -> Parsed {
    let parse = bitstream::parse(&stream, &layout.stream);
    let tail = match parse.end {
        EndOfData::At { bit, .. } => Some(padding::check_tail(layout, &stream, bit)),
        EndOfData::Lost { .. } => None,
    };
    let tail_plausible = tail.as_ref().is_some_and(TailCheck::is_plausible);
    // A stream that ends before its first segment is a message only when
    // what follows is the padding an empty message has. Under a wrong
    // mask or on a wrong placement the stream is noise, and noise begins
    // with a terminator's zeros once in sixteen tries. Zeros to the end
    // explain it too, as an encoder may pad, though the mask's own pattern
    // reads so as well; such a read stands, and is warned of
    // ([`DecodeReport::data_is_the_mask`]).
    let empty_unexplained = parse.segments.is_empty() && !tail_plausible;
    let stream_valid = matches!(parse.end, EndOfData::At { .. })
        && !parse.preamble_damaged
        && !empty_unexplained
        && parse.segments.iter().all(|s| !s.damaged && !s.truncated);
    Parsed { stream, parse, tail, stream_valid, tail_plausible }
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
    fn of(parsed: &Parsed) -> Self {
        let segments = &parsed.parse.segments;
        Self {
            valid: parsed.stream_valid,
            tail_plausible: parsed.tail_plausible,
            ended: matches!(parsed.parse.end, EndOfData::At { .. }),
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
/// taken, standing on its own risk ([`Candidate::held`]), since the
/// message is the same whichever of them is right. Between rivals that
/// disagree and that the message does not tell apart, the block stays
/// as it was. The blocks are taken in order, each with those before it
/// as settled.
///
/// A rival the message tells from the rest stands on what the message
/// says for it, and on its own risk: against each other rival, the
/// margin the block as read gives it, which is less than a tie's and may
/// be less than nothing, and how rarely the data it differs from that
/// rival in, were it wrong, would let the stream parse as well
/// ([`message_bits`]), the rivals' chances and its own
/// ([`Candidate::held`]) added. It once stood on its own risk whatever
/// the message said, as though a stream that parsed better were proof of
/// the data that made it so, and then on the lesser of the two, as
/// though only the likelier way of being wrong could be.
///
/// The solvers that try many ways are list decoders, and a list of two
/// is what they return where the block as read lies as near to two
/// codewords ([`super::rs::DISTANCE_MARGIN`]); the message is the one
/// judge of such a tie that is not the block itself, since a wrong
/// codeword's data parses as a segment of noise, with a header it does
/// not have or a value no character takes, where the right one's reads
/// on as the message does.
///
/// `asked` holds, for each block, every codeword the message has been
/// asked about as a rival of that block so far, and the rivals told
/// apart here are added to it: what tells a round of evidence that
/// brings a block new rivals from one that brings back the old
/// ([`weigh_again`]).
fn arbitrate(
    layout: &SymbolLayout,
    raw: &[CwRead],
    pinned: &[CwRead],
    solves: &mut [BlockSolve],
    codewords: &mut [CodewordReport],
    blocks: &mut [BlockReport],
    asked: &mut [Vec<Vec<u8>>],
) {
    for b in 0..solves.len() {
        if blocks[b].is_verified() || solves[b].rivals.is_empty() {
            continue;
        }
        let spec = &layout.blocks[b];
        let rivals = std::mem::take(&mut solves[b].rivals);
        for rival in &rivals {
            if !asked[b].contains(&rival.corrected) {
                asked[b].push(rival.corrected.clone());
            }
        }
        let mut scored: Vec<(StreamScore, usize)> = rivals
            .iter()
            .enumerate()
            .map(|(r, candidate)| {
                let trial = BlockSolve::solved(candidate.clone());
                blocks[b] = report_block(layout, spec, &trial, raw, pinned, codewords);
                (StreamScore::of(&parsed(layout, codewords, blocks, pinned)), r)
            })
            .collect();
        scored.sort_by(|a, b| b.cmp(a));
        let data = spec.data_len;
        let agree_on_data = rivals.iter().all(|c| c.corrected[..data] == rivals[0].corrected[..data]);
        let taken = match scored.as_slice() {
            &[(best, w), (next, _), ..] if best > next => {
                blocks[b] = report_block(layout, spec, &BlockSolve::solved(rivals[w].clone()), raw, pinned, codewords);
                let risk = told_risk(layout, pinned, codewords, blocks, b, &rivals, w, best);
                Some(rivals[w].clone().at_risk(risk))
            }
            _ if agree_on_data => Some(rivals[0].clone().at_risk(rivals[0].held)),
            _ => None,
        }
        .filter(|candidate| candidate.rank() > solves[b].rank());
        if let Some(candidate) = taken {
            let (near_errors, message_evidence, half) = (solves[b].near_errors, solves[b].message_evidence, solves[b].half_padded_with_ones);
            let (reached, frame) = (std::mem::take(&mut solves[b].reached), std::mem::take(&mut solves[b].frame));
            solves[b] = BlockSolve {
                arbitrated_among: rivals.len(),
                near_errors,
                message_evidence,
                half_padded_with_ones: half,
                reached,
                frame,
                ..BlockSolve::solved(candidate)
            };
        }
        blocks[b] = report_block(layout, spec, &solves[b], raw, pinned, codewords);
    }
}

/// The risk rival `w` of block `b` stands on once the message, parsing
/// to `score` under it as `codewords` and `blocks` lay it, has told it
/// from the other `rivals` ([`arbitrate`]): for each rival that would
/// make the message other than the winner's, the chance it is the right
/// one, by the block as read and by the message ([`message_bits`]), and
/// the winner's own ([`Candidate::held`]), added.
#[allow(clippy::too_many_arguments)]
fn told_risk(
    layout: &SymbolLayout,
    pinned: &[CwRead],
    codewords: &[CodewordReport],
    blocks: &[BlockReport],
    b: usize,
    rivals: &[Candidate],
    w: usize,
    score: StreamScore,
) -> u32 {
    let spec = &layout.blocks[b];
    let data = spec.data_len;
    let winner = &rivals[w];
    let chances: f64 = rivals
        .iter()
        .enumerate()
        .filter(|&(r, rival)| r != w && rival.corrected[..data] != winner.corrected[..data])
        .map(|(r, rival)| {
            let differ: Vec<usize> = (0..data).filter(|&i| rival.corrected[i] != winner.corrected[i]).map(|i| spec.codewords[i]).collect();
            let seed = 0x9E37_79B9_7F4A_7C15 ^ ((b as u64) << 32) ^ r as u64;
            let margin = (f64::from(rival.distance) - f64::from(winner.distance)) / 8.0;
            (-(margin + message_bits(layout, pinned, codewords, blocks, &differ, score, seed))).exp2()
        })
        .sum();
    whole_risk_bits(-(chances + (-f64::from(winner.held)).exp2()).log2())
}

/// Random fillings of the data codewords a rival differs from the
/// winner in that [`message_bits`] parses, at the most: enough to say
/// that noise there parses as well once in two hundred and fifty-six,
/// eight bits, and no further.
const MESSAGE_SAMPLES: u32 = 255;

/// Fillings that parse as well as the winner's data after which
/// [`message_bits`] stops: sixteen say the share to within a quarter of
/// itself, and a share that large is a few bits at the most.
const MESSAGE_ENOUGH: u32 = 16;

/// What the message says for the data a block's winner holds in the
/// codewords `differ`, against a rival's that differ there, in bits:
/// how rarely the stream would parse as well as it does, scoring
/// `score` ([`StreamScore`]), were those codewords noise, as a wrong
/// codeword's are. Counted by filling them with random bytes, from
/// `seed`, their pinned bits kept, as every codeword the block can be
/// keeps them, and parsing the stream with each filling, the rest of it
/// as `codewords` lay it, the winner's data among them: the share of
/// fillings that parse at least as well, one more counted among them and
/// among all, so that what was never seen is held to one in as many as
/// were drawn ([`MESSAGE_SAMPLES`]). The fillings once overturned the
/// pinned bits too, a segment's mode among them where a known beginning
/// pinned it, and noise no codeword of the block could hold parsed worse
/// than any that could, which said more for the winner than the message
/// did.
///
/// The stream is laid once, and each filling laid over the bits those
/// codewords hold in it, each as a solved codeword's bit is
/// ([`codeword_bits`]). It was once laid anew from the codeword reports
/// for every filling, which was four fifths of the work and more: a
/// hundredth of a second for every rival a block of a large symbol was
/// told from.
fn message_bits(
    layout: &SymbolLayout,
    pinned: &[CwRead],
    codewords: &[CodewordReport],
    blocks: &[BlockReport],
    differ: &[usize],
    score: StreamScore,
    mut seed: u64,
) -> f64 {
    let mut stream = stream_of(layout, codewords, blocks, pinned);
    // Where each codeword's bits begin in the stream.
    let mut starts = vec![0; layout.codewords.len()];
    let mut at = 0;
    for &cw in &layout.data_stream {
        starts[cw] = at;
        at += layout.codewords[cw].stream_bits();
    }
    let (mut drawn, mut as_well) = (0u32, 0u32);
    while drawn < MESSAGE_SAMPLES && as_well < MESSAGE_ENOUGH {
        for &cw in differ {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let pin = pinned[cw];
            let filled = (seed >> 24) as u8 & !pin.pinned | pin.value & pin.pinned;
            for (i, (bit, certainty)) in codeword_bits(layout, blocks, pinned, cw, filled, 0xFF, true).enumerate() {
                stream.set(starts[cw] + i, bit, certainty);
            }
        }
        let parsed = parsed_of(layout, stream);
        as_well += u32::from(StreamScore::of(&parsed) >= score);
        stream = parsed.stream;
        drawn += 1;
    }
    (f64::from(drawn + 1) / f64::from(as_well + 1)).log2()
}

/// Rounds of solving again the blocks that did not verify, with what
/// the blocks that did show of how the photo misread
/// ([`Clustering`]) and what the message makes likely of their bits
/// ([`MessageEvidence`]): each block that then verifies locates errors
/// and settles characters of its own for the next.
const EVIDENCE_ROUNDS: usize = 2;

/// What a block solved again in a round of evidence is to the solve in
/// hand: whether it takes that solve's place, and whether the round has
/// improved the decode by it. It takes the place of the solve in hand
/// when it ranks higher, or ranks the same and brings rivals, since the
/// arbitration empties the rivals of the solve it leaves and the
/// message, moved by the other blocks of the round, may now tell them
/// apart. It improves the decode when its rank rose, or when one of its
/// rivals is a codeword the message has not been asked about for the
/// block (`asked`, as [`arbitrate`] keeps it). A solve recalled from the
/// rounds before, bringing back the rivals the message was asked about
/// already, is taken and is no improvement: the rounds once took it for
/// one, and a block with rivals the message could not tell apart kept
/// them from ending early whatever else they found.
fn weigh_again(current: &BlockSolve, again: &BlockSolve, asked: &[Vec<u8>]) -> (bool, bool) {
    let rose = again.rank() > current.rank();
    let taken = rose || (again.rank() == current.rank() && !again.rivals.is_empty());
    let unasked = again.rivals.iter().any(|rival| !asked.contains(&rival.corrected));
    (taken, taken && (rose || unasked))
}

/// The exact decode of a block, when it verifies within the standard:
/// the first thing [`solve_block`] tries, and all a sound block needs,
/// made alone before the hard blocks are shared over the cores, since
/// it is microseconds and a thread is more. Whether it keeps the reserve
/// is counted against `read`, the block as the reserve counts it
/// ([`heard`]).
fn exact_solve(
    block: &[CwRead],
    read: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    reserved: usize,
    opts: &DecodeOptions,
    search_of: Search,
) -> Option<BlockSolve> {
    let solution = rs_decode_reserving(block, ec_len, reserved).ok()?;
    let (own, tried) = (solution.uncharged_bits(), f64::from(solution.charge));
    let shown = solution.clone().among(opts.tries);
    let outcome = BlockOutcome::Solved {
        method: SolveMethod::ReedSolomon,
        errors: shown.errors,
        erasures: shown.erasures,
        slack: shown.slack,
        confidence_bits: shown.confidence_bits(),
        risk_bits: 0,
        beyond_standard: !shown.within_standard(),
    };
    let charges = search_of.charges(opts, block.iter().any(|c| c.pinned == 0xFF), Faint::of(block, bit_reliability));
    let candidate = Candidate::new(outcome, solution.corrected, Vec::new(), own, tried).weigh(block, read, bit_reliability, ec_len, reserved, charges);
    candidate.verified_alone().then(|| BlockSolve::solved(candidate).framed(block, bit_reliability, &Sightings::default()))
}

/// One way a block was solved, from what, and what came of it: from its
/// codewords as pinned and their bits' reliability, searched as read or
/// with the message's evidence laid over them, charged for pins picked
/// among `among` ways ([`PinsPicked`]). The same reads charged otherwise
/// are another solve, as a decode made afresh under the pins would
/// charge them: a solve recalled from a decode under fewer ways was once
/// shown at the risk the pins in force no longer allowed. And `heard`,
/// the codewords as the reserve the standard keeps counts what was read
/// of them ([`heard`]), which the reliabilities a round lowered do not
/// tell: a solve is judged within the standard's limit or past it by
/// that, and a search kept from an earlier decode of a grid whose
/// module was read at a coin's toss then and is read now was once taken
/// with the module counted unread.
#[derive(Clone)]
struct Solved {
    reads: Vec<CwRead>,
    heard: Vec<CwRead>,
    bits: Vec<[u8; 8]>,
    laid: Option<Laid>,
    among: usize,
    solve: BlockSolve,
}

impl Solved {
    fn holds(&self, reads: &[CwRead], heard: &[CwRead], bits: &[[u8; 8]], laid: Option<&Laid>, among: usize) -> bool {
        self.reads == reads && self.heard == heard && self.bits == bits && self.laid.as_ref() == laid && self.among == among
    }
}

/// Ways of solving each block a decode remembers, at the most, the
/// oldest forgotten first: the rounds of [`Stage::solve`] ask a few
/// of each block, and padding recovery a few more under more pins.
const REMEMBERED: usize = 8;

/// The block as solved before from the same reads and reliabilities,
/// searched the same way and charged for as many ways, when it was.
fn recall(solved: &[Solved], reads: &[CwRead], heard: &[CwRead], bits: &[[u8; 8]], laid: Option<&Laid>, among: usize) -> Option<BlockSolve> {
    solved.iter().find(|s| s.holds(reads, heard, bits, laid, among)).map(|s| s.solve.clone())
}

fn remember(solved: &mut Vec<Solved>, entry: Solved) {
    if solved.len() == REMEMBERED {
        solved.remove(0);
    }
    solved.push(entry);
}

/// Blocks searched harder than a decode's options search them
/// ([`DecodeOptions::at_most`]), each search with the reads,
/// reliabilities and evidence it was made from: for later decodes of the
/// grid to take where a block reads the same and the search ranks better
/// than their own ([`decode_keeping`]). They hold only under the layout,
/// mask and options they were found under, since those say what the
/// codewords are and which solvers were switched on; a decode under
/// others is not to be given them. None is forgotten, as a decode
/// forgets its own ways of solving a block ([`REMEMBERED`]): each was
/// asked for and cost far more than it keeps, and a decode searching a
/// block harder searches it under every reliability and evidence its
/// rounds and its padding pins give it, more than a decode remembers.
/// The first, of the block as read under the caller's pins, which
/// [`BlockReport::tried_harder`] asks after, was once forgotten first.
/// Kept for each stage of a decode, under that stage's options ([`Harder`]).
#[derive(Clone, Default)]
struct Searched {
    solved: Vec<Vec<Solved>>,
}

/// The blocks searched harder in each stage of a decode
/// ([`DecodeOptions::stages`]), by the stage's index, each under its
/// stage's options alone ([`Searched`]): for later decodes of the grid
/// under the same layout, mask and options to take, each stage from its
/// own searches ([`decode_keeping`]).
#[derive(Clone, Default)]
pub struct Harder {
    stages: Vec<Searched>,
}

impl Harder {
    fn stage(&self, s: usize) -> Searched {
        self.stages.get(s).cloned().unwrap_or_default()
    }

    fn keep(&mut self, s: usize, searched: Searched) {
        if self.stages.len() <= s {
            self.stages.resize(s + 1, Searched::default());
        }
        self.stages[s] = searched;
    }
}

impl Searched {
    fn of(&self, b: usize) -> &[Solved] {
        self.solved.get(b).map_or(&[], Vec::as_slice)
    }

    fn keep(&mut self, b: usize, entry: Solved) {
        if self.solved.len() <= b {
            self.solved.resize(b + 1, Vec::new());
        }
        self.solved[b].push(entry);
    }

    /// Block `b` was searched harder from `reads`, `heard` and `bits`,
    /// searched as read or with the evidence `laid` over them, charged
    /// for pins picked among `among` ways.
    fn holds(&self, b: usize, reads: &[CwRead], heard: &[CwRead], bits: &[[u8; 8]], laid: Option<&Laid>, among: usize) -> bool {
        self.of(b).iter().any(|s| s.holds(reads, heard, bits, laid, among))
    }

    /// Of `solve`, the block solved under the decode's own options, and
    /// the block searched harder from the same reads, reliabilities and
    /// evidence, when it was, the better: a search at the most is charged
    /// for its many tries, and may rank below what fewer found.
    #[allow(clippy::too_many_arguments)]
    fn better(
        &self,
        b: usize,
        reads: &[CwRead],
        heard: &[CwRead],
        bits: &[[u8; 8]],
        laid: Option<&Laid>,
        among: usize,
        mut solve: BlockSolve,
    ) -> BlockSolve {
        match recall(self.of(b), reads, heard, bits, laid, among) {
            Some(mut harder) => {
                hold_together(&mut harder, &mut solve);
                if harder.rank() > solve.rank() { harder } else { solve }
            }
            None => solve,
        }
    }

    /// Block `b` was searched harder as it reads from `reads`, `heard` and
    /// `bits`, charged for pins picked among `among` ways.
    fn searched(&self, b: usize, reads: &[CwRead], heard: &[CwRead], bits: &[[u8; 8]], among: usize) -> bool {
        self.holds(b, reads, heard, bits, None, among)
    }
}

/// What each dear solver found of each block it was given and every
/// codeword it reached, by everything it was given: the block searched
/// and its bits' reliability, its error correction and reserve, and the
/// solver's budget and the trials it may make. The stages of a decode
/// give the solvers the same blocks over and again
/// ([`DecodeOptions::stages`]): a stage with the message's evidence
/// searches every block as read just as the stage without it did
/// before any evidence is laid, and the stages with the flips or the
/// trials give the solvers that make neither what the stage without
/// them gave. So each is asked once for each, and what it found then is
/// the same, and reaches the same codewords, as asking it again would:
/// a solver goes by what it is given alone, and only notes what it
/// reaches, which is noted again for each that asks ([`Sightings`]).
/// What a solver is charged for the solvers and searches beside it is
/// no part of what it finds, and is reckoned anew for each ([`Search`]).
/// What one found when the work was called off is not held.
#[derive(Default)]
struct Memo {
    lists: Held<SolverAsked, Vec<(usize, RsSolution)>>,
    chases: Held<SolverAsked, Vec<(Vec<u8>, RsSolution)>>,
    trials: Held<SolverAsked, Vec<(Vec<usize>, RsSolution)>>,
    doubts: Held<DoubtAsked, Option<(usize, RsSolution)>>,
    bit_doubts: Held<BitsAsked, Vec<BitDoubt>>,
    bit_trials: Held<BitsAsked, Vec<BitDoubt>>,
    propagations: Held<PropagationAsked, Vec<Propagated>>,
}

/// What a solver of codewords is given ([`Memo`]): the block, the block
/// as read that the reserve is counted against ([`heard`]), its bits'
/// reliability, its error correction and reserve, and the budget.
type SolverAsked = (Vec<CwRead>, Vec<CwRead>, Vec<[u8; 8]>, usize, usize, usize);
/// What a solver of bits is given ([`Memo`]): the block, its bits'
/// reliability, its error correction, and the trials it may make.
type BitsAsked = (Vec<CwRead>, Vec<[u8; 8]>, usize, Trials);
/// What the doubting of codewords is given ([`Memo`]): the block, the
/// block as read that the reserve is counted against, its codewords'
/// reliability, and its error correction and reserve.
type DoubtAsked = (Vec<CwRead>, Vec<CwRead>, Vec<u8>, usize, usize);
/// What the propagation is given ([`Memo`]): the block, the block as
/// read that the reserve is counted against, its bits' reliability, its
/// error correction and reserve, the rounds, the trials its doubting may
/// make and the bits its chase flips.
type PropagationAsked = (Vec<CwRead>, Vec<CwRead>, Vec<[u8; 8]>, usize, usize, usize, Trials, usize);
/// What a solver found, and the codewords it reached ([`Held`]).
type Found<F> = (F, Sightings);

/// What one solver found, by what it was asked, with the codewords it
/// reached ([`Memo`]).
struct Held<A, F>(Mutex<HashMap<A, Found<F>>>);

impl<A, F> Default for Held<A, F> {
    fn default() -> Self {
        Self(Mutex::new(HashMap::new()))
    }
}

impl<A: Eq + std::hash::Hash, F: Clone> Held<A, F> {
    /// What `find` finds asked `asked`, as it was found before when it
    /// was asked so before, noting in `seen` every codeword it reached
    /// either way.
    fn found(&self, asked: A, seen: &mut Sightings, find: impl FnOnce(&mut Sightings) -> F) -> F {
        let held = self.0.lock().unwrap().get(&asked).cloned();
        let (found, reached) = match held {
            Some(held) => held,
            None => {
                let mut reached = Sightings::default();
                let found: Found<F> = (find(&mut reached), reached);
                if !work::cancelled() {
                    self.0.lock().unwrap().insert(asked, found.clone());
                }
                found
            }
        };
        seen.see_all(&reached);
        found
    }
}

impl Memo {
    #[allow(clippy::too_many_arguments)]
    fn list(
        &self,
        block: &[CwRead],
        read: &[CwRead],
        bits: &[[u8; 8]],
        ec_len: usize,
        reserved: usize,
        cost: usize,
        seen: &mut Sightings,
    ) -> Vec<(usize, RsSolution)> {
        let asked = (block.to_vec(), read.to_vec(), bits.to_vec(), ec_len, reserved, cost);
        self.lists.found(asked, seen, |seen| rs_decode_listing_against(block, read, bits, ec_len, reserved, cost, seen))
    }

    #[allow(clippy::too_many_arguments)]
    fn chase(
        &self,
        block: &[CwRead],
        read: &[CwRead],
        bits: &[[u8; 8]],
        ec_len: usize,
        reserved: usize,
        flips: usize,
        seen: &mut Sightings,
    ) -> Vec<(Vec<u8>, RsSolution)> {
        let asked = (block.to_vec(), read.to_vec(), bits.to_vec(), ec_len, reserved, flips);
        self.chases.found(asked, seen, |seen| rs_decode_chasing_bits_against(block, read, bits, ec_len, reserved, flips, seen))
    }

    #[allow(clippy::too_many_arguments)]
    fn trial(
        &self,
        block: &[CwRead],
        read: &[CwRead],
        bits: &[[u8; 8]],
        ec_len: usize,
        reserved: usize,
        sets: usize,
        seen: &mut Sightings,
    ) -> Vec<(Vec<usize>, RsSolution)> {
        let asked = (block.to_vec(), read.to_vec(), bits.to_vec(), ec_len, reserved, sets);
        self.trials.found(asked, seen, |seen| rs_decode_trying_against(block, read, bits, ec_len, reserved, sets, seen))
    }

    fn doubt(&self, block: &[CwRead], read: &[CwRead], reliability: &[u8], ec_len: usize, reserved: usize, seen: &mut Sightings) -> Option<(usize, RsSolution)> {
        let asked = (block.to_vec(), read.to_vec(), reliability.to_vec(), ec_len, reserved);
        self.doubts.found(asked, seen, |seen| rs_decode_doubting_against(block, read, reliability, ec_len, reserved, seen))
    }

    fn doubt_bits(&self, block: &[CwRead], bits: &[[u8; 8]], ec_len: usize, trials: Trials, seen: &mut Sightings) -> Vec<BitDoubt> {
        let asked = (block.to_vec(), bits.to_vec(), ec_len, trials);
        self.bit_doubts.found(asked, seen, |seen| solve_bit_erasures_doubting(block, bits, ec_len, trials, seen))
    }

    fn try_bits(&self, block: &[CwRead], bits: &[[u8; 8]], ec_len: usize, trials: Trials, seen: &mut Sightings) -> Vec<BitDoubt> {
        let asked = (block.to_vec(), bits.to_vec(), ec_len, trials);
        self.bit_trials.found(asked, seen, |seen| solve_bit_erasures_trying(block, bits, ec_len, trials, seen))
    }

    #[allow(clippy::too_many_arguments)]
    fn propagate(
        &self,
        block: &[CwRead],
        read: &[CwRead],
        bits: &[[u8; 8]],
        ec_len: usize,
        reserved: usize,
        rounds: usize,
        trials: Trials,
        chase: usize,
        seen: &mut Sightings,
    ) -> Vec<Propagated> {
        let asked = (block.to_vec(), read.to_vec(), bits.to_vec(), ec_len, reserved, rounds, trials, chase);
        self.propagations.found(asked, seen, |seen| rs_decode_propagating_against(block, read, bits, ec_len, reserved, rounds, trials, chase, seen))
    }
}

/// `solve` done on every item, side by side on the spare cores when
/// there are two or more ([`work::map`]).
fn solve_each<T: Sync>(items: &[T], solve: impl Fn(&T) -> BlockSolve + Sync) -> Vec<BlockSolve> {
    if items.len() >= 2 { work::map(items, solve) } else { items.iter().map(solve).collect() }
}

/// The blocks of the layout `asked` for solved from the codewords as
/// pinned and the reliability of every bit, `as_read` where those are
/// the reliabilities as read, and, with `evidence`, each of them that
/// does not verify so searched again with the evidence laid over it
/// ([`solve_block`]), the budgets shared between the two searches
/// ([`DecodeOptions::sharing`]) and the better of the two kept. The
/// searches are of the `round`th round of evidence, the block as read
/// in the first, numbered from nought, and a search with the
/// reliabilities as read is the block as read whatever the round
/// ([`Search::nth`]); each is made twice as many ways where the block is
/// searched harder, now or before ([`Search`]). Every search
/// is charged for the ways the pins it reaches were picked among
/// (`picked`): one `as_read` for those its block holds, and every other
/// for all of them. A block
/// solved before from the same reads and reliabilities, searched the
/// same way, as `solved` remembers, is not solved again: a decode made
/// again under more pins solves only the blocks the pins touch, and the
/// rounds of [`Stage::solve`] only the blocks whose reliabilities or
/// evidence moved. Of the rest, the exact decoder settles the sound ones
/// alone, and the hard ones, each a search of its fallbacks, are solved
/// side by side on the spare cores ([`work::map`]). A hard one that is
/// `trying` and does not verify so is searched again at the most
/// ([`DecodeOptions::at_most`]), unless `harder` holds a search of it
/// from the same inputs already, and the search kept in `harder`; and
/// of a hard one's solve and a search of it at the most from the same
/// inputs that `harder` holds, the better is taken ([`Searched::better`]).
/// `sampled` is what each bit was read at, before any round lowered it,
/// which tells what the codewords the standard reserves count as read
/// ([`heard`]): it is the grid's own however `bit_reliability` moves, and
/// the solves `solved` remembers by their inputs, all of this grid, were
/// made under it too. The searches `harder` keeps may be of an earlier
/// decode of the grid as it then was, and are not told apart by it.
/// None for the blocks not asked for.
#[allow(clippy::too_many_arguments)]
fn solve_blocks(
    layout: &SymbolLayout,
    opts: &DecodeOptions,
    solved: &mut [Vec<Solved>],
    harder: &mut Searched,
    trying: &[bool],
    pinned: &[CwRead],
    bit_reliability: &[[u8; 8]],
    sampled: &[[u8; 8]],
    (as_read, round): (bool, usize),
    picked: &PinsPicked,
    evidence: Option<&MessageEvidence>,
    asked: &dyn Fn(usize) -> bool,
    memo: &Memo,
) -> Vec<Option<BlockSolve>> {
    let inputs = |spec: &BlockSpec| -> (Vec<CwRead>, Vec<CwRead>, Vec<[u8; 8]>) {
        let reads: Vec<CwRead> = spec.codewords.iter().map(|&cw| pinned[cw]).collect();
        let sampled_bits: Vec<[u8; 8]> = spec.codewords.iter().map(|&cw| sampled[cw]).collect();
        let heard = heard(&reads, &sampled_bits);
        (reads, heard, spec.codewords.iter().map(|&cw| bit_reliability[cw]).collect())
    };
    let ways_of = |b: usize| ways(layout.version()) * if trying[b] || !harder.of(b).is_empty() { 2 } else { 1 };
    let nth = if as_read { 1 } else { 2 * round };
    let among: Vec<usize> = (0..layout.blocks.len()).map(|b| if as_read { picked.block[b] } else { picked.all }).collect();
    let search_of: Vec<Search> = (0..layout.blocks.len()).map(|b| Search { as_read, nth, ways: ways_of(b), picked: among[b] }).collect();
    let guided_of: Vec<Search> =
        (0..layout.blocks.len()).map(|b| Search { as_read: false, nth: 2 * round + 1, ways: ways_of(b), picked: among[b] }).collect();
    let opts_of: Vec<DecodeOptions> = among.iter().map(|&among| opts.one_of(among)).collect();
    let mut out: Vec<Option<BlockSolve>> = vec![None; layout.blocks.len()];
    let mut hard = Vec::new();
    for (b, spec) in layout.blocks.iter().enumerate().filter(|&(b, _)| asked(b)) {
        let (reads, heard, bits) = inputs(spec);
        if let Some(solve) = recall(&solved[b], &reads, &heard, &bits, None, among[b]) {
            out[b] = Some(solve);
            continue;
        }
        match exact_solve(&reads, &heard, &bits, spec.ec_len, spec.reserved, &opts_of[b], search_of[b]) {
            Some(solve) => {
                remember(&mut solved[b], Solved { reads, heard, bits, laid: None, among: among[b], solve: solve.clone() });
                out[b] = Some(solve);
            }
            None => hard.push(b),
        }
    }
    let solves = solve_each(&hard, |&b| solve_spec(layout, &layout.blocks[b], pinned, bit_reliability, sampled, None, &opts_of[b], search_of[b], memo));
    let at_most: Vec<DecodeOptions> = opts_of.iter().map(|opts| opts.at_most()).collect();
    let searching = |harder: &Searched, b: usize, solve: &BlockSolve, laid: Option<&Laid>, among: usize| {
        let (reads, heard, bits) = inputs(&layout.blocks[b]);
        trying[b] && !solve.rank().0 && !harder.holds(b, &reads, &heard, &bits, laid, among)
    };
    let harder_of: Vec<usize> = hard.iter().zip(&solves).filter(|&(&b, solve)| searching(harder, b, solve, None, among[b])).map(|(&b, _)| b).collect();
    let found = solve_each(&harder_of, |&b| solve_spec(layout, &layout.blocks[b], pinned, bit_reliability, sampled, None, &at_most[b], search_of[b], memo));
    for (&b, solve) in harder_of.iter().zip(found) {
        let (reads, heard, bits) = inputs(&layout.blocks[b]);
        harder.keep(b, Solved { reads, heard, bits, laid: None, among: among[b], solve });
    }
    for (&b, solve) in hard.iter().zip(solves) {
        let (reads, heard, bits) = inputs(&layout.blocks[b]);
        let solve = harder.better(b, &reads, &heard, &bits, None, among[b], solve);
        remember(&mut solved[b], Solved { reads, heard, bits, laid: None, among: among[b], solve: solve.clone() });
        out[b] = Some(solve);
    }
    let Some(evidence) = evidence else { return out };
    // What the rest of the message makes of a block moves with the pins
    // wherever they are.
    let counted = opts.one_of(picked.all);
    let guided_opts = counted.sharing(2);
    let take = |out: &mut [Option<BlockSolve>], b: usize, mut guided: BlockSolve| match &mut out[b] {
        Some(plain) => {
            hold_together(plain, &mut guided);
            if guided.rank() > plain.rank() {
                *plain = guided;
            }
        }
        None => out[b] = Some(guided),
    };
    let mut again: Vec<(usize, Laid)> = Vec::new();
    for (b, spec) in layout.blocks.iter().enumerate() {
        if out[b].as_ref().is_none_or(|plain| plain.rank().0) {
            continue;
        }
        let (reads, heard, bits) = inputs(spec);
        let Some(laid) = evidence.laid(&spec.codewords, &reads, &bits, opts.reliability) else { continue };
        match recall(&solved[b], &reads, &heard, &bits, Some(&laid), picked.all) {
            Some(solve) => take(&mut out, b, solve),
            None => again.push((b, laid)),
        }
    }
    let guided = |b: usize, laid: &Laid, opts: &DecodeOptions| {
        let mut solve = solve_spec(layout, &layout.blocks[b], pinned, bit_reliability, sampled, Some(laid), opts, guided_of[b], memo);
        if solve.corrected.is_some() {
            solve.message_evidence = laid.moved;
        }
        solve
    };
    let solves = solve_each(&again, |(b, laid)| guided(*b, laid, &guided_opts));
    let at_most_guided = counted.at_most().sharing(2);
    let harder_of: Vec<&(usize, Laid)> =
        again.iter().zip(&solves).filter(|&((b, laid), solve)| searching(harder, *b, solve, Some(laid), picked.all)).map(|(item, _)| item).collect();
    let found = solve_each(&harder_of, |(b, laid)| guided(*b, laid, &at_most_guided));
    for ((b, laid), solve) in harder_of.into_iter().zip(found) {
        let (reads, heard, bits) = inputs(&layout.blocks[*b]);
        harder.keep(*b, Solved { reads, heard, bits, laid: Some(laid.clone()), among: picked.all, solve });
    }
    for ((b, laid), solve) in again.into_iter().zip(solves) {
        let (reads, heard, bits) = inputs(&layout.blocks[b]);
        let solve = harder.better(b, &reads, &heard, &bits, Some(&laid), picked.all, solve);
        remember(&mut solved[b], Solved { reads, heard, bits, laid: Some(laid), among: picked.all, solve: solve.clone() });
        take(&mut out, b, solve);
    }
    out
}

/// One stage of a decode of one grid under one layout and mask
/// ([`DecodeOptions::stages`]), kept so that it can be made again under
/// more pins with only the blocks the pins touch solved anew: what
/// [`decode`] does to pin the padding it finds in a stage
/// ([`DecodeOptions::derive_pins`]).
struct Stage<'a> {
    layout: &'a SymbolLayout,
    mask: u8,
    opts: DecodeOptions,
    has_reliability: bool,
    raw: Vec<CwRead>,
    module_reliability: Vec<u8>,
    bit_reliability: Vec<[u8; 8]>,
    /// The ways each block was solved, from whatever reads and
    /// reliabilities, searched as read or with the message's evidence.
    solved: Vec<Vec<Solved>>,
    /// The blocks searched harder, before this decode or in it.
    harder: Searched,
    /// The blocks this decode searches harder ([`decode_trying_harder`]).
    trying: Vec<bool>,
    /// What the dear solvers found of each block they were given, in
    /// this decode or in another stage of the decode it is one of
    /// ([`Memo`]).
    memo: Arc<Memo>,
}

impl<'a> Stage<'a> {
    /// The stage under `opts`, taking what the dear solvers found as
    /// `memo` holds it, and holding there what they find.
    fn new(grid: &ModuleGrid, layout: &'a SymbolLayout, mask: u8, opts: DecodeOptions, memo: &Arc<Memo>) -> Self {
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
            solved: vec![Vec::new(); layout.blocks.len()],
            harder: Searched::default(),
            trying: vec![false; layout.blocks.len()],
            memo: memo.clone(),
        }
    }

    /// This decode, taking the searches `kept` holds, and searching the
    /// blocks `trying` harder.
    fn trying_harder(mut self, kept: Searched, trying: &[usize]) -> Self {
        self.harder = kept;
        for &b in trying {
            if let Some(trying) = self.trying.get_mut(b) {
                *trying = true;
            }
        }
        self
    }

    /// The grid decoded under `pins`, each search charged for the ways
    /// the pins it reaches were picked among ([`PinsPicked`]): a block
    /// the pins leave as it was, and charged as it was, is the search it
    /// was, and is not solved again; one they touch, or whose evidence
    /// they move, is searched afresh.
    fn solve(&mut self, pins: &Pins) -> DecodeReport {
        let (layout, opts) = (self.layout, self.opts);
        let opts = &opts;
        let raw = &self.raw;
        let pinned = apply_pins(layout, raw, pins);
        let picked = PinsPicked::of(layout, pins);
        let width = layout.func.width;

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

        let learn = self.has_reliability && opts.reliability;
        let mut solves: Vec<BlockSolve> = solve_blocks(
            layout,
            opts,
            &mut self.solved,
            &mut self.harder,
            &self.trying,
            &pinned,
            &self.bit_reliability,
            &self.bit_reliability,
            (true, 0),
            &picked,
            None,
            &|_| true,
            &self.memo,
        )
        .into_iter()
            .map(|solve| solve.expect("every block asked"))
            .collect();
        let mut blocks: Vec<BlockReport> =
            layout.blocks.iter().zip(&solves).map(|(spec, solve)| report_block(layout, spec, solve, raw, &pinned, &mut codewords)).collect();
        // The codewords the message has been asked about as rivals of
        // each block, over every arbitration of this decode.
        let mut asked: Vec<Vec<Vec<u8>>> = vec![Vec::new(); layout.blocks.len()];
        arbitrate(layout, raw, &pinned, &mut solves, &mut codewords, &mut blocks, &mut asked);

        // The blocks that verified say two things of those that did not.
        // Where the photo misread: an error located is a module read as
        // known and overturned, and how often the modules beside one were
        // misread too says how far to doubt the modules beside it in the
        // blocks that did not verify ([`Clustering`]), which only a grid
        // sampled from a photo has to doubt. And what the message holds:
        // the stream parses further, and the characters it reads say what
        // the rest of their kind likely hold ([`MessageEvidence`]). The
        // blocks that did not verify are solved again with both, and what
        // verifies then says more for the next round.
        let mut clustering = None;
        for round in 1..=EVIDENCE_ROUNDS {
            if blocks.iter().all(BlockReport::is_verified) || work::cancelled() {
                break;
            }
            let learnt = if learn { Clustering::learn(layout, raw, &codewords, &blocks, &self.module_reliability) } else { None };
            let lowered = learnt.as_ref().map(|(clustering, beside)| clustering.lowered(&self.module_reliability, beside));
            let evidence = if opts.message_evidence {
                let parsed = parsed(layout, &codewords, &blocks, &pinned);
                MessageEvidence::of(layout, &parsed.stream, &parsed.parse)
            } else {
                None
            };
            if lowered.is_none() && evidence.is_none() {
                break;
            }
            if let Some((learnt, _)) = learnt {
                clustering = Some(learnt);
            }
            let lowered_bits = lowered.as_ref().map(|lowered| bit_reliabilities(layout, lowered));
            let unverified: Vec<bool> = blocks.iter().map(|block| !block.is_verified()).collect();
            let again = solve_blocks(
                layout,
                opts,
                &mut self.solved,
                &mut self.harder,
                &self.trying,
                &pinned,
                lowered_bits.as_deref().unwrap_or(&self.bit_reliability),
                &self.bit_reliability,
                (lowered_bits.is_none(), round),
                &picked,
                evidence.as_ref(),
                &|b| unverified[b],
                &self.memo,
            );
            let mut improved = false;
            // A solve in hand held down by what a search of this round
            // reached, which the decode then reports.
            let mut held = false;
            for (b, spec) in layout.blocks.iter().enumerate() {
                let Some(mut again) = again[b].clone() else { continue };
                if let Some(lowered) = &lowered {
                    again.near_errors = spec
                        .codewords
                        .iter()
                        .flat_map(|&cw| layout.codeword_modules(cw))
                        .filter(|p| lowered[p.y as usize * width + p.x as usize] < self.module_reliability[p.y as usize * width + p.x as usize])
                        .count();
                }
                let (current_held, again_held) = hold_together(&mut solves[b], &mut again);
                let (taken, improves) = weigh_again(&solves[b], &again, &asked[b]);
                if taken {
                    solves[b] = again;
                }
                improved |= improves;
                held |= if taken { again_held } else { current_held };
            }
            if !improved && !held {
                break;
            }
            blocks = layout.blocks.iter().zip(&solves).map(|(spec, solve)| report_block(layout, spec, solve, raw, &pinned, &mut codewords)).collect();
            arbitrate(layout, raw, &pinned, &mut solves, &mut codewords, &mut blocks, &mut asked);
            if !improved {
                break;
            }
        }

        for (b, (spec, block)) in layout.blocks.iter().zip(&mut blocks).enumerate() {
            let reads: Vec<CwRead> = spec.codewords.iter().map(|&cw| pinned[cw]).collect();
            let bits: Vec<[u8; 8]> = spec.codewords.iter().map(|&cw| self.bit_reliability[cw]).collect();
            let heard = heard(&reads, &bits);
            block.tried_harder = self.harder.searched(b, &reads, &heard, &bits, picked.block[b]);
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

        DecodeReport {
            ec: layout.ec,
            mask: self.mask,
            codewords,
            blocks,
            stream,
            parse,
            tail,
            text,
            score,
            derived_pins: Pins::new(),
            clustering,
            tries: opts.tries.max(1),
            pin_tries: picked.all,
        }
    }
}

/// A decode of one grid under one layout and mask, kept so that it can
/// be made again under more pins with only the blocks the pins touch
/// solved anew, as padding recovery and the trials of common beginnings
/// make it: each decode under pins made in the stages [`decode`] makes
/// it in ([`DecodeOptions::stages`]), until one is trustworthy, or else
/// the best ranked, the earlier stage's on a tie, each stage kept to be
/// made again. Its decodes were once made under every option at once,
/// and one of the recovery's ranked otherwise than the decode of the
/// grid under the same pins, once those were pinned.
pub struct Decoding<'a> {
    stages: Vec<Stage<'a>>,
}

impl<'a> Decoding<'a> {
    pub fn new(grid: &ModuleGrid, layout: &'a SymbolLayout, mask: u8, opts: &DecodeOptions) -> Self {
        let memo = Arc::default();
        Self { stages: opts.stages().into_iter().map(|stage| Stage::new(grid, layout, mask, stage, &memo)).collect() }
    }

    /// The grid decoded under `pins` ([`Stage::solve`]), in stages.
    pub fn solve(&mut self, pins: &Pins) -> DecodeReport {
        best_of_stages(self.stages.iter_mut().map(|stage| stage.solve(pins)))
    }
}

/// What a decode made in stages comes to ([`DecodeOptions::stages`]):
/// the decodes of `reports` made in turn until one is trustworthy, which
/// is taken, or else the best ranked, the earlier stage's on a tie.
fn best_of_stages(reports: impl Iterator<Item = DecodeReport>) -> DecodeReport {
    let mut best: Option<DecodeReport> = None;
    for report in reports {
        let settled = report.is_trustworthy();
        if best.as_ref().is_none_or(|held| report.score > held.score) {
            best = Some(report);
        }
        if settled || work::cancelled() {
            break;
        }
    }
    best.expect("every decode has a stage")
}

pub fn decode(grid: &ModuleGrid, layout: &SymbolLayout, mask: u8, pins: &Pins, opts: &DecodeOptions) -> DecodeReport {
    decode_keeping(grid, layout, mask, pins, opts, &Harder::default())
}

/// [`decode`], taking from `kept` the searches of blocks made harder
/// before, where a block reads as it did and the search ranks better
/// ([`Harder`]). `kept` must be of this layout, mask and options.
pub fn decode_keeping(grid: &ModuleGrid, layout: &SymbolLayout, mask: u8, pins: &Pins, opts: &DecodeOptions, kept: &Harder) -> DecodeReport {
    decode_trying_harder(grid, layout, mask, pins, opts, kept, &[]).0
}

/// [`decode_keeping`], with the blocks `trying` searched again, wherever
/// the options leave one unconfirmed, with every dear solver's budget
/// at its most ([`DecodeOptions::at_most`]), the better solution taken:
/// a search that may take a minute on a large block, for the blocks
/// asked about, called off as any work is ([`work::cancelled`]). With
/// `kept` and the searches this decode made, for later decodes to take.
///
/// The decode is made in stages ([`DecodeOptions::stages`]): under each
/// stage's options in turn, its padding derived as those options derive
/// it, until one is trustworthy, which is taken. Short of that, the
/// best ranked is taken, the earlier stage's on a tie. Each stage
/// searches the blocks `trying` harder, and takes and keeps its own
/// searches ([`Harder`]).
pub fn decode_trying_harder(
    grid: &ModuleGrid,
    layout: &SymbolLayout,
    mask: u8,
    pins: &Pins,
    opts: &DecodeOptions,
    kept: &Harder,
    trying: &[usize],
) -> (DecodeReport, Harder) {
    let mut harder = kept.clone();
    let memo = Arc::default();
    let report = best_of_stages(opts.stages().iter().enumerate().map(|(s, stage)| {
        let (report, searched) = decode_stage(grid, layout, mask, pins, stage, kept.stage(s), trying, &memo);
        harder.keep(s, searched);
        report
    }));
    (report, harder)
}

/// One stage of [`decode_trying_harder`]: the decode under `opts` alone,
/// taking the searches `kept` holds, with the searches it made, and
/// taking what the dear solvers found in the stages before it from
/// `memo` and holding there what they find.
#[allow(clippy::too_many_arguments)]
fn decode_stage(
    grid: &ModuleGrid,
    layout: &SymbolLayout,
    mask: u8,
    pins: &Pins,
    opts: &DecodeOptions,
    kept: Searched,
    trying: &[usize],
    memo: &Arc<Memo>,
) -> (DecodeReport, Searched) {
    let mut decoding = Stage::new(grid, layout, mask, *opts, memo).trying_harder(kept, trying);
    let report = decoding.solve(pins);
    let report = if !opts.derive_pins || report.is_trustworthy() || work::cancelled() { report } else { derive_pins(&mut decoding, pins, report) };
    (report, decoding.harder)
}

/// The decode made again with the padding, terminator and headers the
/// stream shows pinned, by padding recovery over the decode in hand,
/// each round solving anew only the blocks its pins touch: taken, with
/// the pins it placed noted, when the pins solve or verify a block more
/// or the segment headers confirm where the padding was placed, and the
/// decode ranks no worse; else the decode as it was. Pins asserted on
/// what the pad pattern alone showed, that solve nothing more, are not
/// taken: they would stand for nothing but themselves.
fn derive_pins(decoding: &mut Stage, pins: &Pins, report: DecodeReport) -> DecodeReport {
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
    /// What the format bits say against this choice's word, in eighths of
    /// a bit of likelihood ([`bch::FormatCandidate::weight`]).
    pub format_weight: u32,
    /// How far the data read under this level and mask is like a message
    /// rather than noise, in bits of likelihood
    /// ([`super::likeness::likeness_bits`]).
    pub likeness_bits: f64,
    /// The grid was reflected this way before decoding: the attempt is of
    /// the symbol's mirror image.
    pub mirror: Option<Mirror>,
    pub report: DecodeReport,
}

impl Attempt {
    /// What the data and the format bits say together for this level and
    /// mask, in bits of likelihood against the bits as read: the data's
    /// likeness to a message, less what the format bits say against the
    /// word. The two are on one scale, and add.
    pub fn odds_bits(&self) -> f64 {
        self.likeness_bits - f64::from(self.format_weight) / 8.0
    }
}

/// Bits of likeness to a message the data must show under some level and
/// mask before the data has a say in ranking them
/// ([`attempt_every_format`]). The likeness is charged already for the
/// runs and alignments it chose among, so that noise comes to nothing;
/// this is a margin on top, a byte of text's worth several times over.
pub const LIKENESS_FLOOR: f64 = 8.0;

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
/// Doubting is dear, erasing codewords by trial dearer, and searching a
/// block again with the message's evidence makes it dearer twice over,
/// and under the wrong mask every block is noise that none of them can
/// help, so a grid is decoded in two passes: every attempt without them,
/// then the few that rank best with them, in the stages a decode is made
/// in ([`DecodeOptions::stages`]). The exact decoder and the bit-level solver,
/// which the first pass keeps, already tell the right mask from the rest
/// when any block of it is clean. Every attempt is one of as many
/// decodes as there are levels and masks ([`DecodeOptions::among`]),
/// and has that share of the dear solvers' budgets.
pub fn brute_force(
    grid: &ModuleGrid,
    func: &Arc<FunctionMap>,
    pins: Option<(EcLevel, &Pins)>,
    opts: &DecodeOptions,
) -> Vec<Attempt> {
    attempt_every_format(grid, func, pins, &opts.among(formats(func)), None)
}

/// [`brute_force`] of a grid read again of a symbol another grid of it
/// was brute forced from, `preferred` the levels and masks that ranked
/// best there: every level and mask decoded as the first pass decodes
/// them, and the second pass, the dear one, made only of those preferred
/// and of the attempt the first pass ranks best here. The level and mask
/// are the symbol's, the same whichever way its modules are read, and
/// under the wrong ones every block is noise that no stage helps; a read
/// of a placement tries its grid under several samplings, and each
/// once brute forced in full spent the most of a read that did not
/// decode on decoding noise in stages. Each attempt is counted as
/// [`brute_force`] counts it, one of every level and mask: which are
/// searched in stages was chosen by the data, of another reading.
pub fn brute_force_preferring(
    grid: &ModuleGrid,
    func: &Arc<FunctionMap>,
    pins: Option<(EcLevel, &Pins)>,
    opts: &DecodeOptions,
    preferred: &[(EcLevel, u8)],
) -> Vec<Attempt> {
    attempt_every_format(grid, func, pins, &opts.among(formats(func)), Some(preferred))
}

/// [`brute_force`] with every attempt given the budgets of `opts` whole
/// ([`DecodeOptions::one_of`]): counted as one of as many decodes as
/// there are levels and masks, for what confirms a solution, but
/// searched as a decode of the one level and mask would be, so that an
/// attempt ranks as it decodes once it is taken.
pub fn brute_force_whole(
    grid: &ModuleGrid,
    func: &Arc<FunctionMap>,
    pins: Option<(EcLevel, &Pins)>,
    opts: &DecodeOptions,
) -> Vec<Attempt> {
    attempt_every_format(grid, func, pins, &opts.one_of(formats(func)), None)
}

/// The levels and masks a brute force of `func`'s version tries: what a
/// decode under the format a brute force settled is one of.
pub fn formats(func: &FunctionMap) -> usize {
    func.version.ec_levels().len() * func.version.mask_count() as usize
}

/// What the format bits say against a level and mask's word: the bits
/// they mismatch in, and the weight of those ([`Attempt::format_weight`]).
type FormatAgainst = (u32, u32);

/// The decodes of [`brute_force`], under `opts` as counted for them,
/// the second pass made of the attempts at the levels and masks
/// `preferred` names and the first attempt where it names any
/// ([`brute_force_preferring`]).
fn attempt_every_format(
    grid: &ModuleGrid,
    func: &Arc<FunctionMap>,
    pins: Option<(EcLevel, &Pins)>,
    opts: &DecodeOptions,
    preferred: Option<&[(EcLevel, u8)]>,
) -> Vec<Attempt> {
    let format = bch::analyze_format(grid, func);
    let no_pins = Pins::new();
    let two_passes =
        (opts.reliability && grid.has_reliability()) || opts.erase_by_trial || opts.list_decode || opts.derive_pins || opts.message_evidence;
    let first_pass = DecodeOptions {
        reliability: opts.reliability && !two_passes,
        erase_by_trial: false,
        list_decode: false,
        derive_pins: false,
        message_evidence: false,
        ..*opts
    };
    let pins_at = |ec: EcLevel| match pins {
        Some((level, pins)) if level == ec => pins,
        _ => &no_pins,
    };
    // Every level and mask, decoded side by side on the cores to spare
    // ([`work::map`]) and taken in their order: each decode stands alone,
    // and on a symbol of a block or two there is nothing else to spread.
    // Each level and mask, with the format bits' mismatches and weight
    // against its word.
    let mut tried: Vec<(EcLevel, u8, FormatAgainst, Arc<SymbolLayout>)> = Vec::new();
    for &ec in func.version.ec_levels() {
        let Some(layout) = SymbolLayout::new(func.clone(), ec) else { continue };
        let layout = Arc::new(layout);
        for mask in 0..func.version.mask_count() {
            let against = format
                .candidates
                .iter()
                .find(|c| c.fits_grid && c.is_standard() && c.fields.ec == ec && c.fields.mask == mask)
                .map_or((0, 0), |c| (c.best_mismatches, c.weight));
            tried.push((ec, mask, against, layout.clone()));
        }
    }
    let reports = work::map(&tried, |(ec, mask, _, layout)| {
        (decode(grid, layout, *mask, pins_at(*ec), &first_pass), likeness_bits(&stream_as_read(grid, layout, *mask)))
    });
    let mut attempts: Vec<(Attempt, Arc<SymbolLayout>)> = tried
        .into_iter()
        .zip(reports)
        .map(|((ec, mask, (format_mismatches, format_weight), layout), (report, likeness_bits))| {
            (Attempt { ec, mask, format_mismatches, format_weight, likeness_bits, mirror: None, report }, layout)
        })
        .collect();
    // Where no block is confirmed under any level and mask, and the data
    // reads as a message under some, the attempts are told apart by the
    // data and the format bits together ([`Attempt::odds_bits`]): under
    // the wrong format every block is noise, and the unconfirmed
    // solutions noise yields, and how many, say nothing of which format
    // is right. Nor does a stream that parses through bits no block
    // confirmed, once a message's likeness is to be had: noise parses so
    // now and then, and of thirty-one wrong formats one did, ahead of the
    // right one, whose stream two hundred bits of likeness spoke for, and
    // whose headers a misread bit had broken. Where a block is confirmed,
    // or the data reads as a message under no level and mask, the
    // attempts are ranked as they decode and then by the format bits
    // ([`ranked_by_data`]).
    let by_data = ranked_by_data(attempts.iter().map(|(a, _)| a));
    attempts.sort_by(|(a, _), (b, _)| rank_attempts(a, b, by_data));
    if two_passes {
        let blocks = attempts.iter().map(|(a, _)| a.report.blocks.len()).max().unwrap_or(1).max(1);
        // An rMQR grid has two attempts, fewer than the minimum: then all of them.
        let retry = (DOUBTING_BUDGET / blocks).max(DOUBTING_MIN_ATTEMPTS).min(attempts.len());
        let again: Vec<usize> = match preferred {
            None => (0..retry).filter(|&i| !attempts[i].0.report.is_trustworthy()).collect(),
            Some(preferred) => (0..attempts.len())
                .filter(|&i| i == 0 || preferred.contains(&(attempts[i].0.ec, attempts[i].0.mask)))
                .filter(|&i| !attempts[i].0.report.is_trustworthy())
                .collect(),
        };
        // In the stages a decode is made in ([`decode_trying_harder`]),
        // each stage of every attempt before the next stage of any, every
        // attempt keeping the best its stages reach as a decode keeps it,
        // until one of them is trustworthy: under the wrong masks every
        // stage reads noise, and a stage past the one the right mask
        // reads in would be a decode of each of them for nothing.
        let again: Vec<(usize, Arc<Memo>)> = again.into_iter().map(|i| (i, Arc::default())).collect();
        for (s, stage) in opts.stages().iter().enumerate() {
            let reports = work::map(&again, |(i, memo)| {
                let (attempt, layout) = &attempts[*i];
                decode_stage(grid, layout, attempt.mask, pins_at(attempt.ec), stage, Searched::default(), &[], memo).0
            });
            for (&(i, _), report) in again.iter().zip(reports) {
                let held = &mut attempts[i].0.report;
                if s == 0 || report.score > held.score {
                    *held = report;
                }
            }
            if again.iter().any(|&(i, _)| attempts[i].0.report.is_trustworthy()) || work::cancelled() {
                break;
            }
        }
        // Ranked again, by the data only if the doubting confirmed no
        // block.
        let by_data = ranked_by_data(attempts.iter().map(|(a, _)| a));
        attempts.sort_by(|(a, _), (b, _)| rank_attempts(a, b, by_data));
    }
    attempts.into_iter().map(|(attempt, _)| attempt).collect()
}

/// Whether `attempts`, every level and mask of a grid, are ranked by the
/// data they read as and the format bits together ([`rank_attempts`]):
/// where no block is confirmed under any of them, and the data reads as
/// a message under some ([`LIKENESS_FLOOR`]). A block confirmed anywhere
/// says more of which is right than the data does, and whether the data
/// reads as a message under some other level and mask, which a patch of
/// one colour once made it seem to, says nothing of how two attempts the
/// blocks tell apart are to be ranked.
fn ranked_by_data<'a>(attempts: impl Iterator<Item = &'a Attempt> + Clone) -> bool {
    !attempts.clone().any(|a| a.report.score.blocks_verified > 0) && attempts.into_iter().any(|a| a.likeness_bits >= LIKENESS_FLOOR)
}

/// How two attempts at a grid rank, the better first, as
/// [`attempt_every_format`] ranks them: with `by_data` ([`ranked_by_data`]),
/// by the data and the format bits together ([`Attempt::odds_bits`]);
/// then as each decodes, every field of [`AttemptScore`] in its turn, and
/// last by the format bits' mismatches. Attempts with a block confirmed
/// are never ranked by the data: of two clean attempts at two levels
/// whose blocks are laid out alike, both confirmed, the one that checked
/// more confirming bits is the real one, which the data, read alike
/// under both, does not say.
fn rank_attempts(a: &Attempt, b: &Attempt, by_data: bool) -> std::cmp::Ordering {
    let told = if by_data { b.odds_bits().total_cmp(&a.odds_bits()) } else { std::cmp::Ordering::Equal };
    told.then(b.report.score.cmp(&a.report.score)).then(a.format_mismatches.cmp(&b.format_mismatches))
}

/// Decode the grid's mirror images under every EC level and mask, best
/// first, as ZXing and ZBar retry a failed decode with the symbol
/// transposed. The format bits of a mirrored grid usually say so, but
/// damage takes the corners they live in first; the blocks that verify
/// say so whatever the state of the format bits. Pins do not carry over:
/// they name codewords of the grid as it stands.
pub fn brute_force_mirrored(grid: &ModuleGrid, func: &Arc<FunctionMap>, opts: &DecodeOptions) -> Vec<Attempt> {
    mirror_every_way(grid, func, opts, brute_force)
}

/// [`brute_force_mirrored`] with every attempt given the budgets of
/// `opts` whole, as [`brute_force_whole`] gives them: to set beside a
/// decode, or a brute force, of the grid as it stands that has them
/// whole, which the mirror images were once held to less than.
pub fn brute_force_mirrored_whole(grid: &ModuleGrid, func: &Arc<FunctionMap>, opts: &DecodeOptions) -> Vec<Attempt> {
    mirror_every_way(grid, func, opts, brute_force_whole)
}

/// A decode of a grid under every level and mask: [`brute_force`] or
/// [`brute_force_whole`].
type BruteForce = fn(&ModuleGrid, &Arc<FunctionMap>, Option<(EcLevel, &Pins)>, &DecodeOptions) -> Vec<Attempt>;

/// The attempts of `brute_force` on each of the grid's mirror images,
/// best first, each counted as one of the mirror images
/// ([`DecodeOptions::one_of`]): the best of them is taken, and each is a
/// fresh chance for noise to pass, which went uncounted once. They are
/// ranked together as [`attempt_every_format`] ranks those of one grid
/// ([`rank_attempts`]): where no block of any mirror image is confirmed
/// and the data reads as a message under some level and mask of one, by
/// the data and the format bits.
fn mirror_every_way(grid: &ModuleGrid, func: &Arc<FunctionMap>, opts: &DecodeOptions, brute_force: BruteForce) -> Vec<Attempt> {
    let mut attempts = Vec::new();
    let opts = &opts.one_of(func.version.mirrors().len());
    for &mirror in func.version.mirrors() {
        let reflected = grid.mirrored(mirror);
        for mut attempt in brute_force(&reflected, func, None, opts) {
            attempt.mirror = Some(mirror);
            attempts.push(attempt);
        }
    }
    let by_data = ranked_by_data(attempts.iter());
    attempts.sort_by(|a, b| rank_attempts(a, b, by_data));
    attempts
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A candidate for `corrected`, solved with `confidence` bits on its
    /// own and nothing yet held against it.
    fn candidate(corrected: Vec<u8>, confidence: u32) -> Candidate {
        let outcome = BlockOutcome::Solved {
            method: SolveMethod::ReedSolomon,
            errors: 0,
            erasures: 0,
            slack: 0,
            confidence_bits: confidence,
            risk_bits: confidence,
            beyond_standard: false,
        };
        Candidate { alone: confidence, held: confidence, ..Candidate::new(outcome, corrected, Vec::new(), confidence, 0.0) }
    }

    /// `candidate`, the block as read lying `distance` from it.
    fn placed(candidate: Candidate, distance: u32) -> Candidate {
        Candidate { distance, ..candidate }
    }

    #[test]
    fn budgets_shared_in_turn_are_shared_once() {
        // A decode one of four, and that one of eight, as a photo's read
        // is one of its placements and then of its samplings, has what
        // one of thirty-two has: the trials' and the curve's budgets were
        // once shared out over the four again, which the floors hide at
        // the defaults and a budget raised by hand shows.
        let raised = DecodeOptions { list_cost: LIST_COST_MAX, trial_sets: TRIAL_SETS_MAX, ..DecodeOptions::default() };
        for opts in [DecodeOptions::default(), raised] {
            assert_eq!(opts.among(4).among(8), opts.among(32));
        }
        assert_eq!(raised.among(4).among(8).list_cost, LIST_COST_MAX / 32);
        // The most a block of such a decode is searched with is shared out
        // as its own budgets are.
        assert_eq!(raised.among(32).at_most(), raised.at_most().among(32));
        assert_eq!(DecodeOptions::default().at_most().trial_sets, TRIAL_SETS_MAX);
    }

    #[test]
    fn a_texture_that_runs_as_the_mask_does_is_read_and_warned_of() {
        // Stripes a row apart, as a screen's rows of pixels make, read
        // through mask 1: every data module the mask's own colour, so the
        // data is the zero codeword, which its error correction confirms,
        // and the stream ends before it begins with zeros after it. That
        // is an empty message an encoder may have padded with zeros, and
        // it stands, warned of.
        let version = SymbolVersion::Qr(2);
        let func = Arc::new(FunctionMap::new(version));
        let layout = SymbolLayout::new(func.clone(), EcLevel::H).expect("2-H");
        let n = version.width();
        let stripes: Vec<bool> = (0..n * n).map(|i| crate::symbol::mask::mask_bit(version, 1, i % n, i / n)).collect();
        let report = decode(&ModuleGrid::from_bools(n, n, &stripes), &layout, 1, &Pins::new(), &DecodeOptions::default());
        assert!(report.is_trustworthy(), "the zero codeword is a codeword");
        assert!(report.parse.segments.is_empty());
        assert!(report.data_is_the_mask());
        // An empty message padded as the standard pads it, 0xEC and 0x11
        // in turn, is no mask's pattern, and is not warned of.
        let symbol = qrcode_generator::qr::Encoder::new(qrcode_generator::qr::ErrorCorrection::High)
            .version(qrcode_generator::qr::Version::new(2).unwrap())
            .boost_error_correction(false)
            .encode_segments(&[])
            .unwrap();
        let grid = ModuleGrid::from_bools(symbol.width(), symbol.height(), symbol.modules());
        let fields = analyze(&grid, &func).format.best().cloned().expect("the format reads").fields;
        assert_eq!(fields.ec, EcLevel::H);
        let report = decode(&grid, &layout, fields.mask, &Pins::new(), &DecodeOptions::default());
        assert!(report.is_trustworthy() && report.parse.segments.is_empty());
        assert!(!report.data_is_the_mask());
    }

    #[test]
    fn a_block_searched_harder_is_never_forgotten() {
        // A decode searching a block harder searches it under each round's
        // reliabilities and evidence, and again under each set of padding
        // pins it tries: more searches than a decode remembers of its own
        // solves. The first, of the block as read, was once forgotten first,
        // and the block said not to have been searched.
        let mut harder = Searched::default();
        let bits = vec![[CERTAIN; 8]];
        let reads = |i: usize| vec![CwRead::known(i as u8)];
        for i in 0..3 * REMEMBERED {
            let solve = BlockSolve::solved(candidate(vec![i as u8], 10));
            harder.keep(1, Solved { reads: reads(i), heard: reads(i), bits: bits.clone(), laid: None, among: 1, solve });
        }
        assert!((0..3 * REMEMBERED).all(|i| harder.searched(1, &reads(i), &reads(i), &bits, 1)));
        assert!(!harder.searched(0, &reads(0), &reads(0), &bits, 1) && !harder.searched(1, &reads(3 * REMEMBERED), &reads(3 * REMEMBERED), &bits, 1));
        // Charged for pins picked among more ways, it is another search.
        assert!(!harder.searched(1, &reads(0), &reads(0), &bits, 2));
        // A module read at a coin's toss then, and read now, is another
        // search, which the reserve counts otherwise.
        assert!(!harder.searched(1, &reads(0), &[CwRead::unknown()], &bits, 1));
    }

    #[test]
    fn rivals_are_the_codewords_as_near_as_the_nearest_whatever_their_rank() {
        // Three codewords: one the block as read lies far from, which
        // ranks best, and two it lies near, of which the one that ranks
        // best of all stands last in the pool. The rivals are the two
        // near ones, in their rank's order; the far one, though it ranks
        // between them, is none. The rivals once took their distances by
        // their place in the pool after the pool was sorted by rank, and
        // here the far one was taken for a rival and the nearer of the
        // two near ones was not.
        let far = placed(candidate(vec![0xF0], 40), 200);
        let near_worst = placed(candidate(vec![0x33], 4), 30);
        let near_best = placed(candidate(vec![0x0F], 20), 10);
        let mut distinct = vec![far, near_worst, near_best];
        hold_against_each_other(&mut distinct, &[]);
        // Each is held to its margin over the nearest other: the near
        // ones are twenty apart, two bits, and the far one is held to
        // nothing by both.
        let risks: Vec<(u8, u32)> = distinct.iter().map(|c| (c.corrected[0], c.outcome.risk_bits())).collect();
        assert_eq!(risks, [(0x0F, 2), (0xF0, 0), (0x33, 0)]);
        let rivals = rivals_among(&distinct, &[]);
        assert_eq!(rivals.iter().map(|c| c.corrected[0]).collect::<Vec<_>>(), [0x0F, 0x33]);
        // Nothing as near was left out, so each stands on its own risk
        // once the message has told them apart, but for the chance that
        // the far one is the block's: twenty-three bits off the nearer,
        // an eighth of its own twenty's chance, and twenty-one off the
        // other, a hundred and twenty-eighth of its four's.
        assert_eq!(rivals.iter().map(|c| c.held).collect::<Vec<_>>(), [19, 4]);
        // One codeword alone, or two the block as read tells apart, are
        // no rivals.
        assert!(rivals_among(&distinct[..1], &[]).is_empty());
        let apart = vec![placed(candidate(vec![1], 20), 10), placed(candidate(vec![2], 20), 10 + DISTANCE_MARGIN)];
        assert!(rivals_among(&apart, &[]).is_empty());
    }

    #[test]
    fn a_codeword_as_near_but_left_out_of_the_rivals_holds_them_down_still() {
        // One codeword more than the rivals may number, every one as near
        // the block as read as the rest, a bit apart: the worst ranked
        // is left out, and each rival stands on no more than its margin
        // over it once the message has told the rivals apart.
        let mut distinct: Vec<Candidate> = (0..=MAX_RIVALS as u32).map(|i| placed(candidate(vec![i as u8], 40 - i), 10 + i)).collect();
        hold_against_each_other(&mut distinct, &[]);
        let rivals = rivals_among(&distinct, &[]);
        assert_eq!(rivals.len(), MAX_RIVALS);
        assert_eq!(rivals.iter().map(|c| c.corrected[0] as usize).collect::<Vec<_>>(), (0..MAX_RIVALS).collect::<Vec<_>>());
        for (i, rival) in rivals.iter().enumerate() {
            let left_out = 10 + MAX_RIVALS as u32;
            assert_eq!(rival.held, (left_out - (10 + i as u32)) / 8, "rival {i}");
            assert_eq!(rival.alone, 40 - i as u32, "rival {i}");
        }
        // With the one left out gone, every rival stands on its own.
        distinct.pop();
        assert!(rivals_among(&distinct, &[]).iter().all(|c| c.held == c.alone));
    }

    #[test]
    fn a_block_solved_again_improves_the_round_by_its_rank_or_by_a_rival_not_yet_asked_about() {
        let solve = |corrected: Vec<u8>, confidence: u32, rivals: Vec<Candidate>| BlockSolve { rivals, ..BlockSolve::solved(candidate(corrected, confidence)) };
        // The solve in hand, unverified, its rivals emptied by the
        // arbitration that asked the message about them.
        let current = solve(vec![1], 4, Vec::new());
        let asked = vec![vec![1u8], vec![2]];
        // Recalled from the rounds before: the same rank, and the rivals
        // the message was asked about. Taken, so that it may be
        // arbitrated again under a message other blocks have moved, and
        // no improvement by itself.
        let recalled = solve(vec![1], 4, vec![candidate(vec![1], 4), candidate(vec![2], 3)]);
        assert_eq!(weigh_again(&current, &recalled, &asked), (true, false));
        // A rival the message was never asked about is an improvement,
        // as every rival is before the first arbitration.
        let fresh = solve(vec![1], 4, vec![candidate(vec![1], 4), candidate(vec![3], 3)]);
        assert_eq!(weigh_again(&current, &fresh, &asked), (true, true));
        assert_eq!(weigh_again(&current, &recalled, &[]), (true, true));
        // A rank risen is an improvement with rivals or without; a rank
        // fallen is not taken whatever rivals it brings; and the same
        // rank without rivals is nothing.
        assert_eq!(weigh_again(&current, &solve(vec![1], 6, Vec::new()), &asked), (true, true));
        assert_eq!(weigh_again(&current, &solve(vec![1], 2, vec![candidate(vec![3], 2)]), &asked), (false, false));
        assert_eq!(weigh_again(&current, &solve(vec![1], 4, Vec::new()), &asked), (false, false));
    }

    /// Blocks of noise through every solver of [`solve_block`], and the
    /// risk each solution it keeps claims: a claim of `b` bits is that a
    /// block of noise gets so far no oftener than once in `2^b`. Read
    /// whole or with codewords partly unread, sampled with reliabilities
    /// spread over the scale or mostly faint, or painted by hand. Run by
    /// hand, with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn survey_of_noise_through_every_solver() {
        use rayon::prelude::*;
        let blocks: u64 = std::env::var("SURVEY_BLOCKS").ok().and_then(|n| n.parse().ok()).unwrap_or(400);
        let mut worst = 0f64;
        for (n, ec_len) in [(26usize, 7usize), (26, 13), (26, 17), (33, 18), (46, 28), (70, 15)] {
            for erased in [0, ec_len / 2, ec_len - 2] {
                for kind in ["spread", "faint", "painted"] {
                    let risks: Vec<Option<(u32, SolveMethod)>> = (0..blocks)
                        .into_par_iter()
                        .map(|i| {
                            let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (i << 20) ^ ((n as u64) << 8) ^ ec_len as u64 ^ ((erased as u64) << 40));
                            for _ in 0..4 {
                                rng.next();
                            }
                            let mut read: Vec<CwRead> = (0..n).map(|_| CwRead::known(rng.byte())).collect();
                            let mut unread = 0;
                            while unread < erased {
                                let i = rng.below(n);
                                if read[i].is_erasure() {
                                    continue;
                                }
                                // A bit or three unread, as a stain over
                                // part of a codeword leaves it.
                                for _ in 0..1 + rng.below(3) {
                                    read[i].known &= !(1 << rng.below(8));
                                }
                                read[i].value &= read[i].known;
                                unread += 1;
                            }
                            let bits: Vec<[u8; 8]> = (0..n)
                                .map(|_| {
                                    std::array::from_fn(|_| match kind {
                                        "spread" => rng.byte().min(CERTAIN - 1),
                                        "faint" => rng.byte() % 48,
                                        _ => CERTAIN,
                                    })
                                })
                                .collect();
                            let opts = DecodeOptions::default();
                            let search = Search { as_read: true, nth: 1, ways: ways(SymbolVersion::Qr(5)), picked: 1 };
                            let solve = solve_block(&read, &heard(&read, &bits), &bits, None, ec_len, 0, &opts, search, &Memo::default());
                            match solve.outcome {
                                BlockOutcome::Solved { risk_bits, method, .. } => Some((risk_bits, method)),
                                BlockOutcome::Failed(_) => None,
                            }
                        })
                        .collect();
                    let solved = risks.iter().flatten().count();
                    let mut line = format!("({n:3}, {ec_len:2}) {erased:2} unread, {kind:7}: {solved:4} of {blocks} solved;");
                    for b in [1u32, 2, 4, 6, 8, 10, 12, 16, 24] {
                        let at_least = risks.iter().flatten().filter(|(r, _)| *r >= b).count();
                        let expected = blocks as f64 / f64::from(b).exp2();
                        worst = worst.max(at_least as f64 / expected.max(1.0));
                        line += &format!(" ≥{b}: {at_least}");
                        if at_least as f64 > 2.0 * expected + 3.0 {
                            line += "!";
                        }
                    }
                    let most = risks.iter().flatten().max_by_key(|(r, _)| *r);
                    println!("{line}; most {most:?}");
                }
            }
        }
        println!("worst ratio of claims passed to claims allowed: {worst:.2}");
    }

    #[test]
    fn a_codeword_past_the_tie_but_near_a_far_rival_holds_that_rival_down() {
        // The block as read lies 10 from X, 66 from Y and 74 from Z: X
        // and Y tie, and Z does not, lying past the margin from X. Yet
        // Z lies a bit from Y, and the message, asked to tell X from Y,
        // is never shown Z. Taken, Y stands on no more than that bit,
        // and X on its margin over Z with its own chance added: eight
        // bits and twelve, a little under eight.
        let mut distinct = vec![placed(candidate(vec![1], 12), 10), placed(candidate(vec![2], 20), 66), placed(candidate(vec![3], 20), 74)];
        hold_against_each_other(&mut distinct, &[]);
        let rivals = rivals_among(&distinct, &[]);
        assert_eq!(rivals.iter().map(|c| (c.corrected[0], c.held)).collect::<Vec<_>>(), [(1, 7), (2, 1)]);
        // With Z gone, each stands on its own.
        distinct.retain(|c| c.corrected[0] != 3);
        assert_eq!(rivals_among(&distinct, &[]).iter().map(|c| c.held).collect::<Vec<_>>(), [12, 20]);
    }
    #[test]
    fn a_codeword_no_solver_offered_holds_down_those_offered() {
        // One solution offered, forty bits sure on its own, and two
        // codewords a solver reached and kept back, ten and eleven bits
        // further from the block as read: the solution stands on their
        // chances and its own added, not on its own alone, as it did when
        // what was kept back was forgotten, nor on the nearer alone.
        let mut distinct = vec![placed(candidate(vec![1], 40), 16)];
        hold_against_each_other(&mut distinct, &[16 + 80, 16 + 88]);
        assert_eq!(distinct[0].outcome.risk_bits(), 9);
        // One the block as read lies nearer to leaves it nothing.
        hold_against_each_other(&mut distinct, &[8]);
        assert_eq!(distinct[0].outcome.risk_bits(), 0);
        // As rivals, the ones kept back hold each down as one left out
        // for their number does: six bits from the first, and five from
        // the second, two eighths of a bit nearer.
        let mut tied = vec![placed(candidate(vec![1], 30), 10), placed(candidate(vec![2], 30), 12)];
        hold_against_each_other(&mut tied, &[10 + 48]);
        let rivals = rivals_among(&tied, &[10 + 48]);
        assert_eq!(rivals.iter().map(|c| (c.corrected[0], c.held)).collect::<Vec<_>>(), [(1, 6), (2, 5)]);
    }

    /// A 5-Q block of `wrong` codewords each read wrong in one faint bit,
    /// every other bit read well, with the codeword it is.
    fn faintly_misread(wrong: usize) -> (Vec<CwRead>, Vec<[u8; 8]>, Vec<u8>) {
        use super::super::rs::rs_encode;
        let mut rng = Rng(0x5EED);
        let mut codeword: Vec<u8> = (0..15).map(|_| rng.byte()).collect();
        codeword.extend(rs_encode(&codeword, 18));
        let mut read: Vec<CwRead> = codeword.iter().map(|&c| CwRead::known(c)).collect();
        let mut bits = vec![[56u8; 8]; codeword.len()];
        for i in (0..33).step_by(3).take(wrong) {
            let b = rng.below(8);
            read[i].value ^= 1 << b;
            bits[i][b] = 20;
        }
        (read, bits, codeword)
    }

    #[test]
    fn a_codeword_the_block_verifies_stands_on_that_whatever_found_it() {
        // Eleven faint bits overturned, past the nine the code corrects: a
        // codeword the block itself verifies by the soft count many times
        // over. Found by a solver
        // whose own count is small and whose tries were many, it stands
        // on what the block says of it, charged a bit for the choice
        // between the counts and nothing for the tries or the solvers of
        // the search; once it was charged for both, sixteen bits and five.
        let (read, bits, codeword) = faintly_misread(11);
        let unknown = 0;
        let soft = soft_count_bits(&read, &bits, 8 * 18 - unknown, soft_distance(&read, &bits, &codeword));
        assert!(soft > 60, "{soft}");
        let chased = |errors, beyond_standard| BlockOutcome::Solved {
            method: SolveMethod::Chase { flipped_bits: 3 },
            errors,
            erasures: 0,
            slack: 0,
            confidence_bits: 0,
            risk_bits: 0,
            beyond_standard,
        };
        let as_read = Charges { grid: 0.0, pins: 0.0, share: 4.4, judged: 0.0, as_read: true };
        let weighed = |tried: f64, charges: Charges| {
            let candidate = Candidate::new(chased(11, false), codeword.clone(), Vec::new(), 9, tried);
            candidate.weigh(&read, &read, &bits, 18, 0, charges).alone
        };
        assert_eq!(weighed(16.0, as_read), soft - 1);
        assert_eq!(weighed(0.0, as_read), soft - 1);
        // Charged for the decodes of the grid, and for a search other
        // than the block as read its share, the fourth of five, as many
        // ways as it is made: a bit for two decodes, and four and a
        // little, taken to the bit above once with the choice's.
        assert_eq!(weighed(16.0, Charges { grid: 2f64.log2(), ..as_read }), soft - 2);
        let lowered = Charges { judged: (nth_of(4, SEARCHES_MAX) * 2.0).log2(), as_read: false, ..as_read };
        assert_eq!(weighed(16.0, lowered), soft - 1 - whole_charge(lowered.judged));
        // Charged past verifying, the block's count still stands for what
        // it comes to, the more of the two.
        assert_eq!(weighed(16.0, Charges { grid: f64::from(soft - 4), ..as_read }), 3);
        // Held to the codewords the standard reserves as the standard
        // counts them: eleven wrong spend the one a block keeps, however
        // rarely noise lies so near, and the count of the block stands for
        // nothing; eight leave it unused, and the codeword is within the
        // standard's limit, whatever its solver spent to reach it.
        let candidate = Candidate::new(chased(11, true), codeword.clone(), Vec::new(), 9, 16.0).weigh(&read, &read, &bits, 18, 1, as_read);
        assert!(!candidate.verified_alone());
        let (read, bits, codeword) = faintly_misread(8);
        let candidate = Candidate::new(chased(8, true), codeword.clone(), Vec::new(), 9, 16.0).weigh(&read, &read, &bits, 18, 1, as_read);
        assert!(candidate.verified_alone());
    }

    #[test]
    fn the_decode_of_the_pins_alone_is_charged_for_their_picking_and_not_the_grid() {
        // Every codeword of its block read wrong, the decode of the pins
        // alone stands on its own count: charged two bits for pins picked
        // among four ways, and nothing for the decodes of the grid, which
        // change nothing of it. It was once charged for neither.
        let (mut read, bits, codeword) = faintly_misread(0);
        for cw in &mut read {
            cw.value ^= 0xFF;
        }
        let pinned = BlockOutcome::Solved {
            method: SolveMethod::PinnedOnly,
            errors: 0,
            erasures: 18,
            slack: 5,
            confidence_bits: 40,
            risk_bits: 0,
            beyond_standard: false,
        };
        let charges = Charges { grid: 5.0, pins: 2.0, share: 0.0, judged: 0.0, as_read: true };
        let candidate = Candidate::new(pinned, codeword, Vec::new(), 40, 0.0).weigh(&read, &read, &bits, 18, 0, charges);
        assert_eq!(candidate.alone, 38);
    }

    #[test]
    fn the_decode_of_the_pins_alone_keeps_the_reserve_and_takes_what_others_confirm_of_its_codeword() {
        // The pins name the codeword within the standard and confirm it by
        // three bits; a chase reaches it past the standard's limit as the
        // block as read counts it, by twelve. The codeword stands as the
        // pins' own, within the standard, at twelve, and is verified.
        let (read, bits, codeword) = faintly_misread(0);
        let solved = |method, beyond_standard| BlockOutcome::Solved {
            method,
            errors: 0,
            erasures: 0,
            slack: 0,
            confidence_bits: 0,
            risk_bits: 0,
            beyond_standard,
        };
        let alone = |outcome, bits: u32| Candidate { alone: bits, ..Candidate::new(outcome, codeword.clone(), Vec::new(), 0, 0.0) }.at_risk(bits);
        let pinned = alone(solved(SolveMethod::PinnedOnly, false), 3);
        let chased = alone(solved(SolveMethod::Chase { flipped_bits: 3 }, true), 12);
        for pool in [vec![pinned.clone(), chased.clone()], vec![chased.clone(), pinned.clone()]] {
            let distinct = distinct_codewords(pool, &read, &bits);
            assert_eq!(distinct.len(), 1);
            assert!(matches!(
                distinct[0].outcome,
                BlockOutcome::Solved { method: SolveMethod::PinnedOnly, beyond_standard: false, risk_bits: 12, .. }
            ));
            assert!(distinct[0].verified_alone());
        }
        // Another solution within the standard is ranked against it as any.
        let within = alone(solved(SolveMethod::Chase { flipped_bits: 3 }, false), 12);
        let distinct = distinct_codewords(vec![pinned, within], &read, &bits);
        assert!(matches!(distinct[0].outcome, BlockOutcome::Solved { method: SolveMethod::Chase { .. }, risk_bits: 12, .. }));
    }

    #[test]
    fn the_reserve_is_counted_against_the_block_as_read_whatever_the_evidence_leans() {
        // A 3-L block, fifteen codewords of error correction and one of
        // them kept: six bytes of text unread and five codewords read
        // wrong are past what the standard corrects, six erasures and ten
        // for the errors against fourteen. The message's evidence leans
        // the high three bits of each unread byte toward text, and the
        // block a search with it is judged by takes them at their lean
        // ([`Laid::judged`]). They were then counted as read for the
        // reserve, and cost nothing where the solution bore them out:
        // five eighths a byte for the bits left, 110 eighths in all with
        // the errors, where 112 keep the reserve. Counted against the
        // block as read they are unread, and it is spent.
        use super::super::rs::rs_encode;
        let mut rng = Rng(0x3E5E);
        let mut codeword: Vec<u8> = (0..55).map(|_| 0x20 + rng.below(0x5F) as u8).collect();
        codeword.extend(rs_encode(&codeword, 15));
        let mut read: Vec<CwRead> = codeword.iter().map(|&c| CwRead::known(c)).collect();
        read[..6].fill(CwRead::unknown());
        for i in [10, 20, 30, 40, 60] {
            read[i].value ^= 0x5A;
        }
        let mut bits = vec![[56u8; 8]; codeword.len()];
        let mut judged = read.clone();
        for i in 0..6 {
            judged[i] = CwRead { value: codeword[i] & 0xE0, known: 0xE0, pinned: 0 };
            bits[i] = [0, 0, 0, 0, 0, 20, 20, 20];
        }
        let solved = BlockOutcome::Solved {
            method: SolveMethod::BitLevel,
            errors: 5,
            erasures: 6,
            slack: 0,
            confidence_bits: 0,
            risk_bits: 0,
            beyond_standard: false,
        };
        let charges = Charges { grid: 0.0, pins: 0.0, share: 0.0, judged: 0.0, as_read: false };
        let beyond = |block: &[CwRead], read: &[CwRead]| {
            let candidate = Candidate::new(solved, codeword.clone(), Vec::new(), 0, 0.0).weigh(block, read, &bits, 15, 1, charges);
            matches!(candidate.outcome, BlockOutcome::Solved { beyond_standard: true, .. })
        };
        assert!(beyond(&judged, &read));
        assert!(beyond(&read, &read));
        assert!(!beyond(&judged, &judged), "the leaned bits taken for read keep the reserve");
    }

    #[test]
    fn the_reserve_counts_what_was_read_however_a_round_lowered_it() {
        // A 1-M block, ten codewords of error correction and two of them
        // kept: five codewords read wrong, a solid bit each, are past the
        // four the standard corrects and within the five the exact decoder
        // reaches, spending the reserve. A round that learnt the modules
        // beside errors to be misread lowers them, a solid one to nought
        // where most are ([`Clustering::lowered`]). Counted from those, the
        // five wrong bits were unread, an eighth of a codeword each, and
        // the block kept the reserve and verified.
        use super::super::rs::rs_encode;
        let version = SymbolVersion::Qr(1);
        let layout = SymbolLayout::new(Arc::new(FunctionMap::new(version)), EcLevel::M).expect("1-M");
        let spec = &layout.blocks[0];
        assert_eq!((spec.ec_len, spec.reserved), (10, 2));
        let mut rng = Rng(0x10E5);
        let mut codeword: Vec<u8> = (0..spec.data_len).map(|_| rng.byte()).collect();
        codeword.extend(rs_encode(&codeword, spec.ec_len));
        let mut pinned = vec![CwRead::unknown(); layout.codewords.len()];
        let mut sampled = vec![[CERTAIN; 8]; layout.codewords.len()];
        for (i, &cw) in spec.codewords.iter().enumerate() {
            pinned[cw] = CwRead::known(codeword[i]);
            sampled[cw] = [200; 8];
        }
        let mut lowered = sampled.clone();
        for &i in &[0, 5, 10, 15, 20] {
            let cw = spec.codewords[i];
            pinned[cw].value ^= 1 << 3;
            lowered[cw][3] = 0;
        }
        let opts = DecodeOptions::default();
        let search = Search { as_read: false, nth: 2, ways: ways(version), picked: 1 };
        let beyond = |sampled: &[[u8; 8]]| {
            let solve = solve_spec(&layout, spec, &pinned, &lowered, sampled, None, &opts, search, &Memo::default());
            assert_eq!(solve.corrected.as_deref(), Some(&codeword[..]));
            matches!(solve.outcome, BlockOutcome::Solved { beyond_standard: true, .. }) && !solve.outcome.is_verified()
        };
        assert!(beyond(&sampled));
        assert!(!beyond(&lowered), "counted from the lowered bits, the reserve is kept");
        // The exact decoder's solve alone, which a sound block stops at.
        let reads: Vec<CwRead> = spec.codewords.iter().map(|&cw| pinned[cw]).collect();
        let bits_of = |bits: &[[u8; 8]]| -> Vec<[u8; 8]> { spec.codewords.iter().map(|&cw| bits[cw]).collect() };
        let exact = |sampled: &[[u8; 8]]| exact_solve(&reads, &heard(&reads, &bits_of(sampled)), &bits_of(&lowered), 10, 2, &opts, search);
        assert!(exact(&sampled).is_none());
        assert!(exact(&lowered).is_some());
    }

    #[test]
    fn a_codeword_reached_and_kept_back_is_a_solution_where_the_block_verifies_it() {
        // Nothing offered, and the block's own codeword among those a
        // solver reached: the block itself verifies it, and it is a
        // solution of its own, where the block once failed with it in hand.
        let (read, bits, codeword) = faintly_misread(11);
        let as_read = Charges { grid: 0.0, pins: 0.0, share: 4.4, judged: 0.0, as_read: true };
        let mut seen = Sightings::default();
        seen.see(&codeword);
        let mut distinct = Vec::new();
        promote_reached(&mut distinct, &seen, &read, &read, &bits, 18, 0, as_read);
        assert_eq!(distinct.len(), 1);
        assert!(matches!(distinct[0].outcome, BlockOutcome::Solved { method: SolveMethod::Reached, .. }) && distinct[0].verified_alone());
        assert_eq!(distinct[0].distance, soft_distance(&read, &bits, &codeword));
        // Offered already, it is not offered twice.
        promote_reached(&mut distinct, &seen, &read, &read, &bits, 18, 0, as_read);
        assert_eq!(distinct.len(), 1);
        let mut other = codeword.clone();
        other[..15].copy_from_slice(&[0x11; 15]);
        let ec = super::super::rs::rs_encode(&other[..15], 18);
        other[15..].copy_from_slice(&ec);
        // A codeword further from the block as read than one offered by
        // the tie or more is left to hold it down, which is all it could
        // do, though the block verifies it: the block's own, with another
        // offered as though the tie nearer. Within the tie it is offered.
        // This was once asked of a codeword the block does not verify,
        // which is never offered however near, and showed nothing.
        let near = soft_distance(&read, &bits, &codeword);
        assert!(near >= DISTANCE_MARGIN, "{near}");
        for (nearer_by, offered) in [(DISTANCE_MARGIN, false), (DISTANCE_MARGIN - 1, true)] {
            let mut held = vec![Candidate { corrected: other.clone(), distance: near - nearer_by, ..distinct[0].clone() }];
            promote_reached(&mut held, &seen, &read, &read, &bits, 18, 0, as_read);
            assert_eq!(held.iter().any(|c| c.corrected == codeword), offered, "nearer by {nearer_by}");
        }
        // A codeword the block does not verify is never offered.
        let mut far = Sightings::default();
        far.see(&other);
        let mut none = Vec::new();
        promote_reached(&mut none, &far, &read, &read, &bits, 18, 0, as_read);
        assert!(none.is_empty());
    }

    #[test]
    fn several_codewords_nearly_as_near_hold_a_solution_down_together() {
        // Four codewords each nine bits further from the block as read
        // than the one ranked best: each alone leaves it nine bits, and
        // the four together seven, their chances added.
        let mut distinct = vec![placed(candidate(vec![0], 30), 10)];
        distinct.extend((1..=4u8).map(|i| placed(candidate(vec![i], 4), 10 + 72)));
        hold_against_each_other(&mut distinct, &[]);
        assert_eq!((distinct[0].corrected[0], distinct[0].outcome.risk_bits()), (0, 7));
        assert_eq!(held_against(30, 10, [82]), 9);
    }
    #[test]
    fn a_solve_is_held_against_what_another_search_of_its_block_reached() {
        // One search of a block solved it as read, thirty bits sure; one
        // made otherwise reached a codeword a bit further off, read a
        // byte inside its colour, and took it twenty bits sure. Each is
        // held against the other's: the first stands on the nine bits by
        // which the block as read lies nearer it, and the second on
        // nothing, the block lying nearer the first. Held so once, the two
        // have reached the same codewords, and holding them again changes
        // nothing.
        let block = vec![CwRead::known(0x00)];
        let bits = vec![[64u8; 8]];
        let mut first = BlockSolve::solved(candidate(vec![0x00], 30)).framed(&block, &bits, &Sightings::default());
        let mut second = BlockSolve::solved(candidate(vec![0x01], 20)).framed(&block, &bits, &Sightings::default());
        assert_eq!(hold_together(&mut first, &mut second), (true, true));
        assert_eq!((first.outcome.risk_bits(), second.outcome.risk_bits()), (9, 0));
        assert_eq!(hold_together(&mut first, &mut second), (false, false));
        assert_eq!(first.reached, second.reached);
    }

    #[test]
    fn a_solution_that_takes_another_of_its_codewords_place_keeps_the_distance() {
        // Two solvers reach one codeword, the later ranking above the
        // earlier, and a third reaches another. The one kept for the
        // codeword carries how far the block as read lies from it, as the
        // other does: what the message's arbitration measures its margins
        // by. The one that took the first's place once carried nought.
        let block = vec![CwRead::known(0x00)];
        let bits = vec![[64u8; 8]];
        let pool = vec![candidate(vec![0x03], 10), candidate(vec![0x01], 4), candidate(vec![0x03], 20)];
        let distinct = distinct_codewords(pool, &block, &bits);
        assert_eq!(distinct.iter().map(|c| (c.corrected[0], c.alone)).collect::<Vec<_>>(), [(0x03, 20), (0x01, 4)]);
        for candidate in &distinct {
            assert_eq!(candidate.distance, soft_distance(&block, &bits, &candidate.corrected), "{:#04x}", candidate.corrected[0]);
            assert!(candidate.distance > 0, "{:#04x}", candidate.corrected[0]);
        }
    }

    #[test]
    fn the_ways_the_pins_were_picked_among_are_charged_only_where_they_reach() {
        // A clean version 5-Q symbol of four blocks, a codeword of the
        // first pinned as read, the pick of a thousand ways, and one of
        // the second, the pick of four: the first block is charged ten
        // bits for its pin, the second two for its own, and the other
        // two, whose plain search as read is the same whichever way was
        // picked, nothing. Each was once charged as the first is: one
        // count stood for every pin in force, and a beginning pinned in
        // one block left the rest short of what they had stood on, and
        // those holding pins picked before it shorter.
        use crate::decode::codewords::PinSource;
        use qrcode_generator::{Segment, qr};
        let (layout, grid, mask) = clean(5, qr::ErrorCorrection::Quartile, Segment::bytes("the ways the pins were picked among"));
        assert_eq!(layout.blocks.len(), 4);
        let decode_under = |pins: &Pins, opts: &DecodeOptions| decode(&grid, &layout, mask, pins, opts);
        let plain = decode_under(&Pins::new(), &DecodeOptions::default());
        assert!(plain.is_trustworthy());
        let pinned = |among: [usize; 2]| -> Pins {
            (0..2)
                .map(|b| {
                    let cw = layout.blocks[b].codewords[0];
                    (cw, Pin { among: among[b], ..Pin::full(plain.codewords[cw].final_value.unwrap(), PinSource::Text) })
                })
                .collect()
        };
        let risks = |report: &DecodeReport| report.blocks.iter().map(|b| b.outcome.risk_bits()).collect::<Vec<_>>();
        let once = decode_under(&pinned([1, 1]), &DecodeOptions::default());
        assert_eq!(once.pin_tries, 1);
        let once = risks(&once);
        let picked = decode_under(&pinned([1000, 4]), &DecodeOptions::default());
        assert_eq!((picked.tries, picked.pin_tries), (1, 1000));
        let picked = risks(&picked);
        assert_eq!(picked[..2], [once[0] - tries_charge(1000), once[1] - tries_charge(4)], "{once:?} {picked:?}");
        assert_eq!(picked[2..], once[2..]);
    }

    #[test]
    fn noise_in_place_of_a_rivals_data_keeps_the_bits_pinned_there() {
        // A clean version 2 symbol, its first data codeword, the byte
        // segment's mode and the top of its count, pinned in all but its
        // lowest bit. A rival differing from the winner there differs in
        // that bit alone, and noise in its place is the winner's byte half
        // the time: the message says next to nothing for the winner. The
        // fillings once overturned the pinned bits too, the mode with
        // them, and parsed as well all but never.
        use qrcode_generator::{Segment, qr};
        let (layout, grid, mask) = clean(2, qr::ErrorCorrection::Low, Segment::bytes("forensics"));
        let report = decode(&grid, &layout, mask, &Pins::new(), &DecodeOptions::default());
        assert!(report.is_trustworthy());
        let cw = layout.data_stream[0];
        let mut pinned: Vec<CwRead> = report.codewords.iter().map(|c| CwRead::known(c.final_value.unwrap())).collect();
        let score = StreamScore::of(&parsed(&layout, &report.codewords, &report.blocks, &pinned));
        let bits = |pinned: &[CwRead]| message_bits(&layout, pinned, &report.codewords, &report.blocks, &[cw], score, 0x9E37_79B9_7F4A_7C15);
        let unpinned = bits(&pinned);
        assert!(unpinned > 6.0, "{unpinned}");
        pinned[cw].pinned = 0xFE;
        let kept = bits(&pinned);
        assert!(kept < 2.0, "{kept}");
    }

    /// [`message_bits`] as it once was: the stream laid anew from the
    /// codeword reports for every filling, the filling put in place in
    /// them and taken out after.
    fn message_bits_laid_anew(
        layout: &SymbolLayout,
        pinned: &[CwRead],
        codewords: &mut [CodewordReport],
        blocks: &[BlockReport],
        differ: &[usize],
        score: StreamScore,
        mut seed: u64,
    ) -> f64 {
        let held: Vec<Option<u8>> = differ.iter().map(|&cw| codewords[cw].final_value).collect();
        let (mut drawn, mut as_well) = (0u32, 0u32);
        while drawn < MESSAGE_SAMPLES && as_well < MESSAGE_ENOUGH {
            for &cw in differ {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let pin = pinned[cw];
                codewords[cw].final_value = Some((seed >> 24) as u8 & !pin.pinned | pin.value & pin.pinned);
            }
            as_well += u32::from(StreamScore::of(&parsed(layout, codewords, blocks, pinned)) >= score);
            drawn += 1;
        }
        for (&cw, value) in differ.iter().zip(held) {
            codewords[cw].final_value = value;
        }
        (f64::from(drawn + 1) / f64::from(as_well + 1)).log2()
    }

    #[test]
    fn a_filling_laid_over_the_stream_parses_as_the_stream_laid_anew_with_it() {
        // Bytes, where noise parses as well as the message, and digits,
        // where it seldom does and every filling is drawn; one block and
        // four, interleaved, the data a rival differs in taken from the
        // first block and from the last, a codeword of it and several, with
        // the pins of a known beginning and without. The stream laid once
        // and each filling laid over it says what laying it anew with each
        // filling said, to the filling.
        use qrcode_generator::{Segment, qr};
        let mut cases = 0;
        for (v, ec, segment) in [
            (2, qr::ErrorCorrection::Low, Segment::bytes("forensics")),
            (5, qr::ErrorCorrection::Quartile, Segment::numeric("31415926535897932384626433832795028841971693993751").unwrap()),
        ] {
            let (layout, grid, mask) = clean(v, ec, segment);
            let report = decode(&grid, &layout, mask, &Pins::new(), &DecodeOptions::default());
            assert!(report.is_trustworthy(), "version {v}");
            let mut pinned: Vec<CwRead> = report.codewords.iter().map(|c| CwRead::known(c.final_value.unwrap())).collect();
            let score = StreamScore::of(&parsed(&layout, &report.codewords, &report.blocks, &pinned));
            for spec in [layout.blocks.first().unwrap(), layout.blocks.last().unwrap()] {
                for count in [1, 3, spec.data_len] {
                    let differ = &spec.codewords[..count];
                    for pin in [0x00, 0xF0] {
                        pinned[differ[0]].pinned = pin;
                        let laid_over = message_bits(&layout, &pinned, &report.codewords, &report.blocks, differ, score, 0x2545_F491);
                        let mut codewords = report.codewords.clone();
                        let laid_anew = message_bits_laid_anew(&layout, &pinned, &mut codewords, &report.blocks, differ, score, 0x2545_F491);
                        assert_eq!(laid_over, laid_anew, "version {v}, {count} codewords, pinned {pin:#04x}");
                        cases += 1;
                    }
                    pinned[differ[0]].pinned = 0;
                }
            }
        }
        assert_eq!(cases, 24);
    }

    #[test]
    fn a_solver_that_goes_by_reliability_is_counted_only_where_it_has_a_bit_to_take() {
        // A block painted by hand but for one faint bit, in a codeword
        // read in part: the doubting of bits, the flips of solid bits and
        // the propagation take it, and the chase and the doubting of
        // codewords, which look only at the codewords read in full, have
        // nothing, and are not counted. Every faint bit was once counted
        // for all of them.
        let opts = DecodeOptions::default();
        let mut block = vec![CwRead::known(0x5A); 4];
        let mut bits = vec![[CERTAIN; 8]; 4];
        block[1] = CwRead { value: 0x50, known: 0xF0, pinned: 0 };
        bits[1][7] = 40;
        let in_part = Faint::of(&block, &bits);
        assert_eq!(in_part, Faint { bits: true, read_whole: false, unpinned_whole: false });
        assert_eq!(routes(&opts, false, Faint::sampled(true)) - routes(&opts, false, in_part), 2);
        // Read in full, it is the chase's and the doubting of codewords'
        // too; and with another bit of its codeword pinned, the chase's
        // alone, the doubting of codewords leaving a pinned codeword be.
        block[1] = CwRead::known(0x50);
        assert_eq!(Faint::of(&block, &bits), Faint::sampled(true));
        block[1].pinned = 0x01;
        assert_eq!(Faint::of(&block, &bits), Faint { bits: true, read_whole: true, unpinned_whole: false });
        // Pinned, the bit is none of theirs.
        block[1].pinned = 0x80;
        assert_eq!(Faint::of(&block, &bits), Faint::sampled(false));
    }

    #[test]
    fn the_flips_the_trials_and_the_evidence_are_each_a_stage_of_their_own() {
        let all = DecodeOptions::default();
        let ways = |opts: DecodeOptions| {
            opts.stages().iter().map(|stage| (stage.flip_bits, stage.erase_by_trial, stage.message_evidence)).collect::<Vec<_>>()
        };
        let (f, t) = (false, true);
        assert_eq!(ways(all), [(f, f, f), (f, f, t), (t, f, f), (t, f, t), (f, t, f), (f, t, t), (t, t, f), (t, t, t)]);
        assert_eq!(ways(DecodeOptions { erase_by_trial: false, ..all }), [(f, f, f), (f, f, t), (t, f, f), (t, f, t)]);
        assert_eq!(ways(DecodeOptions { flip_bits: false, ..all }), [(f, f, f), (f, f, t), (f, t, f), (f, t, t)]);
        assert_eq!(ways(DecodeOptions { message_evidence: false, ..all }), [(f, f, f), (t, f, f), (f, t, f), (t, t, f)]);
        assert_eq!(ways(DecodeOptions { flip_bits: false, erase_by_trial: false, message_evidence: false, ..all }), [(f, f, f)]);
        // Without reliability the flips do nothing, and are no stage.
        assert_eq!(ways(DecodeOptions { reliability: false, ..all }), [(f, f, f), (f, f, t), (f, t, f), (f, t, t)]);
        for opts in [all, DecodeOptions { reliability: false, ..all }, DecodeOptions { erase_by_trial: false, message_evidence: false, ..all }] {
            assert_eq!(opts.stages()[0], opts.first_stage());
        }
        // The stages of options with any of the three switched off are
        // stages of the options with it on, in their order: what a decode
        // with it off reads, a decode with it on reads in the same stage.
        let every: Vec<(bool, bool, bool)> = (0..8).map(|i| (i & 1 != 0, i & 2 != 0, i & 4 != 0)).collect();
        let of = |(flip_bits, erase_by_trial, message_evidence): (bool, bool, bool)| {
            DecodeOptions { flip_bits, erase_by_trial, message_evidence, ..all }.stages()
        };
        for &fewer in &every {
            for &more in &every {
                if (fewer.0 && !more.0) || (fewer.1 && !more.1) || (fewer.2 && !more.2) {
                    continue;
                }
                let (fewer, more) = (of(fewer), of(more));
                let at: Vec<usize> = fewer.iter().map(|stage| more.iter().position(|s| s == stage).expect("a stage of fewer is one of more")).collect();
                assert!(at.is_sorted(), "{at:?}");
            }
        }
    }

    #[test]
    fn switching_on_the_flips_the_trials_or_the_evidence_never_loses_a_read() {
        // Version 2-M symbols read from a photo, a few modules misread
        // faintly, one to three solidly and a score not read. Made as one
        // decode, as a decode once was, the flips or the trials switched
        // on once lost what the others read without them, every solution
        // charged for the solvers they added, the counts that judge a
        // codeword by the block itself among them; charged for no solver
        // ([`Search::judged_share`]), one decode with more switched on
        // reads what it read with fewer. The evidence once lost the
        // flips' read and the trials' too, a solution found in the block
        // as read being charged for the rounds of evidence that might
        // follow it; charged for its own search's share alone
        // ([`Search::share`]), it keeps them. In stages, every read made
        // with fewer of them is made with more, and so is the brute
        // force's. Of the blocks of version 1-L these were once, which
        // reserves three codewords, none is within what the standard
        // corrects, as it counts a solution's errors.
        let way = |flips: bool, trials: bool, evidence: bool| usize::from(flips) | usize::from(trials) << 1 | usize::from(evidence) << 2;
        let with = |i: usize| DecodeOptions { flip_bits: i & 1 != 0, erase_by_trial: i & 2 != 0, message_evidence: i & 4 != 0, ..DecodeOptions::default() };
        let more = |a: usize, b: usize| a != b && a & !b == 0;
        let (f, t) = (false, true);
        // Each case: what a decode reads with one of them alone, and what
        // it does not.
        let cases = [
            (0, vec![way(t, f, f), way(f, t, f)], vec![way(f, f, f)]),
            (40, vec![way(t, f, f)], vec![way(f, t, f)]),
            (62, vec![way(f, t, f)], vec![way(t, f, f)]),
            (22, vec![way(f, f, f)], vec![]),
        ];
        for (seed, read, unread) in cases {
            let (layout, grid, mask) = damaged(2, qrcode_generator::qr::ErrorCorrection::Medium, seed);
            let pins = Pins::new();
            let made: Vec<DecodeReport> = (0..8).map(|i| decode_stage(&grid, &layout, mask, &pins, &with(i), Searched::default(), &[], &Arc::default()).0).collect();
            let once: Vec<bool> = made.iter().map(DecodeReport::is_trustworthy).collect();
            assert!(made.iter().filter(|report| report.is_trustworthy()).all(|report| report.text.plain() == "forensics"), "seed {seed}");
            for i in read {
                assert!(once[i], "seed {seed}: {once:?}");
            }
            for i in unread {
                assert!(!once[i], "seed {seed}: {once:?}");
            }
            for fewer in 0..8 {
                for with_more in (0..8).filter(|&b| more(fewer, b)) {
                    assert!(!once[fewer] || once[with_more], "seed {seed}: one decode {once:?}");
                }
            }
            let staged: Vec<bool> = (0..8).map(|i| decode(&grid, &layout, mask, &pins, &with(i)).is_trustworthy()).collect();
            let forced: Vec<bool> =
                (0..8).map(|i| brute_force(&grid, &layout.func, None, &with(i)).first().is_some_and(|attempt| attempt.report.is_trustworthy())).collect();
            for fewer in 0..8 {
                for with_more in (0..8).filter(|&b| more(fewer, b)) {
                    assert!(!staged[fewer] || staged[with_more], "seed {seed}: {staged:?}");
                    assert!(!forced[fewer] || forced[with_more], "seed {seed}: brute force {forced:?}");
                }
            }
            assert!(staged[7], "seed {seed}");
        }
    }

    /// Xorshift, so that the damage and the noise are the same at every
    /// run.
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
    }

    /// A symbol of version `v` at `level` holding `segment`, painted by
    /// hand: with its layout and mask, as its format bits settle them.
    fn clean(v: u8, level: qrcode_generator::qr::ErrorCorrection, segment: qrcode_generator::Segment) -> (SymbolLayout, ModuleGrid, u8) {
        use qrcode_generator::qr;
        let symbol = qr::Encoder::new(level)
            .version(qr::Version::new(v).unwrap())
            .boost_error_correction(false)
            .encode_segments(&[segment])
            .unwrap();
        let grid = ModuleGrid::from_bools(symbol.width(), symbol.height(), symbol.modules());
        let func = Arc::new(FunctionMap::new(SymbolVersion::Qr(v)));
        let best = analyze(&grid, &func).format.best().cloned().unwrap();
        (SymbolLayout::new(func, best.fields.ec).unwrap(), grid, best.fields.mask)
    }

    #[test]
    fn attempts_with_a_block_confirmed_rank_as_they_decode_whatever_the_data_says() {
        use std::cmp::Ordering;
        // A decode with its block confirmed and its stream parsing, as an
        // attempt checked by more confirming bits than another and one
        // the data and the format bits favour over it: of two levels whose
        // blocks are laid out alike, the one checked more is the real
        // one, and the data read under both says nothing of which. Ranked
        // by the data and the format bits first, the other once came
        // first.
        let (layout, grid, mask) = clean(1, qrcode_generator::qr::ErrorCorrection::Medium, qrcode_generator::Segment::bytes("forensics"));
        let report = decode(&grid, &layout, mask, &Pins::new(), &DecodeOptions::default());
        assert!(report.score.blocks_verified == 1 && report.score.all_verified && report.score.stream_valid, "{:?}", report.score);
        let attempt = |more_bits: u32, blocks_verified: usize, likeness_bits: f64| Attempt {
            ec: layout.ec,
            mask,
            format_mismatches: 0,
            format_weight: 0,
            likeness_bits,
            mirror: None,
            report: DecodeReport {
                score: AttemptScore {
                    confidence_bits: report.score.confidence_bits + more_bits,
                    blocks_verified,
                    all_verified: blocks_verified > 0,
                    ..report.score
                },
                ..report.clone()
            },
        };
        let (surer, likelier) = (attempt(8, 1, 10.0), attempt(0, 1, 200.0));
        assert!(!ranked_by_data([&surer, &likelier].into_iter()));
        assert_eq!(rank_attempts(&surer, &likelier, false), Ordering::Less);
        // Of attempts with no block confirmed under any format, the data
        // and the format bits rank first, the decode after.
        let (noise, text) = (attempt(8, 0, 10.0), attempt(0, 0, 200.0));
        assert!(ranked_by_data([&noise, &text].into_iter()));
        assert_eq!(rank_attempts(&text, &noise, true), Ordering::Less);
        assert_eq!(rank_attempts(&text, &noise, false), Ordering::Greater);
        // A block confirmed under one format, and the data reading as a
        // message under another: every attempt is ranked as it decodes,
        // as where the data reads as a message nowhere. The data once had
        // its say wherever it read as one, and put every attempt with a
        // block confirmed before every one whose stream parsed with none,
        // the other way round from where it read as none.
        let parsing = attempt(0, 0, 0.0);
        assert!(parsing.report.score.stream_valid);
        let confirmed = attempt(0, 1, 0.0);
        let blocked = Attempt { report: DecodeReport { score: AttemptScore { stream_valid: false, ..confirmed.report.score }, ..confirmed.report.clone() }, ..confirmed };
        for elsewhere in [0.0, 200.0] {
            let other = attempt(0, 0, elsewhere);
            let by_data = ranked_by_data([&parsing, &blocked, &other].into_iter());
            assert!(!by_data, "{elsewhere}");
            assert_eq!(rank_attempts(&parsing, &blocked, by_data), Ordering::Less, "{elsewhere}");
        }
    }

    /// A symbol of version `v` at `level` holding `forensics`, sampled
    /// from a photo with every module read well, and then, by `seed`, as
    /// many of its data modules as twice its error correction codewords
    /// read wrong faintly at the most, one to half as many wrong
    /// solidly, and as many as four times as many not read: with its
    /// layout and mask.
    pub(super) fn damaged(v: u8, level: qrcode_generator::qr::ErrorCorrection, seed: u64) -> (SymbolLayout, ModuleGrid, u8) {
        let (layout, mut grid, mask) = clean(v, level, qrcode_generator::Segment::bytes("forensics"));
        let ec: usize = layout.blocks.iter().map(|block| block.ec_len).sum();
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        for y in 0..grid.height() {
            for x in 0..grid.width() {
                grid.set_sampled(x, y, grid.get(x, y), 150 + rng.below(100) as u8);
            }
        }
        let module = |rng: &mut Rng| {
            let modules = layout.codeword_modules(rng.below(layout.codewords.len()));
            let at = modules[rng.below(modules.len())];
            (at.x as usize, at.y as usize)
        };
        for _ in 0..rng.below(2 * ec + 1) {
            let (x, y) = module(&mut rng);
            grid.set_sampled(x, y, grid.get(x, y).inverted(), rng.below(60) as u8);
        }
        for _ in 0..1 + rng.below(ec / 2) {
            let (x, y) = module(&mut rng);
            grid.set_sampled(x, y, grid.get(x, y).inverted(), 200);
        }
        for _ in 0..rng.below(4 * ec + 1) {
            let (x, y) = module(&mut rng);
            grid.set(x, y, crate::symbol::grid::Module::Unknown);
        }
        (layout, grid, mask)
    }
}
