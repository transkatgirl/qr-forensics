//! Grid to text: read codewords, solve each RS block as far as possible,
//! assemble the data stream with per-bit certainty, and parse it.

use std::sync::Arc;

use super::bch::{self, FormatAnalysis, VersionAnalysis};
use super::bitstream::{self, Certainty, EndOfData, ParseReport, StreamBits};
use super::codewords::{Pin, Pins, apply_pins, codeword_bit_mask, read_codewords};
use super::padding::{self, TailCheck};
use super::rs::{
    CwRead, RsFailure, VERIFIED_BITS, rs_decode_chasing, rs_decode_doubting, rs_decode_reserving,
    solve_bit_erasures, solve_bit_erasures_doubting,
};
use super::text::{self, Charset, TextReport};
use crate::symbol::EcLevel;
use crate::symbol::function_map::{FunctionMap, POLARITY_MIN_KNOWN, looks_inverted};
use crate::symbol::grid::{CERTAIN, Mirror, ModuleGrid, Symmetry};
use crate::symbol::layout::SymbolLayout;
use crate::symbol::tables;

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
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self { charset: Charset::Auto, bit_level: true, reliability: true }
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
    /// tried, and the one solution the patterns that solved agreed on
    /// kept, with this many bits flipped.
    Chase { flipped_bits: usize },
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
        /// Independent bits confirming the solution. Zero means any input
        /// would have produced a "solution".
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
}

impl BlockReport {
    pub fn is_solved(&self) -> bool {
        matches!(self.outcome, BlockOutcome::Solved { .. })
    }

    /// Solved with at least a byte's worth of confirmation, within the
    /// standard's limit.
    pub fn is_verified(&self) -> bool {
        matches!(
            self.outcome,
            BlockOutcome::Solved { confidence_bits, beyond_standard: false, .. } if confidence_bits >= VERIFIED_BITS
        )
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
/// stream reads as noise. A wrong placement manages the latter, with the
/// fallbacks that erase all but one codeword of a block's redundancy and
/// are confirmed by the eight bits left; BoofCV takes a symbol as read
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

struct BlockSolve {
    outcome: BlockOutcome,
    corrected: Option<Vec<u8>>,
    /// When the block did not solve: bits settled anyway, as (value, mask)
    /// per codeword.
    inferred: Vec<(u8, u8)>,
}

impl BlockSolve {
    fn solved(outcome: BlockOutcome, corrected: Vec<u8>) -> Self {
        let inferred = vec![(0, 0); corrected.len()];
        Self { outcome, corrected: Some(corrected), inferred }
    }
}

/// How block solutions rank: verified solutions within the standard first,
/// then by confirmation. A failure ranks below every solution.
fn solve_rank(outcome: &BlockOutcome) -> (bool, u32) {
    let verified = matches!(
        *outcome,
        BlockOutcome::Solved { confidence_bits, beyond_standard: false, .. } if confidence_bits >= VERIFIED_BITS
    );
    (verified, outcome.confidence_bits())
}

/// Solve one block, trying the cheap exact decoder first and then every
/// fallback, keeping the solution with the most confirmation. `reserved`
/// EC codewords are the standard's misdecode protection: a solution that
/// spends them is reported, but marked as beyond the standard, and ranks
/// below any verified solution within it.
///
/// An exact solution ends the search only when it is verified within the
/// standard. One with nothing to spare, or one that spent the reserved
/// codewords, is kept as a last resort while the fallbacks look for
/// better: doubting the one wrong codeword that read faintly turns a
/// zero-slack solution into a confirmed one.
fn solve_block(
    block: &[CwRead],
    reliability: &[u8],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    reserved: usize,
    opts: &DecodeOptions,
) -> BlockSolve {
    let mut best: Option<(BlockOutcome, Vec<u8>)> = None;
    fn consider(best: &mut Option<(BlockOutcome, Vec<u8>)>, outcome: BlockOutcome, corrected: Vec<u8>) {
        if best.as_ref().is_none_or(|(b, _)| solve_rank(&outcome) > solve_rank(b)) {
            *best = Some((outcome, corrected));
        }
    }

    let failure = match rs_decode_reserving(block, ec_len, reserved) {
        Ok(solution) => {
            let outcome = BlockOutcome::Solved {
                method: SolveMethod::ReedSolomon,
                errors: solution.errors,
                erasures: solution.erasures,
                slack: solution.slack,
                confidence_bits: solution.confidence_bits(),
                beyond_standard: !solution.within_standard(),
            };
            if solve_rank(&outcome).0 {
                return BlockSolve::solved(outcome, solution.corrected);
            }
            consider(&mut best, outcome, solution.corrected);
            RsFailure::BudgetExceeded
        }
        Err(failure) => failure,
    };

    let erased = block.iter().filter(|c| c.is_erasure()).count();
    let mut partial = None;
    if opts.bit_level {
        match solve_bit_erasures(block, ec_len) {
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
                consider(&mut best, outcome, solution.corrected);
            }
            Ok(solution) => partial = Some(solution),
            Err(_) => {}
        }
    }

    if opts.reliability
        && let Some((doubted, solution)) = rs_decode_doubting(block, reliability, ec_len, reserved)
    {
        let outcome = BlockOutcome::Solved {
            method: SolveMethod::Reliability { doubted },
            errors: solution.errors,
            erasures: solution.erasures,
            slack: solution.slack,
            confidence_bits: solution.confidence_bits(),
            beyond_standard: !solution.within_standard(),
        };
        consider(&mut best, outcome, solution.corrected);
    }

    // Doubting bits rather than codewords reaches damage spread over more
    // codewords than there are EC codewords, where no erasure decoder can
    // follow. It competes with doubting codewords on confirmation: a
    // codeword solution with one of slack has one syndrome byte behind
    // it, which a wrong codeword passes once in a few hundred tries, and
    // the bit solution, when there is one, has every spare equation.
    if opts.reliability
        && opts.bit_level
        && let Some(doubt) = solve_bit_erasures_doubting(block, bit_reliability, ec_len)
    {
        let outcome = BlockOutcome::Solved {
            method: SolveMethod::BitReliability { doubted_bits: doubt.doubted_bits },
            errors: doubt.errors(),
            erasures: erased,
            slack: 0,
            confidence_bits: doubt.confidence_bits(),
            beyond_standard: doubt.solution.spare_equations < 8 * reserved,
        };
        consider(&mut best, outcome, doubt.solution.corrected);
    }

    // Flipping the faintest bits outright costs no redundancy where
    // doubting a codeword costs a unit and doubting a bit an equation: a
    // block wrong in a few more codewords than the code corrects is
    // reached when the extra ones read faintly, after doubting has run
    // out of erasures. Every pattern that solves must agree, since many
    // tries are many chances of a misdecode.
    if opts.reliability
        && let Some((flipped_bits, solution)) = rs_decode_chasing(block, bit_reliability, ec_len, reserved)
    {
        let outcome = BlockOutcome::Solved {
            method: SolveMethod::Chase { flipped_bits },
            errors: solution.errors,
            erasures: solution.erasures,
            slack: solution.slack,
            confidence_bits: solution.confidence_bits(),
            beyond_standard: !solution.within_standard(),
        };
        consider(&mut best, outcome, solution.corrected);
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
            consider(&mut best, outcome, solution.corrected);
        }
    }

    if let Some((outcome, corrected)) = best {
        return BlockSolve::solved(outcome, corrected);
    }
    let inferred = match partial {
        Some(solution) => solution.corrected.iter().zip(&solution.solved).map(|(&v, &m)| (v & m, m)).collect(),
        None => vec![(0, 0); block.len()],
    };
    BlockSolve { outcome: BlockOutcome::Failed(failure), corrected: None, inferred }
}

pub fn decode(
    grid: &ModuleGrid,
    layout: &SymbolLayout,
    mask: u8,
    pins: &Pins,
    opts: &DecodeOptions,
) -> DecodeReport {
    let raw = read_codewords(grid, layout, mask);
    let pinned = apply_pins(layout, &raw, pins);
    // Each bit is as reliable as the module it was read from; bit `b` of
    // a codeword is its module `7 - b` (less the bits before its first
    // module), and a bit the symbol does not hold, in a four-bit
    // codeword, is certain. A codeword as a whole is as reliable as its
    // least reliable module.
    let bit_reliability: Vec<[u8; 8]> = (0..raw.len())
        .map(|cw| {
            let mut bits = [CERTAIN; 8];
            let offset = layout.codewords[cw].bit_offset as usize;
            for (i, p) in layout.codeword_modules(cw).iter().enumerate() {
                bits[7 - offset - i] = grid.reliability(p.x as usize, p.y as usize);
            }
            bits
        })
        .collect();
    let reliability: Vec<u8> = bit_reliability.iter().map(|bits| *bits.iter().min().unwrap_or(&CERTAIN)).collect();

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
            status: CwStatus::Unresolved,
        })
        .collect();

    let mut blocks = Vec::with_capacity(layout.blocks.len());
    for spec in &layout.blocks {
        let block: Vec<CwRead> = spec.codewords.iter().map(|&cw| pinned[cw]).collect();
        let block_reliability: Vec<u8> = spec.codewords.iter().map(|&cw| reliability[cw]).collect();
        let block_bits: Vec<[u8; 8]> = spec.codewords.iter().map(|&cw| bit_reliability[cw]).collect();
        let mut solve = solve_block(&block, &block_reliability, &block_bits, spec.ec_len, spec.reserved, opts);
        // The final four-bit codeword of an M1 or M3 symbol enters error
        // correction widened to a byte, its missing nibble the standard's
        // zeros. An encoder in the field computes its error correction
        // over the nibble filled with ones instead, which leaves a block
        // of M3 one codeword wrong, and its redundancy corrects that, but
        // a block of M1, which detects errors and corrects none, cannot
        // solve. So a block that did not solve with confirmation is tried
        // again with the nibble filled with ones, and the better solution
        // kept.
        let mut half_padded_with_ones = false;
        if tables::has_half_codeword(layout.version()) && !solve_rank(&solve.outcome).0 {
            let i = spec.data_len - 1;
            let missing = !codeword_bit_mask(layout, spec.codewords[i]);
            let mut filled = block.clone();
            filled[i].value |= missing;
            let again = solve_block(&filled, &block_reliability, &block_bits, spec.ec_len, spec.reserved, opts);
            if solve_rank(&again.outcome) > solve_rank(&solve.outcome) {
                solve = again;
                half_padded_with_ones = true;
            }
        }
        for (i, &cw) in spec.codewords.iter().enumerate() {
            let exists = codeword_bit_mask(layout, cw);
            let report = &mut codewords[cw];
            let read_in_full = raw[cw].known & exists == exists;
            match &solve.corrected {
                Some(values) => {
                    report.final_value = Some(values[i]);
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
        blocks.push(BlockReport {
            data_len: spec.data_len,
            ec_len: spec.ec_len,
            reserved: spec.reserved,
            outcome: solve.outcome,
            erased_codewords: block.iter().filter(|c| c.is_erasure()).count(),
            unknown_bits: block.iter().map(|c| c.known.count_zeros() as usize).sum(),
            inferred_bits: solve.inferred.iter().map(|(_, m)| m.count_ones() as usize).sum(),
            half_padded_with_ones,
        });
    }

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
    let text = text::decode_text_of(layout.version().family(), &parse.segments, opts.charset);

    let blocks_solved = blocks.iter().filter(|b| b.is_solved()).count();
    let blocks_verified = blocks.iter().filter(|b| b.is_verified()).count();
    let errata = codewords.iter().filter(|c| matches!(c.status, CwStatus::Corrected | CwStatus::Filled)).count();
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

    DecodeReport { ec: layout.ec, mask, codewords, blocks, stream, parse, tail, text, score }
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
/// Doubting is dear, and under the wrong mask every block is noise that
/// doubting cannot help, so a grid with sampled modules is decoded in two
/// passes: every attempt without doubting, then the few that rank best
/// with it. The exact decoder and the bit-level solver, which the first
/// pass keeps, already tell the right mask from the rest when any block
/// of it is clean.
pub fn brute_force(
    grid: &ModuleGrid,
    func: &Arc<FunctionMap>,
    pins: Option<(EcLevel, &Pins)>,
    opts: &DecodeOptions,
) -> Vec<Attempt> {
    let format = bch::analyze_format(grid, func);
    let no_pins = Pins::new();
    let two_passes = opts.reliability && grid.has_reliability();
    let first_pass = DecodeOptions { reliability: opts.reliability && !two_passes, ..*opts };
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
