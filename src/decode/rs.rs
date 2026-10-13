//! Reed-Solomon over GF(256) as used by QR, Micro QR and rMQR: generator
//! roots α^0..α^(r-1), first codeword is the highest-degree coefficient.
//!
//! Four solvers are provided. [`rs_decode`] is the classical
//! errors-and-erasures decoder, where any codeword with an unknown bit is an
//! erasure. [`solve_bit_erasures`] treats each unknown *bit* as an unknown
//! over GF(2), which is far cheaper when damage is scattered, but assumes
//! every known bit is correct. [`rs_decode_doubting`] goes the other way:
//! it assumes the least reliable codewords may be wrong and erases them
//! until the block solves, and [`solve_bit_erasures_doubting`] does the
//! same one bit at a time, which spends one equation per doubted bit
//! instead of eight. [`rs_decode_chasing`] spends nothing: it flips the
//! least reliable bits outright, every way, and decodes each result.

use super::gf256::{self, div, exp, mul, poly_degree, poly_eval, poly_mul};
use crate::symbol::grid::CERTAIN;

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
    fn among(mut self, tries: usize) -> Self {
        self.located_bits = self.located_bits.saturating_sub(tries.max(1).next_power_of_two().trailing_zeros());
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
    let n = block.len();
    assert!(n <= 255 && ec_len <= n, "block of {n} with {ec_len} EC codewords");
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

    let syndromes = rs_syndromes(&received, ec_len);

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
    let mut omega = poly_mul(&syndromes, &psi);
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

    if rs_syndromes(&corrected, ec_len).iter().any(|&s| s != 0) {
        return Err(RsFailure::ResidualSyndrome);
    }

    let slack = ec_len - f - 2 * l;
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
        } else if cw.is_erasure() {
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

    Ok(RsSolution {
        corrected,
        errata,
        errors: l,
        erasures: f,
        slack,
        check_bits,
        flipped_known_bits,
        located_bits: located_error_bits(n - f, l),
        reserved,
    })
}

/// Decode by doubting: erase the least reliable fully read codewords, one
/// more at a time, and keep the solution with the most confirmation,
/// among those that show [`DOUBT_CONFIRMING_BITS`] of it and stay within
/// the standard's limit when `reserved` codewords are kept for detection.
/// Returns how many codewords were doubted. Codewords at [`CERTAIN`]
/// reliability and pinned codewords are never doubted.
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
    for k in 1..=doubtful.len() {
        if let Ok(solution) = rs_decode_doubted(block, &doubtful[..k], ec_len, reserved).map(|s| s.among(doubtful.len()))
            && solution.confidence_bits() >= DOUBT_CONFIRMING_BITS
            && solution.within_standard()
            && best.as_ref().is_none_or(|(_, b)| solution.confidence_bits() > b.confidence_bits())
        {
            best = Some((k, solution));
        }
    }
    best
}

/// Bits [`rs_decode_chasing`] flips: every pattern of the least reliable
/// ones is tried, so a block costs `2^CHASE_BITS` decodes at most.
pub const CHASE_BITS: usize = 6;
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
/// codewords, at most [`CHASE_BITS`] of them, flip every non-empty subset
/// in turn and decode what results as errors and erasures. Doubting a
/// faint codeword erases it, which spends a unit of redundancy to let
/// its bits be anything; flipping spends none, so a block with a few more
/// wrong codewords than the code corrects is reached when the extra ones
/// read faintly. This is Chase's second algorithm, over the faintest bits
/// of the block rather than of each codeword.
///
/// Trying many patterns is list decoding, and a wrong pattern can land
/// on a wrong codeword that verifies, so every pattern's verified
/// solution within the standard is kept and they must all name the same
/// codeword: a disagreement means the patterns cannot tell, and nothing
/// is returned. Returns how many bits the best solution flipped, with the
/// solution restated against the block as read: a flipped bit it kept is
/// an erratum it found, and `errors` counts every codeword it changed.
/// The slack stays that of the flipped block's decode, since the flipped
/// bits were guessed, not located by the code, and so does what the
/// errors that decode located confirm, less the cost of the patterns
/// tried. Bits that are pinned, and bits at [`CERTAIN`], are never
/// flipped.
///
/// A block whose every EC codeword the standard reserves for detection,
/// as M1's two are, is not chased at all: flipping bits until the check
/// passes is correction by another name, and with sixteen check bits and
/// sixty-three patterns to try it passes on noise once in a thousand
/// blocks.
pub fn rs_decode_chasing(
    block: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    reserved: usize,
) -> Option<(usize, RsSolution)> {
    let (flipped, solution) = rs_decode_chasing_bits(block, bit_reliability, ec_len, reserved)?;
    Some((flipped.iter().map(|bits| bits.count_ones() as usize).sum(), solution))
}

/// [`rs_decode_chasing`], with the bits of each codeword the best
/// solution's pattern flipped in place of their count: for showing which
/// they were.
pub fn rs_decode_chasing_bits(
    block: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
    reserved: usize,
) -> Option<(Vec<u8>, RsSolution)> {
    assert_eq!(block.len(), bit_reliability.len());
    if reserved >= ec_len {
        return None;
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
        return None;
    }
    doubtful.sort();
    doubtful.truncate(CHASE_BITS);

    let received: Vec<u8> = block.iter().map(|c| c.value & c.known).collect();
    let patterns = (1usize << doubtful.len()) - 1;
    let mut best: Option<(u32, RsSolution)> = None;
    for pattern in 1u32..1 << doubtful.len() {
        let mut trial = block.to_vec();
        for (k, &(_, i, b)) in doubtful.iter().enumerate() {
            if pattern >> k & 1 != 0 {
                trial[i].value ^= 1 << b;
            }
        }
        let Ok(mut solution) = rs_decode_doubted(&trial, &[], ec_len, reserved).map(|s| s.among(patterns)) else { continue };
        // Sixty-odd trials are sixty-odd chances for a wrong codeword to
        // lie within reach of the flipped block, so a chased solution must
        // show more redundancy than one found at the first try, and must
        // reproduce every known bit of the erased codewords: one that
        // overturns them has been read into the noise.
        if !solution.within_standard() || solution.slack < CHASE_MIN_SLACK || solution.flipped_known_bits > 0 {
            continue;
        }
        if best.as_ref().is_some_and(|(_, kept)| kept.corrected != solution.corrected) {
            return None;
        }
        solution.errata = (0..block.len())
            .filter(|&i| block[i].is_erasure() || solution.corrected[i] != received[i])
            .map(|i| Erratum {
                index: i,
                magnitude: solution.corrected[i] ^ received[i],
                was_erasure: block[i].is_erasure(),
            })
            .collect();
        solution.errors = solution.errata.iter().filter(|e| !e.was_erasure).count();
        if best.as_ref().is_none_or(|(_, b)| solution.confidence_bits() > b.confidence_bits()) {
            best = Some((pattern, solution));
        }
    }
    let (pattern, solution) = best?;
    let mut flipped = vec![0u8; block.len()];
    for (k, &(_, i, b)) in doubtful.iter().enumerate() {
        if pattern >> k & 1 != 0 {
            flipped[i] |= 1 << b;
        }
    }
    Some((flipped, solution))
}

/// A block solved by doubting single bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitDoubt {
    pub solution: BitSolution,
    /// How many bits, read as known, were doubted.
    pub doubted_bits: usize,
    /// Which bits of each codeword were doubted.
    pub doubted: Vec<u8>,
    /// Which doubted bits of each codeword the solution overturned.
    pub changed: Vec<u8>,
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

    /// Independent bits confirming the solution: the spare equations, and
    /// the doubted bits' agreement beyond chance, as [`doubted_agreement`]
    /// weighs it.
    pub fn confidence_bits(&self) -> u32 {
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
pub fn solve_bit_erasures_doubting(
    block: &[CwRead],
    bit_reliability: &[[u8; 8]],
    ec_len: usize,
) -> Option<BitDoubt> {
    assert_eq!(block.len(), bit_reliability.len());
    let unknown_bits: usize = block.iter().map(|c| c.known.count_zeros() as usize).sum();
    let equations = 8 * ec_len;
    let budget = equations.saturating_sub(unknown_bits + DOUBT_SPARE_EQUATIONS);
    if budget == 0 {
        return None;
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
        return None;
    }
    doubtful.sort();
    doubtful.truncate(budget);

    let attempt = |k: usize| -> Option<BitDoubt> {
        let mut trial = block.to_vec();
        let mut doubted = vec![0u8; block.len()];
        for &(_, i, b) in &doubtful[..k] {
            trial[i].known &= !(1 << b);
            trial[i].value &= !(1 << b);
            doubted[i] |= 1 << b;
        }
        let solution = solve_bit_erasures(&trial, ec_len).ok().filter(BitSolution::is_complete)?;
        if solution.spare_equations < DOUBT_SPARE_EQUATIONS {
            return None;
        }
        let changed = solution.corrected.iter().zip(block).zip(&doubted).map(|((&got, cw), &m)| (got ^ cw.value) & m).collect();
        Some(BitDoubt { solution, doubted_bits: k, doubted, changed })
    };

    // Fewer doubted bits leave more spare equations, so the first count
    // that solves is the best: find it by growing the count geometrically
    // and then narrowing down between the last failure and that success.
    let mut last_failed = 0;
    let mut k = 1;
    let mut found = None;
    while k <= doubtful.len() {
        match attempt(k) {
            Some(doubt) => {
                found = Some((k, doubt));
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
            Some(doubt) => {
                hi = mid;
                best = doubt;
            }
            None => lo = mid,
        }
    }
    Some(best)
}

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
    let unknowns: Vec<(usize, u8)> = block
        .iter()
        .enumerate()
        .flat_map(|(i, cw)| (0..8u8).filter(move |b| cw.known >> b & 1 == 0).map(move |b| (i, b)))
        .collect();
    let u = unknowns.len();
    let equations = 8 * ec_len;
    if u == 0 {
        return Err(BitSolveFailure::NoUnknowns);
    }
    if u > equations + PARTIAL_MARGIN {
        return Err(BitSolveFailure::TooManyUnknowns { unknown_bits: u, equations });
    }

    let received: Vec<u8> = block.iter().map(|c| c.value & c.known).collect();
    let syndromes = rs_syndromes(&received, ec_len);

    // Row e = 8j + k is bit k of syndrome j; column u holds the right-hand side.
    let words = u / 64 + 1;
    let mut rows = vec![vec![0u64; words]; equations];
    for (col, &(i, bit)) in unknowns.iter().enumerate() {
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
    for (j, &s) in syndromes.iter().enumerate() {
        for k in 0..8 {
            if s >> k & 1 != 0 {
                rows[8 * j + k][u / 64] |= 1 << (u % 64);
            }
        }
    }

    let mut rank = 0;
    let mut pivot_row = vec![usize::MAX; u];
    for col in 0..u {
        let Some(p) = (rank..equations).find(|&r| rows[r][col / 64] >> (col % 64) & 1 != 0) else {
            continue;
        };
        rows.swap(rank, p);
        let pivot = rows[rank].clone();
        for (r, row) in rows.iter_mut().enumerate() {
            if r != rank && row[col / 64] >> (col % 64) & 1 != 0 {
                for (a, b) in row.iter_mut().zip(&pivot) {
                    *a ^= b;
                }
            }
        }
        pivot_row[col] = rank;
        rank += 1;
    }

    let rhs = |row: &[u64]| row[u / 64] >> (u % 64) & 1 != 0;
    if rows[rank..].iter().any(|row| rhs(row)) {
        return Err(BitSolveFailure::Inconsistent);
    }

    // The rows are reduced, so a pivot's unknown is settled exactly when its
    // row involves no other unknown.
    let mut corrected = received;
    let mut solved = vec![0u8; n];
    let mut solved_bits = 0;
    for (col, &(i, bit)) in unknowns.iter().enumerate() {
        if pivot_row[col] == usize::MAX {
            continue;
        }
        let row = &rows[pivot_row[col]];
        let involved: u32 = (0..words)
            .map(|w| {
                // The right-hand side sits at column `u`; leave it out.
                let mask = if w == u / 64 { (1u64 << (u % 64)) - 1 } else { u64::MAX };
                (row[w] & mask).count_ones()
            })
            .sum();
        if involved != 1 {
            continue;
        }
        solved[i] |= 1 << bit;
        solved_bits += 1;
        if rhs(row) {
            corrected[i] |= 1 << bit;
        }
    }
    if solved_bits == 0 {
        return Err(BitSolveFailure::Underdetermined { rank, unknown_bits: u });
    }
    debug_assert!(solved_bits < u || rs_syndromes(&corrected, ec_len).iter().all(|&s| s == 0));
    Ok(BitSolution { corrected, solved, unknown_bits: u, solved_bits, spare_equations: equations - rank })
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
        // beyond the standard's limit, and doubting will not offer it.
        assert_eq!(rs_decode_doubting(&read, &reliability, ec_len, 2), None);
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
        assert_eq!(rs_decode_chasing(&read, &reliability, 2, 2), None);
        assert!(rs_decode_chasing(&read, &reliability, 2, 0).is_some_and(|(_, s)| s.corrected == block));
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
            let doubt = solve_bit_erasures_doubting(&read, &reliability, ec_len).unwrap_or_else(|| panic!("round {round}"));
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
        assert_eq!(solve_bit_erasures_doubting(&read, &vec![[CERTAIN; 8]; block.len()], ec_len), None);
        let mut reliability = vec![[200u8; 8]; block.len()];
        reliability[4][3] = 1;
        assert!(solve_bit_erasures_doubting(&read, &reliability, ec_len).is_some());
        read[4].pinned = 0x08;
        assert!(solve_bit_erasures_doubting(&read, &reliability, ec_len).is_none_or(|d| d.doubted[4] & 0x08 == 0));
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
        assert_eq!(solve_bit_erasures_doubting(&read, &reliability, ec_len), None);
        let (flipped, sol) = rs_decode_chasing(&read, &reliability, ec_len, 0).unwrap();
        assert_eq!(sol.corrected, block);
        assert_eq!((flipped, sol.errors, sol.erasures, sol.slack), (3, 7, 0, 2));
        // Two codewords spare, and the four solid errors the code located
        // among the forty read, less six bits for the sixty-three
        // patterns of the six faintest bits that were tried.
        assert_eq!(sol.located_bits, located_error_bits(40, 4) - 6);
        assert_eq!(sol.confidence_bits(), 16 + 9);
        assert_eq!(sol.errata.len(), 7);
        for e in &sol.errata {
            assert!(!e.was_erasure && e.magnitude == read[e.index].value ^ block[e.index]);
        }
        // Certain bits are never flipped; with three codewords reserved
        // the solution is beyond the standard and not offered.
        assert_eq!(rs_decode_chasing(&read, &vec![[CERTAIN; 8]; block.len()], ec_len, 0), None);
        assert_eq!(rs_decode_chasing(&read, &reliability, ec_len, 3), None);
        // A pinned bit is never flipped either, however faint.
        read[picked[4]].pinned = 1 << 4;
        assert!(rs_decode_chasing(&read, &reliability, ec_len, 0).is_none_or(|(_, s)| s.corrected != block));
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
