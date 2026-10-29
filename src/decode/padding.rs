//! Padding recovery.
//!
//! After the last segment an encoder writes a terminator, zero-fills to a
//! byte boundary, and fills the remaining data codewords with 0xEC and 0x11
//! alternately. Those codewords are known without reading them, so once the
//! start of padding is located they can be pinned, which stops damage there
//! from using up error-correction capacity. A segment header whose bits
//! are unreadable but which has only one reading under which the stream
//! parses is pinned the same way.
//!
//! Not every encoder fills as the standard does. BoofCV stopped checking
//! the padding of the symbols it reads, by default, over "a bug in a
//! popular encoder where for messages of a certain length it would be off
//! by one", and where it does check it lets the pattern begin again; some
//! encoders fill with zeros. A read does not ask what the padding is, so
//! such symbols read; but padding that is not the standard's was not
//! found here, and damage to it was paid for in error correction that a
//! standard symbol is spared. So a small family of fillers is looked for
//! ([`Filler`]), and the one a symbol shows is pinned as the standard's
//! would be.

use super::bitstream::{Certainty, EndOfData, PAD_BYTES, StreamBits, TailAgreement, standard_tail, tail_agreement};
use super::codewords::{Pin, PinSource, Pins, codeword_bit_mask, picked_among};
use super::pipeline::{DecodeOptions, DecodeReport, Decoding, decode};
use super::rs::rs_encode;
use crate::symbol::grid::ModuleGrid;
use crate::symbol::layout::{BlockSpec, SymbolLayout};

/// A mismatch counts this many times a match when scoring a candidate. Data
/// agrees with the pad pattern half the time by chance, so anything above 1
/// makes data score negative on average.
const MISMATCH_PENALTY: i64 = 3;

/// Minimum score for a candidate found without help from segment headers:
/// two fully matching codewords. A filler other than the standard's must
/// score as much wherever it is found, since the headers that say where
/// the padding starts do not say what it is.
const MIN_SCAN_SCORE: i64 = 16;

/// What an encoder fills the data codewords after the message with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filler {
    /// 0xEC and 0x11 in turn from the first codeword after the terminator,
    /// as ISO/IEC 18004 has it.
    Standard,
    /// The standard's turn begun one codeword late: the codeword where
    /// the segment headers put the first of the padding is something
    /// else, and is left as read.
    Late,
    /// 0x11 and 0xEC in turn: the standard's two codewords, the second
    /// first.
    Swapped,
    /// Zeros to the end.
    Zeros,
}

impl Filler {
    /// Every filler looked for, in the order they stand in where the
    /// evidence does not tell them apart: the standard's own first, and
    /// of the two that differ only in the first codeword the one that
    /// leaves it alone.
    pub const ALL: [Filler; 4] = [Filler::Standard, Filler::Late, Filler::Swapped, Filler::Zeros];

    /// The filler in a few words, for saying what was pinned.
    pub fn label(self) -> &'static str {
        match self {
            Filler::Standard => "0xEC and 0x11 in turn",
            Filler::Late => "0xEC and 0x11 in turn from one codeword late",
            Filler::Swapped => "0x11 and 0xEC in turn",
            Filler::Zeros => "zeros",
        }
    }
}

/// Value of pad codeword `stream_index` when padding starts at `first_pad`
/// and is `filler`; `None` for the codeword a late start leaves as read.
/// The four-bit final codeword of M1 and M3 is zero when it is padding.
fn pad_value(layout: &SymbolLayout, stream_index: usize, first_pad: usize, filler: Filler) -> Option<u8> {
    let turn = stream_index - first_pad;
    if filler == Filler::Late && turn == 0 {
        None
    } else if layout.codewords[layout.data_stream[stream_index]].nbits == 4 {
        Some(0)
    } else {
        Some(match filler {
            Filler::Standard => PAD_BYTES[turn % 2],
            Filler::Late => PAD_BYTES[(turn - 1) % 2],
            Filler::Swapped => PAD_BYTES[(turn + 1) % 2],
            Filler::Zeros => 0,
        })
    }
}

/// What the stream should contain from the end of data onward.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PadTail {
    /// Stream index of the first pad codeword; equal to the number of data
    /// codewords when there is no room for padding.
    pub first_pad_codeword: usize,
    /// Expected value of each stream bit starting at the end of data.
    pub expected: Vec<bool>,
}

pub fn pad_tail(layout: &SymbolLayout, end_bit: usize) -> PadTail {
    let total = layout.data_bits;
    let count = layout.data_stream.len();
    let terminator_bits = layout.stream.terminator_bits as usize;
    let end_bit = end_bit.min(total);
    let terminator_end = (end_bit + terminator_bits).min(total);
    let first_pad = terminator_end.div_ceil(8).min(count);
    PadTail { first_pad_codeword: first_pad, expected: standard_tail(terminator_bits, total, end_bit) }
}

/// How the bits after the end of data compare with standard padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TailCheck {
    pub first_pad_codeword: usize,
    pub matches: u32,
    pub mismatches: u32,
    /// Mismatches in bits that error correction has confirmed. Wrong reads
    /// cannot explain these; the tail is not standard padding.
    pub verified_mismatches: u32,
    pub unknown: u32,
    /// Every readable bit after the end of data is zero. Some encoders pad
    /// with zeros instead of the standard pattern.
    pub zero_filled: bool,
}

impl TailCheck {
    /// The tail looks like padding: either all zeros, or the standard
    /// pattern with no confirmed contradiction and few unconfirmed ones.
    pub fn is_plausible(&self) -> bool {
        TailAgreement {
            matches: self.matches,
            mismatches: self.mismatches,
            verified_mismatches: self.verified_mismatches,
            unknown: self.unknown,
            zero_filled: self.zero_filled,
        }
        .is_plausible()
    }
}

pub fn check_tail(layout: &SymbolLayout, stream: &StreamBits, end_bit: usize) -> TailCheck {
    let tail = pad_tail(layout, end_bit);
    let agreement = tail_agreement(stream, end_bit.min(layout.data_bits), &tail.expected);
    TailCheck {
        first_pad_codeword: tail.first_pad_codeword,
        matches: agreement.matches,
        mismatches: agreement.mismatches,
        verified_mismatches: agreement.verified_mismatches,
        unknown: agreement.unknown,
        zero_filled: agreement.zero_filled,
    }
}

/// A place where padding appears to start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PadCandidate {
    /// Stream index of the first pad codeword.
    pub first_pad_codeword: usize,
    pub pad_codewords: usize,
    /// What the padding appears to be.
    pub filler: Filler,
    /// End of data according to segment headers, when they led here.
    pub end_bit: Option<usize>,
    /// Certainty of those headers.
    pub header_certainty: Option<Certainty>,
    /// Readable bits in the padded region (data and, for blocks that are
    /// padding throughout, their EC codewords) that agree with the pattern.
    pub matches: u32,
    pub mismatches: u32,
}

impl PadCandidate {
    fn score(&self) -> i64 {
        self.matches as i64 - MISMATCH_PENALTY * self.mismatches as i64
    }
}

type Evidence = (u32, u32);

fn add(a: Evidence, b: Evidence) -> Evidence {
    (a.0 + b.0, a.1 + b.1)
}

fn compare(value: u8, known: u8, want: u8) -> Evidence {
    let wrong = ((value ^ want) & known).count_ones();
    (known.count_ones() - wrong, wrong)
}

/// Evidence for padding starting at each possible codeword, gathered once.
struct Scanner<'a> {
    layout: &'a SymbolLayout,
    /// `suffix[phase][j]`: evidence over stream codewords `j..` when codeword
    /// `j` is at `phase` of the pad pattern.
    suffix: [Vec<Evidence>; 2],
    /// `zeros[j]`: evidence over stream codewords `j..` for zeros there.
    zeros: Vec<Evidence>,
    /// Per block and phase of its first data codeword: evidence from its EC
    /// codewords, assuming all of its data is padding.
    block_ec: Vec<[Evidence; 2]>,
    /// Per block: evidence from its EC codewords, assuming all of its data
    /// is zeros, whose error correction is zeros too.
    block_zero_ec: Vec<Evidence>,
}

impl<'a> Scanner<'a> {
    fn new(layout: &'a SymbolLayout, report: &DecodeReport) -> Self {
        let count = layout.data_stream.len();
        let exists = |cw| codeword_bit_mask(layout, cw);

        let mut suffix = [vec![(0, 0); count + 1], vec![(0, 0); count + 1]];
        let mut zeros = vec![(0, 0); count + 1];
        for j in (0..count).rev() {
            let cw = layout.data_stream[j];
            let (value, known) = report.codewords[cw].best();
            for phase in 0..2 {
                let want = if layout.codewords[cw].nbits == 4 { 0 } else { PAD_BYTES[phase] };
                let here = compare(value, known & exists(cw), want);
                suffix[phase][j] = add(here, suffix[1 - phase][j + 1]);
            }
            zeros[j] = add(compare(value, known & exists(cw), 0), zeros[j + 1]);
        }

        let ec_evidence = |block: &BlockSpec, ec: Vec<u8>| {
            block.codewords[block.data_len..].iter().zip(ec).fold((0, 0), |acc, (&cw, want)| {
                let (value, known) = report.codewords[cw].best();
                add(acc, compare(value, known, want))
            })
        };
        let block_ec = layout
            .blocks
            .iter()
            .map(|block| {
                [0, 1].map(|phase| {
                    let data: Vec<u8> = (0..block.data_len)
                        .map(|i| {
                            if layout.codewords[block.codewords[i]].nbits == 4 {
                                0
                            } else {
                                PAD_BYTES[(phase + i) % 2]
                            }
                        })
                        .collect();
                    ec_evidence(block, rs_encode(&data, block.ec_len))
                })
            })
            .collect();
        let block_zero_ec = layout.blocks.iter().map(|block| ec_evidence(block, vec![0; block.ec_len])).collect();

        Self { layout, suffix, zeros, block_ec, block_zero_ec }
    }

    fn evidence(&self, first_pad: usize, filler: Filler) -> Evidence {
        // Where the turn of the two pad codewords begins, and with which
        // of them.
        let (start, phase) = match filler {
            Filler::Standard => (first_pad, 0),
            Filler::Late => (first_pad + 1, 0),
            Filler::Swapped => (first_pad, 1),
            Filler::Zeros => {
                let whole = self.layout.blocks.iter().zip(&self.block_zero_ec).filter(|(block, _)| block.stream_offset >= first_pad);
                return whole.fold(self.zeros[first_pad], |total, (_, &ec)| add(total, ec));
            }
        };
        let mut total = self.suffix[phase][start];
        for (block, ec) in self.layout.blocks.iter().zip(&self.block_ec) {
            if block.stream_offset >= start {
                total = add(total, ec[(block.stream_offset - start + phase) % 2]);
            }
        }
        total
    }

    fn candidate(&self, first_pad: usize, filler: Filler) -> PadCandidate {
        let (matches, mismatches) = self.evidence(first_pad, filler);
        PadCandidate {
            first_pad_codeword: first_pad,
            pad_codewords: self.layout.data_stream.len() - first_pad,
            filler,
            end_bit: None,
            header_certainty: None,
            matches,
            mismatches,
        }
    }

    /// The candidate at `first_pad` that scores best among `fillers`, the
    /// earlier of them where two score alike.
    fn best(&self, first_pad: usize, fillers: &[Filler]) -> PadCandidate {
        let mut best = self.candidate(first_pad, fillers[0]);
        for &filler in &fillers[1..] {
            let candidate = self.candidate(first_pad, filler);
            if candidate.score() > best.score() {
                best = candidate;
            }
        }
        best
    }
}

/// Locate the start of padding, from segment headers when they can be read
/// and otherwise from the pad pattern itself.
///
/// Where the headers say the data ends, the padding is taken for the
/// standard's unless another filler shows itself there: one that scores
/// better than the standard's on what can be read, and as well as a
/// pattern found with no headers must. Without headers only the
/// standard's turn of two codewords is looked for, wherever it begins.
/// It marks its own start, which a run of zeros does not: zero is what a
/// message's own bytes may be, and the zeros of a terminator run into
/// those of a zero filler with nothing between. Whether the turn then
/// begins where the headers, once read, put the padding, a codeword
/// late, or with its second codeword, is settled when they are
/// ([`recover_padding_after`]).
pub fn detect_padding(layout: &SymbolLayout, report: &DecodeReport) -> Option<PadCandidate> {
    let count = layout.data_stream.len();
    let scanner = Scanner::new(layout, report);

    if let EndOfData::At { bit, certainty } = report.parse.end {
        let first_pad = pad_tail(layout, bit).first_pad_codeword;
        if first_pad == count {
            // The data runs to capacity: there is no padding to recover.
            if certainty == Certainty::Verified {
                return None;
            }
        } else {
            let standard = scanner.candidate(first_pad, Filler::Standard);
            let other = scanner.best(first_pad, &Filler::ALL[1..]);
            let mut candidate =
                if other.score() >= MIN_SCAN_SCORE && other.score() > standard.score() { other } else { standard };
            candidate.end_bit = Some(bit);
            candidate.header_certainty = Some(certainty);
            // The terminator and zero-fill before the first pad codeword
            // count as evidence too, once headers say where they are.
            for i in bit..(8 * first_pad).min(layout.data_bits) {
                if report.stream.certainty[i] != Certainty::Unknown {
                    if report.stream.values[i] {
                        candidate.mismatches += 1;
                    } else {
                        candidate.matches += 1;
                    }
                }
            }
            let accepted = match certainty {
                // Trusted headers need only a tail that does not contradict them.
                Certainty::Verified => candidate.mismatches * 4 <= candidate.matches,
                _ => candidate.score() >= MIN_SCAN_SCORE,
            };
            if accepted {
                return Some(candidate);
            }
            if certainty == Certainty::Verified {
                // The headers are right and the tail is no padding that
                // is looked for.
                return None;
            }
        }
    }

    // On equal scores prefer the later start: it pins less.
    (0..count)
        .map(|first_pad| scanner.candidate(first_pad, Filler::Standard))
        .filter(|c| c.score() >= MIN_SCAN_SCORE)
        .max_by_key(|c| (c.score(), c.first_pad_codeword))
}

/// Pins for every pad codeword of a candidate. Pins made by hand, by the
/// codeword or as known text, are left alone.
fn pad_pins(layout: &SymbolLayout, candidate: &PadCandidate, pins: &mut Pins) -> usize {
    let mut added = 0;
    for j in candidate.first_pad_codeword..layout.data_stream.len() {
        let cw = layout.data_stream[j];
        if pins.get(&cw).is_some_and(|p| p.source.is_by_hand()) {
            continue;
        }
        let Some(value) = pad_value(layout, j, candidate.first_pad_codeword, candidate.filler) else { continue };
        let pin = Pin::new(codeword_bit_mask(layout, cw), value, PinSource::Padding);
        if pins.insert(cw, pin).is_none_or(|old| !old.asserts_as(&pin)) {
            added += 1;
        }
    }
    added
}

/// Pin the terminator and zero-fill bits between the end of data and the
/// first pad codeword. Returns how many codewords gained pinned bits.
fn terminator_pins(layout: &SymbolLayout, end_bit: usize, first_pad: usize, pins: &mut Pins) -> usize {
    let mut changed = 0;
    let stop = (8 * first_pad).min(layout.data_bits);
    let mut bit = end_bit;
    while bit < stop {
        let j = bit / 8;
        let cw = layout.data_stream[j];
        let upto = (8 * (j + 1)).min(stop);
        let mask = (bit..upto).fold(0u8, |m, b| m | 0x80 >> (b % 8)) & codeword_bit_mask(layout, cw);
        bit = upto;
        let pin = match pins.get(&cw) {
            // Header bits inferred in the same codeword, where a segment
            // of four bits, FNC1 in first position say, ends the data:
            // the terminator's bits join them, under the header's name,
            // which is left as it is. A codeword pinned by hand, or by
            // the padding, is left alone. The codeword was once left
            // alone whatever had pinned it, and the terminator's bits
            // beside a header's went unpinned.
            Some(existing) if existing.source == PinSource::Header => {
                if existing.value & mask != 0 {
                    continue;
                }
                Pin { mask: existing.mask | mask, ..*existing }
            }
            Some(existing) if existing.source != PinSource::Terminator => continue,
            _ => Pin::new(mask, 0, PinSource::Terminator),
        };
        if pins.get(&cw).is_none_or(|old| !old.asserts_as(&pin)) {
            pins.insert(cw, pin);
            changed += 1;
        }
    }
    changed
}

/// Pin the header bits that every clean reading of an unreadable header
/// agrees on, as (stream bit, value) pairs. A codeword already pinned
/// from another source than the terminator's is left alone. Returns how
/// many codewords gained pinned bits.
fn header_pins(layout: &SymbolLayout, inferred: &[(usize, bool)], pins: &mut Pins) -> usize {
    let mut changed = std::collections::BTreeSet::new();
    for &(bit, value) in inferred {
        let j = bit / 8;
        let Some(&cw) = layout.data_stream.get(j) else { continue };
        let mask = (0x80u8 >> (bit % 8)) & codeword_bit_mask(layout, cw);
        if mask == 0 {
            continue;
        }
        let mut pin = match pins.get(&cw) {
            Some(existing) if existing.source == PinSource::Header => *existing,
            // The terminator's bits in the same codeword, pinned before
            // the header was inferred: the header's join them, and the
            // codeword goes under the header's name, which the terminator
            // then leaves as it is ([`terminator_pins`]). Under the
            // terminator's name it would be pinned afresh as the
            // terminator alone each pass, and the header laid over it
            // again, with no end.
            Some(existing) if existing.source == PinSource::Terminator => Pin { source: PinSource::Header, ..*existing },
            Some(_) => continue,
            None => Pin::new(0, 0, PinSource::Header),
        };
        let before = pin;
        pin.mask |= mask;
        pin.value = if value { pin.value | mask } else { pin.value & !mask };
        if !pin.asserts_as(&before) {
            pins.insert(cw, pin);
            changed.insert(cw);
        }
    }
    changed.len()
}

#[derive(Clone, Debug)]
pub struct PaddingRecovery {
    pub candidate: PadCandidate,
    /// The complete pin set to adopt: the pins passed in plus the new ones.
    pub pins: Pins,
    /// Codewords that gained a pin.
    pub added: usize,
    /// Of those, codewords that gained inferred header bits: the one
    /// reading of an unreadable header under which the stream parses.
    pub header_pins: usize,
    /// Segment headers, once readable and verified, agree with where the
    /// padding was placed. Headers that were inferred count only once the
    /// blocks holding them solve with the inferred bits in place.
    pub confirmed: bool,
    pub solved_before: usize,
    pub solved_after: usize,
    /// Decode result with the new pins applied.
    pub report: DecodeReport,
}

/// Find the padding, pin it, and re-solve. Returns `None` when no padding
/// can be located or the result contradicts verified segment headers.
pub fn recover_padding(
    grid: &ModuleGrid,
    layout: &SymbolLayout,
    mask: u8,
    pins: &Pins,
    opts: &DecodeOptions,
) -> Option<PaddingRecovery> {
    recover_padding_after(grid, layout, mask, pins, opts, &decode(grid, layout, mask, pins, opts))
}

/// [`recover_padding`] for a caller that has already decoded the grid
/// under these pins: `before` is that decode. The decodes made here
/// derive no pins of their own ([`DecodeOptions::derive_pins`]): this
/// is the derivation. Each solves anew only the blocks whose pins or
/// evidence moved since the pass before ([`Decoding`]), the pins each
/// pass placed picked among as many ways as the pass's share of the
/// recovery's chances says ([`pass_tries`]): every pass was once a
/// decode afresh, and charged every block.
pub fn recover_padding_after(
    grid: &ModuleGrid,
    layout: &SymbolLayout,
    mask: u8,
    pins: &Pins,
    opts: &DecodeOptions,
    before: &DecodeReport,
) -> Option<PaddingRecovery> {
    let opts = DecodeOptions { derive_pins: false, ..*opts };
    let mut decoding = Decoding::new(grid, layout, mask, &opts);
    recover_padding_with(layout, pins, before, |pins| decoding.solve(pins))
}

/// What the pins the `pass`th decode of padding recovery is the first
/// to make under are picked among ([`Pin::among`]), times the ways the
/// pins given were: `pass` times one more. Every decode under pins of
/// the recovery's placing is a fresh chance for a block of noise to
/// verify, and the recovery is kept where one verifies; the decodes are
/// made in turn, as many as the pins keep giving more to, and the
/// `pass`th is given that share of the chances, which over every pass
/// there may be comes to one: half to the first, a sixth to the second,
/// and so on. The decodes under pins once went uncharged, each a chance
/// the recovery did not count.
pub fn pass_tries(pass: usize) -> usize {
    pass.max(1) * (pass.max(1) + 1)
}

/// Each pin of `pins` placed or changed since `decoded`, the pins the
/// last decode was made under, picked among `among` ways, and every
/// other as it was ([`Pin::among`]). The pins a recovery leaves as they
/// were are the picks they were.
fn stamp(pins: &mut Pins, decoded: &Pins, among: usize) {
    for (cw, pin) in pins.iter_mut() {
        pin.among = match decoded.get(cw) {
            Some(old) if old.asserts_as(pin) => old.among,
            _ => among,
        };
    }
}

/// [`recover_padding_after`] with the grid decoded by `redecode`, given
/// the pins to decode under, each carrying what it was picked among
/// ([`pass_tries`]): for a caller holding a decode it can make again
/// under more pins without solving every block anew
/// ([`super::pipeline::Decoding`]).
pub fn recover_padding_with(
    layout: &SymbolLayout,
    pins: &Pins,
    before: &DecodeReport,
    mut redecode: impl FnMut(&Pins) -> DecodeReport,
) -> Option<PaddingRecovery> {
    let given = picked_among(pins);
    let (mut passes, mut decoded) = (0, pins.clone());
    let mut redecode = |pins: &mut Pins| {
        passes += 1;
        stamp(pins, &decoded, given.saturating_mul(pass_tries(passes)));
        decoded = pins.clone();
        redecode(pins)
    };
    let mut candidate = detect_padding(layout, before)?;
    let mut first_pad = candidate.first_pad_codeword;

    let mut new_pins = pins.clone();
    let mut added = pad_pins(layout, &candidate, &mut new_pins);
    let mut header_pins_added = 0;
    let mut header_codewords: Vec<usize> = Vec::new();
    let mut report = redecode(&mut new_pins);
    let mut confirmed = false;

    // Each pass can verify more headers, which can locate the terminator,
    // which can solve another block; and a header that stays unreadable
    // but has only one reading that parses gives up its bits the same way.
    for _ in 0..layout.blocks.len() + 16 {
        if let Some(inference) = &report.parse.header_inference
            && inference.complete
            && !inference.inferred.is_empty()
        {
            let changed = header_pins(layout, &inference.inferred, &mut new_pins);
            if changed > 0 {
                added += changed;
                header_pins_added += changed;
                header_codewords.extend(inference.inferred.iter().filter_map(|&(bit, _)| layout.data_stream.get(bit / 8)));
                report = redecode(&mut new_pins);
                continue;
            }
        }
        let EndOfData::At { bit, certainty } = report.parse.end else { break };
        if certainty != Certainty::Verified {
            break;
        }
        let headers_first_pad = pad_tail(layout, bit).first_pad_codeword;
        if headers_first_pad != first_pad {
            // The pad pattern could only be seen from some way into the
            // padding; the headers now say it starts earlier, in codewords
            // that were unreadable. Padding alternates, so an earlier
            // start of the same parity continues the pattern the scan
            // found, and the codewords between are padding too. An
            // earlier start of the other parity continues it as well,
            // with its two codewords the other way about where the
            // headers put the first of them: the turn begun a codeword
            // late, or with its second codeword, whichever the codeword
            // there bears out, and it is left as read where it bears out
            // neither. Anything else contradicts the headers, as does
            // any start but the one a filler was found at when it is one
            // that was looked for only where the headers then stood.
            if headers_first_pad > first_pad || candidate.filler != Filler::Standard {
                return None;
            }
            let scanner = Scanner::new(layout, &report);
            let same_parity = (first_pad - headers_first_pad).is_multiple_of(2);
            first_pad = headers_first_pad;
            candidate = if same_parity {
                scanner.candidate(first_pad, Filler::Standard)
            } else {
                scanner.best(first_pad, &[Filler::Late, Filler::Swapped])
            };
            candidate.end_bit = Some(bit);
            candidate.header_certainty = Some(certainty);
            added += pad_pins(layout, &candidate, &mut new_pins);
            report = redecode(&mut new_pins);
            continue;
        }
        // Inferred header bits are taken on trust by the pins; the blocks
        // holding them must solve with them in place before they count as
        // confirmation.
        confirmed = header_codewords.iter().all(|&cw| report.blocks[layout.codewords[cw].block as usize].is_solved());
        let changed = terminator_pins(layout, bit, first_pad, &mut new_pins);
        if changed == 0 {
            break;
        }
        added += changed;
        report = redecode(&mut new_pins);
    }

    Some(PaddingRecovery {
        candidate,
        pins: new_pins,
        added,
        header_pins: header_pins_added,
        confirmed,
        solved_before: before.blocks_solved(),
        solved_after: report.blocks_solved(),
        report,
    })
}

/// What can be read where the padding would lie were it `filler` from
/// stream codeword `first_pad` on: for saying how well a start and a
/// filler given by hand fit before they are pinned. The start is held to
/// the data codewords the symbol has.
pub fn padding_at(layout: &SymbolLayout, report: &DecodeReport, first_pad: usize, filler: Filler) -> PadCandidate {
    let first_pad = first_pad.min(layout.data_stream.len().saturating_sub(1));
    Scanner::new(layout, report).candidate(first_pad, filler)
}

/// Pin the padding where it is said to start, as the filler it is said
/// to be: for a symbol whose padding [`detect_padding`] does not find,
/// too little of it being readable or its headers too damaged to say
/// where the data ends, when the user can say. The pins are `pins` with
/// what padding recovery placed taken away, and every pad codeword from
/// stream codeword `first_pad` on pinned at the filler's value. Pins made
/// by hand are left alone, and so is the terminator, whose place only the
/// headers give. Returns the pins and how many codewords were pinned.
pub fn pin_padding_at(layout: &SymbolLayout, first_pad: usize, filler: Filler, pins: &Pins) -> (Pins, usize) {
    let count = layout.data_stream.len();
    let first_pad = first_pad.min(count);
    let mut new_pins = pins.clone();
    clear_padding_pins(&mut new_pins);
    let candidate = PadCandidate {
        first_pad_codeword: first_pad,
        pad_codewords: count - first_pad,
        filler,
        end_bit: None,
        header_certainty: None,
        matches: 0,
        mismatches: 0,
    };
    let added = pad_pins(layout, &candidate, &mut new_pins);
    (new_pins, added)
}

/// The filler the padding pins of a decode hold: what
/// [`recover_padding`] settled on, read back from the pins it left and
/// from where the segment headers of `report`, the decode made under
/// them, put the end of the data. `None` when no pad codeword is pinned,
/// or the pins follow no filler from there, as pins since changed by hand
/// may not.
pub fn pinned_filler(layout: &SymbolLayout, report: &DecodeReport, pins: &Pins) -> Option<Filler> {
    let EndOfData::At { bit, .. } = report.parse.end else { return None };
    let first_pad = pad_tail(layout, bit).first_pad_codeword;
    let pinned: Vec<(usize, u8)> = (first_pad..layout.data_stream.len())
        .filter_map(|j| Some((j, pins.get(&layout.data_stream[j]).filter(|pin| pin.source == PinSource::Padding)?.value)))
        .collect();
    if pinned.is_empty() {
        return None;
    }
    Filler::ALL.into_iter().find(|&filler| pinned.iter().all(|&(j, value)| pad_value(layout, j, first_pad, filler) == Some(value)))
}

/// Drop every pin that padding recovery placed.
pub fn clear_padding_pins(pins: &mut Pins) {
    pins.retain(|_, pin| pin.source.is_by_hand());
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::symbol::function_map::FunctionMap;
    use crate::symbol::{EcLevel, SymbolVersion};

    #[test]
    fn header_and_terminator_bits_share_a_codeword() {
        // FNC1 in first position is a segment of four bits and no more,
        // and the terminator follows it in the same codeword: the header
        // inferred and the terminator each want bits of stream codeword
        // 1, bits 8 to 11 and 12 to 15. Laid in either order, the two
        // sets of bits end up in one pin, under the header's name; laid
        // again, nothing changes. Each once left the codeword alone
        // where the other had pinned it, and the bits went unpinned.
        let layout = SymbolLayout::new(Arc::new(FunctionMap::new(SymbolVersion::Qr(1))), EcLevel::L).unwrap();
        let cw = layout.data_stream[1];
        let inferred = [(8, false), (9, true), (10, false), (11, true)];
        let both = Pin::new(0xFF, 0x50, PinSource::Header);

        let mut pins = Pins::new();
        assert_eq!(terminator_pins(&layout, 12, 2, &mut pins), 1);
        assert_eq!(pins[&cw], Pin::new(0x0F, 0, PinSource::Terminator));
        assert_eq!(header_pins(&layout, &inferred, &mut pins), 1);
        assert_eq!(pins[&cw], both);
        assert_eq!(terminator_pins(&layout, 12, 2, &mut pins), 0);
        assert_eq!(header_pins(&layout, &inferred, &mut pins), 0);
        assert_eq!(pins[&cw], both);

        let mut pins = Pins::new();
        assert_eq!(header_pins(&layout, &inferred, &mut pins), 1);
        assert_eq!(terminator_pins(&layout, 12, 2, &mut pins), 1);
        assert_eq!((pins[&cw], pins.len()), (both, 1));

        // A pin by hand, or the padding's, is left alone by both.
        for source in [PinSource::Manual, PinSource::Text, PinSource::Padding] {
            let mut pins = Pins::new();
            let held = Pin::new(0xFF, 0xEC, source);
            pins.insert(cw, held);
            assert_eq!(terminator_pins(&layout, 12, 2, &mut pins), 0);
            assert_eq!(header_pins(&layout, &inferred, &mut pins), 0);
            assert_eq!(pins[&cw], held);
        }
    }
}
