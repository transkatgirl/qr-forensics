//! Padding recovery.
//!
//! After the last segment an encoder writes a terminator, zero-fills to a
//! byte boundary, and fills the remaining data codewords with 0xEC and 0x11
//! alternately. Those codewords are known without reading them, so once the
//! start of padding is located they can be pinned, which stops damage there
//! from using up error-correction capacity. A segment header whose bits
//! are unreadable but which has only one reading under which the stream
//! parses is pinned the same way.

use super::bitstream::{Certainty, EndOfData, PAD_BYTES, StreamBits, TailAgreement, standard_tail, tail_agreement};
use super::codewords::{Pin, PinSource, Pins, codeword_bit_mask};
use super::pipeline::{DecodeOptions, DecodeReport, decode};
use super::rs::rs_encode;
use crate::symbol::grid::ModuleGrid;
use crate::symbol::layout::SymbolLayout;

/// A mismatch counts this many times a match when scoring a candidate. Data
/// agrees with the pad pattern half the time by chance, so anything above 1
/// makes data score negative on average.
const MISMATCH_PENALTY: i64 = 3;

/// Minimum score for a candidate found without help from segment headers:
/// two fully matching codewords.
const MIN_SCAN_SCORE: i64 = 16;

/// Value of pad codeword `stream_index` when padding starts at `first_pad`.
/// The four-bit final codeword of M1 and M3 is zero when it is padding.
fn pad_value(layout: &SymbolLayout, stream_index: usize, first_pad: usize) -> u8 {
    if layout.codewords[layout.data_stream[stream_index]].nbits == 4 {
        0
    } else {
        PAD_BYTES[(stream_index - first_pad) % 2]
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
    /// Per block and phase of its first data codeword: evidence from its EC
    /// codewords, assuming all of its data is padding.
    block_ec: Vec<[Evidence; 2]>,
}

impl<'a> Scanner<'a> {
    fn new(layout: &'a SymbolLayout, report: &DecodeReport) -> Self {
        let count = layout.data_stream.len();
        let exists = |cw| codeword_bit_mask(layout, cw);

        let mut suffix = [vec![(0, 0); count + 1], vec![(0, 0); count + 1]];
        for j in (0..count).rev() {
            let cw = layout.data_stream[j];
            let (value, known) = report.codewords[cw].best();
            for phase in 0..2 {
                let want = if layout.codewords[cw].nbits == 4 { 0 } else { PAD_BYTES[phase] };
                let here = compare(value, known & exists(cw), want);
                suffix[phase][j] = add(here, suffix[1 - phase][j + 1]);
            }
        }

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
                    let ec = rs_encode(&data, block.ec_len);
                    block.codewords[block.data_len..].iter().zip(ec).fold((0, 0), |acc, (&cw, want)| {
                        let (value, known) = report.codewords[cw].best();
                        add(acc, compare(value, known, want))
                    })
                })
            })
            .collect();

        Self { layout, suffix, block_ec }
    }

    fn evidence(&self, first_pad: usize) -> Evidence {
        let mut total = self.suffix[0][first_pad];
        for (block, ec) in self.layout.blocks.iter().zip(&self.block_ec) {
            if block.stream_offset >= first_pad {
                total = add(total, ec[(block.stream_offset - first_pad) % 2]);
            }
        }
        total
    }

    fn candidate(&self, first_pad: usize) -> PadCandidate {
        let (matches, mismatches) = self.evidence(first_pad);
        PadCandidate {
            first_pad_codeword: first_pad,
            pad_codewords: self.layout.data_stream.len() - first_pad,
            end_bit: None,
            header_certainty: None,
            matches,
            mismatches,
        }
    }
}

/// Locate the start of padding, from segment headers when they can be read
/// and otherwise from the pad pattern itself.
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
            let mut candidate = scanner.candidate(first_pad);
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
                // The headers are right and the tail is not standard padding.
                return None;
            }
        }
    }

    // On equal scores prefer the later start: it pins less.
    (0..count)
        .map(|first_pad| scanner.candidate(first_pad))
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
        let pin = Pin {
            mask: codeword_bit_mask(layout, cw),
            value: pad_value(layout, j, candidate.first_pad_codeword),
            source: PinSource::Padding,
        };
        if pins.insert(cw, pin) != Some(pin) {
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
        match pins.get(&cw) {
            Some(existing) if existing.source != PinSource::Terminator => continue,
            Some(existing) if existing.mask == mask && existing.value & mask == 0 => continue,
            _ => {}
        }
        pins.insert(cw, Pin { mask, value: 0, source: PinSource::Terminator });
        changed += 1;
    }
    changed
}

/// Pin the header bits that every clean reading of an unreadable header
/// agrees on, as (stream bit, value) pairs. A codeword already pinned
/// from another source is left alone. Returns how many codewords gained
/// pinned bits.
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
            Some(existing) if existing.source != PinSource::Header => continue,
            Some(existing) => *existing,
            None => Pin { mask: 0, value: 0, source: PinSource::Header },
        };
        let before = pin;
        pin.mask |= mask;
        pin.value = if value { pin.value | mask } else { pin.value & !mask };
        if pin != before {
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
/// under these pins: `before` is that decode.
pub fn recover_padding_after(
    grid: &ModuleGrid,
    layout: &SymbolLayout,
    mask: u8,
    pins: &Pins,
    opts: &DecodeOptions,
    before: &DecodeReport,
) -> Option<PaddingRecovery> {
    let mut candidate = detect_padding(layout, before)?;
    let mut first_pad = candidate.first_pad_codeword;

    let mut new_pins = pins.clone();
    let mut added = pad_pins(layout, &candidate, &mut new_pins);
    let mut header_pins_added = 0;
    let mut header_codewords: Vec<usize> = Vec::new();
    let mut report = decode(grid, layout, mask, &new_pins, opts);
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
                report = decode(grid, layout, mask, &new_pins, opts);
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
            // found, and the codewords between are padding too. Anything
            // else contradicts the headers.
            let continues = headers_first_pad < first_pad && (first_pad - headers_first_pad).is_multiple_of(2);
            if !continues {
                return None;
            }
            first_pad = headers_first_pad;
            candidate = Scanner::new(layout, &report).candidate(first_pad);
            candidate.end_bit = Some(bit);
            candidate.header_certainty = Some(certainty);
            added += pad_pins(layout, &candidate, &mut new_pins);
            report = decode(grid, layout, mask, &new_pins, opts);
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
        report = decode(grid, layout, mask, &new_pins, opts);
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

/// Drop every pin that padding recovery placed.
pub fn clear_padding_pins(pins: &mut Pins) {
    pins.retain(|_, pin| pin.source.is_by_hand());
}
