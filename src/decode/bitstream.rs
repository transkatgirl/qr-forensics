//! Segment parser for the data bit stream.
//!
//! Every bit carries a certainty, so a stream assembled partly from blocks
//! that failed error correction still parses: characters built from doubtful
//! or missing bits are marked rather than dropped. A header with unreadable
//! bits would stop the parse where it stands; instead every reading it
//! could have is tried, fewest guessed bits first, and when only one of
//! them parses cleanly to a plausible end its bits are offered as inferred.

use std::ops::Range;

use crate::symbol::Family;
use crate::symbol::tables::{Mode, StreamSpec};

/// How much a bit can be trusted. Ordered so that `min` combines them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Certainty {
    /// The module was not readable.
    Unknown,
    /// Read from the symbol but not confirmed by error correction.
    Unverified,
    /// Confirmed by error correction, or asserted by a pin.
    Verified,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StreamBits {
    pub values: Vec<bool>,
    pub certainty: Vec<Certainty>,
}

/// A run of bits read as an unsigned number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Field {
    /// Unknown bits read as zero.
    pub value: u32,
    /// The least certain bit in the run.
    pub certainty: Certainty,
}

impl StreamBits {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn push(&mut self, value: bool, certainty: Certainty) {
        self.values.push(value && certainty != Certainty::Unknown);
        self.certainty.push(certainty);
    }

    pub fn field(&self, range: Range<usize>) -> Field {
        let mut field = Field { value: 0, certainty: Certainty::Verified };
        for i in range {
            field.value = field.value << 1 | u32::from(self.values[i]);
            field.certainty = field.certainty.min(self.certainty[i]);
        }
        field
    }
}

/// One decoded unit with the bits it came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cell<T> {
    /// `None` when a bit is unknown or the bits do not form a valid value.
    pub value: Option<T>,
    pub certainty: Certainty,
    pub bits: Range<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SegmentContent {
    /// Numeric and alphanumeric characters.
    Chars(Vec<Cell<char>>),
    /// Byte-mode data, to be decoded with a character set.
    Bytes(Vec<Cell<u8>>),
    /// Kanji mode characters as Shift-JIS double-byte codes.
    Kanji(Vec<Cell<u16>>),
    /// Hanzi mode characters as GB 2312 double-byte codes.
    Hanzi(Vec<Cell<u16>>),
    /// ECI designator; `None` when its value bits are unreadable.
    Eci(Option<u32>),
    StructuredAppend { index: u8, total: u8, parity: u8 },
    Fnc1First,
    /// Application indicator.
    Fnc1Second(u8),
    /// Header-only segment whose bits are unreadable.
    Unreadable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    pub mode: Mode,
    /// Mode indicator and character count.
    pub header: Range<usize>,
    pub payload: Range<usize>,
    /// Character count from the header, for data modes.
    pub count: Option<usize>,
    /// The least certain bit of the header, and of the fields of a
    /// control segment (ECI length prefix, structured append position,
    /// application indicator).
    pub header_certainty: Certainty,
    pub content: SegmentContent,
    /// Some known bits do not form a valid value (for example a numeric
    /// group above 999). Evidence of a wrong read or wrong format.
    pub damaged: bool,
    /// The header promises more data than the symbol holds.
    pub truncated: bool,
    /// The header had unreadable bits, and this is the one reading of
    /// them under which the stream parses cleanly: a guess the rest of the
    /// stream bears out, not a read. `header_certainty` is `Unknown`.
    pub header_inferred: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LostReason {
    /// A header contains unreadable bits, so segment boundaries are unknown
    /// from here on.
    UnknownHeader,
    InvalidMode(u32),
    /// An ECI designator starts with a lead byte no length is defined for.
    InvalidEci,
    /// A segment claims to run past the end of the symbol.
    Overrun,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndOfData {
    /// Data ends at `bit`, where the terminator starts or capacity runs out.
    /// `certainty` is the least certain header bit on the way there,
    /// `Unknown` when a header was inferred rather than read.
    At { bit: usize, certainty: Certainty },
    /// Parsing could not continue past the header starting at `bit`.
    Lost { bit: usize, reason: LostReason },
}

/// What trying the readings of a header with unreadable bits found. The
/// readings are tried in order of how many bits they guess, and the search
/// stops at the first depth where one parses cleanly, so the candidates
/// are the most parsimonious readings of the stream. A reading counts
/// only when the terminator it ends at was read, not guessed: the known
/// bits it explains are what set it apart from the readings that merely
/// stop somewhere consistent with the padding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeaderInference {
    /// Readings that parse cleanly to the end with a plausible tail.
    pub candidates: usize,
    /// Header bits each candidate had to guess.
    pub guessed_bits: usize,
    /// Every reading at that depth was tried. When false the search ran
    /// out of budget: `candidates` is a lower bound and nothing is inferred.
    pub complete: bool,
    /// Stream bits, as (position, value), that every candidate agrees on.
    /// Empty unless the search is complete and found a candidate.
    pub inferred: Vec<(usize, bool)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseReport {
    pub segments: Vec<Segment>,
    pub end: EndOfData,
    /// Set when a header had unreadable bits and the stream after it held
    /// something to test their readings against. When exactly one reading
    /// parses, `segments` and `end` are that reading's, marked as inferred.
    pub header_inference: Option<HeaderInference>,
    /// A known bit of the preamble, the zero bits a Model 1 stream begins
    /// with, read as one: evidence of a wrong read or wrong format, as a
    /// damaged segment is.
    pub preamble_damaged: bool,
}

const ALPHANUMERIC: &[u8; 45] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:";

/// The pad codewords an encoder writes after the terminator, in turn.
pub const PAD_BYTES: [u8; 2] = [0xEC, 0x11];

/// Most header bits one reading may guess: a mode indicator, a count and
/// a terminator come to sixteen in QR Code.
const MAX_GUESSED_HEADER_BITS: usize = 16;
/// Most parses one inference may run. Past this it gives up rather than
/// stall a decode, which brute force runs once per EC level and mask.
const MAX_HEADER_TRIALS: usize = 2048;
/// Known bits beyond the terminator that a reading must explain as
/// padding before it counts: a reading that swallows the padding as
/// payload and stops within a few bits of capacity explains nothing.
const MIN_TAIL_EVIDENCE: u32 = 4;

/// Payload length in bits for `count` characters.
pub fn payload_bits(mode: Mode, count: usize) -> usize {
    match mode {
        Mode::Numeric => 10 * (count / 3) + [0, 4, 7][count % 3],
        Mode::Alphanumeric => 11 * (count / 2) + 6 * (count % 2),
        Mode::Byte => 8 * count,
        Mode::Kanji | Mode::Hanzi => 13 * count,
        _ => 0,
    }
}

/// The Shift-JIS code a 13-bit kanji value stands for, and whether it is
/// one a kanji can have. The value is the lead byte less 0x81 (or 0xC1
/// from 0xE0) times 0xC0, plus the trail byte less 0x40, so a trail of
/// 0x7F or above 0xFC cannot come from an encoder, and is evidence of a
/// wrong read. Every lead the thirteen bits reach is allowed: ISO/IEC
/// 18004 gives kanji mode the range 8140 to 9FFC and E040 to EBBF, and a
/// code in it that JIS X 0208 leaves unassigned is a character the set
/// lacks, which the text decoder notes, not a wrong read. ZXing and
/// zxing-cpp pass every value to the character set converter, which
/// turns an unassigned one into a replacement character.
pub fn kanji_code(value: u32) -> (u16, bool) {
    let (high, low) = (value / 0xC0, value % 0xC0);
    let packed = high << 8 | low;
    let code = packed + if packed < 0x1F00 { 0x8140 } else { 0xC140 };
    let trail = low + 0x40;
    let valid = trail != 0x7F && trail <= 0xFC;
    (code as u16, valid)
}

/// The GB 2312 code a 13-bit hanzi value stands for, and whether GB 2312
/// can have it. GB/T 18284: the two bytes less 0xA1A1 (rows up to 0x09)
/// or 0xA6A1 (rows from 0x0A), the first times 0x60. Columns run from
/// 0xA1 to 0xFE, so a low part above 0x5D is impossible (and would carry
/// into the row), and rows end at 0xF7.
pub fn hanzi_code(value: u32) -> (u16, bool) {
    let (high, low) = (value / 0x60, value % 0x60);
    let packed = high << 8 | low;
    let code = packed + if packed < 0x0A00 { 0xA1A1 } else { 0xA6A1 };
    let valid = low <= 0x5D && code >> 8 <= 0xF7;
    (code as u16, valid)
}

/// What a standard encoder writes after the data in a stream of `total`
/// bits whose data ends at `end_bit`: a terminator of up to
/// `terminator_bits` zeros, zeros to the next codeword boundary, then the
/// pad codewords 0xEC and 0x11 in turn, with a final four-bit codeword, as
/// M1 and M3 have, zero. One expected value per bit from `end_bit` on.
pub fn standard_tail(terminator_bits: usize, total: usize, end_bit: usize) -> Vec<bool> {
    let end_bit = end_bit.min(total);
    let terminator_end = (end_bit + terminator_bits).min(total);
    let first_pad = terminator_end.div_ceil(8);
    let mut expected = vec![false; total - end_bit];
    let mut j = first_pad;
    while 8 * j < total {
        let nbits = (total - 8 * j).min(8);
        let value = if nbits == 4 { 0 } else { PAD_BYTES[(j - first_pad) % 2] };
        for bit in 0..nbits {
            expected[8 * j + bit - end_bit] = value & (0x80 >> bit) != 0;
        }
        j += 1;
    }
    expected
}

/// How the readable bits after the end of data compare with what should
/// be there.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TailAgreement {
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

impl TailAgreement {
    /// The tail looks like padding: either all zeros, or the standard
    /// pattern with no confirmed contradiction and few unconfirmed ones.
    pub fn is_plausible(&self) -> bool {
        let compared = self.matches + self.mismatches;
        compared == 0 || self.zero_filled || (self.verified_mismatches == 0 && self.mismatches * 8 <= compared)
    }
}

/// Compare the stream from `end_bit` with `expected`, one value per bit.
pub fn tail_agreement(stream: &StreamBits, end_bit: usize, expected: &[bool]) -> TailAgreement {
    let mut check = TailAgreement { zero_filled: true, ..TailAgreement::default() };
    for (i, &want) in expected.iter().enumerate() {
        let bit = end_bit + i;
        if stream.certainty[bit] == Certainty::Unknown {
            check.unknown += 1;
        } else {
            check.zero_filled &= !stream.values[bit];
            if stream.values[bit] == want {
                check.matches += 1;
            } else {
                check.mismatches += 1;
                check.verified_mismatches += u32::from(stream.certainty[bit] == Certainty::Verified);
            }
        }
    }
    check.zero_filled &= check.matches + check.mismatches > 0;
    check
}

/// One pass of the parser.
struct Parsed {
    segments: Vec<Segment>,
    end: EndOfData,
    /// The header field whose unreadable bits stopped the parse.
    blocked: Option<Range<usize>>,
    preamble_damaged: bool,
}

pub fn parse(stream: &StreamBits, spec: &StreamSpec) -> ParseReport {
    let first = parse_once(stream, spec);
    let preamble_damaged = first.preamble_damaged;
    let Some(field) = first.blocked.clone() else {
        return ParseReport { segments: first.segments, end: first.end, header_inference: None, preamble_damaged };
    };
    // With nothing readable after the header, no reading of it can be
    // told from another.
    if stream.certainty[field.end..].iter().all(|&c| c == Certainty::Unknown) {
        return ParseReport { segments: first.segments, end: first.end, header_inference: None, preamble_damaged };
    }
    let (inference, unique) = infer_headers(stream, spec, field);
    match unique {
        Some((guessed, parsed)) => {
            let mut segments = parsed.segments;
            for segment in &mut segments {
                let span = segment.header.start..segment.payload.end;
                if guessed.iter().any(|(bit, _)| span.contains(bit)) {
                    segment.header_inferred = true;
                    segment.header_certainty = Certainty::Unknown;
                }
            }
            let end = match parsed.end {
                EndOfData::At { bit, .. } => EndOfData::At { bit, certainty: Certainty::Unknown },
                lost => lost,
            };
            ParseReport { segments, end, header_inference: Some(inference), preamble_damaged }
        }
        None => ParseReport {
            segments: first.segments,
            end: first.end,
            header_inference: Some(inference),
            preamble_damaged,
        },
    }
}

fn parse_once(stream: &StreamBits, spec: &StreamSpec) -> Parsed {
    let total = stream.len();
    let micro = spec.version.family() == Family::Micro;
    let mode_bits = spec.mode_bits as usize;
    // Micro QR has no terminator indicator; its terminator reads as a numeric
    // header with a zero count, so that many bits must remain for a header.
    let min_header = if micro { spec.terminator_bits as usize } else { mode_bits };

    let mut segments = Vec::new();
    // A Model 1 stream begins with four zero bits, which zxing-cpp drops;
    // here they are stepped over, and a one among them noted as damage.
    let preamble = (spec.preamble_bits as usize).min(total);
    let preamble_damaged = (0..preamble).any(|i| stream.certainty[i] != Certainty::Unknown && stream.values[i]);
    let mut pos = preamble;
    let mut certainty = Certainty::Verified;
    let mut blocked = None;

    let end = loop {
        if total - pos < min_header {
            break EndOfData::At { bit: pos, certainty };
        }
        let lost = move |reason| EndOfData::Lost { bit: pos, reason };

        let indicator = stream.field(pos..pos + mode_bits);
        if indicator.certainty == Certainty::Unknown {
            blocked = Some(pos..pos + mode_bits);
            break lost(LostReason::UnknownHeader);
        }
        let Some(mode) = spec.mode(indicator.value) else {
            break lost(LostReason::InvalidMode(indicator.value));
        };
        let mut header_certainty = indicator.certainty;
        let mut cursor = pos + mode_bits;

        match mode {
            Mode::Terminator => break EndOfData::At { bit: pos, certainty: certainty.min(header_certainty) },

            Mode::Numeric | Mode::Alphanumeric | Mode::Byte | Mode::Kanji | Mode::Hanzi => {
                let mut damaged = false;
                if mode == Mode::Hanzi {
                    // A four-bit subset indicator comes before the count;
                    // only GB 2312 (0001) is defined.
                    if total - cursor < 4 {
                        break lost(LostReason::Overrun);
                    }
                    let subset = stream.field(cursor..cursor + 4);
                    if subset.certainty == Certainty::Unknown {
                        blocked = Some(cursor..cursor + 4);
                        break lost(LostReason::UnknownHeader);
                    }
                    header_certainty = header_certainty.min(subset.certainty);
                    damaged |= subset.value != 1;
                    cursor += 4;
                }
                let count_bits = spec.count_bits(mode).expect("data mode has a count") as usize;
                if total - cursor < count_bits {
                    break lost(LostReason::Overrun);
                }
                let count_field = stream.field(cursor..cursor + count_bits);
                if count_field.certainty == Certainty::Unknown {
                    blocked = Some(cursor..cursor + count_bits);
                    break lost(LostReason::UnknownHeader);
                }
                header_certainty = header_certainty.min(count_field.certainty);
                let count = count_field.value as usize;
                if micro && mode == Mode::Numeric && count == 0 {
                    break EndOfData::At { bit: pos, certainty: certainty.min(header_certainty) };
                }
                cursor += count_bits;

                let need = payload_bits(mode, count);
                let truncated = need > total - cursor;
                let payload_end = if truncated { total } else { cursor + need };
                let (content, payload_damaged) = parse_payload(stream, mode, count, cursor, payload_end);
                damaged |= payload_damaged;
                segments.push(Segment {
                    mode,
                    header: pos..cursor,
                    payload: cursor..payload_end,
                    count: Some(count),
                    header_certainty,
                    content,
                    damaged,
                    truncated,
                    header_inferred: false,
                });
                certainty = certainty.min(header_certainty);
                if truncated {
                    break lost(LostReason::Overrun);
                }
                pos = payload_end;
            }

            Mode::Eci => {
                // One, two or three bytes, chosen by the leading bits:
                // 0, 10 and 110. A lead of 111 is not defined, and with it
                // the length of the designator is unknown.
                if total - cursor < 8 {
                    break lost(LostReason::Overrun);
                }
                let mut length = 8;
                let mut examined = 1;
                let mut prefix_certainty = stream.certainty[cursor];
                if stream.values[cursor] {
                    length = 16;
                    examined = 2;
                    prefix_certainty = prefix_certainty.min(stream.certainty[cursor + 1]);
                    if stream.values[cursor + 1] {
                        length = 24;
                        examined = 3;
                        prefix_certainty = prefix_certainty.min(stream.certainty[cursor + 2]);
                        if stream.values[cursor + 2] {
                            break lost(LostReason::InvalidEci);
                        }
                    }
                }
                if prefix_certainty == Certainty::Unknown {
                    blocked = Some(cursor..cursor + examined);
                    break lost(LostReason::UnknownHeader);
                }
                if total - cursor < length {
                    break lost(LostReason::Overrun);
                }
                header_certainty = header_certainty.min(prefix_certainty);
                let field = stream.field(cursor..cursor + length);
                let value_bits = match length {
                    8 => 7,
                    16 => 14,
                    _ => 21,
                };
                // Assignment numbers run to 999999; the three-byte form has
                // room for more, and anything above cannot occur.
                let value = (field.certainty != Certainty::Unknown).then(|| field.value & ((1 << value_bits) - 1));
                let damaged = value.is_some_and(|v| v >= 1_000_000);
                segments.push(Segment {
                    mode,
                    header: pos..cursor,
                    payload: cursor..cursor + length,
                    count: None,
                    header_certainty,
                    content: SegmentContent::Eci(value),
                    damaged,
                    truncated: false,
                    header_inferred: false,
                });
                certainty = certainty.min(header_certainty);
                pos = cursor + length;
            }

            Mode::StructuredAppend | Mode::Fnc1First | Mode::Fnc1Second => {
                let length = match mode {
                    Mode::StructuredAppend => 16,
                    Mode::Fnc1Second => 8,
                    _ => 0,
                };
                if total - cursor < length {
                    break lost(LostReason::Overrun);
                }
                let field = stream.field(cursor..cursor + length);
                let content = match mode {
                    Mode::Fnc1First => SegmentContent::Fnc1First,
                    _ if field.certainty == Certainty::Unknown => SegmentContent::Unreadable,
                    Mode::StructuredAppend => SegmentContent::StructuredAppend {
                        index: (field.value >> 12) as u8 + 1,
                        total: (field.value >> 8 & 0xF) as u8 + 1,
                        parity: field.value as u8,
                    },
                    _ => SegmentContent::Fnc1Second(field.value as u8),
                };
                if content != SegmentContent::Unreadable {
                    header_certainty = header_certainty.min(field.certainty);
                }
                // An application indicator is two digits (0 to 99) or a
                // letter (A to Z as 165 to 190, a to z as 197 to 222).
                let damaged = matches!(content, SegmentContent::Fnc1Second(ai) if !application_indicator_valid(ai));
                segments.push(Segment {
                    mode,
                    header: pos..cursor,
                    payload: cursor..cursor + length,
                    count: None,
                    header_certainty,
                    content,
                    damaged,
                    truncated: false,
                    header_inferred: false,
                });
                certainty = certainty.min(header_certainty);
                pos = cursor + length;
            }
        }
    };

    Parsed { segments, end, blocked, preamble_damaged }
}

/// The state of one search through the readings of unreadable headers.
struct Search {
    trials: usize,
    /// Readings that parsed cleanly: the bits each guessed, and the parse.
    survivors: Vec<(Vec<(usize, bool)>, Parsed)>,
    /// Some reading needed more guesses than the depth allows.
    cut: bool,
    /// The trial budget ran out.
    exhausted: bool,
}

/// A reading of a header that parses: the (bit, value) pairs it guessed,
/// and the parse under them.
type Reading = (Vec<(usize, bool)>, Parsed);

/// Try the readings of the header blocked at `field`, fewest guessed bits
/// first, stopping at the first depth where some reading parses cleanly.
/// Returns what was found and, when exactly one reading parses, that
/// reading with the bits it guessed.
fn infer_headers(stream: &StreamBits, spec: &StreamSpec, field: Range<usize>) -> (HeaderInference, Option<Reading>) {
    let mut trial = stream.clone();
    let mut search = Search { trials: 0, survivors: Vec::new(), cut: false, exhausted: false };
    for depth in 1..=MAX_GUESSED_HEADER_BITS {
        search.survivors.clear();
        search.cut = false;
        explore(&mut trial, stream, spec, field.clone(), &mut Vec::new(), depth, &mut search);
        let candidates = search.survivors.len();
        if search.exhausted {
            let inference = HeaderInference { candidates, guessed_bits: depth, complete: false, inferred: Vec::new() };
            return (inference, None);
        }
        if candidates > 0 {
            let inferred = agreed_bits(&search.survivors);
            let unique = (candidates == 1).then(|| search.survivors.pop().expect("one survivor"));
            return (HeaderInference { candidates, guessed_bits: depth, complete: true, inferred }, unique);
        }
        if !search.cut {
            // Every reading ran into an impossibility: there is nothing
            // deeper to find, and the stream is wrong somewhere.
            return (HeaderInference { candidates: 0, guessed_bits: depth, complete: true, inferred: Vec::new() }, None);
        }
    }
    let inference =
        HeaderInference { candidates: 0, guessed_bits: MAX_GUESSED_HEADER_BITS, complete: false, inferred: Vec::new() };
    (inference, None)
}

/// Depth-first over the unreadable bits of `field`, then of whatever
/// header blocks next, up to `depth` guessed bits in all. `original` is
/// the stream as read, for telling guessed bits from known ones.
fn explore(
    trial: &mut StreamBits,
    original: &StreamBits,
    spec: &StreamSpec,
    field: Range<usize>,
    path: &mut Vec<(usize, bool)>,
    depth: usize,
    search: &mut Search,
) {
    let unknown: Vec<usize> = field.filter(|&i| trial.certainty[i] == Certainty::Unknown).collect();
    if path.len() + unknown.len() > depth {
        search.cut = true;
        return;
    }
    let base = path.len();
    for assignment in 0..1u32 << unknown.len() {
        if search.trials >= MAX_HEADER_TRIALS {
            search.exhausted = true;
            break;
        }
        search.trials += 1;
        for (k, &i) in unknown.iter().enumerate() {
            let value = assignment >> k & 1 != 0;
            trial.values[i] = value;
            trial.certainty[i] = Certainty::Unverified;
            path.push((i, value));
        }
        let parsed = parse_once(trial, spec);
        match parsed.blocked.clone() {
            Some(next) => explore(trial, original, spec, next, path, depth, search),
            None => {
                if let Some(end) = survives(&parsed, trial, original, spec) {
                    let guessed = path.iter().copied().filter(|&(bit, _)| bit < end).collect();
                    search.survivors.push((guessed, parsed));
                }
            }
        }
        path.truncate(base);
        for &i in &unknown {
            trial.values[i] = false;
            trial.certainty[i] = Certainty::Unknown;
        }
        if search.exhausted {
            break;
        }
    }
}

/// A reading parses cleanly: it reaches a terminator that was read, with
/// no impossible value or overrun on the way, and enough of what follows
/// is readable and looks like padding. Returns where its data ends. A
/// reading that has to guess its own terminator, or that swallows the
/// padding as payload and stops near capacity, explains nothing the
/// stream shows: a control segment followed by such a guess would pass
/// for any message, so neither counts.
fn survives(parsed: &Parsed, trial: &StreamBits, original: &StreamBits, spec: &StreamSpec) -> Option<usize> {
    let EndOfData::At { bit, .. } = parsed.end else { return None };
    if parsed.segments.iter().any(|s| s.damaged || s.truncated) {
        return None;
    }
    let terminator_bits = spec.terminator_bits as usize;
    let terminator = bit..(bit + terminator_bits).min(original.len());
    if original.certainty[terminator].contains(&Certainty::Unknown) {
        return None;
    }
    let expected = standard_tail(terminator_bits, trial.len(), bit);
    let agreement = tail_agreement(trial, bit, &expected);
    (agreement.is_plausible() && agreement.matches >= terminator_bits as u32 + MIN_TAIL_EVIDENCE).then_some(bit)
}

/// The guessed bits every survivor shares.
fn agreed_bits(survivors: &[(Vec<(usize, bool)>, Parsed)]) -> Vec<(usize, bool)> {
    let Some(((first, _), rest)) = survivors.split_first() else { return Vec::new() };
    first.iter().copied().filter(|guess| rest.iter().all(|(path, _)| path.contains(guess))).collect()
}

/// True for the values an FNC1 second-position application indicator can take.
pub fn application_indicator_valid(ai: u8) -> bool {
    ai < 100 || (165..=190).contains(&ai) || (197..=222).contains(&ai)
}

/// The application indicator as text: two digits or one letter.
pub fn application_indicator_text(ai: u8) -> Option<String> {
    match ai {
        0..=99 => Some(format!("{ai:02}")),
        165..=190 | 197..=222 => Some(char::from(ai - 100).to_string()),
        _ => None,
    }
}

fn parse_payload(
    stream: &StreamBits,
    mode: Mode,
    count: usize,
    start: usize,
    end: usize,
) -> (SegmentContent, bool) {
    let mut cursor = start;
    let mut damaged = false;
    // Reads the next group, or `None` once the payload is cut short.
    let mut take = |width: usize| -> Option<(Field, Range<usize>)> {
        (cursor + width <= end).then(|| {
            let bits = cursor..cursor + width;
            cursor += width;
            (stream.field(bits.clone()), bits)
        })
    };

    match mode {
        Mode::Numeric => {
            let mut cells = Vec::with_capacity(count);
            let mut left = count;
            while left > 0 {
                let digits = left.min(3);
                let Some((field, bits)) = take([4, 7, 10][digits - 1]) else { break };
                let valid = field.value < [10, 100, 1000][digits - 1];
                let known = field.certainty != Certainty::Unknown;
                damaged |= known && !valid;
                for d in 0..digits {
                    let digit = field.value / 10u32.pow((digits - 1 - d) as u32) % 10;
                    cells.push(Cell {
                        value: (known && valid).then(|| char::from(b'0' + digit as u8)),
                        certainty: field.certainty,
                        bits: bits.clone(),
                    });
                }
                left -= digits;
            }
            (SegmentContent::Chars(cells), damaged)
        }
        Mode::Alphanumeric => {
            let mut cells = Vec::with_capacity(count);
            let mut left = count;
            while left > 0 {
                let chars = left.min(2);
                let Some((field, bits)) = take([6, 11][chars - 1]) else { break };
                let valid = field.value < [45, 45 * 45][chars - 1];
                let known = field.certainty != Certainty::Unknown;
                damaged |= known && !valid;
                for c in 0..chars {
                    let index = if chars == 2 && c == 0 { field.value / 45 } else { field.value % 45 };
                    cells.push(Cell {
                        value: (known && valid).then(|| char::from(ALPHANUMERIC[index as usize % 45])),
                        certainty: field.certainty,
                        bits: bits.clone(),
                    });
                }
                left -= chars;
            }
            (SegmentContent::Chars(cells), damaged)
        }
        Mode::Byte => {
            let mut cells = Vec::with_capacity(count);
            for _ in 0..count {
                let Some((field, bits)) = take(8) else { break };
                cells.push(Cell {
                    value: (field.certainty != Certainty::Unknown).then_some(field.value as u8),
                    certainty: field.certainty,
                    bits,
                });
            }
            (SegmentContent::Bytes(cells), false)
        }
        Mode::Kanji => {
            let mut cells = Vec::with_capacity(count);
            for _ in 0..count {
                let Some((field, bits)) = take(13) else { break };
                let (code, valid) = kanji_code(field.value);
                let known = field.certainty != Certainty::Unknown;
                damaged |= known && !valid;
                cells.push(Cell { value: (known && valid).then_some(code), certainty: field.certainty, bits });
            }
            (SegmentContent::Kanji(cells), damaged)
        }
        Mode::Hanzi => {
            let mut cells = Vec::with_capacity(count);
            for _ in 0..count {
                let Some((field, bits)) = take(13) else { break };
                let (code, valid) = hanzi_code(field.value);
                let known = field.certainty != Certainty::Unknown;
                damaged |= known && !valid;
                cells.push(Cell { value: (known && valid).then_some(code), certainty: field.certainty, bits });
            }
            (SegmentContent::Hanzi(cells), damaged)
        }
        _ => unreachable!("not a data mode"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::SymbolVersion;

    fn bits(text: &str) -> StreamBits {
        let mut stream = StreamBits::default();
        for ch in text.chars() {
            match ch {
                '0' => stream.push(false, Certainty::Verified),
                '1' => stream.push(true, Certainty::Verified),
                'u' => stream.push(false, Certainty::Unverified),
                'U' => stream.push(true, Certainty::Unverified),
                '?' => stream.push(false, Certainty::Unknown),
                _ => {}
            }
        }
        stream
    }

    fn chars(segment: &Segment) -> String {
        match &segment.content {
            SegmentContent::Chars(cells) => cells.iter().map(|c| c.value.unwrap_or('~')).collect(),
            other => panic!("not chars: {other:?}"),
        }
    }

    #[test]
    fn numeric_groups() {
        // ISO 18004 example: "01234567" in a 1-H symbol.
        let stream = bits("0001 0000001000 0000001100 0101011001 1000011 0000");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Qr(1)));
        assert_eq!(report.segments.len(), 1);
        assert_eq!(chars(&report.segments[0]), "01234567");
        assert_eq!(report.segments[0].header, 0..14);
        assert_eq!(report.segments[0].payload, 14..41);
        assert_eq!(report.end, EndOfData::At { bit: 41, certainty: Certainty::Verified });
    }

    #[test]
    fn alphanumeric_pairs() {
        // "AC-42": (10,12) (41,4) (2)
        let stream = bits("0010 000000101 00111001110 11100111001 000010 0000");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Qr(1)));
        assert_eq!(chars(&report.segments[0]), "AC-42");
        assert!(!report.segments[0].damaged);
    }

    #[test]
    fn unknown_payload_keeps_segment_boundaries() {
        let stream = bits("0100 00000010 0100?001 01000010 0001 0000000001 0111 0000");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Qr(1)));
        assert_eq!(report.segments.len(), 2);
        let SegmentContent::Bytes(cells) = &report.segments[0].content else { panic!() };
        assert_eq!(cells[0].value, None);
        assert_eq!(cells[1].value, Some(b'B'));
        assert_eq!(chars(&report.segments[1]), "7");
        assert!(matches!(report.end, EndOfData::At { bit: 46, .. }));
    }

    #[test]
    fn unknown_header_loses_the_stream() {
        let stream = bits("0100 0000?010 01000001");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Qr(1)));
        assert!(report.segments.is_empty());
        assert_eq!(report.end, EndOfData::Lost { bit: 0, reason: LostReason::UnknownHeader });
    }

    #[test]
    fn unverified_headers_lower_the_end_certainty() {
        let stream = bits("0100 0000000U 01000001 0000");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Qr(1)));
        assert_eq!(report.segments[0].header_certainty, Certainty::Unverified);
        assert_eq!(report.end, EndOfData::At { bit: 20, certainty: Certainty::Unverified });
    }

    #[test]
    fn invalid_values_are_flagged() {
        // Numeric group 1111111111 = 1023.
        let stream = bits("0001 0000000011 1111111111 0000");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Qr(1)));
        assert!(report.segments[0].damaged);
        assert_eq!(chars(&report.segments[0]), "~~~");
        // Mode 6 does not exist in QR.
        let report = parse(&bits("0110 0000"), &StreamSpec::new(SymbolVersion::Qr(1)));
        assert_eq!(report.end, EndOfData::Lost { bit: 0, reason: LostReason::InvalidMode(6) });
    }

    #[test]
    fn overrun_is_reported_with_partial_content() {
        let stream = bits("0100 00000101 01000001 01000010");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Qr(1)));
        assert!(report.segments[0].truncated);
        let SegmentContent::Bytes(cells) = &report.segments[0].content else { panic!() };
        assert_eq!(cells.len(), 2);
        assert_eq!(report.end, EndOfData::Lost { bit: 0, reason: LostReason::Overrun });
    }

    #[test]
    fn eci_and_control_segments() {
        // ECI 26, then FNC1 first position, then one byte.
        let stream = bits("0111 00011010 0101 0100 00000001 11000011 0000");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Qr(1)));
        assert_eq!(report.segments[0].content, SegmentContent::Eci(Some(26)));
        assert_eq!(report.segments[1].content, SegmentContent::Fnc1First);
        assert_eq!(report.segments[2].mode, Mode::Byte);
        // Two-byte ECI 1000 = 10 000011 11101000.
        let stream = bits("0111 10000011 11101000 0000");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Qr(1)));
        assert_eq!(report.segments[0].content, SegmentContent::Eci(Some(1000)));
        // Structured append: symbol 2 of 3, parity 0x5A.
        let stream = bits("0011 0001 0010 01011010 0000");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Qr(1)));
        assert_eq!(
            report.segments[0].content,
            SegmentContent::StructuredAppend { index: 2, total: 3, parity: 0x5A }
        );
    }

    #[test]
    fn hanzi_segments_decode_to_gb2312() {
        // Mode 1101, subset 0001, count 2, then 中 (0xD6D0) and 文 (0xCEC4):
        // 0xD6D0 - 0xA6A1 = 0x302F -> 0x30 * 0x60 + 0x2F = 4655;
        // 0xCEC4 - 0xA6A1 = 0x2823 -> 0x28 * 0x60 + 0x23 = 3875.
        let stream = bits("1101 0001 00000010 1001000101111 0111100100011 0000");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Qr(1)));
        assert_eq!(report.segments.len(), 1);
        let segment = &report.segments[0];
        assert_eq!(segment.mode, Mode::Hanzi);
        assert_eq!(segment.header, 0..16);
        assert!(!segment.damaged);
        let SegmentContent::Hanzi(cells) = &segment.content else { panic!("{:?}", segment.content) };
        assert_eq!(cells.iter().map(|c| c.value).collect::<Vec<_>>(), [Some(0xD6D0), Some(0xCEC4)]);
        assert!(matches!(report.end, EndOfData::At { bit: 42, .. }));
        // A subset other than GB 2312 is not defined, and a row past 0xF7
        // cannot occur: both are evidence of damage.
        let report = parse(&bits("1101 0010 00000001 1001000101111 0000"), &StreamSpec::new(SymbolVersion::Qr(1)));
        assert!(report.segments[0].damaged);
        let report = parse(&bits("1101 0001 00000001 1111111111111 0000"), &StreamSpec::new(SymbolVersion::Qr(1)));
        assert!(report.segments[0].damaged);
        let SegmentContent::Hanzi(cells) = &report.segments[0].content else { panic!() };
        assert_eq!(cells[0].value, None);
        // Not a mode in the other families: Micro QR's three-bit indicators
        // stop at kanji.
        let report = parse(&bits("101 0001 0000"), &StreamSpec::new(SymbolVersion::Micro(4)));
        assert_eq!(report.end, EndOfData::Lost { bit: 0, reason: LostReason::InvalidMode(5) });
    }

    #[test]
    fn eci_designators_are_checked() {
        // A 111 lead byte has no defined length.
        let report = parse(&bits("0111 11100000 00000000 00000000 0000"), &StreamSpec::new(SymbolVersion::Qr(1)));
        assert_eq!(report.end, EndOfData::Lost { bit: 0, reason: LostReason::InvalidEci });
        // Assignment numbers stop at 999999: 1100 0000 ... is 1048576 and up.
        let report = parse(&bits("0111 11010000 00000000 00000000 0000"), &StreamSpec::new(SymbolVersion::Qr(1)));
        assert!(report.segments[0].damaged);
        assert_eq!(report.segments[0].content, SegmentContent::Eci(Some(0x100000)));
        // 999999 = 0x0F423F is the last valid one.
        let report = parse(&bits("0111 11001111 01000010 00111111 0000"), &StreamSpec::new(SymbolVersion::Qr(1)));
        assert!(!report.segments[0].damaged);
        assert_eq!(report.segments[0].content, SegmentContent::Eci(Some(999_999)));
        // An unknown length bit is tried both ways: three bytes would run
        // past the end, so the two-byte reading is inferred.
        let report = parse(&bits("0111 1?000000 00000000 0000 0000"), &StreamSpec::new(SymbolVersion::Qr(1)));
        assert_eq!(report.end, EndOfData::At { bit: 20, certainty: Certainty::Unknown });
        assert_eq!(report.segments[0].content, SegmentContent::Eci(Some(0)));
        assert!(report.segments[0].header_inferred);
        let inference = report.header_inference.unwrap();
        assert_eq!((inference.candidates, inference.guessed_bits, inference.complete), (1, 1, true));
        assert_eq!(inference.inferred, [(5, false)]);
    }

    #[test]
    fn terminator_certainty_counts() {
        // An unverified terminator indicator is not proof of where the
        // data ends: the end is as sure as the terminator's own bits.
        let stream = bits("0100 00000001 01000001 u000");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Qr(1)));
        assert_eq!(report.end, EndOfData::At { bit: 20, certainty: Certainty::Unverified });
        let stream = bits("011 010 01001000 01101001 u00");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Rmqr(0)));
        assert_eq!(report.end, EndOfData::At { bit: 22, certainty: Certainty::Unverified });
    }

    #[test]
    fn impossible_kanji_and_hanzi_values_are_damage() {
        // 13-bit kanji values whose trail byte would be 0x7F or 0xFF.
        for value in [0x3Fu32, 0xBF] {
            let (_, valid) = kanji_code(value);
            assert!(!valid, "{value:#x}");
            let stream = bits(&format!("1000 00000001 {value:013b} 0000"));
            let report = parse(&stream, &StreamSpec::new(SymbolVersion::Qr(1)));
            assert!(report.segments[0].damaged, "{value:#x}");
            let SegmentContent::Kanji(cells) = &report.segments[0].content else { panic!() };
            assert_eq!(cells[0].value, None);
        }
        // A lead byte of 0xEB is within the range the standard gives
        // kanji mode, though JIS X 0208 assigns nothing there: 0x1F80 is
        // 0xEB40 and 0x1FFF, the largest value, 0xEBBF.
        assert_eq!(kanji_code(0x1F80), (0xEB40, true));
        assert_eq!(kanji_code(0x1FFF), (0xEBBF, true));
        // 点 (0x935F) and 茗 (0xE4AA) are fine: 0x935F - 0x8140 = 0x121F ->
        // 0x12 * 0xC0 + 0x1F = 3487; 0xE4AA - 0xC140 = 0x236A -> 0x23 * 0xC0 + 0x6A = 6826.
        assert_eq!(kanji_code(3487), (0x935F, true));
        assert_eq!(kanji_code(6826), (0xE4AA, true));
        let stream = bits("1000 00000010 0110110011111 1101010101010 0000");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Qr(1)));
        assert!(!report.segments[0].damaged);
        // Hanzi: a low part of 0x5E or 0x5F would be column 0xFF or carry
        // into the next row, which GB 2312 cannot have.
        assert!(!hanzi_code(0x5E).1);
        assert!(!hanzi_code(0x5F).1);
        assert_eq!(hanzi_code(0x5D), (0xA1FE, true));
        for value in [0x5Eu32, 0x5F] {
            let stream = bits(&format!("1101 0001 00000001 {value:013b} 0000"));
            let report = parse(&stream, &StreamSpec::new(SymbolVersion::Qr(1)));
            assert!(report.segments[0].damaged, "{value:#x}");
        }
    }

    #[test]
    fn standard_tail_matches_the_encoders() {
        // Data ends at bit 20 of 152 (version 1-H): four terminator bits
        // reach byte 3, then 0xEC 0x11 ... to the end.
        let tail = standard_tail(4, 152, 20);
        assert_eq!(tail.len(), 132);
        assert!(tail[..4].iter().all(|&b| !b));
        let byte = |j: usize| tail[8 * j - 20..8 * j - 12].iter().fold(0u8, |a, &b| a << 1 | u8::from(b));
        assert_eq!((byte(3), byte(4), byte(5), byte(18)), (0xEC, 0x11, 0xEC, 0x11));
        // M1 holds 20 data bits whose last four are a four-bit codeword, zero
        // as padding; a terminator cut short by the end counts as zeros too.
        assert_eq!(standard_tail(3, 20, 9), vec![false; 11]);
        assert_eq!(standard_tail(3, 20, 5), [vec![false; 3], vec![true, true, true, false, true, true, false, false], vec![false; 4]].concat());
        // Agreement: unknown bits are neither for nor against.
        // Twenty bits ending at 0: terminator, zeros to byte 1, 0xEC, a
        // zero nibble.
        let stream = bits("0000 ?000 1110 1100 0000");
        let agreement = tail_agreement(&stream, 0, &standard_tail(4, 20, 0));
        assert_eq!((agreement.matches, agreement.mismatches, agreement.unknown), (19, 0, 1));
        assert!(agreement.is_plausible());
        let stream = bits("0000 1111 0000 0000 0000");
        let agreement = tail_agreement(&stream, 0, &standard_tail(4, 20, 0));
        assert_eq!((agreement.matches, agreement.verified_mismatches), (11, 9));
        assert!(!agreement.is_plausible());
        assert!(tail_agreement(&bits("0000 0000 0000"), 0, &standard_tail(4, 12, 0)).zero_filled);
    }

    #[test]
    fn unreadable_header_bits_are_inferred_when_one_reading_parses() {
        // A count of 2 or 10 bytes: ten would run past the end, two end at
        // the terminator with zeros beyond it.
        let stream = bits("0100 0000?010 01000001 01000010 0000 0000");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Qr(1)));
        assert_eq!(report.segments.len(), 1);
        assert_eq!(chars_of_bytes(&report.segments[0]), "AB");
        assert!(report.segments[0].header_inferred);
        assert_eq!(report.segments[0].header_certainty, Certainty::Unknown);
        assert_eq!(report.end, EndOfData::At { bit: 28, certainty: Certainty::Unknown });
        let inference = report.header_inference.as_ref().unwrap();
        assert_eq!((inference.candidates, inference.guessed_bits, inference.complete), (1, 1, true));
        assert_eq!(inference.inferred, [(8, false)]);
        // The payload keeps its own certainties: the guess is the header's.
        let SegmentContent::Bytes(cells) = &report.segments[0].content else { panic!() };
        assert!(cells.iter().all(|c| c.certainty == Certainty::Verified));

        // Two readings parse: one or three bytes, both ending in a
        // terminator with zeros after. Nothing is inferred but the count
        // of them is known; the stream stays lost as before.
        let stream = bits("0100 000000?1 01000001 00000000 00000000 0000 0000");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Qr(1)));
        assert!(report.segments.is_empty());
        assert_eq!(report.end, EndOfData::Lost { bit: 0, reason: LostReason::UnknownHeader });
        let inference = report.header_inference.as_ref().unwrap();
        assert_eq!((inference.candidates, inference.complete), (2, true));
        assert!(inference.inferred.is_empty());

        // Readings that agree on a bit without being unique still infer
        // it: the mode's second bit is unknown but only byte mode parses, so
        // it is settled while the count's unknown bit is not.
        let stream = bits("0?00 000000?1 01000001 00000000 00000000 0000 0000");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Qr(1)));
        let inference = report.header_inference.as_ref().unwrap();
        assert_eq!((inference.candidates, inference.guessed_bits), (2, 2));
        assert_eq!(inference.inferred, [(1, true)]);

        // No reading parses: every count overruns or hits a mode that
        // does not exist, which says the stream is wrong somewhere.
        let stream = bits("0100 000000?1 01000001 01100000");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Qr(1)));
        let inference = report.header_inference.as_ref().unwrap();
        assert_eq!((inference.candidates, inference.complete), (0, true));

        // With nothing readable after the header there is nothing to test
        // a reading against, and no inference is attempted.
        let stream = bits("0100 0000?010 ???????? ???????? ???? ????");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Qr(1)));
        assert_eq!(report.header_inference, None);
        assert_eq!(report.end, EndOfData::Lost { bit: 0, reason: LostReason::UnknownHeader });

        // Within the depth that is searched, the one reading of a whole
        // unknown header is found: eight bytes end at the terminator that
        // can be seen, with the padding after it. No numeric, alphanumeric
        // or kanji count ends there, and a count that runs on into the
        // padding and stops near capacity explains none of it.
        let mut text = String::from("???? ???????? ");
        text.push_str(&"????????".repeat(8));
        text.push_str(" 0000 11101100 00010001 11101100");
        let report = parse(&bits(&text), &StreamSpec::new(SymbolVersion::Qr(1)));
        let inference = report.header_inference.as_ref().unwrap();
        assert_eq!((inference.candidates, inference.guessed_bits, inference.complete), (1, 12, true));
        assert_eq!(report.segments[0].mode, Mode::Byte);
        assert_eq!(report.segments[0].count, Some(8));
        assert_eq!(inference.inferred.len(), 12);
        // Headers beyond the depth that is searched, or past the trial
        // budget, give up rather than guess: the same stream in version
        // 10, where a byte count is sixteen bits, is out of reach.
        let mut text = String::from("???? ???????????????? ");
        text.push_str(&"????????".repeat(8));
        text.push_str(" 0000 11101100 00010001 11101100");
        let report = parse(&bits(&text), &StreamSpec::new(SymbolVersion::Qr(10)));
        let inference = report.header_inference.as_ref().unwrap();
        assert!(!inference.complete, "{inference:?}");
        assert!(inference.inferred.is_empty());
        assert_eq!(report.end, EndOfData::Lost { bit: 0, reason: LostReason::UnknownHeader });
    }

    fn chars_of_bytes(segment: &Segment) -> String {
        match &segment.content {
            SegmentContent::Bytes(cells) => cells.iter().map(|c| c.value.map_or('~', char::from)).collect(),
            other => panic!("not bytes: {other:?}"),
        }
    }

    #[test]
    fn application_indicators_are_checked() {
        // 165 is A, 42 is "42", 150 is nothing.
        for (value, text, valid) in [(165u8, Some("A"), true), (42, Some("42"), true), (150, None, false), (222, Some("z"), true)] {
            let stream = bits(&format!("1001 {value:08b} 0000"));
            let report = parse(&stream, &StreamSpec::new(SymbolVersion::Qr(1)));
            assert_eq!(report.segments[0].content, SegmentContent::Fnc1Second(value));
            assert_eq!(report.segments[0].damaged, !valid, "{value}");
            assert_eq!(application_indicator_text(value).as_deref(), text);
        }
    }

    #[test]
    fn micro_terminator_is_an_empty_numeric_header() {
        // M2: numeric "12" (mode 0, count 0010, 7 bits), then five zeros.
        let stream = bits("0 0010 0001100 00000 0000000");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Micro(2)));
        assert_eq!(chars(&report.segments[0]), "12");
        assert_eq!(report.end, EndOfData::At { bit: 12, certainty: Certainty::Verified });
        // M1 has no mode indicator at all.
        let stream = bits("011 0001111011 000 0000");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Micro(1)));
        assert_eq!(chars(&report.segments[0]), "123");
        assert!(matches!(report.end, EndOfData::At { bit: 13, .. }));
    }

    #[test]
    fn rmqr_modes() {
        // R7x43: byte mode is 011 with a 3-bit count.
        let stream = bits("011 010 01001000 01101001 000");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Rmqr(0)));
        let SegmentContent::Bytes(cells) = &report.segments[0].content else { panic!() };
        assert_eq!(cells.iter().map(|c| c.value.unwrap()).collect::<Vec<_>>(), b"Hi");
        assert!(matches!(report.end, EndOfData::At { bit: 22, .. }));
    }

    #[test]
    fn stream_running_to_capacity_ends_cleanly() {
        let stream = bits("0100 00000001 01000001");
        let report = parse(&stream, &StreamSpec::new(SymbolVersion::Qr(1)));
        assert_eq!(report.end, EndOfData::At { bit: 20, certainty: Certainty::Verified });
    }
}
