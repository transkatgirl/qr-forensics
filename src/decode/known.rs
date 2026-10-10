//! Pinning text the message is already known to hold.
//!
//! Whoever repairs a symbol seldom knows nothing of what it says: the
//! address on a label begins `https://`, the sticker on a router
//! `WIFI:`, and half a serial number can be read off the print beside
//! the symbol. Every character known is bits of the data stream that
//! need not be read, and a pin on them costs no error correction: a
//! version 3-L symbol corrects seven wrong codewords, and with fifteen
//! wrong, twelve of them under a known `https://example.com/`, it
//! solves once those twenty characters are pinned.
//!
//! The codeword inspector pins a whole codeword at a time, in
//! hexadecimal, and text does not lie in the stream by the codeword. The
//! header of a byte segment is twelve bits in versions 1 to 9, so every
//! character straddles two codewords and `ht` is pinned as `87 47`;
//! digits are packed three to ten bits and alphanumeric characters two
//! to eleven. So the pins are worked out here, bit for bit, from the
//! text and where it stands: at a character of the message as read, in
//! the segment and character set that character is read in, or at the
//! very start of the message under a header the user names, for a
//! symbol whose own header is too damaged to parse.
//!
//! [`try_beginnings`] tries the ways messages commonly begin, and keeps
//! one only when the blocks its pins fall in then verify: a pin asserts,
//! it does not check, and it is error correction over the rest of the
//! block that bears a beginning out.

use std::collections::BTreeSet;
use std::fmt;

use super::bitstream::{Certainty, SegmentContent, alphanumeric_value, hanzi_value, kanji_value};
use super::codewords::{Pin, PinSource, Pins, codeword_bit_mask};
use super::pipeline::{DecodeOptions, DecodeReport, decode};
use super::text::{self, Charset, GROUP_SEPARATOR};
use crate::symbol::grid::ModuleGrid;
use crate::symbol::layout::SymbolLayout;
use crate::symbol::tables::Mode;

/// Where known text stands in the message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    /// From this character of the message as read, the first being 0, in
    /// the segment that character is read in.
    Char(usize),
    /// At the very start of the message, which is taken to open with a
    /// segment of this mode whatever its header reads as. The mode
    /// indicator is pinned with the text; the character count is not,
    /// since the text may be only the beginning.
    Start(Mode),
}

/// Why text could not be pinned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextPinError {
    /// No text was given.
    Empty,
    /// The message as read has no character at that place; it has `len`.
    NoSuchChar { len: usize },
    /// The character there is of no data segment: an application
    /// indicator is of the header that carries it.
    NotData,
    /// The segment's mode has no such character.
    NotInMode { ch: char, mode: Mode },
    /// The character set the segment's bytes are read in has no such
    /// character.
    NotInCharset { ch: char },
    /// The text needs `needs` characters of room, bytes in byte mode,
    /// and from that place the segment, or the symbol, holds `room`.
    TooLong { needs: usize, room: usize },
    /// The version has no such mode.
    NoSuchMode(Mode),
    /// Every character given is packed with characters not given, three
    /// digits or two alphanumeric characters to a group, and a group is
    /// fixed only by all of its characters.
    NothingWhole,
}

impl fmt::Display for TextPinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            TextPinError::Empty => write!(f, "there is no text to pin"),
            TextPinError::NoSuchChar { len: 0 } => {
                write!(f, "no text was read to place it in; pin it at the start of the message instead")
            }
            TextPinError::NoSuchChar { len } => write!(f, "the message as read has {len} characters, and none at that place"),
            TextPinError::NotData => write!(f, "the character there is an application indicator, which is no message data"),
            TextPinError::NotInMode { ch, mode } => {
                write!(f, "{ch:?} is not a character {} mode has, and that is how the message is packed there", mode.label().to_lowercase())
            }
            TextPinError::NotInCharset { ch } => write!(f, "the character set the bytes there are read in has no {ch:?}"),
            TextPinError::TooLong { needs, room } => write!(f, "the text takes {needs} characters and there is room for {room} from there"),
            TextPinError::NoSuchMode(mode) => write!(f, "this version has no {} mode", mode.label().to_lowercase()),
            TextPinError::NothingWhole => write!(
                f,
                "digits are packed three to a group and alphanumeric characters two, and a group is fixed only by all of its \
                 characters: give the characters beside these as well"
            ),
        }
    }
}

/// The pins known text makes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextPins {
    /// The pins, to be laid over those in force with [`lay_over`].
    pub pins: Pins,
    /// Characters the pins fix: bytes, in byte mode.
    pub pinned: usize,
    /// Characters given that are not pinned, at either end of the text:
    /// those packed in a group with characters that were not given.
    pub left_out: usize,
    /// Pinned bits that differ from bits error correction had confirmed:
    /// the text contradicts a block that solved.
    pub contradicted: usize,
}

/// How a mode packs characters: the bits of a group of one character, of
/// two, and so on up to a full group.
fn group_widths(mode: Mode) -> &'static [usize] {
    match mode {
        Mode::Numeric => &[4, 7, 10],
        Mode::Alphanumeric => &[6, 11],
        Mode::Byte => &[8],
        Mode::Kanji | Mode::Hanzi => &[13],
        _ => &[],
    }
}

/// The value each character of `text` has in `mode`, a byte to a value
/// in byte mode. Under FNC1 an alphanumeric segment writes the group
/// separator as a percent sign and a percent sign as two.
fn values_of(mode: Mode, text: &str, fnc1: bool, bytes: impl FnOnce(&str) -> Result<Vec<u8>, char>) -> Result<Vec<u32>, TextPinError> {
    let lacking = |ch| TextPinError::NotInMode { ch, mode };
    let double_byte = |ch: char, encoding: &'static encoding_rs::Encoding, value: fn(u16) -> Option<u32>| {
        let mut one = [0; 4];
        let (encoded, _, unmapped) = encoding.encode(ch.encode_utf8(&mut one));
        match (unmapped, &encoded[..]) {
            (false, &[high, low]) => value(u16::from_be_bytes([high, low])).ok_or(lacking(ch)),
            _ => Err(lacking(ch)),
        }
    };
    match mode {
        Mode::Numeric => text.chars().map(|ch| ch.to_digit(10).filter(|_| ch.is_ascii()).ok_or(lacking(ch))).collect(),
        Mode::Alphanumeric => {
            let mut values = Vec::with_capacity(text.len());
            for ch in text.chars() {
                let written: &[char] = match ch {
                    GROUP_SEPARATOR if fnc1 => &['%'],
                    '%' if fnc1 => &['%', '%'],
                    _ => &[ch],
                };
                for &w in written {
                    values.push(alphanumeric_value(w).ok_or(lacking(ch))?);
                }
            }
            Ok(values)
        }
        Mode::Byte => Ok(bytes(text).map_err(|ch| TextPinError::NotInCharset { ch })?.into_iter().map(u32::from).collect()),
        Mode::Kanji => text.chars().map(|ch| double_byte(ch, encoding_rs::SHIFT_JIS, kanji_value)).collect(),
        Mode::Hanzi => text.chars().map(|ch| double_byte(ch, encoding_rs::GB18030, hanzi_value)).collect(),
        _ => Err(TextPinError::NotData),
    }
}

/// The stream bits of `values`, the characters from `first` on of a
/// payload that starts at stream bit `payload` and holds `total`
/// characters in all, where that is known. A group is written only when
/// every character of it is among `values`: the groups the run cuts at
/// either end are left out. Returns the bits, each with its value, and
/// how many characters they fix.
fn pack(mode: Mode, payload: usize, total: Option<usize>, first: usize, values: &[u32]) -> (Vec<(usize, bool)>, usize) {
    let widths = group_widths(mode);
    let per = widths.len();
    let radix = match mode {
        Mode::Numeric => 10,
        Mode::Alphanumeric => 45,
        _ => 0,
    };
    let end = first + values.len();
    let (mut bits, mut fixed) = (Vec::new(), 0);
    for group in first / per.max(1).. {
        let start = group * per;
        // The characters the group holds: a full group, or what is left
        // of the payload in its last.
        let size = total.map_or(per, |total| per.min(total.saturating_sub(start)));
        if start >= end || size == 0 {
            break;
        }
        if start < first || start + size > end {
            continue;
        }
        let value = values[start - first..start + size - first].iter().fold(0, |sum, &v| sum * radix + v);
        let (width, at) = (widths[size - 1], payload + group * widths[per - 1]);
        bits.extend((0..width).map(|i| (at + i, value >> (width - 1 - i) & 1 == 1)));
        fixed += size;
    }
    (bits, fixed)
}

/// The pins that fix `text` at `place` in the message `report` read,
/// with byte data in the character set it is read in there, or, at the
/// start of the message, in `charset`. Nothing is decoded: the pins are
/// to be laid over those in force and the grid decoded under them.
pub fn text_pins(layout: &SymbolLayout, report: &DecodeReport, place: Place, text: &str, charset: Charset) -> Result<TextPins, TextPinError> {
    if text.is_empty() {
        return Err(TextPinError::Empty);
    }
    let segments = &report.parse.segments;
    let (mut bits, given, fixed) = match place {
        Place::Char(at) => {
            let source = report.text.sources.get(at).ok_or(TextPinError::NoSuchChar { len: report.text.sources.len() })?;
            let segment = &segments[source.segment];
            let cells = match &segment.content {
                SegmentContent::Chars(cells) => cells.len(),
                SegmentContent::Bytes(cells) => cells.len(),
                SegmentContent::Kanji(cells) | SegmentContent::Hanzi(cells) => cells.len(),
                _ => return Err(TextPinError::NotData),
            };
            let fnc1 = segments.iter().any(|s| matches!(s.content, SegmentContent::Fnc1First | SegmentContent::Fnc1Second(_)));
            let values = values_of(segment.mode, text, fnc1, |text| text::encode_as_read(segments, source.segment, charset, text))?;
            let room = cells.saturating_sub(source.cell);
            if values.len() > room {
                return Err(TextPinError::TooLong { needs: values.len(), room });
            }
            let total = segment.count.unwrap_or(cells);
            let (bits, fixed) = pack(segment.mode, segment.payload.start, Some(total), source.cell, &values);
            (bits, values.len(), fixed)
        }
        Place::Start(mode) => {
            let spec = &layout.stream;
            let count_bits = spec.count_bits(mode).ok_or(TextPinError::NoSuchMode(mode))? as usize;
            let mode_bits = spec.mode_bits as usize;
            let indicator = (0..1u32 << mode_bits).find(|&v| spec.mode(v) == Some(mode)).ok_or(TextPinError::NoSuchMode(mode))?;
            let mut at = spec.preamble_bits as usize;
            let mut bits: Vec<(usize, bool)> = (0..mode_bits).map(|i| (at + i, indicator >> (mode_bits - 1 - i) & 1 == 1)).collect();
            at += mode_bits;
            if mode == Mode::Hanzi {
                // The subset indicator, GB 2312, comes before the count.
                bits.extend([false, false, false, true].into_iter().enumerate().map(|(i, bit)| (at + i, bit)));
                at += 4;
            }
            let values = values_of(mode, text, false, |text| text::encode_in(charset, text))?;
            let (packed, fixed) = pack(mode, at + count_bits, None, 0, &values);
            bits.extend(packed);
            (bits, values.len(), fixed)
        }
    };
    if fixed == 0 {
        return Err(TextPinError::NothingWhole);
    }
    if let Some(&(last, _)) = bits.iter().max_by_key(|(bit, _)| *bit)
        && last >= layout.data_bits
    {
        let within = bits.iter().filter(|(bit, _)| *bit < layout.data_bits).count();
        return Err(TextPinError::TooLong { needs: given, room: given * within / bits.len() });
    }
    bits.sort_unstable();
    let contradicted = bits.iter().filter(|&&(bit, value)| report.stream.certainty[bit] == Certainty::Verified && report.stream.values[bit] != value).count();
    let mut pins = Pins::new();
    for (bit, value) in bits {
        let Some((cw, place)) = layout.stream_bit_location(bit) else { continue };
        // A bit the symbol has no module for, as the fixed nibble of
        // Model 1's corner codeword is, is no bit to pin.
        let mask = (0x80u8 >> place) & codeword_bit_mask(layout, cw);
        let pin = pins.entry(cw).or_insert(Pin { mask: 0, value: 0, source: PinSource::Text });
        pin.mask |= mask;
        if value {
            pin.value |= mask;
        }
    }
    pins.retain(|_, pin| pin.mask != 0);
    Ok(TextPins { pins, pinned: fixed, left_out: given - fixed, contradicted })
}

/// Lay `more` over `pins`: every bit `more` asserts takes the place of
/// what `pins` had for it, and a codeword it touches counts as pinned
/// from where `more` came.
pub fn lay_over(pins: &mut Pins, more: &Pins) {
    for (&cw, new) in more {
        pins.entry(cw)
            .and_modify(|old| {
                old.value = (old.value & !new.mask) | (new.value & new.mask);
                old.mask |= new.mask;
                old.source = new.source;
            })
            .or_insert(*new);
    }
}

/// The ways messages commonly begin: addresses, the schemes phones act
/// on, contact and calendar cards, Wi-Fi labels, one-time-password keys
/// and the payment formats printed on bills and at tills.
pub const BEGINNINGS: [&str; 34] = [
    "https://www.",
    "https://",
    "http://www.",
    "http://",
    "HTTPS://WWW.",
    "HTTPS://",
    "HTTP://WWW.",
    "HTTP://",
    "www.",
    "WIFI:S:",
    "WIFI:T:WPA;S:",
    "WIFI:T:",
    "WIFI:",
    "mailto:",
    "MAILTO:",
    "MATMSG:TO:",
    "tel:+",
    "tel:",
    "TEL:",
    "sms:",
    "smsto:",
    "SMSTO:",
    "geo:",
    "BEGIN:VCARD\r\nVERSION:",
    "BEGIN:VCARD\nVERSION:",
    "BEGIN:VCARD",
    "BEGIN:VCALENDAR",
    "BEGIN:VEVENT",
    "MECARD:N:",
    "MECARD:",
    "otpauth://totp/",
    "otpauth://",
    "bitcoin:",
    "000201",
];

/// The share of a beginning's bits, among those the symbol shows, that
/// may read otherwise before the beginning is passed over untried: one
/// in three. A message that begins another way differs in about half of
/// them, and damage that left the modules readable has not turned a
/// third of them over; a module that reads as nothing is not compared.
const BEGINNING_MISREAD: (u32, u32) = (1, 3);

/// A beginning that error correction bore out.
#[derive(Clone, Debug)]
pub struct Beginning {
    pub text: &'static str,
    /// Where it was pinned: at the first character as read, or at the
    /// start of the message under a header of its own.
    pub place: Place,
    /// Every pin in force with it: those given, and its own over them.
    pub pins: Pins,
    /// The decode under those pins.
    pub report: DecodeReport,
}

/// Try each of [`BEGINNINGS`] at the start of the message, where it is
/// read to begin and as a byte or alphanumeric segment of its own, and
/// keep the one that does most: the most blocks verified, and of those
/// the longest. A beginning counts only when every block its pins fall
/// in then verifies, one of them a block that had not before, and the
/// message decoded under it begins with it; one that contradicts bits
/// already confirmed is not tried. `pins` are those in force and
/// `before` the decode under them. `None` when no beginning helps.
pub fn try_beginnings(
    grid: &ModuleGrid,
    layout: &SymbolLayout,
    mask: u8,
    pins: &Pins,
    opts: &DecodeOptions,
    before: &DecodeReport,
) -> Option<Beginning> {
    let mut best: Option<Beginning> = None;
    for text in BEGINNINGS {
        let mut places = vec![Place::Start(Mode::Byte)];
        if !before.text.sources.is_empty() {
            places.insert(0, Place::Char(0));
        }
        if text.chars().all(|ch| alphanumeric_value(ch).is_some()) {
            places.push(Place::Start(Mode::Alphanumeric));
        }
        for place in places {
            let Ok(found) = text_pins(layout, before, place, text, opts.charset) else { continue };
            if found.contradicted > 0 || found.left_out > 0 {
                continue;
            }
            // Against the modules as read, before any decode is spent.
            let (mut differing, mut compared) = (0, 0);
            for (&cw, pin) in &found.pins {
                let read = &before.codewords[cw];
                differing += ((read.read ^ pin.value) & pin.mask & read.known).count_ones();
                compared += (pin.mask & read.known).count_ones();
            }
            if differing * BEGINNING_MISREAD.1 > compared * BEGINNING_MISREAD.0 {
                continue;
            }
            let mut trial = pins.clone();
            lay_over(&mut trial, &found.pins);
            let report = decode(grid, layout, mask, &trial, opts);
            let blocks: BTreeSet<usize> = found.pins.keys().map(|&cw| layout.codewords[cw].block as usize).collect();
            let borne_out = blocks.iter().all(|&b| report.blocks[b].is_verified())
                && blocks.iter().any(|&b| !before.blocks[b].is_verified())
                && report.text.plain().starts_with(text);
            let does_more = |held: &Beginning| (report.blocks_verified(), text.len()) > (held.report.blocks_verified(), held.text.len());
            if borne_out && best.as_ref().is_none_or(does_more) {
                best = Some(Beginning { text, place, pins: trial, report });
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_are_written_whole_or_not_at_all() {
        // Digits 3 to 8 of a payload of eleven, at bit 100: the group of
        // digits 3 to 5 and that of 6 to 8, ten bits each.
        let (bits, fixed) = pack(Mode::Numeric, 100, Some(11), 3, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(fixed, 6);
        assert_eq!(bits.iter().map(|b| b.0).collect::<Vec<_>>(), (110..130).collect::<Vec<_>>());
        let value = |bits: &[(usize, bool)]| bits.iter().fold(0u32, |sum, &(_, bit)| sum << 1 | u32::from(bit));
        assert_eq!((value(&bits[..10]), value(&bits[10..])), (123, 456));
        // Digits 4 to 10: the first two are of a group whose first digit
        // was not given, and the two that end the payload are a group of
        // their own, of seven bits.
        let (bits, fixed) = pack(Mode::Numeric, 100, Some(11), 4, &[2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(fixed, 5);
        assert_eq!((bits[0].0, bits.len()), (120, 17));
        assert_eq!((value(&bits[..10]), value(&bits[10..])), (456, 78));
        // With the length unknown, a group cut short by the end of the
        // text is not known to be the last, and is left out.
        assert_eq!(pack(Mode::Numeric, 0, None, 0, &[1, 2, 3, 4]).1, 3);
        assert_eq!(pack(Mode::Numeric, 0, None, 0, &[1, 2]).1, 0);
        // Alphanumeric pairs, and the odd character that ends a payload.
        let (bits, fixed) = pack(Mode::Alphanumeric, 0, Some(3), 0, &[10, 11, 12]);
        assert_eq!((fixed, bits.len()), (3, 17));
        assert_eq!((value(&bits[..11]), value(&bits[11..])), (10 * 45 + 11, 12));
        assert_eq!(pack(Mode::Alphanumeric, 0, Some(4), 1, &[11, 12]).1, 0);
        // Bytes and kanji stand each on their own.
        assert_eq!(pack(Mode::Byte, 12, Some(5), 2, &[0x41, 0x42]).0.first(), Some(&(28, false)));
        assert_eq!(pack(Mode::Kanji, 0, None, 0, &[0x1FFF]).0.len(), 13);
    }

    #[test]
    fn characters_take_the_values_their_mode_gives_them() {
        let none = |_: &str| -> Result<Vec<u8>, char> { unreachable!("not byte mode") };
        assert_eq!(values_of(Mode::Numeric, "907", false, none), Ok(vec![9, 0, 7]));
        assert_eq!(values_of(Mode::Numeric, "9a", false, none), Err(TextPinError::NotInMode { ch: 'a', mode: Mode::Numeric }));
        // A digit of another script is no digit of numeric mode.
        assert!(values_of(Mode::Numeric, "٣", false, none).is_err());
        assert_eq!(values_of(Mode::Alphanumeric, "A:%", false, none), Ok(vec![10, 44, 38]));
        assert!(values_of(Mode::Alphanumeric, "a", false, none).is_err());
        // Under FNC1 the group separator is written as a percent sign,
        // and a percent sign as two.
        assert_eq!(values_of(Mode::Alphanumeric, "1\u{1D}%", true, none), Ok(vec![1, 38, 38, 38]));
        assert!(values_of(Mode::Alphanumeric, "\u{1D}", false, none).is_err());
        assert_eq!(values_of(Mode::Byte, "é", false, |text| Ok(text.bytes().collect())), Ok(vec![0xC3, 0xA9]));
        assert_eq!(values_of(Mode::Byte, "é", false, |_| Err('é')), Err(TextPinError::NotInCharset { ch: 'é' }));
        // 点 is 0x935F in Shift-JIS, and 3487 in kanji mode.
        assert_eq!(values_of(Mode::Kanji, "点", false, none), Ok(vec![3487]));
        assert!(values_of(Mode::Kanji, "a", false, none).is_err());
        assert!(values_of(Mode::Hanzi, "中", false, none).is_ok());
    }

    #[test]
    fn pins_laid_over_others_take_their_bits_and_leave_the_rest() {
        let mut pins = Pins::new();
        pins.insert(3, Pin { mask: 0xF0, value: 0xA0, source: PinSource::Padding });
        pins.insert(4, Pin::full(0x11, PinSource::Manual));
        let mut more = Pins::new();
        more.insert(3, Pin { mask: 0x3C, value: 0x14, source: PinSource::Text });
        more.insert(9, Pin::full(0x77, PinSource::Text));
        lay_over(&mut pins, &more);
        assert_eq!(pins[&3], Pin { mask: 0xFC, value: 0x94, source: PinSource::Text });
        assert_eq!(pins[&4], Pin::full(0x11, PinSource::Manual));
        assert_eq!(pins[&9], Pin::full(0x77, PinSource::Text));
    }
}
