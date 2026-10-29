//! Structured append: a message split over several symbols, each carrying
//! its position in the sequence, the number of symbols, and a parity byte
//! that is the XOR of every data byte of the whole message.
//!
//! The parts are decoded one at a time, each from its own photo or grid,
//! and collected here. ZXing concatenates the parts it finds in one image
//! with no check at all, zxing-cpp checks that the count and the parity
//! byte agree between parts, and ZBar groups parts by count and parity
//! and leaves verifying the parity as a to-do. Here the parity is
//! verified once every part is in and every byte of every part is known,
//! which is what the parity is for: a part decoded wrongly, or belonging
//! to another message, shows up as a parity that does not match.

use super::bitstream::{Certainty, EndOfData, Segment, SegmentContent};
use super::pipeline::{DecodeReport, combined_risk_bits};
use super::text::{Charset, TextReport, TextSpan, decode_text, hex_of, under_fnc1};

/// One symbol's share of a message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Part {
    /// Position in the sequence, from 1.
    pub index: u8,
    /// Symbols in the sequence, as this part's header says.
    pub total: u8,
    /// Parity of the whole message, as this part's header says.
    pub parity: u8,
    /// Where the part came from, for the user: a photo's name, say.
    pub label: String,
    /// The part's own text, with its certainties.
    pub text: TextReport,
    /// How rarely the part's message is wrong, in bits
    /// ([`DecodeReport::message_risk_bits`]): None while a block of it
    /// is unsolved.
    pub risk_bits: Option<u32>,
    /// The part's segments as its stream parsed, for the message the
    /// parts make up to be read whole ([`Sequence::assembled`]), and the
    /// character set its own text was read in.
    pub segments: Vec<Segment>,
    pub charset: Charset,
    /// Whether the part's stream parsed to its end ([`EndOfData::At`]),
    /// and was not lost at a header it could not read: its segments are
    /// then all its data, and the next part's go on from the last of them.
    pub whole: bool,
}

impl Part {
    /// The part of a decode, when its data carries a structured append
    /// header, its byte data read in whatever set it shows
    /// ([`Charset::Auto`]).
    pub fn from_report(report: &DecodeReport, label: impl Into<String>) -> Option<Self> {
        Self::from_report_in(report, label, Charset::Auto)
    }

    /// [`Self::from_report`] of a decode whose byte data was read in
    /// `charset`.
    pub fn from_report_in(report: &DecodeReport, label: impl Into<String>, charset: Charset) -> Option<Self> {
        let header = report.parse.segments.iter().find_map(|s| match s.content {
            SegmentContent::StructuredAppend { index, total, parity } => Some((index, total, parity)),
            _ => None,
        })?;
        Some(Self {
            index: header.0,
            total: header.1,
            parity: header.2,
            label: label.into(),
            text: report.text.clone(),
            risk_bits: report.message_risk_bits(),
            segments: report.parse.segments.clone(),
            charset,
            whole: matches!(report.parse.end, EndOfData::At { .. }),
        })
    }

    /// XOR of the part's own data bytes; `None` while any is unreadable.
    pub fn own_parity(&self) -> Option<u8> {
        self.text.parity
    }

    /// Every character of the part is confirmed.
    pub fn is_verified(&self) -> bool {
        self.text.spans.iter().all(|s| s.certainty == Certainty::Verified)
    }
}

/// How the parts' parity bytes bear out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParityCheck {
    /// No parts yet.
    Empty,
    /// The parts disagree about how many there are or what the parity is:
    /// they are not all from one message.
    Inconsistent,
    /// Some parts are still missing.
    Incomplete { have: usize, total: u8 },
    /// Every part is in but some bytes are unreadable, so the parity
    /// cannot be computed.
    Unreadable,
    /// Every part is in and every byte known: the parity computed over
    /// the whole message against the one the headers claim.
    Complete { computed: u8, claimed: u8 },
}

impl ParityCheck {
    pub fn is_ok(&self) -> bool {
        matches!(self, ParityCheck::Complete { computed, claimed } if computed == claimed)
    }
}

/// The parts of one message collected so far.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Sequence {
    parts: Vec<Part>,
}

impl Sequence {
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    pub fn len(&self) -> usize {
        self.parts.len()
    }

    /// Parts in order of their index.
    pub fn parts(&self) -> &[Part] {
        &self.parts
    }

    /// Add a part, replacing any earlier part with the same index.
    /// Returns whether one was replaced.
    pub fn add(&mut self, part: Part) -> bool {
        let replaced = match self.parts.iter().position(|p| p.index == part.index) {
            Some(i) => {
                self.parts.remove(i);
                true
            }
            None => false,
        };
        self.parts.push(part);
        self.parts.sort_by_key(|p| p.index);
        replaced
    }

    /// Add the parts of one message among `parts`: those whose count and
    /// parity are this sequence's header ([`Self::header`]) while it
    /// holds a message, and otherwise those of the message most of them
    /// are of, the first met where two are as many. The rest come back
    /// in their order, for the caller to say so. A part of another
    /// message is no part of this one, whatever its index: the parts of
    /// two messages read from one photo once went into one sequence
    /// together, a part of one taking the place of the other's by its
    /// index.
    pub fn add_all(&mut self, parts: Vec<Part>) -> Vec<Part> {
        let header = self.header().or_else(|| {
            let mut counts: Vec<((u8, u8), usize)> = Vec::new();
            for part in &parts {
                let key = (part.total, part.parity);
                match counts.iter_mut().find(|(k, _)| *k == key) {
                    Some((_, n)) => *n += 1,
                    None => counts.push((key, 1)),
                }
            }
            // The last of the most from the back, which is the first.
            counts.iter().rev().max_by_key(|(_, n)| *n).map(|(key, _)| *key)
        });
        let mut left_out = Vec::new();
        for part in parts {
            if header == Some((part.total, part.parity)) {
                self.add(part);
            } else {
                left_out.push(part);
            }
        }
        left_out
    }

    pub fn remove(&mut self, index: u8) {
        self.parts.retain(|p| p.index != index);
    }

    pub fn clear(&mut self) {
        self.parts.clear();
    }

    /// What the parts agree the sequence is: (total, parity), when they
    /// agree.
    pub fn header(&self) -> Option<(u8, u8)> {
        let first = self.parts.first()?;
        self.parts.iter().all(|p| p.total == first.total && p.parity == first.parity).then_some((first.total, first.parity))
    }

    /// The indices not yet collected, 1 to the total.
    pub fn missing(&self) -> Vec<u8> {
        match self.header() {
            Some((total, _)) => (1..=total).filter(|i| !self.parts.iter().any(|p| p.index == *i)).collect(),
            None => Vec::new(),
        }
    }

    pub fn check(&self) -> ParityCheck {
        if self.parts.is_empty() {
            return ParityCheck::Empty;
        }
        let Some((total, claimed)) = self.header() else { return ParityCheck::Inconsistent };
        if self.parts.iter().any(|p| p.index > total) {
            return ParityCheck::Inconsistent;
        }
        if !self.missing().is_empty() {
            return ParityCheck::Incomplete { have: self.parts.len(), total };
        }
        let mut computed = 0u8;
        for part in &self.parts {
            match part.own_parity() {
                Some(p) => computed ^= p,
                None => return ParityCheck::Unreadable,
            }
        }
        ParityCheck::Complete { computed, claimed }
    }

    /// How rarely the message the parts make up is wrong, in bits, once
    /// every part is in: the parts' chances added, since any one of them
    /// wrong makes it wrong ([`combined_risk_bits`]), and nought where
    /// the parity says one is. A parity that checks is not counted as
    /// confirming anything besides: it is the XOR of the data bytes
    /// alone, and a part wrong in its other data, or in two bytes that
    /// cancel, passes it. None while a part is missing or has a block
    /// unsolved, or the parts disagree about the sequence.
    pub fn risk_bits(&self) -> Option<u32> {
        match self.check() {
            ParityCheck::Complete { computed, claimed } if computed != claimed => Some(0),
            ParityCheck::Complete { .. } | ParityCheck::Unreadable => combined_risk_bits(self.parts.iter().map(|p| p.risk_bits)),
            ParityCheck::Empty | ParityCheck::Inconsistent | ParityCheck::Incomplete { .. } => None,
        }
    }

    /// The parts' text in order, with a gap marked where a part is
    /// missing: one unknown span per missing part. Once every part is in,
    /// the message is read whole, from the segments of every part in
    /// turn ([`Self::read_whole`]), in the set the first part was read in.
    pub fn assembled(&self) -> TextReport {
        self.assembled_in(self.parts.first().map_or(Charset::Auto, |part| part.charset))
    }

    /// [`Self::assembled`], the message read whole in `charset`, the set in
    /// force, whatever set each part's own text was read in. The parts
    /// read in different sets were once joined as they read, every part
    /// in, and a character split between two of them, or a set the first
    /// declared, was lost to a change of set made between them.
    pub fn assembled_in(&self, charset: Charset) -> TextReport {
        if let Some(whole) = self.read_whole(charset) {
            return whole;
        }
        let mut out = TextReport::default();
        let total = self.header().map(|h| h.0).unwrap_or_else(|| self.parts.iter().map(|p| p.index).max().unwrap_or(0));
        let mut encodings: Vec<String> = Vec::new();
        for index in 1..=total {
            match self.parts.iter().find(|p| p.index == index) {
                Some(part) => {
                    for span in &part.text.spans {
                        append_span(&mut out, span);
                    }
                    out.bytes.extend_from_slice(&part.text.bytes);
                    if let Some(encoding) = &part.text.encoding
                        && !encodings.contains(encoding)
                    {
                        encodings.push(encoding.clone());
                    }
                    out.eci_text.push_str(&part.text.eci_text);
                }
                None => {
                    append_span(&mut out, &TextSpan { text: format!("[part {index} missing]"), certainty: Certainty::Unknown });
                    out.eci_text.push_str(&format!("[part {index} missing]"));
                }
            }
        }
        if !encodings.is_empty() {
            out.encoding = Some(encodings.join(", "));
        }
        out.symbology = self.parts.first().and_then(|p| p.text.symbology.clone());
        // The parity of the whole message is known only once every part
        // of one message is in and read: parts that disagree about the
        // sequence have no missing indices to name, and no parity either.
        out.parity = match self.check() {
            ParityCheck::Complete { computed, .. } => Some(computed),
            _ => None,
        };
        out
    }

    /// The message every part of the sequence makes up, read as one
    /// stream: the segments of each part in turn, their structured append
    /// headers left out, decoded together in `charset`, as zxing-cpp
    /// joins the parts' content before it decodes it. `None` while a part
    /// is missing, the parts disagree about the sequence, or a part was
    /// made with no segments to read or lost its stream short of its end
    /// ([`Part::whole`]): the data between is unknown, and its own text
    /// marks where, which the stream read whole would not, the bytes on
    /// either side of the gap read as one run, a character made of them.
    /// `None` too where some parts are under FNC1 and some are not: a
    /// percent sign in a part's alphanumeric data stands for the group
    /// separator by its own symbol's mode, as zxing-cpp reads each symbol
    /// before it joins them, and read whole, one part's FNC1 put every
    /// other part's percent signs under it, and the text disagreed with
    /// the parity the parts' own texts gave.
    ///
    /// Each part's text was once decoded alone and the texts joined: a
    /// character of UTF-8 whose bytes the encoder split between two
    /// symbols came out as two characters of nothing, the second part,
    /// beginning with a continuation byte, was taken for Latin-1 and read
    /// so, confirmed, and an ECI designator in the first part was not
    /// followed in the second. The ECI protocol keeps a designator in
    /// force to the end of the data, and the data of a sequence is all
    /// its parts'.
    fn read_whole(&self, charset: Charset) -> Option<TextReport> {
        let (total, _) = self.header()?;
        // A part numbered past the total disagrees about the sequence as
        // much as one that names another total ([`Self::check`]): it was
        // once left out, and the rest read as the whole message.
        if !self.missing().is_empty() || self.parts.iter().any(|p| p.index > total) {
            return None;
        }
        if self.parts.iter().any(|p| under_fnc1(&p.segments)) && !self.parts.iter().all(|p| under_fnc1(&p.segments)) {
            return None;
        }
        let mut segments: Vec<Segment> = Vec::new();
        for index in 1..=total {
            let part = self.parts.iter().find(|p| p.index == index)?;
            if part.segments.is_empty() || !part.whole {
                return None;
            }
            let mut own = part.segments.iter().filter(|s| !matches!(s.content, SegmentContent::StructuredAppend { .. })).cloned();
            // The bytes a part begins with go on from those the part
            // before ended with, nothing between them: one run of the
            // message's bytes, which the set is told from and read in
            // whole, a character split between the two among them.
            if let Some(first) = own.next() {
                match (segments.last_mut(), &first.content) {
                    (Some(last), SegmentContent::Bytes(more)) if matches!(last.content, SegmentContent::Bytes(_)) && !last.truncated => {
                        if let SegmentContent::Bytes(cells) = &mut last.content {
                            cells.extend(more.iter().cloned());
                        }
                        last.count = last.count.zip(first.count).map(|(a, b)| a + b);
                        last.damaged |= first.damaged;
                        last.truncated = first.truncated;
                        last.header_certainty = last.header_certainty.min(first.header_certainty);
                    }
                    _ => segments.push(first),
                }
            }
            segments.extend(own);
        }
        let mut out = decode_text(&segments, charset);
        // A character's place is in its own symbol's stream, which the
        // message read whole has no one place for.
        out.sources.clear();
        out.symbology = self.parts.first().and_then(|p| p.text.symbology.clone());
        out.parity = match self.check() {
            ParityCheck::Complete { computed, .. } => Some(computed),
            _ => None,
        };
        Some(out)
    }

    /// The assembled bytes in hexadecimal.
    pub fn hex(&self) -> String {
        hex_of(&self.assembled().bytes)
    }
}

fn append_span(report: &mut TextReport, span: &TextSpan) {
    match report.spans.last_mut() {
        Some(last) if last.certainty == span.certainty => last.text.push_str(&span.text),
        _ => report.spans.push(span.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(index: u8, total: u8, parity: u8, text: &str, certainty: Certainty) -> Part {
        let bytes: Vec<Option<u8>> = text.bytes().map(Some).collect();
        let own = text.bytes().fold(0u8, |a, b| a ^ b);
        Part {
            index,
            total,
            parity,
            label: format!("photo {index}"),
            text: TextReport {
                spans: vec![TextSpan { text: text.to_owned(), certainty }],
                parity: Some(own),
                bytes,
                ..TextReport::default()
            },
            risk_bits: (certainty == Certainty::Verified).then_some(8),
            segments: Vec::new(),
            charset: Charset::Auto,
            whole: true,
        }
    }

    /// A part as a decode makes it, of byte-mode data, with its
    /// segments: its structured append header, then the bytes.
    fn byte_part(index: u8, total: u8, parity: u8, data: &[u8]) -> Part {
        use super::super::bitstream::Cell;
        use crate::symbol::tables::Mode;
        let n = data.len();
        let header = Segment {
            mode: Mode::StructuredAppend,
            header: 0..4,
            payload: 4..20,
            count: None,
            header_certainty: Certainty::Verified,
            content: SegmentContent::StructuredAppend { index, total, parity },
            damaged: false,
            truncated: false,
            header_inferred: false,
        };
        let cells = data.iter().enumerate().map(|(i, &b)| Cell { value: Some(b), certainty: Certainty::Verified, bits: 32 + 8 * i..40 + 8 * i }).collect();
        let bytes = Segment {
            mode: Mode::Byte,
            header: 20..32,
            payload: 32..32 + 8 * n,
            count: Some(n),
            header_certainty: Certainty::Verified,
            content: SegmentContent::Bytes(cells),
            damaged: false,
            truncated: false,
            header_inferred: false,
        };
        let segments = vec![header, bytes];
        Part {
            index,
            total,
            parity,
            label: format!("photo {index}"),
            text: decode_text(&segments, Charset::Auto),
            risk_bits: Some(8),
            segments,
            charset: Charset::Auto,
            whole: true,
        }
    }

    #[test]
    fn a_character_split_between_two_symbols_is_read_whole() {
        // "naïve café" in UTF-8, the two bytes of the ï in different
        // symbols: read alone, neither part holds the character, and the
        // second, beginning with a continuation byte, is not UTF-8 at all.
        let message = "naïve café".as_bytes();
        let split = 3;
        assert_eq!(message[2], 0xC3);
        let parity = message.iter().fold(0u8, |a, &b| a ^ b);
        let (first, second) = (byte_part(1, 2, parity, &message[..split]), byte_part(2, 2, parity, &message[split..]));
        assert_ne!(format!("{}{}", first.text.plain(), second.text.plain()), "naïve café");
        let mut sequence = Sequence::default();
        sequence.add(second);
        // With a part missing, the parts are joined as they read, with
        // the gap marked.
        assert!(sequence.assembled().plain().contains("[part 1 missing]"));
        sequence.add(first);
        assert_eq!(sequence.check(), ParityCheck::Complete { computed: parity, claimed: parity });
        let whole = sequence.assembled();
        assert_eq!(whole.plain(), "naïve café");
        assert_eq!(whole.encoding.as_deref(), Some("UTF-8 (guessed)"));
        assert_eq!(whole.parity, Some(parity));
        assert!(whole.sources.is_empty());
        assert_eq!(whole.bytes, message.iter().copied().map(Some).collect::<Vec<_>>());
        // The parts' own texts read in different sets: the message is read
        // whole all the same, in the set asked for.
        let mut mixed = Sequence::default();
        mixed.add(byte_part(1, 2, parity, &message[..split]));
        mixed.add(Part { charset: Charset::Utf8, ..byte_part(2, 2, parity, &message[split..]) });
        assert_eq!(mixed.assembled().plain(), "naïve café");
        assert_eq!(mixed.assembled_in(Charset::Utf8).plain(), "naïve café");
        // A part whose stream was lost short of its end holds an unknown
        // stretch of the message after its last segment: its bytes and the
        // next part's are not read as one run, which made the lead byte it
        // ended on and the continuation the next began with one character.
        let mut lost = Sequence::default();
        lost.add(Part { whole: false, ..byte_part(1, 2, parity, &message[..split]) });
        lost.add(byte_part(2, 2, parity, &message[split..]));
        assert_ne!(lost.assembled().plain(), "naïve café");
        // A part numbered past the total the others name disagrees about
        // the sequence: the rest are not read as the whole message, which
        // left it out.
        let mut past = Sequence::default();
        past.add(byte_part(1, 2, parity, &message[..split]));
        past.add(byte_part(2, 2, parity, &message[split..]));
        past.add(byte_part(3, 2, parity, b"!"));
        assert_eq!(past.check(), ParityCheck::Inconsistent);
        assert_ne!(past.assembled().plain(), "naïve café");
        assert_eq!(past.assembled().parity, None);
    }

    #[test]
    fn a_percent_sign_stands_for_a_separator_by_its_own_symbols_fnc1() {
        use super::super::bitstream::Cell;
        use crate::symbol::tables::Mode;
        // Alphanumeric data under FNC1 in first position where `fnc1`,
        // after a structured append header.
        let alphanumeric = |index: u8, fnc1: bool, text: &str| -> Part {
            let segment = |mode, content| Segment {
                mode,
                header: 0..0,
                payload: 0..0,
                count: None,
                header_certainty: Certainty::Verified,
                content,
                damaged: false,
                truncated: false,
                header_inferred: false,
            };
            let cells = text.chars().map(|c| Cell { value: Some(c), certainty: Certainty::Verified, bits: 0..0 }).collect();
            let mut segments = vec![segment(Mode::StructuredAppend, SegmentContent::StructuredAppend { index, total: 2, parity: 0 })];
            if fnc1 {
                segments.push(segment(Mode::Fnc1First, SegmentContent::Fnc1First));
            }
            segments.push(segment(Mode::Alphanumeric, SegmentContent::Chars(cells)));
            Part {
                index,
                total: 2,
                parity: 0,
                label: format!("photo {index}"),
                text: decode_text(&segments, Charset::Auto),
                risk_bits: Some(8),
                segments,
                charset: Charset::Auto,
                whole: true,
            }
        };
        // Both under FNC1: read whole, each percent sign a separator.
        let mut both = Sequence::default();
        both.add(alphanumeric(1, true, "A%B"));
        both.add(alphanumeric(2, true, "C%D"));
        assert_eq!(both.assembled().plain(), "A\u{1d}BC\u{1d}D");
        // The first alone: its own percent sign is a separator and the
        // second's a percent sign, as each symbol reads. Read whole, the
        // first part's FNC1 once made the second's a separator too.
        let mut first = Sequence::default();
        first.add(alphanumeric(1, true, "A%B"));
        first.add(alphanumeric(2, false, "C%D"));
        assert_eq!(first.assembled().plain(), "A\u{1d}BC%D");
    }

    #[test]
    fn parts_assemble_in_order_and_parity_is_verified() {
        let message = b"hello, world";
        let parity = message.iter().fold(0u8, |a, &b| a ^ b);
        let mut sequence = Sequence::default();
        assert_eq!(sequence.check(), ParityCheck::Empty);
        sequence.add(part(3, 3, parity, "orld", Certainty::Verified));
        assert_eq!(sequence.check(), ParityCheck::Incomplete { have: 1, total: 3 });
        assert_eq!(sequence.missing(), [1, 2]);
        assert_eq!(sequence.assembled().plain(), "[part 1 missing][part 2 missing]orld");
        sequence.add(part(1, 3, parity, "hello", Certainty::Verified));
        sequence.add(part(2, 3, parity, ", w", Certainty::Unverified));
        assert!(sequence.missing().is_empty());
        let check = sequence.check();
        assert_eq!(check, ParityCheck::Complete { computed: parity, claimed: parity });
        assert!(check.is_ok());
        let assembled = sequence.assembled();
        assert_eq!(assembled.plain(), "hello, world");
        assert_eq!(assembled.spans.len(), 3, "the middle part is unverified");
        assert_eq!(assembled.parity, Some(parity));
        assert_eq!(sequence.hex(), "68 65 6C 6C 6F 2C 20 77 6F 72 6C 64");
        // A wrongly decoded part shows as a parity that does not match.
        assert!(sequence.add(part(2, 3, parity, ", W", Certainty::Verified)), "replaced");
        assert_eq!(sequence.len(), 3);
        assert!(!sequence.check().is_ok());
        assert!(matches!(sequence.check(), ParityCheck::Complete { .. }));
        sequence.remove(2);
        assert_eq!(sequence.missing(), [2]);
    }

    #[test]
    fn inconsistent_and_unreadable_parts_are_told_apart() {
        let mut sequence = Sequence::default();
        sequence.add(part(1, 2, 0x11, "ab", Certainty::Verified));
        sequence.add(part(2, 3, 0x11, "cd", Certainty::Verified));
        assert_eq!(sequence.check(), ParityCheck::Inconsistent);
        assert_eq!(sequence.header(), None);
        // Parts that disagree about the sequence name no missing index,
        // and that is no reason to report the parity of a whole message:
        // there is no one message whose parity it would be.
        assert!(sequence.missing().is_empty());
        assert_eq!(sequence.assembled().parity, None);
        assert_eq!(sequence.assembled().plain(), "abcd");
        sequence.clear();
        // The parts of two messages read from one photo, in the order
        // the symbols were found: the message with more parts goes in,
        // whole, and the other's parts come back. They once went in
        // together, part 2 of the second taking the place of part 2 of
        // the first.
        let a = |index, text: &str| part(index, 2, 0x11, text, Certainty::Verified);
        let b = |index, text: &str| part(index, 2, 0x22, text, Certainty::Verified);
        let left_out = sequence.add_all(vec![a(1, "ab"), b(2, "zz"), a(2, "cd")]);
        assert_eq!(sequence.assembled().plain(), "abcd");
        assert_eq!(left_out.iter().map(|p| (p.index, p.parity)).collect::<Vec<_>>(), [(2, 0x22)]);
        // Holding a message, the sequence takes that message's parts and
        // no other's, however many.
        sequence.clear();
        sequence.add(b(1, "yy"));
        let left_out = sequence.add_all(vec![a(1, "ab"), a(2, "cd"), b(2, "zz")]);
        assert_eq!((sequence.assembled().plain().as_str(), left_out.len()), ("yyzz", 2));
        // As many of each: the first met. Nothing: nothing.
        sequence.clear();
        let left_out = sequence.add_all(vec![b(1, "yy"), a(1, "ab")]);
        assert_eq!((sequence.header(), left_out.len()), (Some((2, 0x22)), 1));
        assert!(Sequence::default().add_all(Vec::new()).is_empty());
        sequence.clear();
        let mut torn = part(2, 2, 0x11, "c?", Certainty::Unknown);
        torn.text.parity = None;
        torn.text.bytes = vec![Some(b'c'), None];
        sequence.add(part(1, 2, 0x11, "ab", Certainty::Verified));
        sequence.add(torn);
        assert_eq!(sequence.check(), ParityCheck::Unreadable);
        assert_eq!(sequence.hex(), "61 62 63 ??");
        assert_eq!(sequence.assembled().parity, None);
    }
    #[test]
    fn the_message_of_the_parts_is_wrong_as_often_as_its_parts_together() {
        let message = b"hello, world";
        let parity = message.iter().fold(0u8, |a, &b| a ^ b);
        let mut sequence = Sequence::default();
        sequence.add(part(1, 3, parity, "hell", Certainty::Verified));
        // Parts still missing say nothing of the whole.
        assert_eq!(sequence.risk_bits(), None);
        sequence.add(part(2, 3, parity, "o, w", Certainty::Verified));
        sequence.add(part(3, 3, parity, "orld", Certainty::Verified));
        // Three parts each wrong once in 2^8: the message once in 2^6.
        assert_eq!(sequence.risk_bits(), Some(6));
        // One part with a block unsolved leaves the whole unsaid.
        sequence.add(part(2, 3, parity, "o, w", Certainty::Unverified));
        assert_eq!(sequence.risk_bits(), None);
        // A parity that does not check says one part is wrong.
        sequence.add(part(2, 3, parity, "o, x", Certainty::Verified));
        assert_eq!(sequence.risk_bits(), Some(0));
    }
}
