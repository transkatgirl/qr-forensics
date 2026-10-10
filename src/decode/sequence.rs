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

use super::bitstream::{Certainty, SegmentContent};
use super::pipeline::DecodeReport;
use super::text::{TextReport, TextSpan, hex_of};

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
}

impl Part {
    /// The part of a decode, when its data carries a structured append
    /// header.
    pub fn from_report(report: &DecodeReport, label: impl Into<String>) -> Option<Self> {
        let header = report.parse.segments.iter().find_map(|s| match s.content {
            SegmentContent::StructuredAppend { index, total, parity } => Some((index, total, parity)),
            _ => None,
        })?;
        Some(Self { index: header.0, total: header.1, parity: header.2, label: label.into(), text: report.text.clone() })
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

    /// The parts' text in order, with a gap marked where a part is
    /// missing: one unknown span per missing part.
    pub fn assembled(&self) -> TextReport {
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
        out.parity = if self.missing().is_empty() {
            self.parts.iter().map(Part::own_parity).try_fold(0u8, |acc, p| p.map(|p| acc ^ p))
        } else {
            None
        };
        out
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
        }
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
}
