//! What the message makes likely of its own bits: soft evidence for the
//! blocks error correction has not settled, from how the message's
//! segments pack their characters and what the characters read so far
//! hold.
//!
//! Whoever repairs a symbol by hand seldom treats its bits as a coin's
//! toss: an address is seven-bit text, so the high bit of every byte of
//! it is nought, and digits packed three to ten bits never pass 999.
//! Asserted as pins, such knowledge costs no error correction but is
//! only as good as the assertion, and a message that is ASCII through
//! its address may carry a name with an accent after it. So it is
//! weighed here instead, as a module's reading is: each group a segment
//! packs its characters in, a byte or ten bits of digits or eleven of
//! alphanumerics or thirteen of Kanji, leans each of its bits toward the
//! value the message's other groups of its kind hold at that place, as
//! many as could be read, and, before any could, toward what such a
//! group holds by nature: a byte toward seven-bit text, one bit in
//! sixteen above 127, and a group of digits or alphanumerics toward the
//! values it can take, a lone digit's four bits never above nine. The
//! segments are taken as the stream parses as read, the headers of
//! blocks not yet solved among them.
//!
//! The solvers search a block whose bits the lean has moved
//! ([`MessageEvidence::laid`]): a bit read in the colour it leans toward
//! is believed the more, one read against it the less, and one that
//! could not be read is believed at the lean, so that it is an equation
//! to the solver that works bit by bit where it was an unknown. What
//! they find is judged as any solution is ([`super::rs::judged`]),
//! by how rarely a block of noise lies so near a codeword, against the
//! block as read with only its unread bits taken at their lean. A bit
//! that was read is judged by its reading alone, so the lean never makes
//! a reading count for more or for less than it did. A bit that was not
//! read is an assumption fixed before any solution was looked for, as a
//! pin is, but held only as strongly as the lean, so that a solution
//! that overturns it pays for that as for a reading so believed: a block
//! of noise is no likelier to lie near a codeword for assumptions it had
//! no hand in, and a solution that bears them out has that much more
//! behind it. So that a block has no hand in them, the leans of a
//! block's bits are learnt from the groups of the other blocks alone,
//! besides what a group is by nature, and a block's own reading says
//! nothing of what it is judged against.

use super::abp::{log_odds, reliability_at};
use super::bitstream::{Certainty, ParseReport, StreamBits};
use super::codewords::codeword_bit_mask;
use super::rs::CwRead;
use crate::symbol::grid::CERTAIN;
use crate::symbol::layout::{CwId, SymbolLayout};
use crate::symbol::tables::Mode;

/// Most a bit is believed for the message's evidence alone, on the
/// sampler's scale: seven bits of likelihood, one such bit in a hundred
/// and twenty-eight leaning the wrong way. A message of a few hundred
/// ASCII bytes would otherwise lean its high bits as hard as a module
/// read well inside its colour, and the search would never doubt one.
pub const EVIDENCE_MAX: u8 = 48;

/// Least a lean must come to, in bits of likelihood, to be weighed: the
/// odds two to one. Short of that a place in a group holds either value
/// about as often, and the lean would only stir the order the solvers
/// doubt bits in.
const EVIDENCE_MIN_BITS: f32 = 1.0;

/// How strongly a group's nature counts against what the message's own
/// groups show, in groups: a byte's high bit as sixteen bytes, one in
/// them above 127, and a place in a group of digits or alphanumerics as
/// eight; any other place in a byte, and every place in a group of
/// Kanji or Hanzi, as two groups split evenly, which leans no way until
/// the message's own groups do.
const BYTE_HIGH_WEIGHT: f64 = 16.0;
const BYTE_HIGH_SET: f64 = 1.0 / 16.0;
const RANGE_WEIGHT: f64 = 8.0;
const EVEN_WEIGHT: f64 = 2.0;

/// The lean of each bit of each codeword, as log-odds that it is
/// nought: positive toward nought, negative toward one, and nought
/// where the message says nothing. Bit `1 << b` of codeword `cw` is
/// `lean[cw][b]`.
#[derive(Clone, Debug, PartialEq)]
pub struct MessageEvidence {
    lean: Vec<[f32; 8]>,
}

/// A block with the evidence laid over it ([`MessageEvidence::laid`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Laid {
    /// The block for the solvers to search, every bit the evidence leans
    /// on moved by it, with its bits' reliability.
    pub searched: Vec<CwRead>,
    pub searched_bits: Vec<[u8; 8]>,
    /// The block to judge what they find against: as read, but for the
    /// bits that could not be read, taken at their lean.
    pub judged: Vec<CwRead>,
    pub judged_bits: Vec<[u8; 8]>,
    /// Bits the evidence moved for the search.
    pub moved: usize,
}

/// A group of `width` bits from stream bit `start`, of a segment of
/// `mode`.
#[derive(Clone, Copy)]
struct Group {
    mode: Mode,
    width: usize,
    start: usize,
}

/// What the groups of one kind that could be read hold at each place:
/// ones and groups, in all and of the groups touching each block.
struct Tally {
    kind: (Mode, usize),
    all: Vec<(f64, f64)>,
    touching: Vec<Vec<(f64, f64)>>,
}

impl MessageEvidence {
    /// The evidence the stream as parsed gives of the bits of `layout`'s
    /// codewords: what each group of each data segment leans toward, a
    /// bit of block `b` learning from the groups that touch no bit of
    /// `b`. None when it leans nowhere: no data segment parsed, or none
    /// of its groups leaning past [`EVIDENCE_MIN_BITS`].
    pub fn of(layout: &SymbolLayout, stream: &StreamBits, parse: &ParseReport) -> Option<Self> {
        let groups: Vec<Group> = parse.segments.iter().flat_map(|segment| groups_of(segment.mode, segment.count, segment.payload.clone())).collect();
        let block_of = |bit: usize| layout.stream_bit_location(bit).map(|(cw, _)| layout.codewords[cw].block as usize);
        let read_whole = |group: &Group| (group.start..group.start + group.width).all(|bit| stream.certainty[bit] != Certainty::Unknown);
        let mut tallies: Vec<Tally> = Vec::new();
        for group in groups.iter().filter(|g| read_whole(g)) {
            let kind = (group.mode, group.width);
            let at = match tallies.iter().position(|t| t.kind == kind) {
                Some(at) => at,
                None => {
                    tallies.push(Tally { kind, all: vec![(0.0, 0.0); group.width], touching: vec![vec![(0.0, 0.0); group.width]; layout.blocks.len()] });
                    tallies.len() - 1
                }
            };
            let tally = &mut tallies[at];
            let mut touched: Vec<usize> = (group.start..group.start + group.width).filter_map(block_of).collect();
            touched.dedup();
            for place in 0..group.width {
                let one = f64::from(u8::from(stream.values[group.start + place]));
                tally.all[place].0 += one;
                tally.all[place].1 += 1.0;
                for &b in &touched {
                    tally.touching[b][place].0 += one;
                    tally.touching[b][place].1 += 1.0;
                }
            }
        }
        let mut lean = vec![[0f32; 8]; layout.codewords.len()];
        let ceiling = log_odds(EVIDENCE_MAX);
        let mut any = false;
        for group in &groups {
            let tally = tallies.iter().find(|t| t.kind == (group.mode, group.width));
            for place in 0..group.width {
                let bit = group.start + place;
                // A bit error correction confirmed, or a pin asserted, is
                // settled; the solvers never doubt it.
                if stream.certainty[bit] == Certainty::Verified {
                    continue;
                }
                let Some((cw, at)) = layout.stream_bit_location(bit) else { continue };
                let mask = 0x80u8 >> at;
                if codeword_bit_mask(layout, cw) & mask == 0 {
                    continue;
                }
                let block = layout.codewords[cw].block as usize;
                let (ones, seen) = tally.map_or((0.0, 0.0), |t| (t.all[place].0 - t.touching[block][place].0, t.all[place].1 - t.touching[block][place].1));
                let (set, weight) = nature(group.mode, group.width, place);
                let p = (ones + set * weight) / (seen + weight);
                let toward_nought = ((1.0 - p) / p).ln() as f32;
                if toward_nought.abs() < EVIDENCE_MIN_BITS * std::f32::consts::LN_2 {
                    continue;
                }
                lean[cw][mask.trailing_zeros() as usize] = toward_nought.clamp(-ceiling, ceiling);
                any = true;
            }
        }
        any.then_some(Self { lean })
    }

    /// The block of `codewords`, read as `block` with `bit_reliability`,
    /// with the evidence laid over it: to search, each bit the message
    /// leans on believed as its reading and the lean together would have
    /// it, in whichever colour they favour, and a bit that could not be
    /// read believed at the lean alone; to judge against, the block as
    /// read with its unread bits at their lean. Pinned bits, and bits read
    /// at [`CERTAIN`], are as they were, and so is every bit read unless
    /// `move_reads`, when the decode doubts by reliability. None when the
    /// evidence moves no bit of the block.
    pub fn laid(&self, codewords: &[CwId], block: &[CwRead], bit_reliability: &[[u8; 8]], move_reads: bool) -> Option<Laid> {
        let mut laid = Laid {
            searched: block.to_vec(),
            searched_bits: bit_reliability.to_vec(),
            judged: block.to_vec(),
            judged_bits: bit_reliability.to_vec(),
            moved: 0,
        };
        for (i, &cw) in codewords.iter().enumerate() {
            for (b, &reliability) in bit_reliability[i].iter().enumerate() {
                let lean = self.lean[cw][b];
                let read = block[i];
                let known = read.known >> b & 1 != 0;
                // A reading painted as certain is never moved; a module
                // that could not be read has no reading to be certain of,
                // whatever the grid holds for it.
                if lean == 0.0 || read.pinned >> b & 1 != 0 || (known && (!move_reads || reliability == CERTAIN)) {
                    continue;
                }
                let set = read.value >> b & 1 != 0;
                let reading = if !known {
                    0.0
                } else if set {
                    -log_odds(reliability)
                } else {
                    log_odds(reliability)
                };
                let belief = reading + lean;
                let value = if belief == 0.0 { set } else { belief < 0.0 };
                let searched = &mut laid.searched[i];
                searched.known |= 1 << b;
                searched.value = searched.value & !(1 << b) | u8::from(value) << b;
                laid.searched_bits[i][b] = reliability_at(belief).min(CERTAIN - 1);
                if !known {
                    let judged = &mut laid.judged[i];
                    judged.known |= 1 << b;
                    judged.value = judged.value & !(1 << b) | u8::from(lean < 0.0) << b;
                    laid.judged_bits[i][b] = reliability_at(lean).min(CERTAIN - 1);
                }
                laid.moved += 1;
            }
        }
        (laid.moved > 0).then_some(laid)
    }

    /// The lean of bit `1 << bit` of codeword `cw`, as log-odds that it
    /// is nought: nought where the message says nothing of it.
    pub fn lean(&self, cw: CwId, bit: u8) -> f32 {
        self.lean[cw][usize::from(bit)]
    }

    /// The bits the message leans on.
    pub fn bits(&self) -> usize {
        self.lean.iter().flatten().filter(|&&lean| lean != 0.0).count()
    }
}

/// The groups a segment of `mode` holding `count` characters packs them
/// in, over its payload, as far as the payload runs: a truncated
/// segment's last groups are cut off where the symbol ends.
fn groups_of(mode: Mode, count: Option<usize>, payload: std::ops::Range<usize>) -> Vec<Group> {
    let Some(count) = count else { return Vec::new() };
    let widths: Vec<usize> = match mode {
        Mode::Numeric => std::iter::repeat_n(10, count / 3).chain([None, Some(4), Some(7)][count % 3]).collect(),
        Mode::Alphanumeric => std::iter::repeat_n(11, count / 2).chain((count % 2 == 1).then_some(6)).collect(),
        Mode::Byte => vec![8; count],
        Mode::Kanji | Mode::Hanzi => vec![13; count],
        _ => Vec::new(),
    };
    let mut start = payload.start;
    let mut groups = Vec::with_capacity(widths.len());
    for width in widths {
        if start + width > payload.end {
            break;
        }
        groups.push(Group { mode, width, start });
        start += width;
    }
    groups
}

/// What a group of `width` bits of `mode` holds at `place`, counted
/// from its most significant bit, by its nature: the share of its values
/// with that bit set, and as how many groups that counts against those
/// the message shows.
fn nature(mode: Mode, width: usize, place: usize) -> (f64, f64) {
    let range = match (mode, width) {
        (Mode::Byte, _) if place == 0 => return (BYTE_HIGH_SET, BYTE_HIGH_WEIGHT),
        (Mode::Numeric, 10) => 1000,
        (Mode::Numeric, 7) => 100,
        (Mode::Numeric, 4) => 10,
        (Mode::Alphanumeric, 11) => 45 * 45,
        (Mode::Alphanumeric, 6) => 45,
        _ => return (0.5, EVEN_WEIGHT),
    };
    let bit = 1u32 << (width - 1 - place);
    let set = (0..range).filter(|v| v & bit != 0).count();
    (set as f64 / f64::from(range), RANGE_WEIGHT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lone_digit_leans_its_top_bit_toward_nought_and_a_byte_its_high_bit() {
        // Two of the ten digits set the top bit of four.
        assert_eq!(nature(Mode::Numeric, 4, 0), (0.2, RANGE_WEIGHT));
        // The top of ten bits is set by 512 to 999.
        let (set, _) = nature(Mode::Numeric, 10, 0);
        assert!((set - 0.488).abs() < 1e-9, "{set}");
        assert_eq!(nature(Mode::Byte, 8, 0), (BYTE_HIGH_SET, BYTE_HIGH_WEIGHT));
        assert_eq!(nature(Mode::Byte, 8, 3), (0.5, EVEN_WEIGHT));
    }

    #[test]
    fn groups_follow_the_packing_of_each_mode() {
        let widths = |mode, count, len: usize| groups_of(mode, Some(count), 100..100 + len).iter().map(|g| g.width).collect::<Vec<_>>();
        assert_eq!(widths(Mode::Numeric, 7, 34), [10, 10, 4]);
        assert_eq!(widths(Mode::Numeric, 8, 27), [10, 10, 7]);
        assert_eq!(widths(Mode::Numeric, 6, 20), [10, 10]);
        assert_eq!(widths(Mode::Alphanumeric, 5, 28), [11, 11, 6]);
        assert_eq!(widths(Mode::Byte, 3, 24), [8, 8, 8]);
        // A truncated segment keeps the groups that fit.
        assert_eq!(widths(Mode::Byte, 3, 20), [8, 8]);
        assert!(groups_of(Mode::Eci, None, 0..8).is_empty());
    }

    #[test]
    fn a_read_bit_is_moved_for_the_search_alone_and_an_unread_one_for_both() {
        let mut lean = vec![[0f32; 8]; 2];
        // Both codewords' high bits lean toward nought by six bits of
        // likelihood.
        let toward = (63f32).ln();
        lean[0][7] = toward;
        lean[1][7] = toward;
        let evidence = MessageEvidence { lean };
        // The first read as one, faintly; the second's high bit unread.
        let block = [CwRead { value: 0x80 | 0x41, known: 0xFF, pinned: 0 }, CwRead { value: 0x21, known: 0x7F, pinned: 0 }];
        let bits = [[200, 200, 200, 200, 200, 200, 200, 4], [200; 8]];
        let laid = evidence.laid(&[0, 1], &block, &bits, true).unwrap();
        // Not doubting by reliability, the reading stands for the search
        // too, and only the unread bit is filled.
        let unmoved = evidence.laid(&[0, 1], &block, &bits, false).unwrap();
        assert_eq!((unmoved.moved, unmoved.searched[0]), (1, block[0]));
        assert_eq!(laid.moved, 2);
        // Searched: the faint one turned to nought, the unread one filled.
        assert_eq!(laid.searched[0].value, 0x41);
        assert!(laid.searched_bits[0][7] > 0 && laid.searched_bits[0][7] < 40, "{:?}", laid.searched_bits[0]);
        assert_eq!(laid.searched[1], CwRead { value: 0x21, known: 0xFF, pinned: 0 });
        assert_eq!(laid.searched_bits[1][7], reliability_at(toward));
        // Judged: the reading stands as read, the unread bit is filled.
        assert_eq!((laid.judged[0], laid.judged_bits[0]), (block[0], bits[0]));
        assert_eq!(laid.judged[1], laid.searched[1]);
        assert_eq!(laid.judged_bits[1][7], reliability_at(toward));
        // A pinned bit and a certain one are left alone.
        let block = [CwRead { value: 0x80, known: 0xFF, pinned: 0x80 }, CwRead { value: 0x80, known: 0xFF, pinned: 0 }];
        let bits = [[200; 8], [CERTAIN; 8]];
        assert!(evidence.laid(&[0, 1], &block, &bits, true).is_none());
    }
}
