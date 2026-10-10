//! BCH codes protecting format and version information, and ranking of the
//! valid codewords against bits read with gaps.

use std::cmp::Reverse;

use crate::symbol::function_map::{FunctionMap, InfoCopy};
use crate::symbol::grid::{CERTAIN, ModuleGrid};
use crate::symbol::{EcLevel, Family, SymbolVersion, tables};

const FORMAT_GENERATOR: u32 = 0x537;
const INFO_GENERATOR: u32 = 0x1F25;

/// The XOR mask QR Code Model 1 put on its format word, where Model 2 puts
/// [`MODEL2_FORMAT_MASK`]: zxing-cpp's `FORMAT_INFO_MASK_MODEL1`. The two
/// models share their finder and timing patterns and symbol sizes, so the
/// mask is what tells a grid of one model from the other.
pub const MODEL1_FORMAT_MASK: u32 = 0x2825;
/// The XOR mask of QR Code Model 2 (ISO/IEC 18004 7.9.1).
pub const MODEL2_FORMAT_MASK: u32 = 0x5412;

/// The format XOR mask of the other QR Code model, for a version of one
/// of them.
fn other_model_mask(version: SymbolVersion) -> Option<u32> {
    match version.family() {
        Family::Qr => Some(MODEL1_FORMAT_MASK),
        Family::Model1 => Some(MODEL2_FORMAT_MASK),
        Family::Micro | Family::Rmqr => None,
    }
}

/// Errors either BCH code corrects in one codeword.
const CORRECTABLE: u32 = 3;
/// Known bits a read needs before agreeing with a codeword means anything:
/// fewer than this agree with some word by chance too easily.
pub const MIN_INFO_BITS: u32 = 8;
/// How much less the bits a candidate overturns must weigh than those the
/// runner-up overturns, at the same count, for the weights to decide
/// between them: a quarter of the sampler's scale, so that one bit read
/// well inside its colour against one read near the threshold settles
/// it, and two bits of middling margin do not. ZXing and ZBar rank format
/// words by Hamming distance alone; a module sampled from a photo comes
/// with a margin, and a faint bit is the one most likely misread.
pub const WEIGHT_MARGIN: u32 = 64;

/// Remainder of `data · x^degree` divided by `generator`.
pub fn bch_remainder(data: u32, generator: u32, degree: u32) -> u32 {
    let mut value = data << degree;
    for bit in (degree..32).rev() {
        if value >> bit & 1 != 0 {
            value ^= generator << (bit - degree);
        }
    }
    value
}

/// (15,5) codeword for QR and Micro QR format data, before the XOR mask.
pub fn format15(data: u32) -> u32 {
    data << 10 | bch_remainder(data, FORMAT_GENERATOR, 10)
}

/// (18,6) codeword for QR version info and rMQR format data, before any XOR.
pub fn info18(data: u32) -> u32 {
    data << 12 | bch_remainder(data, INFO_GENERATOR, 12)
}

/// Bits of one info copy as painted, with the copy's XOR already removed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InfoRead {
    pub bits: u32,
    pub known: u32,
    pub len: u8,
    /// The XOR the standard applies to this copy, kept so that a word
    /// painted without it can still be recognised.
    pub xor: u32,
    /// How sure each known bit is, least significant first: the sampler's
    /// reliability for a module read from a photo, [`CERTAIN`] for one
    /// painted by hand. Overturning a faint bit costs a candidate little,
    /// overturning a solid one much.
    pub weights: [u8; 18],
}

impl InfoRead {
    pub fn known_count(&self) -> u32 {
        self.known.count_ones()
    }

    /// (mismatches, compared) against a codeword.
    pub fn distance(&self, word: u32) -> (u32, u32) {
        self.distance_within(word, u32::MAX)
    }

    /// (mismatches, compared) against a codeword over the bits in `mask`.
    pub fn distance_within(&self, word: u32, mask: u32) -> (u32, u32) {
        let known = self.known & mask;
        (((self.bits ^ word) & known).count_ones(), known.count_ones())
    }

    /// Summed weights of the known bits in `mask` that disagree with a
    /// codeword: what overturning them costs.
    pub fn weight_within(&self, word: u32, mask: u32) -> u32 {
        let wrong = (self.bits ^ word) & self.known & mask;
        (0..self.len as usize).filter(|&i| wrong >> i & 1 != 0).map(|i| u32::from(self.weights[i])).sum()
    }

    /// Summed weights of the known bits that disagree with a codeword.
    pub fn weight(&self, word: u32) -> u32 {
        self.weight_within(word, u32::MAX)
    }

    /// Most of the copy was read and every module of it is one colour: ink
    /// or a stain over the copy, glare, or bare paper where it should be,
    /// and no word. BoofCV's Micro QR reader refuses a format read of all
    /// zeros or all ones before any decoding, as "typical failure modes
    /// for reading all white or black regions". Here such a copy passed
    /// for a word: light throughout it is, to the bit, the first format
    /// word of a QR Code or rMQR symbol painted without its XOR mask, dark
    /// throughout the last of a QR Code's, and either way it lies three
    /// bits from a Micro QR word painted the standard way. A symbol whose
    /// format bits were inked over was then decoded under that one phantom
    /// format, and every other format left untried.
    pub fn is_blank(&self) -> bool {
        let painted = (self.bits ^ self.xor) & self.known;
        3 * self.known_count() >= 2 * u32::from(self.len) && (painted == 0 || painted == self.known)
    }

    /// The copy as it is ranked: with nothing known when it [is
    /// blank](Self::is_blank), since it says nothing.
    fn heard(&self) -> Self {
        if self.is_blank() { Self { bits: 0, known: 0, weights: [0; 18], ..*self } } else { *self }
    }
}

/// Build an info read from the colour of each bit's module, least
/// significant first, `None` where it is not known. Every known bit
/// weighs as a certain one.
pub fn info_read_from(copy: &InfoCopy, dark: impl Fn(usize) -> Option<bool>) -> InfoRead {
    info_read_weighted(copy, |i| dark(i).map(|d| (d, CERTAIN)))
}

/// Build an info read from the colour of each bit's module and how sure
/// it is, least significant first, `None` where it is not known.
pub fn info_read_weighted(copy: &InfoCopy, dark: impl Fn(usize) -> Option<(bool, u8)>) -> InfoRead {
    let mut read = InfoRead { len: copy.bits.len() as u8, xor: copy.xor, ..Default::default() };
    for i in 0..copy.bits.len() {
        if let Some((is_dark, weight)) = dark(i) {
            if is_dark {
                read.bits |= 1 << i;
            }
            read.known |= 1 << i;
            read.weights[i] = weight;
        }
    }
    read.bits = (read.bits ^ copy.xor) & read.known;
    read
}

pub fn read_info(grid: &ModuleGrid, copy: &InfoCopy) -> InfoRead {
    info_read_weighted(copy, |i| {
        let (x, y) = (copy.bits[i].x as usize, copy.bits[i].y as usize);
        grid.get(x, y).as_dark().map(|d| (d, grid.reliability(x, y)))
    })
}

/// Bits of a copy in each of its two physical halves. The halves of a QR
/// format copy sit by different finder patterns (bits 0..8 by the top
/// right, 8..15 by the bottom left in the second copy), and damage usually
/// takes one corner, so a clean word can often be assembled from the
/// halves of two copies, as ZBar does. Micro QR has one copy and no split.
fn half_masks(version: SymbolVersion, len: u8) -> [u32; 2] {
    let split = match version.family() {
        Family::Qr | Family::Model1 => 8,
        Family::Rmqr => 9,
        Family::Micro => u32::from(len),
    };
    let all = (1u32 << len) - 1;
    let low = (1u32 << split) - 1;
    [low, all & !low]
}

/// What a format codeword says about the symbol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FormatFields {
    /// Version implied by the format data. For QR this is always the grid's
    /// version, since QR format info does not carry one.
    pub version: SymbolVersion,
    pub ec: EcLevel,
    pub mask: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormatCandidate {
    pub fields: FormatFields,
    /// The codeword with no XOR applied.
    pub word: u32,
    /// Mismatches summed over every copy.
    pub mismatches: u32,
    pub compared: u32,
    /// (mismatches, compared) for each copy.
    pub per_copy: Vec<(u32, u32)>,
    /// Mismatches of the cleanest word the copies can be assembled into,
    /// each half taken from the copy that agrees best, and the bits that
    /// assembly compared. One clean copy makes this zero whatever the
    /// other copy says, and two copies damaged in different halves still
    /// make a clean word between them.
    pub best_mismatches: u32,
    pub best_compared: u32,
    /// Summed weights of the bits that assembly overturns: what the
    /// mismatches cost, in the sampler's reliability.
    pub best_weight: u32,
    /// The word matches only as painted without the standard's XOR mask,
    /// which some encoders leave out.
    pub unmasked: bool,
    /// The word matches only as painted with the XOR mask of the other
    /// QR Code model: Model 1's on a Model 2 grid, or Model 2's on a
    /// Model 1 grid. The two models lay their data out differently, so
    /// the grid wants reading as the other model.
    pub other_model: bool,
    /// False when the candidate implies a different symbol size than the grid.
    pub fits_grid: bool,
}

impl FormatCandidate {
    /// The assembled read lies beyond what the code can correct. A reading
    /// under the other model's mask is always doubtful as a format to
    /// decode under: the grid's model lays its data out otherwise, and a
    /// damaged word of the grid's own model lies near some word of the
    /// other too often for the reading to decide anything. It serves as a
    /// warning, through [`FormatAnalysis::other_model_hint`].
    fn doubtful(&self) -> bool {
        self.other_model || self.best_mismatches > CORRECTABLE
    }

    /// Painted as the standard paints it for the grid's family, with the
    /// family's mask.
    pub fn is_standard(&self) -> bool {
        !self.unmasked && !self.other_model
    }

    /// How far from the standard the painting is: 0 standard, 1 without the
    /// mask, 2 with the other model's mask.
    fn quirk_rank(&self) -> u8 {
        u8::from(self.unmasked) + 2 * u8::from(self.other_model)
    }
}

#[derive(Clone, Debug, Default)]
pub struct FormatAnalysis {
    pub copies: Vec<InfoRead>,
    /// Best first. Candidates that fit the grid sort ahead of those that do
    /// not; then those within correction range, as painted the standard way
    /// before the unmasked quirk; then fewer assembled mismatches.
    pub candidates: Vec<FormatCandidate>,
}

impl FormatAnalysis {
    pub fn best(&self) -> Option<&FormatCandidate> {
        self.candidates.first().filter(|c| c.fits_grid)
    }

    /// The standard word nearest the bits, whatever symbol size it implies:
    /// for Micro QR and rMQR the word names the version, so this is what
    /// the bits say the symbol is.
    pub fn closest(&self) -> Option<&FormatCandidate> {
        self.candidates
            .iter()
            .filter(|c| c.is_standard())
            .min_by_key(|c| (c.best_mismatches, Reverse(c.best_compared), c.best_weight))
    }

    /// Known bits over every copy that says anything: a [blank
    /// copy](InfoRead::is_blank) has none.
    pub fn known_bits(&self) -> u32 {
        self.copies.iter().map(|c| c.heard().known_count()).sum()
    }

    /// Mismatches of the runner-up less those of `first`, among candidates
    /// that pass `admit`. Zero means the bits cannot tell them apart.
    fn margin_among(&self, first: &FormatCandidate, admit: impl Fn(&FormatCandidate) -> bool) -> u32 {
        self.candidates
            .iter()
            .filter(|c| admit(c) && c.fields != first.fields)
            .map(|c| c.best_mismatches)
            .min()
            .map_or(u32::MAX, |m| m.saturating_sub(first.best_mismatches))
    }

    /// How much more the bits the runner-up overturns weigh than those
    /// `first` overturns, among candidates that pass `admit` and overturn
    /// as many bits as `first`: what tells them apart when the count
    /// cannot. `u32::MAX` when no such candidate exists.
    fn weight_margin_among(&self, first: &FormatCandidate, admit: impl Fn(&FormatCandidate) -> bool) -> u32 {
        self.candidates
            .iter()
            .filter(|c| admit(c) && c.fields != first.fields && c.best_mismatches == first.best_mismatches)
            .map(|c| c.best_weight)
            .min()
            .map_or(u32::MAX, |w| w.saturating_sub(first.best_weight))
    }

    fn rivals(best: &FormatCandidate) -> impl Fn(&FormatCandidate) -> bool {
        let rank = best.quirk_rank();
        move |c| c.fits_grid && c.quirk_rank() == rank
    }

    /// How many more mismatches the runner-up has than the best candidate.
    /// Zero means the format bits cannot tell them apart by count alone.
    pub fn margin(&self) -> u32 {
        match self.best() {
            Some(best) => self.margin_among(best, Self::rivals(best)),
            None => u32::MAX,
        }
    }

    /// How much more the runner-up's overturned bits weigh than the best
    /// candidate's, when they overturn as many.
    pub fn weight_margin(&self) -> u32 {
        match self.best() {
            Some(best) => self.weight_margin_among(best, Self::rivals(best)),
            None => u32::MAX,
        }
    }

    /// True when nothing distinguishes the best candidate from the next:
    /// neither the count of bits each overturns nor, at the same count,
    /// a clear difference in how faint those bits were.
    pub fn is_ambiguous(&self) -> bool {
        self.margin() == 0 && self.weight_margin() < WEIGHT_MARGIN
    }

    /// Bits the best candidate would have to overturn, beyond which the
    /// read is noise: both BCH codes correct three errors in a word, and
    /// the assembled read is one word.
    pub fn correctable(&self) -> u32 {
        if self.known_bits() > 0 { CORRECTABLE } else { 0 }
    }

    /// True when the best candidate differs from the bits read by more than
    /// the code can correct, so the bits do not support it.
    pub fn is_doubtful(&self) -> bool {
        self.best().is_some_and(|b| b.best_mismatches > self.correctable())
    }

    /// The format bits settle the EC level and mask: a candidate fits the
    /// grid, nothing else comes as close, and it is within correction range.
    pub fn is_reliable(&self) -> bool {
        self.best().is_some() && !self.is_ambiguous() && !self.is_doubtful()
    }

    /// The best candidate only matches as painted without the XOR mask.
    pub fn unmasked_in_use(&self) -> bool {
        self.best().is_some_and(|b| b.unmasked)
    }

    /// A reading under the other QR Code model's mask that is exact over a
    /// whole word where the best reading under the grid's own mask is at
    /// least two bits off: the mark of a grid of the other model, a
    /// pre-2000 Model 1 symbol read as Model 2 or the reverse. A clean word
    /// of one model never reads as a word of the other, since the two masks
    /// differ in ten bits and the words in at least seven; a damaged one
    /// does so only by a particular pattern of damage, and a partial read
    /// is not asked, since a few bits agree with one of thirty-two words
    /// too easily.
    pub fn other_model_hint(&self) -> Option<&FormatCandidate> {
        let standard = self.closest().map_or(u32::MAX, |c| c.best_mismatches);
        self.candidates
            .iter()
            .find(|c| c.fits_grid && c.other_model && c.best_compared >= 15 && c.best_mismatches == 0)
            .filter(|_| standard >= 2)
    }

    /// What the bits say about the read as a whole: `Some(true)` when they
    /// spell a valid word with enough known bits to mean it, `Some(false)`
    /// when they lie beyond what the code can correct, and `None` when too
    /// few are known to tell. Judged on the nearest standard word of any
    /// size, since a read taken the wrong way round or at the wrong
    /// version still spells some word if it is read right.
    pub fn verdict(&self) -> Option<bool> {
        let closest = self.closest()?;
        if self.known_bits() == 0 {
            return None;
        }
        if closest.doubtful() {
            return Some(false);
        }
        let margin = self.margin_among(closest, FormatCandidate::is_standard);
        let decided = margin > 0 || self.weight_margin_among(closest, FormatCandidate::is_standard) >= WEIGHT_MARGIN;
        (closest.best_compared >= MIN_INFO_BITS && decided).then_some(true)
    }
}

/// Every format codeword valid for the symbol family, with its meaning.
pub fn format_words(version: SymbolVersion) -> Vec<(FormatFields, u32)> {
    match version.family() {
        Family::Qr | Family::Model1 => (0..32)
            .map(|data| {
                let ec = [EcLevel::M, EcLevel::L, EcLevel::H, EcLevel::Q][(data >> 3) as usize];
                (FormatFields { version, ec, mask: (data & 7) as u8 }, format15(data))
            })
            .collect(),
        Family::Micro => (0..32)
            .map(|data| {
                let (v, ec) = tables::MICRO_SYMBOLS[(data >> 2) as usize];
                let fields = FormatFields { version: SymbolVersion::Micro(v), ec, mask: (data & 3) as u8 };
                (fields, format15(data))
            })
            .collect(),
        Family::Rmqr => (0..64)
            .map(|data| {
                let ec = if data & 32 != 0 { EcLevel::H } else { EcLevel::M };
                let fields =
                    FormatFields { version: SymbolVersion::Rmqr((data & 31) as u8), ec, mask: 0 };
                (fields, info18(data))
            })
            .collect(),
    }
}

/// The format data value that encodes the given choice.
pub fn format_word_for(version: SymbolVersion, ec: EcLevel, mask: u8) -> Option<u32> {
    format_words(version)
        .into_iter()
        .find(|(f, _)| f.version == version && f.ec == ec && f.mask == mask)
        .map(|(_, word)| word)
}

pub fn analyze_format(grid: &ModuleGrid, func: &FunctionMap) -> FormatAnalysis {
    let copies: Vec<InfoRead> = func.format.iter().map(|c| read_info(grid, c)).collect();
    analyze_format_reads(func.version, copies)
}

/// Rank every format word of the family against copies already read, for
/// a grid of `version`. A copy that [is blank](InfoRead::is_blank) is kept
/// as read, for showing, and ranks nothing: when no copy says anything no
/// word is nearer than another, the read settles no format, and the
/// caller's search of every format decides, which a symbol whose word
/// really is the one a blank copy spells, painted without its mask, comes
/// through as well.
pub fn analyze_format_reads(version: SymbolVersion, copies: Vec<InfoRead>) -> FormatAnalysis {
    let read = copies;
    let copies: Vec<InfoRead> = read.iter().map(InfoRead::heard).collect();
    let halves = copies.first().map_or([0, 0], |c| half_masks(version, c.len));
    let mut candidates = Vec::with_capacity(2 * 64);
    // The word may have been painted the standard way, without the XOR mask
    // that some encoders leave out, or, for either QR Code model, with the
    // mask of the other model, as zxing-cpp tries all three.
    let other = other_model_mask(version);
    let quirks: &[u8] = if other.is_some() { &[0, 1, 2] } else { &[0, 1] };
    for (fields, word) in format_words(version) {
        for &quirk in quirks {
            // The word as each copy would hold it, once the copy's own XOR
            // has been removed as the standard way requires.
            let target = |copy: &InfoRead| match quirk {
                0 => word,
                1 => word ^ copy.xor,
                _ => word ^ copy.xor ^ other.unwrap_or(0),
            };
            let per_copy: Vec<(u32, u32)> = copies.iter().map(|c| c.distance(target(c))).collect();
            let (mut best_mismatches, mut best_compared, mut best_weight) = (0, 0, 0);
            for &mask in &halves {
                let pick = copies
                    .iter()
                    .map(|c| {
                        let (wrong, compared) = c.distance_within(target(c), mask);
                        (wrong, compared, c.weight_within(target(c), mask))
                    })
                    .filter(|&(_, compared, _)| compared > 0)
                    .min_by_key(|&(wrong, compared, weight)| (wrong, Reverse(compared), weight));
                if let Some((wrong, compared, weight)) = pick {
                    best_mismatches += wrong;
                    best_compared += compared;
                    best_weight += weight;
                }
            }
            candidates.push(FormatCandidate {
                fields,
                word,
                mismatches: per_copy.iter().map(|p| p.0).sum(),
                compared: per_copy.iter().map(|p| p.1).sum(),
                per_copy,
                best_mismatches,
                best_compared,
                best_weight,
                unmasked: quirk == 1,
                other_model: quirk == 2,
                fits_grid: fields.version == version,
            });
        }
    }
    candidates.sort_by_key(|c| {
        (!c.fits_grid, c.doubtful(), c.quirk_rank(), c.best_mismatches, c.best_weight, c.mismatches)
    });
    FormatAnalysis { copies: read, candidates }
}

/// Result of reading QR version information (versions 7 and up). Each
/// version is judged by the cleanest read the copies assemble into, half
/// by half, as the format copies are: one intact copy decides however the
/// other was damaged, and two copies damaged in different halves still
/// make a clean word between them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VersionAnalysis {
    pub copies: Vec<InfoRead>,
    pub best_version: u8,
    /// Mismatches of the cleanest read the copies assemble into for the
    /// best version.
    pub mismatches: u32,
    /// Known bits that assembly compared.
    pub compared: u32,
    /// Mismatches of the next best version minus those of the best.
    pub margin: u32,
    /// Summed weights of the bits the best version's assembly overturns.
    pub weight: u32,
    /// How much more the next best version's overturned bits weigh than
    /// the best version's, when the two overturn as many; `u32::MAX` when
    /// the counts differ.
    pub weight_margin: u32,
}

impl VersionAnalysis {
    /// Known bits over every copy.
    pub fn known_bits(&self) -> u32 {
        self.copies.iter().map(InfoRead::known_count).sum()
    }

    /// The best read still lies beyond what the code can correct.
    pub fn is_doubtful(&self) -> bool {
        self.compared > 0 && self.mismatches > CORRECTABLE
    }

    /// The bits name a version: within correction range, on enough known
    /// bits, with nothing else as close, by count or, at the same count,
    /// by how faint the overturned bits were.
    pub fn is_reliable(&self) -> bool {
        self.compared >= MIN_INFO_BITS
            && self.mismatches <= CORRECTABLE
            && (self.margin > 0 || self.weight_margin >= WEIGHT_MARGIN)
    }

    /// As [`FormatAnalysis::verdict`], for the version bits.
    pub fn verdict(&self) -> Option<bool> {
        if self.known_bits() == 0 {
            None
        } else if self.is_doubtful() {
            Some(false)
        } else {
            self.is_reliable().then_some(true)
        }
    }
}

pub fn analyze_version(grid: &ModuleGrid, func: &FunctionMap) -> Option<VersionAnalysis> {
    if func.version_info.is_empty() {
        return None;
    }
    let copies: Vec<InfoRead> = func.version_info.iter().map(|c| read_info(grid, c)).collect();
    Some(analyze_version_reads(copies))
}

/// (mismatches, compared, weight) of a version's assembled read, ordered
/// worst last.
type VersionScore = (u32, Reverse<u32>, u32);

/// Bits of a version copy in each of its two physical halves. The bits
/// run three to a row down the 6 by 3 block, so bits 0..9 are its first
/// three rows (columns, in the transposed copy) and 9..18 the last three,
/// and damage that takes one end of the block leaves the other half.
const VERSION_HALVES: [u32; 2] = [0x1FF, 0x3FFFF & !0x1FF];

/// Rank every version against version-information copies already read.
pub fn analyze_version_reads(copies: Vec<InfoRead>) -> VersionAnalysis {
    let mut scored: Vec<(VersionScore, u8)> = (7..=40u8)
        .map(|v| {
            let word = info18(v as u32);
            let (mut mismatches, mut compared, mut weight) = (0, 0, 0);
            for &mask in &VERSION_HALVES {
                let pick = copies
                    .iter()
                    .map(|c| {
                        let (wrong, compared) = c.distance_within(word, mask);
                        (wrong, Reverse(compared), c.weight_within(word, mask))
                    })
                    .filter(|&(_, Reverse(compared), _)| compared > 0)
                    .min();
                if let Some((wrong, Reverse(known), cost)) = pick {
                    mismatches += wrong;
                    compared += known;
                    weight += cost;
                }
            }
            ((mismatches, Reverse(compared), weight), v)
        })
        .collect();
    scored.sort();
    let (best, next) = (scored[0].0, scored[1].0);
    VersionAnalysis {
        copies,
        best_version: scored[0].1,
        mismatches: best.0,
        compared: best.1.0,
        margin: next.0 - best.0,
        weight: best.2,
        weight_margin: if next.0 == best.0 { next.2.saturating_sub(best.2) } else { u32::MAX },
    }
}

/// What one format read says, in brief: whether it spells a word, and how
/// many bits the nearest standard word is off by.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FormatVerdict {
    pub valid: Option<bool>,
    /// Assembled mismatches of the nearest standard word; `None` when no
    /// bit is known.
    pub mismatches: Option<u32>,
}

impl FormatAnalysis {
    pub fn summary(&self) -> FormatVerdict {
        FormatVerdict {
            valid: self.verdict(),
            mismatches: (self.known_bits() > 0).then(|| self.closest().map_or(0, |c| c.best_mismatches)),
        }
    }
}

/// Whether the format bits read in mirror order are to be believed over
/// the bits read as is. Being within three bits of some word proves
/// little: the words' correction balls cover more than half of all
/// fifteen-bit strings, so noise reads as "a word with errors" more often
/// than not. A mirrored read is believed only when it is a word to within
/// one bit, which noise manages a few times in a hundred, and when it is
/// clearly nearer a word than the plain read, by two bits. Reversing a
/// clean word moves it at least three bits from every word (two, once the
/// dark module intrudes on the second QR copy), so a clean mirror image
/// always wins, and a clean plain read never loses.
pub fn mirror_wins(plain: FormatVerdict, mirrored: FormatVerdict) -> bool {
    let Some(m) = mirrored.mismatches else { return false };
    if mirrored.valid != Some(true) || m > 1 {
        return false;
    }
    plain.mismatches.is_none_or(|p| p >= m + 2)
}

/// What the format bits, and the version bits where the symbol has them,
/// say about a grid read as a whole: `Some(false)` when either lies beyond
/// correction, `Some(true)` when at least one spells a valid word and
/// neither contradicts it, `None` when too little is known.
pub fn info_verdict(format: &FormatAnalysis, version: Option<&VersionAnalysis>) -> Option<bool> {
    let verdicts = [format.verdict(), version.and_then(VersionAnalysis::verdict)];
    if verdicts.contains(&Some(false)) {
        Some(false)
    } else if verdicts.contains(&Some(true)) {
        Some(true)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_codewords() {
        // ISO/IEC 18004 Annex C: EC level M, mask 5 -> 100000011001110 after XOR.
        assert_eq!(format15(0b00101) ^ 0x5412, 0b100000011001110);
        // Annex D: version 7 -> 000111110010010100.
        assert_eq!(info18(7), 0b000111110010010100);
        assert_eq!(info18(40), 0b101000110001101001);
    }

    #[test]
    fn halves_assemble_the_cleanest_word() {
        use crate::symbol::Pos;
        let version = SymbolVersion::Qr(3);
        let word = format15(0b01101);
        let copy = |xor| InfoCopy { bits: (0..15).map(|i| Pos::new(i, 0)).collect(), xor };
        let read = |painted: u32, known: u32, xor: u32| {
            info_read_from(&copy(xor), |i| (known >> i & 1 != 0).then(|| painted >> i & 1 != 0))
        };
        let all = 0x7FFF;
        // Copy one trashed in its low half, copy two trashed in its high
        // half: no copy is within range alone, the assembly is clean.
        let a = read(word ^ 0x5412 ^ 0x00F0, all, 0x5412);
        let b = read(word ^ 0x5412 ^ 0x7800, all, 0x5412);
        let analysis = analyze_format_reads(version, vec![a, b]);
        let best = analysis.best().unwrap();
        assert_eq!(best.word, word);
        assert_eq!((best.best_mismatches, best.best_compared), (0, 15));
        assert_eq!(best.mismatches, 8);
        assert!(analysis.is_reliable() && !analysis.unmasked_in_use());
        // One copy unreadable, the other clean.
        let analysis = analyze_format_reads(version, vec![read(0, 0, 0x5412), b]);
        assert_eq!(analysis.best().unwrap().best_mismatches, 4);
        assert!(analysis.is_doubtful());
        assert_eq!(analysis.verdict(), Some(false));
        let analysis = analyze_format_reads(version, vec![read(0, 0, 0x5412), a]);
        assert_eq!(analysis.best().unwrap().word, word);
        assert_eq!(analysis.verdict(), Some(false), "four wrong bits in one half is beyond the code");
        // Nothing known: nothing to say.
        let analysis = analyze_format_reads(version, vec![read(0, 0, 0x5412), read(0, 0, 0x5412)]);
        assert_eq!(analysis.verdict(), None);
        assert_eq!(analysis.correctable(), 0);
        // Painted without the XOR mask: recognised, flagged, and ranked
        // behind any standard word within range.
        let quirk = read(word ^ 0x5412 ^ 0x5412, all, 0x5412);
        let analysis = analyze_format_reads(version, vec![quirk, quirk]);
        let best = analysis.best().unwrap();
        assert!(best.unmasked && best.word == word && best.best_mismatches == 0);
        assert!(analysis.is_reliable());
        assert_eq!(analysis.verdict(), Some(false), "as a standard read it is noise");
    }

    #[test]
    fn a_copy_of_one_colour_is_no_word() {
        use crate::symbol::Pos;
        let copy = |len: usize, xor| InfoCopy { bits: (0..len).map(|i| Pos::new(i, 0)).collect(), xor };
        let read = |len: usize, painted: u32, known: u32, xor: u32| {
            info_read_from(&copy(len, xor), |i| (known >> i & 1 != 0).then(|| painted >> i & 1 != 0))
        };
        let all = 0x7FFF;
        let version = SymbolVersion::Qr(5);
        // Both copies inked over, or both bare paper: to the bit a word
        // painted without the mask, and it settles nothing.
        for painted in [0, all] {
            let blot = read(15, painted, all, 0x5412);
            assert!(blot.is_blank());
            let analysis = analyze_format_reads(version, vec![blot, blot]);
            assert!(!analysis.is_reliable() && !analysis.unmasked_in_use(), "{painted:#x}");
            assert_eq!((analysis.verdict(), analysis.known_bits()), (None, 0));
            assert_eq!(analysis.summary().mismatches, None);
            // The copies are kept as they were read.
            assert_eq!(analysis.copies[0].known_count(), 15);
            // One copy blotted and the other clean: the clean one decides.
            let word = format15(0b11010);
            let analysis = analyze_format_reads(version, vec![blot, read(15, word ^ 0x5412, all, 0x5412)]);
            assert!(analysis.is_reliable());
            assert_eq!(analysis.best().unwrap().word, word);
        }
        // A blot with a module or two unread is a blot still; a few
        // modules of one colour, with the rest unread, are not yet one.
        assert!(read(15, 0, all & !0b101, 0x5412).is_blank());
        assert!(!read(15, 0, 0x01FF, 0x5412).is_blank());
        // A word of the standard's own is never all one colour as painted.
        for (_, word) in format_words(version) {
            assert!(!read(15, word ^ 0x5412, all, 0x5412).is_blank());
        }
        // Bare paper where a Micro QR symbol's format bits should be lies
        // three bits from a word of its own, and named a version; the
        // same for ink.
        for painted in [0, all] {
            let micro = analyze_format_reads(SymbolVersion::Micro(4), vec![read(15, painted, all, 0x4445)]);
            assert_eq!(micro.verdict(), None);
            assert!(!micro.is_reliable());
        }
        // And an rMQR symbol's read light throughout is its first word
        // without the masks.
        let rmqr = SymbolVersion::Rmqr(0);
        let bare = analyze_format_reads(rmqr, vec![read(18, 0, 0x3FFFF, 0x1FAB2), read(18, 0, 0x3FFFF, 0x20A7B)]);
        assert!(!bare.is_reliable() && bare.verdict().is_none());
    }

    #[test]
    fn model1_mask_is_recognised() {
        use crate::symbol::Pos;
        let version = SymbolVersion::Qr(2);
        // A word whose Model 1 painting reads as noise the standard way:
        // beyond correction for every standard word.
        let data = (0..32u32)
            .find(|&d| {
                let painted_as_read = format15(d) ^ MODEL1_FORMAT_MASK ^ 0x5412;
                (0..32).all(|e| (painted_as_read ^ format15(e)).count_ones() > CORRECTABLE)
            })
            .expect("some word reads as noise");
        let word = format15(data);
        let copy = |xor| InfoCopy { bits: (0..15).map(|i| Pos::new(i, 0)).collect(), xor };
        let read = |painted: u32, xor: u32| info_read_from(&copy(xor), |i| Some(painted >> i & 1 != 0));
        let old = read(word ^ MODEL1_FORMAT_MASK, 0x5412);
        let analysis = analyze_format_reads(version, vec![old, old]);
        // The Model 1 word is recognised and pointed out, but never becomes
        // the format in effect on a Model 2 grid.
        let hint = analysis.other_model_hint().expect("an exact Model 1 word");
        assert!(hint.other_model && !hint.unmasked && hint.word == word && hint.best_mismatches == 0);
        assert!(analysis.best().unwrap().is_standard());
        assert!(analysis.is_doubtful() && !analysis.unmasked_in_use());
        assert_eq!(analysis.verdict(), Some(false), "as a standard read it is noise");
        // A standard painting is never taken for Model 1; a Model 1 word
        // with bits missing is not asked; the small symbols have no other
        // model at all.
        let standard = analyze_format_reads(version, vec![read(word ^ 0x5412, 0x5412)]);
        assert!(standard.best().unwrap().is_standard() && standard.other_model_hint().is_none());
        let partial = info_read_from(&copy(0x5412), |i| (i < 10).then(|| (word ^ MODEL1_FORMAT_MASK) >> i & 1 != 0));
        assert!(analyze_format_reads(version, vec![partial]).other_model_hint().is_none());
        let micro = analyze_format_reads(SymbolVersion::Micro(2), vec![read(word ^ MODEL1_FORMAT_MASK, 0x4445)]);
        assert!(micro.candidates.iter().all(|c| !c.other_model));
        // On a Model 1 grid the roles swap: the Model 1 painting is the
        // standard one, and a Model 2 painting is the hint.
        let model1 = SymbolVersion::Model1(2);
        let own = analyze_format_reads(model1, vec![read(word ^ MODEL1_FORMAT_MASK, MODEL1_FORMAT_MASK)]);
        let best = own.best().unwrap();
        assert!(best.is_standard() && best.word == word && best.best_mismatches == 0);
        assert!(own.is_reliable() && own.other_model_hint().is_none());
        let foreign = analyze_format_reads(model1, vec![read(word ^ 0x5412, MODEL1_FORMAT_MASK)]);
        let hint = foreign.other_model_hint().expect("an exact Model 2 word");
        assert!(hint.other_model && hint.word == word);
        assert!(foreign.is_doubtful());
    }

    #[test]
    fn mirrored_reads_must_win_clearly() {
        let v = |valid, mismatches| FormatVerdict { valid, mismatches };
        // A clean mirror image against its reversal.
        assert!(mirror_wins(v(Some(true), Some(3)), v(Some(true), Some(0))));
        assert!(mirror_wins(v(Some(false), Some(5)), v(Some(true), Some(0))));
        assert!(mirror_wins(v(None, None), v(Some(true), Some(1))));
        // A clean plain read never loses, nor a close call, and a mirrored
        // read that is merely within range is noise as often as not.
        assert!(!mirror_wins(v(Some(true), Some(0)), v(Some(true), Some(3))));
        assert!(!mirror_wins(v(Some(true), Some(2)), v(Some(true), Some(1))));
        assert!(!mirror_wins(v(Some(false), Some(5)), v(Some(true), Some(2))));
        assert!(!mirror_wins(v(Some(false), Some(5)), v(Some(false), Some(4))));
        assert!(!mirror_wins(v(None, None), v(None, None)));
    }

    #[test]
    fn version_halves_assemble_the_cleanest_word() {
        use crate::symbol::Pos;
        let copy = InfoCopy { bits: (0..18).map(|i| Pos::new(i, 0)).collect(), xor: 0 };
        let read = |painted: u32, known: u32| info_read_from(&copy, |i| (known >> i & 1 != 0).then(|| painted >> i & 1 != 0));
        let all = 0x3FFFF;
        let clean = read(info18(23), all);
        // Eight flips, five in one half and three in the other: beyond the
        // code, and within three bits of no word.
        let trashed = read(info18(23) ^ 0x0F0F, all);
        let analysis = analyze_version_reads(vec![trashed, clean]);
        assert_eq!((analysis.best_version, analysis.mismatches, analysis.compared), (23, 0, 18));
        assert!(analysis.is_reliable());
        assert_eq!(analysis.verdict(), Some(true));
        let analysis = analyze_version_reads(vec![trashed, read(0, 0)]);
        assert!(analysis.mismatches > CORRECTABLE);
        assert!(analysis.is_doubtful());
        assert_eq!(analysis.verdict(), Some(false));
        // One copy trashed in its low half, the other in its high half:
        // neither is within range alone, the assembly is clean.
        let low = read(info18(23) ^ 0x00F0, all);
        let high = read(info18(23) ^ 0x1E000, all);
        for pair in [vec![low, high], vec![high, low]] {
            let analysis = analyze_version_reads(pair);
            assert_eq!((analysis.best_version, analysis.mismatches, analysis.compared), (23, 0, 18));
            assert!(analysis.is_reliable());
        }
        // Damage across the split is not assembled around: the half with
        // the fewer wrong bits is taken from each copy.
        let across = read(info18(23) ^ 0x00300, all);
        let analysis = analyze_version_reads(vec![across, low]);
        assert_eq!((analysis.best_version, analysis.mismatches), (23, 1));
        assert!(analysis.is_reliable());
        let analysis = analyze_version_reads(vec![read(0, 0), read(0, 0)]);
        assert_eq!((analysis.compared, analysis.verdict()), (0, None));
        // Too few bits to mean anything, even when they agree.
        let few = read(info18(23), 0x3F);
        assert_eq!(analyze_version_reads(vec![few]).verdict(), None);
    }

    #[test]
    fn minimum_distances() {
        let f: Vec<u32> = (0..32).map(format15).collect();
        let v: Vec<u32> = (0..64).map(info18).collect();
        let min = |words: &[u32]| {
            let mut best = u32::MAX;
            for (i, a) in words.iter().enumerate() {
                for b in &words[i + 1..] {
                    best = best.min((a ^ b).count_ones());
                }
            }
            best
        };
        assert_eq!(min(&f), 7);
        assert_eq!(min(&v), 8);
    }
}
