//! Per-version constants for QR Code (ISO/IEC 18004), Micro QR Code, rMQR
//! (ISO/IEC 23941) and QR Code Model 1 (AIM ITS 97/001).
//!
//! The rMQR tables were cross-checked against zint and zxing-cpp, and the
//! Model 1 tables are zxing-cpp's; the `module_count_identity` test in
//! `function_map.rs` guards every entry here against the symbol geometry.

use super::{EcLevel, Family, SymbolVersion};

/// EC codewords per block for QR versions 1..=40, indexed `[level][version]`
/// with levels in L, M, Q, H order. Index 0 is unused.
#[rustfmt::skip]
const QR_EC_PER_BLOCK: [[u8; 41]; 4] = [
    [0,  7, 10, 15, 20, 26, 18, 20, 24, 30, 18, 20, 24, 26, 30, 22, 24, 28, 30, 28, 28, 28, 28, 30, 30, 26, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
    [0, 10, 16, 26, 18, 24, 16, 18, 22, 22, 26, 30, 22, 22, 24, 24, 28, 28, 26, 26, 26, 26, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28],
    [0, 13, 22, 18, 26, 18, 24, 18, 22, 20, 24, 28, 26, 24, 20, 30, 24, 28, 28, 26, 30, 28, 30, 30, 30, 30, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
    [0, 17, 28, 22, 16, 22, 28, 26, 26, 24, 28, 24, 28, 22, 24, 24, 30, 28, 28, 26, 28, 30, 24, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
];

/// Number of RS blocks for QR versions 1..=40, same indexing.
#[rustfmt::skip]
const QR_BLOCKS: [[u8; 41]; 4] = [
    [0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4,  4,  4,  4,  4,  6,  6,  6,  6,  7,  8,  8,  9,  9, 10, 12, 12, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 24, 25],
    [0, 1, 1, 1, 2, 2, 4, 4, 4, 5, 5,  5,  8,  9,  9, 10, 10, 11, 13, 14, 16, 17, 17, 18, 20, 21, 23, 25, 26, 28, 29, 31, 33, 35, 37, 38, 40, 43, 45, 47, 49],
    [0, 1, 1, 2, 2, 4, 4, 6, 6, 8, 8,  8, 10, 12, 16, 12, 17, 16, 18, 21, 20, 23, 23, 25, 27, 29, 34, 34, 35, 38, 40, 43, 45, 48, 51, 53, 56, 59, 62, 65, 68],
    [0, 1, 1, 2, 4, 4, 4, 5, 6, 8, 8, 11, 11, 16, 16, 18, 16, 19, 21, 25, 25, 25, 34, 30, 32, 35, 37, 40, 42, 45, 48, 51, 54, 57, 60, 63, 66, 70, 74, 77, 81],
];

/// QR Code Model 1 block structure for versions 1..=14, indexed
/// `[version][level]` with levels in L, M, Q, H order, as (EC codewords per
/// block, blocks, data codewords per block). Index 0 is unused. From
/// zxing-cpp's `Version::Model1` table (QRVersion.cpp), the one open
/// implementation; the version 2 entry is confirmed by zxing-cpp's sample
/// symbol, the rest are taken on trust. Model 1 divides a symbol into
/// equal blocks only, and where the blocks do not fill the symbol the last
/// codeword slots go unused (see [`unused_codewords`]).
#[rustfmt::skip]
const MODEL1_BLOCKS: [[(u8, u8, u8); 4]; 15] = [
    [(0, 0, 0); 4],
    [(7, 1, 19), (10, 1, 16), (13, 1, 13), (17, 1, 9)],
    [(10, 1, 36), (16, 1, 30), (22, 1, 24), (30, 1, 16)],
    [(15, 1, 57), (28, 1, 44), (36, 1, 36), (48, 1, 24)],
    [(20, 1, 80), (40, 1, 60), (50, 1, 50), (66, 1, 34)],
    [(26, 1, 108), (52, 1, 82), (66, 1, 68), (44, 2, 23)],
    [(34, 1, 136), (32, 2, 53), (42, 2, 43), (56, 2, 29)],
    [(42, 1, 170), (40, 2, 66), (52, 2, 54), (46, 3, 24)],
    [(24, 2, 104), (48, 2, 80), (64, 2, 64), (56, 3, 29)],
    [(30, 2, 123), (60, 2, 93), (50, 3, 52), (68, 3, 34)],
    [(34, 2, 145), (68, 2, 111), (58, 3, 61), (58, 4, 31)],
    [(40, 2, 168), (40, 4, 64), (52, 4, 52), (54, 5, 29)],
    [(46, 2, 192), (46, 4, 73), (58, 4, 61), (62, 5, 33)],
    [(36, 3, 144), (52, 4, 83), (66, 4, 69), (58, 6, 32)],
    [(40, 3, 163), (60, 4, 92), (60, 5, 62), (66, 6, 35)],
];

/// Micro QR symbol numbers 0..=7 as (version, EC level).
pub const MICRO_SYMBOLS: [(u8, EcLevel); 8] = [
    (1, EcLevel::L),
    (2, EcLevel::L),
    (2, EcLevel::M),
    (3, EcLevel::L),
    (3, EcLevel::M),
    (4, EcLevel::L),
    (4, EcLevel::M),
    (4, EcLevel::Q),
];

/// (total codewords, data codewords) per Micro QR symbol number. In M1 and M3
/// the last data codeword is four bits long.
const MICRO_CODEWORDS: [(u8, u8); 8] =
    [(5, 3), (10, 5), (10, 4), (17, 11), (17, 9), (24, 16), (24, 14), (24, 10)];

/// rMQR (height, width) by version indicator.
#[rustfmt::skip]
pub const RMQR_SIZES: [(usize, usize); 32] = [
    (7, 43), (7, 59), (7, 77), (7, 99), (7, 139),
    (9, 43), (9, 59), (9, 77), (9, 99), (9, 139),
    (11, 27), (11, 43), (11, 59), (11, 77), (11, 99), (11, 139),
    (13, 27), (13, 43), (13, 59), (13, 77), (13, 99), (13, 139),
    (15, 43), (15, 59), (15, 77), (15, 99), (15, 139),
    (17, 43), (17, 59), (17, 77), (17, 99), (17, 139),
];

#[rustfmt::skip]
const RMQR_TOTAL: [u16; 32] = [
    13, 21, 32, 44, 68,
    21, 33, 49, 66, 99,
    15, 31, 47, 67, 89, 132,
    21, 41, 60, 85, 113, 166,
    51, 74, 103, 136, 199,
    61, 88, 122, 160, 232,
];

/// Data codewords at levels M and H.
#[rustfmt::skip]
const RMQR_DATA: [[u16; 32]; 2] = [
    [
        6, 12, 20, 28, 44,
        12, 21, 31, 42, 63,
        7, 19, 31, 43, 57, 84,
        12, 27, 38, 53, 73, 106,
        33, 48, 67, 88, 127,
        39, 56, 78, 100, 152,
    ],
    [
        3, 7, 10, 14, 24,
        7, 11, 17, 22, 33,
        5, 11, 15, 23, 29, 42,
        7, 13, 20, 29, 35, 54,
        15, 26, 31, 48, 69,
        21, 28, 38, 56, 76,
    ],
];

/// RS blocks at levels M and H.
#[rustfmt::skip]
const RMQR_BLOCKS: [[u8; 32]; 2] = [
    [
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 2,
        1, 1, 1, 1, 2, 2,
        1, 1, 1, 2, 2, 3,
        1, 1, 2, 2, 3,
        1, 2, 2, 3, 4,
    ],
    [
        1, 1, 1, 1, 2,
        1, 1, 2, 2, 3,
        1, 1, 2, 2, 2, 3,
        1, 1, 2, 2, 3, 4,
        2, 2, 3, 4, 5,
        2, 2, 3, 4, 6,
    ],
];

/// Character count widths for numeric, alphanumeric, byte, kanji.
#[rustfmt::skip]
const RMQR_COUNT_BITS: [[u8; 32]; 4] = [
    [
        4, 5, 6, 7, 7,
        5, 6, 7, 7, 8,
        4, 6, 7, 7, 8, 8,
        5, 6, 7, 7, 8, 8,
        7, 7, 8, 8, 9,
        7, 8, 8, 8, 9,
    ],
    [
        3, 5, 5, 6, 6,
        5, 5, 6, 6, 7,
        4, 5, 6, 6, 7, 7,
        5, 6, 6, 7, 7, 8,
        6, 7, 7, 7, 8,
        6, 7, 7, 8, 8,
    ],
    [
        3, 4, 5, 5, 6,
        4, 5, 5, 6, 6,
        3, 5, 5, 6, 6, 7,
        4, 5, 6, 6, 7, 7,
        6, 6, 7, 7, 7,
        6, 6, 7, 7, 8,
    ],
    [
        2, 3, 4, 5, 5,
        3, 4, 5, 5, 6,
        2, 4, 5, 5, 6, 6,
        3, 5, 5, 6, 6, 7,
        5, 5, 6, 6, 7,
        5, 6, 6, 6, 7,
    ],
];

/// How the codewords of a symbol divide into RS blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockShape {
    pub total_codewords: usize,
    pub data_codewords: usize,
    pub blocks: usize,
    pub ec_per_block: usize,
}

impl BlockShape {
    /// Data codewords in block `b`. Short blocks come first.
    pub fn data_len(&self, b: usize) -> usize {
        let short = self.data_codewords / self.blocks;
        let long_blocks = self.data_codewords % self.blocks;
        short + usize::from(b >= self.blocks - long_blocks)
    }
}

fn level_index(ec: EcLevel) -> usize {
    ec as usize
}

fn micro_symbol(version: u8, ec: EcLevel) -> Option<usize> {
    MICRO_SYMBOLS.iter().position(|&(v, e)| v == version && e == ec)
}

/// Codewords in the symbol, counting a four-bit codeword as one. For
/// Model 1 this is the number of codeword slots the placement has, which
/// the blocks of some levels do not all fill.
pub fn total_codewords(version: SymbolVersion) -> usize {
    match version {
        SymbolVersion::Qr(v) => qr_raw_data_modules(v) / 8,
        SymbolVersion::Micro(v) => [5, 10, 17, 24][v as usize - 1],
        SymbolVersion::Rmqr(v) => RMQR_TOTAL[v as usize] as usize,
        SymbolVersion::Model1(v) => model1_codeword_slots(v),
    }
}

/// Extension patterns of a Model 1 symbol: one along the right edge for
/// every other four-row step of the placement, and one along the bottom
/// edge for every other four-column step, eight modules each. See
/// `FunctionMap::model1`.
pub fn model1_extension_patterns(version: u8) -> usize {
    2 * (version as usize / 2)
}

/// Codeword slots in a Model 1 symbol: every module that is not a finder
/// pattern with its separator, a timing pattern, format information, the
/// dark module, an extension pattern or the fixed corner, in eights. The
/// corner codeword has four modules, its high nibble being fixed at zero,
/// so the modules come to four short of a whole number of codewords.
pub fn model1_codeword_slots(version: u8) -> usize {
    let n = 17 + 4 * version as usize;
    let fixed = 3 * 64 + 2 * (n - 16) + 31 + 8 * model1_extension_patterns(version) + 4;
    (n * n - fixed + 4) / 8
}

/// Codeword slots of the placement that the blocks of this level leave
/// unused, at the end of the placement: Model 1 has equal blocks only, and
/// for some versions and levels they do not fill the symbol. zxing-cpp
/// reads past them (and refuses the versions whose level L leaves some).
/// Zero for every other family.
pub fn unused_codewords(version: SymbolVersion, ec: EcLevel) -> usize {
    match block_shape(version, ec) {
        Some(shape) => total_codewords(version) - shape.total_codewords,
        None => 0,
    }
}

/// Modules available for codewords and remainder bits in a QR symbol.
pub fn qr_raw_data_modules(version: u8) -> usize {
    let v = version as usize;
    let mut result = (16 * v + 128) * v + 64;
    if v >= 2 {
        let aligns = v / 7 + 2;
        result -= (25 * aligns - 10) * aligns - 55;
        if v >= 7 {
            result -= 36;
        }
    }
    result
}

/// Block structure, or `None` when the EC level does not exist for the version.
pub fn block_shape(version: SymbolVersion, ec: EcLevel) -> Option<BlockShape> {
    match version {
        SymbolVersion::Qr(v) => {
            let blocks = QR_BLOCKS[level_index(ec)][v as usize] as usize;
            let ec_per_block = QR_EC_PER_BLOCK[level_index(ec)][v as usize] as usize;
            let total_codewords = total_codewords(version);
            Some(BlockShape {
                total_codewords,
                data_codewords: total_codewords - blocks * ec_per_block,
                blocks,
                ec_per_block,
            })
        }
        SymbolVersion::Micro(v) => {
            let (total, data) = MICRO_CODEWORDS[micro_symbol(v, ec)?];
            Some(BlockShape {
                total_codewords: total as usize,
                data_codewords: data as usize,
                blocks: 1,
                ec_per_block: (total - data) as usize,
            })
        }
        SymbolVersion::Rmqr(v) => {
            let level = match ec {
                EcLevel::M => 0,
                EcLevel::H => 1,
                _ => return None,
            };
            let total_codewords = RMQR_TOTAL[v as usize] as usize;
            let data_codewords = RMQR_DATA[level][v as usize] as usize;
            let blocks = RMQR_BLOCKS[level][v as usize] as usize;
            Some(BlockShape {
                total_codewords,
                data_codewords,
                blocks,
                ec_per_block: (total_codewords - data_codewords) / blocks,
            })
        }
        SymbolVersion::Model1(v) => {
            let (ec_per_block, blocks, data_per_block) = MODEL1_BLOCKS[v as usize][level_index(ec)];
            let (ec_per_block, blocks, data_per_block) = (ec_per_block as usize, blocks as usize, data_per_block as usize);
            Some(BlockShape {
                total_codewords: blocks * (data_per_block + ec_per_block),
                data_codewords: blocks * data_per_block,
                blocks,
                ec_per_block,
            })
        }
    }
}

/// True when the last data codeword is four bits long (M1 and M3).
pub fn has_half_codeword(version: SymbolVersion) -> bool {
    matches!(version, SymbolVersion::Micro(1 | 3))
}

/// Error correction codewords per block that ISO/IEC 18004 (Table 9, `p`)
/// reserves against misdecodes in the smallest symbols: they detect errors
/// but must not be spent correcting them, so a block may correct at most
/// `(ec - p) / 2` wrong codewords. rMQR reserves none. The Model 1
/// standard is not to hand; its version 1 at every level, version 2 at L
/// and version 3 at L have exactly the EC codewords of Model 2's, and are
/// given Model 2's protection, and every other Model 1 symbol none.
pub fn misdecode_protection(version: SymbolVersion, ec: EcLevel) -> usize {
    match (version, ec) {
        (SymbolVersion::Qr(1) | SymbolVersion::Model1(1), EcLevel::L) => 3,
        (SymbolVersion::Qr(1) | SymbolVersion::Model1(1), EcLevel::M)
        | (SymbolVersion::Qr(2) | SymbolVersion::Model1(2), EcLevel::L) => 2,
        (SymbolVersion::Qr(1) | SymbolVersion::Model1(1), EcLevel::Q | EcLevel::H)
        | (SymbolVersion::Qr(3) | SymbolVersion::Model1(3), EcLevel::L) => 1,
        (SymbolVersion::Micro(1), _) => 2,
        (SymbolVersion::Micro(2), EcLevel::L) => 3,
        (SymbolVersion::Micro(2), EcLevel::M) | (SymbolVersion::Micro(3 | 4), EcLevel::L) => 2,
        _ => 0,
    }
}

/// Centre coordinates of QR alignment patterns along one axis.
pub fn qr_alignment_centers(version: u8) -> Vec<usize> {
    let v = version as usize;
    if v == 1 {
        return Vec::new();
    }
    let count = v / 7 + 2;
    let step = if v == 32 { 26 } else { (v * 4 + count * 2 + 1) / (count * 2 - 2) * 2 };
    let size = 17 + 4 * v;
    let mut centers: Vec<usize> = (0..count - 1).map(|i| size - 7 - i * step).collect();
    centers.push(6);
    centers.reverse();
    centers
}

/// Centre columns of rMQR alignment patterns for a symbol width.
pub fn rmqr_alignment_centers(width: usize) -> &'static [usize] {
    match width {
        27 => &[],
        43 => &[21],
        59 => &[19, 39],
        77 => &[25, 51],
        99 => &[23, 49, 75],
        139 => &[27, 55, 83, 111],
        _ => panic!("not an rMQR width: {width}"),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    Numeric,
    Alphanumeric,
    Byte,
    Kanji,
    /// Chinese characters in GB 2312, from GB/T 18284. QR Code only.
    Hanzi,
    Eci,
    StructuredAppend,
    Fnc1First,
    Fnc1Second,
    Terminator,
}

impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Mode::Numeric => "Numeric",
            Mode::Alphanumeric => "Alphanumeric",
            Mode::Byte => "Byte",
            Mode::Kanji => "Kanji",
            Mode::Hanzi => "Hanzi",
            Mode::Eci => "ECI",
            Mode::StructuredAppend => "Structured append",
            Mode::Fnc1First => "FNC1 (first position)",
            Mode::Fnc1Second => "FNC1 (second position)",
            Mode::Terminator => "Terminator",
        }
    }
}

/// Everything the bitstream parser needs to know about a version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamSpec {
    pub version: SymbolVersion,
    /// Width of the mode indicator; zero for M1, which is numeric only.
    pub mode_bits: u8,
    pub terminator_bits: u8,
    /// Bits before the first mode indicator that carry no data: the four
    /// zero bits a Model 1 stream begins with, the high nibble of its
    /// corner codeword, which zxing-cpp drops before parsing. Zero for
    /// every other family.
    pub preamble_bits: u8,
}

impl StreamSpec {
    pub fn new(version: SymbolVersion) -> Self {
        let (mode_bits, terminator_bits, preamble_bits) = match version {
            SymbolVersion::Qr(_) => (4, 4, 0),
            SymbolVersion::Micro(v) => (v - 1, 2 * v + 1, 0),
            SymbolVersion::Rmqr(_) => (3, 3, 0),
            SymbolVersion::Model1(_) => (4, 4, 4),
        };
        Self { version, mode_bits, terminator_bits, preamble_bits }
    }

    /// Meaning of a mode indicator value, if it has one. Model 1 is read
    /// with Model 2's indicators, as zxing-cpp reads it.
    pub fn mode(&self, indicator: u32) -> Option<Mode> {
        match self.version.family() {
            Family::Qr | Family::Model1 => match indicator {
                0 => Some(Mode::Terminator),
                1 => Some(Mode::Numeric),
                2 => Some(Mode::Alphanumeric),
                3 => Some(Mode::StructuredAppend),
                4 => Some(Mode::Byte),
                5 => Some(Mode::Fnc1First),
                7 => Some(Mode::Eci),
                8 => Some(Mode::Kanji),
                9 => Some(Mode::Fnc1Second),
                13 => Some(Mode::Hanzi),
                _ => None,
            },
            // Micro QR has no terminator indicator: the terminator is a run of
            // zeros that reads as a numeric segment of length zero.
            Family::Micro => [Mode::Numeric, Mode::Alphanumeric, Mode::Byte, Mode::Kanji]
                .get(indicator as usize)
                .copied()
                .filter(|&m| self.count_bits(m).is_some()),
            Family::Rmqr => match indicator {
                0 => Some(Mode::Terminator),
                1 => Some(Mode::Numeric),
                2 => Some(Mode::Alphanumeric),
                3 => Some(Mode::Byte),
                4 => Some(Mode::Kanji),
                5 => Some(Mode::Fnc1First),
                6 => Some(Mode::Fnc1Second),
                7 => Some(Mode::Eci),
                _ => None,
            },
        }
    }

    /// Width of the character count for a data mode, or `None` when the mode
    /// carries no count or is unavailable in this version.
    pub fn count_bits(&self, mode: Mode) -> Option<u8> {
        let column = match mode {
            Mode::Numeric => 0,
            Mode::Alphanumeric => 1,
            Mode::Byte => 2,
            Mode::Kanji => 3,
            Mode::Hanzi => 4,
            _ => return None,
        };
        match self.version {
            // Model 1 takes Model 2's widths for the same version number,
            // as zxing-cpp reads it.
            SymbolVersion::Qr(v) | SymbolVersion::Model1(v) => {
                let group = match v {
                    1..=9 => 0,
                    10..=26 => 1,
                    _ => 2,
                };
                Some([[10, 12, 14], [9, 11, 13], [8, 16, 16], [8, 10, 12], [8, 10, 12]][column][group])
            }
            _ if mode == Mode::Hanzi => None,
            SymbolVersion::Micro(v) => {
                let bits = [[3, 4, 5, 6], [0, 3, 4, 5], [0, 0, 4, 5], [0, 0, 3, 4]][column][v as usize - 1];
                (bits > 0).then_some(bits)
            }
            SymbolVersion::Rmqr(v) => Some(RMQR_COUNT_BITS[column][v as usize]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qr_alignment_positions() {
        assert!(qr_alignment_centers(1).is_empty());
        assert_eq!(qr_alignment_centers(2), [6, 18]);
        assert_eq!(qr_alignment_centers(7), [6, 22, 38]);
        assert_eq!(qr_alignment_centers(32), [6, 34, 60, 86, 112, 138]);
        assert_eq!(qr_alignment_centers(36), [6, 24, 50, 76, 102, 128, 154]);
        assert_eq!(qr_alignment_centers(40), [6, 30, 58, 86, 114, 142, 170]);
    }

    #[test]
    fn qr_capacities() {
        assert_eq!(total_codewords(SymbolVersion::Qr(1)), 26);
        assert_eq!(total_codewords(SymbolVersion::Qr(7)), 196);
        assert_eq!(total_codewords(SymbolVersion::Qr(40)), 3706);
        let data = |v, ec| block_shape(SymbolVersion::Qr(v), ec).unwrap().data_codewords;
        assert_eq!(data(1, EcLevel::L), 19);
        assert_eq!(data(1, EcLevel::H), 9);
        assert_eq!(data(5, EcLevel::Q), 62);
        assert_eq!(data(40, EcLevel::L), 2956);
        assert_eq!(data(40, EcLevel::H), 1276);
    }

    #[test]
    fn block_lengths() {
        // 5-Q: two blocks of 15 data codewords, two of 16.
        let shape = block_shape(SymbolVersion::Qr(5), EcLevel::Q).unwrap();
        assert_eq!((0..4).map(|b| shape.data_len(b)).collect::<Vec<_>>(), [15, 15, 16, 16]);
        // R13x139-H: two blocks of 13, two of 14.
        let shape = block_shape(SymbolVersion::Rmqr(21), EcLevel::H).unwrap();
        assert_eq!(shape.ec_per_block, 28);
        assert_eq!((0..4).map(|b| shape.data_len(b)).collect::<Vec<_>>(), [13, 13, 14, 14]);
    }

    #[test]
    fn misdecode_protection_leaves_a_whole_number_of_correctable_errors() {
        // Table 9: the capacity after protection, in codewords.
        let capacity = |version, ec| {
            let shape = block_shape(version, ec).unwrap();
            (shape.ec_per_block - misdecode_protection(version, ec)) / 2
        };
        assert_eq!(capacity(SymbolVersion::Qr(1), EcLevel::L), 2);
        assert_eq!(capacity(SymbolVersion::Qr(1), EcLevel::M), 4);
        assert_eq!(capacity(SymbolVersion::Qr(1), EcLevel::Q), 6);
        assert_eq!(capacity(SymbolVersion::Qr(1), EcLevel::H), 8);
        assert_eq!(capacity(SymbolVersion::Qr(2), EcLevel::L), 4);
        assert_eq!(capacity(SymbolVersion::Qr(3), EcLevel::L), 7);
        assert_eq!(capacity(SymbolVersion::Micro(1), EcLevel::L), 0);
        assert_eq!(capacity(SymbolVersion::Micro(2), EcLevel::L), 1);
        assert_eq!(capacity(SymbolVersion::Micro(2), EcLevel::M), 2);
        assert_eq!(capacity(SymbolVersion::Micro(3), EcLevel::L), 2);
        assert_eq!(capacity(SymbolVersion::Micro(3), EcLevel::M), 4);
        assert_eq!(capacity(SymbolVersion::Micro(4), EcLevel::L), 3);
        assert_eq!(capacity(SymbolVersion::Micro(4), EcLevel::Q), 7);
        for family in Family::ALL {
            for version in family.versions() {
                for &ec in version.ec_levels() {
                    let p = misdecode_protection(version, ec);
                    assert!(p <= block_shape(version, ec).unwrap().ec_per_block, "{version} {ec:?}");
                }
            }
        }
        assert_eq!(misdecode_protection(SymbolVersion::Qr(4), EcLevel::L), 0);
        assert_eq!(misdecode_protection(SymbolVersion::Rmqr(0), EcLevel::M), 0);
        assert_eq!(capacity(SymbolVersion::Model1(1), EcLevel::L), 2);
        assert_eq!(capacity(SymbolVersion::Model1(2), EcLevel::L), 4);
        assert_eq!(capacity(SymbolVersion::Model1(3), EcLevel::L), 7);
        assert_eq!(misdecode_protection(SymbolVersion::Model1(2), EcLevel::H), 0);
    }

    #[test]
    fn model1_capacities() {
        // Codeword slots: the version 2 sample has 46, and the zxing-cpp
        // table fills them exactly at every level.
        assert_eq!(total_codewords(SymbolVersion::Model1(1)), 26);
        assert_eq!(total_codewords(SymbolVersion::Model1(2)), 46);
        assert_eq!(total_codewords(SymbolVersion::Model1(14)), 610);
        for v in 1..=14u8 {
            let version = SymbolVersion::Model1(v);
            for &ec in version.ec_levels() {
                let shape = block_shape(version, ec).unwrap();
                assert!(shape.total_codewords <= total_codewords(version), "{version} {ec:?}");
                assert_eq!(shape.total_codewords + unused_codewords(version, ec), total_codewords(version));
                // Equal blocks only.
                assert_eq!(shape.data_codewords % shape.blocks, 0, "{version} {ec:?}");
            }
        }
        // Versions 1 to 6 and 9 fill every slot; the rest leave a few at
        // some levels, most of all 13, whose blocks leave two at every level.
        assert_eq!(unused_codewords(SymbolVersion::Model1(7), EcLevel::H), 2);
        assert_eq!(unused_codewords(SymbolVersion::Model1(7), EcLevel::L), 0);
        assert_eq!(unused_codewords(SymbolVersion::Model1(13), EcLevel::L), 2);
        assert_eq!(unused_codewords(SymbolVersion::Model1(14), EcLevel::H), 4);
        assert_eq!(unused_codewords(SymbolVersion::Qr(7), EcLevel::H), 0);
        assert_eq!(block_shape(SymbolVersion::Model1(2), EcLevel::M).unwrap().data_codewords, 30);
    }

    #[test]
    fn every_block_fits_the_field() {
        for family in Family::ALL {
            for version in family.versions() {
                for &ec in version.ec_levels() {
                    let shape = block_shape(version, ec).unwrap();
                    assert_eq!(
                        (shape.total_codewords - shape.data_codewords) % shape.blocks,
                        0,
                        "{version} {ec:?}"
                    );
                    for b in 0..shape.blocks {
                        assert!(shape.data_len(b) + shape.ec_per_block <= 255);
                        assert!(shape.data_len(b) >= 1);
                    }
                    let sum: usize = (0..shape.blocks).map(|b| shape.data_len(b)).sum();
                    assert_eq!(sum, shape.data_codewords);
                }
            }
        }
        assert!(block_shape(SymbolVersion::Micro(1), EcLevel::M).is_none());
        assert!(block_shape(SymbolVersion::Rmqr(0), EcLevel::L).is_none());
    }

    #[test]
    fn stream_specs() {
        let m1 = StreamSpec::new(SymbolVersion::Micro(1));
        assert_eq!((m1.mode_bits, m1.terminator_bits), (0, 3));
        assert_eq!(m1.count_bits(Mode::Numeric), Some(3));
        assert_eq!(m1.count_bits(Mode::Byte), None);
        let m4 = StreamSpec::new(SymbolVersion::Micro(4));
        assert_eq!((m4.mode_bits, m4.terminator_bits), (3, 9));
        assert_eq!(m4.mode(3), Some(Mode::Kanji));
        let m2 = StreamSpec::new(SymbolVersion::Micro(2));
        assert_eq!(m2.mode(1), Some(Mode::Alphanumeric));
        let qr = StreamSpec::new(SymbolVersion::Qr(27));
        assert_eq!(qr.count_bits(Mode::Numeric), Some(14));
        assert_eq!(qr.mode(7), Some(Mode::Eci));
        assert_eq!(qr.mode(6), None);
        assert_eq!(qr.mode(13), Some(Mode::Hanzi));
        assert_eq!(qr.count_bits(Mode::Hanzi), Some(12));
        assert_eq!(StreamSpec::new(SymbolVersion::Qr(5)).count_bits(Mode::Hanzi), Some(8));
        assert_eq!(m4.count_bits(Mode::Hanzi), None);
        let m1 = StreamSpec::new(SymbolVersion::Model1(12));
        assert_eq!((m1.mode_bits, m1.terminator_bits, m1.preamble_bits), (4, 4, 4));
        assert_eq!(m1.count_bits(Mode::Byte), Some(16));
        assert_eq!(StreamSpec::new(SymbolVersion::Model1(2)).count_bits(Mode::Byte), Some(8));
        assert_eq!(m1.mode(4), Some(Mode::Byte));
        assert_eq!(qr.preamble_bits, 0);
        let r = StreamSpec::new(SymbolVersion::Rmqr(31));
        assert_eq!(r.count_bits(Mode::Kanji), Some(7));
        assert_eq!(r.mode(4), Some(Mode::Kanji));
        assert_eq!(r.count_bits(Mode::Hanzi), None);
        assert_eq!(r.mode(13), None);
    }
}
