//! Symbol geometry: which module does what, for every supported version.

pub mod function_map;
pub mod grid;
pub mod layout;
pub mod mask;
pub mod tables;

use std::fmt;

use grid::Mirror;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Family {
    /// QR Code Model 2, the symbol ISO/IEC 18004 describes.
    Qr,
    Micro,
    Rmqr,
    /// QR Code Model 1, the original 1994 symbol (AIM ITS 97/001, JIS X
    /// 0510:1999): the same finder and timing patterns as Model 2, no
    /// alignment patterns, extension patterns along the right and bottom
    /// edges, and its own codeword placement and block tables. zxing-cpp
    /// reads it; most readers do not.
    Model1,
}

impl Family {
    pub const ALL: [Family; 4] = [Family::Qr, Family::Micro, Family::Rmqr, Family::Model1];

    /// The families a search of a photo looks for. Model 1 is left out:
    /// its finder and timing patterns are those of Model 2, so every
    /// Model 2 symbol would be found a second time as a Model 1 one, and
    /// a version 1 symbol of either reads the same as the other until
    /// its format word is read. A Model 1 symbol found as Model 2 is
    /// told apart by its format word, which the inspector reports, and
    /// the grid is then reinterpreted as Model 1 at the same size. QR
    /// Code comes first: the finders of a QR Code found are not then
    /// taken for those of the families with one finder.
    pub const SEARCHED: [Family; 3] = [Family::Qr, Family::Micro, Family::Rmqr];

    pub fn label(self) -> &'static str {
        match self {
            Family::Qr => "QR Code",
            Family::Micro => "Micro QR",
            Family::Rmqr => "rMQR",
            Family::Model1 => "QR Model 1",
        }
    }

    pub fn versions(self) -> Vec<SymbolVersion> {
        match self {
            Family::Qr => (1..=40).map(SymbolVersion::Qr).collect(),
            Family::Micro => (1..=4).map(SymbolVersion::Micro).collect(),
            Family::Rmqr => (0..32).map(SymbolVersion::Rmqr).collect(),
            Family::Model1 => (1..=14).map(SymbolVersion::Model1).collect(),
        }
    }

    /// Three finder patterns in the corners, as both QR Code models have.
    pub fn has_three_finders(self) -> bool {
        matches!(self, Family::Qr | Family::Model1)
    }

    /// The other QR Code model, whose symbols are the same size and have
    /// the same finder and timing patterns; `None` for the small symbols.
    pub fn other_model(self) -> Option<Family> {
        match self {
            Family::Qr => Some(Family::Model1),
            Family::Model1 => Some(Family::Qr),
            Family::Micro | Family::Rmqr => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EcLevel {
    L,
    M,
    Q,
    H,
}

impl EcLevel {
    pub fn letter(self) -> &'static str {
        match self {
            EcLevel::L => "L",
            EcLevel::M => "M",
            EcLevel::Q => "Q",
            EcLevel::H => "H",
        }
    }
}

/// A symbol version. `Qr` holds 1..=40, `Micro` holds 1..=4 (M1..M4),
/// `Rmqr` holds the version indicator 0..=31 (R7x43..R17x139) and
/// `Model1` holds 1..=14, the versions QR Code Model 1 had.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SymbolVersion {
    Qr(u8),
    Micro(u8),
    Rmqr(u8),
    Model1(u8),
}

impl SymbolVersion {
    pub fn family(self) -> Family {
        match self {
            SymbolVersion::Qr(_) => Family::Qr,
            SymbolVersion::Micro(_) => Family::Micro,
            SymbolVersion::Rmqr(_) => Family::Rmqr,
            SymbolVersion::Model1(_) => Family::Model1,
        }
    }

    pub fn is_valid(self) -> bool {
        match self {
            SymbolVersion::Qr(v) => (1..=40).contains(&v),
            SymbolVersion::Micro(v) => (1..=4).contains(&v),
            SymbolVersion::Rmqr(v) => v < 32,
            SymbolVersion::Model1(v) => (1..=14).contains(&v),
        }
    }

    pub fn width(self) -> usize {
        match self {
            SymbolVersion::Qr(v) | SymbolVersion::Model1(v) => 17 + 4 * v as usize,
            SymbolVersion::Micro(v) => 9 + 2 * v as usize,
            SymbolVersion::Rmqr(v) => tables::RMQR_SIZES[v as usize].1,
        }
    }

    /// The same symbol size under the other QR Code model: a Model 2 grid
    /// read as Model 1 or the reverse. `None` for the small symbols.
    pub fn other_model(self) -> Option<SymbolVersion> {
        match self {
            SymbolVersion::Qr(v) if v <= 14 => Some(SymbolVersion::Model1(v)),
            SymbolVersion::Model1(v) => Some(SymbolVersion::Qr(v)),
            _ => None,
        }
    }

    /// QR version information, the (18,6) word beside two finders, exists
    /// from Model 2 version 7 up; Model 1 and the small symbols have none.
    pub fn has_version_info(self) -> bool {
        matches!(self, SymbolVersion::Qr(v) if v >= 7)
    }

    pub fn height(self) -> usize {
        match self {
            SymbolVersion::Rmqr(v) => tables::RMQR_SIZES[v as usize].0,
            _ => self.width(),
        }
    }

    pub fn from_size(family: Family, width: usize, height: usize) -> Option<Self> {
        family.versions().into_iter().find(|v| v.width() == width && v.height() == height)
    }

    /// EC levels that exist for this version.
    pub fn ec_levels(self) -> &'static [EcLevel] {
        match self {
            SymbolVersion::Qr(_) | SymbolVersion::Model1(_) => &[EcLevel::L, EcLevel::M, EcLevel::Q, EcLevel::H],
            SymbolVersion::Micro(1) => &[EcLevel::L],
            SymbolVersion::Micro(4) => &[EcLevel::L, EcLevel::M, EcLevel::Q],
            SymbolVersion::Micro(_) => &[EcLevel::L, EcLevel::M],
            SymbolVersion::Rmqr(_) => &[EcLevel::M, EcLevel::H],
        }
    }

    pub fn mask_count(self) -> u8 {
        match self {
            SymbolVersion::Qr(_) | SymbolVersion::Model1(_) => 8,
            SymbolVersion::Micro(_) => 4,
            SymbolVersion::Rmqr(_) => 1,
        }
    }

    /// The reflections a symbol of this family can be seen in while its
    /// finder pattern stays in the corner the grid expects: a mirrored
    /// square symbol reads as its transpose, a mirrored rMQR as a flip.
    pub fn mirrors(self) -> &'static [Mirror] {
        match self {
            SymbolVersion::Rmqr(_) => &[Mirror::Horizontal, Mirror::Vertical],
            _ => &[Mirror::Transpose],
        }
    }

    /// How the EC level is written for this version. M1 has error detection
    /// only, which the tables store under `L`.
    pub fn ec_label(self, ec: EcLevel) -> &'static str {
        match self {
            SymbolVersion::Micro(1) => "detect",
            _ => ec.letter(),
        }
    }
}

impl fmt::Display for SymbolVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymbolVersion::Qr(v) | SymbolVersion::Model1(v) => write!(f, "{v}"),
            SymbolVersion::Micro(v) => write!(f, "M{v}"),
            SymbolVersion::Rmqr(_) => write!(f, "R{}x{}", self.height(), self.width()),
        }
    }
}

/// Module coordinate: `x` is the column, `y` the row, origin top-left.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pos {
    pub x: u16,
    pub y: u16,
}

impl Pos {
    pub const fn new(x: usize, y: usize) -> Self {
        Self { x: x as u16, y: y as u16 }
    }
}
