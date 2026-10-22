//! Reading codewords off the grid, and pins: values the user or a recovery
//! step asserts for specific codeword bits.

use std::collections::BTreeMap;

use super::rs::CwRead;
use crate::symbol::grid::ModuleGrid;
use crate::symbol::layout::{CwId, SymbolLayout};
use crate::symbol::mask::mask_bit;
use crate::symbol::tables;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PinSource {
    /// Typed into the codeword inspector.
    Manual,
    /// Bits of text the user knows the message to hold, placed by
    /// [`super::known`].
    Text,
    /// A pad codeword placed by padding recovery.
    Padding,
    /// Terminator and zero-fill bits placed by padding recovery.
    Terminator,
    /// Header bits placed by padding recovery: the one reading of a header
    /// with unreadable bits under which the stream parses.
    Header,
}

/// Asserted bits of one codeword. Bits outside `mask` are left as read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Pin {
    pub mask: u8,
    pub value: u8,
    pub source: PinSource,
    /// Ways the pin was picked among, keeping the best, the picks of the
    /// pins in force when it was counted in: the beginnings tried, the
    /// passes of padding recovery. A search the pin reaches is a fresh
    /// chance under each way, and is charged as one of them
    /// ([`super::pipeline::DecodeReport::pin_tries`]). One for a pin
    /// placed by hand, which nothing picked. The ways were once one count
    /// for every pin in force, and the latest pick was charged to blocks
    /// that held only pins picked before it.
    pub among: usize,
}

impl Pin {
    /// Bits `value` asserted where `mask` is set, picked among nothing.
    pub fn new(mask: u8, value: u8, source: PinSource) -> Self {
        Self { mask, value, source, among: 1 }
    }

    pub fn full(value: u8, source: PinSource) -> Self {
        Self::new(0xFF, value, source)
    }

    /// Whether the two assert the same bits, from the same source,
    /// however many ways each was picked among.
    pub fn asserts_as(&self, other: &Pin) -> bool {
        (self.mask, self.value, self.source) == (other.mask, other.value, other.source)
    }
}

/// The most ways any of `pins` was picked among ([`Pin::among`]), one
/// where there are none: what a pick made under them compounds.
pub fn picked_among(pins: &Pins) -> usize {
    pins.values().map(|pin| pin.among.max(1)).max().unwrap_or(1)
}

impl PinSource {
    /// The user's own assertion, by the codeword or as known text: what
    /// padding recovery neither replaces nor takes away with its own.
    pub fn is_by_hand(self) -> bool {
        matches!(self, PinSource::Manual | PinSource::Text)
    }
}

pub type Pins = BTreeMap<CwId, Pin>;

/// Bits of a codeword that exist in the symbol: the top four for the short
/// final data codeword of M1 and M3, the bottom four for the corner
/// codeword of Model 1, all eight otherwise.
pub fn codeword_bit_mask(layout: &SymbolLayout, cw: CwId) -> u8 {
    layout.codewords[cw].module_mask()
}

/// Codewords in placement order, unmasked, exactly as the modules show them.
pub fn read_codewords(grid: &ModuleGrid, layout: &SymbolLayout, mask: u8) -> Vec<CwRead> {
    let version = layout.version();
    (0..layout.codewords.len())
        .map(|cw| {
            let mut read = CwRead::default();
            let offset = layout.codewords[cw].bit_offset as usize;
            for (i, pos) in layout.codeword_modules(cw).iter().enumerate() {
                let (x, y) = (pos.x as usize, pos.y as usize);
                if let Some(dark) = grid.get(x, y).as_dark() {
                    let bit = 0x80u8 >> (offset + i);
                    read.known |= bit;
                    if dark != mask_bit(version, mask, x, y) {
                        read.value |= bit;
                    }
                }
            }
            read
        })
        .collect()
}

/// Overlay pins on codewords as read. A four-bit codeword is widened to a
/// byte with its missing nibble known as zero, which is how it enters RS:
/// ISO/IEC 18004 has the error correction of an M1 or M3 symbol computed
/// over the final data codeword's four bits followed by four zeros. Those
/// zeros are not pinned. An encoder in the field computes its error
/// correction over another filling of the nibble, all ones, and its
/// symbols then solve with the widened codeword as one wrong codeword, as
/// zxing-cpp and BoofCV, which pin nothing, solve them; pinned, the only
/// solution was refused. The corner codeword of Model 1 has its high
/// nibble fixed at zero by the symbol's own layout, and stays pinned.
pub fn apply_pins(layout: &SymbolLayout, reads: &[CwRead], pins: &Pins) -> Vec<CwRead> {
    let half = tables::has_half_codeword(layout.version());
    reads
        .iter()
        .enumerate()
        .map(|(cw, &read)| {
            let exists = codeword_bit_mask(layout, cw);
            let mut out = CwRead { value: read.value, known: read.known | !exists, pinned: if half { 0 } else { !exists } };
            if let Some(pin) = pins.get(&cw) {
                let mask = pin.mask & exists;
                out.value = (out.value & !mask) | (pin.value & mask);
                out.known |= mask;
                out.pinned |= mask;
            }
            out
        })
        .collect()
}
