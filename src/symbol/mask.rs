//! Data mask predicates.

use super::SymbolVersion;

/// QR mask pattern `pattern` (0..=7) at column `x`, row `y`. True means the
/// module is inverted.
// Written as the standard writes the conditions.
#[allow(clippy::manual_is_multiple_of)]
pub fn qr_mask(pattern: u8, x: usize, y: usize) -> bool {
    match pattern {
        0 => (x + y) % 2 == 0,
        1 => y % 2 == 0,
        2 => x % 3 == 0,
        3 => (x + y) % 3 == 0,
        4 => (y / 2 + x / 3) % 2 == 0,
        5 => (x * y) % 2 + (x * y) % 3 == 0,
        6 => ((x * y) % 2 + (x * y) % 3) % 2 == 0,
        7 => ((x + y) % 2 + (x * y) % 3) % 2 == 0,
        _ => panic!("QR mask pattern out of range: {pattern}"),
    }
}

/// Mask bit for a symbol's own numbering: QR uses patterns 0..=7 directly
/// (Model 1 the same eight, as zxing-cpp applies them), Micro QR numbers
/// four of them 0..=3, and rMQR has a single fixed mask.
pub fn mask_bit(version: SymbolVersion, mask: u8, x: usize, y: usize) -> bool {
    match version {
        SymbolVersion::Qr(_) | SymbolVersion::Model1(_) => qr_mask(mask, x, y),
        SymbolVersion::Micro(_) => qr_mask([1, 4, 6, 7][mask as usize], x, y),
        SymbolVersion::Rmqr(_) => qr_mask(4, x, y),
    }
}
