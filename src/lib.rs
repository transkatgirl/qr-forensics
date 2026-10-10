//! Decode core for damaged QR Code, Micro QR Code, rMQR and QR Code Model 1
//! symbols.
//!
//! The pipeline works from a tri-state module grid (dark, light, unknown) and
//! keeps track of what it knows at every stage, so partial results survive
//! when error correction fails.

pub mod decode;
pub mod imaging;
pub mod symbol;
pub mod work;
