//! The decode pipeline, from module grid to text, with every intermediate
//! stage kept for inspection.

pub mod abp;
pub mod bch;
pub mod bitstream;
pub mod clustering;
pub mod codewords;
pub mod evidence;
pub mod gf256;
pub mod known;
pub mod list;
pub mod padding;
pub mod pipeline;
pub mod repair;
pub mod rs;
pub mod sequence;
pub mod text;
