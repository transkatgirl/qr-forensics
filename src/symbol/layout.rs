//! Codeword layout for one (version, EC level): which path modules make up
//! each codeword, and how codewords interleave into RS blocks.

use std::sync::Arc;

use super::function_map::{FunctionMap, Role};
use super::tables::{self, StreamSpec};
use super::{EcLevel, Pos, SymbolVersion};

/// Codeword index in placement (interleaved) order.
pub type CwId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodewordSpan {
    /// First module of the codeword in [`FunctionMap::data_path`]; the
    /// codeword's most significant bit, or its bit `7 - bit_offset`.
    pub path_start: u32,
    /// Modules the codeword has: eight, except for the four-bit final data
    /// codeword of M1 and M3 and the corner codeword of Model 1.
    pub nbits: u8,
    /// Bits of the codeword before its first module, counted from the
    /// most significant: four for the corner codeword of Model 1, whose
    /// high nibble is fixed at zero and has no modules; zero otherwise.
    pub bit_offset: u8,
    pub block: u16,
    /// Position within the block, data codewords first.
    pub index_in_block: u16,
    pub is_data: bool,
}

impl CodewordSpan {
    /// Bits the codeword contributes to the data stream: its modules and
    /// the fixed bits before them.
    pub fn stream_bits(&self) -> usize {
        (self.bit_offset + self.nbits) as usize
    }

    /// Mask of the bits that have modules, most significant bit 0x80.
    pub fn module_mask(&self) -> u8 {
        (((1u16 << self.nbits) - 1) << (8 - self.bit_offset - self.nbits)) as u8
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockSpec {
    pub data_len: usize,
    pub ec_len: usize,
    /// EC codewords the standard reserves against misdecodes: a solution
    /// must leave at least this many unused to be trusted.
    pub reserved: usize,
    /// Codewords in RS order: data, then error correction.
    pub codewords: Vec<CwId>,
    /// Index of this block's first data codeword in [`SymbolLayout::data_stream`].
    pub stream_offset: usize,
}

#[derive(Clone, Debug)]
pub struct SymbolLayout {
    pub func: Arc<FunctionMap>,
    pub ec: EcLevel,
    pub codewords: Vec<CodewordSpan>,
    pub blocks: Vec<BlockSpec>,
    /// Data codewords in message order: all of block 0, then block 1, and so on.
    pub data_stream: Vec<CwId>,
    /// Length of the data bit stream.
    pub data_bits: usize,
    pub stream: StreamSpec,
    /// Codeword slots at the end of the placement that belong to no block
    /// at this EC level; Model 1 leaves some at some levels. Their modules
    /// hold no codeword.
    pub unused_codewords: usize,
}

impl SymbolLayout {
    /// Returns `None` when the EC level does not exist for the version.
    pub fn new(func: Arc<FunctionMap>, ec: EcLevel) -> Option<Self> {
        let version = func.version;
        let shape = tables::block_shape(version, ec)?;
        let half = tables::has_half_codeword(version);
        let reserved = tables::misdecode_protection(version, ec);

        let mut blocks: Vec<BlockSpec> = (0..shape.blocks)
            .map(|b| BlockSpec {
                data_len: shape.data_len(b),
                ec_len: shape.ec_per_block,
                reserved,
                codewords: Vec::new(),
                stream_offset: 0,
            })
            .collect();
        let mut offset = 0;
        for block in &mut blocks {
            block.stream_offset = offset;
            offset += block.data_len;
            block.codewords = vec![0; block.data_len + block.ec_len];
        }

        // Interleave: one data codeword from each block in turn (short blocks
        // drop out first), then the same for EC codewords. QR Code Model 1
        // does not interleave: its blocks, all of one length, lie back to
        // back, every data block and then every EC block, as zxing-cpp
        // reads them since its fix for issue #940, whose sample symbol,
        // `tests/blackbox/cpp_qrcode-2_#940.png`, decodes only that way.
        let mut order = Vec::with_capacity(shape.total_codewords);
        let longest = blocks.iter().map(|b| b.data_len).max().unwrap_or(0);
        if matches!(version, SymbolVersion::Model1(_)) {
            for (b, block) in blocks.iter().enumerate() {
                for i in 0..block.data_len {
                    order.push((b, i, true));
                }
            }
            for (b, block) in blocks.iter().enumerate() {
                for i in 0..shape.ec_per_block {
                    order.push((b, block.data_len + i, false));
                }
            }
        } else {
            for i in 0..longest {
                for (b, block) in blocks.iter().enumerate() {
                    if i < block.data_len {
                        order.push((b, i, true));
                    }
                }
            }
            for i in 0..shape.ec_per_block {
                for (b, block) in blocks.iter().enumerate() {
                    order.push((b, block.data_len + i, false));
                }
            }
        }
        assert_eq!(order.len(), shape.total_codewords);

        let mut codewords = Vec::with_capacity(order.len());
        let mut path = 0u32;
        let corner = matches!(version, SymbolVersion::Model1(_));
        for (id, &(b, index, is_data)) in order.iter().enumerate() {
            let (nbits, bit_offset) = if half && is_data && index == shape.data_codewords - 1 {
                (4, 0)
            } else if corner && id == 0 {
                // The first codeword placed is the corner one, with its
                // high nibble fixed at zero.
                (4, 4)
            } else {
                (8, 0)
            };
            codewords.push(CodewordSpan {
                path_start: path,
                nbits,
                bit_offset,
                block: b as u16,
                index_in_block: index as u16,
                is_data,
            });
            blocks[b].codewords[index] = id;
            path += nbits as u32;
        }
        let unused_codewords = tables::unused_codewords(version, ec);
        assert_eq!(path as usize + 8 * unused_codewords, func.codeword_bits(), "{version} {ec:?}");

        let data_stream: Vec<CwId> =
            blocks.iter().flat_map(|b| b.codewords[..b.data_len].iter().copied()).collect();
        let data_bits = data_stream.iter().map(|&cw| codewords[cw].stream_bits()).sum();

        Some(Self {
            func,
            ec,
            codewords,
            blocks,
            data_stream,
            data_bits,
            stream: StreamSpec::new(version),
            unused_codewords,
        })
    }

    pub fn version(&self) -> SymbolVersion {
        self.func.version
    }

    /// Modules of a codeword, most significant bit first.
    pub fn codeword_modules(&self, cw: CwId) -> &[Pos] {
        let span = &self.codewords[cw];
        let start = span.path_start as usize;
        &self.func.data_path[start..start + span.nbits as usize]
    }

    /// The codeword a module belongs to and the bit it holds, counted from
    /// the most significant bit. `None` for a module of a codeword slot
    /// this EC level leaves unused.
    pub fn module_codeword(&self, x: usize, y: usize) -> Option<(CwId, u8)> {
        let Role::Data { path } = self.func.role(x, y) else {
            return None;
        };
        let cw = self.codewords.partition_point(|span| span.path_start + span.nbits as u32 <= path);
        let span = self.codewords.get(cw)?;
        Some((cw, (path - span.path_start) as u8 + span.bit_offset))
    }

    /// Modules of the codeword slots this EC level leaves unused.
    pub fn unused_modules(&self) -> &[Pos] {
        let used = self.codewords.last().map_or(0, |span| span.path_start as usize + span.nbits as usize);
        &self.func.data_path[used..self.func.codeword_bits()]
    }

    /// Position of a data codeword in the message stream.
    pub fn stream_index(&self, cw: CwId) -> Option<usize> {
        let span = &self.codewords[cw];
        span.is_data
            .then(|| self.blocks[span.block as usize].stream_offset + span.index_in_block as usize)
    }

    /// The codeword and bit (from the most significant) holding a stream bit.
    pub fn stream_bit_location(&self, bit: usize) -> Option<(CwId, u8)> {
        (bit < self.data_bits).then(|| (self.data_stream[bit / 8], (bit % 8) as u8))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::Family;

    fn layout(version: SymbolVersion, ec: EcLevel) -> SymbolLayout {
        SymbolLayout::new(Arc::new(FunctionMap::new(version)), ec).unwrap()
    }

    #[test]
    fn every_layout_builds_and_is_consistent() {
        for family in Family::ALL {
            for version in family.versions() {
                let func = Arc::new(FunctionMap::new(version));
                for &ec in version.ec_levels() {
                    let l = SymbolLayout::new(func.clone(), ec).unwrap();
                    let mut seen = vec![false; l.codewords.len()];
                    for (b, block) in l.blocks.iter().enumerate() {
                        for (i, &cw) in block.codewords.iter().enumerate() {
                            assert!(!seen[cw]);
                            seen[cw] = true;
                            let span = l.codewords[cw];
                            assert_eq!((span.block as usize, span.index_in_block as usize), (b, i));
                            assert_eq!(span.is_data, i < block.data_len);
                        }
                    }
                    assert!(seen.iter().all(|&s| s));
                    for (j, &cw) in l.data_stream.iter().enumerate() {
                        assert_eq!(l.stream_index(cw), Some(j));
                    }
                }
            }
        }
    }

    #[test]
    fn interleave_order_5q() {
        let l = layout(SymbolVersion::Qr(5), EcLevel::Q);
        // First four codewords are the first data codeword of each block.
        for b in 0..4 {
            assert_eq!(l.blocks[b].codewords[0], b);
        }
        // Data codeword 15 exists only in the two long blocks.
        assert_eq!(l.blocks[2].codewords[15], 60);
        assert_eq!(l.blocks[3].codewords[15], 61);
        assert_eq!(l.blocks[0].codewords[15], 62);
        assert_eq!(l.data_stream.len(), 62);
        assert_eq!(l.data_bits, 62 * 8);
        assert_eq!(l.blocks[0].reserved, 0);
        assert_eq!(layout(SymbolVersion::Qr(1), EcLevel::L).blocks[0].reserved, 3);
        assert_eq!(layout(SymbolVersion::Micro(1), EcLevel::L).blocks[0].reserved, 2);
    }

    #[test]
    fn micro_half_codeword() {
        let l = layout(SymbolVersion::Micro(3), EcLevel::L);
        assert_eq!(l.codewords[10].nbits, 4);
        assert!(l.codewords[10].is_data);
        assert_eq!(l.codewords[11].path_start, 84);
        assert_eq!(l.data_bits, 84);
        let m = layout(SymbolVersion::Micro(3), EcLevel::M);
        assert_eq!(m.codewords[8].nbits, 4);
        assert_eq!(m.data_bits, 68);
        assert_eq!(layout(SymbolVersion::Micro(1), EcLevel::L).data_bits, 20);
        assert_eq!(layout(SymbolVersion::Micro(2), EcLevel::L).data_bits, 40);
        assert!(SymbolLayout::new(Arc::new(FunctionMap::new(SymbolVersion::Micro(2))), EcLevel::Q).is_none());
    }

    #[test]
    fn module_lookup_inverts_codeword_modules() {
        for (version, ec) in [
            (SymbolVersion::Qr(3), EcLevel::M),
            (SymbolVersion::Micro(3), EcLevel::L),
            (SymbolVersion::Rmqr(14), EcLevel::H),
            (SymbolVersion::Model1(2), EcLevel::M),
            (SymbolVersion::Model1(13), EcLevel::H),
        ] {
            let l = layout(version, ec);
            for cw in 0..l.codewords.len() {
                let offset = l.codewords[cw].bit_offset;
                for (bit, pos) in l.codeword_modules(cw).iter().enumerate() {
                    assert_eq!(l.module_codeword(pos.x as usize, pos.y as usize), Some((cw, bit as u8 + offset)));
                }
            }
            assert_eq!(l.module_codeword(0, 0), None);
            for pos in l.unused_modules() {
                assert_eq!(l.module_codeword(pos.x as usize, pos.y as usize), None);
            }
        }
    }

    #[test]
    fn model1_blocks_lie_back_to_back() {
        // 5-H: two blocks of 23 data and 44 EC codewords. The first block's
        // data is the first 23 codewords placed, the second's the next 23,
        // and then the EC codewords of each block in turn, where Model 2
        // would take one codeword from each block by turns.
        let l = layout(SymbolVersion::Model1(5), EcLevel::H);
        assert_eq!(l.blocks.len(), 2);
        assert_eq!(l.blocks[0].codewords, (0..23).chain(46..90).collect::<Vec<_>>());
        assert_eq!(l.blocks[1].codewords, (23..46).chain(90..134).collect::<Vec<_>>());
        assert_eq!(l.data_stream, (0..46).collect::<Vec<_>>());
        // Model 2 interleaves, as before.
        let m = layout(SymbolVersion::Qr(5), EcLevel::H);
        assert_eq!(m.blocks.len(), 4);
        assert_eq!(&m.blocks[1].codewords[..3], &[1, 5, 9]);
    }

    #[test]
    fn model1_corner_codeword_and_unused_slots() {
        let l = layout(SymbolVersion::Model1(2), EcLevel::M);
        assert_eq!((l.codewords[0].nbits, l.codewords[0].bit_offset), (4, 4));
        assert_eq!(l.codewords[0].module_mask(), 0x0F);
        assert_eq!(l.codewords[0].stream_bits(), 8);
        assert_eq!(l.codewords[1].path_start, 4);
        assert_eq!(l.codewords[1].module_mask(), 0xFF);
        assert!(l.codewords[0].is_data);
        assert_eq!(l.data_stream[0], 0);
        assert_eq!(l.data_bits, 30 * 8);
        assert_eq!(l.unused_codewords, 0);
        assert!(l.unused_modules().is_empty());
        // 13-H: six blocks of 32 + 58, two slots over.
        let l = layout(SymbolVersion::Model1(13), EcLevel::H);
        assert_eq!(l.blocks.len(), 6);
        assert_eq!(l.codewords.len(), 540);
        assert_eq!(l.unused_codewords, 2);
        assert_eq!(l.unused_modules().len(), 16);
        assert_eq!(l.blocks[0].reserved, 0);
        assert_eq!(layout(SymbolVersion::Model1(1), EcLevel::L).blocks[0].reserved, 3);
        // Micro's short codeword is at the end, with its low nibble fixed.
        let m = layout(SymbolVersion::Micro(3), EcLevel::L);
        assert_eq!(m.codewords[10].module_mask(), 0xF0);
        assert_eq!(m.codewords[10].stream_bits(), 4);
    }
}
