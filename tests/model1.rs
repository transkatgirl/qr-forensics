//! QR Code Model 1: symbols built here from the tables and the codeword
//! placement, decoded by the pipeline and by zxing-cpp's reader (the
//! `rxing` port of it), the only other implementation of the model. No
//! encoder for Model 1 exists to serve as an oracle, so the reader is the
//! oracle: a symbol it reads as the message it was built from has its
//! codewords where zxing-cpp expects them. The port still interleaves the
//! blocks of a Model 1 symbol as Model 2's are, which upstream zxing-cpp
//! stopped doing with its fix for issue #940: Model 1 lays its blocks
//! back to back. So the port is put only to symbols of one block, and the
//! sample symbol of that issue, which has two, is the oracle for the rest.

mod common;

use std::sync::Arc;

use common::*;
use qr_forensics::decode::bch::{analyze_format, format_word_for};
use qr_forensics::decode::pipeline::{BlockOutcome, SolveMethod, analyze};
use qr_forensics::decode::rs::rs_encode;
use qr_forensics::symbol::function_map::FunctionMap;
use qr_forensics::symbol::grid::{Module, ModuleGrid};
use qr_forensics::symbol::layout::SymbolLayout;
use qr_forensics::symbol::mask::mask_bit;
use qr_forensics::symbol::tables::{self, Mode};
use qr_forensics::symbol::{EcLevel, Family, SymbolVersion};

/// Most bytes a byte-mode message can hold at a version and level.
fn byte_capacity_model1(layout: &SymbolLayout) -> usize {
    let header = layout.stream.preamble_bits as usize
        + layout.stream.mode_bits as usize
        + layout.stream.count_bits(Mode::Byte).unwrap() as usize;
    (layout.data_bits - header) / 8
}

/// A Model 1 symbol holding `text` in byte mode: the data stream as
/// zxing-cpp reads it (four zero bits, then Model 2's segments and
/// padding), Reed-Solomon over equal blocks, interleaved as Model 2
/// interleaves, placed along the function map's path with the data mask,
/// the format word under Model 1's XOR mask, and the unused codeword
/// slots left light once unmasked.
fn encode_model1(version: u8, ec: EcLevel, mask: u8, text: &[u8]) -> ModuleGrid {
    let symbol = SymbolVersion::Model1(version);
    let func = Arc::new(FunctionMap::new(symbol));
    let layout = SymbolLayout::new(func.clone(), ec).expect("every level exists");
    assert!(text.len() <= byte_capacity_model1(&layout));

    let mut bits: Vec<bool> = Vec::with_capacity(layout.data_bits);
    let push = |value: u32, width: usize, bits: &mut Vec<bool>| {
        for i in (0..width).rev() {
            bits.push(value >> i & 1 != 0);
        }
    };
    push(0, layout.stream.preamble_bits as usize, &mut bits);
    push(4, 4, &mut bits);
    push(text.len() as u32, layout.stream.count_bits(Mode::Byte).unwrap() as usize, &mut bits);
    for &byte in text {
        push(byte as u32, 8, &mut bits);
    }
    let terminator = (layout.data_bits - bits.len()).min(4);
    push(0, terminator, &mut bits);
    while !bits.len().is_multiple_of(8) {
        bits.push(false);
    }
    let mut pad = 0;
    while bits.len() < layout.data_bits {
        push([0xEC, 0x11][pad % 2], 8, &mut bits);
        pad += 1;
    }
    assert_eq!(bits.len(), layout.data_bits);
    let data: Vec<u8> = bits.chunks(8).map(|c| c.iter().fold(0u8, |v, &b| v << 1 | u8::from(b))).collect();
    assert_eq!(data.len(), layout.data_stream.len());
    assert_eq!(data[0] & 0xF0, 0, "the corner codeword's high nibble is zero");

    let mut values = vec![0u8; layout.codewords.len()];
    for block in &layout.blocks {
        let block_data = &data[block.stream_offset..block.stream_offset + block.data_len];
        let ec_words = rs_encode(block_data, block.ec_len);
        for (i, &cw) in block.codewords.iter().enumerate() {
            values[cw] = if i < block.data_len { block_data[i] } else { ec_words[i - block.data_len] };
        }
    }

    let mut grid = ModuleGrid::new(func.width, func.height);
    func.fill_fixed(&mut grid);
    for (cw, span) in layout.codewords.iter().enumerate() {
        for (i, pos) in layout.codeword_modules(cw).iter().enumerate() {
            let (x, y) = (pos.x as usize, pos.y as usize);
            let bit = values[cw] & (0x80 >> (span.bit_offset as usize + i)) != 0;
            grid.set(x, y, Module::from_dark(bit != mask_bit(symbol, mask, x, y)));
        }
    }
    for pos in layout.unused_modules() {
        let (x, y) = (pos.x as usize, pos.y as usize);
        grid.set(x, y, Module::from_dark(mask_bit(symbol, mask, x, y)));
    }
    let word = format_word_for(symbol, ec, mask).unwrap();
    for copy in &func.format {
        let painted = word ^ copy.xor;
        for (i, pos) in copy.bits.iter().enumerate() {
            grid.set(pos.x as usize, pos.y as usize, Module::from_dark(painted >> i & 1 != 0));
        }
    }
    assert_eq!(grid.count(Module::Unknown), 0);
    grid
}

/// zxing-cpp's reading of a grid, through rxing's port of its QR decoder,
/// which tells Model 1 from Model 2 by the format mask.
fn zxing_cpp_reads(grid: &ModuleGrid) -> Result<(String, u32, String), String> {
    let mut bits = rxing::common::BitMatrix::with_single_dimension(grid.width() as u32).map_err(|e| e.to_string())?;
    for y in 0..grid.height() {
        for x in 0..grid.width() {
            if grid.get(x, y) == Module::Dark {
                bits.set(x as u32, y as u32);
            }
        }
    }
    let result = rxing::qrcode::cpp_port::decoder::Decode(&bits).map_err(|e| e.to_string())?;
    Ok((result.text(), result.versionNumber(), result.ecLevel().to_owned()))
}

#[test]
fn every_version_and_level_decodes_and_reads_as_zxing_cpp_reads_it() {
    let mut rng = Rng(0x4d6f64656c31);
    let mut through_zxing = 0;
    for version in 1..=14u8 {
        let symbol = SymbolVersion::Model1(version);
        for &ec in symbol.ec_levels() {
            let layout = common::layout(symbol, ec);
            let capacity = byte_capacity_model1(&layout);
            // A message that fills the symbol, and one that leaves padding.
            for (length, mask) in [(capacity, (version + ec as u8) % 8), (capacity / 3, 3)] {
                let text: Vec<u8> = (0..length).map(|_| b' ' + rng.below(95) as u8).collect();
                let grid = encode_model1(version, ec, mask, &text);
                let func = layout.func.clone();

                // The pipeline: the format word names the level and mask,
                // and every block is clean.
                let analysis = analyze(&grid, &func);
                assert!(analysis.format.is_reliable(), "{symbol}-{ec:?}: {:?}", analysis.format.summary());
                let best = analysis.format.best().unwrap();
                assert_eq!((best.fields.ec, best.fields.mask), (ec, mask), "{symbol}");
                assert!(best.is_standard() && analysis.format.other_model_hint().is_none());
                let report = decode_plain(&grid, &layout, mask);
                for (b, block) in report.blocks.iter().enumerate() {
                    assert!(
                        matches!(
                            block.outcome,
                            BlockOutcome::Solved { method: SolveMethod::ReedSolomon, errors: 0, erasures: 0, .. }
                        ),
                        "{symbol}-{ec:?} mask {mask}: block {b} is {:?}",
                        block.outcome
                    );
                }
                assert!(report.is_trustworthy(), "{symbol}-{ec:?}: {:?}", report.score);
                assert!(!report.parse.preamble_damaged);
                assert_eq!(report.text.plain().as_bytes(), text, "{symbol}-{ec:?} mask {mask}");
                assert!(report.score.tail_plausible || length == capacity, "{symbol}-{ec:?}: {:?}", report.tail);

                // zxing-cpp's reader, for the symbols of one block, whose
                // layout the port and upstream agree on.
                if layout.blocks.len() == 1 {
                    let (read, v, level) = zxing_cpp_reads(&grid)
                        .unwrap_or_else(|e| panic!("{symbol}-{ec:?} mask {mask}: zxing-cpp refused: {e}"));
                    assert_eq!(read.as_bytes(), text, "{symbol}-{ec:?} mask {mask}: zxing-cpp read {read:?}");
                    assert_eq!((v, level.as_str()), (version as u32, ec.letter()), "{symbol}-{ec:?}");
                    through_zxing += 1;
                }
            }
        }
    }
    // Versions 1 to 4 at every level, 5 at L, M and Q, and 6 and 7 at L
    // have one block: twenty-one symbols, each built twice.
    assert_eq!(through_zxing, 2 * 21);
}

/// A render pixel for pixel, with `quiet` modules of quiet zone, as a grid.
fn grid_of_render(path: &std::path::Path, quiet: usize) -> ModuleGrid {
    let image = image::open(path).unwrap().to_luma8();
    let (w, h) = (image.width() as usize - 2 * quiet, image.height() as usize - 2 * quiet);
    let mut grid = ModuleGrid::new(w, h);
    for y in 0..h {
        for x in 0..w {
            grid.set(x, y, Module::from_dark(image.get_pixel((x + quiet) as u32, (y + quiet) as u32)[0] < 128));
        }
    }
    grid
}

#[test]
fn the_zxing_cpp_issue_940_symbol_decodes_with_its_blocks_back_to_back() {
    // zxing-cpp issue #940: a Model 1 symbol of version 5 at level H, two
    // blocks of 23 data and 44 EC codewords, which no reader decoded until
    // zxing-cpp found that Model 1 does not interleave its blocks. The
    // image is the issue's sample, a two-module quiet zone around a
    // module a pixel.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/blackbox");
    let grid = grid_of_render(&dir.join("cpp_qrcode-2_#940.png"), 2);
    let expected = std::fs::read_to_string(dir.join("cpp_qrcode-2_#940.txt")).unwrap();
    assert_eq!((grid.width(), grid.height()), (37, 37));
    let func = Arc::new(FunctionMap::new(SymbolVersion::Model1(5)));
    let analysis = analyze(&grid, &func);
    assert!(analysis.format.is_reliable(), "{:?}", analysis.format.summary());
    let best = analysis.format.best().unwrap();
    assert_eq!((best.fields.ec, best.fields.mask), (EcLevel::H, 7));
    let layout = SymbolLayout::new(func.clone(), EcLevel::H).unwrap();
    assert_eq!(layout.blocks.len(), 2);
    let report = decode_plain(&grid, &layout, 7);
    for (b, block) in report.blocks.iter().enumerate() {
        assert!(block.is_verified(), "block {b}: {:?}", block.outcome);
    }
    assert!(report.is_trustworthy(), "{:?}", report.score);
    assert_eq!(report.text.plain(), expected.trim_end());
    assert_eq!(report.text.symbology.as_deref(), Some("]Q0"));
    // Found as a QR Code first, as the photo pipeline finds any square
    // symbol, its format word is the hint that it is Model 1.
    let as_model2 = analyze(&grid, &FunctionMap::new(SymbolVersion::Qr(5)));
    assert!(as_model2.format.other_model_hint().is_some());
    assert!(!as_model2.format.is_reliable());
}

#[test]
fn a_model_2_grid_is_told_from_a_model_1_one_by_its_format_word() {
    // The same text at version 4-Q under each model.
    let text = b"Model 1 or Model 2?";
    let model1 = encode_model1(4, EcLevel::Q, 2, text);
    let model2 = encode(SymbolVersion::Qr(4), EcLevel::Q, Some(2), &[qrcode_generator::Segment::bytes(text)]).unwrap();
    for (grid, own, other) in [(&model1, SymbolVersion::Model1(4), SymbolVersion::Qr(4)), (&model2, SymbolVersion::Qr(4), SymbolVersion::Model1(4))] {
        let own_map = FunctionMap::new(own);
        let own_analysis = analyze_format(grid, &own_map);
        assert!(own_analysis.is_reliable() && own_analysis.other_model_hint().is_none(), "{own}");
        // Read as the other model, the format word is the hint.
        let other_map = FunctionMap::new(other);
        let other_analysis = analyze_format(grid, &other_map);
        let hint = other_analysis.other_model_hint().unwrap_or_else(|| panic!("{own} read as {other}: no hint"));
        assert_eq!((hint.fields.ec, hint.fields.mask), (EcLevel::Q, 2));
        assert!(!other_analysis.is_reliable());
        assert_eq!(own.other_model(), Some(other));
    }
    // Fixed patterns: a Model 1 grid fits its own map in full, and the
    // Model 2 map everywhere but the alignment pattern.
    let (matching, known) = FunctionMap::new(SymbolVersion::Model1(4)).fixed_agreement(&model1);
    assert_eq!(matching, known);
    let (matching, known) = FunctionMap::new(SymbolVersion::Qr(4)).fixed_agreement(&model1);
    assert!(known - matching <= 25 && matching * 10 > known * 9, "{matching} of {known}");
}

#[test]
fn model1_is_a_family_of_its_own_and_not_searched_for() {
    assert_eq!(Family::Model1.versions().len(), 14);
    assert!(Family::ALL.contains(&Family::Model1));
    assert!(!Family::SEARCHED.contains(&Family::Model1));
    assert_eq!(SymbolVersion::from_size(Family::Model1, 25, 25), Some(SymbolVersion::Model1(2)));
    assert_eq!(SymbolVersion::from_size(Family::Model1, 77, 77), None);
    assert_eq!(SymbolVersion::Qr(14).other_model(), Some(SymbolVersion::Model1(14)));
    assert_eq!(SymbolVersion::Qr(15).other_model(), None);
    assert!(!SymbolVersion::Model1(14).has_version_info());
    assert_eq!(SymbolVersion::Model1(7).to_string(), "7");
    assert_eq!(tables::total_codewords(SymbolVersion::Model1(2)), 46);
}
