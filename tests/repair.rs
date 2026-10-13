//! Writing corrections back: a damaged symbol, once decoded, repairs to
//! exactly what the encoder produced.

mod common;

use common::*;
use qr_forensics::decode::bch::analyze_format;
use qr_forensics::decode::repair::{known_modules, quiet_zone, render_rgba, repaired_grid};
use qr_forensics::symbol::grid::{Module, ModuleGrid};
use qr_forensics::symbol::{EcLevel, SymbolVersion};
use qrcode_generator::Segment;

/// Erase and corrupt within capacity in every block, with the three
/// codewords to spare that verify a block some of whose codewords were
/// filled in, wipe one copy of the format information (or, where there
/// is only one, flip three of its bits), and scribble on the fixed
/// patterns.
fn damage(grid: &mut ModuleGrid, layout: &qr_forensics::symbol::layout::SymbolLayout, rng: &mut Rng) {
    for block in &layout.blocks {
        let budget = block.ec_len - 3;
        let erasures = rng.below(budget / 2 + 1);
        let errors = (budget - erasures) / 2;
        for (n, &i) in rng.distinct(erasures + errors, block.codewords.len()).iter().enumerate() {
            if n < erasures {
                erase_codeword(grid, layout, block.codewords[i]);
            } else {
                corrupt_codeword(grid, layout, block.codewords[i], n);
            }
        }
    }
    let copy = &layout.func.format[0];
    if layout.func.format.len() > 1 {
        for pos in &copy.bits {
            grid.set(pos.x as usize, pos.y as usize, Module::Unknown);
        }
    } else {
        for &b in &[0, 5, 10] {
            let (x, y) = (copy.bits[b].x as usize, copy.bits[b].y as usize);
            grid.set(x, y, grid.get(x, y).inverted());
        }
    }
    for pos in layout.func.version_info.iter().flat_map(|c| c.bits.iter().take(5)) {
        let (x, y) = (pos.x as usize, pos.y as usize);
        grid.set(x, y, grid.get(x, y).inverted());
    }
    for (x, y) in [(0, 0), (3, 3), (6, 8), (layout.func.width - 1, 0)] {
        if layout.func.expected(x, y).is_some() {
            grid.set(x, y, Module::Unknown);
        }
    }
}

#[test]
fn repaired_symbols_match_the_encoder_exactly() {
    let mut rng = Rng(0x4EFA);
    for (version, ec) in [
        (SymbolVersion::Qr(5), EcLevel::Q),
        (SymbolVersion::Qr(7), EcLevel::H),
        (SymbolVersion::Qr(26), EcLevel::L),
        (SymbolVersion::Micro(3), EcLevel::L),
        (SymbolVersion::Micro(4), EcLevel::Q),
        (SymbolVersion::Rmqr(12), EcLevel::H),
        (SymbolVersion::Rmqr(31), EcLevel::M),
    ] {
        let layout = layout(version, ec);
        let mask = rng.below(version.mask_count() as usize) as u8;
        let text = rng.alphanumeric(byte_capacity(&layout, 3).max(1)).to_lowercase();
        let clean = encode(version, ec, Some(mask), &[Segment::bytes(&text)]).unwrap();
        let what = format!("{version}-{ec:?} mask {mask}");

        let mut grid = clean.clone();
        damage(&mut grid, &layout, &mut rng);
        assert_ne!(grid, clean);
        // The surviving format copy still identifies the symbol.
        let format = analyze_format(&grid, &layout.func);
        assert_eq!((format.best().unwrap().fields.ec, format.best().unwrap().fields.mask), (ec, mask), "{what}");

        let report = decode_plain(&grid, &layout, mask);
        assert!(report.is_trustworthy(), "{what}");
        let repaired = repaired_grid(&grid, &layout, mask, &report);
        assert_eq!(repaired, clean, "{what}");
        assert!(!repaired.has_reliability());
    }
}

#[test]
fn unsolved_blocks_are_left_alone() {
    let (version, ec, mask) = (SymbolVersion::Qr(10), EcLevel::M, 3);
    let layout = layout(version, ec);
    let clean = encode(version, ec, Some(mask), &[Segment::bytes("partly repaired")]).unwrap();
    let mut grid = clean.clone();
    let lost = &layout.blocks[2];
    for &cw in &lost.codewords[..lost.ec_len + 1] {
        erase_codeword(&mut grid, &layout, cw);
    }
    corrupt_codeword(&mut grid, &layout, layout.blocks[0].codewords[0], 0);
    let report = decode_plain(&grid, &layout, mask);
    assert!(!report.blocks[2].is_solved());
    let repaired = repaired_grid(&grid, &layout, mask, &report);
    assert_ne!(repaired, clean);
    let unknown_before = grid.count(Module::Unknown);
    assert_eq!(repaired.count(Module::Unknown), unknown_before, "nothing else became unknown");
    // Everything outside the lost block's codewords matches the encoder.
    let lost_modules: std::collections::HashSet<(u16, u16)> =
        lost.codewords.iter().flat_map(|&cw| layout.codeword_modules(cw).iter().map(|p| (p.x, p.y))).collect();
    for y in 0..grid.height() {
        for x in 0..grid.width() {
            if !lost_modules.contains(&(x as u16, y as u16)) {
                assert_eq!(repaired.get(x, y), clean.get(x, y), "({x}, {y})");
            }
        }
    }
    // Sampled reliability does not survive a correction.
    let mut sampled = grid.clone();
    sampled.set_sampled(0, 0, Module::Light, 3);
    let repaired = repaired_grid(&sampled, &layout, mask, &report);
    assert_eq!(repaired.reliability(0, 0), qr_forensics::symbol::grid::CERTAIN);
    assert_eq!(repaired.get(0, 0), Module::Dark);

    let (w, h, pixels) = render_rgba(&repaired, 4, quiet_zone(version));
    assert_eq!((w, h), ((57 + 8) * 4, (57 + 8) * 4));
    assert_eq!(pixels.len(), w * h * 4);
}

#[test]
fn known_modules_follow_the_blocks_that_verified() {
    let (version, ec, mask) = (SymbolVersion::Qr(5), EcLevel::Q, 3);
    let layout = layout(version, ec);
    let clean = encode(version, ec, Some(mask), &[Segment::bytes("known modules")]).unwrap();
    let report = decode_plain(&clean, &layout, mask);
    assert!(report.is_trustworthy());
    // A clean decode knows every module, as the encoder painted it.
    let known = known_modules(&layout, mask, &report);
    assert_eq!(known.len(), clean.width() * clean.height());
    for (i, module) in known.iter().enumerate() {
        assert_eq!(*module, clean.get(i % clean.width(), i / clean.width()).as_dark(), "module {i}");
    }
    // Erase more of one block than it can bear: the modules of its
    // codewords, data and error correction alike, become unknown, and
    // the fixed patterns, the format information and the other blocks
    // stay known from what the decode established, not from the grid.
    let mut torn = clean.clone();
    let lost = &layout.blocks[1];
    for &cw in &lost.codewords[..lost.ec_len + 1] {
        erase_codeword(&mut torn, &layout, cw);
    }
    corrupt_codeword(&mut torn, &layout, layout.blocks[0].codewords[2], 5);
    let report = decode_plain(&torn, &layout, mask);
    assert!(!report.blocks[1].is_solved());
    assert!(report.blocks[0].is_verified());
    let known = known_modules(&layout, mask, &report);
    for (cw, span) in layout.codewords.iter().enumerate() {
        for pos in layout.codeword_modules(cw) {
            let (x, y) = (pos.x as usize, pos.y as usize);
            let want = if span.block == 1 { None } else { clean.get(x, y).as_dark() };
            assert_eq!(known[y * clean.width() + x], want, "codeword {cw}");
        }
    }
    for y in 0..clean.height() {
        for x in 0..clean.width() {
            if layout.func.expected(x, y).is_some() || matches!(layout.func.role(x, y), qr_forensics::symbol::function_map::Role::Format { .. }) {
                assert_eq!(known[y * clean.width() + x], clean.get(x, y).as_dark(), "({x}, {y})");
            }
        }
    }
}

#[test]
fn a_guessed_format_is_not_written_back() {
    // Nothing of the symbol but its fixed patterns: no block solves and
    // the format bits say nothing, so the level and mask in effect are a
    // guess, and repairing must not paint their format word as if it had
    // been read.
    let (version, ec, mask) = (SymbolVersion::Qr(2), EcLevel::L, 0);
    let layout = layout(version, ec);
    let blank = layout.func.blank_grid();
    let report = decode_plain(&blank, &layout, mask);
    assert_eq!(report.blocks_solved(), 0);
    let repaired = repaired_grid(&blank, &layout, mask, &report);
    for pos in layout.func.format.iter().flat_map(|c| &c.bits) {
        assert_eq!(repaired.get(pos.x as usize, pos.y as usize), Module::Unknown, "{pos:?}");
    }
    assert!(!analyze_format(&repaired, &layout.func).is_reliable());

    // Format bits that name the level and mask themselves, three of them
    // wrong, are put right even though nothing solved.
    let clean = encode(version, ec, Some(mask), &[Segment::bytes("format")]).unwrap();
    let mut grid = blank.clone();
    for copy in &layout.func.format {
        for (i, pos) in copy.bits.iter().enumerate() {
            let module = clean.get(pos.x as usize, pos.y as usize);
            grid.set(pos.x as usize, pos.y as usize, if i < 3 { module.inverted() } else { module });
        }
    }
    let report = decode_plain(&grid, &layout, mask);
    assert_eq!(report.blocks_solved(), 0);
    let repaired = repaired_grid(&grid, &layout, mask, &report);
    for pos in layout.func.format.iter().flat_map(|c| &c.bits) {
        assert_eq!(repaired.get(pos.x as usize, pos.y as usize), clean.get(pos.x as usize, pos.y as usize), "{pos:?}");
    }
    // The same bits under another mask's layout are not that layout's.
    let repaired = repaired_grid(&grid, &layout, 5, &decode_plain(&grid, &layout, 5));
    for pos in layout.func.format.iter().flat_map(|c| &c.bits) {
        assert_eq!(repaired.get(pos.x as usize, pos.y as usize), grid.get(pos.x as usize, pos.y as usize), "{pos:?}");
    }
}
