//! Grids painted the wrong way round, or seen from the back with their
//! format bits gone, are told apart by trial: every turn and reflection on
//! the fixed patterns, and every mirror image by decoding.

mod common;

use common::*;
use qr_forensics::decode::pipeline::{DecodeOptions, analyze, brute_force, brute_force_mirrored};
use qr_forensics::symbol::grid::{Mirror, Module, Symmetry};
use qr_forensics::symbol::{EcLevel, SymbolVersion};
use qrcode_generator::Segment;

#[test]
fn turned_grids_are_recognised_from_their_fixed_patterns() {
    let qr = layout(SymbolVersion::Qr(3), EcLevel::M);
    let grid = encode(SymbolVersion::Qr(3), EcLevel::M, Some(2), &[Segment::bytes("turned")]).unwrap();
    assert_eq!(analyze(&grid, &qr.func).orientation, None, "the right way up needs nothing");
    for (turned, back) in [
        (grid.rotated_cw(), Symmetry::RotateCcw),
        (grid.rotated_ccw(), Symmetry::RotateCw),
        (grid.rotated_180(), Symmetry::Rotate180),
        (grid.flipped_vertical(), Symmetry::FlipVertical),
        (grid.flipped_horizontal(), Symmetry::FlipHorizontal),
        (grid.anti_transposed(), Symmetry::AntiTranspose),
    ] {
        let analysis = analyze(&turned, &qr.func);
        let orientation = analysis.orientation.unwrap_or_else(|| panic!("{back:?}: nothing suggested"));
        assert_eq!(orientation.symmetry, back, "{back:?}");
        assert!(!orientation.inverted);
        assert_eq!(orientation.agreement.0, orientation.agreement.1, "{back:?}: every fixed module reads right once turned");
        assert_eq!(turned.symmetric(back), grid, "{back:?}: the suggestion restores the grid");
    }
    // Turned and seen as a negative: the inversion is named first, and once
    // the colours are swapped back the turn is.
    let analysis = analyze(&grid.rotated_cw().inverted(), &qr.func);
    assert!(analysis.inverted && analysis.orientation.is_none());
    let analysis = analyze(&grid.rotated_cw().inverted().inverted(), &qr.func);
    assert_eq!(analysis.orientation.map(|o| o.symmetry), Some(Symmetry::RotateCcw));
    // A plain negative is the inversion warning's job, not this one's.
    let analysis = analyze(&grid.inverted(), &qr.func);
    assert!(analysis.inverted && analysis.orientation.is_none());
    // The transpose keeps a square symbol's fixed patterns in place, so it
    // is the format bits' job to tell, and nothing is suggested here.
    let analysis = analyze(&grid.transposed(), &qr.func);
    assert_eq!(analysis.orientation, None);
    assert_eq!(analysis.mirrored, Some(Mirror::Transpose));
    // A rectangular symbol has fewer ways to be wrong.
    let rmqr = layout(SymbolVersion::Rmqr(11), EcLevel::M);
    let grid = encode(SymbolVersion::Rmqr(11), EcLevel::M, None, &[Segment::bytes("wide")]).unwrap();
    let analysis = analyze(&grid.rotated_180(), &rmqr.func);
    assert_eq!(analysis.orientation.map(|o| o.symmetry), Some(Symmetry::Rotate180));
}

#[test]
fn mirror_images_with_no_format_bits_decode_by_trial() {
    let opts = DecodeOptions::default();
    for (version, ec, text, mirror) in [
        (SymbolVersion::Qr(4), EcLevel::M, "seen from the back", Mirror::Transpose),
        (SymbolVersion::Micro(4), EcLevel::M, "micro", Mirror::Transpose),
        (SymbolVersion::Rmqr(13), EcLevel::M, "rmqr from behind", Mirror::Horizontal),
    ] {
        let layout = layout(version, ec);
        let func = &layout.func;
        let mask = if version.mask_count() > 1 { 1 } else { 0 };
        let clean = encode(version, ec, Some(mask), &[Segment::bytes(text)]).unwrap();
        // Every format bit unreadable, wherever the mirror image puts it:
        // the usual tell is gone.
        let mut torn = clean.clone();
        for pos in func.format.iter().flat_map(|c| c.bits.iter()) {
            torn.set(pos.x as usize, pos.y as usize, Module::Unknown);
        }
        let grid = torn.mirrored(mirror);
        let analysis = analyze(&grid, func);
        assert_eq!(analysis.mirrored, None, "{version}: the format bits cannot tell");
        let plain = brute_force(&grid, func, None, &opts);
        assert!(!plain[0].report.is_trustworthy(), "{version}: nothing decodes as painted");
        let mirrored = brute_force_mirrored(&grid, func, &opts);
        let best = &mirrored[0];
        assert!(best.report.is_trustworthy(), "{version}: {:?}", best.report.score);
        assert_eq!(best.mirror, Some(mirror), "{version}");
        assert_eq!(best.report.text.plain(), text, "{version}");
        assert!(best.report.blocks_verified() > plain[0].report.blocks_verified());
        // The right way round the mirror images decode no better.
        let mirrored = brute_force_mirrored(&clean, func, &opts);
        assert!(!mirrored[0].report.is_trustworthy(), "{version}");
    }
}
