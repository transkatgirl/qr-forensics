//! Padding recovery: a short message in a large symbol leaves most data
//! codewords as known filler, which can rescue blocks that are otherwise
//! past saving.

mod common;

use common::*;
use qr_forensics::decode::bitstream::{Certainty, EndOfData, LostReason};
use qr_forensics::decode::codewords::{PinSource, Pins};
use qr_forensics::decode::codewords::Pin;
use qr_forensics::decode::padding::{
    Filler, check_tail, clear_padding_pins, detect_padding, pad_tail, padding_at, pin_padding_at, pinned_filler, recover_padding,
};
use qr_forensics::decode::pipeline::{BlockOutcome, DecodeOptions, SolveMethod, decode};
use qr_forensics::decode::repair::render_rgba;
use qr_forensics::imaging::Luma;
use qr_forensics::imaging::adjust::Adjust;
use qr_forensics::imaging::detect::{SearchParams, find_symbols_with};
use qr_forensics::imaging::read::{read_detection, read_outline};
use qr_forensics::imaging::sample::SampleParams;
use qr_forensics::symbol::Family;
use qr_forensics::symbol::grid::{Module, ModuleGrid};
use qr_forensics::symbol::layout::SymbolLayout;
use qr_forensics::symbol::{EcLevel, SymbolVersion};
use qrcode_generator::Segment;

fn recover(grid: &ModuleGrid, layout: &SymbolLayout, mask: u8) -> Option<qr_forensics::decode::padding::PaddingRecovery> {
    recover_padding(grid, layout, mask, &Pins::new(), &DecodeOptions::default())
}

/// Erase `count` data codewords of a block, starting from its last.
fn erase_data_tail(grid: &mut ModuleGrid, layout: &SymbolLayout, block: usize, count: usize) {
    let spec = &layout.blocks[block];
    for &cw in spec.codewords[..spec.data_len].iter().rev().take(count) {
        erase_codeword(grid, layout, cw);
    }
}

#[test]
fn expected_tail_matches_what_encoders_write() {
    let mut rng = Rng(0x7A11);
    let mut cases: Vec<(SymbolVersion, EcLevel)> = vec![
        (SymbolVersion::Qr(1), EcLevel::L),
        (SymbolVersion::Qr(5), EcLevel::Q),
        (SymbolVersion::Qr(20), EcLevel::M),
        (SymbolVersion::Rmqr(3), EcLevel::M),
        (SymbolVersion::Rmqr(25), EcLevel::H),
    ];
    for v in 1..=4 {
        cases.extend(SymbolVersion::Micro(v).ec_levels().iter().map(|&ec| (SymbolVersion::Micro(v), ec)));
    }
    for (version, ec) in cases {
        let layout = layout(version, ec);
        // Every payload length from one character up to full.
        let capacity_digits = match version {
            SymbolVersion::Micro(1) => 5,
            _ => (layout.data_bits - 20) * 3 / 10,
        };
        for len in 1..=capacity_digits.min(60) {
            let digits = rng.digits(len);
            let Ok(grid) = encode(version, ec, Some(0), &[Segment::numeric(&digits).unwrap()]) else {
                continue;
            };
            let report = decode_plain(&grid, &layout, 0);
            let what = format!("{version}-{ec:?} with {len} digits");
            assert_eq!(report.text.plain(), digits, "{what}");
            let EndOfData::At { bit, certainty } = report.parse.end else { panic!("{what}") };
            assert_eq!(certainty, Certainty::Verified);
            let tail = check_tail(&layout, &report.stream, bit);
            assert_eq!((tail.mismatches, tail.unknown), (0, 0), "{what}: {tail:?}");
            assert_eq!(report.tail, Some(tail));
            assert!(report.score.tail_plausible, "{what}");
            // The first pad codeword, when there is one, is always 0xEC.
            let first_pad = pad_tail(&layout, bit).first_pad_codeword;
            if let Some(&cw) = layout.data_stream.get(first_pad)
                && layout.codewords[cw].nbits == 8
            {
                assert_eq!(report.codewords[cw].final_value, Some(0xEC), "{what}");
            }
        }
    }
}

#[test]
fn rescues_blocks_whose_damage_is_in_the_padding() {
    // 10-M: five blocks of 43 or 44 data codewords with 26 EC codewords each.
    // A short message leaves blocks 1 to 4 as pure padding.
    let (version, ec, mask) = (SymbolVersion::Qr(10), EcLevel::M, 4);
    let layout = layout(version, ec);
    let text = "short message";
    let clean = encode(version, ec, Some(mask), &[Segment::bytes(text)]).unwrap();

    let mut grid = clean.clone();
    for block in 1..5 {
        erase_data_tail(&mut grid, &layout, block, 30);
    }
    let before = decode_plain(&grid, &layout, mask);
    assert_eq!(before.blocks_solved(), 1);
    assert_eq!(before.text.plain(), text);

    let candidate = detect_padding(&layout, &before).unwrap();
    assert_eq!(candidate.header_certainty, Some(Certainty::Verified));
    assert_eq!(candidate.mismatches, 0);

    let recovery = recover(&grid, &layout, mask).unwrap();
    assert!(recovery.confirmed);
    assert_eq!((recovery.solved_before, recovery.solved_after), (1, 5));
    assert!(recovery.report.is_trustworthy());
    assert_eq!(recovery.report.text.plain(), text);
    assert!(recovery.pins.values().all(|p| p.source != PinSource::Manual));
    // Every recovered codeword equals the clean symbol's.
    let truth = decode_plain(&clean, &layout, mask);
    for (got, want) in recovery.report.codewords.iter().zip(&truth.codewords) {
        assert_eq!(got.final_value, want.final_value);
    }

    let mut pins = recovery.pins.clone();
    clear_padding_pins(&mut pins);
    assert!(pins.is_empty());
}

#[test]
fn finds_padding_without_readable_headers() {
    // Destroy block 0, which holds every segment header, far past repair.
    // The pad pattern in the other blocks still shows where padding starts.
    let (version, ec, mask) = (SymbolVersion::Qr(10), EcLevel::M, 0);
    let layout = layout(version, ec);
    let text = "the first block is gone";
    let clean = encode(version, ec, Some(mask), &[Segment::bytes(text)]).unwrap();
    let truth = decode_plain(&clean, &layout, mask);

    let mut grid = clean.clone();
    let first = &layout.blocks[0];
    // Unknown: the message itself and a little beyond. Block 0 loses 27
    // codewords, one more than it can restore.
    for &cw in &first.codewords[..27] {
        erase_codeword(&mut grid, &layout, cw);
    }
    // And two more blocks lose more padding than they can restore.
    erase_data_tail(&mut grid, &layout, 2, 30);
    erase_data_tail(&mut grid, &layout, 4, 28);

    let before = decode_plain(&grid, &layout, mask);
    assert_eq!(before.blocks_solved(), 2);
    assert!(matches!(before.parse.end, EndOfData::Lost { .. }));

    let candidate = detect_padding(&layout, &before).unwrap();
    assert_eq!(candidate.end_bit, None);
    // The first pad codeword is itself erased. The alternation of the ones
    // that can be read fixes its parity, and the scan picks the latest start
    // consistent with them, which here is the true one.
    let EndOfData::At { bit, .. } = truth.parse.end else { panic!() };
    let true_first_pad = pad_tail(&layout, bit).first_pad_codeword;
    assert_eq!(true_first_pad, 26);
    assert_eq!(candidate.first_pad_codeword, true_first_pad);

    let recovery = recover(&grid, &layout, mask).unwrap();
    // Blocks 2 and 4 are rescued. Block 0 is left with 26 unknown codewords
    // against 26 EC codewords: solvable, with nothing to spare, so its text
    // is reported as unverified.
    assert_eq!(recovery.solved_after, 5, "{:?}", recovery.report.blocks);
    assert!(!recovery.report.blocks[0].is_verified());
    assert!(!recovery.confirmed);
    assert!(!recovery.report.is_trustworthy());
    assert_eq!(recovery.report.text.plain(), text);
    for (got, want) in recovery.report.codewords.iter().zip(&truth.codewords) {
        assert_eq!(got.final_value, want.final_value);
    }
}

#[test]
fn padding_misread_as_data_is_corrected_for_free() {
    // Wrong modules, not unknown ones: 14 bad pad codewords in a block that
    // can correct 13.
    let (version, ec, mask) = (SymbolVersion::Qr(10), EcLevel::M, 7);
    let layout = layout(version, ec);
    let clean = encode(version, ec, Some(mask), &[Segment::bytes("x")]).unwrap();
    let mut grid = clean.clone();
    let block = &layout.blocks[3];
    for i in 0..14 {
        corrupt_codeword(&mut grid, &layout, block.codewords[3 * i], i);
    }
    let before = decode_plain(&grid, &layout, mask);
    assert!(matches!(before.blocks[3].outcome, BlockOutcome::Failed(_)));

    let recovery = recover(&grid, &layout, mask).unwrap();
    assert!(recovery.confirmed);
    assert_eq!(recovery.solved_after, 5);
    assert!(recovery.candidate.mismatches >= 14);
    assert!(recovery.report.is_trustworthy());
}

#[test]
fn pure_pad_blocks_solve_from_pins_even_with_ruined_error_correction() {
    let (version, ec, mask) = (SymbolVersion::Qr(10), EcLevel::M, 1);
    let layout = layout(version, ec);
    let clean = encode(version, ec, Some(mask), &[Segment::bytes("tiny")]).unwrap();
    let mut grid = clean.clone();
    // Misread every EC codeword of block 2. Nothing about the block's data
    // is damaged, but Reed-Solomon cannot confirm any of it.
    let block = &layout.blocks[2];
    for (i, &cw) in block.codewords[block.data_len..].iter().enumerate() {
        corrupt_codeword(&mut grid, &layout, cw, i);
    }
    assert!(!decode_plain(&grid, &layout, mask).blocks[2].is_solved());

    let recovery = recover(&grid, &layout, mask).unwrap();
    assert!(matches!(
        recovery.report.blocks[2].outcome,
        BlockOutcome::Solved { method: SolveMethod::PinnedOnly, .. }
    ));
    let truth = decode_plain(&clean, &layout, mask);
    for &cw in &block.codewords {
        assert_eq!(recovery.report.codewords[cw].final_value, truth.codewords[cw].final_value);
    }
}

#[test]
fn works_in_micro_qr_and_rmqr() {
    // M4-L: 16 data codewords, 8 EC. Erase nine pad codewords.
    let (version, ec) = (SymbolVersion::Micro(4), EcLevel::L);
    let layout_m4 = layout(version, ec);
    let mut grid = encode(version, ec, Some(2), &[Segment::numeric("42").unwrap()]).unwrap();
    erase_data_tail(&mut grid, &layout_m4, 0, 9);
    assert!(!decode_plain(&grid, &layout_m4, 2).is_complete());
    let recovery = recover(&grid, &layout_m4, 2).unwrap();
    assert!(recovery.report.is_complete());
    assert_eq!(recovery.report.text.plain(), "42");

    // M3-L ends its data with a four-bit codeword, which is zero as padding.
    let (version, ec) = (SymbolVersion::Micro(3), EcLevel::L);
    let layout_m3 = layout(version, ec);
    let mut grid = encode(version, ec, Some(0), &[Segment::numeric("7").unwrap()]).unwrap();
    erase_data_tail(&mut grid, &layout_m3, 0, 7);
    assert!(!decode_plain(&grid, &layout_m3, 0).is_complete());
    let recovery = recover(&grid, &layout_m3, 0).unwrap();
    assert!(recovery.report.is_complete());
    assert_eq!(recovery.report.text.plain(), "7");
    let half = *layout_m3.data_stream.last().unwrap();
    assert_eq!(recovery.pins[&half].mask, 0xF0);
    assert_eq!(recovery.pins[&half].value, 0);

    // R17x139-M: four blocks of 38 data codewords, 20 EC codewords each.
    let (version, ec) = (SymbolVersion::Rmqr(31), EcLevel::M);
    let layout_r = layout(version, ec);
    let mut grid = encode(version, ec, None, &[Segment::bytes("rMQR")]).unwrap();
    for block in 0..4 {
        erase_data_tail(&mut grid, &layout_r, block, 24);
    }
    assert_eq!(decode_plain(&grid, &layout_r, 0).blocks_solved(), 0);
    let recovery = recover(&grid, &layout_r, 0).unwrap();
    assert!(recovery.report.is_trustworthy());
    assert_eq!(recovery.report.text.plain(), "rMQR");
}

/// `clean` as an encoder that fills otherwise than the standard would
/// have written it: `fill` gives the pad codeword each place from the
/// first, and the error correction is computed over the data so filled.
fn refilled(layout: &SymbolLayout, clean: &ModuleGrid, mask: u8, fill: impl Fn(usize) -> u8) -> ModuleGrid {
    let truth = decode_plain(clean, layout, mask);
    let EndOfData::At { bit, .. } = truth.parse.end else { panic!() };
    let first_pad = pad_tail(layout, bit).first_pad_codeword;
    let mut grid = clean.clone();
    for block in &layout.blocks {
        let mut data: Vec<u8> =
            block.codewords[..block.data_len].iter().map(|&cw| truth.codewords[cw].final_value.unwrap()).collect();
        for (i, value) in data.iter_mut().enumerate() {
            if block.stream_offset + i >= first_pad {
                *value = fill(block.stream_offset + i - first_pad);
            }
        }
        let ec_words = qr_forensics::decode::rs::rs_encode(&data, block.ec_len);
        for (&cw, value) in block.codewords.iter().zip(data.into_iter().chain(ec_words)) {
            for (i, pos) in layout.codeword_modules(cw).iter().enumerate() {
                let (x, y) = (pos.x as usize, pos.y as usize);
                let bit = value & (0x80 >> i) != 0;
                let masked = bit != qr_forensics::symbol::mask::mask_bit(layout.version(), mask, x, y);
                grid.set(x, y, Module::from_dark(masked));
            }
        }
    }
    grid
}

/// A pad codeword by its place from the first.
type Fill = fn(usize) -> u8;

/// The fillers looked for, each as the pad codeword it puts a number of
/// places from the first. The codeword a late start leaves is zero here.
fn fillers() -> [(Filler, Fill); 4] {
    [
        (Filler::Standard, |place| [0xEC, 0x11][place % 2]),
        (Filler::Late, |place| if place == 0 { 0 } else { [0x11, 0xEC][place % 2] }),
        (Filler::Swapped, |place| [0x11, 0xEC][place % 2]),
        (Filler::Zeros, |_| 0),
    ]
}

#[test]
fn a_filler_that_is_not_the_standards_is_found_and_pinned() {
    // Padding of zeros, the standard's two codewords with the second
    // first, and the standard's begun a codeword late: none of them is
    // asked of a read, and each is known without reading it as the
    // standard's is, once it is seen which a symbol has.
    let (version, ec, mask) = (SymbolVersion::Qr(6), EcLevel::M, 0);
    let layout = layout(version, ec);
    let clean = encode(version, ec, Some(mask), &[Segment::bytes("zeros")]).unwrap();
    let EndOfData::At { bit, .. } = decode_plain(&clean, &layout, mask).parse.end else { panic!() };
    let first_pad = pad_tail(&layout, bit).first_pad_codeword;
    for (filler, fill) in fillers() {
        let what = format!("{filler:?}");
        let whole = refilled(&layout, &clean, mask, fill);
        let truth = decode_plain(&whole, &layout, mask);
        assert!(truth.is_trustworthy(), "{what}");
        assert_eq!(truth.text.plain(), "zeros", "{what}");
        assert_eq!(truth.tail.unwrap().zero_filled, filler == Filler::Zeros, "{what}");
        let candidate = detect_padding(&layout, &truth).unwrap();
        assert_eq!((candidate.filler, candidate.first_pad_codeword, candidate.mismatches), (filler, first_pad, 0), "{what}");
        assert_eq!(candidate.header_certainty, Some(Certainty::Verified), "{what}");

        // A block that is padding throughout loses more of it than its
        // error correction restores, and the block the message is in
        // loses pad codewords as well.
        let mut grid = whole.clone();
        erase_data_tail(&mut grid, &layout, 1, 20);
        erase_data_tail(&mut grid, &layout, 0, 12);
        let damaged = decode_plain(&grid, &layout, mask);
        assert!(!damaged.blocks[1].is_solved(), "{what}");
        assert_eq!(detect_padding(&layout, &damaged).map(|c| c.filler), Some(filler), "{what}");
        let recovery = recover(&grid, &layout, mask).unwrap();
        assert!(recovery.confirmed && recovery.report.is_trustworthy(), "{what}: {:?}", recovery.report.blocks);
        assert_eq!(recovery.candidate.filler, filler, "{what}");
        assert_eq!(pinned_filler(&layout, &recovery.report, &recovery.pins), Some(filler), "{what}");
        assert_eq!(recovery.report.text.plain(), "zeros", "{what}");
        for (cw, (got, want)) in recovery.report.codewords.iter().zip(&truth.codewords).enumerate() {
            assert_eq!(got.final_value, want.final_value, "{what} codeword {cw}");
        }
        // The codeword a late start leaves is not pinned, whatever it
        // holds, and every pad codeword after it is.
        let stray = layout.data_stream[first_pad];
        assert_eq!(recovery.pins.contains_key(&stray), filler != Filler::Late, "{what}");
        assert!(layout.data_stream[first_pad + 1..].iter().all(|cw| recovery.pins[cw].source == PinSource::Padding), "{what}");
    }
}

#[test]
fn padding_that_cannot_be_seen_is_taken_for_the_standards() {
    // A filler is told by what can be read of it: its own codewords,
    // those error correction restores in a block that solves, and the
    // error correction of the blocks that are padding throughout. Here
    // the message fills the first block and the others cannot be read
    // at all: the headers alone say where the padding is, and it is
    // taken for the standard's, as it was when that was the only filler
    // known. Two codewords of it are the least that name another.
    let (version, ec, mask) = (SymbolVersion::Qr(6), EcLevel::M, 0);
    let layout = layout(version, ec);
    let clean = encode(version, ec, Some(mask), &[Segment::bytes("twenty-five bytes of text")]).unwrap();
    let EndOfData::At { bit, .. } = decode_plain(&clean, &layout, mask).parse.end else { panic!() };
    let first_pad = pad_tail(&layout, bit).first_pad_codeword;
    assert_eq!(first_pad, layout.blocks[1].stream_offset);
    for (filler, fill) in fillers() {
        let what = format!("{filler:?}");
        let whole = refilled(&layout, &clean, mask, fill);
        let mut grid = whole.clone();
        let pure = layout.blocks[1..].iter().flat_map(|block| &block.codewords[block.data_len..]);
        for &cw in layout.data_stream[first_pad..].iter().chain(pure) {
            erase_codeword(&mut grid, &layout, cw);
        }
        let blind = decode_plain(&grid, &layout, mask);
        assert!(blind.blocks[0].is_verified() && blind.blocks_solved() == 1, "{what}");
        let candidate = detect_padding(&layout, &blind).unwrap();
        assert_eq!((candidate.filler, candidate.header_certainty), (Filler::Standard, Some(Certainty::Verified)), "{what}");

        // Two pad codewords of the second block come back. A turn that
        // begins with the second of the standard's codewords and one
        // that begins a codeword late differ in the first pad codeword
        // alone, which is still unreadable, and of the two the one that
        // leaves it alone is taken.
        for &cw in &layout.data_stream[40..42] {
            for pos in layout.codeword_modules(cw) {
                let (x, y) = (pos.x as usize, pos.y as usize);
                grid.set(x, y, whole.get(x, y));
            }
        }
        let candidate = detect_padding(&layout, &decode_plain(&grid, &layout, mask)).unwrap();
        let seen = if filler == Filler::Swapped { Filler::Late } else { filler };
        assert_eq!((candidate.filler, candidate.matches >= 16, candidate.mismatches), (seen, true, 0), "{what}");
    }
}

#[test]
fn a_filler_found_without_headers_is_fitted_to_them_once_they_read() {
    // The codewords that hold the header are unreadable, and with pad
    // codewords of its block gone too the block is past restoring: only
    // the turn of the two pad codewords, which marks its own place,
    // shows where the padding is. Pinned from where it can be seen, it
    // restores the block, and the header then says where the padding
    // starts: where the turn does, a codeword before it, or at the
    // second of its two codewords.
    let (version, ec, mask) = (SymbolVersion::Qr(10), EcLevel::M, 0);
    let layout = layout(version, ec);
    let text = "the first block is gone";
    let clean = encode(version, ec, Some(mask), &[Segment::bytes(text)]).unwrap();
    let EndOfData::At { bit, .. } = decode_plain(&clean, &layout, mask).parse.end else { panic!() };
    let first_pad = pad_tail(&layout, bit).first_pad_codeword;
    assert_eq!(first_pad, 26);
    for (filler, fill) in fillers() {
        let what = format!("{filler:?}");
        let whole = refilled(&layout, &clean, mask, fill);
        let truth = decode_plain(&whole, &layout, mask);
        let mut grid = whole.clone();
        let first = &layout.blocks[0];
        for &cw in first.codewords[..12].iter().chain(&first.codewords[28..first.data_len]) {
            erase_codeword(&mut grid, &layout, cw);
        }
        let before = decode_plain(&grid, &layout, mask);
        assert!(!before.blocks[0].is_solved(), "{what}");
        assert!(matches!(before.parse.end, EndOfData::Lost { .. }), "{what}");
        let found = detect_padding(&layout, &before);
        if filler == Filler::Zeros {
            // A run of zeros marks no place, and is not looked for
            // without the headers to say where it starts.
            assert_eq!(found, None);
            assert!(recover(&grid, &layout, mask).is_none());
            continue;
        }
        let found = found.unwrap();
        assert_eq!((found.filler, found.end_bit), (Filler::Standard, None), "{what}");
        assert_eq!(found.first_pad_codeword, first_pad + usize::from(filler != Filler::Standard), "{what}");
        let recovery = recover(&grid, &layout, mask).unwrap();
        assert!(recovery.confirmed && recovery.report.is_trustworthy(), "{what}: {:?}", recovery.report.blocks);
        assert_eq!((recovery.candidate.filler, recovery.candidate.first_pad_codeword), (filler, first_pad), "{what}");
        assert_eq!(pinned_filler(&layout, &recovery.report, &recovery.pins), Some(filler), "{what}");
        assert_eq!(recovery.report.text.plain(), text, "{what}");
        for (cw, (got, want)) in recovery.report.codewords.iter().zip(&truth.codewords).enumerate() {
            assert_eq!(got.final_value, want.final_value, "{what} codeword {cw}");
        }
    }
}

#[test]
fn padding_is_pinned_where_it_is_said_to_start() {
    // Two characters in a symbol with room for fifty-three: all but four
    // data codewords are padding. With every one of them torn away there
    // is far more gone than error correction restores, the header that
    // says where the data ends is unconfirmed, and no padding is left to
    // show itself: nothing is found.
    let (version, ec, mask) = (SymbolVersion::Qr(3), EcLevel::L, 2);
    let layout = layout(version, ec);
    let clean = encode(version, ec, Some(mask), &[Segment::bytes("hi")]).unwrap();
    let count = layout.data_stream.len();
    let first = pad_tail(&layout, 4 + 8 + 16).first_pad_codeword;
    assert_eq!((first, count), (4, 55));
    let mut grid = clean.clone();
    for &cw in &layout.data_stream[first..] {
        erase_codeword(&mut grid, &layout, cw);
    }
    let before = decode_plain(&grid, &layout, mask);
    assert!(!before.blocks[0].is_solved());
    assert_eq!(detect_padding(&layout, &before), None);
    assert!(recover(&grid, &layout, mask).is_none());
    // Said by hand, the start is pinned from as recovery would pin it,
    // and the block solves.
    let (pins, added) = pin_padding_at(&layout, first, Filler::Standard, &Pins::new());
    assert_eq!((added, pins.len()), (count - first, count - first));
    assert!(pins.values().all(|pin| pin.source == PinSource::Padding));
    let report = decode(&grid, &layout, mask, &pins, &DecodeOptions::default());
    assert!(report.is_trustworthy());
    assert_eq!(report.text.plain(), "hi");
    assert_eq!(pinned_filler(&layout, &report, &pins), Some(Filler::Standard));
    // Said again from further on, the pins before the new start go, and
    // a pin made by hand stays as it was made.
    let by_hand = Pin::full(0x5A, PinSource::Manual);
    let mut held = pins.clone();
    held.insert(layout.data_stream[10], by_hand);
    let (later, added) = pin_padding_at(&layout, first + 2, Filler::Standard, &held);
    assert_eq!(added, count - first - 3);
    assert!(!later.contains_key(&layout.data_stream[first]) && !later.contains_key(&layout.data_stream[first + 1]));
    assert_eq!(later[&layout.data_stream[10]], by_hand);
    // A start past the last data codeword pins nothing.
    assert_eq!(pin_padding_at(&layout, count + 3, Filler::Standard, &Pins::new()).1, 0);
    // What can be read says how well a start and a filler fit: on the
    // symbol as printed the padding matches bit for bit where it starts
    // and as what it is, and neither a codeword early nor as zeros.
    let whole = decode_plain(&clean, &layout, mask);
    let fit = padding_at(&layout, &whole, first, Filler::Standard);
    assert_eq!((fit.pad_codewords, fit.matches, fit.mismatches), (count - first, 8 * (count - first) as u32, 0));
    assert!(padding_at(&layout, &whole, first - 1, Filler::Standard).mismatches > 0);
    assert!(padding_at(&layout, &whole, first, Filler::Zeros).mismatches > 100);
    // A start past the end is held to the last data codeword.
    assert_eq!(padding_at(&layout, &whole, count + 3, Filler::Standard).first_pad_codeword, count - 1);
}

#[test]
fn full_symbols_have_nothing_to_recover() {
    let (version, ec, mask) = (SymbolVersion::Qr(4), EcLevel::L, 3);
    let layout = layout(version, ec);
    let mut rng = Rng(0xF011);
    let payload = rng.bytes(byte_capacity(&layout, 0));
    let mut grid = encode(version, ec, Some(mask), &[Segment::bytes(&payload)]).unwrap();
    assert_eq!(detect_padding(&layout, &decode_plain(&grid, &layout, mask)), None);
    erase_data_tail(&mut grid, &layout, 0, 25);
    assert!(recover(&grid, &layout, mask).is_none());
}

#[test]
fn manual_pins_are_kept_and_recovery_is_idempotent() {
    use qr_forensics::decode::codewords::Pin;
    let (version, ec, mask) = (SymbolVersion::Qr(10), EcLevel::M, 2);
    let layout = layout(version, ec);
    let mut grid = encode(version, ec, Some(mask), &[Segment::bytes("pins")]).unwrap();
    erase_data_tail(&mut grid, &layout, 3, 30);

    let mut pins = Pins::new();
    let manual_cw = *layout.blocks[3].codewords[..layout.blocks[3].data_len].last().unwrap();
    let truth = decode(
        &encode(version, ec, Some(mask), &[Segment::bytes("pins")]).unwrap(),
        &layout,
        mask,
        &Pins::new(),
        &DecodeOptions::default(),
    );
    let value = truth.codewords[manual_cw].final_value.unwrap();
    pins.insert(manual_cw, Pin::full(value, PinSource::Manual));

    let first = recover_padding(&grid, &layout, mask, &pins, &DecodeOptions::default()).unwrap();
    assert_eq!(first.pins[&manual_cw].source, PinSource::Manual);
    assert!(first.report.is_complete());
    let second = recover_padding(&grid, &layout, mask, &first.pins, &DecodeOptions::default()).unwrap();
    assert_eq!(second.pins, first.pins);
    assert_eq!(second.added, 0);
}

/// Make the first `count` modules of a codeword unknown: its high bits.
fn erase_leading_bits(grid: &mut ModuleGrid, layout: &SymbolLayout, cw: usize, count: usize) {
    for pos in layout.codeword_modules(cw).iter().take(count) {
        grid.set(pos.x as usize, pos.y as usize, Module::Unknown);
    }
}

#[test]
fn an_unreadable_header_with_one_reading_is_pinned() {
    // 9-M, "short message": a 12-bit byte header, 104 bits of text, the
    // terminator at bit 116 inside codeword 14, and padding from codeword
    // 15. Block 0 (36 data and 22 EC codewords) loses the low nibble of
    // the count (in codeword 1), the text in codewords 2 to 13, pad
    // codewords 16 to 27, and 10 of its EC codewords; codewords 14 and 15
    // stay readable.
    let (version, ec, mask) = (SymbolVersion::Qr(9), EcLevel::M, 3);
    let layout = layout(version, ec);
    let text = "short message";
    let clean = encode(version, ec, Some(mask), &[Segment::bytes(text)]).unwrap();
    let truth = decode_plain(&clean, &layout, mask);
    let EndOfData::At { bit: true_end, .. } = truth.parse.end else { panic!() };
    assert_eq!(true_end, 116);

    let mut grid = clean.clone();
    erase_leading_bits(&mut grid, &layout, layout.data_stream[1], 4);
    for j in (2..14).chain(16..28) {
        erase_codeword(&mut grid, &layout, layout.data_stream[j]);
    }
    let block = &layout.blocks[0];
    assert_eq!((block.data_len, block.ec_len), (36, 22));
    for &cw in &block.codewords[block.data_len..block.data_len + 10] {
        erase_codeword(&mut grid, &layout, cw);
    }

    let before = decode_plain(&grid, &layout, mask);
    assert!(!before.blocks[0].is_solved());
    // The count could read 0 to 15; only 13 reaches the terminator that
    // codeword 14 still shows and the padding after it. That one reading
    // is taken, marked as a guess, and what it says parses.
    let inference = before.parse.header_inference.clone().expect("a header was inferred");
    assert_eq!((inference.candidates, inference.guessed_bits, inference.complete), (1, 4, true));
    assert_eq!(inference.inferred, [(8, true), (9, true), (10, false), (11, true)]);
    assert_eq!(before.parse.segments.len(), 1);
    assert!(before.parse.segments[0].header_inferred);
    assert_eq!(before.parse.end, EndOfData::At { bit: 116, certainty: Certainty::Unknown });
    assert!(!before.score.all_verified);
    let candidate = detect_padding(&layout, &before).unwrap();
    assert_eq!((candidate.first_pad_codeword, candidate.end_bit, candidate.header_certainty), (15, Some(116), Some(Certainty::Unknown)));

    // Pinning the padding leaves block 0 one codeword short of plain
    // erasure decoding; the four inferred header bits complete codeword 1
    // and make up the difference.
    let recovery = recover(&grid, &layout, mask).unwrap();
    assert_eq!(recovery.header_pins, 1, "one codeword gained header bits");
    let header_pin = recovery.pins[&layout.data_stream[1]];
    assert_eq!((header_pin.source, header_pin.mask, header_pin.value), (PinSource::Header, 0xF0, 0xD0));
    assert_eq!(recovery.solved_after, 5, "{:?}", recovery.report.blocks);
    assert_eq!(recovery.report.text.plain(), text);
    // Nothing to spare: the solve is not verified, so the headers do not
    // count as confirmed.
    assert!(!recovery.report.blocks[0].is_verified());
    assert!(!recovery.confirmed);
    for (got, want) in recovery.report.codewords.iter().zip(&truth.codewords) {
        assert_eq!(got.final_value, want.final_value);
    }
    // The inferred pins go with the rest of the recovery's pins.
    let mut pins = recovery.pins.clone();
    clear_padding_pins(&mut pins);
    assert!(pins.is_empty());

    // With codeword 14 unreadable too, the terminator is not there to be
    // seen: every odd count ends in step with the padding if a terminator
    // is guessed for it, which explains nothing the stream shows. No
    // reading counts, nothing is inferred, and the stream stays lost.
    erase_codeword(&mut grid, &layout, layout.data_stream[14]);
    let before = decode_plain(&grid, &layout, mask);
    assert_eq!(before.parse.end, EndOfData::Lost { bit: 0, reason: LostReason::UnknownHeader });
    let inference = before.parse.header_inference.clone().unwrap();
    assert_eq!(inference.candidates, 0, "{inference:?}");
    assert!(inference.inferred.is_empty());
    assert_eq!(before.blocks_solved(), 4);
    assert_eq!(recover(&grid, &layout, mask).unwrap().solved_after, 4, "still one codeword short");
}

/// A grid as a photo of it: eight pixels a module on white, with its
/// quiet zone, and the outline of the whole photo.
fn photo_of(grid: &ModuleGrid) -> (Luma, [[f64; 2]; 4]) {
    let (w, h, pixels) = render_rgba(grid, 8, 4);
    let luma = Adjust::default().apply(&pixels, w, h).1;
    (luma, [[0.0, 0.0], [w as f64, 0.0], [w as f64, h as f64], [0.0, h as f64]])
}

#[test]
fn a_read_of_a_photo_pins_the_padding_when_that_is_what_it_takes() {
    // 10-M with a short address: five blocks, four of them all padding.
    let (version, ec, mask) = (SymbolVersion::Qr(10), EcLevel::M, 4);
    let text = "https://example.com/t/8F3K2";
    let clean = encode(version, ec, Some(mask), &[Segment::bytes(text)]).unwrap();
    // A strip of white tape down the symbol, over a third of its width:
    // every block has more wrong codewords than it corrects.
    let mut taped = clean.clone();
    for y in 0..taped.height() {
        for x in 12..32 {
            taped.set(x, y, Module::Light);
        }
    }
    let (params, opts) = (SampleParams::default(), DecodeOptions::default());
    let (image, whole) = photo_of(&taped);
    let read = read_outline(&image, SymbolVersion::Qr(2), whole, &params, &opts).expect("the symbol is found");
    assert_eq!(read.version(), version);
    assert!(read.is_trustworthy());
    assert!(read.padding_pinned());
    assert_eq!(read.text(), text);
    // The pins are the recovery's own, and without them the grid as
    // read solves no block.
    assert!(read.pins.values().all(|pin| !pin.source.is_by_hand()));
    let layout = layout(version, read.report.ec);
    assert_eq!(decode(&read.grid, &layout, read.report.mask, &Pins::new(), &opts).blocks_verified(), 0);
    assert_eq!(decode(&read.grid, &layout, read.report.mask, &read.pins, &opts).text.plain(), text);
    // A search of the photo reads it the same way.
    let found = find_symbols_with(&image, Family::Qr, &params, &SearchParams::default());
    let read = found.into_iter().find_map(|found| read_detection(&image, found, &params, &opts)).expect("the symbol is found");
    assert!(read.is_trustworthy() && read.padding_pinned());
    assert_eq!(read.text(), text);
    // A symbol that error correction reads is read by it, and nothing
    // is pinned.
    let (image, whole) = photo_of(&clean);
    let read = read_outline(&image, SymbolVersion::Qr(2), whole, &params, &opts).unwrap();
    assert!(read.is_trustworthy() && !read.padding_pinned());
    assert!(read.pins.is_empty());
    // Tape over more than half the symbol leaves too little of the
    // error correction codewords to bear the padding out: no read is
    // claimed.
    for y in 0..taped.height() {
        for x in 8..40 {
            taped.set(x, y, Module::Light);
        }
    }
    let (image, whole) = photo_of(&taped);
    let read = read_outline(&image, SymbolVersion::Qr(2), whole, &params, &opts);
    assert!(read.is_none_or(|read| !read.is_trustworthy() && !read.padding_pinned()));
}
