//! Injected errors and erasures: exact recovery inside capacity, honest
//! failure outside it, and partial text when a block is lost.

mod common;

use common::*;
use qr_forensics::decode::bitstream::Certainty;
use qr_forensics::decode::codewords::Pins;
use qr_forensics::decode::pipeline::{BlockOutcome, CwStatus, DecodeOptions, SolveMethod, decode};
use qr_forensics::decode::rs::{CHASE_BITS, RsFailure};
use qr_forensics::decode::text::UNKNOWN_CHAR;
use qr_forensics::symbol::grid::{CERTAIN, Module, ModuleGrid};
use qr_forensics::symbol::layout::SymbolLayout;
use qr_forensics::symbol::{EcLevel, SymbolVersion};
use qrcode_generator::Segment;

fn sample(version: SymbolVersion, ec: EcLevel, mask: u8, rng: &mut Rng) -> (SymbolLayout, ModuleGrid, String) {
    let layout = layout(version, ec);
    let text = rng.alphanumeric(byte_capacity(&layout, 1).max(1)).to_lowercase();
    let grid = encode(version, ec, Some(mask), &[Segment::bytes(&text)]).unwrap();
    (layout, grid, text)
}

const SAMPLES: [(SymbolVersion, EcLevel); 10] = [
    (SymbolVersion::Qr(1), EcLevel::H),
    (SymbolVersion::Qr(4), EcLevel::M),
    (SymbolVersion::Qr(8), EcLevel::Q),
    (SymbolVersion::Qr(15), EcLevel::L),
    (SymbolVersion::Qr(25), EcLevel::H),
    (SymbolVersion::Micro(4), EcLevel::Q),
    (SymbolVersion::Micro(3), EcLevel::M),
    (SymbolVersion::Rmqr(9), EcLevel::H),
    (SymbolVersion::Rmqr(20), EcLevel::M),
    (SymbolVersion::Rmqr(31), EcLevel::H),
];

#[test]
fn recovers_exactly_within_capacity() {
    let mut rng = Rng(0xDA4A6E);
    for (version, ec) in SAMPLES {
        for round in 0..20 {
            let mask = rng.below(version.mask_count() as usize) as u8;
            let (layout, clean, text) = sample(version, ec, mask, &mut rng);
            let mut grid = clean.clone();
            let mut plan = Vec::new();
            for block in &layout.blocks {
                // Leave at least one codeword of slack, and the ones the
                // standard reserves, so the result is verified.
                let budget = block.ec_len - block.reserved.max(1);
                let erasures = rng.below(budget + 1);
                let errors = rng.below((budget - erasures) / 2 + 1);
                let picked = rng.distinct(erasures + errors, block.codewords.len());
                for (n, &i) in picked.iter().enumerate() {
                    let cw = block.codewords[i];
                    if n < erasures {
                        erase_codeword(&mut grid, &layout, cw);
                    } else {
                        corrupt_codeword(&mut grid, &layout, cw, rng.below(8));
                    }
                }
                plan.push((erasures, errors, picked));
            }

            let what = format!("{version}-{ec:?} round {round}");
            let report = decode_plain(&grid, &layout, mask);
            for (b, (erasures, errors, picked)) in plan.iter().enumerate() {
                let block = &layout.blocks[b];
                let slack = block.ec_len - erasures - 2 * errors;
                match report.blocks[b].outcome {
                    BlockOutcome::Solved { method, errors: e, erasures: f, slack: s, confidence_bits, beyond_standard } => {
                        assert_eq!((method, e, f, s), (SolveMethod::ReedSolomon, *errors, *erasures, slack), "{what} block {b}");
                        assert_eq!(beyond_standard, slack < block.reserved, "{what} block {b}");
                        // An erased four-bit codeword adds its fixed low
                        // nibble as four check bits; nothing else can.
                        assert!((8 * slack as u32..=8 * slack as u32 + 4).contains(&confidence_bits));
                    }
                    other => panic!("{what} block {b}: {other:?}"),
                }
                for (i, &cw) in block.codewords.iter().enumerate() {
                    let want = match picked.iter().position(|&p| p == i) {
                        Some(n) if n < *erasures => CwStatus::Filled,
                        Some(_) => CwStatus::Corrected,
                        None => CwStatus::Clean,
                    };
                    assert_eq!(report.codewords[cw].status, want, "{what} block {b} codeword {i}");
                }
            }
            // Every corrected value matches what the clean symbol holds.
            let truth = decode_plain(&clean, &layout, mask);
            for (cw, (got, want)) in report.codewords.iter().zip(&truth.codewords).enumerate() {
                assert_eq!(got.final_value, want.final_value, "{what} codeword {cw}");
            }
            assert_eq!(report.text.plain(), text, "{what}");
            assert!(report.is_trustworthy(), "{what}");
        }
    }
}

#[test]
fn zero_slack_solutions_are_flagged_unverified() {
    let mut rng = Rng(0x51AC);
    let (layout, mut grid, text) = sample(SymbolVersion::Qr(4), EcLevel::M, 3, &mut rng);
    for block in &layout.blocks {
        for i in rng.distinct(block.ec_len, block.codewords.len()) {
            erase_codeword(&mut grid, &layout, block.codewords[i]);
        }
    }
    let report = decode_plain(&grid, &layout, 3);
    assert!(report.is_complete());
    assert!(report.blocks.iter().all(|b| !b.is_verified()));
    assert!(!report.is_trustworthy());
    assert_eq!(report.text.plain(), text);
    assert!(report.text.spans.iter().all(|s| s.certainty == Certainty::Unverified));
}

#[test]
fn a_zero_slack_solution_is_improved_by_flipping_the_faint_bit() {
    // Sixteen codewords of a 4-M block erased and one more read wrong and
    // faint: exact decoding takes the wrong one as an error and lands on
    // the right values with nothing to spare. Doubting the faint codeword
    // instead leaves a codeword of slack, and flipping its faint bit two,
    // and the solution is verified. An exact solution used to end the
    // search before doubting was tried.
    let mut rng = Rng(0x51AD);
    let (layout, clean, text) = sample(SymbolVersion::Qr(4), EcLevel::M, 5, &mut rng);
    let block = &layout.blocks[0];
    assert_eq!(block.ec_len, 18);
    let mut grid = clean.clone();
    for y in 0..grid.height() {
        for x in 0..grid.width() {
            grid.set_sampled(x, y, grid.get(x, y), 200);
        }
    }
    let picked = rng.distinct(block.ec_len - 1, block.codewords.len());
    for &i in &picked[..block.ec_len - 2] {
        erase_codeword(&mut grid, &layout, block.codewords[i]);
    }
    let faint = layout.codeword_modules(block.codewords[picked[block.ec_len - 2]])[3];
    let (x, y) = (faint.x as usize, faint.y as usize);
    grid.set_sampled(x, y, grid.get(x, y).inverted(), 12);

    let strict = DecodeOptions { reliability: false, ..Default::default() };
    let exact = decode(&grid, &layout, 5, &Pins::new(), &strict);
    match exact.blocks[0].outcome {
        BlockOutcome::Solved { method: SolveMethod::ReedSolomon, errors: 1, erasures: 16, slack: 0, .. } => {}
        other => panic!("{other:?}"),
    }
    assert!(!exact.blocks[0].is_verified());
    assert_eq!(exact.text.plain(), text);

    let report = decode_plain(&grid, &layout, 5);
    match report.blocks[0].outcome {
        BlockOutcome::Solved { method: SolveMethod::Chase { flipped_bits: 1 }, errors: 1, erasures: 16, slack: 2, .. } => {}
        other => panic!("{other:?}"),
    }
    assert!(report.blocks[0].is_verified());
    assert!(report.is_trustworthy());
    assert_eq!(report.text.plain(), text);
    assert_eq!(report.codewords.iter().filter(|c| c.status == CwStatus::Corrected).count(), 1);
}

#[test]
fn erasures_past_capacity_fail_deterministically() {
    let mut rng = Rng(0xFA11);
    for (version, ec) in SAMPLES {
        let (layout, mut grid, _) = sample(version, ec, 0, &mut rng);
        let block = &layout.blocks[0];
        for i in rng.distinct(block.ec_len + 1, block.codewords.len()) {
            erase_codeword(&mut grid, &layout, block.codewords[i]);
        }
        let report = decode_plain(&grid, &layout, 0);
        assert_eq!(
            report.blocks[0].outcome,
            BlockOutcome::Failed(RsFailure::TooManyErasures { erasures: block.ec_len + 1, capacity: block.ec_len }),
            "{version}-{ec:?}"
        );
        assert_eq!(report.blocks[0].codewords_short(), 1);
        assert!(report.blocks[1..].iter().all(|b| b.is_verified()));
        assert!(!report.is_complete());
    }
}

#[test]
fn scattered_unknown_modules_are_solved_bit_by_bit() {
    let mut rng = Rng(0xB175);
    for (version, ec) in SAMPLES {
        let mask = rng.below(version.mask_count() as usize) as u8;
        let (layout, mut grid, text) = sample(version, ec, mask, &mut rng);
        // One unreadable module in most codewords of every block: hopeless
        // for codeword erasures, easy at the bit level.
        for block in &layout.blocks {
            let hits = (2 * block.ec_len).min(block.codewords.len());
            for i in rng.distinct(hits, block.codewords.len()) {
                let modules = layout.codeword_modules(block.codewords[i]);
                let pos = modules[rng.below(modules.len())];
                grid.set(pos.x as usize, pos.y as usize, Module::Unknown);
            }
        }
        let what = format!("{version}-{ec:?}");
        let report = decode_plain(&grid, &layout, mask);
        for (b, block) in report.blocks.iter().enumerate() {
            let hits = (2 * block.ec_len).min(block.data_len + block.ec_len);
            if hits > block.ec_len {
                assert!(
                    matches!(block.outcome, BlockOutcome::Solved { method: SolveMethod::BitLevel, .. }),
                    "{what} block {b}: {:?}",
                    block.outcome
                );
            }
            assert!(block.is_verified(), "{what} block {b}");
        }
        assert_eq!(report.text.plain(), text, "{what}");

        // With the fallback off the same grid does not decode.
        if layout.blocks.iter().any(|b| 2 * b.ec_len <= b.codewords.len() && b.ec_len > 0) {
            let strict = DecodeOptions { bit_level: false, ..Default::default() };
            assert!(!decode(&grid, &layout, mask, &Pins::new(), &strict).is_complete(), "{what}");
        }
    }
}

#[test]
fn partly_readable_codewords_past_capacity_still_yield_inferred_bits() {
    let mut rng = Rng(0x1F3);
    let mut inferred_total = 0;
    for (version, ec) in [(SymbolVersion::Qr(4), EcLevel::M), (SymbolVersion::Qr(12), EcLevel::L), (SymbolVersion::Rmqr(20), EcLevel::M)] {
        let mask = rng.below(version.mask_count() as usize) as u8;
        let (layout, clean, _) = sample(version, ec, mask, &mut rng);
        let truth = decode_plain(&clean, &layout, mask);
        let mut grid = clean.clone();
        let block = &layout.blocks[0];
        // One codeword more than can be erased, with seven readable modules
        // left among them: one unknown bit more than there are equations.
        // No solver closes the block, but the solutions form a single line,
        // and the bits it does not touch are settled.
        for (n, i) in rng.distinct(block.ec_len + 1, block.codewords.len()).into_iter().enumerate() {
            let cw = block.codewords[i];
            for (bit, pos) in layout.codeword_modules(cw).iter().enumerate() {
                if n >= 7 || bit != n {
                    grid.set(pos.x as usize, pos.y as usize, Module::Unknown);
                }
            }
        }
        let what = format!("{version}-{ec:?}");
        let report = decode_plain(&grid, &layout, mask);
        let first = &report.blocks[0];
        assert_eq!(first.unknown_bits, 8 * block.ec_len + 1, "{what}");
        assert!(!first.is_solved(), "{what}");
        assert!(first.inferred_bits > 0 && first.inferred_bits < first.unknown_bits, "{what}: {first:?}");
        inferred_total += first.inferred_bits;
        let mut inferred_codewords = 0;
        for &cw in &block.codewords {
            let item = &report.codewords[cw];
            assert!(item.final_value.is_none());
            assert_eq!(item.inferred_known & item.pinned_known, 0, "{what}: inference only fills unknown bits");
            if item.inferred_known != 0 {
                assert_eq!(item.status, CwStatus::Inferred, "{what}");
                let want = truth.codewords[cw].final_value.unwrap();
                assert_eq!(item.inferred_value, want & item.inferred_known, "{what} codeword {cw}");
                inferred_codewords += 1;
            } else {
                assert!(matches!(item.status, CwStatus::Unresolved | CwStatus::Unverified), "{what}");
            }
        }
        assert!(inferred_codewords > 0, "{what}");
        // Inferred bits reach the stream as unverified, never as verified.
        for &cw in &block.codewords[..block.data_len] {
            let item = &report.codewords[cw];
            let j = layout.stream_index(cw).unwrap();
            for bit in 0..8 {
                let mask_bit = 0x80 >> bit;
                let certainty = report.stream.certainty[8 * j + bit];
                if item.inferred_known & mask_bit != 0 {
                    assert_eq!(certainty, Certainty::Unverified, "{what}");
                    assert_eq!(report.stream.values[8 * j + bit], item.inferred_value & mask_bit != 0);
                } else if item.pinned_known & mask_bit == 0 {
                    assert_eq!(certainty, Certainty::Unknown, "{what}");
                }
            }
        }
        // Switching the bit-level solver off removes the inference.
        let strict = DecodeOptions { bit_level: false, ..Default::default() };
        let plain = decode(&grid, &layout, mask, &Pins::new(), &strict);
        assert_eq!(plain.blocks[0].inferred_bits, 0);
    }
    assert!(inferred_total >= 30, "{inferred_total}");
}

#[test]
fn doubting_recovers_from_misread_sampled_modules() {
    let mut rng = Rng(0xD0B7);
    for (version, ec) in SAMPLES {
        let mask = rng.below(version.mask_count() as usize) as u8;
        let (layout, clean, text) = sample(version, ec, mask, &mut rng);
        let block = &layout.blocks[0];
        if block.ec_len < 4 {
            continue;
        }
        let mut grid = clean.clone();
        // Every module counts as sampled and confident...
        for y in 0..grid.height() {
            for x in 0..grid.width() {
                grid.set_sampled(x, y, grid.get(x, y), 200);
            }
        }
        // ...except one module in each of more codewords than errors alone
        // can fix, which read wrong and faint. One short of the EC count,
        // so that erasing them all still leaves something to verify with.
        let wrong = block.ec_len / 2 + 1 + rng.below((block.ec_len / 2 - 1).max(1));
        for i in rng.distinct(wrong, block.codewords.len()) {
            let modules = layout.codeword_modules(block.codewords[i]);
            let pos = modules[rng.below(modules.len())];
            let (x, y) = (pos.x as usize, pos.y as usize);
            grid.set_sampled(x, y, grid.get(x, y).inverted(), 5 + rng.below(20) as u8);
        }
        let what = format!("{version}-{ec:?} with {wrong} misread");
        let strict = DecodeOptions { reliability: false, ..Default::default() };
        assert!(!decode(&grid, &layout, mask, &Pins::new(), &strict).blocks[0].is_solved(), "{what}");
        let report = decode_plain(&grid, &layout, mask);
        // Doubting the faint codewords, doubting the faint bits and
        // flipping them all reach the answer; whichever is confirmed by
        // more bits is kept.
        match report.blocks[0].outcome {
            BlockOutcome::Solved { method: SolveMethod::Reliability { doubted }, .. } => {
                assert!(doubted >= wrong && doubted <= block.ec_len, "{what}: doubted {doubted}");
            }
            BlockOutcome::Solved { method: SolveMethod::BitReliability { doubted_bits }, .. } => {
                assert!(doubted_bits >= wrong && doubted_bits < 8 * block.ec_len, "{what}: doubted {doubted_bits} bits");
            }
            BlockOutcome::Solved { method: SolveMethod::Chase { flipped_bits }, errors, .. } => {
                assert!(flipped_bits >= 1 && flipped_bits <= wrong.min(CHASE_BITS), "{what}: flipped {flipped_bits} bits");
                assert_eq!(errors, wrong, "{what}");
            }
            other => panic!("{what}: {other:?}"),
        }
        assert!(report.is_trustworthy(), "{what}");
        assert_eq!(report.text.plain(), text, "{what}");
        let truth = decode_plain(&clean, &layout, mask);
        for (got, want) in report.codewords.iter().zip(&truth.codewords) {
            assert_eq!(got.final_value, want.final_value, "{what}");
        }
        assert_eq!(report.codewords.iter().filter(|c| c.status == CwStatus::Corrected).count(), wrong, "{what}");
    }
    // A certain module is never doubted, so painting a wrong module by hand
    // keeps the failure.
    let (version, ec, mask) = (SymbolVersion::Qr(4), EcLevel::M, 1);
    let (layout, mut grid, _) = sample(version, ec, mask, &mut rng);
    let block = &layout.blocks[0];
    for i in 0..block.ec_len / 2 + 1 {
        corrupt_codeword(&mut grid, &layout, block.codewords[i], i);
    }
    assert!(grid.cells().iter().all(|_| true) && grid.reliability(0, 0) == CERTAIN);
    assert!(!decode_plain(&grid, &layout, mask).blocks[0].is_solved());
}

#[test]
fn doubting_bits_recovers_faint_modules_in_more_codewords_than_erasures_allow() {
    let mut rng = Rng(0xB17D);
    for (version, ec) in SAMPLES {
        let mask = rng.below(version.mask_count() as usize) as u8;
        let (layout, clean, text) = sample(version, ec, mask, &mut rng);
        let block = &layout.blocks[0];
        if block.ec_len < 4 {
            continue;
        }
        let mut grid = clean.clone();
        for y in 0..grid.height() {
            for x in 0..grid.width() {
                grid.set_sampled(x, y, grid.get(x, y), 200);
            }
        }
        // One faint wrong module in more codewords than there are EC
        // codewords: beyond any erasure decoder, since erasing them all
        // would leave nothing to solve with, but a few dozen bits of the
        // block's hundreds of equations.
        let wrong = (block.ec_len + 3).min(block.codewords.len() - 1);
        for i in rng.distinct(wrong, block.codewords.len()) {
            let modules = layout.codeword_modules(block.codewords[i]);
            let pos = modules[rng.below(modules.len())];
            let (x, y) = (pos.x as usize, pos.y as usize);
            grid.set_sampled(x, y, grid.get(x, y).inverted(), 5 + rng.below(20) as u8);
        }
        let what = format!("{version}-{ec:?} with {wrong} misread");
        let strict = DecodeOptions { reliability: false, ..Default::default() };
        assert!(!decode(&grid, &layout, mask, &Pins::new(), &strict).blocks[0].is_solved(), "{what}");
        let report = decode_plain(&grid, &layout, mask);
        match report.blocks[0].outcome {
            BlockOutcome::Solved { method: SolveMethod::BitReliability { doubted_bits }, errors, confidence_bits, .. } => {
                assert!(doubted_bits >= wrong && doubted_bits < 8 * block.ec_len, "{what}: doubted {doubted_bits} bits");
                assert_eq!(errors, wrong, "{what}");
                assert!(confidence_bits as usize >= 8 * block.ec_len - doubted_bits, "{what}: {confidence_bits} checks");
            }
            other => panic!("{what}: {other:?}"),
        }
        assert!(report.is_trustworthy(), "{what}");
        assert_eq!(report.text.plain(), text, "{what}");
        let truth = decode_plain(&clean, &layout, mask);
        for (got, want) in report.codewords.iter().zip(&truth.codewords) {
            assert_eq!(got.final_value, want.final_value, "{what}");
        }
        assert_eq!(report.codewords.iter().filter(|c| c.status == CwStatus::Corrected).count(), wrong, "{what}");
        // With the bit-level solver off there is nothing to doubt bits with.
        let no_bits = DecodeOptions { bit_level: false, ..Default::default() };
        assert!(!decode(&grid, &layout, mask, &Pins::new(), &no_bits).blocks[0].is_solved(), "{what}");
    }
    // A hand-painted grid has nothing faint in it: the same damage painted
    // by hand is certain, and stays a failure.
    let (version, ec, mask) = (SymbolVersion::Qr(8), EcLevel::Q, 2);
    let (layout, mut grid, _) = sample(version, ec, mask, &mut rng);
    let block = &layout.blocks[0];
    for i in 0..block.ec_len + 3 {
        corrupt_codeword(&mut grid, &layout, block.codewords[i], i);
    }
    assert!(!grid.has_reliability());
    assert!(matches!(decode_plain(&grid, &layout, mask).blocks[0].outcome, BlockOutcome::Failed(_)));
}

#[test]
fn chasing_flips_faint_bits_where_erasing_them_costs_too_much() {
    let mut rng = Rng(0xC4A5E);
    let mut chased = 0;
    for (version, ec) in SAMPLES {
        let mask = rng.below(version.mask_count() as usize) as u8;
        let (layout, clean, text) = sample(version, ec, mask, &mut rng);
        let block = &layout.blocks[0];
        if block.ec_len < 4 {
            continue;
        }
        let mut grid = clean.clone();
        for y in 0..grid.height() {
            for x in 0..grid.width() {
                grid.set_sampled(x, y, grid.get(x, y), 200);
            }
        }
        // As many solid wrong codewords as the code corrects with two
        // codewords to spare (what a chased solution must leave), at the
        // end of the block where doubting by index never reaches them,
        // and the two or three beyond that wrong in one faint module
        // each, beside a faint module that is right: more wrong codewords
        // than the code corrects, and erasing the faint ones costs more
        // redundancy than is left.
        let n = block.codewords.len();
        let solid = (block.ec_len - block.reserved.max(2)) / 2;
        let faint = block.ec_len / 2 + 2 - solid;
        for (k, i) in (n - solid..n).enumerate() {
            corrupt_codeword(&mut grid, &layout, block.codewords[i], k);
        }
        for (k, i) in rng.distinct(faint + 1, n - solid).into_iter().enumerate() {
            let modules = layout.codeword_modules(block.codewords[i]);
            let pos = modules[rng.below(modules.len())];
            let (x, y) = (pos.x as usize, pos.y as usize);
            let module = if k < faint { grid.get(x, y).inverted() } else { grid.get(x, y) };
            grid.set_sampled(x, y, module, 5 + rng.below(20) as u8);
        }
        let what = format!("{version}-{ec:?} with {solid} solid and {faint} faint misreads");
        let strict = DecodeOptions { reliability: false, ..Default::default() };
        assert!(!decode(&grid, &layout, mask, &Pins::new(), &strict).blocks[0].is_solved(), "{what}");
        let report = decode_plain(&grid, &layout, mask);
        match report.blocks[0].outcome {
            BlockOutcome::Solved { method: SolveMethod::Chase { flipped_bits }, errors, slack, .. } => {
                assert_eq!((flipped_bits, errors, slack), (faint, solid + faint, block.ec_len - 2 * solid), "{what}");
                chased += 1;
            }
            BlockOutcome::Solved { .. } => {}
            other => panic!("{what}: {other:?}"),
        }
        assert!(report.is_trustworthy(), "{what}");
        assert_eq!(report.text.plain(), text, "{what}");
        let truth = decode_plain(&clean, &layout, mask);
        for (got, want) in report.codewords.iter().zip(&truth.codewords) {
            assert_eq!(got.final_value, want.final_value, "{what}");
        }
        assert_eq!(report.codewords.iter().filter(|c| c.status == CwStatus::Corrected).count(), solid + faint, "{what}");
    }
    assert!(chased >= 1, "no block was solved by chasing");
    // Painted by hand, every module is certain: nothing to flip, and the
    // same count of wrong codewords stays a failure.
    let (version, ec, mask) = (SymbolVersion::Qr(8), EcLevel::Q, 2);
    let (layout, mut grid, _) = sample(version, ec, mask, &mut rng);
    let block = &layout.blocks[0];
    for i in 0..block.ec_len / 2 + 2 {
        corrupt_codeword(&mut grid, &layout, block.codewords[i], i);
    }
    assert!(!grid.has_reliability());
    assert!(matches!(decode_plain(&grid, &layout, mask).blocks[0].outcome, BlockOutcome::Failed(_)));
}

#[test]
fn a_lost_block_still_yields_the_rest_of_the_text() {
    let mut rng = Rng(0x10057);
    // 10-M has five blocks; the message is laid out block after block.
    let (layout, mut grid, text) = sample(SymbolVersion::Qr(10), EcLevel::M, 2, &mut rng);
    assert_eq!(layout.blocks.len(), 5);
    let lost = &layout.blocks[2];
    // Erase the whole data part of the middle block.
    for &cw in &lost.codewords[..lost.data_len] {
        erase_codeword(&mut grid, &layout, cw);
    }
    let report = decode_plain(&grid, &layout, 2);
    assert_eq!(report.blocks_solved(), 4);
    assert!(!report.blocks[2].is_solved());

    let plain = report.text.plain();
    assert_eq!(plain.chars().count(), text.chars().count());
    let unknown = plain.chars().filter(|&c| c == UNKNOWN_CHAR).count();
    // Byte boundaries do not line up with codewords, so the damage can reach
    // one character past each end of the block.
    assert!((lost.data_len - 1..=lost.data_len + 1).contains(&unknown), "{unknown}");
    for (got, want) in plain.chars().zip(text.chars()) {
        assert!(got == want || got == UNKNOWN_CHAR);
    }
    let (verified, _, missing) = report.text.counts();
    assert_eq!(missing, unknown);
    assert!(verified >= text.len() - lost.data_len - 1);
    assert!(!report.is_trustworthy());
}

#[test]
fn a_misread_block_is_shown_as_unverified_text() {
    let mut rng = Rng(0xBAD);
    let (layout, mut grid, text) = sample(SymbolVersion::Qr(10), EcLevel::M, 5, &mut rng);
    let block = &layout.blocks[1];
    // Wrong modules in more codewords than the block can correct, all in its
    // EC part, so the data is intact but cannot be confirmed.
    for i in 0..block.ec_len / 2 + 2 {
        corrupt_codeword(&mut grid, &layout, block.codewords[block.data_len + i], i);
    }
    let report = decode_plain(&grid, &layout, 5);
    assert!(matches!(report.blocks[1].outcome, BlockOutcome::Failed(_)), "{:?}", report.blocks[1].outcome);
    assert_eq!(report.text.plain(), text);
    let (verified, unverified, unknown) = report.text.counts();
    assert_eq!(unknown, 0);
    assert!((block.data_len - 1..=block.data_len + 1).contains(&unverified));
    assert_eq!(verified + unverified, text.len());
    for &cw in &block.codewords {
        assert_eq!(report.codewords[cw].status, CwStatus::Unverified);
    }
}

#[test]
fn garbage_never_panics() {
    let mut rng = Rng(0xF022);
    for (version, ec) in SAMPLES {
        let layout = layout(version, ec);
        for _ in 0..40 {
            let mut grid = ModuleGrid::new(layout.func.width, layout.func.height);
            let unknown_rate = rng.below(4);
            for cell in grid.cells_mut() {
                *cell = match rng.below(8) {
                    n if n < unknown_rate => Module::Unknown,
                    n if n % 2 == 0 => Module::Dark,
                    _ => Module::Light,
                };
            }
            let mut pins = Pins::new();
            for _ in 0..rng.below(6) {
                use qr_forensics::decode::codewords::{Pin, PinSource};
                let pin = Pin { mask: rng.next() as u8, value: rng.next() as u8, source: PinSource::Manual };
                pins.insert(rng.below(layout.codewords.len()), pin);
            }
            for mask in 0..version.mask_count() {
                let report = decode(&grid, &layout, mask, &pins, &DecodeOptions::default());
                let _ = report.text.plain();
                let _ = qr_forensics::decode::padding::recover_padding(
                    &grid,
                    &layout,
                    mask,
                    &pins,
                    &DecodeOptions::default(),
                );
            }
            let _ = qr_forensics::decode::pipeline::analyze(&grid, &layout.func);
        }
    }
}

/// Development aid, not a test: how long brute force takes on the largest
/// symbol with faint misreads. Run with `--ignored --nocapture`.
#[test]
#[ignore]
fn brute_force_timing() {
    use std::time::Instant;
    let mut rng = Rng(0x7131);
    let (version, ec, mask) = (SymbolVersion::Qr(40), EcLevel::H, 3);
    let (layout, clean, text) = sample(version, ec, mask, &mut rng);
    assert_eq!(layout.blocks.len(), 81);
    for (name, per_block) in [("clean", 0usize), ("few faint misreads", 4), ("faint misreads past erasure capacity", 33)] {
        let mut grid = clean.clone();
        for y in 0..grid.height() {
            for x in 0..grid.width() {
                grid.set_sampled(x, y, grid.get(x, y), 200);
            }
        }
        let mut wrong = 0;
        for block in &layout.blocks {
            for i in rng.distinct(per_block.min(block.codewords.len()), block.codewords.len()) {
                let modules = layout.codeword_modules(block.codewords[i]);
                let pos = modules[rng.below(modules.len())];
                let (x, y) = (pos.x as usize, pos.y as usize);
                grid.set_sampled(x, y, grid.get(x, y).inverted(), 5 + rng.below(20) as u8);
                wrong += 1;
            }
        }
        let start = Instant::now();
        let report = decode_plain(&grid, &layout, mask);
        let single = start.elapsed();
        let start = Instant::now();
        let attempts = qr_forensics::decode::pipeline::brute_force(&grid, &layout.func, None, &DecodeOptions::default());
        let brute = start.elapsed();
        let best = &attempts[0];
        println!(
            "{name}: {wrong} faint wrong modules: single decode {single:?} ({} of 81 blocks verified); brute force {brute:?}: best {:?} mask {} {}, {} verified, text {}",
            report.blocks_verified(),
            best.ec,
            best.mask,
            if best.report.is_trustworthy() { "trustworthy" } else { "not trustworthy" },
            best.report.blocks_verified(),
            if best.report.text.plain() == text { "right" } else { "wrong" },
        );
    }
}
