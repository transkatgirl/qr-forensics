//! Injected errors and erasures: exact recovery inside capacity, honest
//! failure outside it, and partial text when a block is lost.

mod common;

use common::*;
use qr_forensics::decode::bitstream::Certainty;
use qr_forensics::decode::codewords::Pins;
use qr_forensics::decode::pipeline::{BlockOutcome, CwStatus, DecodeOptions, SolveMethod, decode};
use qr_forensics::decode::rs::{
    AIDED_VERIFIED_BITS, CHASE_BITS, RsFailure, TRIAL_SETS, VERIFIED_BITS, located_error_bits, sets_of, tries_charge,
};
use qr_forensics::decode::text::UNKNOWN_CHAR;
use qr_forensics::symbol::grid::{CERTAIN, Module, ModuleGrid};
use qr_forensics::symbol::layout::SymbolLayout;
use qr_forensics::symbol::{EcLevel, Family, SymbolVersion};
use qrcode_generator::Segment;

fn sample(version: SymbolVersion, ec: EcLevel, mask: u8, rng: &mut Rng) -> (SymbolLayout, ModuleGrid, String) {
    let layout = layout(version, ec);
    let text = rng.alphanumeric(byte_capacity(&layout, 1).max(1)).to_lowercase();
    let grid = encode(version, ec, Some(mask), &[Segment::bytes(&text)]).unwrap();
    (layout, grid, text)
}

/// Decoding without what the message makes likely of its bits: error
/// correction and the doubting of what was read, on their own, for the
/// tests that look at how far one way of solving reaches. The message's
/// evidence takes the high bit of every unread byte of the lowercase
/// text these symbols hold ([`sample`]) at nought, an equation a byte,
/// and reaches past them.
fn unguided() -> DecodeOptions {
    DecodeOptions { message_evidence: false, ..DecodeOptions::default() }
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
                // standard reserves, so the result is within the
                // standard, and verified where every codeword was read.
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
            let report = decode(&grid, &layout, mask, &Pins::new(), &unguided());
            let mut confirmed = true;
            for (b, (erasures, errors, picked)) in plan.iter().enumerate() {
                let block = &layout.blocks[b];
                let slack = block.ec_len - erasures - 2 * errors;
                match report.blocks[b].outcome {
                    BlockOutcome::Solved { method, errors: e, erasures: f, slack: s, confidence_bits, beyond_standard, .. } => {
                        assert_eq!((method, e, f, s), (SolveMethod::ReedSolomon, *errors, *erasures, slack), "{what} block {b}");
                        assert_eq!(beyond_standard, slack < block.reserved, "{what} block {b}");
                        // The slack, and what the errors located among
                        // the codewords read confirm. An erased four-bit
                        // codeword adds its fixed low nibble as four
                        // check bits; nothing else can.
                        let sure = 8 * slack as u32 + located_error_bits(block.codewords.len() - erasures, *errors);
                        assert!((sure..=sure + 4).contains(&confidence_bits), "{what} block {b}: {confidence_bits}");
                        // A block read whole is verified by any of that,
                        // and one with codewords filled in by three
                        // codewords' worth.
                        let needed = if *erasures == 0 { VERIFIED_BITS } else { AIDED_VERIFIED_BITS };
                        // Verified by that, or by the soft count of the
                        // bits it overturned, which may be the more.
                        assert!(report.blocks[b].is_verified() >= (confidence_bits >= needed), "{what} block {b}: {confidence_bits}");
                        confirmed &= report.blocks[b].is_verified();
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
            assert_eq!(report.is_trustworthy(), confirmed, "{what}");
        }
    }
}

#[test]
fn a_block_corrected_to_the_limit_of_its_error_correction_is_confirmed() {
    // As many wrong codewords in every block as the standard has its
    // error correction correct, which where it reserves none, and their
    // count is even, is every one of them spent: the blocks solve with
    // nothing to spare, and are confirmed by the errors they located, as
    // every reader that follows the standard takes them. One wrong
    // codeword more in a block is past the code, and no read, but for
    // the trials of erasures, which reach one more in any block, by
    // the smallest set of erasures that goes past the code or the next
    // where the standard's reserve refuses that one, when enough is
    // located to confirm it.
    let mut rng = Rng(0x11A17);
    for (version, ec) in SAMPLES {
        let mask = rng.below(version.mask_count() as usize) as u8;
        let (layout, clean, text) = sample(version, ec, mask, &mut rng);
        let mut grid = clean.clone();
        let mut spare = Vec::new();
        for block in &layout.blocks {
            let limit = (block.ec_len - block.reserved) / 2;
            let picked = rng.distinct(limit + 1, block.codewords.len());
            for &i in &picked[..limit] {
                corrupt_codeword(&mut grid, &layout, block.codewords[i], rng.below(8));
            }
            spare.push(block.codewords[picked[limit]]);
        }
        let what = format!("{version}-{ec:?}");
        let report = decode_plain(&grid, &layout, mask);
        for (b, block) in layout.blocks.iter().enumerate() {
            let limit = (block.ec_len - block.reserved) / 2;
            match report.blocks[b].outcome {
                BlockOutcome::Solved { method: SolveMethod::ReedSolomon, errors, erasures: 0, slack, confidence_bits, beyond_standard: false, .. } => {
                    assert_eq!((errors, slack), (limit, block.ec_len - 2 * limit), "{what} block {b}");
                    assert_eq!(confidence_bits, 8 * slack as u32 + located_error_bits(block.codewords.len(), limit), "{what} block {b}");
                }
                other => panic!("{what} block {b}: {other:?}"),
            }
            assert!(report.blocks[b].is_verified(), "{what} block {b}");
        }
        assert!(report.is_trustworthy(), "{what}");
        assert_eq!(report.text.plain(), text, "{what}");

        corrupt_codeword(&mut grid, &layout, spare[0], rng.below(8));
        let no_trial = DecodeOptions { erase_by_trial: false, list_decode: false, ..Default::default() };
        let report = decode(&grid, &layout, mask, &Pins::new(), &no_trial);
        assert!(!report.blocks[0].is_verified(), "{what}: {:?}", report.blocks[0].outcome);
        assert!(!report.is_trustworthy(), "{what}");
        // The trials alone: the algebra reaches the same block, and is
        // tried first, so it is turned off to see them.
        let no_list = DecodeOptions { list_decode: false, ..Default::default() };
        let report = decode(&grid, &layout, mask, &Pins::new(), &no_list);
        let limit = (layout.blocks[0].ec_len - layout.blocks[0].reserved) / 2;
        match report.blocks[0].outcome {
            BlockOutcome::Solved { method: SolveMethod::Trial { erased }, errors, erasures: 0, .. } => {
                let smallest = 2 - layout.blocks[0].ec_len % 2;
                assert!(erased == smallest || erased == smallest + 2, "{what}: {:?}", report.blocks[0].outcome);
                assert_eq!(errors, limit + 1, "{what}: {:?}", report.blocks[0].outcome);
                assert_eq!(report.text.plain(), text, "{what}");
                assert_eq!(report.blocks[0].is_verified(), report.is_trustworthy(), "{what}");
            }
            BlockOutcome::Failed(_) => {}
            other => panic!("{what}: {other:?}"),
        }
    }
}

#[test]
fn the_limit_confirms_in_every_symbol_the_standard_reserves_nothing_in() {
    // In every block of every symbol whose error correction the standard
    // lets be spent to the last codeword, the errors located at the
    // limit are evidence enough: the least of them, the four an M3-M
    // symbol corrects among its seventeen codewords, are found in one
    // block of noise in nearly two million. Where the standard reserves
    // codewords it is the reserve that speaks, not this.
    let mut least = (u32::MAX, String::new());
    for family in Family::ALL {
        for version in family.versions() {
            for &ec in version.ec_levels() {
                let layout = layout(version, ec);
                for block in layout.blocks.iter().filter(|block| block.reserved == 0 && block.ec_len % 2 == 0) {
                    let bits = located_error_bits(block.codewords.len(), block.ec_len / 2);
                    assert!(bits >= 2 * VERIFIED_BITS, "{version}-{ec:?}: {bits} bits");
                    if bits < least.0 {
                        least = (bits, format!("{version}-{ec:?}"));
                    }
                }
            }
        }
    }
    assert_eq!(least, (20, "M3-M".to_owned()));
}

#[test]
fn every_block_but_m1s_can_be_verified_with_codewords_filled_in() {
    // A block some of whose codewords were filled in is verified with
    // three codewords to spare, and every block has them, within what
    // the standard lets it spend, with one codeword unknown at the
    // least. All but M1's: both of its two are reserved for detection,
    // it fills in nothing, and is verified read whole or not at all, by
    // the sixteen bits they are.
    for family in Family::ALL {
        for version in family.versions() {
            for &ec in version.ec_levels() {
                for block in &layout(version, ec).blocks {
                    let fillable = block.ec_len.saturating_sub(block.reserved.max((AIDED_VERIFIED_BITS / 8) as usize));
                    assert_eq!(fillable == 0, version == SymbolVersion::Micro(1), "{version}-{ec:?}: {} EC codewords, {} reserved", block.ec_len, block.reserved);
                }
            }
        }
    }
    let (layout, text) = (layout(SymbolVersion::Micro(1), EcLevel::L), "12345");
    let mut grid = encode(SymbolVersion::Micro(1), EcLevel::L, Some(1), &[Segment::numeric(text).unwrap()]).unwrap();
    let report = decode_plain(&grid, &layout, 1);
    match report.blocks[0].outcome {
        BlockOutcome::Solved { method: SolveMethod::ReedSolomon, errors: 0, erasures: 0, slack: 2, confidence_bits: 16, beyond_standard: false, .. } => {}
        other => panic!("{other:?}"),
    }
    assert!(report.is_trustworthy());
    assert_eq!(report.text.plain(), text);
    erase_codeword(&mut grid, &layout, layout.blocks[0].codewords[0]);
    let report = decode_plain(&grid, &layout, 1);
    assert!(report.blocks[0].is_beyond_standard() && !report.is_trustworthy(), "{:?}", report.blocks[0].outcome);
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
fn a_solution_with_little_to_spare_is_improved_by_doubting_the_faint_bit() {
    // Fifteen codewords of a 4-M block erased and one more read wrong and
    // faint: exact decoding takes the wrong one as an error and lands on
    // the right values with one codeword to spare, which is too little
    // behind a solution that filled fifteen. Doubting the faint bit
    // instead, or flipping it, leaves nearly three, and the solution
    // comes within a bit of verified. An exact solution used to end the
    // search before the fallbacks were tried.
    let mut rng = Rng(0x51AD);
    let (layout, clean, text) = sample(SymbolVersion::Qr(4), EcLevel::M, 5, &mut rng);
    let block = &layout.blocks[0];
    assert_eq!(block.ec_len, 18);
    let sampled = |grid: &ModuleGrid| {
        let mut grid = grid.clone();
        for y in 0..grid.height() {
            for x in 0..grid.width() {
                grid.set_sampled(x, y, grid.get(x, y), 200);
            }
        }
        grid
    };
    let picked = rng.distinct(block.ec_len - 1, block.codewords.len());
    let torn = |erased: usize| {
        let mut grid = sampled(&clean);
        for &i in &picked[..erased] {
            erase_codeword(&mut grid, &layout, block.codewords[i]);
        }
        let faint = layout.codeword_modules(block.codewords[picked[block.ec_len - 2]])[3];
        let (x, y) = (faint.x as usize, faint.y as usize);
        grid.set_sampled(x, y, grid.get(x, y).inverted(), 12);
        grid
    };
    let grid = torn(block.ec_len - 3);

    let strict = DecodeOptions { reliability: false, message_evidence: false, ..Default::default() };
    let exact = decode(&grid, &layout, 5, &Pins::new(), &strict);
    match exact.blocks[0].outcome {
        BlockOutcome::Solved { method: SolveMethod::ReedSolomon, errors: 1, erasures: 15, slack: 1, .. } => {}
        other => panic!("{other:?}"),
    }
    assert!(!exact.blocks[0].is_verified());
    assert_eq!(exact.text.plain(), text);

    // Doubting the faint bit as a single bit solves the block to the
    // right values with twenty-three equations to spare; flipping it by
    // the chase leaves three codewords, twenty-four bits, less six for
    // the sixty-three patterns of flips tried, and ranks below. After
    // the selection charge neither reaches the byte a verified block
    // needs: the block stands solved and unconfirmed, a bit short. The
    // chase once stood on its three codewords uncharged for the
    // patterns, and the block passed for verified on them.
    let report = decode(&grid, &layout, 5, &Pins::new(), &unguided());
    match report.blocks[0].outcome {
        BlockOutcome::Solved { method: SolveMethod::BitReliability { doubted_bits: 1 }, errors: 1, erasures: 15, risk_bits: 7, .. } => {}
        other => panic!("{other:?}"),
    }
    assert!(report.is_complete() && !report.blocks[0].is_verified());
    assert!(!report.is_trustworthy());
    assert_eq!(report.text.plain(), text);
    assert_eq!(report.codewords.iter().filter(|c| c.status == CwStatus::Corrected).count(), 1);
    // As one of thirty-two decodes, which a search of every format
    // makes, the solution is charged the five bits that costs, as the
    // plain decode is, from both counts of it alike: the doubting
    // solution stands five bits lower. The soft count once went
    // uncharged for the search, and a fallback's solution that stood on
    // it kept its risk whatever the search, where the plain decode's
    // fell.
    let among = decode(&grid, &layout, 5, &Pins::new(), &unguided().among(32));
    assert_eq!(among.blocks[0].outcome.risk_bits(), 7 - tries_charge(32), "{:?}", among.blocks[0].outcome);

    // One codeword fewer erased, and the exact solution has two to
    // spare and the one error located, which with the soft count for
    // the one faint bit it overturns is verified as it stands.
    let grid = torn(block.ec_len - 4);
    let report = decode(&grid, &layout, 5, &Pins::new(), &unguided());
    match report.blocks[0].outcome {
        BlockOutcome::Solved { method: SolveMethod::ReedSolomon, errors: 1, erasures: 14, slack: 2, .. } => {}
        other => panic!("{other:?}"),
    }
    assert!(report.blocks[0].is_verified() && report.is_trustworthy());
    assert_eq!(report.text.plain(), text);

    // One codeword more erased, and flipping the faint bit leaves two to
    // spare: sixteen bits, less the six for the patterns tried, which
    // the sixty-three patterns let a block of noise pass once in a
    // thousand; doubting it leaves fifteen equations. The block is still
    // solved so, to the right values, and stands unconfirmed.
    let grid = torn(block.ec_len - 2);
    let report = decode(&grid, &layout, 5, &Pins::new(), &unguided());
    match report.blocks[0].outcome {
        BlockOutcome::Solved { errors: 1, erasures: 16, risk_bits: 0, .. } => {}
        other => panic!("{other:?}"),
    }
    assert!(report.is_complete() && !report.blocks[0].is_verified());
    assert!(report.blocks[1..].iter().all(|b| b.is_verified()));
    assert!(!report.is_trustworthy());
    assert_eq!(report.text.plain(), text);
}

#[test]
fn a_block_filled_with_one_codeword_to_spare_is_no_read() {
    // Seventeen codewords of a 5-Q block unknown, which leaves one of
    // its eighteen of error correction to check the sixteen read, and
    // two of those wrong. Whatever the first is off by, one value of the
    // second in 255 leaves that one syndrome zero: the block solves with
    // a codeword to spare, to codewords the symbol never held, the other
    // three blocks are clean, and the stream parses, the wrong bytes
    // lying inside its one segment. That was a read. A block so solved
    // is one in 256 of all blocks of noise, and the symbol stands as
    // decoded without confirmation.
    let mut rng = Rng(0x17E5);
    let (layout, clean, text) = sample(SymbolVersion::Qr(5), EcLevel::Q, 2, &mut rng);
    let block = &layout.blocks[0];
    assert_eq!((layout.blocks.len(), block.codewords.len(), block.ec_len), (4, 33, 18));
    let mut torn = clean.clone();
    for &cw in &block.codewords[16..] {
        erase_codeword(&mut torn, &layout, cw);
    }
    corrupt_codeword(&mut torn, &layout, block.codewords[5], 7);
    let mut solved = 0;
    for magnitude in 1..=255usize {
        let mut grid = torn.clone();
        for bit in (0..8).filter(|bit| magnitude >> bit & 1 != 0) {
            corrupt_codeword(&mut grid, &layout, block.codewords[6], bit);
        }
        let report = decode_plain(&grid, &layout, 2);
        assert!(!report.blocks[0].is_verified() && !report.is_trustworthy(), "{magnitude:#x}: {:?}", report.blocks[0].outcome);
        assert!(report.blocks[1..].iter().all(|b| b.is_verified()), "{magnitude:#x}");
        if report.blocks[0].is_solved() {
            solved += 1;
            match report.blocks[0].outcome {
                BlockOutcome::Solved { method: SolveMethod::ReedSolomon, errors: 0, erasures: 17, slack: 1, confidence_bits: 8, beyond_standard: false, .. } => {}
                other => panic!("{magnitude:#x}: {other:?}"),
            }
            // Every block solved and the stream parsing: `unconfirmed`,
            // as the command line puts it, with the message as solved.
            assert!(report.is_complete() && report.score.stream_valid && !report.score.all_verified, "{magnitude:#x}");
            assert_ne!(report.text.plain(), text, "{magnitude:#x}");
            assert!(report.text.spans.iter().any(|s| s.certainty == Certainty::Unverified), "{magnitude:#x}");
        }
    }
    assert_eq!(solved, 1);

    // The same block with fifteen codewords unknown and the rest read
    // right has three to spare, and is a read.
    let mut grid = clean.clone();
    for &cw in &block.codewords[18..] {
        erase_codeword(&mut grid, &layout, cw);
    }
    let report = decode_plain(&grid, &layout, 2);
    match report.blocks[0].outcome {
        BlockOutcome::Solved { method: SolveMethod::ReedSolomon, errors: 0, erasures: 15, slack: 3, confidence_bits: 24, beyond_standard: false, .. } => {}
        other => panic!("{other:?}"),
    }
    assert!(report.is_trustworthy());
    assert_eq!(report.text.plain(), text);
    // With sixteen it has two, and is not.
    erase_codeword(&mut grid, &layout, block.codewords[17]);
    let report = decode_plain(&grid, &layout, 2);
    assert!(report.is_complete() && !report.blocks[0].is_verified() && !report.is_trustworthy());
    assert_eq!(report.text.plain(), text);
}

#[test]
fn erasures_past_capacity_fail_deterministically() {
    let mut rng = Rng(0xFA11);
    let mut reached = 0;
    for (version, ec) in SAMPLES {
        let (layout, mut grid, text) = sample(version, ec, 0, &mut rng);
        let block = &layout.blocks[0];
        for i in rng.distinct(block.ec_len + 1, block.codewords.len()) {
            erase_codeword(&mut grid, &layout, block.codewords[i]);
        }
        // The message's evidence takes the high bit of each erased byte
        // at nought, an equation more a byte, which may reach a codeword
        // past the capacity: what it reads is the message, or nothing
        // confirmed.
        let guided = decode_plain(&grid, &layout, 0);
        assert!(!guided.is_trustworthy() || guided.text.plain() == text, "{version}-{ec:?}");
        reached += usize::from(guided.blocks[0].is_verified());
        let report = decode(&grid, &layout, 0, &Pins::new(), &unguided());
        assert_eq!(
            report.blocks[0].outcome,
            BlockOutcome::Failed(RsFailure::TooManyErasures { erasures: block.ec_len + 1, capacity: block.ec_len }),
            "{version}-{ec:?}"
        );
        assert_eq!(report.blocks[0].codewords_short(), 1);
        assert!(report.blocks[1..].iter().all(|b| b.is_verified()));
        assert!(!report.is_complete());
    }
    // Where enough of the erased codewords are bytes of the message: a
    // large block, as 15-L's is.
    assert!(reached >= 1, "{reached}");
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
        let report = decode(&grid, &layout, mask, &Pins::new(), &unguided());
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
        // Without reliability only the trials and the hard-decision
        // algebra can reach it, and they reach it in the smallest
        // blocks alone.
        let strict = DecodeOptions { reliability: false, ..Default::default() };
        let unaided = decode(&grid, &layout, mask, &Pins::new(), &strict).blocks[0].outcome;
        assert!(
            matches!(
                unaided,
                BlockOutcome::Failed(_) | BlockOutcome::Solved { method: SolveMethod::Trial { .. } | SolveMethod::List { .. }, .. }
            ),
            "{what}: {unaided:?}"
        );
        let report = decode_plain(&grid, &layout, mask);
        // Doubting the faint codewords, doubting the faint bits, flipping
        // them and doubting some while flipping the rest all reach the
        // answer; whichever is confirmed by more bits is kept.
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
            BlockOutcome::Solved { method: SolveMethod::BitFlip { doubted_bits, flipped_bits }, errors, .. } => {
                assert!(flipped_bits >= 1 && flipped_bits <= 3 && doubted_bits + flipped_bits >= wrong, "{what}: {doubted_bits} doubted, {flipped_bits} flipped");
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
        // Which modules were doubted is told with the solution: the
        // codewords erased whole, or the bits taken as unknown or
        // flipped, as many as the method counts. A clean read doubts
        // none.
        let doubted_codewords = report.codewords.iter().filter(|c| c.doubted != 0).count();
        let doubted_bits: usize = report.codewords.iter().map(|c| c.doubted.count_ones() as usize).sum();
        match report.blocks[0].outcome {
            BlockOutcome::Solved { method: SolveMethod::Reliability { doubted }, .. } => assert_eq!(doubted_codewords, doubted, "{what}"),
            BlockOutcome::Solved { method: SolveMethod::BitReliability { doubted_bits: bits }, .. } => assert_eq!(doubted_bits, bits, "{what}"),
            BlockOutcome::Solved { method: SolveMethod::Chase { flipped_bits }, .. } => assert_eq!(doubted_bits, flipped_bits, "{what}"),
            BlockOutcome::Solved { method: SolveMethod::BitFlip { doubted_bits: bits, flipped_bits }, .. } => {
                assert_eq!(doubted_bits, bits + flipped_bits, "{what}")
            }
            other => panic!("{what}: {other:?}"),
        }
        assert!(truth.codewords.iter().all(|c| c.doubted == 0), "{what}");
        // Each of the three ways is left out by itself: with none of
        // them the block fails as it does with doubting off, and with
        // one alone it solves that way or not at all.
        let none = DecodeOptions {
            doubt_codewords: false,
            doubt_bits: false,
            flip_bits: false,
            list_decode: false,
            belief_propagation: false,
            ..Default::default()
        };
        let unaided = decode(&grid, &layout, mask, &Pins::new(), &none).blocks[0].outcome;
        assert!(
            matches!(unaided, BlockOutcome::Failed(_) | BlockOutcome::Solved { method: SolveMethod::Trial { .. }, .. }),
            "{what}: {unaided:?}"
        );
        let alone = [
            DecodeOptions { doubt_codewords: true, ..none },
            DecodeOptions { doubt_bits: true, ..none },
            DecodeOptions { flip_bits: true, ..none },
        ];
        for (way, only) in alone.iter().enumerate() {
            let own = match decode(&grid, &layout, mask, &Pins::new(), only).blocks[0].outcome {
                BlockOutcome::Solved { method: SolveMethod::Reliability { .. }, .. } => 0,
                BlockOutcome::Solved { method: SolveMethod::BitReliability { .. } | SolveMethod::BitTrial { .. }, .. } => 1,
                BlockOutcome::Solved { method: SolveMethod::Chase { .. } | SolveMethod::BitFlip { .. }, .. } => 2,
                BlockOutcome::Solved { method: SolveMethod::Trial { .. }, .. } | BlockOutcome::Failed(_) => continue,
                BlockOutcome::Solved { method, .. } => panic!("{what}: {method:?}"),
            };
            assert_eq!(own, way, "{what}");
        }
    }
    // A certain module is never doubted, so painting a wrong module by hand
    // keeps the failure: two past the code, which is one past what the
    // trials of erasures reach.
    let (version, ec, mask) = (SymbolVersion::Qr(4), EcLevel::M, 1);
    let (layout, mut grid, _) = sample(version, ec, mask, &mut rng);
    let block = &layout.blocks[0];
    for i in 0..block.ec_len / 2 + 2 {
        corrupt_codeword(&mut grid, &layout, block.codewords[i], i);
    }
    assert!(grid.cells().iter().all(|_| true) && grid.reliability(0, 0) == CERTAIN);
    assert!(!decode_plain(&grid, &layout, mask).blocks[0].is_solved());
}

#[test]
fn erasing_by_trial_reaches_one_wrong_codeword_past_the_code() {
    // Painted by hand, every module is certain, and nothing that doubts
    // or flips by reliability has anything to go on; erasing codewords
    // by trial, every pair of the block's, reaches one wrong codeword
    // more than the code corrects, and the errors the code then locates
    // among the rest confirm it. One more again is reached by the sets
    // two larger, where the block is small enough for those to fit the
    // budget with the smaller ones, and is past the trials where it is
    // not.
    let mut rng = Rng(0x7A1A1);
    let mut reached = 0;
    for (version, ec) in SAMPLES {
        let mask = rng.below(version.mask_count() as usize) as u8;
        let (layout, clean, text) = sample(version, ec, mask, &mut rng);
        let block = &layout.blocks[0];
        let n = block.codewords.len();
        if block.reserved > 0 {
            continue;
        }
        let limit = block.ec_len / 2;
        let mut grid = clean.clone();
        let picked = rng.distinct(limit + 2, n);
        for &i in &picked[..limit + 1] {
            corrupt_codeword(&mut grid, &layout, block.codewords[i], rng.below(8));
        }
        let what = format!("{version}-{ec:?} with {} wrong", limit + 1);
        assert!(!grid.has_reliability());
        let no_trial = DecodeOptions { erase_by_trial: false, list_decode: false, ..Default::default() };
        assert!(matches!(decode(&grid, &layout, mask, &Pins::new(), &no_trial).blocks[0].outcome, BlockOutcome::Failed(_)), "{what}");
        // The trials alone, the algebra turned off: it reaches the same
        // block and is tried first.
        let no_list = DecodeOptions { list_decode: false, ..Default::default() };
        let report = decode(&grid, &layout, mask, &Pins::new(), &no_list);
        let smallest = 2 - block.ec_len % 2;
        let erased = match report.blocks[0].outcome {
            BlockOutcome::Solved { method: SolveMethod::Trial { erased }, errors, erasures: 0, beyond_standard: false, .. } => {
                // The smallest set that goes past the code, or the next,
                // where the smallest locates too little to be offered.
                assert!(erased == smallest || erased == smallest + 2, "{what}: {:?}", report.blocks[0].outcome);
                assert_eq!(errors, limit + 1, "{what}");
                erased
            }
            // Too little located to confirm a solution found among so
            // many tries: the smallest blocks.
            BlockOutcome::Failed(_) if block.ec_len < 16 => continue,
            other => panic!("{what}: {other:?}"),
        };
        assert_eq!(report.text.plain(), text, "{what}");
        // The smallest blocks locate too little to confirm a solution
        // found among hundreds of tries, and the largest too little for
        // one found among thousands of pairs: it stands unconfirmed.
        if !report.blocks[0].is_verified() {
            assert!(!report.is_trustworthy(), "{what}: {:?}", report.blocks[0].outcome);
            continue;
        }
        assert!(report.is_trustworthy(), "{what}");
        let truth = decode_plain(&clean, &layout, mask);
        for (got, want) in report.codewords.iter().zip(&truth.codewords) {
            assert_eq!(got.final_value, want.final_value, "{what}");
        }
        assert_eq!(report.codewords.iter().filter(|c| c.status == CwStatus::Corrected).count(), limit + 1, "{what}");
        // The codewords erased by trial are shown as doubted, as the
        // codewords doubted for their reliability are.
        let doubted = report.codewords.iter().filter(|c| c.doubted != 0).count();
        assert_eq!(doubted, erased, "{what}");
        reached += 1;

        corrupt_codeword(&mut grid, &layout, block.codewords[picked[limit + 1]], rng.below(8));
        let past = decode(&grid, &layout, mask, &Pins::new(), &no_list);
        let smallest = 2 - block.ec_len % 2;
        let fits = sets_of(n, smallest, TRIAL_SETS).zip(sets_of(n, smallest + 2, TRIAL_SETS)).is_some_and(|(small, large)| small + large <= TRIAL_SETS);
        match past.blocks[0].outcome {
            BlockOutcome::Solved { method: SolveMethod::Trial { erased: again }, errors, .. } if fits => {
                assert_eq!((again, errors), (erased + 2, limit + 2), "{what} and one more");
                assert_eq!(past.text.plain(), text, "{what} and one more");
            }
            BlockOutcome::Failed(_) => {}
            other => panic!("{what} and one more: {other:?}"),
        }
    }
    assert!(reached >= 4, "{reached} blocks reached by trial");
}

#[test]
fn erasing_by_trial_frees_a_wrong_codeword_among_scattered_unknown_bits() {
    // A codeword to a module unknown over more codewords than there is
    // redundancy to erase, which solves bit by bit as long as every bit
    // read is right, and a stain over one codeword, or two, read wrong
    // in every module: the bit-level solver finds its equations
    // inconsistent, and erasing by trial finds the codewords that make
    // them agree, with no reliability to go on.
    let mut rng = Rng(0x57A1);
    let mut freed = 0;
    for (version, ec) in SAMPLES {
        let mask = rng.below(version.mask_count() as usize) as u8;
        let (layout, clean, text) = sample(version, ec, mask, &mut rng);
        let block = &layout.blocks[0];
        let n = block.codewords.len();
        if block.ec_len < 10 {
            continue;
        }
        for stains in [1usize, 2] {
            let mut grid = clean.clone();
            let unknown = (block.ec_len + 2).min(n - stains - 1);
            let picked = rng.distinct(unknown + stains, n);
            for &i in &picked[..unknown] {
                let modules = layout.codeword_modules(block.codewords[i]);
                let pos = modules[rng.below(modules.len())];
                grid.set(pos.x as usize, pos.y as usize, Module::Unknown);
            }
            for &i in &picked[unknown..] {
                for pos in layout.codeword_modules(block.codewords[i]) {
                    let (x, y) = (pos.x as usize, pos.y as usize);
                    grid.set(x, y, grid.get(x, y).inverted());
                }
            }
            let what = format!("{version}-{ec:?} with {unknown} unknown bits and {stains} stained codewords");
            let no_trial = DecodeOptions { erase_by_trial: false, list_decode: false, ..unguided() };
            assert!(!decode(&grid, &layout, mask, &Pins::new(), &no_trial).blocks[0].is_solved(), "{what}");
            let report = decode(&grid, &layout, mask, &Pins::new(), &unguided());
            match report.blocks[0].outcome {
                BlockOutcome::Solved { method: SolveMethod::BitTrial { doubted_bits: 0, erased }, errors, .. } => {
                    assert_eq!((erased, errors), (stains, stains), "{what}");
                }
                // Pairs of a block too large to try them all, or too
                // few equations left to confirm one found among them.
                BlockOutcome::Failed(_) if stains == 2 && (n * (n - 1) / 2 > TRIAL_SETS || block.ec_len < 16) => continue,
                other => panic!("{what}: {other:?}"),
            }
            assert!(report.blocks[0].is_verified(), "{what}: {:?}", report.blocks[0].outcome);
            assert_eq!(report.text.plain(), text, "{what}");
            let truth = decode_plain(&clean, &layout, mask);
            for (got, want) in report.codewords.iter().zip(&truth.codewords) {
                assert_eq!(got.final_value, want.final_value, "{what}");
            }
            assert_eq!(report.codewords.iter().filter(|c| c.doubted != 0).count(), stains, "{what}");
            freed += 1;
        }
    }
    assert!(freed >= 8, "{freed} blocks freed by trial");
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
        // Without reliability only the trials and the hard-decision
        // algebra can reach it, and they reach it in the smallest
        // blocks alone.
        let strict = DecodeOptions { reliability: false, ..unguided() };
        let unaided = decode(&grid, &layout, mask, &Pins::new(), &strict).blocks[0].outcome;
        assert!(
            matches!(
                unaided,
                BlockOutcome::Failed(_) | BlockOutcome::Solved { method: SolveMethod::Trial { .. } | SolveMethod::List { .. }, .. }
            ),
            "{what}: {unaided:?}"
        );
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
        // With the bit-level solver off there is nothing to doubt bits
        // with, and only chasing, which flips ten at the most, can reach
        // the block.
        let no_bits = DecodeOptions { bit_level: false, belief_propagation: false, ..unguided() };
        let chased = decode(&grid, &layout, mask, &Pins::new(), &no_bits).blocks[0].outcome;
        assert!(
            matches!(chased, BlockOutcome::Failed(_) | BlockOutcome::Solved { method: SolveMethod::Chase { .. }, .. }),
            "{what}: {chased:?}"
        );
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
        let unaided = decode(&grid, &layout, mask, &Pins::new(), &strict).blocks[0].outcome;
        assert!(
            matches!(
                unaided,
                BlockOutcome::Failed(_) | BlockOutcome::Solved { method: SolveMethod::Trial { .. } | SolveMethod::List { .. }, .. }
            ),
            "{what}: {unaided:?}"
        );
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
    // same count of wrong codewords stays a failure, but for the algebra,
    // which reaches thirteen wrong of the forty codewords of an 8-Q block
    // from the bits as painted, at the edge of its bound, given four
    // times the usual budget.
    let (version, ec, mask) = (SymbolVersion::Qr(8), EcLevel::Q, 2);
    let (layout, mut grid, text) = sample(version, ec, mask, &mut rng);
    let block = &layout.blocks[0];
    for i in 0..block.ec_len / 2 + 2 {
        corrupt_codeword(&mut grid, &layout, block.codewords[i], i);
    }
    assert!(!grid.has_reliability());
    let no_list = DecodeOptions { list_decode: false, ..Default::default() };
    assert!(matches!(decode(&grid, &layout, mask, &Pins::new(), &no_list).blocks[0].outcome, BlockOutcome::Failed(_)));
    let listing = DecodeOptions { list_cost: 8192, ..Default::default() };
    let report = decode(&grid, &layout, mask, &Pins::new(), &listing);
    assert!(matches!(report.blocks[0].outcome, BlockOutcome::Solved { method: SolveMethod::List { .. }, errors: 13, .. }), "{:?}", report.blocks[0].outcome);
    assert!(report.blocks[0].is_verified() && report.text.plain() == text);
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
    // With the fewest sets a decode among many may erase by trial: the
    // full budget spends a tenth of a second on every hopeless block,
    // and there are thousands here.
    let opts = DecodeOptions { trial_sets: qr_forensics::decode::rs::TRIAL_SETS_MIN, ..DecodeOptions::default() };
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
                let report = decode(&grid, &layout, mask, &pins, &opts);
                let _ = report.text.plain();
                let _ = qr_forensics::decode::padding::recover_padding(&grid, &layout, mask, &pins, &opts);
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

#[test]
fn errors_located_in_one_block_make_the_modules_beside_them_doubtful_in_another() {
    // Damage is contiguous, and the blocks are interleaved, so a stain
    // over a few modules lies across codewords of different blocks.
    // Block A is wrong in three modules its code locates. Block B is
    // wrong in one module more than its code corrects, every one read
    // as solidly as the rest, two of them beside A's wrong modules.
    // Alone B fails: nothing reads faint, so doubting and chasing by
    // reliability pick modules by their index and miss. With A's errors
    // located, the modules beside them are doubted first, and chasing
    // the two of B's among them flips them and locates the rest.
    let (version, ec, mask) = (SymbolVersion::Qr(5), EcLevel::Q, 3);
    let (layout, clean, text) = sample(version, ec, mask, &mut Rng(0x5EA7));
    let (a, b) = (&layout.blocks[0], &layout.blocks[1]);
    let limit = b.ec_len / 2;
    // Modules of A whose eight neighbours include modules of B, in two
    // codewords of B that are not the first few, where chasing by index
    // would reach them anyway.
    let mut beside: Vec<((usize, usize), (usize, usize), usize)> = Vec::new();
    for (i, &cw) in a.codewords.iter().enumerate().take(a.codewords.len() - 2) {
        for pos in layout.codeword_modules(cw) {
            let (x, y) = (pos.x as usize, pos.y as usize);
            for ny in y.saturating_sub(1)..(y + 2).min(layout.func.height) {
                for nx in x.saturating_sub(1)..(x + 2).min(layout.func.width) {
                    let Some((other, _)) = layout.module_codeword(nx, ny) else { continue };
                    let span = layout.codewords[other];
                    if span.block as usize == 1 && span.index_in_block as usize >= 4 && (nx, ny) != (x, y) {
                        beside.push(((x, y), (nx, ny), i));
                    }
                }
            }
        }
    }
    beside.dedup_by_key(|&mut (_, b, _)| b);
    let mut picked: Vec<((usize, usize), (usize, usize), usize)> = Vec::new();
    for item in beside {
        let (_, (nx, ny), i) = item;
        let (b_cw, _) = layout.module_codeword(nx, ny).unwrap();
        let distinct_a = picked.iter().all(|&(_, _, j)| j != i);
        let distinct_b = picked.iter().all(|&(_, (px, py), _)| layout.module_codeword(px, py).unwrap().0 != b_cw);
        if distinct_a && distinct_b {
            picked.push(item);
        }
        if picked.len() == 2 {
            break;
        }
    }
    assert_eq!(picked.len(), 2, "two modules of B beside modules of A");
    let b_beside: Vec<usize> = picked.iter().map(|&(_, (nx, ny), _)| layout.module_codeword(nx, ny).unwrap().0).collect();
    // The rest of B's wrong codewords: from the high end of the block,
    // one module each.
    let mut b_wrong: Vec<usize> = b_beside.clone();
    for &cw in b.codewords.iter().rev() {
        if b_wrong.len() == limit + 1 {
            break;
        }
        if !b_wrong.contains(&cw) {
            b_wrong.push(cw);
        }
    }
    let mut rng = Rng(0xB51DE);
    let mut paint = |grid: &mut ModuleGrid, with_a: bool| {
        for y in 0..grid.height() {
            for x in 0..grid.width() {
                grid.set_sampled(x, y, clean.get(x, y), 200);
            }
        }
        for &cw in &b_wrong {
            let (x, y) = if let Some(&(_, (nx, ny), _)) = picked.iter().find(|&&(_, (nx, ny), _)| layout.module_codeword(nx, ny).unwrap().0 == cw) {
                (nx, ny)
            } else {
                let modules = layout.codeword_modules(cw);
                let pos = modules[2 + rng.below(modules.len() - 2)];
                (pos.x as usize, pos.y as usize)
            };
            grid.set_sampled(x, y, clean.get(x, y).inverted(), 200);
        }
        if with_a {
            for &((x, y), _, _) in &picked {
                grid.set_sampled(x, y, clean.get(x, y).inverted(), 200);
            }
            let third = layout.codeword_modules(a.codewords[a.codewords.len() - 1])[3];
            grid.set_sampled(third.x as usize, third.y as usize, clean.get(third.x as usize, third.y as usize).inverted(), 200);
        }
    };
    // Without the trials of erasures or the algebra, which reach one
    // past the code on their own: what the located errors add is the
    // point here.
    let opts = DecodeOptions { erase_by_trial: false, list_decode: false, ..DecodeOptions::default() };
    let mut alone = clean.clone();
    paint(&mut alone, false);
    let report = decode(&alone, &layout, mask, &Pins::new(), &opts);
    assert!(report.blocks[0].is_verified(), "A clean: {:?}", report.blocks[0].outcome);
    assert!(!report.blocks[1].is_solved(), "B alone: {:?}", report.blocks[1].outcome);
    assert_eq!(report.blocks[1].near_errors, 0);

    let mut stained = clean.clone();
    paint(&mut stained, true);
    let report = decode(&stained, &layout, mask, &Pins::new(), &opts);
    match report.blocks[0].outcome {
        BlockOutcome::Solved { method: SolveMethod::ReedSolomon, errors: 3, .. } => {}
        other => panic!("A: {other:?}"),
    }
    match report.blocks[1].outcome {
        // Chasing flips the two and locates the rest; doubting the two
        // codewords whole reaches the same codeword, and whichever is
        // the less risky stands for it.
        BlockOutcome::Solved { method: SolveMethod::Chase { flipped_bits: 2 } | SolveMethod::Reliability { doubted: 2 }, errors, erasures, .. } => {
            assert_eq!(errors + erasures, limit + 1, "B");
        }
        other => panic!("B beside A's errors: {other:?}"),
    }
    assert!(report.blocks[1].is_verified(), "B: {:?}", report.blocks[1].outcome);
    assert!(report.blocks[1].near_errors >= 2, "B: {} modules doubted beside errors", report.blocks[1].near_errors);
    assert!(report.is_trustworthy());
    assert_eq!(report.text.plain(), text);
    // The modules doubted are told with the solution: the two flipped.
    let doubted: Vec<usize> = b.codewords.iter().copied().filter(|&cw| report.codewords[cw].doubted != 0).collect();
    assert_eq!(doubted.len(), 2);
    assert!(doubted.iter().all(|cw| b_beside.contains(cw)), "{doubted:?} among {b_beside:?}");
}

#[test]
fn rivals_the_block_as_read_lies_as_near_to_are_told_apart_by_the_message() {
    // Two codewords of the code, as far apart as any two are, and a
    // block read halfway between them: as many codewords wrong by one as
    // by the other, with as many bits overturned either way. Erasing by
    // trial reaches both, with the same confirmation and the same
    // distance from the block as read, and the block alone cannot say
    // which it is. The message can: under one the stream parses to its
    // terminator and the padding after it is the standard's, and under
    // the other the header reads as noise.
    use qr_forensics::decode::rs::{rs_generator, soft_distance};
    let (version, ec, mask) = (SymbolVersion::Qr(1), EcLevel::H, 0);
    let layout = layout(version, ec);
    let text = "AB12CD";
    let clean = encode(version, ec, Some(mask), &[Segment::alphanumeric(text).unwrap()]).unwrap();
    let truth = decode_plain(&clean, &layout, mask);
    assert!(truth.is_trustworthy());
    let block = &layout.blocks[0];
    let (n, ec_len) = (block.codewords.len(), block.ec_len);
    let read: Vec<u8> = block.codewords.iter().map(|&cw| truth.codewords[cw].read).collect();
    // The other codeword: the generator polynomial, times a constant,
    // shifted to lie over the first eighteen positions of the block,
    // which take in every data codeword and the header among them.
    // Position i carries x^(n - 1 - i).
    let generator = rs_generator(ec_len);
    let span = ec_len + 1;
    let difference = |c: u8| -> Vec<u8> {
        (0..n)
            .map(|i| {
                let degree = n - 1 - i;
                let shift = n - span;
                if degree >= shift && degree - shift <= ec_len { qr_forensics::decode::gf256::mul(c, generator[degree - shift]) } else { 0 }
            })
            .collect()
    };
    // A split of the eighteen positions into nine and nine, the header's
    // among the nine taken from the other codeword, overturning as many
    // bits each way.
    let limit = ec_len / 2;
    let mut found = None;
    'search: for c in 1..=255u8 {
        let d = difference(c);
        assert!(d[..span].iter().all(|&v| v != 0) && d[span..].iter().all(|&v| v == 0), "the generator has no zero coefficient");
        let weights: Vec<u32> = d[..span].iter().map(|v| v.count_ones()).collect();
        let total: u32 = weights.iter().sum();
        for set in 0u32..1 << span {
            if set.count_ones() as usize != limit + 1 || set & 1 == 0 {
                continue;
            }
            let taken: u32 = (0..span).filter(|&i| set >> i & 1 != 0).map(|i| weights[i]).sum();
            if 2 * taken == total {
                found = Some((d, set));
                break 'search;
            }
        }
    }
    let (d, set) = found.expect("a split that overturns as many bits each way");
    let mut grid = clean.clone();
    for i in (0..span).filter(|&i| set >> i & 1 != 0) {
        let cw = block.codewords[i];
        let offset = layout.codewords[cw].bit_offset as usize;
        for (k, pos) in layout.codeword_modules(cw).iter().enumerate() {
            if d[i] & (0x80 >> (offset + k)) != 0 {
                let (x, y) = (pos.x as usize, pos.y as usize);
                grid.set(x, y, grid.get(x, y).inverted());
            }
        }
    }
    assert!(!grid.has_reliability());
    // Both codewords lie as far from the block as read.
    let as_read: Vec<qr_forensics::decode::rs::CwRead> =
        block.codewords.iter().map(|&cw| qr_forensics::decode::rs::CwRead::known(decode_plain(&grid, &layout, mask).codewords[cw].read)).collect();
    let certain = vec![[CERTAIN; 8]; n];
    let other: Vec<u8> = read.iter().zip(&d).map(|(r, v)| r ^ v).collect();
    assert_eq!(soft_distance(&as_read, &certain, &read), soft_distance(&as_read, &certain, &other));

    let report = decode_plain(&grid, &layout, mask);
    match report.blocks[0].outcome {
        BlockOutcome::Solved { method: SolveMethod::Trial { erased: 3 }, errors, .. } => assert_eq!(errors, limit + 1),
        other => panic!("{other:?}"),
    }
    assert_eq!(report.blocks[0].rivals, 2, "told from one rival by the message");
    assert!(report.blocks[0].is_verified());
    assert!(report.is_trustworthy());
    assert_eq!(report.text.plain(), text);
    for (cw, want) in block.codewords.iter().zip(&read) {
        assert_eq!(report.codewords[*cw].final_value, Some(*want));
    }
}

#[test]
fn list_decoding_reaches_a_block_wrong_in_more_codewords_than_the_trials_reach() {
    // A version 5-H block: thirty-three codewords, twenty-two of error
    // correction, so half the distance is eleven; erasing codewords by
    // trial reaches twelve by the pairs and thirteen by the fours; the
    // curve through the codewords reaches fourteen, which is the bound
    // of the algebra for the block, and wants a multiplicity of fifteen
    // to reach it with room to spare: twice the usual budget. Painted by
    // hand, every module is certain, and nothing that doubts by
    // reliability has anything to go on.
    let mut rng = Rng(0x115D);
    let (version, ec, mask) = (SymbolVersion::Qr(5), EcLevel::H, 2);
    let (layout, clean, text) = sample(version, ec, mask, &mut rng);
    let block = &layout.blocks[0];
    assert_eq!((block.codewords.len(), block.data_len, block.ec_len), (33, 11, 22));
    let listing = DecodeOptions { list_cost: 4096, ..DecodeOptions::default() };
    for round in 0..5 {
        let mut grid = clean.clone();
        for &i in &rng.distinct(14, 33) {
            corrupt_codeword(&mut grid, &layout, block.codewords[i], rng.below(8));
        }
        let what = format!("round {round}");
        let no_list = DecodeOptions { list_decode: false, ..DecodeOptions::default() };
        let without = decode(&grid, &layout, mask, &Pins::new(), &no_list);
        assert!(!without.blocks[0].is_verified(), "{what}: {:?}", without.blocks[0].outcome);
        let report = decode(&grid, &layout, mask, &Pins::new(), &listing);
        match report.blocks[0].outcome {
            BlockOutcome::Solved { method: SolveMethod::List { multiplicity }, errors, erasures: 0, beyond_standard: false, .. } => {
                assert_eq!(errors, 14, "{what}");
                assert!(multiplicity >= 15, "{what}: multiplicity {multiplicity}");
            }
            other => panic!("{what}: {other:?}"),
        }
        assert!(report.blocks[0].is_verified(), "{what}");
        assert!(report.is_trustworthy(), "{what}");
        assert_eq!(report.text.plain(), text, "{what}");
        // The Corrections overlay has the codewords it changed.
        let corrected = report.codewords.iter().filter(|c| c.status == CwStatus::Corrected).count();
        assert_eq!(corrected, 14, "{what}");
        assert!(report.codewords.iter().filter(|c| c.doubted != 0).count() == 14, "{what}");
    }
    // Sampled faintly, the faint modules' other colours are points of
    // their own: a block wrong in sixteen codewords, each in one faint
    // module, past the algebra on the bits as read, solves on the bits
    // weighed, with every other way of doubting by reliability and the
    // trials turned off.
    let mut grid = clean.clone();
    for y in 0..grid.height() {
        for x in 0..grid.width() {
            grid.set_sampled(x, y, grid.get(x, y), 200);
        }
    }
    for &i in &rng.distinct(16, 33) {
        let modules = layout.codeword_modules(block.codewords[i]);
        let pos = modules[rng.below(modules.len())];
        let (x, y) = (pos.x as usize, pos.y as usize);
        grid.set_sampled(x, y, grid.get(x, y).inverted(), rng.below(10) as u8);
    }
    let alone = DecodeOptions {
        doubt_codewords: false,
        doubt_bits: false,
        flip_bits: false,
        erase_by_trial: false,
        belief_propagation: false,
        ..DecodeOptions::default()
    };
    let report = decode(&grid, &layout, mask, &Pins::new(), &alone);
    assert!(matches!(report.blocks[0].outcome, BlockOutcome::Solved { method: SolveMethod::List { .. }, errors: 16, .. }), "{:?}", report.blocks[0].outcome);
    assert!(report.blocks[0].is_verified() && report.text.plain() == text);
    // With doubting by reliability off the curve is the hard decision
    // one, and sixteen is past it.
    let hard = DecodeOptions { reliability: false, erase_by_trial: false, ..DecodeOptions::default() };
    assert!(!decode(&grid, &layout, mask, &Pins::new(), &hard).blocks[0].is_verified());
}

#[test]
fn a_fallback_spends_the_reserve_as_the_plain_decoder_does() {
    // A version 1-H symbol painted by hand: seventeen EC codewords, one
    // reserved for detection, so the standard corrects eight. Eleven
    // wrong codewords are past the plain decoder, past the trials of
    // erasures within their budget, and within the curve of list
    // decoding, whose solution is held to spend every codeword of the
    // redundancy, the reserve among them. The curve offers it as the
    // plain decoder offers a solution that spends the reserve: marked
    // beyond the standard, never confirmed, and the message read off it
    // all the same.
    let mut rng = Rng(0x5E5E);
    let (version, ec) = (SymbolVersion::Qr(1), EcLevel::H);
    let mask = rng.below(version.mask_count() as usize) as u8;
    let (layout, clean, text) = sample(version, ec, mask, &mut rng);
    let block = &layout.blocks[0];
    assert_eq!((block.ec_len, block.reserved), (17, 1));
    let mut grid = clean.clone();
    for i in rng.distinct(11, block.codewords.len()) {
        corrupt_codeword(&mut grid, &layout, block.codewords[i], rng.below(8));
    }
    let report = decode_plain(&grid, &layout, mask);
    match report.blocks[0].outcome {
        BlockOutcome::Solved { method: SolveMethod::List { .. }, errors: 11, erasures: 0, slack: 0, beyond_standard: true, .. } => {}
        other => panic!("{other:?}"),
    }
    assert!(report.blocks[0].is_beyond_standard() && !report.blocks[0].is_verified());
    assert_eq!(report.text.plain(), text);
    assert!(!report.is_trustworthy());
    // Eight wrong are within the plain decoder, which spends the
    // reserve itself: the same marking, from it.
    let mut grid = clean.clone();
    for i in rng.distinct(8, block.codewords.len()) {
        corrupt_codeword(&mut grid, &layout, block.codewords[i], rng.below(8));
    }
    let report = decode_plain(&grid, &layout, mask);
    assert!(matches!(report.blocks[0].outcome, BlockOutcome::Solved { method: SolveMethod::ReedSolomon, errors: 8, slack: 1, beyond_standard: false, .. }), "{:?}", report.blocks[0].outcome);
    assert!(report.blocks[0].is_verified());
    // A fallback prefers a solution within the standard: nine wrong are
    // reached by the trials erasing three, which leaves two to spare,
    // over the curve's and the trials' own that leave nothing.
    let mut grid = clean.clone();
    for i in rng.distinct(9, block.codewords.len()) {
        corrupt_codeword(&mut grid, &layout, block.codewords[i], rng.below(8));
    }
    let report = decode_plain(&grid, &layout, mask);
    assert!(matches!(report.blocks[0].outcome, BlockOutcome::Solved { method: SolveMethod::Trial { erased: 3 }, errors: 9, slack: 2, beyond_standard: false, .. }), "{:?}", report.blocks[0].outcome);
    assert!(report.blocks[0].is_verified());
    // A block whose every EC codeword is reserved is held to the
    // reserve still: an M1 symbol wrong in two codewords, past the one
    // its two EC codewords detect, has no solution from anyone.
    let (layout, text) = (common::layout(SymbolVersion::Micro(1), EcLevel::L), "12345");
    let mut grid = encode(SymbolVersion::Micro(1), EcLevel::L, Some(1), &[Segment::numeric(text).unwrap()]).unwrap();
    for cw in [0, 1] {
        corrupt_codeword(&mut grid, &layout, layout.blocks[0].codewords[cw], 2);
    }
    let report = decode_plain(&grid, &layout, 1);
    assert!(matches!(report.blocks[0].outcome, BlockOutcome::Failed(_)), "{:?}", report.blocks[0].outcome);
}

#[test]
fn the_chances_of_the_blocks_being_wrong_add_up_for_the_message() {
    use qr_forensics::decode::pipeline::combined_risk_bits;
    // Two blocks each at one in 2^8 make the message one in 2^7.
    assert_eq!(combined_risk_bits([Some(8), Some(8)]), Some(7));
    // The eighty-one blocks of a version 40-H symbol each at one in 2^8
    // make it worse than one in four, though no further than one in two.
    assert_eq!(combined_risk_bits(vec![Some(8); 81]), Some(1));
    // A block far surer than the rest costs the whole nothing, however
    // sure: its chance is not lost to the arithmetic either.
    assert_eq!(combined_risk_bits([Some(8), Some(200)]), Some(8));
    assert_eq!(combined_risk_bits([Some(2000), Some(3000)]), Some(2000));
    // A block unconfirmed leaves nothing to confirm the message, and one
    // unsolved leaves no message to be wrong about yet.
    assert_eq!(combined_risk_bits([Some(0), Some(50)]), Some(0));
    assert_eq!(combined_risk_bits([Some(8), None]), None);
}

#[test]
fn a_clean_message_is_as_sure_as_its_least_sure_block_or_nearly() {
    let (version, ec, mask) = (SymbolVersion::Qr(5), EcLevel::Q, 3);
    let (layout, clean, _) = sample(version, ec, mask, &mut Rng(0xC1EA));
    let report = decode(&clean, &layout, mask, &Pins::new(), &DecodeOptions::default());
    assert!(report.is_trustworthy() && !report.message_risk_falls_short());
    let least = report.blocks.iter().map(|b| b.outcome.risk_bits()).min().unwrap();
    let message = report.message_risk_bits().unwrap();
    // Four blocks: no more than two bits short of the least sure.
    assert!(message <= least && message + 2 >= least, "{message} of {least}");
}

/// A grid of `version` sampled at a solid reliability everywhere, with
/// block `failing` wrong in one solidly misread module of each of its
/// codewords from the high end, one more than its code corrects, so
/// that it fails, and every module of `misread` misread solidly too:
/// what the blocks that verify have to learn from.
fn with_misreads(layout: &SymbolLayout, clean: &ModuleGrid, failing: usize, misread: &[(usize, usize)]) -> ModuleGrid {
    let mut grid = clean.clone();
    for y in 0..grid.height() {
        for x in 0..grid.width() {
            grid.set_sampled(x, y, clean.get(x, y), 200);
        }
    }
    let spec = &layout.blocks[failing];
    for &cw in spec.codewords.iter().rev().take(spec.ec_len / 2 + 1) {
        let pos = layout.codeword_modules(cw)[3];
        let (x, y) = (pos.x as usize, pos.y as usize);
        grid.set_sampled(x, y, clean.get(x, y).inverted(), 200);
    }
    for &(x, y) in misread {
        grid.set_sampled(x, y, clean.get(x, y).inverted(), 200);
    }
    grid
}

#[test]
fn how_far_misreads_spread_is_learnt_from_the_blocks_that_verify() {
    // 5-Q: four blocks of eighteen error correction codewords. The last
    // fails; the other three verify, each with misreads to learn from.
    let (version, ec, mask) = (SymbolVersion::Qr(5), EcLevel::Q, 3);
    let (layout, clean, _) = sample(version, ec, mask, &mut Rng(0x57A1));
    let failing = layout.blocks.len() - 1;
    let in_verified = |x: usize, y: usize| layout.module_codeword(x, y).is_some_and(|(cw, _)| (layout.codewords[cw].block as usize) < failing);
    // Without the trials, the curve or the message's evidence, which
    // reach the failing block on their own: the point is what the other
    // blocks teach.
    let opts = DecodeOptions { erase_by_trial: false, list_decode: false, message_evidence: false, ..DecodeOptions::default() };

    // Two stains of two by two modules, in the verified blocks alone.
    let (width, height) = (layout.func.width, layout.func.height);
    let mut stains: Vec<(usize, usize)> = Vec::new();
    'search: for y in (0..height - 1).step_by(5) {
        for x in (0..width - 1).step_by(7) {
            let square = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)];
            if square.iter().all(|&(x, y)| in_verified(x, y)) {
                stains.extend(square);
                if stains.len() == 8 {
                    break 'search;
                }
            }
        }
    }
    assert_eq!(stains.len(), 8, "two stains");
    let report = decode(&with_misreads(&layout, &clean, failing, &stains), &layout, mask, &Pins::new(), &opts);
    assert!(report.blocks[..failing].iter().all(|b| b.is_verified()), "{:?}", report.blocks);
    let stained = report.clustering.expect("learnt from the verified blocks");
    assert_eq!(stained.located, 8);
    // Beside two or more misread modules, a third of them were misread
    // too: far more than one in eight.
    assert!(stained.gross[1] > 0.2, "{stained:?}");

    // As many specks, none beside another: what spreads is next to
    // nothing, and the modules beside a speck are doubted far less than
    // the one in eight that stood before anything was learnt.
    let mut specks: Vec<(usize, usize)> = Vec::new();
    for y in (1..height - 1).step_by(4) {
        for x in (1..width - 1).step_by(4) {
            if specks.len() < 8 && in_verified(x, y) {
                specks.push((x, y));
            }
        }
    }
    let report = decode(&with_misreads(&layout, &clean, failing, &specks), &layout, mask, &Pins::new(), &opts);
    assert!(report.blocks[..failing].iter().all(|b| b.is_verified()), "{:?}", report.blocks);
    let specked = report.clustering.expect("learnt from the verified blocks");
    assert!(specked.gross[0] < 1.0 / 32.0, "{specked:?}");
    assert!(specked.gross[0] < stained.gross[0] && specked.gross[1] <= stained.gross[1]);
}

#[test]
fn noise_under_the_message_evidence_passes_no_oftener_than_its_risk_says() {
    // 3-L: one block of 55 data and 15 error correction codewords, read
    // as the evidence test reads a real message, the high bit of every
    // byte and ninety bits more unread, but with everything after the
    // byte segment's header noise. Taking the high bits at their lean
    // leaves thirty equations, and noise is consistent with them once in
    // 2^30; doubting the leans to make it consistent costs a lean's
    // worth of likelihood each, which the confirmation is charged. So a
    // block that verifies here is a wrong codeword, and must do so no
    // oftener than a block's risk, one in 2^8 at the most, says.
    use qr_forensics::symbol::grid::Module;
    let (version, ec) = (SymbolVersion::Qr(3), EcLevel::L);
    let layout = layout(version, ec);
    let text = "https://example.com/q?id=7Q2M-XK4P-99AD";
    let block = &layout.blocks[0];
    let module_of = |bit: usize| {
        let (cw, place) = layout.stream_bit_location(bit).unwrap();
        let pos = layout.codeword_modules(cw)[place as usize];
        (pos.x as usize, pos.y as usize)
    };
    let mut rng = Rng(0x0153);
    let (trials, mut passed, mut solved) = (256, 0, 0);
    for _ in 0..trials {
        let mask = rng.below(8) as u8;
        let mut grid = encode(version, ec, Some(mask), &[Segment::bytes(text)]).unwrap();
        // The header is in the first two codewords: the rest is noise.
        for &cw in &block.codewords[2..] {
            for pos in layout.codeword_modules(cw) {
                let noise = if rng.below(2) == 0 { Module::Dark } else { Module::Light };
                grid.set(pos.x as usize, pos.y as usize, noise);
            }
        }
        for k in 0..text.len() {
            let (x, y) = module_of(12 + 8 * k);
            grid.set(x, y, Module::Unknown);
        }
        let mut others = 0;
        while others < 90 {
            let bit = 16 + rng.below(8 * text.len() - 4);
            let (x, y) = module_of(bit);
            if bit % 8 == 4 || grid.get(x, y) == Module::Unknown {
                continue;
            }
            grid.set(x, y, Module::Unknown);
            others += 1;
        }
        let report = decode_plain(&grid, &layout, mask);
        solved += usize::from(report.blocks[0].is_solved());
        passed += usize::from(report.blocks[0].is_verified());
    }
    // Two hundred and fifty-six blocks at one in 2^8 each: one expected
    // at the most, and none at all with the risks they would claim.
    assert!(passed <= 2, "{passed} of {trials} blocks of noise verified, {solved} solved");
}
