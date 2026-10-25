//! Format and version information under damage, and brute force when the
//! format bits are gone.

mod common;

use common::*;
use qr_forensics::decode::bch::{analyze_format, analyze_version};
use qr_forensics::decode::pipeline::{DecodeOptions, LIKENESS_FLOOR, analyze, brute_force, brute_force_mirrored};
use qr_forensics::symbol::function_map::InfoCopy;
use qr_forensics::symbol::grid::{Mirror, Module, ModuleGrid};
use qr_forensics::symbol::{EcLevel, Family, SymbolVersion};
use qrcode_generator::Segment;

fn flip(grid: &mut ModuleGrid, copy: &InfoCopy, bits: &[usize]) {
    for &b in bits {
        let pos = copy.bits[b];
        let (x, y) = (pos.x as usize, pos.y as usize);
        grid.set(x, y, grid.get(x, y).inverted());
    }
}

fn wipe(grid: &mut ModuleGrid, copy: &InfoCopy) {
    for pos in &copy.bits {
        grid.set(pos.x as usize, pos.y as usize, Module::Unknown);
    }
}

fn cases() -> Vec<(SymbolVersion, EcLevel, u8)> {
    let mut cases = Vec::new();
    for (v, ec) in [(1, EcLevel::L), (3, EcLevel::M), (9, EcLevel::Q), (22, EcLevel::H)] {
        for mask in 0..8 {
            cases.push((SymbolVersion::Qr(v), ec, mask));
        }
    }
    for (v, ec) in [(1, EcLevel::L), (2, EcLevel::M), (3, EcLevel::L), (4, EcLevel::Q)] {
        for mask in 0..4 {
            cases.push((SymbolVersion::Micro(v), ec, mask));
        }
    }
    for v in [0, 7, 15, 19, 31] {
        cases.push((SymbolVersion::Rmqr(v), EcLevel::M, 0));
        cases.push((SymbolVersion::Rmqr(v), EcLevel::H, 0));
    }
    cases
}

fn symbol(version: SymbolVersion, ec: EcLevel, mask: u8) -> ModuleGrid {
    // Short enough to leave padding in most symbols, and to fit the smallest.
    let room = byte_capacity(&layout(version, ec), 0);
    let segment = match version {
        SymbolVersion::Micro(1 | 2) => Segment::numeric("314").unwrap(),
        _ => Segment::bytes(&b"format"[..room.min(6)]),
    };
    encode(version, ec, Some(mask), &[segment]).unwrap()
}

#[test]
fn survives_three_flips_in_every_copy() {
    let mut rng = Rng(0xF0);
    for (version, ec, mask) in cases() {
        let layout = layout(version, ec);
        let clean = symbol(version, ec, mask);
        for _ in 0..10 {
            let mut grid = clean.clone();
            // BCH(15,5) has distance 7 and BCH(18,6) distance 8: three wrong
            // bits in a copy still leave the true word strictly closest.
            for copy in &layout.func.format {
                flip(&mut grid, copy, &rng.distinct(3, copy.bits.len()));
            }
            let analysis = analyze_format(&grid, &layout.func);
            let best = analysis.best().unwrap();
            assert_eq!((best.fields.ec, best.fields.mask), (ec, mask), "{version}-{ec:?} mask {mask}");
            assert_eq!(best.mismatches as usize, 3 * layout.func.format.len());
            assert!(!analysis.is_ambiguous());
        }
    }
}

#[test]
fn reads_beyond_correction_range_are_not_trusted() {
    for (version, ec, mask) in cases() {
        let layout = layout(version, ec);
        let clean = symbol(version, ec, mask);
        let analysis = analyze_format(&clean, &layout.func);
        assert!(analysis.is_reliable(), "{version}-{ec:?} mask {mask}");
        assert!(!analysis.is_doubtful());
        assert_eq!(analysis.correctable(), 3);

        // The same four flips in every copy: more than the code corrects,
        // and no mixing of halves gets round flips the copies share. Four
        // flips often land within three bits of another word, where the
        // code is simply past its range and reads that word; but every
        // word has some run of four flips that lands at least four bits
        // from all of them, and such a read must not be trusted.
        let mut shown = false;
        for shift in 0..=layout.func.format[0].bits.len() - 4 {
            let mut grid = clean.clone();
            for copy in &layout.func.format {
                let bits: Vec<usize> = (shift..shift + 4).collect();
                flip(&mut grid, copy, &bits);
            }
            let analysis = analyze_format(&grid, &layout.func);
            let best = analysis.best().unwrap();
            if best.best_mismatches <= 3 {
                continue;
            }
            assert_eq!(best.best_mismatches, 4, "{version}-{ec:?} mask {mask}");
            assert!(analysis.is_doubtful(), "{version}-{ec:?} mask {mask}");
            assert!(!analysis.is_reliable());
            shown = true;
            break;
        }
        assert!(shown, "{version}-{ec:?} mask {mask}: every run of four flips read as another word");

        // With nothing painted there is nothing to doubt, only ambiguity.
        let blank = layout.func.blank_grid();
        let analysis = analyze_format(&blank, &layout.func);
        assert!(!analysis.is_doubtful() && analysis.is_ambiguous() && !analysis.is_reliable());
        assert_eq!(analysis.correctable(), 0);
    }
}

#[test]
fn one_clean_copy_outweighs_a_trashed_one() {
    for (version, ec, mask) in cases() {
        let layout = layout(version, ec);
        if layout.func.format.len() < 2 {
            continue;
        }
        for keep in 0..2 {
            let mut grid = symbol(version, ec, mask);
            // Five wrong bits in one half of one copy: beyond correction on
            // its own, and before, summed with the clean copy, enough to
            // make the whole read doubtful. Five flips in one half can
            // never assemble into another word, which would need seven.
            let trashed = &layout.func.format[1 - keep];
            flip(&mut grid, trashed, &[0, 1, 2, 3, 4]);
            let analysis = analyze_format(&grid, &layout.func);
            let best = analysis.best().unwrap();
            assert_eq!((best.fields.ec, best.fields.mask), (ec, mask), "{version}-{ec:?} mask {mask}");
            assert_eq!((best.best_mismatches, best.mismatches), (0, 5));
            assert_eq!(best.per_copy[keep], (0, trashed.bits.len() as u32));
            assert!(analysis.is_reliable(), "{version}-{ec:?} mask {mask}");
        }
    }
}

#[test]
fn halves_of_two_damaged_copies_assemble_into_a_clean_word() {
    for (version, ec, mask) in cases() {
        let layout = layout(version, ec);
        if version.family() != Family::Qr {
            continue;
        }
        let mut grid = symbol(version, ec, mask);
        // Four wrong bits in the low half of one copy and four in the high
        // half of the other: neither copy is within range, the assembly is.
        flip(&mut grid, &layout.func.format[0], &[0, 2, 4, 6]);
        flip(&mut grid, &layout.func.format[1], &[8, 10, 12, 14]);
        let analysis = analyze_format(&grid, &layout.func);
        let best = analysis.best().unwrap();
        assert_eq!((best.fields.ec, best.fields.mask, best.best_mismatches, best.mismatches), (ec, mask, 0, 8));
        assert!(analysis.is_reliable(), "{version}-{ec:?} mask {mask}");
    }
}

#[test]
fn format_words_painted_without_the_mask_are_recognised() {
    for (version, ec, mask) in cases() {
        let layout = layout(version, ec);
        let clean = symbol(version, ec, mask);
        let mut grid = clean.clone();
        // Undo the XOR the encoder applied, as some encoders forget it.
        for copy in &layout.func.format {
            for (i, pos) in copy.bits.iter().enumerate() {
                if copy.xor >> i & 1 != 0 {
                    let (x, y) = (pos.x as usize, pos.y as usize);
                    grid.set(x, y, grid.get(x, y).inverted());
                }
            }
        }
        let analysis = analyze_format(&grid, &layout.func);
        // A word that comes out all light or all dark without its mask is
        // what bare paper or a blot of ink reads as, and settles nothing:
        // the search of every format finds it, as it finds the format of a
        // symbol whose format bits are gone.
        if analysis.copies.iter().all(|c| c.is_blank()) {
            assert!(!analysis.is_reliable() && !analysis.unmasked_in_use(), "{version}-{ec:?} mask {mask}");
            let attempts = brute_force(&grid, &layout.func, None, &DecodeOptions::default());
            assert_eq!((attempts[0].ec, attempts[0].mask), (ec, mask), "{version}-{ec:?} mask {mask}");
            assert!(attempts[0].report.is_trustworthy(), "{version}-{ec:?} mask {mask}");
            continue;
        }
        let best = analysis.best().unwrap();
        assert!(best.unmasked, "{version}-{ec:?} mask {mask}");
        assert_eq!((best.fields.ec, best.fields.mask, best.best_mismatches), (ec, mask, 0));
        assert!(analysis.is_reliable() && analysis.unmasked_in_use(), "{version}-{ec:?} mask {mask}");
        // The standard reading is never ranked behind the quirk.
        assert!(!analyze_format(&clean, &layout.func).unmasked_in_use());
    }
}

#[test]
fn a_blotted_format_strip_leaves_the_format_to_the_search() {
    // Ink over every copy of the format bits, or bare paper where they
    // were: the bits spell a word painted without the mask, to the bit,
    // and were once taken at that word, the symbol decoded under a level
    // and mask it never had and every other format left untried.
    let options = DecodeOptions::default();
    for (version, ec, mask) in cases() {
        let layout = layout(version, ec);
        let clean = symbol(version, ec, mask);
        for blot in [Module::Dark, Module::Light] {
            let mut grid = clean.clone();
            for copy in &layout.func.format {
                for pos in &copy.bits {
                    grid.set(pos.x as usize, pos.y as usize, blot);
                }
            }
            let what = format!("{version}-{ec:?} mask {mask}, {blot:?}");
            let analysis = analyze_format(&grid, &layout.func);
            assert!(!analysis.is_reliable(), "{what}");
            let attempts = brute_force(&grid, &layout.func, None, &options);
            assert_eq!((attempts[0].ec, attempts[0].mask), (ec, mask), "{what}");
            assert!(attempts[0].report.is_trustworthy(), "{what}");
        }
    }
}

#[test]
fn mirrored_grids_are_told_by_their_format_bits() {
    for (version, ec, mask) in cases() {
        let layout = layout(version, ec);
        let clean = symbol(version, ec, mask);
        let what = format!("{version}-{ec:?} mask {mask}");
        assert_eq!(analyze(&clean, &layout.func).mirrored, None, "{what}");
        // An inverted grid is judged once its colours are right.
        assert_eq!(analyze(&clean.inverted(), &layout.func).mirrored, None, "{what}");
        for &mirror in version.mirrors() {
            let seen_from_behind = clean.mirrored(mirror);
            let analysis = analyze(&seen_from_behind, &layout.func);
            assert_eq!(analysis.mirrored, Some(mirror), "{what}");
            assert_eq!(seen_from_behind.mirrored(mirror), clean, "{what}: the reflection undoes itself");
            // The fixed patterns of a square symbol cannot tell: only the
            // format bits can.
            if version.family() != Family::Rmqr {
                let (matching, known) = analysis.fixed;
                assert!(known - matching <= 1, "{what}: {matching} of {known}");
            }
        }
    }
    assert_eq!(SymbolVersion::Qr(3).mirrors(), [Mirror::Transpose]);
    assert_eq!(SymbolVersion::Rmqr(3).mirrors(), [Mirror::Horizontal, Mirror::Vertical]);
}

#[test]
fn one_surviving_copy_is_enough() {
    for (version, ec, mask) in cases() {
        let layout = layout(version, ec);
        if layout.func.format.len() < 2 {
            continue;
        }
        for keep in 0..2 {
            let mut grid = symbol(version, ec, mask);
            wipe(&mut grid, &layout.func.format[1 - keep]);
            let analysis = analyze_format(&grid, &layout.func);
            let best = analysis.best().unwrap();
            assert_eq!((best.fields.ec, best.fields.mask, best.mismatches), (ec, mask, 0));
            assert_eq!(analysis.copies[1 - keep].known_count(), 0);
            assert!(analysis.margin() >= 7);
        }
    }
}

#[test]
fn wiped_format_is_ambiguous_and_brute_force_finds_it() {
    for (version, ec, mask) in cases() {
        let layout = layout(version, ec);
        let mut grid = symbol(version, ec, mask);
        for copy in &layout.func.format {
            wipe(&mut grid, copy);
        }
        let analysis = analyze_format(&grid, &layout.func);
        // M1 has a single EC level, and rMQR a single mask, but there is
        // always more than one candidate to choose between.
        assert!(analysis.is_ambiguous(), "{version}");
        assert_eq!(analysis.candidates[0].compared, 0);

        let attempts = brute_force(&grid, &layout.func, None, &DecodeOptions::default());
        assert_eq!(attempts.len(), version.ec_levels().len() * version.mask_count() as usize);
        let best = &attempts[0];
        assert_eq!((best.ec, best.mask), (ec, mask), "{version}-{ec:?} mask {mask}");
        assert!(best.report.is_trustworthy());
        // A lower EC level with the same block count accepts the same
        // codewords, so the runner-up may also look clean; it must still
        // rank strictly below, and no other mask may come close.
        assert!(best.report.score > attempts[1].report.score, "{version}-{ec:?} mask {mask}");
        for other in attempts.iter().filter(|a| a.mask != mask) {
            assert!(!other.report.score.all_verified, "{version}-{ec:?}: mask {} also verifies", other.mask);
        }
    }
}

#[test]
fn brute_force_works_on_a_damaged_symbol() {
    let mut rng = Rng(0xB2);
    for (version, ec, mask) in
        [(SymbolVersion::Qr(6), EcLevel::Q, 6), (SymbolVersion::Qr(18), EcLevel::M, 1), (SymbolVersion::Micro(4), EcLevel::M, 2)]
    {
        let layout = layout(version, ec);
        let text = rng.alphanumeric(byte_capacity(&layout, 4)).to_lowercase();
        let mut grid = encode(version, ec, Some(mask), &[Segment::bytes(&text)]).unwrap();
        for copy in &layout.func.format {
            wipe(&mut grid, copy);
        }
        for block in &layout.blocks {
            let picked = rng.distinct(block.ec_len / 2, block.codewords.len());
            for (n, &i) in picked.iter().enumerate() {
                if n % 3 == 0 {
                    corrupt_codeword(&mut grid, &layout, block.codewords[i], n);
                } else {
                    erase_codeword(&mut grid, &layout, block.codewords[i]);
                }
            }
        }
        let attempts = brute_force(&grid, &layout.func, None, &DecodeOptions::default());
        assert_eq!((attempts[0].ec, attempts[0].mask), (ec, mask), "{version}");
        assert_eq!(attempts[0].report.text.plain(), text);
        assert!(attempts[0].report.score > attempts[1].report.score);
    }
}

#[test]
fn format_implying_another_size_is_flagged() {
    // A Micro QR format word says which version it belongs to. Paint the M4
    // word onto an M2 grid and the best-matching candidate no longer fits.
    let m4 = symbol(SymbolVersion::Micro(4), EcLevel::L, 1);
    let m4_func = layout(SymbolVersion::Micro(4), EcLevel::L).func;
    let m2_layout = layout(SymbolVersion::Micro(2), EcLevel::L);
    let mut grid = symbol(SymbolVersion::Micro(2), EcLevel::L, 0);
    for (from, to) in m4_func.format[0].bits.iter().zip(&m2_layout.func.format[0].bits) {
        grid.set(to.x as usize, to.y as usize, m4.get(from.x as usize, from.y as usize));
    }
    let analysis = analyze_format(&grid, &m2_layout.func);
    let exact = analysis.candidates.iter().find(|c| c.mismatches == 0 && !c.unmasked).unwrap();
    assert!(!exact.fits_grid);
    assert_eq!(exact.fields.version, SymbolVersion::Micro(4));
    assert_eq!(analysis.closest().unwrap().fields.version, SymbolVersion::Micro(4));
    // Candidates that fit still sort first, at a distance.
    assert!(analysis.candidates[0].fits_grid);
    assert!(analysis.candidates[0].best_mismatches >= 7);

    // rMQR format words carry the version too; every word is tried both
    // as the standard paints it and without the mask.
    let r = layout(SymbolVersion::Rmqr(11), EcLevel::H);
    let analysis = analyze_format(&symbol(SymbolVersion::Rmqr(11), EcLevel::H, 0), &r.func);
    assert_eq!(analysis.candidates.iter().filter(|c| c.fits_grid && !c.unmasked).count(), 2);
    assert_eq!(analysis.candidates.len(), 128);
    assert_eq!(analysis.best().unwrap().fields.version, SymbolVersion::Rmqr(11));
}

/// Paint one info copy so that its known bits lie one bit from the true
/// word `a` and one bit from another word `b`: of the bits where the two
/// words differ, all but two are made unknown, and of those two the one
/// where the read is wrong for `a` is sampled faintly, the one where it is
/// wrong for `b` solidly. Returns the two bit indices.
fn paint_tie(grid: &mut ModuleGrid, copy: &InfoCopy, a: u32, b: u32, faint: u8) -> (usize, usize) {
    let differ: Vec<usize> = (0..copy.bits.len()).filter(|&i| (a ^ b) >> i & 1 != 0).collect();
    let (p, q) = (differ[0], differ[1]);
    for (i, pos) in copy.bits.iter().enumerate() {
        let (x, y) = (pos.x as usize, pos.y as usize);
        let painted = (a ^ copy.xor) >> i & 1 != 0;
        if differ[2..].contains(&i) {
            grid.set(x, y, Module::Unknown);
        } else if i == p {
            grid.set_sampled(x, y, Module::from_dark(!painted), faint);
        } else {
            grid.set_sampled(x, y, Module::from_dark(painted), 200);
        }
    }
    (p, q)
}

#[test]
fn faint_bits_decide_between_words_at_the_same_distance() {
    // Two words one bit from the read each: by count the bits cannot
    // tell them apart, but the bit the true word overturns was read near
    // the threshold and the bit the other overturns well inside its
    // colour, and a faint bit is the one most likely misread.
    use qr_forensics::decode::bch::{WEIGHT_MARGIN, format_word_for, format_words};
    use qr_forensics::decode::rs::bit_cost;
    let mut shown = 0;
    for (version, ec, mask) in cases() {
        let layout = layout(version, ec);
        let a = format_word_for(version, ec, mask).unwrap();
        let Some((fields_b, b)) = format_words(version)
            .into_iter()
            .find(|&(f, w)| f.version == version && (7..=9).contains(&(a ^ w).count_ones()))
        else {
            continue;
        };
        let mut grid = symbol(version, ec, mask);
        for copy in &layout.func.format[1..] {
            wipe(&mut grid, copy);
        }
        let copy = &layout.func.format[0];
        let (p, _) = paint_tie(&mut grid, copy, a, b, 10);
        let analysis = analyze_format(&grid, &layout.func);
        let what = format!("{version}-{ec:?} mask {mask} against {:?}", fields_b);
        let best = analysis.best().unwrap();
        let rival = analysis.candidates.iter().find(|c| c.fields == fields_b && c.is_standard()).unwrap();
        assert_eq!((best.best_mismatches, rival.best_mismatches), (1, 1), "{what}");
        if analysis.candidates.iter().filter(|c| c.fits_grid && c.is_standard() && c.best_mismatches <= 1).count() != 2 {
            continue;
        }
        shown += 1;
        assert_eq!((best.fields.ec, best.fields.mask), (ec, mask), "{what}");
        assert_eq!((best.best_weight, rival.best_weight), (bit_cost(10), bit_cost(200)), "{what}");
        assert_eq!(analysis.margin(), 0, "{what}");
        assert!(analysis.weight_margin() >= WEIGHT_MARGIN, "{what}");
        assert!(!analysis.is_ambiguous() && analysis.is_reliable(), "{what}");
        assert_eq!(analysis.verdict(), Some(true), "{what}");

        // Read with the same margin, the two bits weigh the same and the
        // tie stands.
        let pos = copy.bits[p];
        let (x, y) = (pos.x as usize, pos.y as usize);
        grid.set_sampled(x, y, grid.get(x, y), 200);
        let analysis = analyze_format(&grid, &layout.func);
        assert_eq!((analysis.margin(), analysis.weight_margin()), (0, 0), "{what}");
        assert!(analysis.is_ambiguous() && !analysis.is_reliable(), "{what}");
        assert_eq!(analysis.verdict(), None, "{what}");
        // Painted by hand, every bit is certain, and the tie stands too.
        let pos = copy.bits[p];
        grid.set(pos.x as usize, pos.y as usize, grid.get(pos.x as usize, pos.y as usize));
        assert!(analyze_format(&grid, &layout.func).is_ambiguous(), "{what}");
    }
    assert!(shown >= 20, "{shown} ties shown");
}

#[test]
fn faint_bits_decide_between_versions_at_the_same_distance() {
    use qr_forensics::decode::bch::{WEIGHT_MARGIN, info18};
    use qr_forensics::decode::rs::bit_cost;
    let mut shown = 0;
    for v in [7u8, 11, 19, 27, 33, 40] {
        let version = SymbolVersion::Qr(v);
        let layout = layout(version, EcLevel::M);
        let a = info18(v as u32);
        let Some(other) = (7..=40u8).find(|&o| o != v && (8..=10).contains(&(a ^ info18(o as u32)).count_ones())) else {
            continue;
        };
        let mut grid = encode(version, EcLevel::M, Some(0), &[Segment::bytes("v")]).unwrap();
        wipe(&mut grid, &layout.func.version_info[1]);
        let copy = &layout.func.version_info[0];
        let (p, _) = paint_tie(&mut grid, copy, a, info18(other as u32), 10);
        let analysis = analyze_version(&grid, &layout.func).unwrap();
        let what = format!("version {v} against {other}");
        if analysis.mismatches != 1 || analysis.margin != 0 {
            continue;
        }
        shown += 1;
        assert_eq!((analysis.best_version, analysis.weight), (v, bit_cost(10)), "{what}");
        assert!(analysis.weight_margin >= WEIGHT_MARGIN, "{what}: {}", analysis.weight_margin);
        assert!(analysis.is_reliable(), "{what}");
        let pos = copy.bits[p];
        let (x, y) = (pos.x as usize, pos.y as usize);
        grid.set_sampled(x, y, grid.get(x, y), 200);
        let analysis = analyze_version(&grid, &layout.func).unwrap();
        assert_eq!((analysis.margin, analysis.weight_margin), (0, 0), "{what}");
        assert!(!analysis.is_reliable(), "{what}");
    }
    assert!(shown >= 3, "{shown} ties shown");
}

#[test]
fn version_information_survives_damage() {
    use qr_forensics::decode::bch::info18;
    let mut rng = Rng(0x7E);
    for v in [7u8, 13, 26, 40] {
        let version = SymbolVersion::Qr(v);
        let layout = layout(version, EcLevel::M);
        let clean = encode(version, EcLevel::M, Some(0), &[Segment::bytes("v")]).unwrap();
        let mut grid = clean.clone();
        for copy in &layout.func.version_info {
            flip(&mut grid, copy, &rng.distinct(3, 18));
        }
        let analysis = analyze_version(&grid, &layout.func).unwrap();
        // Three flips in each copy: each copy alone is three off, and the
        // halves assemble into a read no worse.
        assert_eq!(analysis.best_version, v);
        assert!(analysis.mismatches <= 3, "{}", analysis.mismatches);
        for copy in &analysis.copies {
            assert_eq!(copy.distance(info18(v as u32)), (3, 18));
        }
        assert!(analysis.margin >= 2);
        assert!(analysis.is_reliable());

        let mut grid = clean.clone();
        wipe(&mut grid, &layout.func.version_info[0]);
        let analysis = analyze_version(&grid, &layout.func).unwrap();
        assert_eq!((analysis.best_version, analysis.mismatches, analysis.compared), (v, 0, 18));
    }
    let v6 = layout(SymbolVersion::Qr(6), EcLevel::M);
    assert!(analyze_version(&v6.func.blank_grid(), &v6.func).is_none());
    assert_eq!(Family::Qr.versions().len(), 40);
}

#[test]
fn halves_of_two_damaged_version_copies_assemble_into_a_clean_word() {
    use qr_forensics::decode::bch::info18;
    for v in [7u8, 13, 26, 40] {
        let version = SymbolVersion::Qr(v);
        let layout = layout(version, EcLevel::M);
        let mut grid = encode(version, EcLevel::M, Some(0), &[Segment::bytes("v")]).unwrap();
        // Four wrong bits in the low half of one copy and four in the high
        // half of the other: neither copy is within range, the assembly is.
        flip(&mut grid, &layout.func.version_info[0], &[0, 2, 4, 6]);
        flip(&mut grid, &layout.func.version_info[1], &[9, 11, 13, 15]);
        let analysis = analyze_version(&grid, &layout.func).unwrap();
        for copy in &analysis.copies {
            assert_eq!(copy.distance(info18(v as u32)), (4, 18), "version {v}");
        }
        assert_eq!((analysis.best_version, analysis.mismatches, analysis.compared), (v, 0, 18));
        assert!(analysis.margin >= 2 && analysis.is_reliable(), "version {v}");
        assert_eq!(analysis.verdict(), Some(true));
        // The same damage in the same half of both copies is beyond the
        // code, and assembling cannot get round it.
        let mut grid = encode(version, EcLevel::M, Some(0), &[Segment::bytes("v")]).unwrap();
        for copy in &layout.func.version_info {
            flip(&mut grid, copy, &[0, 2, 4, 6]);
        }
        let analysis = analyze_version(&grid, &layout.func).unwrap();
        assert!(analysis.mismatches >= 4 && !analysis.is_reliable(), "version {v}");
    }
}

/// A sentence of text in a version 10 symbol at level M under `mask`,
/// both copies of its format bits gone, and one module in twelve of its
/// codewords read the other colour, more than any block corrects: no
/// block verifies under any level and mask. With the symbol's layout.
fn text_past_correcting(mask: u8) -> (qr_forensics::symbol::layout::SymbolLayout, ModuleGrid) {
    let (version, ec) = (SymbolVersion::Qr(10), EcLevel::M);
    let layout = layout(version, ec);
    let sentence = "the quick brown fox jumps over the lazy dog, and a photograph of it was taken from across the street. ";
    let text: String = sentence.repeat(4).chars().take(byte_capacity(&layout, 2)).collect();
    let mut grid = encode(version, ec, Some(mask), &[Segment::bytes(&text)]).unwrap();
    for copy in &layout.func.format {
        wipe(&mut grid, copy);
    }
    let mut rng = Rng(11 + u64::from(mask));
    for cw in 0..layout.codewords.len() {
        for pos in layout.codeword_modules(cw) {
            if rng.below(12) == 0 {
                let (x, y) = (pos.x as usize, pos.y as usize);
                grid.set(x, y, grid.get(x, y).inverted());
            }
        }
    }
    (layout, grid)
}

#[test]
fn with_the_format_bits_gone_and_no_block_confirmed_the_text_tells_the_format() {
    // The attempts at a symbol no block of which verifies were once
    // ranked by the unconfirmed solutions noise yields, and then by the
    // format bits, which say nothing. Under the right level and mask
    // the data reads as text, misread as it is; under the others it is
    // noise, or the text scrambled by the difference of two masks.
    let ec = EcLevel::M;
    for mask in [0, 3, 5] {
        let (layout, grid) = text_past_correcting(mask);
        let attempts = brute_force(&grid, &layout.func, None, &DecodeOptions::default());
        assert!(attempts.iter().all(|a| a.report.blocks_verified() == 0), "mask {mask}: a block verified");
        assert_eq!((attempts[0].ec, attempts[0].mask), (ec, mask), "mask {mask}: {:?}", attempts.iter().take(3).map(|a| (a.ec, a.mask, a.likeness_bits)).collect::<Vec<_>>());
        assert!(attempts[0].likeness_bits >= LIKENESS_FLOOR, "mask {mask}: {}", attempts[0].likeness_bits);
    }
}

#[test]
fn seen_from_behind_with_the_format_bits_gone_the_text_tells_the_format() {
    // The same symbol seen from the back: its mirror images' attempts are
    // ranked together as those of one grid are, by the data and the
    // format bits where no block is confirmed, and not by the
    // unconfirmed solutions noise yields, as they once were.
    let ec = EcLevel::M;
    for mask in [0, 3, 5] {
        let (layout, grid) = text_past_correcting(mask);
        let attempts = brute_force_mirrored(&grid.mirrored(Mirror::Transpose), &layout.func, &DecodeOptions::default());
        assert!(attempts.iter().all(|a| a.report.blocks_verified() == 0), "mask {mask}: a block verified");
        let best = &attempts[0];
        assert_eq!(
            (best.ec, best.mask, best.mirror),
            (ec, mask, Some(Mirror::Transpose)),
            "mask {mask}: {:?}",
            attempts.iter().take(3).map(|a| (a.ec, a.mask, a.likeness_bits)).collect::<Vec<_>>()
        );
    }
}
