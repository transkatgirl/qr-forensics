//! Clean symbols from an independent encoder must decode exactly, for every
//! version, EC level and mask, and agree with an independent decoder.

mod common;

use common::*;
use qr_forensics::decode::bch::{analyze_format, analyze_version};
use qr_forensics::decode::bitstream::{Certainty, EndOfData};
use qr_forensics::decode::pipeline::{BlockOutcome, CwStatus, DecodeReport, SolveMethod, analyze};
use qr_forensics::decode::text::Charset;
use qr_forensics::symbol::grid::{Module, ModuleGrid};
use qr_forensics::symbol::layout::SymbolLayout;
use qr_forensics::symbol::{EcLevel, Family, SymbolVersion};
use qrcode_generator::Segment;

/// Checks that hold for any undamaged symbol.
fn assert_clean(grid: &ModuleGrid, layout: &SymbolLayout, ec: EcLevel, mask: u8, what: &str) -> DecodeReport {
    let func = &layout.func;
    let (matching, known) = func.fixed_agreement(grid);
    assert_eq!(matching, known, "{what}: fixed patterns");

    let format = analyze_format(grid, func);
    let best = format.best().unwrap_or_else(|| panic!("{what}: no format candidate fits"));
    assert_eq!((best.fields.ec, best.fields.mask, best.mismatches), (ec, mask, 0), "{what}: format");
    assert!(format.margin() >= 7, "{what}: format margin");
    if let Some(version) = analyze_version(grid, func) {
        assert_eq!(SymbolVersion::Qr(version.best_version), func.version, "{what}: version info");
        assert_eq!(version.mismatches, 0);
    }

    let report = decode_plain(grid, layout, mask);
    for (b, block) in report.blocks.iter().enumerate() {
        assert_eq!(
            block.outcome,
            BlockOutcome::Solved {
                method: SolveMethod::ReedSolomon,
                errors: 0,
                erasures: 0,
                slack: block.ec_len,
                confidence_bits: 8 * block.ec_len as u32,
                risk_bits: 8 * block.ec_len as u32,
                beyond_standard: false,
            },
            "{what}: block {b}"
        );
    }
    assert!(report.codewords.iter().all(|c| c.status == CwStatus::Clean), "{what}: codewords");
    assert!(report.stream.certainty.iter().all(|&c| c == Certainty::Verified));
    assert!(matches!(report.parse.end, EndOfData::At { .. }), "{what}: {:?}", report.parse.end);
    assert!(report.is_trustworthy(), "{what}: score {:?}, tail {:?}", report.score, report.tail);
    report
}

#[test]
fn qr_every_version_level_and_mask() {
    let mut rng = Rng(0x5EED);
    for v in 1..=40u8 {
        let version = SymbolVersion::Qr(v);
        for &ec in version.ec_levels() {
            let layout = layout(version, ec);
            for mask in 0..8u8 {
                // Alternate between nearly full and mostly padding.
                let capacity = byte_capacity(&layout, 0);
                let len = if mask % 2 == 0 { capacity } else { 1 + rng.below(capacity.div_ceil(3)) };
                let payload = rng.bytes(len);
                let what = format!("{v}-{ec:?} mask {mask}");
                let grid = encode(version, ec, Some(mask), &[Segment::bytes(&payload)])
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                let report = assert_clean(&grid, &layout, ec, mask, &what);
                let got: Vec<u8> = payload_bytes(&report).into_iter().map(Option::unwrap).collect();
                assert_eq!(got, payload, "{what}");
            }
        }
    }
}

#[test]
fn micro_every_version_level_and_mask() {
    let mut rng = Rng(0xA11CE);
    for v in 1..=4u8 {
        let version = SymbolVersion::Micro(v);
        for &ec in version.ec_levels() {
            let layout = layout(version, ec);
            for mask in 0..4u8 {
                for fill in [true, false] {
                    let what = format!("M{v}-{ec:?} mask {mask} fill {fill}");
                    // Digits everywhere; bytes too where the version has them.
                    let (segment, expect) = if v >= 3 && mask % 2 == 0 {
                        let capacity = byte_capacity(&layout, 0);
                        let len = if fill { capacity } else { 1 };
                        let text = rng.alphanumeric(len).to_lowercase().replace(['$', '%', '*'], "x");
                        (Segment::bytes(text.as_bytes()), text)
                    } else {
                        let header =
                            (layout.stream.mode_bits + [3, 4, 5, 6][v as usize - 1]) as usize;
                        let room = layout.data_bits - header;
                        let capacity = room / 10 * 3 + [0, 0, 0, 0, 1, 1, 1, 2, 2, 2][room % 10];
                        let len = if fill { capacity } else { 1 };
                        let digits = rng.digits(len);
                        (Segment::numeric(&digits).unwrap(), digits)
                    };
                    let grid = encode(version, ec, Some(mask), &[segment])
                        .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                    let report = assert_clean(&grid, &layout, ec, mask, &what);
                    assert_eq!(report.text.plain(), expect, "{what}");
                }
            }
        }
    }
}

#[test]
fn rmqr_every_version_and_level() {
    let mut rng = Rng(0xB0B);
    for v in 0..32u8 {
        let version = SymbolVersion::Rmqr(v);
        for &ec in version.ec_levels() {
            let layout = layout(version, ec);
            for fill in [true, false] {
                let capacity = byte_capacity(&layout, 0);
                let len = if fill { capacity } else { 1 + rng.below(capacity.div_ceil(3)) };
                let payload = rng.bytes(len);
                let what = format!("{version}-{ec:?} fill {fill}");
                let grid = encode(version, ec, None, &[Segment::bytes(&payload)])
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                let report = assert_clean(&grid, &layout, ec, 0, &what);
                let got: Vec<u8> = payload_bytes(&report).into_iter().map(Option::unwrap).collect();
                assert_eq!(got, payload, "{what}");
            }
        }
    }
}

#[test]
fn every_mode_in_every_family() {
    use qrcode_generator::qr::EciAssignment;
    let cases: Vec<(SymbolVersion, EcLevel, Vec<Segment>, &str)> = vec![
        (
            SymbolVersion::Qr(5),
            EcLevel::M,
            vec![
                Segment::numeric("0123456789012").unwrap(),
                Segment::alphanumeric("HELLO WORLD $%*+-./:").unwrap(),
                Segment::bytes("mixed, ünïcödé ✓"),
                Segment::kanji("点茗日本語").unwrap(),
            ],
            "0123456789012HELLO WORLD $%*+-./:mixed, ünïcödé ✓点茗日本語",
        ),
        (
            SymbolVersion::Qr(12),
            EcLevel::Q,
            vec![Segment::numeric("7").unwrap(), Segment::alphanumeric("A").unwrap(), Segment::numeric("42").unwrap()],
            "7A42",
        ),
        (
            SymbolVersion::Qr(30),
            EcLevel::H,
            vec![Segment::alphanumeric("VERSION 30 USES WIDER COUNTS").unwrap(), Segment::kanji("漢字").unwrap()],
            "VERSION 30 USES WIDER COUNTS漢字",
        ),
        (
            SymbolVersion::Qr(3),
            EcLevel::L,
            vec![Segment::eci(EciAssignment::new(7).unwrap()), Segment::bytes([0xBF, 0xE0, 0xD8, 0xD2, 0xD5, 0xE2])],
            "Привет",
        ),
        (
            SymbolVersion::Micro(2),
            EcLevel::L,
            vec![Segment::alphanumeric("AB").unwrap(), Segment::numeric("12").unwrap()],
            "AB12",
        ),
        (
            SymbolVersion::Micro(4),
            EcLevel::M,
            vec![Segment::kanji("点茗").unwrap(), Segment::bytes("ok"), Segment::numeric("5").unwrap()],
            "点茗ok5",
        ),
        (
            SymbolVersion::Micro(3),
            EcLevel::L,
            vec![Segment::kanji("日本").unwrap(), Segment::alphanumeric("X1").unwrap()],
            "日本X1",
        ),
        (
            SymbolVersion::Rmqr(13),
            EcLevel::M,
            vec![
                Segment::numeric("31415926").unwrap(),
                Segment::alphanumeric("RMQR").unwrap(),
                Segment::bytes("bytes"),
                Segment::kanji("長方形").unwrap(),
            ],
            "31415926RMQRbytes長方形",
        ),
        (
            SymbolVersion::Rmqr(31),
            EcLevel::H,
            vec![Segment::eci(qrcode_generator::rmqr::EciAssignment::UTF_8), Segment::bytes("ECI ✓ in rMQR")],
            "ECI ✓ in rMQR",
        ),
    ];
    for (version, ec, segments, expect) in cases {
        let layout = layout(version, ec);
        let mask = if version.family() == Family::Rmqr { 0 } else { 1 };
        let what = format!("{version}-{ec:?}");
        let grid = encode(version, ec, Some(mask), &segments).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        let report = assert_clean(&grid, &layout, ec, mask, &what);
        assert_eq!(report.text.plain(), expect, "{what}");
        assert!(report.text.notes.is_empty(), "{what}: {:?}", report.text.notes);
    }
}

#[test]
fn light_on_dark_symbols_are_flagged_and_decode_once_inverted() {
    let mut rng = Rng(0x1A7E);
    for (version, ec) in [
        (SymbolVersion::Qr(1), EcLevel::H),
        (SymbolVersion::Qr(7), EcLevel::M),
        (SymbolVersion::Qr(23), EcLevel::L),
        (SymbolVersion::Micro(4), EcLevel::M),
        (SymbolVersion::Rmqr(12), EcLevel::H),
        (SymbolVersion::Rmqr(31), EcLevel::M),
    ] {
        let layout = layout(version, ec);
        let mask = rng.below(version.mask_count() as usize) as u8;
        let text = rng.alphanumeric(byte_capacity(&layout, 2).max(1)).to_lowercase();
        let what = format!("{version}-{ec:?} mask {mask}");
        let clean = encode(version, ec, Some(mask), &[Segment::bytes(&text)]).unwrap();
        assert!(!analyze(&clean, &layout.func).inverted, "{what}");

        // A light-on-dark symbol taken as is: every fixed module is wrong,
        // and the analysis says why nothing will decode.
        let inverted = clean.inverted();
        let analysis = analyze(&inverted, &layout.func);
        assert!(analysis.inverted, "{what}");
        assert_eq!(analysis.fixed, (0, analysis.fixed.1), "{what}");
        assert!(!decode_plain(&inverted, &layout, mask).is_trustworthy(), "{what}");

        // The verdict survives damage to the fixed patterns, and swapping
        // dark and light back gives the symbol exactly.
        let mut damaged = inverted.clone();
        for (x, y) in [(0, 0), (3, 3), (6, 8), (layout.func.width - 1, 0)] {
            damaged.set(x, y, Module::Unknown);
        }
        assert!(analyze(&damaged, &layout.func).inverted, "{what}");
        assert_eq!(inverted.inverted(), clean);
        assert_eq!(assert_clean(&inverted.inverted(), &layout, ec, mask, &what).text.plain(), text);
    }
}

#[test]
fn charset_override_does_not_change_the_decode() {
    let layout = layout(SymbolVersion::Qr(2), EcLevel::L);
    let grid = encode(SymbolVersion::Qr(2), EcLevel::L, Some(0), &[Segment::bytes("naïve")]).unwrap();
    let auto = decode_plain(&grid, &layout, 0);
    assert_eq!(auto.text.plain(), "naïve");
    let latin1 = qr_forensics::decode::text::decode_text(&auto.parse.segments, Charset::Latin1);
    assert_eq!(latin1.plain(), "naÃ¯ve");
}

fn rxing_text(grid: &ModuleGrid) -> Option<String> {
    use rxing::common::BitMatrix;
    let mut bits = BitMatrix::new(grid.width() as u32, grid.height() as u32).ok()?;
    for y in 0..grid.height() {
        for x in 0..grid.width() {
            if grid.get(x, y).as_dark() == Some(true) {
                bits.set(x as u32, y as u32);
            }
        }
    }
    rxing::qrcode::cpp_port::decoder::Decode(&bits).ok().map(|r| r.text())
}

#[test]
fn agrees_with_rxing_on_clean_and_lightly_damaged_symbols() {
    let mut rng = Rng(0xD1FF);
    let mut compared = 0;
    let mut versions: Vec<SymbolVersion> = [1, 2, 6, 7, 10, 14, 21, 27, 32, 40].map(SymbolVersion::Qr).to_vec();
    versions.extend([2, 4].map(SymbolVersion::Micro));
    versions.extend([0, 4, 9, 12, 16, 21, 26, 31].map(SymbolVersion::Rmqr));
    for version in versions {
        for &ec in version.ec_levels() {
            let layout = layout(version, ec);
            let mask = rng.below(version.mask_count() as usize) as u8;
            let len = (byte_capacity(&layout, 2) / 2).max(1);
            let text = if version == SymbolVersion::Micro(2) {
                rng.alphanumeric(3)
            } else {
                rng.alphanumeric(len).to_lowercase()
            };
            let segment = if version == SymbolVersion::Micro(2) {
                Segment::alphanumeric(&text).unwrap()
            } else {
                Segment::bytes(&text)
            };
            let what = format!("{version}-{ec:?}");
            let mut grid = encode(version, ec, Some(mask), &[segment]).unwrap_or_else(|e| panic!("{what}: {e:?}"));

            assert_eq!(decode_plain(&grid, &layout, mask).text.plain(), text, "{what}");
            assert_eq!(rxing_text(&grid).as_deref(), Some(text.as_str()), "{what}: rxing on clean symbol");

            // One wrong module in a codeword of each block: within reach of
            // both decoders whenever the block can correct anything at all.
            if layout.blocks[0].ec_len >= 4 {
                for block in &layout.blocks {
                    let cw = block.codewords[rng.below(block.codewords.len())];
                    corrupt_codeword(&mut grid, &layout, cw, rng.below(8));
                }
                let ours = decode_plain(&grid, &layout, mask);
                assert!(ours.is_complete(), "{what}");
                assert_eq!(ours.text.plain(), text, "{what}: damaged");
                assert_eq!(rxing_text(&grid).as_deref(), Some(text.as_str()), "{what}: rxing on damaged symbol");
            }
            compared += 1;
        }
    }
    assert!(compared > 50);
}
