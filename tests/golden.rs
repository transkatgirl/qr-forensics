//! Symbols from encoders other than the round-trip oracle, including the
//! figures printed in ISO/IEC 18004 and ISO/IEC 23941.

mod common;

use std::collections::BTreeSet;
use std::path::Path;

use common::*;
use qr_forensics::decode::bch::analyze_format;
use qr_forensics::decode::pipeline::{BlockOutcome, SolveMethod};
use qr_forensics::symbol::grid::ModuleGrid;
use qr_forensics::symbol::{Family, SymbolVersion};

struct Fixture {
    name: String,
    family: Family,
    text: Option<String>,
    grid: ModuleGrid,
}

fn fixtures() -> Vec<Fixture> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "txt") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let content = std::fs::read_to_string(&path).unwrap();
        let header = |key: &str| {
            content.lines().find_map(|l| l.strip_prefix(&format!("; {key}: "))).map(str::to_owned)
        };
        let family = match header("family").as_deref() {
            Some("qr") => Family::Qr,
            Some("micro") => Family::Micro,
            Some("rmqr") => Family::Rmqr,
            Some("model1") => Family::Model1,
            other => panic!("{name}: family {other:?}"),
        };
        let grid = ModuleGrid::from_text(&content).unwrap_or_else(|e| panic!("{name}: {e}"));
        out.push(Fixture { name, family, text: header("text"), grid });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

#[test]
fn every_golden_symbol_decodes_cleanly() {
    let fixtures = fixtures();
    assert!(fixtures.len() >= 50, "only {} fixtures found", fixtures.len());
    let mut seen = BTreeSet::new();
    let mut with_text = 0;

    for fixture in &fixtures {
        let name = &fixture.name;
        let (w, h) = (fixture.grid.width(), fixture.grid.height());
        let version = SymbolVersion::from_size(fixture.family, w, h)
            .unwrap_or_else(|| panic!("{name}: no {:?} version is {w}x{h}", fixture.family));

        // The EC level and mask come from the symbol itself.
        let func = layout(version, version.ec_levels()[0]).func;
        let (matching, known) = func.fixed_agreement(&fixture.grid);
        assert_eq!(matching, known, "{name}: fixed patterns");
        let format = analyze_format(&fixture.grid, &func);
        let best = format.best().unwrap_or_else(|| panic!("{name}: format does not fit a {version}"));
        assert_eq!(best.mismatches, 0, "{name}: format");
        assert_eq!(best.compared as usize, func.format.iter().map(|c| c.bits.len()).sum::<usize>());

        let layout = layout(version, best.fields.ec);
        let report = decode_plain(&fixture.grid, &layout, best.fields.mask);
        for (b, block) in report.blocks.iter().enumerate() {
            assert!(
                matches!(
                    block.outcome,
                    BlockOutcome::Solved { method: SolveMethod::ReedSolomon, errors: 0, erasures: 0, .. }
                ),
                "{name} ({version}-{:?} mask {}): block {b} is {:?}",
                best.fields.ec,
                best.fields.mask,
                block.outcome
            );
        }
        assert!(report.is_trustworthy(), "{name}: {:?}", report.score);
        assert!(report.score.tail_plausible, "{name}: tail {:?}", report.tail);
        // Remainder bits are zero once unmasked.
        for pos in &func.data_path[func.codeword_bits()..] {
            let (x, y) = (pos.x as usize, pos.y as usize);
            let masked = qr_forensics::symbol::mask::mask_bit(version, best.fields.mask, x, y);
            assert_eq!(fixture.grid.get(x, y).as_dark(), Some(masked), "{name}: remainder bit at ({x},{y})");
        }
        if let Some(text) = &fixture.text {
            assert_eq!(&report.text.plain(), text, "{name} ({version}-{:?})", best.fields.ec);
            with_text += 1;
        }
        seen.insert((version, best.fields.ec));
    }

    assert!(with_text >= 40, "{with_text} fixtures carried text");
    // The cases the tables are most likely to get wrong.
    use qr_forensics::symbol::EcLevel::*;
    for must in [
        (SymbolVersion::Micro(1), L),
        (SymbolVersion::Micro(3), L),
        (SymbolVersion::Micro(3), M),
        (SymbolVersion::Micro(4), Q),
        (SymbolVersion::Rmqr(16), M), // R13x27
        (SymbolVersion::Rmqr(27), M), // R17x43
        (SymbolVersion::Rmqr(12), H), // R11x59, two unequal blocks
        (SymbolVersion::Rmqr(19), M), // R13x77, two unequal blocks
        (SymbolVersion::Rmqr(26), H), // R15x139, five blocks
        (SymbolVersion::Rmqr(31), H), // R17x139, six blocks
        (SymbolVersion::Rmqr(31), M),
        (SymbolVersion::Qr(5), Q),
        (SymbolVersion::Qr(7), H),
        (SymbolVersion::Qr(40), H),
        (SymbolVersion::Model1(2), M), // zxing-cpp's sample
    ] {
        assert!(seen.contains(&must), "no golden symbol covers {}-{:?}", must.0, must.1);
    }
}
