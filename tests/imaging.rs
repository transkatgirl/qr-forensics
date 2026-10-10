//! Photo to grid: render a known symbol under a known perspective, then get
//! the modules back through corner placement and sampling.

mod common;

use common::*;
use qr_forensics::decode::codewords::Pins;
use qr_forensics::decode::pipeline::{BlockOutcome, DecodeOptions, SolveMethod, analyze, decode};
use qr_forensics::imaging::Luma;
use qr_forensics::imaging::adjust::Filter;
use qr_forensics::imaging::detect::{AnchorKind, Retry, detect_outline, find_symbols, find_symbols_fine, refine_anchors};
use qr_forensics::imaging::enhance::{Enhance, Sharpen};
use qr_forensics::imaging::read::{drop_contradicted_anchors, read_detection, read_outline};
use qr_forensics::imaging::refine::Template;
use qr_forensics::symbol::function_map::{FunctionMap, Role};
use qr_forensics::imaging::homography::Homography;
use qr_forensics::imaging::projection::{ControlPoint, Projection};
use qr_forensics::imaging::refine::{refine_corners, refine_corners_holding};
use qr_forensics::imaging::sample::{SampleParams, classify, sample_modules, sample_modules_as, suggest_versions};
use qr_forensics::symbol::grid::{CERTAIN, Module, ModuleGrid};
use qr_forensics::symbol::{EcLevel, Family, SymbolVersion};
use qrcode_generator::Segment;

const QUAD: [[f64; 2]; 4] = [[150.5, 70.25], [655.0, 120.5], [610.75, 560.0], [95.25, 500.5]];

/// Photograph `grid` so its outline lands on `quad`, with 3×3 supersampling
/// and an optional lighting function of image position.
fn render(grid: &ModuleGrid, quad: [[f64; 2]; 4], size: (usize, usize), light: impl Fn(f64, f64, f32) -> f32) -> Luma {
    let (gw, gh) = (grid.width() as f64, grid.height() as f64);
    let to_grid = Homography::rect_to_quad(gw, gh, quad).unwrap().inverse().unwrap();
    let mut data = Vec::with_capacity(size.0 * size.1);
    for py in 0..size.1 {
        for px in 0..size.0 {
            let mut sum = 0.0;
            for sy in 0..3 {
                for sx in 0..3 {
                    let (x, y) = (px as f64 + (sx as f64 + 0.5) / 3.0, py as f64 + (sy as f64 + 0.5) / 3.0);
                    let [mx, my] = to_grid.apply(x, y);
                    let inside = mx >= 0.0 && my >= 0.0 && mx < gw && my < gh;
                    // Unknown modules print as mid grey, which no threshold
                    // can call.
                    let value = match inside.then(|| grid.get(mx as usize, my as usize)) {
                        Some(Module::Dark) => 0.08,
                        Some(Module::Unknown) => 0.5,
                        _ => 0.92,
                    };
                    sum += light(x, y, value);
                }
            }
            data.push(sum / 9.0);
        }
    }
    Luma::new(size.0, size.1, data)
}

fn symbol(version: SymbolVersion, ec: EcLevel) -> ModuleGrid {
    let layout = layout(version, ec);
    let text = Rng(version.width() as u64 + 1).alphanumeric(byte_capacity(&layout, 2)).to_lowercase();
    encode(version, ec, Some(0), &[Segment::bytes(&text)]).unwrap()
}

/// Sample the photo as a grid, with the reliability of every module.
fn read_back_sampled(image: &Luma, quad: [[f64; 2]; 4], grid: &ModuleGrid, params: &SampleParams) -> ModuleGrid {
    let h = Homography::rect_to_quad(grid.width() as f64, grid.height() as f64, quad).unwrap();
    let sampled = sample_modules(image, &h, grid.width(), grid.height());
    classify(&sampled, params).to_grid(grid.width(), grid.height())
}

/// The modules alone, comparable with an encoder's output.
fn read_back(image: &Luma, quad: [[f64; 2]; 4], grid: &ModuleGrid, params: &SampleParams) -> ModuleGrid {
    let sampled = read_back_sampled(image, quad, grid, params);
    let mut out = ModuleGrid::new(grid.width(), grid.height());
    for y in 0..grid.height() {
        for x in 0..grid.width() {
            out.set(x, y, sampled.get(x, y));
        }
    }
    out
}

#[test]
fn recovers_every_module_under_perspective() {
    for (version, ec) in [
        (SymbolVersion::Qr(2), EcLevel::M),
        (SymbolVersion::Qr(10), EcLevel::Q),
        (SymbolVersion::Micro(3), EcLevel::L),
        (SymbolVersion::Rmqr(13), EcLevel::M),
    ] {
        let grid = symbol(version, ec);
        let image = render(&grid, QUAD, (760, 640), |_, _, v| v);
        assert_eq!(read_back(&image, QUAD, &grid, &SampleParams::default()), grid, "{version}");
    }
}

#[test]
fn corner_order_handles_rotation_and_mirroring() {
    let grid = symbol(SymbolVersion::Qr(4), EcLevel::M);
    let image = render(&grid, QUAD, (760, 640), |_, _, v| v);
    let [a, b, c, d] = QUAD;
    // Start from a different corner: the symbol comes out rotated.
    let rotated = read_back(&image, [d, a, b, c], &grid, &SampleParams::default());
    assert_eq!(rotated, grid.rotated_cw());
    // Reverse the winding: the symbol comes out mirrored.
    let mirrored = read_back(&image, [b, a, d, c], &grid, &SampleParams::default());
    assert_eq!(mirrored, grid.flipped_horizontal());
}

#[test]
fn local_threshold_copes_with_glare_and_shadow() {
    let grid = symbol(SymbolVersion::Qr(8), EcLevel::M);
    // Shadow darkens the left, glare lifts the right: dark modules on the
    // right end up brighter than light modules on the left.
    let image = render(&grid, QUAD, (760, 640), |x, _, v| {
        let t = (x / 760.0) as f32;
        v * (0.3 + 0.1 * t) + 0.6 * t
    });
    // One threshold for the whole symbol still gets modules wrong, though
    // fewer than the gradient alone would cost: a reading this poorly
    // separated is sharpened against its neighbours before it is
    // classified, which takes most of the gradient out with the blur.
    let global = read_back(&image, QUAD, &grid, &SampleParams { band: 0.0, local: false, ..Default::default() });
    let wrong = global.cells().iter().zip(grid.cells()).filter(|(a, b)| a != b).count();
    assert!(wrong > 0, "global threshold got nothing wrong");

    let params = SampleParams { local: true, band: 0.0, ..Default::default() };
    let local = read_back(&image, QUAD, &grid, &params);
    let wrong = local.cells().iter().zip(grid.cells()).filter(|(a, b)| a != b).count();
    assert_eq!(wrong, 0);
}

#[test]
fn uncertain_modules_become_unknown_not_wrong() {
    let grid = symbol(SymbolVersion::Qr(3), EcLevel::M);
    // Wash out a patch to mid grey.
    let image = render(&grid, QUAD, (760, 640), |x, y, v| {
        if (300.0..380.0).contains(&x) && (250.0..330.0).contains(&y) { 0.5 + (v - 0.5) * 0.05 } else { v }
    });
    let read = read_back(&image, QUAD, &grid, &SampleParams::default());
    let unknown = read.count(Module::Unknown);
    assert!(unknown > 4, "{unknown}");
    for (got, want) in read.cells().iter().zip(grid.cells()) {
        assert!(*got == Module::Unknown || got == want);
    }
    // Erasures only, well within reach: the symbol still decodes.
    let layout = layout(SymbolVersion::Qr(3), EcLevel::M);
    assert!(decode_plain(&read, &layout, 0).is_complete());
}

#[test]
fn refinement_recovers_misplaced_corners() {
    for (version, ec) in [(SymbolVersion::Qr(4), EcLevel::M), (SymbolVersion::Qr(15), EcLevel::Q), (SymbolVersion::Rmqr(19), EcLevel::M)] {
        let grid = symbol(version, ec);
        let quad = if version.family() == Family::Rmqr {
            [[40.0, 200.0], [720.0, 230.0], [715.0, 330.0], [45.0, 290.0]]
        } else {
            QUAD
        };
        let image = render(&grid, quad, (760, 640), |x, _, v| v * (0.7 + 0.3 * (x / 760.0) as f32));
        let func = layout(version, ec).func;
        // Each corner is off by about a third of a module, in its own direction.
        let module_px = (quad[1][0] - quad[0][0]) / grid.width() as f64;
        let offsets = [[0.3, -0.2], [-0.25, 0.3], [0.2, 0.3], [-0.3, -0.2]];
        let shifted: [[f64; 2]; 4] =
            std::array::from_fn(|i| [quad[i][0] + offsets[i][0] * module_px, quad[i][1] + offsets[i][1] * module_px]);
        let params = SampleParams::default();

        let refined = refine_corners(&image, &func, shifted, &[], &params).unwrap();
        assert!(!refined.inverted, "{version}");
        assert_eq!(refined.after.1, refined.before.1);
        assert!(refined.after.0 >= refined.before.0, "{version}: {refined:?}");
        assert_eq!(refined.after.0, refined.after.1, "{version}: every fixed module reads right");
        for i in 0..4 {
            let error = |c: [[f64; 2]; 4]| ((c[i][0] - quad[i][0]).powi(2) + (c[i][1] - quad[i][1]).powi(2)).sqrt();
            let (before, after) = (error(shifted), error(refined.corners));
            assert!(after < before && after < 1.0, "{version} corner {i}: {before:.2} px before, {after:.2} px after");
        }
        assert_eq!(read_back(&image, refined.corners, &grid, &params), grid, "{version}");
        assert!(refined.evaluations < 3000, "{}", refined.evaluations);
    }
    let degenerate = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
    let grid = symbol(SymbolVersion::Qr(2), EcLevel::M);
    let image = render(&grid, QUAD, (200, 200), |_, _, v| v);
    let func = layout(SymbolVersion::Qr(2), EcLevel::M).func;
    assert_eq!(refine_corners(&image, &func, degenerate, &[], &SampleParams::default()), None);
}

#[test]
fn doubting_faint_modules_solves_a_smudged_block() {
    // A patch where the contrast collapses and inverts: dark modules read
    // faintly light and light ones faintly dark. Sampled with no unsure
    // band they come out wrong, in more codewords than the block can
    // correct as errors but fewer than it can take as erasures.
    let (version, ec) = (SymbolVersion::Qr(3), EcLevel::M);
    let grid = symbol(version, ec);
    let layout = layout(version, ec);
    let image = render(&grid, QUAD, (760, 640), |x, y, v| {
        if (290.0..470.0).contains(&x) && (230.0..410.0).contains(&y) { 0.5 - (v - 0.5) * 0.1 } else { v }
    });
    let read = read_back_sampled(&image, QUAD, &grid, &SampleParams { band: 0.0, local: false, ..Default::default() });
    assert!(read.has_reliability());
    let wrong: Vec<(usize, usize)> = (0..grid.height())
        .flat_map(|y| (0..grid.width()).map(move |x| (x, y)))
        .filter(|&(x, y)| read.get(x, y) != grid.get(x, y))
        .collect();
    assert!(wrong.len() > 30, "{} wrong modules", wrong.len());
    let affected: std::collections::BTreeSet<usize> =
        wrong.iter().filter_map(|&(x, y)| layout.module_codeword(x, y).map(|(cw, _)| cw)).collect();
    let ec_len = layout.blocks[0].ec_len;
    assert!(
        affected.len() > ec_len / 2 && affected.len() < ec_len,
        "{} codewords affected, EC {ec_len}",
        affected.len()
    );
    // Reliability is the margin as a share of the symbol's contrast: a
    // tenth of it, here.
    assert!(wrong.iter().all(|&(x, y)| read.reliability(x, y) < 40), "wrong modules read faintly");
    let confident = (0..grid.height())
        .flat_map(|y| (0..grid.width()).map(move |x| (x, y)))
        .filter(|&(x, y)| read.reliability(x, y) > 60)
        .count();
    assert!(confident > grid.cells().len() * 3 / 4);

    let truth = decode_plain(&grid, &layout, 0).text.plain();
    let strict = DecodeOptions { reliability: false, ..Default::default() };
    let before = decode(&read, &layout, 0, &Pins::new(), &strict);
    assert!(matches!(before.blocks[0].outcome, BlockOutcome::Failed(_)), "{:?}", before.blocks[0].outcome);
    assert_ne!(before.text.plain(), truth);

    let after = decode(&read, &layout, 0, &Pins::new(), &DecodeOptions::default());
    // Doubting the faint codewords and doubting the faint bits both reach
    // the answer; whichever is confirmed by more bits is kept.
    match after.blocks[0].outcome {
        BlockOutcome::Solved { method: SolveMethod::Reliability { doubted }, slack, .. } => {
            assert!(doubted > layout.blocks[0].ec_len / 2, "doubted {doubted}");
            assert!(slack >= 1);
        }
        BlockOutcome::Solved { method: SolveMethod::BitReliability { doubted_bits }, .. } => {
            assert!(doubted_bits >= wrong.len(), "doubted {doubted_bits} bits for {} wrong modules", wrong.len());
        }
        other => panic!("{other:?}"),
    }
    assert!(after.is_trustworthy());
    assert_eq!(after.text.plain(), truth);

    // Painting the smudge by hand, even wrongly, makes it certain and
    // therefore beyond doubt.
    let mut painted = read.clone();
    for &(x, y) in &wrong {
        painted.set(x, y, read.get(x, y));
        assert_eq!(painted.reliability(x, y), CERTAIN);
    }
    let stubborn = decode(&painted, &layout, 0, &Pins::new(), &DecodeOptions::default());
    assert!(matches!(stubborn.blocks[0].outcome, BlockOutcome::Failed(_)));
}

#[test]
fn suggests_the_right_version() {
    for version in [SymbolVersion::Qr(1), SymbolVersion::Qr(6), SymbolVersion::Qr(17)] {
        let grid = symbol(version, EcLevel::M);
        let image = render(&grid, QUAD, (760, 640), |_, _, v| v);
        let guesses = suggest_versions(&image, QUAD, Family::Qr, &SampleParams::default());
        assert_eq!(guesses.len(), 40);
        assert_eq!(guesses[0].version, version);
        assert_eq!(guesses[0].matching, guesses[0].fixed);
        assert!(!guesses[0].inverted);
        assert!(guesses[0].score() - guesses[1].score() > 0.1, "{version}: {:?}", &guesses[..2]);
    }
    let grid = symbol(SymbolVersion::Micro(3), EcLevel::M);
    let image = render(&grid, QUAD, (760, 640), |_, _, v| v);
    let guesses = suggest_versions(&image, QUAD, Family::Micro, &SampleParams::default());
    assert_eq!(guesses[0].version, SymbolVersion::Micro(3));

    let wide = [[40.0, 200.0], [720.0, 230.0], [715.0, 330.0], [45.0, 290.0]];
    let grid = symbol(SymbolVersion::Rmqr(19), EcLevel::M);
    let image = render(&grid, wide, (760, 640), |_, _, v| v);
    let guesses = suggest_versions(&image, wide, Family::Rmqr, &SampleParams::default());
    assert_eq!(guesses[0].version, SymbolVersion::Rmqr(19));
}

/// The outline a family's test symbol is photographed on.
fn quad_for(version: SymbolVersion) -> [[f64; 2]; 4] {
    if version.family() == Family::Rmqr { [[40.0, 200.0], [720.0, 230.0], [715.0, 330.0], [45.0, 290.0]] } else { QUAD }
}

/// The photo with dark and light swapped, as the Invert control does.
fn negative(image: &Luma) -> Luma {
    Luma::new(image.width, image.height, image.data.iter().map(|v| 1.0 - v).collect())
}

#[test]
fn light_on_dark_photos_are_recognised() {
    for (version, ec) in [(SymbolVersion::Qr(4), EcLevel::M), (SymbolVersion::Micro(3), EcLevel::L), (SymbolVersion::Rmqr(13), EcLevel::M)] {
        let grid = symbol(version, ec);
        let layout = layout(version, ec);
        let func = &layout.func;
        let quad = quad_for(version);
        let normal = render(&grid, quad, (760, 640), |_, _, v| v);
        let inverted = render(&grid, quad, (760, 640), |_, _, v| 1.0 - v);
        let params = SampleParams::default();

        // Sampled as is, every module comes out the wrong way round, and
        // the fixed patterns say so; a dark-on-light photo does not.
        let read = read_back_sampled(&inverted, quad, &grid, &params);
        assert_eq!(read_back(&inverted, quad, &grid, &params), grid.inverted(), "{version}");
        assert!(func.is_inverted(&read), "{version}");
        let analysis = analyze(&read, func);
        assert!(analysis.inverted, "{version}");
        assert_eq!(analysis.fixed.0, 0, "{version}: no fixed module reads right");
        assert!(!analyze(&read_back_sampled(&normal, quad, &grid, &params), func).inverted, "{version}");

        // Taken as is, nothing decodes; after either remedy the symbol is
        // whole. Inverting the grid keeps the sampled reliabilities.
        let truth = decode_plain(&grid, &layout, 0).text.plain();
        assert!(!decode_plain(&read, &layout, 0).is_trustworthy(), "{version}");
        let swapped = read.inverted();
        assert!(swapped.has_reliability());
        let report = decode_plain(&swapped, &layout, 0);
        assert!(report.is_trustworthy(), "{version}: {:?}", report.score);
        assert_eq!(report.text.plain(), truth, "{version}");
        assert_eq!(read_back(&negative(&inverted), quad, &grid, &params), grid, "{version}");
    }
}

#[test]
fn version_suggestion_sees_through_inversion() {
    for version in [SymbolVersion::Qr(1), SymbolVersion::Qr(6), SymbolVersion::Qr(17)] {
        let grid = symbol(version, EcLevel::M);
        let image = render(&grid, QUAD, (760, 640), |_, _, v| 1.0 - v);
        let guesses = suggest_versions(&image, QUAD, Family::Qr, &SampleParams::default());
        assert_eq!(guesses[0].version, version, "{version}: {:?}", &guesses[..2]);
        assert!(guesses[0].inverted, "{version}");
        assert_eq!(guesses[0].matching, guesses[0].fixed, "{version}");
        assert!(guesses[0].score() - guesses[1].score() > 0.1, "{version}: {:?}", &guesses[..2]);
        // The same photo inverted back ranks the same version, read normally.
        let guesses = suggest_versions(&negative(&image), QUAD, Family::Qr, &SampleParams::default());
        assert_eq!((guesses[0].version, guesses[0].inverted), (version, false));
    }
    for (version, ec) in [(SymbolVersion::Micro(3), EcLevel::M), (SymbolVersion::Rmqr(19), EcLevel::M)] {
        let grid = symbol(version, ec);
        let quad = quad_for(version);
        let image = render(&grid, quad, (760, 640), |_, _, v| 1.0 - v);
        let guesses = suggest_versions(&image, quad, version.family(), &SampleParams::default());
        assert_eq!((guesses[0].version, guesses[0].inverted), (version, true), "{version}");
    }
}

#[test]
fn refinement_finds_the_outline_of_a_light_on_dark_symbol() {
    for (version, ec) in [(SymbolVersion::Qr(4), EcLevel::M), (SymbolVersion::Qr(15), EcLevel::Q), (SymbolVersion::Rmqr(19), EcLevel::M)] {
        let grid = symbol(version, ec);
        let quad = quad_for(version);
        let image = render(&grid, quad, (760, 640), |x, _, v| 1.0 - v * (0.7 + 0.3 * (x / 760.0) as f32));
        let func = layout(version, ec).func;
        let module_px = (quad[1][0] - quad[0][0]) / grid.width() as f64;
        let offsets = [[0.3, -0.2], [-0.25, 0.3], [0.2, 0.3], [-0.3, -0.2]];
        let shifted: [[f64; 2]; 4] =
            std::array::from_fn(|i| [quad[i][0] + offsets[i][0] * module_px, quad[i][1] + offsets[i][1] * module_px]);
        let params = SampleParams::default();

        let refined = refine_corners(&image, &func, shifted, &[], &params).unwrap();
        assert!(refined.inverted, "{version}");
        assert_eq!(refined.after.1, refined.before.1);
        assert!(refined.after.0 >= refined.before.0, "{version}: {refined:?}");
        assert_eq!(refined.after.0, refined.after.1, "{version}: every fixed module reads inverted");
        for i in 0..4 {
            let error = |c: [[f64; 2]; 4]| ((c[i][0] - quad[i][0]).powi(2) + (c[i][1] - quad[i][1]).powi(2)).sqrt();
            let (before, after) = (error(shifted), error(refined.corners));
            assert!(after < before && after < 1.0, "{version} corner {i}: {before:.2} px before, {after:.2} px after");
        }
        // The refined outline reads the symbol back exactly once the photo
        // is inverted.
        assert_eq!(read_back(&negative(&image), refined.corners, &grid, &params), grid, "{version}");
        // And the same corners come out of the un-inverted photo, read normally.
        let plain = refine_corners(&negative(&image), &func, shifted, &[], &params).unwrap();
        assert!(!plain.inverted, "{version}");
        for i in 0..4 {
            let gap = ((plain.corners[i][0] - refined.corners[i][0]).powi(2) + (plain.corners[i][1] - refined.corners[i][1]).powi(2)).sqrt();
            assert!(gap < 0.2, "{version} corner {i}: {gap:.3} px apart");
        }
        // Both polarities are searched, so this is about twice one search.
        assert!(refined.evaluations < 3000, "{version}: {}", refined.evaluations);
    }
}

/// An outline around `quad` as a hand would draw it: scaled about the
/// centre by `scale` and pushed off by `shift`, every corner a little off.
fn loosen(quad: [[f64; 2]; 4], scale: f64, shift: [f64; 2]) -> [[f64; 2]; 4] {
    let centre = quad.iter().fold([0.0, 0.0], |a, c| [a[0] + c[0] / 4.0, a[1] + c[1] / 4.0]);
    let wobble = [[3.0, -2.0], [-4.0, 3.0], [2.0, 4.0], [-3.0, -3.0]];
    std::array::from_fn(|i| {
        std::array::from_fn(|axis| centre[axis] + (quad[i][axis] - centre[axis]) * scale + shift[axis] + wobble[i][axis])
    })
}

/// Largest distance between corresponding corners, in photo pixels.
fn corner_error(a: [[f64; 2]; 4], b: [[f64; 2]; 4]) -> f64 {
    (0..4).map(|i| ((a[i][0] - b[i][0]).powi(2) + (a[i][1] - b[i][1]).powi(2)).sqrt()).fold(0.0, f64::max)
}

/// Outlines whose module aspect matches the symbol's, as a photo of a
/// printed symbol would have.
fn detection_quad(version: SymbolVersion) -> [[f64; 2]; 4] {
    match version {
        SymbolVersion::Rmqr(11) => [[60.0, 180.0], [700.0, 195.0], [696.0, 365.0], [64.0, 350.0]],
        SymbolVersion::Rmqr(19) => [[40.0, 220.0], [720.0, 235.0], [716.0, 350.0], [44.0, 335.0]],
        SymbolVersion::Rmqr(31) => [[30.0, 260.0], [730.0, 272.0], [728.0, 357.0], [32.0, 345.0]],
        _ => QUAD,
    }
}

#[test]
fn detection_places_the_corners_from_a_loose_outline() {
    for (version, ec) in [
        (SymbolVersion::Qr(1), EcLevel::M),
        (SymbolVersion::Qr(4), EcLevel::M),
        (SymbolVersion::Qr(10), EcLevel::Q),
        (SymbolVersion::Qr(25), EcLevel::L),
        (SymbolVersion::Micro(3), EcLevel::L),
        (SymbolVersion::Rmqr(11), EcLevel::M),
        (SymbolVersion::Rmqr(19), EcLevel::M),
        (SymbolVersion::Rmqr(31), EcLevel::M),
    ] {
        let grid = symbol(version, ec);
        let quad = detection_quad(version);
        let image = render(&grid, quad, (760, 640), |x, _, v| v * (0.75 + 0.25 * (x / 760.0) as f32));
        let module_px = (quad[1][0] - quad[0][0]) / grid.width() as f64;
        let params = SampleParams::default();
        let func = layout(version, ec).func;
        // A generous outline and a slightly tight one.
        for (scale, shift) in [(1.12, [9.0, -6.0]), (0.97, [-5.0, 7.0])] {
            let region = loosen(quad, scale, shift);
            let found = detect_outline(&image, version, region, &params)
                .unwrap_or_else(|| panic!("{version} ×{scale}: nothing found"));
            assert_eq!(found.version, version, "{version} ×{scale}");
            assert!(!found.inverted, "{version}");
            assert_eq!(found.turns, 0, "{version}");
            let finders = if version.family() == Family::Qr { 3 } else { 1 };
            assert_eq!(found.count(AnchorKind::Finder), finders, "{version} ×{scale}: {:?}", found.anchors);
            match version {
                SymbolVersion::Qr(v) if v >= 2 => assert!(found.count(AnchorKind::Alignment) >= 1, "{version}"),
                SymbolVersion::Rmqr(_) => assert_eq!(found.count(AnchorKind::FinderSub), 1, "{version}"),
                _ => {}
            }
            let coarse = corner_error(found.corners, quad) / module_px;
            assert!(coarse < 0.75, "{version} ×{scale}: corners {coarse:.2} modules off");
            assert!(found.score() > 0.9, "{version} ×{scale}: {:?}", found.agreement);

            let refined = refine_corners(&image, &func, found.corners, &[], &params).unwrap();
            let fine = corner_error(refined.corners, quad);
            assert!(fine < (0.05 * module_px).max(1.0), "{version} ×{scale}: {fine:.2} px off after refinement");
            assert_eq!(read_back(&image, refined.corners, &grid, &params), grid, "{version} ×{scale}");
        }
    }
}

#[test]
fn detection_tells_the_version_from_the_finder_spacing() {
    for (photographed, assumed, ec) in [
        (SymbolVersion::Qr(10), SymbolVersion::Qr(2), EcLevel::Q),
        (SymbolVersion::Qr(3), SymbolVersion::Qr(20), EcLevel::M),
        (SymbolVersion::Micro(3), SymbolVersion::Micro(2), EcLevel::L),
        (SymbolVersion::Rmqr(19), SymbolVersion::Rmqr(0), EcLevel::M),
        (SymbolVersion::Rmqr(11), SymbolVersion::Rmqr(31), EcLevel::M),
    ] {
        let grid = symbol(photographed, ec);
        let quad = detection_quad(photographed);
        let image = render(&grid, quad, (760, 640), |_, _, v| v);
        let region = loosen(quad, 1.08, [4.0, 3.0]);
        let found = detect_outline(&image, assumed, region, &SampleParams::default())
            .unwrap_or_else(|| panic!("{photographed} as {assumed}: nothing found"));
        assert_eq!(found.version, photographed, "assumed {assumed}");
        let module_px = (quad[1][0] - quad[0][0]) / grid.width() as f64;
        let error = corner_error(found.corners, quad) / module_px;
        assert!(error < 0.75, "{photographed} as {assumed}: corners {error:.2} modules off");
        // The version ranking on those corners agrees.
        let guesses = suggest_versions(&image, found.corners, photographed.family(), &SampleParams::default());
        assert_eq!(guesses[0].version, photographed, "{photographed}: {:?}", &guesses[..2]);
    }
}

#[test]
fn detection_reads_the_version_from_the_symbols_own_bits() {
    for (photographed, assumed, ec) in [
        (SymbolVersion::Qr(10), SymbolVersion::Qr(2), EcLevel::Q),
        (SymbolVersion::Micro(3), SymbolVersion::Micro(2), EcLevel::L),
        (SymbolVersion::Micro(4), SymbolVersion::Micro(1), EcLevel::L),
        (SymbolVersion::Rmqr(19), SymbolVersion::Rmqr(0), EcLevel::M),
        (SymbolVersion::Rmqr(11), SymbolVersion::Rmqr(31), EcLevel::M),
    ] {
        let grid = symbol(photographed, ec);
        let quad = detection_quad(photographed);
        let image = render(&grid, quad, (760, 640), |_, _, v| v);
        let region = loosen(quad, 1.08, [4.0, 3.0]);
        let found = detect_outline(&image, assumed, region, &SampleParams::default()).unwrap();
        assert_eq!(found.version, photographed, "assumed {assumed}");
        assert_eq!(found.info.named_version, Some(photographed), "{photographed}: {:?}", found.info);
        assert_eq!(found.info.format.valid, Some(true), "{photographed}: {:?}", found.info);
        assert!(!found.mirrored);
    }
    // Below version 7 a QR symbol names no version; the fit alone decides.
    let grid = symbol(SymbolVersion::Qr(3), EcLevel::M);
    let image = render(&grid, QUAD, (760, 640), |_, _, v| v);
    let found = detect_outline(&image, SymbolVersion::Qr(20), loosen(QUAD, 1.08, [4.0, 3.0]), &SampleParams::default()).unwrap();
    assert_eq!(found.version, SymbolVersion::Qr(3));
    assert_eq!(found.info.named_version, None);
    assert_eq!(found.info.format.valid, Some(true));
}

#[test]
fn detection_copes_with_unreadable_information_bits() {
    // The version and format information torn away, and then printed too
    // faint to call: neither reads as anything, and the fit still decides.
    let (version, ec) = (SymbolVersion::Qr(10), EcLevel::Q);
    let clean = symbol(version, ec);
    let func = layout(version, ec).func;
    let module_px = (QUAD[1][0] - QUAD[0][0]) / clean.width() as f64;
    for (what, module) in [("bare paper", Module::Light), ("too faint", Module::Unknown)] {
        let mut grid = clean.clone();
        for pos in func.format.iter().chain(&func.version_info).flat_map(|c| c.bits.iter()) {
            grid.set(pos.x as usize, pos.y as usize, module);
        }
        let image = render(&grid, QUAD, (760, 640), |_, _, v| v);
        let region = loosen(QUAD, 1.08, [4.0, 3.0]);
        let found = detect_outline(&image, SymbolVersion::Qr(2), region, &SampleParams::default()).unwrap();
        assert_eq!(found.version, version, "{what}");
        assert_eq!(found.info.named_version, None, "{what}: {:?}", found.info);
        assert_ne!(found.info.format.valid, Some(true), "{what}: {:?}", found.info);
        if module == Module::Unknown {
            assert_eq!(found.info.format, Default::default(), "{what}: nothing known, nothing claimed");
        }
        assert!(!found.mirrored, "{what}");
        let error = corner_error(found.corners, QUAD) / module_px;
        assert!(error < 0.75, "{what}: corners {error:.2} modules off");
    }
}

#[test]
fn detection_sees_a_mirrored_photo() {
    for (version, ec) in [
        (SymbolVersion::Qr(4), EcLevel::M),
        (SymbolVersion::Qr(10), EcLevel::Q),
        (SymbolVersion::Micro(3), EcLevel::L),
        (SymbolVersion::Rmqr(11), EcLevel::M),
        (SymbolVersion::Rmqr(19), EcLevel::M),
    ] {
        let grid = symbol(version, ec);
        let quad = detection_quad(version);
        // The symbol as seen from the back of the paper.
        let image = render(&grid.flipped_horizontal(), quad, (760, 640), |x, _, v| v * (0.8 + 0.2 * (x / 760.0) as f32));
        let region = loosen(quad, 1.1, [-5.0, 4.0]);
        let params = SampleParams::default();
        let found = detect_outline(&image, version, region, &params).unwrap();
        assert!(found.mirrored, "{version}: {:?}", found.info);
        assert_eq!(found.version, version);
        // The corners run in mirror order, so the grid comes back the right
        // way round.
        let func = layout(version, ec).func;
        let refined = refine_corners(&image, &func, found.corners, &[], &params).unwrap();
        assert_eq!(read_back(&image, refined.corners, &grid, &params), grid, "{version}");
        // And a photo the right way round is never taken for a mirror image.
        let plain = render(&grid, quad, (760, 640), |_, _, v| v);
        let found = detect_outline(&plain, version, region, &params).unwrap();
        assert!(!found.mirrored, "{version}: {:?}", found.info);
    }
}

#[test]
fn detection_rejects_a_misleading_alignment_pattern() {
    let (version, ec) = (SymbolVersion::Qr(10), EcLevel::Q);
    let mut grid = symbol(version, ec);
    // The alignment pattern at (28, 28) is gone, and a module away sits a
    // pattern that looks just like one.
    for y in 26..=30 {
        for x in 26..=30 {
            grid.set(x, y, Module::Light);
        }
    }
    let (fx, fy) = (29i64, 29i64);
    for y in fy - 2..=fy + 2 {
        for x in fx - 2..=fx + 2 {
            let ring = (x - fx).abs().max((y - fy).abs());
            grid.set(x as usize, y as usize, Module::from_dark(ring != 1));
        }
    }
    let image = render(&grid, QUAD, (760, 640), |_, _, v| v);
    let truth = Homography::rect_to_quad(grid.width() as f64, grid.height() as f64, QUAD).unwrap();
    let module_px = (QUAD[1][0] - QUAD[0][0]) / grid.width() as f64;
    let region = loosen(QUAD, 1.1, [5.0, -4.0]);
    let found = detect_outline(&image, version, region, &SampleParams::default()).unwrap();
    assert_eq!(found.version, version);
    // Every anchor kept lies where the photo really has its pattern; the
    // impostor was found, and dropped as the one the map could not explain.
    for anchor in &found.anchors {
        let [tx, ty] = truth.apply(anchor.at[0], anchor.at[1]);
        let off = ((anchor.center[0] - tx).powi(2) + (anchor.center[1] - ty).powi(2)).sqrt() / module_px;
        assert!(off < 0.5, "{:?} at {:?} is {off:.2} modules off", anchor.kind, anchor.at);
    }
    assert!(found.anchors.iter().all(|a| a.at != [28.0, 28.0]), "{:?}", found.anchors);
    assert!(found.count(AnchorKind::Alignment) >= 5, "{:?}", found.anchors);
    let error = corner_error(found.corners, QUAD) / module_px;
    assert!(error < 0.75, "corners {error:.2} modules off");
}

#[test]
fn detection_turns_an_outline_drawn_a_quarter_turn_out() {
    for (version, ec) in [(SymbolVersion::Qr(5), EcLevel::M), (SymbolVersion::Micro(4), EcLevel::L)] {
        let grid = symbol(version, ec);
        let image = render(&grid, QUAD, (760, 640), |_, _, v| v);
        let module_px = (QUAD[1][0] - QUAD[0][0]) / grid.width() as f64;
        let [a, b, c, d] = loosen(QUAD, 1.1, [0.0, 0.0]);
        // The handle called TL on the symbol's bottom-left corner, and so on
        // round: the symbol sits a quarter turn clockwise in the outline.
        for (region, turns) in [([d, a, b, c], 1), ([c, d, a, b], 2), ([b, c, d, a], 3)] {
            let found = detect_outline(&image, version, region, &SampleParams::default()).unwrap();
            assert_eq!(found.turns, turns, "{version}");
            assert_eq!(found.version, version);
            let error = corner_error(found.corners, QUAD) / module_px;
            assert!(error < 0.75, "{version} turned {turns}: corners {error:.2} modules off");
        }
    }
}

#[test]
fn detection_sees_through_inversion() {
    for (version, ec) in [(SymbolVersion::Qr(6), EcLevel::M), (SymbolVersion::Rmqr(19), EcLevel::M)] {
        let grid = symbol(version, ec);
        let quad = detection_quad(version);
        let image = render(&grid, quad, (760, 640), |_, _, v| 1.0 - v);
        let region = loosen(quad, 1.1, [-6.0, 5.0]);
        let found = detect_outline(&image, version, region, &SampleParams::default()).unwrap();
        assert!(found.inverted, "{version}");
        assert_eq!(found.version, version);
        let module_px = (quad[1][0] - quad[0][0]) / grid.width() as f64;
        let error = corner_error(found.corners, quad) / module_px;
        assert!(error < 0.75, "{version}: corners {error:.2} modules off");
        let func = layout(version, ec).func;
        let refined = refine_corners(&image, &func, found.corners, &[], &SampleParams::default()).unwrap();
        assert!(refined.inverted);
        assert_eq!(read_back(&negative(&image), refined.corners, &grid, &SampleParams::default()), grid, "{version}");
    }
}

#[test]
fn detection_survives_a_destroyed_finder_pattern() {
    let (version, ec) = (SymbolVersion::Qr(6), EcLevel::M);
    let mut grid = symbol(version, ec);
    let n = grid.width();
    // The top-right finder and its surroundings torn away to bare paper.
    for y in 0..9 {
        for x in n - 9..n {
            grid.set(x, y, Module::Light);
        }
    }
    let image = render(&grid, QUAD, (760, 640), |_, _, v| v);
    let region = loosen(QUAD, 1.1, [5.0, -4.0]);
    let found = detect_outline(&image, version, region, &SampleParams::default()).unwrap();
    assert_eq!(found.version, version);
    assert_eq!(found.count(AnchorKind::Finder), 2, "{:?}", found.anchors);
    assert_eq!(found.count(AnchorKind::Alignment), 1);
    let module_px = (QUAD[1][0] - QUAD[0][0]) / n as f64;
    let error = corner_error(found.corners, QUAD) / module_px;
    assert!(error < 0.75, "corners {error:.2} modules off");
}

#[test]
fn detection_needs_a_finder_pattern_in_the_outline() {
    let grid = symbol(SymbolVersion::Qr(4), EcLevel::M);
    let image = render(&grid, QUAD, (760, 640), |_, _, v| v);
    // An outline on the bare paper beside the symbol.
    let region = [[660.0, 10.0], [750.0, 10.0], [750.0, 60.0], [660.0, 60.0]];
    assert_eq!(detect_outline(&image, SymbolVersion::Qr(4), region, &SampleParams::default()), None);
    // No symbol at all.
    let blank = Luma::new(200, 200, vec![0.9; 40_000]);
    let region = [[20.0, 20.0], [180.0, 20.0], [180.0, 180.0], [20.0, 180.0]];
    assert_eq!(detect_outline(&blank, SymbolVersion::Qr(4), region, &SampleParams::default()), None);
    // A degenerate outline.
    let line = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
    assert_eq!(detect_outline(&image, SymbolVersion::Qr(4), line, &SampleParams::default()), None);
}

/// Photograph `grid` with a smooth bulge of strength `k`, as a curled label
/// has: pixel p shows the flat symbol's content at c + (p - c)(1 + k|p - c|²/R²).
fn render_warped(grid: &ModuleGrid, quad: [[f64; 2]; 4], size: (usize, usize), k: f64) -> Luma {
    let (gw, gh) = (grid.width() as f64, grid.height() as f64);
    let to_grid = Homography::rect_to_quad(gw, gh, quad).unwrap().inverse().unwrap();
    let c = quad.iter().fold([0.0, 0.0], |a, p| [a[0] + p[0] / 4.0, a[1] + p[1] / 4.0]);
    let r2 = quad.iter().map(|p| (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2)).fold(0.0, f64::max);
    let mut data = Vec::with_capacity(size.0 * size.1);
    for py in 0..size.1 {
        for px in 0..size.0 {
            let mut sum = 0.0;
            for sy in 0..3 {
                for sx in 0..3 {
                    let (x, y) = (px as f64 + (sx as f64 + 0.5) / 3.0, py as f64 + (sy as f64 + 0.5) / 3.0);
                    let f = 1.0 + k * ((x - c[0]).powi(2) + (y - c[1]).powi(2)) / r2;
                    let [mx, my] = to_grid.apply(c[0] + (x - c[0]) * f, c[1] + (y - c[1]) * f);
                    let inside = mx >= 0.0 && my >= 0.0 && mx < gw && my < gh;
                    sum += match inside.then(|| grid.get(mx as usize, my as usize)) {
                        Some(Module::Dark) => 0.08f32,
                        Some(Module::Unknown) => 0.5,
                        _ => 0.92,
                    };
                }
            }
            data.push(sum / 9.0);
        }
    }
    Luma::new(size.0, size.1, data)
}

/// Photograph `grid` with no quiet zone: module-sized dark and light
/// blocks, as of neighbouring print, run right up to its edge.
fn render_busy(grid: &ModuleGrid, quad: [[f64; 2]; 4], size: (usize, usize)) -> Luma {
    let (gw, gh) = (grid.width() as f64, grid.height() as f64);
    let to_grid = Homography::rect_to_quad(gw, gh, quad).unwrap().inverse().unwrap();
    let mut data = Vec::with_capacity(size.0 * size.1);
    for py in 0..size.1 {
        for px in 0..size.0 {
            let mut sum = 0.0;
            for sy in 0..3 {
                for sx in 0..3 {
                    let (x, y) = (px as f64 + (sx as f64 + 0.5) / 3.0, py as f64 + (sy as f64 + 0.5) / 3.0);
                    let [mx, my] = to_grid.apply(x, y);
                    let inside = mx >= 0.0 && my >= 0.0 && mx < gw && my < gh;
                    let dark = if inside {
                        grid.get(mx as usize, my as usize) == Module::Dark
                    } else {
                        let (i, j) = (mx.floor() as i64, my.floor() as i64);
                        (i.wrapping_mul(73_856_093) ^ j.wrapping_mul(19_349_663)).rem_euclid(5) < 2
                    };
                    sum += if dark { 0.08f32 } else { 0.92 };
                }
            }
            data.push(sum / 9.0);
        }
    }
    Luma::new(size.0, size.1, data)
}

/// Photograph `grid` over-exposed: light has bled `shrink` of a module into
/// every dark region, so dark runs are that much shorter and light runs
/// that much longer.
fn render_bloomed(grid: &ModuleGrid, quad: [[f64; 2]; 4], size: (usize, usize), shrink: f64) -> Luma {
    let (gw, gh) = (grid.width() as f64, grid.height() as f64);
    let to_grid = Homography::rect_to_quad(gw, gh, quad).unwrap().inverse().unwrap();
    let dark_at = |mx: f64, my: f64| {
        mx >= 0.0 && my >= 0.0 && mx < gw && my < gh && grid.get(mx as usize, my as usize) == Module::Dark
    };
    let r = shrink / 2.0;
    let mut data = Vec::with_capacity(size.0 * size.1);
    for py in 0..size.1 {
        for px in 0..size.0 {
            let mut sum = 0.0;
            for sy in 0..3 {
                for sx in 0..3 {
                    let (x, y) = (px as f64 + (sx as f64 + 0.5) / 3.0, py as f64 + (sy as f64 + 0.5) / 3.0);
                    let [mx, my] = to_grid.apply(x, y);
                    // Dark only where everything within the bleed is dark.
                    let dark = [[0.0, 0.0], [r, 0.0], [-r, 0.0], [0.0, r], [0.0, -r], [r, r], [-r, -r], [r, -r], [-r, r]]
                        .iter()
                        .all(|[dx, dy]| dark_at(mx + dx, my + dy));
                    sum += if dark { 0.08f32 } else { 0.92 };
                }
            }
            data.push(sum / 9.0);
        }
    }
    Luma::new(size.0, size.1, data)
}

/// (wrong, unknown) modules of `read` against `truth`.
fn tally(read: &ModuleGrid, truth: &ModuleGrid) -> (usize, usize) {
    read.cells().iter().zip(truth.cells()).fold((0, 0), |(wrong, unknown), (a, b)| match a {
        Module::Unknown => (wrong, unknown + 1),
        _ if a != b => (wrong + 1, unknown),
        _ => (wrong, unknown),
    })
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

#[test]
fn anchors_follow_a_curled_symbol() {
    let params = SampleParams::default();
    for (version, ec, k) in [(SymbolVersion::Qr(15), EcLevel::M, 0.03), (SymbolVersion::Qr(25), EcLevel::L, 0.02)] {
        let grid = symbol(version, ec);
        let layout = layout(version, ec);
        let (w, h) = (grid.width(), grid.height());
        let image = render_warped(&grid, QUAD, (760, 640), k);
        let region = loosen(QUAD, 1.1, [5.0, -4.0]);
        let found = detect_outline(&image, version, region, &params).unwrap_or_else(|| panic!("{version}: nothing found"));
        assert_eq!(found.version, version);
        assert!(found.count(AnchorKind::Alignment) >= 10, "{version}: {:?}", found.anchors.len());
        let func = &layout.func;
        let controls: Vec<ControlPoint> = found.anchors.iter().map(|a| a.control_point()).collect();
        let refined = refine_corners(&image, func, found.corners, &controls, &params).unwrap();
        // One flat homography through the corners misreads the middle.
        let flat = Homography::rect_to_quad(w as f64, h as f64, refined.corners).unwrap();
        let flat_read = classify(&sample_modules(&image, &flat, w, h), &params).to_grid(w, h);
        let (flat_wrong, flat_unknown) = tally(&flat_read, &grid);
        assert!(flat_wrong + flat_unknown > 20, "{version}: flat {flat_wrong} wrong, {flat_unknown} unknown");
        // Bent through the anchors, every module reads.
        let bent = Projection::new(w, h, refined.corners, &controls).unwrap();
        let bent_read = classify(&sample_modules(&image, &bent, w, h), &params).to_grid(w, h);
        let (wrong, unknown) = tally(&bent_read, &grid);
        assert_eq!(wrong, 0, "{version}: {wrong} wrong through the anchors");
        assert!(unknown <= 3, "{version}: {unknown} unknown through the anchors");
        let report = decode_plain(&bent_read, &layout, 0);
        assert!(report.is_trustworthy(), "{version}");
        // Refining the anchors against the photo leaves them on their
        // patterns.
        let module_px = (QUAD[1][0] - QUAD[0][0]) / w as f64;
        let again = refine_anchors(&image, func, refined.corners, &found.anchors, &params);
        assert_eq!(again.len(), found.anchors.len());
        for (a, b) in again.iter().zip(&found.anchors) {
            assert!(distance(a.center, b.center) < 0.3 * module_px, "{version}: {:?} moved {:.2} px", a.at, distance(a.center, b.center));
        }
    }
}

/// Where a flat point of the symbol shows in a photo made by
/// `render_warped`: the inverse of its bulge along the ray from the centre.
fn warped_position(quad: [[f64; 2]; 4], k: f64, flat: [f64; 2]) -> [f64; 2] {
    let c = quad.iter().fold([0.0, 0.0], |a, p| [a[0] + p[0] / 4.0, a[1] + p[1] / 4.0]);
    let r2 = quad.iter().map(|p| (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2)).fold(0.0, f64::max);
    let target = distance(flat, c);
    // r (1 + k r² / R²) = target, by Newton's method.
    let mut r = target;
    for _ in 0..30 {
        let f = r * (1.0 + k * r * r / r2) - target;
        let df = 1.0 + 3.0 * k * r * r / r2;
        r -= f / df;
    }
    let scale = if target > 0.0 { r / target } else { 0.0 };
    [c[0] + (flat[0] - c[0]) * scale, c[1] + (flat[1] - c[1]) * scale]
}

#[test]
fn timing_anchors_follow_a_curled_small_symbol() {
    // A small QR symbol has no alignment pattern, or one: on curled paper
    // the map had nothing to bend through between the finders. The dark
    // modules of the timing patterns are anchors now, as OpenCV
    // straightens a curved symbol along its sides and zxing-cpp holds a
    // Micro QR symbol's transform by the ends of its timing patterns. A
    // Micro QR symbol's timing patterns lie on its edges, where they place
    // its corners (see `a_curled_edge_still_places_the_far_corner`) but
    // say little of its middle, so it is not asked here.
    let params = SampleParams::default();
    for (version, ec, k) in [(SymbolVersion::Qr(1), EcLevel::M, 0.15), (SymbolVersion::Qr(2), EcLevel::M, 0.15)] {
        let grid = symbol(version, ec);
        let layout = layout(version, ec);
        let (w, h) = (grid.width(), grid.height());
        let image = render_warped(&grid, QUAD, (760, 640), k);
        let region = loosen(QUAD, 1.1, [5.0, -4.0]);
        let found = detect_outline(&image, version, region, &params).unwrap_or_else(|| panic!("{version}: nothing found"));
        assert_eq!(found.version, version);
        assert!(found.count(AnchorKind::Timing) >= 4, "{version}: {:?}", found.anchors);
        let func = &layout.func;
        let controls: Vec<ControlPoint> = found.anchors.iter().map(|a| a.control_point()).collect();
        // The finder corners are the capstones' own, held through the
        // refinement.
        assert!(found.measured_corners.contains(&0), "{version}: {:?}", found.measured_corners);
        let refined = refine_corners_holding(&image, func, found.corners, &controls, &found.held_corners(), &params).unwrap();
        let flat = Homography::rect_to_quad(w as f64, h as f64, refined.corners).unwrap();
        let flat_read = classify(&sample_modules(&image, &flat, w, h), &params).to_grid(w, h);
        let (flat_wrong, flat_unknown) = tally(&flat_read, &grid);
        let bent = Projection::new(w, h, refined.corners, &controls).unwrap();
        let bent_read = classify(&sample_modules(&image, &bent, w, h), &params).to_grid(w, h);
        let (wrong, unknown) = tally(&bent_read, &grid);
        // Unknown modules are erasures the code corrects; wrong ones cost
        // twice as much, and the map through the anchors must leave few.
        assert!(
            flat_wrong + flat_unknown >= 10 && 2 * (wrong + unknown) < flat_wrong + flat_unknown,
            "{version}: flat {flat_wrong} wrong {flat_unknown} unknown; bent {wrong} wrong {unknown} unknown"
        );
        assert!(wrong <= 2, "{version}: bent {wrong} wrong {unknown} unknown");
        let report = decode_plain(&bent_read, &layout, 0);
        assert!(report.is_trustworthy(), "{version}: bent {wrong} wrong {unknown} unknown");
    }
}

#[test]
fn a_curled_edge_still_places_the_far_corner() {
    // A Micro QR symbol's far corner is placed by tracing its edges to
    // where they meet. Curled paper bows the edges, and a line fitted to
    // the crossings along one no longer passes; a bend fitted to them
    // gives the edge's tangent at the corner instead.
    let params = SampleParams::default();
    let (version, ec, k) = (SymbolVersion::Micro(4), EcLevel::L, 0.15);
    let grid = symbol(version, ec);
    let (w, h) = (grid.width() as f64, grid.height() as f64);
    let image = render_warped(&grid, QUAD, (760, 640), k);
    let region = loosen(QUAD, 1.1, [5.0, -4.0]);
    let found = detect_outline(&image, version, region, &params).unwrap();
    assert_eq!(found.version, version);
    let flat = Homography::rect_to_quad(w, h, QUAD).unwrap();
    let truth = warped_position(QUAD, k, flat.apply(w, h));
    let module_px = distance(QUAD[0], QUAD[1]) / w;
    let error = distance(found.corners[2], truth);
    assert!(error < 0.3 * module_px, "far corner {:?} is {:.2} modules from {truth:?}", found.corners[2], error / module_px);
    // The other corners are the finder's own and the ends of the timing
    // patterns, which the edge trace and the capstone fit place as well.
    for (i, flat_corner) in [(0usize, [0.0, 0.0]), (1, [w, 0.0]), (3, [0.0, h])] {
        let t = warped_position(QUAD, k, flat.apply(flat_corner[0], flat_corner[1]));
        let e = distance(found.corners[i], t) / module_px;
        assert!(e < 0.4, "corner {i} is {e:.2} modules off");
    }
    assert!(found.edged_corners.contains(&2), "the far corner was not traced along the edges: {:?}", found.edged_corners);
}

#[test]
fn a_corner_nothing_holds_is_not_walked_by_the_refinement() {
    // On a blurred photo the corner refinement used to walk a Micro QR
    // symbol's far corner, which no fixed pattern is near, two modules
    // off, and lose the symbol the detector had placed. A loose corner
    // is held through the descent now and settled by the edge after it.
    let params = SampleParams::default();
    for (version, ec) in [(SymbolVersion::Micro(4), EcLevel::L), (SymbolVersion::Micro(3), EcLevel::L), (SymbolVersion::Qr(1), EcLevel::M)] {
        let grid = symbol(version, ec);
        let layout = layout(version, ec);
        let (w, h) = (grid.width(), grid.height());
        let sharp = render(&grid, QUAD, (760, 640), |_, _, v| v);
        let blurred = Luma::new(sharp.width, sharp.height, Filter::Blur.apply(&sharp.data, sharp.width, sharp.height, 3));
        let blurred = Luma::new(sharp.width, sharp.height, Filter::Blur.apply(&blurred.data, sharp.width, sharp.height, 3));
        let region = loosen(QUAD, 1.1, [5.0, -4.0]);
        let found = detect_outline(&blurred, version, region, &params).unwrap_or_else(|| panic!("{version}: nothing found"));
        assert_eq!(found.version, version);
        let func = &layout.func;
        let controls: Vec<ControlPoint> = found.anchors.iter().map(|a| a.control_point()).collect();
        let refined = refine_corners(&blurred, func, found.corners, &controls, &params).unwrap();
        assert!(refined.after.0 >= refined.before.0, "{version}: agreement fell from {:?} to {:?}", refined.before, refined.after);
        let module_px = distance(QUAD[0], QUAD[1]) / w as f64;
        for (i, (a, b)) in refined.corners.iter().zip(&QUAD).enumerate() {
            let off = distance(*a, *b) / module_px;
            assert!(off < 0.4, "{version}: corner {i} is {off:.2} modules off after refinement");
        }
        let read = read_back(&blurred, refined.corners, &grid, &params);
        let (wrong, unknown) = tally(&read, &grid);
        assert!(wrong == 0 && unknown <= 2, "{version}: {wrong} wrong, {unknown} unknown");
        let _ = h;
    }
}

#[test]
fn a_displaced_anchor_is_refined_back_onto_its_pattern() {
    let (version, ec) = (SymbolVersion::Qr(10), EcLevel::Q);
    let grid = symbol(version, ec);
    let image = render(&grid, QUAD, (760, 640), |_, _, v| v);
    let params = SampleParams::default();
    let found = detect_outline(&image, version, loosen(QUAD, 1.08, [4.0, 3.0]), &params).unwrap();
    let func = layout(version, ec).func;
    let module_px = (QUAD[1][0] - QUAD[0][0]) / grid.width() as f64;
    let truth = Homography::rect_to_quad(grid.width() as f64, grid.height() as f64, QUAD).unwrap();
    let mut anchors = found.anchors.clone();
    let k = anchors.iter().position(|a| a.kind == AnchorKind::Alignment).unwrap();
    // Half a module off, as a hand might drop it.
    anchors[k].center[0] += 0.5 * module_px;
    anchors[k].center[1] -= 0.4 * module_px;
    let refined = refine_anchors(&image, &func, found.corners, &anchors, &params);
    let want = truth.apply(anchors[k].at[0], anchors[k].at[1]);
    let off = distance(refined[k].center, want) / module_px;
    assert!(off < 0.1, "refined anchor {off:.3} modules off");
    assert!(distance(anchors[k].center, want) / module_px > 0.5);
}

#[test]
fn an_anchor_the_decoded_modules_contradict_is_dropped() {
    // A version 7 symbol, eight pixels a module, found from a loose
    // outline. With every module known, as a decode that solved the
    // symbol knows them, every anchor reads right and none is dropped.
    // One alignment anchor moved two modules off its pattern bends the
    // map there, and the modules about it read wrong through the bent
    // map and right through the map without it: that anchor alone is
    // dropped, and the others are kept as they were.
    let (version, ec) = (SymbolVersion::Qr(7), EcLevel::M);
    let grid = symbol(version, ec);
    let quad = quad_at(version, 8.0, [260.0, 260.0]);
    let image = render(&grid, quad, (520, 520), |_, _, v| v);
    let params = SampleParams::default();
    let found = detect_outline(&image, SymbolVersion::Qr(2), loosen(quad, 1.1, [4.0, -3.0]), &params).unwrap();
    assert_eq!(found.version, version);
    let func = FunctionMap::new(version);
    let template = Template {
        width: func.width,
        height: func.height,
        expected: (0..func.height).flat_map(|y| (0..func.width).map(move |x| (x, y))).map(|(x, y)| grid.get(x, y).as_dark()).collect(),
    };
    let kept = drop_contradicted_anchors(&image, &func, &template, found.corners, &found.anchors, found.inverted, &params);
    assert_eq!(kept, found.anchors);
    let k = found
        .anchors
        .iter()
        .position(|a| a.kind == AnchorKind::Alignment && a.at == [22.5, 22.5])
        .unwrap_or_else(|| panic!("no middle alignment anchor among {:?}", found.anchors));
    let mut anchors = found.anchors.clone();
    anchors[k].center[0] += 2.0 * 8.0;
    let kept = drop_contradicted_anchors(&image, &func, &template, found.corners, &anchors, found.inverted, &params);
    assert_eq!(kept.len(), anchors.len() - 1, "{kept:?}");
    assert!(!kept.contains(&anchors[k]));
    assert!(anchors.iter().enumerate().filter(|(i, _)| *i != k).all(|(_, a)| kept.contains(a)), "{kept:?}");
}

#[test]
fn a_finder_with_a_torn_corner_still_anchors() {
    let params = SampleParams::default();
    for (version, ec) in [(SymbolVersion::Qr(6), EcLevel::M), (SymbolVersion::Micro(3), EcLevel::L)] {
        let mut grid = symbol(version, ec);
        let n = grid.width();
        // Four by four modules torn off a finder's corner to bare paper: on
        // the QR symbol the top-right finder's outer corner, on the Micro
        // QR symbol the only finder's.
        let (x0, y0) = if version.family() == Family::Qr { (n - 4, 0) } else { (0, 0) };
        for y in y0..y0 + 4 {
            for x in x0..x0 + 4 {
                grid.set(x, y, Module::Light);
            }
        }
        let image = render(&grid, QUAD, (760, 640), |_, _, v| v);
        let found = detect_outline(&image, version, loosen(QUAD, 1.1, [5.0, -4.0]), &params)
            .unwrap_or_else(|| panic!("{version}: nothing found with a torn finder"));
        assert_eq!(found.version, version);
        let finders = if version.family() == Family::Qr { 3 } else { 1 };
        assert_eq!(found.count(AnchorKind::Finder), finders, "{version}: {:?}", found.anchors);
        assert_eq!(found.partial(), 1, "{version}: the torn finder is placed by its other three quarters");
        let module_px = (QUAD[1][0] - QUAD[0][0]) / n as f64;
        let error = corner_error(found.corners, QUAD) / module_px;
        assert!(error < 0.75, "{version}: corners {error:.2} modules off");
        let func = layout(version, ec).func;
        let controls: Vec<ControlPoint> = found.anchors.iter().map(|a| a.control_point()).collect();
        let refined = refine_corners(&image, &func, found.corners, &controls, &params).unwrap();
        assert_eq!(read_back(&image, refined.corners, &grid, &params), grid, "{version}");
    }
}

#[test]
fn loose_corners_are_traced_along_the_edge_or_settled_without_a_quiet_zone() {
    let params = SampleParams::default();
    for (version, ec) in [(SymbolVersion::Qr(1), EcLevel::M), (SymbolVersion::Micro(3), EcLevel::L)] {
        let grid = symbol(version, ec);
        let module_px = (QUAD[1][0] - QUAD[0][0]) / grid.width() as f64;
        let func = layout(version, ec).func;
        // On bare paper the bottom-right corner, which no fixed pattern
        // reaches, is placed by tracing the edge.
        let image = render(&grid, QUAD, (760, 640), |x, _, v| v * (0.8 + 0.2 * (x / 760.0) as f32));
        let found = detect_outline(&image, version, loosen(QUAD, 1.1, [5.0, -4.0]), &params).unwrap();
        assert!(found.edged_corners.contains(&2), "{version}: {:?}", found.edged_corners);
        let br = distance(found.corners[2], QUAD[2]) / module_px;
        assert!(br < 0.35, "{version}: traced corner {br:.2} modules off");
        // Hemmed in by other print, there is no edge to trace; the corner
        // settles where the data modules read crisply instead.
        let busy = render_busy(&grid, QUAD, (760, 640));
        let found = detect_outline(&busy, version, loosen(QUAD, 1.1, [5.0, -4.0]), &params)
            .unwrap_or_else(|| panic!("{version}: nothing found without a quiet zone"));
        assert_eq!(found.version, version);
        assert!(found.edged_corners.is_empty(), "{version}: traced {:?} with no quiet zone", found.edged_corners);
        let error = corner_error(found.corners, QUAD) / module_px;
        assert!(error < 0.75, "{version}: corners {error:.2} modules off without a quiet zone");
        let controls: Vec<ControlPoint> = found.anchors.iter().map(|a| a.control_point()).collect();
        let refined = refine_corners(&busy, &func, found.corners, &controls, &params).unwrap();
        assert_eq!(read_back(&busy, refined.corners, &grid, &params), grid, "{version}: without a quiet zone");
    }
}

#[test]
fn bloomed_finder_patterns_are_still_found() {
    let (version, ec) = (SymbolVersion::Qr(4), EcLevel::M);
    let grid = symbol(version, ec);
    // Light has bled six tenths of a module into the dark: every dark run
    // of a finder pattern is well outside the run-by-run tolerance.
    let image = render_bloomed(&grid, QUAD, (760, 640), 0.6);
    let params = SampleParams::default();
    let found = detect_outline(&image, version, loosen(QUAD, 1.1, [5.0, -4.0]), &params).expect("bloomed symbol found");
    assert_eq!(found.version, version);
    assert_eq!(found.count(AnchorKind::Finder), 3, "{:?}", found.anchors);
    let module_px = (QUAD[1][0] - QUAD[0][0]) / grid.width() as f64;
    let error = corner_error(found.corners, QUAD) / module_px;
    assert!(error < 0.75, "corners {error:.2} modules off");
}

#[test]
fn detection_decodes_both_ways_round_when_the_format_bits_are_gone() {
    let (version, ec) = (SymbolVersion::Qr(4), EcLevel::M);
    let clean = symbol(version, ec);
    let func = layout(version, ec).func;
    let mut grid = clean.clone();
    // Both copies of the format information torn away to bare paper.
    for pos in func.format.iter().flat_map(|c| c.bits.iter()) {
        grid.set(pos.x as usize, pos.y as usize, Module::Light);
    }
    let params = SampleParams::default();
    // Seen from the back: nothing says so but the decode.
    let image = render(&grid.flipped_horizontal(), QUAD, (760, 640), |_, _, v| v);
    let found = detect_outline(&image, version, loosen(QUAD, 1.1, [-5.0, 4.0]), &params).unwrap();
    assert_ne!(found.info.format.valid, Some(true), "{:?}", found.info);
    assert!(found.mirrored && found.mirrored_by_decode, "{:?}", found.info);
    let controls: Vec<ControlPoint> = found.anchors.iter().map(|a| a.control_point()).collect();
    let refined = refine_corners(&image, &func, found.corners, &controls, &params).unwrap();
    assert_eq!(read_back(&image, refined.corners, &grid, &params), grid);
    // The right way round, it is never taken for a mirror image.
    let plain = render(&grid, QUAD, (760, 640), |_, _, v| v);
    let found = detect_outline(&plain, version, loosen(QUAD, 1.1, [-5.0, 4.0]), &params).unwrap();
    assert!(!found.mirrored, "{:?}", found.info);
}

/// A square outline of `side` pixels turned `degrees` clockwise about
/// `centre`, as top-left, top-right, bottom-right, bottom-left of the
/// symbol drawn in it.
fn turned_quad(centre: [f64; 2], side: f64, degrees: f64) -> [[f64; 2]; 4] {
    let (s, c) = degrees.to_radians().sin_cos();
    let h = side / 2.0;
    [[-h, -h], [h, -h], [h, h], [-h, h]].map(|[x, y]| [centre[0] + x * c - y * s, centre[1] + x * s + y * c])
}

/// The box a hand draws around a symbol without turning it: the bounding
/// box of `quad`, widened by `share` of itself.
fn bounding_box(quad: [[f64; 2]; 4], share: f64) -> [[f64; 2]; 4] {
    let (x0, x1) = quad.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p[0]), a.1.max(p[0])));
    let (y0, y1) = quad.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p[1]), a.1.max(p[1])));
    let (dx, dy) = ((x1 - x0) * share / 2.0, (y1 - y0) * share / 2.0);
    [[x0 - dx, y0 - dy], [x1 + dx, y0 - dy], [x1 + dx, y1 + dy], [x0 - dx, y1 + dy]]
}

/// The photo read through a detection as the app samples it, with the
/// corners refined and the anchors kept.
fn read_through(image: &Luma, found: &qr_forensics::imaging::detect::Detection, grid: &ModuleGrid, params: &SampleParams) -> ModuleGrid {
    let func = FunctionMap::new(found.version);
    let controls: Vec<ControlPoint> = found.anchors.iter().map(|a| a.control_point()).collect();
    let corners = refine_corners(image, &func, found.corners, &controls, params).map_or(found.corners, |r| r.corners);
    let map = Projection::new(func.width, func.height, corners, &controls).unwrap();
    let sampled = classify(&sample_modules(image, &map, func.width, func.height), params).to_grid(func.width, func.height);
    let mut out = ModuleGrid::new(grid.width(), grid.height());
    for y in 0..grid.height() {
        for x in 0..grid.width() {
            out.set(x, y, sampled.get(x, y));
        }
    }
    out
}

#[test]
fn detection_copes_with_a_turned_symbol_in_an_axis_aligned_box() {
    // A symbol photographed at an angle, outlined by a box drawn square to
    // the photo: every finder lies far from where the box expects one, and
    // ZXing and ZBar, which place the symbol from the finder centres alone,
    // read it without noticing.
    let (version, ec) = (SymbolVersion::Qr(4), EcLevel::M);
    let grid = symbol(version, ec);
    let params = SampleParams::default();
    let module_px = 12.0;
    for degrees in [15.0, 30.0, 45.0, 60.0] {
        let quad = turned_quad([380.0, 320.0], 33.0 * module_px, degrees);
        let image = render(&grid, quad, (760, 640), |_, _, v| v);
        let region = bounding_box(quad, 0.1);
        let found = detect_outline(&image, version, region, &params).unwrap_or_else(|| panic!("{degrees}°: nothing found"));
        assert_eq!(found.version, version, "{degrees}°");
        assert_eq!(found.count(AnchorKind::Finder), 3, "{degrees}°: {:?}", found.anchors);
        let coarse = corner_error(found.corners, quad) / module_px;
        assert!(coarse < 0.75, "{degrees}°: corners {coarse:.2} modules off");
        let func = layout(version, ec).func;
        let controls: Vec<ControlPoint> = found.anchors.iter().map(|a| a.control_point()).collect();
        let refined = refine_corners_holding(&image, &func, found.corners, &controls, &found.held_corners(), &params).unwrap();
        // The three finder corners are the capstones' own, to a fifth of
        // a pixel; the fourth, placed by the contrast of the whole outline
        // alone, lands within a tenth of a module on edges this jagged.
        for &i in &found.measured_corners {
            let off = distance(refined.corners[i], quad[i]);
            assert!(off < 0.5, "{degrees}°: finder corner {i} {off:.2} px off after refinement");
        }
        let fine = corner_error(refined.corners, quad);
        assert!(fine < 0.1 * module_px, "{degrees}°: corners {fine:.2} px off after refinement");
        assert_eq!(read_back(&image, refined.corners, &grid, &params), grid, "{degrees}°");
        // The same symbol found with no outline at all.
        let all = find_symbols(&image, Family::Qr, &params);
        assert_eq!(all.len(), 1, "{degrees}°: {all:?}");
        assert_eq!(all[0].version, version, "{degrees}°");
        let coarse = corner_error(all[0].corners, quad) / module_px;
        assert!(coarse < 0.75, "{degrees}°: found corners {coarse:.2} modules off");
    }
}

#[test]
fn a_loose_box_around_a_small_symbol_never_gives_a_wrong_version() {
    let (version, ec) = (SymbolVersion::Qr(2), EcLevel::M);
    let grid = symbol(version, ec);
    let params = SampleParams::default();
    for module_px in [2.0, 2.5, 3.0] {
        let side = 25.0 * module_px;
        let quad = [[100.0, 80.0], [100.0 + side, 80.0], [100.0 + side, 80.0 + side], [100.0, 80.0 + side]];
        let image = render(&grid, quad, (500, 400), |_, _, v| v);
        // A box 1.7 times the symbol, as a hand draws one around something small.
        let region = bounding_box(quad, 0.7);
        match detect_outline(&image, version, region, &params) {
            None => panic!("{module_px} px a module: nothing found in a loose box"),
            Some(found) => {
                assert_eq!(found.version, version, "{module_px} px a module: {found:?}");
                let error = corner_error(found.corners, quad) / module_px;
                assert!(error < 0.75, "{module_px} px a module: corners {error:.2} modules off");
                assert_eq!(read_through(&image, &found, &grid, &params), grid, "{module_px} px a module");
            }
        }
    }
}

#[test]
fn a_symbol_a_pixel_a_module_is_found_by_enlarging() {
    let (version, ec) = (SymbolVersion::Qr(2), EcLevel::M);
    let grid = symbol(version, ec);
    let params = SampleParams::default();
    let quad = [[100.0, 80.0], [125.0, 80.0], [125.0, 105.0], [100.0, 105.0]];
    let image = render(&grid, quad, (500, 400), |_, _, v| v);
    // Through a tight box.
    let found = detect_outline(&image, version, bounding_box(quad, 0.2), &params).expect("found through a tight box");
    assert_eq!(found.version, version);
    // A render this clean is read straight off its pixels, with no
    // anchors and no enlargement; a photo of one would be enlarged.
    assert!(found.retry.is_some() || found.anchors.is_empty(), "{found:?}");
    let error = corner_error(found.corners, quad);
    assert!(error < 1.0, "corners {error:.2} px off");
    assert_eq!(read_as_found(&image, &found, &grid, &params), grid);
    // And with no outline at all.
    let all = find_symbols(&image, Family::Qr, &params);
    assert_eq!(all.len(), 1, "{all:?}");
    assert_eq!(all[0].version, version);
    assert!(all[0].retry.is_some() || all[0].anchors.is_empty(), "{:?}", all[0]);
    let error = corner_error(all[0].corners, quad);
    assert!(error < 1.0, "found corners {error:.2} px off");
}

/// `image` out of focus: blurred by a Gaussian `sigma` pixels wide.
fn defocus(image: &Luma, sigma: f64) -> Luma {
    let reach = (3.0 * sigma).ceil() as i64;
    let raw: Vec<f64> = (-reach..=reach).map(|k| (-0.5 * (k as f64 / sigma).powi(2)).exp()).collect();
    let total: f64 = raw.iter().sum();
    let (w, h) = (image.width as i64, image.height as i64);
    let pass = |from: &[f32], across: bool| -> Vec<f32> {
        (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let at = |k: i64| {
                    let (nx, ny) = if across { ((x + k).clamp(0, w - 1), y) } else { (x, (y + k).clamp(0, h - 1)) };
                    f64::from(from[(ny * w + nx) as usize])
                };
                ((-reach..=reach).zip(&raw).map(|(k, weight)| weight * at(k)).sum::<f64>() / total) as f32
            })
            .collect()
    };
    Luma::new(image.width, image.height, pass(&pass(&image.data, true), false))
}

/// How many modules of `read` differ from `truth`'s, unknown ones
/// counted with the wrong.
fn misread(read: &ModuleGrid, truth: &ModuleGrid) -> usize {
    let (wrong, unknown) = tally(read, truth);
    wrong + unknown
}

#[test]
fn a_symbol_out_of_focus_reads_through_the_photo_sharpened() {
    // A version 5 symbol three pixels a module, a little turned, out of
    // focus by seven tenths of a module: its finder patterns' rings have
    // run into their stones, and a dark module among light ones is grey.
    // As the photo lies it does not decode; the read is made again
    // through the photo enlarged and sharpened, and says so.
    let (version, ec) = (SymbolVersion::Qr(5), EcLevel::M);
    let grid = symbol(version, ec);
    let truth = decode_plain(&grid, &layout(version, ec), 0).text.plain();
    let quad = turned_quad([200.0, 180.0], 3.0 * version.width() as f64, 7.0);
    let image = defocus(&render(&grid, quad, (400, 360), |_, _, v| v), 2.1);
    let (params, opts) = (SampleParams::default(), DecodeOptions::default());
    let read = read_outline(&image, SymbolVersion::Qr(2), bounding_box(quad, 0.2), &params, &opts).expect("found");
    assert!(read.is_trustworthy(), "{:?} of {} blocks", read.report.blocks_verified(), read.report.blocks.len());
    assert_eq!(read.text(), truth);
    assert_eq!(read.version(), version);
    assert!(read.enhance.sharpen != Sharpen::Off && read.enhance.enlarge > 1, "read through {:?}", read.enhance);
    assert_eq!(read.detection.retry, Some(Retry::Enhanced(read.enhance)));
    // The placement is in the photo's own pixels.
    let error = corner_error(read.detection.corners, quad);
    assert!(error < 1.5, "corners {error:.2} px off");
    // Sampled through the same copy, with the kernel and threshold rule
    // of the read, the placement gives the read's grid again: what the
    // window's Sampling does once it has taken the read's settings over.
    // Sampled off the photo as it lies, the same placement reads more
    // than twice as many modules wrong or not at all.
    let sample = |enhance: Enhance| {
        let how = SampleParams { kernel: read.kernel, local: read.local, enhance, ..params };
        let map = read.detection.projection().expect("the placement maps");
        classify(&sample_modules_as(&image, &map, grid.width(), grid.height(), &how), &how).to_grid(grid.width(), grid.height())
    };
    let (again, plain) = (sample(read.enhance), sample(Enhance::NONE));
    let differing = (0..grid.width() * grid.height()).filter(|i| again.get(i % grid.width(), i / grid.width()) != read.grid.get(i % grid.width(), i / grid.width())).count();
    assert!(differing <= 2, "{differing} modules read otherwise the second time");
    assert!(misread(&plain, &grid) > 2 * misread(&again, &grid).max(10), "{} as it lies, {} sharpened", misread(&plain, &grid), misread(&again, &grid));
    // Asked for from the start, the same copy is what is read first.
    let asked = SampleParams { enhance: read.enhance, ..params };
    let direct = read_outline(&image, SymbolVersion::Qr(2), bounding_box(quad, 0.2), &asked, &opts).expect("found");
    assert!(direct.is_trustworthy() && direct.enhance == read.enhance && direct.text() == truth);
    // And a search of the whole photo finds the symbol and reads it.
    let found = find_symbols(&image, Family::Qr, &params);
    assert_eq!(found.len(), 1, "{found:?}");
    let read = read_detection(&image, found[0].clone(), &params, &opts).expect("read where it was found");
    assert!(read.is_trustworthy() && read.text() == truth, "{:?}", read.report.text.plain());
    assert!(corner_error(read.detection.corners, quad) < 1.5);
}

#[test]
fn small_modules_a_lens_has_run_together_read_enlarged() {
    // A version 3 symbol a pixel and a quarter a module, blurred by
    // seven tenths of a pixel, as a lens and a sensor leave a symbol far
    // from the camera: read through the photo enlarged four times and
    // sharpened, with its corners placed in the photo's own pixels to
    // about a module.
    let (version, ec) = (SymbolVersion::Qr(3), EcLevel::M);
    let grid = symbol(version, ec);
    let truth = decode_plain(&grid, &layout(version, ec), 0).text.plain();
    let side = 1.25 * version.width() as f64;
    let quad = [[150.3, 120.6], [150.3 + side, 120.6], [150.3 + side, 120.6 + side], [150.3, 120.6 + side]];
    let image = defocus(&render(&grid, quad, (360, 300), |_, _, v| v), 0.7);
    let (params, opts) = (SampleParams::default(), DecodeOptions::default());
    let read = read_outline(&image, SymbolVersion::Qr(2), bounding_box(quad, 0.25), &params, &opts).expect("found");
    assert!(read.is_trustworthy(), "{:?} of {} blocks", read.report.blocks_verified(), read.report.blocks.len());
    assert_eq!((read.text(), read.version()), (truth, version));
    assert_eq!(read.enhance.enlarge, 4, "read through {:?}", read.enhance);
    assert!(read.enhance.sharpen != Sharpen::Off);
    let error = corner_error(read.detection.corners, quad);
    assert!(error < 1.5, "corners {error:.2} px off");
}

#[test]
fn a_scratch_down_a_finders_centre_column_is_survived() {
    // A vertical scratch through the centre of the top-left and bottom-left
    // finders, two pixels wide: every row across either finder's centre
    // reads broken, and only a scan down the columns, as ZBar runs one,
    // still sees the 1:1:3:1:1 of the finder.
    let (version, ec) = (SymbolVersion::Qr(4), EcLevel::M);
    let grid = symbol(version, ec);
    let module_px = 12.0;
    let side = 33.0 * module_px;
    let quad = [[100.0, 80.0], [100.0 + side, 80.0], [100.0 + side, 80.0 + side], [100.0, 80.0 + side]];
    let mut image = render(&grid, quad, (760, 640), |_, _, v| v);
    let scratch_x = (100.0 + 3.5 * module_px) as usize;
    for y in 60..(80.0 + side) as usize + 20 {
        for x in scratch_x - 1..=scratch_x {
            image.data[y * image.width + x] = 0.95;
        }
    }
    let params = SampleParams::default();
    let found = detect_outline(&image, version, loosen(quad, 1.1, [5.0, -4.0]), &params).expect("found despite the scratch");
    assert_eq!(found.version, version);
    assert_eq!(found.count(AnchorKind::Finder), 3, "{:?}", found.anchors);
    assert_eq!(found.retry, None, "found without the retry ladder: {:?}", found.retry);
    let error = corner_error(found.corners, quad) / module_px;
    assert!(error < 0.75, "corners {error:.2} modules off");
    let read = read_through(&image, &found, &grid, &params);
    let layout = layout(version, ec);
    let report = decode_plain(&read, &layout, 0);
    assert!(report.is_trustworthy(), "{:?}", report.score);
    assert_eq!(report.text.plain(), decode_plain(&grid, &layout, 0).text.plain());
}

/// Photograph `grid` as a coarse halftone print: every dark module is a
/// grid of dots two pixels square, two pixels apart, so no row or column
/// has a run of dark longer than two pixels, and resampling cannot bridge
/// the gaps either.
fn render_halftone(grid: &ModuleGrid, quad: [[f64; 2]; 4], size: (usize, usize)) -> Luma {
    let (gw, gh) = (grid.width() as f64, grid.height() as f64);
    let to_grid = Homography::rect_to_quad(gw, gh, quad).unwrap().inverse().unwrap();
    let mut data = Vec::with_capacity(size.0 * size.1);
    for py in 0..size.1 {
        for px in 0..size.0 {
            let [mx, my] = to_grid.apply(px as f64 + 0.5, py as f64 + 0.5);
            let inside = mx >= 0.0 && my >= 0.0 && mx < gw && my < gh;
            let dark = inside && grid.get(mx as usize, my as usize) == Module::Dark && px % 4 < 2 && py % 4 < 2;
            data.push(if dark { 0.08 } else { 0.92 });
        }
    }
    Luma::new(size.0, size.1, data)
}

#[test]
fn a_halftone_print_is_found_by_the_retry_ladder() {
    let (version, ec) = (SymbolVersion::Qr(4), EcLevel::M);
    let grid = symbol(version, ec);
    let module_px = 12.0;
    let side = 33.0 * module_px;
    let quad = [[100.0, 80.0], [100.0 + side, 80.0], [100.0 + side, 80.0 + side], [100.0, 80.0 + side]];
    let image = render_halftone(&grid, quad, (760, 640));
    let params = SampleParams::default();
    // The plain scan sees no runs; the regions of the binarised image,
    // closed over the gaps between the dots, propose the finders without
    // the ladder, and the ladder stands behind them.
    let found = detect_outline(&image, version, loosen(quad, 1.1, [5.0, -4.0]), &params).expect("found through the regions or the ladder");
    assert_eq!(found.version, version);
    let error = corner_error(found.corners, quad) / module_px;
    assert!(error < 0.75, "corners {error:.2} modules off, via {:?}", found.retry);
    let all = find_symbols(&image, Family::Qr, &params);
    assert_eq!(all.len(), 1, "{all:?}");
    assert_eq!(all[0].version, version);
    let error = corner_error(all[0].corners, quad) / module_px;
    assert!(error < 0.75, "found corners {error:.2} modules off, via {:?}", all[0].retry);
}

/// The finder patterns of `version` and nothing else, on blank paper:
/// marks that stand where a symbol's finders would.
fn finders_alone(version: SymbolVersion) -> ModuleGrid {
    let func = FunctionMap::new(version);
    let mut grid = ModuleGrid::new(func.width, func.height);
    for y in 0..func.height {
        for x in 0..func.width {
            let dark = func.role(x, y) == Role::Finder && func.expected(x, y) == Some(true);
            grid.set(x, y, Module::from_dark(dark));
        }
    }
    grid
}

#[test]
fn finder_marks_with_nothing_between_them_are_no_symbol() {
    // Three marks of nested squares at the spacing of a version 1
    // symbol's finders, as registration marks or a logo may stand. The
    // finders and separators are 192 of that symbol's 203 fixed modules,
    // and 64 of a Micro QR M4 symbol's 82, so the marks fit either at
    // nine parts in ten; but no timing pattern runs between them and no
    // format word lies beside them, and a search offers neither.
    let version = SymbolVersion::Qr(1);
    let image = render(&finders_alone(version), quad_at(version, 10.0, [250.0, 200.0]), (500, 400), |_, _, v| v);
    let params = SampleParams::default();
    for family in [Family::Qr, Family::Micro] {
        let found = find_symbols(&image, family, &params);
        let offered: Vec<_> = found.iter().map(|f| (f.version, f.score())).collect();
        assert!(found.is_empty(), "as {}: {offered:?}", family.label());
    }
}

#[test]
fn the_finders_of_a_qr_code_are_not_micro_qr_symbols() {
    // Each finder of a QR Code is the finder of a Micro QR symbol too,
    // and fits one hung off it any way round; turned to lie along the
    // QR Code's edge, the Micro QR symbol even finds a timing pattern
    // under one of its own, the QR Code's. Its other one lies on blank
    // paper or across the data.
    let (version, ec) = (SymbolVersion::Qr(4), EcLevel::M);
    let image = render(&symbol(version, ec), quad_at(version, 10.0, [330.0, 280.0]), (660, 560), |_, _, v| v);
    let params = SampleParams::default();
    let found = find_symbols(&image, Family::Qr, &params);
    assert_eq!(found.iter().map(|f| f.version).collect::<Vec<_>>(), [version]);
    let micro = find_symbols(&image, Family::Micro, &params);
    let offered: Vec<_> = micro.iter().map(|f| (f.version, f.score())).collect();
    assert!(micro.is_empty(), "{offered:?}");
}

#[test]
fn a_symbol_with_a_finder_gone_is_found_in_the_photo_around_it() {
    // A finder torn off, or under a sticker: an outline drawn round the
    // symbol has always found it from the two that are left, and a search
    // of the photo proposed from three finders or not at all. Two along a
    // side have the timing pattern between them to vouch for the pair,
    // and two across the diagonal lie turned as a symbol's corners do.
    let (version, ec) = (SymbolVersion::Qr(4), EcLevel::M);
    let n = version.width();
    let quad = quad_at(version, 10.0, [330.0, 280.0]);
    let params = SampleParams::default();
    for (gone, which) in [([n - 8, 0], "top right"), ([0, n - 8], "bottom left"), ([0, 0], "top left")] {
        let mut grid = symbol(version, ec);
        for y in gone[1]..gone[1] + 8 {
            for x in gone[0]..gone[0] + 8 {
                grid.set(x, y, Module::Light);
            }
        }
        let image = render(&grid, quad, (660, 560), |_, _, v| v);
        let found = find_symbols(&image, Family::Qr, &params);
        assert_eq!(found.iter().map(|f| f.version).collect::<Vec<_>>(), [version], "{which}");
        let error = corner_error(found[0].corners, quad) / 10.0;
        assert!(error < 0.75, "{which}: corners {error:.2} modules off");
    }
}

#[test]
fn every_symbol_of_a_sheet_is_found() {
    // Twelve symbols on one sheet, thirty-six finder patterns: the search
    // kept the twenty-five strongest clusters its scan threw up and gave
    // up after a dozen proposals, whatever had become of them, and found
    // five.
    let (version, ec) = (SymbolVersion::Qr(1), EcLevel::M);
    let (module, pitch, size) = (6.0, 31.0 * 6.0, (4 * 31 * 6 + 60, 3 * 31 * 6 + 60));
    let centres: Vec<[f64; 2]> = (0..12).map(|i| [30.0 + pitch * ((i % 4) as f64 + 0.5), 30.0 + pitch * ((i / 4) as f64 + 0.5)]).collect();
    let grid = symbol(version, ec);
    let mut image = Luma::new(size.0, size.1, vec![0.92; size.0 * size.1]);
    for &centre in &centres {
        let one = render(&grid, quad_at(version, module, centre), size, |_, _, v| v);
        for (pixel, ink) in image.data.iter_mut().zip(&one.data) {
            *pixel = pixel.min(*ink);
        }
    }
    let found = find_symbols(&image, Family::Qr, &SampleParams::default());
    assert_eq!(found.len(), centres.len(), "{:?}", found.iter().map(|f| (f.version, f.score())).collect::<Vec<_>>());
    for centre in centres {
        let at = |f: &qr_forensics::imaging::detect::Detection| {
            let c = f.corners.iter().fold([0.0, 0.0], |a, p| [a[0] + p[0] / 4.0, a[1] + p[1] / 4.0]);
            distance(c, centre) < 2.0 * module
        };
        assert_eq!(found.iter().filter(|f| f.version == version && at(f)).count(), 1, "{centre:?}");
    }
}

#[test]
fn a_mirrored_symbol_reads_with_its_format_word_damaged() {
    // A symbol seen from the back of the paper, with bits of its format
    // word misread the same in both copies. Read in mirror order its
    // format bits are still within reach of their word; read as it
    // lies, they are within reach of another word more often than not,
    // and the symbol was decoded under that word, plain, and no other
    // way.
    let (version, ec) = (SymbolVersion::Qr(3), EcLevel::M);
    let layout = layout(version, ec);
    let clean = symbol(version, ec);
    let text = decode_plain(&clean, &layout, 0).text.plain();
    let quad = detection_quad(version);
    let region = loosen(quad, 1.1, [-5.0, 4.0]);
    let params = SampleParams::default();
    for bits in MIRROR_FLIPS {
        let mut grid = clean.clone();
        for copy in &layout.func.format {
            for &b in bits.iter() {
                let (x, y) = (copy.bits[b].x as usize, copy.bits[b].y as usize);
                grid.set(x, y, grid.get(x, y).inverted());
            }
        }
        // The same damage on a symbol the right way round is corrected.
        let plain = decode_photo(&render(&grid, quad, (760, 640), |_, _, v| v), Family::Qr, region, &params).expect("found");
        assert!(plain.report.is_trustworthy() && !plain.detection.mirrored, "{bits:?}");
        let behind = render(&grid.flipped_horizontal(), quad, (760, 640), |_, _, v| v);
        let read = decode_photo(&behind, Family::Qr, region, &params).expect("found");
        assert!(read.report.is_trustworthy(), "{bits:?}: {:?}", read.report.score);
        assert!(read.detection.mirrored, "{bits:?}");
        assert_eq!(read.text(), text, "{bits:?}");
    }
}

/// Format bits flipped in both copies of the word by
/// [`a_mirrored_symbol_reads_with_its_format_word_damaged`]: six of the
/// seventy-nine pairs and triples, of a hundred and ten tried, that left
/// the mirrored symbol unread while the format bits alone were asked
/// which way round it lay.
const MIRROR_FLIPS: &[&[usize]] = &[&[0, 10], &[2, 9], &[3, 4], &[12, 14], &[0, 2, 3], &[1, 6, 10]];

#[test]
fn the_finder_of_a_symbol_already_found_proposes_nothing() {
    let (version, ec) = (SymbolVersion::Micro(4), EcLevel::M);
    let image = render(&symbol(version, ec), quad_at(version, 12.0, [250.0, 200.0]), (500, 400), |_, _, v| v);
    let params = SampleParams::default();
    let found = find_symbols(&image, Family::Micro, &params);
    assert_eq!(found.iter().map(|f| f.version).collect::<Vec<_>>(), [version]);
    // A search told that this symbol is found leaves its finder alone,
    // as a search of every family tells the families after QR Code.
    assert!(find_symbols_fine(&image, Family::Micro, &params, &found).is_empty());
}

#[test]
fn a_symbol_without_its_timing_patterns_is_vouched_for_by_its_format_word() {
    // The timing patterns are all a symbol with one finder shows of
    // itself beyond that finder, and a crease along each takes them.
    // The format word beside the finder still reads as a word outright,
    // which print that is no symbol does once in a thousand tries.
    let (version, ec) = (SymbolVersion::Micro(4), EcLevel::M);
    let func = FunctionMap::new(version);
    let mut grid = symbol(version, ec);
    for y in 0..func.height {
        for x in 0..func.width {
            if func.role(x, y) == Role::Timing {
                grid.set(x, y, Module::Light);
            }
        }
    }
    let image = render(&grid, quad_at(version, 12.0, [250.0, 200.0]), (500, 400), |_, _, v| v);
    let found = find_symbols(&image, Family::Micro, &SampleParams::default());
    assert_eq!(found.iter().map(|f| f.version).collect::<Vec<_>>(), [version]);
}

/// Not a test of the code: how long a search of a whole photograph takes.
/// `cargo test --test imaging timing -- --ignored --nocapture`.
#[test]
#[ignore]
fn timing_of_a_whole_photo_search() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/photos/zxing_qrcode-4_32.png");
    let luma = load_photo(&path).unwrap();
    let params = SampleParams::default();
    for _ in 0..3 {
        let start = std::time::Instant::now();
        let found = find_symbols(&luma, Family::Qr, &params);
        println!("find_symbols: {:?}, {} found", start.elapsed(), found.len());
    }
    let start = std::time::Instant::now();
    let found = find_symbols(&luma, Family::Micro, &params);
    println!("find_symbols (Micro, none expected): {:?}, {} found", start.elapsed(), found.len());
}

/// The photo read through a detection as found, corners and anchors as
/// they are: what the app samples before any refinement.
fn read_as_found(image: &Luma, found: &qr_forensics::imaging::detect::Detection, grid: &ModuleGrid, params: &SampleParams) -> ModuleGrid {
    let func = FunctionMap::new(found.version);
    let map = found.projection().unwrap();
    let sampled = classify(&sample_modules(image, &map, func.width, func.height), params).to_grid(func.width, func.height);
    let mut out = ModuleGrid::new(grid.width(), grid.height());
    for y in 0..grid.height() {
        for x in 0..grid.width() {
            out.set(x, y, sampled.get(x, y));
        }
    }
    out
}

/// Not a test of the code: decode one photo of the corpus through its
/// sidecar outline, for looking into a failure. `QR_DIR` names the corpus
/// (`photos`, the default, or `blackbox`) and `QR_TURN` the quarter turns
/// clockwise to read the photo at (0 to 3, default 0), as the corpus tests
/// turn it.
/// `QR_PHOTO=cpp_micro_7 QR_DIR=photos QR_TURN=1 cargo test --test imaging one_photo -- --ignored --nocapture`.
#[test]
#[ignore]
fn one_photo_of_the_corpus() {
    let Some(name) = std::env::var_os("QR_PHOTO") else { return };
    let corpus = std::env::var("QR_DIR").unwrap_or_else(|_| "photos".to_owned());
    assert!(["photos", "blackbox"].contains(&corpus.as_str()), "QR_DIR is photos or blackbox, not {corpus:?}");
    let turns: u8 = std::env::var("QR_TURN").map_or(0, |t| t.parse().expect("QR_TURN is 0 to 3"));
    assert!(turns < 4, "QR_TURN is 0 to 3, not {turns}");
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join(&corpus);
    let name = name.to_string_lossy().into_owned();
    let path = PHOTO_EXTENSIONS.iter().map(|ext| dir.join(format!("{name}.{ext}"))).find(|p| p.exists()).expect("photo");
    let luma = load_photo(&path).unwrap();
    let sidecar = PhotoSidecar::parse(&std::fs::read_to_string(dir.join(format!("{name}.outline"))).unwrap()).unwrap();
    let expected = std::fs::read_to_string(dir.join(format!("{name}.txt"))).unwrap();
    let turned = rotate_luma(&luma, turns);
    let outline = rotate_outline(sidecar.outline, turns, luma.width, luma.height);
    let start = std::time::Instant::now();
    let result = decode_photo(&turned, sidecar.family, outline, &SampleParams::default());
    println!(
        "{corpus}/{name} at {}°: {:?}; the sidecar expects {}",
        degrees(turns),
        start.elapsed(),
        if sidecar.reads_at(turns) { "a read" } else { "a failure" }
    );
    match result {
        None => println!("nothing found"),
        Some(d) => {
            println!(
                "{} with {} anchors, turned {}, retry {:?}{}",
                d.detection.version,
                d.detection.anchors.len(),
                d.detection.turns,
                d.detection.retry,
                d.reinterpreted.map(|v| format!(", read as {v}")).unwrap_or_default()
            );
            println!("corners {:?}", d.detection.corners);
            for y in 0..d.grid.height() {
                let row: String = (0..d.grid.width())
                    .map(|x| match d.grid.get(x, y) {
                        Module::Dark => '#',
                        Module::Light => '.',
                        Module::Unknown => '?',
                    })
                    .collect();
                println!("{row}");
            }
            for (i, block) in d.report.blocks.iter().enumerate() {
                println!("block {i}: {:?}, {} codewords erased", block.outcome, block.erased_codewords);
            }
            println!("score {:?}, {}", d.report.score, if d.report.is_trustworthy() { "trustworthy" } else { "not trustworthy" });
            println!("text {:?} (expected {:?})", d.text(), expected);
        }
    }
}

/// A render of QR4 at `module_px` pixels a module, through a loose box,
/// read at its four turns: the pipeline must not care which way up the
/// camera was held, so each turn, through the outline turned with it,
/// finds the symbol that many quarter turns round, decodes it, and
/// samples the same grid, which the detection has already turned back.
fn assert_turns_read_the_same_grid(module_px: f64) {
    let (version, ec) = (SymbolVersion::Qr(4), EcLevel::M);
    let grid = symbol(version, ec);
    let truth = decode_plain(&grid, &layout(version, ec), 0).text.plain();
    let params = SampleParams::default();
    let side = 33.0 * module_px;
    let quad = [[100.0, 80.0], [100.0 + side, 80.0], [100.0 + side, 80.0 + side], [100.0, 80.0 + side]];
    let image = render(&grid, quad, (500, 400), |_, _, v| v);
    let outline = bounding_box(quad, 0.3);
    let base = decode_photo(&image, Family::Qr, outline, &params).unwrap_or_else(|| panic!("{module_px} px a module: nothing found unturned"));
    assert_eq!((base.detection.version, base.detection.turns), (version, 0), "{module_px} px a module");
    assert!(base.report.is_trustworthy(), "{module_px} px a module: {:?}", base.report.score);
    assert_eq!(base.text(), truth, "{module_px} px a module");
    for turns in 1..4u8 {
        let turned = rotate_luma(&image, turns);
        let turned_outline = rotate_outline(outline, turns, image.width, image.height);
        let d = decode_photo(&turned, Family::Qr, turned_outline, &params)
            .unwrap_or_else(|| panic!("{module_px} px a module, turned {}°: nothing found", degrees(turns)));
        assert_eq!((d.detection.version, d.detection.turns), (version, turns), "{module_px} px a module, turned {}°", degrees(turns));
        assert!(d.report.is_trustworthy(), "{module_px} px a module, turned {}°: {:?}", degrees(turns), d.report.score);
        assert_eq!(d.text(), truth, "{module_px} px a module, turned {}°", degrees(turns));
        // The modules alone: the reliabilities may differ in the last
        // place, as the sampling points land a rounding apart.
        assert!(d.grid.cells() == base.grid.cells(), "{module_px} px a module, turned {}°: a different grid", degrees(turns));
    }
}

#[test]
fn a_turned_photo_reads_the_same_grid() {
    assert_turns_read_the_same_grid(8.0);
}

#[test]
fn a_turned_photo_of_a_small_symbol_reads_the_same_grid() {
    assert_turns_read_the_same_grid(2.5);
}


/// Photograph `grid` with a light scratch `width` pixels wide through the
/// centre of every finder pattern, along the row and the column both.
fn render_cross_scratched(grid: &ModuleGrid, quad: [[f64; 2]; 4], size: (usize, usize), width: f64) -> Luma {
    let (gw, gh) = (grid.width() as f64, grid.height() as f64);
    let map = Homography::rect_to_quad(gw, gh, quad).unwrap();
    let centres = [[3.5, 3.5], [gw - 3.5, 3.5], [3.5, gh - 3.5]].map(|[x, y]| map.apply(x, y));
    let plain = render(grid, quad, size, |_, _, v| v);
    let data = plain
        .data
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let (x, y) = ((i % size.0) as f64 + 0.5, (i / size.0) as f64 + 0.5);
            let scratched = centres.iter().any(|c| ((x - c[0]).abs() < width / 2.0 || (y - c[1]).abs() < width / 2.0) && (x - c[0]).abs() < 45.0 && (y - c[1]).abs() < 45.0);
            if scratched { 0.92 } else { v }
        })
        .collect();
    Luma::new(size.0, size.1, data)
}

#[test]
fn a_finder_scratched_through_both_axes_is_found_by_its_region() {
    // A light scratch through the centre of a finder along its row and its
    // column leaves no line through the centre with the 1:1:3:1:1 runs the
    // scan looks for. The region of the ring survives, as BoofCV finds a
    // position pattern by its square, and the template fit reads it.
    let params = SampleParams::default();
    let (version, ec) = (SymbolVersion::Qr(2), EcLevel::M);
    let grid = symbol(version, ec);
    let image = render_cross_scratched(&grid, QUAD, (760, 640), 2.0);
    let found = detect_outline(&image, version, loosen(QUAD, 1.1, [5.0, -4.0]), &params).expect("the scratched symbol is found");
    assert_eq!(found.version, version);
    assert_eq!(found.count(AnchorKind::Finder), 3, "{:?}", found.anchors);
    let module_px = (QUAD[1][0] - QUAD[0][0]) / grid.width() as f64;
    let error = corner_error(found.corners, QUAD) / module_px;
    assert!(error < 0.5, "corners {error:.2} modules off");
}

/// `image` with Gaussian sensor noise of standard deviation `sigma` on
/// every pixel, clamped to the range of a photo.
fn add_noise(image: &Luma, sigma: f32, rng: &mut Rng) -> Luma {
    let mut uniform = || (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
    let mut data = Vec::with_capacity(image.data.len());
    for pair in image.data.chunks(2) {
        // Box–Muller: two uniforms make two independent normals.
        let (u1, u2) = (1.0 - uniform(), uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        let (s, c) = (std::f64::consts::TAU * u2).sin_cos();
        for (&v, n) in pair.iter().zip([r * c, r * s]) {
            data.push((v + sigma * n as f32).clamp(0.0, 1.0));
        }
    }
    Luma::new(image.width, image.height, data)
}

/// `image` saved as an 8-bit grey JPEG at `quality` and loaded back, with
/// the blocking and ringing that leaves.
fn jpeg_roundtrip(image: &Luma, quality: u8) -> Luma {
    let bytes: Vec<u8> = image.data.iter().map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8).collect();
    let mut file = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut file, quality)
        .encode(&bytes, image.width as u32, image.height as u32, image::ExtendedColorType::L8)
        .unwrap();
    let decoded = image::load_from_memory_with_format(&file, image::ImageFormat::Jpeg).unwrap().into_luma8();
    assert_eq!((decoded.width() as usize, decoded.height() as usize), (image.width, image.height));
    Luma::new(image.width, image.height, decoded.as_raw().iter().map(|&b| f32::from(b) / 255.0).collect())
}

/// Where `image`'s lens shows the point `[x, y]` of a flat scene under
/// Brown radial distortion `k1` about the centre: radii are measured in
/// half-diagonals, and the pixel at radius r shows the scene at
/// r (1 + k1 r²). Positive `k1` bows straight edges outward, as a barrel;
/// negative bows them inward, as a pincushion.
fn lens_position(size: (usize, usize), k1: f64, flat: [f64; 2]) -> [f64; 2] {
    let c = [size.0 as f64 / 2.0, size.1 as f64 / 2.0];
    let half_diagonal = (c[0] * c[0] + c[1] * c[1]).sqrt();
    let target = distance(flat, c) / half_diagonal;
    // r (1 + k1 r²) = target, by Newton's method.
    let mut r = target;
    for _ in 0..30 {
        r -= (r * (1.0 + k1 * r * r) - target) / (1.0 + 3.0 * k1 * r * r);
    }
    let scale = if target > 0.0 { r / target } else { 0.0 };
    [c[0] + (flat[0] - c[0]) * scale, c[1] + (flat[1] - c[1]) * scale]
}

/// `image` through a lens with Brown radial distortion `k1` (see
/// `lens_position`): every pixel resampled bilinearly from where the flat
/// scene lies, white where that falls outside the image.
fn lens_distort(image: &Luma, k1: f64) -> Luma {
    let c = [image.width as f64 / 2.0, image.height as f64 / 2.0];
    let half_diagonal = (c[0] * c[0] + c[1] * c[1]).sqrt();
    let mut data = Vec::with_capacity(image.data.len());
    for py in 0..image.height {
        for px in 0..image.width {
            let (x, y) = (px as f64 + 0.5, py as f64 + 0.5);
            let r2 = ((x - c[0]).powi(2) + (y - c[1]).powi(2)) / (half_diagonal * half_diagonal);
            let f = 1.0 + k1 * r2;
            data.push(image.sample(c[0] + (x - c[0]) * f, c[1] + (y - c[1]) * f).unwrap_or(1.0));
        }
    }
    Luma::new(image.width, image.height, data)
}

/// `image` smeared `length` pixels along `degrees` from the x axis, as a
/// camera moved during the exposure: every pixel the mean of the samples
/// along that stretch, centred on it.
fn motion_blur(image: &Luma, length: f64, degrees: f64) -> Luma {
    let (s, c) = degrees.to_radians().sin_cos();
    let taps = (length * 2.0).ceil().max(1.0) as usize;
    let mut data = Vec::with_capacity(image.data.len());
    for py in 0..image.height {
        for px in 0..image.width {
            let (x, y) = (px as f64 + 0.5, py as f64 + 0.5);
            let (mut sum, mut n) = (0.0, 0);
            for i in 0..taps {
                let t = ((i as f64 + 0.5) / taps as f64 - 0.5) * length;
                if let Some(v) = image.sample(x + t * c, y + t * s) {
                    sum += v;
                    n += 1;
                }
            }
            data.push(if n > 0 { sum / n as f32 } else { image.data[py * image.width + px] });
        }
    }
    Luma::new(image.width, image.height, data)
}

/// A quad the shape of `QUAD`, scaled so that `version` is photographed
/// at about `module_px` pixels a module, about `centre`.
fn quad_at(version: SymbolVersion, module_px: f64, centre: [f64; 2]) -> [[f64; 2]; 4] {
    let scale = module_px * version.width() as f64 / distance(QUAD[0], QUAD[1]);
    let c = QUAD.iter().fold([0.0, 0.0], |a, p| [a[0] + p[0] / 4.0, a[1] + p[1] / 4.0]);
    QUAD.map(|p| [centre[0] + (p[0] - c[0]) * scale, centre[1] + (p[1] - c[1]) * scale])
}

/// What the whole pipeline, as the app runs it from a loose outline, makes
/// of a photo of `grid` drawn on `quad`: the decode and the (wrong,
/// unknown) modules of the grid it sampled, when it found the symbol at
/// the right size.
fn read_degraded(image: &Luma, grid: &ModuleGrid, outline: [[f64; 2]; 4]) -> Option<(PhotoDecode, (usize, usize))> {
    let d = decode_photo(image, Family::Qr, outline, &SampleParams::default())?;
    let counts = ((d.grid.width(), d.grid.height()) == (grid.width(), grid.height())).then(|| tally(&d.grid, grid));
    Some((d, counts?))
}

/// One line on how a degraded photo of `grid` read, for the thresholds
/// below and the messages of the tests.
fn degradation_summary(result: &Option<(PhotoDecode, (usize, usize))>, truth: &str) -> String {
    match result {
        None => "nothing found".to_owned(),
        Some((d, (wrong, unknown))) => format!(
            "{} with {} anchors, {wrong} wrong {unknown} unknown, {}, text {}",
            d.detection.version,
            d.detection.anchors.len(),
            if d.report.is_trustworthy() { "trustworthy" } else { "not trustworthy" },
            if d.text() == truth { "right" } else { "wrong" }
        ),
    }
}

// The levels the degradation tests hold the pipeline to are about two
// thirds of where `degradation_thresholds` found it failing, so that they
// pass today and catch a pipeline that grows weaker.

#[test]
fn sensor_noise_is_averaged_away() {
    // Grain as a phone's sensor adds in poor light: at eight pixels a
    // module the sampler's points average most of it away. Every module
    // reads as printed to a third of the contrast; the symbol still
    // decodes at half, where the first modules go wrong and nothing is
    // found much beyond.
    let (version, ec) = (SymbolVersion::Qr(6), EcLevel::M);
    let grid = symbol(version, ec);
    let truth = decode_plain(&grid, &layout(version, ec), 0).text.plain();
    let quad = quad_at(version, 8.0, [380.0, 320.0]);
    let clean = render(&grid, quad, (760, 640), |_, _, v| v);
    for (sigma, seed, exact) in [(0.26, 11, true), (0.40, 12, false)] {
        let image = add_noise(&clean, sigma, &mut Rng(seed));
        let result = read_degraded(&image, &grid, loosen(quad, 1.1, [5.0, -4.0]));
        let summary = degradation_summary(&result, &truth);
        let (d, (wrong, unknown)) = result.unwrap_or_else(|| panic!("σ {sigma}: {summary}"));
        assert_eq!(d.detection.version, version, "σ {sigma}: {summary}");
        if exact {
            assert_eq!((wrong, unknown), (0, 0), "σ {sigma}: {summary}");
        }
        assert!(d.report.is_trustworthy(), "σ {sigma}: {summary}");
        assert_eq!(d.text(), truth, "σ {sigma}: {summary}");
    }
}

#[test]
fn jpeg_compression_is_read_through() {
    // A heavily compressed photo: the blocks and ringing of JPEG at a low
    // quality muddy every edge, and the modules still read as printed. At
    // eight pixels a module no quality the encoder offers, down to 1,
    // loses a module, so this is held well below any camera's.
    let (version, ec) = (SymbolVersion::Qr(6), EcLevel::M);
    let grid = symbol(version, ec);
    let truth = decode_plain(&grid, &layout(version, ec), 0).text.plain();
    let quad = quad_at(version, 8.0, [380.0, 320.0]);
    let image = jpeg_roundtrip(&render(&grid, quad, (760, 640), |_, _, v| v), 10);
    let result = read_degraded(&image, &grid, loosen(quad, 1.1, [5.0, -4.0]));
    let summary = degradation_summary(&result, &truth);
    let (d, (wrong, unknown)) = result.unwrap_or_else(|| panic!("{summary}"));
    assert_eq!(d.detection.version, version, "{summary}");
    assert_eq!((wrong, unknown), (0, 0), "{summary}");
    assert!(d.report.is_trustworthy(), "{summary}");
    assert_eq!(d.text(), truth, "{summary}");
}

#[test]
fn lens_distortion_is_followed_by_the_alignment_anchors() {
    // A wide lens bows the symbol's edges: a flat map through the corners
    // misses the middle of a large symbol, and the alignment patterns,
    // found where the lens put them, bend the map back. Past about a
    // tenth either way the detector loses most of the patterns, and the
    // middle of the symbol with them.
    let (version, ec) = (SymbolVersion::Qr(15), EcLevel::M);
    let grid = symbol(version, ec);
    let truth = decode_plain(&grid, &layout(version, ec), 0).text.plain();
    let size = (1000, 900);
    let quad = quad_at(version, 10.0, [500.0, 450.0]);
    let flat = render(&grid, quad, size, |_, _, v| v);
    for k1 in [-0.08, 0.08] {
        let image = lens_distort(&flat, k1);
        let bent = quad.map(|p| lens_position(size, k1, p));
        let result = read_degraded(&image, &grid, loosen(bent, 1.1, [5.0, -4.0]));
        let summary = degradation_summary(&result, &truth);
        let (d, (wrong, unknown)) = result.unwrap_or_else(|| panic!("k1 {k1}: {summary}"));
        assert_eq!(d.detection.version, version, "k1 {k1}: {summary}");
        assert!(d.detection.count(AnchorKind::Alignment) >= 8, "k1 {k1}: {summary}");
        assert_eq!(wrong, 0, "k1 {k1}: {summary}");
        assert!(unknown <= 3, "k1 {k1}: {summary}");
        assert!(d.report.is_trustworthy(), "k1 {k1}: {summary}");
        assert_eq!(d.text(), truth, "k1 {k1}: {summary}");
    }
}

#[test]
fn motion_blur_is_read_through() {
    // The camera moved a module during the exposure, across the grid at an
    // angle: every edge is a ramp a module long, and the modules still
    // read. Half as much again and the finder patterns are lost.
    let (version, ec) = (SymbolVersion::Qr(6), EcLevel::M);
    let grid = symbol(version, ec);
    let truth = decode_plain(&grid, &layout(version, ec), 0).text.plain();
    let quad = quad_at(version, 8.0, [380.0, 320.0]);
    let image = motion_blur(&render(&grid, quad, (760, 640), |_, _, v| v), 8.0, 20.0);
    let result = read_degraded(&image, &grid, loosen(quad, 1.1, [5.0, -4.0]));
    let summary = degradation_summary(&result, &truth);
    let (d, (wrong, unknown)) = result.unwrap_or_else(|| panic!("{summary}"));
    assert_eq!(d.detection.version, version, "{summary}");
    assert_eq!((wrong, unknown), (0, 0), "{summary}");
    assert!(d.report.is_trustworthy(), "{summary}");
    assert_eq!(d.text(), truth, "{summary}");
}

#[test]
fn a_phone_photo_is_read() {
    // All of a phone camera's faults at once, each mild: a little lens
    // distortion, sensor grain, and the JPEG the camera saves.
    let (version, ec) = (SymbolVersion::Qr(6), EcLevel::M);
    let grid = symbol(version, ec);
    let truth = decode_plain(&grid, &layout(version, ec), 0).text.plain();
    let size = (760, 640);
    let quad = quad_at(version, 8.0, [380.0, 320.0]);
    let (k1, sigma, quality) = (-0.06, 0.04, 40);
    let image = jpeg_roundtrip(&add_noise(&lens_distort(&render(&grid, quad, size, |_, _, v| v), k1), sigma, &mut Rng(13)), quality);
    let bent = quad.map(|p| lens_position(size, k1, p));
    let result = read_degraded(&image, &grid, loosen(bent, 1.1, [5.0, -4.0]));
    let summary = degradation_summary(&result, &truth);
    let (d, (wrong, unknown)) = result.unwrap_or_else(|| panic!("{summary}"));
    assert_eq!(d.detection.version, version, "{summary}");
    assert_eq!(wrong, 0, "{summary}");
    assert!(unknown <= 2, "{summary}");
    assert!(d.report.is_trustworthy(), "{summary}");
    assert_eq!(d.text(), truth, "{summary}");
}

/// Not a test of the code: how much of each degradation the pipeline
/// takes before a photo stops reading, for setting the levels the tests
/// above hold it to (about two thirds of where it fails). With `QR_DUMP`
/// naming a directory, every degraded photo is saved there as a PNG.
/// `cargo test --test imaging degradation_thresholds -- --ignored --nocapture`.
#[test]
#[ignore]
fn degradation_thresholds() {
    let dump = |name: &str, image: &Luma| {
        if let Some(dir) = std::env::var_os("QR_DUMP") {
            let bytes: Vec<u8> = image.data.iter().map(|v| (v * 255.0).round() as u8).collect();
            image::save_buffer(std::path::Path::new(&dir).join(format!("{name}.png")), &bytes, image.width as u32, image.height as u32, image::ExtendedColorType::L8).unwrap();
        }
    };
    let (version, ec) = (SymbolVersion::Qr(6), EcLevel::M);
    let grid = symbol(version, ec);
    let truth = decode_plain(&grid, &layout(version, ec), 0).text.plain();
    let size = (760, 640);
    let quad = quad_at(version, 8.0, [380.0, 320.0]);
    let clean = render(&grid, quad, size, |_, _, v| v);
    let outline = loosen(quad, 1.1, [5.0, -4.0]);
    for sigma in [0.06, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80] {
        let image = add_noise(&clean, sigma, &mut Rng(12));
        dump(&format!("noise_{sigma}"), &image);
        println!("noise σ {sigma}: {}", degradation_summary(&read_degraded(&image, &grid, outline), &truth));
    }
    for quality in [40, 25, 10, 5, 1] {
        let image = jpeg_roundtrip(&clean, quality);
        dump(&format!("jpeg_{quality}"), &image);
        println!("jpeg quality {quality}: {}", degradation_summary(&read_degraded(&image, &grid, outline), &truth));
    }
    for length in [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0] {
        let image = motion_blur(&clean, length, 20.0);
        dump(&format!("blur_{length}"), &image);
        println!("motion blur {length} px: {}", degradation_summary(&read_degraded(&image, &grid, outline), &truth));
    }
    let (version, ec) = (SymbolVersion::Qr(15), EcLevel::M);
    let grid = symbol(version, ec);
    let truth = decode_plain(&grid, &layout(version, ec), 0).text.plain();
    let size = (1000, 900);
    let quad = quad_at(version, 10.0, [500.0, 450.0]);
    let flat = render(&grid, quad, size, |_, _, v| v);
    for k1 in [-0.06, -0.08, -0.09, -0.10, -0.11, -0.12, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.24] {
        let image = lens_distort(&flat, k1);
        dump(&format!("lens_{k1}"), &image);
        let bent = quad.map(|p| lens_position(size, k1, p));
        let result = read_degraded(&image, &grid, loosen(bent, 1.1, [5.0, -4.0]));
        let anchors = result.as_ref().map_or(0, |(d, _)| d.detection.count(AnchorKind::Alignment));
        println!("lens k1 {k1}: {} ({anchors} alignment)", degradation_summary(&result, &truth));
    }
}

#[test]
fn a_search_or_a_read_called_off_comes_back_at_once_with_nothing() {
    use qr_forensics::work::Cancel;
    // A photo a search finds its symbol in, and a read reads.
    let version = SymbolVersion::Qr(3);
    let image = render(&symbol(version, EcLevel::M), QUAD, (800, 640), |_, _, v| v);
    let params = SampleParams::default();
    let whole = [[0.0, 0.0], [800.0, 0.0], [800.0, 640.0], [0.0, 640.0]];
    let request = Cancel::new();
    // Answering to a request no one has made, the work is done as ever.
    let found = request.scope(|| find_symbols(&image, Family::Qr, &params));
    assert_eq!(found.iter().map(|found| found.version).collect::<Vec<_>>(), [version]);
    let read = request.scope(|| read_outline(&image, SymbolVersion::Qr(2), whole, &params, &DecodeOptions::default()));
    assert!(read.is_some_and(|read| read.is_trustworthy()));
    // Called off before it starts, it climbs no rung and fits nothing.
    request.cancel();
    let started = std::time::Instant::now();
    assert!(request.scope(|| find_symbols(&image, Family::Qr, &params)).is_empty());
    assert!(request.scope(|| find_symbols(&image, Family::Micro, &params)).is_empty());
    let read = request.scope(|| read_outline(&image, SymbolVersion::Qr(2), whole, &params, &DecodeOptions::default()));
    assert!(read.is_none_or(|read| !read.is_trustworthy()));
    assert!(started.elapsed().as_secs() < 5, "work called off went on for {:?}", started.elapsed());
    // Outside the request's scope, work is its own again.
    assert!(!find_symbols(&image, Family::Qr, &params).is_empty());
}
