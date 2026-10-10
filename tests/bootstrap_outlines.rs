//! A development aid, not a test of the code: write the outline sidecars
//! for a directory of photos. Runs only when `QR_BOOTSTRAP_DIR` names the
//! directory, and does nothing otherwise.
//!
//! For each image with an expected-text file beside it, a few loose
//! outlines are tried: around what the `rxing` crate reports, through its
//! port of ZXing's QR reader and of zxing-cpp's, which also knows Micro QR
//! and rMQR; around what the app's own search of the whole photo finds;
//! and boxes of several sizes about the centre of the frame. The first
//! outline the pipeline reads the expected text through becomes the
//! sidecar's; when none does, the first outline tried is kept, so that the
//! photo stays in the corpus and a change that reads it is noticed.
//!
//! The photo is then turned 90, 180 and 270 degrees and read through the
//! same outline turned with it, as ZXing's blackbox suite does, and the
//! sidecar records the turns that read and how far the others got. Its
//! status follows: `pass` when all four read, `partial` when some do,
//! `known-failure` when none does.
//!
//! Each photo costs up to a few seconds an outline, and the corpora run to
//! hundreds of photos, so run it in release mode, over a copy of the
//! corpus directory, and copy back the sidecars it rewrites:
//!
//! ```text
//! QR_BOOTSTRAP_DIR=path/to/copy cargo test --release --test bootstrap_outlines -- --nocapture
//! ```
mod common;

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use common::*;
use qr_forensics::imaging::Luma;
use qr_forensics::imaging::detect::find_symbols;
use qr_forensics::imaging::sample::SampleParams;
use qr_forensics::symbol::Family;
use rxing::{BarcodeFormat, Reader};

/// The points rxing reports for the symbol: through its port of ZXing's
/// Java QR reader, three finder centres; through its port of zxing-cpp's
/// reader (`cpp`), the symbol's four corners, for any of the three
/// families.
fn rxing_points(luma: &Luma, family: Family, cpp: bool) -> Option<Vec<[f64; 2]>> {
    let bytes: Vec<u8> = luma
        .data
        .iter()
        .map(|v| (v * 255.0).round() as u8)
        .collect();
    let source = rxing::Luma8Source::new(bytes, luma.width as u32, luma.height as u32).ok()?;
    let mut bitmap = rxing::BinaryBitmap::new(rxing::common::HybridBinarizer::new(source));
    let format = match family {
        Family::Qr | Family::Model1 => BarcodeFormat::QR_CODE,
        Family::Micro => BarcodeFormat::MICRO_QR_CODE,
        Family::Rmqr => BarcodeFormat::RECTANGULAR_MICRO_QR_CODE,
    };
    let hints = rxing::DecodeHints {
        TryHarder: Some(true),
        PossibleFormats: Some(HashSet::from([format])),
        ..Default::default()
    };
    let result = if cpp {
        let mut reader = rxing::qrcode::cpp_port::QrReader;
        reader.decode_with_hints(&mut bitmap, &hints).ok()?
    } else {
        rxing::qrcode::QRCodeReader::new()
            .decode_with_hints(&mut bitmap, &hints)
            .ok()?
    };
    Some(
        result
            .getPoints()
            .iter()
            .map(|p| [p.x as f64, p.y as f64])
            .collect(),
    )
}

/// A loose outline around the points reported. From ZXing's reader, the
/// finder centres bottom-left, top-left, top-right (a fourth point, the
/// alignment pattern, is not needed), extended outward by a fifth of the
/// finder spacing; from zxing-cpp's (`corners`), the symbol's four
/// corners, put in order and widened about their centre by the same share.
fn outline_from_points(points: &[[f64; 2]], corners: bool) -> Option<[[f64; 2]; 4]> {
    let a = 0.2;
    if corners {
        return (points.len() >= 4).then(|| widened(ordered_quad(&points[..4]), a));
    }
    if points.len() < 3 {
        return None;
    }
    let (bl, tl, tr) = (points[0], points[1], points[2]);
    let u = [tr[0] - tl[0], tr[1] - tl[1]];
    let v = [bl[0] - tl[0], bl[1] - tl[1]];
    Some([
        [tl[0] - a * (u[0] + v[0]), tl[1] - a * (u[1] + v[1])],
        [tr[0] + a * (u[0] - v[0]), tr[1] + a * (u[1] - v[1])],
        [
            tr[0] + v[0] + a * (u[0] + v[0]),
            tr[1] + v[1] + a * (u[1] + v[1]),
        ],
        [bl[0] + a * (v[0] - u[0]), bl[1] + a * (v[1] - u[1])],
    ])
}

/// Four points as top-left, top-right, bottom-right, bottom-left on screen:
/// clockwise about their centre from the one nearest the top left.
fn ordered_quad(points: &[[f64; 2]]) -> [[f64; 2]; 4] {
    let centre = quad_centre(points);
    let mut ordered: Vec<[f64; 2]> = points.to_vec();
    ordered.sort_by(|a, b| {
        (a[1] - centre[1])
            .atan2(a[0] - centre[0])
            .total_cmp(&(b[1] - centre[1]).atan2(b[0] - centre[0]))
    });
    let start = (0..4)
        .min_by(|&i, &j| {
            (ordered[i][0] + ordered[i][1]).total_cmp(&(ordered[j][0] + ordered[j][1]))
        })
        .unwrap();
    ordered.rotate_left(start);
    [ordered[0], ordered[1], ordered[2], ordered[3]]
}

fn quad_centre(points: &[[f64; 2]]) -> [f64; 2] {
    let n = points.len() as f64;
    [
        points.iter().map(|p| p[0]).sum::<f64>() / n,
        points.iter().map(|p| p[1]).sum::<f64>() / n,
    ]
}

/// The quadrilateral pushed outward from its centre by `share` of itself.
fn widened(quad: [[f64; 2]; 4], share: f64) -> [[f64; 2]; 4] {
    let centre = quad_centre(&quad);
    quad.map(|p| {
        [
            centre[0] + (p[0] - centre[0]) * (1.0 + share),
            centre[1] + (p[1] - centre[1]) * (1.0 + share),
        ]
    })
}

#[test]
fn bootstrap_outlines() {
    let Some(dir) = std::env::var_os("QR_BOOTSTRAP_DIR") else {
        return;
    };
    let dir = Path::new(&dir);
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| is_photo_file(p))
        .collect();
    entries.sort();
    let params = SampleParams::default();
    // The running summary: how many photos read at each turn, and how many
    // sidecars of each status have been written.
    let total = entries.len();
    let mut reads = [0usize; 4];
    let mut statuses: BTreeMap<&'static str, usize> = BTreeMap::new();
    for (index, path) in entries.iter().enumerate() {
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        // Byte for byte: a message may end in a line break.
        let expected = std::fs::read_to_string(dir.join(format!("{stem}.txt"))).ok();
        let Some(expected) = expected else {
            println!("{stem}: no expected text");
            continue;
        };
        let luma = match load_photo(path) {
            Ok(l) => l,
            Err(e) => {
                println!("{stem}: {e}");
                continue;
            }
        };
        let family = if stem.contains("micro") {
            Family::Micro
        } else if stem.contains("rmqr") {
            Family::Rmqr
        } else if stem.contains("model-1") || stem.contains("model1") {
            Family::Model1
        } else {
            Family::Qr
        };
        let (w, h) = (luma.width as f64, luma.height as f64);
        type Outlines = Vec<(String, [[f64; 2]; 4])>;
        // Outlines are tried in stages, the dearer only when the cheaper
        // fail: rxing's readers, then the app's own search of the whole
        // photo, then boxes about the centre of the frame.
        let rxing_stage = || -> Outlines {
            let mut out = Outlines::new();
            if family.has_three_finders()
                && let Some(outline) =
                    rxing_points(&luma, family, false).and_then(|p| outline_from_points(&p, false))
            {
                out.push(("rxing".to_owned(), outline));
            }
            if let Some(outline) =
                rxing_points(&luma, family, true).and_then(|p| outline_from_points(&p, true))
            {
                out.push(("rxing-cpp".to_owned(), outline));
            }
            out
        };
        // What the app's own search of the whole photo finds, outlined
        // loosely as a hand would draw around it.
        let found_stage = || -> Outlines {
            find_symbols(&luma, family, &params)
                .iter()
                .take(3)
                .enumerate()
                .map(|(i, found)| (format!("found{}", i + 1), widened(found.corners, 0.2)))
                .collect()
        };
        let boxes_stage = || -> Outlines {
            let inset = 0.04;
            let mut out = Outlines::new();
            out.push((
                "whole".to_owned(),
                [
                    [w * inset, h * inset],
                    [w * (1.0 - inset), h * inset],
                    [w * (1.0 - inset), h * (1.0 - inset)],
                    [w * inset, h * (1.0 - inset)],
                ],
            ));
            // Boxes of several sizes about the centre, for a symbol that
            // does not fill the frame.
            let side = w.min(h);
            for (name, scale) in [
                ("box90", 0.9),
                ("box70", 0.7),
                ("box55", 0.55),
                ("box40", 0.4),
                ("box30", 0.3),
            ] {
                let s = side * scale;
                let (x0, y0) = ((w - s) / 2.0, (h - s) / 2.0);
                out.push((
                    name.to_owned(),
                    [[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s]],
                ));
            }
            // And, for a wide symbol, boxes of the frame's width and a few heights.
            for (name, frac) in [("wide50", 0.5), ("wide30", 0.3)] {
                let hh = h * frac;
                let y0 = (h - hh) / 2.0;
                out.push((
                    name.to_owned(),
                    [
                        [w * inset, y0],
                        [w * (1.0 - inset), y0],
                        [w * (1.0 - inset), y0 + hh],
                        [w * inset, y0 + hh],
                    ],
                ));
            }
            out
        };
        let stages: [&dyn Fn() -> Outlines; 3] = [&rxing_stage, &found_stage, &boxes_stage];
        // The ladder: the first outline that reads is the sidecar's; failing
        // all, the first tried, with how far the photo got through it.
        let mut first: Option<([[f64; 2]; 4], Outcome)> = None;
        let mut chosen: Option<([[f64; 2]; 4], Outcome)> = None;
        'stages: for stage in stages {
            for (source, outline) in stage() {
                let attempt = try_decode_photo(&luma, family, outline, &params);
                let outcome = Outcome::of_attempt(&attempt, &expected);
                let detail = match &attempt {
                    Ok(Some(d)) => format!(
                        " ({} anchors, got {:?})",
                        d.detection.anchors.len(),
                        d.text().chars().take(40).collect::<String>()
                    ),
                    _ => String::new(),
                };
                println!("{stem} [{source}]: {}{detail}", outcome.describe());
                if first.is_none() {
                    first = Some((outline, outcome.clone()));
                }
                if outcome.is_read() {
                    chosen = Some((outline, outcome));
                    break 'stages;
                }
            }
        }
        let (outline, outcome_as_is) = chosen.or(first).expect("the boxes always give an outline");
        // The other three turns go through the same outline turned with the
        // photo, whether or not the photo read as it is: a known failure may
        // still read the other way up, and that is worth knowing.
        let mut rotations = Vec::new();
        let mut failures = Vec::new();
        for turns in 0..4u8 {
            let outcome = if turns == 0 {
                outcome_as_is.clone()
            } else {
                let turned = rotate_luma(&luma, turns);
                let turned_outline = rotate_outline(outline, turns, luma.width, luma.height);
                Outcome::of_attempt(
                    &try_decode_photo(&turned, family, turned_outline, &params),
                    &expected,
                )
            };
            if outcome.is_read() {
                rotations.push(turns);
                reads[usize::from(turns)] += 1;
            } else {
                failures.push((turns, outcome.describe()));
            }
        }
        let sidecar = PhotoSidecar {
            family,
            rotations,
            failures,
            outline,
        };
        std::fs::write(dir.join(format!("{stem}.outline")), sidecar.to_text()).unwrap();
        *statuses.entry(sidecar.status()).or_default() += 1;
        let turns: Vec<String> = (0..4u8)
            .map(|t| {
                format!(
                    "{}:{}",
                    degrees(t),
                    if sidecar.reads_at(t) { "read" } else { "-" }
                )
            })
            .collect();
        println!(
            "[{}/{total}] {stem}: {} ({}) | so far {} read at 0/90/180/270, {:?}",
            index + 1,
            sidecar.status(),
            turns.join(" "),
            reads.map(|n| n.to_string()).join("/"),
            statuses
        );
    }
    println!(
        "{total} photos: {:?}; read at 0/90/180/270: {}",
        statuses,
        reads.map(|n| n.to_string()).join("/")
    );
}
