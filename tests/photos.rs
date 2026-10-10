//! Real photographs: the pipeline from a loose outline to text, on camera
//! shots from the ZXing and zxing-cpp test suites. The synthetic renders in
//! `imaging.rs` have no blur, noise, lens distortion or uneven paper; these
//! have all of them.
//!
//! Each photo in `tests/photos` and `tests/blackbox` comes with the text
//! the suite says it holds and a sidecar outline: the symbol's family, a
//! loose quadrilateral around it as a hand would draw one, and the quarter
//! turns of the photo at which the pipeline is expected to read it. Photos
//! it cannot yet read are kept as known failures, so that a change which
//! reads them is noticed as much as one which loses a photo it could read.
//!
//! Both tests run in the manner of ZXing's `AbstractBlackBoxTestCase`:
//! every photo turned 0, 90, 180 and 270 degrees, a table of reads,
//! misreads, failures and not-founds for each set and turn, and a misread
//! counted apart from and as worse than a failure. The fast test reads the
//! smaller corpus; the ignored `blackbox_corpus_reads_as_its_sidecars_say`
//! reads both corpora, takes minutes, and so runs by hand, and in release
//! mode:
//!
//! ```text
//! cargo test --release --test photos -- --ignored --nocapture
//! ```
//!
//! A turn that reads without its sidecar saying so is reported as a
//! promotion: re-run `tests/bootstrap_outlines.rs` over the corpus and
//! copy the sidecars it rewrites.

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use common::*;
use qr_forensics::imaging::Luma;
use qr_forensics::imaging::sample::SampleParams;
use qr_forensics::symbol::Family;

struct Photo {
    name: String,
    /// The source set, for the summary: `zxing_qrcode-3`, `cpp_micro`.
    set: String,
    path: PathBuf,
    expected: String,
    sidecar: PhotoSidecar,
}

/// The set a photo came from: its name up to the second underscore, so
/// that a stem with underscores of its own, like `R7x43-H_inverted`, stays
/// in its set; a name with fewer is a set of its own.
fn set_of(name: &str) -> String {
    match name.match_indices('_').nth(1) {
        Some((at, _)) => name[..at].to_owned(),
        None => name.to_owned(),
    }
}

/// The photos of one corpus directory, in name order.
fn photos_in(dir: &Path) -> Vec<Photo> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if !is_photo_file(&path) {
            continue;
        }
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        // The file holds the message byte for byte: a message may end in a
        // line break, as one in the ZXing suite does, and trimming it
        // would fail a decode that is exactly right.
        let expected = std::fs::read_to_string(dir.join(format!("{name}.txt")))
            .unwrap_or_else(|e| panic!("{name}: expected text: {e}"));
        let sidecar = std::fs::read_to_string(dir.join(format!("{name}.outline")))
            .map_err(|e| e.to_string())
            .and_then(|t| PhotoSidecar::parse(&t))
            .unwrap_or_else(|e| panic!("{name}: outline sidecar: {e}"));
        out.push(Photo {
            set: set_of(&name),
            name,
            path,
            expected,
            sidecar,
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

fn corpus(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(name)
}

/// The photo turned clockwise by `turns` quarter turns, decoded through
/// its outline turned the same way.
fn decode_turned(
    luma: &Luma,
    photo: &Photo,
    turns: u8,
    params: &SampleParams,
) -> (Outcome, String) {
    let turned = rotate_luma(luma, turns);
    let outline = rotate_outline(photo.sidecar.outline, turns, luma.width, luma.height);
    let attempt = try_decode_photo(&turned, photo.sidecar.family, outline, params);
    let outcome = Outcome::of_attempt(&attempt, &photo.expected);
    let detail = match &attempt {
        Ok(Some(d)) => format!(
            "{} ({} with {} anchors{}, {:?}: {:?})",
            outcome.describe(),
            version_label(d.detection.version),
            d.detection.anchors.len(),
            d.reinterpreted
                .map(|v| format!(", read as {}", version_label(v)))
                .unwrap_or_default(),
            d.report.score,
            d.text().chars().take(60).collect::<String>()
        ),
        _ => outcome.describe(),
    };
    (outcome, detail)
}

/// Every photo decoded at the turns `turns_of` gives it, one photo's turns
/// on one thread and as many threads as the machine has cores. Decoding is
/// pure, so the threads share nothing but the queue of photos; the results
/// come back in the photos' own order, so what is printed and found is
/// what a sequential run prints and finds.
fn decode_all(
    photos: &[Photo],
    turns_of: impl Fn(usize) -> Vec<u8> + Sync,
    params: &SampleParams,
) -> Vec<Vec<(u8, Outcome, String)>> {
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(photos.len().max(1));
    let next = AtomicUsize::new(0);
    let mut results: Vec<Option<Vec<(u8, Outcome, String)>>> =
        (0..photos.len()).map(|_| None).collect();
    let done = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(photo) = photos.get(index) else {
                            break;
                        };
                        let luma = load_photo(&photo.path).unwrap();
                        let turned: Vec<_> = turns_of(index)
                            .into_iter()
                            .map(|turns| {
                                let (outcome, detail) = decode_turned(&luma, photo, turns, params);
                                (turns, outcome, detail)
                            })
                            .collect();
                        mine.push((index, turned));
                    }
                    mine
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|w| w.join().unwrap())
            .collect::<Vec<_>>()
    });
    for (index, turned) in done {
        results[index] = Some(turned);
    }
    results
        .into_iter()
        .map(|r| r.expect("every photo was decoded"))
        .collect()
}

/// Read `photos` at the turns `turns_of` gives each, print a line per
/// photo and the summary table, and gather what the sidecars did not
/// expect.
fn read_corpus(
    photos: &[Photo],
    turns_of: impl Fn(usize) -> Vec<u8> + Sync,
) -> (Summary, Findings) {
    let params = SampleParams::default();
    let mut summary = Summary::default();
    let mut findings = Findings::default();
    for (photo, turned) in photos.iter().zip(decode_all(photos, turns_of, &params)) {
        let mut line = format!("{:<40}", photo.name);
        for (turns, outcome, detail) in &turned {
            let mark = match (outcome, photo.sidecar.reads_at(*turns)) {
                (Outcome::Read, true) => "read",
                (Outcome::Read, false) => "READ+",
                (Outcome::Misread, true) => "MISREAD",
                (Outcome::Misread, false) => "misread",
                (Outcome::Failed(_), true) => "FAILED",
                (Outcome::Failed(_), false) => "failed",
                (Outcome::NotFound, true) => "NONE",
                (Outcome::NotFound, false) => "none",
            };
            line.push_str(&format!(" {:>3}:{mark:<8}", degrees(*turns)));
            summary.add(&photo.set, *turns, outcome);
            findings.note(photo, *turns, outcome, detail);
        }
        println!("{line}");
    }
    println!();
    summary.print();
    (summary, findings)
}

/// Counts of outcomes, as ZXing's blackbox summary keeps them.
#[derive(Clone, Copy, Default)]
struct Tally {
    read: usize,
    misread: usize,
    failed: usize,
    not_found: usize,
}

impl Tally {
    fn add(&mut self, outcome: &Outcome) {
        match outcome {
            Outcome::Read => self.read += 1,
            Outcome::Misread => self.misread += 1,
            Outcome::Failed(_) => self.failed += 1,
            Outcome::NotFound => self.not_found += 1,
        }
    }

    fn total(&self) -> usize {
        self.read + self.misread + self.failed + self.not_found
    }

    fn merge(&mut self, other: &Tally) {
        self.read += other.read;
        self.misread += other.misread;
        self.failed += other.failed;
        self.not_found += other.not_found;
    }
}

/// Outcomes by set and turn, printed as a table in the manner of ZXing's
/// blackbox runner, with the turns a set was not tried at left out.
#[derive(Default)]
struct Summary {
    sets: BTreeMap<String, [Tally; 4]>,
}

impl Summary {
    fn add(&mut self, set: &str, turns: u8, outcome: &Outcome) {
        self.sets.entry(set.to_owned()).or_default()[usize::from(turns % 4)].add(outcome);
    }

    fn print(&self) {
        let row = |name: &str, angle: &str, t: &Tally| {
            println!(
                "{name:<24} {angle:>5} {:>6} {:>8} {:>7} {:>10} {:>6}",
                t.read,
                t.misread,
                t.failed,
                t.not_found,
                t.total()
            )
        };
        println!(
            "{:<24} {:>5} {:>6} {:>8} {:>7} {:>10} {:>6}",
            "set", "turn", "read", "misread", "failed", "not-found", "total"
        );
        let mut all = [Tally::default(); 4];
        for (set, tallies) in &self.sets {
            for (turns, tally) in tallies.iter().enumerate() {
                if tally.total() == 0 {
                    continue;
                }
                row(set, &degrees(turns as u8).to_string(), tally);
                all[turns].merge(tally);
            }
        }
        let mut grand = Tally::default();
        for (turns, tally) in all.iter().enumerate() {
            if tally.total() == 0 {
                continue;
            }
            row("all", &degrees(turns as u8).to_string(), tally);
            grand.merge(tally);
        }
        row("all", "all", &grand);
    }
}

/// What the run found that the sidecars did not say: a turn expected to
/// read that does not, a wrong text where a right one was expected, and a
/// turn that reads without the sidecar expecting it.
#[derive(Default)]
struct Findings {
    regressions: Vec<String>,
    misreads: Vec<String>,
    promotions: Vec<String>,
}

impl Findings {
    fn note(&mut self, photo: &Photo, turns: u8, outcome: &Outcome, detail: &str) {
        let line = format!("{} at {}: {detail}", photo.name, degrees(turns));
        match (photo.sidecar.reads_at(turns), outcome) {
            (true, Outcome::Read) | (false, Outcome::Failed(_) | Outcome::NotFound) => {}
            (true, Outcome::Misread) => self.misreads.push(line),
            (true, _) => self.regressions.push(line),
            (false, Outcome::Read) => self.promotions.push(line),
            // A wrong text at a turn the sidecar does not expect to read
            // is a known failure, which the bootstrap records as such.
            (false, Outcome::Misread) => {}
        }
    }

    fn assert_none(&self) {
        assert!(
            self.regressions.is_empty(),
            "photos that no longer read:\n{}",
            self.regressions.join("\n")
        );
        assert!(
            self.misreads.is_empty(),
            "photos read as the wrong text:\n{}",
            self.misreads.join("\n")
        );
        assert!(
            self.promotions.is_empty(),
            "known failures that now read, so their sidecars should say so (re-run \
             tests/bootstrap_outlines.rs over the corpus and copy the sidecars):\n{}",
            self.promotions.join("\n")
        );
    }
}

/// Every photo of the smaller corpus at all four turns.
#[test]
fn every_photo_in_the_corpus_reads_as_its_sidecar_says() {
    let photos = photos_in(&corpus("photos"));
    assert!(photos.len() >= 20, "only {} photos", photos.len());
    let (summary, findings) = read_corpus(&photos, |_| vec![0, 1, 2, 3]);
    findings.assert_none();
    let passed = summary.sets.values().map(|t| t[0].read).sum::<usize>();
    assert!(passed >= 15, "{passed} photos read of {}", photos.len());
}

/// Every photo of both corpora at all four turns. Ignored because it takes
/// minutes; see the file header for how to run it.
#[test]
#[ignore]
fn blackbox_corpus_reads_as_its_sidecars_say() {
    let mut photos = photos_in(&corpus("photos"));
    photos.extend(photos_in(&corpus("blackbox")));
    assert!(photos.len() >= 100, "only {} photos", photos.len());
    let (_, findings) = read_corpus(&photos, |_| vec![0, 1, 2, 3]);
    findings.assert_none();
}

#[test]
fn a_luma_turns_clockwise_pixel_for_pixel() {
    // Two columns, three rows.
    let image = Luma::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let once = rotate_luma(&image, 1);
    assert_eq!((once.width, once.height), (3, 2));
    // The left column, read upward, is now the top row.
    assert_eq!(once.data, vec![5.0, 3.0, 1.0, 6.0, 4.0, 2.0]);
    let twice = rotate_luma(&image, 2);
    assert_eq!((twice.width, twice.height), (2, 3));
    assert_eq!(twice.data, vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    assert_eq!(
        rotate_luma(&once, 3),
        image,
        "four quarter turns come back round"
    );
    assert_eq!(rotate_luma(&image, 4), image);
    assert_eq!(rotate_luma(&image, 0), image);
}

#[test]
fn an_outline_turns_with_its_photo() {
    // The frame of a 2 by 3 image, drawn from the top left.
    let frame = [[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]];
    let turned = rotate_outline(frame, 1, 2, 3);
    assert_eq!(
        turned,
        [[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]],
        "the frame of the 3 by 2 image, top-left first"
    );
    assert_eq!(
        rotate_outline(frame, 2, 2, 3),
        frame,
        "a half turn of the frame is the frame"
    );
    // A quad that is not the frame: the point nearest the top left of the
    // turned image comes first, and the corners keep their clockwise run.
    let quad = [[1.0, 0.5], [1.8, 1.0], [1.5, 2.5], [0.2, 2.0]];
    let once = rotate_outline(quad, 1, 2, 3);
    assert_eq!(once, [[1.0, 0.2], [2.5, 1.0], [2.0, 1.8], [0.5, 1.5]]);
    assert_eq!(rotate_outline(quad, 4, 2, 3), quad);
    // Turning the image again is turning the outline again, with the
    // turned image's own dimensions.
    assert_eq!(rotate_outline(once, 1, 3, 2), rotate_outline(quad, 2, 2, 3));
}

#[test]
fn sidecars_keep_their_meaning_with_and_without_rotations() {
    let old_pass = "; family: qr\n; status: pass\n0 0\n10 0\n10 10\n0 10\n";
    let sidecar = PhotoSidecar::parse(old_pass).unwrap();
    assert_eq!(
        sidecar.rotations,
        vec![0],
        "an old pass is a pass of the photo as it is"
    );
    assert!(sidecar.expected_to_pass());
    assert_eq!(sidecar.status(), "partial");
    let old_failure = "; family: micro\n; status: known-failure\n0 0\n10 0\n10 10\n0 10\n";
    let sidecar = PhotoSidecar::parse(old_failure).unwrap();
    assert!(sidecar.rotations.is_empty());
    assert!(!sidecar.expected_to_pass());
    assert_eq!(sidecar.status(), "known-failure");

    let partial = PhotoSidecar {
        family: Family::Rmqr,
        rotations: vec![0, 2],
        failures: vec![
            (1, "not found".to_owned()),
            (3, "V3, 2 of 4 blocks solved, text differs".to_owned()),
        ],
        outline: [[0.5, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
    };
    let text = partial.to_text();
    assert_eq!(
        text,
        "; family: rmqr\n; status: partial\n; rotations: 0 180\n; failure: 90 not found\n\
         ; failure: 270 V3, 2 of 4 blocks solved, text differs\n0.5 0\n10 0\n10 10\n0 10\n"
    );
    assert_eq!(PhotoSidecar::parse(&text).unwrap(), partial);
    assert!(partial.reads_at(2) && !partial.reads_at(1));
    // Unknown lines are for other readers; a bad angle is a mistake.
    assert!(PhotoSidecar::parse("; family: qr\n; note: hello\n0 0\n1 0\n1 1\n0 1\n").is_ok());
    assert!(PhotoSidecar::parse("; family: qr\n; rotations: 45\n0 0\n1 0\n1 1\n0 1\n").is_err());
    let full = PhotoSidecar {
        rotations: vec![0, 1, 2, 3],
        failures: Vec::new(),
        ..partial
    };
    assert_eq!(full.status(), "pass");
    assert!(full.to_text().contains("; rotations: 0 90 180 270\n"));
}
