//! The search of a whole photo, with no outline to say where a symbol is:
//! the command line reader run over OpenCV's own QR Code test images, as
//! `qr-forensics decode <image>` reads one. The photos of
//! `tests/photos` and `tests/blackbox` are each read through an outline
//! drawn round the symbol, which is the window's Find in outline; nothing
//! there asks whether the symbol would have been found in the photo
//! around it, and a search that lost every symbol seen at an angle read
//! that corpus as well as ever. These images come with what each holds
//! and no outline, and several hold more symbols than one.
//!
//! Each image in `tests/opencv` has a sidecar, `<name>.search`, that lists
//! the messages the image holds, one to a line, as `--escaped` prints
//! them, and says how many of them the search reads when that is not all
//! of them: see `tests/opencv/README.md`. An image the search cannot yet
//! read whole is kept, with what it does read, so that a change which
//! reads more is noticed as much as one which reads less; and a message
//! read that the image does not hold fails the test whatever the sidecar
//! says, since a misread is worse than a miss.
//!
//! The quick test searches the images whose sidecars are marked `quick`,
//! a few seconds each; the ignored one searches them all, takes a few
//! minutes, and so runs by hand, in release mode:
//!
//! ```text
//! cargo test --release --test search -- --ignored --nocapture
//! ```
//!
//! With `QR_SEARCH_BOOTSTRAP` set, the ignored test writes into each
//! sidecar how many messages the search reads now, in place of failing
//! where that has changed.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

/// One image and what its sidecar says of it.
struct Case {
    name: String,
    image: PathBuf,
    sidecar: PathBuf,
    /// Searched by the quick test.
    quick: bool,
    /// The messages the image holds, as `--escaped` prints them.
    messages: Vec<String>,
    /// How many of them the search is expected to read.
    reads: usize,
}

/// What a search of one image came to.
struct Searched {
    /// Messages read, and to be trusted, that the image holds.
    matched: usize,
    /// Messages read, and to be trusted, that it does not hold.
    misread: Vec<String>,
    /// Every line the reader printed, for the report.
    lines: Vec<String>,
}

fn is_image(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("png" | "jpg" | "jpeg" | "webp")
    )
}

/// The images of the suite, in name order.
fn cases() -> Vec<Case> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/opencv");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap() {
        let image = entry.unwrap().path();
        if !is_image(&image) {
            continue;
        }
        let name = image.file_stem().unwrap().to_string_lossy().into_owned();
        let sidecar = dir.join(format!("{name}.search"));
        let text = std::fs::read_to_string(&sidecar).unwrap_or_else(|e| panic!("{name}: sidecar: {e}"));
        let (mut quick, mut reads, mut messages) = (false, None, Vec::new());
        for line in text.lines() {
            match line.strip_prefix(';').map(str::trim) {
                Some("quick") => quick = true,
                Some(header) => {
                    if let Some(count) = header.strip_prefix("reads:") {
                        reads = Some(count.trim().parse().unwrap_or_else(|e| panic!("{name}: reads: {e}")));
                    }
                }
                // A message is the whole line: one may end in spaces.
                None => messages.push(line.to_owned()),
            }
        }
        assert!(!messages.is_empty(), "{name}: the sidecar lists no message");
        let reads = reads.unwrap_or(messages.len());
        assert!(reads <= messages.len(), "{name}: reads {reads} of {} messages", messages.len());
        out.push(Case {
            name,
            image,
            sidecar,
            quick,
            messages,
            reads,
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Search one image as the command line does, and set what it reads
/// against what the image holds.
fn search(case: &Case) -> Searched {
    let output = Command::new(env!("CARGO_BIN_EXE_qr-forensics"))
        .args(["decode", "--escaped", "--partial"])
        .arg(&case.image)
        .output()
        .unwrap_or_else(|e| panic!("{}: the reader did not run: {e}", case.name));
    let printed = String::from_utf8_lossy(&output.stdout);
    let prefix = format!("{}: ", case.image.display());
    let mut left = case.messages.clone();
    let (mut matched, mut misread, mut lines) = (0, Vec::new(), Vec::new());
    for line in printed.lines() {
        let Some(rest) = line.strip_prefix(&prefix) else { continue };
        lines.push(rest.to_owned());
        // What was found, how the read stands, and the message, which
        // may itself hold the separator.
        let mut fields = rest.splitn(3, ": ");
        let (Some(what), Some(standing), Some(message)) = (fields.next(), fields.next(), fields.next()) else {
            continue;
        };
        if what.starts_with("structured append") || standing != "read" {
            continue;
        }
        match left.iter().position(|m| m == message) {
            Some(i) => {
                left.remove(i);
                matched += 1;
            }
            None => misread.push(message.to_owned()),
        }
    }
    Searched {
        matched,
        misread,
        lines,
    }
}

/// Every case searched, one image to a thread and as many threads as the
/// machine has cores, the results in the cases' own order.
fn search_all(cases: &[Case]) -> Vec<Searched> {
    let next = AtomicUsize::new(0);
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get()).min(cases.len().max(1));
    let mut found: Vec<(usize, Searched)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(case) = cases.get(i) else { break };
                        mine.push((i, search(case)));
                    }
                    mine
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    found.sort_by_key(|(i, _)| *i);
    found.into_iter().map(|(_, searched)| searched).collect()
}

/// The sidecar of a case with its count of messages read set to `reads`.
fn rewrite(case: &Case, reads: usize) {
    let text = std::fs::read_to_string(&case.sidecar).unwrap();
    let mut lines: Vec<String> = text
        .lines()
        .filter(|l| !l.strip_prefix(';').is_some_and(|h| h.trim().starts_with("reads:")))
        .map(str::to_owned)
        .collect();
    if reads != case.messages.len() {
        // After the other headers, before the messages.
        let at = lines.iter().position(|l| !l.starts_with(';')).unwrap_or(lines.len());
        lines.insert(at, format!("; reads: {reads}"));
    }
    std::fs::write(&case.sidecar, lines.join("\n") + "\n").unwrap();
}

/// Search the cases and hold each to its sidecar. With `bootstrap`, a
/// count that has changed is written to the sidecar in place of failing;
/// a misread fails either way.
fn check(cases: &[Case], bootstrap: bool) {
    let found = search_all(cases);
    let (mut lost, mut gained, mut misread) = (Vec::new(), Vec::new(), Vec::new());
    let (mut symbols, mut read) = (0, 0);
    for (case, searched) in cases.iter().zip(&found) {
        symbols += case.messages.len();
        read += searched.matched;
        let detail = || format!("{}: {} of {} read, sidecar says {}: {:?}", case.name, searched.matched, case.messages.len(), case.reads, searched.lines);
        if !searched.misread.is_empty() {
            misread.push(format!("{}: {:?}", case.name, searched.misread));
        }
        if searched.matched == case.reads {
            continue;
        }
        if bootstrap {
            rewrite(case, searched.matched);
            println!("rewritten: {}", detail());
        } else if searched.matched < case.reads {
            lost.push(detail());
        } else {
            gained.push(detail());
        }
    }
    println!("{} images, {symbols} symbols, {read} read", cases.len());
    assert!(misread.is_empty(), "messages read that the image does not hold:\n{}", misread.join("\n"));
    assert!(lost.is_empty(), "images that no longer read as their sidecars say:\n{}", lost.join("\n"));
    assert!(
        gained.is_empty(),
        "images that read more than their sidecars say (set QR_SEARCH_BOOTSTRAP and run the ignored test to record it):\n{}",
        gained.join("\n")
    );
}

#[test]
fn the_quick_images_are_searched_as_their_sidecars_say() {
    let quick: Vec<Case> = cases().into_iter().filter(|c| c.quick).collect();
    assert!(quick.len() >= 10, "{} quick images", quick.len());
    check(&quick, false);
}

#[test]
#[ignore = "searches every image of the suite: minutes"]
fn every_image_is_searched_as_its_sidecar_says() {
    check(&cases(), std::env::var_os("QR_SEARCH_BOOTSTRAP").is_some());
}

#[test]
fn every_sidecar_has_its_image_and_every_image_its_sidecar() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/opencv");
    let cases = cases();
    let sidecars = std::fs::read_dir(&dir)
        .unwrap()
        .filter(|e| e.as_ref().unwrap().path().extension().is_some_and(|x| x == "search"))
        .count();
    assert_eq!(sidecars, cases.len());
    // One of the images is a PNG under a name that says JPEG, as OpenCV
    // ships it: the reader goes by the bytes.
    let misnamed = cases.iter().find(|c| c.name == "curved_curved_6").expect("the misnamed image");
    let bytes = std::fs::read(&misnamed.image).unwrap();
    assert!(bytes.starts_with(b"\x89PNG") && misnamed.image.extension().is_some_and(|x| x == "jpg"));
}
