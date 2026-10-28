//! The benchmark of Benito-Altamirano and others' "Reading QR Codes on
//! challenging surfaces using thin-plate splines" (Pattern Recognition
//! Letters 184, 2024), searched as `qr-forensics decode <image>` searches
//! a photo: one printed symbol photographed on flat surfaces, wrapped
//! round tubes and bottles and bent by hand, and symbols of versions 1 to
//! 13 rendered and seen in perspective, 911 images in all. BoofCV's
//! benchmark marks where each of its symbols lies and says nothing of
//! what it holds; this one says what each symbol holds and nothing of
//! where it lies, and each image holds one symbol, so a read is held to
//! its message as the reads of `tests/opencv` are, and a symbol found
//! or a lead counts wherever in the image it lies.
//!
//! The images are four hundred megabytes and are not kept here: the test
//! fetches the archive the authors published the first time it runs, into
//! the target directory, and reads them from there after
//! (`tests/datasets/mod.rs`), or reads the directory the archive unpacks
//! to wherever `QR_SURFACES_DIR` names it. It takes five minutes on a
//! machine of fourteen cores, and so is left out of `cargo test` and runs
//! by hand, in release mode:
//!
//! ```text
//! cargo test --release --test surfaces -- --ignored --nocapture
//! ```
//!
//! `QR_SURFACES_ONLY` names the sets to search when that is not all of
//! them, with commas between, as `flat,random`, and `QR_SURFACES_READER`
//! the reader to run in place of the one this crate builds, as
//! `QR_BOOFCV_READER` does for BoofCV's benchmark.
//!
//! What is held to account is how far the search got with each image's
//! symbol: `read`, when a read that is to be trusted gives its message;
//! `found`, when a symbol was found that did not decode; `lead`, when the
//! search found finder patterns and could fit no symbol to them; and `not
//! found`. `tests/surfaces/expected.txt` records that for every image, and
//! the test fails where an image now fares worse than recorded, or better
//! without the record saying so. With `QR_SURFACES_BOOTSTRAP` set it
//! writes what it finds into the record instead, for the sets it
//! searched. A read to be trusted that gives a message the symbol does
//! not hold fails the test either way, since a misread is worse than a
//! miss.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use qr_forensics::decode::text::escape_controls;

mod datasets;

/// The archive the paper's authors published on Mendeley Data,
/// doi:10.17632/m6mfwc52vk.1, under CC BY 4.0.
const ARCHIVE: datasets::Archive = datasets::Archive {
    name: "qr_codes_in_surfaces",
    url: "https://data.mendeley.com/public-files/datasets/m6mfwc52vk/files/67b47b85-0e10-4095-8c61-331de5039791/file_downloaded",
    sha256: "706352654a744217b6853c77362f4a32cc318d941b715423948ba2108aae7523",
};

/// How far the search got with an image's symbol, least first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Fared {
    NotFound,
    Lead,
    Found,
    Read,
}

impl Fared {
    const ALL: [Fared; 4] = [Fared::NotFound, Fared::Lead, Fared::Found, Fared::Read];

    fn label(self) -> &'static str {
        match self {
            Fared::NotFound => "not found",
            Fared::Lead => "lead",
            Fared::Found => "found",
            Fared::Read => "read",
        }
    }

    fn named(label: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|fared| fared.label() == label)
    }
}

/// One image of the benchmark and the symbol it holds.
struct Case {
    /// `set/image.jpg`, as the record names it.
    name: String,
    set: String,
    /// How the symbol was bent, as its annotation says: `affine` or
    /// `perspective` for a rendered one, `perspective` for one
    /// photographed flat, `cylindric` or `surface` for one wrapped round
    /// something or bent by hand.
    deformation: String,
    image: PathBuf,
    /// The symbol's message, as `--escaped` prints it.
    message: String,
}

fn json(path: &Path) -> serde_json::Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The string `key` holds in a file of the archive.
fn field(value: &serde_json::Value, key: &str, path: &Path) -> String {
    value.get(key).and_then(|v| v.as_str()).unwrap_or_else(|| panic!("{}: no {key}", path.display())).to_owned()
}

fn files(dir: &Path, extension: &str) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|e| e == extension))
        .collect();
    out.sort();
    out
}

/// The images of the sets asked for, in name order. A set holds the
/// symbols as rendered in `bitmaps`, each with a `.json` of its message,
/// and an annotation in `annotations` for each image in `images`, naming
/// the symbol it is of. Eight more photos lie loose in `flat` with no
/// annotation, and are not searched.
fn cases(dir: &Path, only: Option<&[String]>) -> Vec<Case> {
    let mut sets: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}: QR_SURFACES_DIR should name the directory the archive unpacks to", dir.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.join("annotations").is_dir())
        .collect();
    sets.sort();
    let mut out = Vec::new();
    for set_dir in sets {
        let set = set_dir.file_name().unwrap().to_string_lossy().into_owned();
        if only.is_some_and(|only| !only.contains(&set)) {
            continue;
        }
        let messages: BTreeMap<String, String> = files(&set_dir.join("bitmaps"), "json")
            .iter()
            .map(|path| {
                let symbol = json(path);
                (field(&symbol, "id", path), escape_controls(&field(&symbol, "data", path)))
            })
            .collect();
        for annotation in files(&set_dir.join("annotations"), "json") {
            let value = json(&annotation);
            let id = field(&value, "id", &annotation);
            let of = field(&value, "id_original", &annotation);
            let image = set_dir.join("images").join(format!("{id}.jpg"));
            assert!(image.is_file(), "{}: no image {}", annotation.display(), image.display());
            out.push(Case {
                name: format!("{set}/{id}.jpg"),
                set: set.clone(),
                deformation: field(&value, "deformation", &annotation),
                image,
                message: messages.get(&of).unwrap_or_else(|| panic!("{}: no symbol {of} in bitmaps", annotation.display())).clone(),
            });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// The reader the test runs: the one `QR_SURFACES_READER` names, or this
/// crate's own.
fn reader() -> OsString {
    std::env::var_os("QR_SURFACES_READER").unwrap_or_else(|| env!("CARGO_BIN_EXE_qr-forensics").into())
}

/// How far what the reader printed of an image, its lines each beginning
/// `prefix`, got with a symbol holding `message`, and the messages it
/// read, and would have trusted, that are not that one.
fn fared(printed: &str, prefix: &str, message: &str) -> (Fared, Vec<String>) {
    let (mut best, mut misread) = (Fared::NotFound, Vec::new());
    for line in printed.lines() {
        let Some(rest) = line.strip_prefix(prefix) else { continue };
        // What was found, how it stands and the message, which may
        // itself hold the separator.
        let mut fields = rest.splitn(3, ": ");
        let (Some(what), Some(standing), Some(read)) = (fields.next(), fields.next(), fields.next()) else { continue };
        if what.starts_with("structured append") {
            continue;
        }
        let this = if standing.ends_with("no symbol fitted") {
            Fared::Lead
        } else if standing != "read" && !standing.starts_with("read, ") {
            Fared::Found
        } else if read == message {
            Fared::Read
        } else {
            misread.push(read.to_owned());
            Fared::Found
        };
        best = best.max(this);
    }
    (best, misread)
}

/// An image searched: how far the search got, what it misread, and how
/// many seconds it took.
struct Searched {
    fared: Fared,
    misread: Vec<String>,
    seconds: f64,
}

fn search(case: &Case) -> Searched {
    let start = Instant::now();
    let output = Command::new(reader())
        .args(["decode", "--partial", "--escaped"])
        .arg(&case.image)
        .output()
        .unwrap_or_else(|e| panic!("{}: the reader did not run: {e}", case.name));
    let (fared, misread) = fared(&String::from_utf8_lossy(&output.stdout), &format!("{}: ", case.image.display()), &case.message);
    Searched { fared, misread, seconds: start.elapsed().as_secs_f64() }
}

/// `job` done for every item, a few at a time, each having cores of its
/// own to fit proposals side by side on: the results in the items' own
/// order.
fn each_few_at_a_time<T: Sync, R: Send>(items: &[T], job: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let next = AtomicUsize::new(0);
    let workers = (std::thread::available_parallelism().map_or(4, |n| n.get()) / 2).clamp(1, items.len().max(1));
    let mut done: Vec<(usize, R)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = items.get(i) else { break };
                        mine.push((i, job(item)));
                    }
                    mine
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    done.sort_by_key(|(i, _)| *i);
    done.into_iter().map(|(_, result)| result).collect()
}

fn record() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/surfaces/expected.txt")
}

/// The record: how the search of each image fared.
fn recorded() -> BTreeMap<String, Fared> {
    let text = std::fs::read_to_string(record()).unwrap_or_default();
    let mut out = BTreeMap::new();
    for line in text.lines().filter(|l| !l.starts_with(';') && !l.trim().is_empty()) {
        let (name, fared) = line.split_once(' ').unwrap_or_else(|| panic!("the record: {line}"));
        out.insert(name.to_owned(), Fared::named(fared).unwrap_or_else(|| panic!("the record: {line}")));
    }
    out
}

/// Write the record: what was recorded for the sets not searched, and
/// what was found for those that were.
fn rewrite(before: &BTreeMap<String, Fared>, searched: &[String], now: &BTreeMap<String, Fared>) {
    let in_searched = |name: &str| searched.iter().any(|set| name.split('/').next() == Some(set.as_str()));
    let mut all: BTreeMap<String, Fared> = before.iter().filter(|(name, _)| !in_searched(name)).map(|(k, v)| (k.clone(), *v)).collect();
    all.extend(now.iter().map(|(k, v)| (k.clone(), *v)));
    let mut text = String::from(
        "; How the search of each image of the benchmark of \"Reading QR Codes on challenging surfaces using\n\
         ; thin-plate splines\" fares with the symbol it holds: read, found, lead or not found.\n\
         ; Written by tests/surfaces.rs with QR_SURFACES_BOOTSTRAP set; see tests/surfaces/README.md.\n",
    );
    for (name, fared) in &all {
        text.push_str(&format!("{name} {}\n", fared.label()));
    }
    std::fs::write(record(), text).unwrap();
}

#[test]
#[ignore = "fetches four hundred megabytes of images the first time and searches them: minutes"]
fn the_benchmark_is_searched_as_recorded() {
    let dir = std::env::var_os("QR_SURFACES_DIR").map_or_else(|| datasets::fetched(&ARCHIVE), PathBuf::from);
    let only: Option<Vec<String>> = std::env::var("QR_SURFACES_ONLY").ok().map(|sets| sets.split(',').map(|set| set.trim().to_owned()).collect());
    let bootstrap = std::env::var_os("QR_SURFACES_BOOTSTRAP").is_some();
    let cases = cases(&dir, only.as_deref());
    assert!(!cases.is_empty(), "no images of the sets asked for under {}", dir.display());
    let start = Instant::now();
    let found = each_few_at_a_time(&cases, search);
    let searching = start.elapsed().as_secs_f64();

    let before = recorded();
    let mut now: BTreeMap<String, Fared> = BTreeMap::new();
    // By set and deformation: how many images fared each way, and the
    // seconds they took.
    let mut tally: BTreeMap<(String, String), ([usize; 4], f64)> = BTreeMap::new();
    let (mut lost, mut gained, mut misread, mut unrecorded) = (Vec::new(), Vec::new(), Vec::new(), 0usize);
    for (case, searched) in cases.iter().zip(&found) {
        let row = tally.entry((case.set.clone(), case.deformation.clone())).or_default();
        row.0[searched.fared as usize] += 1;
        row.1 += searched.seconds;
        if !searched.misread.is_empty() {
            misread.push(format!("{}: {:?}, where it holds {:?}", case.name, searched.misread, case.message));
        }
        match before.get(&case.name) {
            Some(&was) if searched.fared < was => lost.push(format!("{}: {}, recorded as {}", case.name, searched.fared.label(), was.label())),
            Some(&was) if searched.fared > was => gained.push(format!("{}: {}, recorded as {}", case.name, searched.fared.label(), was.label())),
            Some(_) => {}
            None => unrecorded += 1,
        }
        now.insert(case.name.clone(), searched.fared);
    }

    println!("{:30}{:>8}{:>7}{:>7}{:>7}{:>11}{:>11}", "set", "images", "read", "found", "lead", "not found", "seconds");
    let mut total = ([0usize; 4], 0.0);
    for ((set, deformation), (counts, seconds)) in &tally {
        println!(
            "{:30}{:>8}{:>7}{:>7}{:>7}{:>11}{:>11.1}",
            format!("{set} {deformation}"),
            counts.iter().sum::<usize>(),
            counts[3],
            counts[2],
            counts[1],
            counts[0],
            seconds
        );
        for (sum, count) in total.0.iter_mut().zip(counts) {
            *sum += count;
        }
        total.1 += seconds;
    }
    let (counts, seconds) = total;
    println!("{:30}{:>8}{:>7}{:>7}{:>7}{:>11}{:>11.1}", "all", counts.iter().sum::<usize>(), counts[3], counts[2], counts[1], counts[0], seconds);
    println!("searched in {searching:.1} seconds, {} images a few at a time; the slowest:", cases.len());
    let mut slowest: Vec<(&Case, f64)> = cases.iter().zip(&found).map(|(case, searched)| (case, searched.seconds)).collect();
    slowest.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (case, seconds) in slowest.iter().take(10) {
        println!("  {seconds:>8.1}  {}", case.name);
    }

    if bootstrap {
        let searched: Vec<String> = tally.keys().map(|(set, _)| set.clone()).collect();
        rewrite(&before, &searched, &now);
        println!("recorded: {} images fare worse than before, {} better, {unrecorded} were not in the record", lost.len(), gained.len());
        for line in lost.iter().chain(&gained) {
            println!("  {line}");
        }
    }
    assert!(misread.is_empty(), "messages read, and to be trusted, that the image does not hold:\n{}", misread.join("\n"));
    if bootstrap {
        return;
    }
    assert!(lost.is_empty(), "images that fare worse than recorded:\n{}", lost.join("\n"));
    assert!(
        gained.is_empty() && unrecorded == 0,
        "{unrecorded} images are not in the record, and these fare better than recorded (set QR_SURFACES_BOOTSTRAP to record it):\n{}",
        gained.join("\n")
    );
}

#[test]
fn what_the_reader_prints_is_held_to_the_message() {
    let prefix = "flat/images/a.jpg: ";
    let lines = |lines: &[&str]| lines.iter().map(|line| format!("{prefix}{line}\n")).collect::<String>();
    // A read that gives the symbol's message, which may itself hold the
    // separator, beside a lead elsewhere in the photo.
    let printed = lines(&["QR Code 7: read: https://example.com: a", "QR Code: 3 finder patterns, no symbol fitted: at 1,2,3,4,5,6,7,8"]);
    assert_eq!(fared(&printed, prefix, "https://example.com: a"), (Fared::Read, Vec::new()));
    assert_eq!(fared(&lines(&["QR Code 7, mirrored: read, padding pinned: x"]), prefix, "x"), (Fared::Read, Vec::new()));
    // A symbol that did not decode is found, whatever its message.
    assert_eq!(fared(&lines(&["QR Code 7: unconfirmed, 3 of 4 blocks confirmed: x<?>"]), prefix, "xy"), (Fared::Found, Vec::new()));
    // A read to be trusted of another message is a misread, and a symbol
    // found all the same.
    assert_eq!(fared(&lines(&["QR Code 7: read: y"]), prefix, "x"), (Fared::Found, vec!["y".to_owned()]));
    assert_eq!(fared(&lines(&["QR Code: 2 finder patterns, no symbol fitted: at 1,2,3,4,5,6,7,8"]), prefix, "x"), (Fared::Lead, Vec::new()));
    // A placement the search fitted and could not read is a symbol found
    // that did not decode, and no lead.
    let fitted = lines(&["QR Code 1: 3 finder patterns, fitted and not read: at 1,2,3,4,5,6,7,8"]);
    assert_eq!(fared(&fitted, prefix, "x"), (Fared::Found, Vec::new()));
    // Lines of another image, and none at all, are nothing found.
    assert_eq!(fared("other.jpg: QR Code 7: read: x\n", prefix, "x"), (Fared::NotFound, Vec::new()));
    assert_eq!(Fared::ALL.map(|fared| Fared::named(fared.label())), Fared::ALL.map(Some));
}
