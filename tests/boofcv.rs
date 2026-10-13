//! BoofCV's QR Code benchmark, searched as `qr-forensics decode <image>`
//! searches a photo: 536 photographs in sixteen sets, blurred, curved,
//! damaged, seen steeply, lit badly, many to a sheet, each with the
//! corners of every symbol it holds marked by hand. The photos of
//! `tests/photos` and `tests/blackbox` are read through outlines and
//! say nothing of the search; `tests/opencv` is searched whole, and is
//! fifty-odd images of mostly sound symbols. This is the corpus the
//! search was tuned against, and the figures quoted for it in the source
//! were counted by hand until this test counted them.
//!
//! The photographs are two hundred megabytes and are not kept here. The
//! test runs only when `QR_BOOFCV_DIR` names the `qrcodes` directory the
//! archive unpacks to, and does nothing otherwise: see
//! `tests/boofcv/README.md` for where the archive is. It takes half an
//! hour over the whole benchmark on a machine of fourteen cores, in
//! release mode:
//!
//! ```text
//! QR_BOOFCV_DIR=path/to/qrcodes cargo test --release --test boofcv -- --nocapture
//! ```
//!
//! `QR_BOOFCV_ONLY` names the sets to search when that is not all of
//! them, with commas between, as `perspective,close`.
//!
//! The benchmark gives no messages, only where each symbol lies, so what
//! is held to account is how far the search got with each symbol marked:
//! `read`, when a read that is to be trusted lies on it; `found`, when a
//! symbol that did not decode does; `lead`, when the search found its
//! finder patterns and could fit no symbol to them; and `not found`.
//! `tests/boofcv/expected.txt` records that for every symbol, and the
//! test fails where a symbol now fares worse than recorded, or better
//! without the record saying so. With `QR_BOOFCV_BOOTSTRAP` set it
//! writes what it finds into the record instead, for the sets it
//! searched.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use image::metadata::Orientation;
use image::{ImageDecoder, ImageReader};

type Quad = [[f64; 2]; 4];

/// How far the search got with a symbol the benchmark marks, least first.
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

/// One photograph of the benchmark and the symbols marked in it.
struct Case {
    /// `set/image.jpg`, as the record names it.
    name: String,
    set: String,
    image: PathBuf,
    /// Each symbol's corners, in the pixels the reader prints corners in.
    symbols: Vec<Quad>,
}

fn is_image(path: &Path) -> bool {
    matches!(path.extension().and_then(|e| e.to_str()), Some("png" | "jpg" | "jpeg"))
}

/// The corners marked in a label file, four to a symbol: a line of eight
/// numbers for each after a `SETS` line, or, for a photo of one symbol,
/// four lines of two.
fn marked(labels: &Path) -> Vec<Quad> {
    let text = std::fs::read_to_string(labels).unwrap_or_else(|e| panic!("{}: {e}", labels.display()));
    let (mut quads, mut loose) = (Vec::new(), Vec::new());
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#') && *l != "SETS") {
        let numbers: Vec<f64> = line.split_whitespace().map(|n| n.parse().unwrap_or_else(|e| panic!("{}: {n}: {e}", labels.display()))).collect();
        match numbers.len() {
            8 => quads.push(std::array::from_fn(|i| [numbers[2 * i], numbers[2 * i + 1]])),
            2 => loose.push([numbers[0], numbers[1]]),
            other => panic!("{}: a line of {other} numbers", labels.display()),
        }
    }
    quads.extend(loose.chunks_exact(4).map(|four| std::array::from_fn(|i| four[i])));
    quads
}

/// A point marked in a photo as it is stored, `width` by `height`, in the
/// pixels the reader prints: the photo's own, turned upright as its EXIF
/// orientation asks. A small photo is read enlarged, and its corners are
/// printed in its own pixels all the same. The benchmark's marks are in
/// the stored pixels, as BoofCV loads them.
fn as_read(point: [f64; 2], orientation: Orientation, width: f64, height: f64) -> [f64; 2] {
    let [x, y] = point;
    match orientation {
        Orientation::NoTransforms => [x, y],
        Orientation::Rotate90 => [height - y, x],
        Orientation::Rotate180 => [width - x, height - y],
        Orientation::Rotate270 => [y, width - x],
        Orientation::FlipHorizontal => [width - x, y],
        Orientation::FlipVertical => [x, height - y],
        Orientation::Rotate90FlipH => [y, x],
        Orientation::Rotate270FlipH => [height - y, width - x],
    }
}

/// The photographs of the sets asked for, in name order.
fn cases(dir: &Path, only: Option<&[String]>) -> Vec<Case> {
    let detection = dir.join("detection");
    let mut sets: Vec<PathBuf> = std::fs::read_dir(&detection)
        .unwrap_or_else(|e| panic!("{}: {e}: QR_BOOFCV_DIR should name the benchmark's qrcodes directory", detection.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.is_dir())
        .collect();
    sets.sort();
    let mut out = Vec::new();
    for set_dir in sets {
        let set = set_dir.file_name().unwrap().to_string_lossy().into_owned();
        if only.is_some_and(|only| !only.contains(&set)) {
            continue;
        }
        let mut images: Vec<PathBuf> = std::fs::read_dir(&set_dir).unwrap().map(|entry| entry.unwrap().path()).filter(|path| is_image(path)).collect();
        images.sort();
        for image in images {
            let labels = image.with_extension("txt");
            if !labels.exists() {
                continue;
            }
            let mut decoder = ImageReader::open(&image)
                .and_then(|reader| reader.with_guessed_format())
                .unwrap_or_else(|e| panic!("{}: {e}", image.display()))
                .into_decoder()
                .unwrap_or_else(|e| panic!("{}: {e}", image.display()));
            let (width, height) = decoder.dimensions();
            let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
            let symbols = marked(&labels).into_iter().map(|quad| quad.map(|p| as_read(p, orientation, f64::from(width), f64::from(height)))).collect();
            let name = format!("{set}/{}", image.file_name().unwrap().to_string_lossy());
            out.push(Case { name, set: set.clone(), image, symbols });
        }
    }
    out
}

fn centre(quad: &Quad) -> [f64; 2] {
    [quad.iter().map(|p| p[0]).sum::<f64>() / 4.0, quad.iter().map(|p| p[1]).sum::<f64>() / 4.0]
}

fn area(quad: &Quad) -> f64 {
    (0..4).map(|i| quad[i][0] * quad[(i + 1) % 4][1] - quad[(i + 1) % 4][0] * quad[i][1]).sum::<f64>().abs() / 2.0
}

/// Whether an outline the reader printed lies on a symbol marked: about
/// the same place and the same size. The marks are a hand's, and an
/// outline about finder patterns is drawn loose.
fn lies_on(symbol: &Quad, outline: &Quad) -> bool {
    let (marked, printed) = (area(symbol), area(outline));
    if marked <= 0.0 || printed <= 0.0 {
        return false;
    }
    let (a, b) = (centre(symbol), centre(outline));
    let apart = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    apart < 0.3 * marked.sqrt() && (0.4..2.5).contains(&(printed / marked))
}

/// What the reader printed of one photo: each outline with how the
/// symbol in it fared.
fn search(case: &Case) -> Vec<(Quad, Fared)> {
    let output = Command::new(env!("CARGO_BIN_EXE_qr-forensics"))
        .args(["decode", "--corners", "--partial", "--escaped"])
        .arg(&case.image)
        .output()
        .unwrap_or_else(|e| panic!("{}: the reader did not run: {e}", case.name));
    let printed = String::from_utf8_lossy(&output.stdout);
    let prefix = format!("{}: ", case.image.display());
    let mut out = Vec::new();
    for line in printed.lines() {
        let Some(rest) = line.strip_prefix(&prefix) else { continue };
        // What was found, how it stands and where it lies; a message
        // after that may itself hold the separator.
        let mut fields = rest.splitn(4, ": ");
        let (Some(what), Some(standing), Some(at)) = (fields.next(), fields.next(), fields.next()) else { continue };
        let Some(numbers) = at.strip_prefix("at ") else { continue };
        let numbers: Vec<f64> = numbers.split(',').filter_map(|n| n.parse().ok()).collect();
        if what.starts_with("structured append") || numbers.len() != 8 {
            continue;
        }
        let fared = if standing.ends_with("no symbol fitted") {
            Fared::Lead
        } else if standing == "read" || standing == "read, padding pinned" {
            Fared::Read
        } else {
            Fared::Found
        };
        out.push((std::array::from_fn(|i| [numbers[2 * i], numbers[2 * i + 1]]), fared));
    }
    out
}

/// Every case searched, a few photos at a time, each search having cores
/// of its own to fit proposals side by side on: the results in the
/// cases' own order.
fn search_all(cases: &[Case]) -> Vec<Vec<(Quad, Fared)>> {
    let next = AtomicUsize::new(0);
    let workers = (std::thread::available_parallelism().map_or(4, |n| n.get()) / 2).clamp(1, cases.len().max(1));
    let mut found: Vec<(usize, Vec<(Quad, Fared)>)> = std::thread::scope(|scope| {
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

fn record() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/boofcv/expected.txt")
}

/// The record: how each symbol fared, by photo and by its place among
/// the symbols marked in it.
fn recorded() -> BTreeMap<(String, usize), Fared> {
    let text = std::fs::read_to_string(record()).unwrap_or_default();
    let mut out = BTreeMap::new();
    for line in text.lines().filter(|l| !l.starts_with(';') && !l.trim().is_empty()) {
        let mut fields = line.splitn(3, ' ');
        let (Some(name), Some(index), Some(fared)) = (fields.next(), fields.next(), fields.next()) else { panic!("the record: {line}") };
        let fared = Fared::named(fared).unwrap_or_else(|| panic!("the record: {line}"));
        out.insert((name.to_owned(), index.parse().unwrap_or_else(|e| panic!("the record: {line}: {e}"))), fared);
    }
    out
}

/// Write the record: what was recorded for the sets not searched, and
/// what was found for those that were.
fn rewrite(before: &BTreeMap<(String, usize), Fared>, searched: &[String], now: &BTreeMap<(String, usize), Fared>) {
    let in_searched = |name: &str| searched.iter().any(|set| name.split('/').next() == Some(set.as_str()));
    let mut all: BTreeMap<(String, usize), Fared> = before.iter().filter(|((name, _), _)| !in_searched(name)).map(|(k, v)| (k.clone(), *v)).collect();
    all.extend(now.iter().map(|(k, v)| (k.clone(), *v)));
    let mut text = String::from(
        "; How the search of each photo of BoofCV's QR Code benchmark fares with each symbol marked in it:\n\
         ; the photo, the symbol's place among those its label file marks, and read, found, lead or not found.\n\
         ; Written by tests/boofcv.rs with QR_BOOFCV_BOOTSTRAP set; see tests/boofcv/README.md.\n",
    );
    for ((name, index), fared) in &all {
        text.push_str(&format!("{name} {index} {}\n", fared.label()));
    }
    std::fs::write(record(), text).unwrap();
}

#[test]
fn the_benchmark_is_searched_as_recorded() {
    let Some(dir) = std::env::var_os("QR_BOOFCV_DIR") else {
        return;
    };
    let only: Option<Vec<String>> = std::env::var("QR_BOOFCV_ONLY").ok().map(|sets| sets.split(',').map(|set| set.trim().to_owned()).collect());
    let bootstrap = std::env::var_os("QR_BOOFCV_BOOTSTRAP").is_some();
    let cases = cases(Path::new(&dir), only.as_deref());
    assert!(!cases.is_empty(), "no photos of the sets asked for under {}", Path::new(&dir).display());
    let found = search_all(&cases);

    let before = recorded();
    let mut now: BTreeMap<(String, usize), Fared> = BTreeMap::new();
    // By set: the symbols marked, and how many fared each way.
    let mut tally: BTreeMap<String, [usize; 4]> = BTreeMap::new();
    let (mut lost, mut gained, mut unrecorded) = (Vec::new(), Vec::new(), 0usize);
    for (case, outlines) in cases.iter().zip(&found) {
        for (index, symbol) in case.symbols.iter().enumerate() {
            let fared = outlines.iter().filter(|(outline, _)| lies_on(symbol, outline)).map(|(_, fared)| *fared).max().unwrap_or(Fared::NotFound);
            tally.entry(case.set.clone()).or_default()[fared as usize] += 1;
            let key = (case.name.clone(), index);
            match before.get(&key) {
                Some(&was) if fared < was => lost.push(format!("{} symbol {index}: {}, recorded as {}", case.name, fared.label(), was.label())),
                Some(&was) if fared > was => gained.push(format!("{} symbol {index}: {}, recorded as {}", case.name, fared.label(), was.label())),
                Some(_) => {}
                None => unrecorded += 1,
            }
            now.insert(key, fared);
        }
    }

    println!("{:14}{:>9}{:>7}{:>7}{:>7}{:>11}", "set", "symbols", "read", "found", "lead", "not found");
    let mut total = [0usize; 4];
    for (set, counts) in &tally {
        println!("{set:14}{:>9}{:>7}{:>7}{:>7}{:>11}", counts.iter().sum::<usize>(), counts[3], counts[2], counts[1], counts[0]);
        for (sum, count) in total.iter_mut().zip(counts) {
            *sum += count;
        }
    }
    println!("{:14}{:>9}{:>7}{:>7}{:>7}{:>11}", "all", total.iter().sum::<usize>(), total[3], total[2], total[1], total[0]);

    if bootstrap {
        let searched: Vec<String> = tally.keys().cloned().collect();
        rewrite(&before, &searched, &now);
        println!("recorded: {} symbols fare worse than before, {} better, {unrecorded} were not in the record", lost.len(), gained.len());
        for line in lost.iter().chain(&gained) {
            println!("  {line}");
        }
        return;
    }
    assert!(lost.is_empty(), "symbols that fare worse than recorded:\n{}", lost.join("\n"));
    assert!(
        gained.is_empty() && unrecorded == 0,
        "{unrecorded} symbols are not in the record, and these fare better than recorded (set QR_BOOFCV_BOOTSTRAP to record it):\n{}",
        gained.join("\n")
    );
}

#[test]
fn marks_are_carried_into_the_pixels_the_reader_prints() {
    // A photo stored on its side, to be turned a quarter clockwise for
    // showing: the stored top-left corner comes out at the top right.
    assert_eq!(as_read([0.0, 0.0], Orientation::Rotate90, 4032.0, 3024.0), [3024.0, 0.0]);
    assert_eq!(as_read([4032.0, 3024.0], Orientation::Rotate90, 4032.0, 3024.0), [0.0, 4032.0]);
    assert_eq!(as_read([10.0, 20.0], Orientation::NoTransforms, 4032.0, 3024.0), [10.0, 20.0]);
    assert_eq!(as_read([10.0, 20.0], Orientation::Rotate180, 400.0, 300.0), [390.0, 280.0]);
    // A photo under 256 pixels on its shorter side is read enlarged, and
    // its corners are printed in its own pixels.
    assert_eq!(as_read([10.0, 20.0], Orientation::NoTransforms, 95.0, 126.0), [10.0, 20.0]);
    // An outline lies on a mark it is drawn a little loose about, and
    // not on its neighbour on a sheet.
    let square = |x: f64, y: f64, side: f64| [[x, y], [x + side, y], [x + side, y + side], [x, y + side]];
    assert!(lies_on(&square(100.0, 100.0, 50.0), &square(95.0, 96.0, 62.0)));
    assert!(!lies_on(&square(100.0, 100.0, 50.0), &square(160.0, 100.0, 50.0)));
    assert!(!lies_on(&square(100.0, 100.0, 50.0), &square(60.0, 60.0, 130.0)));
    assert_eq!(Fared::ALL.map(|fared| Fared::named(fared.label())), Fared::ALL.map(Some));
    assert!(Fared::Read > Fared::Found && Fared::Found > Fared::Lead && Fared::Lead > Fared::NotFound);
}
