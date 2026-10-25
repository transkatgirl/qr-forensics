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
//! Beside the table of how the symbols fared, the test prints how long
//! the search of each set took, the photos searched a few at a time as
//! they are, and the slowest photos. Three more variables measure more:
//!
//! - `QR_BOOFCV_METRICS` has the reader print what each read made of
//!   the modules its decode came to know (`--metrics`), and the test
//!   adds them up by set: the share of those modules read the wrong
//!   colour and left unknown, the share of its correction the worst
//!   block of each read spent, and, over all the reads, whether the
//!   reliability modules were read at is borne out, bit by bit. A change
//!   that reads every symbol it read before, and reads them better,
//!   shows there, where the table shows nothing.
//! - `QR_BOOFCV_ORACLE` reads every symbol the search did not read once
//!   more, through the corners marked for it (`--placed`), with no
//!   search: a symbol that reads so was lost in the search, and one that
//!   does not, in the reading. The test prints how each fared both ways.
//! - `QR_BOOFCV_REPORT` names a file to write a line for each symbol to,
//!   with tabs between: the photo, the symbol's place, how it fared, how
//!   it fared through its marks when that was asked, how long its photo
//!   took to search, and what its read printed with `--metrics`. Two
//!   such files, from before a change and after it, can be set side by
//!   side.
//!
//! `QR_BOOFCV_READER` names the reader to run in place of the one this
//! crate builds, so that the same test measures a reader built from
//! another revision; one that knows no `--metrics` or `--placed` is run
//! without asking for them.
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
use std::ffi::OsString;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

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

/// What `--metrics` printed of one read ([`qr_forensics::imaging::measure::ReadMeasures`]).
#[derive(Clone, Debug, Default)]
struct Measures {
    modules: usize,
    misread: usize,
    unknown: usize,
    load_max: f64,
    /// By whole bits of reliability: (bits, modules, misread, expected).
    calibration: Vec<(u32, usize, usize, f64)>,
    /// The line as printed, for the report.
    line: String,
}

impl Measures {
    fn parse(line: &str) -> Option<Self> {
        let mut out = Measures { line: line.to_owned(), ..Measures::default() };
        for pair in line.split_whitespace() {
            let (name, value) = pair.split_once('=')?;
            match name {
                "modules" => out.modules = value.parse().ok()?,
                "misread" => out.misread = value.parse().ok()?,
                "unknown" => out.unknown = value.parse().ok()?,
                "load_max" => out.load_max = value.parse().ok()?,
                "calibration" => {
                    for bin in value.split(',').filter(|b| !b.is_empty()) {
                        let fields: Vec<&str> = bin.split(':').collect();
                        let [bits, modules, misread, expected] = fields[..] else { return None };
                        out.calibration.push((bits.parse().ok()?, modules.parse().ok()?, misread.parse().ok()?, expected.parse().ok()?));
                    }
                }
                _ => {}
            }
        }
        Some(out)
    }
}

/// One symbol the reader printed: where it lies, how it fared, and what
/// its read printed with `--metrics`.
type Printed = (Quad, Fared, Option<Measures>);

/// The reader the test runs: the one `QR_BOOFCV_READER` names, or this
/// crate's own.
fn reader() -> OsString {
    std::env::var_os("QR_BOOFCV_READER").unwrap_or_else(|| env!("CARGO_BIN_EXE_qr-forensics").into())
}

/// Whether the reader takes `option`, as its usage says.
fn reader_takes(option: &str) -> bool {
    let usage = Command::new(reader()).args(["decode", "--help"]).output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
    usage.contains(option)
}

/// What the reader printed of one photo, asked as `extra` says besides
/// the options every search is made with: each outline with how the
/// symbol in it fared and what its read printed with `--metrics`.
fn read_photo(case: &Case, extra: &[String]) -> Vec<Printed> {
    let output = Command::new(reader())
        .args(["decode", "--corners", "--partial", "--escaped"])
        .args(extra)
        .arg(&case.image)
        .output()
        .unwrap_or_else(|e| panic!("{}: the reader did not run: {e}", case.name));
    let printed = String::from_utf8_lossy(&output.stdout);
    let prefix = format!("{}: ", case.image.display());
    let mut out: Vec<Printed> = Vec::new();
    for line in printed.lines() {
        // The figures of the read on the line before.
        if let Some(figures) = line.strip_prefix("  metrics: ") {
            if let Some(last) = out.last_mut() {
                last.2 = Measures::parse(figures);
            }
            continue;
        }
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
        out.push((std::array::from_fn(|i| [numbers[2 * i], numbers[2 * i + 1]]), fared, None));
    }
    out
}

/// A photo searched: what the reader printed, and how many seconds it
/// took.
struct Searched {
    printed: Vec<Printed>,
    seconds: f64,
}

/// The search of one photo, timed.
fn search(case: &Case, metrics: bool) -> Searched {
    let extra: Vec<String> = if metrics { vec!["--metrics".to_owned()] } else { Vec::new() };
    let start = Instant::now();
    let printed = read_photo(case, &extra);
    Searched { printed, seconds: start.elapsed().as_secs_f64() }
}

/// How a symbol fared read through the corners marked for it, with no
/// search, and what that read printed with `--metrics`.
fn read_through_marks(case: &Case, symbol: &Quad, metrics: bool) -> (Fared, Option<Measures>) {
    let corners: Vec<String> = symbol.iter().flat_map(|p| [p[0], p[1]]).map(|v| format!("{v}")).collect();
    let mut extra = vec![format!("--placed={}", corners.join(","))];
    if metrics {
        extra.push("--metrics".to_owned());
    }
    read_photo(case, &extra).into_iter().map(|(_, fared, measures)| (fared, measures)).max_by_key(|(fared, _)| *fared).unwrap_or((Fared::NotFound, None))
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

/// Modules of the reads of one set, or of all, as `--metrics` counted
/// them, and the worst block of each read.
#[derive(Default)]
struct Tally {
    reads: usize,
    modules: usize,
    misread: usize,
    unknown: usize,
    /// The share of its correction the worst block of each read spent.
    loads: Vec<f64>,
}

impl Tally {
    fn add(&mut self, measures: &Measures) {
        self.reads += 1;
        self.modules += measures.modules;
        self.misread += measures.misread;
        self.unknown += measures.unknown;
        self.loads.push(measures.load_max);
    }

    /// The figures of a row of the table: reads measured, the shares of
    /// modules read wrong and left unknown, in parts per thousand, and the
    /// mean and the ninetieth centile of the worst block's load.
    fn row(&self) -> String {
        let per_mille = |n: usize| 1000.0 * n as f64 / self.modules.max(1) as f64;
        let mut loads = self.loads.clone();
        loads.sort_by(f64::total_cmp);
        let mean = loads.iter().sum::<f64>() / loads.len().max(1) as f64;
        let high = loads.get((loads.len() * 9 / 10).min(loads.len().saturating_sub(1))).copied().unwrap_or(0.0);
        format!("{:>7}{:>10.2}{:>10.2}{:>9.3}{:>9.3}", self.reads, per_mille(self.misread), per_mille(self.unknown), mean, high)
    }
}

#[test]
fn the_benchmark_is_searched_as_recorded() {
    let Some(dir) = std::env::var_os("QR_BOOFCV_DIR") else {
        return;
    };
    let only: Option<Vec<String>> = std::env::var("QR_BOOFCV_ONLY").ok().map(|sets| sets.split(',').map(|set| set.trim().to_owned()).collect());
    let bootstrap = std::env::var_os("QR_BOOFCV_BOOTSTRAP").is_some();
    let metrics = std::env::var_os("QR_BOOFCV_METRICS").is_some() && reader_takes("--metrics");
    let oracle = std::env::var_os("QR_BOOFCV_ORACLE").is_some() && reader_takes("--placed");
    let report = std::env::var_os("QR_BOOFCV_REPORT").map(PathBuf::from);
    let cases = cases(Path::new(&dir), only.as_deref());
    assert!(!cases.is_empty(), "no photos of the sets asked for under {}", Path::new(&dir).display());
    let start = Instant::now();
    let found = each_few_at_a_time(&cases, |case| search(case, metrics));
    let searching = start.elapsed().as_secs_f64();

    let before = recorded();
    let mut now: BTreeMap<(String, usize), Fared> = BTreeMap::new();
    // By set: the symbols marked, and how many fared each way; the
    // seconds its photos took; and the modules of its reads.
    let mut tally: BTreeMap<String, [usize; 4]> = BTreeMap::new();
    let mut seconds: BTreeMap<String, f64> = BTreeMap::new();
    let mut measured: BTreeMap<String, Tally> = BTreeMap::new();
    let mut calibration: BTreeMap<u32, (usize, usize, f64)> = BTreeMap::new();
    // Each symbol: its photo, its place, how it fared and what its read printed.
    let mut symbols: Vec<(&Case, usize, Fared, Option<Measures>)> = Vec::new();
    let (mut lost, mut gained, mut unrecorded) = (Vec::new(), Vec::new(), 0usize);
    for (case, searched) in cases.iter().zip(&found) {
        *seconds.entry(case.set.clone()).or_default() += searched.seconds;
        for (index, symbol) in case.symbols.iter().enumerate() {
            let best = searched.printed.iter().filter(|(outline, ..)| lies_on(symbol, outline)).max_by_key(|(_, fared, _)| *fared);
            let fared = best.map_or(Fared::NotFound, |(_, fared, _)| *fared);
            let measures = best.and_then(|(.., measures)| measures.clone());
            tally.entry(case.set.clone()).or_default()[fared as usize] += 1;
            if let Some(measures) = &measures {
                measured.entry(case.set.clone()).or_default().add(measures);
                for &(bits, modules, misread, expected) in &measures.calibration {
                    let bin = calibration.entry(bits).or_default();
                    *bin = (bin.0 + modules, bin.1 + misread, bin.2 + expected);
                }
            }
            let key = (case.name.clone(), index);
            match before.get(&key) {
                Some(&was) if fared < was => lost.push(format!("{} symbol {index}: {}, recorded as {}", case.name, fared.label(), was.label())),
                Some(&was) if fared > was => gained.push(format!("{} symbol {index}: {}, recorded as {}", case.name, fared.label(), was.label())),
                Some(_) => {}
                None => unrecorded += 1,
            }
            now.insert(key, fared);
            symbols.push((case, index, fared, measures));
        }
    }

    println!("{:14}{:>9}{:>7}{:>7}{:>7}{:>11}{:>11}", "set", "symbols", "read", "found", "lead", "not found", "seconds");
    let mut total = [0usize; 4];
    for (set, counts) in &tally {
        println!(
            "{set:14}{:>9}{:>7}{:>7}{:>7}{:>11}{:>11.1}",
            counts.iter().sum::<usize>(),
            counts[3],
            counts[2],
            counts[1],
            counts[0],
            seconds.get(set).copied().unwrap_or(0.0)
        );
        for (sum, count) in total.iter_mut().zip(counts) {
            *sum += count;
        }
    }
    println!(
        "{:14}{:>9}{:>7}{:>7}{:>7}{:>11}{:>11.1}",
        "all",
        total.iter().sum::<usize>(),
        total[3],
        total[2],
        total[1],
        total[0],
        seconds.values().sum::<f64>()
    );
    println!("searched in {searching:.1} seconds, {} photos a few at a time; the slowest:", cases.len());
    let mut slowest: Vec<(&Case, f64)> = cases.iter().zip(&found).map(|(case, searched)| (case, searched.seconds)).collect();
    slowest.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (case, seconds) in slowest.iter().take(10) {
        println!("  {seconds:>8.1}  {}", case.name);
    }

    if metrics {
        println!();
        println!("modules of the confirmed blocks of each read: read wrong and left unknown, per thousand, and the worst block's load");
        println!("{:14}{:>7}{:>10}{:>10}{:>9}{:>9}", "set", "reads", "misread", "unknown", "load", "load 90%");
        let mut all = Tally::default();
        for (set, set_tally) in &measured {
            println!("{set:14}{}", set_tally.row());
            all.reads += set_tally.reads;
            all.modules += set_tally.modules;
            all.misread += set_tally.misread;
            all.unknown += set_tally.unknown;
            all.loads.extend(&set_tally.loads);
        }
        println!("{:14}{}", "all", all.row());
        println!();
        println!("reliability borne out: the modules read at each whole number of bits, misread and expected to be");
        println!("{:>6}{:>10}{:>9}{:>12}{:>8}", "bits", "modules", "misread", "expected", "ratio");
        for (bits, (modules, misread, expected)) in &calibration {
            // The ratio says nothing where not one misread is expected.
            let ratio = if *expected >= 1.0 { format!("{:.2}", *misread as f64 / expected) } else { "-".to_owned() };
            println!("{bits:>6}{modules:>10}{misread:>9}{expected:>12.4}{ratio:>8}");
        }
    }

    // The symbols the search did not read, each read again through its
    // marks.
    let mut through_marks: BTreeMap<(String, usize), (Fared, Option<Measures>)> = BTreeMap::new();
    if oracle {
        let unread: Vec<(&Case, usize)> = symbols.iter().filter(|(.., fared, _)| *fared < Fared::Read).map(|(case, index, ..)| (*case, *index)).collect();
        let read = each_few_at_a_time(&unread, |(case, index)| read_through_marks(case, &case.symbols[*index], metrics));
        println!();
        println!("the symbols the search did not read, read through the corners marked for them:");
        let mut both: BTreeMap<(Fared, Fared), usize> = BTreeMap::new();
        for ((case, index), (marked, measures)) in unread.iter().zip(read) {
            let searched = now[&(case.name.clone(), *index)];
            *both.entry((searched, marked)).or_default() += 1;
            println!("  {} symbol {index}: {} in the search, {} through its marks", case.name, searched.label(), marked.label());
            through_marks.insert((case.name.clone(), *index), (marked, measures));
        }
        for ((searched, marked), count) in &both {
            println!("  {count:>4} {} in the search, {} through the marks", searched.label(), marked.label());
        }
    }

    if let Some(report) = &report {
        let mut text = String::from("photo\tsymbol\tfared\tthrough marks\tseconds\tmetrics\n");
        let seconds_of: BTreeMap<&str, f64> = cases.iter().zip(&found).map(|(case, searched)| (case.name.as_str(), searched.seconds)).collect();
        for (case, index, fared, measures) in &symbols {
            let marked = through_marks.get(&(case.name.clone(), *index)).map_or("", |(marked, _)| marked.label());
            let line = measures.as_ref().map_or("", |m| m.line.as_str());
            let _ = writeln!(text, "{}\t{index}\t{}\t{marked}\t{:.2}\t{line}", case.name, fared.label(), seconds_of[case.name.as_str()]);
        }
        std::fs::write(report, text).unwrap_or_else(|e| panic!("{}: {e}", report.display()));
    }

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
