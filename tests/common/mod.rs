//! Shared helpers: an independent encoder as oracle, and deterministic damage.
#![allow(dead_code)]

use std::sync::Arc;

use qr_forensics::decode::bitstream::SegmentContent;
use qr_forensics::decode::codewords::Pins;
use qr_forensics::decode::pipeline::{DecodeOptions, DecodeReport, decode};
use qr_forensics::imaging::Luma;
use qr_forensics::imaging::adjust::Adjust;
use qr_forensics::imaging::detect::Detection;
use qr_forensics::imaging::read::read_outline;
use qr_forensics::imaging::sample::SampleParams;
use qr_forensics::symbol::function_map::FunctionMap;
use qr_forensics::symbol::grid::{Module, ModuleGrid};
use qr_forensics::symbol::layout::{CwId, SymbolLayout};
use qr_forensics::symbol::{EcLevel, Family, SymbolVersion};
use qrcode_generator::{Segment, Symbol, micro, qr, rmqr};

/// Xorshift generator, so runs are reproducible without another dependency.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    pub fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next() >> 24) as u8).collect()
    }

    pub fn digits(&mut self, n: usize) -> String {
        (0..n)
            .map(|_| char::from(b'0' + self.below(10) as u8))
            .collect()
    }

    pub fn alphanumeric(&mut self, n: usize) -> String {
        const SET: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:";
        (0..n)
            .map(|_| char::from(SET[self.below(SET.len())]))
            .collect()
    }

    /// `count` distinct values below `n`.
    pub fn distinct(&mut self, count: usize, n: usize) -> Vec<usize> {
        assert!(count <= n);
        let mut pool: Vec<usize> = (0..n).collect();
        for i in 0..count {
            let j = i + self.below(n - i);
            pool.swap(i, j);
        }
        pool.truncate(count);
        pool
    }
}

#[rustfmt::skip]
pub const RMQR_VERSIONS: [rmqr::Version; 32] = {
    use rmqr::Version::*;
    [
        R7x43, R7x59, R7x77, R7x99, R7x139,
        R9x43, R9x59, R9x77, R9x99, R9x139,
        R11x27, R11x43, R11x59, R11x77, R11x99, R11x139,
        R13x27, R13x43, R13x59, R13x77, R13x99, R13x139,
        R15x43, R15x59, R15x77, R15x99, R15x139,
        R17x43, R17x59, R17x77, R17x99, R17x139,
    ]
};

pub fn to_grid(symbol: &Symbol) -> ModuleGrid {
    ModuleGrid::from_bools(symbol.width(), symbol.height(), symbol.modules())
}

/// Encode with the oracle at exactly the given version and EC level. `mask`
/// is ignored for rMQR, which has only one.
pub fn encode(
    version: SymbolVersion,
    ec: EcLevel,
    mask: Option<u8>,
    segments: &[Segment],
) -> Result<ModuleGrid, qrcode_generator::EncodeError> {
    let symbol = match version {
        SymbolVersion::Qr(v) => {
            let level = match ec {
                EcLevel::L => qr::ErrorCorrection::Low,
                EcLevel::M => qr::ErrorCorrection::Medium,
                EcLevel::Q => qr::ErrorCorrection::Quartile,
                EcLevel::H => qr::ErrorCorrection::High,
            };
            let mut encoder = qr::Encoder::new(level)
                .version(qr::Version::new(v)?)
                .boost_error_correction(false);
            if let Some(mask) = mask {
                encoder = encoder.mask(qr::Mask::new(mask)?);
            }
            encoder.encode_segments(segments)?
        }
        SymbolVersion::Micro(v) => {
            let level = match (v, ec) {
                (1, _) => micro::ErrorCorrection::DetectionOnly,
                (_, EcLevel::L) => micro::ErrorCorrection::Low,
                (_, EcLevel::M) => micro::ErrorCorrection::Medium,
                _ => micro::ErrorCorrection::Quartile,
            };
            let micro_version = [
                micro::Version::M1,
                micro::Version::M2,
                micro::Version::M3,
                micro::Version::M4,
            ][v as usize - 1];
            let mut encoder = micro::Encoder::new(level)
                .version(micro_version)
                .boost_error_correction(false);
            if let Some(mask) = mask {
                encoder = encoder.mask(micro::Mask::new(mask)?);
            }
            encoder.encode_segments(segments)?
        }
        SymbolVersion::Rmqr(v) => {
            let level = if ec == EcLevel::H {
                rmqr::ErrorCorrection::High
            } else {
                rmqr::ErrorCorrection::Medium
            };
            rmqr::Encoder::new(level)
                .version(RMQR_VERSIONS[v as usize])
                .boost_error_correction(false)
                .encode_segments(segments)?
        }
        // No encoder for Model 1 exists; tests/model1.rs builds its own
        // symbols and checks them against zxing-cpp's reader.
        SymbolVersion::Model1(_) => panic!("the oracle encoder has no QR Code Model 1"),
    };
    Ok(to_grid(&symbol))
}

pub fn layout(version: SymbolVersion, ec: EcLevel) -> SymbolLayout {
    SymbolLayout::new(Arc::new(FunctionMap::new(version)), ec).expect("EC level exists")
}

pub fn decode_plain(grid: &ModuleGrid, layout: &SymbolLayout, mask: u8) -> DecodeReport {
    decode(grid, layout, mask, &Pins::new(), &DecodeOptions::default())
}

/// Byte-mode payload of a report; `None` for bytes that are not known.
pub fn payload_bytes(report: &DecodeReport) -> Vec<Option<u8>> {
    report
        .parse
        .segments
        .iter()
        .filter_map(|s| match &s.content {
            SegmentContent::Bytes(cells) => Some(cells.iter().map(|c| c.value)),
            _ => None,
        })
        .flatten()
        .collect()
}

/// Make every module of a codeword unknown.
pub fn erase_codeword(grid: &mut ModuleGrid, layout: &SymbolLayout, cw: CwId) {
    for pos in layout.codeword_modules(cw) {
        grid.set(pos.x as usize, pos.y as usize, Module::Unknown);
    }
}

/// Flip one module of a codeword, chosen by `bit` (0 is the first module).
pub fn corrupt_codeword(grid: &mut ModuleGrid, layout: &SymbolLayout, cw: CwId, bit: usize) {
    let modules = layout.codeword_modules(cw);
    let pos = modules[bit % modules.len()];
    let (x, y) = (pos.x as usize, pos.y as usize);
    grid.set(x, y, grid.get(x, y).inverted());
}

/// Largest byte-mode payload that fits, less `spare` bytes.
pub fn byte_capacity(layout: &SymbolLayout, spare: usize) -> usize {
    let header = layout.stream.mode_bits as usize
        + layout
            .stream
            .count_bits(qr_forensics::symbol::tables::Mode::Byte)
            .unwrap_or(0) as usize;
    (layout.data_bits.saturating_sub(header) / 8).saturating_sub(spare)
}

/// A photo as the app loads it: the file's pixels as luminance, with the
/// default tone adjustments.
pub fn load_photo(path: &std::path::Path) -> Result<Luma, String> {
    let image = image::open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .to_rgba8();
    let image = qr_forensics::imaging::photo::normalize_size(image);
    let (width, height) = (image.width() as usize, image.height() as usize);
    let (_, luma) = Adjust::default().apply(image.as_raw(), width, height);
    Ok(luma)
}

/// What the photo pipeline made of one outline.
#[derive(Debug)]
pub struct PhotoDecode {
    pub detection: Detection,
    pub report: DecodeReport,
    pub grid: ModuleGrid,
    /// The grid was read as the other QR Code model, of this version,
    /// because its format bits said so and that read ranked better.
    pub reinterpreted: Option<SymbolVersion>,
}

impl PhotoDecode {
    pub fn text(&self) -> String {
        self.report.text.plain()
    }
}

/// The whole photo pipeline as the app runs it from Find in outline onward:
/// detect the symbol inside the loose outline, refine the corners with the
/// anchors held, refine the anchors, sample through the bent projection,
/// settle the format from the bits or by brute force, and decode; and,
/// when that is not trustworthy, the retries the decode arbitrates, as
/// `imaging::read` describes them. `None` when no symbol is found.
pub fn decode_photo(
    image: &Luma,
    family: Family,
    outline: [[f64; 2]; 4],
    params: &SampleParams,
) -> Option<PhotoDecode> {
    let start = match family {
        Family::Qr => SymbolVersion::Qr(2),
        Family::Micro => SymbolVersion::Micro(2),
        Family::Rmqr => SymbolVersion::Rmqr(11),
        Family::Model1 => SymbolVersion::Model1(2),
    };
    let result = read_outline(image, start, outline, params, &DecodeOptions::default())?;
    Some(PhotoDecode {
        detection: result.detection,
        report: result.report,
        grid: result.grid,
        reinterpreted: result.reinterpreted,
    })
}

/// `decode_photo` with a panic inside the pipeline caught and handed back
/// as its message, so that one photo which trips a bug does not end a run
/// over hundreds; the run's log still has the backtrace.
pub fn try_decode_photo(
    image: &Luma,
    family: Family,
    outline: [[f64; 2]; 4],
    params: &SampleParams,
) -> Result<Option<PhotoDecode>, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        decode_photo(image, family, outline, params)
    }))
    .map_err(|payload| {
        payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown panic".to_owned())
    })
}

/// The image formats the photo corpora hold, by extension; a `.txt` with
/// the expected text and an `.outline` sidecar sit beside each image.
pub const PHOTO_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp"];

/// Whether a path names one of the corpus's images.
pub fn is_photo_file(path: &std::path::Path) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    PHOTO_EXTENSIONS.contains(&ext.as_str())
}

/// The image turned clockwise by `turns` quarter turns, pixel for pixel.
/// ZXing's blackbox suite reads every image at its four turns, since a
/// symbol should read whichever way up the camera held it, and the same
/// is asked here.
pub fn rotate_luma(image: &Luma, turns: u8) -> Luma {
    let mut out = image.clone();
    for _ in 0..turns % 4 {
        let (w, h) = (out.width, out.height);
        // Clockwise, each column of the source becomes a row of the
        // result, read from the bottom of the source upward.
        let mut data = Vec::with_capacity(w * h);
        for row in 0..w {
            for col in 0..h {
                data.push(out.data[(h - 1 - col) * w + row]);
            }
        }
        out = Luma::new(h, w, data);
    }
    out
}

/// A point of the unturned image, in the pixel coordinates the outlines
/// use (the image spans `0..width` by `0..height`), in the image turned
/// clockwise by `turns` quarter turns.
pub fn rotate_point(point: [f64; 2], turns: u8, width: usize, height: usize) -> [f64; 2] {
    let (mut p, mut dims) = (point, [width as f64, height as f64]);
    for _ in 0..turns % 4 {
        // The top-left corner goes to the top right, so x becomes y and y
        // counts from the far edge.
        p = [dims[1] - p[1], p[0]];
        dims.swap(0, 1);
    }
    p
}

/// An outline turned with its photo: the corners mapped into the turned
/// image and put back in the order a hand would draw them there, top-left
/// first. A clockwise turn carries the old bottom-left corner to the top
/// left, so the order shifts by one place a turn.
pub fn rotate_outline(
    outline: [[f64; 2]; 4],
    turns: u8,
    width: usize,
    height: usize,
) -> [[f64; 2]; 4] {
    let mut out = outline.map(|p| rotate_point(p, turns, width, height));
    out.rotate_right(usize::from(turns % 4));
    out
}

/// The angle a number of quarter turns is written as in a sidecar.
pub fn degrees(turns: u8) -> u16 {
    u16::from(turns % 4) * 90
}

/// Quarter turns from an angle in a sidecar.
pub fn turns_from_degrees(text: &str) -> Result<u8, String> {
    match text.trim() {
        "0" => Ok(0),
        "90" => Ok(1),
        "180" => Ok(2),
        "270" => Ok(3),
        other => Err(format!("rotation {other:?} is not a quarter turn")),
    }
}

/// A symbol version named the way the sidecars and summaries do: `V3`,
/// `M4`, `R7x43`, `Model1-V3`.
pub fn version_label(version: SymbolVersion) -> String {
    match version {
        SymbolVersion::Qr(_) => format!("V{version}"),
        SymbolVersion::Model1(_) => format!("Model1-V{version}"),
        _ => version.to_string(),
    }
}

/// What one decode made of a photo, measured against the text it holds.
/// ZXing's blackbox suite keeps the same distinction: a wrong text is
/// counted apart from, and as worse than, no text at all, because a reader
/// that is confidently wrong misleads where one that gives up does not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The expected text, byte for byte, in a report that vouches for it:
    /// what the app shows as read, and the command line prints as one.
    Read,
    /// A text the report vouches for that is not the expected one.
    Misread,
    /// A symbol was found but the decode did not reach a text it could
    /// vouch for; the string says how far it got. The expected text out
    /// of a decode that does not vouch for it belongs here: a reader of
    /// the report cannot tell it from the wrong text of another partial
    /// decode.
    Failed(String),
    /// No symbol found through the outline.
    NotFound,
}

impl Outcome {
    pub fn of(result: Option<&PhotoDecode>, expected: &str) -> Self {
        let Some(d) = result else {
            return Self::NotFound;
        };
        let text = d.text();
        if d.report.is_trustworthy() {
            if text == expected {
                Self::Read
            } else {
                Self::Misread
            }
        } else {
            let mut stage = format!(
                "{}, {} of {} blocks solved",
                version_label(d.detection.version),
                d.report.blocks_solved(),
                d.report.blocks.len()
            );
            // Some text came out of a partial decode, so the failure is
            // late in the pipeline rather than at detection.
            if text == expected {
                stage.push_str(", text right but not vouched for");
            } else if !text.is_empty() {
                stage.push_str(", text differs");
            }
            Self::Failed(stage)
        }
    }

    /// As `of`, for an attempt that may have ended in a panic, which is a
    /// failure of its own kind.
    pub fn of_attempt(attempt: &Result<Option<PhotoDecode>, String>, expected: &str) -> Self {
        match attempt {
            Ok(result) => Self::of(result.as_ref(), expected),
            Err(panic) => Self::Failed(format!("panic: {panic}")),
        }
    }

    pub fn is_read(&self) -> bool {
        matches!(self, Self::Read)
    }

    /// A short account, as a sidecar's `; failure:` line records it.
    pub fn describe(&self) -> String {
        match self {
            Self::Read => "read".to_owned(),
            Self::Misread => "misread".to_owned(),
            Self::Failed(stage) => stage.clone(),
            Self::NotFound => "not found".to_owned(),
        }
    }
}

/// A sidecar describing a photo in the corpus: the symbol family, the
/// quarter turns of the photo at which the pipeline is expected to read
/// it, how far it got at the turns it does not, and a loose outline around
/// the symbol in the pixels of the unturned photo at its normalised size.
///
/// ```text
/// ; family: qr
/// ; status: partial
/// ; rotations: 0 180
/// ; failure: 90 not found
/// ; failure: 270 V3, 2 of 4 blocks solved, text differs
/// x y                      (top-left, then clockwise)
/// ```
///
/// `status` is derived when writing: `pass` when all four turns read,
/// `partial` when some do, `known-failure` when none does. A sidecar from
/// before turns were recorded has no `rotations` line and keeps its
/// meaning: `pass` reads the photo as it is, `known-failure` not at all.
#[derive(Clone, Debug, PartialEq)]
pub struct PhotoSidecar {
    pub family: Family,
    /// Quarter turns clockwise, 0 to 3, in order, at which the photo reads.
    pub rotations: Vec<u8>,
    /// For each turn that does not read, how far the decode got.
    pub failures: Vec<(u8, String)>,
    pub outline: [[f64; 2]; 4],
}

impl PhotoSidecar {
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut family = None;
        let mut status = "pass".to_owned();
        let mut rotations = None;
        let mut failures = Vec::new();
        let mut points = Vec::new();
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
            if let Some(value) = line.strip_prefix("; family:") {
                family = Some(match value.trim() {
                    "qr" => Family::Qr,
                    "micro" => Family::Micro,
                    "rmqr" => Family::Rmqr,
                    "model1" => Family::Model1,
                    other => return Err(format!("unknown family {other:?}")),
                });
            } else if let Some(value) = line.strip_prefix("; status:") {
                status = value.trim().to_owned();
            } else if let Some(value) = line.strip_prefix("; rotations:") {
                let mut turns = value
                    .split_whitespace()
                    .map(turns_from_degrees)
                    .collect::<Result<Vec<_>, _>>()?;
                turns.sort_unstable();
                turns.dedup();
                rotations = Some(turns);
            } else if let Some(value) = line.strip_prefix("; failure:") {
                let value = value.trim();
                let (angle, stage) = value.split_once(char::is_whitespace).unwrap_or((value, ""));
                failures.push((turns_from_degrees(angle)?, stage.trim().to_owned()));
            } else if line.starts_with(';') {
                // Lines this reader does not know are kept for whoever does.
                continue;
            } else {
                let mut parts = line.split_whitespace().map(|v| v.parse::<f64>());
                match (parts.next(), parts.next()) {
                    (Some(Ok(x)), Some(Ok(y))) => points.push([x, y]),
                    _ => return Err(format!("bad outline line {line:?}")),
                }
            }
        }
        let outline: [[f64; 2]; 4] = points
            .try_into()
            .map_err(|p: Vec<_>| format!("{} outline points", p.len()))?;
        // Without a rotations line the status alone says what is expected,
        // and only of the photo as it is.
        let rotations = rotations.unwrap_or_else(|| {
            if status == "known-failure" {
                Vec::new()
            } else {
                vec![0]
            }
        });
        Ok(Self {
            family: family.ok_or("no family")?,
            rotations,
            failures,
            outline,
        })
    }

    /// Whether the photo is expected to read turned clockwise by `turns`
    /// quarter turns.
    pub fn reads_at(&self, turns: u8) -> bool {
        self.rotations.contains(&(turns % 4))
    }

    /// Whether the photo is expected to read as it is.
    pub fn expected_to_pass(&self) -> bool {
        self.reads_at(0)
    }

    pub fn status(&self) -> &'static str {
        match self.rotations.len() {
            4 => "pass",
            0 => "known-failure",
            _ => "partial",
        }
    }

    pub fn to_text(&self) -> String {
        let family = match self.family {
            Family::Qr => "qr",
            Family::Micro => "micro",
            Family::Rmqr => "rmqr",
            Family::Model1 => "model1",
        };
        let mut out = format!("; family: {family}\n; status: {}\n", self.status());
        if !self.rotations.is_empty() {
            let angles: Vec<String> = self
                .rotations
                .iter()
                .map(|&t| degrees(t).to_string())
                .collect();
            out.push_str(&format!("; rotations: {}\n", angles.join(" ")));
        }
        for (turns, stage) in &self.failures {
            out.push_str(&format!("; failure: {} {stage}\n", degrees(*turns)));
        }
        for [x, y] in self.outline {
            out.push_str(&format!("{x} {y}\n"));
        }
        out
    }
}
