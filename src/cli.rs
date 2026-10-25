//! A headless reader: `qr-forensics decode [options] <image>...`
//! reads every image given as the window reads one, searching the whole
//! photo for symbols of every family as Search whole photo does or
//! taking an outline from the command line as Find in outline does, and
//! prints what it finds, one symbol a line. ZXing,
//! zxing-cpp and OpenCV each ship such a tool; this one runs the pipeline
//! the window runs, so a batch of photos can be read, and a result
//! reproduced, without a display.
//!
//! ```text
//! qr-forensics decode [--family qr|micro|rmqr|model1]
//!                     [--outline x,y,x,y,x,y,x,y | --placed x,y,x,y,x,y,x,y]
//!                     [--hex | --escaped] [--charset name] [--corners]
//!                     [--details] [--grid] [--metrics] [--partial]
//!                     [--surface] [--enlarge n] [--sharpen light|strong|deblur]
//!                     [--full-search] [--] <image>...
//! ```
//!
//! An option that takes a value takes the next argument, or what follows
//! an equals sign, as `--outline=1,2,3,4,5,6,7,8`. `--` ends the options,
//! as it does for the shell's own tools: every argument after it names an
//! image, whatever it begins with.
//!
//! `--corners` adds to each line where the symbol lies: its four corners
//! in the pixels `--outline` takes, top-left first as the symbol has
//! them, so that a symbol a search found can be read again through its
//! own outline, and a batch checked for where its symbols were, as
//! OpenCV's `decodeMulti` hands back each symbol's points beside its
//! text and BoofCV each symbol's bounds. `--grid` prints the modules as
//! read below the line, a row to a line and each indented by two
//! spaces, `#` for dark, `.` for light and `?` for a module that could
//! not be read, as OpenCV hands back the straightened symbol; a
//! light-on-dark symbol's grid is dark where the symbol prints dark.
//!
//! `--details` adds what the decode settled on the way to the message:
//! the error correction level and mask, `risk 1 in 2^n` once every block
//! is solved (an upper bound on the chance the message is wrong: the
//! blocks' chances added up), the character set the bytes were read in,
//! the symbology identifier, the symbol's place in a structured append
//! sequence when it has one, and how many codewords of padding were
//! pinned when the read took that, and as what when the padding is not
//! the standard's, and what the photo was read through when that was not
//! the photo as it is, enlarged or sharpened, and `threshold surface`
//! when its modules were read against one, as OpenCV's detector hands
//! back each symbol's ECI and BoofCV its level, mask and encoding.
//! If every block is confirmed but their combined risk is worse than the
//! 1 in 2^8 each block must meet, the read still stands, and a warning
//! is printed to standard error; the read's own line is unchanged.
//! `--charset` names the character set to read byte data in when the
//! symbol declares none, where it is otherwise guessed: `utf-8`,
//! `iso-8859-1`, `windows-1252`, `shift_jis`, `gb18030`, `big5`, `euc-kr`,
//! any other name the ECI table has, or an ECI assignment number. A set
//! the symbol declares with an ECI designator is read as declared
//! whatever is named, as ZXing takes a character set hint.
//! `--partial` prints the symbols that did not decode beside those that
//! did, where they are otherwise printed only when none did: BoofCV
//! returns its failures beside its detections for the same reason, that
//! a damaged symbol among sound ones is still a symbol. With them go the
//! finder patterns a search of the whole photo found standing as one
//! symbol's, its format word reading outright beside them, and could fit
//! no symbol to, a line for each set: the family they were taken for,
//! how many, `no symbol fitted`, and the outline the search proposed
//! about them, which `--outline` takes as it is. A read through an
//! outline tries more than a search of the whole photo does.
//!
//! `--metrics` prints read-quality figures under each symbol's line,
//! indented two spaces, for measuring the pipeline over a corpus
//! ([`ReadMeasures`]). Only modules in confirmed blocks count, since only
//! there does the decode settle their true colours:
//!
//! - `modules`: how many; `misread` and `unknown`: how many were read as
//!   the wrong colour, or left unknown.
//! - `blocks`: how many blocks were confirmed; `load_mean` and
//!   `load_max`: the mean and largest share of a block's error correction
//!   used.
//! - `calibration`: one `bits:modules:misread:expected` entry per whole
//!   number of bits of reliability: the modules read at it, how many were
//!   wrong, and how many it predicted would be.
//!
//! ```text
//!   metrics: modules=1234 misread=12 unknown=30 blocks=4 load_mean=0.210 load_max=0.550 calibration=1:20:9:10.000000,2:41:5:10.250000
//! ```
//!
//! A read that confirmed no block prints no metrics line.
//!
//! `--placed` takes the symbol's four corners, as you'd place them in the
//! Align stage, and reads from them with no search. It tries the few
//! versions that best fit the corners, then refines, samples and decodes
//! each as it would a placement the search found, retrying on an
//! enlarged, sharpened copy if the modules are small, and on a deblurred
//! copy if edges blur by a fifth of a module or more. The corners can
//! start anywhere and go either way round. On a benchmark with
//! hand-marked corners this works as an oracle: a symbol that reads from
//! its marks but not by searching was lost by the search.
//!
//! `--surface` reads every symbol first against one threshold that lies
//! as a smooth surface across it, the window's Threshold surface, where
//! each module is otherwise judged against its neighbourhood first, then
//! against one level threshold, and against the surface last: for a
//! shadow or a fall-off of light across an evenly printed symbol. It is
//! not needed for such a symbol to be read, and the other rules are
//! still tried when the surface does not decode; it sets what is tried
//! first, and reproduces a read that `--details` says was made against
//! the surface.
//!
//! `--enlarge` and `--sharpen` read every symbol off a copy of the
//! photo about it, sharpened by an unsharp mask and enlarged two, three
//! or four times with Magic Kernel Sharp 2013: for modules a pixel or
//! two across, or a symbol out of focus. The window makes the same of
//! the whole photo, with Enlarge and Sharpen under Photo, where `light`
//! is 0.5 and `strong` 1; here the outline stays in the photo's pixels.
//! `--sharpen deblur` deconvolves instead of masking: it measures the
//! blur's width and direction from edges around the symbol and undoes
//! it, on a copy scaled so the blur is at most a couple of pixels wide.
//! Use it for motion blur, or a photo too far out of focus for a mask to
//! help. The window's Deblur filter, under Photo, does the same to the
//! whole photo.
//! Neither option is needed for such a symbol to be read: a failed read
//! is retried on an enlarged, sharpened copy when the modules are small
//! (as wechat_qrcode retries a small crop enlarged), and on a deblurred
//! copy when edges blur by a fifth of a module or more, at any module
//! size. `--details` says which copy each symbol was read from, if not
//! the original photo. The options only change what is tried first,
//! which helps to reproduce a read.
//!
//! `--full-search` searches a photo at the full size a search goes to
//! from the start, where the search otherwise looks at a copy no larger
//! than a thousand pixels on its longer side first, finds what it shows
//! again at full size, and then asks the photo at full size only what
//! finder patterns its regions show beside those symbols, which on a
//! large photo is many times faster. The full size a search goes to is 4096 pixels on the longer
//! side: a larger photo is searched in a copy reduced to that, and the
//! symbols the copy shows are read from the photo's own pixels. A symbol
//! too small to show in the copy is read through `--outline`: what is
//! inside an outline is placed and read in the photo's own pixels,
//! however large the photo.
//!
//! The outline is drawn top-left first and clockwise, in the photo's own
//! pixels, upright as its EXIF orientation has it. A photo under 256
//! pixels on its shorter side is worked on enlarged by a whole factor,
//! as the window's Enlarge starts for one, and its outline is taken and
//! its corners are printed in its own pixels all the same, as OpenCV and
//! wechat_qrcode hand back a symbol's points in the pixels of the image
//! they were given whatever they read it at: to as many more decimal
//! places as keep the tenth of a pixel of the copy. What is inside it is read as
//! the window's Find in outline reads it: as a QR Code first and, when that
//! does not decode, as a Micro QR or rMQR symbol too, on the evidence
//! Find in outline asks of another family, the best of the reads being
//! kept. `--family` names the one family to look for, with an outline or
//! without one, and then nothing else is tried. Without an outline the
//! whole photo is searched, and every symbol that reads is printed; when
//! none does, the partial reads are printed instead, since the search
//! offers what might be a symbol as readily as what is one. Each line
//! names the image, the family and version, whether the symbol was light
//! on dark or mirrored, how the read stands, with `--details` what it
//! settled, with `--corners` where it lies, and the message: as text,
//! but for its control characters other than tab, spelt out as `<CR>`
//! and `<LF>` are, so that a line break inside a message keeps the
//! symbol to its line; with `--escaped` every control character and the
//! unreadable mark spelt out; or with `--hex` the message's bytes. A
//! read stands as `read` when every block is confirmed
//! and the message parses; `read, padding pinned` when it is that only
//! with the filler after a short message taken as known, which the
//! standard fixes, and an encoder that fills otherwise fixes as well
//! once what is left of its filler shows which it is, so that damage
//! there costs no error correction: the window's Pin the padding, tried
//! of every decode that is no read, and kept when every block then
//! confirms and the segment headers bear out where the padding lies;
//! `unconfirmed` when every block solved and the message parses, but
//! some block's chance of being wrong is worse than 1 in 256. Each
//! spare error correction codeword is a 1 in 256 check, so a block that
//! spent them all filling in unreadable codewords can't be confirmed.
//! Locating wrong codewords counts too, so a block with nothing unknown
//! can be confirmed even when it corrected as many as it can. Each
//! doubling of the decodes tried, and a photo read tries many, costs a
//! bit. In the smallest symbols, a solution that uses the codewords the
//! standard reserves for detecting misdecodes is never trusted. A read
//! is `malformed` when every block is confirmed and the message does
//! not parse to its end, and `partial` otherwise, with the blocks that
//! solved.
//!
//! Symbols that carry a structured append header are parts of one
//! message, and after the symbols of an image a line gives each message
//! its parts make up, in order, with the parts found and whether the
//! parity the headers claim is the parity of the bytes, as OpenCV's
//! `decodeMulti` joins the parts it finds in one image, unchecked. With
//! `--details`, once every part is in and solved, the line also gives the
//! message's `risk 1 in 2^n`: the parts' chances added up.
//!
//! The exit status is 0 when every image gave a trustworthy read, 1 when
//! one did not, and 2 for a bad command line or an image that would not
//! load. `--help` prints the usage and exits with 0. When standard output
//! is closed, as `| head` closes it, the run stops there without
//! complaint, with the status of the images read so far.

use std::ffi::OsString;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use qr_forensics::decode::padding::{Filler, pinned_filler};
use qr_forensics::decode::pipeline::{DecodeOptions, VERIFIED_RISK_BITS};
use qr_forensics::decode::sequence::{ParityCheck, Part, Sequence};
use qr_forensics::decode::text::{Charset, escape_controls};
use qr_forensics::imaging::Luma;
use qr_forensics::imaging::adjust::Enlarge;
use qr_forensics::imaging::detect::{Lead, SearchParams, search_for_families};
use qr_forensics::imaging::enhance::{Enhance, Sharpen};
use qr_forensics::imaging::measure::ReadMeasures;
use qr_forensics::imaging::photo;
use qr_forensics::imaging::read::{ReadResult, read_detection, read_outline, read_placed};
use qr_forensics::imaging::sample::{Rule, SampleParams};
use qr_forensics::symbol::function_map::FunctionMap;
use qr_forensics::symbol::layout::SymbolLayout;
use qr_forensics::symbol::{Family, SymbolVersion};
use qr_forensics::work;

const USAGE: &str = "usage: qr-forensics decode [--family qr|micro|rmqr|model1] \
                     [--outline x,y,x,y,x,y,x,y | --placed x,y,x,y,x,y,x,y] \
                     [--hex | --escaped] [--charset name] [--corners] [--details] [--grid] [--metrics] [--partial] \
                     [--surface] [--enlarge n] [--sharpen light|strong|deblur] [--full-search] [--] <image>...\n\
                     A message is printed on its symbol's line, its control characters other than tab spelt out as \
                     <CR>, <LF> and so on; --escaped spells out every one and the unreadable mark, --hex prints the \
                     bytes.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Show {
    Text,
    Hex,
    Escaped,
}

struct Options {
    family: Option<Family>,
    outline: Option<[[f64; 2]; 4]>,
    /// The symbol's own corners, read through with no search.
    placed: Option<[[f64; 2]; 4]>,
    show: Show,
    /// The character set byte data is read in when the symbol declares
    /// none.
    charset: Charset,
    /// Print where each symbol lies.
    corners: bool,
    /// Print what each decode settled: level, mask and character set.
    details: bool,
    /// Print the modules as read below each line.
    grid: bool,
    /// Print what each read made of the modules its decode came to know.
    metrics: bool,
    /// Print the symbols that did not decode beside those that did.
    partial: bool,
    /// Read each symbol against a threshold surface first.
    surface: bool,
    /// Read each symbol off the photo enlarged or sharpened first.
    enhance: Enhance,
    search: SearchParams,
    images: Vec<OsString>,
}

/// The character set a name on the command line means: a name as the app
/// lists them, whatever its case and with or without its hyphens, or an
/// ECI assignment number.
fn charset_named(name: &str) -> Option<Charset> {
    let plain = |text: &str| text.chars().filter(char::is_ascii_alphanumeric).collect::<String>().to_ascii_lowercase();
    let wanted = plain(name);
    Charset::ALL
        .into_iter()
        .find(|c| plain(c.label()) == wanted)
        .or_else(|| name.parse().ok().map(Charset::Eci).filter(|c| c.label() != "unsupported"))
}

/// Why a command line is not run: it asked for the usage, or it is wrong,
/// and how.
#[derive(Debug, PartialEq, Eq)]
enum Unparsed {
    Help,
    Bad(String),
}

fn parse(args: &[OsString]) -> Result<Options, Unparsed> {
    match parse_options(args) {
        Ok(Some(options)) => Ok(options),
        Ok(None) => Err(Unparsed::Help),
        Err(message) => Err(Unparsed::Bad(message)),
    }
}

/// The options the command line gives, or `None` when it asks for the
/// usage instead.
fn parse_options(args: &[OsString]) -> Result<Option<Options>, String> {
    let mut options = Options {
        family: None,
        outline: None,
        placed: None,
        show: Show::Text,
        charset: Charset::Auto,
        corners: false,
        details: false,
        grid: false,
        metrics: false,
        partial: false,
        surface: false,
        enhance: Enhance::NONE,
        search: SearchParams::default(),
        images: Vec::new(),
    };
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        let text = arg.to_string_lossy();
        // `--` ends the options: what follows names images.
        if text == "--" {
            options.images.extend(args.cloned());
            break;
        }
        // `--name=value` is `--name value`.
        let (name, inline) = match text.split_once('=') {
            Some((name, value)) if name.starts_with("--") => (name, Some(value)),
            _ => (text.as_ref(), None),
        };
        let mut value = |what: &str| -> Result<String, String> {
            match inline {
                Some(value) => Ok(value.to_owned()),
                None => args.next().map(|a| a.to_string_lossy().into_owned()).ok_or_else(|| format!("{name} needs {what}")),
            }
        };
        let flag = || if inline.is_some() { Err(format!("{name} takes no value")) } else { Ok(()) };
        match name {
            "--family" => {
                let value = value("a value")?;
                options.family = Some(match value.as_str() {
                    "qr" => Family::Qr,
                    "micro" => Family::Micro,
                    "rmqr" => Family::Rmqr,
                    "model1" => Family::Model1,
                    other => return Err(format!("unknown family {other:?}")),
                });
            }
            "--outline" => {
                options.outline = Some(four_points("an", "outline", &value("eight numbers")?)?);
            }
            "--placed" => {
                options.placed = Some(four_points("a", "placement", &value("eight numbers")?)?);
            }
            "--hex" => {
                flag()?;
                if options.show == Show::Escaped {
                    return Err("--hex and --escaped each print the message one way; give one of them".to_owned());
                }
                options.show = Show::Hex;
            }
            "--escaped" => {
                flag()?;
                if options.show == Show::Hex {
                    return Err("--hex and --escaped each print the message one way; give one of them".to_owned());
                }
                options.show = Show::Escaped;
            }
            "--charset" => {
                let value = value("a character set")?;
                options.charset = charset_named(&value).ok_or_else(|| format!("unknown character set {value:?}"))?;
            }
            "--corners" => {
                flag()?;
                options.corners = true;
            }
            "--details" => {
                flag()?;
                options.details = true;
            }
            "--grid" => {
                flag()?;
                options.grid = true;
            }
            "--metrics" => {
                flag()?;
                options.metrics = true;
            }
            "--partial" => {
                flag()?;
                options.partial = true;
            }
            "--surface" => {
                flag()?;
                options.surface = true;
            }
            "--enlarge" => {
                let value = value("a factor")?;
                options.enhance.enlarge = match value.parse() {
                    Ok(factor) if (1..=Enhance::MAX_ENLARGE).contains(&factor) => factor,
                    _ => return Err(format!("--enlarge takes a factor from 1 to {}, not {value}", Enhance::MAX_ENLARGE)),
                };
            }
            "--sharpen" => {
                let value = value("light, strong or deblur")?;
                options.enhance.sharpen = match value.as_str() {
                    "off" => Sharpen::Off,
                    "light" => Sharpen::Light,
                    "strong" => Sharpen::Strong,
                    "deblur" => Sharpen::Deblur,
                    other => return Err(format!("--sharpen takes light, strong or deblur, not {other}")),
                };
            }
            "--full-search" => {
                flag()?;
                options.search.coarse_to_fine = false;
            }
            "--help" | "-h" => return Ok(None),
            other if other.starts_with("--") => return Err(format!("unknown option {other}")),
            _ => options.images.push(arg.clone()),
        }
    }
    if options.images.is_empty() {
        return Err("no image given".to_owned());
    }
    if options.outline.is_some() && options.placed.is_some() {
        return Err("--outline and --placed can't be used together".to_owned());
    }
    Ok(Some(options))
}

/// Four points in the photo's pixels, as `--outline` and `--placed` take
/// them: eight numbers with commas between, `x` before `y`. `what` names
/// them in what is said of a bad one, after `article`.
fn four_points(article: &str, what: &str, value: &str) -> Result<[[f64; 2]; 4], String> {
    let numbers: Vec<f64> = value.split(',').map(|v| v.trim().parse::<f64>().map_err(|e| format!("{what} {value:?}: {e}"))).collect::<Result<_, _>>()?;
    let [x0, y0, x1, y1, x2, y2, x3, y3] = numbers[..] else {
        return Err(format!("{article} {what} is eight numbers, not {}", numbers.len()));
    };
    if let Some(bad) = numbers.iter().find(|n| !n.is_finite()) {
        return Err(format!("{what} {value:?}: {bad} is no pixel position"));
    }
    Ok([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
}

/// A photo as the app loads it: the file's pixels, enlarged when the
/// photo is small and never shrunk, as luminance, with the default tone
/// adjustments; and how many times its own size it was loaded at.
fn load(path: &Path) -> Result<(Luma, f64), String> {
    let image = photo::open(path).map_err(|e| e.to_string())?;
    let times = Enlarge::for_photo(image.width(), image.height()).factor();
    Ok((photo::luma_as_loaded(image), times))
}

/// Four points of a photo worked on at `times` its own size, as
/// `--corners` prints them and `--outline` takes them back: in the
/// photo's own pixels, to the tenth of a pixel of the copy worked on,
/// which of a photo enlarged is a decimal place more for each ten times.
/// A symbol in a photo of thirty pixels a side is placed to a fraction
/// of a module there, and a tenth of the photo's own pixel is a tenth of
/// a module of it.
fn at(points: &[[f64; 2]; 4], times: f64) -> String {
    let places = 1 + if times > 1.0 { (times.log10().ceil() as usize).max(1) } else { 0 };
    let at: Vec<String> = points.iter().flat_map(|p| [p[0], p[1]]).map(|v| format!("{:.places$}", v / times)).collect();
    at.join(",")
}

/// The version a search of `family` starts from, as the window starts.
fn start_version(family: Family) -> SymbolVersion {
    match family {
        Family::Qr => SymbolVersion::Qr(2),
        Family::Micro => SymbolVersion::Micro(2),
        Family::Rmqr => SymbolVersion::Rmqr(11),
        Family::Model1 => SymbolVersion::Model1(2),
    }
}

/// How a read stands, as the module describes: `read`, or what keeps it
/// from being one.
pub(crate) fn standing(result: &ReadResult) -> String {
    let report = &result.report;
    let blocks = report.blocks.len();
    if result.is_trustworthy() && result.padding_pinned() {
        "read, padding pinned".to_owned()
    } else if result.is_trustworthy() {
        "read".to_owned()
    } else if report.is_complete() && report.score.stream_valid {
        format!("unconfirmed, {} of {blocks} blocks confirmed", report.blocks_verified())
    } else if report.blocks_verified() == blocks {
        format!("malformed, {blocks} of {blocks} blocks solved")
    } else {
        format!("partial, {} of {blocks} blocks solved", report.blocks_solved())
    }
}

/// What a decode settled on the way to the message, as `--details`
/// prints it: the level and mask, the character set byte data was read
/// in, the symbology identifier, the place in a structured append
/// sequence, the padding that was pinned, what `--enlarge`, `--sharpen`
/// and `--surface` would have to ask for to read the symbol the same
/// again at the first try, and `unsure modules kept` when the read
/// stands only with the modules its unsure band would leave unknown
/// kept at the colour they fell on, which the window reads again with
/// the band at nought. The standard's filler goes without saying, and
/// another is named.
fn details(result: &ReadResult) -> String {
    let (version, report) = (result.version(), &result.report);
    let mut parts = vec![format!("EC {}", version.ec_label(report.ec))];
    if version.mask_count() > 1 {
        parts.push(format!("mask {}", report.mask));
    }
    if let Some(bits) = report.message_risk_bits() {
        parts.push(format!("risk 1 in 2^{bits}"));
    }
    parts.extend(report.text.encoding.clone());
    parts.extend(report.text.symbology.clone());
    if let Some(part) = Part::from_report(report, String::new()) {
        parts.push(format!("part {} of {}", part.index, part.total));
    }
    if result.padding_pinned() {
        let mut pinned = format!("{} padding codewords pinned", result.pins.len());
        let layout = SymbolLayout::new(Arc::new(FunctionMap::new(version)), report.ec);
        let filler = layout.and_then(|layout| pinned_filler(&layout, report, &result.pins));
        if let Some(filler) = filler.filter(|&filler| filler != Filler::Standard) {
            pinned.push_str(&format!(" as {}", filler.label()));
        }
        parts.push(pinned);
    }
    if !result.enhance.is_none() {
        parts.push(format!("photo {}", result.enhance));
    }
    if result.rule == Rule::Surface {
        parts.push("threshold surface".to_owned());
    }
    if result.unsure_kept {
        parts.push("unsure modules kept".to_owned());
    }
    parts.join(", ")
}

/// The warning for a read whose every block is confirmed by itself and
/// whose message, the blocks' chances of being wrong added up, is not
/// ([`DecodeReport::message_risk_falls_short`]): printed to standard
/// error, so that the line the read prints is as it was. None for any
/// other read.
fn risk_warning(path: &Path, result: &ReadResult) -> Option<String> {
    let report = &result.report;
    if !result.is_trustworthy() || !report.message_risk_falls_short() {
        return None;
    }
    let bits = report.message_risk_bits()?;
    Some(format!(
        "{}: warning: all {} blocks are confirmed, but their combined chance of being wrong is 1 in 2^{bits}, \
         worse than the 1 in 2^{VERIFIED_RISK_BITS} required of each block",
        path.display(),
        report.blocks.len()
    ))
}

/// The message as the default output prints it: as it is, but for its
/// control characters other than tab, spelt out as `--escaped` spells
/// them, so that a line break inside a message, as a vCard or an
/// iCalendar entry has, keeps the symbol to its one line, and a row of
/// `--grid` below it is never taken for part of the message. Tab stays,
/// a field separator a line keeps, and so does the unreadable mark,
/// which `--escaped` alone spells out.
fn one_line(text: &str) -> String {
    text.chars()
        .map(|ch| if ch.is_control() && ch != '\t' { escape_controls(ch.encode_utf8(&mut [0; 4])) } else { ch.to_string() })
        .collect()
}

/// One symbol's line: the image, what was found, how the read stands,
/// what it settled when `details` is asked, where it lies when `corners`
/// is, in the pixels of a photo read at `times` its size, and the
/// message.
fn describe(path: &Path, result: &ReadResult, options: &Options, times: f64) -> String {
    let version = result.version();
    let mut what = format!("{} {}", version.family().label(), version);
    if result.detection.inverted {
        what.push_str(", light on dark");
    }
    if result.detection.mirrored {
        what.push_str(", mirrored");
    }
    let message = match options.show {
        Show::Text => one_line(&result.text()),
        Show::Hex => result.report.text.hex(),
        Show::Escaped => result.report.text.escaped(),
    };
    let mut line = format!("{}: {what}: {}", path.display(), standing(result));
    if options.details {
        line.push_str(&format!(": {}", details(result)));
    }
    if options.corners {
        line.push_str(&format!(": at {}", at(&result.detection.corners, times)));
    }
    format!("{line}: {message}")
}

/// A line for finder patterns the search could fit no symbol to: the
/// family they were taken for, how many there are, and the outline the
/// search proposed about them, always, since where they lie is all there
/// is to say of them, in the form `--outline` takes it back.
fn describe_lead(path: &Path, lead: &Lead, times: f64) -> String {
    format!("{}: {}: {} finder patterns, no symbol fitted: at {}", path.display(), lead.family.label(), lead.finders.len(), at(&lead.outline, times))
}

/// The messages the parts make up, one sequence for each count and
/// parity the parts' headers claim, as ZBar groups them: parts of two
/// messages photographed together are not parts of one.
fn sequences(parts: Vec<Part>) -> Vec<Sequence> {
    let mut all: Vec<Sequence> = Vec::new();
    for part in parts {
        let header = (part.total, part.parity);
        match all.iter_mut().find(|s| s.header() == Some(header)) {
            Some(sequence) => {
                sequence.add(part);
            }
            None => {
                let mut sequence = Sequence::default();
                sequence.add(part);
                all.push(sequence);
            }
        }
    }
    all
}

/// A sequence's line: the image, the parts found, what the parity says,
/// with `details` how rarely the message the parts make up is wrong
/// once every part is in ([`Sequence::risk_bits`]), and the parts'
/// messages in order, with each missing part marked.
fn describe_sequence(path: &Path, sequence: &Sequence, show: Show, details: bool) -> String {
    let total = sequence.header().map_or(0, |h| h.0);
    let found = format!("structured append, {} of {total} parts", sequence.len());
    let parity = match sequence.check() {
        ParityCheck::Complete { computed, claimed } if computed == claimed => ", parity verified".to_owned(),
        ParityCheck::Complete { computed, claimed } => format!(", parity {computed:02X} where the parts claim {claimed:02X}"),
        ParityCheck::Unreadable => ", parity unreadable".to_owned(),
        ParityCheck::Inconsistent => ", parts disagree".to_owned(),
        ParityCheck::Empty | ParityCheck::Incomplete { .. } => String::new(),
    };
    let risk = match sequence.risk_bits() {
        Some(bits) if details => format!(", risk 1 in 2^{bits}"),
        _ => String::new(),
    };
    let message = match show {
        Show::Text => one_line(&sequence.assembled().plain()),
        Show::Hex => sequence.hex(),
        Show::Escaped => sequence.assembled().escaped(),
    };
    format!("{}: {found}{parity}{risk}: {message}", path.display())
}

/// The modules of a grid as `--grid` prints them: a row to a line,
/// indented, `#` dark, `.` light and `?` unknown.
fn grid_lines(result: &ReadResult) -> Vec<String> {
    result
        .grid
        .to_text()
        .lines()
        .map(|row| format!("  {}", row.chars().map(|c| match c { '1' => '#', '0' => '.', other => other }).collect::<String>()))
        .collect()
}

/// Run the reader over the arguments after `decode`; the exit status.
pub fn run(args: &[OsString]) -> i32 {
    run_to(args, &mut std::io::stdout().lock())
}

/// A line for standard error, when it takes one: with it closed there is
/// no one to tell, and the status says what happened, where `eprintln!`
/// would end the run in a panic instead.
fn warn(message: &str) {
    let _ = writeln!(std::io::stderr(), "{message}");
}

/// [`run`], printing what is read to `out`. When `out` takes no more, as
/// a pipe whose reader has gone does not, the run stops quietly with the
/// status so far: `println!` would panic there, and there is no one left
/// to read the rest.
fn run_to(args: &[OsString], out: &mut impl Write) -> i32 {
    let options = match parse(args) {
        Ok(options) => options,
        Err(Unparsed::Help) => {
            // Asked for, the usage is the output, and the run went well.
            let _ = writeln!(out, "{USAGE}");
            return 0;
        }
        Err(Unparsed::Bad(message)) => {
            warn(&message);
            warn(USAGE);
            return 2;
        }
    };
    let asked = SampleParams { enhance: options.enhance, ..SampleParams::default() };
    let params = if options.surface { asked.under(Rule::Surface) } else { asked };
    let opts = DecodeOptions { charset: options.charset, derive_pins: true, ..DecodeOptions::default() };
    let mut status = 0;
    for image in &options.images {
        let path = Path::new(image);
        let (luma, times) = match load(path) {
            Ok(loaded) => loaded,
            Err(message) => {
                warn(&format!("{}: {message}", path.display()));
                status = status.max(2);
                continue;
            }
        };
        // The finder patterns a search fitted no symbol to: an outline
        // given is read, and nothing is searched for.
        let mut leads: Vec<Lead> = Vec::new();
        // An outline is given in the photo's own pixels, and read in
        // those the photo was loaded at; so is a placement.
        let placed = options.placed.map(|corners| corners.map(|[x, y]| [x * times, y * times]));
        let results: Vec<ReadResult> = match options.outline.map(|outline| outline.map(|[x, y]| [x * times, y * times])) {
            // A placement is read as the family named, or as a QR Code,
            // and nothing is searched for.
            _ if placed.is_some() => {
                read_placed(&luma, options.family.unwrap_or(Family::Qr), placed.expect("a placement"), &params, &opts).into_iter().collect()
            }
            // A family named is the one family the outline is read as;
            // with none named it is read as Find in outline reads one, as a
            // QR Code and then as the other families.
            Some(outline) => match options.family {
                Some(family) => read_outline(&luma, start_version(family), outline, &params, &opts).into_iter().collect(),
                None => crate::app::read_outline_of_any_family(&luma, start_version(Family::Qr), outline, &params, &opts)
                    .into_iter()
                    .map(|(read, _)| read)
                    .collect(),
            },
            None => {
                let found = match options.family {
                    Some(family) => {
                        let sought = search_for_families(&luma, &[family], &params, &options.search, &|_| true);
                        leads = sought.leads;
                        sought.detections
                    }
                    None => {
                        let sought = crate::app::search_for_every_family(&luma, &params, &options.search);
                        leads = sought.leads;
                        sought.found
                    }
                };
                // Each symbol found read side by side on the cores to
                // spare, as Read all reads them, and kept in its order.
                work::map(&found, |detection| read_detection(&luma, detection.clone(), &params, &opts)).into_iter().flatten().collect()
            }
        };
        if results.is_empty() {
            status = status.max(1);
            if writeln!(out, "{}: no symbol found", path.display()).is_err() {
                return status;
            }
            for lead in &leads {
                if writeln!(out, "{}", describe_lead(path, lead, times)).is_err() {
                    return status;
                }
            }
            continue;
        }
        let any_read = results.iter().any(ReadResult::is_trustworthy);
        if !any_read {
            status = status.max(1);
        }
        for result in results.iter().filter(|r| r.is_trustworthy() || !any_read || options.partial) {
            if writeln!(out, "{}", describe(path, result, &options, times)).is_err() {
                return status;
            }
            if let Some(warning) = risk_warning(path, result) {
                warn(&warning);
            }
            if options.grid {
                for line in grid_lines(result) {
                    if writeln!(out, "{line}").is_err() {
                        return status;
                    }
                }
            }
            if options.metrics
                && let Some(measures) = ReadMeasures::of(result)
                && writeln!(out, "  metrics: {measures}").is_err()
            {
                return status;
            }
        }
        // What the search found the finders of and fitted no symbol to
        // goes with the symbols that did not decode.
        for lead in leads.iter().filter(|_| !any_read || options.partial) {
            if writeln!(out, "{}", describe_lead(path, lead, times)).is_err() {
                return status;
            }
        }
        // The parts of a message spread over several symbols, joined: of
        // the reads that are to be believed, whose headers are.
        let parts = results.iter().filter(|r| r.is_trustworthy()).filter_map(|r| Part::from_report(&r.report, String::new()));
        for sequence in sequences(parts.collect()) {
            if writeln!(out, "{}", describe_sequence(path, &sequence, options.show, options.details)).is_err() {
                return status;
            }
        }
    }
    status
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_are_parsed() {
        let args: Vec<OsString> = ["--family", "rmqr", "--outline", "1,2, 3,4,5,6,7,8", "--hex", "a.png", "b.jpg"].iter().map(OsString::from).collect();
        let options = parse(&args).unwrap();
        assert_eq!(options.family, Some(Family::Rmqr));
        assert_eq!(options.outline, Some([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]));
        assert_eq!(options.show, Show::Hex);
        assert!(options.search.coarse_to_fine);
        assert_eq!(options.images, [OsString::from("a.png"), OsString::from("b.jpg")]);
        assert!(!options.corners && !options.grid);
        let full: Vec<OsString> = ["--full-search", "a.png"].iter().map(OsString::from).collect();
        assert!(!parse(&full).unwrap().search.coarse_to_fine);
        // A value after an equals sign, and flags that take none.
        let joined: Vec<OsString> =
            ["--family=micro", "--outline=1,2,3,4,5,6,7,8", "--corners", "--grid", "a.png"].iter().map(OsString::from).collect();
        let options = parse(&joined).unwrap();
        assert_eq!(options.family, Some(Family::Micro));
        assert_eq!(options.outline, Some([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]));
        assert!(options.corners && options.grid);
        assert_eq!((options.charset, options.details, options.partial, options.surface), (Charset::Auto, false, false, false));
        let more: Vec<OsString> =
            ["--charset", "Shift_JIS", "--details", "--partial", "--surface", "a.png"].iter().map(OsString::from).collect();
        let options = parse(&more).unwrap();
        assert_eq!((options.charset, options.details, options.partial, options.surface), (Charset::ShiftJis, true, true, true));
        assert!(parse(&[OsString::from("--charset=klingon"), OsString::from("a.png")]).is_err());
        assert!(parse(&[OsString::from("--details=yes"), OsString::from("a.png")]).is_err());
        assert!(parse(&[OsString::from("--hex=1"), OsString::from("a.png")]).is_err());
        assert!(parse(&[OsString::from("--family")]).is_err());
        assert!(parse(&[OsString::from("--outline"), OsString::from("1,2,3"), OsString::from("a.png")]).is_err());
        assert!(parse(&[OsString::from("--what"), OsString::from("a.png")]).is_err());
        // `--` ends the options: an image may be named as an option is.
        let ended: Vec<OsString> = ["--corners", "a.png", "--", "--hex", "--", "-b.png"].iter().map(OsString::from).collect();
        let options = parse(&ended).unwrap();
        assert_eq!(options.images, ["a.png", "--hex", "--", "-b.png"].map(OsString::from));
        assert!(options.corners && options.show == Show::Text);
        assert!(parse(&[OsString::from("--")]).is_err());
        // A placement is read through, an outline looked inside: not both.
        let placed: Vec<OsString> = ["--placed=1,2,3,4,5,6,7,8", "--metrics", "a.png"].iter().map(OsString::from).collect();
        let options = parse(&placed).unwrap();
        assert_eq!(options.placed, Some([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]));
        assert!(options.metrics && options.outline.is_none());
        assert!(parse(&[OsString::from("--placed=1,2,3,4,5,6,7,8"), OsString::from("--outline=1,2,3,4,5,6,7,8"), OsString::from("a.png")]).is_err());
        assert!(parse(&[OsString::from("--placed=1,2,3"), OsString::from("a.png")]).is_err());
        assert!(parse(&[OsString::from("--metrics=yes"), OsString::from("a.png")]).is_err());
        // The photo enlarged and sharpened, each on its own or both.
        assert_eq!(options.enhance, Enhance::NONE);
        let enhanced: Vec<OsString> = ["--enlarge", "3", "--sharpen=strong", "a.png"].iter().map(OsString::from).collect();
        assert_eq!(parse(&enhanced).unwrap().enhance, Enhance { enlarge: 3, sharpen: Sharpen::Strong });
        let sharpened = parse(&[OsString::from("--sharpen"), OsString::from("light"), OsString::from("a.png")]).unwrap();
        assert_eq!(sharpened.enhance, Enhance { enlarge: 1, sharpen: Sharpen::Light });
        for bad in ["--enlarge=5", "--enlarge=0", "--enlarge=two", "--sharpen=very"] {
            assert!(parse(&[OsString::from(bad), OsString::from("a.png")]).is_err(), "{bad}");
        }
        assert!(parse(&[OsString::from("--sharpen")]).is_err());
        assert!(parse(&[]).is_err());
        // The usage asked for is printed as output, and is no error; a
        // bad command line is one, and says what is bad.
        assert!(matches!(parse(&[OsString::from("--help")]), Err(Unparsed::Help)));
        assert!(matches!(parse(&[OsString::from("-h"), OsString::from("a.png")]), Err(Unparsed::Help)));
        let mut out = Vec::new();
        assert_eq!(run_to(&[OsString::from("--help")], &mut out), 0);
        assert_eq!(String::from_utf8(out).unwrap(), format!("{USAGE}\n"));
        assert_eq!(run(&[OsString::from("/nowhere/at/all.png")]), 2);
        assert!(matches!(parse(&[OsString::from("--what"), OsString::from("a.png")]), Err(Unparsed::Bad(message)) if message.contains("--what")));
        // One way of printing the message at a time.
        for both in [["--hex", "--escaped"], ["--escaped", "--hex"]] {
            let args: Vec<OsString> = both.iter().chain(&["a.png"]).map(OsString::from).collect();
            assert!(matches!(parse(&args), Err(Unparsed::Bad(message)) if message.contains("one of them")), "{both:?}");
        }
        let twice: Vec<OsString> = ["--hex", "--hex", "a.png"].iter().map(OsString::from).collect();
        assert_eq!(parse(&twice).unwrap().show, Show::Hex);
        // An outline is made of pixel positions, which are finite.
        for bad in ["nan,2,3,4,5,6,7,8", "1,2,3,4,inf,6,7,8", "1,2,3,4,5,6,7,-inf"] {
            let args = [OsString::from(format!("--outline={bad}")), OsString::from("a.png")];
            assert!(matches!(parse(&args), Err(Unparsed::Bad(message)) if message.contains("no pixel position")), "{bad}");
        }
        // Model 1 is searched for by name as any family is: a search of
        // the whole photo for it alone finds a Model 1 symbol by its
        // finders, which are those of a QR Code, and fits it as Model 1.
        let model1: Vec<OsString> = ["--family", "model1", "a.png"].iter().map(OsString::from).collect();
        let options = parse(&model1).unwrap();
        assert_eq!((options.family, options.outline), (Some(Family::Model1), None));
    }

    #[test]
    fn the_default_output_keeps_a_message_to_its_line() {
        // A line break, a group separator and an escape are spelt out;
        // a tab, the unreadable mark and everything printable stay.
        assert_eq!(one_line("BEGIN:VCARD\r\nN:A\tB\u{1D}\u{1B}\u{FFFD}é"), "BEGIN:VCARD<CR><LF>N:A\tB<GS><ESC>\u{FFFD}é");
        assert_eq!(one_line("plain"), "plain");
        assert_eq!(one_line(""), "");
    }

    #[test]
    fn character_sets_are_named_loosely() {
        // As the app lists them, in any case, with or without hyphens.
        assert_eq!(charset_named("UTF-8"), Some(Charset::Utf8));
        assert_eq!(charset_named("utf8"), Some(Charset::Utf8));
        assert_eq!(charset_named("shift-jis"), Some(Charset::ShiftJis));
        assert_eq!(charset_named("EUC_KR"), Some(Charset::EucKr));
        assert_eq!(charset_named("auto"), Some(Charset::Auto));
        // Any set the ECI table names, by its name or its number.
        for charset in Charset::ALL {
            assert_eq!(charset_named(charset.label()).map(Charset::label), Some(charset.label()));
        }
        assert_eq!(charset_named("17").map(Charset::label), Some(Charset::Eci(17).label()));
        assert_eq!(charset_named("klingon"), None);
        assert_eq!(charset_named("14"), None, "an assignment the table leaves empty");
    }

    /// A part of a structured append message as a decode would make it.
    fn part(index: u8, total: u8, parity: u8, text: &str) -> Part {
        use qr_forensics::decode::bitstream::Certainty;
        use qr_forensics::decode::text::{TextReport, TextSpan};
        let report = TextReport {
            spans: vec![TextSpan { text: text.to_owned(), certainty: Certainty::Verified }],
            parity: Some(text.bytes().fold(0, |a, b| a ^ b)),
            bytes: text.bytes().map(Some).collect(),
            ..TextReport::default()
        };
        Part { index, total, parity, label: String::new(), text: report, risk_bits: None }
    }

    #[test]
    fn the_parts_of_a_message_are_joined_in_order() {
        let message = "The quick brown fox jumps over the lazy dog";
        let parity = message.bytes().fold(0, |a, b| a ^ b);
        let path = Path::new("sheet.png");
        // Found out of order, with a part of some other message among
        // them: each message is joined of its own parts.
        let found = vec![
            part(3, 3, parity, &message[30..]),
            part(1, 2, 0x55, "another "),
            part(1, 3, parity, &message[..15]),
            part(2, 3, parity, &message[15..30]),
        ];
        let joined = sequences(found);
        assert_eq!(joined.len(), 2);
        assert_eq!(
            describe_sequence(path, &joined[0], Show::Text, false),
            format!("sheet.png: structured append, 3 of 3 parts, parity verified: {message}")
        );
        assert_eq!(
            describe_sequence(path, &joined[1], Show::Text, false),
            "sheet.png: structured append, 1 of 2 parts: another [part 2 missing]"
        );
        // Every part in and a byte of one misread: the parity says so.
        let wrong = sequences(vec![part(1, 2, parity, &message[..20]), part(2, 2, parity, &message[20..].replace('z', "s"))]);
        let line = describe_sequence(path, &wrong[0], Show::Text, false);
        assert!(line.starts_with(&format!("sheet.png: structured append, 2 of 2 parts, parity {:02X} where the parts claim {parity:02X}: ", parity ^ b'z' ^ b's')), "{line}");
    }

    #[test]
    fn details_say_what_the_decode_settled() {
        let image = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/photos/cpp_micro_M3-Mixed.png").into_os_string();
        let mut out = Vec::new();
        assert_eq!(run_to(&[OsString::from("--details"), OsString::from("--corners"), image], &mut out), 0);
        let text = String::from_utf8(out).unwrap();
        let (fields, _) = fields_and_grid(&text);
        assert_eq!(fields.len(), 6, "{text}");
        assert_eq!(&fields[1..3], ["Micro QR M3", "read"]);
        // The level and the mask, and no character set: the symbol holds
        // no byte data.
        let settled: Vec<&str> = fields[3].split(", ").collect();
        assert!(settled.len() >= 2 && settled[0].starts_with("EC ") && settled[1].starts_with("mask "), "{fields:?}");
        assert!(!settled.contains(&"threshold surface"), "{fields:?}");
        assert!(fields[4].starts_with("at "));
        assert_eq!(fields[5], "1234 ABCD");
        // Read against the threshold surface, as `--surface` asks first,
        // the details say so, and say what would read it so again.
        let image = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/photos/cpp_micro_M3-Mixed.png").into_os_string();
        let mut out = Vec::new();
        assert_eq!(run_to(&[OsString::from("--details"), OsString::from("--surface"), image], &mut out), 0);
        let text = String::from_utf8(out).unwrap();
        let (fields, _) = fields_and_grid(&text);
        assert_eq!(fields.len(), 5, "{text}");
        assert_eq!(fields[3].split(", ").last(), Some("threshold surface"), "{fields:?}");
        assert_eq!(fields[4], "1234 ABCD");
    }

    /// A pipe whose reader has gone.
    struct Closed;

    impl Write for Closed {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::ErrorKind::BrokenPipe.into())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// The fields of a line, split at the separators, and the lines of
    /// the grid printed below it.
    fn fields_and_grid(text: &str) -> (Vec<String>, Vec<String>) {
        let mut lines = text.lines();
        let fields = lines.next().unwrap().split(": ").map(str::to_owned).collect();
        let grid = lines.take_while(|l| l.starts_with("  ")).map(|l| l.trim().to_owned()).collect();
        (fields, grid)
    }

    #[test]
    fn a_symbol_found_reads_again_through_its_corners() {
        // A search prints where it found the symbol, and the corners
        // printed, given back as the outline, read the same symbol: as
        // the symbol is, and mirrored, when the corners run the other
        // way round the photo.
        let image = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/photos/cpp_micro_M3-Mixed.png");
        let mirrored = std::env::temp_dir().join(format!("qr-forensics-mirrored-{}.png", std::process::id()));
        image::open(&image).unwrap().fliph().save(&mirrored).unwrap();
        for (path, expect_mirrored) in [(image, false), (mirrored.clone(), true)] {
            let mut out = Vec::new();
            let args = [OsString::from("--corners"), OsString::from("--grid"), path.clone().into_os_string()];
            assert_eq!(run_to(&args, &mut out), 0);
            let text = String::from_utf8(out).unwrap();
            let (fields, grid) = fields_and_grid(&text);
            assert_eq!(fields.len(), 5, "{text}");
            assert_eq!(fields[1], if expect_mirrored { "Micro QR M3, mirrored" } else { "Micro QR M3" });
            assert_eq!(fields[2], "read");
            assert_eq!(fields[4], "1234 ABCD");
            let at = fields[3].strip_prefix("at ").unwrap();
            assert_eq!(at.split(',').count(), 8, "{at}");
            // The photo is thirty-four pixels a side and is read eight
            // times that: the corners are in its own pixels, to the
            // hundredth of one.
            let numbers: Vec<f64> = at.split(',').map(|n| n.parse().unwrap()).collect();
            assert!(numbers.iter().all(|n| (0.0..=34.0).contains(n)), "{at}");
            assert!(at.split(',').all(|n| n.split_once('.').is_some_and(|(_, places)| places.len() == 2)), "{at}");
            // Fifteen rows of fifteen modules, every one read.
            assert_eq!(grid.len(), 15, "{text}");
            assert!(grid.iter().all(|row| row.len() == 15 && row.chars().all(|c| c == '#' || c == '.')), "{grid:?}");
            let again = [OsString::from(format!("--outline={at}")), path.into_os_string()];
            let mut out = Vec::new();
            assert_eq!(run_to(&again, &mut out), 0);
            let (fields, _) = fields_and_grid(&String::from_utf8(out).unwrap());
            assert_eq!(&fields[2..], ["read", "1234 ABCD"], "{fields:?}");
        }
        let _ = std::fs::remove_file(mirrored);
    }

    #[test]
    fn a_symbol_the_search_places_badly_is_read_again_as_an_outline() {
        // The search of this photo of a crumpled sheet finds its symbol,
        // and the one placement it keeps solves nothing; the outline of
        // that placement, drawn a little wide and read as a hand-drawn
        // one is, with every placement the detector offers, reads.
        let image = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/blackbox/cpp_qrcode-4_wrinkle1-05x.webp");
        let expected = std::fs::read_to_string(image.with_extension("txt")).unwrap();
        let mut out = Vec::new();
        assert_eq!(run_to(&[OsString::from("--family=qr"), image.into_os_string()], &mut out), 0);
        let text = String::from_utf8(out).unwrap();
        let (fields, _) = fields_and_grid(&text);
        assert_eq!(&fields[1..], ["QR Code 3", "read", expected.trim_end()], "{text}");
    }

    #[test]
    fn a_read_that_took_the_padding_pinned_says_so() {
        use qr_forensics::decode::bitstream::EndOfData;
        use qr_forensics::decode::codewords::{Pin, PinSource, Pins};
        use qr_forensics::decode::padding::pad_tail;
        use qr_forensics::decode::pipeline::decode;
        use qr_forensics::decode::repair::{render_rgba, repaired_grid};
        use qr_forensics::symbol::EcLevel;
        use qr_forensics::symbol::grid::{Module, ModuleGrid};
        use qrcode_generator::{Segment, qr};
        // A short address in a version 10-M symbol, with a strip of white
        // tape down a third of it: more wrong codewords in every block
        // than the block corrects, nearly all of them padding.
        let message = "https://example.com/t/8F3K2";
        let mask = 4;
        let symbol = qr::Encoder::new(qr::ErrorCorrection::Medium)
            .version(qr::Version::new(10).unwrap())
            .mask(qr::Mask::new(mask).unwrap())
            .boost_error_correction(false)
            .encode_segments(&[Segment::bytes(message)])
            .unwrap();
        let standard = ModuleGrid::from_bools(symbol.width(), symbol.height(), symbol.modules());
        // The same symbol as an encoder that fills with zeros writes it:
        // its data codewords pinned into a blank grid, zeros from the
        // first of the padding on, and the grid error correction makes
        // of them written out.
        let layout = SymbolLayout::new(Arc::new(FunctionMap::new(SymbolVersion::Qr(10))), EcLevel::M).unwrap();
        let (opts, none) = (DecodeOptions::default(), Pins::new());
        let read = decode(&standard, &layout, mask, &none, &opts);
        let EndOfData::At { bit, .. } = read.parse.end else { panic!("{:?}", read.parse.end) };
        let first_pad = pad_tail(&layout, bit).first_pad_codeword;
        let pins: Pins = layout
            .data_stream
            .iter()
            .enumerate()
            .map(|(j, &cw)| (cw, Pin::full(if j < first_pad { read.codewords[cw].final_value.unwrap() } else { 0 }, PinSource::Manual)))
            .collect();
        let blank = layout.func.blank_grid();
        let zero_filled = repaired_grid(&blank, &layout, mask, &decode(&blank, &layout, mask, &pins, &opts));
        assert!(decode(&zero_filled, &layout, mask, &none, &opts).is_trustworthy());

        for (mut grid, pinned) in [(standard, " padding codewords pinned"), (zero_filled, " padding codewords pinned as zeros")] {
            for y in 0..grid.height() {
                for x in 12..32 {
                    grid.set(x, y, Module::Light);
                }
            }
            let (width, height, pixels) = render_rgba(&grid, 8, 4);
            let path = std::env::temp_dir().join(format!("qr-forensics-taped-{}-{}.png", std::process::id(), pinned.len()));
            image::save_buffer(&path, &pixels, width as u32, height as u32, image::ExtendedColorType::Rgba8).unwrap();
            let mut out = Vec::new();
            let status = run_to(&[OsString::from("--details"), path.clone().into_os_string()], &mut out);
            let _ = std::fs::remove_file(&path);
            let text = String::from_utf8(out).unwrap();
            let (fields, _) = fields_and_grid(&text);
            // A read, by the status and by the line, which says how it
            // stands and what it took: the padding pinned, and as what
            // when it is not the standard's.
            assert_eq!(status, 0, "{text}");
            assert_eq!(&fields[1..3], ["QR Code 10", "read, padding pinned"], "{text}");
            assert!(fields[3].ends_with(pinned), "{text}");
            assert_eq!(fields.last().map(String::as_str), Some(message), "{text}");
        }
    }

    #[test]
    fn a_sound_symbol_whose_message_does_not_parse_is_malformed() {
        use qr_forensics::decode::codewords::{Pin, PinSource, Pins};
        use qr_forensics::decode::pipeline::decode;
        use qr_forensics::decode::repair::{render_rgba, repaired_grid};
        use qr_forensics::symbol::EcLevel;
        // A version 1-L symbol sound in every codeword, whose data
        // begins with an ECI header that names no assignment: its one
        // block is confirmed, and its message does not parse so far as
        // its first segment. It stood as `partial`, with one block of
        // one solved, which a symbol with nothing left to solve is not.
        let layout = SymbolLayout::new(Arc::new(FunctionMap::new(SymbolVersion::Qr(1))), EcLevel::L).unwrap();
        let (opts, mask) = (DecodeOptions::default(), 0);
        let pins: Pins = layout
            .data_stream
            .iter()
            .enumerate()
            .map(|(j, &cw)| (cw, Pin::full(if j == 0 { 0x7E } else { 0x00 }, PinSource::Manual)))
            .collect();
        let blank = layout.func.blank_grid();
        let grid = repaired_grid(&blank, &layout, mask, &decode(&blank, &layout, mask, &pins, &opts));
        let read = decode(&grid, &layout, mask, &Pins::new(), &opts);
        assert!(read.blocks[0].is_verified() && read.parse.segments.is_empty() && !read.score.stream_valid);
        let (width, height, pixels) = render_rgba(&grid, 8, 4);
        let path = std::env::temp_dir().join(format!("qr-forensics-malformed-{}.png", std::process::id()));
        image::save_buffer(&path, &pixels, width as u32, height as u32, image::ExtendedColorType::Rgba8).unwrap();
        let mut out = Vec::new();
        let status = run_to(&[path.clone().into_os_string()], &mut out);
        let _ = std::fs::remove_file(&path);
        let text = String::from_utf8(out).unwrap();
        let (fields, _) = fields_and_grid(&text);
        // No read, by the status, and the line says why not.
        assert_eq!(status, 1, "{text}");
        assert_eq!(&fields[1..3], ["QR Code 1", "malformed, 1 of 1 blocks solved"], "{text}");
    }

    #[test]
    fn a_character_set_the_symbol_declares_is_read_as_declared() {
        // OpenCV's symbol declares UTF-8 with an ECI designator: with
        // another set named or with none, it is read in the set it
        // declares.
        let image = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/opencv/encode_version1_mode7.png").into_os_string();
        let mut lines = Vec::new();
        for charset in [None, Some("--charset=shift_jis")] {
            let mut args: Vec<OsString> = charset.into_iter().map(OsString::from).collect();
            args.extend([OsString::from("--details"), image.clone()]);
            let mut out = Vec::new();
            assert_eq!(run_to(&args, &mut out), 0);
            let text = String::from_utf8(out).unwrap();
            let (fields, _) = fields_and_grid(&text);
            assert_eq!(&fields[1..3], ["QR Code 1", "read"], "{text}");
            assert!(fields[3].contains("UTF-8 (ECI)"), "{text}");
            lines.push(text);
        }
        assert_eq!(lines[0], lines[1]);
    }

    #[test]
    fn a_closed_output_ends_the_run_quietly() {
        let image = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/photos/cpp_micro_M3-Mixed.png").into_os_string();
        let args = [image, OsString::from("/nowhere/at/all.png")];
        let mut out = Vec::new();
        assert_eq!(run_to(&args, &mut out), 2, "the second image does not load");
        assert!(String::from_utf8_lossy(&out).ends_with("Micro QR M3: read: 1234 ABCD\n"));
        // The first image reads and its line cannot be written: the run
        // ends there, with that image's status, and the second is never
        // come to.
        assert_eq!(run_to(&args, &mut Closed), 0);
    }
}
