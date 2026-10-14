//! Development aid, compiled only with the `selftest` feature: runs a script
//! of UI steps and saves screenshots of the window, so the interface can be
//! checked without someone at the mouse.
//!
//! ```text
//! QR_FORENSICS_SELFTEST=<dir> cargo run --features selftest
//! ```
//!
//! `<dir>/script.txt` holds one step per line; screenshots are written to
//! `<dir>` as PNG. Steps:
//!
//! ```text
//! shot <name>                 save <name>.png
//! open <file>                 load an image from <dir>
//! version qr|micro|rmqr <n>   choose the symbol version
//! corners x y x y x y x y     place TL TR BR BL, in photo pixels
//! stage align|edit
//! show fixed|sample|lines|binarised
//!                             Align overlay
//! binarise local|global|window <px>
//!                             the binarised view's rule, a window of so
//!                             many pixels either side
//! binarise-scan on|off        read the photo against the binarised
//!                             view's threshold alone
//! window <w> <h>              resize the window, in points
//! wait <frames>               let the window settle for so many frames
//! layers both|photo|grid
//! opacity <grid> <photo>      blend of the Both view, each 0 to 1
//! overlay none|roles|blocks|corrections|reliability|mask|unmasked|status|
//!         segments|pins|photo
//! sample                      sample every module from the photo
//! sample block <n>            or the modules of the n-th block alone
//! sample rect x0 y0 x1 y1     or paint a rectangle of modules with the
//!                             From photo ink
//! sharpening on|off           sharpen small or blurred modules as they
//!                             are sampled
//! tool brush|rect|inspect
//! ink dark|light|unknown|photo
//! guess on|off                paint dark, light and the From photo ink as
//!                             guesses
//! enlarge <n>                 the photo's Enlarge: n times its own size,
//!                             0.25, 0.5 or a whole number, as far as its
//!                             size allows; the outline goes with it
//! sharpen <amount>            the photo's Sharpen, from -1 (blurred) to 1
//! grey on|off                 show the photo in greyscale, or in colour
//! invert photo|grid           toggle the photo's inversion, or swap dark
//!                             and light in the grid
//! unknown x0 y0 x1 y1         paint a rectangle of modules unknown
//! select <codeword>
//! pinbits <codeword> <mask> <value>
//!                             pin the bits of a codeword that the mask
//!                             names, both in hexadecimal
//! block <n>|none              outline a block's unknown codewords, or
//!                             the modules doubted to solve it
//! segment <n>|none            outline the modules of the n-th segment
//! doubt codewords|bits|flips on|off
//!                             one of the three ways of doubting
//! padfrom <byte> [late|swapped|zeros]
//!                             pin the padding from that message byte, as
//!                             the standard's filler or another
//! format auto | <L|M|Q|H> <mask>
//! find                        find the symbol inside the outline and place
//!                             the corners on it
//! symbols                     search the whole photo and list the symbols
//!                             found
//! use <n>                     adopt the n-th symbol found, counting from 1;
//!                             while autoload is on it is read through,
//!                             and a read that decodes is loaded
//! placements                  list the placements inside the outline
//! place <n>                   adopt the n-th of them, as use adopts a
//!                             symbol found
//! autoload on|off             load a read that decodes into the grid, as
//!                             find, symbols, readall, use and place make
//!                             one
//! only on|off                 look for the family in use alone
//! readall                     read every symbol found through to its message
//! crop | whole                cut the photo down to the outline, or show
//!                             all of it again
//! known <text>                the text known of the message: the rest of
//!                             the line
//! at <n> | start <mode>       where it starts: the n-th character, counting
//!                             from 1, or the start of the message as
//!                             byte, alphanumeric, numeric or kanji
//! pintext | beginnings        pin the known text, or try the ways messages
//!                             commonly begin
//! pick <n>|none               outline the modules of the n-th character
//! export png|svg <name>       save the repaired symbol to <dir>
//! refine                      refine the corners against the fixed patterns
//! apply                       write corrections into the grid
//! say                         print the version, decode and status line
//! brute | padding | suggest | fit | quit
//! ```

use std::path::PathBuf;

use eframe::egui;
use qr_forensics::decode::padding::Filler;
use qr_forensics::decode::pipeline::DecodeOptions;
use qr_forensics::imaging::adjust::{Adjust, Enlarge};
use qr_forensics::symbol::grid::{CERTAIN, Module};
use qr_forensics::symbol::tables::Mode;
use qr_forensics::symbol::{EcLevel, SymbolVersion};

use super::state::*;
use super::{Action, App, ExportFormat};

pub(super) struct SelfTest {
    dir: PathBuf,
    steps: std::vec::IntoIter<String>,
    /// Frames left to let the window settle before the next step.
    wait: u32,
    /// Screenshot asked for and not yet delivered.
    pending: Option<PathBuf>,
}

impl SelfTest {
    pub(super) fn from_env() -> Option<Self> {
        let dir = PathBuf::from(std::env::var_os("QR_FORENSICS_SELFTEST")?);
        let script = std::fs::read_to_string(dir.join("script.txt")).expect("selftest script.txt");
        let steps: Vec<String> = script
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(str::to_owned)
            .collect();
        Some(Self { dir, steps: steps.into_iter(), wait: 20, pending: None })
    }
}

impl App {
    pub(super) fn selftest_tick(&mut self, ui: &egui::Ui, actions: &mut Vec<Action>) {
        let Some(mut test) = self.selftest.take() else { return };
        ui.ctx().request_repaint();

        if let Some(path) = &test.pending {
            let shot = ui.input(|i| {
                i.raw.events.iter().find_map(|event| match event {
                    egui::Event::Screenshot { image, .. } => Some(image.clone()),
                    _ => None,
                })
            });
            if let Some(shot) = shot {
                image::save_buffer(
                    path,
                    shot.as_raw(),
                    shot.width() as u32,
                    shot.height() as u32,
                    image::ExtendedColorType::Rgba8,
                )
                .expect("save screenshot");
                test.pending = None;
            }
            self.selftest = Some(test);
            return;
        }
        if test.wait > 0 {
            test.wait -= 1;
            self.selftest = Some(test);
            return;
        }
        // A search of the photo, and the steps that find, adopt or refine
        // the corners, run on their own threads; steps wait for them,
        // since the next one usually builds on what they find, and then
        // let the window settle, as after any other step, so that the
        // result has been taken in and the decode refreshed.
        if self.search.is_some() || self.job.is_some() {
            test.wait = 6;
            self.selftest = Some(test);
            return;
        }
        let Some(step) = test.steps.next() else {
            actions.push(Action::Quit);
            return;
        };

        let mut words = step.split_whitespace();
        let command = words.next().unwrap_or_default();
        let rest: Vec<&str> = words.collect();
        let word = |i: usize| rest.get(i).copied().unwrap_or_else(|| panic!("selftest: `{step}` needs more arguments"));
        let number = |i: usize| word(i).parse::<f64>().unwrap_or_else(|_| panic!("selftest: bad number in `{step}`"));
        // A place in a list, counted from one as the lists show them.
        let ordinal = |i: usize| (number(i) as usize).checked_sub(1).unwrap_or_else(|| panic!("selftest: `{step}` counts from 1"));
        match command {
            "shot" => {
                test.pending = Some(test.dir.join(format!("{}.png", word(0))));
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            }
            "open" => self.load_path(ui.ctx(), &test.dir.join(word(0))),
            "version" => {
                let n = number(1) as u8;
                actions.push(Action::SetVersion(match word(0) {
                    "qr" => SymbolVersion::Qr(n),
                    "micro" => SymbolVersion::Micro(n),
                    "rmqr" => SymbolVersion::Rmqr(n),
                    "model1" => SymbolVersion::Model1(n),
                    other => panic!("selftest: unknown family {other}"),
                }));
            }
            "corners" => {
                assert!(self.photo.is_some(), "selftest: corners need a photo");
                for (i, corner) in self.doc.corners.iter_mut().enumerate() {
                    *corner = [number(2 * i), number(2 * i + 1)];
                }
            }
            "find" => actions.push(Action::FindCorners),
            "symbols" => actions.push(Action::FindSymbols),
            "use" => actions.push(Action::UseProposal(ordinal(0))),
            "placements" => actions.push(Action::ListPlacements),
            "place" => actions.push(Action::UsePlacement(ordinal(0))),
            "autoload" => self.view.load_reads = word(0) == "on",
            "only" => self.view.own_family_only = word(0) == "on",
            "readall" => actions.push(Action::ReadFound),
            "crop" => actions.push(Action::CropToOutline),
            "whole" => actions.push(Action::ShowWholePhoto),
            "known" => self.view.known_text = step.strip_prefix("known").unwrap_or_default().trim().to_owned(),
            "at" => (self.view.known_at, self.view.known_from_start) = (ordinal(0), false),
            "start" => {
                self.view.known_from_start = true;
                self.view.known_mode = match word(0) {
                    "alphanumeric" => Mode::Alphanumeric,
                    "numeric" => Mode::Numeric,
                    "kanji" => Mode::Kanji,
                    _ => Mode::Byte,
                };
            }
            "pintext" => actions.push(Action::PinText),
            "beginnings" => actions.push(Action::TryBeginnings),
            // Counted from one; `pick 0` picks nothing.
            "pick" => self.view.picked_char = word(0).parse::<usize>().ok().filter(|&n| n > 0).map(|n| n - 1),
            "export" => {
                let format = if word(0) == "svg" { ExportFormat::Svg } else { ExportFormat::Png };
                self.export_image(&test.dir.join(word(1)), format);
            }
            "enlarge" => {
                assert!(self.photo.is_some(), "selftest: enlarge needs a photo");
                let enlarge = Enlarge::parse(word(0)).unwrap_or_else(|| panic!("selftest: bad size in `{step}`"));
                self.checkpoint();
                actions.push(Action::SetAdjust(Adjust { enlarge, ..self.doc.adjust }));
            }
            "sharpen" => {
                assert!(self.photo.is_some(), "selftest: sharpen needs a photo");
                self.checkpoint();
                actions.push(Action::SetAdjust(Adjust { sharpen: (number(0) as f32).clamp(-1.0, 1.0), ..self.doc.adjust }));
            }
            "grey" => self.view.greyscale = word(0) == "on",
            "refine" => actions.push(Action::RefineCorners),
            "apply" => actions.push(Action::ApplyCorrections),
            "stage" => self.view.stage = if word(0) == "align" { Stage::Align } else { Stage::Edit },
            "show" => {
                self.view.align_overlay = match word(0) {
                    "sample" => AlignOverlay::SamplePreview,
                    "lines" => AlignOverlay::LinesOnly,
                    "binarised" => AlignOverlay::Binarised,
                    _ => AlignOverlay::FixedPatterns,
                }
            }
            "binarise-scan" => self.view.binarize_for_scan = word(0) == "on",
            "binarise" => {
                self.view.binarize_auto = false;
                (self.view.binarize.global, self.view.binarize.window) = match word(0) {
                    "global" => (true, None),
                    "window" => (false, Some(number(1) as usize)),
                    _ => (false, None),
                };
            }
            "layers" => {
                self.view.layers = match word(0) {
                    "photo" => Layers::Photo,
                    "grid" => Layers::Grid,
                    _ => Layers::Both,
                }
            }
            "window" => {
                let size = egui::vec2(number(0) as f32, number(1) as f32);
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::InnerSize(size));
            }
            "wait" => {
                test.wait = number(0) as u32;
                self.selftest = Some(test);
                return;
            }
            "opacity" => {
                self.view.grid_opacity = (number(0) as f32).clamp(0.1, 1.0);
                self.view.photo_opacity = (number(1) as f32).clamp(0.0, 1.0);
            }
            "overlay" => {
                self.view.overlay = match word(0) {
                    "roles" => Overlay::Roles,
                    "blocks" => Overlay::Blocks,
                    "corrections" => Overlay::Corrections,
                    "reliability" => Overlay::Reliability,
                    "mask" => Overlay::Mask,
                    "unmasked" => Overlay::Unmasked,
                    "status" => Overlay::Status,
                    "segments" => Overlay::Segments,
                    "pins" => Overlay::Pins,
                    "photo" => Overlay::Photo,
                    _ => Overlay::None,
                }
            }
            "sample" => match rest.first().copied() {
                Some("rect") => {
                    self.update_preview();
                    self.begin_edit();
                    for y in number(2) as i32..=number(4) as i32 {
                        for x in number(1) as i32..=number(3) as i32 {
                            self.paint(x, y, Ink::Photo);
                        }
                    }
                    self.warn_if_painted_inverted();
                }
                Some("block") => actions.push(Action::Sample { only_unknown: false, scope: SampleScope::Block(number(1) as usize - 1) }),
                _ => actions.push(Action::Sample { only_unknown: false, scope: SampleScope::All }),
            },
            "sharpening" => self.view.sample.sharpen_modules = word(0) == "on",
            "tool" => {
                self.view.tool = match word(0) {
                    "rect" => Tool::Rect,
                    "inspect" => Tool::Inspect,
                    _ => Tool::Brush,
                }
            }
            "ink" => {
                self.view.ink = match word(0) {
                    "light" => Ink::Module(Module::Light),
                    "unknown" => Ink::Module(Module::Unknown),
                    "photo" => Ink::Photo,
                    "doubt" => Ink::Doubt,
                    _ => Ink::Module(Module::Dark),
                }
            }
            // `guess on` paints guesses, `guess off` certain modules, and
            // `guess <n>` at a reliability of n.
            "guess" => {
                self.view.paint_reliability = match word(0) {
                    "on" => 0,
                    "off" => CERTAIN,
                    _ => number(0).clamp(0.0, 255.0) as u8,
                }
            }
            "invert" => match word(0) {
                "photo" => {
                    assert!(self.photo.is_some(), "selftest: invert photo needs a photo");
                    self.checkpoint();
                    let adjust = self.doc.adjust;
                    actions.push(Action::SetAdjust(Adjust { invert: !adjust.invert, ..adjust }));
                }
                _ => actions.push(Action::Transform(super::Transform::Invert)),
            },
            "unknown" => {
                self.begin_edit();
                for y in number(1) as i32..=number(3) as i32 {
                    for x in number(0) as i32..=number(2) as i32 {
                        self.paint(x, y, Ink::Module(Module::Unknown));
                    }
                }
            }
            "select" => {
                self.view.selected = Some(number(0) as usize);
                self.view.reveal_selected = true;
            }
            "pinbits" => {
                let hex = |i: usize| u8::from_str_radix(word(i), 16).unwrap_or_else(|_| panic!("selftest: bad byte in `{step}`"));
                actions.push(Action::PinBits { cw: number(0) as usize, mask: hex(1), value: hex(2) });
            }
            "block" => self.view.highlight_block = word(0).parse::<usize>().ok().map(|b| b - 1),
            "segment" => self.view.picked_segment = word(0).parse::<usize>().ok().map(|n| n - 1),
            "doubt" => {
                let on = word(1) == "on";
                actions.push(Action::SetDecodeOptions(match word(0) {
                    "codewords" => DecodeOptions { doubt_codewords: on, ..self.opts },
                    "bits" => DecodeOptions { doubt_bits: on, ..self.opts },
                    "flips" => DecodeOptions { flip_bits: on, ..self.opts },
                    other => panic!("selftest: unknown way of doubting {other}"),
                }));
            }
            "padfrom" => {
                let filler = match rest.get(1).copied() {
                    Some("late") => Filler::Late,
                    Some("swapped") => Filler::Swapped,
                    Some("zeros") => Filler::Zeros,
                    _ => Filler::Standard,
                };
                actions.push(Action::PinPaddingAt { first: number(0) as usize, filler });
            }
            "format" => {
                actions.push(Action::SetFormat(match word(0) {
                    "auto" => FormatSel::Auto,
                    level => FormatSel::Manual {
                        ec: match level {
                            "L" => EcLevel::L,
                            "M" => EcLevel::M,
                            "Q" => EcLevel::Q,
                            _ => EcLevel::H,
                        },
                        mask: number(1) as u8,
                    },
                }));
            }
            "brute" => actions.push(Action::BruteForce),
            "padding" => actions.push(Action::ApplyPadding),
            "suggest" => actions.push(Action::SuggestVersions),
            "fit" => actions.push(Action::FitView),
            "say" => {
                // What a screenshot would show, in words, for a run with no
                // display to look at: the status line and the decode.
                let notice = self.message.as_ref().map_or("", |n| n.text.as_str());
                let report = &self.derived.report;
                println!(
                    "say: version {} {}; {} of {} blocks solved; text {:?}; notice: {notice}",
                    self.doc.version.family().label(),
                    self.doc.version,
                    report.blocks_solved(),
                    report.blocks.len(),
                    report.text.plain()
                );
            }
            "quit" => {
                actions.push(Action::Quit);
                return;
            }
            other => panic!("selftest: unknown step `{other}`"),
        }
        test.wait = 6;
        self.selftest = Some(test);
    }
}
