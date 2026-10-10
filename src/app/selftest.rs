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
//! window <w> <h>              resize the window, in points
//! wait <frames>               let the window settle for so many frames
//! layers both|photo|grid
//! opacity <grid> <photo>      blend of the Both view, each 0 to 1
//! overlay none|roles|blocks|corrections|reliability
//! sample                      sample every module from the photo
//! invert photo|grid           toggle the photo's inversion, or swap dark
//!                             and light in the grid
//! unknown x0 y0 x1 y1         paint a rectangle of modules unknown
//! select <codeword>
//! block <n>|none              outline a block's unknown codewords
//! format auto | <L|M|Q|H> <mask>
//! find                        find the symbol inside the outline and place
//!                             the corners on it
//! symbols                     search the whole photo and list the symbols
//!                             found
//! use <n>                     adopt the n-th symbol found, counting from 1
//! refine                      refine the corners against the fixed patterns
//! apply                       write corrections into the grid
//! say                         print the version, decode and status line
//! brute | padding | suggest | fit | quit
//! ```

use std::path::PathBuf;

use eframe::egui;
use qr_forensics::imaging::adjust::Adjust;
use qr_forensics::symbol::grid::Module;
use qr_forensics::symbol::{EcLevel, SymbolVersion};

use super::state::*;
use super::{Action, App};

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
            "use" => actions.push(Action::UseProposal(number(0) as usize - 1)),
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
                    _ => Overlay::None,
                }
            }
            "sample" => actions.push(Action::Sample { only_unknown: false }),
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
                        self.paint(x, y, Module::Unknown);
                    }
                }
            }
            "select" => {
                self.view.selected = Some(number(0) as usize);
                self.view.reveal_selected = true;
            }
            "block" => self.view.highlight_block = word(0).parse::<usize>().ok().map(|b| b - 1),
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
