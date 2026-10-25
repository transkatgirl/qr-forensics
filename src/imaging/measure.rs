//! How well a read read its photo: figures for measuring the pipeline
//! over a corpus, rather than for reading any one symbol.
//!
//! A benchmark that counts symbols read and lost says nothing of a
//! change that reads every symbol it read before a little better, and
//! once nearly every symbol of a corpus reads, nearly every change looks
//! like no change. The smoothing of the map through the anchors read the
//! same at nought, one, five and twenty-five
//! ([`super::projection::ANCHOR_SMOOTHING`]), and was set by reasoning
//! for want of anything to tell the four apart. What a read can say of
//! itself tells them apart. Once a block is confirmed, every module of
//! its codewords is known, and what the sampling made of each can be
//! held against what it is:
//!
//! - how many of those modules were read the wrong colour, and how many
//!   were left unknown, which a better placement or sampling lowers long
//!   before it reads another symbol;
//! - how much of each confirmed block's correction its solution spent:
//!   twice its wrong codewords and once its unknown ones, against the
//!   codewords it may spend, so that a read with room to spare is told
//!   from one that barely passed;
//! - whether the reliability each module was given is borne out: of the
//!   modules at a reliability, as many should be wrong as the chance it
//!   stands for says ([`crate::symbol::grid::misread_chance`]). The
//!   solvers weigh modules by that chance, and a corpus says whether it
//!   is right.
//!
//! The figures are taken over the codewords of confirmed blocks alone,
//! and of a read that confirmed none there are none.

use std::fmt;
use std::sync::Arc;

use super::read::ReadResult;
use super::sample::reliability_bits;
use crate::decode::pipeline::BlockOutcome;
use crate::decode::repair::known_modules;
use crate::symbol::function_map::FunctionMap;
use crate::symbol::grid::{CERTAIN, misread_chance};
use crate::symbol::layout::SymbolLayout;

/// Whole bits of reliability past which the modules are counted
/// together: the most any sampled module is believed
/// ([`super::sample::SAMPLED_MAX`]), though the scale itself runs to
/// about thirty-three ([`reliability_bits`]).
const MOST_BITS: u32 = 8;

/// What a read made of the modules a decode came to know.
#[derive(Clone, Debug, PartialEq)]
pub struct ReadMeasures {
    /// Modules of the codewords of the confirmed blocks.
    pub modules: usize,
    /// Of those, the ones read the other colour.
    pub misread: usize,
    /// Of those, the ones left unknown.
    pub unknown: usize,
    /// Confirmed blocks.
    pub blocks: usize,
    /// The share of its correction a confirmed block spent, twice its
    /// wrong codewords and once its unknown ones over the codewords it
    /// may spend, as a mean over the blocks: nought for a block read
    /// clean, one for a block that took all it had, and more for one a
    /// solver took past what plain Reed-Solomon reaches.
    pub load_mean: f64,
    /// The most any confirmed block spent, on the same scale.
    pub load_max: f64,
    /// The modules read a colour, by the whole bits of reliability they
    /// were read at ([`reliability_bits`]), up to [`MOST_BITS`]: for each,
    /// how many there were, how many were read wrong, and how many of
    /// them the reliability expected to be wrong. Modules painted by
    /// hand, at no reliability, are not counted.
    pub calibration: Vec<CalibrationBin>,
}

/// The modules of a read believed to one whole number of bits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationBin {
    pub bits: u32,
    pub modules: usize,
    pub misread: usize,
    /// The misreads expected of them: each module's chance of being wrong
    /// ([`crate::symbol::grid::misread_chance`]), added up.
    pub expected: f64,
}

impl ReadMeasures {
    /// What `result` made of the modules its decode came to know, or
    /// `None` when it confirmed no block.
    pub fn of(result: &ReadResult) -> Option<Self> {
        let report = &result.report;
        let func = Arc::new(FunctionMap::new(result.version()));
        let layout = SymbolLayout::new(func.clone(), report.ec)?;
        let truth = known_modules(&layout, report.mask, report);
        let grid = &result.grid;
        let (mut modules, mut misread, mut unknown) = (0, 0, 0);
        let mut bins: Vec<CalibrationBin> =
            (1..=MOST_BITS).map(|bits| CalibrationBin { bits, modules: 0, misread: 0, expected: 0.0 }).collect();
        for (cw, span) in layout.codewords.iter().enumerate() {
            if !report.blocks.get(span.block as usize).is_some_and(|block| block.is_verified()) {
                continue;
            }
            for pos in layout.codeword_modules(cw) {
                let (x, y) = (pos.x as usize, pos.y as usize);
                let Some(dark) = truth[y * func.width + x] else { continue };
                modules += 1;
                let Some(read) = grid.get(x, y).as_dark() else {
                    unknown += 1;
                    continue;
                };
                let wrong = read != dark;
                misread += usize::from(wrong);
                let reliability = grid.reliability(x, y);
                if !grid.has_reliability() || reliability == CERTAIN {
                    continue;
                }
                let bits = reliability_bits(reliability);
                let bin = &mut bins[(bits.floor() as u32).clamp(1, MOST_BITS) as usize - 1];
                bin.modules += 1;
                bin.misread += usize::from(wrong);
                bin.expected += misread_chance(reliability);
            }
        }
        let loads: Vec<f64> = report
            .blocks
            .iter()
            .filter(|block| block.is_verified())
            .filter_map(|block| match block.outcome {
                BlockOutcome::Solved { errors, erasures, .. } => {
                    Some((2 * errors + erasures) as f64 / block.ec_len.saturating_sub(block.reserved).max(1) as f64)
                }
                BlockOutcome::Failed(_) => None,
            })
            .collect();
        if loads.is_empty() {
            return None;
        }
        Some(Self {
            modules,
            misread,
            unknown,
            blocks: loads.len(),
            load_mean: loads.iter().sum::<f64>() / loads.len() as f64,
            load_max: loads.iter().copied().fold(0.0, f64::max),
            calibration: bins.into_iter().filter(|bin| bin.modules > 0).collect(),
        })
    }
}

/// As `--metrics` prints the figures: `name=value` pairs, the calibration
/// as `bits:modules:misread:expected` for each whole number of bits,
/// with commas between, the misreads expected to six places: among the
/// modules of a small symbol believed to eight bits, a few dozen, a
/// tenth of one.
impl fmt::Display for ReadMeasures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let calibration: Vec<String> =
            self.calibration.iter().map(|bin| format!("{}:{}:{}:{:.6}", bin.bits, bin.modules, bin.misread, bin.expected)).collect();
        write!(
            f,
            "modules={} misread={} unknown={} blocks={} load_mean={:.3} load_max={:.3} calibration={}",
            self.modules,
            self.misread,
            self.unknown,
            self.blocks,
            self.load_mean,
            self.load_max,
            calibration.join(",")
        )
    }
}
