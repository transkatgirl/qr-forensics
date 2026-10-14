//! How the photo's misreads cluster, learnt from the blocks that
//! verified.
//!
//! A block that verifies says of every module it holds whether the
//! photo read it right: a module read as known whose bit the solution
//! overturned was misread, and every other was not. Damage is often
//! contiguous, a stain or a tear or a fold, and the blocks are
//! interleaved, so a module of a block that did not verify, beside one
//! a verified block shows misread, is one to doubt first. How much
//! first is what the verified blocks say of themselves. Among their own
//! modules beside a misread one, as many are misread as their readings
//! explain, the reliability the sampler gave each, and some more: those
//! are misread outright, whatever their reading, as a stain's modules
//! are. That share is learnt ([`Clustering::gross`]), apart for modules
//! beside one misread module and beside two or more, and every sampled
//! module beside a misread one is then believed no more than a reading
//! that is misread that often besides its own odds
//! ([`Clustering::lowered`]). Specks that do not spread teach a share
//! near nought, and leave the modules beside them nearly as read; a
//! stain teaches a large one, and its edge is doubted before anything
//! else. Until the verified blocks show enough, the share stands at one
//! in eight, as eight modules' worth of evidence.
//!
//! The lowering is evidence about the photo from blocks other than the
//! one solved, so it stands in the reliabilities a solution is judged
//! by as well as those it is searched by: what confirms a solution asks
//! nothing of the reliabilities but that they were fixed before the
//! block was read ([`super::rs::soft_count_bits`]).

use super::codewords::codeword_bit_mask;
use super::list::misread;
use super::pipeline::{BlockReport, CodewordReport};
use super::rs::CwRead;
use crate::symbol::grid::CERTAIN;
use crate::symbol::layout::SymbolLayout;

/// The share of the modules beside a misread one taken as misread
/// outright before the verified blocks say otherwise: one in eight,
/// three bits of likelihood, which is what such a module was held at
/// before the share was learnt.
const PRIOR_GROSS: f64 = 1.0 / 8.0;
/// As how many modules the prior share counts against those the
/// verified blocks show.
const PRIOR_MODULES: f64 = 8.0;

/// The modules of the verified blocks beside some number of misread
/// ones: how many, how many of them were misread, and how many their
/// readings alone would have had misread.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tally {
    pub modules: usize,
    pub misread: usize,
    pub expected: f64,
}

/// What the verified blocks show of how the photo's misreads cluster.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clustering {
    /// Misread modules the verified blocks located.
    pub located: usize,
    /// Their sampled modules beside one misread module, and beside two
    /// or more.
    pub beside: [Tally; 2],
    /// The share of a module beside one misread module, and beside two
    /// or more, misread outright whatever its reading, as learnt: what
    /// [`Self::lowered`] holds such modules to.
    pub gross: [f64; 2],
}

impl Clustering {
    /// Learn the clustering from the blocks that verified, with how many
    /// misread modules lie beside each module of the grid, row by row.
    /// None when the verified blocks located no misread module.
    pub(super) fn learn(
        layout: &SymbolLayout,
        raw: &[CwRead],
        codewords: &[CodewordReport],
        blocks: &[BlockReport],
        module_reliability: &[u8],
    ) -> Option<(Self, Vec<u8>)> {
        let (width, height) = (layout.func.width, layout.func.height);
        // Whether each module of a verified block, read as known, was
        // misread.
        let mut misread_at: Vec<Option<bool>> = vec![None; width * height];
        for (b, spec) in layout.blocks.iter().enumerate() {
            if !blocks[b].is_verified() {
                continue;
            }
            for &cw in &spec.codewords {
                let Some(value) = codewords[cw].final_value else { continue };
                let exists = codeword_bit_mask(layout, cw);
                let overturned = (value ^ raw[cw].value) & raw[cw].known & exists;
                let offset = layout.codewords[cw].bit_offset as usize;
                for (i, pos) in layout.codeword_modules(cw).iter().enumerate() {
                    let bit = 0x80 >> (offset + i);
                    if raw[cw].known & bit != 0 {
                        misread_at[pos.y as usize * width + pos.x as usize] = Some(overturned & bit != 0);
                    }
                }
            }
        }
        let located = misread_at.iter().filter(|&&m| m == Some(true)).count();
        if located == 0 {
            return None;
        }
        let mut beside = vec![0u8; width * height];
        for (at, _) in misread_at.iter().enumerate().filter(|(_, m)| **m == Some(true)) {
            let (x, y) = (at % width, at / width);
            for ny in y.saturating_sub(1)..(y + 2).min(height) {
                for nx in x.saturating_sub(1)..(x + 2).min(width) {
                    if (nx, ny) != (x, y) {
                        beside[ny * width + nx] = beside[ny * width + nx].saturating_add(1);
                    }
                }
            }
        }
        let mut tallies = [Tally::default(); 2];
        for (at, misread_here) in misread_at.iter().enumerate() {
            let (Some(misread_here), reliability) = (*misread_here, module_reliability[at]) else { continue };
            if beside[at] == 0 || reliability == CERTAIN {
                continue;
            }
            let tally = &mut tallies[usize::from(beside[at] >= 2)];
            tally.modules += 1;
            tally.misread += usize::from(misread_here);
            tally.expected += misread(reliability);
        }
        let one = gross_of(tallies[0]);
        let gross = [one, gross_of(tallies[1]).max(one)];
        Some((Self { located, beside: tallies, gross }, beside))
    }

    /// The reliabilities of `module_reliability` with every sampled
    /// module beside a misread one, as `beside` counts them, believed no
    /// more than a reading of its reliability that is misread outright
    /// as often as [`Self::gross`] says besides. Never raised; modules
    /// painted by hand stay certain.
    pub(super) fn lowered(&self, module_reliability: &[u8], beside: &[u8]) -> Vec<u8> {
        module_reliability
            .iter()
            .zip(beside)
            .map(|(&reliability, &count)| {
                if count == 0 || reliability == CERTAIN {
                    reliability
                } else {
                    reliability.min(doubted(reliability, self.gross[usize::from(count >= 2)]))
                }
            })
            .collect()
    }
}

/// The share misread outright among the modules `tally` counts: those
/// misread beyond what their readings explain, over those their readings
/// do not have misread, with the prior's eight modules at one in eight
/// counted in.
fn gross_of(tally: Tally) -> f64 {
    let beyond = (tally.misread as f64 - tally.expected).max(0.0);
    let unexplained = (tally.modules as f64 - tally.expected).max(0.0);
    (beyond + PRIOR_GROSS * PRIOR_MODULES) / (unexplained + PRIOR_MODULES)
}

/// The reliability, on the sampler's scale, of a module read at
/// `reliability` that is also misread outright one time in `1 / gross`.
fn doubted(reliability: u8, gross: f64) -> u8 {
    let wrong = gross + (1.0 - gross) * misread(reliability);
    ((-wrong.log2() - 1.0) * 8.0).round().clamp(0.0, f64::from(CERTAIN - 1)) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_nothing_learnt_a_solid_module_is_held_at_one_in_eight() {
        let gross = gross_of(Tally::default());
        assert_eq!(gross, PRIOR_GROSS);
        assert_eq!(doubted(200, gross), 16);
        // A faint module is doubted the more for it, and never raised.
        assert!(doubted(8, gross) < 8);
    }

    #[test]
    fn specks_teach_a_small_share_and_a_stain_a_large_one() {
        // Forty solid modules beside isolated misreads, none of them
        // misread: the share falls toward nought.
        let specks = gross_of(Tally { modules: 40, misread: 0, expected: 0.0 });
        assert!(specks < 0.025, "{specks}");
        assert!(doubted(200, specks) > 35);
        // Forty beside a stain's modules, twenty of them misread.
        let stain = gross_of(Tally { modules: 40, misread: 20, expected: 0.0 });
        assert!(stain > 0.4, "{stain}");
        assert!(doubted(200, stain) <= 2);
        // Misreads the modules' own readings explain teach nothing: forty
        // faint modules, each read at a coin's toss, half of them wrong.
        let faint = gross_of(Tally { modules: 40, misread: 20, expected: 20.0 });
        assert!((faint - PRIOR_GROSS * PRIOR_MODULES / (20.0 + PRIOR_MODULES)).abs() < 1e-9, "{faint}");
    }
}
