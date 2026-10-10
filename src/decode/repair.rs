//! Writing back what decoding established: the grid with every solved
//! codeword's modules set to their corrected values and the function
//! patterns regenerated, and a rendering of it that an ordinary reader can
//! scan.

use super::bch;
use super::pipeline::{BlockReport, DecodeReport};
use crate::symbol::function_map::InfoCopy;
use crate::symbol::grid::{Module, ModuleGrid};
use crate::symbol::layout::SymbolLayout;
use crate::symbol::mask::mask_bit;
use crate::symbol::{Family, SymbolVersion};

fn paint_info(grid: &mut ModuleGrid, copies: &[InfoCopy], word: u32) {
    for copy in copies {
        let painted = word ^ copy.xor;
        for (i, pos) in copy.bits.iter().enumerate() {
            grid.set(pos.x as usize, pos.y as usize, Module::from_dark(painted >> i & 1 != 0));
        }
    }
}

/// Whether the decode vouches for the layout's EC level and mask: a block
/// solved under them, or the grid's own format bits name them beyond
/// doubt. A level and mask that were only guessed, with nothing solved to
/// bear them out, are no more than the layout in effect, and writing
/// their format word back would turn the guess into bits that read as
/// evidence for it.
fn format_established(grid: &ModuleGrid, layout: &SymbolLayout, mask: u8, report: &DecodeReport) -> bool {
    if report.blocks.iter().any(BlockReport::is_solved) {
        return true;
    }
    let format = bch::analyze_format(grid, &layout.func);
    format.is_reliable() && format.best().is_some_and(|b| b.fields.ec == layout.ec && b.fields.mask == mask)
}

/// The grid as error correction says it should be: fixed patterns and
/// version information regenerated, the format information regenerated
/// for the layout in effect once the decode vouches for it (a block
/// solved under it, or the grid's own format bits naming it), the modules
/// of every solved codeword set to their corrected values, and remainder
/// bits cleared. Modules of unsolved codewords are left as they are, so
/// the result is complete exactly when the decode was.
pub fn repaired_grid(grid: &ModuleGrid, layout: &SymbolLayout, mask: u8, report: &DecodeReport) -> ModuleGrid {
    let func = &layout.func;
    let version = func.version;
    let mut out = grid.clone();
    func.fill_fixed(&mut out);
    if format_established(grid, layout, mask, report)
        && let Some(word) = bch::format_word_for(version, layout.ec, mask)
    {
        paint_info(&mut out, &func.format, word);
    }
    if let SymbolVersion::Qr(v) = version
        && v >= 7
    {
        paint_info(&mut out, &func.version_info, bch::info18(v as u32));
    }
    for (cw, item) in report.codewords.iter().enumerate() {
        let Some(value) = item.final_value else { continue };
        let offset = layout.codewords[cw].bit_offset as usize;
        for (bit, pos) in layout.codeword_modules(cw).iter().enumerate() {
            let (x, y) = (pos.x as usize, pos.y as usize);
            let dark = (value & (0x80 >> (offset + bit)) != 0) != mask_bit(version, mask, x, y);
            out.set(x, y, Module::from_dark(dark));
        }
    }
    for pos in &func.data_path[func.codeword_bits()..] {
        let (x, y) = (pos.x as usize, pos.y as usize);
        out.set(x, y, Module::from_dark(mask_bit(version, mask, x, y)));
    }
    out
}

/// What the decode established of every module, row-major over the
/// symbol: the colour of each fixed module, of the format and version
/// information under the layout in effect, of every module of a codeword
/// in a block that verified, corrected, and of the remainder bits; `None`
/// for the modules of codewords in blocks that did not verify. What
/// [`repaired_grid`] writes back, as far as it is trusted, and nothing of
/// the grid as read: a template of the symbol that a placement can be
/// refitted to.
pub fn known_modules(layout: &SymbolLayout, mask: u8, report: &DecodeReport) -> Vec<Option<bool>> {
    let func = &layout.func;
    let version = func.version;
    let width = func.width;
    let mut out: Vec<Option<bool>> =
        (0..func.height).flat_map(|y| (0..width).map(move |x| (x, y))).map(|(x, y)| func.expected(x, y)).collect();
    fn paint(out: &mut [Option<bool>], width: usize, copies: &[InfoCopy], word: u32) {
        for copy in copies {
            let painted = word ^ copy.xor;
            for (i, pos) in copy.bits.iter().enumerate() {
                out[pos.y as usize * width + pos.x as usize] = Some(painted >> i & 1 != 0);
            }
        }
    }
    if let Some(word) = bch::format_word_for(version, layout.ec, mask) {
        paint(&mut out, width, &func.format, word);
    }
    if let SymbolVersion::Qr(v) = version
        && v >= 7
    {
        paint(&mut out, width, &func.version_info, bch::info18(v as u32));
    }
    for (cw, item) in report.codewords.iter().enumerate() {
        let span = layout.codewords[cw];
        if !report.blocks[span.block as usize].is_verified() {
            continue;
        }
        let Some(value) = item.final_value else { continue };
        for (bit, pos) in layout.codeword_modules(cw).iter().enumerate() {
            let (x, y) = (pos.x as usize, pos.y as usize);
            out[y * width + x] = Some((value & (0x80 >> (span.bit_offset as usize + bit)) != 0) != mask_bit(version, mask, x, y));
        }
    }
    for pos in &func.data_path[func.codeword_bits()..] {
        let (x, y) = (pos.x as usize, pos.y as usize);
        out[y * width + x] = Some(mask_bit(version, mask, x, y));
    }
    out
}

/// Width of the quiet zone the standard asks for, in modules.
pub fn quiet_zone(version: SymbolVersion) -> usize {
    match version.family() {
        Family::Qr | Family::Model1 => 4,
        Family::Micro | Family::Rmqr => 2,
    }
}

/// Render a grid as RGBA pixels, `scale` pixels per module with `quiet`
/// modules of white around it. Unknown modules come out mid-grey.
pub fn render_rgba(grid: &ModuleGrid, scale: usize, quiet: usize) -> (usize, usize, Vec<u8>) {
    let width = (grid.width() + 2 * quiet) * scale;
    let height = (grid.height() + 2 * quiet) * scale;
    let mut pixels = vec![255u8; width * height * 4];
    for y in 0..grid.height() {
        for x in 0..grid.width() {
            let shade = match grid.get(x, y) {
                Module::Dark => 0,
                Module::Light => continue,
                Module::Unknown => 128,
            };
            for py in 0..scale {
                let row = ((y + quiet) * scale + py) * width + (x + quiet) * scale;
                for px in 0..scale {
                    pixels[(row + px) * 4..(row + px) * 4 + 3].fill(shade);
                }
            }
        }
    }
    (width, height, pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rendering_scales_and_pads() {
        let grid = ModuleGrid::from_text("1?\n01").unwrap();
        let (w, h, pixels) = render_rgba(&grid, 3, 1);
        assert_eq!((w, h), (12, 12));
        assert_eq!(pixels.len(), 12 * 12 * 4);
        let at = |x: usize, y: usize| &pixels[(y * w + x) * 4..(y * w + x) * 4 + 4];
        assert_eq!(at(0, 0), [255, 255, 255, 255], "quiet zone");
        assert_eq!(at(3, 3), [0, 0, 0, 255], "dark module");
        assert_eq!(at(5, 5), [0, 0, 0, 255], "last pixel of the dark module");
        assert_eq!(at(6, 3), [128, 128, 128, 255], "unknown module");
        assert_eq!(at(3, 6), [255, 255, 255, 255], "light module");
        assert_eq!(at(8, 8), [0, 0, 0, 255]);
        assert_eq!(quiet_zone(SymbolVersion::Qr(1)), 4);
        assert_eq!(quiet_zone(SymbolVersion::Rmqr(0)), 2);
    }
}
