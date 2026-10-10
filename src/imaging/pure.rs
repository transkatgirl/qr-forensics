//! Symbols rendered pixel for pixel, with no camera between: screenshots
//! and files written by an encoder. The photo pipeline fits such a symbol
//! as it fits any other, and lands a fraction of a pixel off, which at a
//! pixel or two a module is a module's worth. Here the symbol is read off
//! the pixels themselves, as zxing-cpp's `DetectPureQR` reads one: the box
//! the ink fills is the symbol, the 1:1:3:1:1 runs down the diagonal from
//! a corner of it measure the module, and the box divided by the module is
//! the size. A photo never passes: its ink has no clean box and no clean
//! runs, and the fit of the fixed patterns, which the caller checks, is
//! the last word.
//!
//! A small photo is enlarged before the pipeline sees it
//! ([`super::photo::enlarged`]), smoothly, and a render so enlarged has
//! no clean runs either: its edges are ramps and its corners round. It
//! is read off the pixels it had, which the enlarged image keeps beside
//! it ([`Luma::unenlarged`]), each counted as the pixels it was made
//! into, so that what is measured and where is in the pixels the photo
//! has now.

use super::Luma;
use super::sample::otsu;
use crate::symbol::{Family, SymbolVersion};

/// Each of the five runs down a finder's diagonal may be off its ideal
/// length by this share of a module, and by a pixel whatever the module.
const RUN_TOLERANCE: f64 = 0.5;
/// How far the box's sides may be from a whole number of modules.
const SIZE_TOLERANCE: f64 = 0.35;
/// Fewest pixels across the region worth looking into.
const MIN_REGION_PX: usize = 8;

/// A symbol read straight off the pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct PureSymbol {
    /// Top-left, top-right, bottom-right, bottom-left in photo pixels, in
    /// the symbol's own order, on the outer edges of its outermost pixels.
    pub corners: [[f64; 2]; 4],
    pub version: SymbolVersion,
    /// Light on dark.
    pub inverted: bool,
    /// Quarter turns clockwise the symbol sits at in the photo.
    pub turns: u8,
    /// Pixels a module, along the photo's axes.
    pub module: f64,
}

/// The symbol of `family` rendered pixel for pixel inside `region`, if
/// there is one: the first of [`find_each`].
pub fn find(image: &Luma, family: Family, region: [[f64; 2]; 4]) -> Option<PureSymbol> {
    find_each(image, family, region).into_iter().next()
}

/// What `region` holds read as a symbol of `family` rendered pixel for
/// pixel, dark on light and then light on dark. Either reading may be a
/// false one, which the fit of the fixed patterns tells: the dark margin
/// of a light-on-dark Micro QR symbol, taken as ink, has the runs of a
/// finder down its diagonal often enough, and must not hide the symbol
/// the other reading finds.
pub fn find_each(image: &Luma, family: Family, region: [[f64; 2]; 4]) -> Vec<PureSymbol> {
    // An enlarged photo is read at its own size, each of its pixels
    // `times` of the enlarged across.
    let (image, times) = match &image.unenlarged {
        Some(own) => (&**own, (image.width / own.width.max(1)).max(1)),
        None => (image, 1),
    };
    let (width, height) = (image.width, image.height);
    let edge = |axis: usize, limit: usize, up: bool| {
        let v = region.iter().map(|p| p[axis]).fold(if up { f64::NEG_INFINITY } else { f64::INFINITY }, if up { f64::max } else { f64::min });
        let v = v / times as f64;
        if !v.is_finite() {
            return None;
        }
        Some((if up { v.ceil() } else { v.floor() }).clamp(0.0, limit as f64) as usize)
    };
    let (Some(x0), Some(x1), Some(y0), Some(y1)) =
        (edge(0, width, false), edge(0, width, true), edge(1, height, false), edge(1, height, true))
    else {
        return Vec::new();
    };
    if (x1 - x0) * times < MIN_REGION_PX || (y1 - y0) * times < MIN_REGION_PX {
        return Vec::new();
    }
    let threshold = otsu((y0..y1).flat_map(|y| (x0..x1).map(move |x| image.data[y * width + x])));
    // A symbol is dark on light or, whole, light on dark.
    [false, true]
        .into_iter()
        .filter_map(|inverted| {
            let ink = |x: usize, y: usize| (image.data[y * width + x] < threshold) != inverted;
            find_ink(family, [x0, y0, x1, y1], &ink, inverted, times)
        })
        .collect()
}

/// The symbol whose ink fills `bounds` (`[x0, y0, x1, y1]`, exclusive at
/// the far side), read with the `ink` predicate. Each pixel `ink` reads
/// is `times` pixels of the photo across, and the symbol is measured and
/// placed in those.
fn find_ink(family: Family, bounds: [usize; 4], ink: &dyn Fn(usize, usize) -> bool, inverted: bool, times: usize) -> Option<PureSymbol> {
    let [x0, y0, x1, y1] = bounds;
    // The box the ink fills, inclusive.
    let (mut l, mut t, mut r, mut b) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for y in y0..y1 {
        for x in x0..x1 {
            if ink(x, y) {
                l = l.min(x);
                r = r.max(x);
                t = t.min(y);
                b = b.max(y);
            }
        }
    }
    if l == usize::MAX || (r - l + 1) * times < 7 || (b - t + 1) * times < 7 {
        return None;
    }
    let scale = times as f64;
    let (bw, bh) = ((r - l + 1) as f64 * scale, (b - t + 1) as f64 * scale);
    // A step past the box, where the paper ends the last run of a finder
    // that fills a symbol's whole height, as an rMQR's does.
    let reach = (r - l).min(b - t) + 2;
    // The runs down the diagonal in from each corner of the box, as
    // (start, step) along each axis for the top-left, top-right,
    // bottom-right and bottom-left corners in turn.
    let diagonal = |corner: usize| -> (usize, usize, i64, i64) {
        match corner {
            0 => (l, t, 1, 1),
            1 => (r, t, -1, 1),
            2 => (r, b, -1, -1),
            _ => (l, b, 1, -1),
        }
    };
    let along = |corner: usize, i: i64| {
        let (sx, sy, dx, dy) = diagonal(corner);
        let (x, y) = (sx as i64 + i * dx, sy as i64 + i * dy);
        // Beyond the region is paper.
        x >= x0 as i64 && x < x1 as i64 && y >= y0 as i64 && y < y1 as i64 && ink(x as usize, y as usize)
    };
    let finders: Vec<(usize, f64)> =
        (0..4).filter_map(|c| finder_module(runs::<5>(|i| along(c, i), reach, scale)?).map(|m| (c, m))).collect();
    // The placements the finders allow, as the quarter turns the symbol
    // sits at and the module each measures. A symbol only seven modules
    // tall has its finder at two corners of the box at once, and the box
    // itself then says which placement is real.
    let placements: Vec<(usize, f64)> = match family {
        Family::Qr | Family::Model1 => {
            // Three finders, and the corner without one says which way up:
            // the symbol's empty corner, its bottom right, turns with it.
            if finders.len() != 3 {
                return None;
            }
            let missing = (0..4).find(|c| finders.iter().all(|f| f.0 != *c))?;
            vec![((missing + 2) % 4, finders.iter().map(|f| f.1).sum::<f64>() / 3.0)]
        }
        Family::Micro => finders,
        // The finder at one corner and the sub-pattern at the opposite one:
        // dark, light, dark, light down its own diagonal, a module each,
        // and then its dark centre, which runs on into whatever data
        // module lies beyond, so is not asked.
        Family::Rmqr => finders
            .into_iter()
            .filter(|&(corner, module)| {
                sub_module(runs::<4>(|i| along((corner + 2) % 4, i), reach, scale).unwrap_or([0.0; 4]))
                    .is_some_and(|sub| (sub - module).abs() <= (RUN_TOLERANCE * module).max(1.0))
            })
            .collect(),
    };
    let edges = [[l as f64, t as f64], [r as f64 + 1.0, t as f64], [r as f64 + 1.0, b as f64 + 1.0], [l as f64, b as f64 + 1.0]]
        .map(|[x, y]| [x * scale, y * scale]);
    let mut found: Vec<PureSymbol> = placements
        .into_iter()
        .filter_map(|(turns, module)| {
            let whole = |px: f64| {
                let n = px / module;
                (n - n.round()).abs() <= SIZE_TOLERANCE.max(1.0 / module) || module <= 1.0
            };
            if !whole(bw) || !whole(bh) {
                return None;
            }
            let (across, down) = ((bw / module).round() as i64, (bh / module).round() as i64);
            // The box's sides in the symbol's own frame: on end, its width
            // runs down the photo.
            let (sw, sh) = if turns % 2 == 1 { (down, across) } else { (across, down) };
            let mut fitting: Vec<(i64, SymbolVersion)> = family
                .versions()
                .into_iter()
                .filter_map(|v| {
                    let (dw, dh) = ((v.width() as i64 - sw).abs(), (v.height() as i64 - sh).abs());
                    (dw <= 1 && dh <= 1).then_some((dw + dh, v))
                })
                .collect();
            fitting.sort_by_key(|f| f.0);
            let version = match fitting[..] {
                [(_, v)] => v,
                [(a, v), (b, _), ..] if a < b => v,
                _ => return None,
            };
            // The symbol's top-left corner sits at the photo corner it was
            // turned to, and the rest follow round.
            let corners = std::array::from_fn(|i| edges[(i + turns) % 4]);
            Some(PureSymbol { corners, version, inverted, turns: turns as u8, module })
        })
        .collect();
    match found.len() {
        1 => found.pop(),
        _ => None,
    }
}

/// The first `N` runs of ink, paper, ink, ... along a line, by length in
/// pixels of `step` to each step along it, starting on ink; `None` when
/// the line does not start on ink or the last run has not ended within
/// `reach` steps.
fn runs<const N: usize>(ink: impl Fn(i64) -> bool, reach: usize, step: f64) -> Option<[f64; N]> {
    if !ink(0) {
        return None;
    }
    let mut out = [0.0; N];
    let mut run = 0;
    for i in 0..reach as i64 {
        let on = ink(i);
        if on != (run % 2 == 0) {
            run += 1;
            if run == N {
                return Some(out);
            }
        }
        out[run] += step;
    }
    None
}

/// The module the runs down a finder's diagonal imply when they are in
/// the finder's 1:1:3:1:1 proportion.
fn finder_module(runs: [f64; 5]) -> Option<f64> {
    let module = runs.iter().sum::<f64>() / 7.0;
    let tolerance = (RUN_TOLERANCE * module).max(1.0);
    let ideal = [1.0, 1.0, 3.0, 1.0, 1.0];
    runs.iter().zip(ideal).all(|(&run, want)| (run - want * module).abs() <= tolerance).then_some(module)
}

/// The module the first four runs down an rMQR finder sub-pattern's
/// diagonal imply when they are in its 1:1:1:1 proportion.
fn sub_module(runs: [f64; 4]) -> Option<f64> {
    let module = runs.iter().sum::<f64>() / 4.0;
    let tolerance = (RUN_TOLERANCE * module).max(1.0);
    runs.iter().all(|&run| (run - module).abs() <= tolerance).then_some(module)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::function_map::FunctionMap;

    /// A version's fixed patterns drawn `px` pixels a module with a quiet
    /// zone of `quiet` modules, turned `turns` quarter turns clockwise,
    /// with every other module light.
    fn render(version: SymbolVersion, px: usize, quiet: usize, turns: usize, inverted: bool) -> Luma {
        let func = FunctionMap::new(version);
        let (w, h) = (func.width, func.height);
        let (tw, th) = if turns % 2 == 1 { (h, w) } else { (w, h) };
        let (width, height) = ((tw + 2 * quiet) * px, (th + 2 * quiet) * px);
        let mut data = vec![if inverted { 0.0 } else { 1.0 }; width * height];
        for y in 0..h {
            for x in 0..w {
                if func.expected(x, y) != Some(true) {
                    continue;
                }
                // Where module (x, y) lands after the turn.
                let (tx, ty) = match turns % 4 {
                    0 => (x, y),
                    1 => (h - 1 - y, x),
                    2 => (w - 1 - x, h - 1 - y),
                    _ => (y, w - 1 - x),
                };
                for dy in 0..px {
                    for dx in 0..px {
                        data[((ty + quiet) * px + dy) * width + (tx + quiet) * px + dx] = if inverted { 1.0 } else { 0.0 };
                    }
                }
            }
        }
        Luma::new(width, height, data)
    }

    #[test]
    fn renders_are_read_off_their_pixels_at_every_turn_and_size() {
        for (version, family) in [
            (SymbolVersion::Qr(1), Family::Qr),
            (SymbolVersion::Qr(5), Family::Qr),
            (SymbolVersion::Qr(23), Family::Qr),
            (SymbolVersion::Micro(3), Family::Micro),
            (SymbolVersion::Rmqr(0), Family::Rmqr),
            (SymbolVersion::Rmqr(31), Family::Rmqr),
        ] {
            for turns in 0..4 {
                for (px, quiet) in [(1, 2), (3, 4), (7, 0)] {
                    for inverted in [false, true] {
                        let image = render(version, px, quiet, turns, inverted);
                        let whole = [[0.0, 0.0], [image.width as f64, 0.0], [image.width as f64, image.height as f64], [0.0, image.height as f64]];
                        let found = find(&image, family, whole)
                            .unwrap_or_else(|| panic!("{version} at {px} px, turn {turns}, quiet {quiet}, inverted {inverted}"));
                        assert_eq!(found.version, version);
                        assert_eq!(found.turns as usize, turns);
                        assert_eq!(found.inverted, inverted);
                        assert_eq!(found.module, px as f64);
                        let q = (quiet * px) as f64;
                        // The symbol's own top-left corner is at the photo
                        // corner the turn carried it to.
                        let (tw, th) = ((image.width as f64) - 2.0 * q, (image.height as f64) - 2.0 * q);
                        let edges = [[q, q], [q + tw, q], [q + tw, q + th], [q, q + th]];
                        assert_eq!(found.corners[0], edges[turns], "{version} turn {turns}");
                    }
                }
            }
        }
    }

    #[test]
    fn an_enlarged_render_reads_as_it_would_with_each_pixel_repeated() {
        for (version, family, px, quiet) in [
            (SymbolVersion::Qr(1), Family::Qr, 1, 2),
            (SymbolVersion::Qr(2), Family::Qr, 1, 2),
            (SymbolVersion::Qr(5), Family::Qr, 1, 2),
            (SymbolVersion::Qr(12), Family::Qr, 1, 0),
            (SymbolVersion::Micro(1), Family::Micro, 1, 0),
            (SymbolVersion::Micro(3), Family::Micro, 2, 2),
            (SymbolVersion::Rmqr(0), Family::Rmqr, 1, 0),
            (SymbolVersion::Rmqr(31), Family::Rmqr, 3, 1),
        ] {
            for (turns, times, inverted) in [(0, 2, false), (1, 3, true), (2, 7, false), (3, 24, false)] {
                let own = render(version, px, quiet, turns, inverted);
                let (width, height) = (own.width * times, own.height * times);
                // Every pixel repeated: what enlarging once made of it.
                let repeated = Luma::new(width, height, (0..width * height).map(|i| own.data[i / width / times * own.width + i % width / times]).collect());
                // Enlarged as it is now, with its own pixels beside it.
                let enlarged = Luma { unenlarged: Some(std::sync::Arc::new(own.clone())), ..own.resized(width, height) };
                let whole = [[0.0, 0.0], [width as f64, 0.0], [width as f64, height as f64], [0.0, height as f64]];
                let name = format!("{version} at {px} px, quiet {quiet}, turn {turns}, {times} times, inverted {inverted}");
                let found = find(&enlarged, family, whole).unwrap_or_else(|| panic!("{name}"));
                assert_eq!((found.version, found.turns as usize, found.inverted, found.module), (version, turns, inverted, (px * times) as f64), "{name}");
                assert_eq!(find_each(&enlarged, family, whole), find_each(&repeated, family, whole), "{name}");
                // An outline in the enlarged pixels, a little way into
                // the quiet zone, says where to look as the whole does.
                let inset = (quiet * px * times) as f64 / 2.0;
                let outline = [[inset, inset], [width as f64 - inset, inset], [width as f64 - inset, height as f64 - inset], [inset, height as f64 - inset]];
                assert_eq!(find_each(&enlarged, family, outline), find_each(&repeated, family, outline), "{name}, in an outline");
                // Asked for another family, it is no more one enlarged
                // than repeated.
                for other in [Family::Qr, Family::Micro, Family::Rmqr].into_iter().filter(|&other| other != family) {
                    assert_eq!(find_each(&enlarged, other, whole), find_each(&repeated, other, whole), "{name}, as {other:?}");
                }
            }
        }
    }

    #[test]
    fn a_photo_or_a_wrong_box_is_not_a_pure_symbol() {
        let version = SymbolVersion::Qr(2);
        let mut image = render(version, 4, 2, 0, false);
        let whole = [[0.0, 0.0], [image.width as f64, 0.0], [image.width as f64, image.height as f64], [0.0, image.height as f64]];
        assert!(find(&image, Family::Qr, whole).is_some());
        // Print beside the symbol stretches the box.
        image.data[5 * image.width + 2] = 0.0;
        assert!(find(&image, Family::Qr, whole).is_none());
        let image = render(version, 4, 2, 0, false);
        // Only a corner of it in the region.
        let part = [[0.0, 0.0], [40.0, 0.0], [40.0, 40.0], [0.0, 40.0]];
        assert!(find(&image, Family::Qr, part).is_none());
        // Asked for the wrong family.
        assert!(find(&image, Family::Micro, whole).is_none());
        assert!(find(&image, Family::Rmqr, whole).is_none());
    }
}
