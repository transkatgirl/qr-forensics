//! Anchors on patches of data: where no fixed pattern is, the modules of
//! the data say where the grid runs, by reading crisply dark or light
//! where it lies on them and grey where it straddles them.
//!
//! The map from modules to the photo is bent through the anchors the
//! detector finds ([`super::projection`]): the finder patterns, the
//! alignment patterns, a few dark modules of each timing pattern, and
//! points along the symbol's edges. Between them it is the spline's
//! guess. On a large symbol at two pixels a module, its alignment
//! patterns twenty-odd modules apart, a tenth of a module of curl
//! between two of them is a fifth of a pixel, and the curl of a creased
//! page is several tenths: the middle of each cell between the patterns
//! is read a fraction of a module off, where the patterns themselves
//! read right. A photograph of a version 34 symbol curled across its
//! width, at two pixels a module, read four blocks of thirty-seven so.
//!
//! The data modules are everywhere the patterns are not, and each says
//! where the grid runs about it. A grid laid true samples each module in
//! its middle, where it reads its own colour; one a fraction of a module
//! off samples the boundaries between them, which read grey wherever
//! two neighbours differ. So about each point of a lattice a few
//! modules apart, far from any anchor, the grid is moved a little each
//! way, and the move under which the modules about the point read most
//! crisply ([`super::detect`]'s corner search asks the same of a
//! quarter of the symbol) sets the grid on them. No module's colour
//! is assumed; that they have colours is enough.
//!
//! How crisply they read says only that the grid is on them, not where
//! on them: each module is read about its middle, two tenths of a module
//! either way, and every move that keeps those points inside their
//! modules reads as crisply as the next. On a sharp photo that is a
//! fifth of a module and more either way of the middle, and the most
//! crisply read move, which was the anchor, landed anywhere in it: on a
//! clean render at seven pixels a module, patches stood up to a sixth of
//! a module off where the finders and alignment patterns stood to a few
//! hundredths. So each module is read where the grid is on them, dark
//! or light, and the anchor is placed where the contrast across the
//! boundaries between those of opposite colour is greatest, as every
//! fixed pattern is placed ([`super::refine`]): that falls off as soon
//! as the grid leaves the boundaries, either way, and puts the patches
//! as near as the patterns.
//!
//! A move of a whole module reads as crisply as none, since the data is
//! as crisp a module over: the search reaches less than half a module
//! each way, and the points are taken nearest the anchors first, a band
//! at a time, each band read through the map the bands before it bent,
//! so that a point is moved from where its neighbours have put it and
//! not from where the corners alone would. A point is anchored only
//! where its modules read clearly crisper there than half a module off
//! either way, which a patch of one colour, or of noise, does not.
//!
//! The read anchors every placement so before it samples it, once its
//! corners and its patterns' anchors are refined ([`super::read`]), and
//! the decode judges what it reads as it judges every other read. A
//! placement read again without one of its anchors, or expecting a
//! bleed, has its patches placed again through the map it then makes;
//! one whose patched reads settle nothing is read without its patches as
//! well, since on a photo too blurred or dim for its data to read crisply
//! anywhere they are placed by noise.

use super::Luma;
use super::detect::{Anchor, AnchorKind, Detection};
use super::projection::{ControlPoint, Project, Projection};
use super::refine::{Moved, Scorer, Shifted, Template, descend_by};
use super::sample::otsu;
use crate::symbol::function_map::FunctionMap;
use crate::work;

/// Fewest modules between the points of the lattice; a large symbol has
/// them further apart, about [`ACROSS`] of them across it.
const STEP: usize = 6;
/// Points of the lattice across a symbol whose side is more than
/// [`STEP`] times this.
const ACROSS: usize = 10;
/// Modules either side of a point that its patch reaches: a patch of
/// seven by seven.
const REACH: i64 = 3;
/// Modules a point must lie from every anchor, and from every corner, to
/// be anchored: nearer, the anchor already holds the map there.
const CLEAR: f64 = 4.0;
/// Furthest move tried each way, in modules: short of half a module, at
/// which the patch reads as crisply as it would a whole module over.
const MOST_MOVE: f64 = 0.4;
/// Moves tried each way of the furthest, a tenth of a module apart; the
/// best is then settled on the boundaries between the modules.
const MOVES: i64 = 4;
/// How much crisper the patch must read where it is placed than half a
/// module off along either axis, as a share of how crisp it reads
/// there: a patch of data reads about half as crisp straddling its
/// modules, and a patch of one colour, or of noise, as crisp either way.
const PEAK: f64 = 0.25;
/// Fewest modules of a patch that must be on the photo and not of a
/// fixed pattern.
const LEAST_DATA: usize = 24;
/// Most patches anchored: the map is solved over every anchor, and read
/// through each of them for every point of the symbol it places.
const MOST: usize = 100;
/// Share of a module's side, either side of its middle, that a module's
/// reading is taken over, three by three points: the middle of it, as
/// the sampler reads it ([`super::sample::INNER_FRACTION`]).
const SPAN: f64 = 0.2;

/// The points of the photo each module of `patch` is read at through
/// `map`, three by three about its middle ([`SPAN`]): where they lie
/// does not hang on the move tried, which is added to each.
fn read_at(map: &impl Project, patch: &[[f64; 2]]) -> Vec<[[f64; 2]; 9]> {
    let span = [-SPAN, 0.0, SPAN];
    patch.iter().map(|&[mx, my]| std::array::from_fn(|k| map.apply(mx + span[k % 3], my + span[k / 3]))).collect()
}

/// What each module at `points` ([`read_at`]) reads moved by `shift`
/// pixels: the mean of its nine points, `None` where any is off the
/// photo.
fn readings(image: &Luma, points: &[[[f64; 2]; 9]], shift: [f64; 2]) -> Vec<Option<f32>> {
    points
        .iter()
        .map(|module| {
            let mut total = 0.0f32;
            for &[x, y] in module {
                total += image.sample(x + shift[0], y + shift[1])?;
            }
            Some(total / 9.0)
        })
        .collect()
}

/// How crisply the modules read at `points` ([`read_at`]) read moved by
/// `shift` pixels: the mean distance of their readings from the split
/// between dark and light among them. `None` when too few are on the
/// photo.
fn crispness(image: &Luma, points: &[[[f64; 2]; 9]], shift: [f64; 2]) -> Option<f64> {
    let values: Vec<f32> = readings(image, points, shift).into_iter().flatten().collect();
    if values.len() < LEAST_DATA {
        return None;
    }
    let split = otsu(values.iter().copied());
    Some(values.iter().map(|v| f64::from((v - split).abs())).sum::<f64>() / values.len() as f64)
}

/// The anchor a patch about `at` places through `map`, as the module
/// describes, its boundaries expected `bloom` modules into what reads
/// dark; `None` where the patch says nothing clearly.
fn anchor_at(image: &Luma, func: &FunctionMap, map: &impl Project, at: [f64; 2], bloom: f64) -> Option<Anchor> {
    let (cx, cy) = (at[0].floor() as i64, at[1].floor() as i64);
    let modules: Vec<(i64, i64)> = (cy - REACH..=cy + REACH)
        .flat_map(|y| (cx - REACH..=cx + REACH).map(move |x| (x, y)))
        .filter(|&(x, y)| x >= 0 && y >= 0 && (x as usize) < func.width && (y as usize) < func.height)
        .filter(|&(x, y)| func.expected(x as usize, y as usize).is_none())
        .collect();
    let patch: Vec<[f64; 2]> = modules.iter().map(|&(x, y)| [x as f64 + 0.5, y as f64 + 0.5]).collect();
    if patch.len() < LEAST_DATA {
        return None;
    }
    let centre = map.apply(at[0], at[1]);
    let step = |dx: f64, dy: f64| {
        let [x, y] = map.apply(at[0] + dx, at[1] + dy);
        [x - centre[0], y - centre[1]]
    };
    let (across, down) = (step(1.0, 0.0), step(0.0, 1.0));
    let shift = |a: f64, b: f64| [a * across[0] + b * down[0], a * across[1] + b * down[1]];
    let points = read_at(map, &patch);
    let read = |a: f64, b: f64| crispness(image, &points, shift(a, b));
    let tenth = MOST_MOVE / MOVES as f64;
    let mut best: Option<(f64, f64, f64)> = None;
    for j in -MOVES..=MOVES {
        for i in -MOVES..=MOVES {
            let (a, b) = (i as f64 * tenth, j as f64 * tenth);
            if let Some(c) = read(a, b)
                && best.is_none_or(|(held, ..)| c > held)
            {
                best = Some((c, a, b));
            }
        }
    }
    let (_, a, b) = best?;
    // A best at the edge of the moves tried may lie beyond them, a
    // module's worth of data over.
    if a.abs() >= MOST_MOVE - 1e-9 || b.abs() >= MOST_MOVE - 1e-9 {
        return None;
    }
    // Every move that keeps the points inside their modules reads as
    // crisply as the next, and the best of them may be any: settled
    // where the boundaries between the modules as they read there are.
    let by = settled(image, map, &modules, &points, [cx - REACH, cy - REACH], shift(a, b), bloom)?;
    let read_by = |s: [f64; 2]| crispness(image, &points, [by[0] + s[0], by[1] + s[1]]);
    let crisp = read_by([0.0, 0.0])?;
    let straddled = [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5)].iter().map(|&(da, db)| read_by(shift(da, db))).collect::<Option<Vec<f64>>>()?;
    if straddled.iter().any(|&off| crisp - off < PEAK * crisp) {
        return None;
    }
    Some(Anchor { kind: AnchorKind::Data, at, center: [centre[0] + by[0], centre[1] + by[1]], partial: false, inferred: false })
}

/// The move of `map`, in pixels, that sets the grid on the boundaries
/// between `modules` (read at `points`, [`read_at`]), from `start`, at
/// which each reads inside its own module: each is read there, dark or
/// light against the split of them all, and the move is the one under
/// which the contrast across every boundary between two of opposite
/// colour is greatest ([`Scorer`]), which falls off at once either way
/// of the boundaries, where how crisply the modules read does not. The
/// boundaries are expected `bloom` modules into what reads dark, as the
/// patterns' are ([`Detection::bloom`]): a photo that has bled moves
/// every one so, and expecting none the patch would be fitted as far
/// off as the bleed. `origin` is the top-left module of the patch.
/// `None` when the boundaries cannot be scored, or are half a module and
/// more from `start`.
fn settled(
    image: &Luma,
    map: &impl Project,
    modules: &[(i64, i64)],
    points: &[[[f64; 2]; 9]],
    origin: [i64; 2],
    start: [f64; 2],
    bloom: f64,
) -> Option<[f64; 2]> {
    let side = 2 * REACH as usize + 1;
    let read = readings(image, points, start);
    let split = otsu(read.iter().flatten().copied());
    let mut expected = vec![None; side * side];
    for (&(x, y), value) in modules.iter().zip(&read) {
        expected[(y - origin[1]) as usize * side + (x - origin[0]) as usize] = value.map(|v| v < split);
    }
    let template = Template { width: side, height: side, expected };
    let (dx, dy) = (origin[0] as f64, origin[1] as f64);
    let s = side as f64;
    let frame = [[0.0, 0.0], [s, 0.0], [s, s], [0.0, s]].map(|[x, y]| {
        let [px, py] = map.apply(x + dx, y + dy);
        [px + start[0], py + start[1]]
    });
    let scorer = Scorer::new(image, &template, frame, None).with_along(6).with_bloom(bloom);
    let module_px = scorer.module_px();
    // Dark is what reads dark, whichever way round the symbol is.
    let score = |trial: [[f64; 2]; 4]| scorer.contrast_with(&Shifted { map: &Moved { map, by: trial[0] }, dx, dy }, 1.0);
    let mut evaluations = 0;
    let (moved, _) = descend_by([start; 4], &[[1.0, 0.0, 0.0, 0.0]], module_px * 0.25, &mut evaluations, score)?;
    let by = moved[0];
    ((by[0] - start[0]).hypot(by[1] - start[1]) < 0.5 * module_px).then_some(by)
}

/// `found` with anchors on patches of its data, as the module describes:
/// the anchors it had, and those the patches far from every one of them
/// place. `None` when no patch places one.
pub fn with_patches(image: &Luma, found: &Detection) -> Option<Detection> {
    let func = FunctionMap::new(found.version);
    let (w, h) = (func.width as f64, func.height as f64);
    let corners = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
    let held: Vec<[f64; 2]> = found.anchors.iter().map(|a| a.at).chain(corners).collect();
    let nearest = |at: [f64; 2], to: &[[f64; 2]]| to.iter().map(|p| (p[0] - at[0]).hypot(p[1] - at[1])).fold(f64::INFINITY, f64::min);
    let step = STEP.max(func.width.max(func.height).div_ceil(ACROSS));
    let mut points: Vec<([f64; 2], f64)> = (0..func.height)
        .step_by(step)
        .flat_map(|y| (0..func.width).step_by(step).map(move |x| [x as f64 + 0.5 + (step / 2) as f64, y as f64 + 0.5 + (step / 2) as f64]))
        .filter(|at| at[0] < w && at[1] < h)
        .map(|at| (at, nearest(at, &held)))
        .filter(|&(_, apart)| apart >= CLEAR)
        .collect();
    points.sort_by(|p, q| p.1.total_cmp(&q.1));
    // The bleed into what reads dark: a light-on-dark symbol's dark
    // modules read light, and its bleed is into those.
    let bloom = if found.inverted { -found.bloom } else { found.bloom };
    let mut anchors = found.anchors.clone();
    let mut placed = 0;
    // A band of points at a time, each band nearer the anchors than the
    // next, read through the map the bands before it bent.
    let mut from = 0;
    while from < points.len() && placed < MOST && !work::cancelled() {
        let edge = points[from].1 + step as f64;
        let to = points[from..].iter().position(|p| p.1 >= edge).map_or(points.len(), |k| from + k);
        let controls: Vec<ControlPoint> = anchors.iter().map(Anchor::control_point).collect();
        let Some(map) = Projection::new(func.width, func.height, found.corners, &controls) else { break };
        let band: Vec<[f64; 2]> = points[from..to].iter().map(|p| p.0).collect();
        let found_here: Vec<Anchor> = work::map(&band, |&at| anchor_at(image, &func, &map, at, bloom)).into_iter().flatten().take(MOST - placed).collect();
        placed += found_here.len();
        anchors.extend(found_here);
        from = to;
    }
    (anchors.len() > found.anchors.len()).then(|| Detection { anchors, ..found.clone() })
}
