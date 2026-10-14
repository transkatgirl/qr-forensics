//! Reading module colours off a photo through a projection.

use super::Luma;
use super::binarize::Binarizer;
use super::enhance::{Enhance, Enhanced};
use super::homography::Homography;
use super::projection::Project;
use crate::symbol::function_map::{FunctionMap, looks_inverted};
use crate::symbol::grid::{CERTAIN, Module, ModuleGrid};
use crate::symbol::{Family, Pos, SymbolVersion};

/// Points sampled per module along each axis, spread over the middle of the
/// module so neighbours and edge blur stay out of the average.
pub const POINTS: usize = 5;
pub const INNER_FRACTION: f64 = 0.6;

/// How a module's luminance is read: averaged over the middle of the
/// module; taken at its centre alone, which is how ZXing and zxing-cpp
/// read the binarised image; or the median of five points about the
/// centre, which is how BoofCV reads the grey one, by a majority of the
/// five. The average is steadiest against noise; the centre alone reads
/// a smeared print or a symbol drawn as dots, where the average takes in
/// the neighbours or the paper between the dots; the vote outvotes a
/// speck or a scratch narrower than half the module, which pulls the
/// average and may fall on the centre.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Kernel {
    Area,
    Centre,
    #[default]
    Vote,
}

impl Kernel {
    pub const ALL: [Kernel; 3] = [Kernel::Area, Kernel::Centre, Kernel::Vote];

    pub fn label(self) -> &'static str {
        match self {
            Kernel::Area => "Area average",
            Kernel::Centre => "Module centre",
            Kernel::Vote => "Centre vote",
        }
    }

    pub fn describe(self) -> &'static str {
        match self {
            Kernel::Area => {
                "The mean over the middle of each module, which averages noise and specks away"
            }
            Kernel::Centre => {
                "The luminance at the centre of each module alone, as ZXing reads it: for smeared prints and \
                 symbols drawn as dots, where the average takes in the neighbours"
            }
            Kernel::Vote => {
                "The median of five points about the centre of each module, as BoofCV reads one: a majority vote, \
                 which a speck or a scratch narrower than half the module cannot swing"
            }
        }
    }

    /// The other kernel, to retry a read with: the vote falls back on
    /// the average, which it refines.
    pub fn other(self) -> Kernel {
        match self {
            Kernel::Area => Kernel::Centre,
            Kernel::Centre => Kernel::Area,
            Kernel::Vote => Kernel::Area,
        }
    }

    /// This kernel, then the others, in the order a read tries them.
    pub fn retries(self) -> impl Iterator<Item = Kernel> {
        std::iter::once(self).chain(Kernel::ALL.into_iter().filter(move |&k| k != self))
    }

    /// The luminance of the module at column `mx`, row `my`, read as this
    /// kernel reads one; `None` where the module lies off the image.
    pub fn sample(self, image: &Luma, map: &impl Project, mx: usize, my: usize) -> Option<f32> {
        match self {
            Kernel::Area => sample_module_inner(image, map, mx, my, INNER_FRACTION, POINTS),
            Kernel::Centre => sample_module_inner(image, map, mx, my, INNER_FRACTION, 1),
            Kernel::Vote => sample_module_vote(image, map, mx, my),
        }
    }
}

/// What a module's luminance is called dark or light against: the
/// darkest and lightest of its own neighbourhood, as ZXing's
/// HybridBinarizer judges a pixel; one level threshold for the whole
/// symbol, as its GlobalHistogramBinarizer does; or one threshold that
/// lies as a surface across the symbol ([`ThresholdSurface`]), as BoofCV
/// reads a QR Code. The neighbourhood follows shadows, glare and stains,
/// and reads the modules of an evenly blurred symbol against neighbours
/// as grey as themselves; the level reads those, and no shadow; the
/// surface follows light that changes slowly across the symbol and no
/// stain. No one of them reads every photo, so a read tries each
/// ([`SampleParams::rules`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rule {
    Neighbourhood,
    Level,
    Surface,
}

impl Rule {
    pub const ALL: [Rule; 3] = [Rule::Neighbourhood, Rule::Level, Rule::Surface];
}

/// Mean luminance of each module; `None` where the module lies off the image.
#[derive(Clone, Debug, PartialEq)]
pub struct Sampled {
    pub width: usize,
    pub height: usize,
    pub luma: Vec<Option<f32>>,
}

/// Mean luminance over the middle `inner` of the module at column `mx`,
/// row `my`, from `points` × `points` samples; `None` when it lies mostly
/// off the image, since that is not evidence of anything. The map's local
/// correction is taken once, at the module's centre.
pub fn sample_module_inner(
    image: &Luma,
    map: &impl Project,
    mx: usize,
    my: usize,
    inner: f64,
    points: usize,
) -> Option<f32> {
    let [dx, dy] = map.displacement(mx as f64 + 0.5, my as f64 + 0.5);
    let (mut sum, mut count) = (0.0f32, 0u32);
    for sy in 0..points {
        for sx in 0..points {
            let offset = |s: usize| (1.0 - inner) / 2.0 + inner * (s as f64 + 0.5) / points as f64;
            let [px, py] = map.base(mx as f64 + offset(sx), my as f64 + offset(sy));
            if let Some(value) = image.sample(px + dx, py + dy) {
                sum += value;
                count += 1;
            }
        }
    }
    (count as usize * 2 > points * points).then(|| sum / count as f32)
}

/// Mean luminance of the module at column `mx`, row `my`, keeping clear of
/// its edges.
pub fn sample_module(image: &Luma, map: &impl Project, mx: usize, my: usize) -> Option<f32> {
    sample_module_inner(image, map, mx, my, INNER_FRACTION, POINTS)
}

/// How far from the centre of a module the vote's four outer points lie,
/// in modules, as BoofCV places them.
pub const VOTE_OFFSET: f64 = 0.2;

/// The median luminance of five points about the centre of the module at
/// column `mx`, row `my`: the centre and [`VOTE_OFFSET`] to each side of
/// it. Against a threshold the median falls as the majority of the five
/// do, which is how BoofCV votes them. `None` when fewer than three lie
/// on the image.
pub fn sample_module_vote(image: &Luma, map: &impl Project, mx: usize, my: usize) -> Option<f32> {
    let (cx, cy) = (mx as f64 + 0.5, my as f64 + 0.5);
    let [dx, dy] = map.displacement(cx, cy);
    let offsets = [
        [0.0, 0.0],
        [-VOTE_OFFSET, 0.0],
        [VOTE_OFFSET, 0.0],
        [0.0, -VOTE_OFFSET],
        [0.0, VOTE_OFFSET],
    ];
    let mut values: Vec<f32> = offsets
        .iter()
        .filter_map(|[ox, oy]| {
            let [px, py] = map.base(cx + ox, cy + oy);
            image.sample(px + dx, py + dy)
        })
        .collect();
    if values.len() < 3 {
        return None;
    }
    values.sort_by(f32::total_cmp);
    let n = values.len();
    Some(if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    })
}

/// Below this many pixels a module, the modules are sharpened against
/// their neighbours before they are read, and the more the smaller.
pub const SHARPEN_BELOW_PX: f64 = 4.0;
/// Most of the difference between a module and the mean of its four
/// neighbours that is added back, for a module vanishingly small.
const SHARPEN_GAIN: f32 = 0.8;

/// Sample a `width` × `height` module grid through `map` with the area
/// kernel. Where a module is only a few pixels across, the lens has
/// blurred each into its neighbours, and a lone dark module among light
/// ones reads as grey: the modules are then sharpened against their
/// neighbours, a little of what the blur took being put back, as an
/// unsharp mask does to a photo.
pub fn sample_modules(image: &Luma, map: &impl Project, width: usize, height: usize) -> Sampled {
    sample_modules_with(image, map, width, height, Kernel::Area)
}

/// [`sample_modules`] with the kernel given. Whatever the module size, a
/// reading whose luminances separate poorly into dark and light, as a
/// photo blurred by the lens or by movement reads, is sharpened too, as
/// [`Sampled::sharpened_for_separation`] does it.
pub fn sample_modules_with(
    image: &Luma,
    map: &impl Project,
    width: usize,
    height: usize,
    kernel: Kernel,
) -> Sampled {
    sample_modules_noting(image, map, width, height, kernel, true).0
}

/// How far a reading's modules were sharpened against their neighbours
/// after they were read ([`sample_modules_noting`]): the share of the
/// difference between each module and the mean of its four neighbours
/// that was added back, under each of the two rules. Nothing under
/// either for a reading left as it was read.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sharpening {
    /// For modules under [`SHARPEN_BELOW_PX`] pixels across.
    pub for_size: f32,
    /// For a reading whose luminances separated poorly.
    pub for_separation: f32,
    /// The size of a module in the photo, in pixels, as the first rule
    /// reckons it: at the middle of the symbol.
    pub module_px: f32,
}

impl Sharpening {
    /// Whether the reading was sharpened at all.
    pub fn any(&self) -> bool {
        self.for_size > 0.0 || self.for_separation > 0.0
    }
}

/// [`sample_modules_with`], sharpened as it sharpens only when `sharpen`
/// says so, and with what the sharpening came to: a reading is otherwise
/// sharpened with nothing to show that it was, and one that reads worse
/// for it, as specks in a clean print read once they are set against
/// their neighbours, cannot be had as it was read.
pub fn sample_modules_noting(
    image: &Luma,
    map: &impl Project,
    width: usize,
    height: usize,
    kernel: Kernel,
    sharpen: bool,
) -> (Sampled, Sharpening) {
    let luma = (0..height)
        .flat_map(|my| (0..width).map(move |mx| (mx, my)))
        .map(|(mx, my)| kernel.sample(image, map, mx, my))
        .collect();
    let sampled = Sampled {
        width,
        height,
        luma,
    };
    let (cx, cy) = (width as f64 / 2.0, height as f64 / 2.0);
    let step = |dx: f64, dy: f64| {
        let [ax, ay] = map.base(cx, cy);
        let [bx, by] = map.base(cx + dx, cy + dy);
        ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt()
    };
    let module_px = (step(1.0, 0.0) + step(0.0, 1.0)) / 2.0;
    let mut done = Sharpening { module_px: module_px as f32, ..Sharpening::default() };
    if !sharpen {
        return (sampled, done);
    }
    let sampled = if module_px < SHARPEN_BELOW_PX {
        done.for_size = SHARPEN_GAIN * (1.0 - module_px / SHARPEN_BELOW_PX) as f32;
        sampled.sharpened(done.for_size)
    } else {
        sampled
    };
    let (sampled, gain) = sampled.sharpened_for_separation_by();
    done.for_separation = gain;
    (sampled, done)
}

/// [`sample_modules_with`] off the photo as `params` ask for it: with
/// their kernel, and, where they ask for the photo enlarged or
/// sharpened ([`SampleParams::enhance`]), off a copy of it so made about
/// the symbol, through the same map. A symbol a read found only through
/// such a copy reads the same again here, and an outline placed by hand
/// round modules too small or too blurred to read as they lie may be
/// read through one. The photo is read as it is when no copy can be
/// made.
pub fn sample_modules_as(image: &Luma, map: &impl Project, width: usize, height: usize, params: &SampleParams) -> Sampled {
    if !params.enhance.is_none() {
        let (w, h) = (width as f64, height as f64);
        let outline = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]].map(|[x, y]| map.apply(x, y));
        if let Some(view) = Enhanced::around(image, outline, params.enhance) {
            return sample_modules_noting(&view.luma, &view.through(map), width, height, params.kernel, params.sharpen_modules).0;
        }
    }
    sample_modules_noting(image, map, width, height, params.kernel, params.sharpen_modules).0
}

/// Below this separability of a reading's luminances, a sharpened reading
/// is tried: the modules of a blurred photo read grey where a dark one
/// has light neighbours, at any size, where the size rule above sharpens
/// only the smallest.
const SHARPEN_FOR_SEPARABILITY: f32 = POOR_SEPARABILITY;
/// How much better a sharpened reading must separate to be kept over the
/// reading as it is.
const SHARPEN_SEPARATION_GAIN: f32 = 0.1;
/// The gains tried for a poorly separated reading, the larger the
/// strength the size rule gives a module vanishingly small.
const SHARPEN_TRIAL_GAINS: [f32; 2] = [0.4, SHARPEN_GAIN];

impl Sampled {
    /// How clearly the luminances read fall into two groups, as
    /// [`separability`] measures it.
    pub fn separability(&self) -> f32 {
        separability(self.luma.iter().flatten().copied())
    }

    /// The reading sharpened against its neighbours when its luminances
    /// separate poorly into dark and light, as a blurred photo reads at
    /// any module size: a gentle and a strong sharpening are tried and
    /// the reading that separates best is kept, when it separates clearly
    /// better than the reading as it is. A reading that already separates
    /// well, or that no sharpening helps, is returned as it is: a grid
    /// that straddles module boundaries reads grey too, and sharpening
    /// does not place it.
    pub fn sharpened_for_separation(self) -> Self {
        self.sharpened_for_separation_by().0
    }

    /// [`Self::sharpened_for_separation`], with the gain of the
    /// sharpening kept: zero for a reading returned as it is.
    pub fn sharpened_for_separation_by(self) -> (Self, f32) {
        let before = self.separability();
        if before >= SHARPEN_FOR_SEPARABILITY {
            return (self, 0.0);
        }
        let mut best: Option<(f32, f32, Sampled)> = None;
        for &gain in &SHARPEN_TRIAL_GAINS {
            let trial = self.sharpened(gain);
            let after = trial.separability();
            if after >= before + SHARPEN_SEPARATION_GAIN
                && best.as_ref().is_none_or(|(b, _, _)| after > *b)
            {
                best = Some((after, gain, trial));
            }
        }
        match best {
            Some((_, gain, trial)) => (trial, gain),
            None => (self, 0.0),
        }
    }

    /// Each module moved away from the mean of its known four-neighbours
    /// by `gain` times their difference.
    pub fn sharpened(&self, gain: f32) -> Self {
        let (w, h) = (self.width, self.height);
        let at = |x: i64, y: i64| {
            (x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h)
                .then(|| self.luma[y as usize * w + x as usize])
                .flatten()
        };
        let luma = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x as i64, y as i64)))
            .map(|(x, y)| {
                let value = at(x, y)?;
                let around: Vec<f32> = [at(x - 1, y), at(x + 1, y), at(x, y - 1), at(x, y + 1)]
                    .into_iter()
                    .flatten()
                    .collect();
                if around.is_empty() {
                    return Some(value);
                }
                let mean = around.iter().sum::<f32>() / around.len() as f32;
                Some(value + gain * (value - mean))
            })
            .collect();
        Self {
            width: w,
            height: h,
            luma,
        }
    }
}

/// Sample only the given modules, over the middle `inner` of each, from
/// `points` × `points` samples.
pub fn sample_positions(
    image: &Luma,
    map: &impl Project,
    positions: &[Pos],
    inner: f64,
    points: usize,
) -> Vec<Option<f32>> {
    positions
        .iter()
        .map(|p| sample_module_inner(image, map, p.x as usize, p.y as usize, inner, points))
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleParams {
    /// Luminance below which a module is dark. `None` picks it automatically.
    /// Ignored in local mode.
    pub threshold: Option<f32>,
    /// Modules whose distance from the threshold is below this fraction of
    /// the local contrast (half the spread between dark and light) become
    /// unknown rather than guessed.
    pub band: f32,
    /// Threshold each module against its own neighbourhood instead of the
    /// whole symbol. Follows shadows, glare and stains; the default, as in
    /// ZXing. With `threshold` and `surface` this says which [`Rule`] a
    /// module is read by ([`SampleParams::rule`]).
    pub local: bool,
    /// Neighbourhood half-width in modules for local mode.
    pub local_radius: usize,
    /// When the whole symbol is read against one threshold, found
    /// automatically, let it lie as a surface across the symbol rather
    /// than level: see [`ThresholdSurface`]. Ignored in local mode, and
    /// when a threshold is given.
    pub surface: bool,
    /// How each module's luminance is read.
    pub kernel: Kernel,
    /// What is done to the photo about the symbol before it is looked
    /// for and read there: enlarged, sharpened, or neither. A read that
    /// does not decode tries the others of itself
    /// ([`super::read::read_outline`]).
    pub enhance: Enhance,
    /// Sharpen the modules against their neighbours once they are read,
    /// where they are small or their luminances separate poorly
    /// ([`sample_modules_noting`]). On, unless it is turned off by hand.
    pub sharpen_modules: bool,
    /// How the photo is thresholded when the symbol is looked for in it
    /// ([`super::detect`]): the finder scan's three readings pooled, or
    /// one of them alone, as the binarised view shows one, set for the
    /// scan in its place. The reaches are in the photo's pixels, and a
    /// copy made for a look takes them scaled with it.
    pub binarizer: Binarizer,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self {
            threshold: None,
            band: 0.12,
            local: true,
            local_radius: 5,
            surface: false,
            kernel: Kernel::Vote,
            enhance: Enhance::NONE,
            sharpen_modules: true,
            binarizer: Binarizer::Pooled,
        }
    }
}

impl SampleParams {
    /// The rule these parameters read a module by: the neighbourhood's
    /// when they are local, and otherwise the surface when it is asked
    /// for and the threshold is found rather than given, or the level.
    pub fn rule(&self) -> Rule {
        if self.local {
            Rule::Neighbourhood
        } else if self.surface && self.threshold.is_none() {
            Rule::Surface
        } else {
            Rule::Level
        }
    }

    /// These parameters with the module read by `rule` instead. The
    /// surface is found, so a threshold given is let go for it; under
    /// the neighbourhood rule neither is looked at, and both stay.
    pub fn under(self, rule: Rule) -> Self {
        match rule {
            Rule::Neighbourhood => Self { local: true, ..self },
            Rule::Level => Self { local: false, surface: false, ..self },
            Rule::Surface => Self { local: false, surface: true, threshold: None, ..self },
        }
    }

    /// The rules a read tries, in order: the one asked for, then the
    /// others, the neighbourhood before the level before the surface. A
    /// threshold given is the level the symbol is to be read against,
    /// and no surface is found beside it.
    pub fn rules(&self) -> impl Iterator<Item = Rule> {
        let asked = self.rule();
        let given = self.threshold.is_some();
        std::iter::once(asked).chain(Rule::ALL.into_iter().filter(move |&rule| rule != asked && !(given && rule == Rule::Surface)))
    }
}

/// Least spread between the darkest and lightest modules of a
/// neighbourhood for it to hold any information, as a share of the
/// symbol's own contrast, the gap between the means of its dark and its
/// light modules. Below it the modules all look alike, as under an opaque
/// stain or in glare, and calling them dark or light would be guessing.
/// ZXing's HybridBinarizer draws the line at 24 of 255 whatever the
/// photo, and so loses an underexposed photo whole, whose symbol spans
/// less than that from ink to paper; BoofCV's block Otsu draws none at a
/// fixed luminance, and lowers a block's threshold instead in proportion
/// to its own square over the variance between the block's dark and its
/// light, so that what counts as flat goes with the exposure. A share of
/// the symbol's contrast falls where ZXing's line does on a well exposed
/// photo and far lower on a dark one.
pub const FLAT_SHARE: f32 = 0.15;
/// The spread below which a neighbourhood is flat whatever the symbol's
/// contrast: the noise of the sensor alone spreads a flat patch by about
/// this much.
pub const NOISE_RANGE: f32 = 0.02;
/// Fewest readable modules in a neighbourhood for local statistics.
const MIN_LOCAL_COUNT: usize = 6;

#[derive(Clone, Debug, PartialEq)]
pub struct Classified {
    pub modules: Vec<Module>,
    /// Each module as it falls about its threshold with no unsure band
    /// left around that: what `modules` is at a band of nothing, the
    /// colour of every module that was read at all, however near its
    /// threshold. Unknown only where a module could not be sampled or
    /// its neighbourhood had no contrast.
    pub decided: Vec<Module>,
    /// How far each module is from its threshold, as a share of half the
    /// local contrast: nought on the threshold, one where a module of
    /// full contrast lies. Nought where the module could not be sampled
    /// or its neighbourhood had no contrast.
    pub margin: Vec<f32>,
    /// How sure each module's reading is, on the scale the decoder works
    /// in: eight times the bits of likelihood the reading carries, less
    /// eight, so that a module on its threshold is nought and one read
    /// beyond doubt 254 ([`Calibration::reliability`]). From the margin
    /// alone under [`Calibration::DEFAULT`], until the read is
    /// [calibrated](Self::calibrate) against the modules whose colour
    /// was known. Nought where the module could not be sampled or its
    /// neighbourhood had no contrast.
    pub reliability: Vec<u8>,
    /// The misread model the reliabilities were calibrated with, once
    /// they were.
    pub calibration: Option<Calibration>,
    /// The global threshold, automatic or as given.
    pub threshold: f32,
    /// How clearly the module luminances fall into two groups: how empty
    /// the valley between the two peaks of their histogram is, 0 to 1. A
    /// well placed outline on a readable symbol scores high; an outline off
    /// the symbol, which averages neighbouring modules into greys, or a
    /// washed-out photo, scores low. ZXing's global binarizer refuses an
    /// image whose two peaks are not apart for the same reason.
    pub separability: f32,
    /// Modules left unknown because nothing around them had contrast.
    pub flat: usize,
}

impl Classified {
    pub fn to_grid(&self, width: usize, height: usize) -> ModuleGrid {
        let mut grid = ModuleGrid::new(width, height);
        for (i, (&module, &reliability)) in self.modules.iter().zip(&self.reliability).enumerate() {
            grid.set_sampled(i % width, i / width, module, reliability);
        }
        grid
    }

    /// The grid with every module at the colour it fell on, however
    /// near its threshold, at the reliability it was read at: what the
    /// unsure band would leave unknown goes in as a guess instead, which
    /// the decoder may doubt as it doubts a module painted as one. A
    /// codeword with a module left unknown is an erasure whole, and
    /// spends a unit of redundancy however faint the one module was; the
    /// same module kept at its colour spends nothing until the block
    /// fails, and then an equation to doubt. Over blocks of simulated
    /// noise the band never read more than this, and with a stain across
    /// two codewords beside the noise it read far less: a 5-Q block at
    /// six wrong bits in a hundred read 47 times in a hundred through
    /// the band and 71 kept.
    pub fn to_grid_as_decided(&self, width: usize, height: usize) -> ModuleGrid {
        let mut grid = ModuleGrid::new(width, height);
        for (i, (&module, &reliability)) in self.decided.iter().zip(&self.reliability).enumerate() {
            grid.set_sampled(i % width, i / width, module, reliability);
        }
        grid
    }

    /// Modules the unsure band left unknown that were read at all.
    pub fn unsure(&self) -> usize {
        self.modules.iter().zip(&self.decided).filter(|&(&m, &d)| m == Module::Unknown && d != Module::Unknown).count()
    }

    /// The luminances do not separate into two groups; the outline is
    /// probably off the symbol, or the photo too washed out to read.
    pub fn is_poorly_separated(&self) -> bool {
        self.separability < POOR_SEPARABILITY
    }

    /// Calibrate the reliabilities against the modules whose colour is
    /// known: `expected` is the colour of each module of the symbol,
    /// row-major, where it is known, as the fixed patterns' are before
    /// any decoding ([`crate::symbol::function_map::FunctionMap`]) and
    /// every module's of a block that verified is after it
    /// ([`crate::decode::repair::known_modules`]); `inverted` says the
    /// photo shows the symbol light on dark, so that a module read dark
    /// is right where the symbol is light.
    ///
    /// Each labelled module gives a signed margin, its margin toward
    /// the colour it should have: a module read right lies about
    /// [`Calibration::typical`] of half the contrast from its threshold,
    /// one read wrong on the other side. The spread about the typical
    /// margin, taken robustly so that a torn finder pattern does not set
    /// it, is the photo's noise ([`Calibration::sigma`]), from which the
    /// odds of a misread at any margin follow; and the modules read
    /// wrong by more than that noise explains are the gross misreads
    /// ([`Calibration::gross`]), specks and stains and tears, which no
    /// margin protects against and which cap what any module's reading
    /// may be believed. Every sampled module's reliability is then
    /// recomputed from its margin under the model. Nothing is changed,
    /// and `None` returned, with fewer than [`CALIBRATION_MIN_LABELLED`]
    /// labelled modules, or when they read so poorly that the labels
    /// are not to be trusted: the placement is off the symbol.
    pub fn calibrate(&mut self, expected: &[Option<bool>], inverted: bool) -> Option<Calibration> {
        let mut signed: Vec<f32> = expected
            .iter()
            .zip(&self.decided)
            .zip(&self.margin)
            .filter_map(|((want, got), &margin)| {
                let (want, got) = ((*want)?, got.as_dark()?);
                let right = (got != inverted) == want;
                Some(if right { margin } else { -margin })
            })
            .collect();
        if signed.len() < CALIBRATION_MIN_LABELLED {
            return None;
        }
        let typical = median(&mut signed);
        if typical < CALIBRATION_MIN_TYPICAL {
            return None;
        }
        let mut deviations: Vec<f32> = signed.iter().map(|s| (s - typical).abs()).collect();
        let sigma = (MAD_TO_SIGMA * median(&mut deviations)).clamp(CALIBRATION_MIN_SIGMA, CALIBRATION_MAX_SIGMA);
        let misread = signed.iter().filter(|&&s| s < 0.0).count();
        let gross = signed.iter().filter(|&&s| s < -2.0 * sigma).count() as f32 / signed.len() as f32;
        let calibration = Calibration { typical, sigma, gross, labelled: signed.len(), misread };
        for ((reliability, &margin), decided) in self.reliability.iter_mut().zip(&self.margin).zip(&self.decided) {
            if *decided != Module::Unknown {
                *reliability = calibration.reliability(margin);
            }
        }
        self.calibration = Some(calibration);
        Some(calibration)
    }
}

/// The middle value of `values`, which are sorted to find it.
fn median(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    let n = values.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// Fewest labelled modules a calibration is made on: the finder patterns
/// alone of a Micro QR symbol are forty-nine.
pub const CALIBRATION_MIN_LABELLED: usize = 32;
/// The least a typical labelled module may lie from its threshold, as a
/// share of half the contrast, for the labels to be believed: below it
/// the fixed patterns are not where the placement puts them.
pub const CALIBRATION_MIN_TYPICAL: f32 = 0.1;
/// Bounds on the noise fitted, as a share of half the contrast: a photo
/// is never cleaner than the first, which keeps a margin from being
/// believed beyond all reason, nor noisier than the second, which is a
/// photo of nothing.
pub const CALIBRATION_MIN_SIGMA: f32 = 0.05;
pub const CALIBRATION_MAX_SIGMA: f32 = 1.5;
/// The standard deviation of a normal distribution from the median of
/// absolute deviations about its median.
const MAD_TO_SIGMA: f32 = 1.4826;

/// How a module's reading is to be believed, as a function of how far
/// from its threshold it was read: the misread model
/// [`Classified::calibrate`] fits to a photo, or the default for one not
/// yet fitted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Calibration {
    /// Where a module read right lies about its threshold, as a share of
    /// half the local contrast: one on a sharp photo of a clean print,
    /// less as blur draws dark and light together.
    pub typical: f32,
    /// How the readings spread about that, as a share of half the
    /// contrast: the photo's noise.
    pub sigma: f32,
    /// The share of modules read wrong by more than the noise explains:
    /// specks, stains and tears, against which no margin is proof. What
    /// caps the belief in any reading.
    pub gross: f32,
    /// Modules whose colour was known that the fit was made on, and how
    /// many of them read wrong.
    pub labelled: usize,
    pub misread: usize,
}

impl Calibration {
    /// The model for a photo not yet fitted: the noise a photo of
    /// middling quality shows, under which the reliability runs near
    /// enough with the margin as the linear scale it replaced did, two
    /// hundred and fifty-four at a module of full contrast and sixty-one
    /// at a quarter of it.
    pub const DEFAULT: Self = Self { typical: 1.0, sigma: 0.29, gross: 0.0, labelled: 0, misread: 0 };

    /// The bits of likelihood that a module read `margin` from its
    /// threshold was read right: minus the log of the odds it was
    /// misread. Under the model a module's luminance lies about its
    /// colour's mean, `typical` half-contrasts from the threshold, with
    /// a spread of `sigma`, so the log odds of the two colours at a
    /// reading `margin` to one side are `2 · margin · typical / sigma²`;
    /// and besides that a share `gross` of modules are wrong whatever
    /// their margin.
    pub fn bits(&self, margin: f32) -> f32 {
        let ratio = 2.0 * margin.max(0.0) * self.typical / (self.sigma * self.sigma * std::f32::consts::LN_2);
        let wrong = self.gross + (1.0 - self.gross) / (1.0 + 2f32.powf(ratio.min(126.0)));
        -wrong.max(f32::MIN_POSITIVE).log2()
    }

    /// Reliability for a module read `margin` from its threshold: eight
    /// times the bits of likelihood less eight, so that a module on its
    /// threshold, read right as often as not, is nought, and a module
    /// whose reading is beyond doubt reaches the cap of 254, one short
    /// of [`CERTAIN`], which is for modules painted by hand.
    pub fn reliability(&self, margin: f32) -> u8 {
        ((self.bits(margin) - 1.0) * 8.0).round().clamp(0.0, (CERTAIN - 1) as f32) as u8
    }
}

/// The bits of likelihood a sampled module's reliability stands for, the
/// scale [`Calibration::reliability`] puts it on read back: one bit at
/// nought, a coin's toss, and a bit more for every eight. A reading
/// believed to `b` bits is misread about once in `2^b` such readings
/// ([`misread_odds`]).
pub fn reliability_bits(reliability: u8) -> f32 {
    f32::from(reliability.min(CERTAIN - 1)) / 8.0 + 1.0
}

/// How many readings at this reliability are misread, as one in so many:
/// two to the bits of likelihood.
pub fn misread_odds(reliability: u8) -> f64 {
    2f64.powf(f64::from(reliability_bits(reliability)))
}

/// Below this the two groups are not believable: the valley between them
/// holds half as many modules as the smaller peak.
pub const POOR_SEPARABILITY: f32 = 0.5;

/// How clearly the values split in two: one when nothing lies in the
/// middle of the gap between the dark and light groups, zero when as many
/// values lie there as make up the smaller group. Module means from a
/// well placed grid cluster at two levels; a grid that straddles module
/// boundaries averages dark with light and fills the middle.
pub fn separability(values: impl Iterator<Item = f32>) -> f32 {
    let values: Vec<f32> = values.collect();
    if values.len() < 4 {
        return 0.0;
    }
    let (threshold, means, _) = otsu_split(values.iter().copied());
    let band = 0.35 * (means[1] - means[0]).max(1e-3);
    let low = values.iter().filter(|&&v| v < threshold).count();
    let high = values.len() - low;
    let middle = values
        .iter()
        .filter(|&&v| (v - threshold).abs() < band)
        .count();
    let smaller = low.min(high).max(1);
    (1.0 - middle as f32 / smaller as f32).clamp(0.0, 1.0)
}

/// Reliability for a relative margin from the threshold, before any
/// calibration.
fn quantize(margin: f32) -> u8 {
    Calibration::DEFAULT.reliability(margin)
}

/// Otsu's split of `values`, with how well it separates them: the
/// threshold, the means of the two groups, and the share of variance that
/// lies between the groups. Returns a half-grey split with zero
/// separability when there is nothing to separate.
pub fn otsu_split(values: impl Iterator<Item = f32>) -> (f32, [f32; 2], f32) {
    const BINS: usize = 256;
    let mut histogram = [0u32; BINS];
    let mut total = 0u32;
    for value in values {
        histogram[((value.clamp(0.0, 1.0) * (BINS - 1) as f32).round()) as usize] += 1;
        total += 1;
    }
    if total == 0 {
        return (0.5, [0.25, 0.75], 0.0);
    }
    let sum_all: f64 = histogram
        .iter()
        .enumerate()
        .map(|(i, &n)| i as f64 * n as f64)
        .sum();
    let sum_sq: f64 = histogram
        .iter()
        .enumerate()
        .map(|(i, &n)| (i * i) as f64 * n as f64)
        .sum();
    let mean_all = sum_all / total as f64;
    let variance_all = sum_sq / total as f64 - mean_all * mean_all;
    let (mut weight_low, mut sum_low) = (0.0f64, 0.0f64);
    let (mut best, mut best_variance, mut best_means) = (None, 0.0f64, [0.0f64, 0.0f64]);
    // Several adjacent splits tie when the two groups are well apart; take
    // the middle of the tied range so the threshold sits between the groups.
    let mut tie_end = 0;
    for (i, &n) in histogram.iter().enumerate() {
        weight_low += n as f64;
        sum_low += i as f64 * n as f64;
        let weight_high = total as f64 - weight_low;
        if weight_low == 0.0 || weight_high == 0.0 {
            continue;
        }
        let (mean_low, mean_high) = (sum_low / weight_low, (sum_all - sum_low) / weight_high);
        let mean_gap = mean_low - mean_high;
        let variance = weight_low * weight_high * mean_gap * mean_gap;
        if variance > best_variance * (1.0 + 1e-12) {
            best_variance = variance;
            best = Some(i);
            best_means = [mean_low, mean_high];
            tie_end = i;
        } else if best.is_some() && variance >= best_variance * (1.0 - 1e-12) {
            tie_end = i;
        }
    }
    match best {
        Some(start) => {
            let threshold = ((start + tie_end) as f32 / 2.0 + 0.5) / (BINS - 1) as f32;
            let means = best_means.map(|m| (m / (BINS - 1) as f64) as f32);
            // Between-group variance as Otsu defines it, over the total.
            let between = best_variance / (total as f64 * total as f64);
            let separability = if variance_all > 0.0 {
                (between / variance_all).clamp(0.0, 1.0) as f32
            } else {
                0.0
            };
            (threshold, means, separability)
        }
        None => (0.5, [0.25, 0.75], 0.0),
    }
}

/// Otsu's threshold: the split that best separates the values into two
/// groups. Returns 0.5 when there is nothing to separate.
pub fn otsu(values: impl Iterator<Item = f32>) -> f32 {
    otsu_split(values).0
}

/// How far `values` reach from their first to their ninety-ninth
/// percentile, by a histogram of 256 bins: the spread of a photo from ink
/// to paper, with the few pixels beyond either left out. Zero for no
/// values. What a contrast floor is a share of, so that an underexposed
/// photo, which spans a tenth of the scale, is held to a tenth of the
/// floor a well exposed one is.
pub fn robust_spread(values: impl Iterator<Item = f32>) -> f32 {
    const BINS: usize = 256;
    let mut histogram = [0u32; BINS];
    let mut total = 0u32;
    for value in values {
        histogram[((value.clamp(0.0, 1.0) * (BINS - 1) as f32).round()) as usize] += 1;
        total += 1;
    }
    if total == 0 {
        return 0.0;
    }
    let percentile = |share: f64| {
        let want = (share * f64::from(total)).ceil().max(1.0) as u32;
        let mut seen = 0u32;
        for (i, &n) in histogram.iter().enumerate() {
            seen += n;
            if seen >= want {
                return i as f32 / (BINS - 1) as f32;
            }
        }
        1.0
    };
    (percentile(0.99) - percentile(0.01)).max(0.0)
}

/// Means of the values on either side of `threshold`; the group means of
/// the split when a threshold was given rather than found.
fn means_about(values: impl Iterator<Item = f32>, threshold: f32) -> [f32; 2] {
    let (mut sum, mut count) = ([0.0f64; 2], [0u32; 2]);
    for value in values {
        let side = usize::from(value >= threshold);
        sum[side] += value as f64;
        count[side] += 1;
    }
    let mean = |side: usize, fallback: f32| {
        if count[side] > 0 {
            (sum[side] / count[side] as f64) as f32
        } else {
            fallback
        }
    };
    [mean(0, threshold - 0.25), mean(1, threshold + 0.25)]
}

/// Low and high ends of a neighbourhood's luminance: the tenth and
/// ninetieth percentiles, so that a stray module does not set the range.
fn robust_range(values: &mut [f32]) -> (f32, f32) {
    let n = values.len();
    let lo = n / 10;
    let hi = n - 1 - n / 10;
    let (_, low, _) = values.select_nth_unstable_by(lo, f32::total_cmp);
    let low = *low;
    let (_, high, _) = values.select_nth_unstable_by(hi, f32::total_cmp);
    (low, *high)
}

/// Modules on a side of the window at each corner of the symbol that a
/// [`ThresholdSurface`] takes its corner threshold from: a finder pattern
/// with its separator and the format bits beside it, which hold dark and
/// light in fair measure whatever the data.
const SURFACE_WINDOW: usize = 9;

/// A threshold that lies as a smooth surface across the symbol: the
/// thresholds of its four corners, interpolated bilinearly between them,
/// as BoofCV reads a QR Code (`bitIntensityToBitValue`), which takes
/// three of the four from its position patterns, each the midpoint of
/// the luminance just inside and just outside the pattern's outline, and
/// the fourth from the data: the mean of every data module within ten
/// modules of the right or the bottom edge, on the footing that as many
/// of them are dark as light. A shadow that falls across the
/// symbol, or light that fails toward one side, moves every module's
/// luminance by an amount that changes slowly from corner to corner, and
/// the surface follows it. One level threshold cannot; and the
/// neighbourhood rule, which can, follows every stain and smudge as
/// well, and reads the modules of an evenly blurred symbol against
/// neighbours as grey as themselves.
///
/// Each corner's threshold is the midpoint between the darkest and
/// lightest modules of a window [`SURFACE_WINDOW`] modules on a side in
/// that corner, which asks nothing of how many are dark, and stands at
/// the middle of its window; a corner with too few modules read, or none
/// that differ, takes the level threshold of the whole symbol. The
/// contrast the unsure band is a share of is interpolated with it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThresholdSurface {
    /// Where the corner windows stand, in modules: left and right
    /// columns, top and bottom rows.
    columns: [f32; 2],
    rows: [f32; 2],
    /// (threshold, half the contrast) at the top-left, top-right,
    /// bottom-left and bottom-right.
    corners: [(f32, f32); 4],
}

impl ThresholdSurface {
    /// The surface over `sampled`, with `level` the threshold and
    /// contrast of the symbol as a whole, for a corner that has none of
    /// its own, and `flat_range` the least spread of luminance that is
    /// any contrast at all.
    pub fn new(sampled: &Sampled, level: (f32, f32), flat_range: f32) -> Self {
        let (w, h) = (sampled.width, sampled.height);
        let (ww, wh) = (SURFACE_WINDOW.min(w), SURFACE_WINDOW.min(h));
        let corner = |x0: usize, y0: usize| {
            let mut values: Vec<f32> = (y0..y0 + wh)
                .flat_map(|y| (x0..x0 + ww).map(move |x| (x, y)))
                .filter_map(|(x, y)| sampled.luma[y * w + x])
                .collect();
            if values.len() < MIN_LOCAL_COUNT {
                return level;
            }
            let (low, high) = robust_range(&mut values);
            if high - low < flat_range { level } else { ((low + high) / 2.0, (high - low) / 2.0) }
        };
        let (x1, y1) = (w - ww, h - wh);
        Self {
            columns: [(ww as f32 - 1.0) / 2.0, x1 as f32 + (ww as f32 - 1.0) / 2.0],
            rows: [(wh as f32 - 1.0) / 2.0, y1 as f32 + (wh as f32 - 1.0) / 2.0],
            corners: [corner(0, 0), corner(x1, 0), corner(0, y1), corner(x1, y1)],
        }
    }

    /// The threshold, and half the contrast, at the module in column `x`
    /// and row `y`: interpolated between the corners, and carried on at
    /// the same slope beyond the middles of their windows.
    pub fn at(&self, x: usize, y: usize) -> (f32, f32) {
        let share = |v: usize, [a, b]: [f32; 2]| if b > a { (v as f32 - a) / (b - a) } else { 0.5 };
        let (u, v) = (share(x, self.columns), share(y, self.rows));
        let mix = |pick: fn(&(f32, f32)) -> f32| {
            let [tl, tr, bl, br] = self.corners.each_ref().map(pick);
            (1.0 - v) * ((1.0 - u) * tl + u * tr) + v * ((1.0 - u) * bl + u * br)
        };
        // Carried beyond its corners the contrast may run out; a tenth of
        // the least of them is as low as it is let fall.
        let least = self.corners.iter().map(|c| c.1).fold(f32::INFINITY, f32::min);
        (mix(|c| c.0), mix(|c| c.1).max(0.1 * least).max(1e-3))
    }
}

pub fn classify(sampled: &Sampled, params: &SampleParams) -> Classified {
    let (auto_threshold, _, _) = otsu_split(sampled.luma.iter().flatten().copied());
    let separability = separability(sampled.luma.iter().flatten().copied());
    let global = params.threshold.unwrap_or(auto_threshold);
    let global_means = means_about(sampled.luma.iter().flatten().copied(), global);
    let global_half = ((global_means[1] - global_means[0]) / 2.0).max(1e-3);
    let flat_range = (FLAT_SHARE * (global_means[1] - global_means[0])).max(NOISE_RANGE);
    let (w, h) = (sampled.width, sampled.height);
    let surface = (params.surface && !params.local && params.threshold.is_none() && w > 0 && h > 0)
        .then(|| ThresholdSurface::new(sampled, (global, global_half), flat_range));

    let mut modules = Vec::with_capacity(w * h);
    let mut decided = Vec::with_capacity(w * h);
    let mut margins = Vec::with_capacity(w * h);
    let mut reliability = Vec::with_capacity(w * h);
    let mut flat = 0;
    let mut neighbourhood: Vec<f32> = Vec::new();
    for i in 0..w * h {
        let Some(value) = sampled.luma[i] else {
            modules.push(Module::Unknown);
            decided.push(Module::Unknown);
            margins.push(0.0);
            reliability.push(0);
            continue;
        };
        let (threshold, half) = if params.local {
            let (x, y) = (i % w, i / w);
            let r = params.local_radius;
            neighbourhood.clear();
            for ny in y.saturating_sub(r)..(y + r + 1).min(h) {
                for nx in x.saturating_sub(r)..(x + r + 1).min(w) {
                    if let Some(v) = sampled.luma[ny * w + nx] {
                        neighbourhood.push(v);
                    }
                }
            }
            if neighbourhood.len() < MIN_LOCAL_COUNT {
                (global, global_half)
            } else {
                let (low, high) = robust_range(&mut neighbourhood);
                if high - low < flat_range {
                    // Nothing around here has contrast: no evidence either way.
                    modules.push(Module::Unknown);
                    decided.push(Module::Unknown);
                    margins.push(0.0);
                    reliability.push(0);
                    flat += 1;
                    continue;
                }
                // The midpoint rather than the mean, so a neighbourhood with
                // more of one colour does not pull the threshold towards it.
                ((low + high) / 2.0, (high - low) / 2.0)
            }
        } else if let Some(surface) = &surface {
            surface.at(i % w, i / w)
        } else {
            (global, global_half)
        };
        let margin = (value - threshold).abs() / half;
        let side = Module::from_dark(value < threshold);
        modules.push(if margin < params.band { Module::Unknown } else { side });
        decided.push(side);
        margins.push(margin);
        reliability.push(quantize(margin));
    }
    Classified {
        modules,
        decided,
        margin: margins,
        reliability,
        calibration: None,
        threshold: global,
        separability,
        flat,
    }
}

/// How well a photo fits one version's fixed patterns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VersionGuess {
    pub version: SymbolVersion,
    /// Fixed modules that came out the right colour, or, when `inverted`,
    /// the right colour with dark and light swapped.
    pub matching: usize,
    /// Fixed modules in the version.
    pub fixed: usize,
    /// The fixed patterns read inverted at this version: the photo shows
    /// a light symbol on a dark background.
    pub inverted: bool,
}

impl VersionGuess {
    pub fn score(&self) -> f32 {
        self.matching as f32 / self.fixed.max(1) as f32
    }
}

/// Try every version of a family on the same four corners and rank them by
/// how many fixed-pattern modules land on the right colour. A version whose
/// fixed patterns read decidedly inverted is scored with dark and light
/// swapped and marked as such, so a light-on-dark symbol still ranks its
/// true version first.
pub fn suggest_versions(
    image: &Luma,
    corners: [[f64; 2]; 4],
    family: Family,
    params: &SampleParams,
) -> Vec<VersionGuess> {
    let mut guesses: Vec<VersionGuess> = family
        .versions()
        .into_iter()
        .filter_map(|version| {
            let func = FunctionMap::new(version);
            let h = Homography::rect_to_quad(func.width as f64, func.height as f64, corners)?;
            let sampled = sample_modules(image, &h, func.width, func.height);
            let grid = classify(&sampled, params).to_grid(func.width, func.height);
            let (matching, known) = func.fixed_agreement(&grid);
            let inverted = looks_inverted((matching, known));
            let matching = if inverted { known - matching } else { matching };
            let fixed = (0..func.height)
                .flat_map(|y| (0..func.width).map(move |x| (x, y)))
                .filter(|&(x, y)| func.expected(x, y).is_some())
                .count();
            Some(VersionGuess {
                version,
                matching,
                fixed,
                inverted,
            })
        })
        .collect();
    guesses.sort_by(|a, b| b.score().total_cmp(&a.score()));
    guesses
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn otsu_splits_between_groups() {
        let values = [0.1f32, 0.12, 0.09, 0.11, 0.8, 0.82, 0.79, 0.85, 0.81];
        let (t, means, eta) = otsu_split(values.iter().copied());
        assert!((0.3..0.6).contains(&t), "{t}");
        assert!(
            (means[0] - 0.105).abs() < 0.01 && (means[1] - 0.814).abs() < 0.01,
            "{means:?}"
        );
        assert!(eta > 0.95, "{eta}");
        assert_eq!(otsu(std::iter::empty()), 0.5);
        assert_eq!(otsu([0.4f32; 5].into_iter()), 0.5);
        // Two clean groups leave the valley between them empty; one blob
        // of noise has no valley at all, and neither has a grid that
        // straddles module boundaries and averages dark with light.
        assert!(separability(values.iter().copied()) > 0.99);
        let noise: Vec<f32> = (0..200)
            .map(|i| 0.4 + 0.2 * ((i * 7919) % 101) as f32 / 101.0)
            .collect();
        assert!(
            separability(noise.iter().copied()) < POOR_SEPARABILITY,
            "{}",
            separability(noise.iter().copied())
        );
        let straddled: Vec<f32> = (0..350)
            .map(|i| [0.1, 0.23, 0.37, 0.5, 0.63, 0.77, 0.9][(i * 3) % 7])
            .collect();
        assert!(
            separability(straddled.iter().copied()) < POOR_SEPARABILITY,
            "{}",
            separability(straddled.iter().copied())
        );
        assert_eq!(separability(std::iter::empty()), 0.0);
    }

    #[test]
    fn band_marks_uncertain_modules() {
        let sampled = Sampled {
            width: 5,
            height: 1,
            luma: vec![Some(0.1), Some(0.47), Some(0.9), None, Some(0.55)],
        };
        let global = SampleParams {
            threshold: Some(0.5),
            band: 0.2,
            local: false,
            ..Default::default()
        };
        let out = classify(&sampled, &global);
        assert_eq!(
            out.modules,
            [
                Module::Dark,
                Module::Unknown,
                Module::Light,
                Module::Unknown,
                Module::Light
            ]
        );
        // With no band about the threshold, the module inside it is the
        // colour it falls on, as it is at a band of nothing; the one that
        // could not be sampled stays unknown.
        assert_eq!(out.decided, [Module::Dark, Module::Dark, Module::Light, Module::Unknown, Module::Light]);
        assert_eq!(out.decided, classify(&sampled, &SampleParams { band: 0.0, ..global }).modules);
        // Groups mean 0.285 and 0.725: a margin of 0.03 is about a seventh
        // of the half-contrast of 0.22, a margin of 0.05 about a quarter.
        assert_eq!(out.reliability[0], 254);
        assert!(
            (30..40).contains(&out.reliability[1]),
            "{:?}",
            out.reliability
        );
        assert_eq!(out.reliability[2], 254);
        assert_eq!(out.reliability[3], 0);
        assert!(
            (52..64).contains(&out.reliability[4]),
            "{:?}",
            out.reliability
        );
        let grid = out.to_grid(5, 1);
        assert!(grid.has_reliability());
        assert_eq!(grid.reliability(0, 0), 254);
        assert_eq!(grid.reliability(3, 0), 0);
        let strict = classify(
            &sampled,
            &SampleParams {
                band: 0.0,
                ..global
            },
        );
        assert_eq!(strict.modules[1], Module::Dark);
        assert_eq!(strict.modules[4], Module::Light);
    }

    #[test]
    fn local_threshold_follows_lighting_and_refuses_flat_regions() {
        // A 9×9 symbol-like grid, checkerboard, lit twice as bright on the
        // right; a 3×3 patch in the middle washed out to one grey.
        let (w, h) = (9, 9);
        let mut luma = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let dark = (x + y) % 2 == 0;
                let light = 0.4 + 0.6 * x as f32 / (w - 1) as f32;
                let value = if dark { 0.1 * light } else { 0.9 * light };
                luma.push(Some(if (3..6).contains(&x) && (3..6).contains(&y) {
                    0.5
                } else {
                    value
                }));
            }
        }
        let sampled = Sampled {
            width: w,
            height: h,
            luma,
        };
        let params = SampleParams {
            local: true,
            local_radius: 1,
            ..Default::default()
        };
        let out = classify(&sampled, &params);
        for y in 0..h {
            for x in 0..w {
                let want = if (3..6).contains(&x) && (3..6).contains(&y) {
                    // Inside the wash-out the modules all read the same
                    // grey, which is no evidence; the ring around it still
                    // has neighbours with contrast and reads right.
                    None
                } else {
                    Some((x + y) % 2 == 0)
                };
                let got = out.modules[y * w + x];
                match want {
                    None => assert!(matches!(
                        got,
                        Module::Unknown | Module::Dark | Module::Light
                    )),
                    Some(dark) => assert_eq!(got, Module::from_dark(dark), "({x},{y})"),
                }
            }
        }
        // The washed-out centre itself is unknown and its margin zero, as a
        // flat neighbourhood. The exact count depends on how far the flat
        // patch dominates each neighbourhood; the centre module always is.
        assert_eq!(out.modules[4 * w + 4], Module::Unknown);
        assert_eq!(out.reliability[4 * w + 4], 0);
        assert!(out.flat >= 1, "{}", out.flat);
        // Reliability is relative to the local contrast, so a module in the
        // dim left half is about as sure as one in the bright right half.
        let (left, right) = (out.reliability[w + 1], out.reliability[w + 7]);
        assert!(left > 230 && right > 230, "{left} {right}");
        assert!(left.abs_diff(right) < 12, "{left} {right}");
    }

    #[test]
    fn a_threshold_surface_follows_a_gradient_across_the_symbol() {
        // A 25×25 grid of modules, checkered, in grey ink, under light
        // that falls off to a seventh from the left edge to the right:
        // paper at the dark end reads darker than ink at the light end,
        // and no one threshold tells them apart.
        let (w, h) = (25, 25);
        let dark = |x: usize, y: usize| (x + y).is_multiple_of(2);
        let light = |x: usize, y: usize| 1.0 - 0.85 * (4 * x + y) as f32 / (4 * (w - 1) + h - 1) as f32;
        let luma: Vec<Option<f32>> = (0..w * h)
            .map(|i| (i % w, i / w))
            .map(|(x, y)| Some(light(x, y) * if dark(x, y) { 0.35 } else { 0.9 }))
            .collect();
        let sampled = Sampled {
            width: w,
            height: h,
            luma,
        };
        let wrong = |params: &SampleParams| {
            let out = classify(&sampled, params);
            (0..w * h)
                .filter(|&i| out.modules[i] != Module::from_dark(dark(i % w, i / w)))
                .count()
        };
        let level = SampleParams {
            local: false,
            ..Default::default()
        };
        assert!(wrong(&level) > 50, "{}", wrong(&level));
        // The surface is not what a module is read against first; a
        // read comes to it in its turn.
        assert!(!SampleParams::default().surface);
        let surface = SampleParams {
            surface: true,
            ..level
        };
        assert_eq!(wrong(&surface), 0);
        // The contrast goes with the threshold: a module at the dark end
        // is read nearly as surely as one at the light end, where under
        // one contrast for the whole symbol it would be all but unsure.
        let out = classify(&sampled, &surface);
        let (first, last) = (out.reliability[w + 1], out.reliability[(h - 2) * w + w - 2]);
        assert!(first > 150 && last > 150, "{first} {last}");
        // A threshold given, or the neighbourhood rule, is as it was.
        let given = SampleParams {
            threshold: Some(0.3),
            ..surface
        };
        let plain = SampleParams {
            threshold: Some(0.3),
            ..level
        };
        assert_eq!(classify(&sampled, &given), classify(&sampled, &plain));
        let local = SampleParams {
            local: true,
            ..surface
        };
        assert_eq!(
            classify(&sampled, &local),
            classify(&sampled, &SampleParams::default())
        );
        // A corner with nothing read in it takes the symbol's own level.
        let mut holed = sampled.clone();
        for y in 0..SURFACE_WINDOW {
            for x in 0..SURFACE_WINDOW {
                holed.luma[y * w + x] = None;
            }
        }
        let out = classify(&holed, &surface);
        assert_eq!(out.modules[0], Module::Unknown);
        assert_ne!(out.modules[h * w - 1], Module::Unknown);
        // A symbol no bigger than the window has one for every corner.
        let small = Sampled {
            width: 7,
            height: 5,
            luma: (0..35).map(|i| Some(if i % 2 == 0 { 0.1 } else { 0.9 })).collect(),
        };
        let out = classify(&small, &surface);
        assert!((0..35).all(|i| out.modules[i] == Module::from_dark(i % 2 == 0)));
    }

    #[test]
    fn a_read_tries_the_rule_asked_for_first_and_then_the_others() {
        let asked = SampleParams::default();
        assert_eq!(asked.rule(), Rule::Neighbourhood);
        assert_eq!(asked.rules().collect::<Vec<_>>(), [Rule::Neighbourhood, Rule::Level, Rule::Surface]);
        let surface = asked.under(Rule::Surface);
        assert_eq!((surface.local, surface.surface, surface.rule()), (false, true, Rule::Surface));
        assert_eq!(surface.rules().collect::<Vec<_>>(), [Rule::Surface, Rule::Neighbourhood, Rule::Level]);
        let level = surface.under(Rule::Level);
        assert_eq!((level.local, level.surface, level.rule()), (false, false, Rule::Level));
        assert_eq!(level.rules().collect::<Vec<_>>(), [Rule::Level, Rule::Neighbourhood, Rule::Surface]);
        // Every rule comes back as the rule it was set to.
        for rule in Rule::ALL {
            assert_eq!(asked.under(rule).rule(), rule);
            assert_eq!(surface.under(rule).rule(), rule);
        }
        // A threshold given is one level, with no surface beside it: the
        // surface ticked under it is not what a module is read against,
        // and a read tries none.
        let given = SampleParams { threshold: Some(0.4), local: false, surface: true, ..asked };
        assert_eq!(given.rule(), Rule::Level);
        assert_eq!(given.rules().collect::<Vec<_>>(), [Rule::Level, Rule::Neighbourhood]);
        assert_eq!(given.under(Rule::Neighbourhood).threshold, Some(0.4));
        assert_eq!(given.under(Rule::Surface).threshold, None);
    }

    #[test]
    fn local_midpoint_is_not_pulled_by_the_majority_colour() {
        // Seven dark modules at 0.1 and two light ones at 0.9 in a row of
        // nine, plus one at 0.45 that a mean of the row would call light.
        let luma: Vec<Option<f32>> = [0.1, 0.1, 0.1, 0.1, 0.45, 0.1, 0.1, 0.9, 0.9, 0.1]
            .iter()
            .map(|&v| Some(v))
            .collect();
        let sampled = Sampled {
            width: 10,
            height: 1,
            luma,
        };
        let out = classify(
            &sampled,
            &SampleParams {
                local: true,
                local_radius: 9,
                band: 0.0,
                ..Default::default()
            },
        );
        assert_eq!(out.modules[4], Module::Dark);
    }

    #[test]
    fn sampling_reads_module_centres() {
        // 4×2 modules, 10 px each, with a one-pixel dark border bleeding in
        // from the left neighbour of every light module.
        let (mw, mh, px) = (4, 2, 10);
        let pattern = [true, false, false, true, false, true, true, false];
        let mut data = vec![1.0f32; mw * px * mh * px];
        for y in 0..mh * px {
            for x in 0..mw * px {
                if pattern[(y / px) * mw + x / px] {
                    data[y * mw * px + x] = 0.0;
                }
            }
        }
        let image = Luma::new(mw * px, mh * px, data);
        let h = Homography::rect_to_quad(
            mw as f64,
            mh as f64,
            [[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [0.0, 20.0]],
        )
        .unwrap();
        let sampled = sample_modules(&image, &h, mw, mh);
        for (value, dark) in sampled.luma.iter().zip(pattern) {
            assert_eq!(value.unwrap(), if dark { 0.0 } else { 1.0 });
        }
        let some = sample_positions(
            &image,
            &h,
            &[Pos::new(3, 1), Pos::new(0, 0)],
            INNER_FRACTION,
            POINTS,
        );
        assert_eq!(some, [Some(1.0), Some(0.0)]);
        // Sampled densely over its whole area, a light module's dark
        // neighbours bleed into the average without changing its colour.
        let whole = sample_positions(&image, &h, &[Pos::new(1, 0)], 1.0, 20);
        assert!(
            whole[0].unwrap() < 1.0 && whole[0].unwrap() > 0.5,
            "{whole:?}"
        );
        // Shifted half off the image, the outer column has no evidence.
        let shifted = Homography::rect_to_quad(
            mw as f64,
            mh as f64,
            [[-8.0, 0.0], [32.0, 0.0], [32.0, 20.0], [-8.0, 20.0]],
        )
        .unwrap();
        let sampled = sample_modules(&image, &shifted, mw, mh);
        assert_eq!(sampled.luma[0], None);
        assert!(sampled.luma[1].is_some());
        // The centre kernel reads one point, the module's middle: a dot
        // of ink there is read whole where the area average dilutes it.
        let mut dotted = vec![1.0f32; mw * px * mh * px];
        for y in 4..6 {
            for x in 14..16 {
                dotted[y * mw * px + x] = 0.0;
            }
        }
        let dots = Luma::new(mw * px, mh * px, dotted);
        let centre = sample_modules_with(&dots, &h, mw, mh, Kernel::Centre);
        let area = sample_modules_with(&dots, &h, mw, mh, Kernel::Area);
        assert_eq!(centre.luma[1], Some(0.0));
        assert!(area.luma[1].unwrap() > 0.5, "{:?}", area.luma[1]);
        // The vote reads the centre and four points a fifth of a module
        // out: the dot of two pixels covers the centre alone, and is
        // outvoted; a speck of a pixel beside the centre of a light module
        // is outvoted too, where the centre kernel reads it whole.
        let vote = sample_modules_with(&dots, &h, mw, mh, Kernel::Vote);
        assert_eq!(vote.luma[1], Some(1.0));
        let mut specked = vec![1.0f32; mw * px * mh * px];
        specked[4 * mw * px + 15] = 0.0;
        let speck = Luma::new(mw * px, mh * px, specked);
        assert!(sample_modules_with(&speck, &h, mw, mh, Kernel::Centre).luma[1].unwrap() < 1.0);
        assert_eq!(
            sample_modules_with(&speck, &h, mw, mh, Kernel::Vote).luma[1],
            Some(1.0)
        );
        assert_eq!(Kernel::Area.other(), Kernel::Centre);
        assert_eq!(Kernel::Vote.other(), Kernel::Area);
        assert_eq!(
            Kernel::Centre.retries().collect::<Vec<_>>(),
            [Kernel::Centre, Kernel::Area, Kernel::Vote]
        );
        assert_eq!(SampleParams::default().kernel, Kernel::Vote);
        assert_eq!(Kernel::default(), Kernel::Vote);
    }

    #[test]
    fn a_poorly_separated_reading_is_sharpened_and_a_clean_one_left_alone() {
        // A random pattern of modules, blurred so that each reads half its
        // own colour and half the mean of its four neighbours: a dark
        // module among light ones reads as grey as a light one among dark,
        // and the luminances fill the middle. Sharpened against the
        // neighbours they separate again. The clean pattern is returned
        // as it is.
        let (w, h) = (15, 15);
        let mut seed = 2_463_534_242u32;
        let truth: Vec<bool> = (0..w * h)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed & 1 == 1
            })
            .collect();
        let level = |dark: bool| if dark { 0.1f32 } else { 0.9 };
        let clean = Sampled {
            width: w,
            height: h,
            luma: truth.iter().map(|&d| Some(level(d))).collect(),
        };
        assert!(clean.separability() > 0.99);
        assert_eq!(clean.clone().sharpened_for_separation(), clean);
        let at = |x: i64, y: i64| {
            (x >= 0 && y >= 0 && x < w as i64 && y < h as i64)
                .then(|| level(truth[y as usize * w + x as usize]))
        };
        // Each module reads about half its own colour and half the mean of
        // its four neighbours: a dark module among light ones reads as grey
        // as a light one among dark, the luminances fill the middle, and
        // ten modules fall on the wrong side of the split.
        let blurred: Vec<Option<f32>> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as i64, (i / w) as i64);
                let around: Vec<f32> = [at(x - 1, y), at(x + 1, y), at(x, y - 1), at(x, y + 1)]
                    .into_iter()
                    .flatten()
                    .collect();
                let mean = around.iter().sum::<f32>() / around.len() as f32;
                Some(0.48 * level(truth[i]) + 0.52 * mean)
            })
            .collect();
        let smeared = Sampled {
            width: w,
            height: h,
            luma: blurred,
        };
        let wrong = |s: &Sampled| {
            s.luma
                .iter()
                .zip(&truth)
                .filter(|(v, d)| (v.unwrap() < 0.5) != **d)
                .count()
        };
        let before = smeared.separability();
        assert!(before < SHARPEN_FOR_SEPARABILITY, "{before}");
        assert!(wrong(&smeared) > 0);
        let sharpened = smeared.clone().sharpened_for_separation();
        let after = sharpened.separability();
        assert!(
            after >= before + SHARPEN_SEPARATION_GAIN,
            "{before} -> {after}"
        );
        assert_eq!(wrong(&sharpened), 0, "{} wrong before", wrong(&smeared));
    }

    #[test]
    fn modules_are_left_as_read_when_asked_and_a_sharpening_is_told_of() {
        // Modules two pixels across, under the size at which a reading
        // is sharpened against its neighbours.
        let (mw, mh) = (4usize, 2usize);
        let pattern = [true, false, false, true, false, true, true, false];
        let photo = |px: usize| {
            let data = (0..mw * px * mh * px).map(|i| if pattern[(i / (mw * px) / px) * mw + i % (mw * px) / px] { 0.2 } else { 0.8 }).collect();
            let (w, h) = ((mw * px) as f64, (mh * px) as f64);
            (Luma::new(mw * px, mh * px, data), Homography::rect_to_quad(mw as f64, mh as f64, [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]]).unwrap())
        };
        let (small, map) = photo(2);
        let (plain, none) = sample_modules_noting(&small, &map, mw, mh, Kernel::Area, false);
        assert_eq!(none, Sharpening { module_px: 2.0, ..Sharpening::default() });
        assert!(!none.any());
        for (i, value) in plain.luma.iter().enumerate() {
            assert_eq!(*value, Kernel::Area.sample(&small, &map, i % mw, i / mw), "module {i} is as the kernel read it");
        }
        // Sharpened, as it is unless that is turned off, and by how much
        // is told: half the strength a vanishing module is given.
        let (sharpened, done) = sample_modules_noting(&small, &map, mw, mh, Kernel::Area, true);
        assert!(done.any() && (done.for_size - 0.4).abs() < 1e-6 && done.for_separation == 0.0, "{done:?}");
        assert_eq!(sharpened, plain.sharpened(done.for_size));
        assert_ne!(sharpened, plain);
        assert_eq!(sample_modules_with(&small, &map, mw, mh, Kernel::Area), sharpened);
        // The parameters say which a sampling by them is.
        let off = SampleParams { kernel: Kernel::Area, sharpen_modules: false, ..SampleParams::default() };
        assert_eq!(sample_modules_as(&small, &map, mw, mh, &off), plain);
        assert_eq!(sample_modules_as(&small, &map, mw, mh, &SampleParams { sharpen_modules: true, ..off }), sharpened);
        // Modules large enough, which separate, are not sharpened at all.
        let (large, map) = photo(10);
        let (read, done) = sample_modules_noting(&large, &map, mw, mh, Kernel::Area, true);
        assert_eq!(done, Sharpening { module_px: 10.0, ..Sharpening::default() });
        assert_eq!(read, sample_modules_noting(&large, &map, mw, mh, Kernel::Area, false).0);
    }

    #[test]
    fn reliability_is_calibrated_against_the_modules_whose_colour_is_known() {
        // A symbol's worth of readings: every module read right, as far
        // from its threshold as its noise puts it, and a few read wrong
        // outright, as under specks. The fixed patterns, whose colours
        // are known, say how noisy the photo is and how often a module
        // is wrong whatever its margin, and every module's reliability
        // follows.
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                self.0
            }
        }
        let mut rng = Rng(0xCA11B);
        let (w, h) = (25usize, 25usize);
        let func = FunctionMap::new(SymbolVersion::Qr(2));
        let expected = func.expected_modules().to_vec();
        let truth: Vec<bool> = (0..w * h).map(|i| expected[i].unwrap_or_else(|| rng.next() % 2 == 0)).collect();
        // Readings: a module lies 0.8 of half the contrast from the
        // threshold on its own side, spread by 0.15, and one in forty is
        // on the wrong side by a full margin.
        let noise = |rng: &mut Rng| ((rng.next() % 2001) as f32 / 1000.0 - 1.0) * 0.15 * 1.7;
        let mut speck = vec![false; w * h];
        let mut luma: Vec<Option<f32>> = Vec::with_capacity(w * h);
        for i in 0..w * h {
            let side = if truth[i] { -1.0 } else { 1.0 };
            let gross = rng.next() % 40 == 0;
            speck[i] = gross;
            let margin = if gross { -1.0 } else { 0.8 + noise(&mut rng) };
            luma.push(Some(0.5 + side * margin * 0.4));
        }
        let sampled = Sampled { width: w, height: h, luma };
        let params = SampleParams { local: false, band: 0.0, threshold: Some(0.5), ..SampleParams::default() };
        let mut classified = classify(&sampled, &params);
        assert!(classified.calibration.is_none());
        let before = classified.reliability.clone();
        let calibration = classified.calibrate(&expected, false).expect("enough fixed modules to calibrate on");
        let labelled = expected.iter().flatten().count();
        assert_eq!(calibration.labelled, labelled);
        let misread = (0..w * h).filter(|&i| expected[i].is_some() && speck[i]).count();
        assert_eq!(calibration.misread, misread);
        // The margins are shares of half the contrast the readings
        // themselves show, so a module read right lies about one of them
        // from its threshold whatever the contrast; the noise, 0.15 of
        // the full margin as laid down, is near 0.19 of that.
        assert!((0.9..1.1).contains(&calibration.typical), "{calibration:?}");
        assert!((0.12..0.26).contains(&calibration.sigma), "{calibration:?}");
        assert!((calibration.gross - misread as f32 / labelled as f32).abs() < 1e-6, "{calibration:?}");
        // Under the fit, a module's reliability grows with its margin,
        // never beyond what the gross misreads allow, and the specks
        // among the fixed patterns cap every module's reliability at
        // the bits the gross rate is worth.
        let cap = calibration.reliability(10.0);
        let bits = -(calibration.gross).log2();
        assert_eq!(cap, ((bits - 1.0) * 8.0).round() as u8, "{calibration:?}");
        assert!(classified.reliability.iter().all(|&r| r <= cap));
        let mut by_margin: Vec<(f32, u8)> = classified.margin.iter().copied().zip(classified.reliability.iter().copied()).collect();
        by_margin.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert!(by_margin.windows(2).all(|pair| pair[0].1 <= pair[1].1), "monotone in the margin");
        assert_ne!(classified.reliability, before);
        // A photo with less noise believes the same margin more; the
        // default model lies between.
        let clean = Calibration { sigma: 0.05, gross: 0.0, ..calibration };
        assert!(clean.reliability(0.3) > Calibration::DEFAULT.reliability(0.3));
        assert!(Calibration::DEFAULT.reliability(0.3) > calibration.reliability(0.3));
        assert_eq!((Calibration::DEFAULT.reliability(0.0), Calibration::DEFAULT.reliability(1.0)), (0, 254));
        assert_eq!(Calibration::DEFAULT.reliability(0.25), 61);
        // The scale read back: a coin's toss on the threshold, and a bit
        // more for every eight, which is how often such a reading is
        // misread.
        assert_eq!((reliability_bits(0), reliability_bits(8), reliability_bits(120), reliability_bits(CERTAIN)), (1.0, 2.0, 16.0, reliability_bits(254)));
        assert_eq!((misread_odds(0), misread_odds(120)), (2.0, 65536.0));
        for margin in [0.1f32, 0.3, 0.6] {
            let r = calibration.reliability(margin);
            assert!((reliability_bits(r) - calibration.bits(margin)).abs() <= 0.07, "{margin}: {r} for {} bits", calibration.bits(margin));
        }
        // The symbol shown light on dark: the readings are the other way
        // about, and the fit is the same once that is said.
        let mut inverted = classify(&Sampled { width: w, height: h, luma: sampled.luma.iter().map(|v| v.map(|v| 1.0 - v)).collect() }, &params);
        let fit = inverted.calibrate(&expected, true).expect("fits, read light on dark");
        assert_eq!((fit.labelled, fit.misread), (calibration.labelled, calibration.misread));
        assert!((fit.typical - calibration.typical).abs() < 1e-3 && (fit.sigma - calibration.sigma).abs() < 1e-3, "{fit:?} vs {calibration:?}");
        assert!((fit.gross - calibration.gross).abs() < 1e-6, "{fit:?} vs {calibration:?}");
        assert!(inverted.calibrate(&expected, false).is_none(), "read the wrong way round, the fixed patterns are not where they should be");
        // Too few labelled modules, and nothing is fitted.
        let few: Vec<Option<bool>> = expected.iter().enumerate().map(|(i, &e)| e.filter(|_| i < 40)).collect();
        let mut sparse = classify(&sampled, &params);
        assert_eq!(sparse.calibrate(&few, false), None);
        assert_eq!(sparse.reliability, before);
    }
}
