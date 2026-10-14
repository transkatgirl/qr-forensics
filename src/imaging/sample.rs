//! Reading module colours off a photo through a projection.

use super::Luma;
use super::binarize::Binarizer;
use super::enhance::{Enhance, Enhanced};
use super::homography::{HUBER_K, Homography, REWEIGHTINGS, least_squares};
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
    let module_px = module_px_at_middle(map, width, height);
    if !sharpen {
        return (sampled, Sharpening { module_px: module_px as f32, ..Sharpening::default() });
    }
    sampled.sharpened_as_guessed(module_px)
}

/// The size of a module in the photo, in pixels, at the middle of a
/// `width` × `height` symbol placed by `map`.
fn module_px_at_middle(map: &impl Project, width: usize, height: usize) -> f64 {
    let (cx, cy) = (width as f64 / 2.0, height as f64 / 2.0);
    let step = |dx: f64, dy: f64| {
        let [ax, ay] = map.base(cx, cy);
        let [bx, by] = map.base(cx + dx, cy + dy);
        ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt()
    };
    (step(1.0, 0.0) + step(0.0, 1.0)) / 2.0
}

/// A placement read for a decode: sampled through `map` as `params` ask,
/// through the photo enlarged or sharpened when they ask for that as
/// [`sample_modules_as`] reads one, classified under their rule, and
/// taught by the modules whose colour is known, `expected`, the photo
/// light on dark where `inverted` says, as [`Classified::learn`] teaches
/// it; with what the guess at the blur did to the modules, if anything.
///
/// Where `params` ask for the modules to be sharpened, the blur is
/// learnt from the reading as it was read and undone
/// ([`Classified::equalise`]), and only where there is no blur worth
/// undoing, or too few known modules to learn it from, are the modules
/// sharpened by the guess [`sample_modules_noting`] makes. A blur learnt
/// from a reading the guess has sharpened already is the guess's as much
/// as the lens's, and reaches past the eight neighbours it is fitted to.
/// Over the corpus photographs blurred by half a module, the reading so
/// equalised decoded 325 of 332 symbols, where the guess decoded 308;
/// and on top of the guess the filter decoded as many, its reliabilities
/// believed further than its misreads bore out. Where they ask for the
/// modules as read, the blur is learnt and left in, and says only
/// whether the colours may be calibrated apart ([`Classified::calibrate`]).
pub fn read_modules(
    image: &Luma,
    map: &impl Project,
    width: usize,
    height: usize,
    params: &SampleParams,
    expected: &[Option<bool>],
    inverted: bool,
) -> (Classified, Sharpening) {
    if !params.enhance.is_none() {
        let (w, h) = (width as f64, height as f64);
        let outline = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]].map(|[x, y]| map.apply(x, y));
        if let Some(view) = Enhanced::around(image, outline, params.enhance) {
            return read_modules_as_they_lie(&view.luma, &view.through(map), width, height, params, expected, inverted);
        }
    }
    read_modules_as_they_lie(image, map, width, height, params, expected, inverted)
}

/// [`read_modules`] off the pixels given, as they are.
fn read_modules_as_they_lie(
    image: &Luma,
    map: &impl Project,
    width: usize,
    height: usize,
    params: &SampleParams,
    expected: &[Option<bool>],
    inverted: bool,
) -> (Classified, Sharpening) {
    let (read, mut sharpening) = sample_modules_noting(image, map, width, height, params.kernel, false);
    let mut classified = classify(&read, params);
    if !params.sharpen_modules {
        // Left as read, but with the blur known, which says whether the
        // colours may be fitted apart.
        classified.learn_blur(expected, inverted);
    } else if classified.equalise(expected, inverted).is_none() || !classified.equalised {
        // A reading the guess sharpens is classified afresh and knows no
        // blur: the guess rings about the fixed patterns' thin rings as a
        // blur greys them, and the colours are fitted alike.
        let (sharpened, guessed) = read.sharpened_as_guessed(module_px_at_middle(map, width, height));
        if guessed.any() {
            classified = classify(&sharpened, params);
        }
        sharpening = guessed;
    }
    classified.calibrate(expected, inverted);
    (classified, sharpening)
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

    /// The reading sharpened against its neighbours by the guess
    /// [`sample_modules_noting`] makes at the blur, its modules
    /// `module_px` pixels across: the more the smaller they are under
    /// [`SHARPEN_BELOW_PX`], and again where it separates poorly
    /// ([`Self::sharpened_for_separation`]); with what was done.
    pub fn sharpened_as_guessed(self, module_px: f64) -> (Self, Sharpening) {
        let mut done = Sharpening { module_px: module_px as f32, ..Sharpening::default() };
        let sampled = if module_px < SHARPEN_BELOW_PX {
            done.for_size = SHARPEN_GAIN * (1.0 - module_px / SHARPEN_BELOW_PX) as f32;
            self.sharpened(done.for_size)
        } else {
            self
        };
        let (sampled, gain) = sampled.sharpened_for_separation_by();
        done.for_separation = gain;
        (sampled, done)
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
    /// or its neighbourhood had no contrast. Once the read is
    /// [calibrated](Self::calibrate), the distance from the threshold the
    /// calibration settles, which need not lie where the rule put it.
    pub margin: Vec<f32>,
    /// Where each module lies about the threshold its rule gave it, as a
    /// share of half the local contrast: below nought on the dark side,
    /// above it on the light. What `decided` and `margin` are made from,
    /// and what an [equalising](Self::equalise) works on. Nought where
    /// the module could not be sampled or its neighbourhood had no
    /// contrast.
    pub offset: Vec<f32>,
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
    /// How the photo blurs each module into its neighbours, once it has
    /// been [learnt](Self::learn_blur).
    pub blur: Option<Blur>,
    /// The reading has been [equalised](Self::equalise): the blur undone.
    pub equalised: bool,
    /// Modules to a row of the symbol, which every list here runs through
    /// row by row.
    pub width: usize,
    /// The unsure band the modules were classified with, as a share of
    /// half the local contrast ([`SampleParams::band`]), kept so that a
    /// module moved by equalising or calibration is judged by it again.
    pub band: f32,
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
    /// The labelled modules of each colour give where the readings of
    /// that colour lie about the threshold, and how they spread, taken
    /// robustly so that a torn finder pattern does not set either: the
    /// photo's noise on the dark side and on the light ([`Calibration`]).
    /// From the two follow the odds of a misread at any reading, and the
    /// threshold itself, where a reading is as likely of one colour as of
    /// the other: the rule that gave each module its threshold put it
    /// midway between dark and light, which is where it belongs only
    /// when the two spread alike. And the modules read wrong by more than
    /// that noise explains are the gross misreads, specks and stains and
    /// tears, which no margin protects against and which cap what any
    /// module's reading may be believed. Every sampled module is then
    /// judged again under the model: its colour by the side of the
    /// threshold it lies on, and its reliability by the odds. Nothing is
    /// changed, and `None` returned, with fewer than
    /// [`CALIBRATION_MIN_LABELLED`] labelled modules, or when they read so
    /// poorly that the labels are not to be trusted: the placement is off
    /// the symbol.
    ///
    /// The two colours are fitted apart only once the blur has been
    /// [learnt](Self::learn_blur), and undone or found too slight to be
    /// worth undoing. The fixed patterns hold their dark modules in solid
    /// blocks and their light ones in thin rings and strips between dark,
    /// which a blur greys and leaves the blocks as they were: fitted apart
    /// on a blurred reading, the light side spreads wider than the dark
    /// for the blur's sake and not the photo's, and the threshold moves
    /// to call grey modules light, which over blurred photographs read
    /// fewer symbols than one spread for both. Without that, the colours
    /// take one spread between them, about a threshold midway.
    pub fn calibrate(&mut self, expected: &[Option<bool>], inverted: bool) -> Option<Calibration> {
        let labelled: Vec<(f32, bool)> = expected
            .iter()
            .zip(&self.decided)
            .zip(&self.offset)
            .filter_map(|((want, got), &offset)| {
                got.as_dark()?;
                Some((offset, (*want)? != inverted))
            })
            .collect();
        let apart = self.blur.is_some_and(|blur| self.equalised || !blur.is_worth_undoing());
        let calibration = Calibration::fit(&labelled, apart)?;
        for i in 0..self.offset.len() {
            if self.decided[i] != Module::Unknown {
                self.judge(i, calibration.threshold, &calibration);
            }
        }
        self.calibration = Some(calibration);
        Some(calibration)
    }

    /// Undo the blur the photo shows, as the modules whose colour is
    /// known show it, before the reading is calibrated: `expected` and
    /// `inverted` are as [`Self::calibrate`] takes them.
    ///
    /// A lens out of focus, or a hand that moved, spreads each module
    /// into its neighbours, and a dark module among light ones reads
    /// grey. How far it spreads, and in which direction, is learnt from
    /// the labelled modules whose eight neighbours are labelled too, as
    /// the middles of the finder patterns are ([`Blur::fit`]); the
    /// reading is then put through the filter that best undoes that
    /// spread without raising the noise past what it gains
    /// ([`Blur::undo`]), and every module is judged again by where it
    /// now lies. A photo too sharp for the filter to gain much
    /// ([`Blur::is_worth_undoing`]) is left as it was read. The blur
    /// learnt, undone or not; `None` when none could be learnt
    /// ([`Self::learn_blur`]).
    ///
    /// The fixed patterns are mostly solid, where a blur moves nothing,
    /// and the data mostly is not: a calibration on the blurred reading
    /// would learn from the finder patterns a noise the data does not
    /// have, and believe the data too far. Equalised, a module reads its
    /// own colour wherever it lies, and the patterns speak for the data:
    /// over photographs blurred by a third and by half a module, the
    /// noise the fixed patterns showed was two thirds and three quarters
    /// of the data's as read, and the same as the data's equalised.
    pub fn equalise(&mut self, expected: &[Option<bool>], inverted: bool) -> Option<Blur> {
        let blur = self.learn_blur(expected, inverted)?;
        if blur.is_worth_undoing() && !self.equalised {
            let read = self.readings();
            for (i, offset) in blur.undo(&read, self.width).into_iter().enumerate() {
                if let Some(offset) = offset {
                    self.offset[i] = offset;
                    self.judge(i, 0.0, &Calibration::DEFAULT);
                }
            }
            // A calibration made before is of a reading that is no more.
            self.calibration = None;
            self.equalised = true;
        }
        Some(blur)
    }

    /// Learn how the photo blurs from the modules whose colour is known,
    /// as [`Self::equalise`] does, and keep it, leaving the reading as
    /// it is: what tells [`Self::calibrate`] whether the reading is sharp
    /// enough for each colour's spread to be its own, where the modules
    /// are not to be sharpened. `None`, and nothing kept, when too few
    /// known modules have known neighbours all round, or the fit says
    /// the placement is off the symbol ([`Blur::fit`]). Once the reading
    /// is equalised the blur it was equalised by stands.
    pub fn learn_blur(&mut self, expected: &[Option<bool>], inverted: bool) -> Option<Blur> {
        if self.equalised {
            return self.blur;
        }
        let blur = Blur::fit(&self.readings(), self.width, expected, inverted)?;
        self.blur = Some(blur);
        Some(blur)
    }

    /// Each module's offset where it was read at all.
    fn readings(&self) -> Vec<Option<f32>> {
        self.offset.iter().zip(&self.decided).map(|(&offset, decided)| (*decided != Module::Unknown).then_some(offset)).collect()
    }

    /// Learn from the modules whose colour is known how the photo blurs
    /// and how far its readings are to be believed: the reading
    /// [equalised](Self::equalise), then [calibrated](Self::calibrate).
    /// The calibration, when one could be made.
    pub fn learn(&mut self, expected: &[Option<bool>], inverted: bool) -> Option<Calibration> {
        self.equalise(expected, inverted);
        self.calibrate(expected, inverted)
    }

    /// Judge module `i` again by where it lies: dark below `threshold`
    /// and light above it, its margin how far it lies from it, unknown
    /// within the band, and its reliability what `calibration` makes of
    /// its offset.
    fn judge(&mut self, i: usize, threshold: f32, calibration: &Calibration) {
        let offset = self.offset[i];
        let side = Module::from_dark(offset < threshold);
        self.margin[i] = (offset - threshold).abs();
        self.decided[i] = side;
        self.modules[i] = if self.margin[i] < self.band { Module::Unknown } else { side };
        self.reliability[i] = calibration.reliability(offset);
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
/// Fewest labelled modules of a colour for that colour's readings to be
/// fitted on their own; with fewer, the colour takes the noise of both
/// together, about the typical margin.
pub const CALIBRATION_MIN_EACH: usize = 12;
/// How many labelled modules' worth of weight the noise of both colours
/// together carries in each colour's own: a colour with few labelled
/// modules keeps near the noise of both, one with hundreds speaks for
/// itself.
const CALIBRATION_PRIOR: f32 = 16.0;
/// The most one colour's readings may spread against the other's: glare
/// that flattens the paper, or ink that reads unevenly, spreads one side
/// twice the other, and a fit further apart than this is two colours
/// read off too few modules.
const CALIBRATION_MAX_RATIO: f32 = 3.0;

/// Where the readings of one colour lie about their rule's threshold,
/// and how they spread, as shares of half the local contrast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spread {
    /// The middle of the readings: below nought for the dark, above it
    /// for the light, and further from it the sharper the photo.
    pub mean: f32,
    /// How they spread about it: the photo's noise on that side.
    pub sigma: f32,
}

/// How a module's reading is to be believed, as a function of where it
/// lies about its threshold: the misread model [`Classified::calibrate`]
/// fits to a photo, or the default for one not yet fitted.
///
/// The readings of each colour are taken to lie about their own mean
/// with their own spread. A photo seldom spreads the two alike: glare
/// flattens the paper towards white and leaves the ink as it was, and
/// ink soaked unevenly into the paper, or printed in dots, spreads the
/// dark side where the paper stays even. Where the two spread alike the
/// model is the one the reliabilities were first computed under, odds
/// that grow with a reading's distance from a threshold midway between
/// the means; where they do not, a reading near the threshold is likelier
/// of the colour that spreads the wider, and the threshold itself moves
/// toward the other.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Calibration {
    /// The readings of the modules dark in the photo, and of those light.
    pub dark: Spread,
    pub light: Spread,
    /// Where a reading is as likely dark as light, about the threshold
    /// the rule gave it: nought where the two colours spread alike about
    /// it, toward the colour that spreads the less where they do not.
    pub threshold: f32,
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
    /// middling quality shows, alike on both sides, under which the
    /// reliability runs near enough with the margin as the linear scale
    /// it replaced did, two hundred and fifty-four at a module of full
    /// contrast and sixty-one at a quarter of it.
    pub const DEFAULT: Self = Self {
        dark: Spread { mean: -1.0, sigma: 0.29 },
        light: Spread { mean: 1.0, sigma: 0.29 },
        threshold: 0.0,
        gross: 0.0,
        labelled: 0,
        misread: 0,
    };

    /// The model fitted to `labelled`: each module's offset from its
    /// rule's threshold ([`Classified::offset`]), and whether it is dark
    /// in the photo; each colour apart where `apart` says, and otherwise
    /// both alike, about a threshold midway. `None` with fewer than
    /// [`CALIBRATION_MIN_LABELLED`] of them, or when the typical module
    /// lies within [`CALIBRATION_MIN_TYPICAL`] of its threshold on the
    /// side its colour should be, as one does when the labels are not
    /// where the placement puts them.
    pub fn fit(labelled: &[(f32, bool)], apart: bool) -> Option<Self> {
        if labelled.len() < CALIBRATION_MIN_LABELLED {
            return None;
        }
        // Both colours together, each module's offset toward the colour
        // it should have: what a colour with too few modules of its own
        // falls back on, and what a colour with few leans towards.
        let mut toward: Vec<f32> = labelled.iter().map(|&(offset, dark)| if dark { -offset } else { offset }).collect();
        let typical = median(&mut toward);
        if typical < CALIBRATION_MIN_TYPICAL {
            return None;
        }
        let both = spread_about(&toward, typical);
        let side = |dark: bool, fallback: f32| {
            let mut values: Vec<f32> = labelled.iter().filter(|l| l.1 == dark).map(|l| l.0).collect();
            if !apart || values.len() < CALIBRATION_MIN_EACH {
                return Spread { mean: fallback, sigma: both };
            }
            let mean = median(&mut values);
            let own = spread_about(&values, mean);
            let n = values.len() as f32;
            Spread { mean, sigma: ((n * own * own + CALIBRATION_PRIOR * both * both) / (n + CALIBRATION_PRIOR)).sqrt() }
        };
        let (mut dark, mut light) = (side(true, -typical), side(false, typical));
        if light.mean - dark.mean < 2.0 * CALIBRATION_MIN_TYPICAL {
            return None;
        }
        // The spreads drawn together about their geometric middle as far
        // as the ratio allows.
        let middle = (dark.sigma * light.sigma).sqrt();
        let reach = CALIBRATION_MAX_RATIO.sqrt();
        dark.sigma = dark.sigma.clamp(middle / reach, middle * reach);
        light.sigma = light.sigma.clamp(middle / reach, middle * reach);
        let mut fitted = Self { dark, light, threshold: 0.0, gross: 0.0, labelled: labelled.len(), misread: 0 };
        // Two colours, one spread far wider than the other and both near
        // the threshold, can leave the wider so likely everywhere that a
        // reading at the narrower one's own middle is called the wider
        // colour. Such a fit says little but that the photo is noisy, and
        // the two take one spread between them.
        if fitted.log_odds(dark.mean) >= 0.0 || fitted.log_odds(light.mean) <= 0.0 {
            let one = ((dark.sigma * dark.sigma + light.sigma * light.sigma) / 2.0).sqrt();
            fitted.dark.sigma = one;
            fitted.light.sigma = one;
        }
        // The odds rise steadily from the dark middle to the light, and
        // the threshold is where they cross even: midway between the two,
        // exactly, where they spread alike.
        fitted.threshold = if fitted.dark.sigma == fitted.light.sigma {
            (fitted.dark.mean + fitted.light.mean) / 2.0
        } else {
            let (mut low, mut high) = (fitted.dark.mean, fitted.light.mean);
            for _ in 0..40 {
                let mid = (low + high) / 2.0;
                if fitted.log_odds(mid) < 0.0 {
                    low = mid;
                } else {
                    high = mid;
                }
            }
            (low + high) / 2.0
        };
        let (threshold, dark, light) = (fitted.threshold, fitted.dark, fitted.light);
        fitted.misread = labelled.iter().filter(|&&(offset, is_dark)| (offset < threshold) != is_dark).count();
        let gross = labelled
            .iter()
            .filter(|&&(offset, is_dark)| {
                if is_dark { offset > threshold + 2.0 * dark.sigma } else { offset < threshold - 2.0 * light.sigma }
            })
            .count();
        fitted.gross = gross as f32 / labelled.len() as f32;
        Some(fitted)
    }

    /// How far a module read right lies from the threshold, on average
    /// over the two colours, as a share of half the contrast: one on a
    /// sharp photo of a clean print, less as blur draws dark and light
    /// together.
    pub fn typical(&self) -> f32 {
        (self.light.mean - self.dark.mean) / 2.0
    }

    /// The natural log of the odds that a reading `offset` from its
    /// rule's threshold is light rather than dark, from how the readings
    /// of each colour lie and spread: zero at [`Self::threshold`],
    /// positive above it. Between the two means it is the log of the
    /// ratio of the two colours' normal densities; beyond either it is
    /// carried on at the slope it had there, since two normal densities
    /// of unequal spread would otherwise turn about beyond the narrower
    /// one's mean, and call a reading darker than any dark module light.
    /// With the spreads alike it is a straight line, `2 · (offset -
    /// middle) · half the gap between the means / sigma²`, the odds the
    /// reliabilities were first computed under.
    pub fn log_odds(&self, offset: f32) -> f32 {
        let (dark, light) = (self.dark, self.light);
        let between = |u: f32| {
            (dark.sigma / light.sigma).ln() - (u - light.mean).powi(2) / (2.0 * light.sigma * light.sigma)
                + (u - dark.mean).powi(2) / (2.0 * dark.sigma * dark.sigma)
        };
        let gap = light.mean - dark.mean;
        if offset > light.mean {
            between(light.mean) + (offset - light.mean) * gap / (dark.sigma * dark.sigma)
        } else if offset < dark.mean {
            between(dark.mean) + (offset - dark.mean) * gap / (light.sigma * light.sigma)
        } else {
            between(offset)
        }
    }

    /// The bits of likelihood that a module read `offset` from its
    /// rule's threshold is the colour the side of [`Self::threshold`] it
    /// lies on says: minus the log of the odds it was misread, from the
    /// odds of the two colours there ([`Self::log_odds`]); and besides
    /// that a share `gross` of modules are wrong whatever their reading.
    pub fn bits(&self, offset: f32) -> f32 {
        let odds = self.log_odds(offset).abs().min(87.0);
        let wrong = self.gross + (1.0 - self.gross) / (1.0 + odds.exp());
        -wrong.max(f32::MIN_POSITIVE).log2()
    }

    /// Reliability for a module read `offset` from its rule's threshold:
    /// eight times the bits of likelihood less eight, so that a module on
    /// the threshold, read right as often as not, is nought, and a module
    /// whose reading is beyond doubt reaches the cap of 254, one short of
    /// [`CERTAIN`], which is for modules painted by hand.
    pub fn reliability(&self, offset: f32) -> u8 {
        ((self.bits(offset) - 1.0) * 8.0).round().clamp(0.0, (CERTAIN - 1) as f32) as u8
    }
}

/// The spread of `values` about `centre`, from their median absolute
/// deviation, within the bounds a photo's noise is held to.
fn spread_about(values: &[f32], centre: f32) -> f32 {
    let mut deviations: Vec<f32> = values.iter().map(|v| (v - centre).abs()).collect();
    (MAD_TO_SIGMA * median(&mut deviations)).clamp(CALIBRATION_MIN_SIGMA, CALIBRATION_MAX_SIGMA)
}

/// Fewest modules a blur is fitted on: the modules of one finder pattern
/// whose eight neighbours lie in the pattern or its separator, all the
/// fixed modules of a Micro QR symbol have to give, are thirty-six.
pub const BLUR_MIN_FITTED: usize = 32;
/// The most a neighbour may move a module's reading, as a share of what
/// the module's own colour moves it, for the fit to be believed. A lens
/// that blurs by seven tenths of a module, past what any kernel reads,
/// moves it by about two fifths; a grid half a module off its symbol
/// reads every module as much its neighbour as itself.
const BLUR_MAX_SHARE: f32 = 0.6;
/// Weight, in labelled modules, of what holds each neighbour's share at
/// nought: nothing beside a finder pattern's thirty-six modules, but a
/// fit made on fewer, or on modules too alike to tell the neighbours
/// apart, leans to a sharp photo rather than to a guess.
const BLUR_RIDGE: f64 = 2.0;
/// How much clearer a module's own colour must stand in its reading
/// once equalised, against what its neighbours and the noise put there,
/// for the reading to be equalised: a tenth. A photo near enough sharp
/// gains less, and is better read as it lies than through a filter
/// fitted to it.
pub const BLUR_MIN_GAIN: f32 = 0.1;
/// Modules each way from a module that the equaliser reads: the blur
/// is fitted to the module's eight neighbours, and undoing it reaches
/// one further, where the neighbours' own neighbours spread back.
const EQUALISER_REACH: i64 = 2;
/// Taps of the equaliser, a square [`EQUALISER_REACH`] each way.
const EQUALISER_TAPS: usize = 25;

/// How the photo blurs each module into its neighbours, as the modules
/// whose colour is known show it ([`Blur::fit`]), and the filter that
/// undoes it ([`Blur::undo`]). Everything is a share of half the local
/// contrast, as [`Classified::offset`] is.
///
/// The module's own colour moves its reading by `own`, and each of its
/// eight neighbours moves it by the share of its pair: the two beside it
/// in its row, the two above and below it, and the two corners on each
/// diagonal. A blurred lens spreads alike every way, and a hand that
/// moved spreads along the way it moved. Each pair is taken as alike on
/// both sides, as a blur is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Blur {
    /// The module's own colour, which is all a sharp photo's reading is.
    pub own: f32,
    /// Left and right.
    pub across: f32,
    /// Above and below.
    pub down: f32,
    /// Top left and bottom right.
    pub falling: f32,
    /// Bottom left and top right.
    pub rising: f32,
    /// How the readings spread about what the blur makes of their
    /// neighbourhoods: the photo's noise.
    pub noise: f32,
    /// Modules the blur was fitted on.
    pub fitted: usize,
    /// How much clearer a module's own colour stands against what its
    /// neighbours and the noise put into its reading once equalised than
    /// before: the ratio of the two, of the squares of what is its own to
    /// what is not.
    pub gain: f32,
    /// The equaliser, row-major, [`EQUALISER_REACH`] each way: the share
    /// of each module about it that a module's equalised reading is made
    /// of. They sum to one, so that a patch of one colour reads as it did.
    taps: [f32; EQUALISER_TAPS],
}

impl Blur {
    /// The blur fitted to `read`, each module's offset from its rule's
    /// threshold where it was read ([`Classified::offset`]), over a symbol
    /// `width` modules wide, against `expected`, the colour of each
    /// module where it is known, dark in the photo where `inverted` says
    /// it is not, as [`Classified::calibrate`] takes them. Fitted on the
    /// labelled modules whose eight neighbours are labelled too, by least
    /// squares reweighted as [`super::homography::Homography::fit`] is,
    /// so that a speck on a finder pattern does not set it.
    ///
    /// The fit is of the blur, how each module's colour reaches its
    /// neighbours, and not of the filter that undoes it: the modules it
    /// is fitted on stand mostly among their own colour, as the data does
    /// not, and a filter fitted on them would learn to smooth, which
    /// serves a solid patch and ruins a lone module. The blur is the
    /// lens's and the same everywhere, and the filter is worked out from
    /// it for modules whose neighbours are any colour at all, as the
    /// data's are: the one that leaves a module's reading as near its own
    /// colour as it can come, the neighbours' reach against the noise the
    /// filter raises weighed by the noise the fit found, as a Wiener
    /// filter weighs them.
    ///
    /// `None` when too few modules can be fitted on, or when the fit says
    /// the labels are not where the placement has them: a module's own
    /// colour moves it too little, or a neighbour moves it as much as its
    /// own does.
    pub fn fit(read: &[Option<f32>], width: usize, expected: &[Option<bool>], inverted: bool) -> Option<Self> {
        let height = read.len() / width.max(1);
        let colour = |x: i64, y: i64| expected[y as usize * width + x as usize].map(|dark| if dark != inverted { -1.0 } else { 1.0 });
        let mut rows: Vec<([f64; 6], f64)> = Vec::new();
        for y in 1..height.saturating_sub(1) as i64 {
            for x in 1..width.saturating_sub(1) as i64 {
                let Some(offset) = read[y as usize * width + x as usize] else { continue };
                let around = |dx: i64, dy: i64| colour(x + dx, y + dy);
                let (Some(c), Some(l), Some(r), Some(u), Some(d), Some(ul), Some(dr), Some(ur), Some(dl)) = (
                    around(0, 0),
                    around(-1, 0),
                    around(1, 0),
                    around(0, -1),
                    around(0, 1),
                    around(-1, -1),
                    around(1, 1),
                    around(1, -1),
                    around(-1, 1),
                ) else {
                    continue;
                };
                rows.push(([1.0, c, l + r, u + d, ul + dr, ur + dl], f64::from(offset)));
            }
        }
        if rows.len() < BLUR_MIN_FITTED {
            return None;
        }
        let ridge: Vec<([f64; 6], f64)> = (2..6)
            .map(|k| {
                let mut row = [0.0; 6];
                row[k] = BLUR_RIDGE.sqrt();
                (row, 0.0)
            })
            .collect();
        let solve = |weights: &[f64]| {
            least_squares(
                rows.iter().zip(weights).map(|((row, value), &w)| (row.map(|v| v * w.sqrt()), value * w.sqrt())).chain(ridge.iter().copied()),
            )
        };
        let residuals = |beta: &[f64; 6]| -> Vec<f32> {
            rows.iter().map(|(row, value)| (value - row.iter().zip(beta).map(|(a, b)| a * b).sum::<f64>()) as f32).collect()
        };
        let scale = |residuals: &[f32]| {
            let mut absolute: Vec<f32> = residuals.iter().map(|r| r.abs()).collect();
            (MAD_TO_SIGMA * median(&mut absolute)).max(CALIBRATION_MIN_SIGMA)
        };
        let mut weights = vec![1.0; rows.len()];
        let mut beta = solve(&weights)?;
        for _ in 0..REWEIGHTINGS {
            let residuals = residuals(&beta);
            let threshold = HUBER_K as f32 * scale(&residuals);
            let reweighted: Vec<f64> =
                residuals.iter().map(|r| if r.abs() <= threshold { 1.0 } else { f64::from(threshold / r.abs()) }).collect();
            if reweighted == weights {
                break;
            }
            weights = reweighted;
            beta = solve(&weights)?;
        }
        let noise = scale(&residuals(&beta)).min(CALIBRATION_MAX_SIGMA);
        let [_, own, across, down, falling, rising] = beta.map(|v| v as f32);
        if own < CALIBRATION_MIN_TYPICAL || [across, down, falling, rising].iter().any(|share| share.abs() > BLUR_MAX_SHARE * own) {
            return None;
        }
        let mut blur = Self { own, across, down, falling, rising, noise, fitted: rows.len(), gain: 1.0, taps: [0.0; EQUALISER_TAPS] };
        let taps = blur.equaliser()?;
        blur.gain = (blur.clarity(&taps) / blur.clarity(&Self::identity())) as f32;
        blur.taps = taps.map(|t| t as f32);
        Some(blur)
    }

    /// Whether undoing the blur gains enough, [`BLUR_MIN_GAIN`], to be
    /// worth it: a photo near enough sharp is better read as it lies than
    /// through a filter fitted to it.
    pub fn is_worth_undoing(&self) -> bool {
        self.gain >= 1.0 + BLUR_MIN_GAIN
    }

    /// What a module `dx` across and `dy` down from another adds to that
    /// one's reading, per unit of its colour.
    fn share(&self, dx: i64, dy: i64) -> f64 {
        f64::from(match (dx, dy) {
            (0, 0) => self.own,
            (-1 | 1, 0) => self.across,
            (0, -1 | 1) => self.down,
            (-1, -1) | (1, 1) => self.falling,
            (1, -1) | (-1, 1) => self.rising,
            _ => 0.0,
        })
    }

    /// The offset across and down of the module tap `q` of the
    /// equaliser reads.
    fn tap_offset(q: usize) -> (i64, i64) {
        let side = 2 * EQUALISER_REACH + 1;
        (q as i64 % side - EQUALISER_REACH, q as i64 / side - EQUALISER_REACH)
    }

    /// The equaliser that reads nothing but the module itself.
    fn identity() -> [f64; EQUALISER_TAPS] {
        let mut taps = [0.0; EQUALISER_TAPS];
        taps[EQUALISER_TAPS / 2] = 1.0;
        taps
    }

    /// What a module's colour, and each colour about it, come to in its
    /// reading through `taps` after the blur: the share each module `dx`
    /// across and `dy` down from it has, for every module the two reach.
    fn through(&self, taps: &[f64; EQUALISER_TAPS], dx: i64, dy: i64) -> f64 {
        (0..EQUALISER_TAPS)
            .map(|q| {
                let (qx, qy) = Self::tap_offset(q);
                taps[q] * self.share(dx - qx, dy - qy)
            })
            .sum()
    }

    /// How clearly a module's own colour stands in its reading through
    /// `taps`: the square of its own share against the squares of every
    /// other module's and the noise the taps carry through, the colours
    /// of the data being as likely dark as light and each its own.
    fn clarity(&self, taps: &[f64; EQUALISER_TAPS]) -> f64 {
        let reach = EQUALISER_REACH + 1;
        let mut own = 0.0;
        let mut others = f64::from(self.noise).powi(2) * taps.iter().map(|t| t * t).sum::<f64>();
        for dy in -reach..=reach {
            for dx in -reach..=reach {
                let share = self.through(taps, dx, dy);
                if (dx, dy) == (0, 0) {
                    own = share;
                } else {
                    others += share * share;
                }
            }
        }
        own * own / others.max(f64::MIN_POSITIVE)
    }

    /// The taps that bring a module's reading through the blur nearest
    /// to its own colour alone, by least squares over every module the
    /// taps and the blur reach between them, with the noise each tap
    /// carries through counted against it; scaled to sum to one. `None`
    /// when they cannot be solved for, or sum to nothing.
    fn equaliser(&self) -> Option<[f64; EQUALISER_TAPS]> {
        let reach = EQUALISER_REACH + 1;
        let noise = f64::from(self.noise);
        let reached = (-reach..=reach).flat_map(|dy| (-reach..=reach).map(move |dx| (dx, dy))).map(|(dx, dy)| {
            let row: [f64; EQUALISER_TAPS] = std::array::from_fn(|q| {
                let (qx, qy) = Self::tap_offset(q);
                self.share(dx - qx, dy - qy)
            });
            (row, if (dx, dy) == (0, 0) { 1.0 } else { 0.0 })
        });
        let carried = (0..EQUALISER_TAPS).map(|q| {
            let mut row = [0.0; EQUALISER_TAPS];
            row[q] = noise;
            (row, 0.0)
        });
        let taps = least_squares(reached.chain(carried))?;
        let sum: f64 = taps.iter().sum();
        (sum > 1e-6).then(|| taps.map(|t| t / sum))
    }

    /// The equaliser's taps, row-major, [`EQUALISER_REACH`] modules each
    /// way about the module equalised.
    pub fn taps(&self) -> &[f32; EQUALISER_TAPS] {
        &self.taps
    }

    /// How much of a module's reading its eight neighbours put there,
    /// against what its own colour does.
    pub fn spill(&self) -> f32 {
        2.0 * (self.across + self.down + self.falling + self.rising) / self.own
    }

    /// `read`, each module's offset where it was read, over a symbol
    /// `width` modules wide, equalised: each module's offset moved by
    /// each tap's share of how far the module that tap reads lies from
    /// it. The taps sum to one, so this is the filter laid over the
    /// reading; a module beyond the symbol's edge, or one not read, is
    /// taken to lie where the module equalised does, and changes nothing,
    /// as the quiet zone, which may be anything in a photo, is not
    /// guessed at. A module not read stays so.
    pub fn undo(&self, read: &[Option<f32>], width: usize) -> Vec<Option<f32>> {
        let height = (read.len() / width.max(1)) as i64;
        (0..read.len())
            .map(|i| {
                let value = read[i]?;
                let (x, y) = ((i % width) as i64, (i / width) as i64);
                let mut out = value;
                for (q, &tap) in self.taps.iter().enumerate() {
                    let (dx, dy) = Self::tap_offset(q);
                    let (nx, ny) = (x + dx, y + dy);
                    if (dx, dy) == (0, 0) || tap == 0.0 || nx < 0 || ny < 0 || nx >= width as i64 || ny >= height {
                        continue;
                    }
                    if let Some(there) = read[ny as usize * width + nx as usize] {
                        out += tap * (there - value);
                    }
                }
                Some(out)
            })
            .collect()
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
    let mut offsets = Vec::with_capacity(w * h);
    let mut reliability = Vec::with_capacity(w * h);
    let mut flat = 0;
    let mut neighbourhood: Vec<f32> = Vec::new();
    for i in 0..w * h {
        let Some(value) = sampled.luma[i] else {
            modules.push(Module::Unknown);
            decided.push(Module::Unknown);
            margins.push(0.0);
            offsets.push(0.0);
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
                    offsets.push(0.0);
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
        offsets.push((value - threshold) / half);
        reliability.push(quantize(margin));
    }
    Classified {
        modules,
        decided,
        margin: margins,
        offset: offsets,
        reliability,
        calibration: None,
        blur: None,
        equalised: false,
        width: w,
        band: params.band,
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
        // the full margin as laid down, is near 0.19 of that, on either
        // side, which the noise was laid down alike on, and the threshold
        // stays near the middle: the two spreads are fitted on a hundred
        // modules each, and differ by what so few differ by.
        assert!((0.9..1.1).contains(&calibration.typical()), "{calibration:?}");
        for side in [calibration.dark, calibration.light] {
            assert!((0.12..0.26).contains(&side.sigma), "{calibration:?}");
        }
        assert!(calibration.threshold.abs() < 0.1, "{calibration:?}");
        assert!((calibration.gross - misread as f32 / labelled as f32).abs() < 1e-6, "{calibration:?}");
        // Under the fit, a module's reliability grows with its distance
        // from the threshold on either side, never beyond what the gross
        // misreads allow, and the specks among the fixed patterns cap
        // every module's reliability at the bits the gross rate is worth.
        let cap = calibration.reliability(10.0);
        let bits = -(calibration.gross).log2();
        assert_eq!(cap, ((bits - 1.0) * 8.0).round() as u8, "{calibration:?}");
        assert_eq!(calibration.reliability(-10.0), cap);
        assert!(classified.reliability.iter().all(|&r| r <= cap));
        for dark in [true, false] {
            let mut by_margin: Vec<(f32, u8)> = (0..w * h)
                .filter(|&i| classified.decided[i] == Module::from_dark(dark))
                .map(|i| (classified.margin[i], classified.reliability[i]))
                .collect();
            by_margin.sort_by(|a, b| a.0.total_cmp(&b.0));
            assert!(by_margin.windows(2).all(|pair| pair[0].1 <= pair[1].1), "monotone in the margin");
        }
        assert_ne!(classified.reliability, before);
        // A photo with less noise believes the same margin more; the
        // default model lies between.
        let clean = Calibration {
            dark: Spread { sigma: 0.05, ..calibration.dark },
            light: Spread { sigma: 0.05, ..calibration.light },
            gross: 0.0,
            ..calibration
        };
        assert!(clean.reliability(0.3) > Calibration::DEFAULT.reliability(0.3));
        assert!(Calibration::DEFAULT.reliability(0.3) > calibration.reliability(0.3));
        assert_eq!((Calibration::DEFAULT.reliability(0.0), Calibration::DEFAULT.reliability(1.0)), (0, 254));
        assert_eq!((Calibration::DEFAULT.reliability(0.25), Calibration::DEFAULT.reliability(-0.25)), (61, 61));
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
        // The modules dark in the photo are now those light in the symbol,
        // their readings turned about the threshold.
        assert!((fit.typical() - calibration.typical()).abs() < 1e-3, "{fit:?} vs {calibration:?}");
        assert!((fit.dark.mean + calibration.light.mean).abs() < 1e-3 && (fit.dark.sigma - calibration.light.sigma).abs() < 1e-3, "{fit:?} vs {calibration:?}");
        assert!((fit.light.mean + calibration.dark.mean).abs() < 1e-3 && (fit.light.sigma - calibration.dark.sigma).abs() < 1e-3, "{fit:?} vs {calibration:?}");
        assert!((fit.gross - calibration.gross).abs() < 1e-6, "{fit:?} vs {calibration:?}");
        assert!(inverted.calibrate(&expected, false).is_none(), "read the wrong way round, the fixed patterns are not where they should be");
        // Too few labelled modules, and nothing is fitted.
        let few: Vec<Option<bool>> = expected.iter().enumerate().map(|(i, &e)| e.filter(|_| i < 40)).collect();
        let mut sparse = classify(&sampled, &params);
        assert_eq!(sparse.calibrate(&few, false), None);
        assert_eq!(sparse.reliability, before);
    }

    /// Normal deviates for the tests below.
    struct Normal(u64);

    impl Normal {
        fn uniform(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32 + 0.5) / (1u64 << 24) as f32
        }

        fn next(&mut self) -> f32 {
            let (a, b) = (self.uniform(), self.uniform());
            (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
        }
    }

    #[test]
    fn each_colour_is_calibrated_by_its_own_spread() {
        // Ink read unevenly, its readings spread wide about their middle,
        // and clean paper read evenly: a reading a little to the light
        // side of the rule's midway threshold is likelier ink than paper.
        let mut normal = Normal(0x5EED);
        let labelled: Vec<(f32, bool)> =
            (0..400).map(|i| if i % 2 == 0 { (-0.7 + 0.36 * normal.next(), true) } else { (0.9 + 0.14 * normal.next(), false) }).collect();
        let fit = Calibration::fit(&labelled, true).expect("fits");
        assert!((fit.dark.mean + 0.7).abs() < 0.08 && (fit.light.mean - 0.9).abs() < 0.04, "{fit:?}");
        assert!((0.28..0.44).contains(&fit.dark.sigma) && (0.1..0.18).contains(&fit.light.sigma), "{fit:?}");
        // The threshold moves toward the paper, and the odds rise steadily
        // through it and beyond both middles: a reading darker than any
        // ink is not called paper for lying far from the ink's middle.
        assert!(fit.threshold > 0.2 && fit.threshold < fit.light.mean, "{fit:?}");
        assert!(fit.log_odds(fit.threshold).abs() < 1e-3);
        assert!(fit.log_odds(0.1) < 0.0 && fit.reliability(0.1) > 0);
        let odds: Vec<f32> = (-300..=300).map(|i| fit.log_odds(i as f32 / 100.0)).collect();
        assert!(odds.windows(2).all(|pair| pair[0] < pair[1]), "rising");
        // A colour with too few modules of its own takes both colours'
        // noise about the typical margin.
        let lopsided: Vec<(f32, bool)> = labelled.iter().copied().filter(|l| !l.1).chain(labelled.iter().copied().filter(|l| l.1).take(5)).collect();
        let fit = Calibration::fit(&lopsided, true).expect("fits");
        let mut toward: Vec<f32> = lopsided.iter().map(|&(o, d)| if d { -o } else { o }).collect();
        let typical = median(&mut toward);
        assert_eq!(fit.dark.mean, -typical, "{fit:?}");
        assert!(fit.dark.sigma <= spread_about(&toward, typical) * 1.0001, "{fit:?}");
        // Spreads further apart than the ratio allows are drawn together.
        let far: Vec<(f32, bool)> =
            (0..400).map(|i| if i % 2 == 0 { (-0.8 + 0.6 * normal.next(), true) } else { (0.9 + 0.06 * normal.next(), false) }).collect();
        let fit = Calibration::fit(&far, true).expect("fits");
        assert!(fit.dark.sigma / fit.light.sigma <= CALIBRATION_MAX_RATIO * 1.0001, "{fit:?}");
        // Fitted alike, the two colours take one spread about a threshold
        // midway, the typical margin to each side.
        let alike = Calibration::fit(&labelled, false).expect("fits");
        let mut toward: Vec<f32> = labelled.iter().map(|&(o, d)| if d { -o } else { o }).collect();
        let typical = median(&mut toward);
        assert_eq!((alike.dark.mean, alike.light.mean, alike.threshold), (-typical, typical, 0.0));
        assert_eq!((alike.dark.sigma, alike.light.sigma), (spread_about(&toward, typical), spread_about(&toward, typical)));
        assert_eq!(alike.log_odds(0.3), -alike.log_odds(-0.3));
    }

    #[test]
    fn calibration_moves_the_threshold_to_where_the_colours_cross() {
        // A version 2 symbol read with its ink spread wide and its paper
        // even, against one threshold midway between them: dark modules
        // that strayed past the middle are read light, until the
        // calibration puts the threshold where the two colours cross.
        let (w, h) = (25usize, 25usize);
        let func = FunctionMap::new(SymbolVersion::Qr(2));
        let expected = func.expected_modules().to_vec();
        let mut normal = Normal(0xD0_7712);
        let truth: Vec<bool> = (0..w * h).map(|i| expected[i].unwrap_or_else(|| normal.uniform() < 0.5)).collect();
        let luma: Vec<Option<f32>> =
            truth.iter().map(|&dark| Some(0.5 + 0.35 * if dark { -0.6 + 0.4 * normal.next() } else { 0.9 + 0.15 * normal.next() })).collect();
        let sampled = Sampled { width: w, height: h, luma };
        let params = SampleParams { local: false, band: 0.0, threshold: Some(0.5), ..SampleParams::default() };
        let mut classified = classify(&sampled, &params);
        let wrong = |c: &Classified| (0..w * h).filter(|&i| expected[i].is_none() && c.decided[i] != Module::from_dark(truth[i])).count();
        let before = wrong(&classified);
        // With nothing known of the blur, the colours are fitted alike and
        // the threshold stays where the rule put it.
        let mut unknown_blur = classified.clone();
        assert_eq!(unknown_blur.calibrate(&expected, false).expect("fits").threshold, 0.0);
        assert_eq!(wrong(&unknown_blur), before);
        // The photo is sharp: there is no blur to undo, and the reading,
        // left as it was, is calibrated a colour at a time.
        let calibration = classified.learn(&expected, false).expect("fits");
        assert!(classified.blur.is_some_and(|blur| !blur.is_worth_undoing()) && !classified.equalised, "{:?}", classified.blur);
        assert!(calibration.threshold > 0.2, "{calibration:?}");
        assert!(before >= 8 && wrong(&classified) * 4 <= before, "{before} wrong before, {} after", wrong(&classified));
        // Every module is judged by the calibrated threshold.
        for i in 0..w * h {
            let offset = classified.offset[i];
            assert_eq!(classified.decided[i], Module::from_dark(offset < calibration.threshold));
            assert!((classified.margin[i] - (offset - calibration.threshold).abs()).abs() < 1e-6);
            assert_eq!(classified.reliability[i], calibration.reliability(offset));
        }
    }

    /// Offsets of a version 2 symbol read through `blur` with `noise`,
    /// its data random, the quiet zone light; with the symbol's colours,
    /// dark true, and its fixed modules' colours.
    fn blurred_symbol(own: f32, across: f32, down: f32, falling: f32, rising: f32, noise: f32, seed: u64) -> (Vec<f32>, Vec<bool>, Vec<Option<bool>>) {
        let (w, h) = (25i64, 25i64);
        let func = FunctionMap::new(SymbolVersion::Qr(2));
        let expected = func.expected_modules().to_vec();
        let mut normal = Normal(seed);
        let truth: Vec<bool> = (0..(w * h) as usize).map(|i| expected[i].unwrap_or_else(|| normal.uniform() < 0.5)).collect();
        let colour = |x: i64, y: i64| if x < 0 || y < 0 || x >= w || y >= h || !truth[(y * w + x) as usize] { 1.0 } else { -1.0 };
        let offsets = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                own * colour(x, y)
                    + across * (colour(x - 1, y) + colour(x + 1, y))
                    + down * (colour(x, y - 1) + colour(x, y + 1))
                    + falling * (colour(x - 1, y - 1) + colour(x + 1, y + 1))
                    + rising * (colour(x + 1, y - 1) + colour(x - 1, y + 1))
                    + noise * normal.next()
            })
            .collect();
        (offsets, truth, expected)
    }

    #[test]
    fn a_blur_is_learnt_from_the_fixed_patterns_and_undone() {
        // Spread a quarter of the module's own reach to each side across,
        // less up and down, and a little along one diagonal, as a hand
        // that moved spreads it: a lone module among the other colour
        // reads on the wrong side of the threshold.
        let (offsets, truth, expected) = blurred_symbol(0.5, 0.13, 0.08, 0.05, 0.01, 0.04, 0xB1_0B);
        let read: Vec<Option<f32>> = offsets.iter().copied().map(Some).collect();
        let blur = Blur::fit(&read, 25, &expected, false).expect("blurred enough to fit");
        assert!((blur.own - 0.5).abs() < 0.05, "{blur:?}");
        for (got, want) in [(blur.across, 0.13), (blur.down, 0.08), (blur.falling, 0.05), (blur.rising, 0.01)] {
            assert!((got - want).abs() < 0.03, "{blur:?}");
        }
        assert!(blur.noise < 0.08 && blur.gain > 1.0 + BLUR_MIN_GAIN, "{blur:?}");
        assert!((blur.taps().iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!((blur.spill() - 2.0 * 0.27 / 0.5).abs() < 0.15, "{blur:?}");
        let wrong = |values: &[Option<f32>]| (0..625).filter(|&i| expected[i].is_none() && (values[i].unwrap() < 0.0) != truth[i]).count();
        let undone = blur.undo(&read, 25);
        assert!(wrong(&read) >= 5 && wrong(&undone) * 5 <= wrong(&read), "{} wrong as read, {} undone", wrong(&read), wrong(&undone));
        // A module not read stays so, and a patch of one colour reads as
        // it did.
        let mut holed = read.clone();
        holed[300] = None;
        assert_eq!(blur.undo(&holed, 25)[300], None);
        let solid = vec![Some(0.9f32); 625];
        assert!(blur.undo(&solid, 25).iter().all(|v| (v.unwrap() - 0.9).abs() < 1e-5));
    }

    #[test]
    fn a_sharp_photo_or_a_misplaced_grid_is_left_as_read() {
        // Nothing spread: the blur is learnt, and the filter would gain
        // nothing by undoing it.
        let (offsets, _, expected) = blurred_symbol(0.9, 0.0, 0.0, 0.0, 0.0, 0.06, 0x5AA9);
        let read: Vec<Option<f32>> = offsets.iter().copied().map(Some).collect();
        let blur = Blur::fit(&read, 25, &expected, false).expect("fits");
        assert!(!blur.is_worth_undoing() && blur.spill().abs() < 0.1, "{blur:?}");
        // The labels of another symbol's fixed patterns: a module's own
        // colour says nothing of its reading.
        let (offsets, _, _) = blurred_symbol(0.5, 0.13, 0.08, 0.05, 0.01, 0.04, 0xB1_0B);
        let shifted: Vec<Option<bool>> = (0..625).map(|i| expected[(i + 3) % 625]).collect();
        let read: Vec<Option<f32>> = offsets.iter().copied().map(Some).collect();
        assert_eq!(Blur::fit(&read, 25, &shifted, false), None);
        // Too few modules with all their neighbours labelled: the top four
        // rows hold the upper halves of two finder patterns, twenty-four.
        let few: Vec<Option<bool>> = expected.iter().enumerate().map(|(i, &e)| e.filter(|_| i < 100)).collect();
        assert_eq!(Blur::fit(&read, 25, &few, false), None);
    }

    /// A version 2 symbol drawn `px` pixels a module, its data random,
    /// out of focus by `blur` modules; the map onto it, its colours, dark
    /// true, and its fixed modules' colours.
    fn rendered_symbol(px: usize, blur: f64, seed: u64) -> (Luma, Homography, Vec<bool>, Vec<Option<bool>>) {
        let n = 25;
        let func = FunctionMap::new(SymbolVersion::Qr(2));
        let expected = func.expected_modules().to_vec();
        let mut normal = Normal(seed);
        let truth: Vec<bool> = (0..n * n).map(|i| expected[i].unwrap_or_else(|| normal.uniform() < 0.5)).collect();
        let side = n * px;
        let mut data: Vec<f32> = (0..side * side).map(|i| if truth[(i / side / px) * n + i % side / px] { 0.15 } else { 0.85 }).collect();
        if blur > 0.0 {
            let sigma = blur * px as f64;
            let reach = (3.0 * sigma).ceil() as i64;
            let taps: Vec<f32> = (-reach..=reach).map(|i| (-(i * i) as f64 / (2.0 * sigma * sigma)).exp() as f32).collect();
            let total: f32 = taps.iter().sum();
            for (step, len) in [(1usize, side), (side, side)] {
                let source = data.clone();
                for i in 0..side * side {
                    let along = if step == 1 { i % side } else { i / side } as i64;
                    data[i] = (-reach..=reach)
                        .map(|d| {
                            let at = (along + d).clamp(0, len as i64 - 1);
                            let j = if step == 1 { i - i % side + at as usize } else { at as usize * side + i % side };
                            taps[(d + reach) as usize] * source[j]
                        })
                        .sum::<f32>()
                        / total;
                }
            }
        }
        let corner = side as f64;
        let map = Homography::rect_to_quad(25.0, 25.0, [[0.0, 0.0], [corner, 0.0], [corner, corner], [0.0, corner]]).unwrap();
        (Luma::new(side, side, data), map, truth, expected)
    }

    #[test]
    fn a_blur_learnt_takes_the_place_of_the_guess_at_it() {
        let wrong = |c: &Classified, truth: &[bool], expected: &[Option<bool>]| {
            (0..625).filter(|&i| expected[i].is_none() && c.decided[i] != Module::from_dark(truth[i])).count()
        };
        // Out of focus by half a module: the blur is learnt and undone,
        // and the guess at it is not made besides.
        let (image, map, truth, expected) = rendered_symbol(6, 0.5, 0xF0C5);
        let params = SampleParams::default();
        let (read, sharpening) = read_modules(&image, &map, 25, 25, &params, &expected, false);
        assert!(read.equalised && read.blur.is_some_and(|blur| blur.is_worth_undoing()), "{:?}", read.blur);
        assert!(!sharpening.any() && read.calibration.is_some(), "{sharpening:?}");
        let guessed = classify(&sample_modules_noting(&image, &map, 25, 25, params.kernel, true).0, &params);
        let as_read = classify(&sample_modules_noting(&image, &map, 25, 25, params.kernel, false).0, &params);
        assert!(
            wrong(&read, &truth, &expected) < wrong(&as_read, &truth, &expected),
            "{} wrong equalised, {} as read, {} sharpened by the guess",
            wrong(&read, &truth, &expected),
            wrong(&as_read, &truth, &expected),
            wrong(&guessed, &truth, &expected)
        );
        // With sharpening off the reading is taken as it was read: the
        // blur is learnt, and keeps the colours from being fitted apart.
        let off = SampleParams { sharpen_modules: false, ..params };
        let (left, sharpening) = read_modules(&image, &map, 25, 25, &off, &expected, false);
        assert!(!left.equalised && left.blur.is_some() && !sharpening.any());
        assert_eq!(left.calibration.map(|c| c.threshold), Some(0.0));
        assert_eq!(left.decided, as_read.decided);
        // Sharp, with modules too small for the lens to have left them
        // whole, there is no blur to undo, and the guess sharpens them as
        // it did, its colours fitted alike.
        let (image, map, _, expected) = rendered_symbol(3, 0.0, 0x5_3A9);
        let (read, sharpening) = read_modules(&image, &map, 25, 25, &params, &expected, false);
        assert!(!read.equalised && read.blur.is_none() && sharpening.for_size > 0.0, "{sharpening:?}");
        assert_eq!(read.decided, classify(&sample_modules_noting(&image, &map, 25, 25, params.kernel, true).0, &params).decided);
        assert_eq!(read.calibration.map(|c| c.threshold), Some(0.0));
        // Sharp, with modules large enough, nothing is sharpened, and the
        // colours are fitted apart.
        let (image, map, _, expected) = rendered_symbol(8, 0.0, 0x5_3A9);
        let (read, sharpening) = read_modules(&image, &map, 25, 25, &params, &expected, false);
        assert!(!read.equalised && read.blur.is_some() && !sharpening.any(), "{:?} {sharpening:?}", read.blur);
    }

    #[test]
    fn a_classified_reading_is_equalised_before_it_is_calibrated() {
        let (offsets, truth, expected) = blurred_symbol(0.5, 0.13, 0.08, 0.05, 0.01, 0.04, 0xE0_A1);
        let sampled = Sampled { width: 25, height: 25, luma: offsets.iter().map(|o| Some(0.5 + 0.3 * o)).collect() };
        let params = SampleParams { local: false, threshold: Some(0.5), band: 0.1, ..SampleParams::default() };
        let wrong = |c: &Classified| (0..625).filter(|&i| expected[i].is_none() && c.decided[i] != Module::from_dark(truth[i])).count();
        let plain = classify(&sampled, &params);
        let mut classified = plain.clone();
        let blur = classified.equalise(&expected, false).expect("equalised");
        assert_eq!(classified.blur, Some(blur));
        assert!(wrong(&classified) * 5 <= wrong(&plain), "{} wrong as read, {} equalised", wrong(&plain), wrong(&classified));
        for i in 0..625 {
            let offset = classified.offset[i];
            assert_eq!(classified.decided[i], Module::from_dark(offset < 0.0));
            assert_eq!(classified.margin[i], offset.abs());
            assert_eq!(classified.modules[i], if offset.abs() < 0.1 { Module::Unknown } else { classified.decided[i] });
        }
        // Learning is the two in turn.
        let mut learnt = plain.clone();
        let calibration = learnt.learn(&expected, false).expect("calibrated");
        classified.calibrate(&expected, false);
        assert_eq!(learnt, classified);
        assert_eq!(learnt.calibration, Some(calibration));
        // The fixed patterns, equalised, spread as the data does.
        let spread = |c: &Classified, fixed: bool| {
            let labels: Vec<(f32, bool)> =
                (0..625).filter(|&i| expected[i].is_some() == fixed).map(|i| (c.offset[i], truth[i])).collect();
            let fit = Calibration::fit(&labels, true).unwrap();
            (fit.dark.sigma + fit.light.sigma) / 2.0
        };
        let (before, after) = ((spread(&plain, true), spread(&plain, false)), (spread(&learnt, true), spread(&learnt, false)));
        assert!(after.1 / after.0 < before.1 / before.0, "data against fixed: {before:?} as read, {after:?} equalised");
    }
}
