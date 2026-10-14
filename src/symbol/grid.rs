//! The tri-state module grid the whole pipeline reads from.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Module {
    Dark,
    Light,
    #[default]
    Unknown,
}

impl Module {
    pub fn from_dark(dark: bool) -> Self {
        if dark { Module::Dark } else { Module::Light }
    }

    /// `Some(true)` for dark, `Some(false)` for light.
    pub fn as_dark(self) -> Option<bool> {
        match self {
            Module::Dark => Some(true),
            Module::Light => Some(false),
            Module::Unknown => None,
        }
    }

    pub fn inverted(self) -> Self {
        match self {
            Module::Dark => Module::Light,
            Module::Light => Module::Dark,
            Module::Unknown => Module::Unknown,
        }
    }
}

/// Reliability of a module that was painted by hand, filled from the
/// standard, or settled by error correction. Sampled modules score below
/// it, and so do those painted as a guess.
pub const CERTAIN: u8 = 255;

/// A reflection of the grid: what a symbol looks like seen from the back.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mirror {
    /// Swap rows and columns: the one reflection that leaves a square
    /// symbol's finder patterns where they were.
    Transpose,
    /// Reflect left to right.
    Horizontal,
    /// Reflect top to bottom.
    Vertical,
}

impl Mirror {
    pub fn label(self) -> &'static str {
        match self {
            Mirror::Transpose => "Mirror (swap rows and columns)",
            Mirror::Horizontal => "Flip left to right",
            Mirror::Vertical => "Flip top to bottom",
        }
    }
}

/// A symmetry of the grid's rectangle other than leaving it alone: the
/// ways a symbol can have been painted the wrong way up or from the back.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Symmetry {
    RotateCw,
    Rotate180,
    RotateCcw,
    /// Swap rows and columns.
    Transpose,
    /// Swap rows and columns and turn half round.
    AntiTranspose,
    FlipHorizontal,
    FlipVertical,
}

impl Symmetry {
    /// Every symmetry of a square.
    pub const SQUARE: [Symmetry; 7] = [
        Symmetry::RotateCw,
        Symmetry::Rotate180,
        Symmetry::RotateCcw,
        Symmetry::Transpose,
        Symmetry::AntiTranspose,
        Symmetry::FlipHorizontal,
        Symmetry::FlipVertical,
    ];
    /// Those that keep a rectangle's shape.
    pub const RECTANGULAR: [Symmetry; 3] = [Symmetry::Rotate180, Symmetry::FlipHorizontal, Symmetry::FlipVertical];

    /// The symmetries a symbol of the given shape can have been painted in.
    pub fn of_shape(square: bool) -> &'static [Symmetry] {
        if square { &Self::SQUARE } else { &Self::RECTANGULAR }
    }

    /// A reflection shows the symbol as from the back of the paper.
    pub fn is_reflection(self) -> bool {
        matches!(self, Symmetry::Transpose | Symmetry::AntiTranspose | Symmetry::FlipHorizontal | Symmetry::FlipVertical)
    }

    pub fn label(self) -> &'static str {
        match self {
            Symmetry::RotateCw => "Rotate 90° clockwise",
            Symmetry::Rotate180 => "Rotate 180°",
            Symmetry::RotateCcw => "Rotate 90° anticlockwise",
            Symmetry::Transpose => "Mirror (swap rows and columns)",
            Symmetry::AntiTranspose => "Mirror across the other diagonal",
            Symmetry::FlipHorizontal => "Flip left to right",
            Symmetry::FlipVertical => "Flip top to bottom",
        }
    }

    /// Where a point of a `width` × `height` grid lies in the grid turned
    /// or reflected by this symmetry, as [`ModuleGrid::symmetric`] moves
    /// its modules. Points are in module coordinates, the origin at the
    /// top-left corner of the grid and module (x, y) covering x..x + 1 by
    /// y..y + 1, so the centre of a module goes to the centre of the
    /// module its colour goes to, and a corner of the grid to a corner.
    pub fn map_point(self, width: usize, height: usize, point: [f64; 2]) -> [f64; 2] {
        let (w, h) = (width as f64, height as f64);
        let [x, y] = point;
        match self {
            Symmetry::RotateCw => [h - y, x],
            Symmetry::Rotate180 => [w - x, h - y],
            Symmetry::RotateCcw => [y, w - x],
            Symmetry::Transpose => [y, x],
            Symmetry::AntiTranspose => [h - y, w - x],
            Symmetry::FlipHorizontal => [w - x, y],
            Symmetry::FlipVertical => [x, h - y],
        }
    }

    /// The outline of a grid relabelled for the grid turned or reflected
    /// by this symmetry. `corners` is whatever marks the four corners,
    /// top-left, top-right, bottom-right, bottom-left: their places in a
    /// photo, say. Each corner of the result carries the mark of the
    /// corner the symmetry brings there, so a map from the grid to the
    /// photo laid through the result shows every module where the map
    /// through `corners` showed it before it moved.
    pub fn map_corners<T: Copy>(self, corners: [T; 4]) -> [T; 4] {
        // Where each corner of the unit square goes says where each
        // corner of any grid goes.
        const UNIT: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let mut out = corners;
        for (from, &at) in UNIT.iter().enumerate() {
            let to = self.map_point(1, 1, at);
            let index = UNIT.iter().position(|&corner| corner == to).expect("a symmetry takes corners to corners");
            out[index] = corners[from];
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleGrid {
    width: usize,
    height: usize,
    cells: Vec<Module>,
    /// How sure each module's colour is: [`CERTAIN`] unless it was sampled
    /// from a photo, in which case eight times the bits of likelihood its
    /// reading carries less eight, on a scale of 0..=254
    /// ([`crate::imaging::sample::Calibration::reliability`]), or painted
    /// as a guess, which is 0. `None` until something is sampled or
    /// guessed.
    reliability: Option<Vec<u8>>,
}

impl ModuleGrid {
    /// A grid of unknown modules.
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, cells: vec![Module::Unknown; width * height], reliability: None }
    }

    pub fn from_bools(width: usize, height: usize, dark: &[bool]) -> Self {
        assert_eq!(dark.len(), width * height);
        Self { width, height, cells: dark.iter().map(|&d| Module::from_dark(d)).collect(), reliability: None }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> Module {
        self.cells[y * self.width + x]
    }

    /// Set a module as certain.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, module: Module) {
        let i = y * self.width + x;
        self.cells[i] = module;
        if let Some(reliability) = &mut self.reliability {
            reliability[i] = CERTAIN;
        }
    }

    /// Set a module read from a photo, with how far its reading is to be
    /// believed on a scale of 0..=254.
    pub fn set_sampled(&mut self, x: usize, y: usize, module: Module, reliability: u8) {
        let i = y * self.width + x;
        self.cells[i] = module;
        self.reliability.get_or_insert_with(|| vec![CERTAIN; self.width * self.height])[i] =
            reliability.min(CERTAIN - 1);
    }

    /// Set a module painted as a guess: no surer than one sampled on its
    /// threshold, and so among the first to be doubted.
    pub fn set_guessed(&mut self, x: usize, y: usize, module: Module) {
        self.set_sampled(x, y, module, 0);
    }

    /// How sure the module's colour is; [`CERTAIN`] unless it was sampled
    /// or guessed.
    #[inline]
    pub fn reliability(&self, x: usize, y: usize) -> u8 {
        self.reliability.as_ref().map_or(CERTAIN, |r| r[y * self.width + x])
    }

    /// True once any module has been sampled, or painted as a guess,
    /// rather than painted as certain.
    pub fn has_reliability(&self) -> bool {
        self.reliability.is_some()
    }

    pub fn cells(&self) -> &[Module] {
        &self.cells
    }

    pub fn cells_mut(&mut self) -> &mut [Module] {
        &mut self.cells
    }

    pub fn count(&self, module: Module) -> usize {
        self.cells.iter().filter(|&&m| m == module).count()
    }

    /// Parse rows of `1`/`#` (dark), `0`/`.` (light) and `?` (unknown).
    /// Blank lines and lines starting with `;` are skipped.
    pub fn from_text(text: &str) -> Result<Self, String> {
        let mut cells = Vec::new();
        let mut width = 0;
        let mut height = 0;
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with(';')) {
            let before = cells.len();
            for ch in line.chars() {
                cells.push(match ch {
                    '1' | '#' => Module::Dark,
                    '0' | '.' => Module::Light,
                    '?' => Module::Unknown,
                    other => return Err(format!("unexpected character {other:?} in row {height}")),
                });
            }
            let row = cells.len() - before;
            if height == 0 {
                width = row;
            } else if row != width {
                return Err(format!("row {height} has {row} modules, expected {width}"));
            }
            height += 1;
        }
        if width == 0 {
            return Err("empty grid".into());
        }
        Ok(Self { width, height, cells, reliability: None })
    }

    pub fn to_text(&self) -> String {
        let mut out = String::with_capacity((self.width + 1) * self.height);
        for row in self.cells.chunks(self.width) {
            out.extend(row.iter().map(|m| match m {
                Module::Dark => '1',
                Module::Light => '0',
                Module::Unknown => '?',
            }));
            out.push('\n');
        }
        out
    }

    fn remap(&self, width: usize, height: usize, source: impl Fn(usize, usize) -> (usize, usize)) -> Self {
        let mut out = Self::new(width, height);
        if self.reliability.is_some() {
            out.reliability = Some(vec![CERTAIN; width * height]);
        }
        for y in 0..height {
            for x in 0..width {
                let (sx, sy) = source(x, y);
                out.cells[y * width + x] = self.get(sx, sy);
                if let Some(reliability) = &mut out.reliability {
                    reliability[y * width + x] = self.reliability(sx, sy);
                }
            }
        }
        out
    }

    /// Swap rows and columns: what the symbol looks like read from the back
    /// and turned so the finder lands top-left again.
    pub fn transposed(&self) -> Self {
        self.remap(self.height, self.width, |x, y| (y, x))
    }

    pub fn flipped_horizontal(&self) -> Self {
        self.remap(self.width, self.height, |x, y| (self.width - 1 - x, y))
    }

    pub fn flipped_vertical(&self) -> Self {
        self.remap(self.width, self.height, |x, y| (x, self.height - 1 - y))
    }

    pub fn mirrored(&self, mirror: Mirror) -> Self {
        match mirror {
            Mirror::Transpose => self.transposed(),
            Mirror::Horizontal => self.flipped_horizontal(),
            Mirror::Vertical => self.flipped_vertical(),
        }
    }

    pub fn rotated_cw(&self) -> Self {
        self.remap(self.height, self.width, |x, y| (y, self.height - 1 - x))
    }

    pub fn rotated_180(&self) -> Self {
        self.remap(self.width, self.height, |x, y| (self.width - 1 - x, self.height - 1 - y))
    }

    pub fn rotated_ccw(&self) -> Self {
        self.remap(self.height, self.width, |x, y| (self.width - 1 - y, x))
    }

    /// Reflect across the other diagonal: the transpose turned half round.
    pub fn anti_transposed(&self) -> Self {
        self.remap(self.height, self.width, |x, y| (self.width - 1 - y, self.height - 1 - x))
    }

    /// The grid under one of the symmetries of its rectangle.
    pub fn symmetric(&self, symmetry: Symmetry) -> Self {
        match symmetry {
            Symmetry::RotateCw => self.rotated_cw(),
            Symmetry::Rotate180 => self.rotated_180(),
            Symmetry::RotateCcw => self.rotated_ccw(),
            Symmetry::Transpose => self.transposed(),
            Symmetry::AntiTranspose => self.anti_transposed(),
            Symmetry::FlipHorizontal => self.flipped_horizontal(),
            Symmetry::FlipVertical => self.flipped_vertical(),
        }
    }

    pub fn inverted(&self) -> Self {
        Self {
            width: self.width,
            height: self.height,
            cells: self.cells.iter().map(|m| m.inverted()).collect(),
            reliability: self.reliability.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_round_trip() {
        let grid = ModuleGrid::from_text("10?\n; comment\n\n#.1\n").unwrap();
        assert_eq!((grid.width(), grid.height()), (3, 2));
        assert_eq!(grid.get(2, 0), Module::Unknown);
        assert_eq!(grid.get(0, 1), Module::Dark);
        assert_eq!(grid.to_text(), "10?\n101\n");
        assert!(ModuleGrid::from_text("10\n1").is_err());
        assert!(ModuleGrid::from_text("1x").is_err());
    }

    #[test]
    fn transforms() {
        let grid = ModuleGrid::from_text("100\n0?0").unwrap();
        assert_eq!(grid.transposed().to_text(), "10\n0?\n00\n");
        assert_eq!(grid.flipped_horizontal().to_text(), "001\n0?0\n");
        assert_eq!(grid.flipped_vertical().to_text(), "0?0\n100\n");
        assert_eq!(grid.mirrored(Mirror::Transpose), grid.transposed());
        assert_eq!(grid.mirrored(Mirror::Horizontal), grid.flipped_horizontal());
        assert_eq!(grid.mirrored(Mirror::Vertical).mirrored(Mirror::Vertical), grid);
        assert_eq!(grid.rotated_cw().to_text(), "01\n?0\n00\n");
        assert_eq!(grid.rotated_180().to_text(), "0?0\n001\n");
        assert_eq!(grid.rotated_cw().rotated_cw(), grid.rotated_180());
        assert_eq!(grid.inverted().to_text(), "011\n1?1\n");
        assert_eq!(grid.rotated_ccw(), grid.rotated_cw().rotated_180());
        assert_eq!(grid.rotated_ccw().rotated_cw(), grid);
        assert_eq!(grid.anti_transposed(), grid.transposed().rotated_180());
        for symmetry in Symmetry::SQUARE {
            let once = grid.symmetric(symmetry);
            // Rotations by a quarter turn are the only symmetries that
            // are not their own inverse.
            let back = match symmetry {
                Symmetry::RotateCw => once.rotated_ccw(),
                Symmetry::RotateCcw => once.rotated_cw(),
                other => once.symmetric(other),
            };
            assert_eq!(back, grid, "{symmetry:?}");
            let reflection = !matches!(symmetry, Symmetry::RotateCw | Symmetry::Rotate180 | Symmetry::RotateCcw);
            assert_eq!(symmetry.is_reflection(), reflection, "{symmetry:?}");
        }
        assert_eq!(Symmetry::of_shape(false).len(), 3);
    }

    #[test]
    fn symmetries_move_points_and_corners_as_they_move_modules() {
        let (w, h) = (5, 3);
        for symmetry in Symmetry::SQUARE {
            // One dark module at a time: its centre lands on the centre
            // of the one dark module of the grid turned.
            for y in 0..h {
                for x in 0..w {
                    let mut grid = ModuleGrid::from_bools(w, h, &vec![false; w * h]);
                    grid.set(x, y, Module::Dark);
                    let turned = grid.symmetric(symmetry);
                    let [px, py] = symmetry.map_point(w, h, [x as f64 + 0.5, y as f64 + 0.5]);
                    assert_eq!((px.fract(), py.fract()), (0.5, 0.5), "{symmetry:?} ({x}, {y})");
                    assert_eq!(turned.get(px as usize, py as usize), Module::Dark, "{symmetry:?} ({x}, {y})");
                    assert_eq!(turned.count(Module::Dark), 1);
                }
            }
            // Each corner of the relabelled outline is the corner the
            // symmetry brings to that place.
            let turned = ModuleGrid::new(w, h).symmetric(symmetry);
            let (tw, th) = (turned.width() as f64, turned.height() as f64);
            let outline = [[0.0, 0.0], [w as f64, 0.0], [w as f64, h as f64], [0.0, h as f64]];
            let relabelled = symmetry.map_corners(outline);
            for (corner, at) in relabelled.into_iter().zip([[0.0, 0.0], [tw, 0.0], [tw, th], [0.0, th]]) {
                assert_eq!(symmetry.map_point(w, h, corner), at, "{symmetry:?}");
            }
        }
        // A quarter turn clockwise takes the top-left corner to the top
        // right; the mirror that keeps the finders in place swaps the
        // other two; a flip swaps left with right.
        assert_eq!(Symmetry::RotateCw.map_corners(["tl", "tr", "br", "bl"]), ["bl", "tl", "tr", "br"]);
        assert_eq!(Symmetry::Transpose.map_corners(["tl", "tr", "br", "bl"]), ["tl", "bl", "br", "tr"]);
        assert_eq!(Symmetry::FlipHorizontal.map_corners(["tl", "tr", "br", "bl"]), ["tr", "tl", "bl", "br"]);
    }

    #[test]
    fn reliability_follows_modules() {
        let mut grid = ModuleGrid::from_text("100\n0?0").unwrap();
        assert!(!grid.has_reliability());
        assert_eq!(grid.reliability(2, 1), CERTAIN);
        grid.set_sampled(2, 1, Module::Dark, 40);
        grid.set_sampled(0, 0, Module::Light, 255);
        assert!(grid.has_reliability());
        assert_eq!(grid.reliability(2, 1), 40);
        assert_eq!(grid.reliability(0, 0), CERTAIN - 1, "sampled modules are never certain");
        assert_eq!(grid.reliability(1, 1), CERTAIN);
        // Painting over a sampled module makes it certain again.
        grid.set(2, 1, Module::Light);
        assert_eq!(grid.reliability(2, 1), CERTAIN);
        // A guess is as unsure as a module can be, until it is painted
        // over as certain.
        grid.set_guessed(2, 1, Module::Dark);
        assert_eq!((grid.get(2, 1), grid.reliability(2, 1)), (Module::Dark, 0));
        grid.set(2, 1, Module::Dark);
        assert_eq!(grid.reliability(2, 1), CERTAIN);
        grid.set_sampled(2, 1, Module::Dark, 40);
        assert_eq!(grid.rotated_cw().reliability(0, 2), 40);
        assert_eq!(grid.transposed().reliability(1, 2), 40);
        assert_eq!(grid.inverted().reliability(2, 1), 40);
        assert_ne!(grid.inverted(), grid);
    }
}
