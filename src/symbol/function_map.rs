//! Version-dependent geometry: the role of every module, the colour of the
//! fixed ones, where format and version bits live, and the order in which
//! data modules are filled.

use super::grid::{Module, ModuleGrid};
use super::{Pos, SymbolVersion, tables};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Finder,
    Separator,
    Timing,
    Alignment,
    /// rMQR finder sub-pattern, bottom right.
    FinderSub,
    /// rMQR corner finder patterns, top right and bottom left.
    CornerFinder,
    /// The single always-dark module beside the QR bottom-left finder.
    DarkModule,
    /// QR Code Model 1 extension patterns: a dark line along the right
    /// and bottom edges with a light line inside it, in eight-module
    /// pieces where the codeword placement leaves room, and the fixed
    /// two-by-two corner at bottom right.
    Extension,
    Format { copy: u8, bit: u8 },
    Version { copy: u8, bit: u8 },
    /// Index into [`FunctionMap::data_path`].
    Data { path: u32 },
    /// Trailing data-path modules that belong to no codeword.
    Remainder { path: u32 },
}

impl Role {
    /// Fixed patterns carry no information; their colour is known in advance.
    pub fn is_fixed(self) -> bool {
        matches!(
            self,
            Role::Finder
                | Role::Separator
                | Role::Timing
                | Role::Alignment
                | Role::FinderSub
                | Role::CornerFinder
                | Role::DarkModule
                | Role::Extension
        )
    }

    pub fn label(self) -> &'static str {
        match self {
            Role::Finder => "Finder",
            Role::Separator => "Separator",
            Role::Timing => "Timing",
            Role::Alignment => "Alignment",
            Role::FinderSub => "Finder sub-pattern",
            Role::CornerFinder => "Corner finder",
            Role::DarkModule => "Dark module",
            Role::Extension => "Extension pattern",
            Role::Format { .. } => "Format info",
            Role::Version { .. } => "Version info",
            Role::Data { .. } => "Data",
            Role::Remainder { .. } => "Remainder",
        }
    }
}

/// Fixed modules that must be known before their polarity is judged.
pub const POLARITY_MIN_KNOWN: usize = 8;

/// True when fixed patterns read as the inverse of what the standard
/// specifies, from `(matching, known)` as [`FunctionMap::fixed_agreement`]
/// returns it: three quarters or more of the known fixed modules have the
/// wrong colour. A light-on-dark symbol taken as is, or a photo sampled
/// with the wrong polarity, gets almost every one wrong. Misalignment and
/// the wrong version leave the agreement near one half, and damage takes
/// it down from the top, so neither gets this far.
pub fn looks_inverted((matching, known): (usize, usize)) -> bool {
    known >= POLARITY_MIN_KNOWN && (known - matching) * 4 >= known * 3
}

/// One copy of the format or version information.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InfoCopy {
    /// Module for each bit, least significant first.
    pub bits: Vec<Pos>,
    /// XOR applied to the codeword before it was placed.
    pub xor: u32,
}

#[derive(Clone, Debug)]
pub struct FunctionMap {
    pub version: SymbolVersion,
    pub width: usize,
    pub height: usize,
    roles: Vec<Role>,
    expected: Vec<Option<bool>>,
    /// Data and remainder modules in placement order.
    pub data_path: Vec<Pos>,
    pub format: Vec<InfoCopy>,
    pub version_info: Vec<InfoCopy>,
    /// Trailing entries of `data_path` that carry no codeword bit.
    pub remainder_bits: usize,
}

struct Builder {
    width: usize,
    height: usize,
    roles: Vec<Option<Role>>,
    expected: Vec<Option<bool>>,
}

impl Builder {
    fn new(width: usize, height: usize) -> Self {
        Self { width, height, roles: vec![None; width * height], expected: vec![None; width * height] }
    }

    fn set(&mut self, x: usize, y: usize, role: Role, dark: bool) {
        let i = y * self.width + x;
        self.roles[i] = Some(role);
        self.expected[i] = Some(dark);
    }

    fn set_if_free(&mut self, x: usize, y: usize, role: Role, dark: bool) {
        if self.roles[y * self.width + x].is_none() {
            self.set(x, y, role, dark);
        }
    }

    fn is_free(&self, x: usize, y: usize) -> bool {
        self.roles[y * self.width + x].is_none()
    }

    /// 7×7 finder with its top-left corner at (x0, y0).
    fn finder(&mut self, x0: usize, y0: usize) {
        for dy in 0..7usize {
            for dx in 0..7usize {
                let ring = dx.abs_diff(3).max(dy.abs_diff(3));
                self.set(x0 + dx, y0 + dy, Role::Finder, ring != 2);
            }
        }
    }

    fn info(&mut self, copies: &[InfoCopy], role: impl Fn(u8, u8) -> Role) {
        for (c, copy) in copies.iter().enumerate() {
            for (b, pos) in copy.bits.iter().enumerate() {
                let i = pos.y as usize * self.width + pos.x as usize;
                self.roles[i] = Some(role(c as u8, b as u8));
                self.expected[i] = None;
            }
        }
    }

    /// Walk column pairs right to left, alternating up and down, collecting
    /// every module that has no role yet.
    fn data_path(&self, right_columns: impl Iterator<Item = usize>) -> Vec<Pos> {
        let mut path = Vec::new();
        let mut upward = true;
        for right in right_columns {
            for step in 0..self.height {
                let y = if upward { self.height - 1 - step } else { step };
                for x in [right, right - 1] {
                    if self.is_free(x, y) {
                        path.push(Pos::new(x, y));
                    }
                }
            }
            upward = !upward;
        }
        path
    }

    fn finish(
        mut self,
        version: SymbolVersion,
        data_path: Vec<Pos>,
        format: Vec<InfoCopy>,
        version_info: Vec<InfoCopy>,
        codeword_bits: usize,
    ) -> FunctionMap {
        assert!(codeword_bits <= data_path.len(), "{version}: path shorter than codewords");
        for (i, pos) in data_path.iter().enumerate() {
            let role = if i < codeword_bits {
                Role::Data { path: i as u32 }
            } else {
                Role::Remainder { path: i as u32 }
            };
            self.roles[pos.y as usize * self.width + pos.x as usize] = Some(role);
        }
        FunctionMap {
            version,
            width: self.width,
            height: self.height,
            roles: self.roles.into_iter().map(|r| r.expect("every module has a role")).collect(),
            expected: self.expected,
            remainder_bits: data_path.len() - codeword_bits,
            data_path,
            format,
            version_info,
        }
    }
}

impl FunctionMap {
    pub fn new(version: SymbolVersion) -> Self {
        assert!(version.is_valid(), "invalid version {version:?}");
        match version {
            SymbolVersion::Qr(v) => Self::qr(v),
            SymbolVersion::Micro(v) => Self::micro(v),
            SymbolVersion::Rmqr(v) => Self::rmqr(v),
            SymbolVersion::Model1(v) => Self::model1(v),
        }
    }

    #[inline]
    pub fn role(&self, x: usize, y: usize) -> Role {
        self.roles[y * self.width + x]
    }

    /// Colour of a fixed module: `Some(true)` is dark.
    #[inline]
    pub fn expected(&self, x: usize, y: usize) -> Option<bool> {
        self.expected[y * self.width + x]
    }

    /// The colour of every fixed module, row-major over the symbol, and
    /// `None` for every other.
    pub fn expected_modules(&self) -> &[Option<bool>] {
        &self.expected
    }

    /// Number of modules that carry codeword bits.
    pub fn codeword_bits(&self) -> usize {
        self.data_path.len() - self.remainder_bits
    }

    /// A grid with fixed patterns filled in and everything else unknown.
    pub fn blank_grid(&self) -> ModuleGrid {
        let mut grid = ModuleGrid::new(self.width, self.height);
        self.fill_fixed(&mut grid);
        grid
    }

    /// Overwrite every fixed module with its specified colour.
    pub fn fill_fixed(&self, grid: &mut ModuleGrid) {
        for y in 0..self.height {
            for x in 0..self.width {
                if let Some(dark) = self.expected(x, y) {
                    grid.set(x, y, Module::from_dark(dark));
                }
            }
        }
    }

    /// (matching, known) over fixed modules: how well a grid agrees with the
    /// patterns this version must have.
    pub fn fixed_agreement(&self, grid: &ModuleGrid) -> (usize, usize) {
        let (mut matching, mut known) = (0, 0);
        for y in 0..self.height {
            for x in 0..self.width {
                if let (Some(want), Some(got)) = (self.expected(x, y), grid.get(x, y).as_dark()) {
                    known += 1;
                    matching += usize::from(want == got);
                }
            }
        }
        (matching, known)
    }

    /// True when the grid's fixed patterns read as the inverse of what the
    /// standard specifies. See [`looks_inverted`].
    pub fn is_inverted(&self, grid: &ModuleGrid) -> bool {
        looks_inverted(self.fixed_agreement(grid))
    }

    /// What both QR Code models share: three finder patterns with their
    /// separators, the timing patterns, the dark module and two copies of
    /// the format information, painted with `format_xor`.
    fn qr_frame(b: &mut Builder, n: usize, format_xor: u32) -> Vec<InfoCopy> {
        for i in 8..n - 8 {
            b.set(i, 6, Role::Timing, i % 2 == 0);
            b.set(6, i, Role::Timing, i % 2 == 0);
        }

        for (x0, y0) in [(0, 0), (n - 7, 0), (0, n - 7)] {
            b.finder(x0, y0);
        }
        for i in 0..8 {
            for (x, y) in [(7, i), (i, 7), (n - 8, i), (n - 8 + i, 7), (7, n - 8 + i), (i, n - 8)] {
                b.set(x, y, Role::Separator, false);
            }
        }

        b.set(8, n - 8, Role::DarkModule, true);

        let mut first = Vec::with_capacity(15);
        let mut second = Vec::with_capacity(15);
        for i in 0..15usize {
            first.push(match i {
                0..=5 => Pos::new(8, i),
                6 => Pos::new(8, 7),
                7 => Pos::new(8, 8),
                8 => Pos::new(7, 8),
                _ => Pos::new(14 - i, 8),
            });
            second.push(if i < 8 { Pos::new(n - 1 - i, 8) } else { Pos::new(8, n - 15 + i) });
        }
        let format = vec![InfoCopy { bits: first, xor: format_xor }, InfoCopy { bits: second, xor: format_xor }];
        b.info(&format, |copy, bit| Role::Format { copy, bit });
        format
    }

    fn qr(v: u8) -> Self {
        let version = SymbolVersion::Qr(v);
        let n = version.width();
        let mut b = Builder::new(n, n);
        let format = Self::qr_frame(&mut b, n, 0x5412);

        let centers = tables::qr_alignment_centers(v);
        let last = centers.len().saturating_sub(1);
        for (i, &cx) in centers.iter().enumerate() {
            for (j, &cy) in centers.iter().enumerate() {
                if (i == 0 && j == 0) || (i == 0 && j == last) || (i == last && j == 0) {
                    continue;
                }
                for dy in 0..5usize {
                    for dx in 0..5usize {
                        let ring = dx.abs_diff(2).max(dy.abs_diff(2));
                        b.set(cx + dx - 2, cy + dy - 2, Role::Alignment, ring != 1);
                    }
                }
            }
        }

        let mut version_info = Vec::new();
        if v >= 7 {
            let top_right: Vec<Pos> = (0..18).map(|i| Pos::new(n - 11 + i % 3, i / 3)).collect();
            let bottom_left: Vec<Pos> = (0..18).map(|i| Pos::new(i / 3, n - 11 + i % 3)).collect();
            version_info =
                vec![InfoCopy { bits: top_right, xor: 0 }, InfoCopy { bits: bottom_left, xor: 0 }];
            b.info(&version_info, |copy, bit| Role::Version { copy, bit });
        }

        // Column pairs from the right edge, stepping over the timing column.
        let rights = (1..n).rev().step_by(2).map(|x| if x <= 6 { x - 1 } else { x });
        let path = b.data_path(rights);
        b.finish(version, path, format, version_info, 8 * tables::total_codewords(version))
    }

    /// QR Code Model 1: the Model 2 frame without alignment patterns or
    /// version information, the format word under Model 1's XOR mask
    /// (0x2825), extension patterns, and codewords placed as blocks of two
    /// by four modules up the right and left sides and four by two across
    /// the middle, in the order zxing-cpp's `ReadQRCodewordsModel1` walks
    /// them (QRBitMatrixParser.cpp), the only open reader of the model.
    /// The walk, and the fixed modules it steps over, were checked against
    /// zxing-cpp's sample symbol, which decodes with every syndrome zero
    /// only with the corner left out of its first codeword.
    fn model1(v: u8) -> Self {
        let version = SymbolVersion::Model1(v);
        let n = version.width();
        let mut b = Builder::new(n, n);
        let format = Self::qr_frame(&mut b, n, 0x2825);

        // The corner: the first codeword's high nibble, which is no data.
        // In the sample the corner module is dark and its three
        // neighbours light.
        for (x, y) in [(n - 1, n - 1), (n - 2, n - 1), (n - 1, n - 2), (n - 2, n - 2)] {
            b.set(x, y, Role::Extension, (x, y) == (n - 1, n - 1));
        }

        // Codewords as the reader walks them: bit 7 first, bit `k` of a
        // vertical block at (x - k % 2, y - k / 2) and of a horizontal one
        // at (x - k % 4, y - k / 4), from the block's bottom-right corner
        // (x, y). The extension patterns are the blocks the walk skips, a
        // dark line along the edge with a light line inside.
        let mut path = Vec::new();
        let vertical = |x: usize, y: usize| (0..8usize).map(move |k| Pos::new(x - k % 2, y - k / 2));
        let horizontal = |x: usize, y: usize| (0..8usize).map(move |k| Pos::new(x - k % 4, y - k / 4));
        let columns = n / 4 + 3;
        for j in 0..columns {
            if j <= 1 {
                // Up the right side, two columns at a time, below the
                // top-right finder.
                let rows = (n - 8) / 4;
                for i in 0..rows {
                    let (x, y) = (n - 1 - 2 * j, n - 1 - 4 * i);
                    if j == 0 && i % 2 == 0 && i > 0 && i < rows - 1 {
                        for pos in vertical(x, y) {
                            b.set(pos.x as usize, pos.y as usize, Role::Extension, pos.x as usize == n - 1);
                        }
                        continue;
                    }
                    // The corner block is the first codeword, four
                    // modules of it fixed.
                    path.extend(vertical(x, y).filter(|p| b.is_free(p.x as usize, p.y as usize)));
                }
            } else if columns - j <= 4 {
                // Up the left side between the finders, stepping over the
                // timing column.
                let rows = (n - 16) / 4;
                let x = (columns - j - 1) * 2 + 1 + usize::from(columns - j == 4);
                for i in 0..rows {
                    path.extend(vertical(x, n - 9 - 4 * i));
                }
            } else {
                // Across the middle, four columns at a time, bottom to top,
                // stepping over the timing row.
                let rows = n / 2;
                let x = n - 5 - 4 * (j - 2);
                for i in 0..rows {
                    let y = n - 1 - 2 * i - usize::from(i >= rows - 3);
                    if j == 2 && i >= rows - 4 {
                        // The top-right finder and the format bits beside it.
                        continue;
                    }
                    if i == 0 && j % 2 == 1 && j + 1 != columns - 4 {
                        for pos in horizontal(x, y) {
                            b.set(pos.x as usize, pos.y as usize, Role::Extension, pos.y as usize == n - 1);
                        }
                        continue;
                    }
                    path.extend(horizontal(x, y));
                }
            }
        }
        let bits = path.len();
        b.finish(version, path, format, Vec::new(), bits)
    }

    fn micro(v: u8) -> Self {
        let version = SymbolVersion::Micro(v);
        let n = version.width();
        let mut b = Builder::new(n, n);

        b.finder(0, 0);
        for i in 0..8 {
            b.set(7, i, Role::Separator, false);
            b.set(i, 7, Role::Separator, false);
        }
        for i in 8..n {
            b.set(i, 0, Role::Timing, i % 2 == 0);
            b.set(0, i, Role::Timing, i % 2 == 0);
        }

        let bits: Vec<Pos> =
            (0..15usize).map(|i| if i < 8 { Pos::new(8, 1 + i) } else { Pos::new(15 - i, 8) }).collect();
        let format = vec![InfoCopy { bits, xor: 0x4445 }];
        b.info(&format, |copy, bit| Role::Format { copy, bit });

        let path = b.data_path((1..n).rev().step_by(2));
        // M1 and M3 end their data with a four-bit codeword, so the path is
        // not a whole number of bytes; every module still carries a bit.
        let bits = path.len();
        b.finish(version, path, format, Vec::new(), bits)
    }

    fn rmqr(v: u8) -> Self {
        let version = SymbolVersion::Rmqr(v);
        let (w, h) = (version.width(), version.height());
        let mut b = Builder::new(w, h);

        b.finder(0, 0);
        for y in 0..h.min(8) {
            b.set(7, y, Role::Separator, false);
        }
        if h > 7 {
            for x in 0..8 {
                b.set(x, 7, Role::Separator, false);
            }
        }

        for dy in 0..5usize {
            for dx in 0..5usize {
                let dark = dx == 0 || dx == 4 || dy == 0 || dy == 4 || (dx == 2 && dy == 2);
                b.set(w - 5 + dx, h - 5 + dy, Role::FinderSub, dark);
            }
        }

        b.set(w - 2, 0, Role::CornerFinder, true);
        b.set(w - 1, 0, Role::CornerFinder, true);
        b.set(w - 2, 1, Role::CornerFinder, false);
        b.set(w - 1, 1, Role::CornerFinder, true);
        // At height 7 the bottom-left corner is the finder itself, and at
        // height 9 the separator leaves room for the bottom row only.
        if h >= 11 {
            b.set(0, h - 2, Role::CornerFinder, true);
            b.set(1, h - 2, Role::CornerFinder, false);
        }
        if h >= 9 {
            for x in 0..3 {
                b.set(x, h - 1, Role::CornerFinder, true);
            }
        }

        let centers = tables::rmqr_alignment_centers(w);
        for &cx in centers {
            for row in 0..3usize {
                for dx in 0..3usize {
                    let dark = row != 1 || dx != 1;
                    b.set(cx + dx - 1, row, Role::Alignment, dark);
                    b.set(cx + dx - 1, h - 3 + row, Role::Alignment, dark);
                }
            }
        }

        for x in 0..w {
            b.set_if_free(x, 0, Role::Timing, x % 2 == 0);
            b.set_if_free(x, h - 1, Role::Timing, x % 2 == 0);
        }
        for x in [0, w - 1].into_iter().chain(centers.iter().copied()) {
            for y in 0..h {
                b.set_if_free(x, y, Role::Timing, y % 2 == 0);
            }
        }

        let finder_side: Vec<Pos> = (0..18usize)
            .map(|i| if i < 15 { Pos::new(8 + i / 5, 1 + i % 5) } else { Pos::new(11, i - 14) })
            .collect();
        let sub_side: Vec<Pos> = (0..18usize)
            .map(|i| {
                if i < 15 {
                    Pos::new(w - 8 + i / 5, h - 6 + i % 5)
                } else {
                    Pos::new(w - 5 + (i - 15), h - 6)
                }
            })
            .collect();
        let format =
            vec![InfoCopy { bits: finder_side, xor: 0x1FAB2 }, InfoCopy { bits: sub_side, xor: 0x20A7B }];
        b.info(&format, |copy, bit| Role::Format { copy, bit });

        let path = b.data_path((1..w - 1).rev().step_by(2));
        b.finish(version, path, format, Vec::new(), 8 * tables::total_codewords(version))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::Family;

    #[test]
    fn module_count_identity() {
        // The number of data modules must account for every codeword in the
        // tables, with fewer than eight bits left over.
        for family in Family::ALL {
            for version in family.versions() {
                let map = FunctionMap::new(version);
                let total = tables::total_codewords(version);
                match version {
                    SymbolVersion::Micro(v) => {
                        let half = usize::from(tables::has_half_codeword(version));
                        assert_eq!(map.data_path.len(), 8 * total - 4 * half, "M{v}");
                        assert_eq!(map.remainder_bits, 0);
                    }
                    SymbolVersion::Model1(_) => {
                        // The corner codeword has four modules.
                        assert_eq!(map.data_path.len() + 4, 8 * total, "{version}");
                        assert_eq!(map.remainder_bits, 0);
                    }
                    _ => {
                        assert_eq!(map.data_path.len() / 8, total, "{version}");
                        assert_eq!(map.remainder_bits, map.data_path.len() % 8, "{version}");
                    }
                }
            }
        }
    }

    #[test]
    fn qr_remainder_bits() {
        for v in 1..=40u8 {
            let want = match v {
                1 | 7..=13 | 35..=40 => 0,
                2..=6 => 7,
                14..=20 | 28..=34 => 3,
                _ => 4,
            };
            assert_eq!(FunctionMap::new(SymbolVersion::Qr(v)).remainder_bits, want, "v{v}");
            let raw = tables::qr_raw_data_modules(v);
            assert_eq!(FunctionMap::new(SymbolVersion::Qr(v)).data_path.len(), raw, "v{v}");
        }
    }

    #[test]
    fn rmqr_remainder_bits() {
        #[rustfmt::skip]
        let want = [
            0, 3, 5, 6, 1,
            2, 3, 1, 4, 5,
            2, 1, 0, 2, 7, 6,
            4, 1, 6, 4, 3, 0,
            1, 4, 6, 7, 2,
            1, 2, 0, 3, 4,
        ];
        for (v, &bits) in want.iter().enumerate() {
            let map = FunctionMap::new(SymbolVersion::Rmqr(v as u8));
            assert_eq!(map.remainder_bits, bits, "{}", map.version);
        }
    }

    #[test]
    fn every_data_module_is_on_the_path_once() {
        for version in [
            SymbolVersion::Qr(1),
            SymbolVersion::Qr(7),
            SymbolVersion::Qr(40),
            SymbolVersion::Micro(1),
            SymbolVersion::Micro(4),
            SymbolVersion::Rmqr(0),
            SymbolVersion::Rmqr(5),
            SymbolVersion::Rmqr(10),
            SymbolVersion::Rmqr(31),
            SymbolVersion::Model1(1),
            SymbolVersion::Model1(2),
            SymbolVersion::Model1(7),
            SymbolVersion::Model1(14),
        ] {
            let map = FunctionMap::new(version);
            let mut seen = vec![false; map.width * map.height];
            for (i, pos) in map.data_path.iter().enumerate() {
                let index = pos.y as usize * map.width + pos.x as usize;
                assert!(!seen[index]);
                seen[index] = true;
                match map.role(pos.x as usize, pos.y as usize) {
                    Role::Data { path } | Role::Remainder { path } => assert_eq!(path as usize, i),
                    other => panic!("{version}: path module has role {other:?}"),
                }
            }
            for y in 0..map.height {
                for x in 0..map.width {
                    let role = map.role(x, y);
                    assert_eq!(role.is_fixed(), map.expected(x, y).is_some(), "{version} ({x},{y})");
                    let on_path = matches!(role, Role::Data { .. } | Role::Remainder { .. });
                    assert_eq!(on_path, seen[y * map.width + x]);
                }
            }
        }
    }

    #[test]
    fn polarity_from_fixed_patterns() {
        assert!(looks_inverted((0, 8)));
        assert!(looks_inverted((2, 8)), "three quarters wrong is enough");
        assert!(!looks_inverted((3, 8)));
        assert!(!looks_inverted((0, 7)), "too few known to judge");
        assert!(!looks_inverted((0, 0)));
        assert!(!looks_inverted((100, 100)));
        assert!(!looks_inverted((50, 100)), "misalignment reads as noise, not inversion");

        for version in [SymbolVersion::Qr(1), SymbolVersion::Qr(7), SymbolVersion::Micro(2), SymbolVersion::Rmqr(0)] {
            let map = FunctionMap::new(version);
            let grid = map.blank_grid();
            assert!(!map.is_inverted(&grid), "{version}");
            let inverted = grid.inverted();
            assert!(map.is_inverted(&inverted), "{version}");
            let (matching, known) = map.fixed_agreement(&inverted);
            assert_eq!(matching, 0);
            assert_eq!(known, map.fixed_agreement(&grid).1);
            // A few fixed modules lost to damage do not change the verdict.
            let mut damaged = inverted.clone();
            for (x, y) in [(0, 0), (1, 1), (2, 2), (3, 3)] {
                damaged.set(x, y, Module::Unknown);
            }
            assert!(map.is_inverted(&damaged), "{version}");
            // Nothing painted: nothing to judge.
            assert!(!map.is_inverted(&ModuleGrid::new(map.width, map.height)), "{version}");
        }
    }

    #[test]
    fn version_1_layout() {
        let map = FunctionMap::new(SymbolVersion::Qr(1));
        assert_eq!(map.role(0, 0), Role::Finder);
        assert_eq!(map.role(7, 7), Role::Separator);
        assert_eq!(map.role(8, 6), Role::Timing);
        assert_eq!(map.expected(8, 6), Some(true));
        assert_eq!(map.role(8, 13), Role::DarkModule);
        assert_eq!(map.role(8, 0), Role::Format { copy: 0, bit: 0 });
        assert_eq!(map.role(0, 8), Role::Format { copy: 0, bit: 14 });
        assert_eq!(map.role(20, 8), Role::Format { copy: 1, bit: 0 });
        assert_eq!(map.role(8, 20), Role::Format { copy: 1, bit: 14 });
        // First data bit goes bottom right, then left, then up.
        assert_eq!(&map.data_path[..4], [Pos::new(20, 20), Pos::new(19, 20), Pos::new(20, 19), Pos::new(19, 19)]);
        // The pair to the left of the timing column is (5, 4).
        assert!(map.data_path.iter().all(|p| p.x != 6));
        assert!(map.version_info.is_empty());
        assert_eq!(FunctionMap::new(SymbolVersion::Qr(7)).version_info.len(), 2);
    }

    #[test]
    fn model1_layout() {
        // Version 2, as zxing-cpp's sample symbol has it.
        let map = FunctionMap::new(SymbolVersion::Model1(2));
        assert_eq!(map.role(0, 0), Role::Finder);
        assert_eq!(map.role(8, 6), Role::Timing);
        assert_eq!(map.role(8, 17), Role::DarkModule);
        assert_eq!(map.role(8, 0), Role::Format { copy: 0, bit: 0 });
        assert_eq!(map.role(24, 8), Role::Format { copy: 1, bit: 0 });
        assert_eq!(map.format[0].xor, 0x2825);
        assert!(map.version_info.is_empty());
        // No alignment pattern where Model 2 has one.
        assert!(matches!(map.role(18, 18), Role::Data { .. }));
        // The corner: one dark module, three light.
        assert_eq!(map.role(24, 24), Role::Extension);
        assert_eq!(map.expected(24, 24), Some(true));
        assert_eq!(map.expected(23, 23), Some(false));
        // The right extension pattern at rows 13 to 16: edge dark, inside light.
        for y in 13..=16 {
            assert_eq!((map.role(24, y), map.expected(24, y)), (Role::Extension, Some(true)), "({y})");
            assert_eq!((map.role(23, y), map.expected(23, y)), (Role::Extension, Some(false)), "({y})");
        }
        for x in 13..=16 {
            assert_eq!(map.expected(x, 24), Some(true));
            assert_eq!(map.expected(x, 23), Some(false));
        }
        assert!(matches!(map.role(24, 12), Role::Data { .. }));
        assert!(matches!(map.role(24, 17), Role::Data { .. }));
        // The first codeword is the four modules above the corner, bit 3
        // first; the second is the two-by-four block above those.
        assert_eq!(&map.data_path[..4], [Pos::new(24, 22), Pos::new(23, 22), Pos::new(24, 21), Pos::new(23, 21)]);
        assert_eq!(map.data_path[4], Pos::new(24, 20));
        assert_eq!(map.data_path[11], Pos::new(23, 17));
        // The extension pattern is stepped over; then the top block of
        // the column, and the next column pair from the bottom.
        assert_eq!(map.data_path[12], Pos::new(24, 12));
        assert_eq!(map.data_path[20], Pos::new(22, 24));
        assert_eq!(map.data_path.len(), 46 * 8 - 4);
        assert_eq!(map.codeword_bits(), 46 * 8 - 4);
        assert!(map.data_path.iter().all(|p| p.x != 6 && p.y != 6));

        for v in 1..=14u8 {
            let map = FunctionMap::new(SymbolVersion::Model1(v));
            let extensions = (0..map.height)
                .flat_map(|y| (0..map.width).map(move |x| (x, y)))
                .filter(|&(x, y)| map.role(x, y) == Role::Extension)
                .count();
            assert_eq!(extensions, 8 * tables::model1_extension_patterns(v) + 4, "v{v}");
        }
    }

    #[test]
    fn rmqr_corner_cases() {
        let r7 = FunctionMap::new(SymbolVersion::Rmqr(0));
        assert_eq!(r7.role(0, 6), Role::Finder);
        assert_eq!(r7.role(7, 6), Role::Separator);
        let r9 = FunctionMap::new(SymbolVersion::Rmqr(5));
        assert_eq!(r9.role(1, 8), Role::CornerFinder);
        assert_eq!(r9.expected(1, 8), Some(true));
        assert_eq!(r9.role(1, 7), Role::Separator);
        let r11 = FunctionMap::new(SymbolVersion::Rmqr(10));
        assert_eq!(r11.role(1, 9), Role::CornerFinder);
        assert_eq!(r11.expected(1, 9), Some(false));
        // The last strip is on the data path; here it holds a remainder bit.
        assert_eq!(r11.role(1, 8), Role::Remainder { path: r11.data_path.len() as u32 - 1 });
        assert_eq!(r11.data_path[0], Pos::new(25, 5));
    }
}
