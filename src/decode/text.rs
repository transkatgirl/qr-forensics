//! Turning parsed segments into text, keeping track of which characters can
//! be trusted.

use std::ops::Range;

use encoding_rs::Encoding;

use super::bitstream::{Cell, Certainty, Segment, SegmentContent, application_indicator_text};

/// IBM code page 437 from 0x80 up (ECI 0 and 2). The lower half is ASCII.
const CP437_HIGH: [char; 128] = [
    'Ç', 'ü', 'é', 'â', 'ä', 'à', 'å', 'ç', 'ê', 'ë', 'è', 'ï', 'î', 'ì', 'Ä', 'Å', //
    'É', 'æ', 'Æ', 'ô', 'ö', 'ò', 'û', 'ù', 'ÿ', 'Ö', 'Ü', '¢', '£', '¥', '₧', 'ƒ', //
    'á', 'í', 'ó', 'ú', 'ñ', 'Ñ', 'ª', 'º', '¿', '⌐', '¬', '½', '¼', '¡', '«', '»', //
    '░', '▒', '▓', '│', '┤', '╡', '╢', '╖', '╕', '╣', '║', '╗', '╝', '╜', '╛', '┐', //
    '└', '┴', '┬', '├', '─', '┼', '╞', '╟', '╚', '╔', '╩', '╦', '╠', '═', '╬', '╧', //
    '╨', '╤', '╥', '╙', '╘', '╒', '╓', '╫', '╪', '┘', '┌', '█', '▄', '▌', '▐', '▀', //
    'α', 'ß', 'Γ', 'π', 'Σ', 'σ', 'µ', 'τ', 'Φ', 'Θ', 'Ω', 'δ', '∞', 'φ', 'ε', '∩', //
    '≡', '±', '≥', '≤', '⌠', '⌡', '÷', '≈', '°', '∙', '·', '√', 'ⁿ', '²', '■', '\u{A0}',
];
use crate::symbol::Family;
use crate::symbol::tables::Mode;

/// Stands in for a character whose bits are unreadable or invalid.
pub const UNKNOWN_CHAR: char = '\u{FFFD}';
/// The group separator that FNC1 data uses between fields.
pub const GROUP_SEPARATOR: char = '\u{1D}';

/// Character set for byte-mode data that no ECI designator declares one
/// for. A set the symbol declares is followed whatever is chosen here, as
/// ZXing takes a character set hint and BoofCV an encoding to force.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Charset {
    /// Guess from the bytes.
    #[default]
    Auto,
    Utf8,
    Latin1,
    /// What most encoders mean when they say ISO-8859-1.
    Windows1252,
    ShiftJis,
    Gb18030,
    Big5,
    EucKr,
    /// Any other character set the ECI table names, by its assignment
    /// number: the sets a symbol may declare, offered as a choice for one
    /// that declares none, as ZXing takes any character set as a hint.
    Eci(u32),
}

impl Charset {
    /// Every choice offered: the common sets by name, then every other
    /// set an ECI designator can name, in the order of the assignments.
    pub const ALL: [Charset; 32] = [
        Charset::Auto,
        Charset::Utf8,
        Charset::Latin1,
        Charset::Windows1252,
        Charset::ShiftJis,
        Charset::Gb18030,
        Charset::Big5,
        Charset::EucKr,
        Charset::Eci(0),
        Charset::Eci(4),
        Charset::Eci(5),
        Charset::Eci(6),
        Charset::Eci(7),
        Charset::Eci(8),
        Charset::Eci(9),
        Charset::Eci(10),
        Charset::Eci(11),
        Charset::Eci(12),
        Charset::Eci(13),
        Charset::Eci(15),
        Charset::Eci(16),
        Charset::Eci(17),
        Charset::Eci(18),
        Charset::Eci(21),
        Charset::Eci(22),
        Charset::Eci(24),
        Charset::Eci(25),
        Charset::Eci(31),
        Charset::Eci(33),
        Charset::Eci(34),
        Charset::Eci(35),
        Charset::Eci(899),
    ];

    pub fn label(self) -> &'static str {
        match self {
            Charset::Auto => "Auto",
            Charset::Utf8 => "UTF-8",
            Charset::Latin1 => "ISO-8859-1",
            Charset::Windows1252 => "Windows-1252",
            Charset::ShiftJis => "Shift_JIS",
            Charset::Gb18030 => "GB18030",
            Charset::Big5 => "Big5",
            Charset::EucKr => "EUC-KR",
            Charset::Eci(eci) => eci_codec(eci).map_or("unsupported", Codec::name),
        }
    }

    fn codec(self) -> Option<Codec> {
        Some(match self {
            Charset::Auto => return None,
            Charset::Utf8 => Codec::Rs(encoding_rs::UTF_8),
            Charset::Latin1 => Codec::Latin1,
            Charset::Windows1252 => Codec::Rs(encoding_rs::WINDOWS_1252),
            Charset::ShiftJis => Codec::Rs(encoding_rs::SHIFT_JIS),
            Charset::Gb18030 => Codec::Rs(encoding_rs::GB18030),
            Charset::Big5 => Codec::Rs(encoding_rs::BIG5),
            Charset::EucKr => Codec::Rs(encoding_rs::EUC_KR),
            Charset::Eci(eci) => return eci_codec(eci),
        })
    }
}

#[derive(Clone, Copy)]
enum Codec {
    /// True ISO-8859-1: every byte is the code point of the same value.
    /// `encoding_rs` maps that label to windows-1252, so it is done by hand.
    Latin1,
    /// ECI 899: the bytes are data, not text. Shown as ISO-8859-1 so that
    /// every byte has a character, as zxing-cpp shows binary content.
    Binary,
    /// IBM code page 437 (ECI 0 and 2), which `encoding_rs` lacks.
    Cp437,
    /// UTF-32 (ECI 34 big-endian, 35 little-endian), which `encoding_rs`
    /// lacks.
    Utf32 { big_endian: bool },
    /// One of the ISO/IEC 8859 sets, by its part number, decoded through
    /// `encoding_rs`. Named by the part the symbol declared: the WHATWG
    /// table `encoding_rs` follows folds parts 9 and 11 into the Windows
    /// code pages that extend them, 1254 and 874, and would otherwise
    /// name those instead.
    Iso8859 { part: u8, encoding: &'static Encoding },
    Rs(&'static Encoding),
}

/// The names of the ISO/IEC 8859 parts, indexed by part number.
const ISO_8859_NAMES: [&str; 17] = [
    "", "ISO-8859-1", "ISO-8859-2", "ISO-8859-3", "ISO-8859-4", "ISO-8859-5", "ISO-8859-6", "ISO-8859-7",
    "ISO-8859-8", "ISO-8859-9", "ISO-8859-10", "ISO-8859-11", "ISO-8859-12", "ISO-8859-13", "ISO-8859-14",
    "ISO-8859-15", "ISO-8859-16",
];

impl Codec {
    fn name(self) -> &'static str {
        match self {
            Codec::Latin1 => "ISO-8859-1",
            Codec::Binary => "binary",
            Codec::Cp437 => "IBM437",
            Codec::Utf32 { big_endian: true } => "UTF-32BE",
            Codec::Utf32 { big_endian: false } => "UTF-32LE",
            Codec::Iso8859 { part, .. } => ISO_8859_NAMES[usize::from(part)],
            Codec::Rs(encoding) => encoding.name(),
        }
    }

    /// Bytes per code unit of a fixed-width encoding, whose alignment must
    /// survive an unreadable byte; one for the rest.
    fn unit(self) -> usize {
        match self {
            Codec::Utf32 { .. } => 4,
            Codec::Rs(encoding) if encoding == encoding_rs::UTF_16BE || encoding == encoding_rs::UTF_16LE => 2,
            _ => 1,
        }
    }

    /// The byte order mark this encoding may begin with. It is not content:
    /// ZXing and zxing-cpp drop one at the start of the data.
    fn bom(self) -> &'static [u8] {
        match self {
            Codec::Utf32 { big_endian: true } => &[0x00, 0x00, 0xFE, 0xFF],
            Codec::Utf32 { big_endian: false } => &[0xFF, 0xFE, 0x00, 0x00],
            Codec::Rs(encoding) if encoding == encoding_rs::UTF_8 => &[0xEF, 0xBB, 0xBF],
            Codec::Rs(encoding) if encoding == encoding_rs::UTF_16BE => &[0xFE, 0xFF],
            Codec::Rs(encoding) if encoding == encoding_rs::UTF_16LE => &[0xFF, 0xFE],
            _ => &[],
        }
    }

    /// `text` as bytes of this set, or the first character the set lacks.
    fn encode(self, text: &str) -> Result<Vec<u8>, char> {
        match self {
            Codec::Latin1 | Codec::Binary => text.chars().map(|ch| u8::try_from(u32::from(ch)).map_err(|_| ch)).collect(),
            Codec::Cp437 => text
                .chars()
                .map(|ch| match CP437_HIGH.iter().position(|&high| high == ch) {
                    _ if ch.is_ascii() => Ok(ch as u8),
                    Some(i) => Ok(0x80 + i as u8),
                    None => Err(ch),
                })
                .collect(),
            Codec::Utf32 { big_endian } => {
                Ok(text.chars().flat_map(|ch| if big_endian { u32::from(ch).to_be_bytes() } else { u32::from(ch).to_le_bytes() }).collect())
            }
            // `encoding_rs` decodes UTF-16 and will not encode it.
            Codec::Rs(encoding) if encoding == encoding_rs::UTF_16BE => Ok(text.encode_utf16().flat_map(u16::to_be_bytes).collect()),
            Codec::Rs(encoding) if encoding == encoding_rs::UTF_16LE => Ok(text.encode_utf16().flat_map(u16::to_le_bytes).collect()),
            Codec::Rs(encoding) | Codec::Iso8859 { encoding, .. } => {
                // A character the set lacks comes out of `encoding_rs` as
                // a numeric character reference: a character at a time,
                // so that the one it is can be named.
                let mut bytes = Vec::with_capacity(text.len());
                let mut one = [0; 4];
                for ch in text.chars() {
                    let (encoded, _, lacking) = encoding.encode(ch.encode_utf8(&mut one));
                    if lacking {
                        return Err(ch);
                    }
                    bytes.extend_from_slice(&encoded);
                }
                Ok(bytes)
            }
        }
    }
}

/// Character set for an ECI assignment number, where one is supported:
/// the sets ZXing's `CharacterSetECI` and zxing-cpp's `ECI` know.
fn eci_codec(eci: u32) -> Option<Codec> {
    Some(match eci {
        0 | 2 => Codec::Cp437,
        1 | 3 | 27 | 170 => Codec::Latin1,
        4..=13 | 15..=18 => {
            let part = (eci - 2) as u8;
            Codec::Iso8859 { part, encoding: Encoding::for_label(format!("iso-8859-{part}").as_bytes())? }
        }
        20 => Codec::Rs(encoding_rs::SHIFT_JIS),
        21 => Codec::Rs(encoding_rs::WINDOWS_1250),
        22 => Codec::Rs(encoding_rs::WINDOWS_1251),
        23 => Codec::Rs(encoding_rs::WINDOWS_1252),
        24 => Codec::Rs(encoding_rs::WINDOWS_1256),
        25 => Codec::Rs(encoding_rs::UTF_16BE),
        26 => Codec::Rs(encoding_rs::UTF_8),
        28 => Codec::Rs(encoding_rs::BIG5),
        29 | 31 => Codec::Rs(encoding_rs::GBK),
        30 => Codec::Rs(encoding_rs::EUC_KR),
        32 => Codec::Rs(encoding_rs::GB18030),
        33 => Codec::Rs(encoding_rs::UTF_16LE),
        34 => Codec::Utf32 { big_endian: true },
        35 => Codec::Utf32 { big_endian: false },
        899 => Codec::Binary,
        _ => return None,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextSpan {
    pub text: String,
    pub certainty: Certainty,
}

/// Where a character of the message came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CharSource {
    /// The segment it is of, by its place among the segments parsed.
    pub segment: usize,
    /// The first of the segment's cells it was made from: a byte of byte
    /// data, which a character may take several of, or a character of
    /// the other modes.
    pub cell: usize,
    /// The stream bits it was made from. Digits packed three to a group,
    /// and alphanumeric characters two to a group, have their group's.
    pub bits: Range<usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TextReport {
    /// The message in order; adjacent spans differ in certainty.
    pub spans: Vec<TextSpan>,
    /// Where each character of the message came from, one for every
    /// character of [`plain`](Self::plain) in order. Empty for a message
    /// put together from the parts of a structured append sequence.
    pub sources: Vec<CharSource>,
    /// Character set used for byte-mode data, when there was any.
    pub encoding: Option<String>,
    /// AIM symbology identifier: `]Q0` for a QR Code Model 1 symbol, and
    /// `]Q1` to `]Q6` for the rest, where the digit says whether FNC1 and
    /// ECI were used. `None` when nothing was read.
    pub symbology: Option<String>,
    /// XOR of every data byte of the message, as structured append
    /// computes its parity. `None` when any byte is unreadable.
    pub parity: Option<u8>,
    pub notes: Vec<String>,
    /// The message as bytes, before any character set: numeric and
    /// alphanumeric characters as ASCII, byte-mode data as read, kanji and
    /// hanzi as their two bytes. `None` for a byte that is unreadable.
    pub bytes: Vec<Option<u8>>,
    /// ECI designators and the offset into `bytes` from which each
    /// applies.
    pub eci: Vec<(usize, u32)>,
    /// The message in the ECI protocol's notation, as zxing-cpp's ECI text
    /// mode writes it: every designator as a backslash and six digits
    /// where it stood, and every literal backslash doubled.
    pub eci_text: String,
    /// The message read as GS1 element strings, when FNC1 in first
    /// position says it is one.
    pub gs1: Option<Gs1>,
}

/// GS1 element strings in human readable form, as zxing-cpp's `HRIFromGS1`
/// presents them: each application identifier in parentheses before its
/// data, with the separators dropped.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Gs1 {
    /// The human readable interpretation.
    pub hri: String,
    /// What did not parse: an application identifier the specifications
    /// do not define, a field cut short or holding the wrong characters.
    /// Evidence of a wrong read, as an impossible value in a segment is.
    pub problems: Vec<String>,
}

/// How many digits an application identifier has, from its first two: the
/// GS1 General Specifications fix the length of every identifier by them.
fn ai_digits(first: u8, second: u8) -> Option<usize> {
    Some(match (first, second) {
        (0, _) | (1, _) | (2, 0..=2) | (3, 0) | (3, 7) | (9, _) => 2,
        (2, 3..=5) | (4, 0..=2) | (7, 1) => 3,
        (3, 1..=6) | (3, 9) | (4, 3) | (7, 0) | (7, 2) | (8, 0..=2) => 4,
        _ => return None,
    })
}

/// The whole length, identifier and data, of an element string with a
/// predefined length (GS1 General Specifications, 7.8.5), by the first
/// two digits of the identifier. `None` for the others, whose data runs to
/// a group separator or the end.
fn predefined_length(first: u8, second: u8) -> Option<usize> {
    Some(match (first, second) {
        (0, 0) => 20,
        (0, 1..=3) => 16,
        (0, 4) => 18,
        (1, 1..=9) => 8,
        (2, 0) => 4,
        (3, 1..=6) => 10,
        (4, 1) => 16,
        _ => return None,
    })
}

/// Identifiers whose data is digits only: the item and shipping numbers,
/// dates, counts and measures. A batch number (10) or serial number (21)
/// may hold any character.
fn numeric_data(first: u8, second: u8) -> bool {
    matches!((first, second), (0, 0..=4) | (1, 1..=9) | (2, 0) | (3, _) | (4, 1))
}

/// The GS1 element strings in `text`, where the group separator ends every
/// field of variable length.
pub fn gs1_elements(text: &str) -> Gs1 {
    let chars: Vec<char> = text.chars().collect();
    let mut gs1 = Gs1::default();
    let mut pos = 0;
    while pos < chars.len() {
        if chars[pos] == GROUP_SEPARATOR {
            pos += 1;
            continue;
        }
        let digit = |i: usize| chars.get(i).and_then(|c| c.to_digit(10)).map(|d| d as u8);
        let rest = || chars[pos..].iter().collect::<String>();
        let (Some(first), Some(second)) = (digit(pos), digit(pos + 1)) else {
            gs1.problems.push(format!("an application identifier was expected at character {pos}, not {:?}", rest().chars().take(4).collect::<String>()));
            gs1.hri.push_str(&rest());
            break;
        };
        let Some(digits) = ai_digits(first, second) else {
            gs1.problems.push(format!("no application identifier begins with {first}{second}"));
            gs1.hri.push_str(&rest());
            break;
        };
        if pos + digits > chars.len() || !chars[pos..pos + digits].iter().all(char::is_ascii_digit) {
            gs1.problems.push(format!("the application identifier at character {pos} is cut short"));
            gs1.hri.push_str(&rest());
            break;
        }
        let ai: String = chars[pos..pos + digits].iter().collect();
        let data_start = pos + digits;
        let data_end = match predefined_length(first, second) {
            Some(total) => {
                let end = (pos + total).min(chars.len());
                // A separator before the field is full cuts it short, as
                // does the end of the data.
                let cut = chars[data_start..end].iter().position(|&c| c == GROUP_SEPARATOR).map(|n| data_start + n);
                if cut.is_some() || pos + total > chars.len() {
                    let have = cut.unwrap_or(chars.len()) - data_start;
                    gs1.problems.push(format!("({ai}) needs {} characters of data and has {have}", total - digits));
                }
                cut.unwrap_or(end)
            }
            None => chars[data_start..].iter().position(|&c| c == GROUP_SEPARATOR).map_or(chars.len(), |n| data_start + n),
        };
        let data: String = chars[data_start..data_end].iter().collect();
        if data.is_empty() {
            gs1.problems.push(format!("({ai}) has no data"));
        } else if data.contains(UNKNOWN_CHAR) {
            gs1.problems.push(format!("({ai}) has unreadable characters"));
        } else if numeric_data(first, second) && !data.chars().all(|c| c.is_ascii_digit()) {
            gs1.problems.push(format!("({ai}) must be digits, not {data:?}"));
        }
        gs1.hri.push_str(&format!("({ai}){data}"));
        pos = data_end;
    }
    gs1
}

/// The Unicode control picture for a control character, as zxing-cpp's
/// ISO/IEC 15434 text mode shows them (`HRIFromISO15434`): ␝ for the group
/// separator, ␍ for a carriage return, and so on. `None` for anything
/// else.
pub fn control_picture(c: char) -> Option<char> {
    match c as u32 {
        code @ 0..=0x1F => char::from_u32(0x2400 + code),
        0x7F => Some('\u{2421}'),
        _ => None,
    }
}

impl TextReport {
    /// The whole message, with [`UNKNOWN_CHAR`] for unreadable characters.
    pub fn plain(&self) -> String {
        self.spans.iter().map(|s| s.text.as_str()).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// The message's bytes in hexadecimal, two digits each, `??` where a
    /// byte is unreadable.
    pub fn hex(&self) -> String {
        hex_of(&self.bytes)
    }

    /// The message with control characters spelt out, as `<GS>`, `<CR>`
    /// and so on, and unreadable characters as `<?>`.
    pub fn escaped(&self) -> String {
        escape_controls(&self.plain())
    }

    /// The message's bytes when every one of them is known.
    pub fn raw_bytes(&self) -> Option<Vec<u8>> {
        self.bytes.iter().copied().collect()
    }

    /// Number of characters at each certainty: (verified, unverified, unknown).
    pub fn counts(&self) -> (usize, usize, usize) {
        let mut counts = (0, 0, 0);
        for span in &self.spans {
            let n = span.text.chars().count();
            match span.certainty {
                Certainty::Verified => counts.0 += n,
                Certainty::Unverified => counts.1 += n,
                Certainty::Unknown => counts.2 += n,
            }
        }
        counts
    }

    fn push(&mut self, text: &str, certainty: Certainty, from: &CharSource) {
        if text.is_empty() {
            return;
        }
        match self.spans.last_mut() {
            Some(last) if last.certainty == certainty => last.text.push_str(text),
            _ => self.spans.push(TextSpan { text: text.to_owned(), certainty }),
        }
        for ch in text.chars() {
            if ch == '\\' {
                self.eci_text.push('\\');
            }
            self.eci_text.push(ch);
            self.sources.push(from.clone());
        }
    }

    fn push_unknown(&mut self, from: &CharSource) {
        self.push(UNKNOWN_CHAR.encode_utf8(&mut [0; 4]), Certainty::Unknown, from);
    }

    /// Note an ECI designator taking effect from here on.
    fn push_eci(&mut self, value: u32) {
        self.eci.push((self.bytes.len(), value));
        self.eci_text.push_str(&format!("\\{value:06}"));
    }
}

/// Bytes in hexadecimal, two digits each, separated by spaces, `??` for
/// an unknown byte.
pub fn hex_of(bytes: &[Option<u8>]) -> String {
    bytes.iter().map(|b| b.map_or("??".to_owned(), |v| format!("{v:02X}"))).collect::<Vec<_>>().join(" ")
}

/// Text with control characters spelt out as `<GS>`, `<CR>`, `<LF>`,
/// `<TAB>`, `<ESC>`, `<FS>`, `<RS>`, `<EOT>` or `<xx>` in hexadecimal,
/// and the unknown-character mark as `<?>`.
pub fn escape_controls(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            UNKNOWN_CHAR => out.push_str("<?>"),
            GROUP_SEPARATOR => out.push_str("<GS>"),
            '\u{1C}' => out.push_str("<FS>"),
            '\u{1E}' => out.push_str("<RS>"),
            '\u{04}' => out.push_str("<EOT>"),
            '\u{1B}' => out.push_str("<ESC>"),
            '\r' => out.push_str("<CR>"),
            '\n' => out.push_str("<LF>"),
            '\t' => out.push_str("<TAB>"),
            c if c.is_control() => out.push_str(&format!("<{:02X}>", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// The bytes of a segment's content as the message carries them: ASCII
/// for numeric and alphanumeric characters, the bytes themselves for byte
/// mode, two bytes for each kanji or hanzi character. `None` for anything
/// unreadable.
pub fn segment_bytes(content: &SegmentContent) -> Vec<Option<u8>> {
    match content {
        SegmentContent::Chars(cells) => cells.iter().map(|c| c.value.map(|ch| ch as u8)).collect(),
        SegmentContent::Bytes(cells) => cells.iter().map(|c| c.value).collect(),
        SegmentContent::Kanji(cells) | SegmentContent::Hanzi(cells) => cells
            .iter()
            .flat_map(|c| match c.value {
                Some(code) => code.to_be_bytes().map(Some),
                None => [None, None],
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Flags that the whole message carries.
struct Markers {
    fnc1_first: bool,
    fnc1_second: bool,
    eci: bool,
}

impl Markers {
    fn of(segments: &[Segment]) -> Self {
        let has = |f: &dyn Fn(&SegmentContent) -> bool| segments.iter().any(|s| f(&s.content));
        Self {
            fnc1_first: has(&|c| matches!(c, SegmentContent::Fnc1First)),
            fnc1_second: has(&|c| matches!(c, SegmentContent::Fnc1Second(_))),
            eci: has(&|c| matches!(c, SegmentContent::Eci(_))),
        }
    }

    /// FNC1 in either position puts the group-separator convention on
    /// alphanumeric data.
    fn fnc1(&self) -> bool {
        self.fnc1_first || self.fnc1_second
    }

    /// ISO/IEC 18004 Table 3 (ISO/IEC 15424). Modifier 0 is a Model 1
    /// symbol, as zxing-cpp reports one; Model 1 has neither FNC1 nor
    /// ECI, and a stream that carries one all the same keeps the modifier
    /// that says so.
    fn symbology(&self, family: Family) -> String {
        let modifier = 1 + if self.fnc1_first { 2 } else if self.fnc1_second { 4 } else { 0 } + u8::from(self.eci);
        let modifier = if family == Family::Model1 && modifier == 1 { 0 } else { modifier };
        format!("]Q{modifier}")
    }
}

/// XOR of every data byte, as the parity of a structured append message is
/// computed over its whole input; `None` when any byte is unreadable.
fn parity(segments: &[Segment]) -> Option<u8> {
    let mut parity = 0u8;
    for segment in segments {
        match &segment.content {
            SegmentContent::Chars(cells) => {
                for cell in cells {
                    parity ^= cell.value? as u8;
                }
            }
            SegmentContent::Bytes(cells) => {
                for cell in cells {
                    parity ^= cell.value?;
                }
            }
            SegmentContent::Kanji(cells) | SegmentContent::Hanzi(cells) => {
                for cell in cells {
                    let [high, low] = cell.value?.to_be_bytes();
                    parity ^= high ^ low;
                }
            }
            SegmentContent::Unreadable => return None,
            _ => {}
        }
    }
    Some(parity)
}

pub fn decode_text(segments: &[Segment], charset: Charset) -> TextReport {
    decode_text_of(Family::Qr, segments, charset)
}

/// [`decode_text`] for a symbol of `family`, which the symbology
/// identifier names along with what the segments carry.
pub fn decode_text_of(family: Family, segments: &[Segment], charset: Charset) -> TextReport {
    let mut report = TextReport::default();
    let guessed = guess_codec(segments);
    let markers = Markers::of(segments);
    // What an ECI designator has declared for the byte data after it:
    // `Some(None)` when it names no set this reader has.
    let mut declared: Option<Option<Codec>> = None;
    let mut encodings: Vec<String> = Vec::new();
    let mut first_bytes = true;
    if !segments.is_empty() {
        report.symbology = Some(markers.symbology(family));
        report.parity = parity(segments);
    }

    for (index, segment) in segments.iter().enumerate() {
        report.bytes.extend(segment_bytes(&segment.content));
        let from = |cell: usize, bits: &Range<usize>| CharSource { segment: index, cell, bits: bits.clone() };
        match &segment.content {
            SegmentContent::Chars(cells) if markers.fnc1() && segment.mode == Mode::Alphanumeric => {
                // Under FNC1, a percent sign in alphanumeric data stands
                // for the group separator, and two in a row for one
                // percent sign (ISO/IEC 18004 7.4.8.3). A percent sign
                // waits, with its place, for what follows it.
                let mut pending: Option<(Certainty, CharSource)> = None;
                for (i, cell) in cells.iter().enumerate() {
                    let here = from(i, &cell.bits);
                    match (cell.value, pending.take()) {
                        (Some('%'), Some((certainty, first))) => {
                            let both = CharSource { bits: first.bits.start..cell.bits.end, ..first };
                            report.push("%", certainty.min(cell.certainty), &both);
                        }
                        (Some('%'), None) => pending = Some((cell.certainty, here)),
                        (value, Some((certainty, first))) => {
                            report.push(GROUP_SEPARATOR.encode_utf8(&mut [0; 4]), certainty, &first);
                            match value {
                                Some(ch) => report.push(ch.encode_utf8(&mut [0; 4]), cell.certainty, &here),
                                None => report.push_unknown(&here),
                            }
                        }
                        (Some(ch), None) => report.push(ch.encode_utf8(&mut [0; 4]), cell.certainty, &here),
                        (None, None) => report.push_unknown(&here),
                    }
                }
                if let Some((certainty, first)) = pending {
                    report.push(GROUP_SEPARATOR.encode_utf8(&mut [0; 4]), certainty, &first);
                }
            }
            SegmentContent::Chars(cells) => {
                for (i, cell) in cells.iter().enumerate() {
                    match cell.value {
                        Some(ch) => report.push(ch.encode_utf8(&mut [0; 4]), cell.certainty, &from(i, &cell.bits)),
                        None => report.push_unknown(&from(i, &cell.bits)),
                    }
                }
            }
            SegmentContent::Bytes(cells) => {
                let (codec, origin) = byte_codec(declared, charset, || guessed);
                let label = format!("{} ({origin})", codec.name());
                if !encodings.contains(&label) {
                    encodings.push(label);
                }
                let mut cells: &[Cell<u8>] = cells;
                let mut skipped = 0;
                if first_bytes {
                    first_bytes = false;
                    let bom = codec.bom();
                    if !bom.is_empty() && cells.iter().zip(bom).filter(|(c, b)| c.value == Some(**b)).count() == bom.len() {
                        report.notes.push(format!(
                            "The data begins with a byte order mark ({} bytes), left out of the text as ZXing and zxing-cpp leave it out; the bytes keep it.",
                            bom.len()
                        ));
                        cells = &cells[bom.len()..];
                        skipped = bom.len();
                    }
                }
                decode_bytes(codec, cells, (index, skipped), &mut report);
            }
            SegmentContent::Kanji(cells) | SegmentContent::Hanzi(cells) => {
                let encoding = if matches!(segment.content, SegmentContent::Kanji(_)) {
                    encoding_rs::SHIFT_JIS
                } else {
                    encoding_rs::GB18030
                };
                for (i, cell) in cells.iter().enumerate() {
                    let here = from(i, &cell.bits);
                    let bytes = cell.value.map(u16::to_be_bytes);
                    let decoded = bytes
                        .as_ref()
                        .and_then(|bytes| encoding.decode_without_bom_handling_and_without_replacement(bytes));
                    match (decoded, bytes) {
                        (Some(text), _) => report.push(&text, cell.certainty, &here),
                        (None, Some(bytes)) => {
                            // The parser lets through every value an encoder
                            // could write; the set itself has gaps.
                            report.push_unknown(&here);
                            report.notes.push(format!(
                                "{} value {:02X}{:02X} at bit {} is not an assigned character.",
                                segment.mode.label(),
                                bytes[0],
                                bytes[1],
                                cell.bits.start
                            ));
                        }
                        (None, None) => report.push_unknown(&here),
                    }
                }
            }
            SegmentContent::Eci(Some(value)) => {
                report.push_eci(*value);
                declared = Some(eci_codec(*value));
                if *value == 899 {
                    report.notes.push("ECI 899: the data is binary, not text; its bytes are shown as ISO-8859-1 characters.".into());
                }
                if matches!(declared, Some(None)) {
                    // AIM's table has gaps inside the ISO-8859 run: 14 was
                    // kept for an ISO-8859-12 that was never made, and 19
                    // was never assigned. ZXing's CharacterSetECI and
                    // zxing-cpp's ECI skip both.
                    let why = if matches!(value, 14 | 19) { "is reserved and names no character set" } else { "is not supported" };
                    let (shown, _) = byte_codec(declared, charset, || Codec::Latin1);
                    report.notes.push(format!("ECI {value} {why}; bytes shown as {}.", shown.name()));
                }
            }
            SegmentContent::Eci(None) => {
                report.notes.push("An ECI designator is unreadable; character set guessed.".into());
            }
            SegmentContent::StructuredAppend { index, total, parity } => {
                let check = match report.parity {
                    Some(own) if *total == 1 && own == *parity => "This symbol's data XORs to the same value.".to_owned(),
                    Some(own) if *total == 1 => format!("This symbol's data XORs to {own:#04X}, which does not match."),
                    Some(own) => format!(
                        "This symbol's data XORs to {own:#04X}; the data of all {total} symbols together must XOR to the parity."
                    ),
                    None => "The parity cannot be checked while some of this symbol's data is unreadable.".to_owned(),
                };
                report.notes.push(format!(
                    "Structured append: symbol {index} of {total}, parity {parity:#04X}. The message continues in other symbols. {check}"
                ));
            }
            SegmentContent::Fnc1First => report.notes.push(
                "FNC1 in first position (GS1): in alphanumeric data % is shown as the group separator and %% as %.".into(),
            ),
            SegmentContent::Fnc1Second(indicator) => match application_indicator_text(*indicator) {
                Some(text) => {
                    // ISO/IEC 18004 7.4.8.3: the application indicator is
                    // transmitted ahead of the data, as two digits or a
                    // letter. ZBar and zxing-cpp put it there; ZXing leaves
                    // it out. It is not message data, so it stays out of
                    // the structured append parity.
                    report.bytes.extend(text.bytes().map(Some));
                    report.push(&text, segment.header_certainty, &from(0, &segment.payload));
                    report.notes.push(format!(
                        "FNC1 in second position: application indicator {text}, transmitted ahead of the data."
                    ));
                }
                None => report.notes.push(format!(
                    "FNC1 in second position with application indicator value {indicator}, which is not defined."
                )),
            },
            SegmentContent::Unreadable => {}
        }
        if segment.damaged {
            report.notes.push(format!(
                "{} segment at bit {} contains values that cannot occur; a module or the format is wrong.",
                segment.mode.label(),
                segment.header.start
            ));
        }
    }

    if !encodings.is_empty() {
        report.encoding = Some(encodings.join(", "));
    }
    // FNC1 in first position makes the message GS1 element strings, which
    // read as identifiers and fields; a field that does not parse is one
    // more sign of a wrong read.
    if markers.fnc1_first && !report.spans.is_empty() {
        let gs1 = gs1_elements(&report.plain());
        for problem in &gs1.problems {
            report.notes.push(format!("GS1: {problem}."));
        }
        report.gs1 = Some(gs1);
    }
    report
}

/// The character set byte data is read in, and where it came from: the
/// set an ECI designator before the data declares; without one, the set
/// chosen; without either, the set guessed from the bytes. A symbol that
/// declares its character set has said what its bytes are, and a choice
/// is for the symbol that has not: ZXing takes its character set hint
/// only for a byte segment no ECI precedes, and BoofCV ranks the encoding
/// it is told to force below the one an ECI names. A chosen set once
/// outranked a declared one here, and a batch read with the set its
/// undeclared symbols were written in misread every symbol that declared
/// another.
///
/// `declared` is `Some(None)` after a designator that names no set this
/// reader has. That declares nothing that can be followed, so the set
/// chosen stands there, and with none chosen ISO-8859-1, in which every
/// byte has a character.
fn byte_codec(declared: Option<Option<Codec>>, charset: Charset, guess: impl FnOnce() -> Codec) -> (Codec, &'static str) {
    match (declared, charset.codec()) {
        (Some(Some(codec)), _) => (codec, "ECI"),
        (_, Some(codec)) => (codec, "chosen"),
        (Some(None), None) => (Codec::Latin1, "ECI"),
        (None, None) => (guess(), "guessed"),
    }
}

/// `text` as the bytes of the character set that the byte data of segment
/// `index` is read in, by the rule [`decode_text_of`] follows: the set an
/// ECI designator before the segment declares, or the one chosen, or the
/// one guessed from the bytes. `Err` with the first character that set
/// lacks.
pub fn encode_as_read(segments: &[Segment], index: usize, charset: Charset, text: &str) -> Result<Vec<u8>, char> {
    let declared = segments[..index.min(segments.len())].iter().rev().find_map(|segment| match segment.content {
        SegmentContent::Eci(Some(value)) => Some(eci_codec(value)),
        _ => None,
    });
    byte_codec(declared, charset, || guess_codec(segments)).0.encode(text)
}

/// `text` as bytes of `charset`, UTF-8 when none is chosen. `Err` with
/// the first character the set lacks.
pub fn encode_in(charset: Charset, text: &str) -> Result<Vec<u8>, char> {
    charset.codec().unwrap_or(Codec::Rs(encoding_rs::UTF_8)).encode(text)
}

/// Decode byte cells, leaving a placeholder for each unreadable byte and
/// decoding the readable runs between them. `at` is the segment the
/// cells are of and how many of its cells came before them.
fn decode_bytes(codec: Codec, cells: &[Cell<u8>], at: (usize, usize), report: &mut TextReport) {
    let unit = codec.unit();
    if unit > 1 {
        decode_fixed_width(codec, unit, cells, at, report);
        return;
    }
    // The character made of the cells from `first` to `last`.
    let from = |first: usize, last: usize| CharSource { segment: at.0, cell: at.1 + first, bits: cells[first].bits.start..cells[last].bits.end };
    let mut i = 0;
    while i < cells.len() {
        if cells[i].value.is_none() {
            report.push_unknown(&from(i, i));
            i += 1;
            continue;
        }
        let run_end = cells[i..].iter().position(|c| c.value.is_none()).map_or(cells.len(), |n| i + n);
        let run = &cells[i..run_end];
        match codec {
            Codec::Latin1 | Codec::Binary => {
                for (n, cell) in run.iter().enumerate() {
                    let ch = char::from(cell.value.unwrap());
                    report.push(ch.encode_utf8(&mut [0; 4]), cell.certainty, &from(i + n, i + n));
                }
            }
            Codec::Cp437 => {
                for (n, cell) in run.iter().enumerate() {
                    let byte = cell.value.unwrap();
                    let ch = if byte < 0x80 { char::from(byte) } else { CP437_HIGH[usize::from(byte - 0x80)] };
                    report.push(ch.encode_utf8(&mut [0; 4]), cell.certainty, &from(i + n, i + n));
                }
            }
            Codec::Utf32 { .. } => unreachable!("fixed width"),
            Codec::Rs(encoding) | Codec::Iso8859 { encoding, .. } => {
                // One byte at a time, so each character can inherit the
                // certainty of exactly the bytes that produced it.
                let mut decoder = encoding.new_decoder_without_bom_handling();
                let mut pending = Certainty::Verified;
                // The cell the character under way began at.
                let mut begun = None;
                let mut out = String::with_capacity(32);
                for (n, cell) in run.iter().enumerate() {
                    out.clear();
                    pending = pending.min(cell.certainty);
                    let first = *begun.get_or_insert(i + n);
                    let _ = decoder.decode_to_string(&[cell.value.unwrap()], &mut out, n + 1 == run.len());
                    if !out.is_empty() {
                        report.push(&out, pending, &from(first, i + n));
                        pending = Certainty::Verified;
                        begun = None;
                    }
                }
            }
        }
        i = run_end;
    }
}

/// Decode an encoding whose code units are `unit` bytes wide, keeping the
/// alignment from the start of the segment across unreadable bytes: a gap
/// costs the unit it falls in, and the next unit starts where it should
/// rather than at the next readable byte, which would garble everything
/// after the gap.
fn decode_fixed_width(codec: Codec, unit: usize, cells: &[Cell<u8>], at: (usize, usize), report: &mut TextReport) {
    let mut decoder = match codec {
        Codec::Rs(encoding) => Some(encoding.new_decoder_without_bom_handling()),
        _ => None,
    };
    let mut out = String::with_capacity(8);
    // A character may span two units (a surrogate pair), and inherits the
    // weakest of them.
    let mut pending = Certainty::Verified;
    // The cell the character under way began at.
    let mut begun = None;
    let groups = cells.chunks(unit).count();
    for (n, group) in cells.chunks(unit).enumerate() {
        let (start, last) = (n * unit, n * unit + group.len() - 1);
        let from = |first: usize| CharSource { segment: at.0, cell: at.1 + first, bits: cells[first].bits.start..cells[last].bits.end };
        let bytes: Option<Vec<u8>> = group.iter().map(|c| c.value).collect();
        let certainty = group.iter().map(|c| c.certainty).min().unwrap_or(Certainty::Unknown);
        match bytes {
            Some(bytes) if bytes.len() == unit => {
                pending = pending.min(certainty);
                match codec {
                    Codec::Utf32 { big_endian } => {
                        let raw = bytes.iter().fold(0u32, |acc, &b| acc << 8 | u32::from(b));
                        let raw = if big_endian { raw } else { raw.swap_bytes() };
                        match char::from_u32(raw) {
                            Some(ch) => report.push(ch.encode_utf8(&mut [0; 4]), pending, &from(start)),
                            None => report.push_unknown(&from(start)),
                        }
                        pending = Certainty::Verified;
                    }
                    _ => {
                        let decoder = decoder.as_mut().expect("an encoding_rs codec");
                        out.clear();
                        let first = *begun.get_or_insert(start);
                        let _ = decoder.decode_to_string(&bytes, &mut out, n + 1 == groups);
                        if !out.is_empty() {
                            report.push(&out, pending, &from(first));
                            pending = Certainty::Verified;
                            begun = None;
                        }
                    }
                }
            }
            _ => {
                // A unit with an unreadable byte, or cut short by the end
                // of the data. A character begun in the unit before is lost
                // with it.
                report.push_unknown(&from(begun.take().unwrap_or(start)));
                pending = Certainty::Verified;
                if let Codec::Rs(encoding) = codec {
                    decoder = Some(encoding.new_decoder_without_bom_handling());
                }
            }
        }
    }
}

/// A run of readable bytes of byte data, and whether an unreadable byte
/// lies before it and after it. A run that follows a gap may begin in the
/// middle of a multi-byte character, and one a gap follows may end in the
/// middle of one; a run that ends where its segment does ends where the
/// data ends, and a character cut short there is no character.
struct Run {
    bytes: Vec<u8>,
    after_gap: bool,
    before_gap: bool,
}

/// Pick a character set for byte data that carries no ECI: ASCII and UTF-8
/// when the bytes allow it, then Shift-JIS, then the Chinese double-byte
/// sets, then the standard's default. A kanji segment elsewhere in the
/// symbol tips doubtful byte data towards Shift-JIS, as it does for ZBar,
/// which tries Shift-JIS before Latin-1 in a symbol that has one.
///
/// Shift-JIS is asked before GBK, as ZXing asks it and the copy of ZXing
/// in OpenCV's wechat_qrcode after it. Nearly every pair of bytes is a
/// character of GBK, so bytes that are Shift-JIS are GBK as well, and a
/// lead byte from 0x81 to 0x9F is where Shift-JIS keeps its kana and
/// common kanji and where GBK keeps characters seldom written: when GBK
/// was asked first of a message too short for three characters in a row,
/// `日本` read as `擔杮` and `東京` as `搶嫗`. What is given up is the short
/// message of GBK's rarer characters that is Shift-JIS too, which reads
/// as what it is when the set is chosen.
fn guess_codec(segments: &[Segment]) -> Codec {
    let has_kanji = segments.iter().any(|s| matches!(s.content, SegmentContent::Kanji(_)));
    let mut runs: Vec<Run> = Vec::new();
    for segment in segments {
        match &segment.content {
            SegmentContent::Eci(_) => break,
            SegmentContent::Bytes(cells) => {
                let mut after_gap = false;
                let mut current = Vec::new();
                for cell in cells {
                    match cell.value {
                        Some(byte) => current.push(byte),
                        None => {
                            if !current.is_empty() {
                                runs.push(Run { bytes: std::mem::take(&mut current), after_gap, before_gap: true });
                            }
                            after_gap = true;
                        }
                    }
                }
                if !current.is_empty() {
                    // A segment the symbol has no room to finish ends in
                    // bytes that are not there to read, as a gap does.
                    runs.push(Run { bytes: current, after_gap, before_gap: segment.truncated });
                }
            }
            _ => {}
        }
    }

    if runs.iter().all(|run| run.bytes.is_ascii()) {
        return Codec::Rs(encoding_rs::UTF_8);
    }
    if runs.iter().all(utf8_plausible) {
        return Codec::Rs(encoding_rs::UTF_8);
    }
    // Three double-byte characters in a row is as sure as these sets get.
    match shift_jis_evidence(&runs) {
        Evidence::Strong => return Codec::Rs(encoding_rs::SHIFT_JIS),
        Evidence::Weak if has_kanji => return Codec::Rs(encoding_rs::SHIFT_JIS),
        Evidence::Weak if windows_1252_shaped(&runs) => return Codec::Rs(encoding_rs::WINDOWS_1252),
        Evidence::Weak => return Codec::Rs(encoding_rs::SHIFT_JIS),
        Evidence::None => {}
    }
    if has_kanji && decodes_as_shift_jis(&runs) {
        return Codec::Rs(encoding_rs::SHIFT_JIS);
    }
    if !windows_1252_shaped(&runs)
        && let Some(codec) = double_byte_evidence(&runs)
    {
        return codec;
    }
    // The standard's default, ISO-8859-1, has only control codes from 0x80
    // to 0x9F, which text does not hold; windows-1252 is ISO-8859-1 with
    // the euro sign, the curly quotes and the dashes there, and is what
    // an encoder handed such text writes without declaring it. The
    // WHATWG's Encoding Standard, which `encoding_rs` follows, reads
    // everything labelled ISO-8859-1 so. Bytes in that range are the only
    // ones the two read differently, so only they are asked.
    if runs.iter().flat_map(|run| &run.bytes).any(|byte| (0x80..=0x9F).contains(byte)) {
        return Codec::Rs(encoding_rs::WINDOWS_1252);
    }
    Codec::Latin1
}

/// The bytes are Shift-JIS from end to end. 0x80 is not a byte of the
/// set: the WHATWG's decoder, which `encoding_rs` follows, hands it on as
/// the control code U+0080 where a character may begin, and bytes that
/// decode only so do not decode. The euro sign of windows-1252 is that
/// byte, and `€5` passed for Shift-JIS.
fn decodes_as_shift_jis(runs: &[Run]) -> bool {
    runs.iter().all(|run| {
        encoding_rs::SHIFT_JIS
            .decode_without_bom_handling_and_without_replacement(&run.bytes)
            .is_some_and(|text| !text.contains('\u{80}'))
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Evidence {
    None,
    Weak,
    Strong,
}

/// Shift-JIS and ISO-8859-1 overlap heavily: an accented Latin letter
/// followed by another letter is also a valid double-byte character, and
/// every half-width katakana is a Latin-1 symbol or capital. Believe
/// Shift-JIS outright on a run of three double-byte characters or three
/// half-width katakana, as ZXing does, and weakly on a double-byte
/// character whose lead byte Latin-1 has only a control code for (0x81 to
/// 0x9F), which is where hiragana, katakana and the common kanji lie.
///
/// ZXing believes it weakly besides when a tenth of the bytes are ones
/// Latin-1 has as symbols rather than letters (0xA0 to 0xBF, and the
/// multiplication and division signs), and so did this. One such byte is
/// a tenth of any message of ten bytes or fewer, and each is a half-width
/// katakana: `Price: £5` read as `Price: ｣5`, `25°C` as `25ｰC` and
/// `¡Hola!` as `｡Hola!`. BoofCV, which knows two sets for undeclared
/// bytes, UTF-8 where they are valid UTF-8 and ISO-8859-1 where they are
/// not, reads all three. Western text is not told from a katakana or two
/// by counting symbols, so they are not counted: a message whose only
/// Japanese is one or two half-width katakana reads as Latin-1, and as
/// what it is when Shift_JIS is chosen.
fn shift_jis_evidence(runs: &[Run]) -> Evidence {
    if !decodes_as_shift_jis(runs) {
        return Evidence::None;
    }
    let mut can_be_latin1 = true;
    let (mut longest_double, mut longest_kana) = (0, 0);
    for run in runs {
        let run = &run.bytes;
        let (mut double, mut kana) = (0, 0);
        let mut i = 0;
        while i < run.len() {
            let byte = run[i];
            can_be_latin1 &= !(0x80..=0x9F).contains(&byte);
            match byte {
                0x81..=0x9F | 0xE0..=0xFC => {
                    double += 1;
                    kana = 0;
                    i += 1;
                }
                0xA1..=0xDF => {
                    kana += 1;
                    double = 0;
                }
                _ => (double, kana) = (0, 0),
            }
            longest_double = longest_double.max(double);
            longest_kana = longest_kana.max(kana);
            i += 1;
        }
    }
    // Three double-byte characters in a row, or three half-width katakana,
    // settle it, as they do for ZXing; Latin-1 text has no business
    // looking like either.
    if longest_double >= 3 || longest_kana >= 3 {
        Evidence::Strong
    } else if !can_be_latin1 {
        Evidence::Weak
    } else {
        Evidence::None
    }
}

/// The bytes from 0x80 to 0x9F are all typography of windows-1252 set
/// among ASCII letters as Western text sets it, and there is one at
/// least: the apostrophe of `It’s`, the dash of `word—word`, the
/// ellipsis of `…and`, the letters of `cœur` and `Škoda`. Each is the
/// lead byte of a double-byte character in Shift-JIS and in GBK, and the
/// letter after it a trail byte, so `It’s ok` is `It痴 ok` to the one and
/// `It抯 ok` to the other, and valid in both; ZXing, which knows no
/// windows-1252, reads it as Shift-JIS.
///
/// Only what is followed by a letter is asked about, since a lead byte
/// before a space, a digit or the end of the data is no character of
/// either set and the bytes fall to windows-1252 without this: a closing
/// quote, a dash between spaces or figures, a trade mark sign. And it is
/// asked narrowly, since a kanji after an ASCII letter looks the same:
/// an apostrophe between two letters, a dash after a letter or a figure,
/// an ellipsis before a word, and `œ`, `š`, `ž` and their capitals in a
/// word of three letters or more, so that `市`, which is `Žs`, is still
/// a kanji. An opening quote is not asked about at all: one that is
/// closed is closed before a space, and `田` is `“c`.
fn windows_1252_shaped(runs: &[Run]) -> bool {
    // A letter of windows-1252: those of ASCII and of ISO-8859-1, and the
    // six it adds from 0x80 to 0x9F with the Y of 0x9F.
    let letter = |byte: u8| byte.is_ascii_alphabetic() || matches!(byte, 0x8A | 0x8C | 0x8E | 0x9A | 0x9C | 0x9E | 0x9F | 0xC0..=0xD6 | 0xD8..=0xF6 | 0xF8..=0xFF);
    let mut seen = false;
    for Run { bytes, after_gap, .. } in runs {
        for (i, &byte) in bytes.iter().enumerate() {
            if !(0x80..=0x9F).contains(&byte) {
                continue;
            }
            let Some(&after) = bytes.get(i + 1) else { return false };
            // What stands before the first byte of a run that follows a
            // gap is not known, and is taken for nothing that helps.
            let before = match i.checked_sub(1) {
                Some(j) => Some(bytes[j]),
                None if *after_gap => return false,
                None => None,
            };
            let begins_word = before.is_none_or(|b| b.is_ascii() && !b.is_ascii_alphabetic());
            let in_word = || {
                let from = bytes[..i].iter().rev().take_while(|&&b| letter(b)).count();
                let to = bytes[i..].iter().take_while(|&&b| letter(b)).count();
                from + to >= 3
            };
            let fits = match byte {
                // The apostrophe.
                0x92 => before.is_some_and(|b| b.is_ascii_alphabetic()) && after.is_ascii_alphabetic(),
                // The en dash and the em dash.
                0x96 | 0x97 => before.is_some_and(|b| b.is_ascii_alphanumeric()) && after.is_ascii_alphabetic(),
                // The ellipsis.
                0x85 => before.is_none_or(|b| b.is_ascii()) && after.is_ascii_alphabetic(),
                // š, œ and ž, within a word or at the head of one.
                0x9A | 0x9C | 0x9E => (begins_word || before.is_some_and(|b| b.is_ascii_alphabetic())) && after.is_ascii_lowercase() && in_word(),
                // Š, Œ and Ž, at the head of a word or within one in capitals.
                0x8A | 0x8C | 0x8E => (begins_word || before.is_some_and(|b| b.is_ascii_uppercase())) && after.is_ascii_alphabetic() && in_word(),
                _ => false,
            };
            if !fits {
                return false;
            }
            seen = true;
        }
    }
    seen
}

/// The Chinese double-byte sets, by the validity of the bytes as ZBar
/// checks them, and only on three characters in a row so that accented
/// Latin text is not taken for them. GB 2312 takes both bytes from 0xA1
/// to 0xFE; Big5 takes a lead from 0xA1 to 0xF9 and a trail from 0x40 to
/// 0x7E or 0xA1 to 0xFE; GBK extends GB 2312 with leads down to 0x81,
/// which ISO-8859-1 text never contains, so for those validity is enough
/// once the bytes are known not to be Shift-JIS, which has its commonest
/// characters there, nor windows-1252 text with its typography there
/// ([`windows_1252_shaped`]): the caller asks both first.
fn double_byte_evidence(runs: &[Run]) -> Option<Codec> {
    let longest_run = |lead: &dyn Fn(u8) -> bool, trail: &dyn Fn(u8) -> bool| -> Option<usize> {
        let mut longest = 0;
        for Run { bytes: run, after_gap, before_gap } in runs {
            // A run that follows a gap may start in the middle of a character.
            let mut i = usize::from(*after_gap && run.first().is_some_and(|&b| trail(b) && !lead(b)));
            let mut chars = 0;
            while i < run.len() {
                let byte = run[i];
                if byte < 0x80 {
                    chars = 0;
                    i += 1;
                } else if lead(byte) && run.get(i + 1).is_some_and(|&t| trail(t)) {
                    chars += 1;
                    longest = longest.max(chars);
                    i += 2;
                } else if i + 1 == run.len() && lead(byte) && *before_gap {
                    // Cut short by the gap at the far end.
                    i += 1;
                } else {
                    return None;
                }
            }
        }
        Some(longest)
    };
    let gb2312 = longest_run(&|b| (0xA1..=0xF7).contains(&b), &|b| (0xA1..=0xFE).contains(&b));
    if gb2312.is_some_and(|n| n >= 3) {
        return Some(Codec::Rs(encoding_rs::GB18030));
    }
    let big5 = longest_run(&|b| (0xA1..=0xF9).contains(&b), &|b| (0x40..=0x7E).contains(&b) || (0xA1..=0xFE).contains(&b));
    if big5.is_some_and(|n| n >= 3) {
        return Some(Codec::Rs(encoding_rs::BIG5));
    }
    let has_gbk_lead = runs.iter().flat_map(|run| &run.bytes).any(|&b| (0x81..=0xA0).contains(&b));
    let gbk = longest_run(&|b| (0x81..=0xFE).contains(&b), &|b| (0x40..=0xFE).contains(&b) && b != 0x7F);
    if has_gbk_lead && gbk.is_some_and(|n| n >= 1) {
        return Some(Codec::Rs(encoding_rs::GB18030));
    }
    None
}

/// The run is UTF-8, less a character a gap has cut at either end of it.
/// A sequence left unfinished at the end of the run may be finished past
/// a gap, and only past one: where the data itself ends there, it is not
/// UTF-8. ISO-8859-1 text that ends in an accented letter ends in a byte
/// UTF-8 begins a longer character with, and `José` read as UTF-8 with
/// its last letter lost.
fn utf8_plausible(run: &Run) -> bool {
    let bytes = &run.bytes;
    let skip = if run.after_gap { bytes.iter().take(3).take_while(|&&b| b & 0xC0 == 0x80).count() } else { 0 };
    match std::str::from_utf8(&bytes[skip..]) {
        Ok(_) => true,
        Err(e) => run.before_gap && e.error_len().is_none(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::tables::Mode;

    fn byte_segment(bytes: &[Option<u8>], certainty: Certainty) -> Segment {
        let cells = bytes
            .iter()
            .enumerate()
            .map(|(i, &value)| Cell {
                value,
                certainty: if value.is_some() { certainty } else { Certainty::Unknown },
                bits: 8 * i..8 * i + 8,
            })
            .collect();
        Segment {
            mode: Mode::Byte,
            header: 0..0,
            payload: 0..0,
            count: Some(bytes.len()),
            header_certainty: Certainty::Verified,
            content: SegmentContent::Bytes(cells),
            damaged: false,
            truncated: false,
            header_inferred: false,
        }
    }

    fn known(bytes: &[u8]) -> Vec<Option<u8>> {
        bytes.iter().map(|&b| Some(b)).collect()
    }

    fn control(content: SegmentContent) -> Segment {
        Segment {
            mode: Mode::Eci,
            header: 0..0,
            payload: 0..0,
            count: None,
            header_certainty: Certainty::Verified,
            content,
            damaged: false,
            truncated: false,
            header_inferred: false,
        }
    }

    #[test]
    fn utf8_is_detected() {
        let seg = byte_segment(&known("héllo ✓".as_bytes()), Certainty::Verified);
        let report = decode_text(&[seg], Charset::Auto);
        assert_eq!(report.plain(), "héllo ✓");
        assert_eq!(report.encoding.as_deref(), Some("UTF-8 (guessed)"));
        assert_eq!(report.spans.len(), 1);
    }

    #[test]
    fn latin1_fallback_is_true_latin1() {
        // "café élan": not UTF-8, and although E9 6C is a valid Shift-JIS
        // pair, one stray double-byte character is not enough to believe it.
        let seg = byte_segment(&known(b"caf\xE9 \xE9lan"), Certainty::Verified);
        let report = decode_text(&[seg], Charset::Auto);
        assert_eq!(report.plain(), "café élan");
        assert_eq!(report.encoding.as_deref(), Some("ISO-8859-1 (guessed)"));
        // 0x80 is a C1 control in ISO-8859-1, not the euro sign of windows-1252.
        let seg = byte_segment(&known(&[0x80]), Certainty::Verified);
        assert_eq!(decode_text(&[seg], Charset::Latin1).plain(), "\u{80}");
        let seg = byte_segment(&known(b"\x80 \x93ok\x94"), Certainty::Verified);
        let report = decode_text(&[seg], Charset::Windows1252);
        assert_eq!(report.plain(), "€ “ok”");
        assert_eq!(report.encoding.as_deref(), Some("windows-1252 (chosen)"));
    }

    #[test]
    fn text_that_ends_in_an_accented_letter_is_not_utf8() {
        // 0xE9 begins a three-byte character in UTF-8, and nothing
        // follows it: the data ends there, and is ISO-8859-1.
        for (bytes, text) in [(&b"Jos\xE9"[..], "José"), (b"https://example.com/caf\xE9", "https://example.com/café"), (b"Fu\xDF", "Fuß")] {
            let report = decode_text(&[byte_segment(&known(bytes), Certainty::Verified)], Charset::Auto);
            assert_eq!((report.plain().as_str(), report.encoding.as_deref()), (text, Some("ISO-8859-1 (guessed)")), "{bytes:02X?}");
        }
        // The same bytes before a byte that could not be read may be the
        // start of a character the gap cut, and stay UTF-8, as a run that
        // begins with the end of one does after a gap.
        let mut bytes = known(b"Jos\xC3");
        bytes.push(None);
        bytes.extend(known(b"\xA9 Mar\xC3\xADa"));
        let report = decode_text(&[byte_segment(&bytes, Certainty::Verified)], Charset::Auto);
        assert_eq!(report.encoding.as_deref(), Some("UTF-8 (guessed)"));
        assert!(report.plain().ends_with(" María"), "{:?}", report.plain());
        // A segment the symbol cuts short ends as a gap would end it.
        let mut cut = byte_segment(&known("Jos\u{E9}".as_bytes().split_last().unwrap().1), Certainty::Verified);
        cut.truncated = true;
        assert_eq!(decode_text(&[cut], Charset::Auto).encoding.as_deref(), Some("UTF-8 (guessed)"));
        // And a double-byte set is known by the characters before a gap
        // that cuts the last of them, where it once took a gap before the
        // run to excuse a character cut after it.
        let (encoded, _, _) = encoding_rs::GB18030.encode("中文二维码");
        let mut bytes = known(&encoded[..encoded.len() - 1]);
        bytes.push(None);
        let report = decode_text(&[byte_segment(&bytes, Certainty::Verified)], Charset::Auto);
        assert_eq!(report.encoding.as_deref(), Some("gb18030 (guessed)"));
        assert!(report.plain().starts_with("中文二维"), "{:?}", report.plain());
    }

    #[test]
    fn a_symbol_or_two_among_ascii_is_latin1_and_not_katakana() {
        // Each of these decodes as Shift-JIS, its one high byte a
        // half-width katakana, and a tenth or more of each is bytes
        // Latin-1 has as symbols.
        for (bytes, text) in [
            (&b"Price: \xA35"[..], "Price: £5"),
            (b"25\xB0C", "25°C"),
            (b"\xA1Hola!", "¡Hola!"),
            (b"\xA92024 ACME", "©2024 ACME"),
            (b"1\xBD cups", "1½ cups"),
            (b"\xABoui\xBB", "«oui»"),
        ] {
            let report = decode_text(&[byte_segment(&known(bytes), Certainty::Verified)], Charset::Auto);
            assert_eq!((report.plain().as_str(), report.encoding.as_deref()), (text, Some("ISO-8859-1 (guessed)")), "{bytes:02X?}");
        }
        // Three katakana in a row are Shift-JIS still, as three
        // double-byte characters are.
        for text in ["ｶﾀｶﾅ", "QRｺｰﾄﾞ", "ﾃﾞｻﾞｲﾝQR\r\nhttp://d-qr.net/ex/", "Google モバイル\r\nhttp://google.jp"] {
            let (bytes, _, _) = encoding_rs::SHIFT_JIS.encode(text);
            let report = decode_text(&[byte_segment(&known(&bytes), Certainty::Verified)], Charset::Auto);
            assert_eq!((report.plain().as_str(), report.encoding.as_deref()), (text, Some("Shift_JIS (guessed)")));
        }
        // What is given up: a message whose only Japanese is a katakana
        // or two reads as Latin-1, and as what it is when the set is
        // chosen. A kanji segment beside it still tips it.
        let (bytes, _, _) = encoding_rs::SHIFT_JIS.encode("ｱｲ");
        let seg = byte_segment(&known(&bytes), Certainty::Verified);
        let report = decode_text(std::slice::from_ref(&seg), Charset::Auto);
        assert_eq!((report.plain().as_str(), report.encoding.as_deref()), ("±²", Some("ISO-8859-1 (guessed)")));
        assert_eq!(decode_text(std::slice::from_ref(&seg), Charset::ShiftJis).plain(), "ｱｲ");
        let mut kanji = byte_segment(&[], Certainty::Verified);
        kanji.mode = Mode::Kanji;
        kanji.content = SegmentContent::Kanji(vec![Cell { value: Some(0x935F), certainty: Certainty::Verified, bits: 0..13 }]);
        assert_eq!(decode_text(&[kanji, seg], Charset::Auto).plain(), "点ｱｲ");
    }

    #[test]
    fn bytes_latin1_has_no_text_for_are_windows_1252() {
        // The euro sign, the curly quotes and the dash of windows-1252
        // are control codes in ISO-8859-1. 0x80 is no byte of Shift-JIS,
        // whose decoder hands it on all the same.
        for (bytes, text) in [(&b"\x805"[..], "€5"), (b"It\x92s \x805 \x96 \x93ok\x94", "It’s €5 – “ok”"), (b"caf\xE9 \x97 5\x80", "café — 5€")] {
            let report = decode_text(&[byte_segment(&known(bytes), Certainty::Verified)], Charset::Auto);
            assert_eq!((report.plain().as_str(), report.encoding.as_deref()), (text, Some("windows-1252 (guessed)")), "{bytes:02X?}");
        }
        // Without such a byte the two sets read alike, and the guess is
        // named as the standard's default.
        let report = decode_text(&[byte_segment(&known(b"caf\xE9 5\xA4"), Certainty::Verified)], Charset::Auto);
        assert_eq!(report.encoding.as_deref(), Some("ISO-8859-1 (guessed)"));
        // Chosen, ISO-8859-1 is itself.
        let report = decode_text(&[byte_segment(&known(b"\x805"), Certainty::Verified)], Charset::Latin1);
        assert_eq!(report.plain(), "\u{80}5");
    }

    #[test]
    fn shift_jis_is_detected_and_selectable() {
        let (bytes, _, _) = encoding_rs::SHIFT_JIS.encode("日本語テスト");
        let seg = byte_segment(&known(&bytes), Certainty::Verified);
        let report = decode_text(std::slice::from_ref(&seg), Charset::Auto);
        assert_eq!(report.plain(), "日本語テスト");
        let forced = decode_text(&[seg], Charset::Latin1);
        assert_ne!(forced.plain(), "日本語テスト");
        assert_eq!(forced.encoding.as_deref(), Some("ISO-8859-1 (chosen)"));
    }

    #[test]
    fn short_shift_jis_is_not_taken_for_gbk() {
        // A character or two with a lead byte from 0x81 to 0x9F: each is
        // a character of GBK too, and read as one when GBK was asked
        // first, `日本` as `擔杮`.
        for text in ["日本", "東京", "東京 2020", "大阪", "5円", "駅", "あ", "愛", "〒100-0001", "http://example.jp/東京", "080-1234-5678 田中"] {
            let (bytes, _, _) = encoding_rs::SHIFT_JIS.encode(text);
            let report = decode_text(&[byte_segment(&known(&bytes), Certainty::Verified)], Charset::Auto);
            assert_eq!((report.plain().as_str(), report.encoding.as_deref()), (text, Some("Shift_JIS (guessed)")));
        }
    }

    #[test]
    fn typography_among_ascii_letters_is_windows_1252() {
        // Each of these is Shift-JIS and GBK from end to end, the mark
        // from 0x80 to 0x9F a lead byte and the letter after it a trail.
        for text in ["It’s ok", "don’t", "WE’RE OPEN", "L’Oréal", "rock’n’roll", "word—word", "A–Z", "…and more", "cœur", "œuvre", "Škoda Octavia", "ŠKODA", "Žižek"] {
            let (bytes, _, lossy) = encoding_rs::WINDOWS_1252.encode(text);
            assert!(!lossy, "{text}");
            let report = decode_text(&[byte_segment(&known(&bytes), Certainty::Verified)], Charset::Auto);
            assert_eq!((report.plain().as_str(), report.encoding.as_deref()), (text, Some("windows-1252 (guessed)")));
        }
        // A kanji whose bytes are such a mark and a letter is a kanji
        // still where Western text would not set the mark so: `谷` is an
        // apostrophe with no letter before it, `市` is `Žs`, a word of
        // two letters, `田` an opening quote, and the second character
        // of `地図` no character of windows-1252.
        for text in ["谷", "市", "田", "都", "地図", "QR地図", "知る", "3個"] {
            let (bytes, _, _) = encoding_rs::SHIFT_JIS.encode(text);
            let report = decode_text(&[byte_segment(&known(&bytes), Certainty::Verified)], Charset::Auto);
            assert_eq!((report.plain().as_str(), report.encoding.as_deref()), (text, Some("Shift_JIS (guessed)")));
        }
        // What is not known is not taken to help: after a byte that
        // could not be read, the Š need not begin a word.
        let mut bytes = vec![None];
        bytes.extend(known(b"\x8Akoda x\x96y"));
        let report = decode_text(&[byte_segment(&bytes, Certainty::Verified)], Charset::Auto);
        assert_eq!(report.encoding.as_deref(), Some("Shift_JIS (guessed)"));
    }

    #[test]
    fn chinese_double_byte_sets_are_detected_by_validity() {
        let (bytes, _, _) = encoding_rs::GB18030.encode("中文二维码");
        let report = decode_text(&[byte_segment(&known(&bytes), Certainty::Verified)], Charset::Auto);
        assert_eq!(report.plain(), "中文二维码");
        assert_eq!(report.encoding.as_deref(), Some("gb18030 (guessed)"));
        let (bytes, _, _) = encoding_rs::BIG5.encode("繁體中文");
        let report = decode_text(&[byte_segment(&known(&bytes), Certainty::Verified)], Charset::Auto);
        assert_eq!(report.plain(), "繁體中文");
        assert_eq!(report.encoding.as_deref(), Some("Big5 (guessed)"));
        // A GBK-only lead byte settles it with fewer characters, where
        // the bytes are not Shift-JIS as well: 0x86 leads nothing there.
        let (bytes, _, _) = encoding_rs::GB18030.encode("喆");
        assert!(bytes[0] < 0xA1, "{bytes:?}");
        let report = decode_text(&[byte_segment(&known(&bytes), Certainty::Verified)], Charset::Auto);
        assert_eq!((report.plain().as_str(), report.encoding.as_deref()), ("喆", Some("gb18030 (guessed)")));
        // Where they are, Shift-JIS has them: GBK's 丂 is 0x81 0x40, the
        // ideographic space, and reads as itself when the set is chosen.
        let (bytes, _, _) = encoding_rs::GB18030.encode("丂");
        let seg = byte_segment(&known(&bytes), Certainty::Verified);
        let report = decode_text(std::slice::from_ref(&seg), Charset::Auto);
        assert_eq!((report.plain().as_str(), report.encoding.as_deref()), ("\u{3000}", Some("Shift_JIS (guessed)")));
        assert_eq!(decode_text(&[seg], Charset::Gb18030).plain(), "丂");
        // Two accented letters are still Latin-1.
        let seg = byte_segment(&known(b"caf\xE9 \xE9lan \xC4\xE4"), Certainty::Verified);
        assert_eq!(decode_text(&[seg], Charset::Auto).encoding.as_deref(), Some("ISO-8859-1 (guessed)"));
    }

    #[test]
    fn hanzi_cells_decode() {
        let mut seg = byte_segment(&[], Certainty::Verified);
        seg.mode = Mode::Hanzi;
        seg.content = SegmentContent::Hanzi(vec![
            Cell { value: Some(0xD6D0), certainty: Certainty::Verified, bits: 0..13 },
            Cell { value: None, certainty: Certainty::Unknown, bits: 13..26 },
            Cell { value: Some(0xCEC4), certainty: Certainty::Unverified, bits: 26..39 },
        ]);
        let report = decode_text(&[seg], Charset::Auto);
        assert_eq!(report.plain(), "中\u{FFFD}文");
        assert_eq!(report.parity, None);
        assert_eq!(report.symbology.as_deref(), Some("]Q1"));
    }

    #[test]
    fn fnc1_puts_group_separators_in_alphanumeric_data() {
        let alnum = |text: &str| {
            let cells = text
                .chars()
                .enumerate()
                .map(|(i, ch)| Cell { value: Some(ch), certainty: Certainty::Verified, bits: 6 * i..6 * i + 6 })
                .collect();
            Segment {
                mode: Mode::Alphanumeric,
                header: 0..0,
                payload: 0..0,
                count: Some(text.len()),
                header_certainty: Certainty::Verified,
                content: SegmentContent::Chars(cells),
                damaged: false,
                truncated: false,
                header_inferred: false,
            }
        };
        let report = decode_text(&[control(SegmentContent::Fnc1First), alnum("0112345%10ABC%%5%")], Charset::Auto);
        assert_eq!(report.plain(), "0112345\u{1D}10ABC%5\u{1D}");
        assert_eq!(report.symbology.as_deref(), Some("]Q3"));
        // The message is read as GS1 element strings too, and this one is
        // not a well-formed one: a GTIN needs fourteen digits.
        let gs1 = report.gs1.as_ref().unwrap();
        assert!(gs1.problems.iter().any(|p| p.contains("(01)")), "{:?}", gs1.problems);
        assert!(report.notes.iter().any(|n| n.starts_with("GS1:")));
        assert_eq!(report.parity, Some(b"0112345%10ABC%%5%".iter().fold(0, |a, &b| a ^ b)));
        // Without FNC1 the percent signs are themselves.
        let report = decode_text(&[alnum("50%%")], Charset::Auto);
        assert_eq!(report.plain(), "50%%");
        assert_eq!(report.symbology.as_deref(), Some("]Q1"));
        // FNC1 in second position: the indicator is transmitted ahead of
        // the data and named, and ECI lifts the modifier.
        let report = decode_text(
            &[control(SegmentContent::Fnc1Second(165)), control(SegmentContent::Eci(Some(26))), alnum("A%B")],
            Charset::Auto,
        );
        assert_eq!(report.plain(), "AA\u{1D}B");
        assert_eq!(report.hex(), "41 41 25 42");
        assert_eq!(report.parity, Some(b'A' ^ b'%' ^ b'B'), "the indicator is not message data");
        assert_eq!(report.symbology.as_deref(), Some("]Q6"));
        assert!(report.notes.iter().any(|n| n.contains("application indicator A")), "{:?}", report.notes);
        let report = decode_text(&[control(SegmentContent::Fnc1Second(150))], Charset::Auto);
        assert!(report.notes.iter().any(|n| n.contains("not defined")), "{:?}", report.notes);
    }

    #[test]
    fn gs1_element_strings_read_as_hri() {
        // GTIN, then a batch of variable length ended by GS, then a fixed
        // length best-before date, then a serial number running to the end.
        let text = format!("01095011010209171012345{GROUP_SEPARATOR}17250630211234ABC");
        let gs1 = gs1_elements(&text);
        assert_eq!(gs1.hri, "(01)09501101020917(10)12345(17)250630(21)1234ABC");
        assert!(gs1.problems.is_empty(), "{:?}", gs1.problems);
        // A separator after a fixed-length field is tolerated; a fixed
        // field cut short by one, an unknown identifier, a field cut short
        // by the end of the data, and letters in a numeric field are not.
        let gs1 = gs1_elements(&format!("1725063{GROUP_SEPARATOR}10AB"));
        assert_eq!(gs1.hri, "(17)25063(10)AB");
        assert!(gs1.problems.iter().any(|p| p.contains("(17) needs 6 characters of data and has 5")), "{:?}", gs1.problems);
        assert!(gs1_elements("17250630").problems.is_empty());
        assert!(gs1_elements("1725063A").problems.iter().any(|p| p.contains("must be digits")));
        let gs1 = gs1_elements("8912345");
        assert_eq!(gs1.hri, "8912345");
        assert!(gs1.problems[0].contains("89"));
        let gs1 = gs1_elements("0109501");
        assert!(gs1.problems.iter().any(|p| p.contains("(01) needs 14")), "{:?}", gs1.problems);
        assert!(gs1_elements("3103001234").problems.is_empty());
        assert!(gs1_elements("01095011010209\u{FFFD}7").problems.iter().any(|p| p.contains("unreadable")));
        assert_eq!(control_picture(GROUP_SEPARATOR), Some('␝'));
        assert_eq!(control_picture('\r'), Some('␍'));
        assert_eq!(control_picture('\u{7F}'), Some('␡'));
        assert_eq!(control_picture('a'), None);
    }

    #[test]
    fn structured_append_parity_is_reported() {
        let seg = byte_segment(&known(b"abc"), Certainty::Verified);
        let own = b'a' ^ b'b' ^ b'c';
        let report =
            decode_text(&[control(SegmentContent::StructuredAppend { index: 2, total: 3, parity: 0x5A }), seg.clone()], Charset::Auto);
        assert_eq!(report.parity, Some(own));
        assert!(report.notes[0].contains(&format!("{own:#04X}")), "{}", report.notes[0]);
        let report =
            decode_text(&[control(SegmentContent::StructuredAppend { index: 1, total: 1, parity: own }), seg], Charset::Auto);
        assert!(report.notes[0].contains("same value"), "{}", report.notes[0]);
        let mut bytes = known(b"ab");
        bytes.push(None);
        let report = decode_text(&[byte_segment(&bytes, Certainty::Verified)], Charset::Auto);
        assert_eq!(report.parity, None);
        assert_eq!(decode_text(&[], Charset::Auto).symbology, None);
        // A Model 1 symbol has its own identifier, and keeps the modifier
        // of what its stream carries should that be FNC1 or ECI.
        let seg = byte_segment(&known(b"abc"), Certainty::Verified);
        assert_eq!(decode_text_of(Family::Model1, &[], Charset::Auto).symbology, None);
        let plain = std::slice::from_ref(&seg);
        assert_eq!(decode_text_of(Family::Model1, plain, Charset::Auto).symbology.as_deref(), Some("]Q0"));
        assert_eq!(decode_text_of(Family::Micro, plain, Charset::Auto).symbology.as_deref(), Some("]Q1"));
        let fnc1 = [control(SegmentContent::Fnc1First), seg];
        assert_eq!(decode_text_of(Family::Model1, &fnc1, Charset::Auto).symbology.as_deref(), Some("]Q3"));
    }

    #[test]
    fn unknown_bytes_become_placeholders_between_decoded_runs() {
        let mut bytes = known("añb".as_bytes());
        bytes.push(None);
        bytes.extend(known(b"cd"));
        let report = decode_text(&[byte_segment(&bytes, Certainty::Unverified)], Charset::Auto);
        assert_eq!(report.plain(), "añb\u{FFFD}cd");
        assert_eq!(report.spans.len(), 3);
        assert_eq!(report.spans[1].certainty, Certainty::Unknown);
        assert_eq!(report.counts(), (0, 5, 1));
    }

    #[test]
    fn multibyte_characters_take_the_weakest_byte() {
        let bytes = "aé".as_bytes();
        let cells = vec![
            Cell { value: Some(bytes[0]), certainty: Certainty::Verified, bits: 0..8 },
            Cell { value: Some(bytes[1]), certainty: Certainty::Verified, bits: 8..16 },
            Cell { value: Some(bytes[2]), certainty: Certainty::Unverified, bits: 16..24 },
        ];
        let mut seg = byte_segment(&[], Certainty::Verified);
        seg.content = SegmentContent::Bytes(cells);
        let report = decode_text(&[seg], Charset::Utf8);
        assert_eq!(
            report.spans,
            [
                TextSpan { text: "a".into(), certainty: Certainty::Verified },
                TextSpan { text: "é".into(), certainty: Certainty::Unverified },
            ]
        );
    }

    #[test]
    fn eci_overrides_guessing() {
        // 0xC0 0xC1 is invalid UTF-8; under ECI 7 (ISO-8859-5) it is Cyrillic.
        let report = decode_text(
            &[control(SegmentContent::Eci(Some(7))), byte_segment(&known(&[0xC0, 0xC1]), Certainty::Verified)],
            Charset::Auto,
        );
        assert_eq!(report.plain(), "РС");
        assert_eq!(report.encoding.as_deref(), Some("ISO-8859-5 (ECI)"));
        let unsupported = decode_text(
            &[control(SegmentContent::Eci(Some(123456))), byte_segment(&known(b"x"), Certainty::Verified)],
            Charset::Auto,
        );
        assert_eq!(unsupported.plain(), "x");
        assert_eq!(unsupported.notes.len(), 1);
        assert!(unsupported.notes[0].contains("not supported"), "{:?}", unsupported.notes);
        // Parts 9 and 11, which encoding_rs decodes as windows-1254 and
        // windows-874, are still named as the symbol declared them.
        let turkish = decode_text(
            &[control(SegmentContent::Eci(Some(11))), byte_segment(&known(&[0xD0, 0xFD]), Certainty::Verified)],
            Charset::Auto,
        );
        assert_eq!((turkish.plain().as_str(), turkish.encoding.as_deref()), ("Ğı", Some("ISO-8859-9 (ECI)")));
        let thai = decode_text(
            &[control(SegmentContent::Eci(Some(13))), byte_segment(&known(&[0xA1]), Certainty::Verified)],
            Charset::Auto,
        );
        assert_eq!((thai.plain().as_str(), thai.encoding.as_deref()), ("ก", Some("ISO-8859-11 (ECI)")));
        // 14 and 19 lie inside the ISO-8859 run but name no set.
        for eci in [14, 19] {
            let reserved = decode_text(
                &[control(SegmentContent::Eci(Some(eci))), byte_segment(&known(b"\xE9"), Certainty::Verified)],
                Charset::Auto,
            );
            assert_eq!((reserved.plain().as_str(), reserved.notes.len()), ("é", 1), "ECI {eci}");
            assert!(reserved.notes[0].contains("reserved") && !reserved.notes[0].contains("not supported"), "{:?}", reserved.notes);
        }
    }

    #[test]
    fn a_declared_eci_outranks_a_chosen_charset() {
        // The symbol says ISO-8859-5 and the user UTF-8, in which these
        // bytes are nothing: the symbol knows.
        let cyrillic = [control(SegmentContent::Eci(Some(7))), byte_segment(&known(&[0xC0, 0xC1]), Certainty::Verified)];
        for charset in [Charset::Auto, Charset::Utf8, Charset::ShiftJis, Charset::Eci(9)] {
            let report = decode_text(&cyrillic, charset);
            assert_eq!((report.plain().as_str(), report.encoding.as_deref()), ("РС", Some("ISO-8859-5 (ECI)")), "{charset:?}");
            assert_eq!(encode_as_read(&cyrillic, 1, charset, "С"), Ok(vec![0xC1]), "{charset:?}");
        }
        // Byte data before the designator is not what it declares.
        let before = [byte_segment(&known(&[0xC0, 0xC1]), Certainty::Verified), control(SegmentContent::Eci(Some(7))), byte_segment(&known(&[0xC0]), Certainty::Verified)];
        let report = decode_text(&before, Charset::Latin1);
        assert_eq!((report.plain().as_str(), report.encoding.as_deref()), ("ÀÁР", Some("ISO-8859-1 (chosen), ISO-8859-5 (ECI)")));
        assert_eq!(encode_as_read(&before, 0, Charset::Latin1, "Á"), Ok(vec![0xC1]));
        // A designator that names no set this reader has declares nothing
        // to follow: the set chosen stands, and the note says what the
        // bytes are shown as.
        for (eci, why) in [(123456, "not supported"), (14, "reserved")] {
            let unnamed = [control(SegmentContent::Eci(Some(eci))), byte_segment(&known(&[0xB1, 0xB2]), Certainty::Verified)];
            let report = decode_text(&unnamed, Charset::ShiftJis);
            assert_eq!((report.plain().as_str(), report.encoding.as_deref()), ("ｱｲ", Some("Shift_JIS (chosen)")), "ECI {eci}");
            assert!(report.notes[0].contains(why) && report.notes[0].ends_with("bytes shown as Shift_JIS."), "{:?}", report.notes);
            let report = decode_text(&unnamed, Charset::Auto);
            assert_eq!((report.plain().as_str(), report.encoding.as_deref()), ("±²", Some("ISO-8859-1 (ECI)")), "ECI {eci}");
            assert!(report.notes[0].ends_with("bytes shown as ISO-8859-1."), "{:?}", report.notes);
            assert_eq!(encode_as_read(&unnamed, 1, Charset::ShiftJis, "ｱ"), Ok(vec![0xB1]));
            assert_eq!(encode_as_read(&unnamed, 1, Charset::Auto, "±"), Ok(vec![0xB1]));
        }
    }

    #[test]
    fn kanji_cells_decode() {
        let mut seg = byte_segment(&[], Certainty::Verified);
        seg.mode = Mode::Kanji;
        seg.content = SegmentContent::Kanji(vec![
            Cell { value: Some(0x935F), certainty: Certainty::Verified, bits: 0..13 },
            Cell { value: None, certainty: Certainty::Unknown, bits: 13..26 },
            Cell { value: Some(0xE4AA), certainty: Certainty::Verified, bits: 26..39 },
        ]);
        assert_eq!(decode_text(&[seg], Charset::Auto).plain(), "点\u{FFFD}茗");
    }

    #[test]
    fn every_offered_charset_has_a_codec_and_a_name() {
        let mut labels: Vec<&str> = Vec::new();
        for charset in Charset::ALL {
            assert_eq!(charset.codec().is_some(), charset != Charset::Auto, "{charset:?}");
            assert_ne!(charset.label(), "unsupported", "{charset:?}");
            assert!(!labels.contains(&charset.label()), "{charset:?} repeats {}", charset.label());
            labels.push(charset.label());
        }
        // A set chosen by its ECI number reads as the ECI itself would.
        let report = decode_text(&[byte_segment(&known(&[0xC0, 0xC1]), Certainty::Verified)], Charset::Eci(7));
        assert_eq!((report.plain().as_str(), report.encoding.as_deref()), ("РС", Some("ISO-8859-5 (chosen)")));
        assert_eq!(Charset::Eci(14).label(), "unsupported");
        assert!(Charset::Eci(14).codec().is_none());
    }
}
