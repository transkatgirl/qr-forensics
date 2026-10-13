//! The message in other dress: its bytes, its control characters spelt
//! out, its ECI designators written in, and the parts of a structured
//! append sequence put back together.

mod common;

use common::*;
use qr_forensics::decode::bitstream::{Certainty, SegmentContent};
use qr_forensics::decode::codewords::{Pin, PinSource, Pins};
use qr_forensics::decode::pipeline::{DecodeOptions, DecodeReport, decode};
use qr_forensics::decode::repair::repaired_grid;
use qr_forensics::decode::sequence::{ParityCheck, Part, Sequence};
use qr_forensics::decode::text::UNKNOWN_CHAR;
use qr_forensics::symbol::grid::ModuleGrid;
use qr_forensics::symbol::{EcLevel, SymbolVersion};
use qrcode_generator::Segment;
use qrcode_generator::qr::EciAssignment;

#[test]
fn the_message_is_shown_as_bytes_and_escaped_and_with_eci() {
    let layout = layout(SymbolVersion::Qr(3), EcLevel::L);
    let grid = encode(
        SymbolVersion::Qr(3),
        EcLevel::L,
        Some(0),
        &[
            Segment::numeric("42").unwrap(),
            Segment::bytes(b"a\x1db\n\\c"),
            Segment::eci(EciAssignment::new(7).unwrap()),
            Segment::bytes([0xBF, 0xE0, 0xD8, 0xD2, 0xD5, 0xE2]),
        ],
    )
    .unwrap();
    let report = decode_plain(&grid, &layout, 0);
    assert!(report.is_trustworthy());
    let text = &report.text;
    assert_eq!(text.plain(), "42a\u{1D}b\n\\cПривет");
    assert_eq!(text.escaped(), "42a<GS>b<LF>\\cПривет");
    assert_eq!(text.hex(), "34 32 61 1D 62 0A 5C 63 BF E0 D8 D2 D5 E2");
    assert_eq!(text.raw_bytes().as_deref(), Some(&b"42a\x1db\n\\c\xBF\xE0\xD8\xD2\xD5\xE2"[..]));
    assert_eq!(text.eci, [(8, 7)]);
    assert_eq!(text.eci_text, "42a\u{1D}b\n\\\\c\\000007Привет");
    // An unreadable byte shows as such in every dress and stops the raw
    // export.
    let mut torn = grid.clone();
    // Two codewords inside the byte payload, and every error correction
    // codeword, so that nothing can fill them in.
    for &cw in &layout.data_stream[5..7] {
        erase_codeword(&mut torn, &layout, cw);
    }
    for &other in &layout.blocks[0].codewords[layout.blocks[0].data_len..] {
        erase_codeword(&mut torn, &layout, other);
    }
    let report = decode_plain(&torn, &layout, 0);
    assert!(report.text.bytes.iter().any(Option::is_none));
    assert!(report.text.hex().contains("??"));
    assert!(report.text.escaped().contains("<?>"));
    assert_eq!(report.text.raw_bytes(), None);
}

/// A symbol whose data codewords are given outright: pinned into a blank
/// grid, solved from the pins alone, and written back with its error
/// correction and fixed patterns.
fn symbol_from_data(version: SymbolVersion, ec: EcLevel, mask: u8, data: &[u8]) -> (ModuleGrid, DecodeReport) {
    let layout = layout(version, ec);
    assert!(data.len() <= layout.data_stream.len(), "{} data codewords for {} slots", data.len(), layout.data_stream.len());
    let mut pins = Pins::new();
    for (j, &cw) in layout.data_stream.iter().enumerate() {
        let value = data.get(j).copied().unwrap_or_else(|| [0xEC, 0x11][(j - data.len()) % 2]);
        pins.insert(cw, Pin::full(value, PinSource::Manual));
    }
    let blank = layout.func.blank_grid();
    let report = decode(&blank, &layout, mask, &pins, &DecodeOptions::default());
    assert!(report.is_complete(), "pins alone determine every block");
    let grid = repaired_grid(&blank, &layout, mask, &report);
    let report = decode_plain(&grid, &layout, mask);
    assert!(report.is_trustworthy());
    (grid, report)
}

/// Data codewords for one part of a structured append message in a QR
/// symbol: the header, then the text in byte mode, then the terminator.
fn part_data(index: u8, total: u8, parity: u8, text: &[u8]) -> Vec<u8> {
    let mut bits: Vec<bool> = Vec::new();
    let mut push = |value: u32, n: usize| {
        for i in (0..n).rev() {
            bits.push(value >> i & 1 != 0);
        }
    };
    push(0b0011, 4);
    push((index - 1) as u32, 4);
    push((total - 1) as u32, 4);
    push(parity as u32, 8);
    push(0b0100, 4);
    push(text.len() as u32, 8);
    for &b in text {
        push(b as u32, 8);
    }
    push(0, 4);
    while !bits.len().is_multiple_of(8) {
        bits.push(false);
    }
    bits.chunks(8).map(|byte| byte.iter().fold(0u8, |a, &b| a << 1 | u8::from(b))).collect()
}

#[test]
fn structured_append_parts_are_collected_and_their_parity_verified() {
    let message = b"The quick brown fox jumps over the lazy dog";
    let parity = message.iter().fold(0u8, |a, &b| a ^ b);
    let pieces: [&[u8]; 3] = [&message[..15], &message[15..30], &message[30..]];
    let mut sequence = Sequence::default();
    let mut reports = Vec::new();
    for (i, piece) in pieces.iter().enumerate() {
        let data = part_data(i as u8 + 1, 3, parity, piece);
        let (_, report) = symbol_from_data(SymbolVersion::Qr(2), EcLevel::L, 3, &data);
        assert!(report.parse.segments.iter().any(|s| matches!(s.content, SegmentContent::StructuredAppend { .. })));
        assert_eq!(report.text.plain().as_bytes(), *piece);
        reports.push(report);
    }
    // A symbol with no header makes no part.
    let plain = decode_plain(
        &encode(SymbolVersion::Qr(2), EcLevel::L, Some(0), &[Segment::bytes("alone")]).unwrap(),
        &layout(SymbolVersion::Qr(2), EcLevel::L),
        0,
    );
    assert!(Part::from_report(&plain, "alone").is_none());

    // Parts arrive out of order and are put in order.
    for i in [2usize, 0, 1] {
        let part = Part::from_report(&reports[i], format!("photo {}", i + 1)).unwrap();
        assert_eq!((part.index, part.total, part.parity), (i as u8 + 1, 3, parity));
        assert!(part.is_verified());
        if i == 0 {
            assert_eq!(sequence.check(), ParityCheck::Incomplete { have: 1, total: 3 });
        }
        sequence.add(part);
    }
    assert!(sequence.missing().is_empty());
    assert_eq!(sequence.check(), ParityCheck::Complete { computed: parity, claimed: parity });
    assert!(sequence.check().is_ok());
    let assembled = sequence.assembled();
    assert_eq!(assembled.plain().as_bytes(), message);
    assert!(assembled.spans.iter().all(|s| s.certainty == Certainty::Verified));
    assert_eq!(assembled.parity, Some(parity));
    assert_eq!(assembled.symbology.as_deref(), Some("]Q1"));

    // A part from another message, with the same count and parity byte by
    // chance, is caught by the parity.
    let impostor = part_data(2, 3, parity, b"something else!");
    let (_, report) = symbol_from_data(SymbolVersion::Qr(2), EcLevel::L, 3, &impostor);
    assert!(sequence.add(Part::from_report(&report, "stray").unwrap()));
    assert!(matches!(sequence.check(), ParityCheck::Complete { .. }));
    assert!(!sequence.check().is_ok());

    // A part whose count disagrees cannot belong.
    let other = part_data(1, 2, parity, b"two parts");
    let (_, report) = symbol_from_data(SymbolVersion::Qr(2), EcLevel::L, 3, &other);
    sequence.add(Part::from_report(&report, "other").unwrap());
    assert_eq!(sequence.check(), ParityCheck::Inconsistent);
}

#[test]
fn eci_code_page_437_utf32_and_binary_are_understood() {
    // ECI 2 is IBM code page 437, 34 and 35 are UTF-32 big- and
    // little-endian, 899 says the bytes are binary: the sets ZXing's
    // CharacterSetECI and zxing-cpp's ECI table know.
    let layout = layout(SymbolVersion::Qr(3), EcLevel::L);
    let grid = encode(
        SymbolVersion::Qr(3),
        EcLevel::L,
        Some(1),
        &[
            Segment::eci(EciAssignment::new(2).unwrap()),
            Segment::bytes([0x82, 0xB0, 0xE1, 0x41]),
            Segment::eci(EciAssignment::new(34).unwrap()),
            Segment::bytes([0, 0, 0, 0x41, 0, 1, 0xF6, 0, 0, 0x11, 0, 0]),
            Segment::eci(EciAssignment::new(35).unwrap()),
            Segment::bytes([0x42, 0, 0, 0]),
            Segment::eci(EciAssignment::new(899).unwrap()),
            Segment::bytes([0xFF, 0x00]),
        ],
    )
    .unwrap();
    let report = decode_plain(&grid, &layout, 1);
    assert!(report.is_trustworthy());
    // 0x110000 is past the last code point and shows as unreadable.
    assert_eq!(report.text.plain(), format!("é░ßAA😀{UNKNOWN_CHAR}B\u{FF}\0"));
    let encoding = report.text.encoding.clone().unwrap();
    for name in ["IBM437 (ECI)", "UTF-32BE (ECI)", "UTF-32LE (ECI)", "binary (ECI)"] {
        assert!(encoding.contains(name), "{encoding}");
    }
    assert!(report.text.notes.iter().any(|n| n.contains("ECI 899")), "{:?}", report.text.notes);
    assert!(report.text.notes.iter().all(|n| !n.contains("not supported")), "{:?}", report.text.notes);
}

/// Data codewords of a QR symbol with FNC1 in second position: the
/// indicator, then the text in byte mode, then the terminator.
fn fnc1_second_data(indicator: u8, text: &[u8]) -> Vec<u8> {
    let mut bits: Vec<bool> = Vec::new();
    let mut push = |value: u32, n: usize| {
        for i in (0..n).rev() {
            bits.push(value >> i & 1 != 0);
        }
    };
    push(0b1001, 4);
    push(indicator as u32, 8);
    push(0b0100, 4);
    push(text.len() as u32, 8);
    for &b in text {
        push(b as u32, 8);
    }
    push(0, 4);
    while !bits.len().is_multiple_of(8) {
        bits.push(false);
    }
    bits.chunks(8).map(|byte| byte.iter().fold(0u8, |a, &b| a << 1 | u8::from(b))).collect()
}

#[test]
fn fnc1_in_second_position_transmits_the_application_indicator() {
    // ISO/IEC 18004 7.4.8.3: the indicator goes ahead of the data, as two
    // digits or one letter, as ZBar and zxing-cpp transmit it.
    for (indicator, text) in [(165u8, "Axy"), (42, "42xy"), (222, "zxy")] {
        let (_, report) = symbol_from_data(SymbolVersion::Qr(2), EcLevel::L, 3, &fnc1_second_data(indicator, b"xy"));
        assert_eq!(report.text.plain(), text, "{indicator}");
        assert_eq!(report.text.raw_bytes().unwrap(), text.as_bytes());
        assert_eq!(report.text.symbology.as_deref(), Some("]Q5"));
        // The indicator is not message data: the structured append parity
        // is over the message alone.
        assert_eq!(report.text.parity, Some(b'x' ^ b'y'));
        assert!(report.text.notes.iter().any(|n| n.contains("application indicator")), "{:?}", report.text.notes);
    }
}

#[test]
fn a_byte_order_mark_is_left_out_of_the_text() {
    let layout = layout(SymbolVersion::Qr(1), EcLevel::L);
    let grid = encode(SymbolVersion::Qr(1), EcLevel::L, Some(0), &[Segment::bytes([0xEF, 0xBB, 0xBF, b'h', b'i'])]).unwrap();
    let report = decode_plain(&grid, &layout, 0);
    assert_eq!(report.text.plain(), "hi");
    assert_eq!(report.text.hex(), "EF BB BF 68 69", "the bytes keep it");
    assert_eq!(report.text.encoding.as_deref(), Some("UTF-8 (guessed)"));
    assert!(report.text.notes.iter().any(|n| n.contains("byte order mark")), "{:?}", report.text.notes);
    // UTF-16 by ECI, with its own mark.
    let grid = encode(
        SymbolVersion::Qr(1),
        EcLevel::L,
        Some(0),
        &[Segment::eci(EciAssignment::new(25).unwrap()), Segment::bytes([0xFE, 0xFF, 0, b'A', 0x26, 0x03])],
    )
    .unwrap();
    let report = decode_plain(&grid, &layout, 0);
    assert_eq!(report.text.plain(), "A☃");
    assert!(report.text.notes.iter().any(|n| n.contains("byte order mark")));
}

#[test]
fn utf16_keeps_its_alignment_past_an_unreadable_byte() {
    // "ab☃c" in UTF-16BE under ECI 25: 00 61 00 62 26 03 00 63. Losing the
    // high byte of "b" must cost that character alone, not shift every
    // code unit after it by one byte.
    let (version, ec, mask) = (SymbolVersion::Qr(2), EcLevel::L, 2);
    let layout = layout(version, ec);
    let text = [0x00, 0x61, 0x00, 0x62, 0x26, 0x03, 0x00, 0x63];
    let grid = encode(version, ec, Some(mask), &[Segment::eci(EciAssignment::new(25).unwrap()), Segment::bytes(text)]).unwrap();
    assert_eq!(decode_plain(&grid, &layout, mask).text.plain(), "ab☃c");
    // The ECI header is 12 bits and the byte header 12 more, so byte k of
    // the text is data codeword 3 + k.
    let mut torn = grid.clone();
    erase_codeword(&mut torn, &layout, layout.data_stream[3 + 2]);
    for &cw in &layout.blocks[0].codewords[layout.blocks[0].data_len..] {
        erase_codeword(&mut torn, &layout, cw);
    }
    let report = decode_plain(&torn, &layout, mask);
    assert_eq!(report.text.hex(), "00 61 ?? 62 26 03 00 63");
    assert_eq!(report.text.plain(), format!("a{UNKNOWN_CHAR}☃c"));
    assert_eq!(report.text.counts(), (0, 3, 1));
}

#[test]
fn a_kanji_segment_tips_doubtful_bytes_to_shift_jis() {
    // One double-byte character whose bytes ISO-8859-1 has text for is
    // too little to believe Shift-JIS on its own: E0 40 reads as `à@`.
    // Beside a kanji segment, Shift-JIS is the natural reading, as ZBar
    // takes it.
    let (bytes, _, _) = encoding_rs::SHIFT_JIS.encode("漾");
    assert_eq!(bytes[..], [0xE0, 0x40]);
    let layout = layout(SymbolVersion::Qr(2), EcLevel::L);
    let alone = encode(SymbolVersion::Qr(2), EcLevel::L, Some(0), &[Segment::bytes(&bytes)]).unwrap();
    let report = decode_plain(&alone, &layout, 0);
    assert_eq!(report.text.plain(), "à@");
    let beside = encode(
        SymbolVersion::Qr(2),
        EcLevel::L,
        Some(0),
        &[Segment::kanji("日本").unwrap(), Segment::bytes(&bytes)],
    )
    .unwrap();
    let report = decode_plain(&beside, &layout, 0);
    assert_eq!(report.text.plain(), "日本漾");
    assert_eq!(report.text.encoding.as_deref(), Some("Shift_JIS (guessed)"));
    // A lead byte ISO-8859-1 has only a control code for needs no kanji
    // beside it: 8E 9A is a character of GBK as well, and Shift-JIS is
    // asked first.
    let (bytes, _, _) = encoding_rs::SHIFT_JIS.encode("字");
    assert!((0x81..=0x9F).contains(&bytes[0]), "{bytes:?}");
    let alone = encode(SymbolVersion::Qr(2), EcLevel::L, Some(0), &[Segment::bytes(&bytes)]).unwrap();
    let report = decode_plain(&alone, &layout, 0);
    assert_eq!((report.text.plain().as_str(), report.text.encoding.as_deref()), ("字", Some("Shift_JIS (guessed)")));
}

#[test]
fn every_character_of_the_message_knows_the_bits_it_came_from() {
    let (version, ec) = (SymbolVersion::Qr(4), EcLevel::M);
    let layout = layout(version, ec);
    let grid = encode(
        version,
        ec,
        Some(0),
        &[Segment::numeric("31415").unwrap(), Segment::bytes("naïve é"), Segment::alphanumeric("A1B").unwrap(), Segment::kanji("日本").unwrap()],
    )
    .unwrap();
    let report = decode_plain(&grid, &layout, 0);
    let message = "31415naïve éA1B日本";
    assert_eq!(report.text.plain(), message);
    let sources = &report.text.sources;
    assert_eq!(sources.len(), message.chars().count());
    // In the message's order, each inside the payload of its segment.
    for (source, ch) in sources.iter().zip(message.chars()) {
        let payload = &report.parse.segments[source.segment].payload;
        assert!(payload.start <= source.bits.start && source.bits.end <= payload.end, "{ch:?}: {source:?} outside {payload:?}");
    }
    assert!(sources.windows(2).all(|pair| pair[0].bits.start <= pair[1].bits.start));
    let (segments, cells, widths): (Vec<usize>, Vec<usize>, Vec<usize>) =
        (sources.iter().map(|s| s.segment).collect(), sources.iter().map(|s| s.cell).collect(), sources.iter().map(|s| s.bits.len()).collect());
    assert_eq!(segments, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3]);
    // Digits by the group of three, and the two that are left over; a
    // byte to a character but for the two that take two; alphanumeric
    // characters by the pair, and the odd one; kanji thirteen bits each.
    assert_eq!(widths, [10, 10, 10, 7, 7, 8, 8, 16, 8, 8, 8, 16, 11, 11, 6, 13, 13]);
    // The cell a character begins at: a character of two bytes takes
    // two cells of byte data.
    assert_eq!(cells, [0, 1, 2, 3, 4, 0, 1, 2, 4, 5, 6, 7, 0, 1, 2, 0, 1]);
    // The bits named are the bits that spell the character: with those
    // of the ï unreadable, it alone is lost.
    let mut torn = grid.clone();
    let i_umlaut = &sources[7];
    for bit in i_umlaut.bits.clone() {
        let (cw, place) = layout.stream_bit_location(bit).unwrap();
        let module = layout.codeword_modules(cw)[usize::from(place)];
        torn.set(module.x as usize, module.y as usize, qr_forensics::symbol::grid::Module::Unknown);
    }
    // Every error correction codeword gone too, so that nothing fills
    // them in again.
    for &cw in layout.blocks.iter().flat_map(|block| &block.codewords[block.data_len..]) {
        erase_codeword(&mut torn, &layout, cw);
    }
    let lost = decode_plain(&torn, &layout, 0).text.plain();
    let expected: String = message.chars().map(|ch| if ch == 'ï' { UNKNOWN_CHAR } else { ch }).collect();
    assert_eq!(lost.replace(&format!("{UNKNOWN_CHAR}{UNKNOWN_CHAR}"), &UNKNOWN_CHAR.to_string()), expected);
    // A message put together from parts has no bits of its own.
    assert!(Sequence::default().assembled().sources.is_empty());
}
