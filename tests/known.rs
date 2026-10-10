//! Text the message is known to hold, pinned bit for bit: what it does
//! for a block past error correction, and that the bits it pins are the
//! bits an encoder writes, in every mode and family.

mod common;

use common::*;
use qr_forensics::decode::bitstream::Certainty;
use qr_forensics::decode::codewords::{PinSource, Pins};
use qr_forensics::decode::known::{BEGINNINGS, Place, TextPinError, lay_over, text_pins, try_beginnings};
use qr_forensics::decode::pipeline::{DecodeOptions, DecodeReport, decode};
use qr_forensics::decode::text::Charset;
use qr_forensics::symbol::grid::ModuleGrid;
use qr_forensics::symbol::layout::SymbolLayout;
use qr_forensics::symbol::tables::Mode;
use qr_forensics::symbol::{EcLevel, SymbolVersion};
use qrcode_generator::Segment;

fn decode_pinned(grid: &ModuleGrid, layout: &SymbolLayout, mask: u8, pins: &Pins) -> DecodeReport {
    decode(grid, layout, mask, pins, &DecodeOptions::default())
}

/// One module wrong in each of the data codewords `wrong`, by their
/// places in the message.
fn damaged(clean: &ModuleGrid, layout: &SymbolLayout, wrong: impl IntoIterator<Item = usize>) -> ModuleGrid {
    let mut grid = clean.clone();
    for i in wrong {
        corrupt_codeword(&mut grid, layout, layout.data_stream[i], 3);
    }
    grid
}

#[test]
fn a_known_beginning_solves_a_block_past_error_correction() {
    // 3-L: one block of 55 data and 15 error correction codewords, which
    // corrects seven wrong.
    let (version, ec, mask) = (SymbolVersion::Qr(3), EcLevel::L, 2);
    let layout = layout(version, ec);
    let text = "https://example.com/q?id=7Q2M-XK4P-99AD";
    let known = "https://example.com/";
    let clean = encode(version, ec, Some(mask), &[Segment::bytes(text)]).unwrap();
    // Fifteen wrong: twelve under the known beginning, three beyond it.
    let grid = damaged(&clean, &layout, (2..=13).chain([26, 31, 37]));
    let before = decode_plain(&grid, &layout, mask);
    assert_eq!(before.blocks_solved(), 0);
    assert_ne!(before.text.plain(), text);

    let found = text_pins(&layout, &before, Place::Char(0), known, Charset::Auto).unwrap();
    assert_eq!((found.pinned, found.left_out, found.contradicted), (20, 0, 0));
    // The header is twelve bits, so each character straddles two
    // codewords: nineteen pinned whole, and half of one at either end.
    let whole = found.pins.values().filter(|pin| pin.mask == 0xFF).count();
    assert_eq!((whole, found.pins.len()), (19, 21));
    assert_eq!(found.pins[&layout.data_stream[1]].mask, 0x0F);
    assert_eq!(found.pins[&layout.data_stream[21]].mask, 0xF0);
    // `ht` lies across the third codeword as 8 and 7.
    assert_eq!(found.pins[&layout.data_stream[2]].value, 0x87);
    assert!(found.pins.values().all(|pin| pin.source == PinSource::Text));

    let after = decode_pinned(&grid, &layout, mask, &found.pins);
    assert!(after.is_trustworthy());
    assert_eq!(after.text.plain(), text);
    // What is pinned counts as confirmed, as a pin's bits do.
    assert!(after.text.spans.iter().all(|span| span.certainty == Certainty::Verified));
}

#[test]
fn text_is_pinned_where_it_stands_in_the_message() {
    let (version, ec, mask) = (SymbolVersion::Qr(3), EcLevel::L, 2);
    let layout = layout(version, ec);
    let text = "https://example.com/q?id=7Q2M-XK4P-99AD";
    let clean = encode(version, ec, Some(mask), &[Segment::bytes(text)]).unwrap();
    // Eight wrong, one more than the block corrects, all under the host
    // name: known from its ninth character on, it is restored.
    let grid = damaged(&clean, &layout, 12..=19);
    let before = decode_plain(&grid, &layout, mask);
    assert_eq!(before.blocks_solved(), 0);
    let found = text_pins(&layout, &before, Place::Char(8), "example.com", Charset::Auto).unwrap();
    assert_eq!(found.pinned, 11);
    let after = decode_pinned(&grid, &layout, mask, &found.pins);
    assert!(after.is_trustworthy());
    assert_eq!(after.text.plain(), text);
    // The same text a character out is not the message's, and solves
    // nothing: a pin asserts, and error correction over the rest of the
    // block is what bears it out.
    let misplaced = text_pins(&layout, &before, Place::Char(9), "example.com", Charset::Auto).unwrap();
    assert!(!decode_pinned(&grid, &layout, mask, &misplaced.pins).is_trustworthy());
    // Past the end of the segment there is no room for it.
    let room = text.len() - 30;
    assert_eq!(
        text_pins(&layout, &before, Place::Char(30), "example.com", Charset::Auto),
        Err(TextPinError::TooLong { needs: 11, room })
    );
    assert_eq!(text_pins(&layout, &before, Place::Char(text.len()), "x", Charset::Auto), Err(TextPinError::NoSuchChar { len: text.len() }));
    assert_eq!(text_pins(&layout, &before, Place::Char(0), "", Charset::Auto), Err(TextPinError::Empty));
}

/// The pins of text that a clean symbol holds agree with every bit of
/// it, and of text it does not hold contradict it.
fn assert_agrees(version: SymbolVersion, ec: EcLevel, segments: &[Segment], at: usize, text: &str, pinned: usize, other: &str) {
    let layout = layout(version, ec);
    let grid = encode(version, ec, Some(0), segments).unwrap();
    let clean = decode_plain(&grid, &layout, 0);
    assert!(clean.is_trustworthy(), "{version}");
    let found = text_pins(&layout, &clean, Place::Char(at), text, Charset::Auto).unwrap_or_else(|e| panic!("{version} {text:?}: {e}"));
    assert_eq!((found.pinned, found.contradicted), (pinned, 0), "{version} {text:?}");
    let again = decode_pinned(&grid, &layout, 0, &found.pins);
    assert!(again.is_trustworthy());
    assert_eq!(again.text.plain(), clean.text.plain());
    let wrong = text_pins(&layout, &clean, Place::Char(at), other, Charset::Auto).unwrap();
    assert!(wrong.contradicted > 0, "{version} {other:?}");
}

#[test]
fn every_mode_is_pinned_as_an_encoder_packs_it() {
    let qr = SymbolVersion::Qr(4);
    // Digits three to a group: from the fourth, two whole groups, and a
    // seventh digit whose group was not given in full.
    let digits = [Segment::numeric("31415926535897932384").unwrap()];
    assert_agrees(qr, EcLevel::M, &digits, 3, "1592653", 6, "1592663");
    // The two digits that end a payload of twenty are a group of seven
    // bits, pinned when both are given.
    assert_agrees(qr, EcLevel::M, &digits, 18, "84", 2, "85");
    let one = text_pins(&layout(qr, EcLevel::M), &decode_plain(&encode(qr, EcLevel::M, Some(0), &digits).unwrap(), &layout(qr, EcLevel::M), 0), Place::Char(4), "5", Charset::Auto);
    assert_eq!(one, Err(TextPinError::NothingWhole));
    // Alphanumeric characters two to a group.
    let upper = [Segment::alphanumeric("HTTPS://EXAMPLE.COM/A1").unwrap()];
    assert_agrees(qr, EcLevel::M, &upper, 0, "HTTPS://", 8, "HTTPX://");
    assert_agrees(qr, EcLevel::M, &upper, 1, "TTPS://", 6, "TTPX://");
    let lower = text_pins(&layout(qr, EcLevel::M), &decode_plain(&encode(qr, EcLevel::M, Some(0), &upper).unwrap(), &layout(qr, EcLevel::M), 0), Place::Char(0), "https", Charset::Auto);
    assert_eq!(lower, Err(TextPinError::NotInMode { ch: 'h', mode: Mode::Alphanumeric }));
    // Kanji thirteen bits each, and a character of more than one byte in
    // byte data.
    assert_agrees(qr, EcLevel::M, &[Segment::kanji("点茗日本語").unwrap()], 2, "日本", 2, "日語");
    assert_agrees(qr, EcLevel::M, &[Segment::bytes("naïve café")], 2, "ïve c", 6, "ive ca");
    // The second segment of a message, at the character it begins with.
    let mixed = [Segment::numeric("20261010").unwrap(), Segment::bytes("-order"), Segment::alphanumeric("A7").unwrap()];
    assert_agrees(qr, EcLevel::M, &mixed, 8, "-order", 6, "-ordeR");
    assert_agrees(qr, EcLevel::M, &mixed, 14, "A7", 2, "A8");
    // A wider count in the larger versions, and the other families.
    assert_agrees(SymbolVersion::Qr(12), EcLevel::Q, &[Segment::bytes("wider counts from version ten")], 6, "counts", 6, "counts!");
    assert_agrees(SymbolVersion::Micro(3), EcLevel::L, &[Segment::bytes("micro")], 1, "icr", 3, "icx");
    assert_agrees(SymbolVersion::Micro(2), EcLevel::L, &[Segment::numeric("123456").unwrap()], 0, "123456", 6, "123457");
    assert_agrees(SymbolVersion::Rmqr(11), EcLevel::M, &[Segment::bytes("rMQR")], 0, "rMQR", 4, "rMQr");
}

#[test]
fn a_message_whose_header_is_gone_is_pinned_from_its_start() {
    let (version, ec, mask) = (SymbolVersion::Qr(3), EcLevel::L, 2);
    let layout = layout(version, ec);
    let text = "https://example.com/q?id=7Q2M-XK4P-99AD";
    let clean = encode(version, ec, Some(mask), &[Segment::bytes(text)]).unwrap();
    // The mode indicator and the count unreadable, and nine codewords
    // wrong under the beginning: nothing parses, and no character is
    // there to place known text at.
    let mut grid = damaged(&clean, &layout, 3..=11);
    for &cw in &layout.data_stream[..2] {
        erase_codeword(&mut grid, &layout, cw);
    }
    let before = decode_plain(&grid, &layout, mask);
    assert_eq!(before.blocks_solved(), 0);
    assert!(before.text.is_empty());
    assert_eq!(text_pins(&layout, &before, Place::Char(0), "https://", Charset::Auto), Err(TextPinError::NoSuchChar { len: 0 }));
    // At the start of the message, as byte data: the mode indicator is
    // pinned with the text, and error correction restores the count.
    let found = text_pins(&layout, &before, Place::Start(Mode::Byte), "https://example.com/", Charset::Auto).unwrap();
    assert_eq!(found.pins[&layout.data_stream[0]].mask, 0xF0);
    assert_eq!(found.pins[&layout.data_stream[0]].value & 0xF0, 0x40);
    let after = decode_pinned(&grid, &layout, mask, &found.pins);
    assert!(after.is_trustworthy());
    assert_eq!(after.text.plain(), text);
    // The bits pinned from the start are the bits the encoder wrote, in
    // each mode and family: on a clean symbol they contradict nothing.
    for (version, ec, segment, mode, text) in [
        (SymbolVersion::Qr(2), EcLevel::M, Segment::alphanumeric("WIFI:S:HOME").unwrap(), Mode::Alphanumeric, "WIFI:S"),
        (SymbolVersion::Qr(2), EcLevel::M, Segment::numeric("000201010212").unwrap(), Mode::Numeric, "000201"),
        (SymbolVersion::Qr(11), EcLevel::M, Segment::bytes("geo:51.5,-0.1"), Mode::Byte, "geo:"),
        (SymbolVersion::Micro(4), EcLevel::L, Segment::bytes("tel:+44"), Mode::Byte, "tel:"),
        (SymbolVersion::Micro(1), EcLevel::L, Segment::numeric("12345").unwrap(), Mode::Numeric, "123"),
        (SymbolVersion::Rmqr(14), EcLevel::M, Segment::kanji("日本語").unwrap(), Mode::Kanji, "日本"),
    ] {
        let layout = common::layout(version, ec);
        let grid = encode(version, ec, Some(0), &[segment]).unwrap();
        let clean = decode_plain(&grid, &layout, 0);
        let found = text_pins(&layout, &clean, Place::Start(mode), text, Charset::Auto).unwrap_or_else(|e| panic!("{version}: {e}"));
        assert_eq!(found.contradicted, 0, "{version} {mode:?}");
        assert!(decode_pinned(&grid, &layout, 0, &found.pins).is_trustworthy(), "{version}");
        // Under another mode's header the same text is other bits.
        let other = if mode == Mode::Byte { Mode::Numeric } else { Mode::Byte };
        if let Ok(wrong) = text_pins(&layout, &clean, Place::Start(other), if other == Mode::Numeric { "123456" } else { text }, Charset::Auto) {
            assert!(wrong.contradicted > 0, "{version} {other:?}");
        }
    }
    // A mode the version lacks cannot be pinned.
    let m1 = common::layout(SymbolVersion::Micro(1), EcLevel::L);
    let report = decode_plain(&encode(SymbolVersion::Micro(1), EcLevel::L, Some(0), &[Segment::numeric("1").unwrap()]).unwrap(), &m1, 0);
    assert_eq!(text_pins(&m1, &report, Place::Start(Mode::Byte), "a", Charset::Auto), Err(TextPinError::NoSuchMode(Mode::Byte)));
    // And a character its character set lacks is named.
    assert_eq!(
        text_pins(&layout, &before, Place::Start(Mode::Byte), "日本", Charset::Latin1),
        Err(TextPinError::NotInCharset { ch: '日' })
    );
}

#[test]
fn a_common_beginning_is_kept_only_when_error_correction_bears_it_out() {
    let (version, ec, mask) = (SymbolVersion::Qr(3), EcLevel::L, 2);
    let layout = layout(version, ec);
    let opts = DecodeOptions::default();
    let none = Pins::new();
    // An address, with ten codewords wrong under its beginning: three
    // more than the block corrects.
    let text = "https://www.example.com/q?id=7Q2M-XK4P";
    let clean = encode(version, ec, Some(mask), &[Segment::bytes(text)]).unwrap();
    let grid = damaged(&clean, &layout, 2..=11);
    let before = decode_plain(&grid, &layout, mask);
    assert_eq!(before.blocks_solved(), 0);
    let found = try_beginnings(&grid, &layout, mask, &none, &opts, &before).expect("a beginning is borne out");
    // The longest of those that verify: `https://` verifies too.
    assert_eq!(found.text, "https://www.");
    assert!(found.report.is_trustworthy());
    assert_eq!(found.report.text.plain(), text);
    assert!(found.pins.values().all(|pin| pin.source == PinSource::Text));
    // Pins already in force are kept under those of the beginning.
    let mut held = Pins::new();
    let serial = text.find("7Q2M").unwrap();
    lay_over(&mut held, &text_pins(&layout, &before, Place::Char(serial), "7Q2M", Charset::Auto).unwrap().pins);
    let with_held = try_beginnings(&grid, &layout, mask, &held, &opts, &decode_pinned(&grid, &layout, mask, &held)).unwrap();
    assert!(held.keys().all(|cw| with_held.pins.contains_key(cw)));
    // The same damage under a message that begins no common way: no
    // beginning is forced on it.
    let other = "Invoice 2026-1010 for the lathe, due on receipt.";
    let clean = encode(version, ec, Some(mask), &[Segment::bytes(other)]).unwrap();
    let grid = damaged(&clean, &layout, 2..=11);
    let before = decode_plain(&grid, &layout, mask);
    assert_eq!(before.blocks_solved(), 0);
    assert!(try_beginnings(&grid, &layout, mask, &none, &opts, &before).is_none());
    // A symbol that already decodes has no block left to verify.
    let sound = decode_plain(&clean, &layout, mask);
    assert!(try_beginnings(&clean, &layout, mask, &none, &opts, &sound).is_none());
    // A header that reads as no mode at all, its first bit wrong, with
    // seven codewords wrong under the beginning: nothing parses, and
    // the beginning is pinned under a header of its own.
    let wifi = "WIFI:T:WPA;S:workshop;P:hunter2hunter2;;";
    let clean = encode(version, ec, Some(mask), &[Segment::bytes(wifi)]).unwrap();
    let mut grid = damaged(&clean, &layout, 4..=10);
    corrupt_codeword(&mut grid, &layout, layout.data_stream[0], 0);
    let before = decode_plain(&grid, &layout, mask);
    assert_eq!(before.blocks_solved(), 0);
    assert!(before.text.is_empty());
    let found = try_beginnings(&grid, &layout, mask, &none, &opts, &before).expect("a beginning is borne out");
    assert_eq!((found.text, found.place), ("WIFI:T:WPA;S:", Place::Start(Mode::Byte)));
    assert!(found.report.is_trustworthy());
    assert_eq!(found.report.text.plain(), wifi);
    // Every beginning offered can be written in byte mode.
    for text in BEGINNINGS {
        assert!(text_pins(&layout, &sound, Place::Start(Mode::Byte), text, Charset::Auto).is_ok(), "{text:?}");
    }
}
