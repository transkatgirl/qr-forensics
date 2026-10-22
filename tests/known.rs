//! Text the message is known to hold, pinned bit for bit: what it does
//! for a block past error correction, and that the bits it pins are the
//! bits an encoder writes, in every mode and family.

mod common;

use common::*;
use qr_forensics::decode::bitstream::Certainty;
use qr_forensics::decode::codewords::{Pin, PinSource, Pins};
use qr_forensics::decode::known::{BEGINNINGS, Place, TextPinError, lay_over, text_pins, try_beginnings};
use qr_forensics::decode::pipeline::{BlockOutcome, DecodeOptions, DecodeReport, decode};
use qr_forensics::decode::text::Charset;
use qr_forensics::symbol::grid::ModuleGrid;
use qr_forensics::symbol::layout::SymbolLayout;
use qr_forensics::symbol::tables::Mode;
use qr_forensics::symbol::{EcLevel, SymbolVersion};
use qrcode_generator::Segment;

fn decode_pinned(grid: &ModuleGrid, layout: &SymbolLayout, mask: u8, pins: &Pins) -> DecodeReport {
    decode(grid, layout, mask, pins, &DecodeOptions::default())
}

/// Decoding without the trials of erasures, which reach a wrong codeword
/// or two past the code on their own: for showing what pins do.
fn unaided() -> DecodeOptions {
    DecodeOptions { erase_by_trial: false, ..DecodeOptions::default() }
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
fn the_unread_high_bit_of_every_byte_leans_toward_ascii_without_a_pin() {
    use qr_forensics::symbol::grid::Module;
    let (version, ec, mask) = (SymbolVersion::Qr(3), EcLevel::L, 2);
    let layout = layout(version, ec);
    let text = "https://example.com/q?id=7Q2M-XK4P-99AD";
    let clean = encode(version, ec, Some(mask), &[Segment::bytes(text)]).unwrap();
    // The byte segment's header is twelve bits, so byte k has its high
    // bit at stream bit 12 + 8k.
    let module_of = |bit: usize| {
        let (cw, place) = layout.stream_bit_location(bit).unwrap();
        let pos = layout.codeword_modules(cw)[place as usize];
        (pos.x as usize, pos.y as usize)
    };
    // The high bit of every byte unknown, and ninety more bits of the
    // message besides, scattered: more unknown bits than the block's
    // hundred and twenty equations, in more codewords than its fifteen
    // of error correction can erase.
    let unread = |text: &str, clean: &ModuleGrid| {
        let mut grid = clean.clone();
        for k in 0..text.len() {
            let (x, y) = module_of(12 + 8 * k);
            grid.set(x, y, Module::Unknown);
        }
        let mut rng = Rng(0xA5C11);
        let mut others = 0;
        while others < 90 {
            let bit = 12 + rng.below(8 * text.len());
            let (x, y) = module_of(bit);
            if bit % 8 == 4 || grid.get(x, y) == Module::Unknown {
                continue;
            }
            grid.set(x, y, Module::Unknown);
            others += 1;
        }
        grid
    };
    let grid = unread(text, &clean);
    let without = DecodeOptions { message_evidence: false, ..DecodeOptions::default() };
    let before = decode(&grid, &layout, mask, &Pins::new(), &without);
    assert_eq!(before.blocks_solved(), 0);
    // Taking each unread high bit at the lean of a byte, nought but one
    // time in sixteen, leaves thirty equations to spare, and the block
    // solves bit by bit, with nothing pinned.
    let after = decode_plain(&grid, &layout, mask);
    assert!(after.is_trustworthy(), "{:?}", after.blocks[0].outcome);
    assert_eq!(after.text.plain(), text);
    assert!(after.blocks[0].message_evidence >= text.len(), "{}", after.blocks[0].message_evidence);
    // Confirmed by the thirty equations left over once the high bits are
    // taken at their lean, less what the search and the leans cost, and
    // no more: the known bits the solution was solved from are not
    // counted again as checks of it.
    assert!(matches!(after.blocks[0].outcome, BlockOutcome::Solved { risk_bits: 8..=29, .. }), "{:?}", after.blocks[0].outcome);
    assert!(after.codewords.iter().all(|cw| cw.pin.is_none()));

    // A message with bytes above 127 leans the same way and is wrong in
    // the two bytes of its accented letter: it reads as it is, or is
    // left unconfirmed, and never stands confirmed as something else.
    let accented = "café at https://example.com/";
    let clean = encode(version, ec, Some(mask), &[Segment::bytes(accented)]).unwrap();
    let report = decode_plain(&unread(accented, &clean), &layout, mask);
    assert!(!report.is_trustworthy() || report.text.plain() == accented, "{:?}", report.text.plain());
}

/// The lean the evidence of `report` puts on the high bit of byte `k` of
/// its first segment, and the block that bit lies in.
fn high_bit_lean(layout: &SymbolLayout, report: &DecodeReport, k: usize) -> (f32, usize) {
    use qr_forensics::decode::evidence::MessageEvidence;
    let evidence = MessageEvidence::of(layout, &report.stream, &report.parse).expect("the message leans on its bits");
    let bit = report.parse.segments[0].payload.start + 8 * k;
    let (cw, place) = layout.stream_bit_location(bit).unwrap();
    (evidence.lean(cw, 7 - place), layout.codewords[cw].block as usize)
}

#[test]
fn a_block_leans_toward_what_the_other_blocks_hold() {
    use qr_forensics::symbol::grid::Module;
    // 5-Q: four blocks of fifteen and sixteen data codewords. The last
    // block unread altogether, and the message's bytes in the other
    // three say what its bytes likely hold.
    let (version, ec, mask) = (SymbolVersion::Qr(5), EcLevel::Q, 1);
    let layout = layout(version, ec);
    let last = layout.blocks.len() - 1;
    let lean_in_last = |text: &str| {
        let mut grid = encode(version, ec, Some(mask), &[Segment::bytes(text)]).unwrap();
        for &cw in &layout.blocks[last].codewords {
            for pos in layout.codeword_modules(cw) {
                grid.set(pos.x as usize, pos.y as usize, Module::Unknown);
            }
        }
        let report = decode(&grid, &layout, mask, &Pins::new(), &DecodeOptions { message_evidence: false, ..DecodeOptions::default() });
        assert!(!report.blocks[last].is_solved() && report.blocks[..last].iter().all(|b| b.is_verified()));
        // A byte whose high bit lies in the last block.
        let k = (0..text.len()).find(|&k| high_bit_lean(&layout, &report, k).1 == last).expect("a byte in the last block");
        high_bit_lean(&layout, &report, k).0
    };
    // By nature alone a byte's high bit is set one time in sixteen.
    let nature = 15f32.ln();
    // Plain ASCII in the other blocks: leans toward nought harder.
    let ascii = lean_in_last("https://example.com/path/to/a/page?with=query&and=more&q=1");
    assert!(ascii > nature, "{ascii}");
    // UTF-8 Cyrillic, every byte of a letter above 127: toward one.
    let cyrillic = lean_in_last("Съешь же ещё этих мягких булок");
    assert!(cyrillic < 0.0, "{cyrillic}");
}

#[test]
fn text_is_pinned_where_it_stands_in_the_message() {
    let (version, ec, mask) = (SymbolVersion::Qr(3), EcLevel::L, 2);
    let layout = layout(version, ec);
    let text = "https://example.com/q?id=7Q2M-XK4P-99AD";
    let clean = encode(version, ec, Some(mask), &[Segment::bytes(text)]).unwrap();
    // Eight wrong, one more than the block corrects, all under the host
    // name: known from its ninth character on, it is restored. (The
    // trials of erasures would reach the one more by themselves, and
    // the curve of list decoding reaches it spending the codeword the
    // standard reserves, which is offered unconfirmed; here the pins
    // are to confirm it.)
    let grid = damaged(&clean, &layout, 12..=19);
    let before = decode(&grid, &layout, mask, &Pins::new(), &unaided());
    assert_eq!(before.blocks_verified(), 0);
    assert!(!before.is_trustworthy());
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

#[test]
fn a_character_after_a_bad_lead_byte_is_pinned_at_its_own_byte() {
    // Under Shift_JIS, 0xC2 is a half-width katakana on its own, 0x81 a
    // lead byte, and '!' no trail byte: the lead is given up as a
    // replacement character and '!' read on its own. Clicking the '!'
    // and pinning it once pinned the lead's byte with its value, the
    // two characters having been put under the cell the lead began at.
    let (version, ec, mask) = (SymbolVersion::Qr(2), EcLevel::M, 1);
    let layout = layout(version, ec);
    let bytes = b"\xC2\x81!x";
    let clean = encode(version, ec, Some(mask), &[Segment::bytes(bytes)]).unwrap();
    let report = decode(&clean, &layout, mask, &Pins::new(), &DecodeOptions { charset: Charset::ShiftJis, ..DecodeOptions::default() });
    assert!(report.is_trustworthy());
    assert_eq!(report.text.plain(), "ﾂ\u{FFFD}!x");
    assert_eq!(report.text.sources.iter().map(|s| s.cell).collect::<Vec<_>>(), [0, 1, 2, 3]);
    let found = text_pins(&layout, &report, Place::Char(2), "!x", Charset::ShiftJis).unwrap();
    assert_eq!((found.pinned, found.contradicted), (2, 0));
    // The byte segment's header is twelve bits, so byte k lies at stream
    // bits 12 + 8k: bytes 2 and 3 straddle message codewords 3 to 5.
    let pinned: Vec<usize> = found.pins.keys().map(|&cw| layout.stream_index(cw).unwrap()).collect();
    assert_eq!(pinned, [3, 4, 5]);
    // The low nibble of '!' and the high nibble of 'x'.
    assert_eq!(found.pins[&layout.data_stream[4]], Pin::full(0x17, PinSource::Text));
    // The same under UTF-8, where 0xC3 is a lead that 'A' cannot follow.
    let bytes = b"\xC3AB";
    let clean = encode(version, ec, Some(mask), &[Segment::bytes(bytes)]).unwrap();
    let report = decode(&clean, &layout, mask, &Pins::new(), &DecodeOptions { charset: Charset::Utf8, ..DecodeOptions::default() });
    assert_eq!(report.text.plain(), "\u{FFFD}AB");
    assert_eq!(report.text.sources.iter().map(|s| s.cell).collect::<Vec<_>>(), [0, 1, 2]);
    let found = text_pins(&layout, &report, Place::Char(1), "AB", Charset::Utf8).unwrap();
    assert_eq!((found.pinned, found.contradicted), (2, 0));
    let pinned: Vec<usize> = found.pins.keys().map(|&cw| layout.stream_index(cw).unwrap()).collect();
    assert_eq!(pinned, [2, 3, 4]);
    assert!(decode_pinned(&clean, &layout, mask, &found.pins).is_trustworthy());
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
fn text_pinned_at_the_start_is_encoded_as_the_message_reads() {
    // Byte data that no designator covers is read in the set chosen, or
    // guessed from the bytes: "café élan" in ISO-8859-1 reads as such
    // under Auto, and text pinned at the start of the message is the
    // bytes of that set, as text pinned at a character is. It was once
    // UTF-8 whatever the message read as, and "café" pinned at the start
    // asserted two bytes for its last letter where the symbol held one.
    let (version, ec, mask) = (SymbolVersion::Qr(2), EcLevel::M, 1);
    let layout = layout(version, ec);
    let clean = encode(version, ec, Some(mask), &[Segment::bytes(&b"caf\xE9 \xE9lan"[..])]).unwrap();
    let report = decode_plain(&clean, &layout, mask);
    assert!(report.is_trustworthy());
    assert_eq!((report.text.plain().as_str(), report.text.encoding.as_deref()), ("café élan", Some("ISO-8859-1 (guessed)")));
    for charset in [Charset::Auto, Charset::Latin1] {
        let found = text_pins(&layout, &report, Place::Start(Mode::Byte), "café", charset).unwrap();
        assert_eq!((found.pinned, found.contradicted), (4, 0), "{charset:?}");
        for (&cw, pin) in &found.pins {
            let value = report.codewords[cw].final_value.unwrap();
            assert_eq!((value ^ pin.value) & pin.mask, 0, "{charset:?} codeword {cw}");
        }
        // The same bits as the text pinned at its first character, and
        // the mode indicator besides.
        let at_char = text_pins(&layout, &report, Place::Char(0), "café", charset).unwrap();
        for (cw, pin) in &at_char.pins {
            let start = found.pins[cw];
            assert_eq!((start.mask & pin.mask, (start.value ^ pin.value) & pin.mask), (pin.mask, 0), "{charset:?} codeword {cw}");
        }
    }
    // A set chosen is followed, as it is at a character: UTF-8 chosen
    // makes the é two bytes, which the symbol contradicts.
    let found = text_pins(&layout, &report, Place::Start(Mode::Byte), "café", Charset::Utf8).unwrap();
    assert_eq!(found.pinned, 5);
    assert!(found.contradicted > 0);
    // With nothing read to guess from, under Auto the text is UTF-8, as
    // it always was.
    let (version, ec, mask) = (SymbolVersion::Qr(3), EcLevel::L, 2);
    let layout = common::layout(version, ec);
    let clean = encode(version, ec, Some(mask), &[Segment::bytes("https://example.com/q?id=7Q2M-XK4P-99AD")]).unwrap();
    let mut grid = damaged(&clean, &layout, 3..=11);
    for &cw in &layout.data_stream[..2] {
        erase_codeword(&mut grid, &layout, cw);
    }
    let before = decode_plain(&grid, &layout, mask);
    assert!(before.text.is_empty() && before.parse.segments.is_empty());
    assert_eq!(text_pins(&layout, &before, Place::Start(Mode::Byte), "café", Charset::Auto).unwrap().pinned, 5);
    assert_eq!(text_pins(&layout, &before, Place::Start(Mode::Byte), "café", Charset::Latin1).unwrap().pinned, 4);
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
    // Without the trials of erasures, which reach a wrong codeword or two
    // past the code by themselves: the beginnings are to do it here.
    let opts = unaided();
    let none = Pins::new();
    // An address, with ten codewords wrong under its beginning: three
    // more than the block corrects.
    let text = "https://www.example.com/q?id=7Q2M-XK4P";
    let clean = encode(version, ec, Some(mask), &[Segment::bytes(text)]).unwrap();
    let grid = damaged(&clean, &layout, 2..=11);
    let before = decode(&grid, &layout, mask, &none, &opts);
    assert_eq!(before.blocks_solved(), 0);
    let found = try_beginnings(&grid, &layout, mask, &none, &opts, &before).expect("a beginning is borne out");
    // The longest of those that verify: `https://` verifies too.
    assert_eq!(found.text, "https://www.");
    assert!(found.report.is_trustworthy());
    assert_eq!(found.report.text.plain(), text);
    assert!(found.pins.values().all(|pin| pin.source == PinSource::Text));
    // The pick of every beginning tried, which the searches its pins reach
    // are charged for, and not every decode of the grid.
    assert!(found.report.tries == 1 && found.report.pin_tries > 1, "{} {}", found.report.tries, found.report.pin_tries);
    // The beginning's pins carry that pick with them, and a decode of the
    // grid under them is the decode the beginning was kept on: what the
    // document decodes once they are pinned. The count was once the
    // document's, laid on every pin in force.
    assert!(found.pins.values().all(|pin| pin.among == found.report.pin_tries));
    let again = decode(&grid, &layout, mask, &found.pins, &DecodeOptions { derive_pins: false, ..opts });
    let risks = |report: &DecodeReport| report.blocks.iter().map(|b| b.outcome.risk_bits()).collect::<Vec<_>>();
    assert_eq!(risks(&again), risks(&found.report));
    // Pins already in force are kept under those of the beginning, each
    // as the pick it was.
    let mut held = Pins::new();
    let serial = text.find("7Q2M").unwrap();
    lay_over(&mut held, &text_pins(&layout, &before, Place::Char(serial), "7Q2M", Charset::Auto).unwrap().pins);
    let with_held = try_beginnings(&grid, &layout, mask, &held, &opts, &decode_pinned(&grid, &layout, mask, &held)).unwrap();
    assert!(held.keys().all(|cw| with_held.pins.get(cw).is_some_and(|pin| pin.among == 1)));
    // The same damage under a message that begins no common way: no
    // beginning is forced on it.
    let other = "Invoice 2026-1010 for the lathe, due on receipt.";
    let clean = encode(version, ec, Some(mask), &[Segment::bytes(other)]).unwrap();
    let grid = damaged(&clean, &layout, 2..=11);
    let before = decode(&grid, &layout, mask, &none, &opts);
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
    let before = decode(&grid, &layout, mask, &none, &opts);
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

#[test]
fn an_odd_length_beginning_is_tried_in_alphanumeric_mode() {
    // An address in alphanumeric mode, its header's first bit wrong so
    // that nothing parses, and eight codewords wrong under the beginning:
    // one more than the block corrects. `HTTP://WWW.` is eleven
    // characters, and alphanumeric characters are packed two to a group:
    // five pairs are pinned and the odd full stop left out, which once
    // kept every odd-length beginning from being tried in the mode at
    // all. The message decoded under the pairs must still begin with the
    // whole of it.
    let (version, ec, mask) = (SymbolVersion::Qr(3), EcLevel::L, 2);
    let layout = layout(version, ec);
    let opts = unaided();
    let none = Pins::new();
    let text = "HTTP://WWW.EXAMPLE.COM/Q-7Q2M-XK4P-99AD";
    let clean = encode(version, ec, Some(mask), &[Segment::alphanumeric(text).unwrap()]).unwrap();
    let mut grid = damaged(&clean, &layout, 2..=9);
    corrupt_codeword(&mut grid, &layout, layout.data_stream[0], 0);
    let before = decode(&grid, &layout, mask, &none, &opts);
    assert_eq!(before.blocks_solved(), 0);
    assert!(before.text.is_empty());
    let found = try_beginnings(&grid, &layout, mask, &none, &opts, &before).expect("a beginning is borne out");
    assert_eq!((found.text, found.place), ("HTTP://WWW.", Place::Start(Mode::Alphanumeric)));
    assert!(found.report.is_trustworthy());
    assert_eq!(found.report.text.plain(), text);
    // The pins are the five whole pairs after the mode indicator and the
    // count, and nothing of the full stop: the mode indicator's four
    // bits, then fifty-five from bit thirteen.
    let pins = text_pins(&layout, &before, Place::Start(Mode::Alphanumeric), "HTTP://WWW.", Charset::Auto).unwrap();
    assert_eq!((pins.pinned, pins.left_out), (10, 1));
    let bits: u32 = pins.pins.values().map(|pin| pin.mask.count_ones()).sum();
    assert_eq!(bits, 4 + 5 * 11);
    // At a character as read the odd character is left out the same
    // way, and there a beginning so cut is not tried; a symbol that
    // decodes already has no block left for one to verify.
    let sound = decode_plain(&clean, &layout, mask);
    let at_char = text_pins(&layout, &sound, Place::Char(0), "HTTP://WWW.", Charset::Auto).unwrap();
    assert_eq!((at_char.pinned, at_char.left_out), (10, 1));
    assert!(try_beginnings(&clean, &layout, mask, &none, &opts, &sound).is_none());
}
