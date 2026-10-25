//! How far the data a symbol reads under one error correction level and
//! mask is like a message rather than noise: evidence for choosing the
//! level and mask when not one block under any of them is confirmed.
//!
//! The format bits are fifteen modules beside the finders, two copies of
//! them, and a stain over one corner and a blur over the other leave the
//! word they spell to chance: a photograph of a version 34 symbol, read
//! at two pixels a module, settled on level Q from its search and on M
//! from its outline, and its message of 1,735 bytes, held at M, does not
//! fit at Q.
//! Under the wrong level and mask no block verifies, nor does any under
//! the right one when the modules read as poorly as that, and the blocks
//! have nothing to say between them; what the format bits say is then
//! all there is, and it is wrong. The data says more. Under the right
//! level and mask the codewords spell the message, which is seldom
//! noise, and under any other they spell the message scrambled: the
//! blocks laid out at another length and interleaved otherwise, or every
//! module XORed with the difference of two masks. The modules read wrong
//! scramble the message a little under the right format too, but by
//! much less than a wrong format does.
//!
//! What a message is like is taken narrowly, so that the evidence comes
//! from what most symbols hold and is nothing where a symbol holds
//! something else:
//!
//! - Text. A byte segment of text, an address or a sentence, has most
//!   of its bytes lower-case letters, then spaces, digits and
//!   punctuation, then capitals, and few control characters or bytes past
//!   127. Each byte is judged by its top three bits, which say which of
//!   those it is, against a share for each that text of that kind has.
//!   The share must name the classes and not merely find the bytes
//!   uneven: the difference of two masks can be a fixed pattern along
//!   the codewords, and text XORed with a fixed pattern is as uneven as
//!   before, its lower-case letters turned to digits and punctuation and
//!   its capitals to control characters, but not as like text.
//! - Padding. The codewords after a short message alternate 0xEC and
//!   0x11, which noise matches in all eight bits once in two hundred and
//!   fifty-six times, and a wrong mask, or a wrong level's interleaving,
//!   breaks the alternation.
//!
//! Each test is made over the run of the stream where it scores best,
//! since a message is text in part and padding in part, and is charged
//! for the runs and the alignments it could have chosen among: noise
//! scores a few bits so, and nothing once charged. Neither test reaches
//! a message of digits or Kanji, or bytes that are not text, and there
//! both come to nothing under every format, which leaves the choice to
//! the format bits as before.
//!
//! Nor does either test reach a patch of the symbol read as one colour,
//! as glare, a logo, a sticker, a stain or a tear leaves it. What
//! modules of one colour spell unmasked is the mask's own pattern or its
//! inverse, whatever the symbol holds, and under masks 0 and 1, whose
//! patterns repeat every two modules, that is the bytes `f` and `3` over
//! and over: a version 40 symbol read light throughout came to over four
//! hundred bits of likeness to text under both, at every level, and
//! under no other mask to more than twenty-five. So both tests pass over
//! a byte whose top three bits were each read off a codeword whose
//! modules read nearly all one colour, and off a module of that colour
//! ([`blank`]), which a byte of a message is once in eight.
//!
//! The evidence is in bits of likelihood, as the format bits' own
//! weights are ([`super::bch::FormatCandidate::weight`]), so that the
//! two add: a word the format bits read four bits worse than another
//! and the data reads forty bits better is the likelier by thirty-six.

use super::codewords::read_codewords;
use crate::symbol::grid::ModuleGrid;
use crate::symbol::layout::SymbolLayout;

/// The share of the bytes of text in each class its top three bits name:
/// control characters; space, digits and punctuation; capitals; lower
/// case; and the four classes above 127, which text in UTF-8 has some
/// of and seven-bit text none. Half and more lower case, as addresses
/// and sentences are.
const TEXT_SHARES: [f64; 8] = [0.02, 0.25, 0.12, 0.56, 0.0125, 0.0125, 0.0125, 0.0125];

/// How far each share is taken toward an even eighth, for the bits a
/// module read wrong turns into another class: a byte of text with one
/// of its top three bits misread is a byte of another class, and lands
/// there about as often as the misreads go. Without it one misread bit
/// in a high bit costs a byte five bits, and a symbol read at one module
/// in eight wrong would look less like text than noise does.
const TEXT_BLUR: f64 = 0.3;

/// The share of the bits of a padding codeword a read gets wrong, for
/// what each bit that matches the alternation and each that does not is
/// worth: a tenth, as a symbol read too poorly for a block to verify is.
const PAD_MISREAD: f64 = 0.1;

/// The two padding codewords, in the order they alternate.
const PADS: [u8; 2] = [0xEC, 0x11];

/// The share of a codeword's modules that must read one colour for the
/// codeword to be taken as read off a patch of that colour ([`blank`]):
/// three in four, six of eight. A codeword of a message, its modules
/// masked, reads so about once in four.
const PATCH_SHARE: f64 = 0.75;

/// The data stream of a grid under one level and mask, as read
/// ([`stream_as_read`]).
pub struct Stream {
    /// Every bit of every data codeword, unmasked, in the order the
    /// message lays them: `None` where its module could not be read, and
    /// a bit that has no module, the high nibble of the corner codeword
    /// of Model 1, the nought it is.
    bits: Vec<Option<bool>>,
    /// For each bit, the colour of the patch it was read off, dark or
    /// light: where [`PATCH_SHARE`] or more of its codeword's modules
    /// read one colour, and its own module that colour. `None` elsewhere.
    patch: Vec<Option<bool>>,
}

impl Stream {
    /// `bits` read off modules of the colours `dark`, one to a bit, the
    /// bits in the order the message lays them and every eight from the
    /// first a codeword's: `None` where a module could not be read, or
    /// there is none.
    #[cfg(test)]
    fn of(bits: Vec<Option<bool>>, dark: &[Option<bool>]) -> Self {
        let patch = dark.chunks(8).flat_map(patched).collect();
        Stream { bits, patch }
    }
}

/// The colours of a codeword's modules, `None` where one could not be
/// read or there is none, as the colours of the patch each was read off
/// ([`Stream::patch`]).
fn patched(dark: &[Option<bool>]) -> Vec<Option<bool>> {
    let modules = dark.iter().flatten().count();
    let most = [true, false].into_iter().find(|&colour| {
        let alike = dark.iter().filter(|&&c| c == Some(colour)).count();
        modules > 0 && alike as f64 >= PATCH_SHARE * modules as f64
    });
    dark.iter().map(|&c| c.filter(|&c| Some(c) == most)).collect()
}

/// The data bits of `grid` under `layout` and `mask`, in the order the
/// message lays them, as read, with the patch of one colour each was
/// read off, if any.
pub fn stream_as_read(grid: &ModuleGrid, layout: &SymbolLayout, mask: u8) -> Stream {
    let reads = read_codewords(grid, layout, mask);
    let mut stream = Stream { bits: Vec::with_capacity(8 * layout.data_stream.len()), patch: Vec::with_capacity(8 * layout.data_stream.len()) };
    for &cw in &layout.data_stream {
        let span = &layout.codewords[cw];
        let read = reads[cw];
        let modules = span.module_mask();
        let at = layout.codeword_modules(cw);
        let mut dark = Vec::with_capacity(span.stream_bits());
        for i in 0..span.stream_bits() {
            let bit = 0x80u8 >> i;
            if modules & bit == 0 {
                stream.bits.push(Some(false));
                dark.push(None);
            } else {
                stream.bits.push((read.known & bit != 0).then_some(read.value & bit != 0));
                let pos = at[i - span.bit_offset as usize];
                dark.push(grid.get(pos.x as usize, pos.y as usize).as_dark());
            }
        }
        stream.patch.extend(patched(&dark));
    }
    stream
}

/// Bits of likelihood that `stream`, a data stream as [`stream_as_read`]
/// lays it, is a message of text or padding rather than noise: the two
/// tests the module describes, each once charged for the runs it chose
/// among, added where they come to more than nothing. Nought for a
/// stream neither test finds anything in.
pub fn likeness_bits(stream: &Stream) -> f64 {
    text_bits(stream).max(0.0) + padding_bits(stream).max(0.0)
}

/// Whether the byte of `stream` that begins at `at` was read off a patch
/// of one colour: each of its top three bits, which the text test reads,
/// read off a patch of the same colour ([`Stream::patch`]). What they
/// spell unmasked is then the mask's own pattern there, or that
/// pattern's inverse, and says nothing of what the symbol holds.
fn blank(stream: &Stream, at: usize) -> bool {
    match stream.patch.get(at..at + 3) {
        Some([Some(colour), rest @ ..]) => rest.iter().all(|&c| c == Some(*colour)),
        _ => false,
    }
}

/// The most any contiguous run of `scores` adds up to, nought for none:
/// the run of the stream a test finds most of what it looks for in.
fn best_run(scores: impl Iterator<Item = f64>) -> f64 {
    let (mut best, mut running) = (0.0f64, 0.0f64);
    for score in scores {
        running = (running + score).max(0.0);
        best = best.max(running);
    }
    best
}

/// What choosing the best of the runs among `items` and of `alignments`
/// ways of lining them up costs, in bits: a run is a start and an end,
/// each one of the items.
fn charge(items: usize, alignments: usize) -> f64 {
    (alignments.max(1) as f64).log2() + 2.0 * (items.max(2) as f64).log2()
}

/// The text test: each byte's top three bits against [`TEXT_SHARES`], at
/// each of the eight places a byte may start in the stream, the best
/// run at the best place, less what choosing them costs. Bytes with a
/// top bit unread, or read off a patch of one colour ([`blank`]), are
/// passed over.
fn text_bits(stream: &Stream) -> f64 {
    let worth: [f64; 8] = TEXT_SHARES.map(|share| ((1.0 - TEXT_BLUR) * share + TEXT_BLUR / 8.0).log2() + 3.0);
    let bits = &stream.bits;
    let bytes = bits.len() / 8;
    let best = (0..8)
        .map(|start| {
            best_run((0..).map(|k| start + 8 * k).take_while(|&at| at + 3 <= bits.len()).filter_map(|at| {
                if blank(stream, at) {
                    return None;
                }
                let [a, b, c] = [bits[at], bits[at + 1], bits[at + 2]];
                Some(worth[usize::from(a?) << 2 | usize::from(b?) << 1 | usize::from(c?)])
            }))
        })
        .fold(0.0, f64::max);
    best - charge(bytes, 8)
}

/// The padding test: each data codeword's bits against the padding
/// codeword the alternation puts there, starting either way round, the
/// best run of codewords, less what choosing it costs. The padding
/// starts on a codeword's first bit, so the codewords are taken as the
/// stream lays them, eight bits each from its start. A codeword read off
/// a patch of one colour ([`blank`]) counts for nothing.
fn padding_bits(stream: &Stream) -> f64 {
    let (agree, disagree) = ((2.0 * (1.0 - PAD_MISREAD)).log2(), (2.0 * PAD_MISREAD).log2());
    let bits = &stream.bits;
    let codewords = bits.len() / 8;
    let best = (0..2)
        .map(|phase| {
            best_run((0..codewords).map(|k| {
                if blank(stream, 8 * k) {
                    return 0.0;
                }
                let pad = PADS[(k + phase) % 2];
                (0..8)
                    .filter_map(|i| bits[8 * k + i].map(|bit| if bit == (pad & (0x80 >> i) != 0) { agree } else { disagree }))
                    .sum::<f64>()
            }))
        })
        .fold(0.0, f64::max);
    best - charge(codewords, 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bits of `bytes`, most significant first, after `skip` bits of
    /// header.
    fn bits_of(bytes: &[u8], skip: usize) -> Vec<Option<bool>> {
        let mut out = vec![Some(false); skip];
        out.extend(bytes.iter().flat_map(|b| (0..8).map(move |i| Some(b & (0x80 >> i) != 0))));
        out
    }

    /// A stream of noise: bits from a small generator, the same every run.
    fn noise(len: usize, seed: u64) -> Vec<Option<bool>> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                Some(state >> 63 != 0)
            })
            .collect()
    }

    /// `bits` as read off a symbol, unmasked by a mask that along the
    /// stream is as good as noise: each module's colour the bit XORed
    /// with the mask's.
    fn read(bits: Vec<Option<bool>>) -> Stream {
        let mask = noise(bits.len(), 99);
        let dark: Vec<Option<bool>> = bits.iter().zip(&mask).map(|(bit, masked)| bit.zip(*masked).map(|(bit, masked)| bit != masked)).collect();
        Stream::of(bits, &dark)
    }

    #[test]
    fn text_and_padding_stand_out_from_noise() {
        let text = b"QR code (abbreviated from Quick Response code) is a type of two-dimensional matrix barcode invented in 1994";
        // Byte mode at a version of ten or more: four bits of mode and
        // sixteen of count, so each byte starts four bits into a codeword.
        // About a bit and a third a byte.
        let message = read(bits_of(text, 20));
        assert!(likeness_bits(&message) > 100.0, "{}", likeness_bits(&message));
        // Noise, however much of it, finds nothing once charged.
        for seed in 0..20 {
            assert_eq!(likeness_bits(&read(noise(4000, seed))), 0.0, "seed {seed}");
        }
        // Padding after a short message.
        let mut short = bits_of(b"https://example.com", 20);
        short.resize(short.len().div_ceil(8) * 8, Some(false));
        let pads: Vec<u8> = (0..60).map(|k| PADS[k % 2]).collect();
        short.extend(bits_of(&pads, 0));
        let short = read(short);
        assert!(padding_bits(&short) > 200.0, "{}", padding_bits(&short));
    }

    #[test]
    fn a_fixed_pattern_over_the_text_makes_it_less_like_text() {
        // The difference of two masks along the codewords can be a fixed
        // pattern, here every other bit flipped: the text is as uneven
        // as before, and less like text.
        let text = b"https://en.wikipedia.org/wiki/QR_code was looked up from a poster in the station";
        let right = bits_of(text, 20);
        let flipped: Vec<Option<bool>> = right.iter().enumerate().map(|(i, b)| b.map(|b| b != (i % 2 == 1))).collect();
        let (right, flipped) = (read(right), read(flipped));
        assert!(text_bits(&right) > text_bits(&flipped) + 40.0, "{} against {}", text_bits(&right), text_bits(&flipped));
    }

    #[test]
    fn misread_text_still_stands_out() {
        // One bit in eight read wrong: less like text, and far from noise.
        let text = b"QR code (abbreviated from Quick Response code) is a type of two-dimensional matrix barcode invented in 1994";
        // A bit is flipped where three bits of noise are all set.
        let right = bits_of(text, 20);
        let flips = noise(3 * right.len(), 7);
        let misread: Vec<Option<bool>> =
            right.iter().zip(flips.chunks(3)).map(|(b, three)| b.map(|b| b != three.iter().all(|f| *f == Some(true)))).collect();
        let wrong = right.iter().zip(&misread).filter(|(a, b)| a != b).count();
        assert!((right.len() / 12..right.len() / 5).contains(&wrong), "{wrong} of {} flipped", right.len());
        let misread = read(misread);
        assert!(likeness_bits(&misread) > 40.0, "{}", likeness_bits(&misread));
    }

    /// A symbol of version `v` at `level` holding `segment`, its data
    /// modules masked by `mask` in place of the mask the encoder chose:
    /// with its layout.
    fn masked(v: u8, level: qrcode_generator::qr::ErrorCorrection, segment: qrcode_generator::Segment, mask: u8) -> (SymbolLayout, ModuleGrid) {
        use crate::decode::pipeline::analyze;
        use crate::symbol::SymbolVersion;
        use crate::symbol::function_map::FunctionMap;
        use crate::symbol::grid::Module;
        use crate::symbol::mask::mask_bit;
        use std::sync::Arc;
        let symbol = qrcode_generator::qr::Encoder::new(level)
            .version(qrcode_generator::qr::Version::new(v).unwrap())
            .boost_error_correction(false)
            .encode_segments(&[segment])
            .unwrap();
        let mut grid = ModuleGrid::from_bools(symbol.width(), symbol.height(), symbol.modules());
        let version = SymbolVersion::Qr(v);
        let func = Arc::new(FunctionMap::new(version));
        let chosen = analyze(&grid, &func).format.best().cloned().unwrap().fields;
        let layout = SymbolLayout::new(func, chosen.ec).unwrap();
        for cw in 0..layout.codewords.len() {
            for pos in layout.codeword_modules(cw) {
                let (x, y) = (pos.x as usize, pos.y as usize);
                if mask_bit(version, chosen.mask, x, y) != mask_bit(version, mask, x, y) {
                    grid.set(x, y, Module::from_dark(grid.get(x, y) != Module::Dark));
                }
            }
        }
        (layout, grid)
    }

    #[test]
    fn text_reads_as_text_under_its_own_mask_whichever_it_is() {
        // Under masks 0 and 1 as under the rest: those two put their own
        // pattern under every byte read off a patch of one colour, and
        // passing over such bytes passes over a few of a message's.
        let text = b"https://en.wikipedia.org/wiki/QR_code was looked up from a poster in the station, and then again at home";
        for mask in 0..8 {
            let (layout, grid) = masked(10, qrcode_generator::qr::ErrorCorrection::Low, qrcode_generator::Segment::bytes(text), mask);
            let own = likeness_bits(&stream_as_read(&grid, &layout, mask));
            let others = (0..8).filter(|&m| m != mask).map(|m| likeness_bits(&stream_as_read(&grid, &layout, m))).fold(0.0, f64::max);
            assert!(own > 100.0 && own > others + 80.0, "mask {mask}: {own} against {others}");
        }
    }

    #[test]
    fn a_patch_of_one_colour_reads_as_no_message() {
        use crate::decode::pipeline::LIKENESS_FLOOR;
        use crate::symbol::grid::Module;
        // Digits, which neither test reaches, as many as the symbol holds,
        // with no padding after them; with the right half of the symbol
        // washed out to paper, then flecked with a module in eight read
        // dark, and then the whole symbol inked over: under every level
        // and mask, nothing. Unmasked by mask 0 or 1, the paper spelled
        // `f` and `3` over and over, and came to hundreds of bits of
        // likeness to text under both at every level, ahead of the level
        // and mask the format bits favoured.
        let digits: String = (0..3057).map(|i| char::from(b'0' + ((i * 7 + i / 3) % 10) as u8)).collect();
        let (layout, grid) = masked(25, qrcode_generator::qr::ErrorCorrection::Low, qrcode_generator::Segment::numeric(&digits).unwrap(), 7);
        let mut fleck = 0u64;
        let mut flecked = |x: usize, y: usize| {
            fleck = (fleck ^ (x as u64 * 7919 + y as u64 * 104_729)).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            fleck >> 60 < 2
        };
        let washed = |speckled: bool, flecked: &mut dyn FnMut(usize, usize) -> bool| {
            let mut washed = grid.clone();
            for y in 0..grid.height() {
                for x in grid.width() / 2..grid.width() {
                    washed.set(x, y, if speckled && flecked(x, y) { Module::Dark } else { Module::Light });
                }
            }
            washed
        };
        let inked = ModuleGrid::from_bools(grid.width(), grid.height(), &vec![true; grid.width() * grid.height()]);
        for (what, grid) in [("washed out", washed(false, &mut flecked)), ("washed out and flecked", washed(true, &mut flecked)), ("inked over", inked)] {
            for &ec in layout.func.version.ec_levels() {
                let layout = SymbolLayout::new(layout.func.clone(), ec).unwrap();
                for mask in 0..8 {
                    let likeness = likeness_bits(&stream_as_read(&grid, &layout, mask));
                    assert!(likeness < LIKENESS_FLOOR, "{what}, {ec:?} {mask}: {likeness}");
                }
            }
        }
    }
}

