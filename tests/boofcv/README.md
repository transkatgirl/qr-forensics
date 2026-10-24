# BoofCV's QR Code benchmark

The record `tests/boofcv.rs` holds a search of the whole photo to, over
the photographs BoofCV tests its own QR Code detector on: 536 of them in
sixteen sets, with the corners of each of the 1232 symbols they hold
marked by hand.

| set | what it holds |
|---|---|
| `blurred` | out of focus, and moved during the exposure |
| `bright_spots`, `glare` | a lamp's reflection across the symbol |
| `brightness`, `shadows` | too dark, too bright, or unevenly lit |
| `close` | the symbol filling the frame |
| `curved` | on bottles, cans and creased paper |
| `damaged` | torn, scratched, stained and written over |
| `high_version` | versions up to 40 |
| `lots` | sheets of up to sixty symbols |
| `monitor` | photographed off a screen |
| `nominal` | ordinary photographs of ordinary symbols |
| `noncompliant` | logos over the middle, coloured and rounded modules |
| `pathological` | symbols made to be hard |
| `perspective`, `rotations` | seen from far to one side, and turned |

The photographs are not kept here: the archive is two hundred megabytes,
and BoofCV publishes it without a licence of its own beside it. It is
`qrcodes_v3.zip` from BoofCV's regression data,

```text
https://boofcv.org/notwiki/regression/fiducial/qrcodes_v3.zip
```

and unpacks to a directory `qrcodes` holding `detection/<set>/` with the
photographs and, beside each, a `.txt` of the corners marked in it, and
`decoding/`, twenty-six rendered symbols with their messages, which the
test does not use. Point `QR_BOOFCV_DIR` at `qrcodes` and run the test in
release mode; without the variable it does nothing.

```text
QR_BOOFCV_DIR=path/to/qrcodes cargo test --release --test boofcv -- --nocapture
```

It searches every photograph as `qr-forensics decode --partial --corners`
does, a few at a time, which takes a quarter of an hour on a machine of
fourteen cores, and prints a table of how the symbols of each set fared. `QR_BOOFCV_ONLY=perspective,close`
searches those sets alone.

## The record

`expected.txt` has a line for each symbol marked: the photograph, the
symbol's place among those its label file marks, counted from nothing,
and how the search fared with it.

```text
bright_spots/image026.jpg 0 read
bright_spots/image026.jpg 1 read
bright_spots/image026.jpg 2 not found
```

- `read`: a read that is to be trusted (`read`, or `read, padding
  pinned`) lies on the symbol.
- `found`: a symbol was found there and did not decode, or decoded
  without confirmation.
- `lead`: the search found finder patterns there standing as a symbol's
  and could fit no symbol to them.
- `not found`: nothing the search printed lies there.

What the search prints lies on a symbol marked when its outline is
centred within three tenths of the symbol's width of the symbol's own
centre and has between two fifths of its area and two and a half times
it. The benchmark gives no messages, so a read is not checked against
one; `tests/opencv` and `tests/blackbox` do that.

The marks are in the pixels of each photograph as it is stored, as
BoofCV loads one. Twenty-four of the photographs are stored on their
sides with the turn noted in their EXIF data, and the test turns their
marks with them before it compares them. Three are under 256 pixels on
their shorter side and are read enlarged; the reader prints their
corners in their own pixels all the same, and their marks stand as they
are.

The test fails where a symbol fares worse than recorded, and where one
fares better without the record saying so, so that a change which finds
more is noticed as much as one which finds less. To record what the
search does after a change to the pipeline, run it with
`QR_BOOFCV_BOOTSTRAP` set and look over what it rewrites:

```text
QR_BOOFCV_DIR=path/to/qrcodes QR_BOOFCV_BOOTSTRAP=1 cargo test --release --test boofcv -- --nocapture
```

With `QR_BOOFCV_ONLY` as well, only the lines of the sets searched are
rewritten.

## What BoofCV makes of it

BoofCV 1.5.0, as `FactoryFiducial.qrcode(null, GrayU8.class)` configures
its detector, reads 923 of the 1232 symbols. The search here reads 1200
as the record stands; 14 more it finds and cannot decode, and 18 it
does not find. When it read 1187, every one of BoofCV's was among them,
each with the message BoofCV reads. Three of those 1187 were later
found and not read, `brightness/image012.jpg` 0, `curved/image017.jpg` 0
and `curved/image024.jpg` 0: each decodes only with its padding pinned
and some codewords doubted or filled in, and fell short of the fixed
three codewords' worth of confirmation such a decode needed then. That
fixed bar has since been replaced by charging each solution for every
search that could have found it (`Search` in `src/decode/pipeline.rs`),
and this decoder reads the first two again; `curved/image024.jpg` 0 it
still finds and does not read. Nobody has rechecked whether BoofCV reads
the three now.
BoofCV takes a third of a second over a twelve-megapixel photograph
where this search takes tens of seconds, and where it finds the
position patterns of a symbol and cannot decode it, it says where:
several of the choices in
`src/imaging/detect.rs` are answers to one or the other.
