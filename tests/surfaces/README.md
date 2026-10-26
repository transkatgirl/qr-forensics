# The benchmark of "Reading QR Codes on challenging surfaces"

`tests/surfaces.rs` searches whole the images of the benchmark published
with Benito-Altamirano, Martínez-Carpena, Lizarzaburu-Aguilar, Fàbrega and
Prades, "Reading QR Codes on challenging surfaces using thin-plate
splines", Pattern Recognition Letters 184 (2024) 37–43,
doi:10.1016/j.patrec.2024.06.004: 911 images in three sets, each of one
symbol, with the message it holds.

| set | the paper's name | images | what it holds |
|---|---|---|---|
| `flat` | FLAT | 42 | a printed version 7 symbol at level H, photographed with a phone on flat surfaces, from the side (`perspective`) |
| `random` | SURF | 50 | the same symbol wrapped round tubes and bottles (36, `cylindric`) and bent by hand (14, `surface`) |
| `synthetic_small` | SYNT | 819 | 39 symbols, three of each version from 1 to 13 at every level and with random messages, rendered and never printed: each once as made (`affine`) and twenty times seen in perspective (`perspective`) |

The word in brackets is the deformation each image's annotation names,
which the test's table is divided by.

The images are not kept here: the archive is four hundred megabytes. It
is `qr_codes_in_surfaces.zip` from the authors' dataset on Mendeley
Data, "A dataset of QR Codes on top of different surfaces (flat and
challenging surfaces)", doi:10.17632/m6mfwc52vk.1, under CC BY 4.0,

```text
https://data.mendeley.com/datasets/m6mfwc52vk/1
```

and unpacks to `flat`, `random` and `synthetic_small`, each holding
`images/`, an `annotations/` JSON file for each image naming the
deformation and the symbol it is of, and `bitmaps/`, each symbol as
rendered with a JSON file of its version, level and message. Eight more
photos lie loose in `flat` with no annotation, and are not searched;
`synthetic_small/yolo` holds bounding boxes the test does not use. The
paper cites its code at `github.com/dvmcarpena/qr-surface`, which is
gone.

The test fetches the archive the first time it runs, checks it against
the SHA-256 Mendeley Data gives for it, and unpacks it into
`target/tmp/datasets/qr_codes_in_surfaces`, where every later run finds
it until `cargo clean`. Delete that directory to fetch it again. To use a
copy unpacked elsewhere instead, point `QR_SURFACES_DIR` at the directory
holding the three sets.

The test is ignored by a plain `cargo test`; run it by name, in release
mode:

```text
cargo test --release --test surfaces -- --ignored --nocapture
```

It searches every image as `qr-forensics decode --partial` does, a few at
a time, and prints a table of how the images of each set and deformation
fared, how long they took, and the slowest images.
`QR_SURFACES_ONLY=flat,random` searches those sets alone, and
`QR_SURFACES_READER` runs another reader binary in place of the one this
crate builds, as `QR_BOOFCV_READER` does.

## The record

`expected.txt` has a line for each image: the image, and how the search
fared with the symbol it holds.

```text
flat/IMG_20191225_202803.jpg read
random/IMG_20191229_201332.jpg found
random/IMG_20191229_204134.jpg lead
```

- `read`: a read that is to be trusted (`read`, or `read, padding
  pinned`) gives the symbol's message.
- `found`: a symbol was found and did not decode, or decoded without
  confirmation.
- `lead`: the search found finder patterns standing as a symbol's and
  could fit no symbol to them.
- `not found`: the search printed nothing.

The benchmark marks no corners, so a symbol found or a lead counts
wherever in the image it lies. A read to
be trusted whose message is not the symbol's fails the test whatever the
record says, since a misread is worse than a miss.

The test fails where an image fares worse than recorded, and where one
fares better without the record saying so. To record what the search
does after a change to the pipeline, run it with `QR_SURFACES_BOOTSTRAP`
set and look over what it rewrites; with `QR_SURFACES_ONLY` as well,
only the lines of the sets searched are rewritten.

```text
QR_SURFACES_BOOTSTRAP=1 cargo test --release --test surfaces -- --ignored --nocapture
```

## What the paper makes of it

The paper counts 720 images in each set, where the archive holds fewer
photos and more renders: its FLAT and SURF are 48 photos each, every one
put through fifteen projective transforms of its own, and its SYNT is
twenty views of each of 36 symbols, versions 1 to 12. Its figures and
this test's are of different images, and set beside each other only
roughly. ZBar, it reports, reads SYNT very well, 75% of FLAT and 19% of
SURF; its own correction, thin-plate splines falling back on a
cylinder's, reads a little more of SYNT, all of FLAT and 84% of SURF.

As the record stands, the search here reads every symbol of `flat`,
`synthetic_small` and the hand-bent photos of `random`, and 29 of the 36
wrapped round tubes and bottles; of the other seven it finds six that it
can't decode, and the finder patterns of the last with no symbol it can
fit to them. Five of the seven are the five photos it spends longest
on, two to three and a half minutes each on a machine of fourteen
cores, where the whole benchmark is searched in five minutes.
