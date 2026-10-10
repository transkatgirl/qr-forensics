# OpenCV's QR Code test images

The images OpenCV tests its own QR Code detectors on, for `tests/search.rs`:
photographs of one symbol from five sides, symbols close to the lens and
on a monitor, on curved paper, mirrored, and up to nine in one photograph.
The photos of `tests/photos` and `tests/blackbox` are each read through an
outline drawn round the symbol; these are searched whole, with nothing to
say where a symbol is or how many there are, as `qr-forensics decode
<image>` and the window's Search whole photo search one.

Each `<name>.png` or `.jpg` comes with `<name>.search`, a sidecar that
lists the messages the image holds, one to a line, as the reader's
`--escaped` prints a message, with its control characters spelt out so
that it stays on its line:

```text
; from: testdata/cv/qrcode/multiple/3_qrcodes.png
I Love OpenCV
https://github.com/opencv/opencv/tree/3.4
https://en.wikipedia.org/wiki/Computer_vision
```

Lines beginning `;` are headers, and those the harness does not know are
ignored. `from` is where the image lies in OpenCV's test data. `reads`
says how many of the messages the search is expected to read, as
`; reads: 0` in `version_1_up.search`, and is left out when it reads them
all; an image it cannot read whole is kept with what it does read, so
that a change which reads more is noticed as much as one which reads
less. `quick` marks the images the test that runs
with every `cargo test` searches, a few seconds each; the rest are
searched by the ignored test, by hand. `messages: as this reader reads
them` marks an image OpenCV's tests give no text for, whose message here
is the reader's own: a record of what it reads, not a check that it is
right.

A message the reader prints as read that the sidecar does not list fails
the test whatever `reads` says.

To record what the search reads after a change to the pipeline, run the
ignored test with `QR_SEARCH_BOOTSTRAP` set, in release mode, and look
over the sidecars it rewrites:

```text
QR_SEARCH_BOOTSTRAP=1 cargo test --release --test search -- --ignored --nocapture
```

## What the search does not read

- `version_1_up`: the symbol runs off the bottom of the photograph, and
  its bottom-left finder is cut. The search finds it from the two finders
  that are whole and places its right edge a module out; it reads through
  an outline drawn close round it.
- `err_correct_2L`: five of the symbol's codewords are wrong, which is
  all its ten error correction codewords can correct and two more than
  the standard allows a 2-L symbol, whose last two are for detecting a
  misdecode. The message is right and is printed as `unconfirmed`.
  OpenCV's own decoder spends every codeword and reads it.
- `issue_3478`: after its sixteen bytes the symbol's data goes on with
  the header of a second segment that has no room for its payload, where
  a terminator should be. The message is printed as `malformed`.

## Where the images come from

From [opencv_extra](https://github.com/opencv/opencv_extra),
`testdata/cv/qrcode/`, at commit `6b66aee565e86a0d8831daa313f40d6ac0efbd6c`.
An image in a subdirectory there has the subdirectory's name before its
own here: `multiple/3_qrcodes.png` is `multiple_3_qrcodes.png`. The
messages are those of the `dataset_config.json` beside the images, and
for `flipped_*` and `err_correct_*` those of `modules/objdetect/test/
test_qrcode.cpp` and `test_qrcode_encode.cpp` in OpenCV itself.
`curved_curved_6.jpg` is a PNG under a name that says JPEG, as OpenCV
ships it, and is kept so: the reader goes by the bytes.

Left out are `issue_21287.png`, whose message is a compressed archive and
no text; `issue_22892.png`, which holds no symbol that reads, and which
OpenCV asks only not to crash on; and `decode_encode/`, which holds
tables and no images.

OpenCV is distributed under the Apache License, Version 2.0. The
opencv_extra repository carries no licence file of its own; check before
passing these images on.
