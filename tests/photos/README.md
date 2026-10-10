# Photographs

Real photographs and renders of symbols, with the text each one holds, for
`tests/photos.rs`. The synthetic renders in `tests/imaging.rs` have no blur,
noise, lens distortion or uneven paper; these have all of them, and some
have no quiet zone or only a pixel or two per module.

Each `<name>.png`, `.webp` or `.jpg` comes with:

- `<name>.txt`: the text the symbol holds, byte for byte, as the source
  suite's expected output has it. A message may end in a line break, and
  the test compares the whole file.
- `<name>.outline`: the symbol's family, the turns of the photo at which
  the pipeline is expected to read it, how far it got at the others, and
  four `x y` lines giving a loose outline around the symbol in the pixels
  of the unturned photo at its normalised size, top-left, top-right,
  bottom-right, bottom-left, as a hand would draw one in the app:

  ```text
  ; family: qr
  ; status: partial
  ; rotations: 0 180
  ; failure: 90 not found
  ; failure: 270 V3, 2 of 4 blocks solved, text differs
  83.4 142.275
  364.8 38.15
  451.6 357.35
  170.2 461.475
  ```

  `family` is `qr`, `micro`, `rmqr` or `model1`. `rotations` lists the
  quarter turns clockwise, in degrees, at which the photo reads; each
  `failure` line gives a turn that does not and the stage it failed at
  (`not found`, `misread`, or the version found and the blocks solved).
  `status` follows from the turns: `pass` when all four read, `partial`
  when some do, `known-failure` when none does. A sidecar without a
  `rotations` line keeps the meaning it had before turns were recorded:
  `pass` reads the photo as it is, `known-failure` not at all. Lines
  beginning `;` that the harness does not know are ignored.

  The outlines come from what the `rxing` crate reports, through its port
  of ZXing's QR reader (three finder-pattern centres, extended outward by
  a fifth of the finder spacing) or of zxing-cpp's reader (the symbol's
  four corners, widened by the same share), which also knows Micro QR and
  rMQR; from the app's own search of the whole photo; or, failing those,
  from boxes about the centre of the frame. Every turn a sidecar says
  reads was checked by decoding the photo, turned, through the outline
  turned with it.

The fast test in `tests/photos.rs` reads each photo here at all four
turns, across every core at once, and expects what the sidecar says of
each turn. The ignored blackbox test does the same for this directory and
`tests/blackbox` together; see `tests/blackbox/README.md`.

Photos the pipeline cannot read are kept as known failures, so that a
change which reads them is noticed as much as one which loses a photo.
The sidecars were written by `tests/bootstrap_outlines.rs`, which runs only
when `QR_BOOTSTRAP_DIR` names a directory of photos and expected texts.
After a change to the pipeline, copy this directory somewhere, run the
bootstrap over the copy in release mode, and copy back the sidecars it
rewrites:

```text
QR_BOOTSTRAP_DIR=path/to/copy cargo test --release --test bootstrap_outlines -- --nocapture
```

## `zxing_*`

From the blackbox test suite of [ZXing](https://github.com/zxing/zxing),
`core/src/test/resources/blackbox/qrcode-N/`. The name gives the set and the
image number. ZXing is distributed under the Apache License, Version 2.0:

> Copyright 2007 ZXing authors
>
> Licensed under the Apache License, Version 2.0 (the "License"); you may
> not use this file except in compliance with the License. You may obtain a
> copy of the License at
>
> http://www.apache.org/licenses/LICENSE-2.0
>
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
> WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
> License for the specific language governing permissions and limitations
> under the License.

## `cpp_micro_*`, `cpp_rmqr_*` and `cpp_qr-model-1`

From the test samples of [zxing-cpp](https://github.com/zxing-cpp/zxing-cpp),
`test/samples/microqrcode-1/`, `test/samples/rmqrcode-1/` and, for the QR
Code Model 1 symbol, `test/samples/qrcode-1/qr-model-1.png`, under the same
Apache License, Version 2.0 (copyright the zxing-cpp authors). Where the
suite records the expected text in a `.toml` file rather than a `.txt`, the
`TextPlain` value was copied into the `.txt` here.
