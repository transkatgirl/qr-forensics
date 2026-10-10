# Blackbox corpus

The larger corpus of photographs and renders for the ignored
`blackbox_corpus_reads_as_its_sidecars_say` test in `tests/photos.rs`: the
QR Code, Micro QR Code and rMQR sets of the ZXing and zxing-cpp test suites,
less the images `tests/photos` already holds. The test reads every photo of
both corpora turned 0, 90, 180 and 270 degrees, in the manner of ZXing's
`AbstractBlackBoxTestCase`, and prints a table of reads, misreads, failures
and not-founds for each set and turn. It takes minutes, so it runs by hand:

```text
cargo test --release --test photos -- --ignored --nocapture
```

Each `<name>.png`, `.webp` or `.jpg` comes with `<name>.txt`, the text the
symbol holds byte for byte, and `<name>.outline`, a sidecar with the
symbol's family, the turns at which the pipeline is expected to read it,
how far the others got, and a loose outline around the symbol, in the
grammar `tests/photos/README.md` describes. The sidecars were written by
`tests/bootstrap_outlines.rs`, which infers the family from the file name:
a stem containing `micro` is Micro QR, `rmqr` is rMQR, `model-1` or
`model1` is QR Code Model 1, and anything else is QR Code Model 2. The
sets hold Model 2 symbols apart from `cpp_micro_*` and two Model 1 ones:
zxing-cpp's `qrcode-1/qr-model-1.png`, which is in `tests/photos` as
`cpp_qr-model-1.png`, and `cpp_qrcode-2_#940.png`, a Model 1 symbol of
version 5 at level H (zxing-cpp issue #940) that the name does not give
away. It is read as Model 2 first, as the app reads any square symbol,
and the pipeline then follows the format word's hint and reads the grid
again as Model 1. It has two Reed-Solomon blocks, and reads only with
them laid back to back rather than interleaved as Model 2's are, which
is what zxing-cpp found and fixed for the issue; `tests/model1.rs`
decodes it as the oracle for the multi-block layout.

`zxing_qrcode-2_13.txt` differs from ZXing's own `13.txt` by one word:
the symbol spells "photography", which this decoder reads with four
blocks solved by plain Reed-Solomon and eighteen or more codewords to
spare in each, while ZXing's reader does not read the image at all (its
suite expects 32 of the set's images at 0 degrees), so its text was never
checked against a decode.

To remake the sidecars after a change to the pipeline, copy this directory
somewhere, run the bootstrap over the copy in release mode, and copy back
the `.outline` files it rewrites:

```text
QR_BOOTSTRAP_DIR=path/to/copy cargo test --release --test bootstrap_outlines -- --nocapture
```

The images were downloaded from the suites' GitHub directory listings and
accepted by a script that kept only those with an expected text the
harness can hold, skipped any whose bytes match a file in `tests/photos`,
and copied each expected text into its own `.txt` beside the image.

## `zxing_qrcode-N_*`

From the blackbox test suite of [ZXing](https://github.com/zxing/zxing),
`core/src/test/resources/blackbox/qrcode-N/`, sets 1 to 6: the image
`NN.png` with its `NN.txt`. The suite's `NN.metadata.txt` files are not
needed and were not copied; none of the QR sets has a `.bin` (binary)
expectation, which the harness could not hold. ZXing is distributed under
the Apache License, Version 2.0:

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

## `cpp_qrcode-N_*` and `cpp_micro_*`

From the test samples of [zxing-cpp](https://github.com/zxing-cpp/zxing-cpp),
`test/samples/qrcode-N/` (sets 1 to 4) and `test/samples/microqrcode-1/`,
under the same Apache License, Version 2.0 (copyright the zxing-cpp
authors). All three images of `test/samples/rmqrcode-1/` are already in
`tests/photos`, so no `cpp_rmqr_*` appears here.

zxing-cpp's runner finds an image's expected text in `<stem>.txt` or
`<stem>.toml`, or, failing both, cuts the stem at its last `-`, `_` or `!`
and looks again, so that `rotate-1.png` to `rotate-e.png` share
`rotate.txt`, and `wrinkle1-05!.webp` reads `wrinkle1.txt`. Each image
here has its own copy of the text it resolved to. Where the suite records
the text in a `.toml`, the `TextPlain` value was copied into the `.txt`
(`cpp_qrcode-2_28`, `cpp_qrcode-2_qr-inv-1`, `cpp_qrcode-2_qr-inv-2`).
Three images were left out because their `.toml` has no `TextPlain`:
`qrcode-1/gs1-figure-4.15.1-2.png` (a GS1 text given only in escaped form,
with `<GS>` separators), `qrcode-2/2in1.webp` (two symbols in one image)
and `qrcode-2/StructApp.webp` (four structured-append symbols).

A `!` at the end of a zxing-cpp stem marks an image its own reader is not
expected to read (`!f`: not in its fast mode), and `#NNN` names refer to
issues in its tracker. The `!` is written `x` here, as `tests/photos` did
for `cpp_micro_12x.webp` (from `12!.webp`): `cpp_qrcode-2_13x.webp` came
from `13!.webp`, `cpp_qrcode-2_#709xf.webp` from `#709!f.webp`, and
`cpp_qrcode-4_wrinkle1-05x.webp` from `wrinkle1-05!.webp`. The `#` is kept.
