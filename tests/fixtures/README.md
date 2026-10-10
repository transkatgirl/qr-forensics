# Golden symbols

Module matrices produced by encoders other than the one the round-trip tests
use. `tests/golden.rs` decodes every file here and requires zero syndromes in
every block, an exact format match, and the expected text where one is given.

Format: one row per line, `1` dark and `0` light. Lines starting with `;` are
metadata: `; family:` is `qr`, `micro`, `rmqr` or `model1`, and `; text:` is
the message the symbol should decode to, to the last character, so a line
may end in a space.

## `zint_*.txt`

Expected output from the test suite of [zint](https://github.com/zint/zint)
(`backend/tests/test_qr.c`), which includes the example symbols printed in
ISO/IEC 18004 and ISO/IEC 23941. Each file names the test item it came from.

zint is distributed under the BSD 3-Clause licence:

> libzint - the open source barcode library
> Copyright (C) 2019-2026 Robin Stuart <rstuart114@gmail.com>
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice,
>    this list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice,
>    this list of conditions and the following disclaimer in the documentation
>    and/or other materials provided with the distribution.
> 3. Neither the name of the project nor the names of its contributors may be
>    used to endorse or promote products derived from this software without
>    specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
> ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
> LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
> CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
> SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
> INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
> CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
> ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
> POSSIBILITY OF SUCH DAMAGE.

## `qrencode_*.txt`

Output of `qrencode` 4.1.1 (libqrencode) with the arguments recorded in each
file. These cover the four-bit final codeword of Micro QR M1 and M3.

## `zxingcpp_model1_v2m.txt`

The QR Code Model 1 symbol from the test samples of
[zxing-cpp](https://github.com/zxing-cpp/zxing-cpp),
`test/samples/qrcode-1/qr-model-1.png`, read module by module (two pixels a
module inside a four-module quiet zone), under the Apache License, Version
2.0 (copyright the zxing-cpp authors). The only Model 1 symbol with a known
message to hand; the Model 1 codeword placement and the version 2 block
table were settled against it.
