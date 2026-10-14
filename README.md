# QR Forensics

A desktop tool for recovering QR codes that are too damaged for a
camera-based reader to scan.

An ordinary reader gives you the message or nothing. QR Forensics shows the
whole decode instead: where the grid lies on the photo, what each module was
read as, which format word the symbol carries, how each Reed-Solomon block
fared, and which codewords are still missing. You can step in at any of
those stages (move the grid, repaint modules, set the EC level and mask,
tell the decoder what you already know of the message), and the result
updates as you go. Once enough is recovered it gives you the text, and a
clean copy of the symbol that an ordinary reader can scan.

It works on:

- QR Code (Model 2), versions 1 to 40
- Micro QR Code, M1 to M4
- rMQR, the rectangular Micro QR Code
- QR Code Model 1, versions 1 to 14. The photo search finds these as QR
  Codes, and the grid is read as Model 1 when its format bits say so.

## Building and running

You need a recent stable Rust toolchain (the crate uses the 2024 edition).

```text
git clone https://github.com/transkatgirl/qr-forensics.git
cd qr-forensics
cargo run --release
```

A release build is recommended: searching a photo for symbols is heavy work.

To open an image at startup, give its path:

```text
cargo run --release -- path/to/photo.jpg
```

## The window

| Area | What is there |
| --- | --- |
| Menu bar | **File** (open, paste, export), **Edit** (undo, transforms of the grid), **View** (fit, grid lines, overlay) |
| Left panel | How modules get into the grid: **Symbol**, **Photo**, **Alignment**, **Sampling**, **Paint** and **Decoding** |
| Canvas | Two stages, switched at the top: **Align** places the grid on the photo, **Edit** paints and inspects the modules |
| Right panel | The inspector: what the decoder made of the grid, stage by stage, starting with the **Result** |
| Status bar | Notices, steps under way (each with a Stop button), and a description of the module under the pointer |

The app starts on a blank version 2 QR Code grid with no photo. Alignment
and Sampling appear once a photo is loaded.

Nearly every control has hover text that says what it does and when to reach
for it. Changes to the grid, the pins, the outline, the anchors and the
photo adjustments are all undoable.

## Working through a symbol

### 1. Load a photo

Use **Open…**, **Paste**, or drop an image file on the window. PNG, JPEG,
WebP, GIF, BMP and TIFF are read. Paste takes an image on the clipboard, or
an image file copied in a file manager. A photo is turned upright by its
EXIF orientation, and a small one starts enlarged so that its modules can be
seen.

A photo is optional. With none, choose the family and version under Symbol
and paint the grid by hand in the Edit stage.

### 2. Find the symbol

The photo opens in the Align stage with a square outline in the middle.
There are two ways to put that outline on the symbol:

- **Search whole photo** looks for symbols of every family and lists them
  under Alignment, those that decode first. Click a row, or the symbol's
  numbered outline on the canvas, to take it. **Read all** reads every
  symbol found through to its message.
- **Find in outline** works inside an outline you place. Drag the TL, TR, BR
  and BL handles loosely around the symbol and press it: it finds the
  symbol, picks the version, and puts the corners and anchors on it. This
  tries harder than the whole-photo search, so use it when the search comes
  up empty.

While **Load reads that decode** is ticked, as it is to begin with, either
one writes a symbol that decodes straight into the grid. For a lightly
damaged symbol that is the whole job: the message is under Result.

A photo over 4096 pixels on its longer side is searched in a reduced copy.
If that misses a small symbol, draw the outline round it and press Find in
outline, or press **Crop to outline** and search again.

Also under Alignment: **Other placements** lists the runner-up fits inside
the outline, **Suggest version** ranks every version on the current corners,
and **Refine** nudges corners and anchors to sub-pixel accuracy once they
are within half a module.

### 3. Fix the alignment by hand

When detection fails or lands slightly off, place the grid yourself.

- **Corners.** Drag a corner handle onto the symbol's outer corner. A loupe
  shows the photo under the handle enlarged. Arrow keys nudge the last
  corner touched.
- **Anchors.** An anchor pins a fixed pattern to its place in the photo, and
  the grid bends through it, which is how a symbol on a curved or creased
  surface is followed. There is a handle on every fixed pattern, and three
  along each edge. Drag a hollow handle onto its pattern to set it, or drag
  any other module of the grid to set a custom anchor there. Right-click a handle to remove it. Hovering one shows the
  luminance profile through it, so you can see whether it sits on the middle
  of its pattern.
- **Checking the fit.** Alignment reports how many fixed-pattern modules
  the photo shows in the right colour. The **Show** buttons above the canvas
  switch between the fixed patterns, a preview of what sampling would read
  from each module, the bare grid lines, and the photo binarised as the
  finder search sees it. Hold **H** to see the photo with nothing over it.

The **Photo** section adjusts what is read: colour channel, brightness,
contrast, sharpening (or, with **Sharpen** set below nought, a Gaussian
blur of the photo in its colours), enlargement, radial lens correction
(with an **Estimate** from the corners and anchors), a median, dot-closing
or blur filter, and **Invert photo** for light symbols on a dark
background.

### 4. Sample the modules

Sampling reads a colour for each module off the photo.

- **Local threshold** judges each module against its neighbourhood, which
  copes with shadows, glare and stains. Turn it off for one threshold over
  the whole symbol, set by hand or automatic. The automatic one can lie as
  a **Threshold surface** that varies smoothly across the symbol, for a
  shadow or light that fails toward one side.
- **Unsure band** leaves modules that fall too close to the threshold
  unknown instead of guessing them. A read that does not decode so is
  made again with those modules kept at the colour they fell on, as
  guesses the decoder doubts first, and a read that decodes only that
  way says so and sets the band to nought.
- **Sharpen small or blurred modules** takes back out of each module what
  blur put into it from its neighbours. How far the photo blurs, and
  which way, is learnt from the fixed patterns, whose colours are known,
  and undone where that gains enough to be worth it; where it cannot be
  learnt, the modules are sharpened by a guess at the blur. The section
  says which was done. Off, the modules are taken as they read, which
  suits a clean print with specks in it.
- How far each module's reading is to be believed is learnt from the
  photo itself: the fixed patterns say how far a dark module and a
  light one read right lie from the threshold, how much the readings of
  each colour spread, and how many are wrong by more than that spread
  explains, as specks and stains are. Where one colour spreads wider,
  as ink read unevenly does against clean paper, a reading near the
  threshold is likelier that colour, and the threshold moves toward the
  other. The colours are told apart so only once the blur is undone or
  found slight, since a blur greys the fixed patterns' thin light rings
  and leaves their solid dark blocks as they were. The section reports
  the fit, and the decoder doubts the least reliable modules first by it.
- **Kernel** chooses how a module's luminance is taken: the average over its
  middle, its centre point alone, or a vote of five points.

The section says how many modules would be left unknown and how cleanly dark
and light separate; that figure rises as the grid comes onto the modules, so
watch it while moving a corner.

**Sample all** replaces the grid with what the photo shows. **Sample unknown
only** keeps everything already painted.

### 5. Edit the grid

The Edit stage shows the grid, with the photo rectified beneath it when
there is one. **Both**, **Photo** and **Grid** choose the layers, with
opacity sliders for blending.

Paint with the **Brush** or the **Rectangle**, in one of five inks:
**Dark**, **Light**, **Unknown**, **From photo**, which samples just the
modules you paint over, and **Doubtful**, which keeps each module's colour
but lowers its reliability. The left button paints the ink, the right
button paints the other of dark and light (with Doubtful, it makes modules
certain again), and Shift paints unknown.

When you cannot tell what a module is, paint it unknown. A block solves once
its unknown codewords, plus twice its wrong ones, fit inside its
error-correction count, so a wrong guess costs twice what an honest blank
does.

When you can half tell, lower the **painted reliability** slider. It runs
from a guess (misread 1 in 2, like a module sampled right on its
threshold) up to 8 bits (1 in 256), then certain. Dark, light and From
photo are painted at that reliability, the same as a module sampled at
it, and the decoder may doubt them when a block fails (see **Decoding**,
below). Certain modules are never doubted; paint over a module with the
slider at certain to make it certain again.

For a stain, tear or fold over modules you can read but don't fully
trust, brush them with **Doubtful** at the reliability you'd give them,
and the decoder will doubt them first.

**Lock fixed patterns** keeps finder, timing and alignment patterns at the
colours the standard gives them. If the grid was painted inverted, mirrored
or turned, the Paint section says so and offers the fix as a button; the
same transforms are in the Edit menu.

Overlays colour the grid by what the decoder knows:

| Overlay | Shows |
| --- | --- |
| Module roles | Finder, timing, alignment, format and version modules, each in its own colour |
| RS blocks | Which Reed-Solomon block each module belongs to, with unsolved blocks marked red |
| Corrections | Each module as error correction has it: green rim where it was unknown, red where it was painted otherwise |
| Reliability | Red where a module's reading is least to be believed: sampled close to its threshold, on a photo whose fixed patterns read poorly, or painted below certain. The status bar says to how many bits a sampled module's reading is believed, and how many such readings are misread |
| Mask pattern | Where the mask in effect swaps dark and light |
| Unmasked data | The data bits themselves, with the mask taken off |
| Codeword status | Green: corrected or filled in. Amber: read but unconfirmed. Red: unknown. Violet rim: doubted to solve the block |
| Segments | Which segment of the message each data module belongs to |
| Pins | Pinned bits, by where the pin came from |
| Differs from photo | Where the grid disagrees with what sampling reads now |

### 6. Read the inspector

- **Result.** Whether the message is decoded and verified, decoded but
  unconfirmed, or only partly solved, with the text coloured to match:
  unconfirmed characters in the warning colour, unreadable ones marked.
  Show it as text, with control characters escaped, as hex, with ECI
  designators, or as GS1 element strings. Click a character to outline the
  modules it is made of. Once every block is solved, it also shows the
  chance the message is wrong, as at most 1 in 2^n. That is the sum of
  the blocks' own chances, since one wrong block makes the message
  wrong. If many blocks each only just pass, the sum can be worse than
  the 1 in 256 each block must meet, and Result warns about that even
  though every block is confirmed.
- **Format and version.** Each copy of the format bits as read, the format
  words they lie nearest, and the EC level and mask in effect. Words are
  ranked likeliest first, by their differing bits over both copies, each
  weighted by its reliability. The count of bits differing from the
  cleanest word the two copies assemble shows whether the read is within
  what the code corrects. Choose them from the format bits or by hand.
  **Brute force** decodes under every EC level and mask and ranks the
  results.
- **Blocks.** Each Reed-Solomon block: how it was solved, or what it still
  needs, and the chance the solution is wrong, given as at most 1 in 2^n.
  That figure combines all the evidence for the solution, and is capped
  by how much closer the solution is to the block as read than any other
  codeword the solvers found. A block counts as confirmed at 1 in 256 or
  better (n ≥ 8). **Show** outlines its unknown codewords on the grid,
  or the modules that were doubted to solve it, and **Sample** re-reads
  that block alone from the photo.
- **Codewords.** Every codeword with its bits as read, final value and
  status. Click a row, or a module with the **Inspect** tool, to see where a
  codeword lives.
- **Segments** and **Data codewords** show how the bit stream parses and the
  data in message order.

### 7. Tell the decoder what you know

A pin fixes bits at a value you assert, and a pinned codeword costs no
error-correction capacity. That is often what tips a block that is just
short of solving.

- **Known text.** Type part of the message, say where it starts, and press
  **Pin text**. **Try common beginnings** tries openings such as `https://`,
  `WIFI:` and `BEGIN:VCARD` and keeps one that makes a block verify.
- **Padding recovery and pins.** Unused capacity is filled with a known
  pattern. **Pin the padding** fixes it where the decoder has found it, and
  the controls below pin it from a start you give.
- **Codewords.** Pin a selected codeword to a hex value, or pin its bits one
  at a time.

You don't need to pin what the characters are likely to be, such as an
address being plain ASCII. The decoder uses that as a soft prior itself
(see **Use the message as a prior**, below).

The **Decoding** section on the left sets what is tried on a block that
fails:

- solving unknown bits one by one;
- doubting the least reliable modules, as whole codewords, as single
  bits, or by flipping the faintest every way;
- flipping a bit or two or three read solidly wrong, where the equations
  name them;
- erasing codewords by trial: every set of the smallest size that goes
  past what the code corrects, then the next size, whatever their
  reliability;
- adaptive belief propagation: over several rounds, the block's parity
  checks update each bit's reliability from the other bits in the same
  checks, and the block is decoded from the updated reliabilities. Each
  round the checks are rearranged so the least reliable bits each sit in
  a check of their own;
- soft-decision list decoding, which weights each codeword by its
  reliability and finds every valid codeword close enough to the block as
  read.

Modules painted by hand are never doubted unless they were painted below
certain. Erasing by trial is what reaches a hand-painted block with one
codeword more wrong than its error correction corrects (two in a small
block), or a block whose unknown bits solve bit by bit except for a stain
over a few codewords. List decoding goes further the lower the block's
rate: up to 11 wrong codewords in a version 1-H block, where the usual
limit is 8, and 19 in a 40-H block, where it is 15; at level L it gains
nothing. Belief propagation finds faint misreads among more faint but
correct modules than doubting can treat as unknown, which is typical of
a noisy photo of a small symbol. Sliders set how much the expensive solvers try:
sets erased, bits flipped, and the list decoder's constraint budget.
Every extra try is another chance of a wrong answer, so solutions found
with a bigger budget need more to confirm them.

A solution is confirmed by how rarely random data would pass the same
tests: the redundancy it left unused, the errors it located, and, for a
solution that only flipped faint modules, how few other sets of modules
would be as cheap to flip, however many modules that is. In the
smallest symbols the standard reserves some error correction codewords
for detecting misreads rather than correcting them. A solution that
uses them is marked as beyond the standard and is never confirmed,
whichever solver found it, and each fallback prefers a solution that
leaves them unused if it finds one.

**Pin the padding and headers found**, also under Decoding, does what
Padding recovery does by hand, on every decode: it finds where the
padding starts (from the segment headers or the pad pattern), pins the
pad codewords, the terminator and any header bits with only one possible
reading, and re-solves the blocks those pins touch until nothing more
changes. Pins are kept only when they solve or confirm another block.
They show under Pins as the decoder's own; **Pin the padding** adds them
to yours.

**Use the message as a prior**, also under Decoding, retries a block
that still fails, using a prior on its bits taken from the rest of the
message. Each character group (a byte, a group of digits or
alphanumerics, or a Kanji character) biases each of its bits toward
what the message's other readable groups of the same kind hold at that
position. Bytes start out biased toward plain ASCII (high bit set 1
time in 16) until the message shows otherwise, so a message in UTF-8
Cyrillic soon flips that bias. Digit groups are biased toward the
values they can hold, so a lone digit never exceeds nine. A module read
against the prior is doubted sooner, and an unreadable one is solved
for at the prior, as a pin would do, but without asserting anything.
The prior only guides the search: a solution is still confirmed
against the modules as read, so a good prior finds the answer faster
without making it more certain.

Three more fallbacks use the rest of the symbol:

- **Tie-breaking.** When a solver that tries many candidates finds two
  solutions equally close to the block as read, the stream is parsed
  under each and the one that parses best is kept, rather than giving
  up on the block.
- **Error clustering.** A confirmed block shows exactly which of its
  modules were misread. Stains and tears hit neighbouring modules
  together, so when unconfirmed blocks are retried, modules next to a
  known misread are doubted first. How strongly is estimated from the
  confirmed blocks: among their own modules next to a misread one, how
  many more were misread than their readings predict. Scattered specks
  give a low rate, leaving neighbours nearly as read; a stain gives a
  high one, and its edges are doubted before anything else. With little
  evidence the rate stays near 1 in 8. Blocks shows the estimate.
- **Message prior.** As more of the stream parses, the message prior
  above is rebuilt from it for the next retry.

Blocks says when a block was solved in any of these ways.

### 8. Take the result out

- **Copy** copies the message as shown, and **Save bytes…** writes its raw
  bytes to a file.
- **Apply corrections** writes the corrected modules and regenerated fixed
  patterns back into the grid.
- **Export PNG…** and **Export SVG…** save the repaired symbol with its
  quiet zone. Modules still unknown are drawn grey.
- **Structured append.** For a message split over several symbols, decode
  each and press **Add to sequence**. The parts are assembled, and the
  parity byte checked once all are in.

## Keys and mouse

Ctrl is ⌘ on macOS. Shortcuts are off while a text field has the keyboard.

| Input | Action |
| --- | --- |
| Ctrl+O | Open an image |
| Ctrl+V | Paste an image from the clipboard |
| Ctrl+Z, Ctrl+Shift+Z | Undo, redo |
| Scroll | Pan |
| Pinch, or Ctrl+scroll | Zoom |
| Middle-drag, or Space+drag | Pan |
| F | Fit to window |
| H (hold) | Show the bare photo |
| B, R, I | Brush, Rectangle, Inspect |
| 1, 2, 3, 4, 5 | Dark, Light, Unknown, From photo, Doubtful ink |
| X | Swap the ink between dark and light |
| Right button | Paint the other of dark and light; with Doubtful, make modules certain |
| Shift while painting | Paint unknown |
| Arrow keys, pointer over the Align canvas | Nudge the last corner touched one pixel; ten with Shift, a tenth with Alt |
| Right-click an anchor handle | Remove the anchor |
| Esc | Clear the selected codeword and any outlined block, character or segment |

## Licence

Public domain, under the [Unlicense](LICENSE).
