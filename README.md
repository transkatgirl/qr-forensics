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
To read AVIF photos you also need libdav1d and pkg-config, for example
`brew install dav1d pkg-config` or `apt install libdav1d-dev pkg-config`.
Without them, build with `--no-default-features` and every other format is
still read.

```text
git clone https://github.com/transkatgirl/qr-forensics.git
cd qr-forensics
cargo run --release
```

A release build is recommended: searching a photo for symbols is heavy work.

You also need a graphics card that wgpu can use: Metal on macOS, and
Vulkan, DirectX 12 or OpenGL elsewhere. Every adjustment in the Photo
section runs on it (Enlarge, Lens, Sharpen, the tone and Channel, and the
median, dot-closing and **Deblur** filters), as do the enlarged,
reduced, sharpened, deblurred and closed copies of the photo the search
and the reader make, the measure of the blur that Deblur undoes, and
the search's reading of the photo as dark and light every way the
finder scan reads it, which the binarised view shows. The photo is
resampled through an outline for the search on the processor, in double
precision, and handed the card to be read. The window is drawn by the
same card, and what an adjustment makes to be shown is laid into the
window's textures there. There is no fallback to the processor: without
a card, QR Forensics says so and exits. A photo too large for one of the
card's buffers is worked on in tiles, with the same result. The card
still needs room for the photo and the copies made of it at once: one
that runs out of memory stops QR Forensics with the graphics driver's
message.

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

Use **Open…**, **Paste**, or drop an image file on the window. Every format
the `image` crate decodes is read: PNG, JPEG, WebP, AVIF, GIF, BMP, TIFF,
TGA, ICO, QOI, OpenEXR, Radiance HDR, Farbfeld, DDS (DXT1, DXT3 and DXT5)
and the PNM family (PBM, PGM, PPM, PAM). A file is told by its first bytes,
whatever its extension says, except TGA, which has no signature and is
known only by its `.tga` name. Paste takes an image on the clipboard, or
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

If nothing else decodes, the search tries again on a deblurred copy of
the photo, for symbols that are out of focus or motion-blurred. If only
that copy decodes, the Photo filter is set to **Deblur** so the canvas
shows what was read.

Every read also anchors the grid to small patches of data modules where
they read sharpest. This keeps the grid on track between the alignment
patterns of a large symbol on curled paper. A read that doesn't decode
with these anchors is retried without them. They show as small handles,
like custom anchors.

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
  along each edge. Drag a hollow handle onto its pattern to set it. Drag
  the grid's outline to add an edge point anywhere along it, or drag any
  other module to set a custom anchor on it. Right-click a handle to
  remove it. Hovering one shows the luminance profile through it, so you
  can see whether it sits on the middle of its pattern.
- **Checking the fit.** Alignment reports how many fixed-pattern modules
  the photo shows in the right colour. The **Show** buttons above the canvas
  switch between the fixed patterns, a preview of what sampling would read
  from each module, the bare grid lines, and the photo binarised as the
  finder search sees it. Hold **H** to see the photo with nothing over it.

The **Photo** section adjusts what is read. Its controls are listed in
the order they are applied:

1. **Enlarge**.
2. **Lens**: radial lens correction, applied to the enlarged pixels, with
   an **Estimate** from the corners and anchors.
3. **Sharpen**: an unsharp mask, sized in the photo's own pixels however
   far it's enlarged (or in the reduced pixels, if it's reduced). Below 0
   it becomes a Gaussian blur up to 4 px wide, for grain and halftone
   screens.
4. **Invert photo**, for light symbols on a dark background.
5. **Contrast**, **Brightness** and **Channel**.
6. **Filter**, run last on the enlarged photo: median, dot-closing, or
   **Deblur**.

A photo with transparency also has **Transparent parts**, which fills
transparent pixels with white, black or mid grey before any other
adjustment. The default, grey, works for both dark and light ink; a
white symbol on a transparent background (made for dark pages) vanishes
on white. **Lens** fills whatever it pulls in from beyond the photo's
edges with the same colour (white for a photo without transparency).
Loading a read made under a different setting switches back to that
setting.

**Deblur** measures the blur's width and direction from edges in the
photo and deconvolves it, for a photo that is out of focus or shaken.
It measures over the whole photo (on a photo over 1.5 megapixels, the
parts of it with the most edges, at full resolution), or only inside
the outline if you press **Measure in outline**. The kernel is drawn
under the filter. The direction of a shake is measured from edges
that run every way, such as print around the symbol; the symbol's own
modules only have edges two ways, so a shake slanting across them is
measured as a rounder blur along them. If the edges all run within less
than a quarter turn of each other, as on a barcode or ruled paper, they
say nothing about the blur along them, and it's taken to be round.
**Strength** scales it: 1 undoes the blur as measured, lower undoes
less, and higher undoes more but adds ringing around edges.

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
- Each module's reliability is learnt from the photo itself. The fixed
  patterns, whose colours are known, show how far correctly read dark
  and light modules sit from the threshold, how widely each colour's
  readings spread, and how many readings are wrong by more than that
  spread explains (specks and stains). However clear a reading is, it's
  taken to be wrong at least once in 256 times (8 bits). That's roughly
  how often readings that looked like 1 in a million were wrong on
  BoofCV's benchmark photos: a logo, a stain, a crease or a grid
  slipping on curled paper can make a reading wrong without making it
  faint. If one colour spreads wider, as unevenly read ink does against
  clean paper, a reading near the threshold is more likely that colour,
  so the threshold moves toward the other. That only happens once the
  blur is undone or found to be slight, because blur greys the fixed
  patterns' thin light rings but leaves their solid dark blocks alone.
  The section reports the fit, and the decoder doubts the least reliable
  modules first.
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
| Reliability | Red where a module is least reliable: sampled near its threshold, on a photo whose fixed patterns read poorly, or painted below certain. The tint fades out at 8 bits, the cap for a sampled module |
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
  though every block is confirmed. It also warns when the message is
  empty and every data bit is zero: the modules are then just the mask
  pattern, which an encoder can produce, but so can a striped or checked
  texture such as a screen or fabric, with every block confirmed. The
  command line prints both warnings to stderr.
- **Format and version.** Each copy of the format bits as read, the
  closest valid format words, and the EC level and mask in effect.
  Words are ranked by the bits they disagree on across both copies,
  each weighted by how confidently it was read; hover a count to see
  the weighted total. The count itself is taken against the best mix of
  the two copies, so it shows whether the damage is within what the
  format code corrects. The format bits settle on a word only if it's
  at least 256 times likelier than the runner-up, the same bar a block
  must clear. A few encoders leave out the format mask, so unmasked
  words are considered too, but only within one bit of the read, and one
  must be 256 × 256 times likelier than every masked word: a damaged
  read lands near some unmasked word too often for less to mean
  anything. When no block verifies under any word, the data is also
  considered: under the right level and mask it tends to look like text
  or padding rather than noise, and how much it does is weighed along
  with the format bits. Choose the level and mask from the format bits
  or by hand.
  Below the level in effect, a line gives its capacity: how many wrong
  or unknown codewords each block can correct, and how many more list
  decoding and erasing by trial add when they're on. Hover it for
  details. When choosing by hand, hover a level to see its capacity; the
  level and mask closest to the format bits are underlined. Pins apply
  only under the level they were made under. If another level is in
  effect, the section warns you and offers a button to switch back.
  **Brute force** decodes under every EC level and mask, with the budgets
  set under Decoding, and ranks the results. Because each is one of many
  attempts, it needs a little more evidence to be confirmed.
- **Blocks.** Each Reed-Solomon block: how it was solved, or what it still
  needs, and the chance the solution is wrong, given as at most 1 in 2^n.
  Every other codeword the search turned up counts against the
  solution, even ones too weak to offer, and the closer one is to the
  block as read, the more it counts. A block counts as confirmed at 1
  in 256 or better (n ≥ 8). **Show** outlines its unknown codewords on
  the grid, or the modules that were doubted to solve it, and **Sample**
  re-reads that block alone from the photo. **Try harder** re-decodes an
  unconfirmed block with every Decoding budget at its maximum. It runs in
  the background and can take a minute on a large block (Stop is in the
  status bar). Its result is used only if it beats the normal decode,
  and is kept until the block's modules, the level, the mask or the
  decoding options change. The row then says "searched harder".
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
plain Reed-Solomon decoding doesn't confirm.

**Use module reliability**, at the top, lets the solvers use how
clearly each module was read. Turn it off to treat every module as
certain, as if painted by hand; doubting, flipping and belief
propagation are then skipped. An option that depends on one that's off
is greyed out but keeps its setting, and a note says when an option
loses part of what it does.

The solvers run in two groups. Everything in the first group runs, and
the best-confirmed result is kept:

- **Solve unknown bits individually** solves for unknown bits from the
  parity equations, rather than for whole codewords. If the bits as read
  can't all be right, it also flips up to three solidly misread bits the
  equations point to (if flipping is on), and erases a codeword or two
  by trial (if erasing by trial is on).
- **Doubt the least reliable modules** marks the faintest codewords, or
  single bits, as unknown, a few at a time. Single bits needs the option
  above. Modules painted by hand are never doubted unless they were
  painted below certain.

The second group runs in order, and stops once the block is confirmed:

- **Flip the faintest bits** (Chase decoding) tries every combination of
  flips on the least reliable bits.
- **Belief propagation** (adaptive, after Jiang and Narayanan) uses the
  parity checks to refine each bit's reliability over several rounds,
  re-decoding each time. It suits a noisy photo of a small symbol, with
  more faint modules than doubting can mark unknown. It stops at the
  first round that finds a solution, so raising the rounds can't lose
  one.
- **List decoding** (soft-decision, Koetter–Vardy) finds every valid
  codeword close enough to the block as read, weighting each codeword by
  its reliability. It gains more at higher levels: up to 19 wrong
  codewords in a 40-H block instead of 15. At level L it gains nothing.
  In symbols that reserve codewords for error detection (below), it
  can't confirm more wrong codewords than plain Reed-Solomon can.
- **Erase codewords by trial** erases every set of codewords of the
  smallest size that gets past plain Reed-Solomon, then the next size
  up, regardless of reliability. It reaches a hand-painted block with
  one codeword too many wrong (two in a small block), or a block whose
  unknown bits solve except for a stain over a few codewords, in
  symbols that reserve no codewords for error detection.

List decoding and erasing by trial run only when plain Reed-Solomon
finds no solution at all.

Turning on flipping, erasing by trial or **Use the message as a prior**
(below) never loses a symbol that reads without them. The grid is first
decoded with all three off. If that doesn't read (every block confirmed
and the message parsing), it's decoded again with each combination of
the ones that are on, cheapest first, and the first decode that reads
is kept, or the best-ranked one if none does. Each is exactly the
decode those settings would make on their own. The cost is time: a
symbol that doesn't read takes longer to give up on.

Notes under each solver say how far it reaches in the current symbol's
blocks. Sliders set how hard the expensive solvers try: bits flipped,
belief propagation rounds, list decoding constraints, and sets erased.
More tries mean more chances of a wrong answer, so a solution found with
a bigger budget needs more evidence to be confirmed; the note under each
slider says how much. The list decoding budget costs only time. **Restore
defaults** resets every option and budget, and **Put back** undoes that
until you change something else.

A solution is confirmed by how rarely random data would pass the same
tests: the redundancy it left unused, the errors it located, and, for a
solution that only flipped faint modules, how few other sets of modules
would be as cheap to flip, however many modules that is. This applies
to every codeword a solver comes across, including ones it passed over
without offering: if the block alone confirms one, it becomes a
solution in its own right, and Blocks says so.

In the smallest symbols (some levels of versions 1 to 3 and of Micro
QR) the standard reserves some error correction codewords for detecting
misreads rather than correcting them. A solution that uses them is
marked as beyond the standard and is never confirmed, whichever solver
found it, and each fallback prefers a solution that leaves them unused.

Every solution's use of error correction is measured against the block
as read. Each changed codeword that had a module read costs two
codewords, and each unknown module filled in costs an eighth of one.
A module sampled right on the threshold says nothing either way and
counts as unknown. Doubted codewords count as read, since doubting
only chooses after the fact which modules to treat as unknown. Modules
filled in by **Use the message as a prior** count as unknown. So in
these symbols, no solver can confirm more wrong codewords than plain
Reed-Solomon corrects without the reserve. The one exception is a
block with enough codewords pinned whole to be solved from its pins
alone: that decode reads none of the block's other modules, so it is
measured against its pins, with every other codeword unknown. The
other solvers can still add confirmation to such a solution, and
solving scattered unknown modules bit by bit costs far less than
erasing their whole codewords.
Anything further is an unconfirmed last resort. The capacity line
under the EC level counts only what can be confirmed, and the notes
under Decoding flag what can't be.

Every attempt is another chance for random data to pass, so a solution
needs one more bit of confirmation for each doubling of the attempts
that could have produced it. The decoder counts:

- **The solver's own tries:** patterns flipped, sets erased, rounds
  run, and bits doubted. These are charged against the solver's own
  evidence. Separately, how close a solution lies to the block as read
  says how many codewords lie that close to random noise at all. That
  doesn't depend on which solver found it or how many tries it took, so
  it's charged only for whole-grid decodes, pins and the search (below).
  A solution is judged by whichever of the two is stronger.
- **Other solvers and retries on the same block:** message prior and
  error clustering rounds, the retry with a four-bit codeword padded
  with ones, and **Try harder**. Solvers that work from module
  reliability count only if some module in the block is less than
  certain, since they find nothing in a hand-painted block. The first
  search and the later rounds are charged in turn rather than all at
  once: a solution found early needs less, whether or not later rounds
  are on, and one found late needs more.
- **Other decodes of the whole grid:** each level and mask in a brute
  force, each placement, sampling and retry in a photo read, and each
  symbol type an outline is read as. A photo read tries its placements,
  samplings and retries in a fixed order and charges each by its
  position: half the budget is split by position (1/2 to the first, 1/6
  to the second, 1/(k(k+1)) to the kth, and to the last of n whatever
  the others leave, 1/n) and half evenly. So the retries tried first,
  which read the most symbols, need less than an even split would ask
  (the first never more than two bits, however many there are, where
  an even split of 42 asks five and a half), and none needs more than a
  bit over it at any one level. A retry nested inside others, such as one sampling of
  one placement in an enhanced copy of the photo, can pay that bit at
  each level. A read from hand-placed corners (the command line's
  `--placed`) splits evenly across the versions and orientations it
  tries and the copies of the photo it reads each through, and charges
  the samplings and retries within each by position, as above.
  Placements the search fitted with nothing else to support them are
  charged by position among themselves, as above, whether they're read
  by the search, by **Read all**, or by picking one from the list. A
  placement supported only by a trial decode the search made of it is
  charged, wherever it's read, by that decode's position among every
  trial decode the search made, in its fixed order: the kth as one of
  k(k+1), a bit for the first and about 2 log2 k bits for the kth, so
  that all of them together pass no more often than one read would.
  Those are the decodes that decide whether a placement is a symbol at
  all. A symbol the search first finds in a reduced copy of the photo
  is fitted again at full size, and where nothing else supports the
  refit, a decode of it chooses between the refit and the copy's
  placement scaled up. The symbol is kept either way, so that decode
  isn't counted among them or charged, and the placement keeps the
  charge it had.
- **Other pins tried:** padding recovery passes and common beginnings.
  Each pin carries the count of the search that placed it. A block is
  charged for the pins it holds, and the searches that draw on the
  whole symbol (the message prior and error clustering) for every pin.

These counts stick: a grid loaded from a photo search keeps its count
until you resample the whole grid, and a pin placed by **Pin the
padding** or **Try common beginnings** keeps its count until it's
removed. A few things are exempt:

- The plain Reed-Solomon decode of a block as read, which any reader
  makes, is charged only for other decodes of the whole grid and for
  the pins its block holds.
- The decode of a block from its pins alone, which reads nothing of
  the grid, isn't charged for other decodes of the whole grid, though
  it is for the pins its block holds.
- In a photo read, the detector's first placement, sampled with your
  settings, isn't charged for the search's other reads (or, when an
  outline is read as each symbol type, only as the type you chose).
- The decodes with flipping, erasing by trial and the message prior on
  or off aren't charged for each other, so turning one on can't lose a
  read.
- Anything you do by hand, such as painting or resampling.

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
against the prior is doubted sooner, but its confidence still comes
from its reading alone. An unreadable module is filled in from the
prior, like a pin only as sure as the prior is: a solution that agrees
with it gains that much confirmation, and one that contradicts it is
penalised as if it had overturned a module read that confidently.

Three more fallbacks use the rest of the symbol:

- **Tie-breaking.** When a solver that tries many candidates finds two
  solutions equally close to the block as read, the stream is parsed
  under each and the one that parses best is kept, rather than giving
  up on the block. The pick is confirmed by how rarely random data
  would parse as well where the candidates differ.
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
  parity byte checked once all are in. Once every part is solved, it
  also shows the chance the assembled message is wrong: the parts'
  chances added up.

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
| Drag the grid's outline in the Align stage | Set an edge point there |
| Right-click an anchor handle | Remove the anchor |
| Esc | Clear the selected codeword and any outlined block, character or segment |

## Licence

Public domain, under the [Unlicense](LICENSE).
