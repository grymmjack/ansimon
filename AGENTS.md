# AGENTS.md — working on ansimon

Context for AI agents (and humans) making changes here.

## What this is

A CLI that turns a text prompt into **real ANSI art**: SDXL + an ANSI-art LoRA
produces an image, a custom ComfyUI node quantizes it to a CP437 character grid,
and that grid is written out as `.ans` / `.xb` *and* rendered to a PNG.

ansimon is the third of three sibling projects that deliberately share one
engine:

| | repo | model | its custom node |
|---|---|---|---|
| pixelmon | `~/pixelmon` | SDXL + Pixel Art XL | `pixelart_palette` |
| soundmon | `~/git/soundmon` | Stable Audio Open / ACE-Step | `retro_sfx` |
| **ansimon** | `~/git/ansimon` | SDXL + ANSI Art Style XL | `ansi_quantize` |

They share `~/ComfyUI` (venv, torch, port 8188), `~/launch-comfyui.sh`, and the
`servers.json` render farm. **Match their conventions** — same CLI flag names,
same `install.sh` / `download-models.sh` shape, same `bin/<name>` wrapper that
starts ComfyUI on demand, same seed-in-filename rule.

## Layout

```
ansimon.py                    CLI, ComfyUI graph builder, render farm
bin/ansimon                   wrapper: starts ComfyUI if it isn't up
styles.json                   prompt style guides (--style)
custom_nodes/ansi_quantize/
    nodes.py                  AnsiQuantize (quantize+render) + SaveAnsi (serialize)
    cp437.py                  the 256-glyph 8x16 bitmap table
    ansi.py                   .ANS writer/reader, SAUCE, cell transport blob
    xbin.py                   .XB writer/reader, RLE, embedded font/palette
    palette.py                the 16 ANSI colours, .GPL loading
```

`install.sh` symlinks `ansimon.py`, `styles.json` and `custom_nodes/ansi_quantize`
into `~/ComfyUI/`. **The repo is the source of truth**; the ComfyUI copies are
symlinks. Editing files here takes effect immediately — but see "restart" below.

## Invariants — do not break these

**1. The PNG must be a pixel-exact rendering of the emitted art file.**
This is the project's whole claim. `cp437.glyph_bitmaps()` is the single source
of truth used by *both* the quantizer (as coverage masks) and the renderer (as
bitmaps), which is what makes it true by construction. If you add a code path
that draws glyphs some other way, you have broken it.

Verify:
```python
ch, fg, bg = ansi.parse_ans(open('x.ans','rb').read(), ice=True)
rr = nodes.render_cells(ch, fg, bg, np.asarray(palette.ANSI16, np.uint8))
assert (rr == np.asarray(Image.open('x.png').convert('RGB'))).all()
```

**2. The geometric glyphs are generated, not loaded.**
Debian's console fonts are missing exactly `0xB2 ▓`, `0xDC ▄`, `0xDD ▌`,
`0xDE ▐`, `0xDF ▀` — the glyphs blockart is built from. `_geometric_glyphs()`
generates them and **overrides** any font-supplied version. Do not "fix" this by
preferring the font.

**3. `parse_ans` is a terminal emulator, keep it one.**
It models the cursor, deferred wrap, and a canvas that grows downward. That is
not over-engineering: `ESC[#C` (cursor forward) appears in 111 of 112 real scene
files. Deferred wrap especially — wrapping eagerly at column 80 double-advances
on a full row followed by CRLF and silently drops every other line. Regression-
test with a width the art doesn't use (53), not 80.

**4. `.ANS` space cells intentionally differ from the cell grid.**
`to_ans()` lets a space inherit the current foreground colour instead of
emitting a pointless attribute change. So a parsed `.ans` will not match the
source grid on `fg` where the glyph is blank — that is correct and saves real
bytes. Compare `fg` only where the glyph has ink, or compare *renders*.

**5. Colour index == ANSI attribute number.** Palettes are 16 entries in ANSI
order (black, blue, green, cyan, red, magenta, brown, grey, then brights).
Index 9 *means* "bright blue" to every viewer; a palette only changes the RGB it
maps to. Reject palettes that aren't exactly 16.

## Testing

There is no test suite yet. What exists is a set of round-trip checks that
should be run after touching any format code — all are fast and need no GPU:

```bash
cd custom_nodes
~/ComfyUI/.venv/bin/python -c "
import sys; sys.path.insert(0,'.')
import numpy as np
from ansi_quantize import ansi as A, xbin as X
from ansi_quantize.palette import ANSI16
from ansi_quantize.cp437 import glyph_bitmaps
rng=np.random.default_rng(1); r,c=17,53
ch=rng.choice([0x20,0xDB,0xDF,0xB1,0x41],size=(r,c)).astype(np.uint8)
fg=rng.integers(0,16,(r,c),dtype=np.uint8); bg=rng.integers(0,16,(r,c),dtype=np.uint8)
b=A.pack_cells(ch,fg,bg,ANSI16,True); c2,f2,b2,_,i2=A.unpack_cells(b)
assert (c2==ch).all() and (f2==fg).all() and (b2==bg).all() and i2
d=X.to_xbin(ch,fg,bg,palette=ANSI16,ice=True,compress=True); q=X.parse_xbin(d)
assert (q['ch']==ch).all() and (q['fg']==fg).all() and (q['bg']==bg).all()
assert (q['font']==glyph_bitmaps()).all()
a=A.to_ans(ch,fg,bg,ice=True); e,g,h=A.parse_ans(a,cols=c,rows=r,ice=True)
vis=ch!=0x20
assert (e==ch).all() and (g[vis]==fg[vis]).all() and (h==bg).all()
print('all round-trips OK')"
```

Use **non-standard dimensions** (53×17, not 80×25) — width bugs hide behind 80.

`ansimon --doctor` checks the font source, glyph table, node install, and which
farm boxes answer. Run it first when something looks wrong.

## Gotchas that will waste your time

**Restart ComfyUI after changing node code.** Custom nodes load at startup.
Symlinks mean your edit is already in place, but the running process has the old
module. `--doctor` shows the node as installed either way — check
`/object_info` for the actual registered schema.

**Never `pkill -f "ComfyUI/main.py"`.** The pattern matches the invoking shell's
own command line and kills your shell. Get the PID from the port:
```bash
PID=$(ss -ltnp | grep 8188 | grep -oP 'pid=\K[0-9]+' | head -1)
```

**One GPU, three tools.** An ansimon job queued behind a soundmon ACE-Step song
render (10 GB model) just sits there; `ansimon.py` polls `/history` for 900 s, so
it looks like a hang. On an 8 GB card, loading SDXL after ACE-Step OOMs and kills
the server. Check `curl -s localhost:8188/queue` and
`grep -avE 'it/s|%\|' ~/ComfyUI/server.log | tail` before assuming a bug.

**Use `python -u` when diagnosing.** stdout is block-buffered when piped, so
`timeout` discards everything a hung run had printed.

**ComfyUI caches by node inputs.** Re-running with the same `--seed` but a
different `--charset` reuses the sampled image — first run ~20 s, variations ~1 s.
Great for iterating; also means a "fast" run may not have re-sampled at all.

## Conventions

- Comments explain **why**, not what. The codebase leans on this — e.g. why
  dithering is damped, why the tie-break prefers `█`+bright-fg over space+bright-bg.
- Keep the CLI flag vocabulary aligned with pixelmon/soundmon. New flags should
  feel like they could have existed there.
- The seed goes in every output filename so any result is re-runnable.
- Anything that bounds output (dropping a farm box, truncating) must say so on
  stdout. A silently shrinking farm looks like one slow GPU.
- `servers.json` is gitignored — it holds private LAN addresses. Only ever commit
  `servers.example.json`.

## Known gaps

- No automated test suite (the snippet above is what there is).
- `--from-ans` is verified on 112 files; `ESC[s`/`u`, `ESC[K` and 512-char fonts
  are implemented but untested against real art that uses them.
- Multi-GPU farm execution is unverified end-to-end; dispatch and graceful
  degradation work, but the other boxes need `install.sh` run on them.
- The ANSI LoRA is trained on full BBS-screen compositions, so canvases under
  ~32 columns get a whole scene crammed into a tile. This is a model limitation,
  not a quantizer one — don't try to fix it in `ansi_quantize`.
- Nothing is trained natively on `.ANS`. The obvious next step is 16colo.rs →
  tokenize cells as (glyph, fg, bg) → small autoregressive transformer. That
  would replace the LoRA, not the node.

## Training a style LoRA (`tools/`)

`tools/build-lora-dataset.py` turns a folder of `.ans` into an SDXL training
set; `tools/train-lora.sh` drives kohya sd-scripts. Things that are decisions,
not accidents:

- **2x nearest upscale.** 80 cols x 8 px = 640; 24 rows x 16 px = 384. Doubled
  that is 1280x768, a native SDXL bucket — no cropping or resampling. Nearest,
  because bicubic blurs the block edges that are the entire lesson.
- **`enable_bucket = false`.** Every sample is exactly 1280x768, so bucketing
  would be pure overhead. If you change `--rows` or `--upscale`, re-check this.
- **No `flip_aug`.** Half the corpus is lettering; mirroring teaches backwards
  letterforms. Do not "helpfully" turn it on.
- **`keep_tokens = 3`** pins `<token>, ansi art, <kind>` at the caption front.
- **No "screen N of M" in captions.** Position in a scroll has no visual
  correlate — it is a token the model would have to learn to ignore.
- **`network_train_unet_only`.** Not a style choice: with the text encoders
  cached to disk they are unloaded, so they cannot be trained anyway.

Prefer Ampere+ over more VRAM. Pascal has no bf16 and trains SDXL far slower
even with 12 GB. Stop ComfyUI on the training box — 8 GB SDXL LoRA needs all of it.

## Measuring style (`tools/style-report.py`)

`python3 tools/style-report.py REF_DIR [CAND_DIR]` profiles a corpus, or scores
a candidate against it (foreground palette, background palette, glyph
vocabulary, tonal transitions; 100 = identical distributions).

This is the objective answer to "did that change help?", which matters because
ANSI style is easy to eyeball wrongly. Use it before and after any quantizer
or LoRA change. Recorded baseline, stock `ansi-art-xl` vs a 112-piece human
corpus: **overall 70.1** (fg 64.9, bg 77.3, glyph 66.4, tone 71.9).

The tonal-transition metric is the interesting one: mean |Δluma| between
horizontally adjacent inked cells was 0.068 for the human corpus and 0.152 for
ansimon — the human moves in small steps (modelling volume), the quantizer
jumps. That gap is the numeric signature of "converted from a picture".

### Two training configs, and why they look nothing alike

`tools/lora/grymmjack-sdxl.toml` (8 GB Ampere) and `grymmjack-sd15.toml`
(12 GB Pascal) invert almost every setting, because the constraint inverts:

|  | SDXL / 3070 | SD 1.5 / Titan Xp |
|---|---|---|
| UNet | 2.6B params | 860M |
| precision | bf16 | **fp16** — Pascal has no bf16 |
| gradient checkpointing | required | **off** (VRAM to spare; ~30% faster) |
| batch | 1 | 4 |
| optimiser | AdamW8bit | plain AdamW |
| text encoder | can't (outputs cached) | trained, lr 5e-5 |
| dataset `--upscale` | 2 (1280x768) | **1** (640x384 native) |

Do not "unify" these. The SD1.5 run exists to find hyperparameters cheaply;
copying SDXL's memory-saving settings onto it would throw away the whole point.
Never set `full_fp16` on Pascal — without bf16 to fall back on it diverges.

### Sizing a training run

`repeats x epochs` is how many times the model sees each image. For a style
LoRA on a few hundred samples that should land around **10-15 total passes**.
Defaults here are `--repeats 2` and `max_train_epochs = 6` (12 passes), which
on 215 samples at batch 4 is 648 steps.

This was got wrong first time: 6 repeats x 15 epochs = 90 passes = 4,845 steps,
which on a Titan Xp is a **20-hour** run — for the config whose entire purpose
is to be the quick one. Measure step count before launching:

    steps = images x repeats x epochs / batch_size

Measured throughput, Titan Xp (Pascal, 12 GB), SD1.5 @ 640x384, batch 4:
**~14.5 s/it**. So 648 steps is about 2.6 hours.

One myth to not re-derive: `torch.cuda.is_bf16_supported()` returns **True** on
a Titan Xp, but that reflects CUDA-version support, not tensor cores. Separately,
fp16 vs fp32 matmul on that card benchmarks at 1.07x — i.e. a wash, not the
1/64 disaster the Pascal spec sheet implies, because cuBLAS accumulates in fp32.
Precision is not the lever on Pascal; step count is.

### Testing a LoRA you trained here

Two setup errors will make a good LoRA look broken. Both were hit first time:

1. **Generate at the training resolution.** The dataset builder emits 640x384
   for SD1.5 and 1280x768 for SDXL; generating at a different latent size makes
   SD1.5 in particular fall apart. `gen_size()`'s floor is 384 (not 512) so an
   80x24 canvas at `--res 496` lands on exactly 640x384. Don't raise that floor
   back without checking the SD1.5 path.
2. **Use `--raw-prompt`.** Without it ansimon wraps the subject as
   `ansiart, {subject}, bold shapes, flat colors, ...` — and `ansiart` is the
   *stock* LoRA's trigger word. A LoRA trained from `build-lora-dataset.py`
   captions wants its prompt in that same shape:
   `<token>, ansi art, <kind>, <subject>, cp437 block characters, 16 color ega
   palette, text mode art`.

Then sweep checkpoint x strength and score with `style-report.py` rather than
eyeballing. Strength matters a lot: an undertrained LoRA at 1.0 collapses into
noise bands while the same checkpoint at 0.7 is coherent.

### Measured: does a style LoRA actually help?

Yes, substantially — and with a tradeoff the style score cannot see.

Trained on 215 samples from a 112-piece personal corpus (SD 1.5, Titan Xp,
648 steps, ~2h36m). Scored with `style-report.py` against that same corpus,
6-10 generations per cell:

```
             0.5     0.7     0.9     1.0     1.1     1.2
  epoch 2   66.5    69.9    71.8
  epoch 4   66.3    70.2    72.7
  epoch 6   68.2    72.3    77.6    76.8    77.0    75.7
```

Baseline (stock `ansi-art-xl`): **70.1**. Best: **epoch 6 @ 0.9 = 77.6**
(fg 64.8, bg 82.3, glyph 85.7, tone 77.5). Peaks at 0.9-1.1, turns over at 1.2 —
where background palette keeps improving (88.0) but tonal transitions collapse
(64.8), i.e. correct black canvas filled with harsh high-contrast jumps.

**The catch: prompt adherence falls as strength rises.** Percentage of pixels
that change when only the SUBJECT changes at a fixed seed:

    strength 0.5 -> 47-65%      strength 0.7 -> 39-50%      strength 0.9 -> 30-40%

At 0.9+ the LoRA reproduces learned *layouts* (the corpus has 33 "bbs menu
screen" samples and it learned that composition) and largely ignores the prompt.
`style-report.py` scores style-match only and is blind to this, so a high score
partly rewards memorisation. **Use 0.7 for steerable output, 0.9 for maximum
style.** Fixing the tradeoff needs more varied source art, not more training.

Two predictions made here that the data refuted, recorded so they are not
re-derived: the glyph-vocabulary deficit was called "structural, the LoRA cannot
fix it" (it went 47.4 -> 85.7), and more-trained checkpoints were expected to
need *lower* strength (the opposite held; the apparent collapse at 1.0 was a
resolution bug, not the checkpoint).

### Captioning (`tools/caption-ui.py`, `tools/apply-captions.py`)

The first LoRA learned layouts instead of steerable style because the captions'
subject slot held BBS *names* — "colly 10/96", "borgasm electronic magazine" —
never a description of what is depicted. The model was never shown the word
"skull" attached to a skull, which is exactly why prompt adherence collapsed at
high strength.

**Auto-captioning does not fix this.** Measured on this corpus, BLIP-large
returned "a photo of a city at night" for a BORGASM graffiti logo and "a man
with a skateboard" for FOKUS. General VLMs are trained on photographs and
cannot read 16-colour blockart — their output is worse than nothing as training
signal. Florence-2 would be the next thing to try, but its remote code is
incompatible with `transformers 5.11`; put it in its OWN venv rather than
downgrading ComfyUI's, which pixelmon and soundmon share.

So: human in the loop, loop made fast. `caption-ui.py` emits one self-contained
HTML file (images as data URIs, no server, no network, localStorage autosave).
It pre-fills everything already known — the SAUCE title and the auto-classified
piece type — and asks only for the imagery. `apply-captions.py` writes the
results back, preserving the machine captions as `*.txt.auto` so an A/B against
the automatic run stays possible.

### Multi-artist datasets (`extract-artists.py`, `build-multi-artist.py`)

`extract-artists.py` pulls named artists out of a 16colo.rs archive tree (year
folders of pack `.zip`s), matching on the SAUCE author field with normalisation
— the same person signs as "grymmjack", "grymmjack (gj!)" and "GrymmJack" across
a decade — plus a filename-prefix fallback for files with no SAUCE. Every piece
keeps artist/group/pack/year in `manifest.json`; that metadata is the weighting
knob, not paperwork.

`build-multi-artist.py` then does two things a single folder cannot:

- **Weighting.** kohya's repeat count is per folder, so `4_grymmjack/` beside
  `1_filth/` makes the primary ~43% of an epoch while eight other artists supply
  compositional variety. Extracting a whole crew and training flat gives you a
  "mid-90s scene" model, not one person's.
- **Separate trigger words.** Other artists are captioned under their OWN
  handle. Tagging their work with the primary's token would teach that token to
  mean "any blockart", which is the opposite of the goal.

Dedup is by parsed **cell grid**, not file bytes — the same piece ships in
several packs with different SAUCE and line endings. Measured: a 112-piece
personal backup and a 104-piece archive extract shared only 63, union 153.

Two bugs worth not repeating, both silent:
- **Uppercase `.ANS`.** The archive is DOS-era; a lowercase-only glob dropped 80
  of one artist's 153 files and produced zero samples for three artists while
  reporting success. All art globs are now case-insensitive.
- **Non-80-column art.** A few artists worked at 160-200 columns. The builder now
  splits horizontally as well as vertically, so every sample stays 640x384 and
  `enable_bucket` can stay off.

### Dedup must normalise padding

`cellhash()` trims surrounding blank rows/columns before hashing. This is not a
nicety: an artist in several groups — or guesting on other people's packs —
ships the same piece repeatedly, and each release pads it differently. Hashing
the raw grid treats those as distinct works and over-weights whatever the artist
released most often.

Measured on one artist across three sources (personal backup, archive extract,
personal pack collection): **387 files, 198 unique by raw-grid hash, 186 unique
once padding is normalised** — 201 duplicates in total. Near-duplicate pairs
(same size, >=95% of cells matching, i.e. a re-release with the group tag
redrawn) were only 2, so exact-after-trim is enough; fuzzy matching is not worth
the complexity here.

## Validate against pixelview, never against yourself

Three separate bugs shipped here because the test loop was closed: ansimon's
parser read ansimon's output and ansimon's renderer drew it. They agreed
perfectly with each other while all three were wrong, because they shared the
mistake. "Bit-exact round trip" proved self-consistency, not correctness.

  1. **The ANSI reader.** Audited against pixelview on 40 scene pieces: zero
     pixel-exact, wrong canvas height on 21. Rendering now shells out to
     `pixelview --render`; ansimon does not parse other people's art.
  2. **CGA vs SGR colour order.** Palette index 1 is BLUE but SGR 31 is RED;
     index 3 is CYAN but SGR 33 is YELLOW — the RGB bits run opposite ways.
     Writing `30 + index` swapped red/blue and cyan/brown in every file.
     See `CGA_TO_SGR` in ansi.py.
  3. **Glyph geometry.** The block characters were generated mathematically on
     the reasoning that a half block "is the top 8 rows". The real VGA upper
     half is rows **0-6** and the lower half **7-15**. 111 of 256 glyphs
     disagreed with the font. `vga8x16.bin` is now extracted from pixelview's
     own font by `tools/extract-font.py`.

The standing check, which must stay at 0 differing pixels:

    ansimon "..." --output-to /tmp/t
    pixelview --render /tmp/t/*.ans -o /tmp/t/pv.png
    # compare /tmp/t/*_ansi_*.png against pv.png

## Defaults, and why

- `--charset halfblock` — hard cell-aligned edges read as ANSI; richer charsets
  reproduce the source render more faithfully and therefore look more like pixel
  art. Same reason not to raise the grid past ~120 columns for "quality".
- `--dither` **off** — error diffusion scatters ink into empty cells (blank
  cells 36% -> 25%, against ~60% in real scene art) and reads as mush.
- Shade blending is the *right* way to dither in ANSI: `0xB0/B1/B2` over a
  foreground/background pair yields 720 tones from 16 colours. It must be
  enumerated, not derived — deriving fg/bg from a cell's own pixels gives
  fg == bg on a flat cell, so the blend is never a candidate. `shade_bias`
  defaults low (0.10); at 1.0 it dithers 75% of the canvas against a real
  corpus's 10%.
