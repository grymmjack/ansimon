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
