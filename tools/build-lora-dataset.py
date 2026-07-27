#!/usr/bin/env python3
"""Turn a folder of .ANS art into an SDXL LoRA training set.

    python3 tools/build-lora-dataset.py SRC_DIR OUT_DIR [--token grymmjack]

Why this can exist at all: ansimon already has a terminal-accurate `.ANS`
reader, the CP437 glyph table and the renderer. Training data is just those
three pointed at someone's back catalogue.

Three decisions worth understanding
-----------------------------------
**1. Scale: 2x nearest-neighbour, never bicubic.**
An 80-column canvas is 640 px wide (80 x 8) and a 24-row screen is 384 px tall
(24 x 16). Doubled, that is 1280x768 — which happens to be a *native SDXL
training bucket*, so nothing has to be cropped, padded or resampled to fit.
The upscale is nearest-neighbour because the entire subject of the lesson is
hard block edges; bicubic would blur exactly the feature we want taught.

**2. Split tall pieces into screens, don't train on the whole scroll.**
Art files run to hundreds of rows (one here is 1000). A BBS piece is composed
screen by screen, so a 24-row slice is a real composition; a 1000-row image
squashed into one sample is not.

**3. Captions come from SAUCE first.**
Roughly half of real art carries a human-written title in its SAUCE record
("clockwork orange BBS menu template"). That beats anything a vision model
would invent about a picture made of blocks. Files without one fall back to a
heuristic: the ratio of letters to block glyphs separates a *logo* (nearly all
blocks) from a *menu/screen* (lots of text).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_HERE, "custom_nodes"),
           os.path.expanduser("~/ComfyUI/custom_nodes")):
    if os.path.isdir(os.path.join(_p, "ansi_quantize")):
        sys.path.insert(0, _p)
        break
from ansi_quantize import ansi as A, xbin as X          # noqa: E402
from ansi_quantize.nodes import render_cells            # noqa: E402
from ansi_quantize.palette import ANSI16                # noqa: E402
from ansi_quantize.cp437 import CP437                   # noqa: E402

ROWS_PER_SCREEN = 24          # 24 * 16 px * 2 = 768 -> the 1280x768 SDXL bucket
COLS_PER_SCREEN = 80          # 80 * 8 px -> 640. Scene art is overwhelmingly 80
                              # columns, but a few artists worked at 160-200.
                              # Those get split rather than dropped, so every
                              # sample stays one size and bucketing stays off.
UPSCALE = 2
MIN_INK = 0.06                # skip near-empty screens (trailing blank pages)

_LETTERS = set(range(0x41, 0x5B)) | set(range(0x61, 0x7B)) | set(range(0x30, 0x3A))
_BLOCKS = {0xB0, 0xB1, 0xB2, 0xDB, 0xDC, 0xDD, 0xDE, 0xDF}


def clean(s):
    """SAUCE fields are NUL-padded and occasionally have junk. Make them prose."""
    s = s.replace("\x00", " ").strip()
    return re.sub(r"\s+", " ", s)


def describe(ch, sauce, stem, idx, n_screens):
    """Build a caption. SAUCE title if there is one, else infer from the art."""
    title = clean(sauce.get("title", "")) if sauce else ""

    flat = ch.ravel()
    ink = flat[(flat != 0x20) & (flat != 0x00)]
    if ink.size == 0:
        return None
    letters = np.isin(ink, list(_LETTERS)).mean()
    blocks = np.isin(ink, list(_BLOCKS)).mean()

    # A logo is nearly all block glyphs; a menu/screen carries real text.
    if letters > 0.35:
        kind = "bbs menu screen"
    elif letters > 0.12:
        kind = "screen with text"
    elif blocks > 0.5:
        kind = "blockart logo"
    else:
        kind = "ansi art piece"

    bits = [ARGS.token, "ansi art", kind]
    if title:
        # Strip words the `kind` tag already carries, then repair the wreckage:
        # "logo #1" -> "#1" not "-  #1". Leftover punctuation and double spaces
        # are the sort of thing that quietly becomes a learned token.
        t = re.sub(r"\b(logo|ansi|ans|blockart)\b", " ", title, flags=re.I)
        t = re.sub(r"\s+", " ", t).strip(" -,:;#").strip()
        # A bare date or serial ("10/96", "#1") describes nothing visual.
        if t and not re.fullmatch(r"[\d/#.\- ]+", t):
            bits.append(t)
    # Deliberately NO "screen N of M": position in a scroll has no visual
    # correlate, so it is a token the model would have to learn to ignore.
    bits += ["cp437 block characters", "16 color ega palette", "text mode art"]
    return ", ".join(b for b in bits if b)


def load(path):
    """Read .ans or .xb -> (ch, fg, bg, sauce)."""
    raw = open(path, "rb").read()
    sauce, body = A.read_sauce(raw)
    if raw[:5] == X.MAGIC:
        p = X.parse_xbin(body)
        return p["ch"], p["fg"], p["bg"], sauce
    return (*A.parse_ans(raw), sauce)


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="folder of .ans/.xb files")
    ap.add_argument("out", help="dataset output folder")
    ap.add_argument("--token", default="grymmjack",
                    help="style trigger word. default grymmjack")
    # Repeats x epochs = how many times the model sees each image. For a style
    # LoRA on a few hundred samples that wants to land around 10-15 total, not
    # 90 — which is what 6 repeats x 15 epochs gave, and it turned a "fast
    # iteration" run into a 20-hour one on a Titan Xp.
    ap.add_argument("--repeats", type=int, default=2,
                    help="kohya repeat count baked into the folder name. default 2")
    ap.add_argument("--rows", type=int, default=ROWS_PER_SCREEN,
                    help=f"rows per screen. default {ROWS_PER_SCREEN}")
    ap.add_argument("--cols", type=int, default=COLS_PER_SCREEN,
                    help=f"columns per screen. default {COLS_PER_SCREEN}")
    ap.add_argument("--upscale", type=int, default=UPSCALE)
    ARGS = ap.parse_args()

    src = os.path.abspath(os.path.expanduser(ARGS.src))
    out = os.path.abspath(os.path.expanduser(ARGS.out))
    imgdir = os.path.join(out, f"{ARGS.repeats}_{ARGS.token}")
    os.makedirs(imgdir, exist_ok=True)

    # Case-insensitive: the scene archive is full of DOS-era uppercase .ANS,
    # and a lowercase-only glob silently drops most of a 1994 pack.
    files = sorted(f for f in glob.glob(os.path.join(src, "*"))
                   if f.lower().endswith((".ans", ".xb", ".xbin")) and os.path.isfile(f))
    if not files:
        sys.exit(f"nothing to read in {src}")

    pal = np.asarray(ANSI16, np.uint8)
    made = skipped = 0
    manifest = []

    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        try:
            ch, fg, bg, sauce = load(f)
        except Exception as e:
            print(f"  skip {stem}: {e}")
            continue

        nr = max(1, -(-ch.shape[0] // ARGS.rows))      # ceil
        nc = max(1, -(-ch.shape[1] // ARGS.cols))
        for i in range(nr * nc):
            ri, ci = divmod(i, nc)
            a, b = ri * ARGS.rows, (ri + 1) * ARGS.rows
            x0, x1 = ci * ARGS.cols, (ci + 1) * ARGS.cols
            c2, f2, b2 = ch[a:b, x0:x1], fg[a:b, x0:x1], bg[a:b, x0:x1]
            if c2.shape[0] < ARGS.rows:                # pad the last screen
                padn = ARGS.rows - c2.shape[0]
                pad = lambda arr, v: np.vstack(
                    [arr, np.full((padn, arr.shape[1]), v, np.uint8)])
                c2, f2, b2 = pad(c2, 0x20), pad(f2, 7), pad(b2, 0)
            if c2.shape[1] < ARGS.cols:                # pad a narrow last column
                padw = ARGS.cols - c2.shape[1]
                padh = lambda arr, v: np.hstack(
                    [arr, np.full((arr.shape[0], padw), v, np.uint8)])
                c2, f2, b2 = padh(c2, 0x20), padh(f2, 7), padh(b2, 0)

            ink = ((c2 != 0x20) & (c2 != 0x00)).mean()
            if ink < MIN_INK:
                skipped += 1
                continue

            cap = describe(c2, sauce, stem, i, nr * nc)
            if not cap:
                skipped += 1
                continue

            img = Image.fromarray(render_cells(c2, f2, b2, pal))
            if ARGS.upscale > 1:
                img = img.resize((img.width * ARGS.upscale,
                                  img.height * ARGS.upscale), Image.NEAREST)

            name = f"{stem}_{i:03d}"
            img.save(os.path.join(imgdir, name + ".png"))
            with open(os.path.join(imgdir, name + ".txt"), "w",
                      encoding="utf-8") as fh:
                fh.write(cap + "\n")
            manifest.append({"file": name + ".png", "source": os.path.basename(f),
                             "screen": i, "size": list(img.size),
                             "ink": round(float(ink), 3), "caption": cap})
            made += 1

    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    sizes = {tuple(m["size"]) for m in manifest}
    print(f"\n  {made} samples written to {imgdir}")
    print(f"  {skipped} screens skipped (below {MIN_INK:.0%} ink)")
    print(f"  resolutions: {', '.join(f'{w}x{h}' for w, h in sorted(sizes))}")
    print(f"  manifest:    {os.path.join(out, 'manifest.json')}\n")
    kinds = {}
    for m in manifest:
        k = m["caption"].split(", ")[2]
        kinds[k] = kinds.get(k, 0) + 1
    for k, v in sorted(kinds.items(), key=lambda t: -t[1]):
        print(f"    {v:>4}  {k}")
    print()


if __name__ == "__main__":
    main()
