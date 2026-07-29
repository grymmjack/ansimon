#!/usr/bin/env python3
"""Turn a folder of ANSI art into an SDXL / SD1.5 LoRA training set.

    python3 tools/build-lora-dataset.py SRC_DIR OUT_DIR [--token grymmjack]

Rendering is delegated to pixelview
------------------------------------
ANSI is a terminal protocol, not a file format: cursor save/restore, absolute
positioning, erase, overdraw. A partial implementation does not fail loudly, it
silently puts content in the wrong place — which is worse, because the training
data still looks plausible at a glance.

Audited against `pixelview --render` on 40 real scene pieces, the hand-written
parser this tool used to rely on scored **zero** pixel-exact matches and
produced the **wrong canvas height on 21 of them**. So this shells out to
pixelview, which implements the spec.

(`ansi_quantize` still renders its own output, and that stays: ansimon
generates the cell grid itself, so nothing is parsed and the round-trip is
provably exact. The bug was only ever in *reading* other people's art.)

Three other decisions worth understanding
-----------------------------------------
**Scale: nearest-neighbour, never bicubic.** An 80-column canvas is 640 px wide
and a 24-row screen 384 px tall. Doubled that is 1280x768, a native SDXL
bucket; at 1x it suits SD 1.5. Nearest because the whole lesson is hard block
edges, which bicubic would blur.

**Split tall pieces into screens.** Art runs to hundreds of rows. A BBS piece is
composed screen by screen, so a 24-row slice is a real composition; a 1000-row
scroll squashed into one sample is not. Wide pieces (some artists worked at
160-200 columns) split horizontally too, so every sample stays one size.

**Captions come from SAUCE first**, then a fallback tag. Roughly half of real
art carries a human-written title; the rest should be captioned by hand with
`tools/caption-ui.py`.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_HERE, "custom_nodes"),
           os.path.expanduser("~/ComfyUI/custom_nodes")):
    if os.path.isdir(os.path.join(_p, "ansi_quantize")):
        sys.path.insert(0, _p)
        break
from ansi_quantize import ansi as A                     # noqa: E402  (SAUCE only)

PIXELVIEW = os.environ.get("PIXELVIEW") or os.path.expanduser(
    "~/git/pixel-viewer/target/release/pixelview")

CELL_W, CELL_H = 8, 16
ROWS_PER_SCREEN = 24
COLS_PER_SCREEN = 80
MIN_INK = 0.06                # skip near-empty screens (trailing blank pages)


def art_files(d):
    return sorted(f for f in glob.glob(os.path.join(d, "**", "*"), recursive=True)
                  if os.path.isfile(f)
                  and f.lower().endswith((".ans", ".xb", ".xbin")))


def render_all(paths, outdir, font_9px=False):
    """Batch-render art to PNG with pixelview -> {stem: png_path}."""
    if not os.path.exists(PIXELVIEW):
        sys.exit(f"pixelview not found at {PIXELVIEW}\n"
                 f"  build it:  cd ~/git/pixel-viewer && cargo build --release\n"
                 f"  or set PIXELVIEW=/path/to/pixelview")
    os.makedirs(outdir, exist_ok=True)
    for i in range(0, len(paths), 200):                 # keep argv sane
        cmd = [PIXELVIEW, "--render", *paths[i:i + 200], "--outdir", outdir]
        if font_9px:
            cmd.append("--font-9px")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            print(f"  ! pixelview: {r.stderr.strip()[:200]}")
    got = {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        png = os.path.join(outdir, stem + ".png")
        if os.path.exists(png):
            got[stem] = png
    return got


def sauce_of(path):
    try:
        return A.read_sauce(open(path, "rb").read())[0]
    except Exception:
        return None


def clean(s):
    return re.sub(r"\s+", " ", (s or "").replace("\x00", " ").strip())


def describe(title, kind, token):
    bits = [token, "ansi art", kind]
    t = re.sub(r"\b(logo|ansi|ans|blockart)\b", " ", clean(title), flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" -,:;#")
    # A bare date or serial ("10/96", "#1") describes nothing visual.
    if t and not re.fullmatch(r"[\d/#.\- ]+", t):
        bits.append(t)
    bits += ["cp437 block characters", "16 color ega palette", "text mode art"]
    return ", ".join(b for b in bits if b)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--token", default="grymmjack")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--rows", type=int, default=ROWS_PER_SCREEN)
    ap.add_argument("--cols", type=int, default=COLS_PER_SCREEN)
    ap.add_argument("--upscale", type=int, default=1,
                    help="1 for SD1.5 (640x384), 2 for SDXL (1280x768)")
    ap.add_argument("--font-9px", dest="font_9px", action="store_true",
                    help="render the 9-dot VGA cell, as ansilove and 16colo do")
    ap.add_argument("--kind", default="blockart logo",
                    help="fallback piece type used in captions")
    a = ap.parse_args()

    src = os.path.abspath(os.path.expanduser(a.src))
    out = os.path.abspath(os.path.expanduser(a.out))
    imgdir = os.path.join(out, f"{a.repeats}_{a.token}")
    os.makedirs(imgdir, exist_ok=True)

    files = art_files(src)
    if not files:
        sys.exit(f"no .ans/.xb under {src}")
    print(f"  rendering {len(files)} pieces with pixelview"
          f"{' (9-dot cell)' if a.font_9px else ''} ...")
    rendered = render_all(files, os.path.join(out, "_rendered"), a.font_9px)
    print(f"  rendered {len(rendered)}/{len(files)}")

    cw = a.cols * (9 if a.font_9px else CELL_W)
    chh = a.rows * CELL_H
    made = skipped = missing = 0
    manifest = []

    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        png = rendered.get(stem)
        if not png:
            missing += 1
            continue
        arr = np.asarray(Image.open(png).convert("RGB"))
        H, W = arr.shape[:2]
        title = clean((sauce_of(f) or {}).get("title", ""))

        nr = max(1, -(-H // chh))
        nc = max(1, -(-W // cw))
        for i in range(nr * nc):
            ri, ci = divmod(i, nc)
            tile = arr[ri * chh:(ri + 1) * chh, ci * cw:(ci + 1) * cw]
            if tile.shape[0] < chh or tile.shape[1] < cw:      # pad short edges
                pad = np.zeros((chh, cw, 3), np.uint8)
                pad[:tile.shape[0], :tile.shape[1]] = tile
                tile = pad
            if tile.any(-1).mean() < MIN_INK:
                skipped += 1
                continue

            t = Image.fromarray(tile)
            if a.upscale > 1:
                t = t.resize((t.width * a.upscale, t.height * a.upscale),
                             Image.NEAREST)
            name = f"{stem}_{i:03d}"
            t.save(os.path.join(imgdir, name + ".png"))
            cap = describe(title, a.kind, a.token)
            with open(os.path.join(imgdir, name + ".txt"), "w",
                      encoding="utf-8") as fh:
                fh.write(cap + "\n")
            manifest.append({"file": name + ".png", "source": os.path.basename(f),
                             "screen": i, "size": list(t.size), "caption": cap,
                             "title": title, "render": png})
            made += 1

    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    sizes = {tuple(m["size"]) for m in manifest}
    print(f"\n  {made} samples -> {imgdir}")
    print(f"  {skipped} near-empty screens skipped, {missing} pieces unrendered")
    print(f"  resolutions: {', '.join(f'{w}x{h}' for w, h in sorted(sizes))}\n")


if __name__ == "__main__":
    main()
